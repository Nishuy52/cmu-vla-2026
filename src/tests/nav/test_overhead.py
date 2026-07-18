"""Overhead-clearance layer: band membership, ground fallback, noise rejection,
costmap blocking + inflation, an under-table A* scenario, and real-data validation.

Motivation (verified upstream fact): terrainAnalysis.cpp filters /registered_scan to a
thin slab (~0.2 m above the vehicle) before publishing /terrain_map, so bar-height
tabletops/shelves are ABSENT from the terrain cloud — cells under a table read FREE.
The raw /registered_scan DOES carry those overhang points. integrate_scan_overhead
re-reads the scan and flags them so the planner never routes under furniture.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from core.interfaces import LidarScan, TerrainPatch
from core.nav.costmap import Costmap
from core.nav.occupancy import (
    DEFAULT_OVERHEAD_CONFIG,
    FREE,
    GROUND_OFFSET_WARMUP_PATCHES,
    OBSTACLE,
    VEHICLE_SENSOR_HEIGHT_ENV_VAR,
    OverheadConfig,
    OccupancyGrid,
    integrate_scan_overhead_decimated,
)
from core.nav.planner import astar
from tests.nav.helpers import make_points


def _scan(coords: list[tuple[float, float, float]], t: float = 0.0) -> LidarScan:
    """LidarScan from explicit [(x, y, z), ...] map-frame points."""
    arr = np.array(coords, dtype=np.float32) if coords else np.zeros((0, 3), np.float32)
    return LidarScan(t=t, points=arr)


def _repeat_cell(x: float, y: float, z: float, n: int) -> list[tuple[float, float, float]]:
    """n scan points in the SAME cell at height z (jitter kept within one 0.1 m cell)."""
    return [(x + 0.001 * i, y, z) for i in range(n)]


# --------------------------------------------------------------------------- band membership
def test_point_in_band_flags_overhead():
    grid = OccupancyGrid(cell_m=0.1)
    # Ground at z=0 (fallback: vehicle_z 0 - sensor_height 0.6 = -0.6, but a terrain
    # point pins the cell ground to 0.0). Overhang at 0.5 m is inside [0.25, 1.2].
    grid.integrate_patch(make_points([(0.05, 0.05, 0.02)]))  # ground z=0 in this cell
    grid.integrate_scan_overhead(_scan(_repeat_cell(0.05, 0.05, 0.5, 3)), vehicle_z=0.0)
    assert grid.is_overhead(*grid.world_to_cell(0.05, 0.05))


def test_point_below_band_not_flagged():
    grid = OccupancyGrid(cell_m=0.1)
    grid.integrate_patch(make_points([(0.05, 0.05, 0.02)]))  # ground z=0
    # 0.1 m above ground is below overhead_min (0.25) -> not overhead.
    grid.integrate_scan_overhead(_scan(_repeat_cell(0.05, 0.05, 0.10, 5)), vehicle_z=0.0)
    assert not grid.is_overhead(*grid.world_to_cell(0.05, 0.05))


def test_point_above_band_not_flagged():
    grid = OccupancyGrid(cell_m=0.1)
    grid.integrate_patch(make_points([(0.05, 0.05, 0.02)]))  # ground z=0
    # 1.5 m above ground is above overhead_max (1.2) — a wall/ceiling, not furniture.
    grid.integrate_scan_overhead(_scan(_repeat_cell(0.05, 0.05, 1.5, 5)), vehicle_z=0.0)
    assert not grid.is_overhead(*grid.world_to_cell(0.05, 0.05))


def test_band_edges_inclusive():
    grid = OccupancyGrid(cell_m=0.1)
    grid.integrate_patch(make_points([(0.05, 0.05, 0.02), (0.35, 0.05, 0.02)]))
    # Just inside overhead_min (0.25) and overhead_max (1.20) above ground(=0) -> both
    # flag; a hair OUTSIDE either edge (checked below) does not. (Exact float32 edges
    # are avoided — the boundary semantics are `>=`/`<=`, verified within tolerance.)
    grid.integrate_scan_overhead(_scan(_repeat_cell(0.05, 0.05, 0.26, 3)), vehicle_z=0.0)
    grid.integrate_scan_overhead(_scan(_repeat_cell(0.35, 0.05, 1.19, 3)), vehicle_z=0.0)
    assert grid.is_overhead(*grid.world_to_cell(0.05, 0.05))
    assert grid.is_overhead(*grid.world_to_cell(0.35, 0.05))
    # Just below the lower edge and just above the upper edge -> not flagged.
    grid2 = OccupancyGrid(cell_m=0.1)
    grid2.integrate_patch(make_points([(0.05, 0.05, 0.02), (0.35, 0.05, 0.02)]))
    grid2.integrate_scan_overhead(_scan(_repeat_cell(0.05, 0.05, 0.23, 3)), vehicle_z=0.0)
    grid2.integrate_scan_overhead(_scan(_repeat_cell(0.35, 0.05, 1.22, 3)), vehicle_z=0.0)
    assert not grid2.is_overhead(*grid2.world_to_cell(0.05, 0.05))
    assert not grid2.is_overhead(*grid2.world_to_cell(0.35, 0.05))


# --------------------------------------------------------------------------- local ground
def test_local_ground_fallback_when_no_terrain():
    """No terrain point in the cell -> ground = vehicle_z - vehicle_sensor_height."""
    grid = OccupancyGrid(cell_m=0.1)
    # vehicle_z=0, sensor_height default 0.6 -> ground ~ -0.6. A point at z=0.0 is
    # 0.6 m above that fallback ground -> inside the band, flags even over UNKNOWN.
    grid.integrate_scan_overhead(_scan(_repeat_cell(2.05, 2.05, 0.0, 3)), vehicle_z=0.0)
    assert grid.is_overhead(*grid.world_to_cell(2.05, 2.05))
    # The floor beneath was never classified -> still UNKNOWN (state untouched):
    # an overhead point over UNKNOWN floor still flags (spec §1).
    assert grid.state[grid.world_to_cell(2.05, 2.05)] == 0  # UNKNOWN


def test_terrain_ground_beats_fallback():
    """A terrain-derived ground_z shifts the band relative to the fallback."""
    grid = OccupancyGrid(cell_m=0.1)
    # Terrain point at z=-0.6 pins ground here. A scan point at z=0.0 is then 0.6 m
    # above ground -> in band. (With the -0.6 fallback it would also be in band, so
    # use a raised ground to prove the terrain value is what's used.)
    grid.integrate_patch(
        TerrainPatch(
            t=0.0,
            points=np.array([[0.05, 0.05, 0.30, 0.02]], dtype=np.float32),
            extended=False,
        )
    )
    # ground_z pinned to 0.30. A scan point at z=0.40 is only 0.10 above -> below band.
    grid.integrate_scan_overhead(_scan(_repeat_cell(0.05, 0.05, 0.40, 5)), vehicle_z=0.0)
    assert not grid.is_overhead(*grid.world_to_cell(0.05, 0.05))
    # A scan point at z=0.70 is 0.40 above the pinned ground -> in band.
    grid.integrate_scan_overhead(_scan(_repeat_cell(0.05, 0.05, 0.70, 5)), vehicle_z=0.0)
    assert grid.is_overhead(*grid.world_to_cell(0.05, 0.05))


# --------------------------------------------------------------------------- ground-offset estimator (issue #36)


def _terrain_pt(x: float, y: float, z: float, t: float = 0.0) -> TerrainPatch:
    return TerrainPatch(t=t, points=np.array([[x, y, z, 0.02]], dtype=np.float32), extended=False)


def _warm_up(grid: OccupancyGrid, vehicle_z, *, n: int = GROUND_OFFSET_WARMUP_PATCHES) -> None:
    """Feed n terrain patches (far from x=0..1, y=0..1 probe cells used by callers
    below) each with a vehicle_z sample, so grid.ground_offset_estimate warms up
    without pinning any terrain ground_z on the cells the tests probe."""
    vzs = vehicle_z if isinstance(vehicle_z, list) else [vehicle_z] * n
    assert len(vzs) == n
    for i, vz in enumerate(vzs):
        grid.integrate_patch(_terrain_pt(50.0 + i, 50.0, -0.6, t=float(i)), vehicle_z=vz)


def test_ground_offset_estimate_none_before_warmup():
    grid = OccupancyGrid(cell_m=0.1)
    for i in range(GROUND_OFFSET_WARMUP_PATCHES - 1):
        grid.integrate_patch(_terrain_pt(i, 0.0, -0.6, t=float(i)), vehicle_z=0.0)
    assert grid.ground_offset_estimate is None


def test_ground_offset_estimate_available_at_warmup_threshold():
    grid = OccupancyGrid(cell_m=0.1)
    _warm_up(grid, 0.0)
    # jingfan-like geometry: vehicle z ~= 0.0, floor/ground z ~= -0.6 -> offset ~0.6.
    assert grid.ground_offset_estimate == pytest.approx(0.6)


def test_ground_offset_estimate_ignores_patches_without_vehicle_z():
    """A patch integrated without vehicle_z contributes ground_z as usual but does
    NOT count toward warm-up (no sample to estimate the vehicle-height offset)."""
    grid = OccupancyGrid(cell_m=0.1)
    for i in range(GROUND_OFFSET_WARMUP_PATCHES + 5):
        grid.integrate_patch(_terrain_pt(i, 0.0, -0.6, t=float(i)))  # no vehicle_z
    assert grid.ground_offset_estimate is None


def test_ground_offset_estimate_reproduces_sim_rig_delta():
    """Sim-rig-like geometry (report finding 1): vehicle_z samples clustered around
    0.75-0.78 m above a floor at z=0 -> estimate lands in that range, not 0.60."""
    grid = OccupancyGrid(cell_m=0.1)
    for i, vz in enumerate([0.74, 0.75, 0.76, 0.75, 0.77]):
        grid.integrate_patch(_terrain_pt(50.0 + i, 50.0, 0.0, t=float(i)), vehicle_z=vz)
    assert grid.ground_offset_estimate == pytest.approx(0.75)
    assert 0.75 <= grid.ground_offset_estimate <= 0.78


def test_fallback_uses_warmed_estimator_over_stale_constant():
    """Once warmed, the runtime estimate (not the 0.60 constant) drives the
    fallback ground for cells with no terrain ground_z of their own. Uses a
    floor-at-0 warm-up so vehicle_z=0.75, ground=0.0 -> estimate=0.75, easy to
    reason about."""
    grid = OccupancyGrid(cell_m=0.1)
    for i in range(GROUND_OFFSET_WARMUP_PATCHES):
        grid.integrate_patch(_terrain_pt(50.0 + i, 50.0, 0.0, t=float(i)), vehicle_z=0.75)
    assert grid.ground_offset_estimate == pytest.approx(0.75)

    # Probe cell (0.05, 0.05) never saw terrain -> fallback path.
    # Estimator ground = vehicle_z(0.75) - estimate(0.75) = 0.0; a point at z=0.30
    # is 0.30 above it -> inside [0.25, 1.20] -> flagged.
    # (With the stale 0.60 constant, ground would be 0.15 and the same point only
    # 0.15 above it -> below band -> NOT flagged -- see the next test.)
    grid.integrate_scan_overhead(_scan(_repeat_cell(0.05, 0.05, 0.30, 3)), vehicle_z=0.75)
    assert grid.is_overhead(*grid.world_to_cell(0.05, 0.05))


def test_fallback_uses_constant_before_warmup():
    grid = OccupancyGrid(cell_m=0.1)
    # Only one patch (< warm-up threshold) -> estimator unavailable, falls back to
    # the configured constant (0.60): ground = 0.75 - 0.60 = 0.15.
    grid.integrate_patch(_terrain_pt(50.0, 50.0, 0.0), vehicle_z=0.75)
    assert grid.ground_offset_estimate is None
    # A point at z=0.30 is only 0.15 above that ground -> below band -> not flagged.
    grid.integrate_scan_overhead(_scan(_repeat_cell(0.05, 0.05, 0.30, 3)), vehicle_z=0.75)
    assert not grid.is_overhead(*grid.world_to_cell(0.05, 0.05))


def test_use_ground_offset_estimator_false_pins_the_constant():
    """OverheadConfig(use_ground_offset_estimator=False) bypasses a warmed
    estimator, e.g. to isolate the constant-only hypothesis in a validation run."""
    grid = OccupancyGrid(cell_m=0.1)
    for i in range(GROUND_OFFSET_WARMUP_PATCHES):
        grid.integrate_patch(_terrain_pt(50.0 + i, 50.0, 0.0, t=float(i)), vehicle_z=0.75)
    assert grid.ground_offset_estimate == pytest.approx(0.75)
    cfg = OverheadConfig(use_ground_offset_estimator=False)
    grid.integrate_scan_overhead(_scan(_repeat_cell(0.05, 0.05, 0.30, 3)), vehicle_z=0.75, cfg=cfg)
    assert not grid.is_overhead(*grid.world_to_cell(0.05, 0.05))


def test_env_override_takes_precedence_over_warmed_estimator(monkeypatch):
    """VLA_OVERHEAD_VEHICLE_SENSOR_HEIGHT_M pins the fallback, overriding both the
    default constant and a warmed runtime estimate."""
    monkeypatch.setenv(VEHICLE_SENSOR_HEIGHT_ENV_VAR, "0.60")
    grid = OccupancyGrid(cell_m=0.1)
    for i in range(GROUND_OFFSET_WARMUP_PATCHES):
        grid.integrate_patch(_terrain_pt(50.0 + i, 50.0, 0.0, t=float(i)), vehicle_z=0.75)
    assert grid.ground_offset_estimate == pytest.approx(0.75)
    # Pinned to 0.60 -> ground = 0.75 - 0.60 = 0.15; a point at 0.30 is only 0.15
    # above -> below band -> not flagged (would be flagged if the estimator won).
    grid.integrate_scan_overhead(_scan(_repeat_cell(0.05, 0.05, 0.30, 3)), vehicle_z=0.75)
    assert not grid.is_overhead(*grid.world_to_cell(0.05, 0.05))


def test_env_override_invalid_value_ignored(monkeypatch):
    monkeypatch.setenv(VEHICLE_SENSOR_HEIGHT_ENV_VAR, "not-a-number")
    grid = OccupancyGrid(cell_m=0.1)
    for i in range(GROUND_OFFSET_WARMUP_PATCHES):
        grid.integrate_patch(_terrain_pt(50.0 + i, 50.0, 0.0, t=float(i)), vehicle_z=0.75)
    # Falls through to the warmed estimator (0.75) since the env value doesn't parse.
    grid.integrate_scan_overhead(_scan(_repeat_cell(0.05, 0.05, 0.30, 3)), vehicle_z=0.75)
    assert grid.is_overhead(*grid.world_to_cell(0.05, 0.05))


# --------------------------------------------------------------------------- noise rejection
def test_min_points_noise_rejection():
    grid = OccupancyGrid(cell_m=0.1)
    grid.integrate_patch(make_points([(0.05, 0.05, 0.02)]))  # ground z=0
    # Only 2 in-band points; default min_points_per_cell=3 -> rejected as noise.
    grid.integrate_scan_overhead(_scan(_repeat_cell(0.05, 0.05, 0.5, 2)), vehicle_z=0.0)
    assert not grid.is_overhead(*grid.world_to_cell(0.05, 0.05))
    # A 3rd point (default threshold) tips it over.
    grid.integrate_scan_overhead(_scan(_repeat_cell(0.05, 0.05, 0.5, 3)), vehicle_z=0.0)
    assert grid.is_overhead(*grid.world_to_cell(0.05, 0.05))


def test_min_points_config_override():
    grid = OccupancyGrid(cell_m=0.1)
    grid.integrate_patch(make_points([(0.05, 0.05, 0.02)]))
    cfg = OverheadConfig(min_points_per_cell=1)
    grid.integrate_scan_overhead(_scan(_repeat_cell(0.05, 0.05, 0.5, 1)), vehicle_z=0.0, cfg=cfg)
    assert grid.is_overhead(*grid.world_to_cell(0.05, 0.05))


def test_state_stays_pure_overhead_is_separate():
    """Overhead flag never mutates the terrain-derived FREE/OBSTACLE state (spec §1)."""
    grid = OccupancyGrid(cell_m=0.1)
    grid.integrate_patch(make_points([(0.05, 0.05, 0.02)]))  # FREE floor
    assert grid.is_free(*grid.world_to_cell(0.05, 0.05))
    grid.integrate_scan_overhead(_scan(_repeat_cell(0.05, 0.05, 0.5, 5)), vehicle_z=0.0)
    # Still FREE in `state`; the overhead is a parallel flag only.
    assert grid.is_free(*grid.world_to_cell(0.05, 0.05))
    assert grid.is_overhead(*grid.world_to_cell(0.05, 0.05))


# --------------------------------------------------------------------------- costmap
def test_costmap_softens_overhead_cell_by_default():
    """Default (soft): an overhead cell is TRAVERSABLE at a steep cost, not blocked.

    Redteam H13: hard-blocking overhead sealed doorways on false positives. The
    default now makes overhead a soft high-cost cell — passable, cost-bearing.
    """
    grid = OccupancyGrid(cell_m=0.1)
    grid.integrate_patch(make_points([(0.05, 0.05, 0.02)]))  # FREE floor
    grid.integrate_scan_overhead(_scan(_repeat_cell(0.05, 0.05, 0.5, 5)), vehicle_z=0.0)
    cm = Costmap(grid, vehicle_radius_m=0.0)  # no inflation to isolate the cell
    r, c = grid.world_to_cell(0.05, 0.05)
    # NOT hard-blocked, but flagged soft (A* pays the UNKNOWN_COST_MULT to cross).
    assert cm.passable(r, c)
    assert cm.is_soft_overhead(r, c)
    assert cm.is_unknown(r, c)  # cost-bearing seam A* reads


def test_costmap_overhead_hard_flag_blocks_cell():
    """overhead_hard=True restores the legacy hard-block behaviour."""
    grid = OccupancyGrid(cell_m=0.1)
    grid.integrate_patch(make_points([(0.05, 0.05, 0.02)]))  # FREE floor
    grid.integrate_scan_overhead(_scan(_repeat_cell(0.05, 0.05, 0.5, 5)), vehicle_z=0.0)
    cm = Costmap(grid, vehicle_radius_m=0.0, overhead_hard=True)
    r, c = grid.world_to_cell(0.05, 0.05)
    assert cm.blocked(r, c)
    assert not cm.is_soft_overhead(r, c)


def test_allow_overhead_true_restores_old_behavior():
    grid = OccupancyGrid(cell_m=0.1)
    grid.integrate_patch(make_points([(0.05, 0.05, 0.02)]))  # FREE floor
    grid.integrate_scan_overhead(_scan(_repeat_cell(0.05, 0.05, 0.5, 5)), vehicle_z=0.0)
    cm = Costmap(grid, vehicle_radius_m=0.0, allow_overhead=True)
    r, c = grid.world_to_cell(0.05, 0.05)
    # With overhead allowed, the cell is passable (terrain-only behaviour).
    assert cm.passable(r, c)


def test_overhead_soft_inflation_applies():
    """A SOFT overhead cell inflates like an obstacle so the footprint is discouraged,
    but the inflated cells stay passable (soft), not blocked."""
    grid = OccupancyGrid(cell_m=0.1)
    # Build a small FREE patch so neighbours exist to inflate onto.
    grid.integrate_patch(make_points([(x / 10 + 0.05, 0.05, 0.02) for x in range(6)]))
    grid.integrate_scan_overhead(_scan(_repeat_cell(0.25, 0.05, 0.5, 5)), vehicle_z=0.0)
    cm = Costmap(grid, vehicle_radius_m=0.2)  # 2-cell inflation radius
    r0, c0 = grid.world_to_cell(0.25, 0.05)
    assert cm.is_soft_overhead(r0, c0)
    assert cm.passable(r0, c0)  # soft, not hard
    # A cell one step away carries the SOFT inflation (cost-bearing but passable).
    assert cm.is_soft_overhead(r0, c0 + 1)
    assert cm.passable(r0, c0 + 1)


def test_overhead_hard_inflation_blocks():
    """overhead_hard=True inflates the cell as a hard obstacle (legacy)."""
    grid = OccupancyGrid(cell_m=0.1)
    grid.integrate_patch(make_points([(x / 10 + 0.05, 0.05, 0.02) for x in range(6)]))
    grid.integrate_scan_overhead(_scan(_repeat_cell(0.25, 0.05, 0.5, 5)), vehicle_z=0.0)
    cm = Costmap(grid, vehicle_radius_m=0.2, overhead_hard=True)
    r0, c0 = grid.world_to_cell(0.25, 0.05)
    assert cm.blocked(r0, c0)
    assert cm.blocked(r0, c0 + 1)  # blocked by inflation


def test_clone_preserves_allow_overhead():
    grid = OccupancyGrid(cell_m=0.1)
    grid.integrate_patch(make_points([(0.05, 0.05, 0.02)]))
    grid.integrate_scan_overhead(_scan(_repeat_cell(0.05, 0.05, 0.5, 5)), vehicle_z=0.0)
    cm = Costmap(grid, allow_overhead=True)
    assert cm.clone().allow_overhead is True


# --------------------------------------------------------------------------- A* under-table
def _under_table_grid() -> OccupancyGrid:
    """A walled 1-cell corridor with a table straddling it: the only route from the
    left free room to the right free room passes UNDER a bar table.

    Terrain-only, the corridor floor is FREE and walkable (the base stack sees a
    clear doorway). The tabletop overhang (added by the caller) seals it.

    Layout: a 3-row band. Middle row (y~0.15) is the corridor; top/bottom rows are
    OBSTACLE walls so A* cannot detour through UNKNOWN space. Corridor cols 0..9:
    cols 0 and 9 are the open rooms, cols 1..8 the doorway floor between the table
    legs (which sit at cols 1 and 8 as terrain obstacles are NOT used — the legs are
    outside the corridor; here the whole doorway is FREE floor so terrain-only is
    fully passable, and only the overhead layer can block it).
    """
    grid = OccupancyGrid(cell_m=0.1)
    pts = []
    for col in range(10):
        x = col * 0.1 + 0.05
        pts.append((x, 0.15, 0.02))  # middle-row corridor floor: all FREE
        pts.append((x, 0.05, 0.6))   # bottom wall
        pts.append((x, 0.25, 0.6))   # top wall
    grid.integrate_patch(make_points(pts))
    return grid


def test_astar_passes_gap_terrain_only():
    """Sanity: terrain-only, the corridor floor is walkable end to end."""
    grid = _under_table_grid()
    cm = Costmap(grid, vehicle_radius_m=0.0, allow_overhead=True)
    start = grid.cell_to_world(*grid.world_to_cell(0.05, 0.15))
    goal = grid.cell_to_world(*grid.world_to_cell(0.95, 0.15))
    path = astar(cm, start, goal)
    assert path is not None


def _seal_under_table(grid: OccupancyGrid) -> None:
    """Flag the tabletop overhang across the corridor cols 2..7 (in band, seals gap)."""
    over = []
    for col in range(2, 8):
        x = col * 0.1 + 0.05
        over += _repeat_cell(x, 0.15, 0.5, 4)
    grid.integrate_scan_overhead(_scan(over), vehicle_z=0.0)


def test_astar_soft_overhead_still_crosses_under_table():
    """Redteam H13 default: with the tabletop overhang flagged SOFT, A* can still
    cross the only under-table corridor (route stays REACHABLE, just expensive) —
    where the old hard-block made it unreachable."""
    grid = _under_table_grid()
    _seal_under_table(grid)
    start = grid.cell_to_world(*grid.world_to_cell(0.05, 0.15))
    goal = grid.cell_to_world(*grid.world_to_cell(0.95, 0.15))

    # Old behaviour (hard-block): unreachable.
    cm_hard = Costmap(grid, vehicle_radius_m=0.0, overhead_hard=True)
    assert astar(cm_hard, start, goal) is None

    # New default (soft): still found — the corridor cells are cost-bearing, passable.
    cm_soft = Costmap(grid, vehicle_radius_m=0.0)
    path = astar(cm_soft, start, goal)
    assert path is not None
    # The route does traverse soft-overhead cells (there is no alternative here).
    assert any(cm_soft.is_soft_overhead(*grid.world_to_cell(x, y)) for x, y in path)

    # allow_overhead=True also crosses (terrain-only, zero penalty).
    cm_allow = Costmap(grid, vehicle_radius_m=0.0, allow_overhead=True)
    assert astar(cm_allow, start, goal) is not None


# --------------------------------------------------------------------------- decimation helper
def test_decimated_helper_clips_out_of_bounds():
    """The per-tick wiring helper skips points outside the current grid bounds and
    does not grow the grid for a far overhang."""
    grid = OccupancyGrid(cell_m=0.1)
    grid.integrate_patch(make_points([(0.05, 0.05, 0.02)]))  # tiny bounded grid
    shape_before = grid.shape
    # A far overhang (100 m away) is out of bounds -> ignored, no growth.
    integrate_scan_overhead_decimated(grid, _scan(_repeat_cell(100.0, 100.0, 0.0, 5)), 0.0)
    assert grid.shape == shape_before
    assert not grid.overhead.any()


def test_decimated_helper_flags_in_bounds():
    grid = OccupancyGrid(cell_m=0.1)
    # Seed a bounded region so (0.05,0.05) is inside bounds.
    grid.integrate_patch(make_points([(x / 10 + 0.05, 0.05, 0.02) for x in range(6)]))
    integrate_scan_overhead_decimated(grid, _scan(_repeat_cell(0.05, 0.05, 0.5, 5)), 0.0)
    assert grid.is_overhead(*grid.world_to_cell(0.05, 0.05))


def test_empty_scan_is_noop():
    grid = OccupancyGrid(cell_m=0.1)
    grid.integrate_patch(make_points([(0.05, 0.05, 0.02)]))
    grid.integrate_scan_overhead(_scan([]), vehicle_z=0.0)
    assert not grid.overhead.any()


# --------------------------------------------------------------------------- real data
def _jingfan_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "data" / "fixtures" / "jingfan"


@pytest.mark.skipif(
    not (_jingfan_dir() / "index.json").exists(),
    reason="jingfan fixtures not present",
)
def test_real_data_overhead_flags_more_than_terrain_alone():
    """Real room with bar-height tables + stools: the overhead layer must flag MORE
    blocked area than terrain alone, and specifically catch cells the terrain read as
    FREE floor (under the tabletops). Reports the delta."""
    base = _jingfan_dir()
    idx = json.load(open(base / "index.json"))
    kf = [
        k
        for k in idx["keyframes"]
        if {"scan", "terrain", "odom"} <= set(k["channels"])
    ]
    assert kf, "expected keyframes with scan+terrain+odom"

    grid = OccupancyGrid(cell_m=0.10)
    n = 40
    for k in kf[:n]:
        d = np.load(base / k["file"])
        grid.integrate_patch(TerrainPatch(t=k["t"], points=d["terrain"], extended=False))
        odom = d["odom"]
        integrate_scan_overhead_decimated(
            grid, LidarScan(t=k["t"], points=d["scan"]), float(odom[3])
        )
        grid.mark_pose(float(odom[1]), float(odom[2]))

    cell_a = grid.cell_m * grid.cell_m
    n_overhead = int(grid.overhead.sum())
    over_free = int((grid.overhead & (grid.state == FREE)).sum())
    over_unknown = int((grid.overhead & (grid.state == 0)).sum())

    cm_terrain = Costmap(grid, allow_overhead=True)
    cm_soft = Costmap(grid)  # default: soft overhead
    cm_overhead = Costmap(grid, overhead_hard=True)  # legacy hard-block
    blk_terrain = int(cm_terrain.base_blocked.sum())
    blk_overhead = int(cm_overhead.base_blocked.sum())
    soft_cells = int(cm_soft.overhead_soft.sum())
    delta = blk_overhead - blk_terrain

    print(
        "\n[real-data overhead validation] "
        f"{n} keyframes, grid {grid.shape}\n"
        f"  overhead cells: {n_overhead} ({n_overhead * cell_a:.2f} m^2)\n"
        f"  of which over terrain-FREE floor: {over_free} "
        f"({over_free * cell_a:.2f} m^2)  <-- tables the base stack reads as walkable\n"
        f"  of which over UNKNOWN floor:      {over_unknown} ({over_unknown * cell_a:.2f} m^2)\n"
        f"  costmap blocked terrain-only:  {blk_terrain} ({blk_terrain * cell_a:.2f} m^2)\n"
        f"  costmap blocked +overhead(hard):{blk_overhead} ({blk_overhead * cell_a:.2f} m^2)\n"
        f"  overhead adds if hard (infl):  {delta} ({delta * cell_a:.2f} m^2)\n"
        f"  SOFT-overhead cells (default): {soft_cells} ({soft_cells * cell_a:.2f} m^2)"
    )

    # The overhead layer must catch real furniture: more blocked area under the
    # legacy hard mode, and at least some of it over floor the terrain analysis
    # called FREE (under-table cells).
    assert n_overhead > 0
    assert delta > 0
    assert over_free > 0
    # Default (soft) mode adds ZERO hard-blocked area vs terrain-only, yet marks the
    # furniture cells soft (cost-bearing) so routes stay reachable (redteam H13).
    assert int(cm_soft.base_blocked.sum()) == blk_terrain
    assert soft_cells > 0
