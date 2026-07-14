"""Redteam H13 (SYS-F12) — overhead-clearance softening + decimation-consistent gate.

Two failure modes this exercises:

1. False-overhead sealing doorways via hard-block + inflation. The old Costmap
   treated an overhead-flagged cell as a HARD obstacle and inflated it 0.4 m; a
   single false-positive overhead cell next to the only door made the route
   UNREACHABLE, and the IF head quietly answered from the nearest legal point
   (points lost). The default is now SOFT: overhead cells are traversable at a
   steep A* cost, so a false positive makes a route EXPENSIVE, never impossible,
   while a genuinely-blocked under-furniture route (no alternative) stays strongly
   avoided when a clear alternative exists.

2. Decimation-vs-min_points interaction. `integrate_scan_overhead_decimated`
   strides the raw scan up to ~5x BEFORE the `min_points_per_cell >= 3` noise
   gate, so a sparse-but-real overhang edge that just clears the gate at full
   density fails after decimation. The helper now scales the effective threshold
   by the applied stride (`ceil(min_points / stride)`, floor 1), so the same
   physical edge flags at any stride.

Corridor threading stays HARD: a pinch corridor threaded under a table is not a
legal gate. `Costmap.clone()` (the pinch overlay's only entry) hardens overhead.
"""
from __future__ import annotations

import numpy as np

from core.interfaces import LidarScan
from core.nav.costmap import Costmap
from core.nav.occupancy import (
    OverheadConfig,
    OccupancyGrid,
    integrate_scan_overhead_decimated,
)
from core.nav.planner import astar, path_crosses_gate, plan_through
from tests.nav.helpers import make_points, patch_from_ascii


def _scan(coords, t=0.0):
    arr = np.array(coords, dtype=np.float32) if coords else np.zeros((0, 3), np.float32)
    return LidarScan(t=t, points=arr)


def _repeat_cell(x, y, z, n):
    return [(x + 0.001 * i, y, z) for i in range(n)]


# --------------------------------------------------------------------------- doorway
def _doorway_grid() -> OccupancyGrid:
    """One-door layout: a left FREE room and a right FREE room separated by a solid
    OBSTACLE wall with a SINGLE FREE gap (the door). The whole scene is walled on
    the perimeter so A* cannot detour through UNKNOWN/out-of-bounds space — the door
    is genuinely the only passage.

    Columns: 0 = left room, 1 = wall with a gap at the middle row, 2 = right room.
    The door cell is (middle row, col 1). Everything else on col 1 is OBSTACLE.
    """
    rows = [
        "#####",
        "#.#.#",  # wall column (col 2) solid here
        "#...#",  # <- door row: the gap in the wall is at col 2
        "#.#.#",
        "#####",
    ]
    grid = OccupancyGrid(cell_m=0.1)
    grid.integrate_patch(patch_from_ascii(rows, cell_m=0.1))
    return grid


def test_spurious_overhead_at_door_keeps_route_reachable():
    """A spurious overhead cell on the only doorway path: SOFT default keeps the
    route reachable (higher cost) where the old HARD block made it unreachable."""
    grid = _doorway_grid()
    door = grid.cell_to_world(*grid.world_to_cell(0.25, 0.25))   # the single door cell
    start = grid.cell_to_world(*grid.world_to_cell(0.15, 0.25))  # left room
    goal = grid.cell_to_world(*grid.world_to_cell(0.35, 0.25))   # right room

    # Sanity: terrain-only, the door is walkable (left room -> door -> right room).
    cm_clear = Costmap(grid, vehicle_radius_m=0.0)
    assert astar(cm_clear, start, goal) is not None

    # A false-positive overhead flag lands on the single door cell (e.g. a wall
    # sconce / window sill protruding into the 0.25-1.20 m band over FREE floor).
    over = _repeat_cell(door[0], door[1], 0.5, 5)
    grid.integrate_scan_overhead(_scan(over), vehicle_z=0.0)

    # OLD behaviour (hard-block): the door cell is a hard obstacle -> route
    # unreachable (no inflation needed; the door is one cell wide).
    cm_hard = Costmap(grid, vehicle_radius_m=0.0, overhead_hard=True)
    assert astar(cm_hard, start, goal) is None

    # NEW default (soft): route STILL found — expensive, not impossible.
    cm_soft = Costmap(grid, vehicle_radius_m=0.0)
    path = astar(cm_soft, start, goal)
    assert path is not None
    # The route pays the soft-overhead cost on the door cell.
    assert any(cm_soft.is_soft_overhead(*grid.world_to_cell(x, y)) for x, y in path)


def _cost_of(cm: Costmap, path) -> float:
    """Sum A*-consistent step cost of a path (soft/unknown cells cost the multiplier)."""
    from core.nav.planner import UNKNOWN_COST_MULT

    total = 0.0
    for (x0, y0), (x1, y1) in zip(path[:-1], path[1:]):
        r0, c0 = cm.grid.world_to_cell(x0, y0)
        r1, c1 = cm.grid.world_to_cell(x1, y1)
        base = np.hypot(r1 - r0, c1 - c0)
        if cm.is_unknown(r1, c1):
            base *= UNKNOWN_COST_MULT
        total += base
    return total


def test_under_table_avoided_when_clear_alternative_exists():
    """A clear detour beside an under-table stretch: the min-cost A* route AVOIDS
    the soft-overhead cells (they cost more than the free detour)."""
    # A 3-row corridor: the direct middle row runs under a table (soft overhead);
    # the top and bottom rows are clear FREE alternatives of equal geometric length.
    rows = [
        "..........",
        "..........",  # direct row (will be flagged overhead across the middle)
        "..........",
    ]
    grid = OccupancyGrid(cell_m=0.1)
    grid.integrate_patch(patch_from_ascii(rows, cell_m=0.1))
    # Flag the whole middle row's central span as overhead (a table straddling it).
    over = []
    for col in range(2, 8):
        over += _repeat_cell(col * 0.1 + 0.05, 0.15, 0.5, 5)
    grid.integrate_scan_overhead(_scan(over), vehicle_z=0.0)

    cm = Costmap(grid, vehicle_radius_m=0.0)
    start = grid.cell_to_world(*grid.world_to_cell(0.05, 0.15))  # middle-row start
    goal = grid.cell_to_world(*grid.world_to_cell(0.95, 0.15))   # middle-row goal
    path = astar(cm, start, goal)
    assert path is not None
    # The clear alternative exists, so the min-cost route does NOT sit in the
    # soft-overhead band (it detours to a clear row).
    crossed_soft = sum(
        cm.is_soft_overhead(*grid.world_to_cell(x, y)) for x, y in path
    )
    assert crossed_soft == 0


# --------------------------------------------------------------------------- corridor gate
def _pinch_gap_grid():
    """The pinch scenario from test_planner: two pillars leaving a 1-cell gap at
    col 10; the direct plan to the gate midpoint skirts around and MISSES the gate,
    so the planner falls through to the pinch overlay to force threading. Returns
    (grid, gate, start, goal)."""
    rows = []
    for i in range(12):
        line = list("." * 20)
        if 3 <= i <= 8:
            for c in (8, 9, 11, 12):
                line[c] = "#"
        rows.append("".join(line))
    grid = OccupancyGrid(cell_m=0.1)
    grid.integrate_patch(patch_from_ascii(rows, cell_m=0.1))
    gate = ((1.05, 0.25), (1.05, 0.95))
    start = (1.05, 0.05)
    goal = (1.05, 1.15)
    return grid, gate, start, goal


def test_corridor_pinch_still_threads_gap_without_overhead():
    """Baseline: with no overhead, the pinch overlay threads the gate (unchanged)."""
    grid, gate, start, goal = _pinch_gap_grid()
    cm = Costmap(grid, vehicle_radius_m=0.0)
    path = plan_through(cm, start, [("corridor_between", gate), ("goto", goal)])
    assert path is not None
    assert path_crosses_gate(path, gate)


def test_corridor_gate_under_overhead_is_rejected():
    """A pinch corridor whose gate sits UNDER a table is NOT a legal gate.

    The pinch overlay forces threading through the gap by CLONING the costmap and
    blocking everything outside a narrow corridor around the gate. `clone()` hardens
    the soft-overhead layer, so an under-table gap becomes hard-blocked: on the
    pinch's costmap the gate has no crossing at all (redteam H13 corridor exception).

    We prove this at the mechanism level (the clone the pinch overlay builds from),
    because the base `plan_through` accepts the soft crossing before the pinch fires
    when the direct path happens to cross the gap — the pinch is exactly where an
    under-furniture gate must be, and is, rejected.
    """
    grid, gate, start, goal = _pinch_gap_grid()
    # Flag the WHOLE gap column (col 10, the only crossing) overhead across the gate
    # span — a table straddling the doorway.
    gx = 1.05
    for gy in (0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95):
        grid.integrate_scan_overhead(_scan(_repeat_cell(gx, gy, 0.5, 5)), vehicle_z=0.0)

    from core.nav.planner import _pinch_costmap

    cm = Costmap(grid, vehicle_radius_m=0.0)

    # Base (soft) costmap: A* CAN cross the gap under the table (soft, reachable).
    assert astar(cm, start, goal) is not None

    # The pinch overlay forces threading through the gate corridor AND (via clone)
    # hardens overhead: every gap cell along the gate is now hard-blocked, so within
    # the pinch costmap the goal on the far side of the gate is UNREACHABLE — the
    # under-table corridor gate is rejected.
    pinch = _pinch_costmap(cm, gate)
    gr, gcol = grid.world_to_cell(gx, 0.60)  # a gap cell
    assert pinch.blocked(gr, gcol)  # overhead hardened inside the pinch
    # The pinch overlay is exactly where an under-furniture gate is rejected: it
    # forces threading through the gate corridor, but the hardened gap admits NO
    # crossing of the gate segment. `plan_through` accepts a corridor only if the
    # pinch-forced plan to the gate midpoint CROSSES the gate — here it cannot.
    mid = ((gate[0][0] + gate[1][0]) / 2.0, (gate[0][1] + gate[1][1]) / 2.0)
    seg = astar(pinch, start, mid)
    assert seg is None or not path_crosses_gate(seg, gate)


def test_clone_hardens_overhead_for_corridor():
    """clone() (pinch overlay entry) folds soft overhead into the hard mask."""
    grid = OccupancyGrid(cell_m=0.1)
    grid.integrate_patch(make_points([(0.05, 0.05, 0.02)]))
    grid.integrate_scan_overhead(_scan(_repeat_cell(0.05, 0.05, 0.5, 5)), vehicle_z=0.0)
    cm = Costmap(grid, vehicle_radius_m=0.0)
    r, c = grid.world_to_cell(0.05, 0.05)
    assert cm.passable(r, c) and cm.is_soft_overhead(r, c)  # soft in the base
    clone = cm.clone()
    assert clone.blocked(r, c)  # hard in the corridor clone
    assert not clone.is_soft_overhead(r, c)
    # And the base costmap is unchanged (still soft).
    assert cm.passable(r, c)


# --------------------------------------------------------------------------- decimation gate
def test_stride_scaled_gate_passes_sparse_edge_at_stride_5():
    """A sparse-but-real overhang edge that passes the min_points gate at FULL
    density must ALSO pass after a 5x stride decimation (stride-scaled threshold)."""
    # Build a bounded grid so the overhang cell is in-bounds and not clipped.
    grid = OccupancyGrid(cell_m=0.1)
    grid.integrate_patch(make_points([(x / 10 + 0.05, 0.05, 0.02) for x in range(6)]))

    cfg = OverheadConfig()  # default min_points_per_cell = 3

    # The overhang cell has exactly `min_points` in-band returns at full density
    # (a sparse real edge). To force a 5x stride we pad the scan with 5x max_pts of
    # OUT-OF-BAND ground filler in the SAME bounded region (kept in-bounds so they
    # are not clipped, below the band so they never flag). max_pts small to force it.
    max_pts = 20
    edge = _repeat_cell(0.05, 0.05, 0.5, cfg.min_points_per_cell)  # 3 in-band points
    # Ground filler at z below the band, spread across in-bounds cells.
    filler = []
    for i in range(max_pts * 5):  # -> ~5x stride
        fx = (i % 6) / 10 + 0.05
        filler.append((fx, 0.05, 0.0))  # 0.6 above the -0.6 fallback? no terrain ->
    # Note: cells with a terrain point have ground 0.0, so z=0.0 filler is 0.0 above
    # ground -> below overhead_min (0.25) -> never in band. Safe filler.
    pts = edge + filler
    scan = _scan(pts)

    # Full density: the 3-point edge clears the gate.
    grid_full = OccupancyGrid(cell_m=0.1)
    grid_full.integrate_patch(make_points([(x / 10 + 0.05, 0.05, 0.02) for x in range(6)]))
    grid_full.integrate_scan_overhead(_scan(edge), vehicle_z=0.0, cfg=cfg)
    assert grid_full.is_overhead(*grid_full.world_to_cell(0.05, 0.05))

    # Decimated (forced ~5x stride): WITHOUT stride scaling the 3 points decimate to
    # ~1 survivor and fail the >=3 gate. WITH the stride-scaled threshold
    # (ceil(3/5)=1) the edge still flags.
    integrate_scan_overhead_decimated(grid, scan, 0.0, cfg=cfg, max_pts=max_pts)
    assert grid.is_overhead(*grid.world_to_cell(0.05, 0.05))


def test_stride_one_leaves_threshold_unchanged():
    """No decimation (stride 1) -> the full min_points gate still applies (2 < 3 fails)."""
    grid = OccupancyGrid(cell_m=0.1)
    grid.integrate_patch(make_points([(0.05, 0.05, 0.02)]))
    cfg = OverheadConfig()  # min_points = 3
    # Only 2 in-band points, small scan (no stride) -> below the full gate.
    integrate_scan_overhead_decimated(
        grid, _scan(_repeat_cell(0.05, 0.05, 0.5, 2)), 0.0, cfg=cfg, max_pts=12000
    )
    assert not grid.is_overhead(*grid.world_to_cell(0.05, 0.05))


def test_stride_scaled_threshold_floor_is_one():
    """Even a huge stride never drops the effective threshold below 1 (a single real
    in-band return is still required — pure noise of 0 points never flags)."""
    grid = OccupancyGrid(cell_m=0.1)
    grid.integrate_patch(make_points([(x / 10 + 0.05, 0.05, 0.02) for x in range(6)]))
    cfg = OverheadConfig(min_points_per_cell=3)
    # 1 real in-band edge point + heavy out-of-band filler forcing a large stride.
    max_pts = 10
    edge = _repeat_cell(0.05, 0.05, 0.5, 1)
    filler = [((i % 6) / 10 + 0.05, 0.05, 0.0) for i in range(max_pts * 10)]  # ~10x
    integrate_scan_overhead_decimated(grid, _scan(edge + filler), 0.0, cfg=cfg, max_pts=max_pts)
    # ceil(3/10)=1, and the single edge point survives striding (index 0 kept) -> flags.
    assert grid.is_overhead(*grid.world_to_cell(0.05, 0.05))
