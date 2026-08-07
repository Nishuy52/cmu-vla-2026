"""Frontier detection, clustering, min-size filter, scoring + affinity injection."""
from __future__ import annotations

import numpy as np

from core.nav.frontiers import (
    DEGENERATE_POCKET_CELLS,
    DEGENERATE_POCKET_RATIO,
    MIN_CLUSTER_SIZE,
    _bfs_distances,
    detect_frontiers,
    frontier_mask,
    last_call_info,
)
from core.nav.occupancy import FREE, UNKNOWN, OBSTACLE, OccupancyGrid
from tests.nav.helpers import patch_from_ascii


def _direct_grid(shape: tuple[int, int], cell_m: float = 0.1) -> OccupancyGrid:
    """An OccupancyGrid with `state` set directly to exact FREE/UNKNOWN cell counts,
    bypassing integrate_patch -- lets #83 tests pin exact pocket/component sizes."""
    grid = OccupancyGrid(cell_m=cell_m)
    h, w = shape
    grid.state = np.full((h, w), UNKNOWN, dtype=np.int8)
    grid.intensity = np.full((h, w), -np.inf, dtype=np.float32)
    grid.observed = np.zeros((h, w), dtype=bool)
    grid.overhead = np.zeros((h, w), dtype=bool)
    grid.ground_z = np.full((h, w), np.nan, dtype=np.float32)
    return grid


def _free_unknown_grid():
    """A block of FREE with UNKNOWN to the right: the boundary column is a frontier."""
    grid = OccupancyGrid(cell_m=0.1)
    rows = ["." * 6 for _ in range(6)]  # 6x6 free block, everything else unknown
    grid.integrate_patch(patch_from_ascii(rows, cell_m=0.1))
    return grid


def test_frontier_mask_marks_free_next_to_unknown():
    grid = _free_unknown_grid()
    mask = frontier_mask(grid)
    # Border free cells (adjacent to out-of-block unknown) are frontiers.
    assert mask.any()
    # An interior FREE cell fully surrounded by FREE is NOT a frontier... but in a
    # 6x6 block every cell touches the border/unknown, so check the mask is only on FREE.
    assert np.all(grid.state[mask] == FREE)


def test_frontier_cluster_min_size_filter():
    grid = OccupancyGrid(cell_m=0.1)
    # A tiny isolated free speck (< MIN_CLUSTER_SIZE frontier cells) should be dropped.
    grid.integrate_patch(patch_from_ascii(["."], cell_m=0.1))
    fr = detect_frontiers(grid, vehicle_xy=(0.05, 0.05))
    assert all(f.size >= MIN_CLUSTER_SIZE for f in fr)
    # Single cell -> no clusters survive.
    assert fr == []


def test_frontier_detected_on_big_block():
    grid = _free_unknown_grid()
    fr = detect_frontiers(grid, vehicle_xy=(0.05, 0.05))
    assert len(fr) >= 1
    assert fr[0].size >= MIN_CLUSTER_SIZE


def test_affinity_injection_reorders_frontiers():
    """Two reachable frontier clusters (near room + far room joined by a walled
    corridor). Without affinity the near cluster wins on distance; a strong affinity
    biased to the far room must flip the top ranking."""
    grid = OccupancyGrid(cell_m=0.1)
    # Two 5x5 rooms; corridor row 2 links them, flanked by '#' so the corridor cells
    # are not themselves frontiers -> two distinct, mutually reachable clusters.
    rows = [
        ".....#.....",
        ".....#.....",
        "...........",
        ".....#.....",
        ".....#.....",
    ]
    grid.integrate_patch(patch_from_ascii(rows, cell_m=0.1))
    veh = (0.15, 0.25)  # in the near (left) room

    base = detect_frontiers(grid, veh, affinity=None, min_cluster_size=1)
    assert len(base) >= 2
    # Without affinity, the near (left) frontier ranks first on distance.
    assert base[0].xy[0] < 0.5

    def aff(xy):
        return 100.0 if xy[0] > 0.5 else 0.0

    biased = detect_frontiers(grid, veh, affinity=aff, min_cluster_size=1)
    # Strong affinity to the far (right) room promotes it to the top.
    assert biased[0].xy[0] > 0.5
    assert biased[0].affinity == 100.0


def test_score_formula_components():
    grid = _free_unknown_grid()
    fr = detect_frontiers(grid, vehicle_xy=(0.05, 0.05), w_size=1.0, w_dist=0.5,
                          w_affinity=4.0)
    f = fr[0]
    expected = 1.0 * f.size - 0.5 * f.path_distance + 4.0 * f.affinity
    assert abs(f.score - expected) < 1e-6


def test_frontiers_ranked_descending():
    grid = OccupancyGrid(cell_m=0.1)
    grid.integrate_patch(patch_from_ascii(["." * 5 for _ in range(5)], cell_m=0.1))
    grid.integrate_patch(patch_from_ascii(["." * 5 for _ in range(5)], cell_m=0.1,
                                          origin_x=4.0))
    fr = detect_frontiers(grid, vehicle_xy=(0.2, 0.2))
    scores = [f.score for f in fr]
    assert scores == sorted(scores, reverse=True)


def test_no_frontiers_when_fully_enclosed():
    """FREE block fully ringed by OBSTACLE has no FREE-adjacent-UNKNOWN cells."""
    grid = OccupancyGrid(cell_m=0.1)
    rows = [
        "#####",
        "#...#",
        "#...#",
        "#...#",
        "#####",
    ]
    grid.integrate_patch(patch_from_ascii(rows, cell_m=0.1))
    fr = detect_frontiers(grid, vehicle_xy=(0.25, 0.25))
    assert fr == []


# --------------------------------------------------------------------------- #83
def test_disconnection_repro_reroot_reaches_frontiers():
    """Live #83 repro shape: the vehicle sits on a single FREE cell isolated by an
    UNKNOWN annulus from a large FREE region that carries the actual frontiers
    (the terrain blind-spot-around-spawn mechanism). Pre-fix semantics: the raw
    vehicle-rooted BFS alone reaches only the singleton, so every frontier in the
    big block is stamped with the UNREACHABLE_PD (1e6) sentinel. Post-fix:
    detect_frontiers re-roots off the much larger component and every frontier
    comes back with a finite path_distance."""
    rows = [
        "      ......",
        "      ......",
        "      ......",
        ".     ......",
        "      ......",
        "      ......",
    ]
    grid = OccupancyGrid(cell_m=0.1)
    grid.integrate_patch(patch_from_ascii(rows, cell_m=0.1))
    vehicle_xy = (0.05, 0.25)
    vehicle_cell = grid.world_to_cell(*vehicle_xy)
    assert grid.is_free(*vehicle_cell)

    # Pre-fix semantics: the plain vehicle-rooted BFS reaches only the singleton.
    raw_dist = _bfs_distances(grid, vehicle_cell)
    assert int(np.isfinite(raw_dist).sum()) == 1

    # Post-fix: detect_frontiers re-roots off the 36-cell block and returns
    # finite-distance, reachable frontiers for it (the singleton's OWN frontier
    # cluster is itself disconnected from the new root and legitimately stays
    # unreachable -- the fix reconnects the big block, not the singleton).
    frontiers = detect_frontiers(grid, vehicle_xy, min_cluster_size=1)
    assert frontiers
    assert any(f.path_distance < 1e6 for f in frontiers)
    info = last_call_info()
    assert info["bfs_reroot"] is True
    assert info["pocket_cells"] == 1
    assert info["largest_component_cells"] == 36


def test_degenerate_threshold_boundary_pocket_at_threshold_no_reroot():
    """Pocket size == DEGENERATE_POCKET_CELLS (not <) must NOT re-root, even next
    to a much larger separate FREE component -- the far component's frontiers stay
    stamped unreachable, and the vehicle-rooted distances are untouched.

    The gap between the two components is a genuine OBSTACLE wall (not UNKNOWN,
    see issue #197): a #197-style UNKNOWN-only gap would make the far component
    reachable via the unknown-crossing fallback regardless of reroot, which is
    not what this test is pinning -- this test is about the reroot decision, so
    the far component must stay unreachable by construction (a real wall) to
    isolate that."""
    grid = _direct_grid(shape=(40, 40))
    grid.state[0:5, 0:10] = FREE  # exactly DEGENERATE_POCKET_CELLS (50) cells
    # Wall spans the FULL grid width (not just under the far component) -- a
    # partial wall would leave an UNKNOWN detour around its ends that the #197
    # crossing fallback could legitimately find, which would defeat the point of
    # this test (isolating the reroot decision, not the crossing fallback).
    grid.state[5:10, :] = OBSTACLE  # wall, not an UNKNOWN gap
    grid.state[10:25, 0:20] = FREE  # a separate 300-cell component
    vehicle_xy = grid.cell_to_world(2, 5)  # inside the 50-cell pocket

    frontiers = detect_frontiers(grid, vehicle_xy, min_cluster_size=1)
    info = last_call_info()
    assert info["pocket_cells"] == DEGENERATE_POCKET_CELLS
    assert info["bfs_reroot"] is False
    # Threshold is a strict '<': at-threshold skips the (expensive) full-grid
    # component scan entirely.
    assert info["largest_component_cells"] is None
    # The far, wall-separated component's frontier(s) are still stamped unreachable.
    assert any(f.path_distance >= 1e6 for f in frontiers)
    assert info["unknown_crossing"] is False


def test_degenerate_ratio_boundary_insufficient_ratio_no_reroot():
    """Pocket small enough to qualify, but the largest other component is under
    DEGENERATE_POCKET_RATIO x its size -- must NOT re-root.

    The gap is a genuine OBSTACLE wall (see #197 note on the sibling test above)
    so the far component stays unreachable regardless of the reroot decision."""
    grid = _direct_grid(shape=(40, 40))
    grid.state[0:2, 0:5] = FREE  # 10-cell vehicle pocket
    grid.state[2:10, :] = OBSTACLE  # full-width wall, not an UNKNOWN gap
    grid.state[10:17, 0:5] = FREE  # a separate 35-cell component (< 4x10 = 40)
    vehicle_xy = grid.cell_to_world(0, 2)

    frontiers = detect_frontiers(grid, vehicle_xy, min_cluster_size=1)
    info = last_call_info()
    assert info["pocket_cells"] == 10
    assert info["largest_component_cells"] == 35
    assert 35 < DEGENERATE_POCKET_RATIO * 10
    assert info["bfs_reroot"] is False
    assert any(f.path_distance >= 1e6 for f in frontiers)
    assert info["unknown_crossing"] is False


# --------------------------------------------------------------------------- #197
# A remote FREE component -- not degenerate, not around the vehicle -- separated
# from the vehicle's component by UNKNOWN cells only must become reachable (its
# frontiers must survive SYS-F9); the same shape separated by an OBSTACLE wall
# must NOT (the base autonomy stack still must never be told to plan through a
# wall). See core.heads.explore_step._frontier_unreachable for the SYS-F9 gate
# these tests import read-only (owned by the heads surface, not touched here).
from core.heads.explore_step import UNREACHABLE_PD, _frontier_unreachable  # noqa: E402


def _find_frontier_near(frontiers, xy: tuple[float, float], tol: float = 1.5):
    for f in frontiers:
        if abs(f.xy[0] - xy[0]) <= tol and abs(f.xy[1] - xy[1]) <= tol:
            return f
    return None


def test_remote_pocket_separated_by_unknown_becomes_reachable():
    """A FREE pocket well away from the vehicle, not degenerate (so the #83
    re-root never fires for the vehicle's own -- already large -- pocket), joined
    to the vehicle's component only by a stretch of UNKNOWN cells: the #197
    mechanism. Must come back with a finite path_distance and survive SYS-F9."""
    grid = _direct_grid(shape=(40, 40))
    grid.state[0:10, 0:10] = FREE  # 100-cell vehicle component -- not degenerate
    # rows/cols 10:20 stay UNKNOWN: the seam. No OBSTACLE anywhere in the grid.
    grid.state[20:25, 20:30] = FREE  # 50-cell remote pocket, holds the frontiers
    vehicle_xy = grid.cell_to_world(5, 5)
    vehicle_cell = grid.world_to_cell(*vehicle_xy)

    # Pre-fix semantics: the FREE-only wavefront never crosses the UNKNOWN seam.
    raw_dist = _bfs_distances(grid, vehicle_cell)
    assert int(np.isfinite(raw_dist).sum()) == 100  # only the vehicle's own block

    frontiers = detect_frontiers(grid, vehicle_xy, min_cluster_size=1)
    info = last_call_info()
    assert info["bfs_reroot"] is False  # vehicle's own pocket is not degenerate
    assert info["unknown_crossing"] is True

    remote = _find_frontier_near(frontiers, grid.cell_to_world(22, 25))
    assert remote is not None
    assert remote.path_distance < UNREACHABLE_PD
    assert not _frontier_unreachable(remote)  # SYS-F9 must not drop it


def test_remote_pocket_separated_by_obstacle_wall_stays_unreachable():
    """Same shape as above, but the seam is a genuine OBSTACLE wall spanning the
    full grid width -- not UNKNOWN. The remote pocket must stay unreachable: the
    fix must never route the vehicle through a wall."""
    grid = _direct_grid(shape=(40, 40))
    grid.state[0:10, 0:10] = FREE  # 100-cell vehicle component
    grid.state[10:20, :] = OBSTACLE  # full-width wall, not an UNKNOWN gap
    grid.state[20:25, 20:30] = FREE  # 50-cell remote pocket, walled off
    vehicle_xy = grid.cell_to_world(5, 5)

    frontiers = detect_frontiers(grid, vehicle_xy, min_cluster_size=1)
    info = last_call_info()
    assert info["bfs_reroot"] is False
    assert info["unknown_crossing"] is False

    remote = _find_frontier_near(frontiers, grid.cell_to_world(22, 25))
    assert remote is not None
    assert remote.path_distance >= UNREACHABLE_PD
    assert _frontier_unreachable(remote)  # SYS-F9 must still drop it


def test_office_2_shaped_remote_pocket_recovers():
    """A grid proportioned after the live office_2 G3/slot7 costmap (the 200-of-297
    rejects cited in #197): reports/cluster_verify/714719/debug/7_office_2_inst/
    explore_debug_455284.jsonl, final record -- shape [196, 224], free 3145,
    unknown 39755, obstacle 1004, reachable_pocket_cells 2921 (i.e. 224 of 3145
    FREE cells, ~7%, never joined the vehicle-rooted FREE-only component, for 50
    ticks straight). The dump carries only those aggregate per-tick counts, not
    per-cell grid state, so the exact office_2 geometry cannot be reconstructed
    from it -- this grid is SYNTHETIC, sized (70x80, matching the 196:224 aspect
    ratio) and proportioned (free/obstacle fraction, and an UNKNOWN-only remote
    pocket sized to roughly the same ~7% unreachable-free share) to match those
    stats qualitatively, not a literal replay."""
    grid = _direct_grid(shape=(70, 80))
    grid.state[5:25, 5:24] = FREE  # main room, ~380 cells
    grid.state[10:15, 10:15] = OBSTACLE  # furniture-like clutter inside the room
    grid.state[50:55, 50:56] = FREE  # remote pocket, ~30 cells -- reached only
    # through rows/cols 25:50 UNKNOWN, no OBSTACLE anywhere in the seam.
    vehicle_xy = grid.cell_to_world(7, 7)
    vehicle_cell = grid.world_to_cell(*vehicle_xy)

    raw_dist = _bfs_distances(grid, vehicle_cell)
    pocket_cells = int(np.isfinite(raw_dist).sum())
    assert pocket_cells >= DEGENERATE_POCKET_CELLS  # main room is not degenerate

    frontiers = detect_frontiers(grid, vehicle_xy, min_cluster_size=1)
    info = last_call_info()
    assert info["bfs_reroot"] is False  # not the #83 mechanism -- main room is fine
    assert info["unknown_crossing"] is True  # the #197 mechanism is what fires

    remote = _find_frontier_near(frontiers, grid.cell_to_world(52, 53))
    assert remote is not None
    assert remote.path_distance < UNREACHABLE_PD
    assert not _frontier_unreachable(remote)


def test_197_does_not_regress_83_degenerate_vehicle_pocket():
    """The #83 mechanism (a degenerate pocket AROUND THE VEHICLE re-roots the BFS
    off a much larger separate FREE component) must behave exactly as before the
    #197 fallback was added -- same repro shape as
    test_disconnection_repro_reroot_reaches_frontiers, with the #197 diagnostics
    asserted too so a future change can't silently swap which mechanism did the
    rescuing."""
    rows = [
        "      ......",
        "      ......",
        "      ......",
        ".     ......",
        "      ......",
        "      ......",
    ]
    grid = OccupancyGrid(cell_m=0.1)
    grid.integrate_patch(patch_from_ascii(rows, cell_m=0.1))
    vehicle_xy = (0.05, 0.25)

    frontiers = detect_frontiers(grid, vehicle_xy, min_cluster_size=1)
    info = last_call_info()
    # The #83 re-root is what reconnects the big block -- unchanged from before.
    assert info["bfs_reroot"] is True
    assert info["pocket_cells"] == 1
    assert info["largest_component_cells"] == 36
    assert frontiers
    big_block_frontier = _find_frontier_near(frontiers, grid.cell_to_world(2, 9), tol=0.6)
    assert big_block_frontier is not None
    assert big_block_frontier.path_distance < UNREACHABLE_PD
    assert not _frontier_unreachable(big_block_frontier)
