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
    stamped unreachable, and the vehicle-rooted distances are untouched."""
    grid = _direct_grid(shape=(40, 40))
    grid.state[0:5, 0:10] = FREE  # exactly DEGENERATE_POCKET_CELLS (50) cells
    grid.state[10:25, 0:20] = FREE  # a separate 300-cell component (gap rows 5-9)
    vehicle_xy = grid.cell_to_world(2, 5)  # inside the 50-cell pocket

    frontiers = detect_frontiers(grid, vehicle_xy, min_cluster_size=1)
    info = last_call_info()
    assert info["pocket_cells"] == DEGENERATE_POCKET_CELLS
    assert info["bfs_reroot"] is False
    # Threshold is a strict '<': at-threshold skips the (expensive) full-grid
    # component scan entirely.
    assert info["largest_component_cells"] is None
    # The far, disconnected component's frontier(s) are still stamped unreachable.
    assert any(f.path_distance >= 1e6 for f in frontiers)


def test_degenerate_ratio_boundary_insufficient_ratio_no_reroot():
    """Pocket small enough to qualify, but the largest other component is under
    DEGENERATE_POCKET_RATIO x its size -- must NOT re-root."""
    grid = _direct_grid(shape=(40, 40))
    grid.state[0:2, 0:5] = FREE  # 10-cell vehicle pocket
    grid.state[10:17, 0:5] = FREE  # a separate 35-cell component (< 4x10 = 40)
    vehicle_xy = grid.cell_to_world(0, 2)

    frontiers = detect_frontiers(grid, vehicle_xy, min_cluster_size=1)
    info = last_call_info()
    assert info["pocket_cells"] == 10
    assert info["largest_component_cells"] == 35
    assert 35 < DEGENERATE_POCKET_RATIO * 10
    assert info["bfs_reroot"] is False
    assert any(f.path_distance >= 1e6 for f in frontiers)
