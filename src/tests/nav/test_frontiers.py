"""Frontier detection, clustering, min-size filter, scoring + affinity injection."""
from __future__ import annotations

import numpy as np

from core.nav.frontiers import (
    MIN_CLUSTER_SIZE,
    detect_frontiers,
    frontier_mask,
)
from core.nav.occupancy import FREE, UNKNOWN, OBSTACLE, OccupancyGrid
from tests.nav.helpers import patch_from_ascii


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
