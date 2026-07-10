"""Occupancy grid: classification, max-intensity retention, growth, observed mask."""
from __future__ import annotations

import numpy as np

from core.nav.occupancy import FREE, OBSTACLE, UNKNOWN, OccupancyGrid
from tests.nav.helpers import make_points, patch_from_ascii


def test_classification_free_vs_obstacle():
    grid = OccupancyGrid(cell_m=0.1)
    # (0.05,0.05) low intensity -> FREE; (0.35,0.05) high -> OBSTACLE.
    grid.integrate_patch(make_points([(0.05, 0.05, 0.02), (0.35, 0.05, 0.5)]))
    assert grid.state[grid.world_to_cell(0.05, 0.05)] == FREE
    assert grid.state[grid.world_to_cell(0.35, 0.05)] == OBSTACLE


def test_threshold_boundary_is_free_below_cutoff():
    grid = OccupancyGrid(cell_m=0.1, free_max=0.15)
    grid.integrate_patch(make_points([(0.05, 0.05, 0.149), (0.15, 0.05, 0.15)]))
    assert grid.is_free(*grid.world_to_cell(0.05, 0.05))
    # Exactly at cutoff is NOT < free_max -> obstacle.
    assert grid.is_obstacle(*grid.world_to_cell(0.15, 0.05))


def test_unobserved_cells_are_unknown():
    grid = OccupancyGrid(cell_m=0.1)
    grid.integrate_patch(make_points([(0.05, 0.05, 0.02)]))
    # A far cell that got no point is UNKNOWN.
    assert grid.is_unknown(*grid.world_to_cell(5.0, 5.0))


def test_max_intensity_retained_obstacle_wins():
    grid = OccupancyGrid(cell_m=0.1)
    # First a free reading, then an obstacle reading in the SAME cell.
    grid.integrate_patch(make_points([(0.05, 0.05, 0.02)]))
    assert grid.is_free(*grid.world_to_cell(0.05, 0.05))
    grid.integrate_patch(make_points([(0.05, 0.05, 0.6)]))
    r, c = grid.world_to_cell(0.05, 0.05)
    assert grid.state[r, c] == OBSTACLE
    assert grid.intensity[r, c] >= 0.6


def test_obstacle_not_overwritten_by_later_free():
    grid = OccupancyGrid(cell_m=0.1)
    grid.integrate_patch(make_points([(0.05, 0.05, 0.6)]))
    grid.integrate_patch(make_points([(0.05, 0.05, 0.01)]))  # later free reading
    # Max-intensity retention keeps it an obstacle.
    assert grid.is_obstacle(*grid.world_to_cell(0.05, 0.05))


def test_max_within_single_patch():
    grid = OccupancyGrid(cell_m=0.1)
    # Two points in the same cell within one patch; the higher intensity decides.
    grid.integrate_patch(make_points([(0.05, 0.05, 0.02), (0.06, 0.06, 0.7)]))
    assert grid.is_obstacle(*grid.world_to_cell(0.05, 0.05))


def test_grid_grows_to_negative_coords():
    grid = OccupancyGrid(cell_m=0.1)
    grid.integrate_patch(make_points([(0.05, 0.05, 0.02)]))
    h0, w0 = grid.shape
    # A point at negative world coords forces origin shift + top/left padding.
    grid.integrate_patch(make_points([(-2.0, -2.0, 0.02)]))
    assert grid.is_free(*grid.world_to_cell(-2.0, -2.0))
    # Original cell still classified correctly after growth.
    assert grid.is_free(*grid.world_to_cell(0.05, 0.05))
    assert grid.shape[0] >= h0 and grid.shape[1] >= w0


def test_grid_growth_preserves_world_lattice():
    grid = OccupancyGrid(cell_m=0.1)
    grid.integrate_patch(make_points([(1.05, 1.05, 0.6)]))
    before = grid.cell_to_world(*grid.world_to_cell(1.05, 1.05))
    grid.integrate_patch(make_points([(-5.0, -5.0, 0.02)]))
    after = grid.cell_to_world(*grid.world_to_cell(1.05, 1.05))
    # Cell centre for the same world point is stable across growth.
    assert np.allclose(before, after, atol=1e-6)
    assert grid.is_obstacle(*grid.world_to_cell(1.05, 1.05))


def test_mark_pose_carves_free():
    grid = OccupancyGrid(cell_m=0.1)
    # Put an obstacle reading where the vehicle is, then carve it FREE.
    grid.integrate_patch(make_points([(0.05, 0.05, 0.9)]))
    assert grid.is_obstacle(*grid.world_to_cell(0.05, 0.05))
    grid.mark_pose(0.05, 0.05)
    assert grid.is_free(*grid.world_to_cell(0.05, 0.05))


def test_mark_pose_sets_observed_within_radius():
    grid = OccupancyGrid(cell_m=0.1, observe_radius_m=1.0)
    grid.mark_pose(0.0, 0.0)
    r_in, c_in = grid.world_to_cell(0.5, 0.0)
    r_out, c_out = grid.world_to_cell(3.0, 0.0)
    assert grid.observed[r_in, c_in]
    assert grid.in_bounds(r_out, c_out) is False or not grid.observed[r_out, c_out]


def test_empty_patch_is_noop():
    grid = OccupancyGrid(cell_m=0.1)
    grid.integrate_patch(make_points([]))
    assert grid.state.sum() == 0  # all UNKNOWN


def test_ascii_patch_layout_orientation():
    # Top row is higher y; '#' top-left should map to obstacle at high y, low x.
    p = patch_from_ascii(["#.", ".."], cell_m=0.1, origin_x=0.0, origin_y=0.0)
    grid = OccupancyGrid(cell_m=0.1)
    grid.integrate_patch(p)
    # Top-left char -> (x low, y high): cell (row=1, col=0).
    assert grid.state[1, 0] == OBSTACLE
    assert grid.state[0, 0] == FREE
