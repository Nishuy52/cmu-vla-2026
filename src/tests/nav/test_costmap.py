"""Costmap: inflation, hard capsule stamps (never relax), nearest_reachable_point."""
from __future__ import annotations

import numpy as np

from core.nav.costmap import Costmap
from core.nav.occupancy import OccupancyGrid
from tests.nav.helpers import patch_from_ascii


def _open_grid(n=40):
    """n x n all-FREE grid."""
    grid = OccupancyGrid(cell_m=0.1)
    grid.integrate_patch(patch_from_ascii(["." * n for _ in range(n)], cell_m=0.1))
    return grid


def test_inflation_blocks_around_obstacle():
    grid = OccupancyGrid(cell_m=0.1)
    rows = ["." * 11 for _ in range(11)]
    rows[5] = "....."+"#"+"....."  # single obstacle at centre (col 5, row 5)
    grid.integrate_patch(patch_from_ascii(rows, cell_m=0.1))
    cm = Costmap(grid, vehicle_radius_m=0.4)
    # Obstacle cell blocked, and cells within ~0.4 m inflated too.
    orow, ocol = grid.world_to_cell(0.55, 0.55)
    assert cm.blocked(orow, ocol)
    # A cell 0.2 m away should be inflated (0.4 m radius covers it).
    nr, nc = grid.world_to_cell(0.75, 0.55)
    assert cm.blocked(nr, nc)


def test_capsule_stamp_blocks_segment():
    grid = _open_grid()
    cm = Costmap(grid, vehicle_radius_m=0.4)
    # Stamp a horizontal capsule across the middle.
    cm.stamp_capsule(((0.5, 2.0), (3.5, 2.0)), radius_m=0.2)
    r, c = grid.world_to_cell(2.0, 2.0)
    assert cm.blocked(r, c)
    # Away from the capsule stays passable.
    r2, c2 = grid.world_to_cell(2.0, 0.3)
    assert cm.passable(r2, c2)


def test_capsule_is_cumulative_never_shrinks():
    grid = _open_grid()
    cm = Costmap(grid)
    cm.stamp_capsule(((0.5, 1.0), (3.5, 1.0)), radius_m=0.2)
    blocked_after_first = cm.capsule_blocked.sum()
    # Stamp a second, disjoint capsule; the first region must remain blocked.
    cm.stamp_capsule(((0.5, 3.0), (3.5, 3.0)), radius_m=0.2)
    r, c = grid.world_to_cell(2.0, 1.0)
    assert cm.blocked(r, c)  # first capsule still there
    assert cm.capsule_blocked.sum() > blocked_after_first  # only grew


def test_clone_does_not_mutate_original_capsules():
    grid = _open_grid()
    cm = Costmap(grid)
    cm.stamp_capsule(((0.5, 1.0), (3.5, 1.0)), radius_m=0.2)
    before = cm.capsule_blocked.copy()
    clone = cm.clone()
    clone.stamp_capsule(((0.5, 3.0), (3.5, 3.0)), radius_m=0.2)
    # Original unchanged by the clone's stamp.
    assert np.array_equal(cm.capsule_blocked, before)
    assert clone.capsule_blocked.sum() > before.sum()


def test_nearest_reachable_point_when_goal_walled_off():
    grid = _open_grid()
    cm = Costmap(grid)
    # Wall the whole grid in two by a capsule spanning the full width at y=2.0.
    cm.stamp_capsule(((-1.0, 2.0), (5.0, 2.0)), radius_m=0.3)
    start = (2.0, 0.5)   # below the wall
    goal = (2.0, 3.5)    # above the wall -> unreachable with capsule
    from core.nav.planner import astar

    assert astar(cm, start, goal) is None  # truly unreachable
    nrp = cm.nearest_reachable_point(goal, start)
    # The least-bad point is on the START side of the wall (y < 2.0), not the goal.
    assert nrp[1] < 2.0
    # And it's reachable from start.
    assert astar(cm, start, nrp) is not None


def test_nearest_reachable_does_not_relax_capsule():
    grid = _open_grid()
    cm = Costmap(grid)
    cm.stamp_capsule(((-1.0, 2.0), (5.0, 2.0)), radius_m=0.3)
    blocked_before = cm.capsule_blocked.copy()
    cm.nearest_reachable_point((2.0, 3.5), (2.0, 0.5))
    # The capsule mask is byte-for-byte unchanged — capsules NEVER relax.
    assert np.array_equal(cm.capsule_blocked, blocked_before)
