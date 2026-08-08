"""Costmap: inflation, hard capsule stamps (never relax), nearest_reachable_point."""
from __future__ import annotations

import numpy as np

from core.nav.costmap import BASE_OBSTACLE_CLEARANCE_M, Costmap
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


# --------------------------------------------------------------------------- issue #207
# BASE_OBSTACLE_CLEARANCE_M / obstacle_clearance_m / nearest_clear_point: the base
# stack's own waypointConverter snap-clearance rule (obstacleDisThre = 0.75 m,
# upstream/.../waypoint_converter/src/waypointConverter.cpp:225-233), confirmed by
# reading the upstream source directly.


def _corridor_grid(gap_cells: int = 9, wall_cells: int = 10, n_rows: int = 30, cell_m: float = 0.1):
    """A vertical corridor spanning every row: obstacle columns on both sides, a free
    gap of ``gap_cells`` columns between them. Default ``gap_cells=9`` at ``cell_m=0.1``
    is 0.9 m -- the narrowest doorway FRAME width measured across the challenge scene set
    (chinese_room door frame, size [0.1937..., 0.8995..., 2.034...] --
    data/vla3d/Unity/chinese_room/chinese_room_objects.json -- the 0.8995 m dimension is
    the opening span, the 0.1937 m dimension the wall/frame thickness). Every measured
    door/door-frame object across all 20 scene folders has its wider horizontal dimension
    >= this value, so 0.9 m is the tightest real passage in the data, not a contrived
    extreme.
    """
    n_cols = 2 * wall_cells + gap_cells
    row = "#" * wall_cells + "." * gap_cells + "#" * wall_cells
    grid = OccupancyGrid(cell_m=cell_m)
    grid.integrate_patch(patch_from_ascii([row for _ in range(n_rows)], cell_m=cell_m))
    return grid, n_cols, n_rows


def test_narrow_doorway_gap_stays_traversable_under_planning_inflation():
    """A 0.9 m corridor (the narrowest measured doorway width in the scene set) stays
    PASSABLE under the production 0.4 m planning inflation (VEHICLE_RADIUS_M) -- #207's
    fix must not seal a real doorway just because the base's own 0.75 m snap-clearance
    rule can never be satisfied inside it (see the next test)."""
    grid, n_cols, n_rows = _corridor_grid()
    cm = Costmap(grid, vehicle_radius_m=0.4)
    mid_row, gap_col = n_rows // 2, n_cols // 2
    assert cm.passable(mid_row, gap_col)


def test_narrow_doorway_center_never_clears_base_threshold():
    """At the centre of that same 0.9 m gap, clearance from the nearest jamb is ~0.5 m
    (half the gap width, rounded to the cell lattice) -- below BASE_OBSTACLE_CLEARANCE_M
    (0.75 m) -- and NO point anywhere in the (grid-spanning) corridor does better, since
    every row repeats the identical gap. This is a property of the base's OWN rule, not
    something #207 introduces: ``nearest_clear_point`` must recognise there is nothing to
    nudge to and return None, rather than hunt forever or walk the crumb somewhere
    unrelated."""
    grid, n_cols, n_rows = _corridor_grid()
    cm = Costmap(grid, vehicle_radius_m=0.4)
    mid_row, gap_col = n_rows // 2, n_cols // 2
    x, y = grid.cell_to_world(mid_row, gap_col)
    clearance = cm.obstacle_clearance_m(x, y)
    assert clearance < BASE_OBSTACLE_CLEARANCE_M
    assert abs(clearance - 0.5) < 1e-9  # 5 cells * 0.1 m, exact on this lattice
    assert cm.nearest_clear_point(x, y) is None


def test_nearest_clear_point_finds_alternative_away_from_isolated_obstacle():
    """A single small obstacle (furniture edge, not a doorway) DOES have room around it:
    ``nearest_clear_point`` must find a nearby point that clears
    BASE_OBSTACLE_CLEARANCE_M, unlike the grid-spanning doorway case above."""
    rows = ["." * 60 for _ in range(60)]
    rows[30] = "." * 29 + "#" + "." * 30  # one obstacle cell near the room centre
    grid = OccupancyGrid(cell_m=0.1)
    grid.integrate_patch(patch_from_ascii(rows, cell_m=0.1))
    cm = Costmap(grid, vehicle_radius_m=0.0)
    x, y = grid.cell_to_world(30, 30)  # adjacent to the obstacle cell
    assert cm.obstacle_clearance_m(x, y) < BASE_OBSTACLE_CLEARANCE_M
    nudged = cm.nearest_clear_point(x, y)
    assert nudged is not None
    assert cm.obstacle_clearance_m(*nudged) >= BASE_OBSTACLE_CLEARANCE_M


def test_obstacle_clearance_far_from_everything_reports_search_radius():
    grid = _open_grid(n=60)
    cm = Costmap(grid, vehicle_radius_m=0.0)
    x, y = grid.cell_to_world(30, 30)
    assert cm.obstacle_clearance_m(x, y, search_radius_m=1.0) == 1.0
