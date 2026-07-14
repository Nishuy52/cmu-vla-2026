"""Free-space-gradient via placement (H11 / IF-F7).

``free_space_via_point`` replaces the fixed ``anchor + (+1.2, 0)`` offset that ignored
walls and could land the "path near X" waypoint inside a wall or on the unreachable side.
The replacement picks a max-clearance passable cell at ~near_thresh from the anchor that
is reachable from the current pose.
"""
from __future__ import annotations

from core.nav.costmap import Costmap
from core.nav.occupancy import OccupancyGrid
from core.nav.planner import free_space_via_point
from tests.nav.helpers import patch_from_ascii


def _grid_with_vertical_wall(n=20, wall_col=10) -> OccupancyGrid:
    """An n x n open room with a solid vertical wall at ``wall_col`` (splits left/right)."""
    rows = ["." * n for _ in range(n)]
    for i in range(n):
        r = list(rows[i])
        r[wall_col] = "#"
        rows[i] = "".join(r)
    grid = OccupancyGrid(cell_m=0.1)
    grid.integrate_patch(patch_from_ascii(rows, cell_m=0.1))
    return grid


def test_via_lands_on_reachable_side_of_wall_adjacent_anchor():
    """Anchor sits just LEFT of a vertical wall; the vehicle is in the left region. The via
    must land on the LEFT (reachable) side, never behind the wall (IF-F7)."""
    grid = _grid_with_vertical_wall(n=20, wall_col=10)
    cm = Costmap(grid, vehicle_radius_m=0.0)
    anchor = grid.cell_to_world(10, 9)  # one cell left of the wall
    start = grid.cell_to_world(10, 2)  # left region
    wall_x = grid.cell_to_world(10, 10)[0]

    via = free_space_via_point(cm, anchor, start, near_thresh_m=0.4)

    assert via is not None
    assert via[0] < wall_x, "via placed behind the wall (unreachable side)"
    # And it is a passable cell (in free space), not inside an obstacle.
    r, c = grid.world_to_cell(*via)
    assert cm.passable(r, c), "via must land in free space"


def test_via_prefers_max_clearance_open_cell():
    """Between two near-threshold candidates, the one with more clearance (further from the
    wall) wins (IF-F7 free-space gradient)."""
    grid = _grid_with_vertical_wall(n=30, wall_col=20)
    cm = Costmap(grid, vehicle_radius_m=0.0)
    # Anchor well left of the wall, in open space: the via should be a comfortably-open
    # cell, and (with no obstacle nearby on the open side) still reachable from start.
    anchor = grid.cell_to_world(15, 8)
    start = grid.cell_to_world(15, 2)
    via = free_space_via_point(cm, anchor, start, near_thresh_m=0.5)
    assert via is not None
    r, c = grid.world_to_cell(*via)
    assert cm.passable(r, c)
    # The via sits at ~near_thresh from the anchor (ring placement).
    d = ((via[0] - anchor[0]) ** 2 + (via[1] - anchor[1]) ** 2) ** 0.5
    assert 0.2 <= d <= 0.9, f"via at {d:.2f} m is outside the near-threshold ring"


def test_via_reachable_from_start_across_a_wall_gap():
    """The via must be reachable from the CURRENT pose, not merely near the anchor: with a
    wall between the start region and the anchor's far side, the chosen via is the one the
    vehicle can actually reach from start (IF-F7 — nearest != reachable)."""
    grid = _grid_with_vertical_wall(n=24, wall_col=12)
    cm = Costmap(grid, vehicle_radius_m=0.0)
    anchor = grid.cell_to_world(12, 11)  # left of the wall
    start = grid.cell_to_world(12, 3)  # left region (same side as anchor's left face)
    wall_x = grid.cell_to_world(12, 12)[0]

    via = free_space_via_point(cm, anchor, start, near_thresh_m=0.4)
    assert via is not None
    # Reachable side == left of the wall.
    assert via[0] < wall_x
    r, c = grid.world_to_cell(*via)
    assert cm.passable(r, c)
