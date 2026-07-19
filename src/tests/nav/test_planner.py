"""A*: around obstacles, through UNKNOWN at penalty, corridor pinch threading."""
from __future__ import annotations

import numpy as np

from core.nav.costmap import Costmap
from core.nav.occupancy import OccupancyGrid
from core.nav.planner import (
    PINCH_CORRIDOR_HALF_W_M,
    PINCH_DISC_M,
    UNKNOWN_COST_MULT,
    astar,
    path_crosses_gate,
    plan_through,
    segments_cross,
)
from tests.nav.helpers import patch_from_ascii


def _grid_from(rows):
    grid = OccupancyGrid(cell_m=0.1)
    grid.integrate_patch(patch_from_ascii(rows, cell_m=0.1))
    return grid


def test_astar_straight_line_free():
    grid = _grid_from(["." * 20 for _ in range(3)])
    cm = Costmap(grid, vehicle_radius_m=0.0)
    path = astar(cm, (0.05, 0.15), (1.85, 0.15))
    assert path is not None
    assert path[0] != path[-1]
    # Ends near the goal cell.
    assert abs(path[-1][0] - 1.85) < 0.2


def test_astar_routes_around_obstacle_wall():
    # A vertical wall with a gap forces a detour.
    rows = [
        "....#.....",
        "....#.....",
        "....#.....",
        "..........",  # gap on the bottom row
    ]
    grid = _grid_from(rows)
    cm = Costmap(grid, vehicle_radius_m=0.0)
    path = astar(cm, (0.05, 0.35), (0.85, 0.35))
    assert path is not None
    # Path must dip down to the gap row (y ~ 0.05) to get around the wall.
    ys = [p[1] for p in path]
    assert min(ys) < 0.15


def test_astar_returns_none_when_blocked():
    # A FREE room fully ringed by OBSTACLE, with the goal walled inside a separate
    # sealed FREE cell -> no route (UNKNOWN detour also sealed off by the ring).
    rows = [
        "#######",
        "#..#..#",
        "#..#..#",
        "#..#..#",
        "#######",
    ]
    grid = _grid_from(rows)
    cm = Costmap(grid, vehicle_radius_m=0.0)
    # Start in the left chamber, goal in the right chamber, split by a sealed wall.
    path = astar(cm, (0.15, 0.25), (0.55, 0.25))
    assert path is None


def test_astar_traverses_unknown_at_penalty():
    """Direct route crosses UNKNOWN; a longer all-FREE detour exists. With a 3x
    UNKNOWN penalty the planner should prefer the FREE detour when it's cheaper."""
    # Build a grid: FREE ring around an UNKNOWN gap. Row layout:
    #   cols 0..8. Middle column 4 is UNKNOWN (no points) on the direct row.
    rows = [
        ".........",
        "....X....",  # placeholder; we'll blank the X to unknown
        ".........",
    ]
    # Replace 'X' with a space so that cell stays UNKNOWN.
    rows = [r.replace("X", " ") for r in rows]
    grid = _grid_from(rows)
    cm = Costmap(grid, vehicle_radius_m=0.0)
    # The unknown cell is at (col 4, row 1) -> world ~ (0.45, 0.15).
    ur, uc = grid.world_to_cell(0.45, 0.15)
    assert cm.is_unknown(ur, uc)
    # Plan across it; UNKNOWN is traversable so a path exists.
    path = astar(cm, (0.05, 0.15), (0.85, 0.15))
    assert path is not None
    # With the FREE rows above/below available, the min-cost path should avoid the
    # UNKNOWN cell (detour is cheaper than 3x penalty for a straight crossing).
    crossed_unknown = any(cm.is_unknown(*grid.world_to_cell(x, y)) for x, y in path)
    assert not crossed_unknown


def test_unknown_only_route_still_found():
    grid = OccupancyGrid(cell_m=0.1)
    # A thin FREE start and FREE goal with UNKNOWN in between (single row).
    grid.integrate_patch(patch_from_ascii(["."], cell_m=0.1, origin_x=0.0))
    grid.integrate_patch(patch_from_ascii(["."], cell_m=0.1, origin_x=1.0))
    cm = Costmap(grid, vehicle_radius_m=0.0)
    path = astar(cm, (0.05, 0.05), (1.05, 0.05))
    # Must route through UNKNOWN cells between the two FREE specks.
    assert path is not None


def test_segments_cross_basic():
    assert segments_cross((0, 0), (2, 0), (1, -1), (1, 1))
    assert not segments_cross((0, 0), (2, 0), (0, 1), (2, 1))


def test_path_crosses_gate_detection():
    path = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)]
    gate = ((1.0, -1.0), (1.0, 1.0))
    assert path_crosses_gate(path, gate)
    gate_off = ((5.0, -1.0), (5.0, 1.0))
    assert not path_crosses_gate(path, gate_off)


def test_corridor_pinch_forces_threading_through_gap():
    """Two obstacle blobs with a narrow gap; a naive plan might skirt around, but
    plan_through must produce a path that crosses the gate between them."""
    # 20-wide grid, 12 tall. Put two obstacle pillars leaving a 1-cell gap at col 10.
    rows = []
    for i in range(12):
        line = list("." * 20)
        if 3 <= i <= 8:
            # obstacle columns 8-9 and 11-12, gap at col 10
            for c in (8, 9, 11, 12):
                line[c] = "#"
        rows.append("".join(line))
    grid = _grid_from(rows)
    cm = Costmap(grid, vehicle_radius_m=0.0)
    # Gate spans the gap at col 10 (world x ~1.05), from below to above the pillars.
    gate = ((1.05, 0.25), (1.05, 0.95))
    start = (1.05, 0.05)   # below the gap
    goal = (1.05, 1.15)    # above the gap
    legs = [("corridor_between", gate), ("goto", goal)]
    path = plan_through(cm, start, legs)
    assert path is not None
    assert path_crosses_gate(path, gate)


def test_plan_through_goto_only():
    grid = _grid_from(["." * 20 for _ in range(3)])
    cm = Costmap(grid, vehicle_radius_m=0.0)
    path = plan_through(cm, (0.05, 0.15), [("goto", (1.85, 0.15))])
    assert path is not None
    assert abs(path[-1][0] - 1.85) < 0.2


# --------------------------------------------------------------------------- calibration seams
# nav.unknown_cost_mult / nav.pinch_disc_m / nav.pinch_corridor_half_w_m (docs/calibration.md).
# astar / _pinch_costmap / plan_through now accept these as keyword overrides; the module
# constants remain the defaults so unwired call sites are unaffected.


def test_astar_unknown_cost_mult_override_changes_route_preference():
    """Same grid as test_astar_traverses_unknown_at_penalty, but with the UNKNOWN
    penalty overridden near-zero: the direct straight-through-UNKNOWN route should now
    be cheaper than the FREE detour, so the planner crosses the UNKNOWN cell."""
    rows = [
        ".........",
        "....X....",
        ".........",
    ]
    rows = [r.replace("X", " ") for r in rows]
    grid = _grid_from(rows)
    cm = Costmap(grid, vehicle_radius_m=0.0)
    path = astar(cm, (0.05, 0.15), (0.85, 0.15), unknown_cost_mult=0.01)
    assert path is not None
    crossed_unknown = any(cm.is_unknown(*grid.world_to_cell(x, y)) for x, y in path)
    assert crossed_unknown


def test_astar_default_unknown_cost_mult_matches_module_constant():
    """Omitting the keyword reproduces UNKNOWN_COST_MULT (no behaviour change for
    unwired call sites)."""
    rows = [
        ".........",
        "....X....",
        ".........",
    ]
    rows = [r.replace("X", " ") for r in rows]
    grid = _grid_from(rows)
    cm = Costmap(grid, vehicle_radius_m=0.0)
    default_path = astar(cm, (0.05, 0.15), (0.85, 0.15))
    explicit_path = astar(
        cm, (0.05, 0.15), (0.85, 0.15), unknown_cost_mult=UNKNOWN_COST_MULT
    )
    assert default_path == explicit_path


def _pinch_gap_grid():
    rows = []
    for i in range(12):
        line = list("." * 20)
        if 3 <= i <= 8:
            for c in (8, 9, 11, 12):
                line[c] = "#"
        rows.append("".join(line))
    grid = _grid_from(rows)
    return Costmap(grid, vehicle_radius_m=0.0)


def test_pinch_costmap_disc_m_override_scales_blocked_region():
    from core.nav.planner import _pinch_costmap

    cm = _pinch_gap_grid()
    gate = ((1.05, 0.25), (1.05, 0.95))
    small = _pinch_costmap(cm, gate, pinch_disc_m=0.3)
    large = _pinch_costmap(cm, gate, pinch_disc_m=PINCH_DISC_M)
    # A bigger disc overlays (and therefore can block) at least as much of the grid as
    # a smaller one — the seam actually changes the overlay geometry.
    assert large.capsule_blocked.sum() >= small.capsule_blocked.sum()
    assert large.capsule_blocked.sum() > cm.capsule_blocked.sum()


def test_pinch_costmap_corridor_half_w_m_override_narrows_gap():
    from core.nav.planner import _pinch_costmap

    cm = _pinch_gap_grid()
    gate = ((1.05, 0.25), (1.05, 0.95))
    narrow = _pinch_costmap(cm, gate, pinch_corridor_half_w_m=0.05)
    wide = _pinch_costmap(cm, gate, pinch_corridor_half_w_m=2.0)
    # Widening the allowed corridor blocks strictly less (the wide corridor swallows the
    # whole disc, leaving nothing extra blocked outside it).
    assert narrow.capsule_blocked.sum() >= wide.capsule_blocked.sum()


def test_plan_through_pinch_seams_reproduce_default_when_omitted():
    cm = _pinch_gap_grid()
    gate = ((1.05, 0.25), (1.05, 0.95))
    start = (1.05, 0.05)
    goal = (1.05, 1.15)
    legs = [("corridor_between", gate), ("goto", goal)]
    default_path = plan_through(cm, start, legs)
    explicit_path = plan_through(
        cm,
        start,
        legs,
        pinch_disc_m=PINCH_DISC_M,
        pinch_corridor_half_w_m=PINCH_CORRIDOR_HALF_W_M,
    )
    assert default_path == explicit_path
    assert path_crosses_gate(default_path, gate)


def test_plan_through_open_space_unaffected_by_pinch_seam_overrides():
    """A leg with no gate geometry (goto/via_near) must be identical regardless of the
    pinch overlay knobs — the overlay only ever applies inside a corridor_between leg's
    fallback branch."""
    grid = _grid_from(["." * 20 for _ in range(6)])
    cm = Costmap(grid, vehicle_radius_m=0.0)
    legs = [("goto", (1.85, 0.25))]
    default_path = plan_through(cm, (0.05, 0.25), legs)
    overridden_path = plan_through(
        cm,
        (0.05, 0.25),
        legs,
        pinch_disc_m=50.0,
        pinch_corridor_half_w_m=0.01,
        unknown_cost_mult=999.0,
    )
    assert default_path == overridden_path


def test_plan_through_unreachable_leg_returns_none():
    rows = [
        "#######",
        "#..#..#",
        "#..#..#",
        "#..#..#",
        "#######",
    ]
    grid = _grid_from(rows)
    cm = Costmap(grid, vehicle_radius_m=0.0)
    path = plan_through(cm, (0.15, 0.25), [("goto", (0.55, 0.25))])
    assert path is None
