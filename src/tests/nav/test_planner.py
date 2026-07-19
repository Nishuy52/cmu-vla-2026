"""A*: around obstacles, through UNKNOWN at penalty, corridor pinch threading."""
from __future__ import annotations

import numpy as np

from core.nav.costmap import Costmap
from core.nav.occupancy import OccupancyGrid
from core.nav.planner import (
    MAX_PINCH_RELAX_ROUNDS,
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


def _pinch_gap_grid_with_vehicle_radius(vehicle_radius_m: float):
    rows = []
    for i in range(12):
        line = list("." * 20)
        if 3 <= i <= 8:
            for c in (8, 9, 11, 12):
                line[c] = "#"
        rows.append("".join(line))
    grid = _grid_from(rows)
    return Costmap(grid, vehicle_radius_m=vehicle_radius_m)


def _pinch_gap_grid_with_midpoint_blocker():
    """Same pillar gap as `_pinch_gap_grid`, plus a genuine solid obstacle straddling
    the gate's raw midpoint (issue #63's hotel_room_2 shape: a real third object, not
    an architectural over-stamp, sitting at the gate line) — blocks column 10 (the
    gap) at gy in {4, 5, 6} (y in roughly [0.4, 0.7]), leaving clear column-10 cells
    on either side of the block, still within the gate's own [0.25, 0.95] span."""
    rows = []
    for i in range(12):
        line = list("." * 20)
        if 3 <= i <= 8:
            for c in (8, 9, 11, 12):
                line[c] = "#"
        if i in (5, 6, 7):  # gy in {6, 5, 4} respectively -> col 10 blocked there too
            line[10] = "#"
        rows.append("".join(line))
    grid = _grid_from(rows)
    return Costmap(grid, vehicle_radius_m=0.0)


def test_plan_through_nudges_off_genuine_obstacle_at_gate_midpoint():
    """issue #63: the raw gate midpoint sits inside a genuine solid obstacle cell (not
    an anchor's own footprint, not inflation) — `plan_through` must nudge its via-point
    along the gate's own axis to a clear point on the same verified gate line and still
    thread it, rather than either failing outright or (pre-#63) A*-ing straight at a
    blocked cell and depending on incidental grid-snap luck."""
    cm = _pinch_gap_grid_with_midpoint_blocker()
    gate = ((1.05, 0.25), (1.05, 0.95))
    mid_r, mid_c = cm.grid.world_to_cell(1.05, 0.6)
    assert cm.blocked(mid_r, mid_c), "test setup: raw gate midpoint must be blocked"
    start = (1.05, 0.05)
    goal = (1.05, 1.15)
    legs = [("corridor_between", gate), ("goto", goal)]
    path = plan_through(cm, start, legs)
    assert path is not None, "must thread by nudging off the genuine obstacle"
    assert path_crosses_gate(path, gate)


def test_usable_gate_point_matches_between_toolbox_and_planner():
    """issue #63: the scoring geometry (`toolbox.corridor_gate`) and the planner
    (`plan_through`'s gate-target computation) must compute the SAME usable crossing
    point for an identical gate + identical blocking geometry, since both call the one
    shared `core.geometry.primitives.usable_gate_point` helper — this is the guarantee
    that prevents the #51 scoring/planning mismatch class from reopening."""
    from core.geometry.primitives import usable_gate_point

    p0 = np.array([1.05, 0.25])
    p1 = np.array([1.05, 0.95])
    mid = (p0 + p1) / 2.0

    def blocked(pt):
        return 0.4 <= pt[1] <= 0.7  # matches the planner-grid blocker's y-band

    nudged_generic = usable_gate_point(p0, p1, mid, blocked)

    cm = _pinch_gap_grid_with_midpoint_blocker()
    from core.nav.planner import _raw_obstacle_blocked_xy

    nudged_planner = usable_gate_point(
        p0, p1, mid, lambda pt: _raw_obstacle_blocked_xy(cm, pt)
    )
    # Both should have moved off the raw midpoint, to the same side (nearest clear
    # point found by the identical sweep order) — not necessarily bit-identical (the
    # costmap quantizes to cell centres) but on the same side and clear in both.
    assert not np.allclose(nudged_generic, mid)
    assert not np.allclose(nudged_planner, mid)
    assert (nudged_generic[1] - mid[1]) * (nudged_planner[1] - mid[1]) > 0, (
        "toolbox-style and planner-style nudges must land on the same side of the gate"
    )


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


# --------------------------------------------------------------------------- issue #51


def test_pinch_costmap_clears_inflation_only_cells_inside_corridor_band():
    """A real (GT-verified) gate narrower than 2*vehicle_radius gets sealed end-to-end
    by the vehicle's OWN inflation margin from both anchor pillars, not by the pillars'
    raw footprint — the pinch overlay used to only ever ADD blocking, so it could never
    open a corridor an inflated vehicle would otherwise seal. The overlay must clear
    inflation-only cells (never a raw obstacle cell) inside the forced corridor band."""
    from core.nav.planner import _pinch_costmap

    rows = []
    for i in range(12):
        line = list("." * 20)
        if 3 <= i <= 8:
            for c in (8, 9, 11, 12):
                line[c] = "#"
        rows.append("".join(line))
    grid = _grid_from(rows)
    # vehicle_radius=0.4 (4 cells) merges the two pillars' inflation across the 1-cell
    # (0.1 m) gap at col 10 -> the un-pinched costmap already has the gate mid blocked.
    cm = Costmap(grid, vehicle_radius_m=0.4)
    gate = ((1.05, 0.25), (1.05, 0.95))
    mid_r, mid_c = grid.world_to_cell(1.05, 0.6)
    assert cm.blocked(mid_r, mid_c), "test setup: gate mid should start inflation-sealed"
    assert not cm.raw_blocked[mid_r, mid_c], "test setup: mid is inflation, not a real obstacle"

    pinch = _pinch_costmap(cm, gate)
    assert not pinch.blocked(mid_r, mid_c), "inflation-only gate mid must be cleared"
    # The pillars themselves (raw obstacle cells) must NEVER be cleared.
    for r, c in [grid.world_to_cell(1.05, 0.35), grid.world_to_cell(1.05, 0.85)]:
        if cm.raw_blocked[r, c]:
            assert pinch.blocked(r, c), "a real obstacle cell must never be un-blocked"


def test_pinch_costmap_never_clears_raw_obstacle_outside_disc():
    """The inflation-clearing is scoped to the forced corridor band; nothing beyond the
    disc is touched (unaffected obstacles far from the gate stay exactly as before)."""
    from core.nav.planner import _pinch_costmap

    rows = []
    for i in range(12):
        line = list("." * 20)
        if 3 <= i <= 8:
            for c in (8, 9, 11, 12):
                line[c] = "#"
        rows.append("".join(line))
    grid = _grid_from(rows)
    cm = Costmap(grid, vehicle_radius_m=0.4)
    gate = ((1.05, 0.25), (1.05, 0.95))
    pinch = _pinch_costmap(cm, gate, pinch_disc_m=0.3)
    # A far-away inflation-halo cell (well outside the 0.3 m disc) must be unchanged.
    far_r, far_c = grid.world_to_cell(1.85, 0.05)
    assert pinch.blocked(far_r, far_c) == cm.blocked(far_r, far_c)


def test_plan_through_tries_pinch_even_when_direct_astar_to_mid_fails_outright():
    """issue #51: when the gate midpoint is unreachable in the UN-pinched costmap
    (direct A* returns None, not just 'reached but missed the gate'), the corridor leg
    used to give up immediately without ever trying the pinch fallback that exists to
    force a path through a tight, verified gate. A narrow real gate sealed by inflation
    (mid itself unreachable pre-pinch) must still be threaded once the pinch clears the
    inflation-only halo."""
    rows = []
    for i in range(12):
        line = list("." * 20)
        if 3 <= i <= 8:
            for c in (8, 9, 11, 12):
                line[c] = "#"
        rows.append("".join(line))
    grid = _grid_from(rows)
    cm = Costmap(grid, vehicle_radius_m=0.4)
    gate = ((1.05, 0.25), (1.05, 0.95))
    start = (1.05, 0.05)
    goal = (1.05, 1.15)
    mid_r, mid_c = grid.world_to_cell(1.05, 0.6)
    # Confirm the direct (un-pinched) target is genuinely unreachable-from-blocked, the
    # precondition this test exercises.
    assert cm.blocked(mid_r, mid_c)
    legs = [("corridor_between", gate), ("goto", goal)]
    path = plan_through(cm, start, legs)
    assert path is not None, "the pinch fallback must still be attempted and succeed"
    assert path_crosses_gate(path, gate)


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


# --------------------------------------------------------------------------- issue #54


def _off_axis_gap_grid():
    """20x24 grid: a horizontal gate (gap at row 10, cols 8-12 open) between two solid
    blocks, plus a large open area to the SIDE of the gate (off the corridor axis) so a
    start point placed there is within pinch_disc_m of the gate but off the forced band."""
    rows = []
    for i in range(24):
        line = list("." * 20)
        if i == 10:
            for c in range(0, 8):
                line[c] = "#"
            for c in range(13, 20):
                line[c] = "#"
        rows.append("".join(line))
    grid = _grid_from(rows)
    return Costmap(grid, vehicle_radius_m=0.0)


def test_pinch_costmap_start_xy_exempts_vehicles_own_neighbourhood_from_added_blocking():
    """issue #54: a start point within pinch_disc_m of the gate but off the forced
    corridor band must never be newly blocked by the overlay — the vehicle is already
    standing there, so the overlay adding a block there can only ever seal a route that
    was reachable before the overlay existed."""
    from core.nav.planner import _pinch_costmap

    cm = _off_axis_gap_grid()
    gate = ((1.05, 0.95), (1.05, 1.05))  # the gap at row 10 (x~1.05), thin band
    start = (0.35, 0.35)  # well off the corridor axis, within pinch_disc_m of the gate
    grid = cm.grid
    sr, sc = grid.world_to_cell(*start)
    assert cm.passable(sr, sc), "test setup: start must be passable pre-overlay"

    no_exempt = _pinch_costmap(cm, gate, pinch_disc_m=1.0)
    assert not no_exempt.passable(sr, sc), (
        "test setup: without start_xy, the overlay newly blocks the start cell"
    )

    exempt = _pinch_costmap(cm, gate, pinch_disc_m=1.0, start_xy=start)
    assert exempt.passable(sr, sc), "start_xy must exempt the vehicle's own cell"


def test_pinch_costmap_start_xy_never_clears_a_real_obstacle():
    """The start-disc exemption only ever prevents ADDED blocking / clears inflation-only
    halo — it must never un-block a real (raw) obstacle cell, even one right next to the
    exempted start."""
    from core.nav.planner import _pinch_costmap

    cm = _pinch_gap_grid()
    gate = ((1.05, 0.25), (1.05, 0.95))
    grid = cm.grid
    start = (0.35, 0.6)  # off-axis, near one of the real pillar cells
    obstacle_r, obstacle_c = grid.world_to_cell(0.85, 0.6)  # pillar column 8
    assert cm.raw_blocked[obstacle_r, obstacle_c], "test setup: must be a real obstacle"

    pinch = _pinch_costmap(cm, gate, start_xy=start)
    assert pinch.blocked(obstacle_r, obstacle_c), "a real obstacle must never be un-blocked"


def test_plan_through_pinch_never_seals_vehicles_own_start_cell():
    """Integration shape of issue #54's home_building_1/home_building_2 traces: the
    start sits off the corridor axis, within pinch_disc_m of the gate, and the gate is
    inflation-sealed end-to-end pre-pinch (the direct attempt cannot thread it). Pre-#54
    the pinch fallback would then ALSO fail (its own overlay newly blocks the start
    cell, off the forced corridor band) and the whole leg would be reported
    unreachable; post-#54 the leg must thread."""
    cm = _pinch_gap_grid_with_vehicle_radius(0.4)
    gate = ((1.05, 0.25), (1.05, 0.95))
    start = (0.35, 0.6)  # off-axis, within pinch_disc_m (3.0 default) of the gate
    goal = (1.05, 1.15)
    grid = cm.grid
    sr, sc = grid.world_to_cell(*start)
    assert cm.passable(sr, sc), "test setup: start must be passable"
    mid_r, mid_c = grid.world_to_cell(1.05, 0.6)
    assert cm.blocked(mid_r, mid_c), "test setup: gate mid sealed by inflation pre-pinch"
    from core.nav.planner import _pinch_costmap

    # Without the exemption the overlay newly blocks the start's OWN cell (verified
    # directly, matching test_pinch_costmap_start_xy_exempts_...  above) — a real
    # vehicle standing there would be unable to even begin the forced-corridor plan.
    pinch_no_exempt = _pinch_costmap(cm, gate)
    assert not pinch_no_exempt.passable(sr, sc), (
        "test setup: without start_xy the overlay must newly block the start cell"
    )

    legs = [("corridor_between", gate), ("goto", goal)]
    path = plan_through(cm, start, legs)
    assert path is not None, "the pinch fallback must thread once start-blocking is fixed"
    assert path_crosses_gate(path, gate)


def test_plan_through_nudge_only_engages_when_direct_route_is_wholly_unreachable():
    """Safety guard (issue #54 regression found during verification): nudging the pinch
    retry's target past the gate is only sound when the leg has never reached the gate
    at all pre-pinch (the direct A* returned None). Nudging when the direct attempt
    already reached the gate (just without crossing) can force a path back THROUGH an
    already-threaded gate on a later replan (the vehicle may legitimately already be on
    the far side, having threaded it earlier before a route rebuild) — reproduced during
    verification as a live oscillation/regression in `tests/heads/test_instruction.py`'s
    `test_corridor_route_threads_gate_with_overridden_planner_seams`. When the direct
    attempt reaches the gate without crossing, the pinch retry's target must stay at the
    exact midpoint (the pre-#54 behaviour) — never nudged."""
    from unittest.mock import patch

    from core.nav import planner as P

    cm = _pinch_gap_grid()
    gate = ((1.05, 0.25), (1.05, 0.95))
    legs = [("corridor_between", gate), ("goto", (1.05, 1.15))]
    start = (1.05, 0.05)

    orig_astar = P.astar

    def _direct_call_returns(first_result):
        """First astar() call (the direct, un-pinched corridor attempt) returns
        `first_result`; every other call (pinch retry, other legs) runs for real."""
        calls = {"n": 0}

        def fake(costmap, s, g, **kw):
            calls["n"] += 1
            if calls["n"] == 1:
                return first_result
            return orig_astar(costmap, s, g, **kw)

        return fake

    with (
        patch.object(P, "_nudge_past_gate", wraps=P._nudge_past_gate) as nudge_spy,
        patch.object(P, "astar", side_effect=_direct_call_returns(None)),
    ):
        # Case A: the direct (first) astar call returns None -> nudge IS used on the
        # pinch retry.
        P.plan_through(cm, start, legs)
        assert nudge_spy.call_count == 1, "direct astar was None -> nudge must be attempted"

    near_miss = [(0.2, 0.6), (0.5, 0.6), (0.9, 0.6)]  # stays on one side, never crosses
    assert not path_crosses_gate(near_miss, gate), "test setup: near_miss must not cross"
    with (
        patch.object(P, "_nudge_past_gate", wraps=P._nudge_past_gate) as nudge_spy,
        patch.object(P, "astar", side_effect=_direct_call_returns(near_miss)),
    ):
        # Case B: the direct (first) astar call reaches the gate but doesn't cross it
        # (any non-None, non-crossing path) -> nudge must NOT be used.
        P.plan_through(cm, start, legs)
        assert nudge_spy.call_count == 0, (
            "direct astar succeeded (even without crossing) -> nudge must not engage"
        )


def test_plan_through_nudge_engages_on_genuine_quantization_near_miss():
    """issue #64: the direct attempt reaches the gate but its snapped final vertex lands
    a hair (< 1 grid cell) short of the line on the SAME side `cur` approached from — a
    grid-quantization near-miss, not an already-threaded gate. `cur` starts well away
    from the gate (>= 1 cell), so the #54 guard's "already resumed at the gate" signal is
    absent. The tightened case-B condition must engage the nudge here (unlike the #54
    guard's far near-miss, which stays > 1 cell out and must NOT engage it)."""
    from unittest.mock import patch

    from core.nav import planner as P

    cm = _pinch_gap_grid()
    gate = ((1.05, 0.25), (1.05, 0.95))
    legs = [("corridor_between", gate), ("goto", (1.05, 1.15))]
    start = (1.05, 0.05)  # >= 1 cell (0.1 m) from the gate segment

    orig_astar = P.astar
    # A near-miss whose LAST vertex sits 0.03 m (< 1 cell) short of the gate line
    # (x=1.05), on the same side `cur` approaches from, never crossing.
    quantized_near_miss = [(1.05, 0.05), (0.99, 0.3), (1.02, 0.6)]
    assert not path_crosses_gate(quantized_near_miss, gate), "test setup: must not cross"

    def _direct_call_returns_near_miss(costmap, s, g, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            return quantized_near_miss
        return orig_astar(costmap, s, g, **kw)

    calls = {"n": 0}
    with (
        patch.object(P, "_nudge_past_gate", wraps=P._nudge_past_gate) as nudge_spy,
        patch.object(P, "astar", side_effect=_direct_call_returns_near_miss),
    ):
        path = P.plan_through(cm, start, legs)
    assert nudge_spy.call_count == 1, (
        "a genuine quantization near-miss (< 1 cell, cur far from the gate) must engage the nudge"
    )
    assert path is not None
    assert path_crosses_gate(path, gate)


def test_plan_through_pinch_relax_is_bounded_and_falls_back_when_never_converging():
    """issue #54 termination guard: a geometry whose start-disc exemption can never
    free a route (every relax round's pinch overlay still fails to thread the gate)
    must NOT relax->replan->re-seal forever. `plan_through` must call astar at most
    once for the direct attempt plus MAX_PINCH_RELAX_ROUNDS pinch retries, then fall
    back to the pre-#54 behaviour (leg reported unreachable, i.e. None) — never hang."""
    from unittest.mock import patch

    from core.nav import planner as P

    cm = _pinch_gap_grid()
    gate = ((1.05, 0.25), (1.05, 0.95))
    legs = [("corridor_between", gate), ("goto", (1.05, 1.15))]
    start = (0.35, 0.6)

    calls = {"n": 0}

    def _always_non_crossing(costmap, s, g, **kw):
        calls["n"] += 1
        # Every attempt (direct + every relax round) "reaches" somewhere but never
        # crosses the gate segment — the pathological case that would have cycled
        # relax->replan->re-seal forever pre-cap.
        return [s, s]

    with patch.object(P, "astar", side_effect=_always_non_crossing):
        result = P.plan_through(cm, start, legs)

    assert result is None, "must fall back to leg-unreachable, not hang or fabricate a plan"
    # 1 direct attempt + MAX_PINCH_RELAX_ROUNDS pinch retries, never more.
    assert calls["n"] == 1 + MAX_PINCH_RELAX_ROUNDS, (
        f"relax loop must be capped at {MAX_PINCH_RELAX_ROUNDS} rounds, got {calls['n'] - 1}"
    )


def test_plan_through_pinch_relax_converges_once_disc_widens_enough():
    """A geometry where the DEFAULT pinch_disc_m's start-disc exemption is too small to
    clear the vehicle's local inflation halo, but a later (widened) relax round's disc
    does: the leg must still thread within the cap, proving relaxation is functional
    (not just a no-op cap) and terminates as soon as a round succeeds."""
    cm = _pinch_gap_grid_with_vehicle_radius(0.4)
    gate = ((1.05, 0.25), (1.05, 0.95))
    # A tiny caller-supplied pinch_disc_m means round 0's start-disc exemption is too
    # small to clear the inflation halo around `start`; PINCH_RELAX_GROWTH widens it
    # each round until it does.
    start = (0.35, 0.6)
    legs = [("corridor_between", gate), ("goto", (1.05, 1.15))]

    path = plan_through(cm, start, legs, pinch_disc_m=0.15)
    assert path is not None, "later relax round must thread the gate within the cap"
    assert path_crosses_gate(path, gate)


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
