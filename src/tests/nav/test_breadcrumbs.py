"""Breadcrumbs: spacing <= lookahead, advance on reach, stall detection, LOS."""
from __future__ import annotations

from core.interfaces import WaypointCmd
from core.nav.breadcrumbs import BreadcrumbFollower, ensure_base_clearance, line_of_sight
from core.nav.costmap import BASE_OBSTACLE_CLEARANCE_M, Costmap
from core.nav.occupancy import OccupancyGrid
from tests.nav.helpers import patch_from_ascii


def _open_cm(n=60):
    grid = OccupancyGrid(cell_m=0.1)
    grid.integrate_patch(patch_from_ascii(["." * n for _ in range(n)], cell_m=0.1))
    return Costmap(grid, vehicle_radius_m=0.0)


def _straight_path(n=20, step=0.3):
    return [(step * i + 0.05, 0.05) for i in range(n)]


def test_crumb_within_lookahead():
    cm = _open_cm()
    path = _straight_path()
    f = BreadcrumbFollower(path=path, costmap=cm)
    pose = (0.05, 0.05)
    wp = f.current(pose)
    assert wp is not None
    d = ((wp.x - pose[0]) ** 2 + (wp.y - pose[1]) ** 2) ** 0.5
    assert d <= f.lookahead_m + 1e-6


def test_crumb_is_farthest_within_lookahead():
    cm = _open_cm()
    path = _straight_path(n=30, step=0.3)
    f = BreadcrumbFollower(path=path, costmap=cm)
    pose = (0.05, 0.05)
    wp = f.current(pose)
    # Farthest point within 2.5 m along a straight clear path should be ~2.45 m out.
    d = ((wp.x - pose[0]) ** 2 + (wp.y - pose[1]) ** 2) ** 0.5
    assert 2.0 <= d <= 2.5 + 1e-6


def test_advance_progresses_on_reach():
    cm = _open_cm()
    path = _straight_path(n=10, step=0.3)
    f = BreadcrumbFollower(path=path, costmap=cm)
    idx0 = f._idx
    # Move the vehicle onto the 3rd path point; index should advance past reached ones.
    f.advance(path[3], t=0.2)
    assert f._idx > idx0


def test_advance_returns_none_at_end():
    cm = _open_cm()
    path = _straight_path(n=5, step=0.3)
    f = BreadcrumbFollower(path=path, costmap=cm)
    # Drive to the final point.
    for i, p in enumerate(path):
        f.advance(p, t=0.2 * i)
    wp = f.advance(path[-1], t=10.0)
    assert wp is None
    assert f.at_goal(path[-1])


def test_reached_signal_advances():
    cm = _open_cm()
    path = _straight_path(n=10, step=0.3)
    f = BreadcrumbFollower(path=path, costmap=cm)
    before = f._idx
    f.advance(path[0], t=0.0, reached_signal=True)
    assert f._idx >= before


def test_stall_detection_flags_replan():
    cm = _open_cm()
    path = _straight_path(n=10, step=0.3)
    f = BreadcrumbFollower(path=path, costmap=cm, stall_window_s=10.0, stall_move_m=0.3)
    # Feed the same pose across > stall window: should flag replan.
    for k in range(12):
        f.advance((0.05, 0.05), t=float(k))  # 0..11 s, no movement
    assert f.replan_flag is True


def test_no_stall_when_moving():
    cm = _open_cm()
    path = _straight_path(n=40, step=0.3)
    f = BreadcrumbFollower(path=path, costmap=cm, stall_window_s=10.0, stall_move_m=0.3)
    for k in range(12):
        f.advance((0.3 * k + 0.05, 0.05), t=float(k))  # moving steadily
    assert f.replan_flag is False


def test_stall_not_flagged_before_window_fills():
    cm = _open_cm()
    path = _straight_path()
    f = BreadcrumbFollower(path=path, costmap=cm, stall_window_s=10.0)
    f.advance((0.05, 0.05), t=0.0)
    f.advance((0.05, 0.05), t=2.0)  # only 2 s of history
    assert f.replan_flag is False


def test_line_of_sight_blocked_by_obstacle():
    grid = OccupancyGrid(cell_m=0.1)
    rows = ["." * 10 for _ in range(3)]
    rows[1] = "....#....."  # obstacle in the middle row
    grid.integrate_patch(patch_from_ascii(rows, cell_m=0.1))
    cm = Costmap(grid, vehicle_radius_m=0.0)
    # LOS across the obstacle is blocked.
    assert not line_of_sight(cm, (0.05, 0.15), (0.85, 0.15))
    # LOS along a clear row is open.
    assert line_of_sight(cm, (0.05, 0.05), (0.85, 0.05))


# --------------------------------------------------------------------------- issue #74
# leg_goal_indices: a leg-goal path index is a hard stop — neither crumb selection nor
# progress-index advancement may go past it until the pose dwells within
# leg_goal_tol_m of it, even when a farther, LOS-clear point sits within one ordinary
# lookahead window (the exact skip mechanism the issue names).


def test_leg_boundary_not_selected_past_without_dwelling():
    """A leg goal sits at index 5 of a straight, fully-open path; without leg-boundary
    awareness the farthest-within-lookahead scan would happily select a point well
    past it (the whole path is LOS-clear and short). With ``leg_goal_indices=[5]`` the
    crumb must never be chosen past index 5 while the pose has not yet dwelled there."""
    cm = _open_cm()
    path = _straight_path(n=20, step=0.3)  # spans ~5.7 m; well past LOOKAHEAD_M=2.5
    f = BreadcrumbFollower(path=path, costmap=cm, leg_goal_indices=[5])
    pose = (0.05, 0.05)
    wp = f.current(pose)
    assert wp is not None
    # index 5 -> x = 0.3*5 + 0.05 = 1.55; must not select past it.
    assert wp.x <= path[5][0] + 1e-6


def test_leg_boundary_lifts_once_dwelled():
    cm = _open_cm()
    path = _straight_path(n=20, step=0.3)
    f = BreadcrumbFollower(path=path, costmap=cm, leg_goal_indices=[5])
    # Drive up to (and dwell at) the leg-goal point.
    for k in range(6):
        f.advance(path[k], t=float(k))
    # Now at path[5], well within leg_goal_tol_m of itself -> the boundary should have
    # lifted, and the crumb may again be selected past index 5.
    wp = f.current(path[5])
    assert wp is not None
    assert wp.x > path[5][0] + 1e-6


def test_leg_boundary_caps_idx_advance_even_on_stationary_tick():
    """Mechanism 2 (issue #74): the within/overshoot loop can consume many path
    indices in one ``advance()`` call regardless of real movement (grid resolution far
    finer than ``reach_m``). Ordinarily ``reach_m < leg_goal_tol_m`` already keeps this
    loop from crossing an un-dwelled leg boundary (reaching-within-reach_m of the
    boundary itself always satisfies the looser dwell tolerance too, so
    ``_advance_leg_goals`` pops it before the cap would even matter). Exercise the cap
    directly with an oversized ``reach_m`` (a caller misconfiguration / edge case) to
    prove the ceiling check is load-bearing on its own, not merely redundant with the
    tolerance ordering."""
    cm = _open_cm()
    # step=0.2 m -> index 10 sits 2.0 m out, beyond leg_goal_tol_m (~1.75 m), so it is
    # NOT already dwelled at the start pose; reach_m=5.0 m would otherwise let the
    # within-loop consume straight past it.
    path = [(0.2 * i + 0.05, 0.05) for i in range(60)]
    f = BreadcrumbFollower(path=path, costmap=cm, leg_goal_indices=[10], reach_m=5.0)
    f.advance((0.05, 0.05), t=0.0)
    assert f._idx <= 10


def test_leg_boundary_empty_list_preserves_prior_behaviour():
    """No leg_goal_indices threaded (every pre-#74 caller/test) -> identical to the
    pre-#74 unconstrained follower."""
    cm = _open_cm()
    path = _straight_path(n=30, step=0.3)
    f = BreadcrumbFollower(path=path, costmap=cm)
    pose = (0.05, 0.05)
    wp = f.current(pose)
    d = ((wp.x - pose[0]) ** 2 + (wp.y - pose[1]) ** 2) ** 0.5
    assert 2.0 <= d <= 2.5 + 1e-6


# --------------------------------------------------------------------------- issue #207
# ensure_base_clearance / BASE_OBSTACLE_CLEARANCE_M: a crumb must never sit closer than
# the base's own obstacleDisThre (0.75 m) to an obstacle when a legal alternative exists
# nearby; a genuinely tight passage (no alternative) must be left alone, not blocked.


def test_crumb_never_published_closer_than_base_clearance():
    """A crumb the unconstrained farthest-within-lookahead scan would select right next
    to an obstacle (clearance < BASE_OBSTACLE_CLEARANCE_M) is nudged to a nearby
    alternative that clears the base's own snap-clearance floor, when room to do so
    exists (an open area, not a doorway — see the doorway test below)."""
    from tests.nav.helpers import make_points

    grid = OccupancyGrid(cell_m=0.1)
    grid.integrate_patch(patch_from_ascii(["." * 60 for _ in range(60)], cell_m=0.1))
    # A single obstacle 0.3 m off the path, right where the unconstrained scan would
    # otherwise select its farthest LOS-clear crumb (see test_crumb_is_farthest_within_lookahead).
    grid.integrate_patch(make_points([(2.45, 3.35, 0.5)]))
    cm = Costmap(grid, vehicle_radius_m=0.0)
    path = [(0.3 * i + 0.05, 3.05) for i in range(20)]
    f = BreadcrumbFollower(path=path, costmap=cm)
    wp = f.current((0.05, 3.05))
    assert wp is not None
    assert cm.obstacle_clearance_m(wp.x, wp.y) >= BASE_OBSTACLE_CLEARANCE_M - 1e-9


def test_doorway_width_crumb_still_selected_when_no_wider_alternative_exists():
    """Doorway-safety check: inside a 0.9 m corridor (the narrowest doorway width
    measured across the challenge scene set — see
    ``tests.nav.test_costmap._corridor_grid``), no point clears
    BASE_OBSTACLE_CLEARANCE_M anywhere. ``ensure_base_clearance`` must NOT block or
    discard the crumb in that case (there is nothing better nearby to nudge to) — the
    follower keeps threading the doorway instead of trading one stall (base clearance
    refusal) for a worse one (our own crumb selector giving up on a legal passage)."""
    from tests.nav.test_costmap import _corridor_grid

    grid, n_cols, n_rows = _corridor_grid()
    cm = Costmap(grid, vehicle_radius_m=0.4)
    gap_x, _ = grid.cell_to_world(0, n_cols // 2)
    path = [(gap_x, 0.05 + 0.1 * i) for i in range(n_rows)]  # straight down the corridor
    f = BreadcrumbFollower(path=path, costmap=cm)
    wp = f.current((gap_x, 0.05))
    assert wp is not None
    assert f.replan_flag is False


def test_ensure_base_clearance_returns_point_unchanged_when_no_alternative_exists():
    """Direct unit check of the fallback behaviour ``ensure_base_clearance`` relies on:
    no clear alternative nearby -> the original point comes back untouched."""
    from tests.nav.test_costmap import _corridor_grid

    grid, n_cols, n_rows = _corridor_grid()
    cm = Costmap(grid, vehicle_radius_m=0.4)
    point = grid.cell_to_world(n_rows // 2, n_cols // 2)
    out = ensure_base_clearance(cm, point, pose=point)
    assert out == point


def test_crumb_respects_line_of_sight():
    grid = OccupancyGrid(cell_m=0.1)
    rows = ["." * 30 for _ in range(3)]
    rows[1] = "." * 15 + "#" + "." * 14  # obstacle blocks the far crumb's LOS
    grid.integrate_patch(patch_from_ascii(rows, cell_m=0.1))
    cm = Costmap(grid, vehicle_radius_m=0.0)
    path = [(0.3 * i + 0.05, 0.15) for i in range(10)]
    f = BreadcrumbFollower(path=path, costmap=cm)
    wp = f.current((0.05, 0.15))
    # Chosen crumb must not sit past the obstacle at x~1.55.
    assert wp.x < 1.55
