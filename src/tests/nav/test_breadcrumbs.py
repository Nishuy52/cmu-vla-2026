"""Breadcrumbs: spacing <= lookahead, advance on reach, stall detection, LOS."""
from __future__ import annotations

from core.interfaces import WaypointCmd
from core.nav.breadcrumbs import BreadcrumbFollower, line_of_sight
from core.nav.costmap import Costmap
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
