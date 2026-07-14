"""Exploration robustness pack (H11) — the red-team scenarios as named tests.

Covers the ExploreHead-side items of the hardening backlog H11:
  - SYS-F9 (a): the exploration policy is not constructed/run before the first odom, so a
    pre-odom (0,0)/t=0 anchor never seeds the sweep at the origin.
  - SYS-F9 (b): a far frontier goal is clamped to <= 2.5 m along the line toward it.
  - SYS-F9 (c): unreachable (pd=1e6) frontier goals are dropped, not commanded.
  - SYS-F11: the CP2 provisional waypoint latch clears on arrival / timeout / noun
    appearing in the scene, then frontier flow resumes.

Uses the synthetic-scene mocks (no ROS).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from core.heads.explore_step import (
    ExploreHead,
    FRONTIER_GOAL_CLAMP_M,
    FRONTIER_THROTTLE_S,
    PROVISIONAL_TIMEOUT_S,
    _frontier_unreachable,
)
from core.nav.exploration import ExplorationDecision, ExplorationStatus
from core.interfaces import OdomState, WaypointCmd
from core.mocks.synthetic_scene import SyntheticScene
from core.nav.frontiers import Frontier
from core.perception.scene_index import BasicSceneIndex
from tests.heads._helpers import object_plan


@dataclass
class _MR:
    action: str
    confidence: float = 0.8
    n_obs: int = 1
    score: float = 0.4


class _ExploreIO:
    """RobotIO with a controllable odom (may be None to simulate pre-odom ticks)."""

    def __init__(self, sc, start=(2.5, 2.5), odom_present=True):
        self._sc = sc
        self._x, self._y, self._t = start[0], start[1], 0.0
        self._odom_present = odom_present
        self.waypoints: list[WaypointCmd] = []

    def latest_terrain(self, extended=False):
        return self._sc.terrain_patch(extended=extended, t=self._t)

    def latest_odom(self):
        if not self._odom_present:
            return None
        return OdomState(t=self._t, x=self._x, y=self._y, z=0.0, yaw=0.0)

    def publish_waypoint(self, wp):
        self.waypoints.append(wp)

    def set_odom_present(self, present):
        self._odom_present = present

    def set_pose(self, x, y):
        self._x, self._y = x, y

    def advance_time(self, dt):
        self._t += dt


def _scene_without(noun_absent="unicorn"):
    sc = SyntheticScene(0)
    sc.populate_default(2)
    return sc, BasicSceneIndex(sc.instances())


# --------------------------------------------------------------------------- SYS-F9 (a)
def test_pre_odom_tick_publishes_nothing_and_builds_no_policy():
    """A tick before the first odom must publish nothing and NOT construct the policy —
    otherwise the sweep anchors at (0,0)/t=0 and is later skipped (SYS-F9)."""
    sc, idx = _scene_without()
    head = ExploreHead(plan=object_plan("chair"))
    io = _ExploreIO(sc, odom_present=False)

    head.advance(io, idx)

    assert io.waypoints == [], "pre-odom tick must publish nothing unsafe"
    assert head._policy is None, "policy must not be anchored before the first odom"


def test_policy_anchors_on_first_real_pose_not_origin():
    """Once odom arrives at a non-origin pose the policy anchors there (not (0,0))."""
    sc, idx = _scene_without()
    head = ExploreHead(plan=object_plan("chair"))
    io = _ExploreIO(sc, start=(2.5, 2.5), odom_present=False)
    head.advance(io, idx)  # pre-odom: no policy
    io.set_odom_present(True)
    head.advance(io, idx)  # first real pose
    assert head._policy is not None
    assert head._policy.start_xy == (2.5, 2.5)


# --------------------------------------------------------------------------- SYS-F9 (b)
def test_far_frontier_goal_is_clamped():
    """A goal well beyond the clamp is bounded to <= FRONTIER_GOAL_CLAMP_M along the line
    toward it (SYS-F9): direction preserved, distance bounded."""
    sc, idx = _scene_without()
    head = ExploreHead(plan=object_plan("chair"))
    wp = head._clamp_goal((0.0, 0.0), (10.0, 0.0))
    d = (wp.x ** 2 + wp.y ** 2) ** 0.5
    assert abs(d - FRONTIER_GOAL_CLAMP_M) < 1e-6
    assert wp.y == 0.0  # direction preserved (straight +x)


def test_near_frontier_goal_is_not_clamped():
    head = ExploreHead(plan=object_plan("chair"))
    wp = head._clamp_goal((0.0, 0.0), (1.0, 0.0))
    assert (wp.x, wp.y) == (1.0, 0.0)


# --------------------------------------------------------------------------- SYS-F9 (c)
def test_unreachable_frontier_is_dropped():
    """A frontier carrying the pd=1e6 unreachable sentinel is recognised and dropped."""
    reachable = Frontier(row=0, col=0, xy=(1.0, 1.0), size=10, path_distance=5.0,
                         affinity=0.0, score=10.0)
    unreachable = Frontier(row=0, col=0, xy=(9.0, 9.0), size=10, path_distance=1e6,
                           affinity=0.0, score=1.0)
    assert _frontier_unreachable(unreachable) is True
    assert _frontier_unreachable(reachable) is False
    assert _frontier_unreachable(None) is True


# --------------------------------------------------------------------------- SYS-F11 latch
def _provisioned_head(sc, idx, budget_frac=lambda: 0.8):
    head = ExploreHead(
        plan=object_plan("unicorn"),  # absent -> CP2 fires
        miss_recoverer=lambda n, r, t: _MR("provisional"),
        budget_frac=budget_frac,
    )
    io = _ExploreIO(sc)
    head.advance(io, idx)  # CP2 fires, provisional set
    assert head.provisional_xy is not None
    return head, io


def test_provisional_clears_on_arrival():
    """The provisional latch clears when the vehicle arrives within tolerance (SYS-F11)."""
    sc, idx = _scene_without()
    head, io = _provisioned_head(sc, idx)
    px, py = head.provisional_xy
    io.set_pose(px, py)  # drive to the provisional
    io.advance_time(1.0)
    head.advance(io, idx)
    assert head.provisional_xy is None, "provisional must clear on arrival"


def test_provisional_clears_on_timeout():
    """The provisional latch clears after ~PROVISIONAL_TIMEOUT_S without arrival (SYS-F11)."""
    sc, idx = _scene_without()
    head, io = _provisioned_head(sc, idx)
    # Stay away from the provisional; advance past the timeout.
    io.set_pose(2.5, 2.5)
    io.advance_time(PROVISIONAL_TIMEOUT_S + 1.0)
    head.advance(io, idx)
    assert head.provisional_xy is None, "provisional must clear on timeout"


def test_provisional_clears_when_noun_observed():
    """The provisional latch clears once the stood-in noun appears in the scene (SYS-F11)."""
    sc, idx = _scene_without()
    head, io = _provisioned_head(sc, idx)
    # The unicorn now appears in the scene index.
    sc.place_box("unicorn", 4.0, 4.0, 0.4, 0.4, 0.5)
    idx2 = BasicSceneIndex(sc.instances())
    io.set_pose(2.5, 2.5)  # NOT at the provisional (so not an arrival clear)
    io.advance_time(1.0)
    head.advance(io, idx2)
    assert head.provisional_xy is None, "provisional must clear once the noun is observed"


# --------------------------------------------------------------------------- H14 throttle
class _CountingPolicy:
    """Stub ExplorationPolicy that always returns a FRONTIER decision and counts steps."""

    def __init__(self):
        self.steps = 0
        self.start_xy = (0.0, 0.0)

    def step(self, grid, pose, t, **kw):
        self.steps += 1
        return ExplorationDecision(
            ExplorationStatus.FRONTIER,
            waypoint=WaypointCmd(1.0, 1.0),
            frontier=Frontier(0, 0, (1.0, 1.0), 10, 3.0, 0.0, 10.0),
        )


def test_frontier_detection_throttled_to_1hz():
    """Frontier-phase policy stepping (which runs detect_frontiers) is gated to <= 1 Hz:
    over 5 s of 5 Hz ticks the policy steps ~5 times, not ~25 (H14)."""
    sc, idx = _scene_without()
    head = ExploreHead(plan=object_plan("chair"))
    policy = _CountingPolicy()
    head._policy = policy
    io = _ExploreIO(sc, start=(0.0, 0.0))

    # 25 ticks spaced 0.2 s => 5 s of sim time.
    for _ in range(25):
        head.advance(io, idx)
        io.advance_time(0.2)

    # With a 1 s throttle over 5 s, the policy is stepped at most ~6 times (once per second
    # plus the initial), far below the 25 ticks.
    assert policy.steps <= 6, f"throttle failed: policy stepped {policy.steps} times over 5 s"
    assert policy.steps >= 4, "throttle over-suppressed: expected ~1 Hz refreshes"


def test_provisional_clears_once_and_frontier_resumes():
    """After the latch clears, frontier flow resumes and CP2 does not re-fire (SYS-F11 +
    the once-per-question guard)."""
    sc, idx = _scene_without()
    head, io = _provisioned_head(sc, idx)
    io.set_pose(2.5, 2.5)
    io.advance_time(PROVISIONAL_TIMEOUT_S + 1.0)
    head.advance(io, idx)  # clears
    assert head.provisional_xy is None
    # Subsequent ticks explore (frontier/sweep) and CP2 stays fired-once.
    io.advance_time(1.0)
    head.advance(io, idx)
    assert head._cp2_fired is True
    assert head.provisional_xy is None
