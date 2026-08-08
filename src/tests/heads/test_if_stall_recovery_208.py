"""Issue #208: the H11 replan cap spends itself on a frozen pose and regenerates the
same refused crumb.

Root cause (#205 diagnosis, 8 Aug): ``_replan`` rebuilds the ``BreadcrumbFollower`` with
empty history (the stall window restarts) and re-plans from the SAME frozen pose over an
unchanged costmap -- A* is deterministic, so a stall replan whose pose has not moved since
the previous stall replan reproduces the identical route and the identical first crumb.
Three such replans spend ``MAX_REPLANS_PER_QUESTION`` in about 30 s, after which the wedged
crumb republishes at 4-5 Hz for the rest of the question with no code path left to change it.

Two changes, both covered here:
  (a) ``_replan`` plans from a RECOVERY pose (the nearest point that clears
      ``BASE_OBSTACLE_CLEARANCE_M``, via ``Costmap.nearest_clear_point``) instead of the
      frozen one, when a stall/no-LOS replan's pose has not moved since the previous one.
  (b) The cap now gates ``_replans_since_progress``, which RESETS to 0 once the vehicle has
      moved ``REPLAN_PROGRESS_MOVE_M`` -- a replan that produced no motion no longer spends
      the same budget as one that did. ``MAX_REPLANS_HARD_CAP`` remains an absolute ceiling
      (with the watchdog as the final, unconditional backstop) so a pathological run still
      cannot replan forever.
"""
from __future__ import annotations

from core.heads.instruction import (
    MAX_REPLANS_HARD_CAP,
    MAX_REPLANS_PER_QUESTION,
    REPLAN_PROGRESS_MOVE_M,
    InstructionHead,
)
from core.interfaces import OdomState, WaypointCmd
from core.mocks.synthetic_scene import Room, SyntheticScene
from core.perception.scene_index import BasicSceneIndex
from core.plan_schema import Anchor, LegKind, RouteLeg
from tests.heads._helpers import instruction_plan


def _goto(noun: str) -> RouteLeg:
    return RouteLeg(kind=LegKind.GOTO, anchors=[Anchor(noun=noun)])


_START = (2.15, 3.0)


class _DriveIO:
    """RobotIO whose pose is set explicitly by the test (no auto-motion)."""

    def __init__(self, sc: SyntheticScene, start=_START):
        self._sc = sc
        self._x, self._y = start
        self._t = 0.0
        self.waypoints: list[WaypointCmd] = []

    def latest_terrain(self, extended: bool = False):
        return self._sc.terrain_patch(extended=extended, t=self._t)

    def latest_odom(self):
        return OdomState(t=self._t, x=self._x, y=self._y, z=0.0, yaw=0.0)

    def publish_waypoint(self, wp: WaypointCmd):
        self.waypoints.append(wp)

    def set_pose(self, x: float, y: float):
        self._x, self._y = x, y

    def advance_time(self, dt: float):
        self._t += dt

    @property
    def pose(self):
        return (self._x, self._y)


def _wedge_scene() -> SyntheticScene:
    """A small obstacle 0.55 m from the vehicle's start pose (well clear of the room's own
    boundary walls): outside the 0.4 m planning inflation (the start cell stays passable)
    but inside the base's own 0.75 m snap-clearance floor (BASE_OBSTACLE_CLEARANCE_M) --
    so a crumb pinned near this pose is exactly the #207 shape, letting
    ``nearest_clear_point`` find a genuine nearby recovery point instead of degenerately
    returning the query point itself."""
    sc = SyntheticScene(0)
    sc.rooms = [Room(0.0, 0.0, 10.0, 6.0)]
    sc.place_box("wall_bit", 1.5, 3.0, 0.2, 0.2, 1.0)  # x in [1.4, 1.6], y in [2.9, 3.1]
    sc.place_box("sofa", 8.0, 3.0, 0.5, 0.5, 0.5)
    return sc


def _wedged_head() -> tuple[InstructionHead, _DriveIO, BasicSceneIndex]:
    sc = _wedge_scene()
    idx = BasicSceneIndex(sc.instances())
    head = InstructionHead(plan=instruction_plan([_goto("sofa")]))
    io = _DriveIO(sc)
    head.advance(io, idx)
    assert head._follower is not None
    assert head._replans == 0, "test setup must not already have triggered a replan"
    return head, io, idx


# --------------------------------------------------------------------------- (a) recovery pose
def test_second_stall_replan_from_a_frozen_pose_uses_a_recovery_start():
    """Two stall replans back to back with NO vehicle motion between them: the first
    plans from the pose itself (nothing to recover from yet); the second must plan from
    a different (recovery) start -- not reproduce the exact same route."""
    head, io, idx = _wedged_head()

    assert head._replan("stall_or_no_los", idx) is True
    first_start = head._follower.path[0]
    assert first_start == io.pose  # first replan: no history yet, plans from the pose itself

    # Pose has not moved -- mirrors the #205 frozen-pose replan storm.
    assert head._replan("stall_or_no_los", idx) is True
    second_start = head._follower.path[0]
    assert second_start != io.pose, (
        "second stall replan from an unmoved pose must not plan from the identical "
        "frozen pose again"
    )


def test_stall_replan_recovery_produces_a_different_next_goal():
    """End to end: a wedged follower whose stall flag fires, replanned twice with no
    motion in between, plans its next leg from a DIFFERENT, genuinely clearer start the
    second time -- not the exact frozen pose the first replan already tried (and the
    base presumably refused, #207) -- so the vehicle has a new target instead of the
    same one repeated."""
    head, io, idx = _wedged_head()

    assert head._replan("stall_or_no_los", idx) is True
    first_goal = head._follower.path[0]
    assert first_goal == io.pose  # nothing to recover from yet on the first attempt

    assert head._replan("stall_or_no_los", idx) is True
    second_goal = head._follower.path[0]
    assert second_goal != first_goal, "the replan must not hand the follower the same goal twice"
    assert second_goal != io.pose
    # And the recovery start is genuinely SAFER than the frozen wedge point, not just
    # numerically different.
    assert head._costmap.obstacle_clearance_m(*second_goal) >= head._costmap.obstacle_clearance_m(*io.pose)


def test_recovery_pose_not_used_when_vehicle_has_genuinely_moved():
    """The recovery-pose substitution is specifically for a REPEATED stall from the same
    spot -- once the vehicle has actually moved past REPLAN_PROGRESS_MOVE_M, the next
    stall replan plans from the vehicle's real (moved) pose, not a nudged one."""
    head, io, idx = _wedged_head()
    assert head._replan("stall_or_no_los", idx) is True

    io.set_pose(_START[0] + REPLAN_PROGRESS_MOVE_M + 0.05, 3.0)
    head._pose = io.pose  # mirrors what InstructionHead.advance() records each tick
    assert head._replan("stall_or_no_los", idx) is True
    assert head._follower.path[0] == io.pose


# --------------------------------------------------------------------------- (b) progress-gated cap
def test_replan_budget_not_exhausted_by_attempts_that_never_moved_the_vehicle():
    """Issue #208(b): MAX_REPLANS_PER_QUESTION stall replans fired from a frozen pose
    exhaust ``_can_replan()`` exactly as before (the H11 pathological-storm protection is
    unchanged) -- but ``_replans`` itself (the raw attempt count) is no longer what's
    gated, so a LATER stall (after real motion) still gets its own fresh budget."""
    head, io, idx = _wedged_head()

    for _ in range(MAX_REPLANS_PER_QUESTION):
        assert head._replan("stall_or_no_los", idx) is True
    assert head._can_replan() is False, "the per-stall budget must still cap a frozen-pose storm"
    assert head._replan("stall_or_no_los", idx) is False

    # The vehicle then genuinely moves (a later, unrelated stall elsewhere in the drive).
    io.set_pose(_START[0] + REPLAN_PROGRESS_MOVE_M + 0.05, 3.0)
    head._pose = io.pose
    head._note_pose_progress(io.pose)
    assert head._can_replan() is True, "progress must reset the per-stall replan budget"
    assert head._replan("stall_or_no_los", idx) is True


def test_replans_since_progress_resets_on_motion_but_replans_counter_does_not():
    head, io, idx = _wedged_head()
    assert head._replan("stall_or_no_los", idx) is True
    assert head._replans == 1
    assert head._replans_since_progress == 1

    io.set_pose(_START[0] + REPLAN_PROGRESS_MOVE_M + 0.05, 3.0)
    head._note_pose_progress(io.pose)
    assert head._replans_since_progress == 0
    assert head._replans == 1, "the raw flight-recorder counter is monotonic, unaffected by progress"


def test_hard_cap_still_bounds_a_pathological_run_across_many_progress_resets():
    """Even if the vehicle keeps making just enough progress to reset the per-stall
    budget every time, MAX_REPLANS_HARD_CAP still stops the run eventually (issue #208:
    'keep a hard bound so a pathological run cannot replan forever')."""
    head, io, idx = _wedged_head()
    x = _START[0]
    for _ in range(MAX_REPLANS_HARD_CAP + 5):
        if not head._can_replan():
            break
        assert head._replan("stall_or_no_los", idx) is True
        x += REPLAN_PROGRESS_MOVE_M + 0.05
        io.set_pose(x, 3.0)
        head._pose = io.pose
        head._note_pose_progress(io.pose)
    assert head._replans <= MAX_REPLANS_HARD_CAP
    assert head._can_replan() is False


def test_note_pose_progress_does_not_reset_on_sub_threshold_motion():
    head, io, idx = _wedged_head()
    assert head._replan("stall_or_no_los", idx) is True
    assert head._replans_since_progress == 1

    io.set_pose(_START[0] + REPLAN_PROGRESS_MOVE_M * 0.5, 3.0)  # below the progress floor
    head._note_pose_progress(io.pose)
    assert head._replans_since_progress == 1, "sub-threshold motion must not reset the budget"
