"""Nav/drive robustness pack (H11) — the red-team scenarios as named tests.

Covers the InstructionHead-side items of the hardening backlog H11:
  - IF-F8/SYS-F10: consume ``replan_flag`` (stall / no-LOS) and follower-exhausted-but-
    not-arrived -> rebuild + re-plan from the current pose, capped ~3/question.
  - IF-F5: avoid grounding is a route-commit precondition; runtime capsule tripwire.
  - IF-F4: forced answer assembly keeps breadcrumbs flowing (the answer IS the drive).
  - IF-F7: via placement lands in free space on the reachable side of the anchor.

Each test is a self-contained repro against the synthetic-scene mocks (no ROS).
"""
from __future__ import annotations

from core.heads.instruction import InstructionHead, MAX_REPLANS_PER_QUESTION
from core.interfaces import OdomState, WaypointCmd
from core.mocks.synthetic_scene import Room, SyntheticScene
from core.perception.scene_index import BasicSceneIndex
from core.plan_schema import Anchor, AvoidSpec, LegKind, RouteLeg
from tests.heads._helpers import instruction_plan


def _goto(noun: str) -> RouteLeg:
    return RouteLeg(kind=LegKind.GOTO, anchors=[Anchor(noun=noun)])


class _DriveIO:
    """RobotIO whose pose is set explicitly by the test (no auto-motion)."""

    def __init__(self, sc: SyntheticScene, start=(0.7, 3.0)):
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


def _room(w=10.0, h=6.0) -> SyntheticScene:
    sc = SyntheticScene(0)
    sc.rooms = [Room(0.0, 0.0, w, h)]
    return sc


# --------------------------------------------------------------------------- IF-F8 stall
def test_stall_triggers_replan_from_current_pose():
    """A grounded route whose vehicle stops moving (< 0.3 m over the 10 s window) raises
    the follower's ``replan_flag``; the head consumes it and re-plans (IF-F8/SYS-F10) —
    the old code republished the wedged crumb until the watchdog."""
    sc = _room()
    sc.place_box("sofa", 8.0, 3.0, 0.5, 0.5, 0.5)
    idx = BasicSceneIndex(sc.instances())
    inst = InstructionHead(plan=instruction_plan([_goto("sofa")]))
    io = _DriveIO(sc)

    # Hold the pose fixed across > the stall window (poses spaced 1 s so the follower's
    # stall history spans a full window and the flag fires).
    for _ in range(14):
        inst.advance(io, idx)
        io.advance_time(1.0)

    assert inst._replans >= 1, "a persistent stall must trigger at least one re-plan"
    assert any("stall" in e for e in inst.replan_events())


def test_replans_capped_per_question():
    """A pathological permanent stall cannot burn the whole budget replanning every tick:
    the count is bounded by MAX_REPLANS_PER_QUESTION (IF-F8)."""
    sc = _room()
    sc.place_box("sofa", 8.0, 3.0, 0.5, 0.5, 0.5)
    idx = BasicSceneIndex(sc.instances())
    inst = InstructionHead(plan=instruction_plan([_goto("sofa")]))
    io = _DriveIO(sc)

    for _ in range(120):  # long permanent stall
        inst.advance(io, idx)
        io.advance_time(1.0)

    assert inst._replans == MAX_REPLANS_PER_QUESTION
    # Still publishing (never silent) despite the cap being spent.
    assert io.waypoints


def test_follower_exhausted_not_arrived_replans():
    """If the follower runs out of path while the vehicle has NOT reached the terminal,
    that is a re-plan trigger — not a teleport to a distant terminal waypoint (IF-F8)."""
    sc = _room()
    sc.place_box("sofa", 8.0, 3.0, 0.5, 0.5, 0.5)
    idx = BasicSceneIndex(sc.instances())
    inst = InstructionHead(plan=instruction_plan([_goto("sofa")]))
    io = _DriveIO(sc)
    inst.advance(io, idx)
    assert inst._follower is not None

    # Force the follower to the end of its path (index past the last point) while the pose
    # is nowhere near the terminal, then tick: the exhausted-not-arrived path fires.
    inst._follower._idx = len(inst._follower.path)
    io.set_pose(0.7, 3.0)  # far from the sofa terminal (~8 m)
    io.advance_time(0.2)
    inst.advance(io, idx)

    assert any("exhausted" in e for e in inst.replan_events())


# --------------------------------------------------------------------------- IF-F5 avoid
def test_avoid_unresolvable_keeps_route_uncommitted():
    """An unresolvable avoid anchor keeps the whole route uncommitted while budget remains
    (IF-F5) — so the robot keeps exploring instead of driving a route the forbidden
    capsule can't be stamped into. With no budget signal the commit is forced (never
    strands), so we inject a low budget_frac to exercise the withhold."""
    sc = _room()
    sc.place_box("sofa", 8.0, 3.0, 0.5, 0.5, 0.5)  # leg anchor present
    # "vase" (avoid anchor) absent -> avoid_capsule raises -> not all resolvable.
    idx = BasicSceneIndex(sc.instances())
    route = [_goto("sofa")]
    avoid = [AvoidSpec(near=Anchor(noun="vase"))]
    inst = InstructionHead(
        plan=instruction_plan(route, avoid=avoid),
        budget_frac=lambda: 0.10,  # early: pressure has not forced the commit
    )
    io = _DriveIO(sc)
    emitted = inst.advance(io, idx)

    assert inst._follower is None, "route must stay uncommitted while avoid is ungrounded"
    assert emitted is False, "no waypoint from the route -> caller falls through to explore"


def test_avoid_commit_forced_under_budget_pressure():
    """Once budget pressure forces the commit, the route is built with whatever capsules
    resolved (never strand on a missing avoid anchor)."""
    sc = _room()
    sc.place_box("sofa", 8.0, 3.0, 0.5, 0.5, 0.5)
    idx = BasicSceneIndex(sc.instances())  # vase still absent
    route = [_goto("sofa")]
    avoid = [AvoidSpec(near=Anchor(noun="vase"))]
    inst = InstructionHead(
        plan=instruction_plan(route, avoid=avoid),
        budget_frac=lambda: 0.99,  # late: pressure forces the commit
    )
    io = _DriveIO(sc)
    inst.advance(io, idx)
    assert inst._follower is not None, "budget pressure commits the route despite the gap"


def test_avoid_nouns_exposed_for_affinity_bias():
    """The head exposes avoid nouns so exploration biases toward avoid anchors too (IF-F5)."""
    route = [_goto("sofa")]
    avoid = [AvoidSpec(near=Anchor(noun="tv"))]
    inst = InstructionHead(plan=instruction_plan(route, avoid=avoid))
    assert inst.avoid_nouns() == ["tv"]


# --------------------------------------------------------------------------- IF-F5 tripwire
def test_capsule_breach_mid_drive_triggers_stop_and_replan():
    """A driven pose that ENTERS a stamped avoid capsule mid-route trips the runtime
    tripwire: the head re-plans away (IF-F5). Edge-triggered so a legitimately-engulfed
    start would not, but a clear->violated transition does."""
    sc = _room()
    sc.place_box("sofa", 9.0, 3.0, 0.5, 0.5, 0.5)
    sc.place_box("vase", 5.0, 3.0, 0.3, 0.3, 0.4)
    idx = BasicSceneIndex(sc.instances())
    route = [_goto("sofa")]
    avoid = [AvoidSpec(near=Anchor(noun="vase"))]
    inst = InstructionHead(plan=instruction_plan(route, avoid=avoid))
    io = _DriveIO(sc, start=(0.7, 1.0))  # start clear of the vase capsule

    inst.advance(io, idx)  # build route + stamp the vase capsule
    assert inst._stamped_capsules, "avoid capsule must be stamped"
    replans_before = inst._replans

    # Drive the pose into the capsule centre — a clear->violated transition.
    io.set_pose(5.0, 3.0)
    io.advance_time(0.2)
    inst.advance(io, idx)

    assert inst._replans > replans_before
    assert any("capsule" in e for e in inst.replan_events())


# --------------------------------------------------------------------------- IF-F4 drive
def test_forced_assembly_keeps_crumbs_flowing():
    """When the FSM forces answer assembly, ``terminal_waypoint`` must NOT hand back a
    distant terminal coordinate while the route is unfinished — it returns the next crumb
    so the drive continues (IF-F4: the answer IS the drive)."""
    sc = _room()
    sc.place_box("sofa", 8.0, 3.0, 0.5, 0.5, 0.5)
    idx = BasicSceneIndex(sc.instances())
    inst = InstructionHead(plan=instruction_plan([_goto("sofa")]))
    io = _DriveIO(sc, start=(0.7, 3.0))
    inst.advance(io, idx)

    assert inst._follower is not None
    # Vehicle is far from the terminal and the follower still has path left.
    ans = inst.terminal_waypoint()
    assert ans is not None
    # The answer is the NEXT CRUMB (near the vehicle), not the far terminal (~8 m away).
    d = ((ans.x - io._x) ** 2 + (ans.y - io._y) ** 2) ** 0.5
    assert d <= 2.5 + 1e-6, "forced assembly emitted a distant terminal, abandoning the drive"


def test_terminal_published_when_within_reach():
    """Once the vehicle is within reach of the terminal, ``terminal_waypoint`` returns the
    raw terminal coordinate (the drive is complete)."""
    sc = _room()
    sc.place_box("sofa", 8.0, 3.0, 0.5, 0.5, 0.5)
    idx = BasicSceneIndex(sc.instances())
    inst = InstructionHead(plan=instruction_plan([_goto("sofa")]))
    io = _DriveIO(sc, start=(0.7, 3.0))
    inst.advance(io, idx)
    assert inst._terminal_xy is not None

    # Place the pose at the terminal.
    io.set_pose(*inst._terminal_xy)
    io.advance_time(0.2)
    inst.advance(io, idx)
    ans = inst.terminal_waypoint()
    assert (ans.x, ans.y) == (
        float(inst._terminal_xy[0]),
        float(inst._terminal_xy[1]),
    )
