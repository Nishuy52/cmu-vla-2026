"""InstructionHead #183 reinvest re-drive: DRIVE_OUT must not settle for "the follower
mechanically stopped" while a leg was never individually reached in order. See
core.heads.instruction.InstructionHead._redrive / all_legs_visited and
core.fsm.controller._tick_drive_out (WorldView.legs_visited).
"""
from __future__ import annotations

import numpy as np
import pytest

from core.geometry.toolbox import avoid_capsule, capsule_violated
from core.heads.instruction import MAX_REDRIVE_PASSES, MAX_REPLANS_PER_QUESTION, InstructionHead
from core.interfaces import OdomState, WaypointCmd
from core.mocks.synthetic_scene import Room, SyntheticScene
from core.perception.scene_index import BasicSceneIndex
from core.plan_schema import Anchor, AvoidSpec, LegKind, RouteLeg
from tests.heads._helpers import instruction_plan


def _goto(noun: str) -> RouteLeg:
    return RouteLeg(kind=LegKind.GOTO, anchors=[Anchor(noun=noun)])


class _DriveIO:
    """Minimal RobotIO: fixed terrain patch, kinematic pose that chases the last waypoint."""

    def __init__(self, sc: SyntheticScene, start=(0.7, 3.0), step=0.1):
        self._sc = sc
        self._x, self._y = start
        self._t = 0.0
        self._step = step
        self.waypoints: list[WaypointCmd] = []

    def latest_terrain(self, extended: bool = False):
        return self._sc.terrain_patch(extended=extended, t=self._t)

    def latest_odom(self):
        return OdomState(t=self._t, x=self._x, y=self._y, z=0.0, yaw=0.0)

    def publish_waypoint(self, wp: WaypointCmd):
        self.waypoints.append(wp)

    def tick_motion(self):
        if not self.waypoints:
            self._t += 0.2
            return
        wp = self.waypoints[-1]
        dx, dy = wp.x - self._x, wp.y - self._y
        d = (dx * dx + dy * dy) ** 0.5
        s = min(self._step, d)
        if d > 1e-6:
            self._x += dx / d * s
            self._y += dy / d * s
        self._t += 0.2

    @property
    def pose(self):
        return (self._x, self._y)


def _two_leg_scene() -> tuple[SyntheticScene, BasicSceneIndex]:
    """A wide room with two well-separated GOTO anchors and an avoid anchor sitting
    directly on the straight line between them, so any redrive that ignored the avoid
    capsule (a raw beeline instead of a re-plan through the stamped costmap) would be
    caught crossing it."""
    sc = SyntheticScene(0)
    sc.rooms = [Room(0.0, 0.0, 10.0, 6.0)]
    sc.place_box("lamp", 2.0, 3.0, 0.4, 0.4, 0.5)
    sc.place_box("shelf", 8.0, 3.0, 0.4, 0.4, 0.5)
    sc.place_box("trap", 5.0, 3.0, 0.3, 0.3, 0.4)  # sits on the lamp->shelf straight line
    return sc, BasicSceneIndex(sc.instances())


class _DeadFollower:
    """Stand-in for an exhausted BreadcrumbFollower: no more crumbs, no replan flag."""

    replan_flag = False

    def advance(self, pose, t):
        return None

    def current(self, pose):
        return None


def _stall_the_follower(head: InstructionHead) -> None:
    """Simulate the drive having mechanically stalled: swap in a dead follower and spend
    the H11 replan cap, mirroring the live #181/#183 scenario where the stall/no-LOS
    replans fired during exploration already spent the cap before DRIVE_OUT began.

    Issue #208: ``_can_replan`` now gates on ``_replans_since_progress`` (which resets on
    real vehicle motion), not the raw ``_replans`` counter — set both so this helper still
    spends the cap exactly as before regardless of which one gates it.
    """
    head._follower = _DeadFollower()
    head._replans = MAX_REPLANS_PER_QUESTION
    head._replans_since_progress = MAX_REPLANS_PER_QUESTION


@pytest.mark.slow
def test_redrive_fires_when_stalled_with_unvisited_legs_and_replan_cap_spent():
    """#183: when the follower is exhausted, the H11 replan cap is already spent, and a
    leg has never been individually confirmed reached, ``_drive`` reinvests the #183
    redrive budget (a SEPARATE cap from H11's) to rebuild the full route and keep going,
    instead of the FSM being told the drive is complete."""
    sc, idx = _two_leg_scene()
    head = InstructionHead(plan=instruction_plan([_goto("lamp"), _goto("shelf")]))
    io = _DriveIO(sc)
    head.advance(io, idx)  # grounds both legs, stamps nothing (no avoid), builds a real route
    assert head._follower is not None
    assert head.ungrounded_subgoals() == 0
    assert head._confirmed == set()  # nothing reached yet — pose hasn't moved

    _stall_the_follower(head)
    assert head.all_legs_visited() is False
    # drive_complete() alone is the LOOSER "mechanically stopped" signal (#183 docstring)
    # -- it can and does read True here (follower exhausted, H11 cap spent); the FSM
    # pairs it with all_legs_visited() (WorldView.legs_visited) before treating the
    # question as actually finished. See core.fsm.controller._tick_drive_out.
    assert head.drive_complete() is True
    # can no longer H11-replan (cap spent)...
    assert head._can_replan() is False
    # ...but the SEPARATE #183 reinvest budget is untouched.
    assert head._can_redrive() is True

    published = head._drive(io, io.pose, 0.0)
    assert published is True
    assert head._redrives == 1
    assert head.redrive_events(), "no redrive event recorded"
    assert "unvisited_legs=[0, 1]" in head.redrive_events()[0]
    # The dead follower was replaced by a genuine rebuilt route, not left in place.
    assert not isinstance(head._follower, _DeadFollower)
    assert head._follower is not None


@pytest.mark.slow
def test_redrive_visits_every_leg_in_order_and_skips_nothing():
    """After a #183 redrive fires, driving to completion visits BOTH legs, in the
    plan's order (leg 0 confirmed no later than leg 1) — nothing is skipped."""
    sc, idx = _two_leg_scene()
    head = InstructionHead(plan=instruction_plan([_goto("lamp"), _goto("shelf")]))
    io = _DriveIO(sc)
    head.advance(io, idx)
    _stall_the_follower(head)

    confirm_order: list[int] = []
    for _ in range(6000):
        before = set(head._confirmed)
        head.advance(io, idx)
        newly = sorted(head._confirmed - before)
        confirm_order.extend(newly)
        io.tick_motion()
        if head.all_legs_visited():
            break

    assert head.all_legs_visited() is True, "redrive never reached every leg"
    assert confirm_order == [0, 1], f"legs confirmed out of order or skipped: {confirm_order}"
    assert head._redrives >= 1
    assert head.drive_complete() is True  # genuinely complete once every leg is visited


@pytest.mark.slow
def test_redrive_never_violates_an_avoid_capsule_the_first_pass_would_not_have():
    """The #183 redrive reuses the same stamp-then-plan_through machinery as the first
    pass and every H11 replan — driving through a forced redrive must never cross the
    avoid capsule the original route already had to route around."""
    sc, idx = _two_leg_scene()
    avoid = [AvoidSpec(near=Anchor(noun="trap"))]
    head = InstructionHead(plan=instruction_plan([_goto("lamp"), _goto("shelf")], avoid=avoid))
    io = _DriveIO(sc)
    head.advance(io, idx)
    _stall_the_follower(head)

    traj = [io.pose]
    for _ in range(6000):
        head.advance(io, idx)
        io.tick_motion()
        traj.append(io.pose)
        if head.all_legs_visited():
            break
    assert head.all_legs_visited() is True
    assert head._redrives >= 1, "test setup did not actually exercise the redrive path"

    traj = np.array(traj)
    cap = avoid_capsule(AvoidSpec(near=Anchor(noun="trap")), idx)
    violated, _ = capsule_violated(traj, cap)
    assert violated is False
    wp = np.array([[w.x, w.y] for w in io.waypoints])
    wp_violated, _ = capsule_violated(wp, cap)
    assert wp_violated is False


def test_redrive_budget_is_bounded_and_separate_from_h11_replan_cap():
    """The #183 reinvest budget is its own bounded counter, independent of
    MAX_REPLANS_PER_QUESTION -- spending it does not touch ``_replans`` and it stops
    firing once MAX_REDRIVE_PASSES is spent (the watchdog remains the real backstop)."""
    sc, idx = _two_leg_scene()
    head = InstructionHead(plan=instruction_plan([_goto("lamp"), _goto("shelf")]))
    io = _DriveIO(sc)
    head.advance(io, idx)
    _stall_the_follower(head)

    for _ in range(MAX_REDRIVE_PASSES):
        head._follower = _DeadFollower()  # force another stall each pass
        replans_before = head._replans
        fired = head._redrive("test_forced", idx)
        assert fired is True
        assert head._replans == replans_before  # H11 cap untouched by a #183 redrive
    assert head._can_redrive() is False
    head._follower = _DeadFollower()
    assert head._redrive("test_forced", idx) is False  # budget spent — no more redrives
    assert head._redrives == MAX_REDRIVE_PASSES


def test_all_legs_visited_true_when_route_has_no_legs():
    """#183: a plan with no route legs is vacuously "fully visited" — nothing to redrive."""
    head = InstructionHead(plan=instruction_plan([]))
    assert head.all_legs_visited() is True


def test_all_legs_visited_false_until_every_leg_confirmed():
    sc, idx = _two_leg_scene()
    head = InstructionHead(plan=instruction_plan([_goto("lamp"), _goto("shelf")]))
    io = _DriveIO(sc)
    head.advance(io, idx)
    assert head.all_legs_visited() is False
    head._confirmed = {0}
    assert head.all_legs_visited() is False
    head._confirmed = {0, 1}
    assert head.all_legs_visited() is True
