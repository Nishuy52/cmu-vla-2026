"""InstructionHead: leg grounding, ungrounded tracking, corridor/avoid routing, drive."""
from __future__ import annotations

import numpy as np
import pytest

from core.geometry.toolbox import (
    avoid_capsule,
    capsule_violated,
    corridor_gate,
    threading_check,
)
from core.heads.instruction import InstructionHead
from core.interfaces import WaypointCmd
from core.mocks.mock_io import FakeClock, MockRobotIO
from core.mocks.synthetic_scene import SyntheticScene
from core.perception.scene_index import BasicSceneIndex
from core.plan_schema import Anchor, AvoidSpec, LegKind, RouteLeg
from tests.heads._helpers import inst, instruction_plan, scene


def _goto(noun: str) -> RouteLeg:
    return RouteLeg(kind=LegKind.GOTO, anchors=[Anchor(noun=noun)])


def _corridor(n1: str, n2: str) -> RouteLeg:
    return RouteLeg(kind=LegKind.CORRIDOR_BETWEEN, anchors=[Anchor(noun=n1), Anchor(noun=n2)])


def _via(noun: str) -> RouteLeg:
    return RouteLeg(kind=LegKind.VIA_NEAR, anchors=[Anchor(noun=noun)])


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
        from core.interfaces import OdomState

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


def _if_scene() -> tuple[SyntheticScene, BasicSceneIndex]:
    sc = SyntheticScene(0)
    sc.rooms = [type(sc.rooms[0])(0.0, 0.0, 8.0, 6.0)]
    sc.place_box("table", 3.0, 2.0, 0.4, 0.4, 0.5)
    sc.place_box("chair", 3.0, 4.0, 0.4, 0.4, 0.5)
    sc.place_box("sofa", 6.0, 3.0, 0.5, 0.5, 0.5)
    sc.place_box("vase", 3.0, 5.4, 0.3, 0.3, 0.4)
    return sc, BasicSceneIndex(sc.instances())


# --------------------------------------------------------------------------- grounding


def test_ungrounded_count_before_advance():
    head = InstructionHead(plan=instruction_plan([_goto("sofa"), _goto("chair")]))
    assert head.ungrounded_subgoals() == 2


def test_grounds_present_anchors():
    sc, idx = _if_scene()
    head = InstructionHead(plan=instruction_plan([_goto("sofa")]))
    io = _DriveIO(sc)
    head.advance(io, idx)
    assert head.ungrounded_subgoals() == 0


def test_ungrounded_when_anchor_absent():
    sc, idx = _if_scene()
    head = InstructionHead(plan=instruction_plan([_goto("unicorn")]))
    io = _DriveIO(sc)
    head.advance(io, idx)
    assert head.ungrounded_subgoals() == 1


def test_ungrounded_when_low_nobs():
    sc = scene(inst(1, "sofa", n_obs=1, centroid=(2.0, 2.0, 0.0)))
    head = InstructionHead(plan=instruction_plan([_goto("sofa")]))
    io = _DriveIO(SyntheticScene(0))
    head.advance(io, sc)
    assert head.ungrounded_subgoals() == 1  # present but under-observed


def test_first_anchor_pt_reported():
    sc, idx = _if_scene()
    head = InstructionHead(plan=instruction_plan([_goto("sofa")]))
    io = _DriveIO(sc)
    head.advance(io, idx)
    pt = head.first_anchor_pt()
    assert pt is not None and len(pt) == 2


def test_next_noun_affinity_target():
    sc, idx = _if_scene()
    # sofa present, unicorn absent -> affinity should target the ungrounded noun
    head = InstructionHead(plan=instruction_plan([_goto("sofa"), _goto("unicorn")]))
    io = _DriveIO(sc)
    head.advance(io, idx)
    assert head.next_noun_affinity_target() == "unicorn"


# --------------------------------------------------------------------------- driving


def _run(head: InstructionHead, io: _DriveIO, idx, max_ticks=4000):
    traj = [io.pose]
    for _ in range(max_ticks):
        head.advance(io, idx)
        io.tick_motion()
        traj.append(io.pose)
        if head.terminal_waypoint() is not None:
            tw = head.terminal_waypoint()
            if ((io.pose[0] - tw.x) ** 2 + (io.pose[1] - tw.y) ** 2) ** 0.5 < 0.3:
                break
    return np.array(traj)


@pytest.mark.slow
def test_drives_and_reaches_terminal():
    sc, idx = _if_scene()
    head = InstructionHead(plan=instruction_plan([_goto("sofa")]))
    io = _DriveIO(sc)
    traj = _run(head, io, idx)
    tw = head.terminal_waypoint()
    assert tw is not None
    assert io.waypoints  # published a stream
    d = ((traj[-1][0] - tw.x) ** 2 + (traj[-1][1] - tw.y) ** 2) ** 0.5
    assert d < 0.3


@pytest.mark.slow
def test_corridor_route_threads_gate_and_respects_avoid():
    sc, idx = _if_scene()
    head = InstructionHead(
        plan=instruction_plan(
            [_corridor("table", "chair"), _goto("sofa")],
            avoid=[AvoidSpec(near=Anchor(noun="vase"))],
        )
    )
    io = _DriveIO(sc, start=(0.7, 3.0))
    traj = _run(head, io, idx)
    gate = corridor_gate(idx.by_label("table")[0], idx.by_label("chair")[0])
    cap = avoid_capsule(AvoidSpec(near=Anchor(noun="vase")), idx)
    assert threading_check(traj, gate)[0] is True
    assert capsule_violated(traj, cap)[0] is False


def test_avoid_capsule_stamped_once():
    sc, idx = _if_scene()
    head = InstructionHead(
        plan=instruction_plan([_goto("sofa")], avoid=[AvoidSpec(near=Anchor(noun="vase"))])
    )
    io = _DriveIO(sc)
    head.advance(io, idx)  # builds route + stamps
    cm1 = head._costmap
    head.advance(io, idx)  # second tick must not rebuild
    assert head._costmap is cm1
    assert cm1.capsule_blocked.any()  # a capsule was stamped


def test_via_near_leg_drives():
    sc, idx = _if_scene()
    head = InstructionHead(plan=instruction_plan([_via("table"), _goto("sofa")]))
    io = _DriveIO(sc)
    head.advance(io, idx)
    assert head._follower is not None  # route built with a via leg


@pytest.mark.slow
def test_arrival_confirmation_checkpoint_fires():
    sc, idx = _if_scene()
    seen = []
    head = InstructionHead(
        plan=instruction_plan([_goto("sofa")]),
        anchor_confirm=lambda plan, i, summary: seen.append(i) or True,
    )
    io = _DriveIO(sc)
    _run(head, io, idx)
    assert 0 in seen  # arrival at leg 0 confirmed the anchor
