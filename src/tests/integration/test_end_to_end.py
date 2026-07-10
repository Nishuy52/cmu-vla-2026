"""End-to-end: the full QuestionController on MockRobotIO + SyntheticScene at 5 Hz.

The payoff test (architecture §2 pipeline, §4 per-type strategies). Each case runs a real
QuestionController wired by ``core.heads.build_callables`` over a live BasicSceneIndex, with
a FakeClock ticked at 5 Hz and a kinematic follower that chases the published waypoints so
the instruction-following case produces an actual driven trajectory to check the geometry
oracles against.

Parse is the deterministic regex tier (offline), with training-question phrasings adapted
to the synthetic scene's objects. LLM checkpoints are absent (deterministic path).
"""
from __future__ import annotations

import numpy as np

from core.fsm.controller import QuestionController, State
from core.geometry.toolbox import (
    avoid_capsule,
    capsule_violated,
    corridor_gate,
    threading_check,
)
from core.heads import build_callables
from core.interfaces import IntAnswer, MarkerBox, WaypointCmd
from core.mocks.mock_io import FakeClock, MockRobotIO
from core.mocks.synthetic_scene import Room, SyntheticScene
from core.perception.scene_index import BasicSceneIndex
from core.plan_schema import Anchor, AvoidSpec

TICK_DT = 0.2  # 5 Hz
DRIVE_STEP_M = 0.1  # kinematic follower step per tick


def _run(io: MockRobotIO, ctrl: QuestionController, clk: FakeClock, *, drive: bool,
         max_t: float = 600.0):
    """Tick the controller at 5 Hz; optionally chase waypoints. Return the trajectory."""
    traj = [(io.latest_odom().x, io.latest_odom().y)]
    while ctrl.state is not State.DONE and clk.now() < max_t:
        ctrl.tick(io)
        clk.advance(TICK_DT)
        if drive and io.waypoints:
            wp = io.waypoints[-1]
            od = io.latest_odom()
            dx, dy = wp.x - od.x, wp.y - od.y
            d = (dx * dx + dy * dy) ** 0.5
            s = min(DRIVE_STEP_M, d)
            if d > 1e-6:
                io.set_pose(od.x + dx / d * s, od.y + dy / d * s)
            traj.append((io.latest_odom().x, io.latest_odom().y))
    ctrl.tick(io)  # finalize DONE
    return np.array(traj)


# --------------------------------------------------------------------------- (a) numerical


def test_numerical_publishes_correct_int_before_watchdog():
    sc = SyntheticScene(0)
    sc.place_box("chair", 1.0, 1.0)
    sc.place_box("chair", 3.0, 3.0)
    sc.place_box("table", 2.0, 4.0)
    idx = BasicSceneIndex(sc.instances())
    clk = FakeClock(0.0)
    io = MockRobotIO(sc, clk)
    io.set_question("how many chairs are there")
    ctrl = QuestionController(**build_callables(idx))

    _run(io, ctrl, clk, drive=False)

    assert ctrl.state is State.DONE
    assert io.ints == [IntAnswer(2)]
    assert clk.now() < 570.0  # before the watchdog floor


# --------------------------------------------------------------------------- (b) object-ref


def test_object_reference_publishes_correct_marker():
    sc = SyntheticScene(0)
    sc.place_box("table", 1.0, 1.0, 0.6, 0.6, 0.5)
    sc.place_box("lamp", 1.5, 1.0, 0.2, 0.2, 0.3)
    sc.place_box("table", 4.0, 4.0, 0.6, 0.6, 0.5)  # decoy far table
    idx = BasicSceneIndex(sc.instances())
    clk = FakeClock(0.0)
    io = MockRobotIO(sc, clk)
    io.set_question("find the table near the lamp")
    ctrl = QuestionController(**build_callables(idx))

    _run(io, ctrl, clk, drive=False)

    assert ctrl.state is State.DONE
    assert len(io.markers) == 1
    m = io.markers[0]
    assert isinstance(m, MarkerBox)
    # the near table (1,1), not the decoy at (4,4)
    assert (round(m.cx, 3), round(m.cy, 3)) == (1.0, 1.0)


# --------------------------------------------------------------------------- (c) instruction


def _if_scene() -> tuple[SyntheticScene, BasicSceneIndex]:
    sc = SyntheticScene(0)
    sc.rooms = [Room(0.0, 0.0, 8.0, 6.0)]
    sc.place_box("table", 3.0, 2.0, 0.4, 0.4, 0.5)
    sc.place_box("chair", 3.0, 4.0, 0.4, 0.4, 0.5)
    sc.place_box("sofa", 6.0, 3.0, 0.5, 0.5, 0.5)
    sc.place_box("vase", 3.0, 5.4, 0.3, 0.3, 0.4)  # avoid, off-route
    return sc, BasicSceneIndex(sc.instances())


def test_instruction_following_threads_corridor_avoids_capsule_and_publishes_terminal():
    sc, idx = _if_scene()
    clk = FakeClock(0.0)
    io = MockRobotIO(sc, clk, start_x=0.7, start_y=3.0)
    io.set_question(
        "take the path between the table and the chair to the sofa, avoid the vase"
    )
    ctrl = QuestionController(**build_callables(idx))

    traj = _run(io, ctrl, clk, drive=True)

    assert ctrl.state is State.DONE
    # waypoints streamed
    assert len(io.waypoints) > 1
    # terminal waypoint published as the IF answer
    assert isinstance(io.waypoints[-1], WaypointCmd)
    # threading_check passes on the driven trajectory
    gate = corridor_gate(idx.by_label("table")[0], idx.by_label("chair")[0])
    threaded, _ = threading_check(traj, gate)
    assert threaded is True
    # capsule never violated (geometry.capsule_violated as oracle)
    cap = avoid_capsule(AvoidSpec(near=Anchor(noun="vase")), idx)
    violated, _ = capsule_violated(traj, cap)
    assert violated is False


# --------------------------------------------------------------------------- (d) pathological


def test_absent_target_object_reference_falls_to_floor_never_silent():
    sc = SyntheticScene(0)
    sc.place_box("table", 2.0, 2.0)
    idx = BasicSceneIndex(sc.instances())
    clk = FakeClock(0.0)
    io = MockRobotIO(sc, clk)
    io.set_question("find the unicorn near the window")
    ctrl = QuestionController(**build_callables(idx))

    _run(io, ctrl, clk, drive=False)

    assert ctrl.state is State.DONE
    # never silence: a legal floor marker is still published
    assert len(io.markers) == 1
    assert isinstance(io.markers[0], MarkerBox)


def test_absent_target_numerical_answers_zero_never_silent():
    sc = SyntheticScene(0)
    sc.place_box("table", 2.0, 2.0)
    idx = BasicSceneIndex(sc.instances())
    clk = FakeClock(0.0)
    io = MockRobotIO(sc, clk)
    io.set_question("how many unicorns are there")
    ctrl = QuestionController(**build_callables(idx))

    _run(io, ctrl, clk, drive=False)

    assert ctrl.state is State.DONE
    assert len(io.ints) == 1  # a legal integer, never silence
    assert io.ints[0] == IntAnswer(0)
