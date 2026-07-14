"""IF-F4 acceptance: for instruction-following the drive IS the answer, so the FSM must
KEEP driving the route after the answer-path publishes its first waypoint — until the
route completes (arrival / exhaustion) or the watchdog floor fires.

This is the fresh-context verifier's Priority-D probe made executable. The defect it pins:
for IF questions the controller transitioned to ANSWER at the ~270 s explore budget,
published ONE waypoint, and went DONE — the heads were never ticked again, so the second
leg of a partially-grounded route was never attempted despite ~330 s of remaining budget
and real ordered-leg partial credit. The fix inserts a DRIVE_OUT state after ANSWER (IF
only) that keeps ticking the answer heads.

Scenario for every case: a two-leg route ``go to the table then to the sofa``. Leg 1's
anchor (table) grounds from spawn; leg 2's anchor (sofa) is hidden behind a time gate so
it appears only LATER (or never). We assert:

  (i)   waypoints keep flowing AFTER the explore budget is crossed;
  (ii)  with the late-appearing sofa, leg 2 is attempted and its arrival is marked;
  (iii) with a never-appearing sofa, the watchdog floor still fires on time and the run
        terminates (DRIVE_OUT does not shadow the overlay);
  (v)   the flight recorder shows the answer->drive_out and drive_out->done transitions.

(NUMERICAL + OR unchanged is pinned by the existing tests in test_end_to_end.py — those
go ANSWER -> DONE and never enter DRIVE_OUT.)
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pytest

from core.fsm.controller import QuestionController, State
from core.heads import build_callables
from core.interfaces import InstanceRecord, WaypointCmd
from core.mocks.mock_io import FakeClock, MockRobotIO
from core.mocks.synthetic_scene import Room, SyntheticScene
from core.perception.scene_index import BasicSceneIndex

from ._scaled import BUDGET_SCALE, install_scaled_budget

# Full-controller sims: slow even budget-scaled (A* replans every tick). Fast tier skips.
pytestmark = pytest.mark.slow

TICK_DT = 0.2  # 5 Hz
DRIVE_STEP_M = 0.15  # kinematic follower step per tick
IF_EXPLORE_BUDGET_S = 270.0  # architecture §5: soft IF exploration budget


class _RevealingSceneIndex:
    """A SceneIndex that hides instances whose label is in ``hidden_labels`` until a
    per-label reveal gate opens, driven by the scene's clock.

    Models a late-appearing (or never-appearing) anchor: leg-1's anchor is visible from
    spawn; leg-2's anchor stays out of the index until ``reveal_at_s`` (None == never).
    Delegates every read to an inner BasicSceneIndex, filtering the still-hidden instances.
    """

    def __init__(self, inner: BasicSceneIndex, clock: FakeClock, gates: dict[str, float | None]):
        self._inner = inner
        self._clock = clock
        # label -> reveal time (s), or None to keep hidden for the whole run.
        self._gates = {k.lower(): v for k, v in gates.items()}

    def _visible(self, rec: InstanceRecord) -> bool:
        gate = self._gates.get(rec.label.lower())
        if gate is None and rec.label.lower() in self._gates:
            return False  # explicitly never-revealed
        if gate is None:
            return True  # not gated at all
        return self._clock.now() >= gate

    def all_instances(self) -> Sequence[InstanceRecord]:
        return [r for r in self._inner.all_instances() if self._visible(r)]

    def by_label(self, noun: str) -> Sequence[InstanceRecord]:
        return [r for r in self._inner.by_label(noun) if self._visible(r)]

    def by_label_tiered(self, noun: str):
        vis = {id(r) for r in self.by_label(noun)}
        return [r for r in self._inner.by_label_tiered(noun) if id(r) in vis]

    def marker_for(self, *a, **k):
        return self._inner.marker_for(*a, **k)


def _if_scene() -> SyntheticScene:
    sc = SyntheticScene(0)
    sc.rooms = [Room(0.0, 0.0, 10.0, 6.0)]
    # Leg 1 anchor: near spawn, grounds immediately.
    sc.place_box("table", 2.0, 3.0, 0.5, 0.5, 0.5)
    # Leg 2 anchor: across the room, gated so it appears only later (or never).
    sc.place_box("sofa", 8.0, 3.0, 0.6, 0.6, 0.5)
    return sc


def _drive_tick(io: MockRobotIO) -> None:
    """Chase the latest published waypoint one kinematic step."""
    if not io.waypoints:
        return
    wp = io.waypoints[-1]
    od = io.latest_odom()
    dx, dy = wp.x - od.x, wp.y - od.y
    d = (dx * dx + dy * dy) ** 0.5
    if d > 1e-6:
        s = min(DRIVE_STEP_M, d)
        io.set_pose(od.x + dx / d * s, od.y + dy / d * s)


def _transitions(ctrl: QuestionController) -> str:
    return "|".join(r.detail for r in ctrl.flight_recording() if r.event == "transition")


# --------------------------------------------------------------------------- (i)+(ii) late anchor


def test_if_drive_continues_past_explore_budget_and_attempts_leg_two():
    """Leg-2 anchor appears at ~300 s (after the 270 s explore budget). The FSM must keep
    driving: waypoints flow past the budget, the sofa leg is attempted once it appears, and
    the vehicle arrives at the terminal."""
    sc = _if_scene()
    clk = FakeClock(0.0)
    io = MockRobotIO(sc, clk, start_x=0.7, start_y=3.0)
    io.set_question("go to the table then go to the sofa")
    inner = BasicSceneIndex(sc.instances())
    # Sofa hidden until 300 s of SCALED budget time == 300*BUDGET_SCALE raw ticked seconds.
    scaled_reveal_raw = 300.0 * BUDGET_SCALE
    scene = _RevealingSceneIndex(inner, clk, {"sofa": scaled_reveal_raw})
    ctrl = QuestionController(**build_callables(scene))

    scaled = install_scaled_budget(io)
    max_raw_t = 600.0 * BUDGET_SCALE

    wp_after_budget = 0
    entered_drive_out = False
    while ctrl.state is not State.DONE and clk.now() < max_raw_t:
        ctrl.tick(io)
        if ctrl.state is State.DRIVE_OUT:
            entered_drive_out = True
        # count waypoints published strictly after the explore budget (scaled elapsed).
        if ctrl.budget is not None and ctrl.budget.elapsed() > IF_EXPLORE_BUDGET_S:
            wp_after_budget = max(wp_after_budget, len(io.waypoints))
        clk.advance(TICK_DT)
        _drive_tick(io)
    ctrl.tick(io)

    # (i) DRIVE_OUT was actually used, and waypoints flow past the explore budget.
    assert entered_drive_out, "IF never entered DRIVE_OUT — the drive stopped at ANSWER"
    assert wp_after_budget > 0, "no waypoints published after the 270 s explore budget"
    # (ii) leg 2 attempted: the vehicle drove toward the sofa (x >> the table's x=2.0) and
    # arrived at the terminal (within arrival tolerance of the sofa goal).
    final = io.latest_odom()
    assert final.x > 5.0, f"vehicle never progressed toward the sofa leg (x={final.x:.2f})"
    assert ctrl.state is State.DONE
    # arrival marked -> the head reported drive_complete -> DONE via 'route drive complete'.
    trail = _transitions(ctrl)
    assert "answer->drive_out" in trail
    assert "drive_out->done" in trail


# --------------------------------------------------------------------------- (iii) never anchor


def test_if_watchdog_floor_still_fires_when_terminal_never_appears():
    """Leg-2 anchor NEVER appears. DRIVE_OUT must not shadow the watchdog overlay: the
    >=540 s floor still fires on time and the run terminates with exactly one answer."""
    sc = _if_scene()
    clk = FakeClock(0.0)
    io = MockRobotIO(sc, clk, start_x=0.7, start_y=3.0)
    io.set_question("go to the table then go to the sofa")
    inner = BasicSceneIndex(sc.instances())
    scene = _RevealingSceneIndex(inner, clk, {"sofa": None})  # never revealed
    ctrl = QuestionController(**build_callables(scene))

    scaled = install_scaled_budget(io)
    floor_s = ctrl._watchdog_floor_s
    max_raw_t = 620.0 * BUDGET_SCALE

    entered_drive_out = False
    while ctrl.state is not State.DONE and clk.now() < max_raw_t:
        ctrl.tick(io)
        if ctrl.state is State.DRIVE_OUT:
            entered_drive_out = True
        clk.advance(TICK_DT)
        _drive_tick(io)
    ctrl.tick(io)

    assert entered_drive_out, "IF never entered DRIVE_OUT"
    assert ctrl.state is State.DONE, "watchdog floor did not terminate the hung drive"
    # the floor fired at/after the watchdog gate in the FSM's own (scaled) budget time.
    assert ctrl.budget.elapsed() >= floor_s
    # exactly one answer published (the ANSWER waypoint; the watchdog publish is a latched
    # no-op), and it is a legal WaypointCmd.
    assert len(io.waypoints) >= 1
    assert isinstance(io.waypoints[-1], WaypointCmd)
    events = [r.event for r in ctrl.flight_recording()]
    assert "done" in events


# --------------------------------------------------------------------------- (v) recorder


def test_if_recorder_shows_drive_out_transitions():
    """The flight recorder records the new answer->drive_out and drive_out->done state
    transitions (following the existing 'transition' event convention)."""
    sc = _if_scene()
    clk = FakeClock(0.0)
    io = MockRobotIO(sc, clk, start_x=0.7, start_y=3.0)
    io.set_question("go to the table then go to the sofa")
    inner = BasicSceneIndex(sc.instances())
    # Reveal the sofa early so the drive completes quickly and both transitions appear.
    scene = _RevealingSceneIndex(inner, clk, {"sofa": 5.0 * BUDGET_SCALE})
    ctrl = QuestionController(**build_callables(scene))

    install_scaled_budget(io)
    max_raw_t = 620.0 * BUDGET_SCALE
    while ctrl.state is not State.DONE and clk.now() < max_raw_t:
        ctrl.tick(io)
        clk.advance(TICK_DT)
        _drive_tick(io)
    ctrl.tick(io)

    trail = _transitions(ctrl)
    assert "answer->drive_out" in trail, trail
    assert "drive_out->done" in trail, trail
    # ordering: answer precedes drive_out precedes done.
    assert trail.index("answer->drive_out") < trail.index("drive_out->done")
