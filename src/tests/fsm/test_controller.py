"""QuestionController FSM: happy paths, watchdog in every state, forced assembly,
ledger/floor interaction, dedup, early-answer gate, publish-once latching."""
from __future__ import annotations

import pytest

from core.fsm.controller import QuestionController, StabilitySignal, State, WorldView
from core.fsm.floors import PartialResults
from core.interfaces import IntAnswer, MarkerBox, Question, QType, WaypointCmd
from core.plan_schema import Plan, TargetSpec
from tests.fsm._fakes import FakeClock, FakeRobotIO, FakeScene, make_instance


# --------------------------------------------------------------------------- stubs


def make_plan(qtype: QType, noun: str = "chair") -> Plan:
    if qtype is QType.INSTRUCTION_FOLLOWING:
        from core.plan_schema import Anchor, LegKind, RouteLeg

        return Plan(
            qtype=qtype,
            question_raw="",
            route=[RouteLeg(kind=LegKind.GOTO, anchors=[Anchor(noun=noun)])],
        )
    return Plan(qtype=qtype, question_raw="", target=TargetSpec(noun=noun))


def build_controller(
    *,
    qtype: QType,
    world: WorldView | None = None,
    verify_answer=None,
    parse_plan=None,
):
    """Wire a controller with simple stubs. Returns (controller, calls-dict)."""
    calls = {"parse": 0, "explore": 0, "verify": 0}
    the_world = world if world is not None else WorldView(scene=FakeScene([]))

    def parse(q):
        calls["parse"] += 1
        return parse_plan if parse_plan is not None else make_plan(qtype)

    def explore(io, plan, w):
        calls["explore"] += 1

    def verify(io, plan, w):
        calls["verify"] += 1
        return verify_answer

    def probe(io):
        return the_world

    ctrl = QuestionController(parse=parse, explore=explore, verify=verify, probe=probe)
    return ctrl, calls


def q_of(qtype: QType, text: str | None = None, t: float = 0.0) -> Question:
    default = {
        QType.NUMERICAL: "how many chairs are there",
        QType.OBJECT_REFERENCE: "the chair near the window",
        QType.INSTRUCTION_FOLLOWING: "go to the chair",
    }[qtype]
    return Question(text=text or default, t_received=t)


def run_until_done(ctrl, io, clk, *, step=0.5, max_t=620.0, advance=True):
    """Tick at ~2 Hz advancing the clock until DONE or timeout."""
    while ctrl.state is not State.DONE and clk.now() <= max_t:
        ctrl.tick(io)
        if advance:
            clk.advance(step)
    ctrl.tick(io)  # let DONE finalize the flight recording
    return ctrl


# --------------------------------------------------------------------------- happy paths


@pytest.mark.parametrize("qtype", list(QType))
def test_happy_path_publishes_exactly_one_legal_answer(qtype):
    clk = FakeClock(0.0)
    scene = FakeScene([make_instance(1, "chair"), make_instance(2, "chair")])
    # For IF, provide an anchor point so the floor/verify has something real.
    world = WorldView(scene=scene, partial=PartialResults(first_anchor_pt=(1.0, 2.0, 0.0)))
    ctrl, calls = build_controller(qtype=qtype, world=world)
    io = FakeRobotIO(clk, q_of(qtype))

    run_until_done(ctrl, io, clk)

    assert ctrl.state is State.DONE
    assert io.publish_count == 1
    if qtype is QType.NUMERICAL:
        assert len(io.published_ints) == 1
        assert isinstance(io.published_ints[0], IntAnswer)
    elif qtype is QType.OBJECT_REFERENCE:
        assert len(io.published_markers) == 1
    else:
        assert len(io.published_waypoints) == 1


def test_happy_path_visits_states_in_order():
    clk = FakeClock(0.0)
    ctrl, _ = build_controller(qtype=QType.OBJECT_REFERENCE, world=WorldView(scene=FakeScene([make_instance(1, "chair")])))
    io = FakeRobotIO(clk, q_of(QType.OBJECT_REFERENCE))
    run_until_done(ctrl, io, clk)
    # Some transitions are transient within one tick (parse runs concurrently with
    # orientation), so read the ordered transition trail from the flight recorder.
    trail = "|".join(
        r.detail for r in ctrl.flight_recording() if r.event == "transition"
    )
    for target in ("->parsing", "->orient", "->explore_execute", "->verify", "->answer", "->done"):
        assert target in trail, f"{target} missing from transition trail: {trail}"
    # ordering: each target appears in sequence
    order = ["->parsing", "->orient", "->explore_execute", "->verify", "->answer", "->done"]
    positions = [trail.index(t) for t in order]
    assert positions == sorted(positions)


# --------------------------------------------------------------------------- watchdog


@pytest.mark.parametrize("stuck", [State.PARSING, State.ORIENT, State.EXPLORE_EXECUTE, State.VERIFY])
@pytest.mark.parametrize("qtype", list(QType))
def test_watchdog_fires_at_floor_in_every_state(stuck, qtype):
    """Freeze the FSM in `stuck`, jump the clock past the floor -> one floor answer, DONE.

    Boundaries are read from the controller's effective watchdog gate (pulled in to hedge the
    evaluator-clock skew, SYS-F6) rather than hard-coded, so the test tracks the real default.
    """
    clk = FakeClock(0.0)
    world = WorldView(scene=FakeScene([make_instance(1, "chair")]), partial=PartialResults(first_anchor_pt=(1.0, 1.0, 0.0)))
    ctrl, _ = build_controller(qtype=qtype, world=world)
    io = FakeRobotIO(clk, q_of(qtype))
    floor_s = ctrl._watchdog_floor_s

    # Advance normally but pin the state by monkeypatching the handler to a no-op that
    # also parks the state, EXCEPT we simply drive a couple ticks then force the state.
    ctrl.tick(io)  # intake -> PARSING
    ctrl.state = stuck  # jam the FSM in the target state
    # keep it there right up to the floor
    clk.set(floor_s - 1.0)
    ctrl.tick(io)
    assert io.publish_count == 0  # not yet
    # now cross the watchdog floor
    clk.set(floor_s + 1.0)
    ctrl.tick(io)

    assert ctrl.state is State.DONE
    assert io.publish_count == 1
    # legal answer of the right type
    if qtype is QType.NUMERICAL:
        assert len(io.published_ints) == 1
    elif qtype is QType.OBJECT_REFERENCE:
        assert len(io.published_markers) == 1
    else:
        assert len(io.published_waypoints) == 1


def test_watchdog_publishes_only_once_even_if_ticked_again():
    clk = FakeClock(0.0)
    ctrl, _ = build_controller(qtype=QType.NUMERICAL, world=WorldView(scene=FakeScene([])))
    io = FakeRobotIO(clk, q_of(QType.NUMERICAL))
    ctrl.tick(io)
    clk.set(580.0)
    ctrl.tick(io)
    assert io.publish_count == 1
    for _ in range(5):
        ctrl.tick(io)
    assert io.publish_count == 1  # still one


# --------------------------------------------------------------------------- forced assembly


def test_forced_assembly_forces_verify_at_gate():
    clk = FakeClock(0.0)
    ctrl, calls = build_controller(qtype=QType.OBJECT_REFERENCE, world=WorldView(scene=FakeScene([make_instance(1, "chair")])))
    io = FakeRobotIO(clk, q_of(QType.OBJECT_REFERENCE))
    fa_s, floor_s = ctrl._forced_assembly_s, ctrl._watchdog_floor_s
    ctrl.tick(io)  # -> PARSING
    ctrl.state = State.EXPLORE_EXECUTE  # pretend still exploring past forced-assembly
    clk.set(fa_s + 1.0)
    ctrl.tick(io)  # forced_assembly should push toward VERIFY
    assert ctrl.state in (State.VERIFY, State.ANSWER, State.DONE)
    # ends with a published answer once we keep ticking (still before the watchdog floor)
    clk.set(0.5 * (fa_s + floor_s))
    for _ in range(4):
        ctrl.tick(io)
    assert io.publish_count == 1


# --------------------------------------------------------------------------- ledger vs floor


def test_ledger_exhaustion_blocks_verify_but_floor_still_answers():
    clk = FakeClock(0.0)
    # verify would return a marker, but we exhaust the verification cap first.
    ret = MarkerBox(9, 9, 9, 1, 1, 1, label="verified")
    ctrl, calls = build_controller(qtype=QType.OBJECT_REFERENCE, world=WorldView(scene=FakeScene([make_instance(1, "chair", centroid=(2, 2, 2))])), verify_answer=ret)
    io = FakeRobotIO(clk, q_of(QType.OBJECT_REFERENCE))
    ctrl.tick(io)
    # pre-consume the verification checkpoint
    ctrl.ledger.record("verification", 0.0, "api")
    assert ctrl.ledger.allow("verification") is False
    ctrl.state = State.VERIFY
    clk.set(300.0)
    for _ in range(4):
        ctrl.tick(io)
    # verify callable never invoked (blocked), but a legal floor marker still published
    assert calls["verify"] == 0
    assert len(io.published_markers) == 1


# --------------------------------------------------------------------------- dedup / latch


def test_question_republish_same_text_ignored():
    clk = FakeClock(0.0)
    ctrl, _ = build_controller(qtype=QType.NUMERICAL, world=WorldView(scene=FakeScene([])))
    io = FakeRobotIO(clk, q_of(QType.NUMERICAL, t=0.0))
    ctrl.tick(io)  # latch
    latched_t0 = ctrl.budget.t0
    # simulate 1 Hz republish of identical text at a later receipt time
    io.set_question(q_of(QType.NUMERICAL, t=1.0))
    clk.set(1.0)
    ctrl.tick(io)
    assert ctrl.budget.t0 == latched_t0  # t0 unchanged; not re-latched
    assert ctrl.question.t_received == 0.0  # kept the first receipt


def test_different_text_midrun_is_ignored_not_switched():
    clk = FakeClock(0.0)
    ctrl, _ = build_controller(qtype=QType.NUMERICAL, world=WorldView(scene=FakeScene([])))
    io = FakeRobotIO(clk, q_of(QType.NUMERICAL))
    ctrl.tick(io)
    original = ctrl.question.text
    io.set_question(Question(text="a totally different question", t_received=2.0))
    clk.set(2.0)
    ctrl.tick(io)
    assert ctrl.question.text == original


# --------------------------------------------------------------------------- early-answer gate


def test_numerical_early_answer_when_stable():
    clk = FakeClock(0.0)
    scene = FakeScene([make_instance(1, "chair"), make_instance(2, "chair")])
    world = WorldView(scene=scene, stability=StabilitySignal(winner_margin=0.4, min_contrib_n_obs=3))
    ctrl, _ = build_controller(qtype=QType.NUMERICAL, world=world)
    io = FakeRobotIO(clk, q_of(QType.NUMERICAL))
    # get into EXPLORE well before the explore budget (210 s)
    ctrl.tick(io)  # PARSING
    ctrl.state = State.EXPLORE_EXECUTE
    clk.set(70.0)  # past orientation, far from explore budget
    ctrl.tick(io)  # early gate should open -> VERIFY
    assert ctrl.state in (State.VERIFY, State.ANSWER, State.DONE)
    for _ in range(4):
        ctrl.tick(io)
    assert io.publish_count == 1
    assert clk.now() < 210.0  # answered early, before the explore budget


def test_numerical_no_early_answer_when_margin_unstable():
    clk = FakeClock(0.0)
    scene = FakeScene([make_instance(1, "chair")])
    world = WorldView(scene=scene, stability=StabilitySignal(winner_margin=0.10, min_contrib_n_obs=5))
    ctrl, _ = build_controller(qtype=QType.NUMERICAL, world=world)
    io = FakeRobotIO(clk, q_of(QType.NUMERICAL))
    ctrl.tick(io)
    ctrl.state = State.EXPLORE_EXECUTE
    clk.set(70.0)
    ctrl.tick(io)
    assert ctrl.state is State.EXPLORE_EXECUTE  # margin < 25% -> not early
    assert io.publish_count == 0


def test_numerical_no_early_answer_when_nobs_too_low():
    clk = FakeClock(0.0)
    world = WorldView(scene=FakeScene([make_instance(1, "chair")]), stability=StabilitySignal(winner_margin=0.9, min_contrib_n_obs=2))
    ctrl, _ = build_controller(qtype=QType.NUMERICAL, world=world)
    io = FakeRobotIO(clk, q_of(QType.NUMERICAL))
    ctrl.tick(io)
    ctrl.state = State.EXPLORE_EXECUTE
    clk.set(70.0)
    ctrl.tick(io)
    assert ctrl.state is State.EXPLORE_EXECUTE  # n_obs < 3 -> not early


def test_if_never_early_with_ungrounded_subgoal():
    clk = FakeClock(0.0)
    world = WorldView(
        scene=FakeScene([make_instance(1, "chair")]),
        ungrounded_subgoals=1,
        # even if stability looks great, IF must not early-answer
        stability=StabilitySignal(winner_margin=1.0, min_contrib_n_obs=9),
    )
    ctrl, _ = build_controller(qtype=QType.INSTRUCTION_FOLLOWING, world=world)
    io = FakeRobotIO(clk, q_of(QType.INSTRUCTION_FOLLOWING))
    ctrl.tick(io)
    ctrl.state = State.EXPLORE_EXECUTE
    clk.set(100.0)  # well before IF explore budget (270 s)
    ctrl.tick(io)
    assert ctrl.state is State.EXPLORE_EXECUTE
    assert io.publish_count == 0


# --------------------------------------------------------------------------- publish-once


def test_publish_once_latches_across_answer_and_watchdog():
    clk = FakeClock(0.0)
    ctrl, _ = build_controller(qtype=QType.NUMERICAL, world=WorldView(scene=FakeScene([make_instance(1, "chair")])))
    io = FakeRobotIO(clk, q_of(QType.NUMERICAL))
    # normal answer path
    run_until_done(ctrl, io, clk)
    assert io.publish_count == 1
    assert ctrl.answer_published is True
    # force a late watchdog tick — must not double publish
    clk.set(590.0)
    ctrl.state = State.EXPLORE_EXECUTE  # pretend we somehow re-entered
    ctrl.tick(io)
    assert io.publish_count == 1


# --------------------------------------------------------------------------- misc


def test_no_question_stays_idle():
    clk = FakeClock(0.0)
    ctrl, _ = build_controller(qtype=QType.NUMERICAL)
    io = FakeRobotIO(clk, question=None)
    for _ in range(3):
        ctrl.tick(io)
        clk.advance(1.0)
    assert ctrl.state is State.IDLE
    assert io.publish_count == 0


def test_flight_recording_dumped_on_done():
    clk = FakeClock(0.0)
    ctrl, _ = build_controller(qtype=QType.NUMERICAL, world=WorldView(scene=FakeScene([])))
    io = FakeRobotIO(clk, q_of(QType.NUMERICAL))
    run_until_done(ctrl, io, clk)
    rec = ctrl.flight_recording()
    events = [r.event for r in rec]
    assert "question_latched" in events
    assert "published" in events
    assert "done" in events


# --------------------------------------------------------------------------- defect 1:
# verify returning an undispatchable type must NOT defeat the exactly-one-answer invariant


class _NoneyWrapper:
    """A truthy (not-None) object of a type _dispatch does not recognize."""

    def __repr__(self) -> str:
        return "<NoneyWrapper>"


import numpy as _np  # noqa: E402 — colocated with the defect-1 params below

_UNDISPATCHABLE = [
    pytest.param({"answer": 5}, id="dict"),
    pytest.param((1, 2, 3), id="tuple"),
    pytest.param(_np.int64(7), id="numpy_scalar"),
    pytest.param(_NoneyWrapper(), id="none_typed_wrapper"),
]


@pytest.mark.parametrize("bad_answer", _UNDISPATCHABLE)
@pytest.mark.parametrize("qtype", list(QType))
def test_verify_undispatchable_type_still_publishes_exactly_one_floor(qtype, bad_answer):
    """verify returns a non-None value of an illegal type -> the pending answer is dropped
    and the floor is published instead: exactly ONE legal answer, never zero, never two."""
    clk = FakeClock(0.0)
    scene = FakeScene([make_instance(1, "chair"), make_instance(2, "chair")])
    world = WorldView(scene=scene, partial=PartialResults(first_anchor_pt=(1.0, 2.0, 0.0)))
    ctrl, _ = build_controller(qtype=qtype, world=world, verify_answer=bad_answer)
    io = FakeRobotIO(clk, q_of(qtype))

    run_until_done(ctrl, io, clk)

    assert ctrl.state is State.DONE
    assert ctrl.answer_published is True
    assert io.publish_count == 1  # never zero, never two
    if qtype is QType.NUMERICAL:
        assert len(io.published_ints) == 1
        assert isinstance(io.published_ints[0], IntAnswer)
    elif qtype is QType.OBJECT_REFERENCE:
        assert len(io.published_markers) == 1
        assert isinstance(io.published_markers[0], MarkerBox)
    else:
        assert len(io.published_waypoints) == 1
        assert isinstance(io.published_waypoints[0], WaypointCmd)
    # flight recorder logged the discard
    events = [r.event for r in ctrl.flight_recording()]
    assert "publish_dropped" in events


@pytest.mark.parametrize("qtype", list(QType))
def test_verify_undispatchable_never_starves_watchdog_floor(qtype):
    """If forced assembly / ANSWER somehow did not clear it, the 570 s watchdog floor still
    fires exactly one legal answer when verify returned garbage — the run is never poisoned."""
    clk = FakeClock(0.0)
    world = WorldView(
        scene=FakeScene([make_instance(1, "chair")]),
        partial=PartialResults(first_anchor_pt=(1.0, 1.0, 0.0)),
    )
    ctrl, _ = build_controller(qtype=qtype, world=world, verify_answer={"answer": 5})
    io = FakeRobotIO(clk, q_of(qtype))
    ctrl.tick(io)  # intake -> PARSING
    # jam a poisoned pending answer and pin VERIFY just under the floor
    ctrl._pending_answer = {"answer": 5}
    ctrl.state = State.VERIFY
    clk.set(571.0)
    ctrl.tick(io)
    assert ctrl.state is State.DONE
    assert io.publish_count == 1


def test_happy_path_verify_answer_unchanged():
    """A legal verify answer is still published as-is (happy path unchanged)."""
    clk = FakeClock(0.0)
    ret = MarkerBox(9, 9, 9, 1, 1, 1, label="verified")
    world = WorldView(
        scene=FakeScene([make_instance(1, "chair", centroid=(2, 2, 2))]),
        partial=PartialResults(first_anchor_pt=(2.0, 2.0, 2.0)),
    )
    ctrl, _ = build_controller(qtype=QType.OBJECT_REFERENCE, world=world, verify_answer=ret)
    io = FakeRobotIO(clk, q_of(QType.OBJECT_REFERENCE))
    run_until_done(ctrl, io, clk)
    assert io.publish_count == 1
    assert io.published_markers == [ret]


# --------------------------------------------------------------------------- qtype correction
# The intake seed is a regex guess; a successful parse carrying a different QType must correct
# controller.qtype AND budget.qtype so floor selection + budget gates use the authoritative type.


def test_qtype_corrected_from_plan_overrides_intake_seed():
    """A motion-verbed OR question seeds IF at intake; the OR plan must correct it."""
    from core.fsm.controller import _infer_qtype

    clk = FakeClock(0.0)
    # "walk past" makes _infer_qtype seed INSTRUCTION_FOLLOWING...
    q = Question(text="the picture you walk past near the door", t_received=0.0)
    assert _infer_qtype(q) is QType.INSTRUCTION_FOLLOWING  # the seed genuinely misfires
    # ...but the parse produces an object-reference plan.
    or_plan = make_plan(QType.OBJECT_REFERENCE, "picture")
    ctrl, _ = build_controller(
        qtype=QType.OBJECT_REFERENCE,  # governs stub verify/floor typing
        world=WorldView(scene=FakeScene([make_instance(1, "picture")])),
        parse_plan=or_plan,
    )
    io = FakeRobotIO(clk, q)
    ctrl.tick(io)  # intake seeds IF, then parse runs in the same tick and corrects to OR
    assert ctrl.qtype is QType.OBJECT_REFERENCE
    assert ctrl.budget.qtype is QType.OBJECT_REFERENCE
    events = [r.event for r in ctrl.flight_recording()]
    assert "qtype_corrected" in events
    # the recorded correction names both the seed and the corrected type
    detail = next(r.detail for r in ctrl.flight_recording() if r.event == "qtype_corrected")
    assert "instruction_following" in detail and "object_reference" in detail


def test_qtype_not_logged_when_plan_matches_seed():
    """When the parse agrees with the seed, no correction event is emitted."""
    clk = FakeClock(0.0)
    plan = make_plan(QType.NUMERICAL, "chair")
    ctrl, _ = build_controller(
        qtype=QType.NUMERICAL,
        world=WorldView(scene=FakeScene([])),
        parse_plan=plan,
    )
    io = FakeRobotIO(clk, q_of(QType.NUMERICAL))
    ctrl.tick(io)
    ctrl.tick(io)
    assert ctrl.qtype is QType.NUMERICAL
    events = [r.event for r in ctrl.flight_recording()]
    assert "qtype_corrected" not in events


def test_corrected_qtype_drives_floor_selection():
    """After correction the watchdog floor publishes the CORRECTED type's answer, not the seed's.

    Seed IF would floor a WaypointCmd on the wrong topic; the OR correction must floor a Marker.
    """
    clk = FakeClock(0.0)
    q = Question(text="the picture you walk past", t_received=0.0)
    or_plan = make_plan(QType.OBJECT_REFERENCE, "picture")
    world = WorldView(
        scene=FakeScene([make_instance(1, "picture", centroid=(2, 2, 2))]),
        partial=PartialResults(first_anchor_pt=(2.0, 2.0, 2.0)),
    )
    ctrl, _ = build_controller(qtype=QType.OBJECT_REFERENCE, world=world, parse_plan=or_plan)
    io = FakeRobotIO(clk, q)
    ctrl.tick(io)  # intake seeds IF, parse corrects to OR in the same tick
    assert ctrl.qtype is QType.OBJECT_REFERENCE
    clk.set(ctrl._watchdog_floor_s + 1.0)  # cross the effective watchdog floor
    ctrl.tick(io)
    assert ctrl.state is State.DONE
    assert len(io.published_markers) == 1  # OR floor, not an IF waypoint
    assert io.publish_count == 1


def test_parse_stub_invoked_and_plan_stored():
    clk = FakeClock(0.0)
    plan = make_plan(QType.NUMERICAL, "chair")
    ctrl, calls = build_controller(qtype=QType.NUMERICAL, world=WorldView(scene=FakeScene([])), parse_plan=plan)
    io = FakeRobotIO(clk, q_of(QType.NUMERICAL))
    ctrl.tick(io)  # intake -> PARSING
    ctrl.tick(io)  # parse runs
    assert calls["parse"] >= 1
    assert ctrl.plan is plan
