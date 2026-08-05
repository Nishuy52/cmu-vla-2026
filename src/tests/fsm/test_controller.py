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


# --------------------------------------------------------------------------- orient sweep (#80)


def test_orient_ticks_explore_callable():
    """#80: ORIENT must actually run the sweep, not sit parked for 60 s."""
    clk = FakeClock(0.0)
    ctrl, calls = build_controller(
        qtype=QType.OBJECT_REFERENCE, world=WorldView(scene=FakeScene([make_instance(1, "chair")]))
    )
    io = FakeRobotIO(clk, q_of(QType.OBJECT_REFERENCE))
    ctrl.tick(io)  # -> PARSING (then -> ORIENT same tick)
    assert ctrl.state is State.ORIENT
    before = calls["explore"]
    for _ in range(3):
        clk.advance(1.0)
        ctrl.tick(io)
    assert ctrl.state is State.ORIENT  # still well inside the 60 s window
    assert calls["explore"] > before  # explore was ticked each ORIENT step


def test_orient_to_explore_transition_still_at_60s():
    """Explore ticking during ORIENT must not change the 60 s window boundary."""
    clk = FakeClock(0.0)
    ctrl, _ = build_controller(
        qtype=QType.OBJECT_REFERENCE, world=WorldView(scene=FakeScene([make_instance(1, "chair")]))
    )
    io = FakeRobotIO(clk, q_of(QType.OBJECT_REFERENCE))
    ctrl.tick(io)  # -> PARSING -> ORIENT
    clk.set(59.0)
    ctrl.tick(io)
    assert ctrl.state is State.ORIENT
    clk.set(61.0)
    ctrl.tick(io)
    assert ctrl.state is State.EXPLORE_EXECUTE


def test_orient_explore_tick_does_not_disturb_watchdog_or_forced_assembly():
    """Watchdog/forced-assembly overlays fire unchanged whether jammed in ORIENT or not,
    now that ORIENT itself does explore work each tick."""
    clk = FakeClock(0.0)
    world = WorldView(
        scene=FakeScene([make_instance(1, "chair")]),
        partial=PartialResults(first_anchor_pt=(1.0, 1.0, 0.0)),
    )
    ctrl, _ = build_controller(qtype=QType.OBJECT_REFERENCE, world=world)
    io = FakeRobotIO(clk, q_of(QType.OBJECT_REFERENCE))
    floor_s = ctrl._watchdog_floor_s
    ctrl.tick(io)  # -> PARSING -> ORIENT
    ctrl.state = State.ORIENT  # jam it there
    clk.set(floor_s - 1.0)
    ctrl.tick(io)
    assert io.publish_count == 0
    clk.set(floor_s + 1.0)
    ctrl.tick(io)
    assert ctrl.state is State.DONE
    assert io.publish_count == 1
    assert len(io.published_markers) == 1


def test_orient_explore_tick_does_not_disturb_forced_assembly():
    clk = FakeClock(0.0)
    ctrl, _ = build_controller(
        qtype=QType.OBJECT_REFERENCE, world=WorldView(scene=FakeScene([make_instance(1, "chair")]))
    )
    io = FakeRobotIO(clk, q_of(QType.OBJECT_REFERENCE))
    fa_s = ctrl._forced_assembly_s
    ctrl.tick(io)  # -> PARSING -> ORIENT
    ctrl.state = State.ORIENT  # pretend still orienting past forced-assembly
    clk.set(fa_s + 1.0)
    ctrl.tick(io)
    assert ctrl.state in (State.VERIFY, State.ANSWER, State.DONE)


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


def test_numerical_keeps_exploring_past_old_270s_budget():
    """#150: NUMERICAL's soft explore budget was raised 210 -> 450 s (well past the old
    INSTRUCTION_FOLLOWING-level 270 s ceiling other qtypes still use) because NUMERICAL
    answers in place with no drive-out phase, so extra exploration time is never traded
    away -- it only buys more sensing coverage before the effective forced-assembly gate.
    An unstable count must still be exploring at 300 s (past the OLD 210/270 s budgets,
    under the NEW 450 s one) and only get forced to VERIFY once the new budget is spent.
    """
    clk = FakeClock(0.0)
    scene = FakeScene([make_instance(1, "chair")])
    # Unstable margin (< 25%): never qualifies for the early-answer gate, so only the
    # soft explore budget (or forced assembly) can move EXPLORE_EXECUTE -> VERIFY.
    world = WorldView(scene=scene, stability=StabilitySignal(winner_margin=0.10, min_contrib_n_obs=5))
    ctrl, _ = build_controller(qtype=QType.NUMERICAL, world=world)
    io = FakeRobotIO(clk, q_of(QType.NUMERICAL))
    ctrl.tick(io)  # -> PARSING
    ctrl.state = State.EXPLORE_EXECUTE

    from core.interfaces import EXPLORE_BUDGET_S

    budget_s = EXPLORE_BUDGET_S[QType.NUMERICAL]
    assert budget_s == 450.0  # pins the #150 value this test exercises

    # Past every qtype's OLD budget (210/240/270) and past the other qtypes' current
    # budgets too, but still short of NUMERICAL's new 450 s budget and the 480 s
    # forced-assembly gate: exploration must still be running.
    clk.set(300.0)
    ctrl.tick(io)
    assert ctrl.state is State.EXPLORE_EXECUTE
    assert io.publish_count == 0

    # Cross the new budget -> forced into VERIFY (then answers from the floor/verify path).
    clk.set(budget_s + 1.0)
    ctrl.tick(io)
    assert ctrl.state in (State.VERIFY, State.ANSWER, State.DONE)
    for _ in range(4):
        ctrl.tick(io)
    assert io.publish_count == 1


def test_numerical_early_answer_still_fires_once_stable_within_raised_budget():
    """The raised NUMERICAL budget must not delay an already-stable count: the
    stability-based early-answer gate (`_early_answer_ready`) still exits
    EXPLORE_EXECUTE immediately, well inside the new 450 s ceiling, exactly as it did
    inside the old 210 s one (architecture §1 row 8 / #150's "costs nothing when the
    room is small" claim)."""
    clk = FakeClock(0.0)
    scene = FakeScene([make_instance(1, "chair"), make_instance(2, "chair")])
    world = WorldView(scene=scene, stability=StabilitySignal(winner_margin=0.4, min_contrib_n_obs=3))
    ctrl, _ = build_controller(qtype=QType.NUMERICAL, world=world)
    io = FakeRobotIO(clk, q_of(QType.NUMERICAL))
    ctrl.tick(io)  # -> PARSING
    ctrl.state = State.EXPLORE_EXECUTE
    clk.set(70.0)  # far below both the old (210) and new (450) explore budgets
    ctrl.tick(io)  # stability is met -> early gate fires regardless of the raised budget
    assert ctrl.state in (State.VERIFY, State.ANSWER, State.DONE)
    for _ in range(4):
        ctrl.tick(io)
    assert io.publish_count == 1
    assert clk.now() < 210.0  # answered early, well before even the OLD budget


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


def test_if_never_early_with_ungrounded_subgoal_exits_only_on_budget():
    """#181 (b): with a permanent gap, IF must NEVER early-answer, however many ticks
    pass -- the only exit is the explore budget (or forced assembly/watchdog), exactly
    as before this change."""
    clk = FakeClock(0.0)
    world = WorldView(
        scene=FakeScene([make_instance(1, "chair")]),
        ungrounded_subgoals=1,  # one leg never grounds
        stability=StabilitySignal(winner_margin=1.0, min_contrib_n_obs=9),
    )
    ctrl, _ = build_controller(qtype=QType.INSTRUCTION_FOLLOWING, world=world)
    io = FakeRobotIO(clk, q_of(QType.INSTRUCTION_FOLLOWING))
    ctrl.tick(io)  # -> PARSING
    ctrl.state = State.EXPLORE_EXECUTE

    from core.interfaces import EXPLORE_BUDGET_S

    budget_s = EXPLORE_BUDGET_S[QType.INSTRUCTION_FOLLOWING]
    # Tick repeatedly well past the point 2 stable ticks would have fired the gate; the
    # gap never closes, so the gate must never open before the budget is spent.
    for t in (10.0, 20.0, 50.0, 100.0, 200.0, budget_s - 1.0):
        clk.set(t)
        ctrl.tick(io)
        assert ctrl.state is State.EXPLORE_EXECUTE, f"early-answered with a gap at t={t}"
    clk.set(budget_s + 1.0)
    ctrl.tick(io)
    assert ctrl.state in (State.VERIFY, State.ANSWER, State.DRIVE_OUT, State.DONE)


def test_if_early_answer_when_all_legs_grounded_and_stable():
    """#181 (a): once every leg grounds and that state holds for
    IF_EARLY_ANSWER_STABLE_TICKS consecutive ticks, IF must bank the route early
    (VERIFY -> ANSWER -> DRIVE_OUT) well before the explore budget."""
    clk = FakeClock(0.0)
    world = WorldView(
        scene=FakeScene([make_instance(1, "chair")]),
        ungrounded_subgoals=0,  # every leg grounded
        partial=PartialResults(first_anchor_pt=(1.0, 2.0, 0.0)),
    )
    ctrl, _ = build_controller(qtype=QType.INSTRUCTION_FOLLOWING, world=world)
    io = FakeRobotIO(clk, q_of(QType.INSTRUCTION_FOLLOWING))
    ctrl.tick(io)  # -> PARSING
    ctrl.state = State.EXPLORE_EXECUTE
    clk.set(60.0)  # well before the 270 s IF explore budget

    ctrl.tick(io)  # 1st stable tick: streak == 1, not enough yet
    assert ctrl.state is State.EXPLORE_EXECUTE
    ctrl.tick(io)  # 2nd consecutive stable tick: streak == 2 -> gate opens
    assert ctrl.state in (State.VERIFY, State.ANSWER, State.DRIVE_OUT, State.DONE)
    assert clk.now() < 270.0  # answered early, before the explore budget


def test_if_grounded_but_unstable_flicker_never_early():
    """#181 (c): grounding that flickers (grounded one tick, gapped the next) must
    never accumulate the required consecutive-tick streak, so the gate never opens even
    though every individual reading momentarily shows a fully-grounded route."""
    clk = FakeClock(0.0)
    scene = FakeScene([make_instance(1, "chair")])
    the_world = WorldView(scene=scene, ungrounded_subgoals=0)
    calls = {"parse": 0, "explore": 0, "verify": 0}

    def parse(q):
        calls["parse"] += 1
        return make_plan(QType.INSTRUCTION_FOLLOWING)

    def explore(io, plan, w):
        calls["explore"] += 1

    def verify(io, plan, w):
        calls["verify"] += 1
        return None

    tick_n = {"n": 0}

    def probe(io):
        # Flicker: grounded on even ticks, gapped on odd ticks -- never two grounded
        # ticks in a row, so the debounce streak can never reach 2.
        tick_n["n"] += 1
        the_world.ungrounded_subgoals = 0 if tick_n["n"] % 2 == 0 else 1
        return the_world

    ctrl = QuestionController(parse=parse, explore=explore, verify=verify, probe=probe)
    io = FakeRobotIO(clk, q_of(QType.INSTRUCTION_FOLLOWING))
    ctrl.tick(io)  # -> PARSING (tick 1, odd -> gapped)
    ctrl.state = State.EXPLORE_EXECUTE

    from core.interfaces import EXPLORE_BUDGET_S

    budget_s = EXPLORE_BUDGET_S[QType.INSTRUCTION_FOLLOWING]
    for t in (10.0, 20.0, 50.0, 100.0, 200.0, budget_s - 1.0):
        clk.set(t)
        ctrl.tick(io)
        assert ctrl.state is State.EXPLORE_EXECUTE, f"early-answered on a flicker at t={t}"
    clk.set(budget_s + 1.0)
    ctrl.tick(io)
    assert ctrl.state in (State.VERIFY, State.ANSWER, State.DRIVE_OUT, State.DONE)


def test_numerical_and_or_early_answer_gates_unchanged_by_if_debounce():
    """#181 (d): the IF stability debounce lives entirely behind the IF branch of
    _early_answer_ready; NUMERICAL keeps firing on its very first EXPLORE_EXECUTE tick
    (no debounce added) and OBJECT_REFERENCE keeps never early-firing."""
    clk = FakeClock(0.0)
    scene = FakeScene([make_instance(1, "chair"), make_instance(2, "chair")])
    world = WorldView(scene=scene, stability=StabilitySignal(winner_margin=0.4, min_contrib_n_obs=3))
    ctrl, _ = build_controller(qtype=QType.NUMERICAL, world=world)
    io = FakeRobotIO(clk, q_of(QType.NUMERICAL))
    ctrl.tick(io)  # -> PARSING
    ctrl.state = State.EXPLORE_EXECUTE
    clk.set(70.0)
    ctrl.tick(io)  # single tick: NUMERICAL still fires immediately, no debounce
    assert ctrl.state in (State.VERIFY, State.ANSWER, State.DONE)

    clk2 = FakeClock(0.0)
    ctrl_or, _ = build_controller(
        qtype=QType.OBJECT_REFERENCE, world=WorldView(scene=FakeScene([make_instance(1, "chair")]))
    )
    io_or = FakeRobotIO(clk2, q_of(QType.OBJECT_REFERENCE))
    ctrl_or.tick(io_or)
    ctrl_or.state = State.EXPLORE_EXECUTE
    for t in (50.0, 100.0, 150.0):
        clk2.set(t)
        ctrl_or.tick(io_or)
        assert ctrl_or.state is State.EXPLORE_EXECUTE  # OR never early-finishes


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


# --------------------------------------------------------------------------- IF-F4:
# for INSTRUCTION_FOLLOWING the drive IS the answer — ANSWER inserts a DRIVE_OUT state that
# keeps ticking the answer heads until the head reports drive_complete (or the watchdog
# floor fires). NUMERICAL/OR are unaffected: they go ANSWER -> DONE. These stub tests pin
# the FSM contract directly (fast; no A*).


def _drive_out_controller(*, complete_after: int, qtype=QType.INSTRUCTION_FOLLOWING):
    """Controller whose stub probe flips ``world.drive_complete`` True after
    ``complete_after`` DRIVE_OUT explore ticks, and whose verify returns a legal answer.

    Returns (ctrl, io, calls) where calls tracks explore/verify invocations. The explore
    counter proves the heads keep being ticked during DRIVE_OUT.
    """
    clk = FakeClock(0.0)
    calls = {"explore": 0, "verify": 0}
    scene = FakeScene([make_instance(1, "chair")])
    verify_answer = (
        WaypointCmd(5.0, 5.0)
        if qtype is QType.INSTRUCTION_FOLLOWING
        else (
            IntAnswer(1)
            if qtype is QType.NUMERICAL
            else MarkerBox(1, 1, 1, 1, 1, 1, label="x")
        )
    )
    state = {"drive_out_ticks": 0}
    the_world = WorldView(scene=scene)

    def parse(q):
        return make_plan(qtype)

    def explore(io, plan, w):
        calls["explore"] += 1
        # Only count explore ticks that happen while the FSM is in DRIVE_OUT.
        if ctrl.state is State.DRIVE_OUT:
            state["drive_out_ticks"] += 1
            if state["drive_out_ticks"] >= complete_after:
                the_world.drive_complete = True

    def verify(io, plan, w):
        calls["verify"] += 1
        return verify_answer

    def probe(io):
        return the_world

    ctrl = QuestionController(parse=parse, explore=explore, verify=verify, probe=probe)
    io = FakeRobotIO(clk, q_of(qtype))
    return ctrl, io, clk, calls, state


def test_if_answer_enters_drive_out_and_keeps_ticking_until_complete():
    ctrl, io, clk, calls, state = _drive_out_controller(complete_after=4)
    # Drive into ANSWER via the explore budget, then observe DRIVE_OUT.
    ctrl.tick(io)  # intake -> PARSING
    ctrl.state = State.EXPLORE_EXECUTE
    clk.set(300.0)  # past the IF explore budget (270 s)
    saw_drive_out = False
    for _ in range(12):
        ctrl.tick(io)
        if ctrl.state is State.DRIVE_OUT:
            saw_drive_out = True
        clk.advance(0.5)
        if ctrl.state is State.DONE:
            break
    assert saw_drive_out, "IF ANSWER did not enter DRIVE_OUT"
    assert ctrl.state is State.DONE
    # exactly one waypoint answer, published once at ANSWER (not re-published in DRIVE_OUT).
    assert io.publish_count == 1
    assert len(io.published_waypoints) == 1
    # the heads WERE ticked during DRIVE_OUT (the whole point of the fix).
    assert state["drive_out_ticks"] >= 4
    # transitions recorded: answer->drive_out then drive_out->done.
    trail = "|".join(r.detail for r in ctrl.flight_recording() if r.event == "transition")
    assert "answer->drive_out" in trail
    assert "drive_out->done" in trail
    assert trail.index("answer->drive_out") < trail.index("drive_out->done")


@pytest.mark.parametrize("qtype", [QType.NUMERICAL, QType.OBJECT_REFERENCE])
def test_non_if_answer_goes_straight_to_done_no_drive_out(qtype):
    """NUMERICAL/OR must NOT enter DRIVE_OUT — they answer and finish exactly as before."""
    ctrl, io, clk, calls, state = _drive_out_controller(complete_after=1, qtype=qtype)
    ctrl.tick(io)  # intake -> PARSING
    ctrl.state = State.VERIFY
    clk.set(300.0)
    saw_drive_out = False
    for _ in range(8):
        ctrl.tick(io)
        if ctrl.state is State.DRIVE_OUT:
            saw_drive_out = True
        clk.advance(0.5)
        if ctrl.state is State.DONE:
            break
    assert not saw_drive_out, f"{qtype} wrongly entered DRIVE_OUT"
    assert ctrl.state is State.DONE
    assert io.publish_count == 1
    trail = "|".join(r.detail for r in ctrl.flight_recording() if r.event == "transition")
    assert "drive_out" not in trail
    assert "answer->done" in trail


def test_if_drive_out_watchdog_floor_terminates_a_hung_drive():
    """If the drive never completes (drive_complete stays False), the watchdog floor still
    fires from DRIVE_OUT — the continue-drive state cannot shadow or delay the overlay."""
    # complete_after huge => drive_complete never flips in the tick budget below.
    ctrl, io, clk, calls, state = _drive_out_controller(complete_after=10_000)
    ctrl.tick(io)  # intake -> PARSING
    ctrl.state = State.EXPLORE_EXECUTE
    clk.set(300.0)
    # step into DRIVE_OUT
    for _ in range(6):
        ctrl.tick(io)
        clk.advance(0.5)
        if ctrl.state is State.DRIVE_OUT:
            break
    assert ctrl.state is State.DRIVE_OUT
    # jump past the watchdog floor: overlay must force DONE regardless of drive_complete.
    clk.set(ctrl._watchdog_floor_s + 1.0)
    ctrl.tick(io)
    assert ctrl.state is State.DONE
    assert io.publish_count == 1  # the ANSWER waypoint; watchdog publish is a latched no-op


# --------------------------------------------------------------------------- #183


def _redrive_controller(
    *, complete_after: int, visited_after: int | None, qtype=QType.INSTRUCTION_FOLLOWING
):
    """Like ``_drive_out_controller`` but the stub explore flips ``drive_complete`` and
    ``legs_visited`` independently, so the two-signal DRIVE_OUT exit gate (#183) can be
    pinned directly without a real InstructionHead/planner.

    ``visited_after=None`` means legs_visited never flips (the redrive never finishes
    inside the tick budget) -- only the watchdog floor can end the question.
    """
    clk = FakeClock(0.0)
    calls = {"explore": 0, "verify": 0}
    scene = FakeScene([make_instance(1, "chair")])
    verify_answer = WaypointCmd(5.0, 5.0)
    state = {"drive_out_ticks": 0}
    the_world = WorldView(scene=scene, legs_visited=False)

    def parse(q):
        return make_plan(qtype)

    def explore(io, plan, w):
        calls["explore"] += 1
        if ctrl.state is State.DRIVE_OUT:
            state["drive_out_ticks"] += 1
            if state["drive_out_ticks"] >= complete_after:
                the_world.drive_complete = True
            if visited_after is not None and state["drive_out_ticks"] >= visited_after:
                the_world.legs_visited = True

    def verify(io, plan, w):
        calls["verify"] += 1
        return verify_answer

    def probe(io):
        return the_world

    ctrl = QuestionController(parse=parse, explore=explore, verify=verify, probe=probe)
    io = FakeRobotIO(clk, q_of(qtype))
    return ctrl, io, clk, calls, state, the_world


def test_if_drive_out_does_not_exit_on_drive_complete_while_legs_unvisited():
    """Issue #183: a head report of drive_complete=True is NOT enough on its own to end
    DRIVE_OUT while legs_visited stays False (the ordered visit is unfinished) and
    budget remains — the FSM must keep ticking the heads (reinvesting the budget) rather
    than discard it the moment the looser drive_complete signal fires."""
    # drive_complete flips almost immediately (mirrors the live #181/#183 evidence: an
    # early-fired slot's route already reads drive_complete seconds after answering);
    # legs_visited never flips inside this tick budget -- only the watchdog can end it.
    ctrl, io, clk, calls, state, world = _redrive_controller(complete_after=1, visited_after=None)
    ctrl.tick(io)  # intake -> PARSING
    ctrl.state = State.EXPLORE_EXECUTE
    clk.set(300.0)
    saw_drive_out = False
    for _ in range(10):
        ctrl.tick(io)
        if ctrl.state is State.DRIVE_OUT:
            saw_drive_out = True
        clk.advance(0.5)
        if ctrl.state is State.DONE:
            break
    assert saw_drive_out
    # drive_complete fired long ago but legs_visited never did — the FSM must still be
    # driving, not DONE, well before the watchdog floor.
    assert ctrl.state is State.DRIVE_OUT
    assert world.drive_complete is True
    assert world.legs_visited is False
    assert state["drive_out_ticks"] > 5  # kept reinvesting the budget, not idling out

    # The watchdog floor is still the unconditional hard backstop: it forces DONE
    # regardless of legs_visited ever completing.
    clk.set(ctrl._watchdog_floor_s + 1.0)
    ctrl.tick(io)
    assert ctrl.state is State.DONE
    assert io.publish_count == 1  # the ANSWER waypoint; watchdog publish is a latched no-op


def test_if_drive_out_exits_once_ordered_visit_finishes():
    """Issue #183: once the head reports BOTH drive_complete AND legs_visited (the
    reinvest redrive finished visiting every leg in order), DRIVE_OUT ends right away —
    it does not force the question to idle out to the watchdog floor."""
    # legs_visited flips a few ticks after drive_complete, mirroring a redrive pass that
    # takes a few extra ticks to thread the remaining leg(s) after the loose
    # drive_complete signal first fires.
    ctrl, io, clk, calls, state, world = _redrive_controller(complete_after=2, visited_after=5)
    ctrl.tick(io)  # intake -> PARSING
    ctrl.state = State.EXPLORE_EXECUTE
    clk.set(300.0)
    saw_drive_out = False
    for _ in range(20):
        ctrl.tick(io)
        if ctrl.state is State.DRIVE_OUT:
            saw_drive_out = True
        clk.advance(0.5)
        if ctrl.state is State.DONE:
            break
    assert saw_drive_out
    assert ctrl.state is State.DONE
    assert world.drive_complete is True
    assert world.legs_visited is True
    # Ended well before the watchdog floor — a finished redrive is not forced to idle.
    assert clk.now() < ctrl._watchdog_floor_s
    trail = "|".join(r.detail for r in ctrl.flight_recording() if r.event == "transition")
    assert "drive_out->done" in trail


def test_if_never_early_when_parse_failed_and_plan_is_none():
    """#181 regression (verifier repro, 5 Aug 2026): a failed parse leaves plan None
    and the instruction head unbuilt, so WorldView.ungrounded_subgoals carries the
    assembler's default 0. The gate must read that as the ultimate gap and never
    fire; the only exit stays the explore budget, exactly as before #181."""
    clk = FakeClock(0.0)
    world = WorldView(
        scene=FakeScene([]),
        ungrounded_subgoals=0,  # the default -- NOT a head report; no head exists
    )

    def broken_parse(q):
        raise RuntimeError("parse failed")

    ctrl, _ = build_controller(qtype=QType.INSTRUCTION_FOLLOWING, world=world)
    ctrl._parse = broken_parse
    io = FakeRobotIO(clk, q_of(QType.INSTRUCTION_FOLLOWING))
    ctrl.tick(io)  # -> PARSING; parse raises, plan stays None
    assert ctrl.plan is None
    ctrl.state = State.EXPLORE_EXECUTE

    from core.interfaces import EXPLORE_BUDGET_S

    budget_s = EXPLORE_BUDGET_S[QType.INSTRUCTION_FOLLOWING]
    for t in (10.0, 20.0, 50.0, 100.0, 200.0, budget_s - 1.0):
        clk.set(t)
        ctrl.tick(io)
        assert ctrl.state is State.EXPLORE_EXECUTE, f"early-answered with no plan at t={t}"
    clk.set(budget_s + 1.0)
    ctrl.tick(io)
    assert ctrl.state in (State.VERIFY, State.ANSWER, State.DRIVE_OUT, State.DONE)
