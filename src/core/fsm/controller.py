"""QuestionController — the per-question lifecycle FSM and watchdog overlay.

States (normal flow):
    IDLE -> PARSING -> ORIENT -> EXPLORE_EXECUTE -> VERIFY -> ANSWER -> DONE

WATCHDOG is an *overlay*, not a state:
  * at elapsed >= 570 s (watchdog_floor): publish the FloorAnswers answer for the qtype
    and force DONE, regardless of the current state;
  * at elapsed >= 510 s (forced_assembly): force transition into VERIFY/ANSWER with
    whatever exists, so best-effort assembly runs before the hard floor.

All heavy/external work (parse, one explore step, verify) is injected as callables so the
FSM runs deterministically against stubs in tests; the real wiring is done at integration.

Answer publication is latched: each question publishes exactly one answer. Question intake
latches the first receipt and ignores the 1 Hz republish of the same text (upstream gotcha:
/challenge_question republishes at 1 Hz).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

from core.interfaces import (
    IntAnswer,
    MarkerBox,
    QType,
    Question,
    RobotIO,
    SceneIndex,
    WaypointCmd,
)
from core.fsm.budget import BudgetState, CallLedger
from core.fsm.events import EventLog
from core.fsm.floors import FloorAnswers, PartialResults


class State(str, Enum):
    IDLE = "idle"
    PARSING = "parsing"
    ORIENT = "orient"
    EXPLORE_EXECUTE = "explore_execute"
    VERIFY = "verify"
    ANSWER = "answer"
    DONE = "done"


@dataclass
class StabilitySignal:
    """Injected numerical-count stability evidence (architecture §1 row 8 gate).

    winner_margin: fractional lead of the modal count over the runner-up (0..1).
    min_contrib_n_obs: min n_obs across instances contributing to the winning count.
    A count is 'stable' iff winner_margin >= 0.25 AND min_contrib_n_obs >= 3.
    """

    winner_margin: float = 0.0
    min_contrib_n_obs: int = 0

    @property
    def stable(self) -> bool:
        return self.winner_margin >= 0.25 and self.min_contrib_n_obs >= 3


@dataclass
class WorldView:
    """Everything the FSM reads about progress, refreshed each tick by the injected probes.

    scene:            read view over the instance map.
    partial:          best-effort head outputs for the floor + early answer.
    ungrounded_subgoals: count of IF sub-goals not yet grounded with >=3 obs
                         (0 => fully grounded; gates IF early-answer).
    stability:        numerical count stability evidence.
    """

    scene: SceneIndex | None = None
    partial: PartialResults = field(default_factory=PartialResults)
    ungrounded_subgoals: int = 0
    stability: StabilitySignal = field(default_factory=StabilitySignal)


# Injected callables ---------------------------------------------------------
# parse:   (Question) -> plan | None            (checkpoint 1)
# explore: (RobotIO, plan|None, WorldView) -> None   (one deterministic step per tick)
# verify:  (RobotIO, plan|None, WorldView) -> answer   final publishable answer, or None
# probe:   (RobotIO) -> WorldView               read the live world for this tick
ParseFn = Callable[[Question], object | None]
ExploreFn = Callable[[RobotIO, object | None, WorldView], None]
VerifyFn = Callable[[RobotIO, object | None, WorldView], object | None]
ProbeFn = Callable[[RobotIO], WorldView]


class QuestionController:
    """Drives one question end-to-end. tick(io) is called externally at ~5 Hz."""

    def __init__(
        self,
        *,
        parse: ParseFn,
        explore: ExploreFn,
        verify: VerifyFn,
        probe: ProbeFn,
        floors: FloorAnswers | None = None,
        events: EventLog | None = None,
    ) -> None:
        self._parse = parse
        self._explore = explore
        self._verify = verify
        self._probe = probe
        self.floors = floors if floors is not None else FloorAnswers()
        self.events = events if events is not None else EventLog()

        self.state: State = State.IDLE
        self.budget: BudgetState | None = None
        self.ledger: CallLedger | None = None

        self.question: Question | None = None
        self.qtype: QType | None = None
        self.plan: object | None = None
        self.world: WorldView = WorldView()

        self._answer_published = False
        self._dump_done = False

    # ------------------------------------------------------------------ helpers
    @property
    def answer_published(self) -> bool:
        return self._answer_published

    def _log(self, event: str, detail: str = "") -> None:
        t = self.budget.elapsed() if self.budget is not None else 0.0
        self.events.record(t, self.state.value, event, detail)

    def _to(self, state: State, why: str = "") -> None:
        if state is self.state:
            return
        self._log("transition", f"{self.state.value}->{state.value} {why}".strip())
        self.state = state

    # ------------------------------------------------------------------ intake
    def _intake(self, io: RobotIO) -> None:
        """Latch the first question; ignore 1 Hz republishes of the same text."""
        q = io.question()
        if q is None:
            return
        if self.question is None:
            self.question = q
            self.qtype = _infer_qtype(q)
            self.budget = BudgetState(io.clock(), self.qtype)
            self.budget.latch(getattr(q, "t_received", None))
            self.ledger = CallLedger(self.budget)
            self._log("question_latched", f"{self.qtype.value if self.qtype else '?'}: {q.text!r}")
            self._to(State.PARSING, "question received")
        elif q.text != self.question.text:
            # A genuinely different question mid-run is an upstream error; log, don't switch.
            self._log("question_ignored", f"unexpected new text {q.text!r}")
        # same text republish => silently ignored

    # ------------------------------------------------------------------ tick
    def tick(self, io: RobotIO) -> None:
        """Advance the FSM one step. Idempotent once DONE. Never raises out."""
        if self.state is State.DONE:
            self._finish()
            return

        if self.question is None:
            self._intake(io)
            if self.question is None:
                return  # still idle, no question yet

        # From here budget/ledger exist. Refresh world + floors every tick.
        self.world = _safe_probe(self._probe, io)
        self.floors.update(self.world.scene, self.plan, self.world.partial)

        # WATCHDOG overlay — checked before normal state work.
        if self.budget is not None and self.budget.watchdog_floor:
            self._watchdog_publish(io, reason="watchdog_floor>=570s")
            return
        if self.budget is not None and self.budget.forced_assembly and self.state not in (
            State.VERIFY,
            State.ANSWER,
            State.DONE,
        ):
            self._to(State.VERIFY, "forced_assembly>=510s")

        # Normal per-state work.
        handler = self._HANDLERS.get(self.state)
        if handler is not None:
            handler(self, io)

    # ------------------------------------------------------------------ states
    def _tick_parsing(self, io: RobotIO) -> None:
        if self.plan is None and self.ledger is not None and self.ledger.allow("parse"):
            plan = _safe_call(self._parse, self.question)
            self.ledger.record("parse", 0.0, "api")
            if plan is not None:
                self.plan = plan
                self._log("parsed", f"tier={getattr(plan, 'parse_tier', '?')}")
        # Parse runs concurrently with orientation; move on regardless (floor covers a dark parse).
        self._to(State.ORIENT, "parse attempted")

    def _tick_orient(self, io: RobotIO) -> None:
        # In-place sweep window; seed the map. Leave orientation once the 60 s window closes.
        if self.budget is not None and not self.budget.in_orientation:
            self._to(State.EXPLORE_EXECUTE, "orientation window closed")

    def _tick_explore(self, io: RobotIO) -> None:
        _safe_explore(self._explore, io, self.plan, self.world)
        if self._early_answer_ready():
            self._to(State.VERIFY, "early-answer gate open")
            return
        if self.budget is not None and self.budget.past_explore_budget(self.qtype):
            self._to(State.VERIFY, "explore budget spent")

    def _tick_verify(self, io: RobotIO) -> None:
        ans = None
        if self.ledger is not None and self.ledger.allow("verification"):
            ans = _safe_verify(self._verify, io, self.plan, self.world)
            self.ledger.record("verification", 0.0, "api")
        if ans is not None:
            self._pending_answer = ans
        self._to(State.ANSWER, "verify complete")

    def _tick_answer(self, io: RobotIO) -> None:
        ans = getattr(self, "_pending_answer", None)
        if ans is None:
            ans = self.floors.get(self.qtype)  # best-effort floor if verify produced nothing
            self._log("answer_from_floor", "verify yielded nothing")
        self._publish(io, ans, reason="answer_state")
        if not self._answer_published:
            # The pending (verify) answer was undispatchable and got discarded; fall back to
            # the floor so this state still emits exactly one legal answer.
            self._pending_answer = None
            floor = self.floors.get(self.qtype)
            self._log("answer_from_floor", "pending answer undispatchable; using floor")
            self._publish(io, floor, reason="answer_state_floor")
        self._to(State.DONE, "answered")

    _HANDLERS: dict[State, Callable[["QuestionController", RobotIO], None]] = {
        State.PARSING: _tick_parsing,
        State.ORIENT: _tick_orient,
        State.EXPLORE_EXECUTE: _tick_explore,
        State.VERIFY: _tick_verify,
        State.ANSWER: _tick_answer,
    }

    # ------------------------------------------------------------------ early answer
    def _early_answer_ready(self) -> bool:
        """Asymmetric early-answer gate (architecture §1 row 8).

        NUMERICAL: fire as soon as the injected count is stable (margin >=25% AND all
        contributing instances >=3 obs). IF: NEVER early with any ungrounded sub-goal.
        OBJECT_REFERENCE: no aggressive early-finish (fall through to budget/verify).
        """
        if self.qtype is QType.NUMERICAL:
            return self.world.stability.stable
        if self.qtype is QType.INSTRUCTION_FOLLOWING:
            return False  # only the budget/forced path ends IF; never bank early with a gap
        return False

    # ------------------------------------------------------------------ publish
    def _publish(self, io: RobotIO, ans: object, reason: str) -> None:
        """Publish an answer exactly once; subsequent calls are no-ops (latched).

        The exactly-once latch is only set when dispatch actually reaches a RobotIO sink.
        An unrecognized answer type (e.g. a verify callable that returned a dict) is NOT
        published and does NOT latch — it is discarded so the normal forced-assembly /
        watchdog-floor path can still fire and publish a legal floor answer.
        """
        if self._answer_published:
            self._log("publish_suppressed", f"already published; {reason}")
            return
        if not _dispatch(io, ans):
            self._log(
                "publish_dropped",
                f"undispatchable type {type(ans).__name__} via {reason}",
            )
            return
        self._answer_published = True
        self._log("published", f"{type(ans).__name__} via {reason}")

    def _watchdog_publish(self, io: RobotIO, reason: str) -> None:
        ans = self.floors.get(self.qtype)
        self._publish(io, ans, reason=reason)
        self._to(State.DONE, reason)

    # ------------------------------------------------------------------ finish
    def _finish(self) -> None:
        if not self._dump_done:
            self._log("done", f"published={self._answer_published}")
            self._flight_recording = self.events.dump()
            self._dump_done = True

    def flight_recording(self):
        """Return the dumped event log (available after DONE)."""
        return getattr(self, "_flight_recording", self.events.dump())


# --------------------------------------------------------------------------- module helpers


def _infer_qtype(q: Question) -> QType:
    """Cheap qtype heuristic used until the parse settles it; total, defaults NUMERICAL-safe.

    Real qtype comes from the parse; this only seeds budgets before the plan lands.
    """
    text = (getattr(q, "text", "") or "").lower()
    qtype = getattr(q, "qtype", None)
    if isinstance(qtype, QType):
        return qtype
    if any(w in text for w in ("how many", "number of", "count")):
        return QType.NUMERICAL
    if any(w in text for w in ("go to", "navigate", "take the path", "avoid", "drive")):
        return QType.INSTRUCTION_FOLLOWING
    return QType.OBJECT_REFERENCE


def _dispatch(io: RobotIO, ans: object) -> bool:
    """Route an answer to the matching RobotIO publisher.

    Returns True iff the answer was one of the three legal types and was dispatched to a
    sink; returns False for any other type (including None) without publishing. The caller
    relies on this to avoid latching the exactly-once guarantee on a non-answer.
    """
    if isinstance(ans, IntAnswer):
        io.publish_int(ans)
        return True
    if isinstance(ans, MarkerBox):
        io.publish_marker(ans)
        return True
    if isinstance(ans, WaypointCmd):
        io.publish_waypoint(ans)
        return True
    return False  # unrecognized type: caller discards and falls back to the floor


def _safe_probe(probe: ProbeFn, io: RobotIO) -> WorldView:
    try:
        wv = probe(io)
        return wv if isinstance(wv, WorldView) else WorldView()
    except Exception:
        return WorldView()


def _safe_call(fn, *args):
    try:
        return fn(*args)
    except Exception:
        return None


def _safe_explore(fn: ExploreFn, io: RobotIO, plan, world) -> None:
    try:
        fn(io, plan, world)
    except Exception:
        pass


def _safe_verify(fn: VerifyFn, io: RobotIO, plan, world):
    try:
        return fn(io, plan, world)
    except Exception:
        return None
