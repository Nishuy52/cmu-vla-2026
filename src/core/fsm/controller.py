"""QuestionController — the per-question lifecycle FSM and watchdog overlay.

States (normal flow):
    IDLE -> PARSING -> ORIENT -> EXPLORE_EXECUTE -> VERIFY -> ANSWER -> DONE

For INSTRUCTION_FOLLOWING the flow inserts one extra state before DONE:
    ... -> VERIFY -> ANSWER -> DRIVE_OUT -> DONE
because for IF the drive IS the answer (IF-F4). ANSWER still publishes exactly one
waypoint (latched, unchanged), but instead of going straight to DONE the FSM enters
DRIVE_OUT and KEEPS ticking the answer heads — breadcrumbs, re-grounding, replans — so a
partially-grounded route's later legs are still attempted with the remaining budget
(ordered-leg partial credit is real). DRIVE_OUT ends only once the head reports BOTH the
drive complete (arrival / exhausted-with-no-replan-budget) AND every leg individually
visited in order (issue #183 — a "drive complete" report alone can fire on a capsule-
forced recovery path or a spent H11 replan budget without every ordered leg goal actually
having been reached; see WorldView.legs_visited / InstructionHead.all_legs_visited), or
the watchdog floor fires — whichever is first. While the route reports complete but not
every leg is yet visited and budget remains, the instruction head reinvests the remaining
budget in a bounded re-drive of the unvisited legs (InstructionHead._redrive) instead of
the question ending outright; this applies identically whether DRIVE_OUT was entered via
the #181 early-answer gate or the ordinary explore-budget exit. NUMERICAL and
OBJECT_REFERENCE go ANSWER -> DONE exactly as before.

WATCHDOG is an *overlay*, not a state:
  * at elapsed >= 540 s (watchdog_floor): publish the FloorAnswers answer for the qtype
    and force DONE, regardless of the current state;
  * at elapsed >= 480 s (forced_assembly): force transition into VERIFY/ANSWER with
    whatever exists, so best-effort assembly runs before the hard floor.
  These are the controller's EFFECTIVE, skew-hedged defaults (fsm/budget.py
  DEFAULT_WATCHDOG_FLOOR_S / DEFAULT_FORCED_ASSEMBLY_S); the neutral interface
  constants (core.interfaces WATCHDOG_FLOOR_S / FORCED_ASSEMBLY_S) stay at 570/510
  (see core.calibration.BudgetTunables for the split).

All heavy/external work (parse, one explore step, verify) is injected as callables so the
FSM runs deterministically against stubs in tests; the real wiring is done at integration.

Answer publication is latched: each question publishes exactly one answer. Question intake
latches the first receipt and ignores the 1 Hz republish of the same text (upstream gotcha:
/challenge_question republishes at 1 Hz).
"""
from __future__ import annotations

import traceback
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
from core.fsm.budget import (
    BudgetState,
    CallLedger,
    DEFAULT_FORCED_ASSEMBLY_S,
    DEFAULT_WATCHDOG_FLOOR_S,
)
from core.fsm.events import EventLog
from core.fsm.floors import FloorAnswers, PartialResults


# #181 — IF early-answer stability debounce: how many consecutive ticks
# WorldView.ungrounded_subgoals must read 0 before the IF early-answer gate opens. A
# leg can flip from grounded back to ungrounded mid-run (a CP3 anchor mismatch demotes
# the resolved instance and re-plans to the runner-up, core.heads.instruction), so a
# single tick's zero-gap reading is not proof the route stays resolved. Requiring the
# state to hold twice mirrors the spirit of the NUMERICAL stability gate's per-instance
# observation floor (StabilitySignal.min_contrib_n_obs >= 3) without needing new state
# outside the controller: any tick that reopens a gap resets the streak to zero, so the
# asymmetry invariant (never fire with a gap) cannot be weakened by this debounce.
IF_EARLY_ANSWER_STABLE_TICKS: int = 2


class State(str, Enum):
    IDLE = "idle"
    PARSING = "parsing"
    ORIENT = "orient"
    EXPLORE_EXECUTE = "explore_execute"
    VERIFY = "verify"
    ANSWER = "answer"
    DRIVE_OUT = "drive_out"  # IF only: keep driving the route after the answer (IF-F4)
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
    drive_complete:   IF only — the instruction head reports the route drive is finished
                      (arrived at the terminal, or exhausted with no replan budget left).
                      Read by the DRIVE_OUT state alongside legs_visited to decide when
                      to stop driving and go DONE (IF-F4). Meaningless / False for
                      NUMERICAL and OR.
    legs_visited:     IF only — issue #183: the instruction head reports every route leg
                      has been individually confirmed reached (an ordered visit of each
                      leg's own goal within the rubric-relevant tolerance), or the route
                      has no legs. drive_complete can fire on a looser signal (a
                      capsule-forced recovery path, or a spent H11 replan budget) that
                      does not itself guarantee every leg was reached — DRIVE_OUT exits
                      only once BOTH drive_complete and legs_visited are true (or the
                      watchdog floor overrides). Meaningless / True (vacuously "nothing
                      left unvisited") for NUMERICAL and OR, and defaults True so an
                      unconfigured/stub probe never blocks a pre-#183 caller.
    """

    scene: SceneIndex | None = None
    partial: PartialResults = field(default_factory=PartialResults)
    ungrounded_subgoals: int = 0
    stability: StabilitySignal = field(default_factory=StabilitySignal)
    drive_complete: bool = False
    legs_visited: bool = True


# Injected callables ---------------------------------------------------------
# parse:   (Question) -> plan | None            (checkpoint 1)
# explore: (RobotIO, plan|None, WorldView) -> None   (one deterministic step per tick)
# verify:  (RobotIO, plan|None, WorldView) -> answer   final publishable answer, or None
# probe:   (RobotIO) -> WorldView               read the live world for this tick
ParseFn = Callable[[Question], object | None]
ExploreFn = Callable[[RobotIO, object | None, WorldView], None]
VerifyFn = Callable[[RobotIO, object | None, WorldView], object | None]
ProbeFn = Callable[[RobotIO], WorldView]

# Optional injected logger callback (issue #60): (level, message) -> None. level is
# "info" (state transitions) or "warn" (a swallowed exception in a _safe_* seam).
# Default None preserves the original total silence (pure-core tests stay byte-identical).
LogFn = Callable[[str, str], None]


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
        forced_assembly_s: float = DEFAULT_FORCED_ASSEMBLY_S,
        watchdog_floor_s: float = DEFAULT_WATCHDOG_FLOOR_S,
        logger: LogFn | None = None,
    ) -> None:
        self._parse = parse
        self._explore = explore
        self._verify = verify
        self._probe = probe
        self.floors = floors if floors is not None else FloorAnswers()
        self.events = events if events is not None else EventLog()
        # Runtime observability (issue #60): None (default) is total silence, matching the
        # pre-#60 behaviour exactly, so pure-core tests stay byte-identical without a logger.
        self._logger = logger
        self._swallow_counts: dict[str, int] = {}
        # Answer gates the budget will use (pulled in from the interface constants to hedge
        # the evaluator-clock skew; see core.fsm.budget). Kept as controller params so a
        # launch/config layer can override them from measured Ubuntu-gate skew.
        self._forced_assembly_s = float(forced_assembly_s)
        self._watchdog_floor_s = float(watchdog_floor_s)

        self.state: State = State.IDLE
        self.budget: BudgetState | None = None
        self.ledger: CallLedger | None = None

        self.question: Question | None = None
        self.qtype: QType | None = None
        self.plan: object | None = None
        self.world: WorldView = WorldView()

        self._answer_published = False
        self._dump_done = False
        # #181 — consecutive-tick streak of "all IF legs grounded" (see
        # IF_EARLY_ANSWER_STABLE_TICKS); read/mutated only by _early_answer_ready.
        self._if_grounded_streak = 0

    # ------------------------------------------------------------------ helpers
    @property
    def answer_published(self) -> bool:
        return self._answer_published

    def _log(self, event: str, detail: str = "") -> None:
        t = self.budget.elapsed() if self.budget is not None else 0.0
        self.events.record(t, self.state.value, event, detail)

    def _emit(self, level: str, msg: str) -> None:
        """Forward a line to the injected logger callback, if any (issue #60)."""
        if self._logger is not None:
            self._logger(level, msg)

    def _note_swallowed(self, seam: str, exc: BaseException) -> None:
        """Record a swallowed exception from a _safe_* seam (issue #60).

        Logs the first occurrence per seam with a full traceback, then every 50th repeat
        (count kept per seam) so a persistently failing seam does not silently vanish nor
        spam the log every tick.
        """
        n = self._swallow_counts.get(seam, 0) + 1
        self._swallow_counts[seam] = n
        if n == 1 or n % 50 == 0:
            tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            self._emit(
                "warn",
                f"seam {seam!r} swallowed exception (occurrence {n}): {exc!r}\n{tb}",
            )

    def _to(self, state: State, why: str = "") -> None:
        if state is self.state:
            return
        line = f"{self.state.value}->{state.value} {why}".strip()
        self._log("transition", line)
        self._emit("info", line)
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
            self.budget = BudgetState(
                io.clock(),
                self.qtype,
                forced_assembly_s=self._forced_assembly_s,
                watchdog_floor_s=self._watchdog_floor_s,
            )
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
        self.world = _safe_probe(self._probe, io, self._note_swallowed)
        self.floors.update(self.world.scene, self.plan, self.world.partial)

        # WATCHDOG overlay — checked before normal state work.
        if self.budget is not None and self.budget.watchdog_floor:
            self._watchdog_publish(io, reason=f"watchdog_floor>={self._watchdog_floor_s:.0f}s")
            return
        if self.budget is not None and self.budget.forced_assembly and self.state not in (
            State.VERIFY,
            State.ANSWER,
            State.DRIVE_OUT,  # already answered + driving out (IF-F4); do not re-assemble
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
            plan = _safe_call(self._parse, self.question, on_error=self._note_swallowed)
            self.ledger.record("parse", 0.0, "api")
            if plan is not None:
                self.plan = plan
                self._log("parsed", f"tier={getattr(plan, 'parse_tier', '?')}")
                # The regex _infer_qtype at intake is only a pre-parse seed; the parse carries
                # the authoritative qtype. Adopt it so floor selection and the budget's
                # explore/answer gates key off the real type (SYS-F7: a motion-verbed OR
                # question otherwise gets a WaypointCmd floor on the wrong answer topic).
                plan_qtype = getattr(plan, "qtype", None)
                if isinstance(plan_qtype, QType) and plan_qtype is not self.qtype:
                    old = self.qtype
                    self.qtype = plan_qtype
                    if self.budget is not None:
                        self.budget.qtype = plan_qtype
                    self._log(
                        "qtype_corrected",
                        f"{old.value if old else '?'}->{plan_qtype.value} (parse over seed)",
                    )
        # Parse runs concurrently with orientation; move on regardless (floor covers a dark parse).
        self._to(State.ORIENT, "parse attempted")

    def _tick_orient(self, io: RobotIO) -> None:
        # In-place sweep window; seed the map. Tick the same explore callable used by
        # EXPLORE_EXECUTE so the opening diamond sweep actually runs during this window
        # instead of after it (#80: 60 s of dead air otherwise, live-competition-critical —
        # invisible offline since the battery bypasses the FSM). ExplorationPolicy anchors
        # its own sweep clock on first step (`_elapsed`, core/nav/exploration.py:80-85), so
        # ticking it here for the first time naturally starts the sweep at ORIENT entry
        # rather than at t=60s; no anchor change needed there.
        _safe_explore(self._explore, io, self.plan, self.world, self._note_swallowed)
        # Leave orientation once the 60 s window closes.
        if self.budget is not None and not self.budget.in_orientation:
            self._to(State.EXPLORE_EXECUTE, "orientation window closed")

    def _tick_explore(self, io: RobotIO) -> None:
        _safe_explore(self._explore, io, self.plan, self.world, self._note_swallowed)
        if self._early_answer_ready():
            self._to(State.VERIFY, "early-answer gate open")
            return
        if self.budget is not None and self.budget.past_explore_budget(self.qtype):
            self._to(State.VERIFY, "explore budget spent")

    def _tick_verify(self, io: RobotIO) -> None:
        ans = None
        if self.ledger is not None and self.ledger.allow("verification"):
            ans = _safe_verify(self._verify, io, self.plan, self.world, self._note_swallowed)
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
        # IF-F4: for instruction-following the drive IS the answer. Publishing the first
        # waypoint does not end the question — enter DRIVE_OUT to keep ticking the heads
        # (breadcrumbs, re-grounding, replans) until the route completes or the watchdog
        # fires. NUMERICAL/OR are terminal at ANSWER: they answer and finish as today.
        if self.qtype is QType.INSTRUCTION_FOLLOWING:
            self._to(State.DRIVE_OUT, "answered; continue driving the route (IF-F4)")
        else:
            self._to(State.DONE, "answered")

    def _tick_drive_out(self, io: RobotIO) -> None:
        """IF continue-drive (IF-F4): keep ticking the answer heads so the route's later
        legs are still driven with the remaining budget.

        The answer waypoint was already published in ANSWER (latch untouched). Here we
        advance the heads every tick — the instruction head streams the next breadcrumb,
        re-grounds late-appearing anchors, extends the route over newly grounded legs, and
        replans on stalls — via the same explore callable used during EXPLORE_EXECUTE. We
        leave DRIVE_OUT for DONE only when the head reports BOTH the drive complete
        (arrival, or exhausted with no replan budget) AND every leg individually visited
        in order (issue #183: world.legs_visited — a drive_complete report alone can fire
        on a capsule-forced recovery path or a spent H11 replan budget without every
        ordered leg goal actually having been reached; while that gap remains and budget
        is left, the head reinvests it into a bounded re-drive of the unvisited legs
        rather than the FSM ending the question — see InstructionHead._redrive). This
        applies identically whether DRIVE_OUT was entered via the #181 early-answer gate
        or the ordinary explore-budget exit. The watchdog overlay (checked before this
        handler every tick) remains the hard backstop: if the drive hangs, the >=540 s
        floor forces DONE regardless — this state cannot shadow or delay it, and a
        re-drive that finishes early (every leg visited before the floor) is free to end
        the question right then, with no forced idle wait.
        """
        _safe_explore(self._explore, io, self.plan, self.world, self._note_swallowed)
        # Re-read the world AFTER driving this tick so completion is observed the moment it
        # happens (the top-of-tick probe reflects the PRE-drive state), rather than lagging a
        # tick and emitting one crumb past arrival.
        self.world = _safe_probe(self._probe, io, self._note_swallowed)
        if self.world.drive_complete and self.world.legs_visited:
            self._to(State.DONE, "route drive complete")

    _HANDLERS: dict[State, Callable[["QuestionController", RobotIO], None]] = {
        State.PARSING: _tick_parsing,
        State.ORIENT: _tick_orient,
        State.EXPLORE_EXECUTE: _tick_explore,
        State.VERIFY: _tick_verify,
        State.ANSWER: _tick_answer,
        State.DRIVE_OUT: _tick_drive_out,
    }

    # ------------------------------------------------------------------ early answer
    def _early_answer_ready(self) -> bool:
        """Asymmetric early-answer gate (architecture §1 row 8).

        NUMERICAL: fire as soon as the injected count is stable (margin >=25% AND all
        contributing instances >=3 obs). IF: fire once the resolved route is fully
        grounded AND that state has held stable for IF_EARLY_ANSWER_STABLE_TICKS
        consecutive ticks (#181) -- see _if_route_ready. OBJECT_REFERENCE: no
        aggressive early-finish (fall through to budget/verify).
        """
        if self.qtype is QType.NUMERICAL:
            return self.world.stability.stable
        if self.qtype is QType.INSTRUCTION_FOLLOWING:
            return self._if_route_ready()
        return False

    def _if_route_ready(self) -> bool:
        """IF early-answer readiness (#181): every leg grounded, stably.

        Authoritative source: WorldView.ungrounded_subgoals, which the head/factory
        seam (core.heads.factory._assemble_worldview) fills from
        InstructionHead.ungrounded_subgoals() (core/heads/instruction.py). A leg only
        counts as grounded there once its resolved anchor's contributing instance
        record(s) individually clear MIN_GROUND_OBS (>=3 observations) -- reading this
        field keeps grounding truth in one place (the head owns relaxation/demotion/
        runner-up logic) instead of the controller re-deriving it from plan/scene as a
        proxy that could drift out of sync.

        Stability: a leg can flip back to ungrounded mid-run (a CP3 anchor mismatch
        demotes the resolved instance and re-plans to the runner-up), so a single tick
        reading zero gaps is not proof the route stays resolved. This debounces the
        all-legs-grounded state across IF_EARLY_ANSWER_STABLE_TICKS consecutive ticks,
        mirroring the spirit of the NUMERICAL gate's per-instance observation floor
        (StabilitySignal.min_contrib_n_obs >= 3) as protection against a one-tick
        flicker. Any tick that reopens a gap resets the streak to zero, so the
        asymmetry invariant (never fire with a gap) is never weakened -- the streak can
        only advance while the route is genuinely fully resolved.
        """
        if self.plan is None:
            # No bound route exists (the parse failed or never produced a plan), so
            # WorldView.ungrounded_subgoals is the assembler's default zero, not the
            # instruction head's report -- the head was never constructed. That is
            # the ultimate gap: without this guard a parse failure would bank a
            # degenerate answer two ticks into EXPLORE_EXECUTE (verifier repro,
            # 5 Aug 2026) instead of exploring the budget as before #181.
            self._if_grounded_streak = 0
            return False
        if self.world.ungrounded_subgoals != 0:
            self._if_grounded_streak = 0
            return False
        self._if_grounded_streak += 1
        return self._if_grounded_streak >= IF_EARLY_ANSWER_STABLE_TICKS

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


import re

_NUMERICAL_PHRASES = ("how many", "number of", "count the", "count of")
_NUMERICAL_WORD_RE = re.compile(r"\bcount\b")

# Movement/instruction verbs that open (or open a clause of) an instruction-following
# question, per the upstream questions.json phrasing (e.g. "Go near the X and stop at
# the Y", "First, go to...", "Take the path...", "Head...", "Navigate...", "Move...",
# "Walk...", "Drive...", "Stop at...", "Pass by...", "Avoid...").
_INSTRUCTION_VERBS_RE = re.compile(
    r"\b("
    r"go|goes|going|go near|go to|go between|go past|"
    r"take the path|take a path|"
    r"head|navigate|move|walk|drive|"
    r"stop at|stop by|"
    r"pass by|pass|"
    r"avoid"
    r")\b"
)


def _infer_qtype(q: Question) -> QType:
    """Cheap qtype heuristic used until the parse settles it; total, defaults NUMERICAL-safe.

    Real qtype comes from the parse; this only seeds budgets before the plan lands.
    """
    text = (getattr(q, "text", "") or "").lower()
    qtype = getattr(q, "qtype", None)
    if isinstance(qtype, QType):
        return qtype
    if any(w in text for w in _NUMERICAL_PHRASES) or _NUMERICAL_WORD_RE.search(text):
        return QType.NUMERICAL
    if _INSTRUCTION_VERBS_RE.search(text):
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


def _safe_probe(probe: ProbeFn, io: RobotIO, on_error=None) -> WorldView:
    try:
        wv = probe(io)
        return wv if isinstance(wv, WorldView) else WorldView()
    except Exception as exc:
        if on_error is not None:
            on_error("probe", exc)
        return WorldView()


def _safe_call(fn, *args, on_error=None):
    try:
        return fn(*args)
    except Exception as exc:
        if on_error is not None:
            on_error("parse", exc)
        return None


def _safe_explore(fn: ExploreFn, io: RobotIO, plan, world, on_error=None) -> None:
    try:
        fn(io, plan, world)
    except Exception as exc:
        if on_error is not None:
            on_error("explore", exc)


def _safe_verify(fn: VerifyFn, io: RobotIO, plan, world, on_error=None):
    try:
        return fn(io, plan, world)
    except Exception as exc:
        if on_error is not None:
            on_error("verify", exc)
        return None
