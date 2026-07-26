"""Factory that wires the QuestionController's injected callables from a live scene + nav.

``build_callables(scene_index, ...)`` returns the ``{parse, explore, verify, probe}`` dict
:class:`~core.fsm.controller.QuestionController` expects, implementing the architecture §4
per-type strategies over a live :class:`~core.interfaces.SceneIndex` and the nav stack.

Wiring model
------------
A single mutable :class:`HeadState` is shared by all four callables. The FSM drives them:

* ``parse(question) -> plan``   — checkpoint 1. NOT this module's concern: an injected
  parse fn is used (default = the offline regex tier). Whatever it returns is latched onto
  the shared state so the heads specialise to the qtype.
* ``explore(io, plan, world)``  — one deterministic step: the qtype's head advances (count
  refresh / re-rank / route drive). Binds heads to the plan on first sight.
* ``verify(io, plan, world)``   — final publishable answer for the qtype (checkpoint 4 for
  OR; the terminal WaypointCmd for IF; the stable IntAnswer for NUMERICAL), or None to let
  the FSM fall through to the floor.
* ``probe(io) -> WorldView``    — assembles the WorldView from the heads' current state:
  scene, PartialResults (best_marker/count/first_anchor_pt), ungrounded_subgoals,
  StabilitySignal.

The LLM checkpoints (``llm_verify`` for OR clause verification, ``anchor_confirm`` for IF
arrival) are injected and STUBBED in tests; None == deterministic-only (trust the map).

Determinism / no network: everything runs offline against the injected scene + mocks.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from core.fsm.controller import WorldView
from core.fsm.floors import PartialResults
from core.interfaces import IntAnswer, MarkerBox, QType, Question, RobotIO, SceneIndex, WaypointCmd
from core.geometry.toolbox import DEFAULT_THRESHOLDS, Thresholds
from core.llm.timeout import DEFAULT_CALL_TIMEOUT_S, wrap_call_timeout
from core.parsing.vocab import PHRASES, SINGLE_NOUNS
from core.perception.detector import DetectorFn, refresh_prompt
from core.perception.scene_index import dump_instance_index
from core.plan_schema import Plan

from core.heads.explore_step import (
    AffinityFactory,
    ExploreHead,
    FrontierSelectFn,
    FuseHintFn,
    MissRecoveryFn,
    _plan_nouns,
    uniform_affinity,
)
from core.heads.instruction import AnchorConfirmFn, InstructionHead
from core.heads.numerical import NumericalHead
from core.heads.object_ref import LlmVerifyFn, ObjectRefHead, VerifierFn

#: Issue #34 "standing vocab": every canonical noun the offline parser knows (the 114-noun
#: training vocabulary — core.parsing.vocab's own docstring), single-word + multi-word
#: phrases collapsed to their canonical form. Handed to :func:`build_gdino_prompt` (via
#: :func:`refresh_prompt`) AFTER the latched question's own nouns, so the detector also
#: grounds on the rest of the known vocabulary at lower recall priority — not just the
#: exact nouns this one question mentioned (read-only reuse: core/parsing/vocab.py is
#: out of scope for this fix; nothing here mutates it).
_STANDING_VOCAB_NOUNS: tuple[str, ...] = tuple(sorted(set(SINGLE_NOUNS) | set(PHRASES.values())))


@dataclass
class HeadState:
    """Shared mutable state binding the four callables to one question's heads."""

    scene: SceneIndex
    thresholds: Thresholds = DEFAULT_THRESHOLDS
    affinity_fn: AffinityFactory = uniform_affinity
    llm_verify: LlmVerifyFn | None = None
    anchor_confirm: AnchorConfirmFn | None = None
    #: rich checkpoint seams (design doc wiring map); all None == today's behaviour.
    verifier: VerifierFn | None = None
    miss_recoverer: MissRecoveryFn | None = None
    frontier_selector: FrontierSelectFn | None = None
    #: CP-support hooks the seams need (all optional).
    remaining_s: Callable[[], float] | None = None  # CP4 90 s re-resolve rule
    budget_frac: Callable[[], float] | None = None  # CP2 >=60% coverage trigger
    forced_assembly: Callable[[], bool] | None = None  # issue #33 T-90 commit-obs gate
    tiles_fn: Callable[[], object] | None = None  # CP2 tile supplier
    fuse_hint: FuseHintFn | None = None  # CP2 provisional-instance fusion
    #: Issue #34 — the live perception detector (GroundingDinoDetector / FakeDetector),
    #: if any, whose ``.prompt`` gets rebuilt from the plan's nouns the moment it latches
    #: (see :meth:`bind`). None (default) is a no-op, matching today's behaviour when no
    #: detector is wired (VLA_DETECTOR=none, or an offline/replay caller with none to give).
    detector: DetectorFn | None = None

    plan: Plan | None = None
    numerical: NumericalHead | None = None
    object_ref: ObjectRefHead | None = None
    instruction: InstructionHead | None = None
    explore: ExploreHead | None = None

    def bind(self, plan: Plan | None) -> None:
        """Latch the plan, refresh the detector prompt, and construct the qtype's head(s)
        — each exactly once, the first time a non-None plan reaches either ``explore`` or
        ``verify`` (whichever the FSM calls first this run).

        Issue #34: refreshing ``self.detector``'s prompt HERE — rather than only at
        construction, when no question has latched yet and the prompt is forced empty —
        is what stops a ``GroundingDinoDetector`` from being silently inert for the whole
        run. This is the one seam both the adapter (``ros_adapter/adapter_node.py``,
        real ROS timer/callback threads) and the offline replay path
        (``core/runner/single.py`` -> ``build_callables`` -> this same ``bind``) drive
        identically, so the fix lands once for both callers.
        """
        if plan is None or self.plan is not None:
            return
        self.plan = plan
        refresh_prompt(self.detector, _plan_nouns(plan), _STANDING_VOCAB_NOUNS)
        if plan.qtype is QType.NUMERICAL:
            self.numerical = NumericalHead(plan=plan, thresholds=self.thresholds)
        elif plan.qtype is QType.OBJECT_REFERENCE:
            self.object_ref = ObjectRefHead(
                plan=plan,
                thresholds=self.thresholds,
                llm_verify=self.llm_verify,
                verifier=self.verifier,
                remaining_s=self.remaining_s,
            )
        elif plan.qtype is QType.INSTRUCTION_FOLLOWING:
            self.instruction = InstructionHead(
                plan=plan,
                thresholds=self.thresholds,
                anchor_confirm=self.anchor_confirm,
                budget_frac=self.budget_frac,  # H4c provisional-terminal commit gate
                forced_assembly=self.forced_assembly,  # issue #33 single-obs commit gate
            )
        # The explore head is always built (it may delegate to the IF head).
        self.explore = ExploreHead(
            plan=plan,
            affinity_fn=self.affinity_fn,
            instruction=self.instruction,
            miss_recoverer=self.miss_recoverer,
            fuse_hint=self.fuse_hint,
            budget_frac=self.budget_frac,
            tiles_fn=self.tiles_fn,
            frontier_selector=self.frontier_selector,
        )


def build_callables(
    scene_index: SceneIndex,
    *,
    parse: Callable[[Question], object | None] | None = None,
    affinity_fn: AffinityFactory | None = None,
    llm_verify: LlmVerifyFn | None = None,
    anchor_confirm: AnchorConfirmFn | None = None,
    verifier: VerifierFn | None = None,
    anchor_confirmer: AnchorConfirmFn | None = None,
    miss_recoverer: MissRecoveryFn | None = None,
    frontier_selector: FrontierSelectFn | None = None,
    remaining_s: Callable[[], float] | None = None,
    budget_frac: Callable[[], float] | None = None,
    forced_assembly: Callable[[], bool] | None = None,
    tiles_fn: Callable[[], object] | None = None,
    fuse_hint: FuseHintFn | None = None,
    detector: DetectorFn | None = None,
    thresholds: Thresholds = DEFAULT_THRESHOLDS,
    call_timeout_s: float = DEFAULT_CALL_TIMEOUT_S,
) -> dict:
    """Build the {parse, explore, verify, probe} callables for a QuestionController.

    scene_index: the live SceneIndex the heads resolve/count against.
    parse:       checkpoint-1 parse fn; default = the offline regex tier.
    affinity_fn: nouns -> ((x,y)->float) frontier bias; default uniform.
    detector:    the live perception detector (e.g. GroundingDinoDetector), if any. Its
                 ``.prompt`` is rebuilt from the plan's nouns + the standing vocab the
                 moment the plan latches (issue #34 — see ``HeadState.bind``), so a
                 detector constructed at boot with an empty prompt stops being silently
                 inert once a question arrives. None (default) is a no-op: nothing to
                 refresh, matching today's behaviour.

    Checkpoint seams (design doc wiring map; all default None == today's behaviour):
    * ``verifier``          — rich CP4 pre-answer verification (full contract). Falls back
      to the legacy ``llm_verify`` bool seam when None.
    * ``anchor_confirmer``  — rich CP3 anchor confirmation (vision contract). Accepted as an
      alias of ``anchor_confirm`` (the head auto-detects the legacy bool seam by signature).
    * ``miss_recoverer``    — CP2 detector-miss recovery (explore path).
    * ``frontier_selector`` — CP5 frontier selection (multi-room scenes).

    CP-support hooks: ``remaining_s`` (CP4 90 s re-resolve rule), ``budget_frac`` (CP2
    >=60% coverage trigger), ``forced_assembly`` (issue #33 — zero-arg -> True once the
    T-90 forced-assembly gate is reached; feeds the IF head's single-observation
    route-prefix commit gate), ``tiles_fn`` (CP2 tile supplier), ``fuse_hint`` (CP2 fusion).

    Off-tick-thread safety (SYS-F8): every seam that can trigger a *provider call* (``parse``,
    ``llm_verify``, ``anchor_confirm``/``anchor_confirmer``, ``verifier``, ``miss_recoverer``,
    ``frontier_selector``) is wrapped with a hard per-call timeout (``call_timeout_s``, default
    20 s) HERE, at the injection boundary — so a hung network can never stall the 5 Hz tick
    past that bound regardless of whether the individual call site remembered to wrap its
    ChatFn. The fast local support hooks (``budget_frac``, ``forced_assembly``, ``remaining_s``,
    ``tiles_fn``, ``fuse_hint``, ``affinity_fn``) are NOT wrapped: they are synchronous
    map/clock reads, not provider calls, and wrapping them would only add thread-handoff
    overhead. Unconfigured (``None``) seams pass through untouched so the offline path
    stays deterministic.
    """
    # Wrap only the provider-triggering seams (None -> None; see wrap_call_timeout).
    def _tw(fn):
        return wrap_call_timeout(fn, call_timeout_s)

    parse = _tw(parse)
    llm_verify = _tw(llm_verify)
    anchor_confirm = _tw(anchor_confirm)
    anchor_confirmer = _tw(anchor_confirmer)
    verifier = _tw(verifier)
    miss_recoverer = _tw(miss_recoverer)
    frontier_selector = _tw(frontier_selector)

    state = HeadState(
        scene=scene_index,
        thresholds=thresholds,
        affinity_fn=affinity_fn if affinity_fn is not None else uniform_affinity,
        llm_verify=llm_verify,
        anchor_confirm=anchor_confirmer if anchor_confirmer is not None else anchor_confirm,
        verifier=verifier,
        miss_recoverer=miss_recoverer,
        frontier_selector=frontier_selector,
        remaining_s=remaining_s,
        budget_frac=budget_frac,
        forced_assembly=forced_assembly,
        tiles_fn=tiles_fn,
        fuse_hint=fuse_hint,
        detector=detector,
    )
    # Issue #84 "prompt dead zone": GroundingDinoDetector is constructed at boot with no
    # question latched yet (question_nouns=(), vocab_nouns=()), so its prompt stays "" and
    # every __call__ short-circuits to zero detections until HeadState.bind() first fires
    # on plan latch (core.heads.factory.HeadState.bind). Parsing (LLM ladder or regex) can
    # take real wall-clock time, and exploration/perception keyframes tick throughout that
    # window — so without this, every keyframe before latch runs the detector blind, even
    # though the standing 114-noun vocab (:data:`_STANDING_VOCAB_NOUNS`) is known up front
    # and does not depend on the question at all. Priming it here, immediately at
    # build_callables() time (before any plan exists), closes that window: the detector
    # grounds on the standing vocab from the very first keyframe, and `bind()` still fully
    # rebuilds the prompt (question nouns first, standing vocab after) the moment the plan
    # latches, same as before this fix.
    refresh_prompt(detector, (), _STANDING_VOCAB_NOUNS)
    parse_fn = parse if parse is not None else _default_parse

    def _explore(io: RobotIO, plan, world: WorldView) -> None:
        state.bind(plan if isinstance(plan, Plan) else None)
        _advance_heads(state, io)

    def _verify(io: RobotIO, plan, world: WorldView):
        state.bind(plan if isinstance(plan, Plan) else None)
        _advance_heads(state, io)  # one last refresh so the answer reflects the newest map
        return _final_answer(state)

    def _probe(io: RobotIO) -> WorldView:
        return _assemble_worldview(state)

    return {"parse": parse_fn, "explore": _explore, "verify": _verify, "probe": _probe}


# --------------------------------------------------------------------------- internals


def _advance_heads(state: HeadState, io: RobotIO) -> None:
    """Step whichever head(s) the bound qtype uses for this tick."""
    if state.plan is None:
        return
    qt = state.plan.qtype
    if qt is QType.NUMERICAL and state.numerical is not None:
        state.numerical.advance(state.scene)
    elif qt is QType.OBJECT_REFERENCE and state.object_ref is not None:
        state.object_ref.advance(state.scene)
    # Exploration/execution step (IF drives via its head inside ExploreHead).
    if state.explore is not None:
        state.explore.advance(io, state.scene)


def _final_answer(state: HeadState):
    """The publishable answer for the bound qtype, or None -> FSM floor."""
    if state.plan is None:
        return None
    qt = state.plan.qtype
    answer = None
    if qt is QType.NUMERICAL and state.numerical is not None:
        answer = state.numerical.answer()
    elif qt is QType.OBJECT_REFERENCE and state.object_ref is not None:
        answer = state.object_ref.verify()
    elif qt is QType.INSTRUCTION_FOLLOWING and state.instruction is not None:
        answer = state.instruction.terminal_waypoint()
    if answer is not None and state.scene is not None:
        # Opt-in live diagnostics (issues #84/#89): snapshot the instance index at
        # answer time. No-op unless VLA_INSTANCE_DUMP_PATH is set.
        dump_instance_index(state.scene, tag="answer_time")
    return answer


def _assemble_worldview(state: HeadState) -> WorldView:
    """Assemble the per-tick WorldView from the heads' current state (probe)."""
    partial = PartialResults()
    ungrounded = 0
    stability = None
    drive_complete = False

    if state.numerical is not None:
        partial.count = state.numerical.count
        stability = state.numerical.signal()
    if state.object_ref is not None:
        state.object_ref.publish_partial(partial)
    if state.instruction is not None:
        ungrounded = state.instruction.ungrounded_subgoals()
        pt = state.instruction.first_anchor_pt()
        if pt is not None:
            partial.first_anchor_pt = pt
        # IF-F4: surface the head's continue-drive completion signal so the FSM's
        # DRIVE_OUT state can stop driving once the route is finished (arrival, or
        # exhausted with no replan budget). Read-only.
        drive_complete = state.instruction.drive_complete()

    wv = WorldView(
        scene=state.scene,
        partial=partial,
        ungrounded_subgoals=ungrounded,
        drive_complete=drive_complete,
    )
    if stability is not None:
        wv.stability = stability
    return wv


def _default_parse(question: Question):
    """Offline default parse: the deterministic regex tier (no network)."""
    from core.parsing.regex_tier import parse_regex

    text = getattr(question, "text", question)
    return parse_regex(text)
