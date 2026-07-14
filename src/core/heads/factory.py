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
from core.plan_schema import Plan

from core.heads.explore_step import (
    AffinityFactory,
    ExploreHead,
    FrontierSelectFn,
    FuseHintFn,
    MissRecoveryFn,
    uniform_affinity,
)
from core.heads.instruction import AnchorConfirmFn, InstructionHead
from core.heads.numerical import NumericalHead
from core.heads.object_ref import LlmVerifyFn, ObjectRefHead, VerifierFn


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
    tiles_fn: Callable[[], object] | None = None  # CP2 tile supplier
    fuse_hint: FuseHintFn | None = None  # CP2 provisional-instance fusion

    plan: Plan | None = None
    numerical: NumericalHead | None = None
    object_ref: ObjectRefHead | None = None
    instruction: InstructionHead | None = None
    explore: ExploreHead | None = None

    def bind(self, plan: Plan | None) -> None:
        """Latch the plan and construct the qtype's head(s) exactly once."""
        if plan is None or self.plan is not None:
            return
        self.plan = plan
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
    tiles_fn: Callable[[], object] | None = None,
    fuse_hint: FuseHintFn | None = None,
    thresholds: Thresholds = DEFAULT_THRESHOLDS,
) -> dict:
    """Build the {parse, explore, verify, probe} callables for a QuestionController.

    scene_index: the live SceneIndex the heads resolve/count against.
    parse:       checkpoint-1 parse fn; default = the offline regex tier.
    affinity_fn: nouns -> ((x,y)->float) frontier bias; default uniform.

    Checkpoint seams (design doc wiring map; all default None == today's behaviour):
    * ``verifier``          — rich CP4 pre-answer verification (full contract). Falls back
      to the legacy ``llm_verify`` bool seam when None.
    * ``anchor_confirmer``  — rich CP3 anchor confirmation (vision contract). Accepted as an
      alias of ``anchor_confirm`` (the head auto-detects the legacy bool seam by signature).
    * ``miss_recoverer``    — CP2 detector-miss recovery (explore path).
    * ``frontier_selector`` — CP5 frontier selection (multi-room scenes).

    CP-support hooks: ``remaining_s`` (CP4 90 s re-resolve rule), ``budget_frac`` (CP2
    >=60% coverage trigger), ``tiles_fn`` (CP2 tile supplier), ``fuse_hint`` (CP2 fusion).
    """
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
        tiles_fn=tiles_fn,
        fuse_hint=fuse_hint,
    )
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
    if qt is QType.NUMERICAL and state.numerical is not None:
        return state.numerical.answer()
    if qt is QType.OBJECT_REFERENCE and state.object_ref is not None:
        return state.object_ref.verify()
    if qt is QType.INSTRUCTION_FOLLOWING and state.instruction is not None:
        return state.instruction.terminal_waypoint()
    return None


def _assemble_worldview(state: HeadState) -> WorldView:
    """Assemble the per-tick WorldView from the heads' current state (probe)."""
    partial = PartialResults()
    ungrounded = 0
    stability = None

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

    wv = WorldView(scene=state.scene, partial=partial, ungrounded_subgoals=ungrounded)
    if stability is not None:
        wv.stability = stability
    return wv


def _default_parse(question: Question):
    """Offline default parse: the deterministic regex tier (no network)."""
    from core.parsing.regex_tier import parse_regex

    text = getattr(question, "text", question)
    return parse_regex(text)
