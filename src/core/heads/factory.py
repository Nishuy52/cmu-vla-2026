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

import json
import os
import time
from dataclasses import dataclass, field
from typing import Callable

from core.fsm.controller import WorldView
from core.fsm.floors import PartialResults
from core.interfaces import (
    EXPLORE_BUDGET_S,
    IntAnswer,
    MarkerBox,
    QUESTION_BUDGET_S,
    QType,
    Question,
    RobotIO,
    SceneIndex,
    WaypointCmd,
)
from core.geometry.toolbox import DEFAULT_THRESHOLDS, Thresholds
from core.llm.timeout import DEFAULT_CALL_TIMEOUT_S, wrap_call_timeout
from core.parsing.vocab import PHRASES, SINGLE_NOUNS
from core.perception.detector import DetectorFn, refresh_prompt
from core.perception.scene_index import dump_instance_index
from core.plan_schema import Anchor, Clause, Plan, RouteLeg, TargetSpec, AvoidSpec

from core.heads.explore_step import (
    AffinityFactory,
    ExploreHead,
    FrontierSelectFn,
    FuseHintFn,
    MissRecoveryFn,
    _plan_avoid_nouns,
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


def _numerical_explore_progress(
    budget_frac: Callable[[], float] | None,
) -> Callable[[], float] | None:
    """(#109) Derive NumericalHead's `explore_progress` seam from the already-wired
    whole-question `budget_frac` (elapsed / QUESTION_BUDGET_S, the SAME closure the live
    adapter / gt_battery wire for the IF head's H4c gate -- see core.heads.instruction).

    NUMERICAL's own soft explore budget (`EXPLORE_BUDGET_S[QType.NUMERICAL]`, 210 s) is
    much shorter than the 600 s whole-question window `budget_frac` is scaled to, so this
    rescales: elapsed = budget_frac() * QUESTION_BUDGET_S; progress = elapsed / 210 s,
    clamped to [0, 1]. None in -> None out (no signal wired -> old indefinite-hold
    behaviour, matching every other optional seam in this module).
    """
    if budget_frac is None:
        return None
    explore_budget_s = EXPLORE_BUDGET_S[QType.NUMERICAL]

    def _progress() -> float:
        elapsed = budget_frac() * QUESTION_BUDGET_S
        return max(0.0, min(1.0, elapsed / explore_budget_s))

    return _progress


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
        # Issue #173: avoid-anchor nouns must also reach the SHORT question-noun-only
        # caption, not only the full caption's full_only_nouns slot (issue #108's
        # original routing). A long (~100-phrase) full caption dilutes a phrase's
        # decode score toward zero (detector.py's own probe: 117 phrases decode zero
        # 'teapot' where 2 phrases decode it in 194/211 keyframes); the full caption
        # also only refreshes every 3rd tick at the higher 0.35 threshold, so an avoid
        # anchor routed there alone measured zero live detections across two IF runs
        # while the same phrase, present in another run's question caption, scored
        # 175 raw detections. Folding the avoid nouns into `question_nouns` here (a
        # 4- to ~6-phrase list) puts them in both: `refresh_prompt` derives
        # `.question_prompt` from `question_nouns` alone, and `question_nouns` is also
        # the top-priority (never-budget-dropped) slice of the full caption. This
        # preserves #108's intent -- avoid nouns still reach the full caption's breadth
        # pass (kept via `full_only_nouns` below, additive/unchanged) -- while fixing
        # the actual defect #108 didn't anticipate: the short pass's own low threshold
        # and tiny-caption calibration is exactly what avoid anchors need too, and the
        # short list stays small (target + a couple of anchors), nowhere near the
        # 117-phrase regime #108's docstring warns about.
        avoid_nouns = _plan_avoid_nouns(plan)
        refresh_prompt(
            self.detector,
            [*_plan_nouns(plan), *avoid_nouns],
            _STANDING_VOCAB_NOUNS,
            full_only_nouns=avoid_nouns,
        )
        if plan.qtype is QType.NUMERICAL:
            self.numerical = NumericalHead(
                plan=plan,
                thresholds=self.thresholds,
                # (#109) release hatch for the #93 dropped-disambiguator hold: derive
                # NUMERICAL's own soft-explore-budget-relative elapsed fraction from the
                # already-wired whole-question budget_frac (None-safe -- see
                # _numerical_explore_progress).
                explore_progress=_numerical_explore_progress(self.budget_frac),
            )
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


# --------------------------------------------------------------------------- instrumentation
#
# Issue #102: three investigations this week each hit the same wall -- an answer scored
# wrong, with no artifact showing whether the resolved Plan's restricting clause was ever
# evaluated (and ignored) or never resolved at all. dump_instance_index (#84/#89) already
# snapshots the live instance index at answer time; this adds the resolved Plan itself
# (target/clauses/anchors, recursing fully through nested Anchor.disambiguator clauses) plus
# whatever candidate-count context is cleanly reachable from here. Same contract as every
# other debug dump in this codebase: single env-var lookup, zero I/O when unset, failures
# swallowed -- a debug dump must never break the run it observes.

#: Path to append JSONL resolved-Plan records to. Unset (default) -> dump_plan is a no-op.
ENV_PLAN_DUMP_PATH: str = "VLA_PLAN_DUMP_PATH"

#: Depth guard for walking Anchor.disambiguator nesting (a Clause's Anchor can itself carry
#: a disambiguator Clause, recursively). The schema has no cycle in practice, but a debug
#: dump must never hang/stack-overflow on a pathological or (bugged) cyclic Plan.
_PLAN_DUMP_MAX_DEPTH: int = 16


def _anchor_to_dict(anchor: Anchor, depth: int) -> dict:
    if depth > _PLAN_DUMP_MAX_DEPTH:
        return {
            "noun": anchor.noun,
            "raw": anchor.raw,
            "attributes": list(anchor.attributes),
            "disambiguator": "<max-depth-exceeded>",
        }
    disamb = None
    if anchor.disambiguator is not None:
        disamb = _clause_to_dict(anchor.disambiguator, depth + 1)
    return {
        "noun": anchor.noun,
        "raw": anchor.raw,
        "attributes": list(anchor.attributes),
        "disambiguator": disamb,
    }


def _clause_to_dict(clause: Clause, depth: int) -> dict:
    return {
        "pred": clause.pred.value,
        "negated": clause.negated,
        "anchors": [_anchor_to_dict(a, depth + 1) for a in clause.anchors],
    }


def _target_to_dict(target: TargetSpec | None, depth: int) -> dict | None:
    if target is None:
        return None
    return {
        "noun": target.noun,
        "raw": target.raw,
        "attributes": list(target.attributes),
        "clauses": [_clause_to_dict(c, depth + 1) for c in target.clauses],
    }


def _route_leg_to_dict(leg: RouteLeg, depth: int) -> dict:
    return {"kind": leg.kind.value, "anchors": [_anchor_to_dict(a, depth + 1) for a in leg.anchors]}


def _avoid_to_dict(av: AvoidSpec, depth: int) -> dict:
    return {
        "between": [_anchor_to_dict(a, depth + 1) for a in av.between] if av.between else None,
        "near": _anchor_to_dict(av.near, depth + 1) if av.near is not None else None,
    }


def _plan_to_dict(plan: Plan) -> dict:
    """Full recursive walk of a resolved ``Plan`` (target/clauses/anchors/route/avoid),
    depth-capped against a pathological/cyclic ``Anchor.disambiguator`` nesting."""
    return {
        "qtype": plan.qtype.value,
        "question_raw": plan.question_raw,
        "parse_tier": plan.parse_tier,
        "notes": plan.notes,
        "target": _target_to_dict(plan.target, 0),
        "route": [_route_leg_to_dict(leg, 0) for leg in plan.route],
        "avoid": [_avoid_to_dict(av, 0) for av in plan.avoid],
    }


def dump_plan(state: "HeadState", tag: str = "answer_time", *, answer=None) -> None:
    """Append one JSONL record of the resolved ``Plan`` (+ whatever candidate-count
    context is cleanly reachable), if :data:`ENV_PLAN_DUMP_PATH` is set. No-op (no I/O
    at all) when unset, matching ``core.perception.scene_index.dump_instance_index``.

    Issue #102 honesty note -- true PER-CLAUSE before/after candidate counts (the
    motivating ask: "did the restricting clause resolve and get ignored, or never
    resolve at all") are NOT reachable here without changing scored-path code, so this
    does not fabricate them:

    * NUMERICAL: ``core.geometry.toolbox.counting`` computes one final ``CountResult``
      (cardinality of the fully-AND-filtered set) and never retains an intermediate
      per-clause survivor count; ``NumericalHead.advance`` (``core/heads/numerical.py``,
      not owned by this change) keeps only the final ``count``, discarding even
      ``CountResult.explanations``/``.audit``. What IS dumped: the WHOLE-TARGET
      before-any-clause census (``len(scene.by_label(noun))``) alongside the final
      answer count -- exactly the signal that would have shown the #101/#102 motivating
      defect (a numerical answer equal to the raw class census through a clause that
      never actually restricted anything).
    * OBJECT_REFERENCE: ``ObjectRefHead`` DOES retain the full ``ResolveResult``
      (``self._result``) from its last ``resolve()`` call, so this dumps its relaxation
      ``audit`` trail (which fallback rung, if any, produced the final survivors) and
      the final survivor count alongside the same before-any-clause census. The audit
      trail directly answers "resolved and ignored" (audit empty, clause genuinely
      applied) vs "never resolved" (audit shows category_only/drop_relation).
      ``ResolveResult.pass_matrix`` is per-clause PASS/FAIL but only over the FINAL
      survivor set, not a before-this-clause population, and is deliberately NOT
      reported here as a "per-clause count" -- doing so over ``hard_clauses`` (which
      the fallback ladder may have already dropped clauses from) would silently
      misattribute pass/fail to the wrong original clause index.
    * INSTRUCTION_FOLLOWING: no ``target``/count semantics apply; only the Plan
      structure (route/avoid) is dumped.

    Any failure (bad path, unwritable dir, etc.) is swallowed -- diagnostics must never
    break the run they are observing.

    Issue #159 item 1: called regardless of whether a final answer was produced (the
    ``answer`` kwarg, when passed, is recorded as ``answer_present`` in the record) --
    the "verify yielded nothing" runs are exactly the ones that need this snapshot most.
    """
    path = os.environ.get(ENV_PLAN_DUMP_PATH)
    if not path or state.plan is None:
        return
    try:
        record: dict = {
            "wall_time": time.time(),
            "tag": tag,
            "answer_present": answer is not None,
            "plan": _plan_to_dict(state.plan),
        }
        target = state.plan.target
        if target is not None and state.scene is not None:
            candidates: dict = {
                "noun": target.noun,
                "before_any_filter": len(state.scene.by_label(target.noun)),
            }
            qt = state.plan.qtype
            if qt is QType.NUMERICAL and state.numerical is not None:
                candidates["after_all_filters"] = state.numerical.count
                candidates["granularity"] = "whole_target_aggregate"
            elif qt is QType.OBJECT_REFERENCE and state.object_ref is not None:
                res = state.object_ref._result
                if res is not None:
                    candidates["after_all_filters"] = len(res.candidates_ranked)
                    candidates["relaxation_audit"] = [
                        {"step": r.step, "detail": r.detail} for r in res.audit
                    ]
                candidates["granularity"] = "whole_target_aggregate"
            record["candidates"] = candidates
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "a") as fh:
            fh.write(json.dumps(record) + "\n")
    except Exception:  # noqa: BLE001 - diagnostics must never break the run
        pass


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
    # Opt-in live diagnostics (issues #84/#89/#159): fire whether or not an answer was
    # produced -- the runs where verify/terminal_waypoint yields nothing are exactly the
    # ones that most need a snapshot of what was seen. answer_present in the dumped plan
    # record (see dump_plan) lets a reader tell the two cases apart.
    if state.scene is not None:
        # No-op unless VLA_INSTANCE_DUMP_PATH is set.
        dump_instance_index(state.scene, tag="answer_time")
    # No-op unless VLA_PLAN_DUMP_PATH is set (dump_plan also no-ops if state.plan is None).
    dump_plan(state, tag="answer_time", answer=answer)
    return answer


def _assemble_worldview(state: HeadState) -> WorldView:
    """Assemble the per-tick WorldView from the heads' current state (probe)."""
    partial = PartialResults()
    ungrounded = 0
    stability = None
    drive_complete = False
    legs_visited = True

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
        # Issue #183: the stricter per-leg ordered-visit signal DRIVE_OUT's exit gate
        # pairs drive_complete with (core.fsm.controller._tick_drive_out) — see
        # InstructionHead.all_legs_visited for why drive_complete alone is not enough.
        legs_visited = state.instruction.all_legs_visited()

    wv = WorldView(
        scene=state.scene,
        partial=partial,
        ungrounded_subgoals=ungrounded,
        drive_complete=drive_complete,
        legs_visited=legs_visited,
    )
    if stability is not None:
        wv.stability = stability
    return wv


def _default_parse(question: Question):
    """Offline default parse: the deterministic regex tier (no network)."""
    from core.parsing.regex_tier import parse_regex

    text = getattr(question, "text", question)
    return parse_regex(text)
