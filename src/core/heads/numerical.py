"""Numerical answer head — set-cardinality with cross-tick count-stability tracking.

Architecture §4 (capped effort): the LLM builds the filter once (here the parsed
``TargetSpec`` IS the filter); a deterministic set-cardinality over tracked,
NMS-deduplicated instances via :func:`core.geometry.toolbox.counting` produces the
count; we answer aggressively at count stability, and the FSM's watchdog supplies the
modal-integer floor.

Count stability (architecture §1 row 8, mirrored by ``fsm.StabilitySignal.stable``):
a count is stable iff the SAME count has held for ``>= stable_ticks`` consecutive
observations AND every instance contributing to that count has ``n_obs >= 3``. We
translate that into the ``StabilitySignal`` the FSM's early-answer gate expects:

* ``min_contrib_n_obs`` = min n_obs across the contributing instances (the ``>= 3``
  observation gate);
* ``winner_margin`` = a fraction >= 0.25 ONLY once the count has been held for
  ``stable_ticks`` ticks (so the FSM's ``winner_margin >= 0.25`` gate doubles as the
  "held long enough" gate); 0.0 before that.

The head is deterministic and holds no wall-clock: "ticks" are advance() calls.
Units: metres/`map` frame throughout (inherited from the toolbox).
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable

import numpy as np

from core.fsm.controller import StabilitySignal
from core.heads.scene_established import ESTABLISH_N_OBS, EstablishedView
from core.interfaces import InstanceRecord, IntAnswer, MatchTier, QType, SceneIndex
from core.geometry.toolbox import (
    DEFAULT_THRESHOLDS,
    Thresholds,
    CountResult,
    counting,
    has_unresolved_disambiguator,
)
from core.perception.dimension_priors import prior_for
from core.plan_schema import Plan

STABLE_TICKS: int = 3  # consecutive equal counts required before firing early
MIN_OBS: int = 3  # per-contributor observation floor (architecture §1 row 8)
STABLE_MARGIN: float = 0.30  # winner_margin reported once held (> the FSM's 0.25 gate)

# H15(b) answer-time observation gating. Once ANY instance of the queried noun has been
# seen this many times, the class is "established" and ghosts (n_obs==1) are dropped from
# the count; instances with n_obs>=GATE_MIN_OBS still count. Below the establish
# threshold (cold start) every instance counts, so a genuine cold count is not starved.
# ESTABLISH_N_OBS itself now lives in `core.heads.scene_established` (#184: shared with
# OBJECT_REFERENCE's candidate/anchor gating) and is re-imported above.
GATE_MIN_OBS: int = 2     # once established, count only instances with n_obs >= this

# H15(c) coverage-gated early answer: minimum exploration-coverage fraction required
# (in addition to count stability) before the head reports a firing winner_margin.
COVERAGE_MIN_FRAC: float = 0.5

# (#109) Release fraction for the #93 dropped-disambiguator hold. `_disambiguator_unresolved`
# (below) withholds `stable` indefinitely while the nested disambiguator class hasn't been
# detected -- correct when the class simply hasn't been *found yet* (more ticks help), wrong
# when the class genuinely does not exist in the scene (more ticks buy nothing: `answer()`
# still publishes the same widened count at `past_explore_budget`, so the hold only spends
# NUMERICAL's soft explore budget, `core.interfaces.EXPLORE_BUDGET_S[QType.NUMERICAL]`
# (210 s), for zero accuracy gain -- see issue #109).
#
# The trade this constant names: releasing early answers on incomplete search coverage
# (the exact risk #93 was filed to close); releasing at 1.0 (never early) wastes the whole
# 210 s explore window whenever the class is absent, which is the status quo #109 reports.
# 0.85 reuses `core.heads.instruction.PROVISIONAL_COMMIT_FRAC` -- the codebase's existing
# convention for "how much of the relevant budget must elapse before treating persistent
# absence as signal rather than noise" for this exact family of drop-and-guess gates (see
# instruction.py's H4c gate). Unlike PROVISIONAL_COMMIT_FRAC (verified against issue #99 to
# land on the SAME tick as `forced_assembly`, i.e. no-op there), this constant is scaled
# against NUMERICAL's own 210 s soft explore budget, not the 600 s whole-question budget:
# 0.85 * 210 s = 178.5 s, a real 31.5 s (~15%) short of the 210 s hard cutoff where
# `past_explore_budget` forces VERIFY regardless. So this DOES buy back exploration time in
# the terminal case, unlike the #99 finding for the IF head.
#
# Caveat (generalization protocol): 0.85 is carried over from an existing, independently-
# reasoned constant, not fit to any sample from this issue's own evidence -- flagged as
# thin evidence should it ever need reconsideration for NUMERICAL specifically.
DISAMBIGUATOR_RELEASE_FRAC: float = 0.85

# (#151) Anchor-established observation floor for relation-clause filtering. The
# toolbox's clause evaluation (`counting()` -> `_filter_and` -> `_eval_clause` ->
# `_resolve_anchor`) is EXISTENTIAL over every resolved anchor instance: a
# candidate passes if the relation holds against ANY of them. That is correct
# when the anchor noun genuinely has several real instances (several sofas, say)
# but is exactly wrong when most of the "anchor" pool is single-frame detector
# noise: replaying the live 713413 archive (reports/cluster_verify/713413/debug)
# shows livingroom_3's 'tv cabinet' resolves to 19 instances for a scene with one
# physical TV cabinet, most with n_obs<=3 and scattered across the whole room --
# "on(any of 19)" is then satisfied by nearly every photo in the scene (24 pool
# -> 9 counted, true answer 2). The target side already has exactly this
# established/ghost distinction (ESTABLISH_N_OBS/_answer_min_obs, H15(b)); this
# constant applies the SAME notion to the ANCHOR side of a relation clause,
# reusing the existing threshold rather than fitting a new one to this issue's
# own sample (generalization protocol) -- an anchor instance participates in
# clause matching only once it has been seen ESTABLISH_N_OBS times. See
# `_count()` for the fail-open guarantee that a too-strict floor can only ever
# fall back to the unfiltered count, never fabricate a zero.
ANCHOR_ESTABLISHED_N_OBS: int = ESTABLISH_N_OBS

# (#189) A single established anchor instance can still under-approximate the true
# object it tracks -- the perception under-approximation dimension_priors.py already
# documents (single-viewpoint AABBs only see the observed faces). Live evidence
# (714757's hotel_room_1: "How many pillows are on the bed?" fell 8 -> 1, GT 4)
# traced to exactly this: the surviving n_obs>=3 bed anchor's own footprint sits a
# few tenths of a metre short of the true bed extent on one axis, so pillows that
# really do sit on the bed fail the support test's footprint-overlap gate against
# every established anchor, not because the on() predicate is wrong but because the
# anchor BOX it is evaluated against is too small.
#
# Fix: before the support test, rescue only the X/Y (footprint) axes of an
# established anchor that are MILDLY undersized relative to the anchor's own class
# dimension prior (core.perception.dimension_priors, read-only) -- i.e. below the
# class's 10th-percentile MIN extent, so a modest, expected sizing gap, but not so
# far below it (< ANCHOR_RESCUE_MILDNESS_FLOOR * min) that the box reads as a
# degenerate/mislabelled detection a real rescue would only compound (verified live:
# office_2's control fixture carries an established "table" instance with a 4 cm
# x-extent -- an obviously broken box, not a fragmented table -- and unconditionally
# inflating ANY below-class-min axis toward the class median flips a previously-
# correct exact answer). Rescued ranks jump to the class's TYPICAL (median) extent,
# not merely up to the min floor: the min floor alone (already applied elsewhere via
# core.perception.dimension_priors.clamp_extents) measured too small a nudge to
# close the archived hotel_room_1 gap in the same experiment.
#
# The Z axis is NEVER touched: the support test's vertical gate (on()'s upper-z-band
# check, core.geometry.toolbox.on) is a near-miss-sensitive tolerance, not a
# footprint-overlap gate, and letting a Z rescue quietly re-open it risks pulling in
# an established anchor a genuinely absent relation should not resolve against (this
# generalises past the training sample: 714757's home_building_1 "sofa" anchor pool
# fails the support test on a 3 cm Z near-miss for one candidate -- padding Z there
# would trade a defensible, unaffected fallback-tier answer for a strictly worse
# established-tier one; XY-only rescue leaves that row untouched, exactly preserving
# its current, evidence-ceiling behaviour).
#
# Scoped to NUMERICAL's relation-clause anchor filtering only (this module), not the
# shared `core.heads.scene_established.EstablishedView`/#160 toolbox clustering used
# by OBJECT_REFERENCE (#184): rescuing footprints is a support-test-specific
# correction, not a general anchor-resolution behaviour change.
ANCHOR_RESCUE_MILDNESS_FLOOR: float = 0.5  # only rescue a rank >= this fraction of class-min


def _rescue_undersized_anchor_footprint(record: InstanceRecord) -> InstanceRecord:
    """(#189) Inflate an established anchor's X/Y extent toward its class TYPICAL
    value, but only on ranks that are mildly undersized relative to the class's own
    10th-percentile MIN extent (>= :data:`ANCHOR_RESCUE_MILDNESS_FLOOR` of it, < it).
    Ranks already at/above class-min, and ranks far enough below it to look like a
    broken/degenerate detection rather than an ordinary sizing gap, are left exactly
    as measured. The Z axis is never touched (see the module note above). Fails open
    (returns ``record`` unchanged) when the class has no dimension prior, or when no
    rank actually qualifies.
    """
    prior = prior_for(record.label)
    if prior is None:
        return record
    ext = record.extents
    order = np.argsort(ext, kind="stable")
    sorted_ext = ext[order]
    mildly_undersized = (sorted_ext < prior.min_ext) & (
        sorted_ext >= ANCHOR_RESCUE_MILDNESS_FLOOR * prior.min_ext
    )
    if not np.any(mildly_undersized):
        return record
    adjusted = np.where(mildly_undersized, prior.typ_ext, sorted_ext)
    out = ext.copy()
    for rank, axis in enumerate(order):
        out[axis] = adjusted[rank]
    out[2] = ext[2]  # never touch Z -- see module note
    if np.allclose(out, ext):
        return record
    centre = (record.aabb_min + record.aabb_max) / 2.0
    lo = centre - out / 2.0
    hi = centre + out / 2.0
    lo[2] = record.aabb_min[2]
    hi[2] = record.aabb_max[2]
    return replace(record, aabb_min=lo, aabb_max=hi)


@dataclass(frozen=True)
class _FootprintRescuedAnchorView:
    """(#189) Wraps an :class:`EstablishedView` so ANCHOR-side lookups additionally
    run :func:`_rescue_undersized_anchor_footprint`; the exempt (target) noun passes
    through untouched, exactly mirroring ``EstablishedView``'s own exemption. A
    thin, NUMERICAL-local wrapper rather than a new ``EstablishedView`` option: this
    behaviour is specific to the support-clause anchor-filtering path (#151/#189),
    not something OBJECT_REFERENCE's (#184) shared use of ``EstablishedView`` should
    also pick up.
    """

    inner: EstablishedView

    def _passthrough(self, noun: str) -> bool:
        return self.inner.exempt_noun is not None and noun == self.inner.exempt_noun

    def by_label(self, noun: str) -> list[InstanceRecord]:
        recs = self.inner.by_label(noun)
        if self._passthrough(noun):
            return recs
        return [_rescue_undersized_anchor_footprint(r) for r in recs]

    def by_label_tiered(self, noun: str) -> list[tuple[InstanceRecord, MatchTier]]:
        hits = self.inner.by_label_tiered(noun)
        if self._passthrough(noun):
            return hits
        return [(_rescue_undersized_anchor_footprint(r), t) for r, t in hits]

    def all_instances(self) -> list[InstanceRecord]:
        return self.inner.all_instances()


@dataclass
class NumericalHead:
    """Tracks the running count for a NUMERICAL plan across ticks.

    ``advance(scene)`` recomputes the deterministic count and updates the stability
    run; ``signal()`` returns the ``StabilitySignal`` the FSM reads; ``answer()``
    returns the current best :class:`IntAnswer`.
    """

    plan: Plan | None = None
    thresholds: Thresholds = DEFAULT_THRESHOLDS
    stable_ticks: int = STABLE_TICKS
    min_obs: int = MIN_OBS
    #: H15(c) seam — optional exploration-coverage probe returning a fraction in [0, 1].
    #: None (default) preserves the current stability-only early-fire behaviour. The
    #: FSM/factory owns wiring a real coverage signal here (see module note); the head
    #: only defines and consumes the seam.
    coverage_frac: Callable[[], float] | None = None
    #: (#109) Optional probe: fraction (0..1, expected clamped by the caller) of THIS head's
    #: own soft NUMERICAL explore budget (``EXPLORE_BUDGET_S[QType.NUMERICAL]``, 210 s) that
    #: has elapsed. Distinct from ``coverage_frac`` above (a SPATIAL map-coverage seam that
    #: is not wired anywhere in production today): this is the TIME proxy the factory
    #: derives from the already-wired whole-question ``budget_frac`` closure (see
    #: ``core.heads.factory``). None (default) preserves the pre-#109 behaviour: the
    #: dropped-disambiguator hold never releases early and rides out to
    #: ``past_explore_budget``. See ``DISAMBIGUATOR_RELEASE_FRAC``.
    explore_progress: Callable[[], float] | None = None

    count: int | None = None
    contrib_min_obs: int = 0
    _run_count: int | None = None
    _run_len: int = 0
    # Set by advance() when the scene index holds zero tracked instances overall (perception
    # dark/not-wired). Distinguishes that "no data at all" state from the ordinary "count is
    # None because advance() was never given a usable scene/plan" state below, so answer()
    # can withhold rather than fabricate a zero (see advance()'s empty-index branch).
    _index_empty: bool = False
    # (#93) Set by advance() when the current count rests on a dropped nested
    # disambiguator whose class simply hasn't been detected yet (as opposed to a
    # depth-limit drop -- see has_unresolved_disambiguator). signal() reads this to
    # withhold the "stable" verdict so the FSM doesn't bank an early answer on a
    # filter that silently widened to the whole category; the watchdog/budget floor
    # still guarantees an answer eventually via answer() (never withheld here).
    _disambiguator_unresolved: bool = False

    # ------------------------------------------------------------------ update
    def advance(self, scene: SceneIndex | None) -> None:
        """Recompute the count from the live scene and update the stability run."""
        if scene is None or self.plan is None or self.plan.target is None:
            self._reset_run()
            return
        if not scene.all_instances():
            # An empty index is absence of data, not an observation of zero — e.g. with
            # perception fully dark, EVERY noun would otherwise "count" to 0. Withhold the
            # count so the FSM floor's modal-count fallback answers instead; a genuine
            # zero (index non-empty, target noun just absent) still counts normally below.
            self.count = None
            self._index_empty = True
            self._reset_run()
            return
        self._index_empty = False
        min_obs = self._answer_min_obs(scene)
        result = self._count(scene, min_obs)
        n, ids = result.count, result.ids
        self.count = n
        self.contrib_min_obs = self._min_obs(scene, ids)
        self._disambiguator_unresolved = has_unresolved_disambiguator(result.audit)
        if n == self._run_count:
            self._run_len += 1
        else:
            self._run_count = n
            self._run_len = 1

    def _count(self, scene: SceneIndex, min_obs: int) -> CountResult:
        """(#151) Count against the live instance index, filtering the ANCHOR side
        of any relation clause to established instances, with a three-tier
        fall-open so a stricter filter can only ever shrink the count toward the
        truth and can never fabricate a zero from a filtering failure:

        1. A question with NO relation clause counts exactly as before this
           issue -- a single unfiltered `counting()` call, byte-identical to the
           pre-#151 path.
        2. A question WITH a clause counts first against the established-anchor
           view (`core.heads.scene_established.EstablishedView`, `exempt_noun` set to
           the target noun): the clause's toolbox predicate
           (``on``/``above``/``near``/``with``/...) evaluated only against
           anchor instances seen `ANCHOR_ESTABLISHED_N_OBS` times, so a
           single-frame ghost detection cannot inflate the count (#151's own
           evidence: 19 mostly-single-observation 'tv cabinet' detections
           scattered across a room otherwise satisfy on() for nearly every
           photo in the scene).
        3. If that comes back empty, fall back to the toolbox's plain,
           unfiltered-by-established-floor `counting()` result -- the anchor may
           simply not be established YET (cold start), not genuinely absent.
        4. If THAT is also empty, the clause is unevaluable against this scene
           (anchor class wholly absent, or every candidate fails a predicate the
           head cannot second-guess): per spec, fall all the way back to the
           target's raw category count with the clause dropped entirely, rather
           than answer 0 for a relation that could not be checked at all -- the
           pre-#151 undercount-by-overfiltering failure mode is impossible by
           construction.
        """
        target = self.plan.target
        if not target.clauses:
            return counting(target, scene, min_obs=min_obs, th=self.thresholds)

        established = EstablishedView(
            scene, floor=ANCHOR_ESTABLISHED_N_OBS, exempt_noun=target.noun
        )
        rescued = _FootprintRescuedAnchorView(established)
        filtered = counting(target, rescued, min_obs=min_obs, th=self.thresholds)
        if filtered.count > 0:
            return filtered

        raw = counting(target, scene, min_obs=min_obs, th=self.thresholds)
        if raw.count > 0:
            return raw

        unclaused = replace(target, clauses=[])
        return counting(unclaused, scene, min_obs=min_obs, th=self.thresholds)

    def _answer_min_obs(self, scene: SceneIndex) -> int:
        """H15(b): the observation floor to count at, given how established the noun is.

        Cold start (no instance of the queried noun yet seen ESTABLISH_N_OBS times) ->
        count everything (min_obs=1), so a genuine cold count is not starved. Once ANY
        instance of the noun is established, drop one-frame ghosts (min_obs=GATE_MIN_OBS).
        """
        if self.plan is None or self.plan.target is None:
            return 1
        peak = 0
        for rec in scene.by_label(self.plan.target.noun):
            if rec.n_obs > peak:
                peak = rec.n_obs
        return GATE_MIN_OBS if peak >= ESTABLISH_N_OBS else 1

    def _min_obs(self, scene: SceneIndex, ids: set[int]) -> int:
        if not ids:
            return 0
        by_id = {r.instance_id: r for r in scene.all_instances()}
        obs = [by_id[i].n_obs for i in ids if i in by_id]
        return min(obs) if obs else 0

    def _reset_run(self) -> None:
        self._run_count = None
        self._run_len = 0
        self._disambiguator_unresolved = False

    # ------------------------------------------------------------------ read-out
    def signal(self) -> StabilitySignal:
        """Translate the stability run into the FSM's early-answer StabilitySignal.

        H15(c): when a coverage probe is injected, count stability alone is not enough
        to fire early — exploration must also have covered at least COVERAGE_MIN_FRAC
        of the scene, guarding against locking in an under-count after seeing only part
        of the scene. With no probe injected (default), behaviour is stability-only.

        (#93) Nor is a held run enough when the count rests on a dropped nested
        disambiguator whose class hasn't been detected yet (`_disambiguator_unresolved`):
        a stable-looking count that widened "near the table WITH a vase" to "near any
        table" is not a verified count, it's an unverified filter that happened to hold
        steady. Reporting it as stable would let the FSM bank it early; withholding the
        verdict costs nothing but exploration time (answer() still always publishes the
        current count, so the always-answer guarantee is untouched) and gives perception
        more ticks to find the missing class before commitment.

        (#109) That hold has no escape hatch when the class genuinely is not in the
        scene: waiting then costs the WHOLE remaining explore window for no accuracy
        gain. If `explore_progress` is injected and has crossed
        `DISAMBIGUATOR_RELEASE_FRAC`, we've plausibly searched enough that "the class
        just isn't here" is the better inference, so the hold releases and `stable` can
        fire on the current (widened) count same as any other stable run. With no probe
        injected (default), behaviour is unchanged: the hold never releases early.
        """
        held = self._run_len >= self.stable_ticks and self._run_count is not None
        if held and self.coverage_frac is not None:
            held = self.coverage_frac() >= COVERAGE_MIN_FRAC
        if held and self._disambiguator_unresolved:
            released = (
                self.explore_progress is not None
                and self.explore_progress() >= DISAMBIGUATOR_RELEASE_FRAC
            )
            held = released
        margin = STABLE_MARGIN if held else 0.0
        return StabilitySignal(
            winner_margin=margin,
            min_contrib_n_obs=self.contrib_min_obs,
        )

    def answer(self) -> IntAnswer | None:
        """Current best integer answer, or None if there is nothing to publish.

        None signals the empty-index case (see advance()): the FSM's verify contract
        treats None as "no answer yet" and falls back to the FloorAnswers modal count
        instead of a fabricated 0. Absent that case, falls back to 0 before any count.
        """
        if self._index_empty:
            return None
        return IntAnswer(int(self.count) if self.count is not None else 0)


def is_numerical(plan: Plan | None) -> bool:
    return plan is not None and plan.qtype is QType.NUMERICAL
