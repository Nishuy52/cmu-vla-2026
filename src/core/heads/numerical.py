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

from dataclasses import dataclass, field
from typing import Callable

from core.fsm.controller import StabilitySignal
from core.interfaces import IntAnswer, QType, SceneIndex
from core.geometry.toolbox import (
    DEFAULT_THRESHOLDS,
    Thresholds,
    counting,
    has_unresolved_disambiguator,
)
from core.plan_schema import Plan

STABLE_TICKS: int = 3  # consecutive equal counts required before firing early
MIN_OBS: int = 3  # per-contributor observation floor (architecture §1 row 8)
STABLE_MARGIN: float = 0.30  # winner_margin reported once held (> the FSM's 0.25 gate)

# H15(b) answer-time observation gating. Once ANY instance of the queried noun has been
# seen this many times, the class is "established" and ghosts (n_obs==1) are dropped from
# the count; instances with n_obs>=GATE_MIN_OBS still count. Below the establish
# threshold (cold start) every instance counts, so a genuine cold count is not starved.
ESTABLISH_N_OBS: int = 3  # a noun is "established" once some instance reaches this
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
        result = counting(self.plan.target, scene, min_obs=min_obs, th=self.thresholds)
        n, ids = result.count, result.ids
        self.count = n
        self.contrib_min_obs = self._min_obs(scene, ids)
        self._disambiguator_unresolved = has_unresolved_disambiguator(result.audit)
        if n == self._run_count:
            self._run_len += 1
        else:
            self._run_count = n
            self._run_len = 1

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
