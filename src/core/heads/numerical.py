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

from core.fsm.controller import StabilitySignal
from core.interfaces import IntAnswer, QType, SceneIndex
from core.geometry.toolbox import DEFAULT_THRESHOLDS, Thresholds, counting
from core.plan_schema import Plan

STABLE_TICKS: int = 3  # consecutive equal counts required before firing early
MIN_OBS: int = 3  # per-contributor observation floor (architecture §1 row 8)
STABLE_MARGIN: float = 0.30  # winner_margin reported once held (> the FSM's 0.25 gate)


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

    count: int | None = None
    contrib_min_obs: int = 0
    _run_count: int | None = None
    _run_len: int = 0

    # ------------------------------------------------------------------ update
    def advance(self, scene: SceneIndex | None) -> None:
        """Recompute the count from the live scene and update the stability run."""
        if scene is None or self.plan is None or self.plan.target is None:
            self._reset_run()
            return
        n, ids = counting(self.plan.target, scene, min_obs=1, th=self.thresholds)
        self.count = n
        self.contrib_min_obs = self._min_obs(scene, ids)
        if n == self._run_count:
            self._run_len += 1
        else:
            self._run_count = n
            self._run_len = 1

    def _min_obs(self, scene: SceneIndex, ids: set[int]) -> int:
        if not ids:
            return 0
        by_id = {r.instance_id: r for r in scene.all_instances()}
        obs = [by_id[i].n_obs for i in ids if i in by_id]
        return min(obs) if obs else 0

    def _reset_run(self) -> None:
        self._run_count = None
        self._run_len = 0

    # ------------------------------------------------------------------ read-out
    def signal(self) -> StabilitySignal:
        """Translate the stability run into the FSM's early-answer StabilitySignal."""
        held = self._run_len >= self.stable_ticks and self._run_count is not None
        margin = STABLE_MARGIN if held else 0.0
        return StabilitySignal(
            winner_margin=margin,
            min_contrib_n_obs=self.contrib_min_obs,
        )

    def answer(self) -> IntAnswer:
        """Current best integer answer (falls back to 0 before any count)."""
        return IntAnswer(int(self.count) if self.count is not None else 0)


def is_numerical(plan: Plan | None) -> bool:
    return plan is not None and plan.qtype is QType.NUMERICAL
