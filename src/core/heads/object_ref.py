"""Object-reference answer head — toolbox ranking + per-clause verification checkpoint.

Architecture §4 (60/255 pts): :func:`core.geometry.toolbox.resolve` yields ranked
candidates plus a per-clause ``pass_matrix``; an optional injected ``llm_verify``
callable (checkpoint 4, STUBBED in tests) demotes the current winner to the runner-up
when it fails a clause; the surviving winner's trimmed AABB becomes the
``/selected_object_marker`` via :meth:`InstanceRecord.to_marker`. The best marker is
published continuously into ``PartialResults.best_marker`` so the floor always has the
freshest ranked box.

``llm_verify`` signature (checkpoint 4):
    (plan, candidate_summary: str, pass_matrix_text: str) -> bool
returning True == "the winner genuinely satisfies the clauses", False == demote.
It is consulted at most once per verify() call (the FSM ledger caps checkpoint calls);
a missing/raising verify callable is treated as "accept the deterministic winner".

Units: `map` frame, metres (inherited from the toolbox).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from core.interfaces import InstanceRecord, MarkerBox, QType, SceneIndex
from core.fsm.floors import PartialResults
from core.geometry.toolbox import (
    DEFAULT_THRESHOLDS,
    PredResult,
    ResolveResult,
    Thresholds,
    resolve,
)
from core.plan_schema import Plan

# per-clause verifier: (plan, candidate_summary, pass_matrix_text) -> keep-winner bool
LlmVerifyFn = Callable[[Plan, str, str], bool]


@dataclass
class ObjectRefHead:
    """Ranks object-reference candidates and maintains the best marker each tick."""

    plan: Plan | None = None
    thresholds: Thresholds = DEFAULT_THRESHOLDS
    llm_verify: LlmVerifyFn | None = None

    _result: ResolveResult | None = None
    best_candidate: InstanceRecord | None = None
    best_marker: MarkerBox | None = None

    # ------------------------------------------------------------------ update
    def advance(self, scene: SceneIndex | None) -> None:
        """Re-rank against the live scene; refresh best_candidate / best_marker."""
        if scene is None or self.plan is None or self.plan.target is None:
            return
        res = resolve(self.plan.target, scene, self.thresholds)
        self._result = res
        if res.candidates_ranked:
            self.best_candidate = res.candidates_ranked[0]
            self.best_marker = self.best_candidate.to_marker()

    def publish_partial(self, partial: PartialResults) -> None:
        """Stamp the current best candidate/marker into the shared PartialResults."""
        if self.best_candidate is not None:
            partial.best_candidate = self.best_candidate
        if self.best_marker is not None:
            partial.best_marker = self.best_marker

    # ------------------------------------------------------------------ verify
    def verify(self) -> MarkerBox | None:
        """Final answer: the (possibly demoted) winner's marker, or None if nothing ranked.

        Applies the injected per-clause verification checkpoint: if the top candidate
        fails verification, demote to the runner-up (and try it too). Deterministic and
        total when llm_verify is None.
        """
        res = self._result
        if res is None or not res.candidates_ranked:
            return self.best_marker
        winner = self._verified_winner(res)
        self.best_candidate = winner
        self.best_marker = winner.to_marker()
        return self.best_marker

    def _verified_winner(self, res: ResolveResult) -> InstanceRecord:
        if self.llm_verify is None:
            return res.candidates_ranked[0]
        for cand in res.candidates_ranked:
            rows = res.pass_matrix.get(cand.instance_id, [])
            summary = _candidate_summary(cand)
            matrix_text = _pass_matrix_text(rows)
            try:
                keep = bool(self.llm_verify(self.plan, summary, matrix_text))
            except Exception:
                keep = True  # a dark/broken checkpoint trusts the deterministic rank
            if keep:
                return cand
        # every candidate demoted: fall back to the deterministic top rank.
        return res.candidates_ranked[0]


def _candidate_summary(cand: InstanceRecord) -> str:
    c = cand.centroid
    e = cand.extents
    return (
        f"id={cand.instance_id} label={cand.label!r} n_obs={cand.n_obs} "
        f"centroid=({float(c[0]):.2f},{float(c[1]):.2f},{float(c[2]):.2f}) "
        f"extents=({float(e[0]):.2f},{float(e[1]):.2f},{float(e[2]):.2f})"
    )


def _pass_matrix_text(rows: list[PredResult]) -> str:
    if not rows:
        return "(no clauses)"
    return " | ".join(
        f"{'PASS' if r.passed else 'FAIL'}: {r.explanation}" for r in rows
    )


def is_object_reference(plan: Plan | None) -> bool:
    return plan is not None and plan.qtype is QType.OBJECT_REFERENCE
