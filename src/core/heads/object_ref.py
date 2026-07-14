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

import inspect
import re
from dataclasses import dataclass
from typing import Any, Callable

from core.interfaces import InstanceRecord, MarkerBox, QType, SceneIndex
from core.fsm.floors import PartialResults
from core.geometry.toolbox import (
    DEFAULT_THRESHOLDS,
    PredResult,
    ResolveResult,
    Thresholds,
    resolve,
)
from core.parsing.regex_tier import _REL_TOKENS  # read-only: relation-token -> Pred map
from core.plan_schema import Anchor, Clause, Plan, Pred, TargetSpec
from core.perception.dimension_priors import clamp_record_marker

#: Re-observation gate (OR-F8): a winner with fewer than this many distinct observations is
#: provisional-only (e.g. a CP2 hallucination-recovery instance, n_obs=1) and must never be
#: committed as the published /selected_object_marker without re-observation.
REOBS_MIN_N_OBS: int = 2

# Legacy narrow per-clause verifier (backward compat):
#     (plan, candidate_summary, pass_matrix_text) -> keep-winner bool
LlmVerifyFn = Callable[[Plan, str, str], bool]

# Rich CP4 verifier (design doc §CP4 full contract). Consumes the whole context and
# returns a VerificationOutcome-shaped object (duck-typed: ``.action`` in
# {"keep","runner_up","re_resolve"} and ``.winner``). Called with keyword args so the
# checkpoint module's ``run(question, winner, runner_up, winner_facts, notes,
# resolve_again=None)`` signature binds directly.
#     run(question, winner, runner_up, winner_facts, notes, remaining_s, resolve_again)
#       -> VerificationOutcome
VerifierFn = Callable[..., Any]


@dataclass
class ObjectRefHead:
    """Ranks object-reference candidates and maintains the best marker each tick.

    Two verification seams are supported (checkpoint 4):

    * ``verifier`` — the rich CP4 seam (design doc §CP4 full contract). If set, it is
      consulted once per :meth:`verify` call with the winner, runner-up, the winner's
      per-clause pass matrix, ``plan.notes``, the remaining question-budget seconds, and a
      ``resolve_again`` re-resolve hook; it returns a VerificationOutcome-shaped object
      whose ``.action`` ("keep"/"runner_up"/"re_resolve") + ``.winner`` we apply.
    * ``llm_verify`` — the legacy narrow bool seam ``(plan, summary, matrix) -> bool``,
      preserved for backward compatibility. Used when no rich ``verifier`` is set (or when
      a callable that matches the old 3-arg bool signature is injected as ``verifier``).

    When both are None the head is deterministic (trust the toolbox rank).
    """

    plan: Plan | None = None
    thresholds: Thresholds = DEFAULT_THRESHOLDS
    llm_verify: LlmVerifyFn | None = None
    verifier: VerifierFn | None = None
    #: zero-arg callable -> remaining question-budget seconds (feeds the CP4 re-resolve
    #: 90 s rule). Defaults to +inf (always eligible) when None.
    remaining_s: Callable[[], float] | None = None
    #: live scene retained so the re-resolve hook can re-run resolve on demand.
    _scene: SceneIndex | None = None

    _result: ResolveResult | None = None
    best_candidate: InstanceRecord | None = None
    best_marker: MarkerBox | None = None

    def __post_init__(self) -> None:
        # Backward compat: a callable injected as ``verifier`` that actually matches the
        # legacy narrow bool seam (3 positional params, none named ``winner``) is treated
        # as an ``llm_verify`` so old callers/tests keep working through the rich kwarg.
        if self.verifier is not None and _is_legacy_bool_seam(self.verifier):
            if self.llm_verify is None:
                self.llm_verify = self.verifier  # type: ignore[assignment]
            self.verifier = None

    # ------------------------------------------------------------------ update
    def advance(self, scene: SceneIndex | None) -> None:
        """Re-rank against the live scene; refresh best_candidate / best_marker."""
        if scene is None or self.plan is None or self.plan.target is None:
            return
        self._scene = scene
        res = resolve(self.plan.target, scene, self.thresholds)
        self._result = res
        if res.candidates_ranked:
            self.best_candidate = res.candidates_ranked[0]
            self.best_marker = clamp_record_marker(self.best_candidate)

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

        Provisional-commit guard (OR-F8): a winner that is provisional-only — fewer than
        ``REOBS_MIN_N_OBS`` distinct observations, e.g. a CP2 hallucination-recovery
        instance (n_obs=1) — is never committed as the published marker. We fall through to
        the next non-provisional ranked candidate; if none qualifies, we publish nothing
        (return None) rather than a confident wrong box on empty space.
        """
        res = self._result
        if res is None or not res.candidates_ranked:
            return self.best_marker
        if self.verifier is not None:
            winner = self._cp4_winner(res)
        else:
            winner = self._verified_winner(res)
        winner = self._first_committable(winner, res)
        if winner is None:
            # Only provisional-only candidates remain: refuse to publish an unverified box.
            self.best_candidate = None
            self.best_marker = None
            return None
        self.best_candidate = winner
        self.best_marker = clamp_record_marker(winner)
        return self.best_marker

    def _first_committable(
        self, winner: InstanceRecord, res: ResolveResult
    ) -> InstanceRecord | None:
        """Return the winner if committable, else the first non-provisional ranked
        candidate after it, else None (OR-F8 provisional-commit guard)."""
        if winner is not None and winner.n_obs >= REOBS_MIN_N_OBS:
            return winner
        for cand in res.candidates_ranked:
            if cand.n_obs >= REOBS_MIN_N_OBS:
                return cand
        return None

    # -------------------------------------------------------------- rich CP4 seam
    def _cp4_winner(self, res: ResolveResult) -> InstanceRecord:
        """Apply the rich CP4 verifier's three-way verdict (design doc §CP4).

        confirm -> keep winner; runner_up -> the runner-up (swap once); re_resolve ->
        the outcome's re-resolved winner. Any failure/degenerate reply keeps the
        deterministic top rank (CP4 only ever improves on clause-slip).
        """
        winner = res.candidates_ranked[0]
        runner_up = res.candidates_ranked[1] if len(res.candidates_ranked) > 1 else None
        winner_facts = res.pass_matrix.get(winner.instance_id, [])
        notes = getattr(self.plan, "notes", "") or ""
        question = getattr(self.plan, "question_raw", "") or ""
        try:
            outcome = self.verifier(
                question=question,
                winner=winner,
                runner_up=runner_up,
                winner_facts=winner_facts,
                notes=notes,
                resolve_again=self._make_resolve_again(),
            )
        except TypeError:
            # A verifier that doesn't accept the full kwarg contract (e.g. a bare
            # positional callable) is retried positionally; still failing -> keep winner.
            try:
                outcome = self.verifier(
                    question, winner, runner_up, winner_facts, notes,
                    self._make_resolve_again(),
                )
            except Exception:
                return winner
        except Exception:
            return winner  # a dark/broken checkpoint trusts the deterministic rank

        new_winner = getattr(outcome, "winner", None)
        action = getattr(outcome, "action", "keep")
        if action in ("runner_up", "re_resolve") and isinstance(new_winner, InstanceRecord):
            return new_winner
        return winner

    def _make_resolve_again(self) -> Callable[[str], InstanceRecord | None]:
        """Build the CP4 re-resolve hook (OR-F5): synthesize a REAL clause from the
        missed constraint, append it to a spec copy, and re-run resolve.

        The verifier's ``missed_constraint`` is a ``"<pred_word> <anchor noun>"`` phrase
        (prompt-constrained). We map the pred word through the regex tier's relation tokens
        into a :class:`Pred`, build ``Clause(pred, [Anchor(noun)])`` and append it to a copy
        of the base target. A non-superlative pred filters the pool; a superlative pred
        (``closest_to``/``farthest_from``) is picked up by ``resolve`` as the ranking clause
        — either can change the winner. The constraint string is also recorded on the plan
        notes for the audit trail.

        Falls back to the unchanged re-resolve (keep the deterministic ranking) when the
        constraint does not parse into a clause.
        """
        scene = self._scene
        base_target = self.plan.target if self.plan is not None else None

        def resolve_again(missed: str) -> InstanceRecord | None:
            if scene is None or base_target is None:
                return None
            # Audit trail: record the constraint the verifier surfaced.
            if self.plan is not None:
                add = f"[cp4 re-resolve] missed_constraint: {missed}"
                self.plan.notes = (self.plan.notes + " " + add).strip() if self.plan.notes else add
            clauses = list(base_target.clauses)
            synthesized = _synthesize_clause(missed)
            if synthesized is not None:
                clauses.append(synthesized)
            spec = TargetSpec(
                noun=base_target.noun,
                raw=base_target.raw,
                attributes=list(base_target.attributes),
                clauses=clauses,
            )
            res = resolve(spec, scene, self.thresholds)
            if not res.candidates_ranked:
                return None
            if synthesized is not None:
                # A real clause was applied; trust the re-ranked winner directly.
                return res.candidates_ranked[0]
            # Unparseable constraint: fall back to the text-match skip (keep-winner-ish).
            token = (missed or "").strip().lower()
            if token:
                for cand in res.candidates_ranked:
                    rows = res.pass_matrix.get(cand.instance_id, [])
                    if not _contradicts_constraint(rows, token):
                        return cand
            return res.candidates_ranked[0]

        return resolve_again

    # ------------------------------------------------------------ legacy bool seam
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


def _is_legacy_bool_seam(fn: Callable) -> bool:
    """True if ``fn`` matches the legacy ``(plan, summary, matrix) -> bool`` bool seam.

    Heuristic (design: "inspect.signature"): exactly 3 positional-capable params and none
    named ``winner`` (the rich seam's marker keyword). A ``*args`` callable or an
    unintrospectable one is treated as rich (safer to pass full context).
    """
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    params = list(sig.parameters.values())
    if any(p.kind is inspect.Parameter.VAR_POSITIONAL for p in params):
        return False
    if any(p.name in ("winner", "winner_facts", "runner_up") for p in params):
        return False
    positional = [
        p for p in params
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    return len(positional) == 3


def _pred_from_word(word: str) -> Pred | None:
    """Map a CP4 missed-constraint pred word to a :class:`Pred` (OR-F5).

    Accepts the enum-value form the prompt emits (``closest_to``, ``on``, ``near``,
    ``between``, ...) directly, then falls back to matching the regex tier's relation
    tokens (which use whitespace, e.g. ``closest to``) so a spaced surface form still maps.
    Returns None when the word names no known relation.
    """
    w = (word or "").strip().lower()
    if not w:
        return None
    # 1. exact enum value ("closest_to", "on", "between", ...)
    try:
        return Pred(w)
    except ValueError:
        pass
    # 2. underscore -> space, match against the regex tier's ordered token patterns.
    spaced = w.replace("_", " ")
    for pat, pred in _REL_TOKENS:
        if pred is not None and re.fullmatch(pat, spaced):
            return pred
    return None


def _synthesize_clause(missed: str) -> Clause | None:
    """Build ``Clause(pred, [Anchor(noun)])`` from a ``"<pred_word> <anchor noun>"`` phrase.

    The first whitespace-delimited token is the relation word; the remainder is the anchor
    noun. Returns None when the phrase is empty, has no anchor noun, or the pred word does
    not map to a known relation — the caller then keeps the deterministic ranking.
    """
    parts = (missed or "").strip().split()
    if len(parts) < 2:
        return None
    pred = _pred_from_word(parts[0])
    if pred is None:
        return None
    noun = " ".join(parts[1:]).strip().lower()
    if not noun:
        return None
    return Clause(pred=pred, anchors=[Anchor(noun=noun, raw=noun)])


def _contradicts_constraint(rows: list[PredResult], token: str) -> bool:
    """True if any FAILing clause explanation mentions the missed-constraint token.

    Deterministic text match: a candidate "contradicts" the surfaced constraint when a
    clause it FAILS names the constraint word. Candidates with no contradicting failure
    are preferred by the re-resolve hook.
    """
    for r in rows:
        if not r.passed and token in (r.explanation or "").lower():
            return True
    return False


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
