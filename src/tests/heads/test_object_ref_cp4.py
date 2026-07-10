"""CP4 rich verifier seam on ObjectRefHead: three-way verdict, re-resolve, backward compat.

Seams are exercised with scripted checkpoint-style stubs at the *seam signature level*
(a callable matching ``run(question, winner, runner_up, winner_facts, notes,
resolve_again=None) -> VerificationOutcome``) — the real CP4 builder is not called, so these
tests don't couple to the checkpoint module's internals.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.heads.object_ref import ObjectRefHead
from tests.heads._helpers import closest_clause, inst, near_clause, object_plan, scene


@dataclass
class _Outcome:
    """VerificationOutcome-shaped stub (duck-typed on .action / .winner)."""

    action: str
    winner: Any
    verdict: str = ""
    missed_constraint: str | None = None
    reason: str = ""


def _orhead(sc, plan):
    head = ObjectRefHead(plan=plan)
    head.advance(sc)
    return head


# --------------------------------------------------------------------------- confirm
def test_cp4_confirm_keeps_winner():
    sc = scene(inst(1, "chair", centroid=(0, 0, 0)), inst(2, "chair", centroid=(5, 0, 0)))
    head = _orhead(sc, object_plan("chair"))
    top = head.best_candidate.instance_id

    def verifier(*, question, winner, runner_up, winner_facts, notes, resolve_again):
        return _Outcome("keep", winner, "confirm")

    head.verifier = verifier
    head.verify()
    assert head.best_candidate.instance_id == top


# --------------------------------------------------------------------------- runner_up
def test_cp4_runner_up_swaps_once():
    sc = scene(
        inst(1, "chair", centroid=(0, 0, 0)),
        inst(2, "chair", centroid=(5, 0, 0)),
        inst(3, "table", centroid=(0.5, 0, 0)),
    )
    head = _orhead(sc, object_plan("chair", clauses=[closest_clause("table")]))
    res = head._result
    runner = res.candidates_ranked[1]

    def verifier(*, question, winner, runner_up, winner_facts, notes, resolve_again):
        return _Outcome("runner_up", runner_up, "runner_up")

    head.verifier = verifier
    head.verify()
    assert head.best_candidate.instance_id == runner.instance_id


def test_cp4_runner_up_with_no_runner_up_keeps_winner():
    sc = scene(inst(1, "chair", centroid=(0, 0, 0)))
    head = _orhead(sc, object_plan("chair"))

    # winner is passed through; runner_up is None so the stub returns keep-on-winner.
    def verifier(*, question, winner, runner_up, winner_facts, notes, resolve_again):
        assert runner_up is None
        return _Outcome("keep", winner, "runner_up")

    head.verifier = verifier
    m = head.verify()
    assert m is not None
    assert head.best_candidate.instance_id == 1


# --------------------------------------------------------------------------- re_resolve
def test_cp4_neither_re_resolve_uses_hook_result():
    sc = scene(inst(1, "chair", centroid=(0, 0, 0)), inst(2, "chair", centroid=(5, 0, 0)))
    head = _orhead(sc, object_plan("chair"))
    # The stub calls the head-supplied resolve_again hook and answers with its result.
    seen = {}

    def verifier(*, question, winner, runner_up, winner_facts, notes, resolve_again):
        new = resolve_again("must be red")
        seen["new"] = new
        return _Outcome("re_resolve", new, "neither", missed_constraint="must be red")

    head.verifier = verifier
    head.verify()
    # re_resolve hook re-ran resolve and returned a real candidate; head adopts it.
    assert seen["new"] is not None
    assert head.best_candidate.instance_id == seen["new"].instance_id
    # audit trail records the missed constraint text.
    assert "must be red" in head.plan.notes


def test_cp4_re_resolve_records_constraint_in_audit_even_when_kept():
    sc = scene(inst(1, "chair"))
    head = _orhead(sc, object_plan("chair"))

    def verifier(*, question, winner, runner_up, winner_facts, notes, resolve_again):
        resolve_again("greenish")  # exercise the hook; keep the winner
        return _Outcome("keep", winner, "neither", missed_constraint="greenish")

    head.verifier = verifier
    head.verify()
    assert "greenish" in head.plan.notes


# --------------------------------------------------------------------------- robustness
def test_cp4_verifier_exception_keeps_deterministic_winner():
    sc = scene(inst(1, "chair"), inst(2, "chair", centroid=(5, 0, 0)))
    head = _orhead(sc, object_plan("chair"))

    def boom(**kw):
        raise RuntimeError("dark checkpoint")

    head.verifier = boom
    head.verify()
    assert head.best_candidate.instance_id == 1


def test_cp4_verifier_receives_full_context():
    sc = scene(inst(1, "chair", centroid=(0, 0, 0)), inst(2, "chair", centroid=(5, 0, 0)))
    plan = object_plan("chair", clauses=[near_clause("chair")])
    plan.notes = "indefinite-article ambiguity"
    head = ObjectRefHead(plan=plan)
    head.advance(sc)
    captured = {}

    def verifier(*, question, winner, runner_up, winner_facts, notes, resolve_again):
        captured.update(
            question=question, notes=notes, has_runner=runner_up is not None,
            facts=winner_facts, hook=callable(resolve_again),
        )
        return _Outcome("keep", winner, "confirm")

    head.verifier = verifier
    head.verify()
    assert captured["notes"] == "indefinite-article ambiguity"
    assert captured["question"] == plan.question_raw
    assert captured["has_runner"] is True
    assert captured["hook"] is True


# --------------------------------------------------------------------------- backward compat
def test_legacy_bool_seam_injected_as_verifier_is_detected():
    sc = scene(
        inst(1, "chair", centroid=(0, 0, 0)),
        inst(2, "chair", centroid=(5, 0, 0)),
        inst(3, "table", centroid=(0.5, 0, 0)),
    )
    calls = {"n": 0}

    def legacy(plan, summary, matrix):  # old 3-arg bool seam
        calls["n"] += 1
        return calls["n"] > 1  # reject first, accept second

    # injected via the rich `verifier` kwarg — __post_init__ must reroute it to llm_verify.
    head = ObjectRefHead(plan=object_plan("chair", clauses=[closest_clause("table")]),
                         verifier=legacy)
    assert head.verifier is None
    assert head.llm_verify is legacy
    head.advance(sc)
    head.verify()
    assert calls["n"] == 2
    assert head.best_candidate.instance_id == 2  # legacy demotion walk still works


def test_legacy_llm_verify_still_works_alongside_none_verifier():
    sc = scene(inst(1, "chair"), inst(2, "chair", centroid=(5, 0, 0)))
    head = ObjectRefHead(plan=object_plan("chair"), llm_verify=lambda p, s, m: True)
    head.advance(sc)
    m = head.verify()
    assert m is not None
    assert head.best_candidate.instance_id == 1
