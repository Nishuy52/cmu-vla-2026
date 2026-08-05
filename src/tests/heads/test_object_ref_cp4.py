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


# ------------------------------------------------------------------ OR-F4: keep on neither
def test_cp4_hallucinated_neither_keeps_winner_or_f4():
    """OR-F4 attack: the model returns a schema-valid `neither` with a missed constraint it
    cannot actually verify. The head must KEEP the deterministic winner — never silently
    swap to an unverified runner-up (the deployed-seam inversion this test guards against).
    """
    sc = scene(
        inst(1, "bowl", centroid=(0, 0, 0)),
        inst(2, "bowl", centroid=(5, 0, 0)),
    )
    head = _orhead(sc, object_plan("bowl"))
    winner_id = head.best_candidate.instance_id

    def verifier(*, question, winner, runner_up, winner_facts, notes, resolve_again):
        # keep-on-neither: a non-actionable verdict returns the winner unchanged.
        return _Outcome("keep", winner, "neither",
                        missed_constraint="closest_to folding screen")

    head.verifier = verifier
    head.verify()
    assert head.best_candidate.instance_id == winner_id


# ---------------------------------------------------- OR-F5: re-resolve flips the instance
def test_cp4_reresolve_flips_to_correct_instance_two_tables_or_f5():
    """OR-F5 attack scene: two bowls (one per table); the deterministic rank picks bowl 1
    (lowest id). CP4 diagnoses the missed `closest_to folding screen` constraint; the
    re-resolve hook synthesizes a REAL superlative clause and the winner flips to bowl 2 —
    the bowl on the table nearest the folding screen. Proves the re-resolve is no longer a
    no-op.
    """
    sc = scene(
        inst(1, "bowl", centroid=(0, 0, 0)),           # far table
        inst(2, "bowl", centroid=(5, 0, 0)),           # near table
        inst(3, "folding screen", centroid=(5.5, 0, 0)),  # beside bowl 2's table
    )
    head = _orhead(sc, object_plan("bowl"))
    assert head.best_candidate.instance_id == 1  # deterministic rank: lowest id

    def verifier(*, question, winner, runner_up, winner_facts, notes, resolve_again):
        new = resolve_again("closest_to folding screen")
        return _Outcome("re_resolve", new, "neither",
                        missed_constraint="closest_to folding screen")

    head.verifier = verifier
    head.verify()
    assert head.best_candidate.instance_id == 2  # flipped to the correct instance


def test_cp4_reresolve_unparseable_constraint_keeps_winner_or_f5():
    """OR-F5: a missed constraint that names no known relation does not parse into a clause;
    the re-resolve keeps the deterministic winner rather than crashing."""
    sc = scene(inst(1, "bowl"), inst(2, "bowl", centroid=(5, 0, 0)))
    head = _orhead(sc, object_plan("bowl"))

    def verifier(*, question, winner, runner_up, winner_facts, notes, resolve_again):
        new = resolve_again("gibberish phrase here")
        return _Outcome("re_resolve", new, "neither", missed_constraint="gibberish phrase here")

    head.verifier = verifier
    head.verify()
    assert head.best_candidate.instance_id == 1


# ---------------------------------------------------- OR-F8: provisional never published
def test_cp4_provisional_only_winner_not_published_or_f8():
    """OR-F8: a provisional-only winner (n_obs=1, a CP2 hallucination-recovery instance) is
    never committed as the published marker.

    (#184) The established-instance gate (``ObjectRefHead._resolve``,
    ``core.heads.scene_established.ESTABLISH_N_OBS``) now already ranks the
    established (n_obs>=3) candidate ahead of the n_obs=1 ghost at advance() time, so
    the winner is real from the start here -- OR-F8's own verify()-time
    eligibility fallthrough is not even the deciding mechanism for THIS n_obs=1 case
    any more (it stays load-bearing for a low-SCORE winner instead -- see
    ``test_cp4_low_score_winner_not_published_falls_through`` below, unaffected since
    the established gate only looks at n_obs). The guarantee this test protects
    (a provisional-only instance is never the published marker) still holds, just
    enforced earlier."""
    sc = scene(
        inst(1, "bowl", n_obs=1, centroid=(0, 0, 0)),   # provisional (CP2 hit)
        inst(2, "bowl", n_obs=3, centroid=(5, 0, 0)),   # real, re-observed
    )
    head = _orhead(sc, object_plan("bowl"))
    assert head.best_candidate.instance_id == 2  # (#184) established gate already prefers it
    m = head.verify()
    assert m is not None
    assert head.best_candidate.instance_id == 2  # provisional never published


def test_cp4_all_provisional_falls_back_to_relaxed_target_185():
    """(#185, supersedes the pre-#185 OR-F8 "publishes nothing" verdict) When every
    ranked candidate is provisional-only, verify() no longer refuses to publish:
    returning None here would force the FSM floor to fall through to an ANCHOR-class
    instance instead of the TARGET class (``core.fsm.floors.FloorAnswers.
    _object_reference`` rung 4 -- issue #185's 'trash can for a bowl question'
    defect). Both candidates are equally provisional (n_obs=1, below both the
    established floor and the answer-eligibility floor), so the deterministic
    top-ranked one is published -- provisional, but still the TARGET class, which is
    strictly better than a wrong-class floor answer."""
    sc = scene(inst(1, "bowl", n_obs=1), inst(2, "bowl", n_obs=1, centroid=(5, 0, 0)))
    head = _orhead(sc, object_plan("bowl"))
    m = head.verify()
    assert m is not None
    assert head.best_candidate.instance_id == 1  # deterministic top rank, never silence


# ------------------------------------------------------- issue #43a answer eligibility
def test_cp4_low_score_winner_not_published_falls_through():
    """A winner with n_obs >= 2 but peak score under the 0.30 answer-eligibility floor
    (a weak question-pass recall hit, not a confident detection) is ineligible just like
    a provisional-only (n_obs=1) winner -- verify() falls through to the next eligible
    candidate."""
    sc = scene(
        inst(1, "bowl", n_obs=3, score=0.20, centroid=(0, 0, 0)),   # under the score floor
        inst(2, "bowl", n_obs=3, score=0.90, centroid=(5, 0, 0)),   # eligible
    )
    head = _orhead(sc, object_plan("bowl"))
    assert head.best_candidate.instance_id == 1  # ranks first by id
    m = head.verify()
    assert m is not None
    assert head.best_candidate.instance_id == 2  # low-score winner refused, fell through


def test_cp4_all_low_score_falls_back_to_relaxed_target_185():
    """(#185) Every ranked candidate under the score floor -> verify() no longer
    refuses to publish (see ``test_cp4_all_provisional_falls_back_to_relaxed_target_185``
    for the full rationale): both candidates are established (n_obs=3), so #184's gate
    is a no-op here -- this isolates #185's OR-F8 relaxation on its own."""
    sc = scene(
        inst(1, "bowl", n_obs=3, score=0.10),
        inst(2, "bowl", n_obs=3, score=0.29, centroid=(5, 0, 0)),
    )
    head = _orhead(sc, object_plan("bowl"))
    m = head.verify()
    assert m is not None
    assert head.best_candidate.instance_id == 1  # deterministic top rank, never silence


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
