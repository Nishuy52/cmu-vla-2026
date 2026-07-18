"""Regex tier over the full 75-question training set: total, valid, correctly typed."""
from __future__ import annotations

from core.parsing.regex_tier import classify_qtype, parse_regex


def test_every_training_question_yields_valid_plan(all_questions):
    """parse_regex returns a schema-valid Plan for all 75 questions, zero exceptions."""
    for scene, qtype_label, q in all_questions:
        plan = parse_regex(q)  # must not raise
        errs = plan.validate()
        assert errs == [], f"{scene}/{qtype_label}: {q!r} -> {errs}"
        assert plan.parse_tier == "regex"
        assert plan.question_raw == q


def test_qtype_classification_matches_labels(all_questions):
    """Text-only qtype classification agrees with the JSON section labels on all 75."""
    wrong = [
        (scene, q, classify_qtype(q).value)
        for scene, qtype_label, q in all_questions
        if classify_qtype(q).value != qtype_label
    ]
    assert not wrong, f"misclassified: {wrong}"


def test_instruction_routes_end_in_goto(all_questions):
    """Every instruction-following parse ends with a terminal goto leg."""
    for scene, qtype_label, q in all_questions:
        if qtype_label != "instruction_following":
            continue
        plan = parse_regex(q)
        assert plan.route, f"{scene}: empty route for {q!r}"
        assert plan.route[-1].kind.value == "goto", f"{scene}: {q!r}"


# ----------------------------------------------------------------- issue #23 regression


def test_leading_superlative_resolves_target_noun():
    """'Find the closest X to Y' parses to target noun X, not Y (issue #23).

    The superlative-first NP shape ("the CLOSEST speaker TO the plant") must resolve
    the head noun to 'speaker' with a top-level closest_to clause anchored on 'plant' —
    the pre-fix parser mis-identified the head noun as 'plant' with empty clauses.
    """
    plan = parse_regex("Find the closest speaker to the plant.")
    assert plan.target is not None
    assert plan.target.noun == "speaker"
    assert [cl.pred.value for cl in plan.target.clauses] == ["closest_to"]
    assert plan.target.clauses[0].anchors[0].noun == "plant"


def test_leading_superlative_farthest_from():
    """'Find the farthest X from Y' resolves the same way for the farthest_from family."""
    plan = parse_regex("Find the farthest suitcase from the door.")
    assert plan.target is not None
    assert plan.target.noun == "suitcase"
    assert [cl.pred.value for cl in plan.target.clauses] == ["farthest_from"]
    assert plan.target.clauses[0].anchors[0].noun == "door"


def test_leading_superlative_with_modifier():
    """A modifier between the superlative adjective and the head noun still resolves."""
    plan = parse_regex("Find the nearest small lamp to the sofa.")
    assert plan.target is not None
    assert plan.target.noun == "lamp"
    assert plan.target.attributes == ["small"]
    assert [cl.pred.value for cl in plan.target.clauses] == ["closest_to"]
