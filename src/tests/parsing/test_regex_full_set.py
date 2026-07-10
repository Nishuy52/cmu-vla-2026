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
