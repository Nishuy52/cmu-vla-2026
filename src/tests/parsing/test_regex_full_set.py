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


# ----------------------------------------------------------------- issue #25 regression
#
# These first three cases exercise the trailing-superlative surfacing baseline (a bare
# "on the Y closest to Z" ranks the target, an explicit relative pronoun or a counting
# question keeps it nested under Y) that the guard extensions below are built on top of.


def test_trailing_superlative_surfaces_to_head_target():
    """'<target> on the Y closest to Z' parses to TWO top-level clauses on the target.

    The bare trailing superlative ranks the TARGET (the speakers), not the anchor Y —
    so it surfaces as a second top-level ``closest_to`` clause rather than nesting under
    the ``on`` anchor.
    """
    q = "Find the speaker on the TV cabinet closest to the potted plant on the TV cabinet."
    plan = parse_regex(q)
    assert plan.target is not None
    preds = [cl.pred.value for cl in plan.target.clauses]
    assert preds == ["on", "closest_to"], f"expected two top-level clauses, got {preds}"
    on_cl, sup_cl = plan.target.clauses
    # The 'on' anchor is a bare cabinet — the superlative no longer nests under it.
    assert on_cl.anchors[0].noun == "tv cabinet"
    assert on_cl.anchors[0].disambiguator is None
    # The surfaced superlative ranks the target by distance to the potted plant.
    assert sup_cl.anchors[0].noun == "potted plant"


def test_relative_clause_superlative_stays_nested():
    """An explicit relative pronoun binds the superlative to the anchor, not the target.

    "... on the sofa THAT IS closest to the lamp" means the sofa closest to the lamp —
    the superlative stays nested under the ``on`` anchor (one top-level clause).
    """
    plan = parse_regex("Find the pillow on the sofa that is closest to the lamp.")
    assert plan.target is not None
    assert [cl.pred.value for cl in plan.target.clauses] == ["on"]
    anchor = plan.target.clauses[0].anchors[0]
    assert anchor.noun == "sofa"
    assert anchor.disambiguator is not None
    assert anchor.disambiguator.pred.value == "closest_to"


def test_counting_superlative_does_not_surface():
    """'How many X on the Y closest to Z' keeps the superlative disambiguating Y.

    A superlative cannot rank a cardinality; for counting it selects the anchor Y (the
    table closest to Z), so it must stay nested — surfacing would drop the anchor filter
    and over-count.
    """
    plan = parse_regex(
        "How many computer monitors are on the table closest to the map wall decal?"
    )
    assert plan.target is not None
    assert [cl.pred.value for cl in plan.target.clauses] == ["on"]
    anchor = plan.target.clauses[0].anchors[0]
    assert anchor.disambiguator is not None
    assert anchor.disambiguator.pred.value == "closest_to"


def test_with_anchor_superlative_binds_to_anchor_not_target():
    """A bare WITH-anchor's trailing superlative binds to the anchor, not the target.

    "the pillow WITH a lamp closest to the window" means the lamp closest to the
    window — the superlative must nest under the 'with' anchor (one top-level clause),
    not surface to the pillow (issue #25).
    """
    plan = parse_regex("Find the pillow with a lamp closest to the window.")
    assert plan.target is not None
    assert plan.target.noun == "pillow"
    assert [cl.pred.value for cl in plan.target.clauses] == ["with"]
    anchor = plan.target.clauses[0].anchors[0]
    assert anchor.noun == "lamp"
    assert anchor.disambiguator is not None
    assert anchor.disambiguator.pred.value == "closest_to"
    assert anchor.disambiguator.anchors[0].noun == "window"


def test_relative_clause_superlative_with_trailing_comma_stays_nested():
    """A paused/comma'd relative-clause delivery ("that is, closest to ...") still guards.

    A trailing comma after the copula must not defeat the relative-clause guard —
    the superlative stays bound to the anchor (issue #25).
    """
    plan = parse_regex(
        "Find the tv cabinet on the shelf that is, closest to the window."
    )
    assert plan.target is not None
    assert [cl.pred.value for cl in plan.target.clauses] == ["on"]
    anchor = plan.target.clauses[0].anchors[0]
    assert anchor.noun == "shelf"
    assert anchor.disambiguator is not None
    assert anchor.disambiguator.pred.value == "closest_to"


def test_it_is_copula_superlative_stays_nested():
    """The bare 'it is' copula form guards the trailing superlative like 'that is' does.

    "... on the shelf IT IS closest to the window" binds to the shelf, not the head
    target (issue #25) — _CONNECTOR_RE already recognized 'it is', the relative-clause
    tail guard did not.
    """
    plan = parse_regex(
        "Find the tv cabinet on the shelf it is closest to the window."
    )
    assert plan.target is not None
    assert [cl.pred.value for cl in plan.target.clauses] == ["on"]
    anchor = plan.target.clauses[0].anchors[0]
    assert anchor.disambiguator is not None
    assert anchor.disambiguator.pred.value == "closest_to"
