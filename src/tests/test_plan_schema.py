"""Plan DSL round-trip + validation tests.

Builds a Plan for one real training question of each type (verbatim from
docs/question_analysis.md) and checks to_json -> from_json -> validate() == [],
plus a battery of validation-failure cases exercising every branch in
Plan.validate().
"""
from __future__ import annotations

import json

import pytest

from core.interfaces import QType
from core.plan_schema import (
    Anchor,
    AvoidSpec,
    Clause,
    LegKind,
    Plan,
    PlanSchemaError,
    Pred,
    RouteLeg,
    TargetSpec,
)


def _roundtrip(plan: Plan) -> Plan:
    return Plan.from_json(plan.to_json())


# --------------------------------------------------------------- real questions


def numerical_plan() -> Plan:
    """arabic_room numerical: 'How many sofas are below a window?'"""
    return Plan(
        qtype=QType.NUMERICAL,
        question_raw="How many sofas are below a window?",
        target=TargetSpec(
            noun="sofa",
            raw="sofas",
            clauses=[Clause(pred=Pred.UNDER, anchors=[Anchor(noun="window", raw="window")])],
        ),
    )


def object_reference_plan() -> Plan:
    """arabic_room object_reference:
    'Find the pillow closest to the book on the stool.'"""
    return Plan(
        qtype=QType.OBJECT_REFERENCE,
        question_raw="Find the pillow closest to the book on the stool.",
        target=TargetSpec(
            noun="pillow",
            raw="pillow",
            clauses=[
                Clause(
                    pred=Pred.CLOSEST_TO,
                    anchors=[
                        Anchor(
                            noun="book",
                            raw="book",
                            disambiguator=Clause(
                                pred=Pred.ON, anchors=[Anchor(noun="stool", raw="stool")]
                            ),
                        )
                    ],
                )
            ],
        ),
    )


def instruction_following_plan() -> Plan:
    """arabic_room q2:
    'First, go to the potted plant furthest from the hookah, then take the path
    between the two columns, and stop at the tray on the table.'"""
    return Plan(
        qtype=QType.INSTRUCTION_FOLLOWING,
        question_raw=(
            "First, go to the potted plant furthest from the hookah, then take the "
            "path between the two columns, and stop at the tray on the table."
        ),
        route=[
            RouteLeg(
                kind=LegKind.GOTO,
                anchors=[
                    Anchor(
                        noun="potted plant",
                        raw="potted plant",
                        disambiguator=Clause(
                            pred=Pred.FARTHEST_FROM,
                            anchors=[Anchor(noun="hookah", raw="hookah")],
                        ),
                    )
                ],
            ),
            RouteLeg(
                kind=LegKind.CORRIDOR_BETWEEN,
                anchors=[Anchor(noun="column", raw="columns"), Anchor(noun="column", raw="columns")],
            ),
            RouteLeg(
                kind=LegKind.GOTO,
                anchors=[
                    Anchor(
                        noun="tray",
                        raw="tray",
                        disambiguator=Clause(
                            pred=Pred.ON, anchors=[Anchor(noun="table", raw="table")]
                        ),
                    )
                ],
            ),
        ],
    )


def instruction_following_avoid_plan() -> Plan:
    """livingroom_2 q2: two proximity goals + an avoid-corridor over the traversal."""
    return Plan(
        qtype=QType.INSTRUCTION_FOLLOWING,
        question_raw=(
            "First, go to the chair near the window, then stop at the soccer ball "
            "near the couch, avoiding the path between the TV and the tea table."
        ),
        route=[
            RouteLeg(
                kind=LegKind.GOTO,
                anchors=[
                    Anchor(
                        noun="chair",
                        disambiguator=Clause(pred=Pred.NEAR, anchors=[Anchor(noun="window")]),
                    )
                ],
            ),
            RouteLeg(
                kind=LegKind.GOTO,
                anchors=[
                    Anchor(
                        noun="soccer ball",
                        disambiguator=Clause(pred=Pred.NEAR, anchors=[Anchor(noun="couch")]),
                    )
                ],
            ),
        ],
        avoid=[AvoidSpec(between=[Anchor(noun="tv"), Anchor(noun="tea table")])],
    )


# ------------------------------------------------------------------ round-trips


def test_numerical_roundtrip_valid():
    plan = numerical_plan()
    assert plan.validate() == []
    rt = _roundtrip(plan)
    assert rt.validate() == []
    assert rt.qtype is QType.NUMERICAL
    assert rt.target is not None
    assert rt.target.noun == "sofa"
    assert rt.target.clauses[0].pred is Pred.UNDER
    assert rt.target.clauses[0].anchors[0].noun == "window"


def test_object_reference_roundtrip_valid():
    plan = object_reference_plan()
    assert plan.validate() == []
    rt = _roundtrip(plan)
    assert rt.validate() == []
    # nested disambiguator survives the round-trip
    anchor = rt.target.clauses[0].anchors[0]
    assert anchor.noun == "book"
    assert anchor.disambiguator is not None
    assert anchor.disambiguator.pred is Pred.ON
    assert anchor.disambiguator.anchors[0].noun == "stool"


def test_instruction_following_roundtrip_valid():
    plan = instruction_following_plan()
    assert plan.validate() == []
    rt = _roundtrip(plan)
    assert rt.validate() == []
    assert [leg.kind for leg in rt.route] == [
        LegKind.GOTO,
        LegKind.CORRIDOR_BETWEEN,
        LegKind.GOTO,
    ]
    assert rt.route[-1].kind is LegKind.GOTO  # terminates in GOTO
    corridor = rt.route[1]
    assert len(corridor.anchors) == 2


def test_instruction_following_avoid_roundtrip_valid():
    plan = instruction_following_avoid_plan()
    assert plan.validate() == []
    rt = _roundtrip(plan)
    assert rt.validate() == []
    assert len(rt.avoid) == 1
    assert rt.avoid[0].between is not None
    assert len(rt.avoid[0].between) == 2
    assert rt.avoid[0].near is None


def test_parse_tier_and_notes_roundtrip():
    plan = numerical_plan()
    plan.parse_tier = "regex"
    plan.notes = "degraded: noun 'sofa' low-confidence"
    rt = _roundtrip(plan)
    assert rt.parse_tier == "regex"
    assert rt.notes == "degraded: noun 'sofa' low-confidence"


# --------------------------------------------------------------- failure cases


def test_numerical_missing_target_invalid():
    plan = Plan(qtype=QType.NUMERICAL, question_raw="x")
    errs = plan.validate()
    assert any("requires target" in e for e in errs)


def test_numerical_with_route_invalid():
    plan = numerical_plan()
    plan.route = [RouteLeg(kind=LegKind.GOTO, anchors=[Anchor(noun="door")])]
    errs = plan.validate()
    assert any("must not carry a route" in e for e in errs)


def test_instruction_following_empty_route_invalid():
    plan = Plan(qtype=QType.INSTRUCTION_FOLLOWING, question_raw="x")
    errs = plan.validate()
    assert any("requires >=1 route leg" in e for e in errs)


def test_instruction_following_with_target_invalid():
    plan = instruction_following_plan()
    plan.target = TargetSpec(noun="door")
    errs = plan.validate()
    assert any("must not carry a target" in e for e in errs)


def test_route_must_end_in_goto():
    plan = Plan(
        qtype=QType.INSTRUCTION_FOLLOWING,
        question_raw="x",
        route=[RouteLeg(kind=LegKind.VIA_NEAR, anchors=[Anchor(noun="window")])],
    )
    errs = plan.validate()
    assert any("must terminate in a GOTO leg" in e for e in errs)


def test_corridor_leg_needs_two_anchors():
    plan = Plan(
        qtype=QType.INSTRUCTION_FOLLOWING,
        question_raw="x",
        route=[
            RouteLeg(kind=LegKind.CORRIDOR_BETWEEN, anchors=[Anchor(noun="column")]),
            RouteLeg(kind=LegKind.GOTO, anchors=[Anchor(noun="tray")]),
        ],
    )
    errs = plan.validate()
    assert any("corridor_between needs 2 anchor(s)" in e for e in errs)


def test_between_clause_needs_two_anchors():
    plan = Plan(
        qtype=QType.OBJECT_REFERENCE,
        question_raw="x",
        target=TargetSpec(
            noun="lantern",
            clauses=[Clause(pred=Pred.BETWEEN, anchors=[Anchor(noun="vase")])],
        ),
    )
    errs = plan.validate()
    assert any("between needs 2 anchor(s)" in e for e in errs)


def test_avoid_requires_exactly_one_of_between_near():
    both = Plan(
        qtype=QType.INSTRUCTION_FOLLOWING,
        question_raw="x",
        route=[RouteLeg(kind=LegKind.GOTO, anchors=[Anchor(noun="tray")])],
        avoid=[AvoidSpec(between=[Anchor(noun="tv"), Anchor(noun="table")], near=Anchor(noun="door"))],
    )
    assert any("exactly one of between/near" in e for e in both.validate())

    neither = Plan(
        qtype=QType.INSTRUCTION_FOLLOWING,
        question_raw="x",
        route=[RouteLeg(kind=LegKind.GOTO, anchors=[Anchor(noun="tray")])],
        avoid=[AvoidSpec()],
    )
    assert any("exactly one of between/near" in e for e in neither.validate())


def test_avoid_between_needs_two_anchors():
    plan = Plan(
        qtype=QType.INSTRUCTION_FOLLOWING,
        question_raw="x",
        route=[RouteLeg(kind=LegKind.GOTO, anchors=[Anchor(noun="tray")])],
        avoid=[AvoidSpec(between=[Anchor(noun="tv")])],
    )
    errs = plan.validate()
    assert any("between needs exactly 2 anchors" in e for e in errs)


# ------------------------------------------------------------- enum error messages (#45)


def _plan_json(pred: str) -> str:
    return json.dumps(
        {
            "qtype": "object_reference",
            "question_raw": "x",
            "target": {
                "noun": "lantern",
                "clauses": [{"pred": pred, "anchors": [{"noun": "vase"}]}],
            },
        }
    )


def test_invalid_pred_raises_schema_error_naming_field_value_and_valid_set():
    with pytest.raises(PlanSchemaError) as exc_info:
        Plan.from_json(_plan_json("closer_to"))
    msg = str(exc_info.value)
    assert "'pred'" in msg
    assert "'closer_to'" in msg
    # valid set is derived from the Pred enum itself, not a hand-maintained list
    for member in Pred:
        assert member.value in msg


def test_invalid_qtype_raises_schema_error():
    bad = json.dumps({"qtype": "not_a_real_qtype", "question_raw": "x"})
    with pytest.raises(PlanSchemaError) as exc_info:
        Plan.from_json(bad)
    assert "'qtype'" in str(exc_info.value)
    assert "'not_a_real_qtype'" in str(exc_info.value)


def test_invalid_route_kind_raises_schema_error():
    bad = json.dumps(
        {
            "qtype": "instruction_following",
            "question_raw": "x",
            "route": [{"kind": "walk_to", "anchors": [{"noun": "door"}]}],
        }
    )
    with pytest.raises(PlanSchemaError) as exc_info:
        Plan.from_json(bad)
    assert "'walk_to'" in str(exc_info.value)


# ------------------------------------------------------- pred synonym aliases (parse-robustness)


def test_furthest_from_alias_parses_as_farthest_from():
    plan = Plan.from_json(_plan_json("furthest_from"))
    assert plan.validate() == []
    assert plan.target.clauses[0].pred is Pred.FARTHEST_FROM


def test_nearest_to_alias_parses_as_closest_to():
    plan = Plan.from_json(_plan_json("nearest_to"))
    assert plan.validate() == []
    assert plan.target.clauses[0].pred is Pred.CLOSEST_TO


def test_unknown_pred_value_still_fails():
    with pytest.raises(PlanSchemaError):
        Plan.from_json(_plan_json("closer_to"))
