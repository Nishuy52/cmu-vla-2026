"""Unit tests for core.parsing.normalize -- deterministic LLM-tier post-processing for
issues #47 (via_near/goto mislabeling), #48 (flattened stacked relative clauses), and
#49 (fabricated avoid corridors)."""
from __future__ import annotations

from core.interfaces import QType
from core.parsing.normalize import normalize_llm_plan
from core.plan_schema import Anchor, AvoidSpec, Clause, LegKind, Plan, Pred, RouteLeg, TargetSpec


def _leg(kind: LegKind, noun: str, raw: str | None = None, disambiguator: Clause | None = None) -> RouteLeg:
    return RouteLeg(kind=kind, anchors=[Anchor(noun=noun, raw=raw or noun, disambiguator=disambiguator)])


def _plan(route: list[RouteLeg] | None = None, target=None, avoid=None, qtype=QType.INSTRUCTION_FOLLOWING) -> Plan:
    return Plan(
        qtype=qtype,
        question_raw="",
        target=target,
        route=route or [],
        avoid=avoid or [],
    )


# --------------------------------------------------------------------------- #47


def test_non_terminal_go_near_downgraded_to_goto():
    question = "Go near the stool under the picture and stop at the small table farthest from the columns."
    plan = _plan(
        route=[
            _leg(LegKind.VIA_NEAR, "stool"),
            _leg(LegKind.GOTO, "table"),
        ]
    )
    normalize_llm_plan(plan, question)
    assert [leg.kind for leg in plan.route] == [LegKind.GOTO, LegKind.GOTO]


def test_genuine_pass_by_leg_kept_as_via_near():
    question = "Go near the fireplace, pass by the stairs, then stop at the sphere decoration on the cabinet."
    plan = _plan(
        route=[
            _leg(LegKind.VIA_NEAR, "fireplace"),
            _leg(LegKind.VIA_NEAR, "stairs"),
            _leg(LegKind.GOTO, "sphere decoration"),
        ]
    )
    normalize_llm_plan(plan, question)
    assert [leg.kind for leg in plan.route] == [LegKind.GOTO, LegKind.VIA_NEAR, LegKind.GOTO]


def test_take_the_path_near_kept_as_via_near():
    question = "First, go near the stool, then take the path near the cabinet, and stop at the bowl on the table."
    plan = _plan(
        route=[
            _leg(LegKind.VIA_NEAR, "stool"),
            _leg(LegKind.VIA_NEAR, "cabinet"),
            _leg(LegKind.GOTO, "bowl"),
        ]
    )
    normalize_llm_plan(plan, question)
    assert [leg.kind for leg in plan.route] == [LegKind.GOTO, LegKind.VIA_NEAR, LegKind.GOTO]


def test_terminal_via_near_left_alone():
    # A via_near leg is never downgraded when it's the LAST leg -- schema validation (route
    # must terminate GOTO) is a separate concern this normalizer doesn't need to fix.
    question = "Go near the window."
    plan = _plan(route=[_leg(LegKind.VIA_NEAR, "window")])
    normalize_llm_plan(plan, question)
    assert plan.route[0].kind is LegKind.VIA_NEAR


def test_split_terminal_stop_merged_into_one_goto_leg():
    """Regression per issue #47's hotel_room_2 example: the model splits 'stop by the
    curtain closest to the TV' into a via_near(curtain, closest_to TV) leg followed by a
    bare duplicate goto(curtain) leg. Downgrading the via_near to goto then collapses the
    two same-anchor consecutive goto legs into one, keeping the richer (disambiguated)
    anchor."""
    question = (
        "First, go to the picture closest to the door, then take the path between the "
        "TV cabinet and the bed, and stop by the curtain closest to the TV."
    )
    curtain_with_dis = Anchor(
        noun="curtain", raw="curtain",
        disambiguator=Clause(pred=Pred.CLOSEST_TO, anchors=[Anchor(noun="TV", raw="TV")]),
    )
    plan = _plan(
        route=[
            _leg(LegKind.GOTO, "picture", disambiguator=Clause(pred=Pred.CLOSEST_TO, anchors=[Anchor(noun="door")])),
            RouteLeg(kind=LegKind.CORRIDOR_BETWEEN, anchors=[Anchor(noun="tv cabinet", raw="TV cabinet"), Anchor(noun="bed", raw="bed")]),
            RouteLeg(kind=LegKind.VIA_NEAR, anchors=[curtain_with_dis]),
            _leg(LegKind.GOTO, "curtain"),
        ]
    )
    normalize_llm_plan(plan, question)
    assert [leg.kind for leg in plan.route] == [LegKind.GOTO, LegKind.CORRIDOR_BETWEEN, LegKind.GOTO]
    assert len(plan.route) == 3
    final = plan.route[-1]
    assert final.anchors[0].noun == "curtain"
    assert final.anchors[0].disambiguator is not None
    assert final.anchors[0].disambiguator.pred is Pred.CLOSEST_TO


def test_unmatched_anchor_text_defaults_to_goto():
    """An anchor whose noun/raw doesn't appear in the question text at all (paraphrase,
    truncation) can't be located -- the search window falls back to 'rest of the
    question', which (absent a pass-by/path marker) downgrades to goto, matching rule
    6's more common default rather than guessing via_near is intentional."""
    plan = _plan(route=[_leg(LegKind.VIA_NEAR, "gadget"), _leg(LegKind.GOTO, "thing")])
    normalize_llm_plan(plan, "Some question with none of those nouns in it.")
    assert plan.route[0].kind is LegKind.GOTO


# --------------------------------------------------------------------------- #48


def _flat_target(pred0: Pred, noun0: str, pred1: Pred, noun1: str, qtype=QType.OBJECT_REFERENCE) -> tuple[Plan, str]:
    target = TargetSpec(
        noun="paper cup",
        clauses=[
            Clause(pred=pred0, anchors=[Anchor(noun=noun0, raw=noun0)]),
            Clause(pred=pred1, anchors=[Anchor(noun=noun1, raw=noun1)]),
        ],
    )
    return _plan(target=target, qtype=qtype), ""


def test_relative_pronoun_bound_superlative_gets_nested():
    plan, _ = _flat_target(Pred.ON, "chair", Pred.CLOSEST_TO, "TV")
    question = "Find the pillow on the chair that is closest to the TV."
    normalize_llm_plan(plan, question)
    assert len(plan.target.clauses) == 1
    c0 = plan.target.clauses[0]
    assert c0.pred is Pred.ON
    assert c0.anchors[0].noun == "chair"
    assert c0.anchors[0].disambiguator is not None
    assert c0.anchors[0].disambiguator.pred is Pred.CLOSEST_TO
    assert c0.anchors[0].disambiguator.anchors[0].noun == "TV"


def test_with_clause_always_nested_no_pronoun_needed():
    plan, _ = _flat_target(Pred.ON, "nightstand", Pred.WITH, "photo")
    question = "Find the lamp on the nightstand that has the photo on it."
    normalize_llm_plan(plan, question)
    assert len(plan.target.clauses) == 1
    assert plan.target.clauses[0].anchors[0].disambiguator.pred is Pred.WITH


def test_numerical_always_nests_even_without_relative_pronoun():
    plan, _ = _flat_target(Pred.ON, "sofa", Pred.UNDER, "picture", qtype=QType.NUMERICAL)
    question = "How many pillows are on the sofa under the pictures?"
    normalize_llm_plan(plan, question)
    assert len(plan.target.clauses) == 1
    assert plan.target.clauses[0].anchors[0].disambiguator.pred is Pred.UNDER


def test_bare_trailing_superlative_on_object_reference_left_flat():
    """Matches the regex floor's own current behaviour (issue #25): a BARE (no relative
    pronoun) trailing superlative on an object_reference question ranks the target, not
    the anchor -- normalize must NOT nest this one, or it would disagree with the floor."""
    plan, _ = _flat_target(Pred.ON, "table", Pred.CLOSEST_TO, "projector screen")
    question = "Find the paper cup on the table closest to the projector screen."
    normalize_llm_plan(plan, question)
    assert len(plan.target.clauses) == 2


def test_already_disambiguated_first_anchor_left_alone():
    inner = Clause(pred=Pred.NEAR, anchors=[Anchor(noun="window")])
    target = TargetSpec(
        noun="cup",
        clauses=[
            Clause(pred=Pred.ON, anchors=[Anchor(noun="table", disambiguator=inner)]),
            Clause(pred=Pred.CLOSEST_TO, anchors=[Anchor(noun="screen")]),
        ],
    )
    plan = _plan(target=target, qtype=QType.OBJECT_REFERENCE)
    normalize_llm_plan(plan, "Find the cup on the table near the window closest to the screen.")
    # anchor already has a disambiguator -- normalizer must not clobber it
    assert len(plan.target.clauses) == 2


# --------------------------------------------------------------------------- #49


def test_avoid_dropped_when_no_avoid_language():
    avoid = [AvoidSpec(between=[Anchor(noun="projector screen"), Anchor(noun="window")])]
    plan = _plan(
        route=[_leg(LegKind.GOTO, "potted plant"), _leg(LegKind.GOTO, "water cooler")],
        avoid=avoid,
    )
    question = (
        "Go to the potted plant furthest from the projector screen then stop at the "
        "water cooler near the window."
    )
    normalize_llm_plan(plan, question)
    assert plan.avoid == []
    assert "dropped 1 fabricated avoid" in plan.notes


def test_avoid_kept_when_avoid_language_present():
    avoid = [AvoidSpec(between=[Anchor(noun="tv"), Anchor(noun="tea table")])]
    plan = _plan(
        route=[_leg(LegKind.GOTO, "chair"), _leg(LegKind.GOTO, "soccer ball")],
        avoid=avoid,
    )
    question = (
        "First, go to the chair near the window, then stop at the soccer ball near the "
        "couch, avoiding the path between the TV and the tea table."
    )
    normalize_llm_plan(plan, question)
    assert plan.avoid == avoid


def test_avoid_kept_when_without_language_present():
    avoid = [AvoidSpec(near=Anchor(noun="fireplace"))]
    plan = _plan(route=[_leg(LegKind.GOTO, "chair")], avoid=avoid)
    question = "Go to the chair without passing near the fireplace."
    normalize_llm_plan(plan, question)
    assert plan.avoid == avoid


def test_no_avoid_entries_is_a_noop():
    plan = _plan(route=[_leg(LegKind.GOTO, "chair")])
    normalize_llm_plan(plan, "Go to the chair.")
    assert plan.avoid == []
    assert plan.notes == ""
