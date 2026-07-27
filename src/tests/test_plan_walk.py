"""Unit tests for the shared recursive plan-anchor walker (issue #95)."""
from __future__ import annotations

from core.plan_schema import Anchor, AvoidSpec, Clause, Pred, TargetSpec
from core.plan_walk import (
    MAX_DISAMBIGUATOR_DEPTH,
    iter_anchor_chain,
    iter_avoid_anchors,
    iter_clause_anchors,
    iter_route_anchors,
    iter_target_anchors,
)


def _nested_anchor(depth: int, leaf_noun: str = "leaf") -> Anchor:
    """Build an anchor chain ``depth`` disambiguators deep, innermost named ``leaf_noun``."""
    anchor = Anchor(noun=leaf_noun)
    for i in range(depth):
        anchor = Anchor(noun=f"n{depth - i}", disambiguator=Clause(pred=Pred.NEAR, anchors=[anchor]))
    return anchor


def test_iter_anchor_chain_none_yields_nothing():
    assert list(iter_anchor_chain(None)) == []


def test_iter_anchor_chain_single_anchor():
    a = Anchor(noun="table")
    assert [x.noun for x in iter_anchor_chain(a)] == ["table"]


def test_iter_anchor_chain_descends_disambiguator_in_order():
    """'the potted plant closest to the pyramid candle holder near the window' -- three
    anchors deep, outer to inner."""
    inner = Anchor(noun="window")
    mid = Anchor(noun="pyramid candle holder", disambiguator=Clause(pred=Pred.NEAR, anchors=[inner]))
    outer = Anchor(noun="potted plant", disambiguator=Clause(pred=Pred.CLOSEST_TO, anchors=[mid]))
    assert [x.noun for x in iter_anchor_chain(outer)] == [
        "potted plant",
        "pyramid candle holder",
        "window",
    ]


def test_iter_anchor_chain_between_disambiguator_yields_both_branches():
    """A disambiguator clause can itself be a BETWEEN with 2 anchors -- both must be
    reachable, depth-first."""
    a = Anchor(
        noun="chair",
        disambiguator=Clause(
            pred=Pred.BETWEEN, anchors=[Anchor(noun="lamp"), Anchor(noun="sofa")]
        ),
    )
    assert [x.noun for x in iter_anchor_chain(a)] == ["chair", "lamp", "sofa"]


def test_iter_anchor_chain_depth_cap_stops_pathological_nesting():
    """A disambiguator chain far deeper than any real question stops at the cap instead
    of hanging or blowing the stack."""
    deep = _nested_anchor(MAX_DISAMBIGUATOR_DEPTH + 50)
    out = list(iter_anchor_chain(deep))
    assert len(out) == MAX_DISAMBIGUATOR_DEPTH + 1  # depth 0..max_depth inclusive
    assert out[-1].noun != "leaf"  # capped before reaching the innermost anchor


def test_iter_anchor_chain_within_depth_cap_reaches_leaf():
    shallow = _nested_anchor(3)
    out = [x.noun for x in iter_anchor_chain(shallow)]
    assert out[-1] == "leaf"


def test_iter_anchor_chain_cycle_does_not_hang():
    """A malformed/adversarial plan could share Anchor objects across a cycle (e.g. two
    Anchor instances whose disambiguators point back at each other). The walk must
    terminate instead of looping forever."""
    a = Anchor(noun="a")
    b = Anchor(noun="b")
    a.disambiguator = Clause(pred=Pred.NEAR, anchors=[b])
    b.disambiguator = Clause(pred=Pred.NEAR, anchors=[a])  # cycle
    out = [x.noun for x in iter_anchor_chain(a)]
    assert out == ["a", "b"]  # visits each once, then stops


def test_iter_anchor_chain_self_cycle_does_not_hang():
    a = Anchor(noun="a")
    a.disambiguator = Clause(pred=Pred.NEAR, anchors=[a])  # points at itself
    assert [x.noun for x in iter_anchor_chain(a)] == ["a"]


def test_iter_target_anchors_excludes_target_noun_itself():
    target = TargetSpec(
        noun="teapot",
        clauses=[Clause(pred=Pred.ON, anchors=[Anchor(noun="table")])],
    )
    assert [a.noun for a in iter_target_anchors(target)] == ["table"]


def test_iter_target_anchors_none_target():
    assert list(iter_target_anchors(None)) == []


def test_iter_clause_anchors_order_is_clause_then_anchor():
    clauses = [
        Clause(pred=Pred.NEAR, anchors=[Anchor(noun="a")]),
        Clause(pred=Pred.BETWEEN, anchors=[Anchor(noun="b"), Anchor(noun="c")]),
    ]
    assert [a.noun for a in iter_clause_anchors(clauses)] == ["a", "b", "c"]


def test_iter_route_anchors_recurses():
    from core.plan_schema import LegKind, RouteLeg

    legs = [
        RouteLeg(
            kind=LegKind.GOTO,
            anchors=[Anchor(noun="bowl", disambiguator=Clause(pred=Pred.ON, anchors=[Anchor(noun="table")]))],
        )
    ]
    assert [a.noun for a in iter_route_anchors(legs)] == ["bowl", "table"]


def test_iter_avoid_anchors_between_and_near_recurse():
    avoid = [
        AvoidSpec(
            between=[
                Anchor(noun="col1", disambiguator=Clause(pred=Pred.NEAR, anchors=[Anchor(noun="rug")])),
                Anchor(noun="col2"),
            ]
        ),
        AvoidSpec(near=Anchor(noun="fireplace")),
    ]
    assert [a.noun for a in iter_avoid_anchors(avoid)] == ["col1", "rug", "col2", "fireplace"]
