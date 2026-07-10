"""Resolution engine: filtering, superlative ranking, fallback ladder, counting."""
from __future__ import annotations

from core.geometry import toolbox as T
from core.plan_schema import Anchor, Clause, Pred, TargetSpec
from tests.geometry._helpers import FakeIndex, rec


def _spec(noun, attributes=None, clauses=None):
    return TargetSpec(
        noun=noun,
        attributes=list(attributes or []),
        clauses=list(clauses or []),
    )


# --------------------------------------------------------------------------- basic filter


def test_resolve_noun_only():
    idx = FakeIndex([rec(1, "chair", (0, 0, 0)), rec(2, "table", (5, 0, 0))])
    res = T.resolve(_spec("chair"), idx)
    assert [c.instance_id for c in res.candidates_ranked] == [1]
    assert res.audit == []


def test_resolve_typo_tolerant_noun():
    idx = FakeIndex([rec(1, "refrigerator", (0, 0, 0))])
    res = T.resolve(_spec("refridgerator"), idx)  # misspelled
    assert [c.instance_id for c in res.candidates_ranked] == [1]


def test_resolve_attribute_filter():
    idx = FakeIndex(
        [
            rec(1, "pillow", (0, 0, 0), caption="a red pillow"),
            rec(2, "pillow", (1, 0, 0), caption="a blue pillow"),
        ]
    )
    res = T.resolve(_spec("pillow", attributes=["red"]), idx)
    assert [c.instance_id for c in res.candidates_ranked] == [1]
    assert res.audit == []


def test_resolve_relation_clause_and():
    # bowl ON table: only the bowl actually resting on the table survives.
    table = rec(1, "table", (0, 0, 0.5), (2.0, 2.0, 1.0))  # top z=1.0
    bowl_on = rec(2, "bowl", (0, 0, 1.05), (0.3, 0.3, 0.1))  # on the table
    bowl_off = rec(3, "bowl", (5, 5, 1.05), (0.3, 0.3, 0.1))  # elsewhere
    idx = FakeIndex([table, bowl_on, bowl_off])
    spec = _spec("bowl", clauses=[Clause(Pred.ON, [Anchor(noun="table")])])
    res = T.resolve(spec, idx)
    assert [c.instance_id for c in res.candidates_ranked] == [2]
    assert res.pass_matrix[2][0].passed


def test_resolve_superlative_ranks_not_filters():
    door = rec(1, "door", (0, 0, 0), (0.1, 2.0, 2.0))
    c_far = rec(2, "chair", (4, 0, 0))
    c_near = rec(3, "chair", (1, 0, 0))
    idx = FakeIndex([door, c_far, c_near])
    spec = _spec("chair", clauses=[Clause(Pred.CLOSEST_TO, [Anchor(noun="door")])])
    res = T.resolve(spec, idx)
    # both chairs survive (superlative ranks, does not filter); nearest first
    assert [c.instance_id for c in res.candidates_ranked] == [3, 2]
    assert res.margins[3] > 0


def test_resolve_superlative_with_filter():
    door = rec(1, "door", (0, 0, 0), (0.1, 2.0, 2.0))
    table = rec(4, "table", (2, 0, 0.5), (2.0, 2.0, 1.0))
    bowl_on = rec(2, "bowl", (2, 0, 1.05), (0.3, 0.3, 0.1))  # on table, dist ~2.06
    bowl_free = rec(3, "bowl", (1, 0, 0), (0.3, 0.3, 0.1))  # closer but not on table
    idx = FakeIndex([door, table, bowl_on, bowl_free])
    spec = _spec(
        "bowl",
        clauses=[
            Clause(Pred.ON, [Anchor(noun="table")]),
            Clause(Pred.CLOSEST_TO, [Anchor(noun="door")]),
        ],
    )
    res = T.resolve(spec, idx)
    # only bowl_on passes the ON filter; ranked winner
    assert [c.instance_id for c in res.candidates_ranked] == [2]


# --------------------------------------------------------------------------- fallback ladder


def test_fallback_relax_attributes_first():
    # attribute has no match -> ladder step 1 relaxes attributes and recovers.
    idx = FakeIndex([rec(1, "pillow", (0, 0, 0), caption="a blue pillow")])
    res = T.resolve(_spec("pillow", attributes=["red"]), idx)
    assert [c.instance_id for c in res.candidates_ranked] == [1]
    assert [a.step for a in res.audit] == ["relax_attributes"]


def test_fallback_drop_weakest_relation():
    # Two clauses: a permissive 'near' (both bowls pass) and a strict 'on' that
    # only bowl_on satisfies -- but no bowl satisfies BOTH because bowl_on sits
    # far from the shelf. Dropping the weakest (least selective = 'near') leaves
    # the strict 'on', which bowl_on passes: recovery happens AT drop_relation,
    # not by falling through to category-only.
    table = rec(1, "table", (10, 0, 0.5), (2.0, 2.0, 1.0))  # top z=1.0
    shelf = rec(2, "shelf", (0, 0, 0), (0.5, 0.5, 2.0))
    bowl_on = rec(3, "bowl", (10, 0, 1.05), (0.3, 0.3, 0.1))  # on table, far from shelf
    bowl_near = rec(4, "bowl", (0.7, 0, 0), (0.3, 0.3, 0.1))  # near shelf, not on table
    idx = FakeIndex([table, shelf, bowl_on, bowl_near])
    spec = _spec(
        "bowl",
        clauses=[
            Clause(Pred.NEAR, [Anchor(noun="shelf")]),  # weak: bowl_near passes
            Clause(Pred.ON, [Anchor(noun="table")]),  # strict: bowl_on passes
        ],
    )
    res = T.resolve(spec, idx)
    steps = [a.step for a in res.audit]
    assert steps == ["drop_relation"]  # recovered at the drop step, no category_only
    assert [c.instance_id for c in res.candidates_ranked] == [3]  # bowl_on survives 'on'
    dropped = [a.detail for a in res.audit if a.step == "drop_relation"]
    assert any("near" in d for d in dropped)  # least-selective clause dropped first


def test_fallback_category_only_last_resort():
    # a relation that nothing satisfies -> ladder exhausts to category-only.
    idx = FakeIndex([rec(1, "chair", (0, 0, 0))])  # no 'table' anchor exists at all
    spec = _spec("chair", clauses=[Clause(Pred.ON, [Anchor(noun="table")])])
    res = T.resolve(spec, idx)
    assert [c.instance_id for c in res.candidates_ranked] == [1]
    assert res.audit[-1].step == "category_only"


def test_fallback_ladder_order_attrs_then_relation():
    # both an unmatched attribute AND an unsatisfiable relation are present:
    # attributes must be relaxed before any relation is dropped.
    idx = FakeIndex([rec(1, "chair", (0, 0, 0), caption="wooden chair")])
    spec = _spec(
        "chair",
        attributes=["metal"],  # no match
        clauses=[Clause(Pred.ON, [Anchor(noun="table")])],  # no anchor
    )
    res = T.resolve(spec, idx)
    steps = [a.step for a in res.audit]
    assert steps[0] == "relax_attributes"
    assert "category_only" in steps
    assert steps.index("relax_attributes") < steps.index("category_only")


# --------------------------------------------------------------------------- counting


def test_counting_basic():
    idx = FakeIndex(
        [rec(1, "chair", (0, 0, 0)), rec(2, "chair", (2, 0, 0)), rec(3, "table", (5, 0, 0))]
    )
    n, ids = T.counting(_spec("chair"), idx)
    assert n == 2 and ids == {1, 2}


def test_counting_min_obs_filter():
    idx = FakeIndex(
        [rec(1, "chair", (0, 0, 0), n_obs=3), rec(2, "chair", (2, 0, 0), n_obs=0)]
    )
    n, ids = T.counting(_spec("chair"), idx, min_obs=1)
    assert n == 1 and ids == {1}


def test_counting_dedup_duplicate_id():
    # a survivor list that somehow repeats an instance_id must count once.
    dup = rec(1, "chair", (0, 0, 0))
    idx = FakeIndex([dup, dup, rec(2, "chair", (2, 0, 0))])
    n, ids = T.counting(_spec("chair"), idx)
    assert n == 2 and ids == {1, 2}


def test_counting_zero_allowed():
    idx = FakeIndex([rec(1, "table", (0, 0, 0))])
    n, ids = T.counting(_spec("chair"), idx)
    assert n == 0 and ids == set()
