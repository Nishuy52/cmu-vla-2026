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


# ---------------------------------------------------- H4a: counting must not relax


def test_counting_unsatisfied_relation_returns_zero_not_category_total():
    # NUM-F1 repro: pillows exist but none rest on the sofa. resolve() would fall
    # back to category_only and report the whole-category count; counting() must
    # report 0 (a legal answer) and NAME the failing clause, never relax.
    sofa = rec(1, "sofa", (0, 0, 0.4), (2.0, 1.0, 0.8))  # top z=0.8
    p1 = rec(2, "pillow", (5, 5, 0.2), (0.3, 0.3, 0.2))  # nowhere near the sofa
    p2 = rec(3, "pillow", (7, 7, 0.2), (0.3, 0.3, 0.2))
    idx = FakeIndex([sofa, p1, p2])
    spec = _spec("pillow", clauses=[Clause(Pred.ON, [Anchor(noun="sofa")])])
    # resolve() over-counts via the ladder (category_only -> both pillows)
    assert len(T.resolve(spec, idx).candidates_ranked) == 2
    # counting() does NOT relax: strict filter empties -> 0
    res = T.counting(spec, idx)
    assert res.count == 0 and res.ids == set()
    assert any("on" in e for e in res.explanations)
    # and the ladder relaxations never appear on the counting result
    assert res.audit == []


def test_counting_relaxed_attribute_returns_zero():
    # NUM-F1/F4 repro: a colour-qualified count whose attribute matches nothing.
    # resolve() relaxes the attribute and counts all pillows; counting() returns 0
    # and names the attribute failure.
    idx = FakeIndex(
        [
            rec(1, "pillow", (0, 0, 0), caption="a gray pillow"),
            rec(2, "pillow", (1, 0, 0), caption="an olive pillow"),
        ]
    )
    spec = _spec("pillow", attributes=["black"])
    assert len(T.resolve(spec, idx).candidates_ranked) == 2  # ladder relaxes attrs
    res = T.counting(spec, idx)
    assert res.count == 0 and res.ids == set()
    assert any("black" in e for e in res.explanations)


def test_counting_absent_noun_explains():
    idx = FakeIndex([rec(1, "table", (0, 0, 0))])
    res = T.counting(_spec("chair"), idx)
    assert res.count == 0
    assert any("chair" in e for e in res.explanations)


def test_counting_satisfied_relation_still_counts():
    # regression: a genuinely-satisfied relation counts normally (no over-zealous 0).
    table = rec(1, "table", (0, 0, 0.5), (2.0, 2.0, 1.0))  # top z=1.0
    bowl_on = rec(2, "bowl", (0, 0, 1.05), (0.3, 0.3, 0.1))  # on the table
    bowl_off = rec(3, "bowl", (5, 5, 1.05), (0.3, 0.3, 0.1))
    idx = FakeIndex([table, bowl_on, bowl_off])
    spec = _spec("bowl", clauses=[Clause(Pred.ON, [Anchor(noun="table")])])
    res = T.counting(spec, idx)
    assert res.count == 1 and res.ids == {2}
    assert res.explanations == []


def test_counting_result_unpacks_as_pair():
    # backward-compat: existing callers do `n, ids = counting(...)`.
    idx = FakeIndex([rec(1, "chair", (0, 0, 0)), rec(2, "chair", (2, 0, 0))])
    n, ids = T.counting(_spec("chair"), idx)
    assert n == 2 and ids == {1, 2}


# ---------------------------------------------------- H4c: scene-scope clauses don't filter


def test_counting_in_room_clause_is_vacuous_scope():
    # "how many stools are in the room?" -> in(room), but no instance is
    # labeled "room". The strict AND-filter must not zero this out: the room
    # is the universe of discourse, so the clause is skipped (and audited),
    # not treated as an unsatisfiable relation.
    idx = FakeIndex([rec(1, "stool", (0, 0, 0)), rec(2, "stool", (2, 0, 0))])
    spec = _spec("stool", clauses=[Clause(Pred.IN, [Anchor(noun="room")])])
    res = T.counting(spec, idx)
    assert res.count == 2 and res.ids == {1, 2}
    assert res.explanations == []
    assert any(a.step == "scope_clause" for a in res.audit)


def test_counting_in_kitchen_clause_stays_strict():
    # a NAMED room type is not a scope noun: no "kitchen" instance -> strict 0.
    idx = FakeIndex([rec(1, "stool", (0, 0, 0))])
    spec = _spec("stool", clauses=[Clause(Pred.IN, [Anchor(noun="kitchen")])])
    res = T.counting(spec, idx)
    assert res.count == 0 and res.ids == set()
    assert not any(a.step == "scope_clause" for a in res.audit)


# -------------------------------------------- H4b: nested disambiguators must bind


def _table_with_bowls_scene():
    """Two tables, one bowl on each; a screen near the far table. The 'near' table
    is the one closest to the screen -> its bowl must win."""
    screen = rec(10, "screen", (0, 0, 0.5), (0.1, 1.0, 1.0))
    table_near = rec(1, "table", (2, 0, 0.5), (1.0, 1.0, 1.0))  # top z=1.0, ~2m from screen
    table_far = rec(2, "table", (8, 0, 0.5), (1.0, 1.0, 1.0))  # ~8m from screen
    bowl_near = rec(4, "bowl", (2, 0, 1.05), (0.3, 0.3, 0.1))  # on the near table
    bowl_far = rec(3, "bowl", (8, 0, 1.05), (0.3, 0.3, 0.1))  # on the far table (lower id)
    return FakeIndex([screen, table_near, table_far, bowl_near, bowl_far])


def _bowl_on_closest_table_spec():
    disamb = Clause(Pred.CLOSEST_TO, [Anchor(noun="screen")])
    on_table = Clause(Pred.ON, [Anchor(noun="table", disambiguator=disamb)])
    return _spec("bowl", clauses=[on_table])


def test_nested_superlative_near_table_bowl_wins():
    # OR-F1 repro: "the bowl on the table closest to the screen". The near-table
    # bowl (id 4, HIGHER instance id) must win over the far-table bowl (id 3);
    # without disambiguator binding, existential on(table) keeps both and id order
    # would pick id 3 (the wrong, far bowl).
    idx = _table_with_bowls_scene()
    res = T.resolve(_bowl_on_closest_table_spec(), idx)
    assert res.candidates_ranked[0].instance_id == 4  # near-table bowl


def test_nested_superlative_only_narrowed_bowl_survives():
    # The disambiguator narrows the 'table' anchor to the closest one, so on(table)
    # matches ONLY the near-table bowl -> a single survivor, not two.
    idx = _table_with_bowls_scene()
    res = T.resolve(_bowl_on_closest_table_spec(), idx)
    assert [c.instance_id for c in res.candidates_ranked] == [4]


def test_dropped_disambiguator_recorded_in_audit():
    # OR-F1 audit honesty: when the disambiguator's own anchor noun has no
    # instances, the disambiguator is dropped -- and the drop is visible.
    # No 'screen' in the scene, so closest_to(screen) cannot apply.
    table_a = rec(1, "table", (2, 0, 0.5), (1.0, 1.0, 1.0))
    table_b = rec(2, "table", (8, 0, 0.5), (1.0, 1.0, 1.0))
    bowl_a = rec(3, "bowl", (2, 0, 1.05), (0.3, 0.3, 0.1))
    idx = FakeIndex([table_a, table_b, bowl_a])
    res = T.resolve(_bowl_on_closest_table_spec(), idx)
    steps = [a.step for a in res.audit]
    assert "drop_disambiguator" in steps
    dropped = [a.detail for a in res.audit if a.step == "drop_disambiguator"]
    assert any("screen" in d for d in dropped)


def test_nonsuperlative_disambiguator_filters_anchor():
    # "the bowl on the table NEAR the window": the near-window table is picked, so
    # only the bowl on it survives.
    window = rec(10, "window", (0, 0, 1.0), (0.1, 1.0, 2.0))
    table_near = rec(1, "table", (1.0, 0, 0.5), (1.0, 1.0, 1.0))  # near window
    table_far = rec(2, "table", (20, 0, 0.5), (1.0, 1.0, 1.0))  # far
    bowl_near = rec(4, "bowl", (1.0, 0, 1.05), (0.3, 0.3, 0.1))  # on near table
    bowl_far = rec(3, "bowl", (20, 0, 1.05), (0.3, 0.3, 0.1))  # on far table
    idx = FakeIndex([window, table_near, table_far, bowl_near, bowl_far])
    disamb = Clause(Pred.NEAR, [Anchor(noun="window")])
    spec = _spec(
        "bowl",
        clauses=[Clause(Pred.ON, [Anchor(noun="table", disambiguator=disamb)])],
    )
    res = T.resolve(spec, idx)
    assert [c.instance_id for c in res.candidates_ranked] == [4]


def test_disambiguator_recursion_depth_guarded():
    # A pathological chain of nested disambiguators must terminate (depth limit),
    # not recurse forever. Build a self-similar 'table near a table near a ...'.
    from core.plan_schema import Anchor as A, Clause as C

    inner = A(noun="table")
    for _ in range(8):  # deeper than _MAX_ANCHOR_DEPTH
        inner = A(noun="table", disambiguator=C(Pred.NEAR, [inner]))
    idx = FakeIndex([rec(1, "table", (0, 0, 0)), rec(2, "bowl", (0, 0, 0))])
    spec = _spec("bowl", clauses=[Clause(Pred.ON, [inner])])
    # must return without hanging / RecursionError
    res = T.resolve(spec, idx)
    assert isinstance(res.candidates_ranked, list)


def test_single_relation_resolve_unchanged():
    # regression: a plain single-relation resolve (no disambiguator) is byte-for-byte
    # the same behaviour as before -- winner + empty audit.
    table = rec(1, "table", (0, 0, 0.5), (2.0, 2.0, 1.0))
    bowl_on = rec(2, "bowl", (0, 0, 1.05), (0.3, 0.3, 0.1))
    bowl_off = rec(3, "bowl", (5, 5, 1.05), (0.3, 0.3, 0.1))
    idx = FakeIndex([table, bowl_on, bowl_off])
    spec = _spec("bowl", clauses=[Clause(Pred.ON, [Anchor(noun="table")])])
    res = T.resolve(spec, idx)
    assert [c.instance_id for c in res.candidates_ranked] == [2]
    assert res.audit == []
