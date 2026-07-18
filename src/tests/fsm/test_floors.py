"""FloorAnswers: always-ready, always-legal degraded answers per qtype."""
from __future__ import annotations

from core.fsm.floors import FloorAnswers, MODAL_COUNT, PartialResults
from core.interfaces import IntAnswer, MarkerBox, QType, WaypointCmd
from core.plan_schema import Anchor, Clause, Plan, Pred, TargetSpec
from tests.fsm._fakes import FakeScene, make_instance


def _num_plan(noun: str) -> Plan:
    return Plan(qtype=QType.NUMERICAL, question_raw="", target=TargetSpec(noun=noun))


def test_numerical_counts_matching_noun():
    scene = FakeScene([make_instance(1, "chair"), make_instance(2, "chair"), make_instance(3, "table")])
    f = FloorAnswers()
    f.update(scene, _num_plan("chair"), PartialResults())
    ans = f.get(QType.NUMERICAL)
    assert isinstance(ans, IntAnswer) and ans.value == 2


def test_numerical_modal_when_no_match():
    scene = FakeScene([make_instance(1, "table")])
    f = FloorAnswers()
    f.update(scene, _num_plan("chair"), PartialResults())
    assert f.get(QType.NUMERICAL).value == MODAL_COUNT


def test_numerical_prefers_partial_count():
    scene = FakeScene([make_instance(1, "chair"), make_instance(2, "chair")])
    f = FloorAnswers()
    f.update(scene, _num_plan("chair"), PartialResults(count=7))
    assert f.get(QType.NUMERICAL).value == 7


def test_numerical_modal_on_empty_scene_and_no_plan():
    f = FloorAnswers()
    f.update(FakeScene([]), None, PartialResults())
    assert f.get(QType.NUMERICAL).value == MODAL_COUNT


def test_numerical_modal_on_empty_scene_with_real_plan():
    # Perception dark (zero tracked instances at all): the numerical head is expected to
    # withhold partial.count (leave it None) rather than claim a false "0" — floors.py
    # then falls through to the modal count, exactly as it does with no plan at all.
    f = FloorAnswers()
    f.update(FakeScene([]), _num_plan("chair"), PartialResults(count=None))
    assert f.get(QType.NUMERICAL).value == MODAL_COUNT


def test_numerical_honors_genuine_zero_from_partial_count():
    # A real observed zero (index non-empty, target noun just absent) must still answer
    # 0 -- partial.count is checked with `is not None`, not truthiness, so a real 0 from
    # the numerical head is never mistaken for "no count yet".
    scene = FakeScene([make_instance(1, "table")])
    f = FloorAnswers()
    f.update(scene, _num_plan("unicorn"), PartialResults(count=0))
    assert f.get(QType.NUMERICAL).value == 0


def test_object_ref_best_candidate_marker():
    cand = make_instance(9, "lamp", centroid=(3.0, 4.0, 1.0))
    f = FloorAnswers()
    f.update(FakeScene([cand]), None, PartialResults(best_candidate=cand))
    m = f.get(QType.OBJECT_REFERENCE)
    assert isinstance(m, MarkerBox)
    assert (m.cx, m.cy, m.cz) == (3.0, 4.0, 1.0)
    assert m.label == "lamp"


def test_object_ref_largest_matching_noun():
    small = make_instance(1, "box", extent=(1.0, 1.0, 1.0), centroid=(0, 0, 0))
    big = make_instance(2, "box", extent=(3.0, 3.0, 3.0), centroid=(5, 5, 5))
    f = FloorAnswers()
    plan = Plan(qtype=QType.OBJECT_REFERENCE, question_raw="", target=TargetSpec(noun="box"))
    f.update(FakeScene([small, big]), plan, PartialResults())
    m = f.get(QType.OBJECT_REFERENCE)
    assert (m.cx, m.cy, m.cz) == (5.0, 5.0, 5.0)  # the larger one


def test_object_ref_any_instance_when_noun_absent():
    other = make_instance(1, "sofa", centroid=(2, 2, 2))
    f = FloorAnswers()
    plan = Plan(qtype=QType.OBJECT_REFERENCE, question_raw="", target=TargetSpec(noun="unicorn"))
    f.update(FakeScene([other]), plan, PartialResults())
    m = f.get(QType.OBJECT_REFERENCE)
    assert m.label == "sofa"


def test_object_ref_unit_box_at_most_observed_when_no_match_no_target():
    a = make_instance(1, "a", n_obs=1, centroid=(1, 1, 1))
    b = make_instance(2, "b", n_obs=9, centroid=(8, 8, 8))
    f = FloorAnswers()
    # no plan/noun -> falls to "any instance" (largest). Give equal volume so it uses all-inst path.
    f.update(FakeScene([a, b]), None, PartialResults())
    m = f.get(QType.OBJECT_REFERENCE)
    assert isinstance(m, MarkerBox)  # legal box, never raises


def test_object_ref_empty_scene_unit_box_at_origin():
    f = FloorAnswers()
    f.update(FakeScene([]), None, PartialResults())
    m = f.get(QType.OBJECT_REFERENCE)
    assert (m.cx, m.cy, m.cz) == (0.0, 0.0, 0.0)
    assert (m.sx, m.sy, m.sz) == (1.0, 1.0, 1.0)


def _plan_with_anchor(target_noun: str, anchor_noun: str) -> Plan:
    """e.g. 'find the teapot on the table' -> target='teapot', clause anchor='table'."""
    clause = Clause(pred=Pred.ON, anchors=[Anchor(noun=anchor_noun)])
    return Plan(
        qtype=QType.OBJECT_REFERENCE,
        question_raw="",
        target=TargetSpec(noun=target_noun, clauses=[clause]),
    )


def test_object_ref_anchor_rung_fires_when_target_absent_but_anchor_present():
    # issue #42 repro shape: 'teapot' never grounds, but 'table' does; the anchor rung
    # should mark on the table instead of falling to rung 5's blind largest-any-label
    # guess (which in the real incident landed on an unrelated column).
    table = make_instance(1, "table", centroid=(-0.6, 2.1, 0.3), extent=(1.2, 0.8, 0.4))
    column = make_instance(2, "column", centroid=(2.3, -2.4, 1.5), extent=(0.3, 0.3, 2.0))
    plan = _plan_with_anchor("teapot", "table")
    f = FloorAnswers()
    f.update(FakeScene([table, column]), plan, PartialResults())
    m = f.get(QType.OBJECT_REFERENCE)
    assert m.label == "table"
    assert (m.cx, m.cy, m.cz) == (-0.6, 2.1, 0.3)


def test_object_ref_anchor_rung_picks_largest_anchor_instance():
    small = make_instance(1, "table", centroid=(0, 0, 0), extent=(1.0, 1.0, 1.0))
    big = make_instance(2, "table", centroid=(5, 5, 5), extent=(3.0, 3.0, 3.0))
    plan = _plan_with_anchor("teapot", "table")
    f = FloorAnswers()
    f.update(FakeScene([small, big]), plan, PartialResults())
    m = f.get(QType.OBJECT_REFERENCE)
    assert (m.cx, m.cy, m.cz) == (5.0, 5.0, 5.0)


def test_object_ref_anchor_rung_skipped_when_no_anchor_instances_falls_to_largest_volume():
    # No 'table' instance in the scene either -> the anchor rung has nothing to grab, so
    # this falls through to rung 5 (largest instance of ANY label), same as pre-#42.
    column = make_instance(1, "column", centroid=(2.3, -2.4, 1.5), extent=(0.3, 0.3, 2.0))
    plan = _plan_with_anchor("teapot", "table")
    f = FloorAnswers()
    f.update(FakeScene([column]), plan, PartialResults())
    m = f.get(QType.OBJECT_REFERENCE)
    assert m.label == "column"


def test_object_ref_anchor_rung_skipped_when_plan_has_no_anchors():
    # Target absent, no clause anchors at all (plain 'find the unicorn') -> straight to
    # rung 5, unaffected by the new rung.
    other = make_instance(1, "sofa", centroid=(2, 2, 2))
    plan = Plan(qtype=QType.OBJECT_REFERENCE, question_raw="", target=TargetSpec(noun="unicorn"))
    f = FloorAnswers()
    f.update(FakeScene([other]), plan, PartialResults())
    m = f.get(QType.OBJECT_REFERENCE)
    assert m.label == "sofa"


def test_object_ref_target_match_beats_anchor_rung():
    # Rung 3 (direct target-noun match) still wins over the new anchor rung when the
    # target itself IS grounded.
    teapot = make_instance(1, "teapot", centroid=(-0.6, 2.1, 0.3))
    table = make_instance(2, "table", centroid=(0, 0, 0), extent=(3.0, 3.0, 3.0))
    plan = _plan_with_anchor("teapot", "table")
    f = FloorAnswers()
    f.update(FakeScene([teapot, table]), plan, PartialResults())
    m = f.get(QType.OBJECT_REFERENCE)
    assert m.label == "teapot"


def test_instruction_following_first_anchor_point():
    f = FloorAnswers()
    f.update(FakeScene([]), None, PartialResults(first_anchor_pt=(5.0, 6.0, 0.0)))
    wp = f.get(QType.INSTRUCTION_FOLLOWING)
    assert isinstance(wp, WaypointCmd)
    assert (wp.x, wp.y) == (5.0, 6.0)


def test_instruction_following_scene_centroid():
    scene = FakeScene([make_instance(1, "a", centroid=(0, 0, 0)), make_instance(2, "b", centroid=(4, 8, 0))])
    f = FloorAnswers()
    f.update(scene, None, PartialResults())
    wp = f.get(QType.INSTRUCTION_FOLLOWING)
    assert (wp.x, wp.y) == (2.0, 4.0)


def test_instruction_following_origin_on_empty():
    f = FloorAnswers()
    f.update(FakeScene([]), None, PartialResults())
    wp = f.get(QType.INSTRUCTION_FOLLOWING)
    assert (wp.x, wp.y) == (0.0, 0.0)


def test_update_never_raises_on_garbage():
    f = FloorAnswers()
    # scene raises internally -> swallowed; still returns legal defaults
    class Exploding:
        def all_instances(self):
            raise RuntimeError("boom")

        def by_label(self, noun):
            raise RuntimeError("boom")

    f.update(Exploding(), object(), PartialResults(first_anchor_pt="not-a-point"))
    assert isinstance(f.get(QType.NUMERICAL), IntAnswer)
    assert isinstance(f.get(QType.OBJECT_REFERENCE), MarkerBox)
    assert isinstance(f.get(QType.INSTRUCTION_FOLLOWING), WaypointCmd)


def test_get_before_any_update_is_legal():
    f = FloorAnswers()
    assert f.get(QType.NUMERICAL).value == MODAL_COUNT
    assert isinstance(f.get(QType.OBJECT_REFERENCE), MarkerBox)
    assert isinstance(f.get(QType.INSTRUCTION_FOLLOWING), WaypointCmd)
