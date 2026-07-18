"""FloorAnswers: always-ready, always-legal degraded answers per qtype."""
from __future__ import annotations

from core.fsm.floors import FloorAnswers, MODAL_COUNT, PartialResults
from core.interfaces import IntAnswer, MarkerBox, QType, WaypointCmd
from core.plan_schema import Plan, TargetSpec
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
