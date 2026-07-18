"""ObjectRefHead: ranking, best-marker partial, llm_verify demotion checkpoint."""
from __future__ import annotations

from core.heads.object_ref import ObjectRefHead
from core.fsm.floors import PartialResults
from core.interfaces import MarkerBox
from tests.heads._helpers import closest_clause, inst, near_clause, object_plan, scene


def test_ranks_and_marks_best_candidate():
    # table near lamp vs far table -> the near one wins
    sc = scene(
        inst(1, "table", centroid=(0.0, 0.0, 0.0)),
        inst(2, "table", centroid=(10.0, 0.0, 0.0)),
        inst(3, "lamp", centroid=(0.6, 0.0, 0.0), extent=(0.2, 0.2, 0.3)),
    )
    head = ObjectRefHead(plan=object_plan("table", clauses=[near_clause("lamp")]))
    head.advance(sc)
    assert head.best_candidate.instance_id == 1
    assert isinstance(head.best_marker, MarkerBox)
    assert head.best_marker.label == "table"


def test_publish_partial_populates_best_marker():
    sc = scene(inst(1, "sofa", centroid=(1.0, 1.0, 0.0)))
    head = ObjectRefHead(plan=object_plan("sofa"))
    head.advance(sc)
    partial = PartialResults()
    head.publish_partial(partial)
    assert partial.best_marker is not None
    assert partial.best_candidate.instance_id == 1


def test_verify_returns_winner_marker():
    sc = scene(inst(1, "chair", centroid=(2.0, 3.0, 0.0)))
    head = ObjectRefHead(plan=object_plan("chair"))
    head.advance(sc)
    m = head.verify()
    assert isinstance(m, MarkerBox)
    assert (round(m.cx, 3), round(m.cy, 3)) == (2.0, 3.0)


def test_verify_none_when_nothing_ranked():
    sc = scene(inst(1, "table"))
    head = ObjectRefHead(plan=object_plan("unicorn"))
    head.advance(sc)
    # category-only fallback still returns all 'unicorn' (empty) -> nothing ranked
    assert head.verify() is None


def test_verify_none_on_empty_index():
    # Perception fully dark (empty scene index): resolve()'s noun-match/category-only
    # ladder bottoms out at [] (no instances at all to fall back to), so nothing ranks.
    # The head must not fabricate a marker from nothing — verify() withholds (None) so
    # the FSM floor's richer any-instance fallback (core/fsm/floors.py
    # FloorAnswers._object_reference, rungs 3-5) answers instead. Mirrors the
    # NumericalHead empty-index withhold pattern (core/heads/numerical.py).
    sc = scene()
    head = ObjectRefHead(plan=object_plan("chair"))
    head.advance(sc)
    assert head.best_candidate is None
    assert head.best_marker is None
    assert head.verify() is None


def test_llm_verify_demotes_failed_winner():
    sc = scene(
        inst(1, "chair", centroid=(0.0, 0.0, 0.0)),
        inst(2, "chair", centroid=(5.0, 0.0, 0.0)),
        inst(3, "table", centroid=(0.5, 0.0, 0.0)),
    )
    head = ObjectRefHead(plan=object_plan("chair", clauses=[closest_clause("table")]))
    head.advance(sc)
    # deterministic winner is id 1 (closest to table). Stub verifier rejects it once.
    calls = {"n": 0}

    def verify(plan, summary, matrix):
        calls["n"] += 1
        return calls["n"] > 1  # reject the first candidate, accept the next

    head.llm_verify = verify
    m = head.verify()
    assert calls["n"] == 2
    assert head.best_candidate.instance_id == 2  # demoted to runner-up


def test_llm_verify_all_rejected_falls_back_to_top():
    sc = scene(inst(1, "chair"), inst(2, "chair", centroid=(5, 0, 0)))
    head = ObjectRefHead(plan=object_plan("chair"))
    head.advance(sc)
    head.llm_verify = lambda p, s, m: False  # reject everything
    m = head.verify()
    # every candidate demoted -> deterministic top rank stands (never silence)
    assert m is not None
    assert head.best_candidate.instance_id == 1


def test_llm_verify_exception_trusts_deterministic_rank():
    sc = scene(inst(1, "chair"), inst(2, "chair", centroid=(5, 0, 0)))
    head = ObjectRefHead(plan=object_plan("chair"))
    head.advance(sc)

    def boom(p, s, m):
        raise RuntimeError("dark checkpoint")

    head.llm_verify = boom
    m = head.verify()
    assert head.best_candidate.instance_id == 1  # exception == keep


def test_none_scene_safe():
    head = ObjectRefHead(plan=object_plan("chair"))
    head.advance(None)
    assert head.best_marker is None
    assert head.verify() is None
