"""NumericalHead: count correctness, cross-tick stability run, StabilitySignal semantics."""
from __future__ import annotations

from core.heads.numerical import STABLE_TICKS, NumericalHead
from core.interfaces import IntAnswer
from tests.heads._helpers import inst, near_clause, numerical_plan, scene


def _advance(head: NumericalHead, sc, n: int) -> None:
    for _ in range(n):
        head.advance(sc)


def test_counts_matching_instances():
    sc = scene(inst(1, "chair"), inst(2, "chair"), inst(3, "table"))
    head = NumericalHead(plan=numerical_plan("chair"))
    head.advance(sc)
    assert head.count == 2
    assert head.answer() == IntAnswer(2)


def test_count_zero_for_absent_noun():
    sc = scene(inst(1, "table"))
    head = NumericalHead(plan=numerical_plan("unicorn"))
    head.advance(sc)
    assert head.count == 0
    assert head.answer() == IntAnswer(0)


def test_typo_tolerant_count():
    sc = scene(inst(1, "refrigerator"), inst(2, "fridge"))
    head = NumericalHead(plan=numerical_plan("refridgerator"))
    head.advance(sc)
    # synonym (fridge) + typo (refrigerator canonical) both fold to the fridge head
    assert head.count == 2


def test_not_stable_before_stable_ticks():
    sc = scene(inst(1, "chair"), inst(2, "chair"))
    head = NumericalHead(plan=numerical_plan("chair"))
    head.advance(sc)  # 1 tick
    sig = head.signal()
    assert sig.min_contrib_n_obs == 3
    assert sig.winner_margin == 0.0
    assert sig.stable is False


def test_stable_after_run_holds():
    sc = scene(inst(1, "chair"), inst(2, "chair"))
    head = NumericalHead(plan=numerical_plan("chair"))
    _advance(head, sc, STABLE_TICKS)
    sig = head.signal()
    assert sig.winner_margin >= 0.25
    assert sig.min_contrib_n_obs >= 3
    assert sig.stable is True


def test_changing_count_resets_run():
    sc1 = scene(inst(1, "chair"))
    sc2 = scene(inst(1, "chair"), inst(2, "chair"))
    head = NumericalHead(plan=numerical_plan("chair"))
    _advance(head, sc1, STABLE_TICKS)
    assert head.signal().stable is True
    head.advance(sc2)  # count jumps 1 -> 2: run resets
    assert head.signal().stable is False
    _advance(head, sc2, STABLE_TICKS - 1)
    assert head.signal().stable is True


def test_low_nobs_contributor_blocks_stability():
    # A contributor with n_obs < 3 keeps min_contrib_n_obs below the gate.
    sc = scene(inst(1, "chair", n_obs=3), inst(2, "chair", n_obs=2))
    head = NumericalHead(plan=numerical_plan("chair"))
    _advance(head, sc, STABLE_TICKS)
    sig = head.signal()
    assert sig.min_contrib_n_obs == 2
    assert sig.stable is False  # held long enough but a contributor is under-observed


def test_none_scene_is_safe():
    head = NumericalHead(plan=numerical_plan("chair"))
    head.advance(None)
    assert head.signal().stable is False
    assert head.answer() == IntAnswer(0)


def test_empty_index_withholds_count_not_zero():
    # Perception dark / not wired: zero tracked instances overall is absence of data, not
    # an observed zero — the head must withhold rather than claim a false "0" (the FSM
    # floor's modal count is the correct fallback; see core/fsm/floors.py).
    sc = scene()  # empty index
    head = NumericalHead(plan=numerical_plan("chair"))
    head.advance(sc)
    assert head.count is None
    assert head.answer() is None
    assert head.signal().stable is False


def test_empty_index_recovers_once_scene_populates():
    head = NumericalHead(plan=numerical_plan("chair"))
    head.advance(scene())  # dark tick: withheld
    assert head.answer() is None
    head.advance(scene(inst(1, "chair"), inst(2, "chair")))  # perception comes online
    assert head.count == 2
    assert head.answer() == IntAnswer(2)


def test_clause_filter_narrows_count():
    # two chairs, one near a table -> counting with a NEAR clause yields 1
    sc = scene(
        inst(1, "chair", centroid=(0.0, 0.0, 0.0)),
        inst(2, "chair", centroid=(10.0, 10.0, 0.0)),
        inst(3, "table", centroid=(0.6, 0.0, 0.0)),
    )
    head = NumericalHead(plan=numerical_plan("chair", clauses=[near_clause("table")]))
    head.advance(sc)
    assert head.count == 1
