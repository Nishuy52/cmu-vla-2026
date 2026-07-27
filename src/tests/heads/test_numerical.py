"""NumericalHead: count correctness, cross-tick stability run, StabilitySignal semantics."""
from __future__ import annotations

from core.heads.numerical import STABLE_TICKS, NumericalHead
from core.interfaces import IntAnswer
from core.plan_schema import Anchor, Clause, Plan, Pred
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


# --------------------------------------------------------------------- #93: a dropped
# nested disambiguator must not let the head confidently commit to a silently-widened
# count. Live repro: "how many chairs are near the table with a vase on it?" published
# 28 -- exactly the whole tracked chair census -- because the vase was never detected,
# `with(vase)` was dropped, and near(any table) admitted every chair. Truth was 8.
#
# Scene mirrors the repro: two tables 10 m apart (one carrying a vase), two chairs
# flanking each. Truth for "chairs near the table with a vase" is 2.


def _chairs_near_table_with_vase_plan() -> Plan:
    table_with_vase = Anchor(
        noun="table", disambiguator=Clause(pred=Pred.WITH, anchors=[Anchor(noun="vase")])
    )
    clause = Clause(pred=Pred.NEAR, anchors=[table_with_vase])
    return numerical_plan("chair", clauses=[clause])


def _two_tables_two_chairs_each(with_vase: bool):
    recs = [
        inst(1, "table", centroid=(0.0, 0.0, 0.4), extent=(1.2, 1.2, 0.8)),
        inst(2, "table", centroid=(10.0, 0.0, 0.4), extent=(1.2, 1.2, 0.8)),
        inst(3, "chair", centroid=(1.0, 0.0, 0.5), extent=(0.5, 0.5, 1.0)),
        inst(4, "chair", centroid=(-1.0, 0.0, 0.5), extent=(0.5, 0.5, 1.0)),
        inst(5, "chair", centroid=(11.0, 0.0, 0.5), extent=(0.5, 0.5, 1.0)),
        inst(6, "chair", centroid=(9.0, 0.0, 0.5), extent=(0.5, 0.5, 1.0)),
    ]
    if with_vase:
        recs.append(inst(7, "vase", centroid=(0.0, 0.0, 0.9), extent=(0.2, 0.2, 0.2)))
    return scene(*recs)


def test_dropped_disambiguator_never_reports_stable():
    # The core #93 regression: with the vase never detected, the count holds steady at
    # the widened full census (4) tick after tick -- a run that would otherwise satisfy
    # STABLE_TICKS -- but the head must never report it stable, because that count
    # rests on a hard filter ('with a vase') that could not be verified.
    sc = _two_tables_two_chairs_each(with_vase=False)
    head = NumericalHead(plan=_chairs_near_table_with_vase_plan())
    _advance(head, sc, STABLE_TICKS * 5)  # well past the stable-ticks threshold
    assert head.count == 4  # full chair census -- the widened, unverified count
    assert head.signal().stable is False
    assert head.signal().winner_margin == 0.0


def test_dropped_disambiguator_still_always_publishes_an_answer():
    # Hard constraint: an answer must always be published, even when the constraint
    # backing the count is undecidable. The head withholds the STABLE verdict (above)
    # but never withholds the answer itself -- the FSM's budget/watchdog floor still
    # gets a legal IntAnswer to publish once exploration time runs out.
    sc = _two_tables_two_chairs_each(with_vase=False)
    head = NumericalHead(plan=_chairs_near_table_with_vase_plan())
    _advance(head, sc, STABLE_TICKS * 5)
    ans = head.answer()
    assert ans is not None
    assert ans == IntAnswer(4)


def test_disambiguator_satisfied_still_stabilizes_to_narrowed_count():
    # Control: once the vase IS detected, the disambiguator applies, the count is
    # correctly narrowed to 2, and stability fires normally -- #93's fix must not make
    # every disambiguated count perpetually provisional, only ones that are dropped.
    sc = _two_tables_two_chairs_each(with_vase=True)
    head = NumericalHead(plan=_chairs_near_table_with_vase_plan())
    _advance(head, sc, STABLE_TICKS)
    assert head.count == 2
    assert head.signal().stable is True
    assert head.answer() == IntAnswer(2)


def test_disambiguator_resolving_mid_run_lets_stability_recover():
    # The vase is undetected at first (provisional, unstable full-census count), then
    # perception finds it -- once it's part of the scene the disambiguator applies and
    # the head can stabilize on the now-verified, correctly-narrowed count.
    sc_no_vase = _two_tables_two_chairs_each(with_vase=False)
    sc_with_vase = _two_tables_two_chairs_each(with_vase=True)
    head = NumericalHead(plan=_chairs_near_table_with_vase_plan())
    _advance(head, sc_no_vase, STABLE_TICKS * 2)
    assert head.signal().stable is False
    _advance(head, sc_with_vase, STABLE_TICKS)
    assert head.count == 2
    assert head.signal().stable is True
