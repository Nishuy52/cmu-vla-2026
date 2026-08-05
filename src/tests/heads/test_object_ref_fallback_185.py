"""Issue #185: when the target class has no ELIGIBLE candidate (``answer_present=false``
in the archived plan dump), the forced fallback published the ANCHOR class instead of the
TARGET class.

Live evidence (job 713818, home_building_1 group 5, "Find the bowl closest to the knife
rack near the trash can."): the head logs 'knife rack not found' and
``answer_present=false`` (``reports/cluster_verify/713818/debug/5_home_building_1_obje/
resolved_plan.jsonl``), yet 5 'bowl' instances ARE present in the scene index
(``before_any_filter``/``after_all_filters`` both 5) -- every one of them fails the #43a
answer-eligibility floor (``n_obs>=2`` AND ``score>=0.30``; two are single-observation
ghosts on top of that). Pre-#185, ``ObjectRefHead._first_committable`` (OR-F8's
provisional-commit guard) refused to publish any of the 5 bowls and returned None, which
forced ``core.fsm.floors.FloorAnswers._object_reference`` down to its anchor-guided rung
(rung 4): the plan's clause anchor nouns are 'knife rack' (0 instances -> skipped) then
'trash can' (present, eligible) -> the floor published a TRASH CAN marker for a BOWL
question, 12.36 m from GT.

The fix (``ObjectRefHead._first_committable``): once ANY target-class candidate is ranked
at all, ``verify()`` never again returns None to defer to the floor -- it falls open to the
best-ranked candidate from the relaxed (unestablished / under-score) pool. A relaxed
TARGET-class marker is always closer to correct than a floor-forced ANCHOR-class one.
"""
from __future__ import annotations

import numpy as np

from core.heads.object_ref import ObjectRefHead
from core.fsm.floors import FloorAnswers, PartialResults
from core.interfaces import InstanceRecord
from core.plan_schema import Anchor, Clause, Plan, Pred, QType, TargetSpec
from tests.heads._helpers import scene


def _rec(iid: int, label: str, aabb_min, aabb_max, n_obs: int, score: float) -> InstanceRecord:
    lo = np.array(aabb_min, dtype=float)
    hi = np.array(aabb_max, dtype=float)
    return InstanceRecord(
        instance_id=iid,
        label=label,
        score=score,
        n_obs=n_obs,
        centroid=(lo + hi) / 2.0,
        aabb_min=lo,
        aabb_max=hi,
    )


# --------------------------------------------------------------------------- #185
# archived-data fixtures (verbatim aabb_min/aabb_max/n_obs/score, as archived from
# reports/cluster_verify/713818/debug/5_home_building_1_obje/instance_index.jsonl)
# -- 'knife rack' has zero instances in this scene, matching the archived audit
# ("knife rack not found").

_HOMEBUILDING1_BOWLS = [
    _rec(30, "bowl", (0.067, -2.546, 0.885), (0.451, -2.289, 1.058), 2, 0.2655),
    _rec(62, "bowl", (1.892, 3.071, 0.954), (2.28, 3.449, 1.127), 2, 0.2109),
    _rec(262, "bowl", (-0.407, -6.6, 1.076), (-0.046, -6.572, 1.41), 2, 0.2867),
    _rec(305, "bowl", (-2.357, -6.602, 1.59), (-1.214, -6.324, 3.35), 1, 0.298),
    _rec(307, "bowl", (7.372, 2.923, 0.648), (7.984, 3.268, 1.113), 1, 0.2154),
]

_HOMEBUILDING1_TRASH_CANS = [
    _rec(33, "trash can", (5.162, -4.116, -0.0), (5.869, -3.892, 1.01), 4, 0.3058),
    _rec(51, "trash can", (4.996, -3.486, -0.0), (6.29, -2.827, 0.899), 6, 0.3433),
    _rec(64, "trash can", (-3.168, 3.74, 0.223), (-2.951, 4.164, 0.897), 2, 0.2392),
    _rec(95, "trash can", (1.443, -8.486, 0.054), (2.383, -8.456, 0.943), 2, 0.2045),
    _rec(135, "trash can", (3.003, -12.809, -0.0), (3.8, -12.582, 1.248), 2, 0.2098),
    _rec(195, "trash can", (-0.056, -6.616, 0.497), (0.931, -6.586, 1.503), 5, 0.2943),
    _rec(220, "trash can", (5.301, -0.332, 0.028), (5.74, 0.882, 0.519), 3, 0.3191),
    _rec(306, "trash can", (12.744, -6.179, 0.068), (12.854, -4.871, 1.195), 1, 0.2048),
]


def _plan() -> Plan:
    # "Find the bowl closest to the knife rack near the trash can."
    knife_rack = Anchor(
        noun="knife rack", raw="knife rack",
        disambiguator=Clause(pred=Pred.NEAR, anchors=[Anchor(noun="trash can", raw="trash can")]),
    )
    target = TargetSpec(
        noun="bowl", raw="bowl", attributes=[],
        clauses=[Clause(pred=Pred.CLOSEST_TO, anchors=[knife_rack])],
    )
    return Plan(
        qtype=QType.OBJECT_REFERENCE,
        question_raw="Find the bowl closest to the knife rack near the trash can.",
        target=target,
        notes="indefinite article on 'knife rack' and 'trash can'",
    )


def test_pre_185_head_alone_would_publish_nothing_all_bowls_ineligible():
    """Sanity: every archived bowl genuinely fails the #43a answer-eligibility floor (this
    confirms the fixture reproduces the archived 'no eligible candidate' condition, not
    just an absent target)."""
    from core.perception.detector import is_answer_eligible

    assert not any(is_answer_eligible(b) for b in _HOMEBUILDING1_BOWLS)


def test_bowl_fallback_publishes_bowl_not_trash_can():
    """(#185) ``ObjectRefHead.verify()`` publishes a BOWL marker -- never None, and never
    an anchor-class marker -- even though every ranked bowl candidate is under the
    answer-eligibility floor."""
    sc = scene(*_HOMEBUILDING1_BOWLS, *_HOMEBUILDING1_TRASH_CANS)
    head = ObjectRefHead(plan=_plan())
    head.advance(sc)
    m = head.verify()
    assert m is not None
    assert m.label == "bowl"
    assert head.best_candidate is not None
    assert head.best_candidate.label == "bowl"
    assert head.best_candidate.instance_id in {b.instance_id for b in _HOMEBUILDING1_BOWLS}


def test_bowl_fallback_end_to_end_floor_never_emits_trash_can():
    """(#185) End-to-end: because ``ObjectRefHead.verify()`` no longer returns None for
    this slot, the FSM floor (``core.fsm.floors.FloorAnswers``) never even reaches its
    anchor-guided rung -- the published label stays 'bowl', never 'trash can', for the
    whole pipeline the archived run actually exercises."""
    sc = scene(*_HOMEBUILDING1_BOWLS, *_HOMEBUILDING1_TRASH_CANS)
    plan = _plan()
    head = ObjectRefHead(plan=plan)
    head.advance(sc)
    head.verify()

    partial = PartialResults()
    head.publish_partial(partial)

    floor = FloorAnswers()
    floor.update(sc, plan, partial)
    marker = floor.get(QType.OBJECT_REFERENCE)
    assert marker.label == "bowl"
    assert marker.label != "trash can"


def test_target_wholly_absent_still_falls_through_to_floor():
    """Control: when the TARGET class has zero instances at all (not merely zero eligible
    ones), ``verify()`` still correctly returns None -- #185 only relaxes the
    already-ranked-but-ineligible case, never fabricates a candidate out of nothing."""
    sc = scene(*_HOMEBUILDING1_TRASH_CANS)  # no bowls in the scene at all
    head = ObjectRefHead(plan=_plan())
    head.advance(sc)
    assert head.verify() is None
    assert head.best_candidate is None
