"""H2 scorer/battery repairs — IF rubric proxy + strict count matching (NUM-F6/IF-F2).

Covers the four behaviours the hardening backlog H2 requires unit coverage for:
ordered-leg partial credit, threading/avoid penalty detection on synthetic paths,
strict target-class matching (the "photo frame vs photo" / "tv cabinet vs tv" cases),
and class_only exclusion from the agreement signal. Plus a driven-trajectory smoke
test over a hand-built GT scene (the constant-speed follower actually reaches the legs).
"""
from __future__ import annotations

import numpy as np
import pytest

from core.geometry.toolbox import Capsule, Gate
from core.groundtruth import scoring as S
from core.groundtruth.loader import GTScene
from core.groundtruth.scoring import (
    _class_equal,
    score_instruction_rubric,
    score_numerical,
)
from core.interfaces import InstanceRecord
from core.perception.scene_index import BasicSceneIndex
from core.runner import gt_battery as B


# --------------------------------------------------------------------------- helpers


def _rec(iid: int, label: str, x: float, y: float, hw: float = 0.3) -> InstanceRecord:
    amin = np.array([x - hw, y - hw, 0.0])
    amax = np.array([x + hw, y + hw, 0.5])
    return InstanceRecord(
        instance_id=iid, label=label, score=1.0, n_obs=3,
        centroid=(amin + amax) / 2, aabb_min=amin, aabb_max=amax,
        points=None, caption="",
    )


# --------------------------------------------------------------- ordered-leg credit


def test_rubric_full_credit_when_all_legs_reached_in_order():
    traj = np.array([[0, 0], [1, 0], [2, 0], [3, 0]], dtype=float)
    goals = [("goto", (1.0, 0.0)), ("goto", (3.0, 0.0))]
    r = score_instruction_rubric(traj, goals)
    assert r.rubric_score == pytest.approx(1.0)
    assert r.n_legs_reached_in_order == 2
    assert r.ordered_leg_credit == pytest.approx(1.0)


def test_rubric_partial_credit_one_of_two_legs():
    # Trajectory only reaches the first leg goal; second is far away.
    traj = np.array([[0, 0], [1, 0], [2, 0]], dtype=float)
    goals = [("goto", (2.0, 0.0)), ("goto", (20.0, 0.0))]
    r = score_instruction_rubric(traj, goals)
    assert r.n_legs_reached_in_order == 1
    assert r.rubric_score == pytest.approx(0.5)


def test_rubric_ordering_enforced_out_of_order_gets_partial():
    # Trajectory passes the SECOND goal first, then the first — order is violated, so
    # only one leg can count (whichever comes first in a monotone walk).
    traj = np.array([[3, 0], [2, 0], [1, 0]], dtype=float)
    goals = [("goto", (1.0, 0.0)), ("goto", (3.0, 0.0))]
    r = score_instruction_rubric(traj, goals)
    # leg0 (goal 1,0) reached at index 2; leg1 (goal 3,0) must be reached at/after
    # index 3 — never — so only 1 in order.
    assert r.n_legs_reached_in_order == 1
    assert r.rubric_score == pytest.approx(0.5)


def test_rubric_empty_trajectory_scores_zero():
    goals = [("goto", (1.0, 0.0))]
    r = score_instruction_rubric(np.empty((0, 2)), goals)
    assert r.rubric_score == pytest.approx(0.0)
    assert r.driven_n_poses == 0
    assert "empty" in r.note


# ------------------------------------------------------- threading / avoid penalties


def test_rubric_threading_violation_penalises():
    # Gate spans x=0 (y in [-1,1]); a straight path at x>0 never crosses it.
    gate = Gate(
        np.array([0.0, -1.0]), np.array([0.0, 1.0]), np.array([0.0, 0.0]), 2.0
    )
    traj = np.array([[1, 0], [2, 0], [3, 0]], dtype=float)
    goals = [("corridor_between", (0.0, 0.0))]
    r = score_instruction_rubric(traj, goals, corridor_gates=[(0, gate)])
    assert r.n_threading_violations == 1
    assert r.threading_details  # a human-readable reason is recorded
    assert r.rubric_score == pytest.approx(0.0)  # 1 leg, 1 violation -> clamped 0


def test_rubric_threading_satisfied_no_penalty():
    gate = Gate(
        np.array([0.0, -1.0]), np.array([0.0, 1.0]), np.array([0.0, 0.0]), 2.0
    )
    # Path crosses x=0 at the gate midpoint AND arrives at the goal.
    traj = np.array([[-1, 0], [0, 0], [1, 0]], dtype=float)
    goals = [("corridor_between", (0.0, 0.0))]
    r = score_instruction_rubric(traj, goals, corridor_gates=[(0, gate)])
    assert r.n_threading_violations == 0
    assert r.rubric_score == pytest.approx(1.0)


def test_rubric_avoid_capsule_breach_penalises():
    # Disc capsule around (2, 0) radius 0.5; the path drives straight through it.
    cap = Capsule(np.array([2.0, 0.0]), np.array([2.0, 0.0]), 0.5)
    traj = np.array([[0, 0], [1, 0], [2, 0], [3, 0]], dtype=float)
    goals = [("goto", (3.0, 0.0))]
    r = score_instruction_rubric(traj, goals, avoid_capsules=[cap])
    assert r.n_avoid_violations == 1
    assert r.avoid_details
    assert r.rubric_score == pytest.approx(0.0)  # 1 leg reached, 1 breach -> 0


def test_rubric_avoid_capsule_respected_no_penalty():
    cap = Capsule(np.array([2.0, 5.0]), np.array([2.0, 5.0]), 0.5)  # off to the side
    traj = np.array([[0, 0], [1, 0], [2, 0], [3, 0]], dtype=float)
    goals = [("goto", (3.0, 0.0))]
    r = score_instruction_rubric(traj, goals, avoid_capsules=[cap])
    assert r.n_avoid_violations == 0
    assert r.rubric_score == pytest.approx(1.0)


def test_rubric_frechet_coverage_are_secondary_only(tmp_path):
    ply = tmp_path / "traj.ply"
    ply.write_text(
        "ply\nformat ascii 1.0\nelement vertex 2\nproperty float x\n"
        "property float y\nproperty float z\nend_header\n"
        "0.0 0.0 0.75\n3.0 0.0 0.75\n",
        encoding="utf-8",
    )
    traj = np.array([[0, 0], [1, 0], [2, 0], [3, 0]], dtype=float)
    goals = [("goto", (3.0, 0.0))]
    r = score_instruction_rubric(traj, goals, trajectory_ply=ply)
    # diagnostics are populated but the headline is the rubric, unaffected by them
    assert r.coverage_1m is not None
    assert r.rubric_score == pytest.approx(1.0)


# -------------------------------------------------------- strict class matching (NUM-F6)


def test_class_equal_rejects_substring_photo_frame_vs_photo():
    # NUM-F6: "photo frame" targets must NOT count as "photo" (livingroom_3 inflation).
    assert _class_equal("photo", "photo frame") is False
    assert _class_equal("photo frame", "photo") is False


def test_class_equal_rejects_shared_token_tv_cabinet_vs_tv():
    # NUM-F6: the "tv" token must NOT match a "tv cabinet" to a TV relation.
    assert _class_equal("tv", "tv cabinet") is False
    assert _class_equal("tv cabinet", "tv") is False


def test_class_equal_accepts_normalised_equality_and_bridge():
    assert _class_equal("photo", "photo") is True
    assert _class_equal("photos", "photo") is True  # normalize_label singularises
    # whitelisted vocab bridge (same object, surface drift) still matches
    assert _class_equal("bedside table", "night stand") is True


def _ref(regions):
    return {"regions": regions}


def test_independent_count_strict_class_excludes_photo_frame():
    # 2 "photo" targets + 1 "photo frame" target of the same relation. Strict class
    # equality must count only the 2 photos, not inflate to 3.
    ref = _ref({
        "0": {
            "the photo on the wall": [
                {"target_index": "1", "target_class": "photo",
                 "anchors": {"a": {"class": "wall"}}},
                {"target_index": "2", "target_class": "photo",
                 "anchors": {"a": {"class": "wall"}}},
                {"target_index": "3", "target_class": "photo frame",
                 "anchors": {"a": {"class": "wall"}}},
            ]
        }
    })
    idx = BasicSceneIndex([_rec(1, "photo", 0, 0), _rec(2, "photo", 1, 0)])
    ns = score_numerical("How many photos are on the wall?", idx, referential=ref)
    assert ns.gt_count_independent == 2
    assert ns.independent_source == "referential"


# ----------------------------------------------------- parse-time notes (issue #26)


def test_score_numerical_propagates_plan_parse_notes(monkeypatch):
    """A clause-dropped-at-parse-time signal must survive onto NumericalScore."""
    from core.plan_schema import Plan, TargetSpec
    from core.interfaces import QType as _QType

    fake_plan = Plan(
        qtype=_QType.NUMERICAL, question_raw="q",
        target=TargetSpec(noun="photo", raw="photo"),
        notes="unparsed clause text dropped: 'near the thing'",
    )
    monkeypatch.setattr(S, "parse_regex", lambda text: fake_plan)
    idx = BasicSceneIndex([_rec(1, "photo", 0, 0)])
    ns = score_numerical("How many photos near the thing?", idx)
    assert ns.parse_notes == "unparsed clause text dropped: 'near the thing'"


def test_score_object_reference_propagates_plan_parse_notes(monkeypatch):
    """Same clause-dropped signal must survive onto ObjectRefScore."""
    from core.plan_schema import Plan, TargetSpec
    from core.interfaces import QType as _QType

    fake_plan = Plan(
        qtype=_QType.OBJECT_REFERENCE, question_raw="q",
        target=TargetSpec(noun="photo", raw="photo"),
        notes="unparsed clause text dropped: 'near the thing'",
    )
    monkeypatch.setattr(S, "parse_regex", lambda text: fake_plan)
    instances = [_rec(1, "photo", 0, 0)]
    idx = BasicSceneIndex(instances)
    ors = S.score_object_reference("Go to the photo near the thing.", idx, instances)
    assert ors.parse_notes == "unparsed clause text dropped: 'near the thing'"


# ------------------------------------------------------ class_only exclusion (NUM-F6b)


def test_class_only_row_reported_as_no_evidence_not_disagreement():
    # No statement matches the named relation/anchor, but the class IS annotated ->
    # a relation-agnostic class_only count. It must NOT be scored as a disagreement.
    ref = _ref({
        "0": {
            "the sofa near the lamp": [
                {"target_index": "10", "target_class": "sofa",
                 "anchors": {"a": {"class": "lamp"}}},
                {"target_index": "11", "target_class": "sofa",
                 "anchors": {"a": {"class": "lamp"}}},
                {"target_index": "12", "target_class": "sofa",
                 "anchors": {"a": {"class": "lamp"}}},
            ]
        }
    })
    # Question relation (below a window) is absent from the annotations -> class_only.
    idx = BasicSceneIndex([_rec(10, "sofa", 0, 0)])
    ns = score_numerical("How many sofas are below the window?", idx, referential=ref)
    assert ns.independent_source == "referential_class_only"
    # class_only is not counted as a disagreement even though 1 != 3
    assert "disagreement" not in ns.note
    assert "no independent evidence" in ns.note


def test_annotation_coverage_flags_underannotated_class():
    # Class annotated once as a target but the scene CSV has 3 instances -> deflated.
    ref = _ref({
        "0": {
            "the framed record near the shelf": [
                {"target_index": "1", "target_class": "framed record",
                 "anchors": {"a": {"class": "shelf"}}},
            ]
        }
    })
    idx = BasicSceneIndex([
        _rec(1, "framed record", 0, 0),
        _rec(2, "framed record", 1, 0),
        _rec(3, "framed record", 2, 0),
    ])
    ns = score_numerical(
        "How many framed records are near the shelf?", idx, referential=ref
    )
    assert ns.csv_instances_of_class == 3
    assert ns.annotated_targets_of_class == 1
    assert "under-annotated" in ns.note


# --------------------------------------------------------- driven-trajectory smoke


def test_driven_trajectory_reaches_ordered_legs():
    """The constant-speed follower actually drives through both ordered leg goals."""
    insts = [_rec(0, "door", 1.0, 1.0), _rec(1, "table", 5.0, 5.0)]
    gt = GTScene(scene_name="t", instances=insts, regions=[])
    idx = BasicSceneIndex(insts)
    q = "First go to the door, then go to the table."
    leg_goals, gates, caps = B._if_rubric_geometry(q, gt, idx)
    assert [k for k, _ in leg_goals] == ["goto", "goto"]
    traj = B._drive_if_trajectory(q, gt, idx, start_xy=(0.5, 0.5))
    assert traj.shape[0] > 2  # a real motion stream, not a point
    r = score_instruction_rubric(traj, leg_goals, corridor_gates=gates, avoid_capsules=caps)
    assert r.n_legs_reached_in_order == 2
    assert r.rubric_score == pytest.approx(1.0)
