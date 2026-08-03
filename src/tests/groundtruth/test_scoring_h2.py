"""H2 scorer/battery repairs — IF rubric proxy + strict count matching (NUM-F6/IF-F2).

Covers the four behaviours the hardening backlog H2 requires unit coverage for:
ordered-leg partial credit, threading/avoid penalty detection on synthetic paths,
strict target-class matching (the "photo frame vs photo" / "tv cabinet vs tv" cases),
and class_only exclusion from the agreement signal. Plus a driven-trajectory smoke
test over a hand-built GT scene (the constant-speed follower actually reaches the legs).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from core.geometry.toolbox import Capsule, Gate
from core.groundtruth import scoring as S
from core.groundtruth.loader import GTScene
from core.groundtruth.scoring import (
    ARRIVAL_RESAMPLE_STEP_M,
    _class_equal,
    _densify_polyline,
    _is_pass_by_leg,
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
    # Explicit tol=0.8 (issue #70: the DEFAULT tol is now the much larger derived
    # nominal tolerance, ~1.75 m — big enough that both 2-m-apart goals below would
    # be reachable from nearly anywhere on this short path, which would trivially
    # satisfy ordering and defeat the point of this test). A small explicit tol
    # isolates the ordering-cursor invariant under test from that default.
    traj = np.array([[3, 0], [2, 0], [1, 0]], dtype=float)
    goals = [("goto", (1.0, 0.0)), ("goto", (3.0, 0.0))]
    r = score_instruction_rubric(traj, goals, tol=0.8)
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


def test_rubric_arrival_over_sparse_waypoints_mid_segment_goal_reached():
    # Issue #58 regression: goal sits 0.5 m off the midpoint of a 10 m segment whose
    # endpoints are both farther than tol from the goal. Without densification the
    # sparse 2-pose trajectory never counts as reaching it.
    traj = np.array([[0, 0], [10, 0]], dtype=float)
    goals = [("goto", (5.0, 0.5))]
    r = score_instruction_rubric(traj, goals, tol=0.8)
    assert r.n_legs_reached_in_order == 1
    assert r.rubric_score == pytest.approx(1.0)


# ------------------------------------------------------------- _densify_polyline


def test_densify_polyline_bounds_spacing_and_keeps_endpoints():
    traj = np.array([[0, 0], [10, 0], [10, 4]], dtype=float)
    dense = _densify_polyline(traj, ARRIVAL_RESAMPLE_STEP_M)
    gaps = np.linalg.norm(np.diff(dense[:, :2], axis=0), axis=1)
    assert np.all(gaps <= ARRIVAL_RESAMPLE_STEP_M + 1e-9)
    assert np.allclose(dense[0], traj[0])
    assert np.allclose(dense[-1], traj[-1])
    # original vertices preserved
    for v in traj:
        assert np.any(np.all(np.isclose(dense, v), axis=1))


def test_densify_polyline_single_point_and_empty_pass_through():
    single = np.array([[1.0, 2.0]])
    assert np.array_equal(_densify_polyline(single, 0.25), single)
    empty = np.empty((0, 2))
    assert np.array_equal(_densify_polyline(empty, 0.25), empty)


def test_densify_polyline_does_not_change_threading_verdict():
    gate = Gate(
        np.array([0.0, -1.0]), np.array([0.0, 1.0]), np.array([0.0, 0.0]), 2.0
    )
    # Sparse path crossing the gate exactly at the midpoint.
    traj = np.array([[-5, 0], [5, 0]], dtype=float)
    goals = [("corridor_between", (0.0, 0.0))]
    r_sparse = score_instruction_rubric(traj, goals, corridor_gates=[(0, gate)])
    r_dense = score_instruction_rubric(
        _densify_polyline(traj, ARRIVAL_RESAMPLE_STEP_M), goals,
        corridor_gates=[(0, gate)],
    )
    assert r_sparse.n_threading_violations == r_dense.n_threading_violations == 0
    assert r_sparse.rubric_score == pytest.approx(r_dense.rubric_score)


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


def test_rubric_threading_still_violation_when_gate_is_a_genuine_wide_miss():
    """issue #155 done-criterion: a normal, non-degenerate, genuinely wide gate
    that the trajectory never threads must still score as a violation exactly as
    before -- the degenerate-gate fix must not blanket-relax the rubric."""
    gate = Gate(
        np.array([0.0, -1.0]), np.array([0.0, 1.0]), np.array([0.0, 0.0]), 2.0,
        degenerate=False,
    )
    traj = np.array([[1, 0], [2, 0], [3, 0]], dtype=float)
    goals = [("corridor_between", (0.0, 0.0))]
    r = score_instruction_rubric(traj, goals, corridor_gates=[(0, gate)])
    assert r.n_threading_violations == 1
    assert r.n_threading_unevaluable == 0
    assert r.rubric_score == pytest.approx(0.0)


def test_rubric_degenerate_gate_is_unevaluable_not_a_violation():
    """issue #155 regression: livingroom_1 leg 1's exact diagnosed geometry. The
    driven trajectory never came near the degenerate 1.3cm sliver (closest
    approach 1.08m in the live run) -- with the fix, this must NOT be recorded as
    a threading violation (no robot could ever have threaded a non-existent gap),
    but the case must still be visible via ``n_threading_unevaluable`` rather than
    silently vanishing."""
    from core.geometry.toolbox import corridor_gate

    sofa_min = np.array([-2.588, -3.704, 0.0])
    sofa_max = np.array([-0.704, -0.636, 1.0])
    table_min = np.array([-0.717, -2.833, 0.0])
    table_max = np.array([0.084, -2.032, 1.0])
    sofa = _rec(0, "sofa", *((sofa_min + sofa_max)[:2] / 2.0))
    table = _rec(60, "round table", *((table_min + table_max)[:2] / 2.0))
    sofa.aabb_min, sofa.aabb_max = sofa_min, sofa_max
    table.aabb_min, table.aabb_max = table_min, table_max
    gate = corridor_gate(sofa, table)
    assert gate.degenerate  # sanity: this is the diagnosed degenerate case

    # Driven trajectory bbox from the tracked repro: x in [-0.93, 1.01], y in
    # [-5.01, 0.55] -- reaches the room but never within ~1m of the gate sliver.
    traj = np.array([[0.0, 0.55], [0.0, -2.5], [0.0, -5.0]], dtype=float)
    goals = [("corridor_between", (float(gate.midpoint[0]), float(gate.midpoint[1])))]
    r = score_instruction_rubric(traj, goals, corridor_gates=[(0, gate)], tol=10.0)
    assert r.n_threading_violations == 0
    assert r.n_threading_unevaluable == 1
    assert any("degenerate" in d for d in r.threading_details)
    # not scored as a threading penalty: only ordered-arrival credit applies.
    assert r.penalty == pytest.approx(0.0)


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
    """The constant-speed follower actually drives through both ordered leg goals.

    The table's footprint is a perfect square about its own centroid, so
    ``_nearest_free_goal`` (issue #66: now pushes a leg goal off its OWN anchor's
    footprint too, same as the real ``_goto_point``/``_via_point`` navigation) picks
    its nearest edge by an exact floating-point tie across all four sides — here it
    resolves to the SOUTH edge (verified directly). The pole (leg 1) sits due south
    of the table so leg 2's straight approach lands on that same south edge, keeping
    the rubric's pushed goal and the real driven path in agreement.
    """
    insts = [_rec(0, "pole", 6.0, -2.0), _rec(1, "table", 6.0, 1.0)]
    gt = GTScene(scene_name="t", instances=insts, regions=[])
    idx = BasicSceneIndex(insts)
    q = "First go to the pole, then go to the table."
    leg_goals, gates, caps, _, aabbs = B._if_rubric_geometry(q, gt, idx)
    assert [k for k, _ in leg_goals] == ["goto", "goto"]
    traj = B._drive_if_trajectory(q, gt, idx, start_xy=(6.0, -2.5))
    assert traj.shape[0] > 2  # a real motion stream, not a point
    r = score_instruction_rubric(
        traj, leg_goals, corridor_gates=gates, avoid_capsules=caps,
        leg_instance_aabbs=aabbs,
    )
    assert r.n_legs_reached_in_order == 2
    assert r.rubric_score == pytest.approx(1.0)


# --------------------------------------------------------- stop vs pass-by (issue #70)


def _aabb(cx: float, cy: float, hx: float, hy: float):
    """A synthetic (aabb_min, aabb_max) footprint centred at (cx, cy)."""
    return (
        np.array([cx - hx, cy - hy, 0.0]),
        np.array([cx + hx, cy + hy, 0.5]),
    )


def test_is_pass_by_leg_classification():
    # via_near is ALWAYS pass-by, any position.
    assert _is_pass_by_leg("via_near", 0, 1) is True
    assert _is_pass_by_leg("via_near", 2, 3) is True  # even the last leg
    # goto is pass-by unless it is the route's LAST leg.
    assert _is_pass_by_leg("goto", 0, 3) is True
    assert _is_pass_by_leg("goto", 1, 3) is True
    assert _is_pass_by_leg("goto", 2, 3) is False  # terminal goto -> stop
    assert _is_pass_by_leg("goto", 0, 1) is False  # single-leg route -> terminal
    # corridor_between is always stop (out of scope for this issue's semantics).
    assert _is_pass_by_leg("corridor_between", 0, 3) is False
    assert _is_pass_by_leg("corridor_between", 2, 3) is False


def test_pass_by_leg_widens_tolerance_by_instance_half_diagonal():
    # Non-terminal goto (leg 1 of 3) at (5, 0) with a 2x2 m footprint (half-diag
    # sqrt(8)/2 ~= 1.414 m). Base pass_by_tol=0.5 m; the driven path passes 1.5 m off
    # the goal -- inside the widened pass-by radius (1.914 m) but well outside the
    # bare tol. Only widening the tolerance (not moving the goal point) explains a
    # hit. pass_by_tol is passed explicitly (issue #100: it is a SEPARATE parameter
    # from tol, not the same number reused) -- tol is left at its default and must
    # play no part in this leg's tolerance.
    aabb = _aabb(5.0, 0.0, 1.0, 1.0)
    leg_goals = [
        ("goto", (0.0, 0.0)),
        ("goto", (5.0, 0.0)),  # pass-by (non-terminal)
        ("goto", (10.0, 0.0)),  # stop (terminal)
    ]
    aabbs = [None, aabb, None]
    traj = np.array([[0, 0], [5, 1.5], [10, 0]], dtype=float)
    r = score_instruction_rubric(
        traj, leg_goals, pass_by_tol=0.5, leg_instance_aabbs=aabbs
    )
    leg1 = r.leg_outcomes[1]
    assert leg1.pass_by is True
    assert leg1.tol_used == pytest.approx(math.hypot(2.0, 2.0) / 2.0 + 0.5)
    assert leg1.reached is True


def test_stop_leg_ignores_instance_aabb_even_when_supplied():
    # Same geometry as above but the leg is the route's ONLY (=terminal) leg -- a
    # stop leg must NOT get the pass-by widening even if an AABB is supplied.
    aabb = _aabb(5.0, 0.0, 1.0, 1.0)
    leg_goals = [("goto", (5.0, 0.0))]
    traj = np.array([[5.0, 1.5]], dtype=float)
    r = score_instruction_rubric(traj, leg_goals, tol=0.5, leg_instance_aabbs=[aabb])
    leg0 = r.leg_outcomes[0]
    assert leg0.pass_by is False
    assert leg0.tol_used == pytest.approx(0.5)
    assert leg0.reached is False  # 1.5 m away, outside the bare 0.5 m stop tolerance


def test_pass_by_leg_without_aabb_falls_back_to_plain_pass_by_tol():
    # No leg_instance_aabbs supplied at all (old-caller compatibility) -- a pass-by
    # leg must behave exactly like today: plain pass_by_tol, no widening. Issue #100:
    # the fallback is pass_by_tol, NOT tol -- tol is set to a different value here to
    # prove it plays no part.
    leg_goals = [("goto", (0.0, 0.0)), ("goto", (5.0, 0.0))]
    traj = np.array([[0, 0], [5, 1.5]], dtype=float)
    r = score_instruction_rubric(
        traj, leg_goals, tol=99.0, pass_by_tol=0.5
    )  # no leg_instance_aabbs
    leg0 = r.leg_outcomes[0]
    assert leg0.pass_by is True  # non-terminal goto -- still classified pass-by
    assert leg0.tol_used == pytest.approx(0.5)  # but no AABB -> no widening
    assert leg0.reached is True  # (0,0) is hit exactly regardless


def test_via_near_pass_by_even_as_terminal_leg():
    # via_near stays pass-by even when it is the LAST leg (unlike goto).
    aabb = _aabb(3.0, 0.0, 0.5, 0.5)  # half-diag sqrt(0.5) ~= 0.707
    leg_goals = [("via_near", (3.0, 0.0))]
    traj = np.array([[3.0, 0.8]], dtype=float)  # 0.8 m off, outside bare 0.3 tol
    r = score_instruction_rubric(
        traj, leg_goals, pass_by_tol=0.3, leg_instance_aabbs=[aabb]
    )
    leg0 = r.leg_outcomes[0]
    assert leg0.pass_by is True
    assert leg0.tol_used == pytest.approx(math.hypot(1.0, 1.0) / 2.0 + 0.3)
    assert leg0.reached is True


def test_pass_by_default_tol_no_longer_credits_a_1_5m_miss():
    # Issue #100 regression: livingroom_4 leg1 shape from the offline battery (a
    # non-terminal goto with half-diag ~0.4999 m, driven closest-approach 1.5469 m)
    # used to be credited under the OLD tol reuse (half_diag + LEG_ARRIVAL_TOL_M ~=
    # 1.746 -> tol_used 2.2462, comfortably > 1.5469). Under the new default
    # pass_by_tol it must NOT be credited.
    from core.groundtruth.scoring import PASS_BY_ARRIVAL_TOL_M

    aabb = _aabb(5.0, 0.0, 0.3536, 0.3536)  # half-diag ~= 0.5
    leg_goals = [
        ("goto", (0.0, 0.0)),
        ("goto", (5.0, 0.0)),  # pass-by (non-terminal)
        ("goto", (20.0, 0.0)),  # stop (terminal) -- never reached, irrelevant here
    ]
    aabbs = [None, aabb, None]
    # An out-and-back path held at a fixed y-offset (1.5469 m) as it passes x=5: its
    # closest approach to the goal (5, 0) is EXACTLY 1.5469 m, at the vertex, with no
    # nearer point elsewhere on the (densified) path -- a diagonal straight shot would
    # interpolate a nearer closest-approach than the intended one and contaminate
    # the test.
    traj = np.array([[10.0, 1.5469], [5.0, 1.5469], [10.0, 1.5469]], dtype=float)
    r = score_instruction_rubric(traj, leg_goals, leg_instance_aabbs=aabbs)
    leg1 = r.leg_outcomes[1]
    assert leg1.pass_by is True
    assert leg1.tol_used == pytest.approx(0.5 + PASS_BY_ARRIVAL_TOL_M, abs=1e-3)
    assert leg1.tol_used < 1.5469  # the miss is no longer inside tolerance
    assert leg1.reached is False


def test_pass_by_default_tol_still_credits_a_sub_1m_arrival():
    # Issue #100: the mass of the credited-leg distribution (offline battery:
    # median ~0.7 m closest-approach for PASS-BY legs) must stay credited under the
    # tightened default -- this is not a blanket tightening, only the outlier tail.
    aabb = _aabb(5.0, 0.0, 0.3536, 0.3536)  # half-diag ~= 0.5, same footprint as above
    leg_goals = [
        ("goto", (0.0, 0.0)),
        ("goto", (5.0, 0.0)),  # pass-by (non-terminal)
        ("goto", (20.0, 0.0)),  # stop (terminal) -- never reached, irrelevant here
    ]
    # Same out-and-back construction as the miss test above, at 0.95 m instead.
    traj = np.array([[10.0, 0.95], [5.0, 0.95], [10.0, 0.95]], dtype=float)
    r = score_instruction_rubric(traj, leg_goals, leg_instance_aabbs=[None, aabb, None])
    leg1 = r.leg_outcomes[1]
    assert leg1.pass_by is True
    assert leg1.reached is True


def test_corridor_between_leg_stays_stop_tolerance_with_ordering_intact():
    # corridor_between (leg 0) is a stop leg; the ordered-arrival cursor logic must
    # still enforce sequence across a mix of pass-by (leg 1, non-terminal goto) and
    # stop (leg 2, terminal goto) legs -- (a) changes only the per-leg RADIUS, never
    # the cursor mechanics.
    leg_goals = [
        ("corridor_between", (0.0, 0.0)),
        ("goto", (5.0, 0.0)),
        ("goto", (10.0, 0.0)),
    ]
    traj = np.array([[0, 0], [5, 0], [10, 0]], dtype=float)
    r = score_instruction_rubric(traj, leg_goals, tol=0.3)
    assert [o.pass_by for o in r.leg_outcomes] == [False, True, False]
    assert r.n_legs_reached_in_order == 3
    assert r.rubric_score == pytest.approx(1.0)
