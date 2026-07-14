"""Scorer tests: hand-built IoU / Frechet / coverage + real-loft numerical & OR."""
from __future__ import annotations

import numpy as np
import pytest

from core.groundtruth import scoring as S
from core.groundtruth.loader import load_scene
from core.interfaces import MarkerBox
from core.perception.scene_index import BasicSceneIndex

from core.groundtruth.scoring import Frame2D, align_scene_trajectories, fit_frame

from tests.groundtruth.conftest import requires_loft, LOFT_DIR


# --------------------------------------------------------------------------- scene-graph between pairs (T7-S2)


def _sg_with_between_pair():
    """Minimal VLA-3D-shaped scene graph: a chair BETWEEN two tables.

    ``between`` stores each entry as an [id, id] PAIR (T7 #5a): the scorer must
    expand the pair into its member anchor ids, not str()-garble it into "[t1, t2]".
    """
    return {
        "regions": {
            "0": {
                "objects": [
                    {"object_id": "c1", "raw_label": "chair"},
                    {"object_id": "t1", "raw_label": "table"},
                    {"object_id": "t2", "raw_label": "table"},
                ],
                "relationships": {
                    # target c1 is between the pair (t1, t2) -> nested [id, id]
                    "between": {"c1": [["t1", "t2"]]},
                },
            }
        }
    }


def test_scene_graph_between_pair_not_garbled():
    # The chair between two tables must be counted once when the question names the
    # 'table' anchor: the pair members expand to t1/t2 (both label 'table'), so the
    # anchor-agreement check finds a match. If the pair were str()-flattened into a
    # single "['t1', 'table']"-style token the anchor labels would never resolve and
    # the count would collapse to the class-only fallback.
    sg = _sg_with_between_pair()
    n, src = S._scene_graph_count("How many chairs are between the tables?", sg)
    assert n == 1
    assert src == "scene_graph"  # relation-aware, not the class-only fallback


# --------------------------------------------------------------------------- 3D IoU


def test_iou_identical_boxes_is_one():
    lo = np.array([0.0, 0.0, 0.0])
    hi = np.array([2.0, 2.0, 2.0])
    assert S.iou_from_corners(lo, hi, lo, hi) == pytest.approx(1.0)


def test_iou_zero_overlap():
    a_lo, a_hi = np.array([0.0, 0, 0]), np.array([1.0, 1, 1])
    b_lo, b_hi = np.array([5.0, 5, 5]), np.array([6.0, 6, 6])
    assert S.iou_from_corners(a_lo, a_hi, b_lo, b_hi) == 0.0


def test_iou_half_overlap():
    # two unit cubes overlapping in exactly half their x-extent
    a_lo, a_hi = np.array([0.0, 0, 0]), np.array([2.0, 1, 1])
    b_lo, b_hi = np.array([1.0, 0, 0]), np.array([3.0, 1, 1])
    # inter = 1*1*1 = 1; union = 2 + 2 - 1 = 3
    assert S.iou_from_corners(a_lo, a_hi, b_lo, b_hi) == pytest.approx(1.0 / 3.0)


def test_iou_from_markerbox_matches_corners():
    box = MarkerBox(cx=1.0, cy=1.0, cz=1.0, sx=2.0, sy=2.0, sz=2.0)
    lo = np.array([0.0, 0, 0])
    hi = np.array([2.0, 2, 2])
    assert S.aabb_iou_3d(box, lo, hi) == pytest.approx(1.0)


# --------------------------------------------------------------------------- Frechet


def test_frechet_identical_paths_zero():
    p = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    assert S.discrete_frechet(p, p) == pytest.approx(0.0)


def test_frechet_parallel_offset_paths():
    a = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    b = np.array([[0.0, 1.0], [1.0, 1.0], [2.0, 1.0]])  # constant 1 m above
    assert S.discrete_frechet(a, b) == pytest.approx(1.0)


def test_frechet_empty_is_inf():
    a = np.array([[0.0, 0.0]])
    assert S.discrete_frechet(a, np.empty((0, 2))) == float("inf")


def test_frechet_symmetric():
    a = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0]])
    b = np.array([[0.0, 0.5], [1.0, 0.0], [2.0, 1.5]])
    assert S.discrete_frechet(a, b) == pytest.approx(S.discrete_frechet(b, a))


# --------------------------------------------------------------------------- coverage


def test_coverage_full_when_paths_coincide():
    gt = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    assert S.path_coverage(gt, gt) == pytest.approx(1.0)


def test_coverage_partial_within_radius():
    gt = np.array([[0.0, 0.0], [10.0, 0.0]])  # one near ours, one far
    ours = np.array([[0.0, 0.5]])  # 0.5 m from first, 10 m from second
    assert S.path_coverage(gt, ours, radius=1.0) == pytest.approx(0.5)


def test_coverage_zero_when_our_path_empty():
    gt = np.array([[0.0, 0.0], [1.0, 0.0]])
    assert S.path_coverage(gt, np.empty((0, 2))) == 0.0


# --------------------------------------------------------------------------- PLY parse


@requires_loft
def test_load_trajectory_ply_real_file():
    p = LOFT_DIR.parent.parent.parent / "CMU-VLN-Challenge-2026" / "questions" / "loft" / "trajectory_q4.ply"
    if not p.exists():
        pytest.skip("loft trajectory not present")
    pts = S.load_trajectory_ply(p)
    assert pts.ndim == 2 and pts.shape[1] == 3
    assert pts.shape[0] == 739  # header 'element vertex 739'
    assert np.allclose(pts[:, 2], 0.75)  # fixed-height path


# --------------------------------------------------------------------------- numerical


@requires_loft
def test_numerical_pipeline_count_deterministic(loft_referential):
    scene = load_scene(LOFT_DIR)
    idx = BasicSceneIndex(scene.instances)
    q = "How many black pillows are on the sofa?"
    a = S.score_numerical(q, idx, referential=loft_referential)
    b = S.score_numerical(q, idx, referential=loft_referential)
    assert a.our_count == b.our_count  # deterministic
    assert a.exact_match is True  # our_count == pipeline gt by construction


@requires_loft
def test_numerical_independent_second_opinion_present(loft_referential):
    scene = load_scene(LOFT_DIR)
    idx = BasicSceneIndex(scene.instances)
    a = S.score_numerical("How many black pillows are on the sofa?", idx,
                          referential=loft_referential)
    # loft referential annotates 9 distinct pillow instances (relation-agnostic)
    assert a.gt_count_independent == 9
    assert a.independent_source in ("referential", "referential_class_only")


@requires_loft
def test_numerical_no_referential_yields_no_second_opinion():
    scene = load_scene(LOFT_DIR)
    idx = BasicSceneIndex(scene.instances)
    a = S.score_numerical("How many chairs are there?", idx, referential=None)
    assert a.gt_count_independent is None
    assert a.independent_source == "none"


# --------------------------------------------------------------------------- object ref


@requires_loft
def test_object_reference_produces_marker_or_flags(loft_referential):
    scene = load_scene(LOFT_DIR)
    idx = BasicSceneIndex(scene.instances)
    q = "The blue chair that is closest to the cup of coffee."
    r = S.score_object_reference(q, idx, scene.instances, referential=loft_referential)
    # Our resolver finds a chair candidate (a marker is produced)…
    assert r.our_marker is not None
    assert r.our_marker.label == "chair"
    # …but when the GT target can't be matched we FLAG rather than guess (IoU nan).
    if r.gt_target_id is None:
        assert np.isnan(r.iou)
        assert r.target_source in ("none", "ambiguous")


@requires_loft
def test_object_reference_unique_in_scene_target_scores_iou():
    """A scene-unique target class needs no disambiguation: it IS the GT target."""
    scene = load_scene(LOFT_DIR)
    idx = BasicSceneIndex(scene.instances)
    # 'fireplace' is a single instance in loft -> unique-in-scene GT target.
    r = S.score_object_reference("Find the fireplace.", idx, scene.instances,
                                 referential=None)
    assert r.target_source == "unique_in_scene"
    assert r.gt_target_id is not None
    assert r.iou == pytest.approx(1.0)  # our pick == the unique instance


@requires_loft
def test_object_reference_iou_in_unit_range_when_scored(loft_referential):
    scene = load_scene(LOFT_DIR)
    idx = BasicSceneIndex(scene.instances)
    for q in [
        "The blue chair that is closest to the cup of coffee.",
        "Find the potted plant between a vase and the cabinet with a TV on it.",
    ]:
        r = S.score_object_reference(q, idx, scene.instances, referential=loft_referential)
        if r.gt_target_id is not None and not np.isnan(r.iou):
            assert 0.0 <= r.iou <= 1.0


# --------------------------------------------------------------------------- IF scorer


def test_instruction_score_flags_frame_offset(tmp_path):
    """A GT path far from ours (frame offset) is flagged, numbers still reported."""
    ply = tmp_path / "traj.ply"
    ply.write_text(
        "ply\nformat ascii 1.0\nelement vertex 2\nproperty float x\n"
        "property float y\nproperty float z\nend_header\n"
        "100.0 100.0 0.75\n101.0 100.0 0.75\n",
        encoding="utf-8",
    )
    ours = np.array([[0.0, 0.0], [1.0, 0.0]])
    sc = S.score_instruction_following(ours, ply)
    assert sc.frame_aligned is False
    assert sc.gt_n_waypoints == 2
    assert sc.our_n_waypoints == 2
    assert np.isfinite(sc.frechet_m)


def test_instruction_score_aligned_paths(tmp_path):
    ply = tmp_path / "traj.ply"
    ply.write_text(
        "ply\nformat ascii 1.0\nelement vertex 2\nproperty float x\n"
        "property float y\nproperty float z\nend_header\n"
        "0.0 0.0 0.75\n1.0 0.0 0.75\n",
        encoding="utf-8",
    )
    ours = np.array([[0.0, 0.0], [1.0, 0.0]])
    sc = S.score_instruction_following(ours, ply)
    assert sc.frame_aligned is True
    assert sc.coverage_1m == pytest.approx(1.0)
    assert sc.frechet_m == pytest.approx(0.0)


# --------------------------------------------------------------------------- frame fit


def test_fit_frame_pure_translation():
    """A shifted-but-unrotated correspondence recovers the translation, yaw ~ 0."""
    src = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 3.0]])
    shift = np.array([5.0, -1.5])
    dst = src + shift
    frame, res = fit_frame(src, dst)
    assert res == pytest.approx(0.0, abs=1e-9)
    assert frame.theta == pytest.approx(0.0, abs=1e-9)
    assert np.allclose(frame.t, shift)
    assert np.allclose(frame.apply(src), dst)


def test_fit_frame_recovers_yaw():
    """A rotated+translated correspondence needs yaw; translation-only can't fit it."""
    theta = np.deg2rad(35.0)
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array([[c, -s], [s, c]])
    t = np.array([3.0, -2.0])
    src = np.array([[0.0, 0.0], [4.0, 0.0], [4.0, 2.0], [1.0, 3.0]])
    dst = src @ rot.T + t
    frame, res = fit_frame(src, dst, yaw_residual_gate=0.5)
    assert res == pytest.approx(0.0, abs=1e-6)
    assert frame.theta == pytest.approx(theta, abs=1e-6)
    assert np.allclose(frame.apply(src), dst, atol=1e-6)


def test_fit_frame_translation_only_when_within_gate():
    """When translation residual is under the gate, yaw is NOT introduced."""
    src = np.array([[0.0, 0.0], [3.0, 0.0]])
    dst = src + np.array([1.0, 1.0])
    frame, res = fit_frame(src, dst, yaw_residual_gate=0.5)
    assert frame.theta == pytest.approx(0.0)  # stayed translation-only
    assert res == pytest.approx(0.0, abs=1e-9)


def test_align_scene_two_endpoints_maps_ends_to_goals():
    """Two trajectory endpoints -> two goal centroids fits one scene transform."""
    # sim-frame trajectories sharing a start at origin
    t4 = np.array([[0.0, 0.0, 0.75], [3.0, 0.0, 0.75], [6.0, 1.0, 0.75]])
    t5 = np.array([[0.0, 0.0, 0.75], [1.0, -2.0, 0.75], [7.0, -1.0, 0.75]])
    # object-frame goals = sim endpoints shifted by a known translation
    shift = np.array([-0.15, -0.70])
    goal4 = t4[-1, :2] + shift
    goal5 = t5[-1, :2] + shift
    frame, res = align_scene_trajectories([(t4, goal4), (t5, goal5)])
    assert res == pytest.approx(0.0, abs=1e-9)
    assert np.allclose(frame.apply(t4[-1:])[0], goal4)
    assert np.allclose(frame.apply(t5[-1:])[0], goal5)
    # shared start maps to a single spawn point in the object frame
    s4 = frame.apply(t4[:1])[0]
    s5 = frame.apply(t5[:1])[0]
    assert np.allclose(s4, s5)


def test_align_scene_single_endpoint_translation_only():
    """One usable endpoint => translation-only fit (yaw unidentifiable)."""
    t = np.array([[0.0, 0.0, 0.75], [5.0, 0.0, 0.75]])
    goal = np.array([5.0, 2.0])  # shift (0, 2)
    frame, res = align_scene_trajectories([(t, goal), (t, None)])
    assert frame.theta == pytest.approx(0.0)
    assert res == pytest.approx(0.0, abs=1e-9)
    assert np.allclose(frame.t, np.array([0.0, 2.0]))


def test_align_scene_no_goals_returns_none():
    t = np.array([[0.0, 0.0, 0.75], [5.0, 0.0, 0.75]])
    frame, res = align_scene_trajectories([(t, None)])
    assert frame is None and res is None


def test_score_if_with_frame_transforms_gt(tmp_path):
    """A fitted frame maps the GT path into our frame before scoring (coverage rises)."""
    # GT trajectory in 'sim' frame offset by (10, 10); our path near origin.
    ply = tmp_path / "traj.ply"
    ply.write_text(
        "ply\nformat ascii 1.0\nelement vertex 2\nproperty float x\n"
        "property float y\nproperty float z\nend_header\n"
        "10.0 10.0 0.75\n11.0 10.0 0.75\n",
        encoding="utf-8",
    )
    ours = np.array([[0.0, 0.0], [1.0, 0.0]])
    # frame that subtracts (10, 10): maps GT sim -> our frame exactly onto ours.
    frame = Frame2D(theta=0.0, t=np.array([-10.0, -10.0]))
    sc = S.score_instruction_following(ours, ply, frame=frame, fit_residual_m=0.1)
    assert sc.frame_aligned is True
    assert sc.coverage_1m == pytest.approx(1.0)
    assert sc.frechet_m == pytest.approx(0.0, abs=1e-9)


def test_score_if_high_residual_marks_unaligned(tmp_path):
    ply = tmp_path / "traj.ply"
    ply.write_text(
        "ply\nformat ascii 1.0\nelement vertex 2\nproperty float x\n"
        "property float y\nproperty float z\nend_header\n"
        "0.0 0.0 0.75\n1.0 0.0 0.75\n",
        encoding="utf-8",
    )
    ours = np.array([[0.0, 0.0], [1.0, 0.0]])
    frame = Frame2D(theta=0.0, t=np.array([0.0, 0.0]))
    sc = S.score_instruction_following(ours, ply, frame=frame, fit_residual_m=2.5)
    assert sc.frame_aligned is False  # residual 2.5 m > 1.0 m gate
    assert sc.fit_residual_m == 2.5


# --------------------------------------------------------------------------- OR matching


@requires_loft
def test_or_exact_statement_match(loft_referential):
    """A question equal (up to punctuation/case) to a statement matches exactly."""
    scene = load_scene(LOFT_DIR)
    idx = BasicSceneIndex(scene.instances)
    # take a real statement verbatim and pose it as a question
    stmt = None
    for _rid, stmts in loft_referential["regions"].items():
        for s, anns in stmts.items():
            if isinstance(anns, list) and anns and "plant" in s and "between" in s:
                stmt = s
                break
        if stmt:
            break
    assert stmt is not None
    r = S.score_object_reference(stmt + ".", idx, scene.instances,
                                 referential=loft_referential)
    assert r.match_method == "exact"
    assert r.gt_target_id is not None


@requires_loft
def test_or_relation_fuzzy_match_potted_plant(loft_referential):
    """The 'potted plant between a vase and the cabinet' question matches by relation."""
    scene = load_scene(LOFT_DIR)
    idx = BasicSceneIndex(scene.instances)
    q = "Find the potted plant between a vase and the cabinet with a TV on it."
    r = S.score_object_reference(q, idx, scene.instances, referential=loft_referential)
    assert r.gt_target_id is not None
    assert r.match_method in ("fuzzy", "relation")
    assert not np.isnan(r.iou)


@requires_loft
def test_or_ordinal_phrasing_tiebreak(loft_referential):
    """When 'closest'/'second closest' share the relation string, phrasing picks the
    literal 'closest'. Uses a synthetic referential set to isolate the tie-break."""
    ref = {
        "regions": {
            "0": {
                "the vase that is closest to the guitar": [
                    {"target_index": "5", "target_class": "vase", "relation": "closest",
                     "anchors": {"anchor_1": {"class": "guitar"}}},
                ],
                "the vase that is second closest to the guitar": [
                    {"target_index": "6", "target_class": "vase", "relation": "closest",
                     "anchors": {"anchor_1": {"class": "guitar"}}},
                ],
            }
        }
    }
    tid, source, method = S._gt_target_from_referential(
        "Find the vase closest to the guitar.", ref, []
    )
    assert tid == 5  # the literal 'closest', not the 'second closest'
    assert method == "relation"


@requires_loft
def test_or_no_match_flagged_not_guessed(loft_referential):
    """The 'blue chair closest to the cup of coffee' has no statement -> none, flagged."""
    scene = load_scene(LOFT_DIR)
    idx = BasicSceneIndex(scene.instances)
    q = "The blue chair that is closest to the cup of coffee."
    r = S.score_object_reference(q, idx, scene.instances, referential=loft_referential)
    assert r.match_method == "none"
    assert r.gt_target_id is None
    assert np.isnan(r.iou)


# --------------------------------------------------------------------------- 3rd opinion


@requires_loft
def test_numerical_scenegraph_third_opinion(loft_referential, loft_scene_graph):
    """Scene-graph relation count is reported as a distinct third opinion."""
    scene = load_scene(LOFT_DIR)
    idx = BasicSceneIndex(scene.instances)
    a = S.score_numerical("How many black pillows are on the sofa?", idx,
                          referential=loft_referential, scene_graph=loft_scene_graph)
    assert a.gt_count_scenegraph is not None
    assert a.scenegraph_source in ("scene_graph", "scene_graph_class_only")
    # three opinions visible; disagreement surfaced in the note
    assert a.gt_count_pipeline is not None
    assert a.gt_count_independent is not None
    if len({a.gt_count_pipeline, a.gt_count_independent, a.gt_count_scenegraph}) > 1:
        assert "disagreement" in a.note


@requires_loft
def test_numerical_no_scenegraph_no_third_opinion(loft_referential):
    scene = load_scene(LOFT_DIR)
    idx = BasicSceneIndex(scene.instances)
    a = S.score_numerical("How many black pillows are on the sofa?", idx,
                          referential=loft_referential, scene_graph=None)
    assert a.gt_count_scenegraph is None
    assert a.scenegraph_source == "none"
