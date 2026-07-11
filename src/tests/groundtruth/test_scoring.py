"""Scorer tests: hand-built IoU / Frechet / coverage + real-loft numerical & OR."""
from __future__ import annotations

import numpy as np
import pytest

from core.groundtruth import scoring as S
from core.groundtruth.loader import load_scene
from core.interfaces import MarkerBox
from core.perception.scene_index import BasicSceneIndex

from tests.groundtruth.conftest import requires_loft, LOFT_DIR


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
