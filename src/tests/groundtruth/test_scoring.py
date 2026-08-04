"""Scorer tests: hand-built IoU / Frechet / coverage + real-loft numerical & OR."""
from __future__ import annotations

import numpy as np
import pytest

from core.groundtruth import scoring as S
from core.groundtruth.loader import load_scene
from core.interfaces import InstanceRecord, MarkerBox
from core.perception.scene_index import BasicSceneIndex

from core.groundtruth.scoring import Frame2D, align_scene_trajectories, fit_frame

from tests.groundtruth.conftest import (
    requires_loft,
    requires_full_unity,
    FULL_UNITY_ROOT,
    LOFT_DIR,
)


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
    tid, source, method, _amb = S._gt_target_from_referential(
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


def test_or_superlative_requires_superlative_statement():
    """A superlative question does NOT match a merely-related statement (#20).

    "the speaker ... closest to the potted plant" must be validated by a genuine
    "closest to (potted) plant" statement — a "near the small cabinet" statement (same
    target class, generically-related, head-noun-overlapping anchor) is NOT evidence for
    it. With no speaker-to-plant statement present, the honest result is no match.
    """
    ref = {
        "regions": {
            "0": {
                # generically related, wrong relation + wrong anchor (bare head-noun)
                "the black speaker that is near the small cabinet": [
                    {"target_index": "91", "target_class": "speaker", "relation": "near",
                     "anchors": {"anchor_1": {"class": "cabinet"}}},
                ],
                # superlative, but ranked against an unrelated anchor
                "the speaker that is closest to the ashtray": [
                    {"target_index": "108", "target_class": "speaker", "relation": "closest",
                     "anchors": {"anchor_1": {"class": "ashtray"}}},
                ],
            }
        }
    }
    q = "Find the speaker on the TV cabinet closest to the potted plant on the TV cabinet."
    tid, source, method, _amb = S._gt_target_from_referential(q, ref, [])
    assert tid is None
    assert method == "none"


def test_or_superlative_genuine_statement_still_matches():
    """Control: a genuine superlative-bearing statement still matches under the gate.

    The superlative-aware guard must not over-reject: when a statement DOES express the
    question's superlative against the question's anchor, it is accepted as before.
    """
    ref = {
        "regions": {
            "0": {
                "the vase that is closest to the guitar": [
                    {"target_index": "46", "target_class": "vase", "relation": "closest",
                     "anchors": {"anchor_1": {"class": "guitar"}}},
                ],
                # distractor: same class, different (non-superlative) relation/anchor
                "the vase that is on the shelf": [
                    {"target_index": "50", "target_class": "vase", "relation": "on",
                     "anchors": {"anchor_1": {"class": "shelf"}}},
                ],
            }
        }
    }
    tid, source, method, _amb = S._gt_target_from_referential(
        "Find the vase closest to the guitar.", ref, []
    )
    assert tid == 46
    assert method == "relation"


# --------------------------------------------------------------------------- 3rd opinion


# --------------------------------------------------------------------------- nested
# disambiguator recursion (issue #95): an anchor named only inside another anchor's
# ``disambiguator`` (e.g. "... the table THAT IS closest to the folding screen") must
# still be visible to every ``anchor_nouns`` set these scorers build, not just the
# top-level clause anchors.


def test_independent_count_sees_nested_disambiguator_anchor():
    """Statement anchors only agree with the NESTED anchor ('folding screen'), not the
    top-level one ('table') -> recursion is required for the relation-aware (finer)
    count instead of the class-only (coarser) fallback."""
    ref = {
        "regions": {
            "0": {
                "stmt": [
                    {
                        "target_index": "7",
                        "target_class": "bowl",
                        "anchors": {"anchor_1": {"class": "folding screen"}},
                    },
                ]
            }
        }
    }
    q = "How many bowls are on the table closest to the folding screen?"
    n, source = S._independent_count(q, ref)
    assert n == 1
    assert source == "referential"  # not "referential_class_only"


def test_scene_graph_count_sees_nested_disambiguator_anchor():
    """Scene-graph edge only agrees with the nested anchor -> recursion required for
    the relation-aware (finer) count instead of the class-only fallback."""
    sg = {
        "regions": {
            "0": {
                "objects": [
                    {"object_id": "b1", "raw_label": "bowl"},
                    {"object_id": "t1", "raw_label": "table"},
                    {"object_id": "fs1", "raw_label": "folding screen"},
                ],
                "relationships": {"on": {"b1": ["fs1"]}},
            }
        }
    }
    q = "How many bowls are on the table closest to the folding screen?"
    n, source = S._scene_graph_count(q, sg)
    assert n == 1
    assert source == "scene_graph"  # not "scene_graph_class_only"


def test_gt_target_relation_match_via_nested_disambiguator_anchor():
    """Non-superlative top clause ('on the table') with a nested disambiguator
    ('that is closest to the folding screen') — the referential statement's anchor
    only agrees with the NESTED noun, so the relation match requires recursion."""
    ref = {
        "regions": {
            "0": {
                "stmt1": [
                    {
                        "target_index": "9",
                        "target_class": "bowl",
                        "relation": "on",
                        "anchors": {"anchor_1": {"class": "folding screen"}},
                    },
                ]
            }
        }
    }
    q = "Find the bowl on the table that is closest to the folding screen."
    tid, source, method, _amb = S._gt_target_from_referential(q, ref, [])
    assert tid == 9
    assert method == "relation"


def test_gt_target_superlative_match_via_nested_disambiguator_anchor():
    """Superlative top clause ('closest to the book') where 'book' itself carries a
    nested disambiguator ('on the stool') — the referential statement's anchor only
    agrees with the NESTED noun 'stool', so the superlative-restricted anchor filter
    requires recursion into the superlative clause's own anchors."""
    ref = {
        "regions": {
            "0": {
                "stmt1": [
                    {
                        "target_index": "12",
                        "target_class": "pillow",
                        "relation": "closest",
                        "anchors": {"anchor_1": {"class": "stool"}},
                    },
                ]
            }
        }
    }
    q = "Find the pillow closest to the book on the stool."
    tid, source, method, _amb = S._gt_target_from_referential(q, ref, [])
    assert tid == 12
    assert method == "relation"


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


# --------------------------------------------------------------------------- length-
# scaled tie-break margin (issue #92 fix 2): a FIXED absolute Jaccard floor rejects
# genuine winners on long statements (more tokens -> smaller absolute delta per
# discriminating token) while a length-scaled floor accepts them without opening the
# door to genuine ties (which sit at margin 0 regardless of length).


@requires_full_unity
def test_or_margin_scales_with_statement_length_paper_cup():
    """office_1 'paper cup on the table closest to the projector screen' (issue #92):
    the correct match (id 84, 'the cup that is closest to the projector screen', a
    12-token question) scores 0.583 against a 'second closest' runner-up at 0.538 —
    a 0.045 absolute margin that the OLD fixed 0.05 floor rejected outright. The
    length-scaled margin must accept it."""
    scene_dir = FULL_UNITY_ROOT / "office_1"
    referential = __import__("json").loads(
        (scene_dir / "office_1_referential_statements.json").read_text()
    )
    scene = load_scene(scene_dir)
    idx = BasicSceneIndex(scene.instances)
    q = "Find the paper cup on the table closest to the projector screen."
    r = S.score_object_reference(q, idx, scene.instances, referential=referential)
    assert r.match_method == "relation"
    assert r.target_source == "referential"
    assert r.gt_target_id == 84
    assert not np.isnan(r.iou)


@requires_full_unity
def test_or_genuine_tie_still_ambiguous_livingroom1_pillow():
    """Guard: office_1's neighbour scene livingroom_1 has a genuine wording tie —
    'the olive pillow that is on the sofa' vs 'the maroon pillow that is on the
    sofa' both score identically (0.545) against 'Find the pillow on the sofa that
    is closest to the windows.' — no discriminating token exists (the colour words
    aren't in the question either way), so this must stay ambiguous under the new
    length-scaled margin exactly as it did under the old fixed one."""
    scene_dir = FULL_UNITY_ROOT / "livingroom_1"
    referential = __import__("json").loads(
        (scene_dir / "livingroom_1_referential_statements.json").read_text()
    )
    q = "Find the pillow on the sofa that is closest to the windows."
    tid, source, method, _amb = S._gt_target_from_referential(q, referential, [])
    assert tid is None
    assert source == "ambiguous"
    assert method == "none"


def test_or_margin_synthetic_one_token_ordinal_beats_scaled_floor():
    """Synthetic, data-independent version of the paper-cup case: a longer question
    where the winning statement differs from the runner-up by exactly one token
    ('second') must be accepted even though the absolute Jaccard gap is well under
    the old fixed 0.05 floor, because the length-scaled floor shrinks to match."""
    ref = {
        "regions": {
            "0": {
                "the cup that is closest to the printer stand": [
                    {"target_index": "84", "target_class": "cup", "relation": "closest",
                     "anchors": {"anchor_1": {"class": "printer stand"}}},
                ],
                "the cup that is second closest to the printer stand": [
                    {"target_index": "85", "target_class": "cup", "relation": "closest",
                     "anchors": {"anchor_1": {"class": "printer stand"}}},
                ],
            }
        }
    }
    q = "Find the paper cup on the table closest to the printer stand."
    tid, source, method, _amb = S._gt_target_from_referential(q, ref, [])
    assert tid == 84
    assert method == "relation"


def test_or_margin_synthetic_genuine_tie_stays_ambiguous():
    """Synthetic control: two equally-worded, differently-coloured distractors with
    NO discriminating token vs the question tie at margin 0 and must stay
    ambiguous — the length-scaled floor never drops to (or below) zero."""
    ref = {
        "regions": {
            "0": {
                "the olive pillow that is on the sofa": [
                    {"target_index": "41", "target_class": "pillow", "relation": "on",
                     "anchors": {"anchor_1": {"class": "sofa"}}},
                ],
                "the maroon pillow that is on the sofa": [
                    {"target_index": "42", "target_class": "pillow", "relation": "on",
                     "anchors": {"anchor_1": {"class": "sofa"}}},
                ],
            }
        }
    }
    q = "Find the pillow on the sofa that is closest to the windows."
    tid, source, method, _amb = S._gt_target_from_referential(q, ref, [])
    assert tid is None
    assert source == "ambiguous"
    assert method == "none"


# --------------------------------------------------------------------------- geometry
# fallback GT target (issue #92 fix 3): when the referential-statement ladder finds
# ZERO candidates for a single-anchor physical relation, verify it directly against
# the GT geometry instead of leaving the question unscored.


def _box(iid: int, label: str, xmin, xmax, ymin, ymax, zmin, zmax) -> InstanceRecord:
    amin = np.array([xmin, ymin, zmin])
    amax = np.array([xmax, ymax, zmax])
    return InstanceRecord(
        instance_id=iid, label=label, score=1.0, n_obs=3,
        centroid=(amin + amax) / 2, aabb_min=amin, aabb_max=amax,
        points=None, caption="",
    )


def test_or_geometry_fallback_unique_on_match():
    """No referential statements at all -> the text ladder returns 'none'; a unique
    target-class instance that geometrically satisfies 'on' the unique anchor
    instance is an honest GT target, not a guess."""
    cabinet = _box(1, "file cabinet", 0, 1, 0, 1, 0, 1)
    plant_on = _box(2, "potted plant", 0.3, 0.5, 0.3, 0.5, 1.0, 1.3)
    plant_far = _box(3, "potted plant", 5, 5.3, 5, 5.3, 0, 0.3)
    instances = [cabinet, plant_on, plant_far]
    idx = BasicSceneIndex(instances)
    r = S.score_object_reference(
        "Find the potted plant on the file cabinet.", idx, instances, referential=None,
    )
    assert r.match_method == "geometric"
    assert r.target_source == "geometry"
    assert r.gt_target_id == 2
    assert not np.isnan(r.iou)


def test_or_geometry_fallback_ambiguous_on_stays_none():
    """Guard: two target-class instances both geometrically satisfy the relation ->
    genuinely ambiguous, must NOT guess one -> stays 'none', not fabricated."""
    cabinet = _box(1, "file cabinet", 0, 1, 0, 1, 0, 1)
    plant_a = _box(2, "potted plant", 0.1, 0.3, 0.1, 0.3, 1.0, 1.3)
    plant_b = _box(3, "potted plant", 0.6, 0.8, 0.6, 0.8, 1.0, 1.3)
    instances = [cabinet, plant_a, plant_b]
    idx = BasicSceneIndex(instances)
    r = S.score_object_reference(
        "Find the potted plant on the file cabinet.", idx, instances, referential=None,
    )
    assert r.match_method == "none"
    assert r.gt_target_id is None
    assert np.isnan(r.iou)


def test_or_geometry_fallback_declines_ordinal_relation():
    """Superlative ('closest'/'farthest') questions are declined by the geometry
    fallback even with zero referential candidates -- ranking every same-class
    instance against the anchor would just re-derive our own resolver's answer as
    its own ground truth, not an independent check."""
    cabinet = _box(1, "file cabinet", 0, 1, 0, 1, 0, 1)
    plant_near = _box(2, "potted plant", 1.1, 1.3, 0.4, 0.6, 0, 0.3)
    plant_far = _box(3, "potted plant", 5, 5.3, 5, 5.3, 0, 0.3)
    instances = [cabinet, plant_near, plant_far]
    idx = BasicSceneIndex(instances)
    r = S.score_object_reference(
        "Find the potted plant closest to the file cabinet.",
        idx, instances, referential=None,
    )
    assert r.match_method == "none"
    assert r.gt_target_id is None


def test_or_geometry_fallback_adjudicates_ambiguous_text_result():
    """issue #170: when the text ladder found candidates but couldn't pick one
    ('ambiguous'), the geometry fallback now adjudicates AMONG THOSE candidates —
    here only instance 2 is actually geometrically 'on' the cabinet, so it wins and
    the result is recorded as 'geometry_ambiguous' (distinguishable from a clean
    zero-candidate 'geometry' resolution or a clean text resolution)."""
    cabinet = _box(1, "file cabinet", 0, 1, 0, 1, 0, 1)
    plant_on = _box(2, "potted plant", 0.3, 0.5, 0.3, 0.5, 1.0, 1.3)
    plant_off = _box(3, "potted plant", 5, 5.3, 5, 5.3, 0, 0.3)
    instances = [cabinet, plant_on, plant_off]
    idx = BasicSceneIndex(instances)
    ref = {
        "regions": {
            "0": {
                "the black plant that is near the small cabinet": [
                    {"target_index": "2", "target_class": "plant", "relation": "near",
                     "anchors": {"anchor_1": {"class": "cabinet"}}},
                ],
                "the brown plant that is near the gray cabinet": [
                    {"target_index": "3", "target_class": "plant", "relation": "near",
                     "anchors": {"anchor_1": {"class": "cabinet"}}},
                ],
            }
        }
    }
    r = S.score_object_reference(
        "Find the potted plant on the file cabinet.", idx, instances, referential=ref,
    )
    assert r.target_source == "geometry_ambiguous"
    assert r.match_method == "geometric"
    assert r.gt_target_id == 2
    assert not np.isnan(r.iou)


def test_or_geometry_fallback_still_ambiguous_when_geometry_cant_decide():
    """Guard: an ambiguous text result whose candidates are BOTH geometrically 'on'
    the anchor (so geometry can't break the tie either) must stay honestly
    'ambiguous', not fabricate a winner — same non-guessing guarantee as the
    zero-candidate case, just restricted to the ambiguous candidate set."""
    cabinet = _box(1, "file cabinet", 0, 1, 0, 1, 0, 1)
    plant_a = _box(2, "potted plant", 0.1, 0.3, 0.1, 0.3, 1.0, 1.3)
    plant_b = _box(3, "potted plant", 0.6, 0.8, 0.6, 0.8, 1.0, 1.3)
    instances = [cabinet, plant_a, plant_b]
    idx = BasicSceneIndex(instances)
    ref = {
        "regions": {
            "0": {
                "the black plant that is near the small cabinet": [
                    {"target_index": "2", "target_class": "plant", "relation": "near",
                     "anchors": {"anchor_1": {"class": "cabinet"}}},
                ],
                "the brown plant that is near the gray cabinet": [
                    {"target_index": "3", "target_class": "plant", "relation": "near",
                     "anchors": {"anchor_1": {"class": "cabinet"}}},
                ],
            }
        }
    }
    r = S.score_object_reference(
        "Find the potted plant on the file cabinet.", idx, instances, referential=ref,
    )
    assert r.target_source == "ambiguous"
    assert r.match_method == "none"
    assert r.gt_target_id is None
    assert np.isnan(r.iou)


@requires_full_unity
def test_or_geometry_fallback_real_office1_plant_on_cabinet():
    """office_1 (issue #92 flagship case): 'the potted plant on the file cabinet' has
    zero referential-statement candidates (the corpus's 25 'near'-relation plant
    statements in this scene are all anchored to 'book', and it has zero 'on'
    statements for 'plant' at all) -- but instance 55 IS verifiably 'on' instance 69
    ('file cabinet') by GT geometry, and is the only plant instance that is."""
    scene_dir = FULL_UNITY_ROOT / "office_1"
    scene = load_scene(scene_dir)
    idx = BasicSceneIndex(scene.instances)
    r = S.score_object_reference(
        "Find the potted plant on the file cabinet.",
        idx, scene.instances, referential=None,
    )
    assert r.match_method == "geometric"
    assert r.target_source == "geometry"
    assert r.gt_target_id == 55
    assert not np.isnan(r.iou)


# --------------------------------------------------------------------------- nested-
# anchor geometry adjudication (issue #170): the ambiguity-adjudication geometry
# fallback also resolves a clause whose own anchor carries a nested (non-superlative)
# disambiguator, e.g. "the vase ON the cabinet BELOW the picture" — the outer anchor
# ("cabinet") is pinned by first resolving its own disambiguator's anchor uniquely.


def test_or_geometry_fallback_ambiguous_nested_anchor_resolves():
    """Two ambiguous vase candidates; only one sits on the SPECIFIC cabinet that is
    itself uniquely pinned by its own nested disambiguator ("below the picture")."""
    cabinet_a = _box(1, "cabinet", 0, 1, 0, 1, 0, 1)  # under the picture
    cabinet_b = _box(2, "cabinet", 5, 6, 5, 6, 0, 1)  # not under the picture
    picture = _box(3, "picture", 0.2, 0.8, 0.2, 0.8, 1.5, 2.0)
    vase_a = _box(4, "vase", 0.3, 0.5, 0.3, 0.5, 1.0, 1.3)  # on cabinet_a
    vase_b = _box(5, "vase", 5.3, 5.5, 5.3, 5.5, 1.0, 1.3)  # on cabinet_b
    instances = [cabinet_a, cabinet_b, picture, vase_a, vase_b]
    idx = BasicSceneIndex(instances)
    ref = {
        "regions": {
            "0": {
                "the tall vase that is on a cabinet": [
                    {"target_index": "4", "target_class": "vase", "relation": "on",
                     "anchors": {"anchor_1": {"class": "cabinet"}}},
                ],
                "the short vase that is on a cabinet": [
                    {"target_index": "5", "target_class": "vase", "relation": "on",
                     "anchors": {"anchor_1": {"class": "cabinet"}}},
                ],
            }
        }
    }
    r = S.score_object_reference(
        "Find the vase on the cabinet below the picture.",
        idx, instances, referential=ref,
    )
    assert r.target_source == "geometry_ambiguous"
    assert r.match_method == "geometric"
    assert r.gt_target_id == 4
    assert not np.isnan(r.iou)


def test_or_geometry_fallback_nested_anchor_sub_anchor_not_unique_stays_none():
    """Guard: when the disambiguator's OWN anchor noun matches more than one
    instance (no unique sub-anchor identity), the outer anchor cannot be pinned
    either — stays honestly declined, never guesses."""
    cabinet_a = _box(1, "cabinet", 0, 1, 0, 1, 0, 1)
    cabinet_b = _box(2, "cabinet", 5, 6, 5, 6, 0, 1)
    picture_1 = _box(3, "picture", 0.2, 0.8, 0.2, 0.8, 1.5, 2.0)
    picture_2 = _box(4, "picture", 5.2, 5.8, 5.2, 5.8, 1.5, 2.0)
    vase_a = _box(5, "vase", 0.3, 0.5, 0.3, 0.5, 1.0, 1.3)
    vase_b = _box(6, "vase", 5.3, 5.5, 5.3, 5.5, 1.0, 1.3)
    instances = [cabinet_a, cabinet_b, picture_1, picture_2, vase_a, vase_b]
    idx = BasicSceneIndex(instances)
    ref = {
        "regions": {
            "0": {
                "the tall vase that is on a cabinet": [
                    {"target_index": "5", "target_class": "vase", "relation": "on",
                     "anchors": {"anchor_1": {"class": "cabinet"}}},
                ],
                "the short vase that is on a cabinet": [
                    {"target_index": "6", "target_class": "vase", "relation": "on",
                     "anchors": {"anchor_1": {"class": "cabinet"}}},
                ],
            }
        }
    }
    r = S.score_object_reference(
        "Find the vase on the cabinet below the picture.",
        idx, instances, referential=ref,
    )
    assert r.target_source == "ambiguous"
    assert r.match_method == "none"
    assert r.gt_target_id is None
    assert np.isnan(r.iou)


# --------------------------------------------------------------------------- superlative
# ambiguity adjudication (issue #170): a superlative clause ("closest to"/"farthest
# from") is declined in zero-candidate mode (ranking the whole target class would just
# re-derive our own resolver's answer as "ground truth"), but IS allowed in ambiguity-
# adjudication mode, because there the pool is already the text ladder's own small tied
# candidate set, never the whole scene -- a genuinely independent GT-geometry check.


def test_or_geometry_fallback_ambiguous_superlative_resolves():
    """Two ambiguous fossil-decoration candidates tie in text; only one is actually
    closest to the phone by GT geometry -- the superlative adjudicates the tie."""
    phone = _box(1, "phone", 0, 0.2, 0, 0.2, 1.0, 1.1)
    fossil_near = _box(2, "fossil decoration", 0.3, 0.4, 0.3, 0.4, 1.0, 1.1)
    fossil_far = _box(3, "fossil decoration", 6, 6.2, 6, 6.2, 1.0, 1.1)
    instances = [phone, fossil_near, fossil_far]
    idx = BasicSceneIndex(instances)
    ref = {
        "regions": {
            "0": {
                "the gray fossil decoration that is near the phone": [
                    {"target_index": "2", "target_class": "fossil decoration",
                     "relation": "closest", "anchors": {"anchor_1": {"class": "phone"}}},
                ],
                "the black fossil decoration that is near the phone": [
                    {"target_index": "3", "target_class": "fossil decoration",
                     "relation": "closest", "anchors": {"anchor_1": {"class": "phone"}}},
                ],
            }
        }
    }
    r = S.score_object_reference(
        "Find the fossil decoration closest to the phone.", idx, instances,
        referential=ref,
    )
    assert r.target_source == "geometry_ambiguous"
    assert r.match_method == "geometric"
    assert r.gt_target_id == 2
    assert not np.isnan(r.iou)


def test_or_geometry_fallback_zero_candidate_superlative_still_declines():
    """Guard: superlative clauses stay declined in ZERO-candidate mode -- ranking
    every same-class instance in the scene is still out of scope, unaffected by
    the new ambiguity-adjudication path."""
    phone = _box(1, "phone", 0, 0.2, 0, 0.2, 1.0, 1.1)
    fossil_near = _box(2, "fossil decoration", 0.3, 0.4, 0.3, 0.4, 1.0, 1.1)
    fossil_far = _box(3, "fossil decoration", 6, 6.2, 6, 6.2, 1.0, 1.1)
    instances = [phone, fossil_near, fossil_far]
    idx = BasicSceneIndex(instances)
    r = S.score_object_reference(
        "Find the fossil decoration closest to the phone.", idx, instances,
        referential=None,
    )
    assert r.match_method == "none"
    assert r.gt_target_id is None


def test_or_geometry_fallback_ambiguous_superlative_genuine_tie_stays_none():
    """Guard: two ambiguous candidates EXACTLY equidistant from the anchor must
    not guess a winner via id order -- stays declined."""
    phone = _box(1, "phone", 0, 0.2, 0, 0.2, 1.0, 1.1)
    fossil_a = _box(2, "fossil decoration", 1.0, 1.2, 0.0, 0.2, 1.0, 1.1)
    fossil_b = _box(3, "fossil decoration", -1.0, -0.8, 0.0, 0.2, 1.0, 1.1)
    instances = [phone, fossil_a, fossil_b]
    idx = BasicSceneIndex(instances)
    ref = {
        "regions": {
            "0": {
                "the gray fossil decoration that is near the phone": [
                    {"target_index": "2", "target_class": "fossil decoration",
                     "relation": "closest", "anchors": {"anchor_1": {"class": "phone"}}},
                ],
                "the black fossil decoration that is near the phone": [
                    {"target_index": "3", "target_class": "fossil decoration",
                     "relation": "closest", "anchors": {"anchor_1": {"class": "phone"}}},
                ],
            }
        }
    }
    r = S.score_object_reference(
        "Find the fossil decoration closest to the phone.", idx, instances,
        referential=ref,
    )
    assert r.target_source == "ambiguous"
    assert r.match_method == "none"
    assert r.gt_target_id is None


# --------------------------------------------------------------------------- #177
# residual-set resolution: between-relation support, vertical (above/under)
# narrow-then-rank, and the anchor-uniqueness fallback for an unverifiable nested
# disambiguator. See :func:`S._gt_target_from_geometry`,
# :func:`S._geometry_relation_candidates`, :func:`S._geometry_relation_pool_existential`
# and the #177 update to :func:`S._resolve_geometric_anchor`.


def test_or_between_relation_unique_target_resolves():
    """A single BETWEEN clause with both anchors uniquely resolved pins the ONE
    target-class instance whose centroid falls in the anchor-to-anchor capsule —
    the arabic_room 'wall lamp between a door frame and a window' shape (issue
    #177), synthetic and with unambiguous anchors so it actually resolves."""
    door = _box(1, "door frame", 0, 1, 0, 1, 0, 2)
    window = _box(2, "window", 4, 5, 0, 1, 0, 2)
    lamp_between = _box(3, "wall lamp", 2.4, 2.6, 0.4, 0.6, 1.0, 1.5)
    lamp_far = _box(4, "wall lamp", 20, 20.2, 20, 20.2, 1.0, 1.5)
    instances = [door, window, lamp_between, lamp_far]
    idx = BasicSceneIndex(instances)
    r = S.score_object_reference(
        "Find the wall lamp that is between the door frame and the window.",
        idx, instances, referential=None,
    )
    assert r.target_source == "geometry"
    assert r.match_method == "geometric"
    assert r.gt_target_id == 3
    assert not np.isnan(r.iou)


def test_or_between_relation_ambiguous_anchor_stays_none():
    """Guard: BETWEEN never guesses which anchor instance is meant. Two 'door
    frame' instances in the scene -- the arabic_room real-data shape (issue #177)
    -- means the anchor itself cannot be pinned, so the question stays declined
    even though a target-class instance clearly sits between ONE of the pairs."""
    door_a = _box(1, "door frame", 0, 1, 0, 1, 0, 2)
    door_b = _box(2, "door frame", 10, 11, 10, 11, 0, 2)
    window = _box(3, "window", 4, 5, 0, 1, 0, 2)
    lamp = _box(4, "wall lamp", 2.4, 2.6, 0.4, 0.6, 1.0, 1.5)
    lamp_other = _box(5, "wall lamp", 30, 30.2, 30, 30.2, 1.0, 1.5)
    instances = [door_a, door_b, window, lamp, lamp_other]
    idx = BasicSceneIndex(instances)
    r = S.score_object_reference(
        "Find the wall lamp that is between the door frame and the window.",
        idx, instances, referential=None,
    )
    assert r.target_source == "none"
    assert r.match_method == "none"
    assert r.gt_target_id is None
    assert np.isnan(r.iou)


def test_or_narrow_then_rank_relation_plus_superlative_resolves():
    """Two separate top-level clauses on the same target ('on the tray' + 'closest
    to the window') -- the chinese_room 'bowl on the table closest to the folding
    screen' shape (issue #177). The relation clause narrows to a hard-verified
    pool (two mugs on the tray); the superlative then breaks the tie by real GT
    distance, which is genuine independent evidence, not a re-derived guess."""
    tray = _box(1, "tray", 0, 1, 0, 1, 0, 1)
    mug_near = _box(2, "mug", 0.6, 0.8, 0.6, 0.8, 1.0, 1.3)
    mug_far = _box(3, "mug", 0.3, 0.5, 0.3, 0.5, 1.0, 1.3)
    window = _box(4, "window", 10, 11, 0, 1, 1, 2)
    instances = [tray, mug_near, mug_far, window]
    idx = BasicSceneIndex(instances)
    r = S.score_object_reference(
        "Find the mug on the tray closest to the window.", idx, instances,
        referential=None,
    )
    assert r.target_source == "geometry"
    assert r.match_method == "geometric"
    assert r.gt_target_id == 2
    assert not np.isnan(r.iou)


def test_or_narrow_then_rank_vertical_above_and_farthest_from_resolves():
    """Vertical relation (issue #177): 'picture above the shelf furthest from the
    floor' -- the hotel_room_1 shape. Both pictures satisfy ABOVE the shelf (the
    hard gate); FARTHEST_FROM the floor breaks the tie by height."""
    shelf = _box(1, "shelf", 0, 1, 0, 1, 0, 0.5)
    floor = _box(2, "floor", -5, 5, -5, 5, -0.1, 0.0)
    picture_low = _box(3, "picture", 0.2, 0.4, 0.2, 0.4, 0.6, 0.8)
    picture_high = _box(4, "picture", 0.2, 0.4, 0.2, 0.4, 2.0, 2.2)
    instances = [shelf, floor, picture_low, picture_high]
    idx = BasicSceneIndex(instances)
    r = S.score_object_reference(
        "Find the picture above the shelf furthest from the floor.", idx, instances,
        referential=None,
    )
    assert r.target_source == "geometry"
    assert r.match_method == "geometric"
    assert r.gt_target_id == 4
    assert not np.isnan(r.iou)


def test_or_narrow_then_rank_superlative_anchor_consensus_resolves():
    """Anchor-consensus fallback (issue #177): the superlative's own anchor noun
    ('floor') is not class-unique (two floor regions), but every strongest-tier
    'floor' instance ranks the SAME picture as the clear farthest-from winner --
    the answer does not depend on which literal floor is meant, so consensus
    across every possibility is still an honest, non-guessed resolution."""
    shelf = _box(1, "shelf", 0, 1, 0, 1, 0, 0.5)
    floor_a = _box(2, "floor", -5, -4, -5, -4, -0.1, 0.0)
    floor_b = _box(3, "floor", 4, 5, 4, 5, -0.1, 0.0)
    picture_low = _box(4, "picture", 0.2, 0.4, 0.2, 0.4, 0.6, 0.8)
    picture_high = _box(5, "picture", 0.2, 0.4, 0.2, 0.4, 2.0, 2.2)
    instances = [shelf, floor_a, floor_b, picture_low, picture_high]
    idx = BasicSceneIndex(instances)
    r = S.score_object_reference(
        "Find the picture above the shelf furthest from the floor.", idx, instances,
        referential=None,
    )
    assert r.target_source == "geometry"
    assert r.match_method == "geometric"
    assert r.gt_target_id == 5
    assert not np.isnan(r.iou)


def test_or_narrow_then_rank_genuine_tie_stays_none():
    """Guard: when the two relation-gated pool members are EXACTLY equidistant
    from the superlative anchor, the tie-break margin is zero and the question
    must stay declined, never guessed by id order."""
    tray = _box(1, "tray", 0, 1, 0, 1, 0, 1)
    mug_a = _box(2, "mug", 0.3, 0.5, 0.3, 0.5, 1.0, 1.3)
    mug_b = _box(3, "mug", 0.6, 0.8, 0.6, 0.8, 1.0, 1.3)
    # window centroid sits on the perpendicular bisector of mug_a/mug_b's centroids
    # (both at distance 0.55 along the diagonal) so both mugs are equidistant.
    window = _box(4, "window", 5.45, 5.65, -4.55, -4.35, 1.0, 2.0)
    instances = [tray, mug_a, mug_b, window]
    idx = BasicSceneIndex(instances)
    r = S.score_object_reference(
        "Find the mug on the tray closest to the window.", idx, instances,
        referential=None,
    )
    assert r.target_source == "none"
    assert r.match_method == "none"
    assert r.gt_target_id is None
    assert np.isnan(r.iou)


def test_or_anchor_fallback_unique_bare_anchor_overrides_unverifiable_disambiguator():
    """Anchor-uniqueness fallback (issue #177): 'the cabinet below the picture' --
    the livingroom_1 'vase on the cabinet below the picture' shape. The cabinet
    class is already unique in the scene; 'below the picture' cannot itself be
    checked because 'picture' is not class-unique (two pictures), but that makes
    the disambiguator unverifiable, not contradicted -- the already-unique cabinet
    still pins the anchor, and the vase pool on it is a single honest winner."""
    cabinet = _box(1, "cabinet", 0, 1, 0, 1, 0, 1)
    picture_a = _box(2, "picture", 0, 1, 0, 1, 1.5, 2.0)
    picture_b = _box(3, "picture", 5, 6, 5, 6, 1.5, 2.0)
    vase = _box(4, "vase", 0.3, 0.5, 0.3, 0.5, 1.0, 1.3)
    vase_elsewhere = _box(6, "vase", 8, 8.2, 8, 8.2, 1.0, 1.3)
    instances = [cabinet, picture_a, picture_b, vase, vase_elsewhere]
    idx = BasicSceneIndex(instances)
    r = S.score_object_reference(
        "Find the vase on the cabinet below the picture.", idx, instances,
        referential=None,
    )
    assert r.target_source == "geometry"
    assert r.match_method == "geometric"
    assert r.gt_target_id == 4
    assert not np.isnan(r.iou)


def test_or_anchor_fallback_declines_when_bare_anchor_also_ambiguous():
    """Guard: the anchor-uniqueness fallback only fires when the BARE anchor noun
    is itself unique. Two cabinets, same unverifiable disambiguator shape -- there
    is no honest single anchor to fall back to, so the question stays declined."""
    cabinet_a = _box(1, "cabinet", 0, 1, 0, 1, 0, 1)
    cabinet_b = _box(2, "cabinet", 8, 9, 8, 9, 0, 1)
    picture_a = _box(3, "picture", 0, 1, 0, 1, 1.5, 2.0)
    picture_b = _box(4, "picture", 5, 6, 5, 6, 1.5, 2.0)
    vase = _box(5, "vase", 0.3, 0.5, 0.3, 0.5, 1.0, 1.3)
    vase_elsewhere = _box(6, "vase", 20, 20.2, 20, 20.2, 1.0, 1.3)
    instances = [cabinet_a, cabinet_b, picture_a, picture_b, vase, vase_elsewhere]
    idx = BasicSceneIndex(instances)
    r = S.score_object_reference(
        "Find the vase on the cabinet below the picture.", idx, instances,
        referential=None,
    )
    assert r.target_source == "none"
    assert r.match_method == "none"
    assert r.gt_target_id is None
    assert np.isnan(r.iou)


# --------------------------------------------------------------------------- #177
# real training-scene residual rows (see issue #177's question list): rows that
# were "match_method=none" / "no GT target matched" against the LIVE scorer before
# this fix now resolve to a unique GT target from the real GT annotations, and the
# rows that stay genuinely ambiguous under the real GT data are locked down too.


@requires_full_unity
def test_or_177_chinese_room_bowl_on_table_closest_to_screen_resolves():
    """Residual row: chinese_room 'Find the bowl on the table closest to the
    folding screen.' Two tables in the scene make the relation anchor itself
    ambiguous, but the existential on-table pool has exactly one bowl once
    ranked by the folding-screen superlative."""
    scene_dir = FULL_UNITY_ROOT / "chinese_room"
    referential = __import__("json").loads(
        (scene_dir / "chinese_room_referential_statements.json").read_text()
    )
    scene = load_scene(scene_dir)
    idx = BasicSceneIndex(scene.instances)
    q = "Find the bowl on the table closest to the folding screen."
    r = S.score_object_reference(q, idx, scene.instances, referential=referential)
    assert r.target_source == "geometry"
    assert r.match_method == "geometric"
    assert r.gt_target_id == 91
    assert not np.isnan(r.iou)


@requires_full_unity
def test_or_177_hotel_room_1_picture_above_suitcase_furthest_from_floor_resolves():
    """Residual row: hotel_room_1 'Find the picture above the suitcase furthest
    from the floor.' Two 'floor' region instances make the superlative anchor
    ambiguous, but every floor instance ranks the same picture as the clear
    farthest-from winner (anchor-consensus)."""
    scene_dir = FULL_UNITY_ROOT / "hotel_room_1"
    referential = __import__("json").loads(
        (scene_dir / "hotel_room_1_referential_statements.json").read_text()
    )
    scene = load_scene(scene_dir)
    idx = BasicSceneIndex(scene.instances)
    q = "Find the picture above the suitcase furthest from the floor."
    r = S.score_object_reference(q, idx, scene.instances, referential=referential)
    assert r.target_source == "geometry"
    assert r.match_method == "geometric"
    assert r.gt_target_id == 19
    assert not np.isnan(r.iou)


@requires_full_unity
def test_or_177_livingroom_1_vase_on_cabinet_below_picture_resolves():
    """Residual row: livingroom_1 'Find the vase on the cabinet below the
    picture.' The text ladder ties two vase candidates; the geometry-ambiguous
    adjudication pass now pins the cabinet (already unique) via the #177
    anchor-uniqueness fallback and finds one vase on it."""
    scene_dir = FULL_UNITY_ROOT / "livingroom_1"
    referential = __import__("json").loads(
        (scene_dir / "livingroom_1_referential_statements.json").read_text()
    )
    scene = load_scene(scene_dir)
    idx = BasicSceneIndex(scene.instances)
    q = "Find the vase on the cabinet below the picture."
    r = S.score_object_reference(q, idx, scene.instances, referential=referential)
    assert r.target_source == "geometry_ambiguous"
    assert r.match_method == "geometric"
    assert r.gt_target_id == 23
    assert not np.isnan(r.iou)


@requires_full_unity
def test_or_177_arabic_room_wall_lamp_between_stays_flagged():
    """Guard, real data: arabic_room 'Find the wall lamp that is between a door
    frame and a window.' has TWO door frames in the GT annotation, so the BETWEEN
    anchor cannot be pinned uniquely -- stays honestly declined, not guessed."""
    scene_dir = FULL_UNITY_ROOT / "arabic_room"
    scene = load_scene(scene_dir)
    idx = BasicSceneIndex(scene.instances)
    q = "Find the wall lamp that is between a door frame and a window."
    r = S.score_object_reference(q, idx, scene.instances, referential=None)
    assert r.target_source == "none"
    assert r.match_method == "none"
    assert r.gt_target_id is None
    assert np.isnan(r.iou)


@requires_full_unity
def test_or_177_japanese_room_red_pillow_closest_to_sushi_stays_flagged():
    """Guard, real data: japanese_room 'The red pillow closest to the sushi.' is a
    single top-level superlative clause with zero pre-narrowed candidates -- issue
    #170's circularity rule still applies (ranking the WHOLE class would just
    re-derive our own resolver's answer), so this residual row correctly stays
    declined; #177 does not widen that rule."""
    scene_dir = FULL_UNITY_ROOT / "japanese_room"
    scene = load_scene(scene_dir)
    idx = BasicSceneIndex(scene.instances)
    q = "The red pillow closest to the sushi."
    r = S.score_object_reference(q, idx, scene.instances, referential=None)
    assert r.target_source == "none"
    assert r.match_method == "none"
    assert r.gt_target_id is None
    assert np.isnan(r.iou)
    assert np.isnan(r.iou)
