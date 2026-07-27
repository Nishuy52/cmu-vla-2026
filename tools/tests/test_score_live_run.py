"""Tests for tools/score_live_run.py — the pure/deterministic helpers only.

No bag files or GT dataset needed: covers decimation, question matching,
marker-frame mapping, offline-row pairing/headline extraction, and the
incremental-rerun merge — the logic most likely to silently drift wrong
without a real capture to eyeball against.
"""
from __future__ import annotations

import json

import numpy as np

from core.groundtruth.scoring import Frame2D
from tools.score_live_run import (
    _decimate_xy,
    _load_offline_index,
    _marker_aabb_in_object_frame,
    _match_question,
    _merge_with_existing,
    _offline_headline,
    _squash,
)


def test_decimate_xy_keeps_endpoints_and_drops_close_points():
    pts = [(0.0, 0.0), (0.01, 0.0), (0.02, 0.0), (0.2, 0.0), (0.2, 0.0)]
    out = _decimate_xy(pts, min_move=0.05)
    assert out.shape[0] == 2
    assert tuple(out[0]) == (0.0, 0.0)
    assert tuple(out[-1]) == (0.2, 0.0)


def test_decimate_xy_empty_input():
    assert _decimate_xy([], min_move=0.05).shape == (0, 2)


def test_decimate_xy_single_point():
    out = _decimate_xy([(1.0, 2.0)], min_move=0.05)
    assert out.shape == (1, 2)
    assert tuple(out[0]) == (1.0, 2.0)


def test_squash_ignores_whitespace_and_case():
    assert _squash("How Many  Chairs?") == _squash("how many chairs?")


def test_match_question_exact_and_whitespace_tolerant():
    candidates = ["Find the vase on the cabinet.", "Go to the lamp."]
    assert _match_question("Find the vase on the cabinet.", candidates) == 0
    assert _match_question("  find THE vase on the cabinet.  ", candidates) == 0
    assert _match_question("Go to the lamp.", candidates) == 1
    assert _match_question("no such question", candidates) is None


def test_marker_aabb_no_frame_is_identity():
    marker = {"cx": 1.0, "cy": 2.0, "cz": 0.5, "sx": 2.0, "sy": 1.0, "sz": 1.0}
    a_min, a_max = _marker_aabb_in_object_frame(marker, None)
    np.testing.assert_allclose(a_min, [0.0, 1.5, 0.0])
    np.testing.assert_allclose(a_max, [2.0, 2.5, 1.0])


def test_marker_aabb_translation_only_frame_shifts_xy_not_z():
    marker = {"cx": 0.0, "cy": 0.0, "cz": 0.75, "sx": 1.0, "sy": 1.0, "sz": 0.5}
    frame = Frame2D(theta=0.0, t=np.array([5.0, -3.0]))
    a_min, a_max = _marker_aabb_in_object_frame(marker, frame)
    np.testing.assert_allclose(a_min, [4.5, -3.5, 0.5])
    np.testing.assert_allclose(a_max, [5.5, -2.5, 1.0])


def test_marker_aabb_rotated_frame_rehulls_to_axis_aligned():
    # 90 degree rotation of a non-square footprint must swap the XY half-extents.
    marker = {"cx": 0.0, "cy": 0.0, "cz": 0.0, "sx": 2.0, "sy": 0.4, "sz": 1.0}
    frame = Frame2D(theta=np.pi / 2, t=np.array([0.0, 0.0]))
    a_min, a_max = _marker_aabb_in_object_frame(marker, frame)
    np.testing.assert_allclose(a_max - a_min, [0.4, 2.0, 1.0], atol=1e-9)


def test_offline_headline_per_qtype():
    assert _offline_headline("numerical", {"true_match": True}) == 1.0
    assert _offline_headline("numerical", {"true_match": False}) == 0.0
    assert _offline_headline("numerical", {"true_match": None}) is None
    assert _offline_headline("object_reference", {"iou": 0.42}) == 0.42
    assert _offline_headline("instruction_following", {"rubric_score": 0.75}) == 0.75
    assert _offline_headline("numerical", None) is None


def test_load_offline_index_keys_by_scene_qtype_question(tmp_path):
    path = tmp_path / "results.json"
    path.write_text(
        json.dumps(
            {
                "scores": [
                    {"scene": "a", "qtype": "numerical", "question": "Q1", "true_match": True},
                ]
            }
        ),
        encoding="utf-8",
    )
    idx = _load_offline_index(path)
    assert idx[("a", "numerical", "Q1")]["true_match"] is True


def test_load_offline_index_missing_file_returns_empty(tmp_path):
    assert _load_offline_index(tmp_path / "nope.json") == {}


def test_merge_with_existing_preserves_other_runs(tmp_path):
    existing = {
        "rows": [
            {"scene": "s", "qdir": "nume", "headline_live": 1.0},
            {"scene": "s", "qdir": "obje", "headline_live": None},
        ]
    }
    (tmp_path / "scores.json").write_text(json.dumps(existing), encoding="utf-8")

    new_rows = [{"scene": "s", "qdir": "obje", "headline_live": 0.5}]
    merged = _merge_with_existing(tmp_path, new_rows)

    by_qdir = {r["qdir"]: r for r in merged}
    assert by_qdir["nume"]["headline_live"] == 1.0  # untouched
    assert by_qdir["obje"]["headline_live"] == 0.5  # replaced


def test_merge_with_existing_no_prior_file(tmp_path):
    new_rows = [{"scene": "s", "qdir": "nume", "headline_live": 1.0}]
    assert _merge_with_existing(tmp_path, new_rows) == new_rows


def test_merge_with_existing_corrupt_file_falls_back_to_fresh(tmp_path):
    (tmp_path / "scores.json").write_text("not json", encoding="utf-8")
    new_rows = [{"scene": "s", "qdir": "nume", "headline_live": 1.0}]
    assert _merge_with_existing(tmp_path, new_rows) == new_rows


def test_merge_with_existing_keeps_both_same_type_questions_in_one_scene(tmp_path):
    # Every scene has TWO object_reference questions, both scored under the
    # same qdir ("obje") — the merge key must not collapse them (#97).
    new_rows = [
        {
            "scene": "s", "qdir": "obje", "question": "Find the red chair.",
            "run_dir": "reports/s/obje", "headline_live": 0.2,
        },
        {
            "scene": "s", "qdir": "obje", "question": "Find the blue lamp.",
            "run_dir": "reports/s/obje", "headline_live": 0.9,
        },
    ]
    merged = _merge_with_existing(tmp_path, new_rows)
    assert len(merged) == 2
    by_question = {r["question"]: r["headline_live"] for r in merged}
    assert by_question["Find the red chair."] == 0.2
    assert by_question["Find the blue lamp."] == 0.9


def test_merge_with_existing_rescore_replaces_same_question_row(tmp_path):
    existing = {
        "rows": [
            {
                "scene": "s", "qdir": "obje", "question": "Find the red chair.",
                "run_dir": "reports/s/obje", "headline_live": 0.2,
            },
            {
                "scene": "s", "qdir": "obje", "question": "Find the blue lamp.",
                "run_dir": "reports/s/obje", "headline_live": 0.9,
            },
        ]
    }
    (tmp_path / "scores.json").write_text(json.dumps(existing), encoding="utf-8")

    # Re-scoring only the "red chair" run must replace just that row and
    # leave the "blue lamp" row (from a different question, same scene/qdir)
    # untouched.
    new_rows = [
        {
            "scene": "s", "qdir": "obje", "question": "Find the red chair.",
            "run_dir": "reports/s/obje", "headline_live": 0.6,
        },
    ]
    merged = _merge_with_existing(tmp_path, new_rows)
    by_question = {r["question"]: r["headline_live"] for r in merged}
    assert len(merged) == 2
    assert by_question["Find the red chair."] == 0.6  # replaced
    assert by_question["Find the blue lamp."] == 0.9  # untouched
