"""Unit tests for tools/perception_eval.py's scoring metrics.

Synthetic fixtures only -- no bag, no GT dataset, no GPU. Exercises the pure
metric functions (match_gt_to_live, compute_metrics, pool_metrics, shell_fraction)
plus the JSONL snapshot loader (load_snapshot) against small hand-built inputs.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from core.interfaces import InstanceRecord
from tools.perception_eval import (
    Match,
    compute_metrics,
    load_snapshot,
    match_gt_to_live,
    pool_metrics,
    shell_fraction,
)


def _gt(instance_id, label, cx, cy, cz, half=0.25):
    c = np.array([cx, cy, cz], dtype=float)
    h = np.array([half, half, half], dtype=float)
    return InstanceRecord(
        instance_id=instance_id, label=label, score=1.0, n_obs=3,
        centroid=c, aabb_min=c - h, aabb_max=c + h,
    )


def _live(iid, label, cx, cy, cz, half=0.25, score=0.9):
    return {
        "id": iid,
        "label": label,
        "position": [cx, cy, cz],
        "aabb_min": [cx - half, cy - half, cz - half],
        "aabb_max": [cx + half, cy + half, cz + half],
        "score": score,
        "n_obs": 3,
    }


# --------------------------------------------------------------------------- match_gt_to_live


def test_match_finds_close_same_label_instance():
    gt = [_gt(0, "chair", 0.0, 0.0, 0.0)]
    live = [_live(0, "chair", 0.05, 0.0, 0.0)]
    matches = match_gt_to_live(gt, live)
    assert matches == [Match(gt_index=0, live_index=0, distance=pytest.approx(0.05))]


def test_match_ignores_different_label():
    gt = [_gt(0, "chair", 0.0, 0.0, 0.0)]
    live = [_live(0, "table", 0.0, 0.0, 0.0)]
    matches = match_gt_to_live(gt, live)
    assert matches[0].live_index is None


def test_match_rejects_beyond_tolerance():
    gt = [_gt(0, "chair", 0.0, 0.0, 0.0)]  # half=0.25 -> tol = max(0.5, 0.25*sqrt(3)) = 0.5
    live = [_live(0, "chair", 10.0, 0.0, 0.0)]
    matches = match_gt_to_live(gt, live)
    assert matches[0].live_index is None


def test_match_picks_nearest_over_first():
    gt = [_gt(0, "chair", 0.0, 0.0, 0.0)]
    live = [
        _live(0, "chair", 0.3, 0.0, 0.0),  # farther
        _live(1, "chair", 0.05, 0.0, 0.0),  # nearer -- must win despite index order
    ]
    matches = match_gt_to_live(gt, live)
    assert matches[0].live_index == 1


def test_match_one_live_instance_can_satisfy_multiple_gt():
    gt = [_gt(0, "chair", 0.0, 0.0, 0.0), _gt(1, "chair", 0.05, 0.0, 0.0)]
    live = [_live(0, "chair", 0.0, 0.0, 0.0)]
    matches = match_gt_to_live(gt, live)
    assert matches[0].live_index == 0
    assert matches[1].live_index == 0


# --------------------------------------------------------------------------- shell_fraction


def test_shell_fraction_all_on_shell():
    lo = np.array([[0.0, 0.0, 0.0], [9.0, 0.0, 0.0]])
    hi = np.array([[1.0, 1.0, 1.0], [10.0, 1.0, 1.0]])
    assert shell_fraction(lo, hi) == pytest.approx(1.0)


def test_shell_fraction_none_on_shell():
    lo = np.array([[4.0, 4.0, 4.0]])
    hi = np.array([[5.0, 5.0, 5.0]])
    gmin_box_lo = np.array([[0.0, 0.0, 0.0]])
    gmax_box_hi = np.array([[10.0, 10.0, 10.0]])
    combined_lo = np.vstack([lo, gmin_box_lo])
    combined_hi = np.vstack([hi, gmax_box_hi])
    # the middle box (index 0) is far from both scene extrema on every axis
    frac = shell_fraction(combined_lo, combined_hi)
    assert frac == pytest.approx(0.5)  # only the extremal box itself is "on the shell"


def test_shell_fraction_empty():
    assert np.isnan(shell_fraction(np.zeros((0, 3)), np.zeros((0, 3))))


# --------------------------------------------------------------------------- compute_metrics


def test_compute_metrics_perfect_match_no_duplication():
    gt = [_gt(0, "chair", 0.0, 0.0, 0.0), _gt(1, "table", 5.0, 0.0, 0.0)]
    live = [_live(0, "chair", 0.0, 0.0, 0.0), _live(1, "table", 5.0, 0.0, 0.0)]
    m = compute_metrics("scene_x", live, gt)
    assert m.n_gt == 2
    assert m.n_live == 2
    assert m.n_found == 2
    assert m.recall == pytest.approx(1.0)
    assert m.duplication_factor == pytest.approx(1.0)
    assert m.inflation == pytest.approx(1.0)


def test_compute_metrics_duplication_and_recall():
    gt = [_gt(0, "chair", 0.0, 0.0, 0.0)]
    # 3 live chairs all near the one GT chair -- classic over-production.
    live = [
        _live(0, "chair", 0.0, 0.0, 0.0),
        _live(1, "chair", 0.05, 0.0, 0.0),
        _live(2, "chair", -0.05, 0.0, 0.0),
    ]
    m = compute_metrics("scene_x", live, gt)
    assert m.recall == pytest.approx(1.0)
    assert m.duplication_factor == pytest.approx(3.0)


def test_compute_metrics_missed_object_recall_zero():
    gt = [_gt(0, "chair", 0.0, 0.0, 0.0)]
    live: list[dict] = []
    m = compute_metrics("scene_x", live, gt)
    assert m.n_found == 0
    assert m.recall == pytest.approx(0.0)
    assert np.isnan(m.duplication_factor)
    assert np.isnan(m.inflation)


def test_compute_metrics_inflation_ratio():
    gt = [_gt(0, "chair", 0.0, 0.0, 0.0, half=0.25)]  # gt diag = 0.5*sqrt(3)
    live = [_live(0, "chair", 0.0, 0.0, 0.0, half=0.5)]  # live diag = 1.0*sqrt(3) -> ratio 2x
    m = compute_metrics("scene_x", live, gt)
    assert m.inflation == pytest.approx(2.0)


def test_compute_metrics_per_class_counts_and_unmatched_classes():
    gt = [_gt(0, "chair", 0.0, 0.0, 0.0), _gt(1, "sofa", 5.0, 0.0, 0.0)]
    live = [
        _live(0, "chair", 0.0, 0.0, 0.0),
        _live(1, "chair", 10.0, 10.0, 10.0),  # a chair with no nearby GT chair
        _live(2, "lamp", 1.0, 1.0, 1.0),  # a class GT never has
    ]
    m = compute_metrics("scene_x", live, gt)
    by_label = {r.label: r for r in m.per_class}
    assert by_label["chair"].live == 2 and by_label["chair"].gt == 1
    assert by_label["sofa"].live == 0 and by_label["sofa"].gt == 1
    assert by_label["lamp"].live == 1 and by_label["lamp"].gt == 0
    assert by_label["lamp"].ratio == float("inf")


# --------------------------------------------------------------------------- pool_metrics


def test_pool_metrics_sums_counts_across_scenes():
    a = compute_metrics("a", [_live(0, "chair", 0, 0, 0)], [_gt(0, "chair", 0, 0, 0)])
    b = compute_metrics(
        "b",
        [_live(0, "chair", 0, 0, 0), _live(1, "chair", 0.05, 0, 0)],
        [_gt(0, "chair", 0, 0, 0)],
    )
    pooled = pool_metrics([a, b])
    assert pooled.n_gt == 2
    assert pooled.n_live == 3
    assert pooled.n_found == 2
    assert pooled.recall == pytest.approx(1.0)
    assert pooled.duplication_factor == pytest.approx(3 / 2)


def test_pool_metrics_merges_per_class_rows():
    a = compute_metrics("a", [_live(0, "chair", 0, 0, 0)], [_gt(0, "chair", 0, 0, 0)])
    b = compute_metrics("b", [_live(0, "sofa", 0, 0, 0)], [_gt(0, "sofa", 0, 0, 0)])
    pooled = pool_metrics([a, b])
    labels = {r.label: r for r in pooled.per_class}
    assert labels["chair"].live == 1 and labels["chair"].gt == 1
    assert labels["sofa"].live == 1 and labels["sofa"].gt == 1


# --------------------------------------------------------------------------- load_snapshot


def _write_jsonl(path, records):
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")


def test_load_snapshot_prefers_answer_time_tag(tmp_path):
    path = tmp_path / "idx.jsonl"
    _write_jsonl(
        path,
        [
            {"tag": "periodic", "total_instances": 1},
            {"tag": "answer_time", "total_instances": 5},
            {"tag": "periodic", "total_instances": 9},
        ],
    )
    snap = load_snapshot(path)
    assert snap["total_instances"] == 5


def test_load_snapshot_falls_back_to_last_record(tmp_path):
    path = tmp_path / "idx.jsonl"
    _write_jsonl(
        path,
        [
            {"tag": "periodic", "total_instances": 1},
            {"tag": "replay_final", "total_instances": 32},
        ],
    )
    snap = load_snapshot(path)
    assert snap["total_instances"] == 32


def test_load_snapshot_empty_file_raises(tmp_path):
    path = tmp_path / "idx.jsonl"
    path.write_text("")
    with pytest.raises(ValueError):
        load_snapshot(path)
