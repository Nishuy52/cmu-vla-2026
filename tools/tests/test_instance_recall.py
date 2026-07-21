"""Tests for tools/live_harness/instance_recall.py (issue #84).

Pure helpers only (no bag/capture files needed) plus one integration check
against the real office_1 GT scene (skipped if the VLA-3D dataset symlink is
absent) and a run against the committed pre-#84 office capture to lock in the
graceful "no live_instances field" path.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.live_harness.instance_recall import (
    DEFAULT_GROUNDTRUTH,
    _detected_class_counts,
    _gt_class_counts,
    _load_last_record,
    main,
    render_table,
)

_REPO = Path(__file__).resolve().parents[2]


def test_load_last_record_skips_malformed_lines(tmp_path):
    p = tmp_path / "debug.jsonl"
    p.write_text('{"a": 1}\nnot json\n{"a": 2}\n')
    assert _load_last_record(p) == {"a": 2}


def test_load_last_record_empty_file(tmp_path):
    p = tmp_path / "debug.jsonl"
    p.write_text("")
    assert _load_last_record(p) is None


def test_detected_class_counts_normalises_labels():
    record = {"live_instances": {"by_class": {"tv": 2, "couch": 1}}}
    counts = _detected_class_counts(record)
    # 'tv' -> 'television', 'couch' -> 'sofa' per the scene-index synonym map.
    assert counts == {"television": 2, "sofa": 1}


def test_render_table_recall_and_totals():
    gt = {"chair": 4, "table": 1}
    det = {"chair": 2, "lamp": 1}  # 'lamp' is detected-only: no GT row -> n/a recall
    table = render_table(gt, det)
    assert "chair" in table
    assert "0.50" in table  # 2/4
    lines = {ln.split(" | ")[0].strip(): ln for ln in table.splitlines()}
    assert "0.00" in lines["table"]  # gt=1, det=0 -> 0.00, not n/a
    assert "n/a" in lines["lamp"]  # gt=0 -> recall undefined


def test_main_reports_missing_field_gracefully(capsys):
    capture = _REPO / "reports" / "issue83_live_captures" / "office_1_inst" / "explore_debug.jsonl"
    assert capture.is_file(), "committed pre-#84 office capture is missing"
    rc = main([str(capture), "office_1"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "predates" in out
    assert "live_instances" in out


@pytest.mark.skipif(
    not (DEFAULT_GROUNDTRUTH / "office_1").is_dir(),
    reason="VLA-3D dataset not present in this checkout",
)
def test_gt_class_counts_office_1_has_chair():
    counts = _gt_class_counts(DEFAULT_GROUNDTRUTH / "office_1")
    assert counts.get("chair", 0) > 0


@pytest.mark.skipif(
    not (DEFAULT_GROUNDTRUTH / "office_1").is_dir(),
    reason="VLA-3D dataset not present in this checkout",
)
def test_main_end_to_end_synthetic_record(tmp_path, capsys):
    rec = {
        "question_clock_t": 10.0,
        "keyframes_processed": 3,
        "live_instances": {
            "total_instances": 2,
            "by_class": {"chair": 2},
            "instances": [],
            "truncated": False,
        },
    }
    p = tmp_path / "debug.jsonl"
    p.write_text(json.dumps(rec) + "\n")
    rc = main([str(p), "office_1"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "chair" in out
    assert "keyframes_processed=3" in out
