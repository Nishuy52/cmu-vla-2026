"""Tests for tools/battery_diff.py."""

from __future__ import annotations

import json

from tools.battery_diff import build_report, main


def _base_results(**overrides):
    data = {
        "date": "2026-07-14",
        "n_questions": 1,
        "scenes": ["arabic_room"],
        "missing_scenes": [],
        "aggregate": {
            "numerical": {
                "n": 15,
                "pipeline_determinism_rate": 1.0,
                "n_with_true_answer": 15,
                "true_accuracy": 1.0,
            },
        },
        "scores": [
            {
                "scene": "arabic_room",
                "qtype": "numerical",
                "question": "How many sofas are below a window?",
                "our_count": 0,
                "gt_count_pipeline": 0,
                "exact_match": True,
                "note": "",
            }
        ],
    }
    data.update(overrides)
    return data


def test_identical_inputs_report_no_changes(tmp_path):
    before = _base_results()
    after = json.loads(json.dumps(before))

    report = build_report(before, after)
    assert "no changes" in report
    assert "## Provenance" in report


def test_changed_row_field_detected():
    before = _base_results()
    after = json.loads(json.dumps(before))
    after["scores"][0]["our_count"] = 1
    after["scores"][0]["exact_match"] = False

    report = build_report(before, after)
    assert "## Changed rows" in report
    assert "our_count: 0 -> 1" in report
    assert "exact_match: true -> false" in report


def test_added_row_detected():
    before = _base_results()
    after = json.loads(json.dumps(before))
    after["scores"].append(
        {
            "scene": "arabic_room",
            "qtype": "numerical",
            "question": "How many chairs are near a table?",
            "our_count": 2,
        }
    )

    report = build_report(before, after)
    assert "## Added / removed rows" in report
    assert "### Added" in report
    assert "How many chairs are near a table?" in report


def test_topline_delta_and_one_sided_key():
    before = _base_results()
    after = json.loads(json.dumps(before))
    after["aggregate"]["numerical"]["pipeline_determinism_rate"] = 0.8
    after["aggregate"]["numerical"]["new_metric"] = 0.5

    report = build_report(before, after)
    assert "numerical.pipeline_determinism_rate" in report
    assert "-0.2000" in report
    assert "numerical.new_metric" in report
    # one-sided key: before column shows the missing-value marker
    lines = [l for l in report.splitlines() if "numerical.new_metric" in l]
    assert lines
    assert "—" in lines[0]


def test_unprovenanced_label_and_stamped_after_commit():
    before = _base_results()
    after = _base_results()
    after["provenance"] = {
        "tool": "gt_battery",
        "generated_utc": "2026-07-14T00:00:00Z",
        "git_commit": "abcdef1234567890",
        "git_dirty": False,
        "calibration_sha1": "deadbeef",
    }

    report = build_report(before, after)
    assert "UNPROVENANCED (pre-meth-F7 run)" in report
    assert "abcdef123456" in report


def test_out_flag_writes_file(tmp_path):
    before_path = tmp_path / "before.json"
    after_path = tmp_path / "after.json"
    out_path = tmp_path / "diff.md"

    before_path.write_text(json.dumps(_base_results()), encoding="utf-8")
    after_path.write_text(json.dumps(_base_results()), encoding="utf-8")

    rc = main([str(before_path), str(after_path), "--out", str(out_path)])
    assert rc == 0
    assert out_path.exists()
    assert "no changes" in out_path.read_text(encoding="utf-8")


def test_identical_inputs_cli_exit_zero(tmp_path, capsys):
    before_path = tmp_path / "before.json"
    after_path = tmp_path / "after.json"

    before_path.write_text(json.dumps(_base_results()), encoding="utf-8")
    after_path.write_text(json.dumps(_base_results()), encoding="utf-8")

    rc = main([str(before_path), str(after_path)])
    assert rc == 0
    captured = capsys.readouterr()
    assert "no changes" in captured.out


def test_invalid_json_exits_2(tmp_path):
    bad_path = tmp_path / "bad.json"
    good_path = tmp_path / "good.json"
    bad_path.write_text("{not valid json", encoding="utf-8")
    good_path.write_text(json.dumps(_base_results()), encoding="utf-8")

    rc = main([str(bad_path), str(good_path)])
    assert rc == 2
