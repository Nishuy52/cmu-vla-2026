"""Tests for tools/replay_live_numerical.py.

No cluster data needed: builds tiny synthetic ``instance_index.jsonl`` capture
dirs so every test is fast and deterministic. Covers the specific defect this
harness exists to prevent (a fixed ``min_obs`` instead of the live head's own
dynamic gate) plus the schema/selection traps recorded dumps set for a naive
reader.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from tools.replay_live_numerical import (
    VariantSummary,
    load_answer_time_record,
    record_to_instances,
    run_replay,
    score_scene,
    scene_name_from_slot,
    variant_raw,
)


def _instance(id_, label, n_obs, score=0.9, pos=(0.0, 0.0, 0.0), half=0.1):
    x, y, z = pos
    return {
        "id": id_,
        "label": label,
        "position": [x, y, z],
        "aabb_min": [x - half, y - half, z - half],
        "aabb_max": [x + half, y + half, z + half],
        "score": score,
        "n_obs": n_obs,
        "answer_eligible": True,
        "eligibility_reason": "eligible",
    }


def _answer_time_record(instances, **extra):
    by_class: dict[str, int] = {}
    for inst in instances:
        by_class[inst["label"]] = by_class.get(inst["label"], 0) + 1
    record = {
        "wall_time": 1234.5,
        "tag": "answer_time",
        "keyframes_processed": None,
        "total_instances": len(instances),
        "by_class": by_class,
        "instances": instances,
    }
    record.update(extra)
    return record


def _write_slot(captures_dir, slot, lines):
    debug_dir = captures_dir / "debug" / slot
    debug_dir.mkdir(parents=True)
    path = debug_dir / "instance_index.jsonl"
    with open(path, "w", encoding="utf-8") as fh:
        for line in lines:
            fh.write(json.dumps(line) + "\n")
    return path


# ------------------------------------------------------------- scene naming


def test_scene_name_from_slot_strips_numeric_prefix_and_nume_suffix():
    assert scene_name_from_slot("0_arabic_room_nume") == "arabic_room"
    assert scene_name_from_slot("13_office_2_nume") == "office_2"


# ------------------------------------------------------ answer_time selection


def test_load_answer_time_record_selects_answer_time_not_last_periodic(tmp_path):
    """The answer_time record is NOT the last line in the file — periodic dumps
    keep appending after it fires. Picking the last line (instead of filtering
    by tag) is exactly the kind of defect this harness must not have."""
    periodic_early = {"tag": "periodic", "total_instances": 0, "instances": []}
    answer_time = _answer_time_record([_instance(1, "sofa", 3)])
    periodic_late = {"tag": "periodic", "total_instances": 999, "instances": [
        _instance(i, "ghost", 1) for i in range(999)
    ]}
    path = tmp_path / "instance_index.jsonl"
    with open(path, "w", encoding="utf-8") as fh:
        for row in (periodic_early, answer_time, periodic_late):
            fh.write(json.dumps(row) + "\n")

    record = load_answer_time_record(path)
    assert record["tag"] == "answer_time"
    assert record["total_instances"] == 1
    assert record is not periodic_late


def test_load_answer_time_record_raises_without_answer_time_tag(tmp_path):
    path = tmp_path / "instance_index.jsonl"
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps({"tag": "periodic", "instances": []}) + "\n")
    with pytest.raises(Exception):
        load_answer_time_record(path)


# ------------------------------------------------------------- schema shape


def test_record_to_instances_reads_top_level_schema_not_nested():
    """instances/by_class/total_instances are TOP-LEVEL keys of the record,
    never nested under a 'live_instances' key."""
    record = _answer_time_record(
        [_instance(7, "chair", 4, score=0.6, pos=(1.0, 2.0, 0.5), half=0.2)]
    )
    assert "live_instances" not in record
    assert "instances" in record and "by_class" in record and "total_instances" in record

    recs = record_to_instances(record)
    assert len(recs) == 1
    rec = recs[0]
    assert rec.instance_id == 7
    assert rec.label == "chair"
    assert rec.n_obs == 4
    assert rec.score == pytest.approx(0.6)
    assert np.allclose(rec.centroid, [1.0, 2.0, 0.5])
    assert np.allclose(rec.aabb_min, [0.8, 1.8, 0.3])
    assert np.allclose(rec.aabb_max, [1.2, 2.2, 0.7])


# --------------------------------------------------------- THE dynamic gate


def test_dynamic_gate_drops_ghost_a_fixed_min_obs_would_keep():
    """The key regression test.

    One 'widget' is established (n_obs=3 >= ESTABLISH_N_OBS), a second is a
    one-frame ghost (n_obs=1). NumericalHead's dynamic gate, once a noun is
    established, counts only n_obs >= GATE_MIN_OBS(=2) — so the ghost is
    dropped and the true count is 1. A harness that instead passed a FIXED
    min_obs=1 (the exact defect that broke an earlier throwaway version) would
    count both and report 2 — a materially different, wrong answer.
    """
    instances = record_to_instances(
        _answer_time_record(
            [
                _instance(1, "widget", n_obs=3, pos=(0.0, 0.0, 0.0)),
                _instance(2, "widget", n_obs=1, pos=(5.0, 5.0, 0.0)),
            ]
        )
    )
    result = score_scene("synthetic_scene", "How many widgets are there?", instances, answers=None)
    assert result.min_obs == 2  # GATE_MIN_OBS, not the fixed MIN_OBS=3 nor a naive 1
    assert result.count == 1  # dynamic gate drops the ghost

    # Sanity check the counterfactual: a fixed min_obs=1 gives the WRONG, inflated
    # count this harness must never reproduce.
    from core.geometry.toolbox import counting
    from core.parsing.regex_tier import parse_regex
    from core.perception.scene_index import BasicSceneIndex

    plan = parse_regex("How many widgets are there?")
    fixed_result = counting(plan.target, BasicSceneIndex(instances), min_obs=1)
    assert fixed_result.count == 2
    assert fixed_result.count != result.count


def test_dynamic_gate_cold_start_counts_everything():
    """Below ESTABLISH_N_OBS for every instance of the noun (cold start), min_obs
    is 1 — a genuine cold count is not starved."""
    instances = record_to_instances(
        _answer_time_record(
            [
                _instance(1, "widget", n_obs=1, pos=(0.0, 0.0, 0.0)),
                _instance(2, "widget", n_obs=2, pos=(5.0, 5.0, 0.0)),
            ]
        )
    )
    result = score_scene("synthetic_scene", "How many widgets are there?", instances, answers=None)
    assert result.min_obs == 1
    assert result.count == 2


# -------------------------------------------------------------------- A/B


def _drop_all(instances):
    return []


def test_ab_mode_reports_both_variants(tmp_path):
    captures_dir = tmp_path / "captures"
    _write_slot(
        captures_dir,
        "0_widgetville_nume",
        [_answer_time_record([_instance(1, "widget", n_obs=3, pos=(0.0, 0.0, 0.0))])],
    )
    questions_path = tmp_path / "questions.json"
    questions_path.write_text(
        json.dumps(
            [{"scene": "widgetville", "questions": {"numerical": ["How many widgets are there?"]}}]
        )
    )
    answers_path = tmp_path / "answers.json"
    answers_path.write_text(
        json.dumps(
            {
                "scenes": {
                    "widgetville": {
                        "question_raw": "How many widgets are there?",
                        "answer": 1,
                    }
                }
            }
        )
    )

    summary_a, summary_b = run_replay(
        captures_dir,
        variant_raw,
        _drop_all,
        questions_path=questions_path,
        answers_path=answers_path,
    )

    assert isinstance(summary_a, VariantSummary) and isinstance(summary_b, VariantSummary)
    assert len(summary_a.results) == 1 and len(summary_b.results) == 1
    ra, rb = summary_a.results[0], summary_b.results[0]
    assert ra.scene == rb.scene == "widgetville"
    assert ra.truth == 1

    # variant A (raw) sees the one recorded widget -> exact match.
    assert ra.count == 1
    assert ra.exact_match is True
    assert summary_a.exact_matches == 1
    assert summary_a.mean_abs_err == pytest.approx(0.0)

    # variant B (drops every instance) sees nothing -> count 0, a clear miss —
    # proves the two variants are genuinely scored independently.
    assert rb.count == 0
    assert rb.exact_match is False
    assert summary_b.exact_matches == 0
    assert summary_b.mean_abs_err == pytest.approx(1.0)


def test_variant_raw_does_not_mutate_or_alias_input_list():
    instances = record_to_instances(_answer_time_record([_instance(1, "widget", n_obs=3)]))
    out = variant_raw(instances)
    assert out == instances
    assert out is not instances
