"""Tests for the pure helpers in ``tools.llm_conformance`` (no network)."""
from __future__ import annotations

import json

from tools.llm_conformance import (
    ConformanceRow,
    _CallRecorder,
    build_report,
    load_all_questions,
    run_one,
    sample_questions,
)


def _write_questions(tmp_path, entries):
    path = tmp_path / "questions.json"
    path.write_text(json.dumps(entries), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- load_all_questions


def test_load_all_questions_flattens_every_qtype(tmp_path):
    path = _write_questions(
        tmp_path,
        [
            {
                "scene": "room_a",
                "questions": {
                    "numerical": ["How many chairs?"],
                    "object_reference": ["Find the lamp."],
                    "instruction_following": ["Go to the door."],
                },
            },
            {"scene": "room_b", "questions": {"numerical": ["How many beds?"]}},
        ],
    )
    rows = load_all_questions(path)
    assert len(rows) == 4
    assert {"scene": "room_a", "qtype": "numerical", "text": "How many chairs?"} in rows
    assert {"scene": "room_b", "qtype": "numerical", "text": "How many beds?"} in rows


def test_load_all_questions_missing_questions_key_yields_nothing(tmp_path):
    path = _write_questions(tmp_path, [{"scene": "empty_room"}])
    assert load_all_questions(path) == []


# --------------------------------------------------------------------------- sample_questions


def _pool(n: int) -> list[dict[str, str]]:
    return [{"scene": f"s{i}", "qtype": "numerical", "text": f"q{i}"} for i in range(n)]


def test_sample_questions_is_deterministic_for_a_fixed_seed():
    pool = _pool(75)
    a = sample_questions(pool, n=10, seed=0)
    b = sample_questions(pool, n=10, seed=0)
    assert a == b
    assert len(a) == 10


def test_sample_questions_different_seeds_can_differ():
    pool = _pool(75)
    a = sample_questions(pool, n=10, seed=0)
    b = sample_questions(pool, n=10, seed=1)
    assert a != b


def test_sample_questions_caps_at_pool_size():
    pool = _pool(5)
    out = sample_questions(pool, n=10, seed=0)
    assert len(out) == 5
    # every pool row appears exactly once
    assert sorted(r["text"] for r in out) == sorted(r["text"] for r in pool)


# --------------------------------------------------------------------------- _CallRecorder / run_one


def test_call_recorder_counts_calls_and_latency():
    rec = _CallRecorder(lambda messages: "reply")
    rec(["ignored"])
    rec(["ignored"])
    assert rec.calls == 2
    assert len(rec.latencies_s) == 2
    assert all(lat >= 0.0 for lat in rec.latencies_s)


VALID_NUMERICAL_JSON = json.dumps(
    {
        "qtype": "numerical",
        "question_raw": "How many pillows are on the bed?",
        "target": {
            "noun": "pillow",
            "raw": "pillows",
            "attributes": [],
            "clauses": [
                {
                    "pred": "on",
                    "anchors": [
                        {"noun": "bed", "raw": "bed", "attributes": [], "disambiguator": None}
                    ],
                    "negated": False,
                }
            ],
        },
        "route": [],
        "avoid": [],
        "notes": "",
        "parse_tier": "api",
    }
)


def test_run_one_schema_valid_on_first_try_marks_local_tier_no_repair():
    row_in = {
        "scene": "bedroom_1",
        "qtype": "numerical",
        "text": "How many pillows are on the bed?",
    }
    row = run_one(row_in, lambda messages: VALID_NUMERICAL_JSON)
    assert isinstance(row, ConformanceRow)
    assert row.parse_tier == "local"
    assert row.schema_valid_via_llm is True
    assert row.repair_round_used is False
    assert row.regex_floor_used is False
    assert row.n_llm_calls == 1


def test_run_one_garbage_reply_falls_through_to_regex_floor():
    row_in = {"scene": "bedroom_1", "qtype": "numerical", "text": "How many beds?"}
    row = run_one(row_in, lambda messages: "not json at all")
    assert row.parse_tier == "regex"
    assert row.schema_valid_via_llm is False
    assert row.repair_round_used is True  # first attempt + one repair round, both failed
    assert row.regex_floor_used is True
    assert row.n_llm_calls == 2


# --------------------------------------------------------------------------- build_report


def test_build_report_pass_when_bar_met():
    rows = [
        ConformanceRow(
            scene="s", qtype="numerical", question="q", parse_tier="local",
            schema_valid_via_llm=True, repair_round_used=False, regex_floor_used=False,
            n_llm_calls=1, latency_s=1.0,
        )
        for _ in range(8)
    ] + [
        ConformanceRow(
            scene="s", qtype="numerical", question="q", parse_tier="regex",
            schema_valid_via_llm=False, repair_round_used=True, regex_floor_used=True,
            n_llm_calls=2, latency_s=2.0,
        )
        for _ in range(2)
    ]
    report = build_report(rows, model="m", base_url="u")
    assert "**Result: PASS**" in report
    assert "8/10 schema-valid" in report


def test_build_report_fail_when_bar_missed():
    rows = [
        ConformanceRow(
            scene="s", qtype="numerical", question="q", parse_tier="regex",
            schema_valid_via_llm=False, repair_round_used=True, regex_floor_used=True,
            n_llm_calls=2, latency_s=2.0,
        )
        for _ in range(10)
    ]
    report = build_report(rows, model="m", base_url="u")
    assert "**Result: FAIL**" in report
