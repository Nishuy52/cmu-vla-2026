"""Tests for the pure helpers in ``tools.llm_parse_battery`` (no network)."""
from __future__ import annotations

from core.parsing.regex_tier import parse_regex
from tools.llm_parse_battery import (
    BattleRow,
    _contention_note,
    _recommend,
    aggregate_by_qtype,
    diff_summary,
    plan_summary,
    run_one,
    summaries_agree,
)


# --------------------------------------------------------------------------- plan_summary / diff


def test_plan_summary_numerical_target():
    plan = parse_regex("How many chairs are near the table?")
    s = plan_summary(plan)
    assert s["qtype"] == "numerical"
    assert s["target_noun"] == "chair"
    assert s["n_route_legs"] == 0


def test_plan_summary_instruction_following_route():
    plan = parse_regex("Go to the chair, then go to the door.")
    s = plan_summary(plan)
    assert s["qtype"] == "instruction_following"
    assert s["target_noun"] is None
    assert s["n_route_legs"] == 2
    assert s["route_kinds"] == ["goto", "goto"]


def test_summaries_agree_and_diff_summary():
    a = plan_summary(parse_regex("How many chairs are near the table?"))
    b = plan_summary(parse_regex("How many chairs are near the table?"))
    assert summaries_agree(a, b)
    assert diff_summary(a, b) == []

    c = plan_summary(parse_regex("How many beds are near the table?"))
    assert not summaries_agree(a, c)
    diffs = diff_summary(a, c)
    assert any("target_noun" in d for d in diffs)


# --------------------------------------------------------------------------- run_one


def test_run_one_identical_llm_reply_agrees_with_floor():
    q = "How many chairs are near the table?"
    floor_plan = parse_regex(q)
    reply = floor_plan.to_json()
    row = run_one({"scene": "s1", "qtype": "numerical", "text": q}, lambda messages: reply)
    assert isinstance(row, BattleRow)
    assert row.agree is True
    assert row.llm_used is True
    assert row.llm_final_tier == "local"
    assert row.floor_qtype_correct is True
    assert row.llm_qtype_correct is True


def test_run_one_llm_failure_falls_back_and_still_agrees_with_itself():
    q = "How many chairs are near the table?"
    row = run_one({"scene": "s1", "qtype": "numerical", "text": q}, lambda messages: "garbage")
    assert row.llm_used is False
    assert row.llm_final_tier == "regex"
    # both sides end up regex-parsed from the same text -> identical summaries
    assert row.agree is True


def test_run_one_wrong_qtype_flagged_incorrect():
    q = "How many chairs are near the table?"
    row = run_one({"scene": "s1", "qtype": "object_reference", "text": q}, lambda messages: "garbage")
    assert row.floor_qtype_correct is False
    assert row.llm_qtype_correct is False


# --------------------------------------------------------------------------- aggregate_by_qtype / _recommend


def _row(qtype: str, agree: bool, llm_used: bool, llm_qtype_ok: bool, floor_qtype_ok: bool = True,
         llm_valid: bool = True, floor_valid: bool = True) -> BattleRow:
    return BattleRow(
        scene="s", true_qtype=qtype, question="q",
        floor_qtype_correct=floor_qtype_ok, floor_valid=floor_valid, floor_summary={},
        llm_final_tier="local" if llm_used else "regex",
        llm_qtype_correct=llm_qtype_ok, llm_valid=llm_valid, llm_summary={},
        llm_used=llm_used, llm_repair_used=False, llm_n_calls=1, llm_latency_s=1.0,
        agree=agree, diffs=[],
    )


def test_aggregate_by_qtype_computes_rates():
    rows = [
        _row("numerical", agree=True, llm_used=True, llm_qtype_ok=True),
        _row("numerical", agree=False, llm_used=True, llm_qtype_ok=False),
        _row("object_reference", agree=True, llm_used=False, llm_qtype_ok=True),
    ]
    agg = aggregate_by_qtype(rows)
    assert agg["numerical"]["n"] == 2
    assert agg["numerical"]["agreement_rate"] == 0.5
    assert agg["object_reference"]["n"] == 1
    assert agg["instruction_following"]["n"] == 0


def test_recommend_keeps_floor_when_no_upside():
    agg = {
        "n": 10, "agreement_rate": 1.0, "llm_used_local_tier_rate": 1.0,
        "floor_qtype_accuracy": 1.0, "llm_qtype_accuracy": 1.0,
        "floor_valid_rate": 1.0, "llm_valid_rate": 1.0,
    }
    assert "KEEP FLOOR" in _recommend("numerical", agg)


def test_recommend_uses_llm_when_it_improves_qtype_accuracy():
    agg = {
        "n": 10, "agreement_rate": 0.5, "llm_used_local_tier_rate": 1.0,
        "floor_qtype_accuracy": 0.6, "llm_qtype_accuracy": 0.9,
        "floor_valid_rate": 1.0, "llm_valid_rate": 1.0,
    }
    assert "USE LLM" in _recommend("numerical", agg)


def test_contention_note_flags_low_llm_usage():
    rows = [_row("numerical", agree=True, llm_used=False, llm_qtype_ok=True) for _ in range(10)]
    for r in rows:
        r.llm_latency_s = 40.0
    text = "\n".join(_contention_note(rows))
    assert "0/10" in text
    assert "Caveat" in text
    assert "quiet box" in text


def test_contention_note_silent_when_llm_mostly_used():
    rows = [_row("numerical", agree=True, llm_used=True, llm_qtype_ok=True) for _ in range(10)]
    text = "\n".join(_contention_note(rows))
    assert "10/10" in text
    assert "Caveat" not in text


def test_recommend_keeps_floor_when_llm_regresses():
    agg = {
        "n": 10, "agreement_rate": 0.5, "llm_used_local_tier_rate": 1.0,
        "floor_qtype_accuracy": 0.9, "llm_qtype_accuracy": 0.5,
        "floor_valid_rate": 1.0, "llm_valid_rate": 0.7,
    }
    assert "KEEP FLOOR" in _recommend("numerical", agg)
