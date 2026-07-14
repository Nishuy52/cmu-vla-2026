"""Battery tests: run a 2-scene subset, check report files + aggregate keys + determinism."""
from __future__ import annotations

import json

import pytest

from core.runner.battery import DEFAULT_QUESTIONS, aggregate, run_battery, write_report

# Each test drives a 10-question multi-scene battery end-to-end (>40 s each). These assert
# report-file / aggregate / determinism semantics tied to the battery's own sim_elapsed and
# answered_before_watchdog fields, so they run at real budget scale. Fast tier skips them.
pytestmark = pytest.mark.slow

_SUBSET = ["arabic_room", "chinese_room"]


def test_battery_subset_runs_and_scores_structure():
    runs, specs = run_battery(DEFAULT_QUESTIONS, scenes=_SUBSET)
    assert len(runs) == 10  # 2 scenes x 5 questions
    assert {s.scene_name for s in specs} == set(_SUBSET)
    # every run published a legal answer before the watchdog (never silent)
    assert all(r.answered_before_watchdog for r in runs)
    # offline path => every plan came from the regex tier
    assert all(r.parse_tier_used == "regex" for r in runs)


def test_aggregate_has_expected_keys():
    runs, _ = run_battery(DEFAULT_QUESTIONS, scenes=_SUBSET)
    agg = aggregate(runs)
    assert set(agg) >= {"overall", "per_type", "corridor_built_rate_if", "avoid_built_rate_if", "parse_tiers"}
    for key in ("answered_before_watchdog", "answer_type_correct", "target_grounded", "floor_used"):
        assert key in agg["overall"]
    for qt in ("numerical", "object_reference", "instruction_following"):
        assert qt in agg["per_type"]


def test_battery_writes_report_files(tmp_path):
    runs, specs = run_battery(DEFAULT_QUESTIONS, scenes=_SUBSET)
    md_path, json_path = write_report(runs, specs, tmp_path)
    assert md_path.exists() and json_path.exists()
    text = md_path.read_text(encoding="utf-8")
    assert "Battery structural-health report" in text
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["n_questions"] == 10
    assert payload["n_scenes"] == 2
    assert "aggregate" in payload and "runs" in payload


def test_battery_deterministic_across_two_runs():
    r1, _ = run_battery(DEFAULT_QUESTIONS, scenes=_SUBSET, seed=1)
    r2, _ = run_battery(DEFAULT_QUESTIONS, scenes=_SUBSET, seed=1)
    a1, a2 = aggregate(r1), aggregate(r2)
    assert a1 == a2
    assert [r.sim_elapsed for r in r1] == [r.sim_elapsed for r in r2]
    assert [r.answer_type_correct for r in r1] == [r.answer_type_correct for r in r2]
