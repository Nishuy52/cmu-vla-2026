"""Tests for tools/live_replay.py (#163).

``tools/tests/fixtures/live_replay/job/`` is a TRIMMED real capture: run
``21_livingroom_4_obje`` from ``reports/cluster_verify/701984`` ("Find the
fossil decoration closest to the phone.", livingroom_4), kept down to the 3
``fossil decoration`` instances (ids 23/25/67) whose relative n_obs/score
ordering reproduces the same winner (id 23) the full 9-candidate snapshot
picked — the ``resolved_plan.jsonl`` record and the ``scores.json`` row's
``live_marker`` are the run's own, byte-for-byte. The question's ``closest_to
phone`` clause is unresolvable in this capture (no ``phone`` instance was ever
perceived — see ``resolved_plan.jsonl``'s ``relaxation_audit``), so ``resolve()``
falls through to its no-discriminating-evidence order (lowest instance id
wins), which is exactly what end-to-end Mode 1 replay is meant to catch a
regression in.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from core.geometry import toolbox as TB
from core.interfaces import ColorBin, InstanceRecord
from core.perception.scene_index import BasicSceneIndex
from core.plan_schema import LegKind, Plan, QType
from tools.live_replay import (
    ObjectReferenceReplay,
    _nearest_instance_to_marker,
    _parse_slot_name,
    discover_runs,
    instance_record_from_dict,
    load_final_instance_snapshot,
    load_last_relaxation_audit,
    load_resolved_plan,
    replay_instruction_following,
    replay_numerical,
    replay_object_reference,
    replay_run,
    scene_index_from_snapshot,
)

FIXTURE_JOB = Path(__file__).parent / "fixtures" / "live_replay" / "job"


# --------------------------------------------------------------------------- unit: artifact IO


def test_parse_slot_name_handles_underscored_scene_names():
    assert _parse_slot_name("4_home_building_1_inst") == ("home_building_1", "inst")
    assert _parse_slot_name("21_livingroom_4_obje") == ("livingroom_4", "obje")
    assert _parse_slot_name("0_arabic_room_nume") == ("arabic_room", "nume")


def test_parse_slot_name_rejects_unrecognised_names():
    assert _parse_slot_name("not_a_slot") is None
    assert _parse_slot_name("scene_only") is None


def test_instance_record_from_dict_round_trips_geometry_and_colour():
    d = {
        "id": 7, "label": "fossil decoration", "position": [1.0, 2.0, 3.0],
        "aabb_min": [0.5, 1.5, 2.5], "aabb_max": [1.5, 2.5, 3.5], "score": 0.42, "n_obs": 5,
        "caption": "small", "color_bins": [{"name": "gray", "rgb": [1, 2, 3], "fraction": 0.5}],
    }
    rec = instance_record_from_dict(d)
    assert isinstance(rec, InstanceRecord)
    assert rec.instance_id == 7
    assert rec.label == "fossil decoration"
    assert np.allclose(rec.centroid, [1.0, 2.0, 3.0])
    assert np.allclose(rec.aabb_min, [0.5, 1.5, 2.5])
    assert np.allclose(rec.aabb_max, [1.5, 2.5, 3.5])
    assert rec.score == pytest.approx(0.42)
    assert rec.n_obs == 5
    assert rec.color_bins == (ColorBin(name="gray", rgb=(1, 2, 3), fraction=0.5),)


def test_load_final_instance_snapshot_takes_last_record(tmp_path):
    p = tmp_path / "instance_index.jsonl"
    p.write_text(
        json.dumps({"wall_time": 1.0, "tag": "periodic", "instances": []}) + "\n"
        + json.dumps({"wall_time": 2.0, "tag": "answer_time", "instances": [{"id": 1}]}) + "\n",
        encoding="utf-8",
    )
    snap = load_final_instance_snapshot(p)
    assert snap["wall_time"] == 2.0
    assert snap["instances"] == [{"id": 1}]


def test_load_final_instance_snapshot_missing_file_returns_none(tmp_path):
    assert load_final_instance_snapshot(tmp_path / "nope.jsonl") is None


def test_load_resolved_plan_prefers_answer_time_record(tmp_path):
    p = tmp_path / "resolved_plan.jsonl"
    numerical_plan = {
        "qtype": "numerical", "question_raw": "How many chairs?",
        "target": {"noun": "chair", "raw": "chair", "attributes": [], "clauses": []},
    }
    p.write_text(
        json.dumps({"tag": "periodic", "plan": None}) + "\n"
        + json.dumps({"tag": "answer_time", "plan": numerical_plan}) + "\n",
        encoding="utf-8",
    )
    result = load_resolved_plan(p)
    assert result is not None
    plan, rec = result
    assert isinstance(plan, Plan)
    assert plan.qtype is QType.NUMERICAL
    assert rec["tag"] == "answer_time"


def test_nearest_instance_to_marker_picks_closest_centroid():
    idx = BasicSceneIndex(
        [
            InstanceRecord(
                instance_id=1, label="x", score=1.0, n_obs=5,
                centroid=np.array([0.0, 0.0, 0.0]),
                aabb_min=np.array([-0.1, -0.1, -0.1]), aabb_max=np.array([0.1, 0.1, 0.1]),
            ),
            InstanceRecord(
                instance_id=2, label="x", score=1.0, n_obs=5,
                centroid=np.array([10.0, 10.0, 0.0]),
                aabb_min=np.array([9.9, 9.9, -0.1]), aabb_max=np.array([10.1, 10.1, 0.1]),
            ),
        ]
    )
    marker = {"cx": 9.9, "cy": 9.9, "cz": 0.0}
    best_id, dist = _nearest_instance_to_marker(marker, idx)
    assert best_id == 2
    assert dist == pytest.approx(0.1414, abs=1e-3)


def test_nearest_instance_to_marker_empty_index_returns_none():
    assert _nearest_instance_to_marker({"cx": 0.0, "cy": 0.0}, BasicSceneIndex([])) == (None, None)


# --------------------------------------------------------------------------- unit: replay_numerical


def _numerical_plan(noun: str = "chair") -> Plan:
    return Plan.from_json(
        json.dumps(
            {
                "qtype": "numerical", "question_raw": f"How many {noun}s?",
                "target": {"noun": noun, "raw": noun, "attributes": [], "clauses": []},
            }
        )
    )


def _chair_index(n: int) -> BasicSceneIndex:
    recs = [
        InstanceRecord(
            instance_id=i, label="chair", score=1.0, n_obs=5,
            centroid=np.array([float(i), 0.0, 0.0]),
            aabb_min=np.array([float(i) - 0.1, -0.1, -0.1]),
            aabb_max=np.array([float(i) + 0.1, 0.1, 0.1]),
        )
        for i in range(n)
    ]
    return BasicSceneIndex(recs)


def test_replay_numerical_matches_when_counts_agree():
    result = replay_numerical(_numerical_plan(), _chair_index(3), then_answer=3)
    assert result.now_count == 3
    assert result.match is True


def test_replay_numerical_flags_a_mismatch():
    result = replay_numerical(_numerical_plan(), _chair_index(2), then_answer=5)
    assert result.now_count == 2
    assert result.match is False


def test_replay_numerical_unknown_then_answer_yields_none_match():
    result = replay_numerical(_numerical_plan(), _chair_index(2), then_answer=None)
    assert result.match is None


# --------------------------------------------------------------------------- unit: replay_instruction_following


def test_replay_instruction_following_resolves_goto_and_corridor_legs():
    plan = Plan.from_json(
        json.dumps(
            {
                "qtype": "instruction_following",
                "question_raw": "Go to the sofa then take the path between the two lamps.",
                "route": [
                    {"kind": "goto", "anchors": [{"noun": "sofa", "raw": "sofa", "attributes": []}]},
                    {
                        "kind": "corridor_between",
                        "anchors": [
                            {"noun": "lamp", "raw": "lamp", "attributes": []},
                            {"noun": "lamp", "raw": "lamp", "attributes": []},
                        ],
                    },
                ],
            }
        )
    )
    index = BasicSceneIndex(
        [
            InstanceRecord(
                instance_id=1, label="sofa", score=1.0, n_obs=5,
                centroid=np.array([0.0, 0.0, 0.0]),
                aabb_min=np.array([-0.5, -0.5, -0.5]), aabb_max=np.array([0.5, 0.5, 0.5]),
            ),
            InstanceRecord(
                instance_id=2, label="lamp", score=1.0, n_obs=5,
                centroid=np.array([2.0, 0.0, 0.0]),
                aabb_min=np.array([1.8, -0.2, -0.2]), aabb_max=np.array([2.2, 0.2, 0.2]),
            ),
            InstanceRecord(
                instance_id=3, label="lamp", score=1.0, n_obs=5,
                centroid=np.array([4.0, 0.0, 0.0]),
                aabb_min=np.array([3.8, -0.2, -0.2]), aabb_max=np.array([4.2, 0.2, 0.2]),
            ),
        ]
    )
    result = replay_instruction_following(plan, index, then_audit_record=None)
    assert result.then_source == "unavailable"
    assert len(result.legs) == 2

    goto_leg = result.legs[0]
    assert goto_leg.kind == LegKind.GOTO.value
    assert goto_leg.now_anchor_instance_ids == [1]
    assert goto_leg.now_grounded is True
    assert goto_leg.now_goal_xy == pytest.approx((0.0, 0.0))

    corridor_leg = result.legs[1]
    assert corridor_leg.kind == LegKind.CORRIDOR_BETWEEN.value
    # IF-F6 distinct-instance rule: the two "lamp" anchors of a shared-noun corridor
    # must resolve to two DIFFERENT instances, never the same one twice.
    assert sorted(corridor_leg.now_anchor_instance_ids) == [2, 3]
    assert corridor_leg.now_gate is not None
    assert corridor_leg.now_gate["mid"] == pytest.approx([3.0, 0.0], abs=1e-6)


def test_replay_instruction_following_diffs_against_a_relaxation_audit_record():
    plan = Plan.from_json(
        json.dumps(
            {
                "qtype": "instruction_following",
                "question_raw": "Go to the sofa.",
                "route": [{"kind": "goto", "anchors": [{"noun": "sofa", "raw": "sofa", "attributes": []}]}],
            }
        )
    )
    index = BasicSceneIndex(
        [
            InstanceRecord(
                instance_id=1, label="sofa", score=1.0, n_obs=5,
                centroid=np.array([0.0, 0.0, 0.0]),
                aabb_min=np.array([-0.5, -0.5, -0.5]), aabb_max=np.array([0.5, 0.5, 0.5]),
            )
        ]
    )
    # The run's own last relaxation_audit.jsonl record said this leg was NOT grounded
    # (e.g. the sofa hadn't reached MIN_GROUND_OBS yet at that tick) -- today's code,
    # replayed against the SAME (now fully-observed) snapshot, grounds it.
    then_audit = {
        "legs": [
            {
                "index": 0, "kind": "goto", "grounded": False,
                "anchors": [{"relax_steps": [], "candidate_count": 1}],
            }
        ]
    }
    result = replay_instruction_following(plan, index, then_audit)
    leg = result.legs[0]
    assert result.then_source == "relaxation_audit.jsonl"
    assert leg.then_grounded is False
    assert leg.now_grounded is True
    assert leg.grounded_changed is True


# --------------------------------------------------------------------------- end-to-end: fixture


def test_fixture_job_is_discoverable():
    refs = discover_runs(FIXTURE_JOB)
    assert len(refs) == 1
    ref = refs[0]
    assert ref.scene == "livingroom_4"
    assert ref.qdir == "obje"
    assert ref.qtype == "object_reference"
    assert ref.scores_row is not None
    assert ref.scores_row["live"]["live_marker"]["cx"] == pytest.approx(-2.6485, abs=1e-3)


def test_fixture_job_end_to_end_mode1_replay_matches_the_run():
    """The full Mode 1 pipeline (discover -> load snapshot+plan -> resolve -> match
    against the run's own captured marker) reproduces what this real (trimmed)
    livingroom_4 object_reference run actually answered: instance id 23."""
    refs = discover_runs(FIXTURE_JOB)
    result = replay_run(refs[0])
    assert result.warnings == []
    assert result.snapshot_n_instances == 3
    detail = result.detail
    assert detail["now_instance_id"] == 23
    assert detail["then_matched_instance_id"] == 23
    assert detail["id_match"] is True


def test_fixture_job_replay_object_reference_directly():
    ref = discover_runs(FIXTURE_JOB)[0]
    snapshot = load_final_instance_snapshot(ref.debug_dir / "instance_index.jsonl")
    plan, _ = load_resolved_plan(ref.debug_dir / "resolved_plan.jsonl")
    index = scene_index_from_snapshot(snapshot)
    assert {r.instance_id for r in index.all_instances()} == {23, 25, 67}

    then_marker = ref.scores_row["live"]["live_marker"]
    result = replay_object_reference(plan, index, then_marker)
    assert isinstance(result, ObjectReferenceReplay)
    assert result.now_instance_id == 23
    assert result.id_match is True


def test_fixture_job_has_no_relaxation_audit_for_an_object_reference_run():
    """object_reference runs never write relaxation_audit.jsonl (only
    instruction_following legs do) -- confirms the loader degrades to None rather
    than raising."""
    ref = discover_runs(FIXTURE_JOB)[0]
    assert load_last_relaxation_audit(ref.debug_dir / "relaxation_audit.jsonl") is None
