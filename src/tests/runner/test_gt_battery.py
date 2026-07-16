"""Battery v2 (GT accuracy) smoke tests on the real loft sample scene."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from core.groundtruth.loader import load_scene
from core.interfaces import QType
from core.runner import gt_battery as GB

_SRC = Path(__file__).resolve().parents[2]
_REPO = _SRC.parent
UNITY_SAMPLE_ROOT = _REPO / "upstream" / "VLA-3D" / "sample_data" / "Unity"
LOFT_DIR = UNITY_SAMPLE_ROOT / "loft"
QUESTIONS_JSON = _REPO / "upstream" / "CMU-VLN-Challenge-2026" / "questions" / "questions.json"
QUESTIONS_DIR = QUESTIONS_JSON.parent

_have_loft = LOFT_DIR.is_dir() and (LOFT_DIR / "loft_object_result.csv").exists()
requires_loft = pytest.mark.skipif(not _have_loft, reason="loft sample data not present")

#: Full downloaded VLA-3D Unity root (all 15 training scenes), if extracted.
FULL_UNITY_ROOT = _REPO / "data" / "vla3d" / "Unity"
_have_full = FULL_UNITY_ROOT.is_dir() and (FULL_UNITY_ROOT / "office_1").is_dir()
requires_full_unity = pytest.mark.skipif(
    not _have_full, reason="full VLA-3D Unity dataset not extracted"
)


def _synthetic_gt_scene(objs, scene_name="synthetic"):
    """Build a GTScene from ``[(label, cx, cy, cz, sx, sy, sz), ...]`` for driven-trajectory
    tests — no dataset needed, deterministic. Every instance is fully observed (n_obs=3)."""
    from core.groundtruth.loader import GTScene
    from core.interfaces import InstanceRecord

    recs = []
    for i, (label, cx, cy, cz, sx, sy, sz) in enumerate(objs):
        lo = np.array([cx - sx / 2, cy - sy / 2, cz - sz / 2], dtype=float)
        hi = np.array([cx + sx / 2, cy + sy / 2, cz + sz / 2], dtype=float)
        recs.append(
            InstanceRecord(
                instance_id=i,
                label=label,
                score=1.0,
                n_obs=3,
                centroid=(lo + hi) / 2.0,
                aabb_min=lo,
                aabb_max=hi,
            )
        )
    return GTScene(scene_name=scene_name, instances=recs, regions=[])


def test_drive_if_trajectory_not_truncated_by_build_draining():
    """Regression: the driven trajectory must span the whole committed route, not be cut
    short because the route-build ticks already advanced the follower's progress index.

    The old harness ticked the instruction head from the *stationary* spawn to firm up the
    route, then drove the SAME follower — so every build tick's ``head.advance`` already
    called ``follower.advance`` at the spawn, consuming leading crumbs. For a route whose
    early crumbs sit within reach of the spawn this left an already-advanced (or fully
    exhausted) follower, and the drive produced a 2-pose trajectory that reaches no leg.
    Here spawn sits ON the first leg goal (the worst case for draining); the drive must
    still traverse to the well-separated terminal, i.e. many poses and terminal arrival.
    """
    from core.perception.scene_index import BasicSceneIndex

    # Two GOTO legs 8 m apart on a clear floor; spawn will be placed on leg-0's anchor.
    gt = _synthetic_gt_scene(
        [
            ("stool", -4.0, 0.0, 0.3, 0.4, 0.4, 0.6),  # leg 0 anchor (near spawn)
            ("table", 4.0, 0.0, 0.4, 1.0, 1.0, 0.8),  # leg 1 / terminal anchor (far)
        ]
    )
    idx = BasicSceneIndex(gt.instances)
    text = "Go to the stool and then go to the table."

    driven = GB._drive_if_trajectory(gt=gt, idx=idx, text=text, start_xy=(-4.0, 0.0))

    assert driven.shape[0] > 2, (
        f"driven trajectory spuriously truncated to {driven.shape[0]} poses - "
        "the follower was drained during route build"
    )
    # It must actually reach the far terminal (table at +4,0), well outside any spawn-local
    # draining radius — proof the whole route was driven, not just the near cluster.
    # It must cross the whole 8 m route to the far side. The GOTO goal is projected to a
    # free cell off the table's (1 m) AABB and the follower stops a step short, so the
    # closest approach sits ~1-1.5 m from the centroid; assert the vehicle got to the far
    # neighbourhood (not that it hit the exact centroid) — proof the whole route was driven.
    term = np.array([4.0, 0.0])
    min_term = float(np.min(np.hypot(driven[:, 0] - term[0], driven[:, 1] - term[1])))
    assert min_term <= 1.5, f"driven trajectory never reached the terminal (min {min_term:.2f} m)"


def test_drive_if_trajectory_progresses_with_moving_pose():
    """The closed-loop driver advances the follower only as the vehicle moves: the pose
    stream must be monotonically progressing (net displacement from spawn to end large),
    not a stationary point repeated."""
    from core.perception.scene_index import BasicSceneIndex

    gt = _synthetic_gt_scene(
        [
            ("stool", -4.0, 0.0, 0.3, 0.4, 0.4, 0.6),
            ("table", 4.0, 0.0, 0.4, 1.0, 1.0, 0.8),
        ]
    )
    idx = BasicSceneIndex(gt.instances)
    text = "Go to the stool and then go to the table."
    driven = GB._drive_if_trajectory(gt=gt, idx=idx, text=text, start_xy=(-4.0, 0.0))
    net = float(np.hypot(*(driven[-1] - driven[0])))
    assert net >= 4.0, f"vehicle barely moved (net {net:.2f} m) — drive did not progress"


@requires_loft
@pytest.mark.slow
def test_score_scene_loft_all_five_questions():
    """loft has 1 numerical + 2 object_reference + 2 instruction_following = 5 Qs."""
    gt = load_scene(LOFT_DIR)
    with open(QUESTIONS_JSON, encoding="utf-8") as fh:
        data = json.load(fh)
    entry = next(e for e in data if e["scene"] == "loft")
    ref_path = LOFT_DIR / "loft_referential_statements.json"
    referential = json.loads(ref_path.read_text(encoding="utf-8"))

    scores = GB.score_scene(
        gt, entry["questions"], referential=referential, questions_dir=QUESTIONS_DIR
    )
    assert len(scores) == 5
    kinds = [s.qtype for s in scores]
    assert kinds.count(QType.NUMERICAL.value) == 1
    assert kinds.count(QType.OBJECT_REFERENCE.value) == 2
    assert kinds.count(QType.INSTRUCTION_FOLLOWING.value) == 2


@requires_loft
def test_score_scene_numerical_has_real_count():
    gt = load_scene(LOFT_DIR)
    with open(QUESTIONS_JSON, encoding="utf-8") as fh:
        data = json.load(fh)
    entry = next(e for e in data if e["scene"] == "loft")
    scores = GB.score_scene(gt, entry["questions"], questions_dir=QUESTIONS_DIR,
                            drive_if=False)
    num = next(s for s in scores if s.qtype == QType.NUMERICAL.value)
    assert num.our_count is not None and num.our_count >= 0
    assert num.exact_match is True  # pipeline self-consistency


@requires_loft
def test_numerical_true_answer_from_key():
    """The true numerical yardstick anchors loft's key answer (2) under the guard."""
    gt = load_scene(LOFT_DIR)
    with open(QUESTIONS_JSON, encoding="utf-8") as fh:
        data = json.load(fh)
    entry = next(e for e in data if e["scene"] == "loft")
    answers = json.loads(Path(GB.DEFAULT_ANSWERS).read_text(encoding="utf-8"))
    scores = GB.score_scene(
        gt, entry["questions"], questions_dir=QUESTIONS_DIR, answers=answers,
        drive_if=False,
    )
    num = next(s for s in scores if s.qtype == QType.NUMERICAL.value)
    assert num.gt_answer_true == 2
    assert num.true_source == "questions_pdf_text"
    assert num.true_match == (num.our_count == 2)


@requires_loft
def test_numerical_true_answer_mismatch_guard():
    """A key whose question text doesn't match is refused (null + note), never mis-anchored."""
    gt = load_scene(LOFT_DIR)
    with open(QUESTIONS_JSON, encoding="utf-8") as fh:
        data = json.load(fh)
    entry = next(e for e in data if e["scene"] == "loft")
    tampered = {"scenes": {"loft": {"question_raw": "How many zebras stand here?", "answer": 2}}}
    scores = GB.score_scene(
        gt, entry["questions"], questions_dir=QUESTIONS_DIR, answers=tampered,
        drive_if=False,
    )
    num = next(s for s in scores if s.qtype == QType.NUMERICAL.value)
    assert num.gt_answer_true is None
    assert num.true_match is None
    assert num.true_source == ""
    assert "answer-key question mismatch" in num.note


def test_aggregate_true_accuracy_math():
    """true_accuracy is the mean of true_match over keyed rows; keyless rows excluded."""
    from core.runner.gt_battery import GTQuestionScore, aggregate

    rows = [
        GTQuestionScore(scene="a", qtype=QType.NUMERICAL.value, question="q",
                        our_count=2, gt_answer_true=2, true_match=True,
                        true_source="questions_pdf_text"),
        GTQuestionScore(scene="b", qtype=QType.NUMERICAL.value, question="q",
                        our_count=2, gt_answer_true=2, true_match=True,
                        true_source="questions_pdf_text"),
        GTQuestionScore(scene="c", qtype=QType.NUMERICAL.value, question="q",
                        our_count=1, gt_answer_true=3, true_match=False,
                        true_source="questions_pdf_text"),
        GTQuestionScore(scene="d", qtype=QType.NUMERICAL.value, question="q",
                        our_count=5),  # keyless: excluded from true_accuracy
    ]
    agg = aggregate(rows)["numerical"]
    assert agg["n_with_true_answer"] == 3
    assert agg["true_accuracy"] == 0.6667
    # determinism rate is renamed and retained; the old name is gone
    assert "pipeline_determinism_rate" in agg
    assert "exact_match_rate_pipeline" not in agg


def test_aggregate_true_accuracy_none_without_key():
    """No keyed rows -> true_accuracy None, n_with_true_answer 0 (topline reads n/a)."""
    from core.runner.gt_battery import GTQuestionScore, aggregate

    rows = [
        GTQuestionScore(scene="a", qtype=QType.NUMERICAL.value, question="q", our_count=2),
    ]
    agg = aggregate(rows)["numerical"]
    assert agg["n_with_true_answer"] == 0
    assert agg["true_accuracy"] is None


@requires_loft
@pytest.mark.slow
def test_score_scene_if_produces_two_numbers():
    gt = load_scene(LOFT_DIR)
    with open(QUESTIONS_JSON, encoding="utf-8") as fh:
        data = json.load(fh)
    entry = next(e for e in data if e["scene"] == "loft")
    scores = GB.score_scene(gt, entry["questions"], questions_dir=QUESTIONS_DIR,
                            drive_if=True)
    ifs = [s for s in scores if s.qtype == QType.INSTRUCTION_FOLLOWING.value]
    assert len(ifs) == 2
    for s in ifs:
        # two separate numbers, never a composite
        assert s.frechet_m is not None
        assert s.coverage_1m is not None
        assert s.gt_n_waypoints > 0
        assert s.our_n_waypoints is not None


@requires_loft
@pytest.mark.slow
def test_run_gt_battery_emits_report(tmp_path):
    """End-to-end: discover loft under the sample Unity root, write a report."""
    scores, missing = GB.run_gt_battery(
        UNITY_SAMPLE_ROOT,
        questions_path=QUESTIONS_JSON,
        questions_dir=QUESTIONS_DIR,
        scenes=["loft"],
        drive_if=True,
    )
    assert len(scores) == 5
    assert "loft" not in missing
    md_path, json_path = GB.write_report(scores, missing, tmp_path)
    assert md_path.exists() and json_path.exists()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["n_questions"] == 5
    assert "loft" in payload["scenes"]
    assert "numerical" in payload["aggregate"]
    # report mentions the honest circularity caveat
    assert "Circularity" in md_path.read_text(encoding="utf-8")
    # provenance stamp is present and names this tool (meth-F7/F8)
    assert payload["provenance"]["tool"] == "gt_battery"
    # IF rows carry the per-leg rubric geometry + outcomes
    if_rows = [s for s in payload["scores"] if s["qtype"] == QType.INSTRUCTION_FOLLOWING.value]
    assert if_rows
    assert all(r["leg_goals"] is not None for r in if_rows)
    assert all(r["leg_outcomes"] is not None for r in if_rows)
    a_leg = next(o for r in if_rows for o in r["leg_outcomes"])
    assert {"i", "kind", "goal", "reached_in_order", "threaded"} <= set(a_leg)


@requires_loft
def test_run_gt_battery_reports_missing_scene(tmp_path):
    """A questions.json scene with no folder is reported missing, not crashed."""
    scores, missing = GB.run_gt_battery(
        UNITY_SAMPLE_ROOT,
        questions_path=QUESTIONS_JSON,
        questions_dir=QUESTIONS_DIR,
        scenes=["office_1"],  # not in the sample data
        drive_if=False,
    )
    assert scores == []
    assert "office_1" in missing


@requires_full_unity
@pytest.mark.slow
def test_full_battery_smoke_two_scenes(tmp_path):
    """Full-dataset smoke on 2 real scenes: scores present, aggregate shapes hold."""
    scores, missing = GB.run_gt_battery(
        FULL_UNITY_ROOT,
        questions_path=QUESTIONS_JSON,
        questions_dir=QUESTIONS_DIR,
        scenes=["loft", "office_1"],
        drive_if=True,
    )
    assert "loft" not in missing and "office_1" not in missing
    assert {s.scene for s in scores} == {"loft", "office_1"}
    agg = GB.aggregate(scores)
    # new aggregate keys are present and well-formed
    assert "match_method_breakdown" in agg["object_reference"]
    assert "n_unaligned_scenes" in agg["instruction_following"]
    assert "scenegraph_agreement_rate" in agg["numerical"]
    # every scored OR question carries a match method label
    for s in scores:
        if s.qtype == QType.OBJECT_REFERENCE.value:
            assert s.match_method in ("exact", "fuzzy", "relation", "unique", "none")
    # report writes without error and round-trips
    md_path, json_path = GB.write_report(scores, missing, tmp_path)
    assert md_path.exists() and json_path.exists()


@requires_full_unity
@pytest.mark.slow
def test_full_battery_if_alignment_reports_residual(tmp_path):
    """Instruction-following scores carry a per-scene fit residual when aligned."""
    scores, _ = GB.run_gt_battery(
        FULL_UNITY_ROOT,
        questions_path=QUESTIONS_JSON,
        questions_dir=QUESTIONS_DIR,
        scenes=["loft"],
        drive_if=True,
    )
    ifs = [s for s in scores if s.qtype == QType.INSTRUCTION_FOLLOWING.value]
    assert ifs
    # loft aligns cleanly (endpoints co-locate), so a finite residual is recorded
    assert any(s.fit_residual_m is not None for s in ifs)


@requires_loft
@pytest.mark.slow
def test_cli_groundtruth_flag_delegates(tmp_path):
    """`battery --groundtruth` switches to v2 and writes the gt report."""
    from core.runner import battery

    rc = battery.main([
        "--groundtruth", str(UNITY_SAMPLE_ROOT),
        "--questions", str(QUESTIONS_JSON),
        "--scenes", "loft",
        "--out", str(tmp_path),
    ])
    assert rc == 0
    assert (tmp_path / "gt_battery_report.md").exists()
