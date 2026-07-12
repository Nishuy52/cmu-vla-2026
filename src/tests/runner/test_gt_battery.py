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
