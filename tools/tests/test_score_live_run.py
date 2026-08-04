"""Tests for tools/score_live_run.py — the pure/deterministic helpers only.

No bag files or GT dataset needed: covers decimation, question matching,
marker-frame mapping, offline-row pairing/headline extraction, and the
incremental-rerun merge — the logic most likely to silently drift wrong
without a real capture to eyeball against.
"""
from __future__ import annotations

import json

import numpy as np

from core.groundtruth.scoring import Frame2D
from tools.score_live_run import (
    DEFAULT_BASELINE_DIR,
    _capture_completeness_issues,
    _debug_dir_for_run,
    _decimate_xy,
    _load_offline_index,
    _marker_aabb_in_object_frame,
    _match_question,
    _merge_key,
    _merge_with_existing,
    _offline_headline,
    _squash,
    main,
)


def _write_instance_index(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")


def test_decimate_xy_keeps_endpoints_and_drops_close_points():
    pts = [(0.0, 0.0), (0.01, 0.0), (0.02, 0.0), (0.2, 0.0), (0.2, 0.0)]
    out = _decimate_xy(pts, min_move=0.05)
    assert out.shape[0] == 2
    assert tuple(out[0]) == (0.0, 0.0)
    assert tuple(out[-1]) == (0.2, 0.0)


def test_decimate_xy_empty_input():
    assert _decimate_xy([], min_move=0.05).shape == (0, 2)


def test_decimate_xy_single_point():
    out = _decimate_xy([(1.0, 2.0)], min_move=0.05)
    assert out.shape == (1, 2)
    assert tuple(out[0]) == (1.0, 2.0)


def test_squash_ignores_whitespace_and_case():
    assert _squash("How Many  Chairs?") == _squash("how many chairs?")


def test_match_question_exact_and_whitespace_tolerant():
    candidates = ["Find the vase on the cabinet.", "Go to the lamp."]
    assert _match_question("Find the vase on the cabinet.", candidates) == 0
    assert _match_question("  find THE vase on the cabinet.  ", candidates) == 0
    assert _match_question("Go to the lamp.", candidates) == 1
    assert _match_question("no such question", candidates) is None


def test_marker_aabb_no_frame_is_identity():
    marker = {"cx": 1.0, "cy": 2.0, "cz": 0.5, "sx": 2.0, "sy": 1.0, "sz": 1.0}
    a_min, a_max = _marker_aabb_in_object_frame(marker, None)
    np.testing.assert_allclose(a_min, [0.0, 1.5, 0.0])
    np.testing.assert_allclose(a_max, [2.0, 2.5, 1.0])


def test_marker_aabb_translation_only_frame_shifts_xy_not_z():
    marker = {"cx": 0.0, "cy": 0.0, "cz": 0.75, "sx": 1.0, "sy": 1.0, "sz": 0.5}
    frame = Frame2D(theta=0.0, t=np.array([5.0, -3.0]))
    a_min, a_max = _marker_aabb_in_object_frame(marker, frame)
    np.testing.assert_allclose(a_min, [4.5, -3.5, 0.5])
    np.testing.assert_allclose(a_max, [5.5, -2.5, 1.0])


def test_marker_aabb_rotated_frame_rehulls_to_axis_aligned():
    # 90 degree rotation of a non-square footprint must swap the XY half-extents.
    marker = {"cx": 0.0, "cy": 0.0, "cz": 0.0, "sx": 2.0, "sy": 0.4, "sz": 1.0}
    frame = Frame2D(theta=np.pi / 2, t=np.array([0.0, 0.0]))
    a_min, a_max = _marker_aabb_in_object_frame(marker, frame)
    np.testing.assert_allclose(a_max - a_min, [0.4, 2.0, 1.0], atol=1e-9)


def test_offline_headline_per_qtype():
    assert _offline_headline("numerical", {"true_match": True}) == 1.0
    assert _offline_headline("numerical", {"true_match": False}) == 0.0
    assert _offline_headline("numerical", {"true_match": None}) is None
    assert _offline_headline("object_reference", {"iou": 0.42}) == 0.42
    assert _offline_headline("instruction_following", {"rubric_score": 0.75}) == 0.75
    assert _offline_headline("numerical", None) is None


def test_load_offline_index_keys_by_scene_qtype_question(tmp_path):
    path = tmp_path / "results.json"
    path.write_text(
        json.dumps(
            {
                "scores": [
                    {"scene": "a", "qtype": "numerical", "question": "Q1", "true_match": True},
                ]
            }
        ),
        encoding="utf-8",
    )
    idx = _load_offline_index(path)
    assert idx[("a", "numerical", "Q1")]["true_match"] is True


def test_load_offline_index_missing_file_returns_empty(tmp_path):
    assert _load_offline_index(tmp_path / "nope.json") == {}


def test_merge_with_existing_preserves_other_runs(tmp_path):
    existing = {
        "rows": [
            {"scene": "s", "qdir": "nume", "headline_live": 1.0},
            {"scene": "s", "qdir": "obje", "headline_live": None},
        ]
    }
    (tmp_path / "scores.json").write_text(json.dumps(existing), encoding="utf-8")

    new_rows = [{"scene": "s", "qdir": "obje", "headline_live": 0.5}]
    merged = _merge_with_existing(tmp_path, new_rows)

    by_qdir = {r["qdir"]: r for r in merged}
    assert by_qdir["nume"]["headline_live"] == 1.0  # untouched
    assert by_qdir["obje"]["headline_live"] == 0.5  # replaced


def test_merge_with_existing_no_prior_file(tmp_path):
    new_rows = [{"scene": "s", "qdir": "nume", "headline_live": 1.0}]
    assert _merge_with_existing(tmp_path, new_rows) == new_rows


def test_merge_with_existing_corrupt_file_falls_back_to_fresh(tmp_path):
    (tmp_path / "scores.json").write_text("not json", encoding="utf-8")
    new_rows = [{"scene": "s", "qdir": "nume", "headline_live": 1.0}]
    assert _merge_with_existing(tmp_path, new_rows) == new_rows


def test_merge_with_existing_keeps_both_same_type_questions_in_one_scene(tmp_path):
    # Every scene has TWO object_reference questions, both scored under the
    # same qdir ("obje") — the merge key must not collapse them (#97).
    new_rows = [
        {
            "scene": "s", "qdir": "obje", "question": "Find the red chair.",
            "run_dir": "reports/s/obje", "headline_live": 0.2,
        },
        {
            "scene": "s", "qdir": "obje", "question": "Find the blue lamp.",
            "run_dir": "reports/s/obje", "headline_live": 0.9,
        },
    ]
    merged = _merge_with_existing(tmp_path, new_rows)
    assert len(merged) == 2
    by_question = {r["question"]: r["headline_live"] for r in merged}
    assert by_question["Find the red chair."] == 0.2
    assert by_question["Find the blue lamp."] == 0.9


def test_merge_with_existing_rescore_replaces_same_question_row(tmp_path):
    existing = {
        "rows": [
            {
                "scene": "s", "qdir": "obje", "question": "Find the red chair.",
                "run_dir": "reports/s/obje", "headline_live": 0.2,
            },
            {
                "scene": "s", "qdir": "obje", "question": "Find the blue lamp.",
                "run_dir": "reports/s/obje", "headline_live": 0.9,
            },
        ]
    }
    (tmp_path / "scores.json").write_text(json.dumps(existing), encoding="utf-8")

    # Re-scoring only the "red chair" run must replace just that row and
    # leave the "blue lamp" row (from a different question, same scene/qdir)
    # untouched.
    new_rows = [
        {
            "scene": "s", "qdir": "obje", "question": "Find the red chair.",
            "run_dir": "reports/s/obje", "headline_live": 0.6,
        },
    ]
    merged = _merge_with_existing(tmp_path, new_rows)
    by_question = {r["question"]: r["headline_live"] for r in merged}
    assert len(merged) == 2
    assert by_question["Find the red chair."] == 0.6  # replaced
    assert by_question["Find the blue lamp."] == 0.9  # untouched


def test_merge_key_squashes_question_case_and_whitespace():
    # (#106) The merge key must be at least as tolerant as _match_question —
    # otherwise identical questions that differ only in case/whitespace
    # (a launcher change, a re-encode, a different adapter build) hash to
    # different keys and silently duplicate the row instead of replacing it.
    base = {"scene": "s", "qdir": "inst", "question": "Find the red chair."}
    variants = [
        {"scene": "s", "qdir": "inst", "question": "find the red chair."},
        {"scene": "s", "qdir": "inst", "question": "  Find the red chair.  "},
        {"scene": "s", "qdir": "inst", "question": "FIND THE RED CHAIR."},
        {"scene": "s", "qdir": "inst", "question": "Find   the red\tchair."},
    ]
    base_key = _merge_key(base)
    for v in variants:
        assert _merge_key(v) == base_key


def test_merge_with_existing_collapses_question_case_and_whitespace_variants(tmp_path):
    # (#106) direct end-to-end reproduction via _merge_with_existing: a
    # re-score of the same question with drifted case/whitespace must
    # replace, not duplicate.
    existing = {
        "rows": [
            {
                "scene": "s", "qdir": "inst", "question": "Find the red chair.",
                "run_dir": "reports/s/inst", "headline_live": 0.2,
            },
        ]
    }
    (tmp_path / "scores.json").write_text(json.dumps(existing), encoding="utf-8")

    new_rows = [
        {
            "scene": "s", "qdir": "inst", "question": "find the red chair.  ",
            "run_dir": "reports/s/inst", "headline_live": 0.8,
        },
    ]
    merged = _merge_with_existing(tmp_path, new_rows)
    assert len(merged) == 1
    assert merged[0]["headline_live"] == 0.8  # replaced, not appended


def test_merge_key_no_question_fallback_still_uses_run_dir():
    # The "no question matched" branch must stay keyed on run_dir (unique,
    # canonical) rather than being squashed itself.
    row = {"scene": "s", "qdir": "inst", "question": None, "run_dir": "reports/s/inst"}
    assert _merge_key(row) == ("s", "inst", "<no-question:reports/s/inst>")


def test_write_report_records_offline_baseline_source(tmp_path):
    # (#139) scores.json must record which fixed baseline file the `offline`
    # column came from, so it cannot be mistaken for a per-run recomputation.
    from tools.score_live_run import write_report

    offline_path = tmp_path / "some_baseline" / "gt_battery_results.json"
    rows = [{"scene": "s", "qdir": "inst", "question": "Q1", "headline_live": 0.5}]
    _md, json_path = write_report(rows, tmp_path / "out", offline_results_path=offline_path)

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["offline_baseline_source"] == str(offline_path)


def test_write_report_excludes_rubric_excluded_rows_from_mean_and_shows_unevaluable(tmp_path):
    """Issue #165 (live counterpart of #162's row-rendering defect): a run whose
    IF rubric geometry yielded zero evaluable legs must have ``headline_live: None``
    (never the misleading 0.0), must render as "unevaluable" in the scores.md table
    (not "n/a", not "0.0000"), and must be excluded from the per-type mean line."""
    from tools.score_live_run import write_report

    rows = [
        {
            "scene": "arabic_room", "qdir": "inst", "qtype": "instruction_following",
            "question": "stool/table q", "headline_live": None,
            "live": {"rubric_excluded": True, "rubric_score": None},
        },
        {
            "scene": "loft", "qdir": "inst", "qtype": "instruction_following",
            "question": "Q2", "headline_live": 0.6,
            "live": {"rubric_excluded": False, "rubric_score": 0.6},
        },
    ]
    offline_path = tmp_path / "baseline" / "gt_battery_results.json"
    md_path, json_path = write_report(rows, tmp_path / "out", offline_results_path=offline_path)

    md = md_path.read_text(encoding="utf-8")
    assert "unevaluable" in md
    assert "0.0000" not in md  # never rendered as a scored zero
    # mean over instruction_following counts only the one truly-scored row
    assert "| instruction_following | 2 | 1 | 1 | 0.6000 |" in md

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["rows"][0]["headline_live"] is None


def test_score_instruction_following_run_zero_evaluable_legs_excluded(monkeypatch, tmp_path):
    """Row-level unit test for the score_instruction_following_run fix: when
    ``score_instruction_rubric`` reports ``n_legs == 0`` (every leg unevaluable,
    issue #162), the returned dict must carry ``headline: None`` +
    ``rubric_excluded: True`` -- not the ``rubric_score = 0.0`` the scorer computes
    internally by construction for that case."""
    import numpy as np

    import tools.score_live_run as SLR

    class _Rub:
        rubric_score = 0.0
        ordered_leg_credit = 0.0
        n_legs = 0
        n_legs_reached_in_order = 0
        n_threading_violations = 0
        n_avoid_violations = 0
        driven_n_poses = 3
        frechet_m = None
        coverage_1m = None
        threading_details: list = []
        avoid_details: list = []

    class _Ctx:
        if_texts = ["Go to the stool near the table."]
        frame = SLR.S.Frame2D(theta=0.0, t=np.array([0.0, 0.0]))
        gt = object()
        idx = object()
        spawn_xy = None
        scene = "arabic_room"

    class _Capture:
        odom_xy = np.zeros((3, 2))
        odom_xy_raw_n = 3

    monkeypatch.setattr(SLR.GB, "_IF_TRAJ_INDEX", {0: 0})
    monkeypatch.setattr(SLR.GB, "_if_rubric_geometry", lambda *a, **k: ([], [], [], [], []))
    monkeypatch.setattr(SLR.S, "score_instruction_rubric", lambda *a, **k: _Rub())

    result = SLR.score_instruction_following_run(
        _Ctx(), "Go to the stool near the table.", _Capture(),
        questions_dir=tmp_path,
    )
    assert result["headline"] is None
    assert result["rubric_score"] is None
    assert result["ordered_leg_credit"] is None
    assert result["rubric_excluded"] is True
    assert "excluded" in result["note"]


def test_score_instruction_following_run_emits_live_leg_structures(monkeypatch, tmp_path):
    """Issue #178: the live block must carry leg_goals/leg_outcomes/leg_probe,
    mirroring the offline block's shapes/field names, consistent with the live
    scalars (n_legs, n_legs_reached_in_order) the scorer already derives from
    the same per-leg outcomes."""
    import numpy as np

    import tools.score_live_run as SLR

    class _Outcome:
        def __init__(self, index, kind, goal_xy, reached_in_order, threaded, pass_by, tol_used):
            self.index = index
            self.kind = kind
            self.goal_xy = goal_xy
            self.reached_in_order = reached_in_order
            self.threaded = threaded
            self.pass_by = pass_by
            self.tol_used = tol_used

    _leg_outcomes = [
        _Outcome(0, "goto", (1.0, 2.0), True, True, False, 0.5),
        _Outcome(1, "goto", (3.0, 4.0), False, True, False, 0.5),
    ]

    class _Rub:
        rubric_score = 0.5
        ordered_leg_credit = 0.5
        n_legs = 2
        n_legs_reached_in_order = 1
        n_threading_violations = 0
        n_avoid_violations = 0
        driven_n_poses = 3
        frechet_m = None
        coverage_1m = None
        threading_details: list = []
        avoid_details: list = []
        leg_outcomes = _leg_outcomes

    class _Ctx:
        if_texts = ["Go to the stool, then go to the table."]
        frame = SLR.S.Frame2D(theta=0.0, t=np.array([0.0, 0.0]))
        gt = object()
        idx = object()
        spawn_xy = None
        scene = "arabic_room"

    class _Capture:
        odom_xy = np.zeros((3, 2))
        odom_xy_raw_n = 3

    leg_goals = [("goto", (1.0, 2.0)), ("goto", (3.0, 4.0))]
    leg_instance_ids = [(11,), (22,)]

    monkeypatch.setattr(SLR.GB, "_IF_TRAJ_INDEX", {0: 0})
    monkeypatch.setattr(
        SLR.GB, "_if_rubric_geometry",
        lambda *a, **k: (leg_goals, [], [], leg_instance_ids, [None, None]),
    )
    monkeypatch.setattr(SLR.S, "score_instruction_rubric", lambda *a, **k: _Rub())

    result = SLR.score_instruction_following_run(
        _Ctx(), "Go to the stool, then go to the table.", _Capture(),
        questions_dir=tmp_path,
    )

    assert len(result["leg_outcomes"]) == result["n_legs"]
    assert (
        sum(o["reached_in_order"] for o in result["leg_outcomes"])
        == result["n_legs_reached_in_order"]
    )
    assert result["leg_goals"] == [
        ["goto", [1.0, 2.0]], ["goto", [3.0, 4.0]],
    ]
    assert result["leg_outcomes"][0] == {
        "i": 0, "kind": "goto", "goal": [1.0, 2.0],
        "reached_in_order": True, "threaded": True,
        "pass_by": False, "tol_used": 0.5,
    }
    assert len(result["leg_probe"]) == 2
    probe0 = result["leg_probe"][0]
    assert probe0["our_goal"] == [1.0, 2.0]
    assert probe0["our_instance_id"] == [11]
    assert "min_dist_driven_to_goal_m" in probe0
    assert "dist_goal_to_gt_traj_m" in probe0


def test_main_requires_out_when_target_given(tmp_path, capsys):
    # A targeted invocation (single run dir or non-default root) must never
    # silently fall back to writing into the committed DEFAULT_BASELINE_DIR
    # (#96) — it must error and tell the caller to pass --out.
    run_dir = tmp_path / "some_scene" / "inst"
    (run_dir / "bag").mkdir(parents=True)

    before = DEFAULT_BASELINE_DIR / "scores.json"
    before_bytes = before.read_bytes() if before.is_file() else None

    rc = main([str(run_dir)])

    assert rc != 0
    after_bytes = before.read_bytes() if before.is_file() else None
    assert after_bytes == before_bytes  # baseline dir untouched
    assert "--out" in capsys.readouterr().out


# --------------------------------------------------------------------------- #158 capture-completeness


def _make_batch_job(tmp_path, slot_keyframes_instances):
    """Build a fake harvested batch job dir and return its slots' run dirs.

    ``slot_keyframes_instances`` maps slot name -> (keyframes_processed,
    total_instances) for that slot's final instance_index.jsonl record.
    Mirrors harvest_verify.sh's batch layout: captures/<slot>/<scene>/<qdir>
    next to debug/<slot>/instance_index.jsonl.
    """
    out = tmp_path / "job"
    run_dirs = {}
    for slot, (keyframes, instances) in slot_keyframes_instances.items():
        run_dir = out / "captures" / slot / "scene" / "inst"
        run_dir.mkdir(parents=True)
        _write_instance_index(
            out / "debug" / slot / "instance_index.jsonl",
            [
                {"keyframes_processed": max(1, keyframes // 2), "total_instances": 0},
                {"keyframes_processed": keyframes, "total_instances": instances},
            ],
        )
        run_dirs[slot] = run_dir
    return run_dirs


def test_debug_dir_for_run_batch_layout(tmp_path):
    run_dirs = _make_batch_job(tmp_path, {"3_loft_inst": (12, 0)})
    debug_dir = _debug_dir_for_run(run_dirs["3_loft_inst"])
    assert debug_dir == tmp_path / "job" / "debug" / "3_loft_inst"


def test_debug_dir_for_run_single_job_flat_layout(tmp_path):
    out = tmp_path / "job"
    run_dir = out / "captures" / "scene" / "inst"
    run_dir.mkdir(parents=True)
    (out / "debug").mkdir(parents=True)
    assert _debug_dir_for_run(run_dir) == out / "debug"


def test_debug_dir_for_run_no_captures_segment_returns_none(tmp_path):
    run_dir = tmp_path / "somewhere" / "scene" / "inst"
    run_dir.mkdir(parents=True)
    assert _debug_dir_for_run(run_dir) is None


def test_debug_dir_for_run_no_debug_dir_returns_none(tmp_path):
    run_dir = tmp_path / "job" / "captures" / "scene" / "inst"
    run_dir.mkdir(parents=True)
    assert _debug_dir_for_run(run_dir) is None


def test_capture_completeness_zero_instances_always_flagged(tmp_path):
    # Job 702061 slot 3: 12 keyframes, 0 instances over a full run — this
    # must be flagged unconditionally, not as a function of any threshold.
    run_dirs = _make_batch_job(
        tmp_path,
        {
            "0_slot": (140, 69),
            "1_slot": (150, 90),
            "2_slot": (140, 87),
            "3_starved": (12, 0),
        },
    )
    issues = _capture_completeness_issues(run_dirs["3_starved"])
    assert any("zero instances" in i for i in issues)
    assert any("#158" in i for i in issues)


def test_capture_completeness_healthy_run_not_flagged(tmp_path):
    # A job matching the shape of 702061's slots 4-9: keyframe counts vary
    # scene to scene but none falls far below the job's median, and every
    # slot built a non-trivial instance count. None of these must be flagged.
    run_dirs = _make_batch_job(
        tmp_path,
        {
            "4_office_1": (139, 87),
            "5_office_1": (125, 87),
            "6_office_2": (173, 32),
            "7_office_2": (165, 38),
            "8_studio": (148, 113),
            "9_studio": (166, 77),
        },
    )
    for slot, run_dir in run_dirs.items():
        assert _capture_completeness_issues(run_dir) == [], f"{slot} was flagged"


def test_capture_completeness_starved_keyframes_flagged_relative_to_median(tmp_path):
    # A slot that built SOME instances but processed far fewer keyframes
    # than its job siblings (job 702061 slot 2: 40 keyframes vs a ~145
    # median from the other slots) is still a capture-completeness issue,
    # not just the zero-instance case.
    run_dirs = _make_batch_job(
        tmp_path,
        {
            "0_slot": (140, 69),
            "1_slot": (150, 90),
            "2_starved_nonzero": (41, 39),
            "4_slot": (140, 87),
            "5_slot": (125, 87),
        },
    )
    issues = _capture_completeness_issues(run_dirs["2_starved_nonzero"])
    assert any("median" in i for i in issues)
    assert any("#158" in i for i in issues)


def test_capture_completeness_too_few_slots_skips_relative_check(tmp_path):
    # Below MIN_SLOTS_FOR_MEDIAN, "the job's median" isn't meaningful (a
    # 2-slot batch where one is legitimately half the other) — only the
    # unconditional zero-instance check should still apply.
    run_dirs = _make_batch_job(
        tmp_path,
        {
            "0_slot": (140, 69),
            "1_smaller_scene": (40, 10),
        },
    )
    assert _capture_completeness_issues(run_dirs["1_smaller_scene"]) == []


def test_capture_completeness_no_debug_dir_returns_no_issues(tmp_path):
    run_dir = tmp_path / "job" / "captures" / "scene" / "inst"
    run_dir.mkdir(parents=True)
    assert _capture_completeness_issues(run_dir) == []
