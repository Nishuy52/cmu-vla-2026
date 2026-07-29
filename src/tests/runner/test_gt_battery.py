"""Battery v2 (GT accuracy) smoke tests on the real loft sample scene."""
from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

from core.groundtruth import scoring as S
from core.groundtruth.loader import load_scene
from core.interfaces import QType
from core.perception.scene_index import BasicSceneIndex
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
    # free cell off the table's (1 m) AABB and the follower stops a step short. Issue #132:
    # the terminal standoff now correctly accounts for the costmap's own inflation radius
    # on top of the anchor's half-diagonal, so the stood-off goal (and the closest driven
    # approach) sits measurably farther out than the pre-#132 accounting gave (~1.16 m) --
    # a wide "<= 1.7" upper-bound-only assertion would pass unchanged if #132 were reverted,
    # so this pins a band that EXCLUDES the old, short figure as well as bounding the new
    # one: proof the whole route was driven (not spuriously truncated) AND that the standoff
    # fix is actually in effect, not just "close enough".
    term = np.array([4.0, 0.0])
    min_term = float(np.min(np.hypot(driven[:, 0] - term[0], driven[:, 1] - term[1])))
    assert 1.4 <= min_term <= 1.7, (
        f"driven trajectory closest approach {min_term:.3f} m outside the expected "
        "post-#132 band [1.4, 1.7] m (either truncated short of the terminal, or the "
        "standoff accounting regressed to its pre-#132 ~1.16 m figure)"
    )


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
    # report mentions the honest yardstick caveat (TRUE accuracy primary, meth-F4/F6)
    assert "Yardstick note" in md_path.read_text(encoding="utf-8")
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
            # "geometric" added with issue #92 fix 3: the geometry-grounded
            # fallback used when the referential-statement corpus yields zero
            # candidates for a question's target class (10 of 30 questions).
            assert s.match_method in (
                "exact", "fuzzy", "relation", "unique", "geometric", "none"
            )
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


# --------------------------------------------------------------------------- meth-F11
# IF frame-fit fallback: candidate-search correspondence selection. These use only
# synthetic endpoints/candidates (no dataset) so they run on the fast tier.


def _traj(end_xy):
    """A minimal 2-vertex GT-trajectory array (shared start at origin, given endpoint)."""
    return np.array([[0.0, 0.0, 0.75], [end_xy[0], end_xy[1], 0.75]], dtype=float)


def test_fit_candidates_rescues_terminal_mispick():
    """When the top terminal candidate is wrong, a lower-ranked one restores the fit.

    Two questions, true sim->object transform = identity. Question 0's terminal resolves
    correctly (top candidate right); question 1's TOP candidate is a decoy far from the
    true object, with the correct object at rank 1. The default top-pick fit would blow
    the gate; the candidate search recovers the identity fit at ~0 residual.
    """
    trajs = [_traj((1.0, 0.0)), _traj((5.0, 0.0))]
    # q0: single correct candidate at the endpoint; q1: decoy first, truth second.
    cand_lists = [
        [np.array([1.0, 0.0])],
        [np.array([1.5, 3.0]), np.array([5.0, 0.0])],
    ]
    res = GB._fit_if_frame_over_candidates(trajs, cand_lists, gate_m=1.0)
    assert res is not None
    frame, residual = res
    assert residual < 1e-6
    # identity: endpoints map onto themselves
    mapped = frame.apply(np.array([[5.0, 0.0]]))[0]
    assert np.allclose(mapped, [5.0, 0.0], atol=1e-6)


def test_fit_candidates_returns_none_when_distance_contradiction():
    """No candidate pairing clears the gate when the geometry is contradictory.

    Endpoints are 1.2 m apart (sim) but every candidate pairing is >= 3.3 m apart
    (object) — a distance a rigid transform cannot reconcile (the livingroom_3 shape).
    The two-point residual floor is (3.3-1.2)/2 = 1.05 m > gate, so the search declines.
    """
    trajs = [_traj((5.23, -2.98)), _traj((6.31, -2.48))]  # 1.20 m apart
    cand_lists = [
        [np.array([4.22, -2.11]), np.array([2.47, -2.08])],  # pillows
        [np.array([6.90, -3.99])],  # bowl, >= 3.3 m from any pillow
    ]
    res = GB._fit_if_frame_over_candidates(trajs, cand_lists, gate_m=1.0)
    assert res is None


def test_fit_candidates_needs_two_endpoints():
    """A single usable endpoint yields no rigid fit (yaw unidentifiable) -> None."""
    trajs = [_traj((1.0, 0.0)), None]
    cand_lists = [[np.array([1.0, 0.0])], []]
    assert GB._fit_if_frame_over_candidates(trajs, cand_lists, gate_m=1.0) is None


def test_terminal_goal_centroid_delegates_to_candidates():
    """``_terminal_goal_centroid`` returns exactly the first candidate (k=1 wrapper)."""
    from core.perception.scene_index import BasicSceneIndex

    gt = _synthetic_gt_scene(
        [("painting", 2.0, 1.0, 1.0, 0.4, 0.4, 0.4)], scene_name="syn"
    )
    idx = BasicSceneIndex(gt.instances)
    text = "Go to the painting."
    cands = GB._terminal_goal_candidates(text, idx, k=8)
    top = GB._terminal_goal_centroid(text, idx)
    if cands:
        assert np.allclose(top, cands[0])
    else:
        assert top is None


def test_livingroom_3_registered_as_data_unfittable():
    """livingroom_3 is registered (with a reason) in the DATA-confirmed unfittable set."""
    _DATA = GB._DATA_UNFITTABLE_IF_SCENES
    assert "livingroom_3" in _DATA and _DATA["livingroom_3"]


# --------------------------------------------------------------------------- #16
# Corrupt vs absent answer key: a present-but-unreadable key must be LOUD and must
# read differently in the report topline than a legitimately absent one — the
# TRUE-accuracy yardstick can't be allowed to vanish silently.


def test_load_answers_missing_is_silent(tmp_path, capsys):
    """An absent key file is a soft miss: None, no warning."""
    assert GB._load_answers(tmp_path / "does_not_exist.json") is None
    assert GB._load_answers(None) is None
    assert capsys.readouterr().err == ""


def test_load_answers_corrupt_warns_and_returns_none(tmp_path, capsys):
    """A present-but-unreadable key returns None but emits a loud stderr warning that
    distinguishes 'unreadable' from 'missing' — and never raises."""
    bad = tmp_path / "gt_answers_numerical.json"
    bad.write_text("{ this is not valid json ", encoding="utf-8")
    assert GB._load_answers(bad) is None
    err = capsys.readouterr().err
    assert "UNREADABLE" in err
    assert "yardstick" in err


def test_answer_key_status_classifies(tmp_path):
    """_answer_key_status separates ok / missing / unreadable."""
    assert GB._answer_key_status(None) == "missing"
    assert GB._answer_key_status(tmp_path / "nope.json") == "missing"
    bad = tmp_path / "bad.json"
    bad.write_text("not json", encoding="utf-8")
    assert GB._answer_key_status(bad) == "unreadable"
    good = tmp_path / "good.json"
    good.write_text('{"scenes": {}}', encoding="utf-8")
    assert GB._answer_key_status(good) == "ok"


def _numerical_row_without_true_answer():
    """A keyless numerical row -> n_with_true_answer == 0 (topline reads n/a)."""
    from core.runner.gt_battery import GTQuestionScore

    return GTQuestionScore(
        scene="loft", qtype=QType.NUMERICAL.value, question="How many chairs?",
        our_count=2,
    )


def test_report_topline_says_unreadable_not_absent(tmp_path):
    """With no keyed rows AND an unreadable key, the topline says UNREADABLE, not 'no key'."""
    scores = [_numerical_row_without_true_answer()]
    md_path, _ = GB.write_report(
        scores, [], tmp_path, answer_key_status="unreadable"
    )
    text = md_path.read_text(encoding="utf-8")
    assert "UNREADABLE" in text
    assert "no answer key" not in text


def test_report_topline_says_no_key_when_absent(tmp_path):
    """With no keyed rows and a missing key, the topline keeps the 'no answer key' wording."""
    scores = [_numerical_row_without_true_answer()]
    md_path, _ = GB.write_report(scores, [], tmp_path, answer_key_status="missing")
    text = md_path.read_text(encoding="utf-8")
    assert "no answer key" in text
    assert "UNREADABLE" not in text


def test_json_payload_carries_answer_key_status(tmp_path):
    """The status is surfaced in the JSON payload, not only the markdown (issue #16)."""
    scores = [_numerical_row_without_true_answer()]
    _, json_path = GB.write_report(scores, [], tmp_path, answer_key_status="unreadable")
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["answer_key_status"] == "unreadable"

    _, json_path2 = GB.write_report(
        scores, [], tmp_path / "b", answer_key_status="missing"
    )
    payload2 = json.loads((json_path2).read_text(encoding="utf-8"))
    assert payload2["answer_key_status"] == "missing"


def test_main_stdout_marks_unreadable_key(tmp_path, monkeypatch, capsys):
    """main()'s stdout summary distinguishes an unreadable key from an absent one (issue #16)."""
    scores = [_numerical_row_without_true_answer()]
    monkeypatch.setattr(GB, "run_gt_battery", lambda *a, **k: (scores, []))

    monkeypatch.setattr(GB, "_answer_key_status", lambda p: "unreadable")
    rc = GB.main(["--groundtruth", str(tmp_path), "--out", str(tmp_path / "u")])
    assert rc == 0
    assert "num_true=n/a[UNREADABLE]" in capsys.readouterr().out

    monkeypatch.setattr(GB, "_answer_key_status", lambda p: "missing")
    rc = GB.main(["--groundtruth", str(tmp_path), "--out", str(tmp_path / "m")])
    assert rc == 0
    out = capsys.readouterr().out
    assert "num_true=n/a " in out
    assert "UNREADABLE" not in out


# --------------------------------------------------------------------------- #15
# The write_report `cal` parameter was never wired from main() (provenance always
# stamped the default calibration). gt_battery has no non-default calibration path,
# so the param was dropped: the default fallback is the single documented path.


def test_write_report_has_no_cal_param():
    """The dead `cal` parameter is gone from write_report's signature (#15)."""
    import inspect

    params = inspect.signature(GB.write_report).parameters
    assert "cal" not in params


def test_write_report_stamps_default_calibration(tmp_path):
    """Provenance still records the (default) calibration identity without a cal param."""
    scores = [_numerical_row_without_true_answer()]
    _, json_path = GB.write_report(scores, [], tmp_path)
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    prov = payload["provenance"]
    assert prov["tool"] == "gt_battery"
    # default calibration is stamped via collect_provenance's own fallback
    assert prov["calibration_sha1"] is not None
    assert prov["calibration"] is not None


# --------------------------------------------------------------------------- IF-F2 walls
# Interior-wall derivation off a scene's traversable_area.ply (mirror costmap otherwise
# has object obstacles + an outer boundary but no interior walls, so planned routes can
# cut through where real walls are). Pure-geometry rasterization tests need no dataset.


def _grid_points(x0, x1, y0, y1, n=30):
    xs = np.linspace(x0, x1, n)
    ys = np.linspace(y0, y1, n)
    gx, gy = np.meshgrid(xs, ys)
    return np.column_stack([gx.ravel(), gy.ravel()])


def test_derive_wall_cells_fully_covered_room_has_no_walls():
    """A traversable mesh covering the whole outer rectangle yields zero wall cells."""
    pts = _grid_points(0.05, 3.95, 0.05, 3.95)
    walls = GB._derive_wall_cells(pts, 0.0, 0.0, 4.0, 4.0, cell_m=0.1, dilate_cells=2)
    assert walls == set()


def test_derive_wall_cells_blocks_solid_wall_keeps_doorway_open():
    """A dividing wall band with a doorway gap: cells in the solid band are OBSTACLE,
    cells at the doorway centre are NOT — the rasterization must not seal doorways."""
    # Two 4x4 rooms separated by a 1 m wall band (x in [4, 5]) with a 0.9 m doorway gap
    # at y in [1.55, 2.45].
    pts = [_grid_points(0.05, 3.95, 0.05, 3.95), _grid_points(5.05, 8.95, 0.05, 3.95)]
    pts.append(_grid_points(4.05, 4.95, 1.55, 2.45, n=10))
    pts = np.vstack(pts)

    walls = GB._derive_wall_cells(pts, 0.0, 0.0, 9.0, 4.0, cell_m=0.1, dilate_cells=2)

    def cell(x, y):
        return (round(x / 0.1), round(y / 0.1))

    assert cell(4.5, 0.2) in walls, "solid wall band away from the doorway must block"
    assert cell(4.5, 3.8) in walls, "solid wall band away from the doorway must block"
    assert cell(4.5, 2.0) not in walls, "doorway centre must stay passable"


def test_synthetic_scene_extra_wall_cells_mark_terrain_obstacle():
    """SyntheticScene stamps extra_wall_cells as WALL_HEIGHT intensity in terrain_patch,
    same as the existing boundary/doorway wall logic."""
    from core.mocks.synthetic_scene import FLOOR_SPACING, Room, SyntheticScene, WALL_HEIGHT

    ix, iy = round(2.0 / FLOOR_SPACING), round(2.0 / FLOOR_SPACING)
    sc = SyntheticScene(0, extra_wall_cells={(ix, iy)})
    sc.rooms = [Room(0, 0, 4, 4)]
    sc._split_x = None
    sc.doorway = None
    sc.objects = []

    patch = sc.terrain_patch()
    pts = patch.points
    near = pts[(np.abs(pts[:, 0] - 2.0) < 1e-6) & (np.abs(pts[:, 1] - 2.0) < 1e-6)]
    assert near.shape[0] == 1
    assert near[0, 3] == pytest.approx(WALL_HEIGHT, abs=1e-6)

    # A cell well away from the marked wall cell and the border stays free.
    away = pts[(np.abs(pts[:, 0] - 1.0) < 1e-6) & (np.abs(pts[:, 1] - 1.0) < 1e-6)]
    assert away.shape[0] == 1
    assert away[0, 3] == pytest.approx(0.0, abs=1e-6)


def test_drive_if_trajectory_routes_around_interior_wall():
    """Feeding wall_cells that block the direct line between two GOTO legs forces the
    planner around them — the driven path must not cross the blocked band, proving
    wall_cells actually reaches the costmap the planner reasons over."""
    from core.perception.scene_index import BasicSceneIndex
    from core.mocks.synthetic_scene import FLOOR_SPACING

    gt = _synthetic_gt_scene(
        [
            ("stool", -4.0, 0.0, 0.3, 0.4, 0.4, 0.6),
            ("table", 4.0, 0.0, 0.4, 1.0, 1.0, 0.8),
        ]
    )
    idx = BasicSceneIndex(gt.instances)
    text = "Go to the stool and then go to the table."

    # A solid wall band across x=0 (the straight-line path) spanning y in [-3, 3], with
    # no gap — the vehicle should never end up inside the band.
    wall_cells = set()
    for x in np.arange(-0.3, 0.31, FLOOR_SPACING):
        for y in np.arange(-3.0, 3.01, FLOOR_SPACING):
            wall_cells.add((round(x / FLOOR_SPACING), round(y / FLOOR_SPACING)))

    driven = GB._drive_if_trajectory(
        gt=gt, idx=idx, text=text, start_xy=(-4.0, 0.0), wall_cells=wall_cells
    )
    assert driven.shape[0] > 2
    inside_band = (np.abs(driven[:, 0]) <= 0.25) & (np.abs(driven[:, 1]) <= 2.95)
    assert not inside_band.any(), "driven path crossed the interior wall band"


def test_scene_wall_cells_none_without_frame():
    """No fitted sim->object frame -> walls unavailable (old boundary-only fallback)."""
    gt = _synthetic_gt_scene([("stool", 0.0, 0.0, 0.0, 0.4, 0.4, 0.4)], scene_name="syn")
    assert GB._scene_wall_cells(gt, None) is None


def test_scene_wall_cells_none_without_ply(tmp_path):
    """A fitted frame but no traversable_area.ply on disk -> walls unavailable."""
    from core.groundtruth import scoring as S

    gt = _synthetic_gt_scene(
        [("stool", 0.0, 0.0, 0.0, 0.4, 0.4, 0.4)], scene_name="no_such_scene_xyz"
    )
    frame = S.Frame2D(theta=0.0, t=np.array([0.0, 0.0]))
    assert (
        GB._scene_wall_cells(gt, frame, unity_scenes_ros2_root=tmp_path) is None
    )


# --------------------------------------------------------- issue #77 Stage 1 (border pad)


def test_scene_room_bounds_without_frame_matches_pad_only_footprint():
    """No fitted frame -> the pre-fix instance-AABB-plus-pad rectangle, unchanged."""
    gt = _synthetic_gt_scene([("stool", 0.0, 0.0, 0.0, 0.4, 0.4, 0.4)], scene_name="syn")
    assert GB._scene_room_bounds(gt, None, None) == GB._gt_footprint_bounds(gt, 1.5)


def test_scene_room_bounds_without_trajectories_matches_pad_only_footprint():
    """A fitted frame but no GT trajectories (``if_traj`` empty/None) -> unchanged
    fallback — trajectories are the fix's evidence, not the frame alone."""
    from core.groundtruth import scoring as S

    gt = _synthetic_gt_scene([("stool", 0.0, 0.0, 0.0, 0.4, 0.4, 0.4)], scene_name="syn")
    frame = S.Frame2D(theta=0.0, t=np.array([0.0, 0.0]))
    assert GB._scene_room_bounds(gt, frame, None) == GB._gt_footprint_bounds(gt, 1.5)
    assert GB._scene_room_bounds(gt, frame, []) == GB._gt_footprint_bounds(gt, 1.5)
    assert (
        GB._scene_room_bounds(gt, frame, [None, np.empty((0, 3))])
        == GB._gt_footprint_bounds(gt, 1.5)
    )


def test_scene_room_bounds_expands_to_cover_gt_trajectory(tmp_path):
    """A GT reference trajectory point far past the instance-AABB-plus-pad rectangle
    must widen the room bounds to (at least) cover it — the exact defect the audit
    found (arabic_room's GT trajectory ran to x=8.42 against a mirror border stamped
    at x=5.63 by the old instance-AABB-only pad). Deliberately scoped to the
    trajectory itself, not the scene's full traversable mesh (which can reach well
    past anywhere any GT trajectory goes and was found to perturb unrelated route
    geometry — see the function's docstring)."""
    from core.groundtruth import scoring as S

    gt = _synthetic_gt_scene([("stool", 0.0, 0.0, 0.0, 0.4, 0.4, 0.4)], scene_name="room77")
    frame = S.Frame2D(theta=0.0, t=np.array([0.0, 0.0]))
    # Instance footprint is [-0.2, 0.2] x [-0.2, 0.2]; old pad-only bound would be
    # [-1.7, 1.7] x [-1.7, 1.7]. Put a trajectory point well past that.
    if_traj = [np.array([[5.0, 0.0, 0.0], [0.0, 0.0, 0.0]])]

    x0, y0, x1, y1 = GB._scene_room_bounds(gt, frame, if_traj)
    old_x0, old_y0, old_x1, old_y1 = GB._gt_footprint_bounds(gt, 1.5)
    assert x1 > old_x1, "room bounds must widen to cover the far trajectory point"
    assert x1 >= 5.0 + 1.5 - 1e-9
    # Never shrinks relative to the old (instance-only) rectangle on the other sides.
    assert x0 <= old_x0 and y0 <= old_y0 and y1 >= old_y1


def test_scene_room_bounds_never_shrinks_when_trajectory_inside_old_pad():
    """A trajectory entirely inside the old instance-AABB-plus-pad rectangle must not
    shrink the bounds — the fix only ever grows the rectangle, never tightens it."""
    from core.groundtruth import scoring as S

    gt = _synthetic_gt_scene([("stool", 0.0, 0.0, 0.0, 0.4, 0.4, 0.4)], scene_name="room77b")
    frame = S.Frame2D(theta=0.0, t=np.array([0.0, 0.0]))
    if_traj = [np.array([[0.1, 0.1, 0.0], [-0.1, -0.1, 0.0]])]

    got = GB._scene_room_bounds(gt, frame, if_traj)
    assert got == GB._gt_footprint_bounds(gt, 1.5)


def test_synthetic_from_gt_room_bounds_override_moves_border_wall():
    """``room_bounds`` overrides the default instance-AABB-plus-pad rectangle used for
    the mirror's own border-wall stamp — the mechanism :func:`_scene_room_bounds`
    feeds into :func:`_synthetic_from_gt` (issue #77 Stage 1)."""
    gt = _synthetic_gt_scene([("stool", 0.0, 0.0, 0.0, 0.4, 0.4, 0.4)], scene_name="syn")

    sc_default = GB._synthetic_from_gt(gt)
    assert sc_default.rooms[0] == GB.Room(-1.7, -1.7, 1.7, 1.7)

    wide = (-6.0, -6.0, 6.0, 6.0)
    sc_wide = GB._synthetic_from_gt(gt, room_bounds=wide)
    assert sc_wide.rooms[0] == GB.Room(*wide)

    # A point just past the OLD (tight) border, inside the room by the old rectangle's
    # own logic, is FREE under the widened room but blocked (border wall) under the
    # default — proving room_bounds actually reaches the stamped costmap, not just the
    # Room dataclass.
    px, py = 3.0, 0.0
    assert sc_default._is_wall(px, py) or not sc_default._in_any_room(px, py)
    assert sc_wide._in_any_room(px, py) and not sc_wide._is_wall(px, py)


def test_synthetic_from_gt_zero_room_bounds_default_is_byte_identical():
    """``room_bounds=None`` (every pre-existing caller) reproduces the exact old
    instance-AABB-plus-pad rectangle — regression guard for byte-identical behaviour
    when Stage-1 room-bounds evidence is unavailable."""
    gt = _synthetic_gt_scene(
        [("stool", -4.0, 0.0, 0.0, 0.4, 0.4, 0.4), ("table", 4.0, 0.0, 0.0, 0.4, 0.4, 0.4)],
        scene_name="syn2",
    )
    sc = GB._synthetic_from_gt(gt)
    x0, y0, x1, y1 = GB._gt_footprint_bounds(gt, 0.0)
    assert sc.rooms[0] == GB.Room(x0 - 1.5, y0 - 1.5, x1 + 1.5, y1 + 1.5)


# ------------------------------------------------------ issue #77 Stage 1 (carve)


def test_carve_cells_along_trajectories_empty_without_frame_or_traj():
    """No fitted frame, or no trajectories, -> nothing carved."""
    from core.groundtruth import scoring as S

    frame = S.Frame2D(theta=0.0, t=np.array([0.0, 0.0]))
    traj = [np.array([[1.0, 2.0, 0.0]])]
    assert GB._carve_cells_along_trajectories(traj, None) == set()
    assert GB._carve_cells_along_trajectories(None, frame) == set()
    assert GB._carve_cells_along_trajectories([], frame) == set()
    assert GB._carve_cells_along_trajectories([None, np.empty((0, 3))], frame) == set()


def test_carve_cells_along_trajectories_covers_vehicle_radius_disc():
    """The carved set around a single trajectory point is exactly the
    VEHICLE_RADIUS_M disc on the FLOOR_SPACING lattice — cells inside are carved,
    a cell just past the radius is not."""
    from core.groundtruth import scoring as S
    from core.mocks.synthetic_scene import FLOOR_SPACING
    from core.nav.costmap import VEHICLE_RADIUS_M

    frame = S.Frame2D(theta=0.0, t=np.array([0.0, 0.0]))
    traj = [np.array([[0.0, 0.0, 0.0]])]
    carved = GB._carve_cells_along_trajectories(traj, frame)

    assert (0, 0) in carved, "the trajectory's own cell must be carved"
    r_cells = int(np.ceil(VEHICLE_RADIUS_M / FLOOR_SPACING))
    assert (r_cells, 0) in carved, "a cell exactly at the radius (axis-aligned) is carved"
    assert (r_cells + 5, 0) not in carved, "a cell well past the radius is not carved"


def test_synthetic_from_gt_carved_cells_removes_derived_wall_cell():
    """A cell present in BOTH ``wall_cells`` and ``carved_cells`` is not stamped wall —
    the carve wins over derived-wall geometry (a plain set difference upstream)."""
    from core.mocks.synthetic_scene import FLOOR_SPACING

    gt = _synthetic_gt_scene([("stool", 0.0, 0.0, 0.0, 0.4, 0.4, 0.4)], scene_name="syn77c")
    cell = (round(3.0 / FLOOR_SPACING), round(0.0 / FLOOR_SPACING))
    wall_cells = {cell}
    carved_cells = {cell}

    sc_uncarved = GB._synthetic_from_gt(gt, wall_cells=wall_cells)
    sc_carved = GB._synthetic_from_gt(gt, wall_cells=wall_cells, carved_cells=carved_cells)

    assert sc_uncarved._is_wall(3.0, 0.0), "sanity: the cell is wall without the carve"
    assert not sc_carved._is_wall(3.0, 0.0), "the carve must clear the derived-wall cell"


def test_synthetic_from_gt_carved_cells_clears_stamped_object_terrain():
    """A carved cell inside a solid object's footprint reads FREE (intensity 0) in the
    mirror terrain, while an un-carved cell in the SAME footprint stays solid — proving
    the carve is a per-cell terrain filter, not a whole-object removal."""
    from core.mocks.synthetic_scene import FLOOR_SPACING

    gt = _synthetic_gt_scene(
        [
            ("table", 0.0, 0.0, 0.0, 2.0, 2.0, 0.8),
            # Filler far away so "table" (2x2) isn't the scene's ENTIRE own footprint
            # bounds -- otherwise issue #53's room-scale-AABB skip would drop it from
            # stamping altogether (a fixture artifact, not what this test is about).
            ("filler", 20.0, 20.0, 0.0, 0.2, 0.2, 0.2),
        ],
        scene_name="syn77d",
    )
    # A carved cell near the object's centre, and an un-carved one near its edge —
    # both squarely inside its [-1, 1] x [-1, 1] footprint.
    carved_cell = (round(0.0 / FLOOR_SPACING), round(0.0 / FLOOR_SPACING))
    uncarved_probe = (0.8, 0.8)
    carved_cells = {carved_cell}

    sc = GB._synthetic_from_gt(gt, carved_cells=carved_cells)
    patch = sc.terrain_patch()
    pts = patch.points

    at_carved = pts[(np.abs(pts[:, 0] - 0.0) < 1e-6) & (np.abs(pts[:, 1] - 0.0) < 1e-6)]
    assert at_carved.shape[0] == 1
    assert at_carved[0, 3] == 0.0, "carved cell inside the object footprint must read free"

    at_uncarved = pts[
        (np.abs(pts[:, 0] - uncarved_probe[0]) < 1e-6)
        & (np.abs(pts[:, 1] - uncarved_probe[1]) < 1e-6)
    ]
    assert at_uncarved.shape[0] == 1
    assert at_uncarved[0, 3] > 0.0, "un-carved cell in the same footprint must stay solid"


def test_synthetic_from_gt_carved_cells_never_shrinks_instance_record_aabb():
    """The carve only touches the MIRROR TERRAIN (the costmap the planner BFS/A*
    consult) — the GT instance record itself (what the object IS, used by every other
    head/rubric path) is untouched."""
    from core.groundtruth import scoring as S

    gt = _synthetic_gt_scene(
        [
            ("table", 0.0, 0.0, 0.0, 2.0, 2.0, 0.8),
            ("filler", 20.0, 20.0, 0.0, 0.2, 0.2, 0.2),  # see the terrain-carve test above
        ],
        scene_name="syn77e",
    )
    frame = S.Frame2D(theta=0.0, t=np.array([0.0, 0.0]))
    carved_cells = GB._carve_cells_along_trajectories([np.array([[0.0, 0.0, 0.0]])], frame)
    sc = GB._synthetic_from_gt(gt, carved_cells=carved_cells)
    recs = {r.label: r for r in sc.instances()}
    assert set(recs) == {"table", "filler"}
    assert np.allclose(recs["table"].aabb_min, [-1.0, -1.0, -0.4])
    assert np.allclose(recs["table"].aabb_max, [1.0, 1.0, 0.4])


def test_synthetic_from_gt_zero_carved_cells_default_is_byte_identical():
    """``carved_cells=None`` (every pre-existing caller) reproduces the plain
    ``SyntheticScene`` — regression guard for byte-identical behaviour."""
    gt = _synthetic_gt_scene([("stool", 0.0, 0.0, 0.0, 0.4, 0.4, 0.4)], scene_name="syn77f")
    sc = GB._synthetic_from_gt(gt)
    assert type(sc) is GB.SyntheticScene, "no carve requested -> the plain base class"


def test_drive_if_trajectory_carve_opens_a_path_through_a_blocked_band():
    """Feeding a carved corridor through an otherwise-solid wall band lets the driven
    path cross straight through it — end-to-end proof the carve reaches the costmap
    the planner reasons over, mirroring test_drive_if_trajectory_routes_around_
    interior_wall's un-carved case (which must still detour)."""
    from core.perception.scene_index import BasicSceneIndex
    from core.mocks.synthetic_scene import FLOOR_SPACING

    gt = _synthetic_gt_scene(
        [
            ("stool", -4.0, 0.0, 0.3, 0.4, 0.4, 0.6),
            ("table", 4.0, 0.0, 0.4, 1.0, 1.0, 0.8),
        ]
    )
    idx = BasicSceneIndex(gt.instances)
    text = "Go to the stool and then go to the table."

    wall_cells = set()
    for x in np.arange(-0.3, 0.31, FLOOR_SPACING):
        for y in np.arange(-3.0, 3.01, FLOOR_SPACING):
            wall_cells.add((round(x / FLOOR_SPACING), round(y / FLOOR_SPACING)))

    # Carve a corridor straight across the band at y=0 (as if a GT trajectory drove
    # through there) — wide enough (a few cells either side of x=0) that the vehicle
    # can actually fit through, not just graze it.
    carved_cells = set()
    for x in np.arange(-0.3, 0.31, FLOOR_SPACING):
        for y in np.arange(-0.5, 0.51, FLOOR_SPACING):
            carved_cells.add((round(x / FLOOR_SPACING), round(y / FLOOR_SPACING)))

    driven = GB._drive_if_trajectory(
        gt=gt, idx=idx, text=text, start_xy=(-4.0, 0.0),
        wall_cells=wall_cells, carved_cells=carved_cells,
    )
    assert driven.shape[0] > 2
    crossed_band = (np.abs(driven[:, 0]) <= 0.25).any()
    assert crossed_band, "the carved corridor must let the driven path cross the band"


def test_score_scene_walls_false_skips_wall_derivation(monkeypatch):
    """``walls=False`` (the --no-walls escape) never calls the wall-derivation path."""
    calls = []
    monkeypatch.setattr(
        GB, "_scene_wall_cells", lambda *a, **k: calls.append(1) or None
    )
    gt = _synthetic_gt_scene(
        [("stool", -4.0, 0.0, 0.0, 0.4, 0.4, 0.4), ("table", 4.0, 0.0, 0.0, 0.4, 0.4, 0.4)],
        scene_name="syn",
    )
    GB.score_scene(gt, {"instruction_following": []}, walls=False)
    assert calls == [], "_scene_wall_cells must not be called when walls=False"


def test_cli_no_walls_flag_delegates(tmp_path, monkeypatch):
    """``--no-walls`` reaches run_gt_battery as walls=False."""
    seen = {}

    def _fake_run(*args, **kwargs):
        seen.update(kwargs)
        return [], []

    monkeypatch.setattr(GB, "run_gt_battery", _fake_run)
    rc = GB.main(["--groundtruth", str(tmp_path), "--out", str(tmp_path), "--no-walls"])
    assert rc == 1  # no scores from the stubbed run -> gt_battery reports none found
    assert seen.get("walls") is False


def test_cli_walls_default_on(tmp_path, monkeypatch):
    """Without --no-walls, walls defaults True through the CLI."""
    seen = {}

    def _fake_run(*args, **kwargs):
        seen.update(kwargs)
        return [], []

    monkeypatch.setattr(GB, "run_gt_battery", _fake_run)
    GB.main(["--groundtruth", str(tmp_path), "--out", str(tmp_path)])
    assert seen.get("walls") is True


# ----------------------------------------------------------------------- issue #52


def _score_scene_with_residual(tmp_path, monkeypatch, residual):
    """Drive ``score_scene`` with a fitted frame at the given fixed ``residual`` and
    capture what ``frame`` (or ``None``) ``_scene_wall_cells`` was actually called with —
    the observable proxy for whether wall derivation was declined or attempted."""
    from core.groundtruth import scoring as S

    gt = _synthetic_gt_scene(
        [("stool", -4.0, 0.0, 0.0, 0.4, 0.4, 0.4), ("table", 4.0, 0.0, 0.0, 0.4, 0.4, 0.4)],
        scene_name="syn52",
    )
    # A trajectory file must exist on disk to enter the alignment path; its content is
    # irrelevant because load_trajectory_ply is monkeypatched below.
    scene_dir = tmp_path / "syn52"
    scene_dir.mkdir()
    (scene_dir / "trajectory_q4.ply").write_bytes(b"")

    fixed_frame = S.Frame2D(theta=0.0, t=np.array([0.0, 0.0]))
    monkeypatch.setattr(
        GB.S, "load_trajectory_ply",
        lambda *a, **k: np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float),
    )
    monkeypatch.setattr(
        GB.S, "align_scene_trajectories", lambda pairs: (fixed_frame, residual)
    )
    calls = []
    monkeypatch.setattr(
        GB, "_scene_wall_cells",
        lambda gt_, frame_, **k: calls.append(frame_) or None,
    )
    GB.score_scene(
        gt, {"instruction_following": ["go to the stool"]},
        questions_dir=tmp_path, drive_if=False,
    )
    assert len(calls) == 1
    return calls[0]


def test_wall_derivation_declines_just_above_residual_gate(tmp_path, monkeypatch):
    """Residual just past WALL_FIT_MAX_RESIDUAL_M -> no frame handed to wall derivation
    (falls back to the boundary-only costmap), even though it still clears the looser
    scoring alignment gate (S._ALIGN_RESIDUAL_GATE_M = 1.0)."""
    from core.groundtruth import scoring as S

    residual = GB.WALL_FIT_MAX_RESIDUAL_M + 0.01
    assert residual <= S._ALIGN_RESIDUAL_GATE_M  # still trusted for scoring, not walls
    frame_seen = _score_scene_with_residual(tmp_path, monkeypatch, residual)
    assert frame_seen is None


def test_wall_derivation_proceeds_just_below_residual_gate(tmp_path, monkeypatch):
    """Residual just under WALL_FIT_MAX_RESIDUAL_M -> the fitted frame IS handed to wall
    derivation."""
    residual = GB.WALL_FIT_MAX_RESIDUAL_M - 0.01
    frame_seen = _score_scene_with_residual(tmp_path, monkeypatch, residual)
    assert frame_seen is not None


# ----------------------------------------------------------------------- issue #51


def test_if_rubric_geometry_corridor_leg_resolves_distinct_anchors():
    """A "between the two X" corridor leg (both anchors share a noun) must resolve
    DISTINCT instances for the rubric's gate, mirroring InstructionHead's own
    distinctness enforcement (`_resolve_leg_anchors`'s `used` set) — the old
    independent-per-anchor resolve collapsed both anchors onto the SAME top-ranked
    instance, producing a zero-width gate no real trajectory can ever cross even when
    the actually-driven route correctly threads the real (distinct-instance) gate."""
    from core.perception.scene_index import BasicSceneIndex

    gt = _synthetic_gt_scene(
        [
            ("column", -1.0, 1.0, 0.0, 0.4, 0.4, 2.0),  # column A (lower id)
            ("column", -1.0, -1.0, 0.0, 0.4, 0.4, 2.0),  # column B (higher id)
        ],
        scene_name="syn51",
    )
    idx = BasicSceneIndex(gt.instances)
    text = "Take the path between the two columns."
    leg_goals, corridor_gates, _, _, _ = GB._if_rubric_geometry(text, gt, idx)
    assert len(corridor_gates) == 1
    _, gate = corridor_gates[0]
    assert gate.width > 0.5, f"expected a real (non-degenerate) gate, got width={gate.width}"


def test_if_rubric_geometry_non_corridor_legs_unaffected():
    """Distinctness enforcement is scoped to a corridor leg's OWN two anchors; a plain
    GOTO leg's resolution is unchanged."""
    from core.perception.scene_index import BasicSceneIndex

    gt = _synthetic_gt_scene(
        [("stool", -4.0, 0.0, 0.0, 0.4, 0.4, 0.6)], scene_name="syn51b"
    )
    idx = BasicSceneIndex(gt.instances)
    leg_goals, corridor_gates, _, _, _ = GB._if_rubric_geometry("Go to the stool.", gt, idx)
    assert corridor_gates == []
    assert leg_goals == [("goto", (-4.0, 0.0))]


# ----------------------------------------------------------------------- issue #53


def test_architectural_room_scale_aabb_detects_wide_flat_object():
    """A floor-level GT instance whose footprint spans > ARCHITECTURAL_AABB_ROOM_FRACTION
    of the room in BOTH axes is flagged room-scale — the traced home_building_2 "wall"
    id 126 shape (x:[-10.07,6.07] y:[-1.84,14.05] in a 30x26 m room)."""
    amin = np.array([-10.07, -1.84, -0.20])
    amax = np.array([6.07, 14.05, 4.17])
    assert GB._is_architectural_room_scale_aabb(amin, amax, room_w=30.02, room_h=25.99)


def test_architectural_room_scale_aabb_spares_real_furniture():
    """A real furniture item never spans a large fraction of the room in BOTH axes —
    the traced hotel_room_2 bed-frame shape (0.38/0.28 of its room) sits just under the
    bar on the SHORT axis and must not be suppressed."""
    amin = np.array([-1.95, 0.46, -0.001])
    amax = np.array([1.09, 2.34, 0.696])
    room_w, room_h = (1.09 - -1.95) / 0.38, (2.34 - 0.46) / 0.28
    assert not GB._is_architectural_room_scale_aabb(amin, amax, room_w=room_w, room_h=room_h)


def test_architectural_room_scale_aabb_requires_both_axes():
    """A single-axis-spanning thin real wall panel (one dimension room-scale, the other
    genuinely thin) is NOT room-scale — only an AABB wide in BOTH axes is an aggregate."""
    amin = np.array([-4.0, -0.05, 0.0])
    amax = np.array([4.0, 0.05, 2.4])  # 8 m long, 0.1 m thick: a real wall panel
    assert not GB._is_architectural_room_scale_aabb(amin, amax, room_w=8.4, room_h=6.8)


def test_architectural_room_scale_aabb_spares_elevated_slab():
    """A room-scale slab that sits ABOVE the terrain slab cutoff (a real ceiling) is an
    overhang, not a floor obstacle — it was never the problem and must not be flagged."""
    from core.mocks.synthetic_scene import TERRAIN_SLAB_MAX_Z

    amin = np.array([-10.0, -10.0, TERRAIN_SLAB_MAX_Z + 0.1])
    amax = np.array([10.0, 10.0, TERRAIN_SLAB_MAX_Z + 0.5])
    assert not GB._is_architectural_room_scale_aabb(amin, amax, room_w=20.0, room_h=20.0)


def test_synthetic_from_gt_skips_room_scale_architectural_object():
    """`_synthetic_from_gt` must not stamp a room-scale architectural AABB as a solid
    box: two real anchors with a genuine gap between them, plus a "wall" GT instance
    whose raw AABB spans nearly the whole room (VLA-3D's room-shell-as-one-bbox
    pattern) — the gap between the anchors must stay free, not sealed by the room-scale
    box."""
    gt = _synthetic_gt_scene(
        [
            ("sofa", -2.0, 0.0, 0.0, 1.0, 1.0, 0.8),
            ("coffee table", 2.0, 0.0, 0.0, 1.0, 1.0, 0.5),
            # room-scale "wall" aggregate: spans nearly the whole 10x10 room, floor
            # level (cz well under TERRAIN_SLAB_MAX_Z), covering the gap between the
            # two anchors above.
            ("wall", 0.0, 0.0, 2.0, 9.0, 9.0, 4.0),
        ],
        scene_name="syn53",
    )
    sc = GB._synthetic_from_gt(gt)
    labels = [o.label for o in sc.objects]
    assert "wall" not in labels, "room-scale wall AABB must not be stamped as a solid box"
    assert {"sofa", "coffee table"} <= set(labels)

    # The gap between the two anchors (the corridor's real footprint) must read FREE in
    # the terrain mirror, not sealed by the room-scale box.
    patch = sc.terrain_patch()
    pts = patch.points
    mid = pts[(np.abs(pts[:, 0] - 0.0) < 1e-6) & (np.abs(pts[:, 1] - 0.0) < 1e-6)]
    assert mid.shape[0] == 1
    assert mid[0, 3] == 0.0, "corridor gap under a room-scale architectural AABB must be free"


def test_synthetic_from_gt_still_stamps_real_thin_wall():
    """A genuinely thin, localized wall panel (not room-scale) is unaffected — it still
    stamps as a solid obstacle."""
    gt = _synthetic_gt_scene(
        [
            ("sofa", -2.0, 0.0, 0.0, 1.0, 1.0, 0.8),
            ("wall", 0.0, 3.0, 1.0, 6.0, 0.15, 2.4),  # a real thin wall segment
        ],
        scene_name="syn53b",
    )
    sc = GB._synthetic_from_gt(gt)
    labels = [o.label for o in sc.objects]
    assert "wall" in labels, "a genuinely thin wall panel must still be stamped"


# ----------------------------------------------------------------------- issue #77 OBB raster


def _synthetic_gt_scene_with_obb(objs, scene_name="synthetic_obb"):
    """Like ``_synthetic_gt_scene`` but each entry is
    ``(label, cx, cy, cz, sx, sy, sz, heading)`` and the record's ADDITIVE
    ``obb_center/obb_extents/obb_heading`` fields are populated (mirroring what
    ``core.groundtruth.loader.parse_object_csv`` does), with ``aabb_min/aabb_max``
    the honest ``obb_to_aabb`` over-approximation -- i.e. a real loader-shaped
    GTScene, not a hand-rolled AABB-only one."""
    from core.groundtruth.loader import GTScene, obb_to_aabb
    from core.interfaces import InstanceRecord

    recs = []
    for i, (label, cx, cy, cz, sx, sy, sz, heading) in enumerate(objs):
        center = (cx, cy, cz)
        extents = np.array([sx, sy, sz], dtype=float)
        amin, amax = obb_to_aabb(center, extents, heading)
        recs.append(
            InstanceRecord(
                instance_id=i,
                label=label,
                score=1.0,
                n_obs=3,
                centroid=(amin + amax) / 2.0,
                aabb_min=amin,
                aabb_max=amax,
                obb_center=np.asarray(center, dtype=float),
                obb_extents=extents,
                obb_heading=float(heading),
            )
        )
    return GTScene(scene_name=scene_name, instances=recs, regions=[])


def test_synthetic_from_gt_zero_heading_is_byte_identical_to_aabb_stamp():
    """Regression guard: a GTScene whose instances all carry heading=0.0 (the
    common case -- most VLA-3D objects are axis-aligned) stamps IDENTICAL
    SyntheticScene objects (label/cx/cy/sx/sy/sz/cz/heading) whether or not the
    additive OBB fields are populated."""
    gt_with_obb = _synthetic_gt_scene_with_obb(
        [
            ("sofa", -2.0, 0.0, 0.0, 1.0, 1.2, 0.8, 0.0),
            ("coffee table", 2.0, 0.5, 0.0, 0.9, 0.6, 0.5, 0.0),
        ]
    )
    gt_plain = _synthetic_gt_scene(
        [
            ("sofa", -2.0, 0.0, 0.0, 1.0, 1.2, 0.8),
            ("coffee table", 2.0, 0.5, 0.0, 0.9, 0.6, 0.5),
        ]
    )
    sc_obb = GB._synthetic_from_gt(gt_with_obb)
    sc_plain = GB._synthetic_from_gt(gt_plain)
    assert len(sc_obb.objects) == len(sc_plain.objects) == 2
    for a, b in zip(sc_obb.objects, sc_plain.objects):
        assert a.label == b.label
        assert a.heading == b.heading == 0.0
        assert (a.cx, a.cy, a.sx, a.sy, a.sz, a.cz) == (b.cx, b.cy, b.sx, b.sy, b.sz, b.cz)
    assert np.array_equal(sc_obb.terrain_patch().points, sc_plain.terrain_patch().points)


def test_synthetic_from_gt_rotated_object_stamps_smaller_than_aabb_hull():
    """Issue #77 Pre-Stage 1a core claim: a 45deg-rotated GT instance stamps its
    TRUE oriented footprint (fewer blocked terrain cells) rather than its AABB
    hull -- built through the real ``_synthetic_from_gt`` code path."""
    import math

    heading = math.pi / 4
    gt = _synthetic_gt_scene_with_obb(
        [
            ("sofa", 0.0, 0.0, 0.0, 2.0, 0.6, 0.7, heading),
            # A distant small anchor so the room bounds aren't dominated by the
            # rotated object alone (which would otherwise read "room-scale" and
            # get skipped by the #53 architectural-AABB guard).
            ("lamp", 8.0, 8.0, 0.0, 0.2, 0.2, 0.3, 0.0),
        ],
        scene_name="syn77_rot",
    )
    sc = GB._synthetic_from_gt(gt)
    sc.objects = [o for o in sc.objects if o.label == "sofa"]
    assert len(sc.objects) == 1
    obj = sc.objects[0]
    assert obj.heading == pytest.approx(heading)
    # sx/sy came from the OBB's own local extents, not the (larger-footprint) AABB hull.
    assert obj.sx == pytest.approx(2.0)
    assert obj.sy == pytest.approx(0.6)

    n_true = int(np.sum(sc.terrain_patch().points[:, 3] > 0.0))

    # Compare against the (pre-fix) AABB-hull stamp for the SAME instance.
    gt_hull_only = _synthetic_gt_scene(
        [
            ("sofa", 0.0, 0.0, 0.0,
             float(gt.instances[0].aabb_max[0] - gt.instances[0].aabb_min[0]),
             float(gt.instances[0].aabb_max[1] - gt.instances[0].aabb_min[1]),
             0.7),
            ("lamp", 8.0, 8.0, 0.0, 0.2, 0.2, 0.3),
        ],
        scene_name="syn77_hull",
    )
    sc_hull = GB._synthetic_from_gt(gt_hull_only)
    sc_hull.objects = [o for o in sc_hull.objects if o.label == "sofa"]
    n_hull = int(np.sum(sc_hull.terrain_patch().points[:, 3] > 0.0))
    assert n_true < n_hull


def test_synthetic_from_gt_arabic_room_sofa_covers_fewer_cells_than_hull():
    """The real arabic_room sofa (object_id 56) that motivated this fix: heading
    -3.1328664 rad, ~0.5 deg off axis-aligned. Confirms the rasterized stamp is a
    strict (if modest) improvement over the AABB hull for this exact fixture."""
    import math

    cx, cy, cz = 2.0790001107131033, 0.6679999141611369, 0.33887168842884763
    sx, sy, sz = 2.1899273413427003, 0.7787988067917222, 0.6777391842100893
    heading = -3.1328664111144175
    gt = _synthetic_gt_scene_with_obb(
        [
            ("sofa", cx, cy, cz, sx, sy, sz, heading),
            ("lamp", cx + 8.0, cy + 8.0, 0.0, 0.2, 0.2, 0.3, 0.0),
        ],
        scene_name="arabic_room_sofa",
    )
    sc = GB._synthetic_from_gt(gt)
    sofa_obj = next(o for o in sc.objects if o.label == "sofa")
    assert sofa_obj.heading == pytest.approx(heading)
    assert sofa_obj.sx == pytest.approx(sx)  # true OBB extents, not the AABB hull's

    hull_sx = float(gt.instances[0].aabb_max[0] - gt.instances[0].aabb_min[0])
    hull_sy = float(gt.instances[0].aabb_max[1] - gt.instances[0].aabb_min[1])
    assert hull_sx > sx or hull_sy > sy  # confirms this fixture IS rotated enough to inflate

    # The residual delta from a ~0.5deg tilt is well under the 0.1m terrain grid's
    # resolution, so exercise the exact same fine-grained containment check as
    # tests/mocks/test_synthetic_scene.py's GTObject-level regression test (the
    # true footprint stays a strict subset of its own AABB hull).
    step = 0.02
    xs = np.arange(cx - hull_sx / 2 - 0.1, cx + hull_sx / 2 + 0.1, step)
    ys = np.arange(cy - hull_sy / 2 - 0.1, cy + hull_sy / 2 + 0.1, step)
    n_true = sum(
        1 for x in xs for y in ys if sofa_obj.footprint_contains(float(x), float(y))
    )
    n_hull = sum(
        1
        for x in xs
        for y in ys
        if abs(float(x) - cx) <= hull_sx / 2 and abs(float(y) - cy) <= hull_sy / 2
    )
    assert n_true < n_hull


# ----------------------------------------------------------------------- issue #66


def test_nearest_free_goal_pushes_off_anchors_own_footprint():
    """Issue #66: a GOTO leg goal must not sit inside its OWN resolved anchor's solid
    footprint — the real drive (`InstructionHead._goto_point`) never targets a point
    inside the target's own geometry either (it BFS-snaps to the nearest reachable
    cell), so the rubric goal must match. A bench-sized anchor (half-diagonal well
    over ARRIVAL_TOL_M) with its raw centroid stands in for the traced office_1 case."""
    gt = _synthetic_gt_scene(
        [
            ("bench", 0.0, 0.0, 0.0, 1.58, 0.66, 0.4),
            # A distant marker so the room bounds aren't degenerately equal to the
            # bench's own footprint (which would make it read as "room-scale" itself).
            ("stool", 20.0, 20.0, 0.0, 0.4, 0.4, 0.5),
        ],
        scene_name="syn66own",
    )
    pushed = GB._nearest_free_goal((0.0, 0.0), gt)
    assert pushed != (0.0, 0.0)
    # Pushed clear of the bench's own (inflated) half-extent on the shorter axis.
    dist = (pushed[0] ** 2 + pushed[1] ** 2) ** 0.5
    assert dist >= 0.33 - 1e-6  # half of sy (0.66) -- the nearest edge


def test_nearest_free_goal_spares_room_scale_instances():
    """A room-scale-architectural instance the point falls inside is never a push
    source (issue #53) -- it isn't really stamped."""
    gt = _synthetic_gt_scene(
        [("wall", 0.0, 0.0, 0.0, 9.0, 9.0, 2.4)],  # room-scale in a 10x10 room
        scene_name="syn66spare",
    )
    pushed = GB._nearest_free_goal((0.0, 0.0), gt)
    assert pushed == (0.0, 0.0)


def test_nearest_free_goal_directional_push_prefers_approach_side():
    """Issue #66: with ``anchor_id``/``approach_xy`` given, the anchor's own-footprint
    push picks the SIDE actually approached (the previous leg's goal / route start),
    not just the geometrically nearest of 4 edges — a plain nearest-edge choice on a
    symmetric footprint traced worse against both the driven path and the GT
    reference path (issue #66 classification table: hotel_room_1, livingroom_4)."""
    gt = _synthetic_gt_scene(
        [
            ("table", 0.0, 0.0, 0.0, 2.0, 2.0, 0.4),  # a perfect square -- no single
            # geometrically-nearest edge; direction must be the tie-breaker.
            ("stool", 20.0, 20.0, 0.0, 0.4, 0.4, 0.5),
        ],
        scene_name="syn66dir",
    )
    table_id = gt.instances[0].instance_id
    # Approaching from due east -> pushed goal must land on the table's EAST edge.
    east = GB._nearest_free_goal(
        (0.0, 0.0), gt, anchor_id=table_id, approach_xy=(10.0, 0.0)
    )
    assert east[0] > 0.9  # east of the table's own half-extent (1.0 m)
    assert east[1] == pytest.approx(0.0)
    # Approaching from due south -> pushed goal must land on the table's SOUTH edge.
    south = GB._nearest_free_goal(
        (0.0, 0.0), gt, anchor_id=table_id, approach_xy=(0.0, -10.0)
    )
    assert south[1] < -0.9
    assert south[0] == pytest.approx(0.0)


def test_if_rubric_geometry_goto_goal_clears_own_anchor_footprint():
    """End-to-end: `_if_rubric_geometry`'s resolved GOTO goal for "go to the bench" is
    off the bench's own solid footprint, not at its raw (unreachable) centroid, and
    respects a supplied route start as the first leg's approach direction."""
    from core.perception.scene_index import BasicSceneIndex

    gt = _synthetic_gt_scene(
        [
            ("bench", 3.0, 0.0, 0.0, 1.58, 0.66, 0.4),
            ("stool", 20.0, 20.0, 0.0, 0.4, 0.4, 0.5),
        ],
        scene_name="syn66rubric",
    )
    idx = BasicSceneIndex(gt.instances)
    leg_goals, _, _, _, _ = GB._if_rubric_geometry(
        "Go to the bench.", gt, idx, start_xy=(3.0, -5.0)
    )
    assert len(leg_goals) == 1
    kind, (gx, gy) = leg_goals[0]
    assert kind == "goto"
    assert (gx, gy) != (3.0, 0.0)
    # Approached from due south -> pushed onto the bench's south edge.
    assert gy < 0.0


# --------------------------------------------------------------------------- issue #162


@requires_full_unity
def test_nearest_free_goal_home_building_2_coffee_table_bail_now_fails_explicitly():
    """Issue #162 regression: home_building_2's "coffee table" (instance 70) is the
    exact traced case from the issue -- pushing it off its own AABB plus the
    surrounding "carpet" footprint needs a ~3.32 m push, over the 2.5 m
    ``_RUBRIC_GOAL_MAX_PUSH_M`` cap. The OLD code bailed out and returned the raw,
    unpushed centroid -- a point inside solid furniture, silently treated as a
    valid rubric goal. The fix must report this as a construction failure
    (``None``), never fall back to the raw centroid."""
    gt = load_scene(str(FULL_UNITY_ROOT / "home_building_2"))
    rec = next(r for r in gt.instances if r.label == "coffee table")
    assert rec.instance_id == 70  # pins the exact traced instance
    xy = (float(rec.centroid[0]), float(rec.centroid[1]))
    result = GB._nearest_free_goal(xy, gt, anchor_id=rec.instance_id, approach_xy=None)
    assert result is None
    # And it must never silently equal the raw (unreachable) centroid either.
    assert result != xy


def test_nearest_free_goal_oscillating_footprints_returns_none():
    """Issue #162 regression: a point that oscillates between two overlapping
    footprints (pushing off A lands inside B; pushing off B lands back inside A)
    must be reported as a construction failure, not silently returned after the
    bounded iteration budget runs out. Geometry: a 1x1 "ottoman" at the origin and
    an overlapping 1.6x1.6 "bookshelf" centred 1.7 m east of it -- with the 0.4 m
    rubric clearance, their inflated boxes overlap by 0.4 m, and a point started
    just inside the ottoman's east edge bounces between the two forever."""
    gt = _synthetic_gt_scene(
        [
            ("ottoman", 0.0, 0.0, 0.0, 1.0, 1.0, 0.4),
            ("bookshelf", 1.7, 0.0, 0.0, 1.6, 1.6, 0.4),
            # A distant marker so the room isn't degenerately equal to either
            # footprint (which would misclassify one as room-scale/architectural).
            ("marker", 20.0, 20.0, 0.0, 0.4, 0.4, 0.5),
        ],
        scene_name="syn162osc",
    )
    result = GB._nearest_free_goal((0.3, 0.0), gt)
    assert result is None


def test_nearest_free_goal_normal_single_supporter_push_unchanged():
    """Control (issue #162): a normal, single-supporter push -- the common case
    the fix must NOT disturb -- still succeeds and returns a real, free point.
    Mirrors ``test_nearest_free_goal_pushes_off_anchors_own_footprint`` (issue #66)
    but asserts the point is explicitly non-``None`` and genuinely clear of every
    footprint, pinning that the convergence/post-condition machinery added here
    doesn't turn a legitimately-constructible goal into a false failure."""
    gt = _synthetic_gt_scene(
        [
            ("bench", 0.0, 0.0, 0.0, 1.58, 0.66, 0.4),
            ("stool", 20.0, 20.0, 0.0, 0.4, 0.4, 0.5),
        ],
        scene_name="syn162control",
    )
    result = GB._nearest_free_goal((0.0, 0.0), gt)
    assert result is not None
    x0, y0, x1, y1 = GB._gt_footprint_bounds(gt, 0.0)
    assert GB._blocking_footprint(result[0], result[1], gt, x1 - x0, y1 - y0) is None


def test_if_rubric_geometry_construction_failure_skips_leg_like_unresolved_anchor():
    """Issue #162: a GOTO leg whose anchor resolves but whose goal cannot be
    constructed (oscillating footprints) is dropped from ``leg_goals`` the exact
    same way an anchor that fails to resolve already is -- an unscored, not a
    wrong, leg -- and the failure is surfaced via ``goal_construction_diag``
    rather than silently vanishing."""
    from core.perception.scene_index import BasicSceneIndex

    gt = _synthetic_gt_scene(
        [
            ("ottoman", 0.0, 0.0, 0.0, 1.0, 1.0, 0.4),
            ("bookshelf", 1.7, 0.0, 0.0, 1.6, 1.6, 0.4),
            # A distant marker so the room bounds aren't degenerately equal to the
            # ottoman's own footprint (which would misclassify it as room-scale
            # architectural (#53) and skip pushing off it entirely).
            ("marker", 20.0, 20.0, 0.0, 0.4, 0.4, 0.5),
        ],
        scene_name="syn162rubric",
    )
    # The ottoman itself is the resolved GOTO anchor. Approaching from due east
    # makes the directional own-footprint push (issue #66) go straight toward the
    # bookshelf, triggering the same A<->B oscillation proven in isolation above.
    idx = BasicSceneIndex(gt.instances)
    diag: list[str] = []
    leg_goals, _, _, leg_instance_ids, _ = GB._if_rubric_geometry(
        "Go to the ottoman.", gt, idx,
        start_xy=(10.0, 0.0), goal_construction_diag=diag,
    )
    assert leg_goals == []
    assert leg_instance_ids == []
    assert len(diag) == 1
    assert "ottoman" in diag[0]


def test_score_instruction_rubric_missed_constructible_leg_still_scores_as_before():
    """Control (issue #162): this fix must not blanket-loosen scoring. A leg whose
    goal IS constructible, which the robot simply never drives near, must still
    score exactly as it always has -- not reached, zero ordered-leg credit."""
    from core.groundtruth import scoring as S

    leg_goals = [("goto", (5.0, 5.0))]
    driven = np.array([[0.0, 0.0], [0.1, 0.1], [0.2, 0.2]], dtype=float)
    rub = S.score_instruction_rubric(driven, leg_goals)
    assert rub.n_legs == 1
    assert rub.n_legs_reached_in_order == 0
    assert rub.ordered_leg_credit == 0.0
    assert rub.leg_outcomes[0].reached is False


# ----------------------------------------------------------- issue #162 rubric v3 fix


def test_aggregate_excludes_zero_evaluable_leg_question_from_if_rubric():
    """Issue #162 (v3 correction): a question every one of whose legs failed goal
    construction has ``n_legs == 0`` and ``rubric_score == 0.0`` by construction
    (``core.groundtruth.scoring.score_instruction_rubric`` returns 0.0, not None,
    when ``n_legs`` is 0). Counting that 0.0 in the ``if_rubric`` aggregate would
    depress the battery for exactly the case #162 was fixed to handle honestly —
    refusing to fabricate a goal must not itself read as a scored failure. Mirrors
    the #155 precedent (``n_threading_unevaluable`` legs excluded from the
    threading-violation count): here the whole QUESTION is excluded from the
    rubric means, not scored as 0.0 or 1.0."""
    from core.runner.gt_battery import GTQuestionScore, aggregate

    rows = [
        # Fully unscoreable: both legs failed goal construction. Must NOT drag
        # the mean down to 0.0 for this row.
        GTQuestionScore(
            scene="a", qtype=QType.INSTRUCTION_FOLLOWING.value, question="q0",
            rubric_score=0.0, ordered_leg_credit=0.0, n_legs=0,
            n_legs_reached_in_order=0, n_goal_construction_unevaluable=2,
        ),
        # A normal, fully-scored perfect question.
        GTQuestionScore(
            scene="b", qtype=QType.INSTRUCTION_FOLLOWING.value, question="q1",
            rubric_score=1.0, ordered_leg_credit=1.0, n_legs=2,
            n_legs_reached_in_order=2, n_goal_construction_unevaluable=0,
        ),
        # A legitimately failed leg (goal WAS constructible, robot never arrived)
        # -- must still count as a real 0.0, not be excluded.
        GTQuestionScore(
            scene="c", qtype=QType.INSTRUCTION_FOLLOWING.value, question="q2",
            rubric_score=0.0, ordered_leg_credit=0.0, n_legs=1,
            n_legs_reached_in_order=0, n_goal_construction_unevaluable=0,
        ),
    ]
    agg = aggregate(rows)["instruction_following"]
    assert agg["n_scored"] == 3
    # only the genuinely-scored q1/q2 count toward the rubric means
    assert agg["n_zero_evaluable_legs"] == 1
    assert agg["mean_rubric_score"] == 0.5
    assert agg["mean_ordered_leg_credit"] == 0.5


def test_aggregate_if_rubric_none_when_all_questions_zero_evaluable_legs():
    """Degenerate case: every IF question has zero evaluable legs -- the rubric
    means must be ``None`` (no evaluable data), not ``0.0`` (a false failure
    signal) and not silently omitted."""
    from core.runner.gt_battery import GTQuestionScore, aggregate

    rows = [
        GTQuestionScore(
            scene="a", qtype=QType.INSTRUCTION_FOLLOWING.value, question="q0",
            rubric_score=0.0, ordered_leg_credit=0.0, n_legs=0,
            n_legs_reached_in_order=0, n_goal_construction_unevaluable=1,
        ),
    ]
    agg = aggregate(rows)["instruction_following"]
    assert agg["n_scored"] == 1
    assert agg["n_zero_evaluable_legs"] == 1
    assert agg["mean_rubric_score"] is None
    assert agg["mean_ordered_leg_credit"] is None


def test_if_rubric_geometry_skipped_leg_carries_own_anchor_forward_not_stale_approach():
    """Issue #162 (v3 correction): when a leg's goal fails to construct and is
    skipped, the NEXT leg's push must not be biased by whatever approach
    reference preceded the skipped leg (e.g. ``start_xy``, potentially clear
    across the room) -- the driven trajectory this rubric scores still actually
    passes by the skipped leg's own anchor before continuing on, skip or no
    skip. The skipped leg's own (raw, never-scored) anchor centroid must be
    carried forward as the next leg's approach reference instead.

    Geometry: leg 0 ("ottoman") oscillates against an overlapping "bookshelf"
    and fails construction (proven independently in
    ``test_nearest_free_goal_oscillating_footprints_returns_none``). Leg 1
    ("sofa") is pushed off its OWN footprint (issue #66) -- and that push's
    AXIS depends entirely on which side ``approach_xy`` sits on relative to the
    sofa's centroid: due east (the old, stale ``start_xy=(10, 0)``) pushes
    along X; due south (the ottoman's own centroid at the origin) pushes along
    Y. The two are unambiguously distinguishable outcomes, so this pins the
    fix, not just "a value changed"."""
    from core.perception.scene_index import BasicSceneIndex as _BSI

    gt = _synthetic_gt_scene(
        [
            ("ottoman", 0.0, 0.0, 0.0, 1.0, 1.0, 0.4),
            ("bookshelf", 1.7, 0.0, 0.0, 1.6, 1.6, 0.4),
            # keeps the room bounds from degenerately equalling the ottoman's own
            # footprint (which would misclassify it as room-scale architectural).
            ("marker", 20.0, 20.0, 0.0, 0.4, 0.4, 0.5),
            ("sofa", 0.0, 10.0, 0.0, 2.0, 1.0, 0.5),
        ],
        scene_name="syn162approach",
    )
    idx = _BSI(gt.instances)
    diag: list[str] = []
    leg_goals, _, _, leg_instance_ids, _ = GB._if_rubric_geometry(
        "Go to the ottoman and stop at the sofa.", gt, idx,
        start_xy=(10.0, 0.0), goal_construction_diag=diag,
    )
    # leg 0 (ottoman) skipped; only the sofa leg remains.
    assert len(diag) == 1 and "ottoman" in diag[0]
    assert len(leg_goals) == 1
    kind, (gx, gy) = leg_goals[0]
    assert kind == "goto"
    # Pushed along Y (toward the skipped ottoman leg's own centroid, south of
    # the sofa) -- NOT along X (toward the stale start_xy, east of the sofa).
    # The stale-approach push would have landed near (1.4, 10.0); the fixed
    # push lands near (0.0, 9.1).
    assert gx == pytest.approx(0.0, abs=1e-2)
    assert gy < 10.0
    assert gy == pytest.approx(9.0999, abs=1e-3)


# --------------------------------------------------------------------------- issue #81:
# offline withhold-gate parity (budget_frac/forced_assembly hooks)


def _relaxable_gt_scene():
    """A GTScene where "the lamp near the bench" resolves only by dropping the relation
    clause (a lamp exists, no bench does) -- a provisional terminal grounding, mirroring
    ``tests/heads/test_if_explore_ungrounded.py::_relaxable_scene`` at the gt_battery
    (fully-observed, n_obs=3) layer."""
    from core.perception.scene_index import BasicSceneIndex

    gt = _synthetic_gt_scene(
        [
            ("table", -4.0, 0.0, 0.4, 1.0, 1.0, 0.8),  # leg 0 anchor
            ("lamp", 6.0, 0.0, 0.3, 0.4, 0.4, 0.4),  # leg 1 / terminal anchor (no bench)
        ],
        scene_name="syn81relax",
    )
    return gt, BasicSceneIndex(gt.instances)


_RELAXABLE_TEXT = "Go to the table and then go to the lamp near the bench."


def test_run_instruction_head_commits_provisional_terminal_by_default():
    """Byte-preservation: with ``enable_withhold_gates`` left at its default (False), the
    offline battery's ``InstructionHead`` gets ``budget_frac=None``, so H4c's
    ``_commit_forced()`` defaults True and a provisional terminal commits immediately —
    exactly today's (pre-#81) behaviour, unchanged by this fix."""
    gt, idx = _relaxable_gt_scene()
    head, _io, _plan = GB._run_instruction_head(
        _RELAXABLE_TEXT, gt, idx, start_xy=(-4.0, 0.0), max_build_ticks=GB._IF_MAX_BUILD_TICKS,
    )
    assert head._legs[-1].provisional, "test setup sanity: the terminal must be provisional"
    assert head._driven_prefix == 2, "unhooked H4c must commit the provisional terminal"


def test_run_instruction_head_withholds_provisional_terminal_when_gates_enabled():
    """Issue #81 acceptance: with ``enable_withhold_gates=True`` the offline battery wires
    the SAME kind of budget_frac/forced_assembly hooks the live adapter wires (a real
    ``BudgetState`` keyed to the drive's own clock), so H4c's withhold gate fires offline
    exactly as it does live — the provisional terminal is held back from the committed
    prefix while budget remains fresh, instead of silently committing (the bug this issue
    tracks)."""
    gt, idx = _relaxable_gt_scene()
    head, _io, _plan = GB._run_instruction_head(
        _RELAXABLE_TEXT, gt, idx, start_xy=(-4.0, 0.0), max_build_ticks=GB._IF_MAX_BUILD_TICKS,
        enable_withhold_gates=True,
    )
    assert head._legs[-1].provisional, "test setup sanity: the terminal must be provisional"
    assert head._driven_prefix == 1, (
        "H4c must withhold the provisional terminal offline once the hooks are wired "
        "(driven_prefix should stop at the grounded, non-provisional leg 0)"
    )


def test_drive_if_trajectory_stops_short_of_withheld_provisional_terminal():
    """Same acceptance at the DRIVEN-trajectory level (what the rubric actually scores):
    with the gates enabled the driven path never reaches the withheld terminal anchor."""
    from core.perception.scene_index import BasicSceneIndex

    gt, idx = _relaxable_gt_scene()

    driven_off = GB._drive_if_trajectory(
        gt=gt, idx=idx, text=_RELAXABLE_TEXT, start_xy=(-4.0, 0.0),
    )
    driven_on = GB._drive_if_trajectory(
        gt=gt, idx=idx, text=_RELAXABLE_TEXT, start_xy=(-4.0, 0.0), enable_withhold_gates=True,
    )

    lamp_xy = np.array([6.0, 0.0])
    # Default: the provisional terminal commits, so the driven path reaches near the lamp.
    # Issue #132: the terminal standoff now correctly adds the costmap's inflation radius
    # to the push target, so the committed goal -- and the closest driven approach -- sits
    # measurably farther from the small lamp's centroid (~1.13 m) than the pre-#132 figure
    # (~0.73 m), still comfortably inside the rubric's arrival tolerance. A "< 1.3" upper
    # bound alone would still pass if #132 were reverted, so pin a band that EXCLUDES the
    # old, short figure too: proof the provisional terminal actually committed AND that the
    # #132 standoff fix is in effect.
    min_off = np.min(np.linalg.norm(driven_off[:, :2] - lamp_xy, axis=1))
    assert 0.95 <= min_off < 1.3, (
        f"driven (default) closest approach to the lamp {min_off:.3f} m outside the "
        "expected post-#132 band [0.95, 1.3) m (either the provisional terminal never "
        "committed, or the standoff accounting regressed to its pre-#132 ~0.73 m figure)"
    )
    # Gated: the terminal is withheld, so the driven path stops at leg 0 (the table) and
    # never approaches the lamp -- unaffected by the #132 push-distance change, so a plain
    # bound (well clear of either the pre- or post-#132 committed-terminal figures) suffices.
    assert np.min(np.linalg.norm(driven_on[:, :2] - lamp_xy, axis=1)) > 1.3


def test_offline_budget_hooks_track_a_real_budget_state():
    """``_offline_budget_hooks`` reuses the live ``core.fsm.budget.BudgetState`` type
    (not a new mock) keyed to the supplied clock: ``budget_frac`` rises with elapsed time
    and ``forced_assembly`` flips True once the T-90 gate is crossed."""
    from core.interfaces import FORCED_ASSEMBLY_S, QUESTION_BUDGET_S

    clk = GB.FakeClock(0.0)
    budget_frac, forced_assembly = GB._offline_budget_hooks(clk)

    assert budget_frac() == 0.0
    assert forced_assembly() is False

    # BudgetState's own default forced-assembly gate (core.interfaces.FORCED_ASSEMBLY_S) --
    # _offline_budget_hooks does not override it, so this is what actually fires here.
    clk.advance(FORCED_ASSEMBLY_S)
    assert forced_assembly() is True
    assert budget_frac() == pytest.approx(FORCED_ASSEMBLY_S / QUESTION_BUDGET_S)


# ----------------------------------------------------------------------- issue #124
# The sim->object frame fit is spurious: a single mis-ranked terminal-goal
# correspondence can rotate the whole scene, and the endpoint residual gate does not
# catch it. Fix: try the IDENTITY first (verified against the live sim's own
# object_list.txt by id); only fall back to the endpoint fit when that doesn't
# apply, and gate any fitted (non-identity) frame on a free-space sanity check.


def _write_object_list(tmp_path, scene_name, rows):
    """``rows`` is ``[(id, x, y, z, label), ...]`` -> a real object_list.txt on disk."""
    scene_dir = tmp_path / scene_name / scene_name
    scene_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        f'{oid} {x} {y} {z} 0.3 0.3 0.3 0.0 "{label}"' for oid, x, y, z, label in rows
    ]
    (scene_dir / "object_list.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_identity_frame_chosen_when_object_list_matches(tmp_path):
    """object_list.txt ids match the GT instances within tolerance -> the IDENTITY is
    used directly, no endpoint fit attempted at all."""
    gt = _synthetic_gt_scene(
        [("chair", 1.0, 2.0, 0.5, 0.5, 0.5, 1.0), ("lamp", -3.0, 0.5, 0.4, 0.2, 0.2, 0.4)],
        scene_name="s124a",
    )
    _write_object_list(
        tmp_path, "s124a",
        [(0, 1.01, 1.99, 0.48, "chair"), (1, -2.98, 0.53, 0.42, "lamp")],
    )
    idx = BasicSceneIndex(gt.instances)
    if_traj, if_cands, frame, residual, pairs = GB._fit_scene_if_frame(
        gt, idx, [], None, unity_scenes_ros2_root=tmp_path,
    )
    assert frame is not None
    assert frame.theta == 0.0
    np.testing.assert_allclose(frame.t, [0.0, 0.0])
    assert residual < GB.IDENTITY_MATCH_TOL_M


def test_identity_frame_declined_when_delta_too_large(tmp_path):
    """object_list.txt ids match, but the measured centroid delta is well past the
    identity tolerance -> no identity verdict, caller falls back to the endpoint fit."""
    gt = _synthetic_gt_scene(
        [("chair", 1.0, 2.0, 0.5, 0.5, 0.5, 1.0)], scene_name="s124b",
    )
    _write_object_list(
        tmp_path, "s124b", [(0, 1.0 + 5.0, 2.0, 0.5, "chair")],  # 5 m off -> not identity
    )
    idx = BasicSceneIndex(gt.instances)
    assert GB._identity_frame_if_matched(gt, tmp_path) is None
    # No IF trajectories at all -> the endpoint-fit path also has nothing to fit;
    # the whole thing correctly resolves to "no frame" rather than a false identity.
    if_traj, if_cands, frame, residual, pairs = GB._fit_scene_if_frame(
        gt, idx, [], None, unity_scenes_ros2_root=tmp_path,
    )
    assert frame is None
    assert residual is None


def test_identity_frame_none_when_object_list_missing(tmp_path):
    """No object_list.txt on disk for the scene -> no identity verdict (not an error)."""
    gt = _synthetic_gt_scene(
        [("chair", 1.0, 2.0, 0.5, 0.5, 0.5, 1.0)], scene_name="s124c",
    )
    assert GB._identity_frame_if_matched(gt, tmp_path) is None


def _scene_with_regions(objs, region_box, scene_name):
    """A synthetic GTScene like ``_synthetic_gt_scene`` but with one GT region (the
    "floor") covering ``region_box = (x0, y0, x1, y1)``, needed to exercise the
    free-space gate's "on floor" side against a bounded footprint."""
    from core.groundtruth.loader import GTRegion

    gt = _synthetic_gt_scene(objs, scene_name=scene_name)
    x0, y0, x1, y1 = region_box
    region = GTRegion(
        region_id=0, label="room",
        aabb_min=np.array([x0, y0, 0.0]), aabb_max=np.array([x1, y1, 3.0]),
    )
    gt.regions.append(region)
    return gt


def test_free_space_gate_rejects_nonsense_fit(monkeypatch):
    """Regression (issue #124): a fit whose ENDPOINT residual clears the alignment
    gate, but whose own source trajectory points map almost entirely off the GT
    floor once the fit is applied, must be REJECTED -- not silently scored. This
    reproduces the confirmed defect shape: a large spurious rotation that still
    passes the (endpoint-only) residual check."""
    gt = _scene_with_regions(
        [("chair", 0.0, 0.0, 0.5, 0.5, 0.5, 1.0)],
        region_box=(-2.0, -2.0, 2.0, 2.0),
        scene_name="s124d",
    )
    idx = BasicSceneIndex(gt.instances)

    # Trajectory living entirely inside the floor region in the SIM frame.
    traj = np.array([[0.0, 0.0, 0.75], [1.0, 0.0, 0.75]], dtype=float)

    # A large-rotation, large-translation frame (shape of the confirmed spurious
    # fits: theta up to 154 deg, |t| up to 10.7 m) that maps the trajectory WAY
    # outside the floor region -- but report a tiny residual, as the real defect did
    # (the endpoint correspondence alone can't tell this fit is nonsense).
    bad_frame = S.Frame2D(theta=math.radians(150.0), t=np.array([50.0, 50.0]))
    monkeypatch.setattr(GB.S, "align_scene_trajectories", lambda pairs: (bad_frame, 0.05))
    # Feed one IF question so `pairs` is non-empty and the fit path (not identity,
    # no object_list.txt) actually runs.
    monkeypatch.setattr(GB, "_terminal_goal_candidates", lambda text, idx: [np.array([1.0, 0.0])])
    monkeypatch.setattr(GB.S, "load_trajectory_ply", lambda *a, **k: traj)

    with tempfile.TemporaryDirectory() as tmp:
        qdir = Path(tmp) / gt.scene_name
        qdir.mkdir(parents=True)
        (qdir / "trajectory_q4.ply").write_bytes(b"")
        if_traj, if_cands, frame, residual, pairs = GB._fit_scene_if_frame(
            gt, idx, ["go to the chair"], Path(tmp),
        )

    assert frame is None
    assert residual == float("inf")


def test_free_space_gate_accepts_sane_fit(monkeypatch):
    """Sanity counterpart: a fit whose mapped source points DO land on the GT floor
    (e.g. a small, honest correction) is accepted as before -- the gate only rejects
    genuinely nonsensical fits, not every non-identity one."""
    gt = _scene_with_regions(
        # Chair sits well clear of the trajectory's path (0,0)->(1,0) so a sane fit
        # doesn't also trip the "in furniture" side of the gate.
        [("chair", -1.8, 1.8, 0.5, 0.3, 0.3, 1.0)],
        region_box=(-2.0, -2.0, 2.0, 2.0),
        scene_name="s124e",
    )
    idx = BasicSceneIndex(gt.instances)
    traj = np.array([[0.0, 0.0, 0.75], [1.0, 0.0, 0.75]], dtype=float)

    sane_frame = S.Frame2D(theta=0.0, t=np.array([0.1, 0.0]))
    monkeypatch.setattr(GB.S, "align_scene_trajectories", lambda pairs: (sane_frame, 0.1))
    monkeypatch.setattr(GB, "_terminal_goal_candidates", lambda text, idx: [np.array([1.1, 0.0])])
    monkeypatch.setattr(GB.S, "load_trajectory_ply", lambda *a, **k: traj)

    with tempfile.TemporaryDirectory() as tmp:
        qdir = Path(tmp) / gt.scene_name
        qdir.mkdir(parents=True)
        (qdir / "trajectory_q4.ply").write_bytes(b"")
        if_traj, if_cands, frame, residual, pairs = GB._fit_scene_if_frame(
            gt, idx, ["go to the chair"], Path(tmp),
        )

    assert frame is sane_frame
    assert residual == 0.1


def test_free_space_fraction_uses_footprint_fallback_when_no_regions():
    """A GTScene with no region CSV (regions=[]) still runs the free-space check by
    falling back to the instance-AABB footprint, instead of vacuously scoring
    on_floor=0 for every scene lacking a region file."""
    gt = _synthetic_gt_scene(
        [("chair", 0.0, 0.0, 0.5, 1.0, 1.0, 1.0)], scene_name="s124f",
    )
    pts = np.array([[0.0, 0.0]])  # dead centre of the only instance's footprint
    on_floor, in_furniture = GB._free_space_fraction(pts, gt)
    assert on_floor == 1.0
    assert in_furniture == 1.0  # also inside the (only) furniture item, as expected
