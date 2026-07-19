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
    assert near[0, 3] == WALL_HEIGHT

    # A cell well away from the marked wall cell and the border stays free.
    away = pts[(np.abs(pts[:, 0] - 1.0) < 1e-6) & (np.abs(pts[:, 1] - 1.0) < 1e-6)]
    assert away.shape[0] == 1
    assert away[0, 3] == 0.0


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
    leg_goals, corridor_gates, _, _ = GB._if_rubric_geometry(text, gt, idx)
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
    leg_goals, corridor_gates, _, _ = GB._if_rubric_geometry("Go to the stool.", gt, idx)
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
