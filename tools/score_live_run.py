"""Score LIVE-sim baseline runs (recorded ROS 2 bags) against ground truth.

Companion to ``core.runner.gt_battery`` (the OFFLINE battery, which drives a
*simulated* kinematic follower over the mirror costmap): this tool scores the
*actually driven* trajectory + published answers captured from a real
``ros2 bag record`` of the live sim/adapter stack, using the SAME scorers and
SAME per-scene geometry the offline battery uses (``core.groundtruth.scoring``,
plus the private leg/frame helpers in ``core.runner.gt_battery`` — imported,
never reimplemented) so a live row and its offline counterpart are directly
comparable question-for-question.

Bag reading reuses ``core.replay.bag_reader.BagSource`` (the same machinery
``tools/llm_vision_checkpoint_replay.py`` and the perception replay tooling
use to turn an mcap into core dataclasses), extended with two converters for
the two live-only topics the replay machinery doesn't already know about
(``/numerical_response`` std_msgs/Int32, ``/selected_object_marker``
visualization_msgs/Marker).

Frame note: the live sim publishes ``/state_estimation`` in the SAME
sim/trajectory frame the GT reference trajectories (``trajectory_q{4,5}.ply``)
were recorded in — verified per-scene by comparing the live trajectory's first
pose to the GT trajectory's first vertex (both ~= the scene's fixed sim-origin
spawn, (0, 0, 0.75) for every scene in this dataset). The battery's per-scene
fitted ``Frame2D`` (``core.runner.gt_battery._fit_scene_if_frame`` /
``core.groundtruth.scoring.align_scene_trajectories``) maps that sim frame into
the VLA-3D object frame; we fit it ONCE per scene (from the questions.json
instruction_following texts + GT trajectory PLYs — no live data involved) and
reuse it for every question type in that scene, since it is a property of the
scene's sim<->object registration, not of any one question.

Usage (from the repo root, host venv)::

    python -m tools.score_live_run                                   # whole default baseline dir
    python -m tools.score_live_run reports/live_baseline_2026-07-20 --out reports/live_baseline_2026-07-20  # explicit baseline dir
    python -m tools.score_live_run reports/live_baseline_2026-07-20/livingroom_1/inst --out reports/scratch  # one run

Writes ``<out>/scores.md`` + ``<out>/scores.json``. With no ``target`` this
defaults to the committed baseline dir (merging any existing ``scores.json``
there); any invocation that DOES name a ``target`` must also name ``--out``
(#96) — the default baseline dir is a committed evidence artifact and must
never be silently overwritten by an unrelated/narrower scoring run. Safe to
re-run as more bags land — each invocation rescans its target fresh; nothing
is mutated in place except the chosen ``<out>``.

Pure offline dev tool (tools/ — never part of the scored pipeline). CPU only.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

import tools  # noqa: F401 -- inserts <repo>/src onto sys.path

from core.groundtruth import scoring as S
from core.groundtruth.loader import load_scene
from core.interfaces import MarkerBox
from core.perception.scene_index import BasicSceneIndex
from core.replay.bag_reader import (
    DEFAULT_TOPIC_MAP,
    BagSource,
    TOPIC_ODOM,
    TOPIC_QUESTION,
)
from core.runner import gt_battery as GB

_REPO = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE_DIR = _REPO / "reports" / "live_baseline_2026-07-20"
DEFAULT_GROUNDTRUTH = _REPO / "data" / "vla3d" / "Unity"
DEFAULT_OFFLINE_RESULTS = (
    _REPO / "reports" / "gt_battery_main_post_carve" / "gt_battery_results.json"
)

#: run-dir basename -> QType value (docs/challenge_brief.md naming; the baseline
#: capture layout uses the shortened dir names per scene).
QDIR_TO_QTYPE = {
    "nume": "numerical",
    "obje": "object_reference",
    "inst": "instruction_following",
}

TOPIC_NUM_RESPONSE = "/numerical_response"
TOPIC_MARKER = "/selected_object_marker"

#: Minimum XY movement (m) between kept trajectory samples. Live odom is
#: published at ~100-200 Hz; the rubric scorer's Frechet/threading/capsule
#: checks are O(n*m) Python loops sized for the offline battery's sparse
#: (one-pose-per-planned-waypoint) driven paths, so a raw multi-hundred-
#: thousand-row live trajectory must be shape-preservingly decimated first —
#: well below every tolerance the scorer applies (ARRIVAL_RESAMPLE_STEP_M
#: 0.25 m, leg-arrival tolerances ~1-2 m), so decimation cannot change a
#: verdict, only the cost of computing it.
DECIMATE_MIN_MOVE_M = 0.05


def _int32_to_value(msg: Any, bag_ns: int) -> int:
    return int(msg.data)


def _marker_to_dict(msg: Any, bag_ns: int) -> dict[str, float]:
    p = msg.pose.position
    s = msg.scale
    return {
        "cx": float(p.x), "cy": float(p.y), "cz": float(p.z),
        "sx": float(s.x), "sy": float(s.y), "sz": float(s.z),
        "t": float(bag_ns) * 1e-9,
    }


LIVE_TOPIC_MAP = {
    TOPIC_ODOM: DEFAULT_TOPIC_MAP[TOPIC_ODOM],
    TOPIC_QUESTION: DEFAULT_TOPIC_MAP[TOPIC_QUESTION],
    TOPIC_NUM_RESPONSE: _int32_to_value,
    TOPIC_MARKER: _marker_to_dict,
}


# --------------------------------------------------------------------------- bag reading


@dataclass
class BagCapture:
    """Raw content extracted from one run's bag, plus per-topic message counts."""

    question_text: str | None
    odom_xy_raw_n: int  # message count before decimation (capture diagnostic)
    odom_xy: np.ndarray  # (N, 2) decimated, order-preserving
    numerical_response: int | None
    marker: dict[str, float] | None
    topic_counts: dict[str, int] = field(default_factory=dict)


def _decimate_xy(points: list[tuple[float, float]], min_move: float) -> np.ndarray:
    """Distance-decimate an ordered XY polyline, keeping the first/last points.

    Shape-preserving (never reorders, never drops a >=min_move deviation) —
    see :data:`DECIMATE_MIN_MOVE_M` for why this is safe for the scorers.
    """
    if not points:
        return np.empty((0, 2), dtype=float)
    out = [points[0]]
    last = points[0]
    for p in points[1:]:
        if math.hypot(p[0] - last[0], p[1] - last[1]) >= min_move:
            out.append(p)
            last = p
    if out[-1] != points[-1]:
        out.append(points[-1])
    return np.asarray(out, dtype=float)


def read_bag_capture(bag_dir: Path) -> BagCapture:
    """Read one run's bag into a :class:`BagCapture`.

    ``question_text`` is the LAST ``/challenge_question`` message (there
    should be exactly one on a well-formed capture); ``numerical_response``/
    ``marker`` are likewise the last message on their topic (the final
    published answer, tolerant of a re-publish). ``topic_counts`` reads
    straight off the bag's connection metadata (rosbags ``msgcount``) —
    the capture-completeness signal this tool reports even when a topic's
    payload isn't needed for THIS run's qtype.
    """
    from rosbags.highlevel import AnyReader

    topic_counts: dict[str, int] = {}
    with AnyReader([bag_dir]) as reader:
        for c in reader.connections:
            topic_counts[c.topic] = topic_counts.get(c.topic, 0) + int(c.msgcount)

    question_text: str | None = None
    numerical_response: int | None = None
    marker: dict[str, float] | None = None
    odom_pts: list[tuple[float, float]] = []
    odom_n = 0
    for rec in BagSource(bag_dir, topic_map=LIVE_TOPIC_MAP):
        if rec.topic == TOPIC_ODOM:
            odom_n += 1
            odom_pts.append((rec.msg.x, rec.msg.y))
        elif rec.topic == TOPIC_QUESTION:
            question_text = rec.msg.text
        elif rec.topic == TOPIC_NUM_RESPONSE:
            numerical_response = rec.msg
        elif rec.topic == TOPIC_MARKER:
            marker = rec.msg

    return BagCapture(
        question_text=question_text,
        odom_xy_raw_n=odom_n,
        odom_xy=_decimate_xy(odom_pts, DECIMATE_MIN_MOVE_M),
        numerical_response=numerical_response,
        marker=marker,
        topic_counts=topic_counts,
    )


def _run_log_question(run_dir: Path) -> str | None:
    """Fallback question text from ``run.log``'s ``question=<text>`` line.

    Used when the bag itself carries no ``/challenge_question`` message
    (a capture gap — see the tool's capture-completeness check) but the
    launch log still records what was asked.
    """
    log = run_dir / "run.log"
    if not log.is_file():
        return None
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("question="):
            return line[len("question="):].strip()
    return None


# --------------------------------------------------------------------------- discovery


def discover_runs(root: Path) -> list[tuple[str, str, Path]]:
    """Find ``<scene>/<qdir>/bag`` runs under a baseline dir. Returns (scene, qdir, run_dir)."""
    runs: list[tuple[str, str, Path]] = []
    if not root.is_dir():
        return runs
    for scene_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for qdir in QDIR_TO_QTYPE:
            run_dir = scene_dir / qdir
            if (run_dir / "bag").is_dir():
                runs.append((scene_dir.name, qdir, run_dir))
    return runs


def resolve_targets(target: str | None) -> list[tuple[str, str, Path]]:
    """Resolve a CLI target (baseline root, or one run dir) to a run list."""
    if target is None:
        return discover_runs(DEFAULT_BASELINE_DIR)
    p = Path(target)
    if (p / "bag").is_dir():
        # A single run dir: <scene_dir>/<qdir>.
        return [(p.parent.name, p.name, p)]
    return discover_runs(p)


# --------------------------------------------------------------------------- questions.json


def _load_questions_index(questions_path: Path) -> dict[str, dict[str, list[str]]]:
    with open(questions_path, encoding="utf-8") as fh:
        data = json.load(fh)
    return {e["scene"]: e["questions"] for e in data}


def _squash(s: str) -> str:
    return "".join(s.split()).casefold()


def _match_question(text: str, candidates: list[str]) -> int | None:
    """Index of ``text`` in ``candidates`` under whitespace/case-insensitive equality."""
    target = _squash(text)
    for i, c in enumerate(candidates):
        if _squash(c) == target:
            return i
    return None


# --------------------------------------------------------------------------- per-scene frame


@dataclass
class SceneContext:
    scene: str
    gt: Any
    idx: BasicSceneIndex
    referential: dict | None
    scene_graph: dict | None
    if_texts: list[str]
    frame: S.Frame2D | None
    fit_residual_m: float | None
    spawn_xy: tuple[float, float] | None


_SCENE_CACHE: dict[str, SceneContext] = {}


def _load_scene_context(
    scene: str,
    *,
    groundtruth_root: Path,
    questions_dir: Path,
    questions_index: dict[str, dict[str, list[str]]],
    unity_scenes_ros2_root: Path | None = None,
) -> SceneContext | None:
    if scene in _SCENE_CACHE:
        return _SCENE_CACHE[scene]
    folder = GB._find_scene_folder(groundtruth_root, scene)
    if folder is None:
        return None
    gt = load_scene(folder, scene_name=scene)
    idx = BasicSceneIndex(gt.instances)
    referential = GB._load_referential(folder, scene)
    scene_graph = GB._load_scene_graph(folder, scene)
    entry_questions = questions_index.get(scene, {})
    if_texts = entry_questions.get("instruction_following", [])

    # Issue #124: IDENTITY-first, verified against the live sim's own
    # object_list.txt by id -- see GB._fit_scene_if_frame's docstring. Falls back
    # to the (now free-space-gated) endpoint-correspondence fit only when
    # object_list.txt is unavailable or its ids don't match closely enough.
    _if_traj, _if_cands, frame, residual, pairs = GB._fit_scene_if_frame(
        gt, idx, if_texts, questions_dir,
        unity_scenes_ros2_root=unity_scenes_ros2_root,
    )
    spawn_xy: tuple[float, float] | None = None
    if frame is not None and pairs:
        start_pt = pairs[0][0][0, :2]
        mapped = frame.apply(np.asarray([start_pt], dtype=float))[0]
        spawn_xy = (float(mapped[0]), float(mapped[1]))

    ctx = SceneContext(
        scene=scene, gt=gt, idx=idx, referential=referential, scene_graph=scene_graph,
        if_texts=if_texts, frame=frame, fit_residual_m=residual, spawn_xy=spawn_xy,
    )
    _SCENE_CACHE[scene] = ctx
    return ctx


# --------------------------------------------------------------------------- marker -> AABB


def _marker_aabb_in_object_frame(
    marker: dict[str, float], frame: S.Frame2D | None
) -> tuple[np.ndarray, np.ndarray]:
    """Map a live /selected_object_marker (sim frame) to a GT-comparable AABB.

    z is untouched (the fitted frame is a 2D sim<->object registration only —
    same convention ``score_instruction_rubric``/``align_scene_trajectories``
    use for trajectories). When ``frame`` rotates (theta != 0) the box's four
    XY corners are individually mapped and re-hulled into an axis-aligned
    box — the same "OBB->AABB, an over-approximation" convention the GT
    loader itself uses for oriented ground-truth boxes (see the offline
    report's OBB->AABB note), applied here to keep the two IoUs comparable.
    """
    cx, cy, cz = marker["cx"], marker["cy"], marker["cz"]
    sx, sy, sz = marker["sx"], marker["sy"], marker["sz"]
    if frame is None:
        a_min = np.array([cx - sx / 2, cy - sy / 2, cz - sz / 2])
        a_max = np.array([cx + sx / 2, cy + sy / 2, cz + sz / 2])
        return a_min, a_max
    corners_xy = np.array(
        [
            [cx - sx / 2, cy - sy / 2],
            [cx - sx / 2, cy + sy / 2],
            [cx + sx / 2, cy - sy / 2],
            [cx + sx / 2, cy + sy / 2],
        ]
    )
    mapped = frame.apply(corners_xy)
    xy_min = mapped.min(axis=0)
    xy_max = mapped.max(axis=0)
    a_min = np.array([xy_min[0], xy_min[1], cz - sz / 2])
    a_max = np.array([xy_max[0], xy_max[1], cz + sz / 2])
    return a_min, a_max


# --------------------------------------------------------------------------- per-qtype scoring


def score_numerical_run(
    ctx: SceneContext, text: str, capture: BagCapture, *, answers: dict | None
) -> dict:
    live_value = capture.numerical_response
    ns = S.score_numerical(text, ctx.idx, referential=ctx.referential, scene_graph=ctx.scene_graph)
    gt_true, _pipeline_true_match, true_source, key_note = GB._true_numerical(
        answers, ctx.scene, text, live_value
    )
    true_match = None if (live_value is None or gt_true is None) else (live_value == gt_true)
    headline = None if true_match is None else (1.0 if true_match else 0.0)
    note_parts = [key_note]
    if gt_true is not None:
        note_parts.append(f"live={live_value} true={gt_true}")
    return {
        "live_answer": live_value,
        "gt_answer_true": gt_true,
        "true_source": true_source,
        "true_match": true_match,
        "gt_count_pipeline": ns.gt_count_pipeline,
        "gt_count_independent": ns.gt_count_independent,
        "independent_source": ns.independent_source,
        "headline": headline,
        "note": "; ".join(filter(None, note_parts)),
    }


def score_object_reference_run(ctx: SceneContext, text: str, capture: BagCapture) -> dict:
    ors = S.score_object_reference(text, ctx.idx, ctx.gt.instances, referential=ctx.referential)
    note_parts: list[str] = []
    if capture.marker is None:
        return {
            "live_marker": None,
            "gt_target_id": ors.gt_target_id,
            "target_source": ors.target_source,
            "match_method": ors.match_method,
            "iou": None,
            "headline": None,
            "note": "no /selected_object_marker in bag",
        }
    a_min, a_max = _marker_aabb_in_object_frame(capture.marker, ctx.frame)
    if ctx.frame is None:
        note_parts.append("no scene sim->object frame fit; marker scored in raw sim coords")
    by_id = {r.instance_id: r for r in ctx.gt.instances}
    iou: float | None = None
    if ors.gt_target_id is not None and ors.gt_target_id in by_id:
        gt_rec = by_id[ors.gt_target_id]
        iou = S.iou_from_corners(a_min, a_max, gt_rec.aabb_min, gt_rec.aabb_max)
    else:
        note_parts.append("no GT target matched; IoU undefined (flagged, not guessed)")
    live_marker = MarkerBox(
        cx=float((a_min[0] + a_max[0]) / 2), cy=float((a_min[1] + a_max[1]) / 2),
        cz=float((a_min[2] + a_max[2]) / 2),
        sx=float(a_max[0] - a_min[0]), sy=float(a_max[1] - a_min[1]), sz=float(a_max[2] - a_min[2]),
    )
    return {
        "live_marker": asdict(live_marker),
        "gt_target_id": ors.gt_target_id,
        "target_source": ors.target_source,
        "match_method": ors.match_method,
        "iou": None if iou is None else round(float(iou), 4),
        "headline": None if iou is None else round(float(iou), 4),
        "note": "; ".join(note_parts),
    }


def score_instruction_following_run(
    ctx: SceneContext, text: str, capture: BagCapture, *, questions_dir: Path
) -> dict:
    """Score a live driven trajectory with the SAME per-leg rubric geometry
    (``GB._if_rubric_geometry``) and scene frame (``ctx.frame``, fitted once
    per scene by ``GB._fit_scene_if_frame``) the offline battery uses for this
    question — only the driven trajectory itself (from the bag, mapped into
    the object frame) differs from the offline simulated-follower path.
    """
    if not ctx.if_texts:
        return {"headline": None, "note": "scene has no instruction_following questions"}
    i = _match_question(text, ctx.if_texts)
    if i is None:
        return {"headline": None, "note": "question text did not match questions.json"}
    if ctx.frame is None:
        return {
            "headline": None,
            "note": "no scene sim->object frame fit; cannot map driven trajectory",
        }
    traj_q = GB._IF_TRAJ_INDEX.get(i)
    traj_path = None
    if traj_q is not None:
        cand = questions_dir / ctx.scene / f"trajectory_q{traj_q}.ply"
        if cand.exists():
            traj_path = cand

    leg_goals, corridor_gates, avoid_caps, _leg_ids, leg_instance_aabbs = (
        GB._if_rubric_geometry(text, ctx.gt, ctx.idx, start_xy=ctx.spawn_xy)
    )
    empty_note = ""
    if capture.odom_xy.shape[0] == 0:
        empty_note = "no /state_estimation messages in bag; driven trajectory empty"
        driven_object = np.empty((0, 2))
    else:
        driven_object = ctx.frame.apply(capture.odom_xy)

    rub = S.score_instruction_rubric(
        driven_object,
        leg_goals,
        corridor_gates=corridor_gates,
        avoid_capsules=avoid_caps,
        trajectory_ply=traj_path,
        frame=None,  # driven_object is already in the object frame
        leg_instance_aabbs=leg_instance_aabbs,
    )
    return {
        "rubric_score": round(rub.rubric_score, 4),
        "ordered_leg_credit": round(rub.ordered_leg_credit, 4),
        "n_legs": rub.n_legs,
        "n_legs_reached_in_order": rub.n_legs_reached_in_order,
        "n_threading_violations": rub.n_threading_violations,
        "n_avoid_violations": rub.n_avoid_violations,
        "driven_n_poses_decimated": rub.driven_n_poses,
        "driven_n_poses_raw": capture.odom_xy_raw_n,
        "frechet_m": rub.frechet_m,
        "coverage_1m": round(rub.coverage_1m, 4) if rub.coverage_1m is not None else None,
        "if_question_index": i,
        "headline": round(rub.rubric_score, 4),
        "note": "; ".join(
            filter(
                None,
                [f"legs={rub.n_legs_reached_in_order}/{rub.n_legs}", empty_note]
                + rub.threading_details + rub.avoid_details,
            )
        ),
    }


# --------------------------------------------------------------------------- capture completeness


def _capture_issues(qdir: str, capture: BagCapture) -> list[str]:
    issues: list[str] = []
    if capture.topic_counts.get("/challenge_question", 0) == 0:
        issues.append("missing /challenge_question (0 messages) — question read from run.log")
    if qdir == "nume" and capture.topic_counts.get(TOPIC_NUM_RESPONSE, 0) == 0:
        issues.append("missing /numerical_response (0 messages) — cannot score")
    if qdir == "obje" and capture.topic_counts.get(TOPIC_MARKER, 0) == 0:
        issues.append("missing /selected_object_marker (0 messages) — cannot score")
    if qdir == "inst" and capture.topic_counts.get(TOPIC_ODOM, 0) == 0:
        issues.append("missing /state_estimation (0 messages) — cannot score")
    return issues


#: Final record of debug/<slot>/instance_index.jsonl (or the flat
#: debug/instance_index.jsonl in a single-question harvest) — one JSON
#: object per line, written by the runtime's periodic instance-index
#: dump; the last line carries the run's final tally.
INSTANCE_INDEX_FILENAME = "instance_index.jsonl"

#: A slot processing fewer than this fraction of its job's median
#: keyframe count is treated as starved. Chosen against job 702061 (#158):
#: the two genuinely-starved loft slots ran at 6% and 28% of that job's
#: median keyframe count, while every healthy slot in the same job sat
#: at >=86% of the median — a wide enough margin to absorb legitimate
#: scene-to-scene scan-length variation without false-flagging a slot
#: that's merely scoring a smaller/simpler scene.
STARVED_KEYFRAME_RATIO = 0.5

#: Below this many sibling slots in a batch, "the job's median" isn't a
#: meaningful baseline (e.g. a 2-slot batch where one slot is legitimately
#: half the other) — skip the relative check and rely on the unconditional
#: zero-instance check alone.
MIN_SLOTS_FOR_MEDIAN = 3


def _debug_dir_for_run(run_dir: Path) -> Path | None:
    """Locate the harvested job's debug dir for a scored ``run_dir``.

    Mirrors ``harvest_verify.sh``'s harvest layout: a single-question
    harvest lays out ``<OUT>/captures/<scene>/<qdir>`` next to a flat
    ``<OUT>/debug/``; a batch harvest lays out
    ``<OUT>/captures/<slot>/<scene>/<qdir>`` next to ``<OUT>/debug/<slot>/``.
    Both hang off a ``captures`` path segment, which this walks up to find.
    Returns ``None`` when ``run_dir`` isn't under a harvested ``captures/``
    tree at all (e.g. an ad hoc dev capture with no debug/ sibling) —
    the capture-completeness guard is then simply inapplicable, not an error.
    """
    parts = run_dir.resolve().parts
    if "captures" not in parts:
        return None
    idx = len(parts) - 1 - parts[::-1].index("captures")
    out_root = Path(*parts[:idx]) if idx else None
    if out_root is None:
        return None
    debug_root = out_root / "debug"
    if not debug_root.is_dir():
        return None
    tail = parts[idx + 1:]
    if len(tail) >= 3 and (debug_root / tail[0]).is_dir():
        return debug_root / tail[0]  # batch layout: debug/<slot>/
    return debug_root  # single-job flat layout


def _last_instance_index_record(debug_dir: Path) -> dict | None:
    """Last JSON record in ``debug_dir/instance_index.jsonl``, or ``None``."""
    path = debug_dir / INSTANCE_INDEX_FILENAME
    if not path.is_file():
        return None
    last_line: str | None = None
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                last_line = line
    if last_line is None:
        return None
    try:
        return json.loads(last_line)
    except json.JSONDecodeError:
        return None


def _capture_completeness_issues(run_dir: Path) -> list[str]:
    """Flag a run whose perception build looks starved (#158).

    Reads ONLY the already-written ``debug/<slot>/instance_index.jsonl``
    final record (``keyframes_processed``/``total_instances``) — no new
    runtime instrumentation. Job 702061 slot 3 processed 12 keyframes and
    built 0 instances over a full 780s run, yet reported SUCCESS and was
    scored into the aggregate as if it were a real measurement.

    Zero instances is ALWAYS flagged, unconditionally: a completed run
    that built nothing measured the harness failing, not the pipeline.
    Beyond that, a slot processing far fewer keyframes than its OTHER
    slots in the same job (median, not a fixed constant — scenes
    legitimately vary in scan length) is flagged too; see
    :data:`STARVED_KEYFRAME_RATIO`. This is complementary to the #127
    ``DEGRADED`` sentinel (handled in ``harvest_verify.sh``): that one
    fires when the sim endpoint never came up at all, this one fires when
    the run completed normally but the perception stack saw almost nothing.
    """
    debug_dir = _debug_dir_for_run(run_dir)
    if debug_dir is None:
        return []
    record = _last_instance_index_record(debug_dir)
    if record is None:
        return []

    issues: list[str] = []
    keyframes = record.get("keyframes_processed")
    instances = record.get("total_instances")

    if instances == 0:
        issues.append(
            f"perception built zero instances ({keyframes} keyframes processed) — "
            "capture-completeness failure (#158)"
        )

    is_batch_slot = debug_dir.name != "debug"
    if is_batch_slot and instances != 0 and isinstance(keyframes, (int, float)):
        peer_keyframes = []
        for peer in sorted(debug_dir.parent.iterdir()):
            if not peer.is_dir():
                continue
            peer_record = _last_instance_index_record(peer)
            if peer_record is None:
                continue
            kf = peer_record.get("keyframes_processed")
            if isinstance(kf, (int, float)):
                peer_keyframes.append(kf)
        if len(peer_keyframes) >= MIN_SLOTS_FOR_MEDIAN:
            median_kf = statistics.median(peer_keyframes)
            if median_kf > 0 and keyframes < STARVED_KEYFRAME_RATIO * median_kf:
                issues.append(
                    f"keyframes_processed={keyframes} is far below the job's "
                    f"median ({median_kf:.0f}) — capture-completeness issue (#158)"
                )
    return issues


# --------------------------------------------------------------------------- offline pairing


def _load_offline_index(path: Path) -> dict[tuple[str, str, str], dict]:
    if not path.is_file():
        return {}
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    out: dict[tuple[str, str, str], dict] = {}
    for row in data.get("scores", []):
        out[(row["scene"], row["qtype"], row["question"])] = row
    return out


def _offline_headline(qtype: str, row: dict | None) -> float | None:
    if row is None:
        return None
    if qtype == "numerical":
        tm = row.get("true_match")
        return None if tm is None else (1.0 if tm else 0.0)
    if qtype == "object_reference":
        return row.get("iou")
    if qtype == "instruction_following":
        return row.get("rubric_score")
    return None


# --------------------------------------------------------------------------- one run


def score_run(
    scene: str,
    qdir: str,
    run_dir: Path,
    *,
    groundtruth_root: Path,
    questions_dir: Path,
    questions_index: dict[str, dict[str, list[str]]],
    answers: dict | None,
    offline_index: dict[tuple[str, str, str], dict],
    offline_results_path: Path,
    unity_scenes_ros2_root: Path | None = None,
) -> dict:
    qtype = QDIR_TO_QTYPE[qdir]
    bag_dir = run_dir / "bag"
    capture = read_bag_capture(bag_dir)
    text = capture.question_text or _run_log_question(run_dir)
    capture_issues = _capture_issues(qdir, capture) + _capture_completeness_issues(run_dir)

    row: dict[str, Any] = {
        "scene": scene,
        "qdir": qdir,
        "qtype": qtype,
        "question": text,
        "run_dir": str(run_dir.relative_to(_REPO)) if run_dir.is_relative_to(_REPO) else str(run_dir),
        "capture_topic_counts": capture.topic_counts,
        "capture_issues": capture_issues,
    }

    if text is None:
        row["headline_live"] = None
        row["note"] = "no question text found (bag empty and no run.log)"
        row["offline"] = None
        row["offline_baseline_source"] = None
        row["headline_offline"] = None
        row["delta"] = None
        return row

    ctx = _load_scene_context(
        scene, groundtruth_root=groundtruth_root, questions_dir=questions_dir,
        questions_index=questions_index,
        unity_scenes_ros2_root=unity_scenes_ros2_root,
    )
    if ctx is None:
        row["headline_live"] = None
        row["note"] = f"GT scene folder not found under {groundtruth_root}"
        row["offline"] = None
        row["offline_baseline_source"] = None
        row["headline_offline"] = None
        row["delta"] = None
        return row

    row["frame_fit_residual_m"] = (
        round(ctx.fit_residual_m, 4) if ctx.fit_residual_m is not None else None
    )
    row["frame_available"] = ctx.frame is not None

    entry_questions = questions_index.get(scene, {})
    candidates = entry_questions.get(qtype, [])
    if _match_question(text, candidates) is None:
        row["capture_issues"] = capture_issues + [
            "question text did not match any questions.json entry for this scene/qtype"
        ]

    if qtype == "numerical":
        live = score_numerical_run(ctx, text, capture, answers=answers)
    elif qtype == "object_reference":
        live = score_object_reference_run(ctx, text, capture)
    else:
        live = score_instruction_following_run(ctx, text, capture, questions_dir=questions_dir)

    row["live"] = live
    row["headline_live"] = live.get("headline")

    offline_row = offline_index.get((scene, qtype, text))
    row["offline"] = offline_row
    # (#139) `offline` is NOT recomputed from this run's own bag — it is looked
    # up, by (scene, qtype, question) alone, from a FIXED baseline file that
    # never changes between invocations. Two live captures of the identical
    # question will therefore always show the identical `offline` value, no
    # matter how differently the live run itself was driven; that is expected
    # (the offline battery is a separate simulated-follower pipeline, not a
    # replay of this bag) but reads as authoritative unless its provenance is
    # explicit, so every row that carries an offline value also records
    # exactly which fixed file it came from.
    row["offline_baseline_source"] = (
        str(offline_results_path.relative_to(_REPO))
        if offline_row is not None and offline_results_path.is_relative_to(_REPO)
        else (str(offline_results_path) if offline_row is not None else None)
    )
    headline_offline = _offline_headline(qtype, offline_row)
    row["headline_offline"] = headline_offline
    if row["headline_live"] is not None and headline_offline is not None:
        row["delta"] = round(row["headline_live"] - headline_offline, 4)
    else:
        row["delta"] = None
    row["note"] = live.get("note") or ""
    return row


# --------------------------------------------------------------------------- report


def _fmt(v: Any) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def write_report(
    rows: list[dict], out_dir: Path, *, offline_results_path: Path
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / "scores.md"
    json_path = out_dir / "scores.json"

    offline_source_str = (
        str(offline_results_path.relative_to(_REPO))
        if offline_results_path.is_relative_to(_REPO)
        else str(offline_results_path)
    )

    lines = [
        f"# Live baseline scores ({date.today().isoformat()})",
        "",
        f"{len(rows)} run(s) scored. Headline metric per type: numerical = TRUE-answer "
        "match (1.0/0.0), object_reference = 3D IoU of the live marker vs the GT target "
        "box, instruction_following = rubric-proxy score (same scorers as the offline "
        "`gt_battery`; see module docstring). `delta` = live - offline on the same "
        "headline metric for the same question.",
        "",
        f"**Offline column provenance (#139):** `Offline`/`offline` is NOT recomputed "
        f"from each run's own bag — it is a fixed baseline snapshot read from "
        f"`{offline_source_str}` and looked up by (scene, qtype, question) only. It is "
        "identical across different live captures of the same question by design "
        "(the offline battery is a separate simulated-follower pipeline); it is not "
        "evidence that a change to the live driving/scoring had no effect.",
        "",
        "| Scene | Type | Live | Offline | Delta | Capture issues | Note | Question |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        issues = "; ".join(r.get("capture_issues") or []) or ""
        note = r.get("note") or ""
        q = (r.get("question") or "")[:70]
        lines.append(
            f"| {r['scene']} | {r['qdir']} | {_fmt(r.get('headline_live'))} | "
            f"{_fmt(r.get('headline_offline'))} | {_fmt(r.get('delta'))} | {issues} | "
            f"{note} | {q} |"
        )
    lines.append("")

    n_capture_issues = sum(1 for r in rows if r.get("capture_issues"))
    lines.append(
        f"Runs with a capture-completeness issue: {n_capture_issues}/{len(rows)}."
    )
    lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")

    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "n_runs": len(rows),
        "offline_baseline_source": offline_source_str,
        "rows": rows,
    }
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return md_path, json_path


def _merge_key(r: dict) -> tuple[str, str, str]:
    """Row identity for ``_merge_with_existing``.

    ``(scene, qdir)`` alone is NOT unique: every scene has TWO
    instruction_following and TWO object_reference questions, both scored
    under the same ``qdir`` (``inst``/``obje``), so that key silently
    collapsed the two rows of a type down to whichever was merged last
    (#97). The question text (``r["question"]``) is what actually
    distinguishes them, so it joins the key.

    ``question`` can be ``None``/empty when the bag's question text didn't
    match anything in ``questions.json`` (see ``score_run``/
    ``score_instruction_following_run``) — falling back to ``""`` there
    would reintroduce the same collision for two differently-broken runs
    of the same scene/qdir, so the fallback instead folds in ``run_dir``,
    which is unique per run and stable across re-invocations of that same
    run. That preserves the intended re-score semantics: re-scoring a run
    (matched question or not) recomputes the SAME key and replaces just
    that row, while every other already-scored row is left untouched.
    Every branch returns a plain ``str`` tuple so ``sorted(merged)`` below
    always compares like types.

    The question component is run through :func:`_squash` (#106) — the
    same whitespace/case-insensitive normalisation ``_match_question``
    already applies when matching a bag's captured question text against
    ``questions.json``. Without it, the merge key was STRICTER than the
    matcher that produced the value: two captures of the identical
    question differing only in case or incidental whitespace (a launcher
    change, a re-encode, a different adapter build) hashed to two
    different keys and silently duplicated the row instead of replacing
    it on re-score.
    """
    question = r.get("question")
    key_question = _squash(question) if question else f"<no-question:{r.get('run_dir', '')}>"
    return (r["scene"], r["qdir"], key_question)


def _merge_with_existing(out_dir: Path, new_rows: list[dict]) -> list[dict]:
    """Merge freshly-scored rows into any pre-existing ``scores.json`` at ``out_dir``.

    Keyed by :func:`_merge_key` (scene, qdir, question) — a re-run of one run
    dir (more bags landing, a fix applied) replaces just that run's row and
    leaves every other already-scored run's row untouched, so a partial
    re-invocation never blanks previously-scored runs (the tool is "ready
    for re-run as more bags land" per the task brief).
    """
    existing_path = out_dir / "scores.json"
    merged: dict[tuple[str, str, str], dict] = {}
    if existing_path.is_file():
        try:
            data = json.loads(existing_path.read_text(encoding="utf-8"))
            for r in data.get("rows", []):
                merged[_merge_key(r)] = r
        except (OSError, json.JSONDecodeError, KeyError):
            pass  # a corrupt/old-shape file is not fatal — we just start fresh
    for r in new_rows:
        merged[_merge_key(r)] = r
    return [merged[k] for k in sorted(merged)]


# --------------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="tools.score_live_run",
        description="Score LIVE-sim baseline bag captures against ground truth "
        "(same scorers as core.runner.gt_battery), side by side with the offline battery.",
    )
    ap.add_argument(
        "target", nargs="?", default=None,
        help="a run dir (<scene>/<qdir>) or a baseline root dir "
        f"(default: {DEFAULT_BASELINE_DIR.relative_to(_REPO)})",
    )
    ap.add_argument("--groundtruth", default=str(DEFAULT_GROUNDTRUTH))
    ap.add_argument("--questions", default=str(GB.DEFAULT_QUESTIONS))
    ap.add_argument("--questions-dir", default=str(GB.DEFAULT_QUESTIONS_ROOT))
    ap.add_argument(
        "--unity-scenes-ros2-root", default=str(GB.DEFAULT_UNITY_SCENES_ROS2_ROOT),
        help="root holding <scene>/<scene>/object_list.txt (issue #124: the live "
        "sim's own object poses, used to verify the sim<->object frame is the "
        "identity before falling back to an endpoint-correspondence fit).",
    )
    ap.add_argument("--answers", default=str(GB.DEFAULT_ANSWERS))
    ap.add_argument("--offline-results", default=str(DEFAULT_OFFLINE_RESULTS))
    ap.add_argument(
        "--out", default=None,
        help="output dir. Required whenever `target` is given (a run dir or a "
        "non-default root) — never defaults to the committed baseline dir, to "
        f"avoid clobbering it (#96). Omit only for the no-target whole-baseline "
        f"invocation, which defaults to {DEFAULT_BASELINE_DIR.relative_to(_REPO)} "
        "(merged with any existing scores.json there — see _merge_with_existing).",
    )
    args = ap.parse_args(argv)

    # Default output is the whole baseline dir ONLY for the documented no-target
    # invocation (the deliverable location per the task brief). `DEFAULT_BASELINE_DIR`
    # is a committed evidence artifact (reports/live_baseline_2026-07-20); silently
    # merging an unrelated `target` invocation's rows into it overwrote committed
    # history (#96). So any invocation that names a `target` (a single run dir or a
    # non-default root) MUST also name `--out` — we error instead of guessing where
    # the caller meant the scores to land. Checked before any of the (possibly slow
    # or failing) GT/questions loading below so a missing `--out` fails fast.
    if args.target is not None and args.out is None:
        print(
            "score_live_run: --out is required when a target is given "
            "(pass --out to choose where these scores are written; "
            f"omit target to score/merge the default {DEFAULT_BASELINE_DIR.relative_to(_REPO)})"
        )
        return 1
    out_dir = Path(args.out) if args.out else DEFAULT_BASELINE_DIR

    runs = resolve_targets(args.target)
    if not runs:
        print(f"score_live_run: no bag captures found under {args.target or DEFAULT_BASELINE_DIR}")
        return 1

    groundtruth_root = Path(args.groundtruth)
    questions_dir = Path(args.questions_dir)
    questions_index = _load_questions_index(Path(args.questions))
    answers = GB._load_answers(args.answers)
    offline_results_path = Path(args.offline_results)
    offline_index = _load_offline_index(offline_results_path)

    rows: list[dict] = []
    for scene, qdir, run_dir in runs:
        row = score_run(
            scene, qdir, run_dir,
            groundtruth_root=groundtruth_root, questions_dir=questions_dir,
            questions_index=questions_index, answers=answers, offline_index=offline_index,
            offline_results_path=offline_results_path,
            unity_scenes_ros2_root=Path(args.unity_scenes_ros2_root),
        )
        rows.append(row)
        print(
            f"score_live_run: {scene}/{qdir} live={_fmt(row.get('headline_live'))} "
            f"offline={_fmt(row.get('headline_offline'))} delta={_fmt(row.get('delta'))} "
            f"issues={row.get('capture_issues') or []}"
        )

    rows = _merge_with_existing(out_dir, rows)
    md_path, json_path = write_report(rows, out_dir, offline_results_path=offline_results_path)
    print(f"wrote {md_path}")
    print(f"wrote {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
