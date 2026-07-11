"""Battery v2 — REAL accuracy scoring against ground-truth VLA-3D scenes.

Where :mod:`core.runner.battery` scores structural health over *synthetic* scenes (no
ground truth exists for the training questions), this harness loads the actual VLA-3D
scene folders and scores each question for accuracy:

* NUMERICAL           -> exact-match count + independent second opinion
* OBJECT_REFERENCE    -> 3D IoU of our box vs the GT target box
* INSTRUCTION_FOLLOWING -> discrete Frechet + fraction of GT path within 1.0 m

The scene is *fully observed*: every GT object is an :class:`InstanceRecord` with
``n_obs=3``, so the answer heads resolve directly with no exploration needed. For
NUMERICAL/OBJECT_REFERENCE the scorers call our resolver on the GT index directly
(fast, deterministic). For INSTRUCTION_FOLLOWING we drive the full pipeline via
:func:`core.runner.single.run_question` over a GT-mirrored :class:`MockRobotIO` so the
instruction head plans a real waypoint path through a costmap built from the GT
geometry; the emitted waypoints are our path, scored against ``trajectory_qN.ply``.

Scene discovery: given ``--groundtruth <unity_root>``, we match each ``questions.json``
scene name to a subfolder of the same name (VLA-3D uses a flat ``<scene>/`` layout that
matches the challenge scene names verbatim — see docs/vla3d_notes.md §4). Scenes with no
folder present are skipped (reported as ``missing``), so a partial download still yields
a partial report.

Output: ``reports/gt_battery_<date>/`` with ``gt_battery_report.md`` (per-type accuracy
+ per-question table with failure notes) and ``gt_battery_results.json``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

import numpy as np

from core.groundtruth import scoring as S
from core.groundtruth.loader import GTScene, load_scene
from core.interfaces import QType, WaypointCmd
from core.mocks.mock_io import FakeClock, MockRobotIO
from core.mocks.synthetic_scene import Room, SyntheticScene
from core.perception.scene_index import BasicSceneIndex

_SRC = Path(__file__).resolve().parents[2]
DEFAULT_QUESTIONS = (
    _SRC.parent / "upstream" / "CMU-VLN-Challenge-2026" / "questions" / "questions.json"
)
DEFAULT_QUESTIONS_ROOT = DEFAULT_QUESTIONS.parent
DEFAULT_OUT_ROOT = _SRC.parent / "reports"

# The instruction-following trajectory files sit at questions/<scene>/trajectory_q{4,5}.ply.
# questions.json order is 1 numerical, 2 object_reference, 2 instruction_following, so the
# two IF questions map to q4 and q5 respectively (docs/vla3d_notes.md §5).
_IF_TRAJ_INDEX = {0: 4, 1: 5}

_BATTERY_TICK_HZ = 1.0


# --------------------------------------------------------------------------- scene mirror


def _synthetic_from_gt(gt: GTScene, pad: float = 1.5) -> SyntheticScene:
    """Build a SyntheticScene that mirrors the GT AABBs (for the IF costmap).

    One room bounding all GT footprints (plus padding); one box per GT instance at its
    AABB footprint centre + size. This gives the instruction head a terrain/costmap
    consistent with the geometry the heads resolve against. Region/room walls are not
    reproduced — the challenge trajectories are open-floor paths and the costmap only
    needs object obstacles + an outer boundary.
    """
    mins = np.array([r.aabb_min for r in gt.instances])
    maxs = np.array([r.aabb_max for r in gt.instances])
    x0 = float(mins[:, 0].min()) - pad
    y0 = float(mins[:, 1].min()) - pad
    x1 = float(maxs[:, 0].max()) + pad
    y1 = float(maxs[:, 1].max()) + pad

    sc = SyntheticScene(0)
    # Shift into non-negative coords is unnecessary — Room accepts arbitrary bounds.
    sc.rooms = [Room(x0, y0, x1, y1)]
    sc._split_x = None
    sc.doorway = None
    sc.objects = []
    for rec in gt.instances:
        cx = float((rec.aabb_min[0] + rec.aabb_max[0]) / 2)
        cy = float((rec.aabb_min[1] + rec.aabb_max[1]) / 2)
        sx = float(max(rec.aabb_max[0] - rec.aabb_min[0], 0.05))
        sy = float(max(rec.aabb_max[1] - rec.aabb_min[1], 0.05))
        sz = float(max(rec.aabb_max[2] - rec.aabb_min[2], 0.05))
        sc.place_box(rec.label, cx, cy, sx, sy, sz)
    return sc


_IF_MAX_BUILD_TICKS = 12  # ticks to let the instruction head ground legs + plan the route


def _drive_if_path(
    text: str,
    gt: GTScene,
    idx: BasicSceneIndex,
    *,
    max_build_ticks: int = _IF_MAX_BUILD_TICKS,
    start_xy: tuple[float, float] | None = None,
) -> np.ndarray:
    """Plan an instruction-following path over the GT scene and return it as (N, 2).

    The scene is *fully observed*, so we do NOT run the FSM's explore/watchdog loop
    (which would spin on terrain recomputation with no new information). Instead we
    drive the :class:`InstructionHead` directly for a few ticks — just enough to
    ingest terrain, ground the route legs against the GT index, and plan the path
    through the costmap — then return the head's fully planned route (the
    ``BreadcrumbFollower.path``). This is our system's intended trajectory for the
    instruction, which is what the GT ``trajectory_qN.ply`` is compared against.

    ``start_xy`` sets the robot spawn in the GT (object) frame. When the scene frame
    has been fitted, the battery passes the GT trajectory's shared start mapped into
    the object frame, so our planned path departs from the same spawn the GT path does
    (a fair like-for-like comparison). Absent a fit it defaults to the scene corner.
    """
    from core.parsing.regex_tier import parse_regex
    from core.heads.instruction import InstructionHead

    plan = parse_regex(text)
    if plan.qtype is not QType.INSTRUCTION_FOLLOWING or not plan.route:
        return np.empty((0, 2), dtype=float)

    sc = _synthetic_from_gt(gt)
    clk = FakeClock(0.0)
    if start_xy is not None:
        start_x, start_y = float(start_xy[0]), float(start_xy[1])
    else:
        start_x = float(min(r.aabb_min[0] for r in gt.instances)) + 0.5
        start_y = float(min(r.aabb_min[1] for r in gt.instances)) + 0.5
    io = MockRobotIO(sc, clk, start_x=start_x, start_y=start_y)

    head = InstructionHead(plan=plan)
    for _ in range(max_build_ticks):
        head.advance(io, idx)
        clk.advance(1.0)
        if head._follower is not None and head._follower.path:
            break

    follower = head._follower
    if follower is None or not follower.path:
        # fall back to any streamed waypoints if the follower never firmed up
        if io.waypoints:
            return np.array([[w.x, w.y] for w in io.waypoints], dtype=float)
        return np.empty((0, 2), dtype=float)
    return np.array([[p[0], p[1]] for p in follower.path], dtype=float)


def _terminal_goal_centroid(text: str, idx: BasicSceneIndex) -> np.ndarray | None:
    """Object-frame centroid of an IF question's terminal goal (final GOTO anchor).

    We take the LAST ``GOTO`` route leg's anchor, build a :class:`TargetSpec` from it
    (noun + attributes + disambiguating clause), resolve it on the GT index exactly as
    the instruction head would, and return the top candidate's XY centroid — the point
    the GT ``trajectory_qN.ply`` should end at. Returns None when the route has no GOTO
    leg or the anchor doesn't resolve (so the scene fit simply drops that endpoint).
    """
    from core.parsing.regex_tier import parse_regex
    from core.geometry.toolbox import TargetSpec, resolve
    from core.plan_schema import LegKind

    plan = parse_regex(text)
    if not plan.route:
        return None
    goto_legs = [leg for leg in plan.route if leg.kind is LegKind.GOTO and leg.anchors]
    if not goto_legs:
        return None
    anchor = goto_legs[-1].anchors[0]
    spec = TargetSpec(
        noun=anchor.noun,
        raw=anchor.raw,
        attributes=list(anchor.attributes),
        clauses=[anchor.disambiguator] if anchor.disambiguator is not None else [],
    )
    res = resolve(spec, idx)
    if not res.candidates_ranked:
        return None
    c = res.candidates_ranked[0].centroid
    return np.asarray(c, dtype=float).reshape(-1)[:2]


# --------------------------------------------------------------------------- records


@dataclass
class GTQuestionScore:
    """One scored ground-truth question."""

    scene: str
    qtype: str
    question: str
    # numerical
    our_count: int | None = None
    gt_count_pipeline: int | None = None
    exact_match: bool | None = None
    gt_count_independent: int | None = None
    independent_source: str = ""
    gt_count_scenegraph: int | None = None
    scenegraph_source: str = ""
    # object_reference
    iou: float | None = None
    gt_target_id: int | None = None
    target_source: str = ""
    match_method: str = ""
    # instruction_following
    frechet_m: float | None = None
    coverage_1m: float | None = None
    our_n_waypoints: int | None = None
    gt_n_waypoints: int | None = None
    frame_aligned: bool | None = None
    fit_residual_m: float | None = None
    note: str = ""


# --------------------------------------------------------------------------- per-scene


def score_scene(
    gt: GTScene,
    questions: dict[str, list[str]],
    *,
    referential: dict | None = None,
    scene_graph: dict | None = None,
    questions_dir: os.PathLike | str | None = None,
    drive_if: bool = True,
) -> list[GTQuestionScore]:
    """Score every question of one GT scene."""
    idx = BasicSceneIndex(gt.instances)
    out: list[GTQuestionScore] = []

    for text in questions.get("numerical", []):
        ns = S.score_numerical(text, idx, referential=referential, scene_graph=scene_graph)
        out.append(
            GTQuestionScore(
                scene=gt.scene_name, qtype=QType.NUMERICAL.value, question=text,
                our_count=ns.our_count, gt_count_pipeline=ns.gt_count_pipeline,
                exact_match=ns.exact_match, gt_count_independent=ns.gt_count_independent,
                independent_source=ns.independent_source,
                gt_count_scenegraph=ns.gt_count_scenegraph,
                scenegraph_source=ns.scenegraph_source, note=ns.note,
            )
        )

    for text in questions.get("object_reference", []):
        ors = S.score_object_reference(text, idx, gt.instances, referential=referential)
        out.append(
            GTQuestionScore(
                scene=gt.scene_name, qtype=QType.OBJECT_REFERENCE.value, question=text,
                iou=(None if ors.iou != ors.iou else round(ors.iou, 4)),
                gt_target_id=ors.gt_target_id, target_source=ors.target_source,
                match_method=ors.match_method, note=ors.note,
            )
        )

    # Instruction following: two passes. First resolve each question's terminal goal +
    # load its GT trajectory; fit ONE scene-level sim->object transform from the
    # endpoints (+ shared start); then score each with the fitted frame applied.
    if_texts = questions.get("instruction_following", [])
    if_traj: list[np.ndarray | None] = []
    if_goal: list[np.ndarray | None] = []
    for i, text in enumerate(if_texts):
        traj_q = _IF_TRAJ_INDEX.get(i)
        traj_arr: np.ndarray | None = None
        if questions_dir is not None and traj_q is not None:
            cand = Path(questions_dir) / gt.scene_name / f"trajectory_q{traj_q}.ply"
            if cand.exists():
                traj_arr = S.load_trajectory_ply(cand)
        if_traj.append(traj_arr)
        if_goal.append(_terminal_goal_centroid(text, idx) if traj_arr is not None else None)

    pairs = [
        (t, g) for t, g in zip(if_traj, if_goal) if t is not None and t.shape[0] > 0
    ]
    frame, residual = S.align_scene_trajectories(pairs) if pairs else (None, None)

    # The GT trajectory's (shared) start, mapped into the object frame, is the robot
    # spawn our planner should depart from — feed it so our path and the GT path start
    # at the same place (fair comparison). Fall back to None (scene-corner) with no fit.
    spawn_xy: tuple[float, float] | None = None
    if frame is not None and pairs:
        start_pt = pairs[0][0][0, :2]
        mapped = frame.apply(np.asarray([start_pt], dtype=float))[0]
        spawn_xy = (float(mapped[0]), float(mapped[1]))

    for i, text in enumerate(if_texts):
        traj_q = _IF_TRAJ_INDEX.get(i)
        rec = GTQuestionScore(
            scene=gt.scene_name, qtype=QType.INSTRUCTION_FOLLOWING.value, question=text,
        )
        traj_path = None
        if questions_dir is not None and traj_q is not None:
            cand = Path(questions_dir) / gt.scene_name / f"trajectory_q{traj_q}.ply"
            if cand.exists():
                traj_path = cand
        if traj_path is None:
            rec.note = "no GT trajectory file found; IF unscored"
            out.append(rec)
            continue
        our_path = (
            _drive_if_path(text, gt, idx, start_xy=spawn_xy)
            if drive_if
            else np.empty((0, 2), dtype=float)
        )
        isc = S.score_instruction_following(
            our_path, traj_path, frame=frame, fit_residual_m=residual
        )
        rec.frechet_m = round(isc.frechet_m, 4) if np.isfinite(isc.frechet_m) else None
        rec.coverage_1m = round(isc.coverage_1m, 4)
        rec.our_n_waypoints = isc.our_n_waypoints
        rec.gt_n_waypoints = isc.gt_n_waypoints
        rec.frame_aligned = isc.frame_aligned
        rec.fit_residual_m = round(residual, 4) if residual is not None else None
        rec.note = isc.note
        out.append(rec)

    return out


# --------------------------------------------------------------------------- discovery


def _find_scene_folder(unity_root: Path, scene_name: str) -> Path | None:
    """Locate a scene's folder under the Unity root (verbatim name match)."""
    direct = unity_root / scene_name
    if direct.is_dir():
        return direct
    # Some zips nest under a 'Unity/' dir; probe one level down.
    nested = unity_root / "Unity" / scene_name
    if nested.is_dir():
        return nested
    for child in unity_root.iterdir() if unity_root.is_dir() else []:
        if child.is_dir() and (child / f"{scene_name}_object_result.csv").exists():
            return child
        cand = child / scene_name
        if cand.is_dir():
            return cand
    return None


def _load_referential(folder: Path, scene_name: str) -> dict | None:
    p = folder / f"{scene_name}_referential_statements.json"
    if p.exists():
        with open(p, encoding="utf-8") as fh:
            return json.load(fh)
    return None


def _load_scene_graph(folder: Path, scene_name: str) -> dict | None:
    p = folder / f"{scene_name}_scene_graph.json"
    if p.exists():
        with open(p, encoding="utf-8") as fh:
            return json.load(fh)
    return None


def run_gt_battery(
    unity_root: os.PathLike | str,
    *,
    questions_path: os.PathLike | str = DEFAULT_QUESTIONS,
    questions_dir: os.PathLike | str = DEFAULT_QUESTIONS_ROOT,
    scenes: list[str] | None = None,
    drive_if: bool = True,
) -> tuple[list[GTQuestionScore], list[str]]:
    """Score every question whose scene folder is present under ``unity_root``.

    Returns (scores, missing_scenes).
    """
    root = Path(unity_root)
    with open(questions_path, encoding="utf-8") as fh:
        data = json.load(fh)

    scores: list[GTQuestionScore] = []
    missing: list[str] = []
    for entry in data:
        scene_name = entry["scene"]
        if scenes is not None and scene_name not in scenes:
            continue
        folder = _find_scene_folder(root, scene_name)
        if folder is None:
            missing.append(scene_name)
            continue
        gt = load_scene(folder, scene_name=scene_name)
        referential = _load_referential(folder, scene_name)
        scene_graph = _load_scene_graph(folder, scene_name)
        scores.extend(
            score_scene(
                gt,
                entry["questions"],
                referential=referential,
                scene_graph=scene_graph,
                questions_dir=questions_dir,
                drive_if=drive_if,
            )
        )
    return scores, missing


# --------------------------------------------------------------------------- aggregate


def aggregate(scores: list[GTQuestionScore]) -> dict:
    """Per-type accuracy topline."""
    num = [s for s in scores if s.qtype == QType.NUMERICAL.value]
    obj = [s for s in scores if s.qtype == QType.OBJECT_REFERENCE.value]
    inf = [s for s in scores if s.qtype == QType.INSTRUCTION_FOLLOWING.value]

    def _mean(vals: list[float]) -> float | None:
        vals = [v for v in vals if v is not None]
        return round(float(np.mean(vals)), 4) if vals else None

    num_exact = [1.0 if s.exact_match else 0.0 for s in num]
    num_agree = [
        1.0 if (s.gt_count_independent is not None and s.our_count == s.gt_count_independent) else 0.0
        for s in num
        if s.gt_count_independent is not None
    ]
    num_sg_agree = [
        1.0 if (s.gt_count_scenegraph is not None and s.our_count == s.gt_count_scenegraph) else 0.0
        for s in num
        if s.gt_count_scenegraph is not None
    ]
    obj_iou = [s.iou for s in obj if s.iou is not None]
    obj_scored = [s for s in obj if s.iou is not None]
    # match-method breakdown across ALL object-reference questions
    method_counts: dict[str, int] = {}
    for s in obj:
        m = s.match_method or "none"
        method_counts[m] = method_counts.get(m, 0) + 1

    # Instruction following: aligned vs unaligned scenes (diagnostic-only unaligned).
    inf_aligned = [s for s in inf if s.frame_aligned]
    unaligned_scenes = sorted({s.scene for s in inf if s.frame_aligned is False})
    return {
        "numerical": {
            "n": len(num),
            "exact_match_rate_pipeline": _mean(num_exact),
            "n_with_independent": len(num_agree),
            "independent_agreement_rate": _mean(num_agree) if num_agree else None,
            "n_with_scenegraph": len(num_sg_agree),
            "scenegraph_agreement_rate": _mean(num_sg_agree) if num_sg_agree else None,
        },
        "object_reference": {
            "n": len(obj),
            "n_scored": len(obj_scored),
            "mean_iou": _mean(obj_iou),
            "iou_at_0p25": _mean([1.0 if v >= 0.25 else 0.0 for v in obj_iou]) if obj_iou else None,
            "iou_at_0p5": _mean([1.0 if v >= 0.5 else 0.0 for v in obj_iou]) if obj_iou else None,
            "match_method_breakdown": method_counts,
        },
        "instruction_following": {
            "n": len(inf),
            "n_aligned": len(inf_aligned),
            "n_unaligned_scenes": len(unaligned_scenes),
            "unaligned_scenes": unaligned_scenes,
            "mean_frechet_m_aligned": _mean([s.frechet_m for s in inf_aligned]),
            "mean_coverage_1m_aligned": _mean([s.coverage_1m for s in inf_aligned]),
            "mean_frechet_m_all": _mean([s.frechet_m for s in inf]),
            "mean_coverage_1m_all": _mean([s.coverage_1m for s in inf]),
        },
    }


# --------------------------------------------------------------------------- report


def _md_table(scores: list[GTQuestionScore]) -> str:
    header = (
        "| Scene | Type | Metric(s) | Note | Question |\n"
        "|---|---|---|---|---|\n"
    )
    rows = []
    for s in scores:
        q = s.question if len(s.question) <= 55 else s.question[:52] + "..."
        if s.qtype == QType.NUMERICAL.value:
            indep = (
                f", indep={s.gt_count_independent}({s.independent_source})"
                if s.gt_count_independent is not None
                else ""
            )
            sg = (
                f", sg={s.gt_count_scenegraph}({s.scenegraph_source})"
                if s.gt_count_scenegraph is not None
                else ""
            )
            metric = f"count={s.our_count} pipeline_gt={s.gt_count_pipeline}{indep}{sg}"
        elif s.qtype == QType.OBJECT_REFERENCE.value:
            iou = "n/a" if s.iou is None else f"{s.iou:.3f}"
            metric = (
                f"IoU={iou} tgt={s.gt_target_id}({s.target_source}"
                f"/{s.match_method or 'none'})"
            )
        else:
            fr = "n/a" if s.frechet_m is None else f"{s.frechet_m:.2f}m"
            cov = "n/a" if s.coverage_1m is None else f"{s.coverage_1m:.0%}"
            resid = "" if s.fit_residual_m is None else f" fit={s.fit_residual_m:.2f}m"
            metric = (
                f"Frechet={fr} cover1m={cov}{resid} "
                f"(ours {s.our_n_waypoints}/gt {s.gt_n_waypoints})"
            )
        note = s.note or ""
        rows.append(f"| {s.scene} | {s.qtype[:4]} | {metric} | {note} | {q} |")
    return header + "\n".join(rows) + "\n"


def write_report(
    scores: list[GTQuestionScore], missing: list[str], out_dir: os.PathLike | str
) -> tuple[Path, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    agg = aggregate(scores)

    md_path = out / "gt_battery_report.md"
    json_path = out / "gt_battery_results.json"

    scenes = sorted({s.scene for s in scores})
    lines: list[str] = []
    lines.append(f"# GT battery — REAL accuracy report ({date.today().isoformat()})\n")
    lines.append(
        f"{len(scores)} question(s) across {len(scenes)} ground-truth scene(s): "
        f"{', '.join(scenes) or '(none)'}. "
        f"Missing/skipped scenes: {', '.join(missing) or 'none'}.\n"
    )
    lines.append(
        "> **Circularity note (numerical):** the primary `pipeline_gt` count is OUR "
        "resolver run over the ground-truth geometry, so exact-match validates "
        "*pipeline self-consistency*, not absolute truth. The `indep` column is an "
        "independent second opinion from the referential-statement annotations "
        "(distinct annotated target instances); `referential_class_only` means the "
        "count is relation-agnostic (coarser). Disagreements are the informative "
        "signal.\n"
    )
    lines.append(
        "> **OBB->AABB note:** GT boxes are the axis-aligned hull of each object's "
        "oriented box (rotated corners, min/max), a strict over-approximation for "
        "non-axis-aligned objects — small IoU deficits on rotated targets are partly "
        "this, not localisation error.\n"
    )

    n, o, i = agg["numerical"], agg["object_reference"], agg["instruction_following"]
    method_str = ", ".join(
        f"{k}={v}" for k, v in sorted(o["match_method_breakdown"].items())
    ) or "none"
    lines.append("## Topline (per type)\n")
    lines.append(
        f"- **Numerical** (n={n['n']}): pipeline exact-match "
        f"{_pct(n['exact_match_rate_pipeline'])}; independent (referential) agreement "
        f"{_pct(n['independent_agreement_rate'])} over {n['n_with_independent']}; "
        f"scene-graph agreement {_pct(n['scenegraph_agreement_rate'])} over "
        f"{n['n_with_scenegraph']}.\n"
        f"- **Object reference** (n={o['n']}, scored={o['n_scored']}): mean 3D IoU "
        f"{_num(o['mean_iou'])}; IoU>=0.25 {_pct(o['iou_at_0p25'])}; IoU>=0.5 "
        f"{_pct(o['iou_at_0p5'])}. Match method: {method_str}.\n"
        f"- **Instruction following** (n={i['n']}): aligned scenes score mean discrete "
        f"Frechet {_num(i['mean_frechet_m_aligned'])} m; mean coverage within 1.0 m "
        f"{_pct(i['mean_coverage_1m_aligned'])} (over {i['n_aligned']} aligned Qs). "
        f"{i['n_unaligned_scenes']} scene(s) unaligned (residual > 1 m, diagnostic-only)"
        f"{': ' + ', '.join(i['unaligned_scenes']) if i['unaligned_scenes'] else ''}.\n"
    )
    lines.append("\n## Per-question\n")
    lines.append(_md_table(scores))

    md_path.write_text("\n".join(lines), encoding="utf-8")

    payload = {
        "date": date.today().isoformat(),
        "n_questions": len(scores),
        "scenes": scenes,
        "missing_scenes": missing,
        "aggregate": agg,
        "scores": [asdict(s) for s in scores],
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return md_path, json_path


def _pct(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.0%}"


def _num(v: float | None) -> str:
    return "n/a" if v is None else f"{v:.3f}"


# --------------------------------------------------------------------------- CLI


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="core.runner.gt_battery",
        description="Score the training questions against ground-truth VLA-3D scenes (REAL accuracy).",
    )
    ap.add_argument(
        "--groundtruth", required=True,
        help="Unity root dir containing per-scene folders (matched to questions.json scenes).",
    )
    ap.add_argument("--questions", default=str(DEFAULT_QUESTIONS), help="questions.json path")
    ap.add_argument(
        "--questions-dir", default=str(DEFAULT_QUESTIONS_ROOT),
        help="dir holding <scene>/trajectory_q*.ply (default: the challenge questions dir)",
    )
    ap.add_argument("--out", default=None, help="output dir (default reports/gt_battery_<date>/)")
    ap.add_argument("--scenes", default=None, help="comma-separated scene subset")
    ap.add_argument(
        "--no-drive-if", action="store_true",
        help="skip driving instruction-following paths (score IF as empty path)",
    )
    args = ap.parse_args(argv)

    scenes = [s.strip() for s in args.scenes.split(",")] if args.scenes else None
    out_dir = args.out or (DEFAULT_OUT_ROOT / f"gt_battery_{date.today().isoformat()}")

    scores, missing = run_gt_battery(
        args.groundtruth,
        questions_path=args.questions,
        questions_dir=args.questions_dir,
        scenes=scenes,
        drive_if=not args.no_drive_if,
    )
    if not scores:
        print(f"gt_battery: no scenes found under {args.groundtruth} (missing={missing})")
        return 1
    md_path, json_path = write_report(scores, missing, out_dir)

    agg = aggregate(scores)
    n, o, i = agg["numerical"], agg["object_reference"], agg["instruction_following"]
    print(
        f"gt_battery: {len(scores)} questions / {len({s.scene for s in scores})} scenes  "
        f"num_exact={_pct(n['exact_match_rate_pipeline'])} "
        f"or_iou={_num(o['mean_iou'])} "
        f"if_cover1m_aligned={_pct(i['mean_coverage_1m_aligned'])} "
        f"if_unaligned={i['n_unaligned_scenes']}"
    )
    print(f"wrote {md_path}")
    print(f"wrote {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
