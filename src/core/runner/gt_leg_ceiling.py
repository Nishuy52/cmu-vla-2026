"""GT leg-goal reachability ceiling (T11, reworked for issue #70).

For every instruction-following question in the battery, resolve the scorer's ordered
leg goals (exactly as ``gt_battery._if_rubric_geometry`` does) and measure the GT
REFERENCE trajectory's distance to each — the trajectory mapped into the object frame
by the same per-scene rigid fit the battery uses. A leg counts as "reached" when that
distance is within its DERIVED tolerance (issue #70: no longer the asserted 0.8 m --
see ``core.groundtruth.arrival``), using the SAME per-leg-kind STOP/PASS-BY semantics
``score_instruction_rubric`` applies (terminal goto / corridor_between = fixed-tolerance
STOP; via_near / non-terminal goto = closest-approach PASS-BY, tolerance widened by the
referenced instance's own AABB half-diagonal).

TWO distance metrics are reported per leg (issue #70 methodology fix, from #69's
finding that a WHOLE-TRAJECTORY comparison inflates apparent reachability by letting a
leg match ANY point of the GT path, including one that "belongs" to a different leg):

* ``gt_min_dist_whole_m`` — distance from the leg goal to the nearest point anywhere on
  the full GT trajectory (the ORIGINAL, unrestricted method — kept for comparison only,
  to size the whole-vs-segment artifact; do not treat this as the trustworthy ceiling).
* ``gt_min_dist_segment_m`` — distance from the leg goal to the nearest point on the GT
  trajectory at-or-after an ORDERED CURSOR that only advances past a point once some
  earlier leg has matched it (mirrors ``score_instruction_rubric``'s own ordered-arrival
  cursor, applied to the reference path instead of the driven one). This is the leg-
  attributed, non-inflated measurement -- the recommended ceiling.

Both K/N ratios are reported (``ratio_whole`` / ``ratio_segment``) so the size of the
whole-trajectory artifact is directly visible, not just asserted.

This makes the "GT reference reaches K/N leg goals" ceiling claim reproducible and
auditable: the canonical correct path can only score ordered-leg credit up to K/N, so
the residual battery gap above K/N is a scorer/geometry property, not a pipeline defect.

Run (from ``src/`` with the GT Unity root present)::

    python -m core.runner.gt_leg_ceiling \
        --groundtruth ../data/vla3d/Unity \
        --out ../reports/leg_ceiling_post70.json

Emits a JSON artifact (per-question leg rows + the K/N summary, both metrics) and
prints a short table. ``--groundtruth`` must point at the Unity root that holds the 15
battery scenes (each ``<scene>/`` with its VLA-3D object CSV); ``--questions-dir``
defaults to the challenge questions dir that ships the ``trajectory_q{4,5}.ply``
reference paths.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from core.geometry import primitives as P
from core.groundtruth import scoring as S
from core.groundtruth.loader import load_scene
from core.perception.scene_index import BasicSceneIndex
from core.runner.gt_battery import (
    DEFAULT_QUESTIONS,
    DEFAULT_QUESTIONS_ROOT,
    DEFAULT_UNITY_SCENES_ROS2_ROOT,
    _IF_TRAJ_INDEX,
    _fit_scene_if_frame,
    _find_scene_folder,
    _if_rubric_geometry,
    collect_scene_fit_residuals,
)


def _leg_tol(kind: str, index: int, n_legs: int, aabb, base_tol: float) -> tuple[float, bool]:
    """Per-leg tolerance + pass-by flag, mirroring ``score_instruction_rubric`` (#70)."""
    pass_by = S._is_pass_by_leg(kind, index, n_legs)
    if pass_by and aabb is not None:
        half_diag = P.footprint_diagonal(aabb[0], aabb[1]) / 2.0
        return half_diag + base_tol, pass_by
    return base_tol, pass_by


def measure(
    groundtruth: Path,
    questions: Path,
    questions_dir: Path,
    *,
    tol: float | None = None,
    unity_scenes_ros2_root: Path | None = None,
) -> dict:
    """Measure the GT-reference leg-arrival ceiling.

    ``tol`` (issue #70): the STOP-leg base tolerance. ``None`` (the default) derives
    it from THIS run's own p95 fit residual across scenes
    (:func:`core.runner.gt_battery.collect_scene_fit_residuals` +
    :func:`core.groundtruth.arrival.derived_arrival_tol_m`) -- a cheap no-driving
    pre-pass, same mechanism ``run_gt_battery(..., derive_tol=True)`` uses -- so the
    ceiling probe uses the SAME live yardstick a battery run derives, not a hardcoded
    number. Pass an explicit ``tol`` to override (e.g. to reproduce a past run's fixed
    value for a regression comparison).
    """
    data = json.loads(Path(questions).read_text(encoding="utf-8"))

    tol_source = "explicit"
    p95_residual: float | None = None
    if tol is None:
        residuals = collect_scene_fit_residuals(
            groundtruth, questions_path=questions, questions_dir=questions_dir,
            unity_scenes_ros2_root=unity_scenes_ros2_root,
        )
        finite = [r for r in residuals.values() if r is not None]
        if finite:
            p95_residual = float(np.percentile(finite, 95))
            tol = S.derived_arrival_tol_m(p95_residual)
            tol_source = "derived_live_p95"
        else:
            tol = S.LEG_ARRIVAL_TOL_M
            tol_source = "nominal_fallback_no_residuals"

    per_question: list[dict] = []
    total_legs = 0
    reached_whole = 0
    reached_segment = 0
    scenes_seen: list[str] = []
    for entry in data:
        scene = entry["scene"]
        folder = _find_scene_folder(Path(groundtruth), scene)
        if folder is None:
            continue
        gt = load_scene(str(folder), scene)
        idx = BasicSceneIndex(gt.instances)
        scenes_seen.append(scene)
        if_texts = entry["questions"].get("instruction_following", [])

        # Per-scene sim->object frame, resolved the same way the battery does (issue
        # #146): IDENTITY-first via object_list.txt id match, endpoint-correspondence
        # fit only as a gated fallback (see GB._fit_scene_if_frame's docstring).
        # Calling S.align_scene_trajectories directly here would bypass that fix and
        # can produce a nonsense-but-self-consistent frame (issue #124).
        if_traj, _if_cands, frame, residual, _pairs = _fit_scene_if_frame(
            gt, idx, if_texts, questions_dir,
            unity_scenes_ros2_root=unity_scenes_ros2_root,
        )

        for i, text in enumerate(if_texts):
            arr = if_traj[i]
            row = {
                "scene": scene,
                "traj_q": _IF_TRAJ_INDEX.get(i),
                "question": text,
                "fit_residual_m": None if residual is None else round(float(residual), 4),
                "legs": [],
                "n_legs": 0,
                "n_reached_whole": 0,
                "n_reached_segment": 0,
            }
            if arr is None:
                row["note"] = "no GT trajectory PLY"
                per_question.append(row)
                continue
            gt_xy = frame.apply(arr) if frame is not None else arr[:, :2]
            leg_goals, _gates, _avoid, _iids, leg_aabbs = _if_rubric_geometry(text, gt, idx)
            n_legs = len(leg_goals)
            cursor = 0  # ordered-cursor frontier over gt_xy (mirrors the scorer)
            for k, (kind, goal) in enumerate(leg_goals):
                aabb = leg_aabbs[k] if k < len(leg_aabbs) else None
                leg_tol, pass_by = _leg_tol(kind, k, n_legs, aabb, tol)
                g = np.array(goal)

                # WHOLE-trajectory metric (original method; kept to size the artifact).
                d_whole = float(np.min(np.linalg.norm(gt_xy - g, axis=1)))
                reached_w = d_whole <= leg_tol

                # SEGMENT (ordered-cursor) metric: only the GT path from the cursor
                # onward is attributable to this leg, mirroring how
                # score_instruction_rubric walks the DRIVEN trajectory once with an
                # advancing cursor -- applied here to the REFERENCE trajectory.
                suffix = gt_xy[cursor:]
                if suffix.shape[0] == 0:
                    d_segment = float("inf")
                    reached_s = False
                else:
                    dists = np.linalg.norm(suffix - g, axis=1)
                    j = int(np.argmin(dists))
                    d_segment = float(dists[j])
                    reached_s = d_segment <= leg_tol
                    if reached_s:
                        cursor += j + 1  # advance past the matched point, in order

                row["legs"].append({
                    "index": k, "kind": kind, "pass_by": pass_by,
                    "tol_used_m": round(leg_tol, 4),
                    "goal_xy": [round(float(goal[0]), 4), round(float(goal[1]), 4)],
                    "gt_min_dist_whole_m": round(d_whole, 4),
                    "reached_whole": reached_w,
                    "gt_min_dist_segment_m": (
                        None if not np.isfinite(d_segment) else round(d_segment, 4)
                    ),
                    "reached_segment": reached_s,
                })
                total_legs += 1
                if reached_w:
                    reached_whole += 1
                if reached_s:
                    reached_segment += 1
            row["n_legs"] = n_legs
            row["n_reached_whole"] = sum(1 for l in row["legs"] if l["reached_whole"])
            row["n_reached_segment"] = sum(1 for l in row["legs"] if l["reached_segment"])
            per_question.append(row)

    return {
        "tol_m": tol,
        "tol_source": tol_source,
        "p95_fit_residual_m": (
            None if p95_residual is None else round(p95_residual, 4)
        ),
        "gt_reached_legs_whole": reached_whole,
        "gt_reached_legs_segment": reached_segment,
        "total_legs": total_legs,
        "ratio_whole": (round(reached_whole / total_legs, 4) if total_legs else None),
        "ratio_segment": (round(reached_segment / total_legs, 4) if total_legs else None),
        "scenes": scenes_seen,
        "how_to_regenerate": {
            "cwd": "src/",
            "command": (
                "python -m core.runner.gt_leg_ceiling "
                "--groundtruth <UNITY_ROOT> "
                "--out ../reports/leg_ceiling_post70.json"
            ),
            "data_root_note": (
                "<UNITY_ROOT> is the VLA-3D Unity root holding the 15 battery scenes "
                "(each <scene>/ with its object CSV + scene graph). In this repo tree it "
                "is data/vla3d/Unity (git-ignored fixtures). questions-dir defaults to the "
                "challenge questions dir shipping trajectory_q{4,5}.ply."
            ),
        },
        "per_question": per_question,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--groundtruth", required=True, help="Unity root with the battery scenes")
    ap.add_argument("--questions", default=str(DEFAULT_QUESTIONS))
    ap.add_argument("--questions-dir", default=str(DEFAULT_QUESTIONS_ROOT))
    ap.add_argument("--out", required=True, help="output JSON path")
    ap.add_argument(
        "--tol", type=float, default=None,
        help="override: fixed STOP-leg tolerance (m). Default derives it live from "
             "this run's own p95 fit residual (issue #70).",
    )
    ap.add_argument(
        "--unity-scenes-ros2-root", default=str(DEFAULT_UNITY_SCENES_ROS2_ROOT),
        help="root holding <scene>/object_list.txt, used to resolve the sim->object "
             "frame identity-first (issue #124/#146; see GB._fit_scene_if_frame).",
    )
    args = ap.parse_args(argv)

    result = measure(
        Path(args.groundtruth), Path(args.questions), Path(args.questions_dir),
        tol=args.tol,
        unity_scenes_ros2_root=args.unity_scenes_ros2_root,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    r = result
    print(
        f"GT reference reaches {r['gt_reached_legs_whole']}/{r['total_legs']} "
        f"(whole-trajectory) / {r['gt_reached_legs_segment']}/{r['total_legs']} "
        f"(per-segment) leg goals within {r['tol_m']:.4f} m "
        f"(tol_source={r['tol_source']}, p95_residual={r['p95_fit_residual_m']})"
    )
    print(f"ratio_whole={r['ratio_whole']}  ratio_segment={r['ratio_segment']}")
    print(f"scenes: {len(r['scenes'])}   wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
