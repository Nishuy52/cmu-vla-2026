"""GT leg-goal reachability ceiling (T11).

For every instruction-following question in the battery, resolve the scorer's ordered
leg goals (exactly as ``gt_battery._if_rubric_geometry`` does) and measure the GT
REFERENCE trajectory's minimum distance to each — the trajectory mapped into the object
frame by the same per-scene rigid fit the battery uses. A leg counts as "reached" when
that minimum distance is <= ``LEG_ARRIVAL_TOL_M`` (0.8 m).

This makes the "GT reference reaches K/N leg goals" ceiling claim reproducible and
auditable: the canonical correct path can only score ordered-leg credit up to K/N, so
the residual battery gap above K/N is a scorer/geometry property, not a pipeline defect.

Run (from ``src/`` with the GT Unity root present)::

    python -m core.runner.gt_leg_ceiling \
        --groundtruth ../data/vla3d/Unity \
        --out ../reports/gt_battery_postT11_2026-07-14/gt_leg_ceiling.json

Emits a JSON artifact (per-question leg rows + the K/N summary) and prints a short
table. ``--groundtruth`` must point at the Unity root that holds the 15 battery scenes
(each ``<scene>/`` with its VLA-3D object CSV); ``--questions-dir`` defaults to the
challenge questions dir that ships the ``trajectory_q{4,5}.ply`` reference paths.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from core.groundtruth import scoring as S
from core.groundtruth.loader import load_scene
from core.perception.scene_index import BasicSceneIndex
from core.runner.gt_battery import (
    DEFAULT_QUESTIONS,
    DEFAULT_QUESTIONS_ROOT,
    _IF_TRAJ_INDEX,
    _find_scene_folder,
    _if_rubric_geometry,
    _terminal_goal_centroid,
)


def measure(groundtruth: Path, questions: Path, questions_dir: Path) -> dict:
    data = json.loads(Path(questions).read_text(encoding="utf-8"))
    tol = S.LEG_ARRIVAL_TOL_M
    per_question: list[dict] = []
    total_legs = 0
    reached_legs = 0
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

        # Per-scene rigid fit (trajectory frame -> object frame), exactly as the battery.
        if_traj: list[np.ndarray | None] = []
        if_goal: list[np.ndarray | None] = []
        for i, text in enumerate(if_texts):
            tq = _IF_TRAJ_INDEX.get(i)
            arr: np.ndarray | None = None
            if tq is not None:
                cand = Path(questions_dir) / scene / f"trajectory_q{tq}.ply"
                if cand.exists():
                    arr = S.load_trajectory_ply(cand)
            if_traj.append(arr)
            if_goal.append(_terminal_goal_centroid(text, idx) if arr is not None else None)
        pairs = [(t, g) for t, g in zip(if_traj, if_goal) if t is not None and t.shape[0] > 0]
        frame, residual = S.align_scene_trajectories(pairs) if pairs else (None, None)

        for i, text in enumerate(if_texts):
            arr = if_traj[i]
            row = {
                "scene": scene,
                "traj_q": _IF_TRAJ_INDEX.get(i),
                "question": text,
                "fit_residual_m": None if residual is None else round(float(residual), 4),
                "legs": [],
                "n_legs": 0,
                "n_reached": 0,
            }
            if arr is None:
                row["note"] = "no GT trajectory PLY"
                per_question.append(row)
                continue
            gt_xy = frame.apply(arr) if frame is not None else arr[:, :2]
            leg_goals, _gates, _avoid, _iids = _if_rubric_geometry(text, gt, idx)
            for k, (kind, goal) in enumerate(leg_goals):
                d = float(np.min(np.linalg.norm(gt_xy - np.array(goal), axis=1)))
                reached = d <= tol
                row["legs"].append({
                    "index": k, "kind": kind,
                    "goal_xy": [round(float(goal[0]), 4), round(float(goal[1]), 4)],
                    "gt_min_dist_m": round(d, 4), "reached": reached,
                })
                total_legs += 1
                if reached:
                    reached_legs += 1
            row["n_legs"] = len(leg_goals)
            row["n_reached"] = sum(1 for l in row["legs"] if l["reached"])
            per_question.append(row)

    return {
        "tol_m": tol,
        "gt_reached_legs": reached_legs,
        "total_legs": total_legs,
        "ratio": (round(reached_legs / total_legs, 4) if total_legs else None),
        "scenes": scenes_seen,
        "how_to_regenerate": {
            "cwd": "src/",
            "command": (
                "python -m core.runner.gt_leg_ceiling "
                "--groundtruth <UNITY_ROOT> "
                "--out ../reports/gt_battery_postT11_2026-07-14/gt_leg_ceiling.json"
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
    args = ap.parse_args(argv)

    result = measure(Path(args.groundtruth), Path(args.questions), Path(args.questions_dir))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")

    r = result
    print(f"GT reference reaches {r['gt_reached_legs']}/{r['total_legs']} leg goals "
          f"within {r['tol_m']} m  (ratio {r['ratio']})")
    print(f"scenes: {len(r['scenes'])}   wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
