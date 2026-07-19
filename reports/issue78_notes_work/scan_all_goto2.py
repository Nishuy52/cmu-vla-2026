"""Issue #78 blast-radius scan v2: for every flagged goto leg (>1.0m from rubric
goal), classify whether the head's stale/wrong point is explained by the
pinch-crossing BFS-disconnection mechanism (leg immediately follows a
CORRIDOR_BETWEEN leg whose gate needed pinch relaxation to cross) vs. some
other cause (e.g. legitimately different anchor resolution or a genuinely
far, non-pinch-related nearest-reachable snap)."""
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np

from core.groundtruth.loader import load_scene
import core.runner.gt_battery as GB
from core.perception.scene_index import BasicSceneIndex
from core.plan_schema import LegKind

ROOT = Path("../data/vla3d/Unity")

with open(GB.DEFAULT_QUESTIONS, encoding="utf-8") as fh:
    data = json.load(fh)

results = []
for entry in data:
    scene_name = entry["scene"]
    folder = GB._find_scene_folder(ROOT, scene_name)
    if folder is None:
        continue
    gt = load_scene(folder, scene_name=scene_name)
    idx = BasicSceneIndex(gt.instances)
    if_texts = entry["questions"].get("instruction_following", [])
    if not if_texts:
        continue

    if_traj, if_cands, frame, residual, pairs = GB._fit_scene_if_frame(
        gt, idx, if_texts, GB.DEFAULT_QUESTIONS_ROOT
    )
    frame_for_walls = frame if residual is None or residual <= GB.WALL_FIT_MAX_RESIDUAL_M else None
    wall_cells = GB._scene_wall_cells(
        gt, frame_for_walls, unity_scenes_ros2_root=GB.DEFAULT_UNITY_SCENES_ROS2_ROOT
    )

    for qi, text in enumerate(if_texts):
        spawn_xy = None
        if frame is not None and pairs and qi < len(pairs):
            start_pt = pairs[qi][0][0, :2]
            mapped = frame.apply(np.asarray([start_pt], dtype=float))[0]
            spawn_xy = (float(mapped[0]), float(mapped[1]))
        try:
            head, io, plan = GB._run_instruction_head(
                text, gt, idx, start_xy=spawn_xy, max_build_ticks=GB._IF_MAX_BUILD_TICKS,
                wall_cells=wall_cells,
            )
        except Exception:
            continue
        if head is None or head._legs is None:
            continue
        try:
            rubric = GB._if_rubric_geometry(text, gt, idx, start_xy=spawn_xy)
            leg_goals = rubric[0]
        except Exception:
            leg_goals = None

        cm = head._costmap
        pose = head._pose
        seen = cm.reachable_mask(pose) if cm is not None else None
        passable_total = int((~(cm.base_blocked | cm.capsule_blocked)).sum()) if cm is not None else None
        seen_frac = float(seen.sum()) / passable_total if seen is not None and passable_total else None

        for li, leg in enumerate(head._legs):
            if leg.kind is not LegKind.GOTO or leg.geom is None:
                continue
            gx, gy = leg.geom
            rub_xy = None
            if leg_goals is not None and li < len(leg_goals):
                kind, xy = leg_goals[li]
                if kind == "goto":
                    rub_xy = xy
            dist_to_rubric = math.hypot(gx - rub_xy[0], gy - rub_xy[1]) if rub_xy else None
            prev_kind = head._legs[li - 1].kind.name if li > 0 else None

            results.append({
                "scene": scene_name, "q": qi, "leg": li, "text": text[:70],
                "head_goal": [round(gx, 4), round(gy, 4)],
                "rubric_goal": [round(rub_xy[0], 4), round(rub_xy[1], 4)] if rub_xy else None,
                "dist_to_rubric": round(dist_to_rubric, 4) if dist_to_rubric is not None else None,
                "prev_leg_kind": prev_kind,
                "seen_frac": round(seen_frac, 4) if seen_frac is not None else None,
            })

with open(Path(__file__).parent / "goto_scan2.json", "w") as fh:
    json.dump(results, fh, indent=2)

flagged = [r for r in results if r.get("dist_to_rubric") is not None and r["dist_to_rubric"] > 1.0]
print(f"Total goto legs: {len(results)}, flagged (>1.0m): {len(flagged)}")
after_corridor = [r for r in flagged if r["prev_leg_kind"] == "CORRIDOR_BETWEEN"]
print(f"Flagged AND immediately follow a corridor_between leg: {len(after_corridor)}")
low_seen = [r for r in flagged if r["seen_frac"] is not None and r["seen_frac"] < 0.1]
print(f"Flagged AND seen_frac < 0.1 (tiny disconnected BFS pocket): {len(low_seen)}")
both = [r for r in flagged if r["prev_leg_kind"] == "CORRIDOR_BETWEEN" and r["seen_frac"] is not None and r["seen_frac"] < 0.5]
print(f"Flagged AND after-corridor AND seen_frac<0.5: {len(both)}")
print()
print("-- All flagged, sorted by dist --")
for r in sorted(flagged, key=lambda r: -r["dist_to_rubric"]):
    print(r["scene"], r["q"], r["leg"], "prev=", r["prev_leg_kind"], "seen_frac=", r["seen_frac"],
          "dist=", r["dist_to_rubric"])
