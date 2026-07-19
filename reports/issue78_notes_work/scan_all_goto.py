"""Issue #78 blast-radius scan: for every IF question's goto legs, compare the
head's resolved geom (leg.geom) against the rubric's independently-computed
leg goal, and flag legs whose goal is unreachable from spawn under the plain
(non-pinch) costmap -- the suspected #78 mechanism -- plus large positional
discrepancies vs the rubric goal."""
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np

from core.groundtruth.loader import load_scene
import core.runner.gt_battery as GB
from core.perception.scene_index import BasicSceneIndex

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
        except Exception as e:
            results.append({"scene": scene_name, "q": qi, "text": text, "error": str(e)})
            continue
        if head is None or head._legs is None:
            continue
        try:
            rubric = GB._if_rubric_geometry(text, gt, idx, start_xy=spawn_xy)
            leg_goals = rubric[0]
        except Exception as e:
            leg_goals = None

        cm = head._costmap
        pose = head._pose
        for li, leg in enumerate(head._legs):
            if leg.kind.name != "GOTO" or leg.geom is None:
                continue
            gx, gy = leg.geom
            rub_xy = None
            if leg_goals is not None and li < len(leg_goals):
                kind, xy = leg_goals[li]
                if kind == "goto":
                    rub_xy = xy
            dist_to_rubric = None
            if rub_xy is not None:
                dist_to_rubric = math.hypot(gx - rub_xy[0], gy - rub_xy[1])

            # Check: is the anchor's own (unresolved) cell reachable in the plain
            # (non-pinch) costmap from the grounding pose? (the suspected #78 mechanism)
            rec = leg.record
            anchor_unreachable = None
            if rec is not None and cm is not None:
                from core.geometry import toolbox as TB
                c = TB.P._as3(rec.centroid)
                axy = (float(c[0]), float(c[1]))
                r, col = head.grid.world_to_cell(*axy)
                seen = cm.reachable_mask(pose)
                if seen is not None and 0 <= r < seen.shape[0] and 0 <= col < seen.shape[1]:
                    anchor_unreachable = not bool(seen[r, col])

            results.append({
                "scene": scene_name, "q": qi, "leg": li, "text": text,
                "head_goal": [round(gx, 4), round(gy, 4)],
                "rubric_goal": [round(rub_xy[0], 4), round(rub_xy[1], 4)] if rub_xy else None,
                "dist_to_rubric": round(dist_to_rubric, 4) if dist_to_rubric is not None else None,
                "anchor_cell_unreachable_plain_costmap": anchor_unreachable,
            })

with open(Path(__file__).parent / "goto_scan.json", "w") as fh:
    json.dump(results, fh, indent=2)

print(f"Total goto legs scanned: {len(results)}")
flagged = [r for r in results if r.get("dist_to_rubric") is not None and r["dist_to_rubric"] > 1.0]
print(f"Flagged (>1.0m from rubric goal): {len(flagged)}")
for r in flagged:
    print(r)
print()
unreach = [r for r in results if r.get("anchor_cell_unreachable_plain_costmap")]
print(f"Anchor cell unreachable under plain costmap: {len(unreach)}")
for r in unreach:
    print(r["scene"], r["q"], r["leg"], r["text"][:60], r["head_goal"], r["rubric_goal"], r["dist_to_rubric"])
