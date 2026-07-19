"""Issue #77e: deeper mechanism check for the top-6 legs -- is the head's
resolved goal short because the TRUE anchor cell is unreachable (BFS
disconnection from the grounding-time pose) or because the WRONG anchor was
picked (candidate resolution to a different instance than the rubric's)?
"""
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np

from core.groundtruth.loader import load_scene
import core.runner.gt_battery as GB
from core.perception.scene_index import BasicSceneIndex
from core.geometry import toolbox as TB

ROOT = Path("../data/vla3d/Unity")


def trace(scene, qidx, leg_idx):
    with open(GB.DEFAULT_QUESTIONS, encoding="utf-8") as fh:
        data = json.load(fh)
    entry = next(e for e in data if e["scene"] == scene)
    folder = GB._find_scene_folder(ROOT, scene)
    gt = load_scene(folder, scene_name=scene)
    idx = BasicSceneIndex(gt.instances)

    if_texts = entry["questions"]["instruction_following"]
    text = if_texts[qidx]

    if_traj, if_cands, frame, residual, pairs = GB._fit_scene_if_frame(
        gt, idx, if_texts, GB.DEFAULT_QUESTIONS_ROOT
    )
    spawn_xy = None
    if frame is not None and pairs and qidx < len(pairs):
        start_pt = pairs[qidx][0][0, :2]
        mapped = frame.apply(np.asarray([start_pt], dtype=float))[0]
        spawn_xy = (float(mapped[0]), float(mapped[1]))

    frame_for_walls = frame if residual is None or residual <= GB.WALL_FIT_MAX_RESIDUAL_M else None
    wall_cells = GB._scene_wall_cells(
        gt, frame_for_walls, unity_scenes_ros2_root=GB.DEFAULT_UNITY_SCENES_ROS2_ROOT
    )

    head, io, plan = GB._run_instruction_head(
        text, gt, idx, start_xy=spawn_xy, max_build_ticks=GB._IF_MAX_BUILD_TICKS,
        wall_cells=wall_cells,
    )
    leg_goals, *_rest = GB._if_rubric_geometry(text, gt, idx, start_xy=spawn_xy)
    rubric_kind, rubric_xy = leg_goals[leg_idx]

    leg = head._legs[leg_idx]
    rec = leg.record
    c = TB.P._as3(rec.centroid)
    anchor_xy = (float(c[0]), float(c[1]))
    head_goal = leg.geom

    print(f"\n{'='*90}\n{scene} q{qidx} leg{leg_idx} :: {text[:80]}\n{'='*90}")
    print(f"spawn_xy={spawn_xy}")
    print(f"picked candidate: label={getattr(rec,'label',None)!r} inst={getattr(rec,'instance_id',None)} centroid={anchor_xy}")
    print(f"head resolved goal (post nearest_reachable_point) = {head_goal}")
    print(f"rubric goal = {rubric_xy}  dist(anchor,rubric) = {math.hypot(anchor_xy[0]-rubric_xy[0], anchor_xy[1]-rubric_xy[1]):.4f}")
    print(f"dist(head_goal, anchor_centroid) = {math.hypot(head_goal[0]-anchor_xy[0], head_goal[1]-anchor_xy[1]):.4f}")

    cm = head._costmap
    pose = head._pose
    print(f"grounding pose (self._pose) used for reachable_mask = {pose}")
    seen = cm.reachable_mask(pose)
    r, col = head.grid.world_to_cell(*anchor_xy)
    anchor_reachable = bool(seen[r, col]) if (0 <= r < seen.shape[0] and 0 <= col < seen.shape[1]) else None
    print(f"anchor cell reachable from grounding pose (full BFS)? {anchor_reachable}")
    print(f"anchor cell blocked? base={cm.blocked(r,col) if hasattr(cm,'blocked') else 'n/a'}")

    # Check: is the anchor picked (rec) actually the SAME instance the rubric intends?
    # Compare against every candidate within 0.5m of the rubric goal.
    all_labels_near_rubric = []
    for inst in gt.instances:
        ic = TB.P._as3(inst.centroid)
        d = math.hypot(float(ic[0]) - rubric_xy[0], float(ic[1]) - rubric_xy[1])
        if d < 1.0:
            all_labels_near_rubric.append((round(d, 3), inst.instance_id, inst.label))
    all_labels_near_rubric.sort()
    print(f"instances within 1.0m of rubric goal: {all_labels_near_rubric[:5]}")


if __name__ == "__main__":
    targets = [
        ("studio", 0, 1),
        ("arabic_room", 1, 2),
        ("office_2", 0, 1),
        ("livingroom_4", 1, 0),
        ("chinese_room", 0, 0),
        ("home_building_2", 0, 1),
    ]
    for scene, qidx, leg_idx in targets:
        try:
            trace(scene, qidx, leg_idx)
        except Exception as e:
            import traceback
            traceback.print_exc()
