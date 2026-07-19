"""Issue #77d Family A: probe leg0 goal resolution for the 3 never-approached
threading roots (arabic_room, home_building_1, livingroom_1), each rooted in
a leg0 upstream goal-resolution miss. No predecessor gate exists for leg0 (it
is the first leg), so this checks PLAIN reachable_mask/nearest_reachable_point
behavior (no pinch-relax through a gate applies -- there is no gate).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np

from core.groundtruth.loader import load_scene
import core.runner.gt_battery as GB
from core.perception.scene_index import BasicSceneIndex
from core.geometry import toolbox as TB

ROOT = Path("../data/vla3d/Unity")


def probe(scene, qidx, leg_idx=0):
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
    print(f"=== {scene} q{qidx} leg{leg_idx} :: {text[:70]} ===")
    print(f"  spawn_xy={spawn_xy} frame_residual={residual}")
    if head is None or head._legs is None or leg_idx >= len(head._legs):
        print(f"  could not build head/legs (head={head})")
        return

    leg = head._legs[leg_idx]
    rec = leg.record
    c = TB.P._as3(rec.centroid)
    anchor_xy = (float(c[0]), float(c[1]))
    inst_ids = getattr(rec, "instance_ids", None) or getattr(rec, "matched_ids", None)
    print(f"  candidate raw_label={getattr(rec, 'label', None)} inst={getattr(rec, 'instance_id', None)} centroid={anchor_xy}")
    print(f"  leg.kind={leg.kind} leg.geom(resolved goal)={leg.geom}")

    leg_goals, *_rest = GB._if_rubric_geometry(text, gt, idx, start_xy=spawn_xy)
    rubric_xy = leg_goals[leg_idx][1] if leg_idx < len(leg_goals) else None
    print(f"  rubric_xy={rubric_xy}")
    if rubric_xy is not None:
        dist_anchor_to_rubric = float(np.hypot(anchor_xy[0] - rubric_xy[0], anchor_xy[1] - rubric_xy[1]))
        dist_goal_to_rubric = float(np.hypot(leg.geom[0] - rubric_xy[0], leg.geom[1] - rubric_xy[1]))
        print(f"  dist(anchor,rubric)={dist_anchor_to_rubric:.4f}  dist(resolved_goal,rubric)={dist_goal_to_rubric:.4f}")

    cm = head._costmap
    pose = head._pose
    seen = cm.reachable_mask(pose)
    r, col = head.grid.world_to_cell(*anchor_xy)
    plain_hit = bool(seen[r, col]) if seen is not None else None
    plain_nrp = cm.nearest_reachable_point(anchor_xy, pose)
    print(f"  pose(spawn used by planner)={pose}")
    print(f"  anchor cell reachable(plain BFS from spawn)={plain_hit} pocket_size={int(seen.sum()) if seen is not None else None}")
    print(f"  nearest_reachable_point(anchor)={plain_nrp}")
    if rubric_xy is not None:
        dist_nrp_to_rubric = float(np.hypot(plain_nrp[0] - rubric_xy[0], plain_nrp[1] - rubric_xy[1]))
        print(f"  dist(nearest_reachable_point, rubric)={dist_nrp_to_rubric:.4f}")
    print()


if __name__ == "__main__":
    targets = [
        ("arabic_room", 1, 0),
        ("home_building_1", 1, 0),
        ("livingroom_1", 1, 0),
    ]
    for scene, qidx, leg_idx in targets:
        try:
            probe(scene, qidx, leg_idx)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"{scene} q{qidx} leg{leg_idx}: ERROR {e}")
            print()
