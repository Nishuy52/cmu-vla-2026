"""Issue #77e: per-leg mechanism tracer for the top-6 cheapest-excess legs in
the current fixable arrival-blocked pool. For each (scene, qidx, leg_idx):
  - resolved head goal vs rubric goal (goal-resolution correctness)
  - driven trajectory tail near the goal (execution/controller behavior)
  - predecessor leg kind/outcome (dwell interplay)
  - nearest_reachable_point / reachable-mask diagnostics (route/costmap shape)
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

    print(f"\n{'='*90}\n{scene} q{qidx} leg{leg_idx} :: {text}\n{'='*90}")
    print(f"spawn_xy={spawn_xy} frame_residual={residual}")

    head, io, plan = GB._run_instruction_head(
        text, gt, idx, start_xy=spawn_xy, max_build_ticks=GB._IF_MAX_BUILD_TICKS,
        wall_cells=wall_cells,
    )
    if head is None or head._legs is None or leg_idx >= len(head._legs):
        print(f"could not build head/legs (head={head})")
        return

    leg_goals, corridor_gates, avoid_caps, leg_instance_ids, leg_instance_aabbs = (
        GB._if_rubric_geometry(text, gt, idx, start_xy=spawn_xy)
    )
    rubric_kind, rubric_xy = leg_goals[leg_idx]

    for li in range(len(head._legs)):
        leg = head._legs[li]
        marker = " <== TARGET" if li == leg_idx else ""
        print(f"  leg{li}: kind={leg.kind.name if hasattr(leg.kind,'name') else leg.kind} "
              f"geom={leg.geom}{marker}")

    leg = head._legs[leg_idx]
    head_goal = leg.geom
    dist_head_to_rubric = math.hypot(head_goal[0] - rubric_xy[0], head_goal[1] - rubric_xy[1]) if head_goal else None
    print(f"\nhead resolved goal: {head_goal}")
    print(f"rubric goal:        {rubric_xy}  (kind={rubric_kind})")
    print(f"dist(head_goal, rubric_goal) = {dist_head_to_rubric}")

    rec = leg.record
    c = TB.P._as3(rec.centroid)
    anchor_xy = (float(c[0]), float(c[1]))
    inst_ids = getattr(rec, "instance_ids", None) or getattr(rec, "matched_id", None) or getattr(rec, "instance_id", None)
    print(f"candidate anchor: label={getattr(rec,'label',None)} inst={getattr(rec,'instance_id',None)} centroid={anchor_xy}")

    cm = head._costmap
    pose = head._pose
    seen = cm.reachable_mask(pose) if cm is not None else None
    plain_nrp = cm.nearest_reachable_point(head_goal, pose) if cm is not None and head_goal is not None else None
    print(f"nearest_reachable_point(head_goal) = {plain_nrp}")
    if plain_nrp is not None:
        d = math.hypot(plain_nrp[0]-head_goal[0], plain_nrp[1]-head_goal[1])
        print(f"  dist(nrp, head_goal) = {d:.4f}  (0 => goal already reachable per costmap)")

    # Drive the actual controller and inspect the trajectory tail near this leg's goal.
    driven = GB._drive_if_trajectory(text, gt, idx, start_xy=spawn_xy, wall_cells=wall_cells)
    dist_driven = GB._min_dist_driven_to_goal(driven, head_goal)
    print(f"\nmin_dist_driven_to_goal (to HEAD goal) = {dist_driven:.4f}")
    dist_driven_rubric = GB._min_dist_driven_to_goal(driven, rubric_xy)
    print(f"min_dist_driven_to_goal (to RUBRIC goal) = {dist_driven_rubric:.4f}")

    # driven trajectory closest-approach index + tail
    xs = np.asarray([p[0] for p in driven])
    ys = np.asarray([p[1] for p in driven])
    dists = np.hypot(xs - head_goal[0], ys - head_goal[1])
    closest_i = int(np.argmin(dists))
    print(f"driven n_poses={len(driven)}  closest_approach_idx={closest_i}/{len(driven)-1} "
          f"closest_dist={dists[closest_i]:.4f}")
    print(f"  closest pose: {driven[closest_i][:2]}")
    print(f"  last pose:    {driven[-1][:2]}  dist_to_goal={dists[-1]:.4f}")
    lo = max(0, closest_i - 2)
    hi = min(len(driven), closest_i + 3)
    print(f"  poses around closest approach [{lo}:{hi}]:")
    for k in range(lo, hi):
        print(f"    [{k}] {driven[k][:2]}  d={dists[k]:.4f}")


if __name__ == "__main__":
    targets = [
        ("studio", 0, 1),          # excess 0.0221
        ("arabic_room", 1, 2),     # excess 0.0276
        ("office_2", 0, 1),        # excess 0.1838
        ("livingroom_4", 1, 0),    # excess 0.2150
        ("chinese_room", 0, 0),    # excess 0.8338
        ("home_building_2", 0, 1),  # excess 1.6087
    ]
    for scene, qidx, leg_idx in targets:
        try:
            trace(scene, qidx, leg_idx)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"{scene} q{qidx} leg{leg_idx}: ERROR {e}")
