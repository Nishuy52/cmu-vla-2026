"""Issue #77 (residual v2, post-carve): per-leg mechanism tracer for the
still-failing legs in reports/gt_battery_main_post_carve, adapted from
reports/issue77e_notes_work/trace_leg.py for the post-carve costmap
(_synthetic_from_gt now needs carved_cells threaded through, same as
gt_battery's own battery driver)."""
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

    # Post-carve: carve GT-trajectory corridor cells the same way the battery does.
    carved_cells = None
    carve_fn = getattr(GB, "_carve_cells_along_trajectories", None)
    if carve_fn is not None and if_traj:
        try:
            carved_cells = carve_fn(if_traj, frame)
        except Exception as e:
            print(f"  (carve_fn failed: {e})")

    print(f"\n{'='*90}\n{scene} q{qidx} leg{leg_idx} :: {text}\n{'='*90}")
    print(f"spawn_xy={spawn_xy} frame_residual={residual} carved_cells={len(carved_cells) if carved_cells else 0}")

    head, io, plan = GB._run_instruction_head(
        text, gt, idx, start_xy=spawn_xy, max_build_ticks=GB._IF_MAX_BUILD_TICKS,
        wall_cells=wall_cells, carved_cells=carved_cells,
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
    print(f"candidate anchor: label={getattr(rec,'label',None)} inst={getattr(rec,'instance_id',None)} centroid={anchor_xy}")

    cm = head._costmap
    pose = head._pose
    plain_nrp = cm.nearest_reachable_point(head_goal, pose) if cm is not None and head_goal is not None else None
    print(f"nearest_reachable_point(head_goal) = {plain_nrp}")
    if plain_nrp is not None:
        d = math.hypot(plain_nrp[0]-head_goal[0], plain_nrp[1]-head_goal[1])
        print(f"  dist(nrp, head_goal) = {d:.4f}  (0 => goal already reachable per costmap)")

    driven = GB._drive_if_trajectory(
        text, gt, idx, start_xy=spawn_xy, wall_cells=wall_cells, carved_cells=carved_cells,
    )
    dist_driven = GB._min_dist_driven_to_goal(driven, head_goal)
    print(f"\nmin_dist_driven_to_goal (to HEAD goal) = {dist_driven:.4f}")
    dist_driven_rubric = GB._min_dist_driven_to_goal(driven, rubric_xy)
    print(f"min_dist_driven_to_goal (to RUBRIC goal) = {dist_driven_rubric:.4f}")

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
        ("arabic_room", 0, 0),      # stool leg, rubric=0.5
        ("arabic_room", 1, 0),      # potted plant leg, rubric=0.0 (corridor-threading Q)
        ("chinese_room", 1, 1),     # tea table leg, rubric=0.5
        ("livingroom_2", 1, 1),     # chair/window leg, rubric=0.5
        ("loft", 0, 0),             # cup/avoid leg, rubric=0.0
    ]
    for scene, qidx, leg_idx in targets:
        try:
            trace(scene, qidx, leg_idx)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"{scene} q{qidx} leg{leg_idx}: ERROR {e}")


def deep_dive_arabic():
    import json as _json
    with open(GB.DEFAULT_QUESTIONS, encoding="utf-8") as fh:
        data = _json.load(fh)
    entry = next(e for e in data if e["scene"] == "arabic_room")
    folder = GB._find_scene_folder(ROOT, "arabic_room")
    gt = load_scene(folder, scene_name="arabic_room")
    idx = BasicSceneIndex(gt.instances)
    if_texts = entry["questions"]["instruction_following"]
    if_traj, if_cands, frame, residual, pairs = GB._fit_scene_if_frame(
        gt, idx, if_texts, GB.DEFAULT_QUESTIONS_ROOT
    )
    text = if_texts[0]
    spawn_xy = None
    if frame is not None and pairs:
        start_pt = pairs[0][0][0, :2]
        mapped = frame.apply(np.asarray([start_pt], dtype=float))[0]
        spawn_xy = (float(mapped[0]), float(mapped[1]))
    frame_for_walls = frame if residual is None or residual <= GB.WALL_FIT_MAX_RESIDUAL_M else None
    wall_cells = GB._scene_wall_cells(gt, frame_for_walls, unity_scenes_ros2_root=GB.DEFAULT_UNITY_SCENES_ROS2_ROOT)
    carve_fn = getattr(GB, "_carve_cells_along_trajectories", None)
    carved_cells = carve_fn(if_traj, frame) if carve_fn and if_traj else None
    print("carved_cells:", len(carved_cells) if carved_cells else 0)

    head, io, plan = GB._run_instruction_head(
        text, gt, idx, start_xy=spawn_xy, max_build_ticks=GB._IF_MAX_BUILD_TICKS,
        wall_cells=wall_cells, carved_cells=carved_cells,
    )
    cm = head._costmap
    pose = head._pose
    print("cell_m:", cm.cell_m, "spawn:", pose)
    # cell coords for stool centroid vs head_goal vs spawn
    for label, xy in [("spawn", pose), ("stool_centroid", (-3.739999691864865, -1.6720002847253355)),
                      ("head_goal", (-6.250000000000001, -1.6500000000000004)),
                      ("rubric_goal", (-3.1393839424582937, -1.9811260129585473))]:
        cx = round(xy[0] / cm.cell_m)
        cy = round(xy[1] / cm.cell_m)
        in_carved = (cx, cy) in carved_cells if carved_cells else None
        print(f"  {label}: xy={xy} cell=({cx},{cy}) in_carved={in_carved}")

    mask = cm.reachable_mask(pose)
    print("reachable_mask shape:", None if mask is None else mask.shape)
    # is the GT if_traj (q0) actually near the stool at all?
    if if_traj and len(if_traj) > 0:
        traj0 = if_traj[0]
        mapped_traj = frame.apply(traj0[:, :2]) if frame is not None else traj0[:, :2]
        d = np.hypot(mapped_traj[:, 0] - (-3.74), mapped_traj[:, 1] - (-1.67))
        print("GT traj min dist to stool centroid:", float(d.min()))
        print("GT traj bbox x:", mapped_traj[:,0].min(), mapped_traj[:,0].max(), "y:", mapped_traj[:,1].min(), mapped_traj[:,1].max())

deep_dive_arabic()
