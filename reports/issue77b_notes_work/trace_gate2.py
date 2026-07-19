"""Deeper trace (issue #77b): re-run _drive_if_trajectory's own loop body with
instrumentation, to see exactly why the drive halts after ~8 poses for
hotel_room_2 leg0 (dwell-class corridor leg)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np

from core.groundtruth.loader import load_scene
import core.runner.gt_battery as GB
from core.perception.scene_index import BasicSceneIndex

ROOT = Path("../data/vla3d/Unity")
SCENE = sys.argv[1] if len(sys.argv) > 1 else "hotel_room_2"
QIDX = int(sys.argv[2]) if len(sys.argv) > 2 else 0

with open(GB.DEFAULT_QUESTIONS, encoding="utf-8") as fh:
    data = json.load(fh)
entry = next(e for e in data if e["scene"] == SCENE)
folder = GB._find_scene_folder(ROOT, SCENE)
gt = load_scene(folder, scene_name=SCENE)
idx = BasicSceneIndex(gt.instances)

if_texts = entry["questions"]["instruction_following"]
text = if_texts[QIDX]
print("QUESTION:", text)

if_traj, if_cands, frame, residual, pairs = GB._fit_scene_if_frame(
    gt, idx, if_texts, GB.DEFAULT_QUESTIONS_ROOT
)
spawn_xy = None
if frame is not None and pairs:
    start_pt = pairs[0][0][0, :2]
    mapped = frame.apply(np.asarray([start_pt], dtype=float))[0]
    spawn_xy = (float(mapped[0]), float(mapped[1]))

frame_for_walls = frame if residual is None or residual <= GB.WALL_FIT_MAX_RESIDUAL_M else None
wall_cells = GB._scene_wall_cells(
    gt, frame_for_walls, unity_scenes_ros2_root=GB.DEFAULT_UNITY_SCENES_ROS2_ROOT
)

# ---- inline copy of _drive_if_trajectory's body, instrumented ----
head, io, plan = GB._run_instruction_head(
    text, gt, idx, start_xy=spawn_xy, max_build_ticks=GB._IF_MAX_BUILD_TICKS,
    wall_cells=wall_cells,
)
follower = head._follower
print("initial follower.path len:", len(follower.path))
print("initial leg_goal_indices:", follower.leg_goal_indices)
print("n_legs (head._legs):", len(head._legs) if head._legs else None)
print("head._driven_prefix:", head._driven_prefix)

follower._idx = 0
follower._hist = []
follower._last_crumb_idx = -1

odom = io.latest_odom()
pose = (float(odom.x), float(odom.y)) if odom is not None else (0.0, 0.0)
t = 0.0
poses = [pose]
last_term = (float(follower.path[-1][0]), float(follower.path[-1][1]))
n_legs = len(head._legs) if head._legs else len(plan.route)
head_reticks_left = GB._DRIVE_HEAD_RETICK_BUDGET
stall_ref = pose
stall_ticks = 0

for tick in range(GB._DRIVE_MAX_TICKS):
    route_growing = head._driven_prefix < n_legs and head_reticks_left > 0
    if route_growing:
        head_reticks_left -= 1
        io.set_pose(pose[0], pose[1])
        head.advance(io, idx)
        new_follower = head._follower
        if new_follower is not None and new_follower.path:
            if new_follower is not follower:
                print(f"  [tick {tick}] REPLAN: path len {len(follower.path)} -> {len(new_follower.path)}, "
                      f"leg_goal_indices {follower.leg_goal_indices} -> {new_follower.leg_goal_indices}")
                follower = new_follower
                follower._idx = GB._nearest_forward_idx(follower.path, pose)
                follower._hist = []
                follower._last_crumb_idx = -1
            last_term = (float(follower.path[-1][0]), float(follower.path[-1][1]))

    wp = follower.advance(pose, t)
    ceiling = follower._leg_ceiling()
    print(f"  [tick {tick}] pose=({pose[0]:.4f},{pose[1]:.4f}) idx={follower._idx} "
          f"ceiling={ceiling} next_leg_goal_ptr={follower._next_leg_goal_ptr} "
          f"driven_prefix={head._driven_prefix} wp={None if wp is None else (round(wp.x,4), round(wp.y,4))} "
          f"replan_flag={follower.replan_flag}")
    if wp is None:
        print(f"  [tick {tick}] BREAK: wp is None")
        break
    target = (float(wp.x), float(wp.y))
    dx, dy = target[0] - pose[0], target[1] - pose[1]
    d = (dx * dx + dy * dy) ** 0.5
    if d <= GB._DRIVE_STEP_M:
        pose = target
    else:
        pose = (pose[0] + GB._DRIVE_STEP_M * dx / d, pose[1] + GB._DRIVE_STEP_M * dy / d)
    poses.append(pose)
    t += 1.0

    if ((pose[0] - stall_ref[0]) ** 2 + (pose[1] - stall_ref[1]) ** 2) > (GB._DRIVE_STALL_EPS_M**2):
        stall_ref = pose
        stall_ticks = 0
    else:
        stall_ticks += 1
        if stall_ticks >= GB._DRIVE_STALL_TICKS:
            print(f"  [tick {tick}] BREAK: stall ({stall_ticks} ticks)")
            break

print("\nfinal path (last follower):", follower.path)
print("leg_goal_indices:", follower.leg_goal_indices)
print("last_term:", last_term)
print("poses so far (len=%d):" % len(poses), poses)

if not poses or (poses[-1][0] - last_term[0]) ** 2 + (poses[-1][1] - last_term[1]) ** 2 > (GB._DRIVE_STEP_M**2):
    print("APPEND last_term fixup ->", last_term)
    poses.append((float(last_term[0]), float(last_term[1])))

print("\nFINAL DRIVEN POSES:")
for i, p in enumerate(poses):
    print(f"  [{i}] ({p[0]:.4f}, {p[1]:.4f})")
