"""Issue #78 repro: trace hotel_room_2 leg1's goto goal resolution."""
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
print("spawn_xy:", spawn_xy)

frame_for_walls = frame if residual is None or residual <= GB.WALL_FIT_MAX_RESIDUAL_M else None
wall_cells = GB._scene_wall_cells(
    gt, frame_for_walls, unity_scenes_ros2_root=GB.DEFAULT_UNITY_SCENES_ROS2_ROOT
)

head, io, plan = GB._run_instruction_head(
    text, gt, idx, start_xy=spawn_xy, max_build_ticks=GB._IF_MAX_BUILD_TICKS,
    wall_cells=wall_cells,
)

print("\n--- legs ---")
for i, leg in enumerate(head._legs):
    print(f"leg[{i}] kind={leg.kind} grounded={leg.grounded} geom={leg.geom} "
          f"record_id={getattr(leg.record, 'instance_id', None)} "
          f"record_centroid={getattr(leg.record, 'centroid', None)}")

print("\nfollower.path:")
for i, p in enumerate(head._follower.path):
    print(f"  path[{i}] = ({p[0]:.4f}, {p[1]:.4f})")
print("leg_goal_indices:", head._follower.leg_goal_indices)

print("\n--- rubric geometry ---")
rubric = GB._if_rubric_geometry(text, gt, idx, start_xy=spawn_xy)
print("leg_goals:", rubric.leg_goals if hasattr(rubric, "leg_goals") else rubric)

# leg1 investigation: replicate _goto_point manually
leg1 = head._legs[1]
rec = leg1.record
print("\nleg1 record centroid (3d):", rec.centroid, "instance_id:", rec.instance_id)
from core.geometry import toolbox as TB
c = TB.P._as3(rec.centroid)
anchor_xy = (float(c[0]), float(c[1]))
print("anchor_xy (2d projection of centroid):", anchor_xy)

cm = head._costmap
pose = head._pose
print("head._pose (used for reachability):", pose)
r, col = head.grid.world_to_cell(*anchor_xy)
print("anchor cell (r,c):", r, col, "grid shape:", head.grid.shape)
seen = cm.reachable_mask(pose)
print("seen is None:", seen is None)
if seen is not None:
    in_bounds = 0 <= r < seen.shape[0] and 0 <= col < seen.shape[1]
    print("anchor cell in_bounds:", in_bounds, "seen[r,c]:", seen[r, col] if in_bounds else None)
    print("cm.blocked(r,c):", cm.blocked(r, col) if in_bounds else None)

nrp = cm.nearest_reachable_point(anchor_xy, pose)
print("nearest_reachable_point ->", nrp)
