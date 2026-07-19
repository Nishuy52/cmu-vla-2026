"""Issue #77d Family B: experimental probe for home_building_2 q1 (the corridor
leg0 / goto leg1 trade-off). Checks whether ANY point along the gate segment
(not just the small GATE_CROSSING_MARGINS_M push-past-the-line nudge) permits
BOTH (a) the corridor leg0's own crossing to register as threaded, and (b) an
A* route onward from that crossing point to leg1's narrower/truer goto goal.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np

from core.groundtruth.loader import load_scene
import core.runner.gt_battery as GB
from core.perception.scene_index import BasicSceneIndex
from core.nav import planner as P
from core.geometry import toolbox as TB

ROOT = Path("../data/vla3d/Unity")

scene, qidx = "home_building_2", 1
with open(GB.DEFAULT_QUESTIONS, encoding="utf-8") as fh:
    data = json.load(fh)
entry = next(e for e in data if e["scene"] == scene)
folder = GB._find_scene_folder(ROOT, scene)
gt = load_scene(folder, scene_name=scene)
idx = BasicSceneIndex(gt.instances)
if_texts = entry["questions"]["instruction_following"]
text = if_texts[qidx]

if_traj, if_cands, frame, residual, pairs = GB._fit_scene_if_frame(gt, idx, if_texts, GB.DEFAULT_QUESTIONS_ROOT)
spawn_xy = None
if frame is not None and pairs and qidx < len(pairs):
    start_pt = pairs[qidx][0][0, :2]
    mapped = frame.apply(np.asarray([start_pt], dtype=float))[0]
    spawn_xy = (float(mapped[0]), float(mapped[1]))

frame_for_walls = frame if residual is None or residual <= GB.WALL_FIT_MAX_RESIDUAL_M else None
wall_cells = GB._scene_wall_cells(gt, frame_for_walls, unity_scenes_ros2_root=GB.DEFAULT_UNITY_SCENES_ROS2_ROOT)

head, io, plan = GB._run_instruction_head(
    text, gt, idx, start_xy=spawn_xy, max_build_ticks=GB._IF_MAX_BUILD_TICKS,
    wall_cells=wall_cells,
)
leg0 = head._legs[0]
leg1 = head._legs[1]
gate = leg0.geom
print("gate:", gate)
print("leg1 (wide) goal:", leg1.geom)

cm = head._costmap
pose = head._pose

# Recompute the pinch-relaxed NARROW candidate for leg1, same as _goto_point_pinch_relax
# but WITHOUT the corridor-threading guard, to get the "truer" goal.
rec1 = leg1.record
c = TB.P._as3(rec1.centroid)
anchor_xy = (float(c[0]), float(c[1]))
plain_best = cm.nearest_reachable_point(anchor_xy, pose)
best = plain_best
best_d = ((best[0]-anchor_xy[0])**2 + (best[1]-anchor_xy[1])**2) ** 0.5
relax_disc_m = P.PINCH_DISC_M
for _round in range(P.MAX_PINCH_RELAX_ROUNDS):
    pinch = P._pinch_costmap(cm, gate, pinch_disc_m=relax_disc_m, pinch_corridor_half_w_m=P.PINCH_CORRIDOR_HALF_W_M, start_xy=pose)
    cand = pinch.nearest_reachable_point(anchor_xy, pose)
    d = ((cand[0]-anchor_xy[0])**2 + (cand[1]-anchor_xy[1])**2) ** 0.5
    if d < best_d:
        best, best_d = cand, d
    relax_disc_m *= P.PINCH_RELAX_GROWTH
narrow_goal = best
print("leg1 narrow (pinch-relaxed) goal:", narrow_goal, "d_to_anchor=", best_d)

# Now sweep crossing points ALONG the gate segment (not just the small
# push-past margin) and check: (1) reachable from `cur` (spawn/leg0 start),
# (2) path_crosses_gate for that reach, (3) reachable onward to narrow_goal.
g0, g1 = np.asarray(gate[0]), np.asarray(gate[1])
cur = pose  # leg0 starts from spawn/current pose in this probe
n_samples = 21
print(f"\nsweeping {n_samples} points along the gate segment:")
for t in np.linspace(0.02, 0.98, n_samples):
    pt = tuple((g0 + t * (g1 - g0)).tolist())
    if P._raw_obstacle_blocked_xy(cm, pt):
        print(f"  t={t:.2f} pt={pt} BLOCKED (raw obstacle)")
        continue
    seg_to_gate = P.astar(cm, cur, pt, unknown_cost_mult=head.unknown_cost_mult)
    reach0 = seg_to_gate is not None
    onward = P.astar(cm, pt, narrow_goal, unknown_cost_mult=head.unknown_cost_mult)
    reach1 = onward is not None
    print(f"  t={t:.2f} pt=({pt[0]:.2f},{pt[1]:.2f}) reach_from_start={reach0} reach_to_narrow_goal={reach1}")
