"""Issue #79 (coordinator-flagged #78 spillover): probe whether relaxing
_goto_point's reachability BFS through the previous corridor leg's gate (via
core.nav.planner._pinch_costmap, same relax-round schedule as plan_through's
own #54 mechanism) lands hotel_room_2 leg1's goal near the rubric target."""
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

ROOT = Path("../../data/vla3d/Unity")
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

head, io, plan = GB._run_instruction_head(
    text, gt, idx, start_xy=spawn_xy, max_build_ticks=GB._IF_MAX_BUILD_TICKS,
    wall_cells=wall_cells,
)

for i, leg in enumerate(head._legs):
    print(f"leg[{i}] kind={leg.kind} grounded={leg.grounded} geom={leg.geom}")

leg0 = head._legs[0]
leg1 = head._legs[1]
gate = leg0.geom
rec = leg1.record
c = TB.P._as3(rec.centroid)
anchor_xy = (float(c[0]), float(c[1]))
print("gate:", gate)
print("anchor_xy:", anchor_xy)

leg_goals, *_rest = GB._if_rubric_geometry(text, gt, idx, start_xy=spawn_xy)
print("rubric leg_goals:", leg_goals)
rubric_goal1 = leg_goals[1][1]

cm = head._costmap
pose = head._pose
print("pose:", pose)

seen = cm.reachable_mask(pose)
r, col = head.grid.world_to_cell(*anchor_xy)
print("plain reachable? ", bool(seen[r, col]) if seen is not None else None,
      "pocket size:", int(seen.sum()) if seen is not None else None)
plain_nrp = cm.nearest_reachable_point(anchor_xy, pose)
print("plain nearest_reachable_point:", plain_nrp,
      "dist to rubric:", np.hypot(plain_nrp[0]-rubric_goal1[0], plain_nrp[1]-rubric_goal1[1]))

relax_disc_m = P.PINCH_DISC_M
for round_i in range(P.MAX_PINCH_RELAX_ROUNDS):
    pinch = P._pinch_costmap(
        cm, gate,
        pinch_disc_m=relax_disc_m,
        pinch_corridor_half_w_m=P.PINCH_CORRIDOR_HALF_W_M,
        start_xy=pose,
    )
    pseen = pinch.reachable_mask(pose)
    hit = bool(pseen[r, col]) if pseen is not None else None
    nrp = pinch.nearest_reachable_point(anchor_xy, pose)
    dist = np.hypot(nrp[0]-rubric_goal1[0], nrp[1]-rubric_goal1[1])
    print(f"round {round_i} disc={relax_disc_m:.3f} pocket={int(pseen.sum()) if pseen is not None else None} "
          f"anchor_reachable={hit} nearest={nrp} dist_to_rubric={dist:.4f}")
    relax_disc_m *= P.PINCH_RELAX_GROWTH
