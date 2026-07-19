import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np

from core.groundtruth.loader import load_scene
import core.runner.gt_battery as GB
from core.perception.scene_index import BasicSceneIndex
from core.nav import planner as P

ROOT = Path("../data/vla3d/Unity")
SCENE = "home_building_2"
QIDX = 1

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
if frame is not None and pairs and QIDX < len(pairs):
    start_pt = pairs[QIDX][0][0, :2]
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

gate = head._legs[0].geom
leg1_goal = head._legs[1].geom
leg2_goal = head._legs[2].geom
cm = head._costmap
print("gate:", gate)
print("leg1 goal (resolved, used for planning):", leg1_goal)

# reconstruct the corridor's own gate-crossing extension candidate
g0, g1 = gate
seg_end = P.usable_gate_point(np.asarray(g0), np.asarray(g1), np.asarray(((g0[0]+g1[0])/2, (g0[1]+g1[1])/2)), lambda pt: P._raw_obstacle_blocked_xy(cm, pt))
print("usable_gate_point (seg_end):", seg_end)

for margin in P.GATE_CROSSING_MARGINS_M:
    dx, dy = g1[0]-g0[0], g1[1]-g0[1]
    dlen = (dx**2+dy**2)**0.5
    nx, ny = -dy/dlen, dx/dlen
    # orientation: leg_start is head._pose (approx) -- use spawn_xy as leg_start proxy
    leg_start = head._pose
    if (seg_end[0]-leg_start[0])*nx + (seg_end[1]-leg_start[1])*ny < 0:
        nx, ny = -nx, -ny
    cand = (seg_end[0]+nx*margin, seg_end[1]+ny*margin)
    clear = P._raw_segment_clear(cm, tuple(seg_end), cand)
    direct = P.astar(cm, cand, leg1_goal, unknown_cost_mult=P.UNKNOWN_COST_MULT)
    pinch = P._pinch_costmap(cm, gate, pinch_disc_m=P.PINCH_DISC_M, pinch_corridor_half_w_m=P.PINCH_CORRIDOR_HALF_W_M, start_xy=cand)
    viapinch = P.astar(pinch, cand, leg1_goal, unknown_cost_mult=P.UNKNOWN_COST_MULT)
    print(f"margin={margin} clear={clear} direct_astar={'OK' if direct else None} pinch_astar={'OK' if viapinch else None}")
