"""Diagnostic (issue #77b): trace driven poses near the gate for one dwell-class
corridor leg, alongside the exact gate segment threading_check tests against.

Not part of the scored pipeline / test suite -- ephemeral debug script.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np

from core.groundtruth.loader import load_scene
from core.runner.gt_battery import (
    _drive_if_trajectory,
    _fit_scene_if_frame,
    _if_rubric_geometry,
    _scene_wall_cells,
    _find_scene_folder,
    WALL_FIT_MAX_RESIDUAL_M,
    DEFAULT_QUESTIONS,
    DEFAULT_QUESTIONS_ROOT,
    DEFAULT_UNITY_SCENES_ROS2_ROOT,
)
from core.perception.scene_index import BasicSceneIndex
from core.geometry import primitives as P
from core.geometry.toolbox import threading_check

ROOT = Path("../data/vla3d/Unity")
SCENE = sys.argv[1] if len(sys.argv) > 1 else "hotel_room_2"
QIDX = int(sys.argv[2]) if len(sys.argv) > 2 else 0  # which IF question in the scene (0-based)

with open(DEFAULT_QUESTIONS, encoding="utf-8") as fh:
    data = json.load(fh)
entry = next(e for e in data if e["scene"] == SCENE)
folder = _find_scene_folder(ROOT, SCENE)
gt = load_scene(folder, scene_name=SCENE)
idx = BasicSceneIndex(gt.instances)

if_texts = entry["questions"]["instruction_following"]
text = if_texts[QIDX]
print("QUESTION:", text)

if_traj, if_cands, frame, residual, pairs = _fit_scene_if_frame(
    gt, idx, if_texts, DEFAULT_QUESTIONS_ROOT
)
spawn_xy = None
if frame is not None and pairs:
    start_pt = pairs[0][0][0, :2]
    mapped = frame.apply(np.asarray([start_pt], dtype=float))[0]
    spawn_xy = (float(mapped[0]), float(mapped[1]))
print("spawn_xy:", spawn_xy, "fit residual:", residual)

frame_for_walls = frame if residual is None or residual <= WALL_FIT_MAX_RESIDUAL_M else None
wall_cells = _scene_wall_cells(
    gt, frame_for_walls, unity_scenes_ros2_root=DEFAULT_UNITY_SCENES_ROS2_ROOT
)

driven = _drive_if_trajectory(text, gt, idx, start_xy=spawn_xy, wall_cells=wall_cells)
leg_goals, corridor_gates, avoid_caps, leg_instance_ids, leg_instance_aabbs = (
    _if_rubric_geometry(text, gt, idx, start_xy=spawn_xy)
)

print("n driven poses:", len(driven))
print("leg_goals:", leg_goals)
print("corridor_gates:", [(i, g.p0, g.p1) for i, g in corridor_gates])

for leg_i, gate in corridor_gates:
    p0, p1 = np.asarray(gate.p0, dtype=float), np.asarray(gate.p1, dtype=float)
    ok, msg = threading_check(driven, gate)
    print(f"\n--- corridor leg {leg_i}: gate p0={p0.tolist()} p1={p1.tolist()} ---")
    print("threading_check:", ok, msg)
    # distance of each driven pose to the gate LINE (infinite line through p0,p1) and
    # to the SEGMENT, plus which side of the line it's on (cross-product sign) --
    # a sign flip between consecutive poses without an intersection would be a bug in
    # segments_intersect_2d; no sign flip at all means the poses never actually
    # straddled the line.
    d = p1 - p0
    def side(pt):
        pt = np.asarray(pt, dtype=float)
        v = pt - p0
        return float(d[0] * v[1] - d[1] * v[0])
    dists = [abs(side(pt)) / np.linalg.norm(d) for pt in driven]
    sides = [side(pt) for pt in driven]
    # print the 15 driven poses nearest the gate midpoint
    mid = (p0 + p1) / 2
    order = np.argsort([float(np.linalg.norm(pt - mid)) for pt in driven])[:20]
    order = sorted(order.tolist())
    print("closest-to-goal driven poses (index, x, y, perp_dist_to_line, side_sign, seg_dist):")
    for k in order:
        pt = driven[k]
        seg_d = P.point_segment_distance(pt, p0, p1) if hasattr(P, "point_segment_distance") else None
        print(f"  [{k:4d}] ({pt[0]:.4f}, {pt[1]:.4f})  perp={dists[k]:.4f}  side={sides[k]:+.4f}")
    print("min perp dist to line among ALL poses:", min(dists))
    print("sign range among ALL poses: min side=%.4f max side=%.4f" % (min(sides), max(sides)))
