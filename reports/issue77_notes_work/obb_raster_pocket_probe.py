"""Issue #77 Pre-Stage 1a (OBB rasterization): §0-style BFS-pocket probe for
arabic_room q1 leg0 (the hookah-question pocket) -- OLD (AABB-hull stamp,
``rec.obb_heading`` zeroed so ``_synthetic_from_gt`` takes its pre-fix
axis-aligned branch) vs NEW (current code, real OBB heading), comparing the
grounding-pose BFS-reachable pocket size and nearest-reachable-to-anchor
distance -- the same two numbers §0 reports (536-cell pocket, 7.33 m).
"""
import copy
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

ROOT = Path(__file__).resolve().parents[2] / "data" / "vla3d" / "Unity"

SCENE = "arabic_room"
QIDX = 1
LEG_IDX = 0


def run(gt, idx, if_texts, label):
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
        if_texts[QIDX], gt, idx, start_xy=spawn_xy, max_build_ticks=GB._IF_MAX_BUILD_TICKS,
        wall_cells=wall_cells,
    )
    leg = head._legs[LEG_IDX]
    rec = leg.record
    c = TB.P._as3(rec.centroid)
    anchor_xy = (float(c[0]), float(c[1]))

    cm = head._costmap
    pose = head._pose
    seen = cm.reachable_mask(pose)
    rows, cols = np.where(seen)
    n_pocket = len(rows)
    xs, ys = [], []
    for rr, cc in zip(rows, cols):
        wx, wy = head.grid.cell_to_world(rr, cc)
        xs.append(wx)
        ys.append(wy)
    xs, ys = np.array(xs), np.array(ys)
    d = np.hypot(xs - anchor_xy[0], ys - anchor_xy[1])
    nearest = float(d.min()) if len(d) else None
    r, col = head.grid.world_to_cell(*anchor_xy)
    anchor_reachable = bool(seen[r, col]) if (0 <= r < seen.shape[0] and 0 <= col < seen.shape[1]) else None
    print(f"{label}: pocket={n_pocket} cells  nearest_reachable_to_anchor={nearest:.3f} m  "
          f"anchor_reachable={anchor_reachable}")
    return n_pocket, nearest


def main():
    with open(GB.DEFAULT_QUESTIONS, encoding="utf-8") as fh:
        data = json.load(fh)
    entry = next(e for e in data if e["scene"] == SCENE)
    folder = GB._find_scene_folder(ROOT, SCENE)
    gt = load_scene(folder, scene_name=SCENE)
    idx = BasicSceneIndex(gt.instances)
    if_texts = entry["questions"]["instruction_following"]

    print(f"=== {SCENE} q{QIDX} leg{LEG_IDX} BFS pocket (current NEW code) ===")
    run(gt, idx, if_texts, "NEW (real OBB heading)")

    # OLD: zero every instance's obb_heading so _synthetic_from_gt takes its
    # pre-fix axis-aligned (AABB-hull) branch -- the exact old behaviour, since the
    # heading==0 path is untouched by this change (regression-guard property).
    gt_old = load_scene(folder, scene_name=SCENE)
    for rec in gt_old.instances:
        rec.obb_heading = 0.0
    idx_old = BasicSceneIndex(gt_old.instances)
    run(gt_old, idx_old, if_texts, "OLD (AABB hull, heading zeroed)")


if __name__ == "__main__":
    main()
