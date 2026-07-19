"""Deeper look at the leg0 reachable pocket vs anchor: bounding box of the
plain-BFS reachable region, plus the true nearest FREE (not necessarily
reachable) cell to the anchor, to tell disconnection-by-wall apart from
disconnection-by-real-clutter/geometry."""
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
    print(f"=== {scene} q{qidx} leg{leg_idx} === wall_cells={None if wall_cells is None else len(wall_cells)}")

    head, io, plan = GB._run_instruction_head(
        text, gt, idx, start_xy=spawn_xy, max_build_ticks=GB._IF_MAX_BUILD_TICKS,
        wall_cells=wall_cells,
    )
    leg = head._legs[leg_idx]
    rec = leg.record
    c = TB.P._as3(rec.centroid)
    anchor_xy = (float(c[0]), float(c[1]))

    cm = head._costmap
    pose = head._pose
    seen = cm.reachable_mask(pose)
    r, col = head.grid.world_to_cell(*anchor_xy)
    rows, cols = np.where(seen)
    if len(rows):
        xs, ys = [], []
        for rr, cc in zip(rows, cols):
            wx, wy = head.grid.cell_to_world(rr, cc)
            xs.append(wx); ys.append(wy)
        xs = np.array(xs); ys = np.array(ys)
        print(f"  reachable pocket bbox: x[{xs.min():.2f},{xs.max():.2f}] y[{ys.min():.2f},{ys.max():.2f}] n={len(xs)}")
        # nearest reachable cell to anchor (brute)
        d = np.hypot(xs - anchor_xy[0], ys - anchor_xy[1])
        j = int(np.argmin(d))
        print(f"  nearest reachable cell to anchor: ({xs[j]:.2f},{ys[j]:.2f}) dist={d[j]:.3f}")
    print(f"  anchor_xy={anchor_xy} spawn={pose}")

    # free (not obstacle) mask regardless of reachability, if available
    free = getattr(cm, "free_mask", None) or getattr(cm, "traversable_mask", None)
    grid = head.grid
    print(f"  grid shape={getattr(grid,'shape',None)} cell_m={getattr(grid,'cell_m',None)}")
    print(f"  costmap attrs sample: {[a for a in dir(cm) if 'free' in a.lower() or 'block' in a.lower() or 'wall' in a.lower()]}")
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
        except Exception:
            import traceback
            traceback.print_exc()
