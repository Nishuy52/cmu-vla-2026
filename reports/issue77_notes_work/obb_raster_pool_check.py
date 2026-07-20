"""Issue #77 Pre-Stage 1a (OBB rasterization): re-check the issue77e 14-leg
arrival-blocked pool's "anchor cell reachable from grounding pose?" diagnostic
(trace_leg2.py pattern) against the OBB-rasterized mirror stamp, to count how
many legs become BFS-reachable now that rotated GT footprints stop stamping
their inflated AABB hull.

For each pool leg this reports the SAME check issue77e ran (all 14 were
"reachable? no" under the AABB-hull stamp): is the picked candidate anchor's
grid cell inside the grounding-pose's full BFS-reachable set now.
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

ROOT = Path(__file__).resolve().parents[2] / "data" / "vla3d" / "Unity"

# (scene, question-substring, leg-index) -- the issue77e 14-leg fixable pool.
POOL = [
    ("arabic_room", "stool under the picture", 0),
    ("arabic_room", "potted plant furthest from the hookah", 2),
    ("chinese_room", "potted plant on the table", 0),
    ("chinese_room", "tea table with the elephant figurine", 1),
    ("home_building_2", "magazine on the ottoman", 1),
    ("home_building_2", "Take the path between the sofa", 2),
    ("hotel_room_1", "bedside table closest to the window", 0),
    ("livingroom_2", "chair near the window", 1),
    ("livingroom_4", "near the fireplace, then go to the window", 0),
    ("loft", "cup near the TV remote", 0),
    ("loft", "fireplace, pass by the stairs", 2),
    ("office_1", "potted plant furthest from the projector screen", 0),
    ("office_2", "potted plant on the cabinet", 1),
    ("studio", "vases on the cabinet below the TV", 1),
]


def check(scene, qsub, leg_idx):
    with open(GB.DEFAULT_QUESTIONS, encoding="utf-8") as fh:
        data = json.load(fh)
    entry = next(e for e in data if e["scene"] == scene)
    folder = GB._find_scene_folder(ROOT, scene)
    gt = load_scene(folder, scene_name=scene)
    idx = BasicSceneIndex(gt.instances)

    if_texts = entry["questions"]["instruction_following"]
    qidx = next(i for i, t in enumerate(if_texts) if qsub in t)
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

    head, io, plan = GB._run_instruction_head(
        text, gt, idx, start_xy=spawn_xy, max_build_ticks=GB._IF_MAX_BUILD_TICKS,
        wall_cells=wall_cells,
    )
    if head is None or head._legs is None or leg_idx >= len(head._legs):
        return {"scene": scene, "qidx": qidx, "leg": leg_idx, "error": "no head/legs"}

    leg = head._legs[leg_idx]
    rec = leg.record
    c = TB.P._as3(rec.centroid)
    anchor_xy = (float(c[0]), float(c[1]))

    cm = head._costmap
    pose = head._pose
    seen = cm.reachable_mask(pose)
    r, col = head.grid.world_to_cell(*anchor_xy)
    reachable = bool(seen[r, col]) if (0 <= r < seen.shape[0] and 0 <= col < seen.shape[1]) else None
    return {"scene": scene, "qidx": qidx, "leg": leg_idx, "reachable": reachable}


if __name__ == "__main__":
    n_reachable = 0
    for scene, qsub, leg_idx in POOL:
        try:
            result = check(scene, qsub, leg_idx)
        except Exception as e:
            import traceback
            traceback.print_exc()
            result = {"scene": scene, "leg": leg_idx, "error": str(e)}
        print(result)
        if result.get("reachable"):
            n_reachable += 1
    print(f"\n{n_reachable}/{len(POOL)} pool legs' anchor now BFS-reachable from grounding pose")
