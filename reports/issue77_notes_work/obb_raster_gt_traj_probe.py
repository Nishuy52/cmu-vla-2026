"""Issue #77 Pre-Stage 1a (OBB rasterization): reproduce the §0 probe
(pre_grounding_movement_plan.md) -- "the GT reference trajectory vs the
mirror drives through N mirror-blocked cells" -- for arabic_room q1 (the
hookah-question leg0 pocket), comparing the OLD AABB-hull stamp against the
NEW OBB-rasterized stamp.

For each GT-trajectory sample point, snap to the FLOOR_SPACING lattice and
check membership against every stamped GTObject footprint (old: axis-aligned
box built from AABB min/max i.e. heading forced to 0; new: real OBB heading)
plus the derived-wall cell set, tallying blocked counts by source label.
"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np

from core.groundtruth.loader import load_scene
import core.runner.gt_battery as GB
from core.perception.scene_index import BasicSceneIndex
from core.mocks.synthetic_scene import (
    FLOOR_SPACING,
    TERRAIN_SLAB_MAX_Z,
    Room,
    SyntheticScene,
)

ROOT = Path(__file__).resolve().parents[2] / "data" / "vla3d" / "Unity"

SCENE = "arabic_room"
QIDX = 1  # the hookah-question (leg0 pocket, §0's probe)


def snap(v):
    return round(v / FLOOR_SPACING) * FLOOR_SPACING


def classify(x, y, sc, wall_cells):
    """Return the blocking source label at (x, y), or None if free.

    Mirrors ``SyntheticScene`` cell semantics: wall border/derived-wall cells
    first (as ``terrain_patch``/``_is_wall`` check them ahead of objects),
    then the first floor-level object whose footprint contains the point.
    """
    if sc._is_wall(x, y):
        return "wall"
    for obj in sc.objects:
        if obj.cz >= TERRAIN_SLAB_MAX_Z:
            continue
        if obj.footprint_contains(x, y):
            return obj.label
    return None


def main():
    with open(GB.DEFAULT_QUESTIONS, encoding="utf-8") as fh:
        data = json.load(fh)
    entry = next(e for e in data if e["scene"] == SCENE)
    folder = GB._find_scene_folder(ROOT, SCENE)
    gt = load_scene(folder, scene_name=SCENE)
    idx = BasicSceneIndex(gt.instances)

    if_texts = entry["questions"]["instruction_following"]
    if_traj, if_cands, frame, residual, pairs = GB._fit_scene_if_frame(
        gt, idx, if_texts, GB.DEFAULT_QUESTIONS_ROOT
    )
    traj = if_traj[QIDX]
    assert traj is not None and traj.shape[0] > 0, "no GT trajectory for this qidx"
    mapped = frame.apply(traj[:, :2])

    frame_for_walls = frame if residual is None or residual <= GB.WALL_FIT_MAX_RESIDUAL_M else None
    wall_cells = GB._scene_wall_cells(
        gt, frame_for_walls, unity_scenes_ros2_root=GB.DEFAULT_UNITY_SCENES_ROS2_ROOT
    )

    # NEW: current code (real OBB heading, via _synthetic_from_gt as committed).
    sc_new = GB._synthetic_from_gt(gt, wall_cells=wall_cells)

    # OLD: reproduce the pre-fix AABB-hull stamp verbatim -- axis-aligned box at the
    # AABB min/max centre/size (the ACTUAL pre-fix formula; NOT sc_new's objects with
    # heading merely zeroed out, which for an exact 90deg-heading object would keep
    # the OBB's un-swapped local extents and so understate the old AABB's real size).
    pad = 1.5
    x0, y0, x1, y1 = GB._gt_footprint_bounds(gt, 0.0)
    room_w, room_h = x1 - x0, y1 - y0
    sc_old = SyntheticScene(0, extra_wall_cells=wall_cells)
    sc_old.rooms = [Room(x0 - pad, y0 - pad, x1 + pad, y1 + pad)]
    sc_old._split_x = None
    sc_old.doorway = None
    sc_old.objects = []
    for rec in gt.instances:
        if GB._is_architectural_room_scale_aabb(rec.aabb_min, rec.aabb_max, room_w, room_h):
            continue
        cx = float((rec.aabb_min[0] + rec.aabb_max[0]) / 2)
        cy = float((rec.aabb_min[1] + rec.aabb_max[1]) / 2)
        sx = float(max(rec.aabb_max[0] - rec.aabb_min[0], 0.05))
        sy = float(max(rec.aabb_max[1] - rec.aabb_min[1], 0.05))
        sz = float(max(rec.aabb_max[2] - rec.aabb_min[2], 0.05))
        cz = float(rec.aabb_min[2])
        sc_old.place_box(rec.label, cx, cy, sx, sy, sz, cz=cz)  # heading defaults 0

    def tally(sc, label):
        blocked = Counter()
        n_blocked = 0
        seen_cells = set()
        for x, y in mapped:
            gx, gy = snap(float(x)), snap(float(y))
            if (gx, gy) in seen_cells:
                continue
            seen_cells.add((gx, gy))
            src = classify(gx, gy, sc, wall_cells)
            if src is not None:
                n_blocked += 1
                blocked[src] += 1
        print(f"{label}: {n_blocked} distinct GT-trajectory-lattice cells blocked "
              f"(of {len(seen_cells)} distinct sampled cells)")
        for k, v in blocked.most_common():
            print(f"    {k}: {v}")
        return n_blocked, blocked

    print(f"=== {SCENE} q{QIDX} GT-trajectory vs mirror stamp ===")
    n_old, b_old = tally(sc_old, "OLD (AABB hull, heading forced 0)")
    n_new, b_new = tally(sc_new, "NEW (OBB rasterized)")
    print(f"\ndelta: {n_old} -> {n_new}  ({n_old - n_new} fewer blocked cells)")


if __name__ == "__main__":
    main()
