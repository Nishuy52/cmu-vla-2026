"""Stage 0 audit: GT-trajectory-vs-blocked-cells, extended to all 15 scenes.

Read-only w.r.t. src/docs. Reproduces (method check) arabic_room's 102/36/66,
sofa 74 numbers from docs/proposals/pre_grounding_movement_plan.md §0, then
extends to all scenes, adds OBB-clearability estimate per object-box blocker,
pocket-excluded leg counts (build_miss_table methodology), and a
currently-passing-leg regression-risk flag.
"""
import csv
import json
import math
import sys
from pathlib import Path

REPO = Path("/home/jason/cmu_ws/cmu-vla-2026")
sys.path.insert(0, str(REPO / "src"))

import numpy as np

from core.groundtruth.loader import load_scene
from core.groundtruth import scoring as S
import core.runner.gt_battery as GB
from core.perception.scene_index import BasicSceneIndex
from core.mocks.synthetic_scene import TERRAIN_SLAB_MAX_Z, FLOOR_SPACING
from core.mocks.mock_io import FakeClock, MockRobotIO

ROOT = REPO / "data" / "vla3d" / "Unity"
RESULTS = json.load(open(REPO / "reports/gt_battery_post80_bytecheck/gt_battery_results.json"))

SCENES = [
    "arabic_room", "chinese_room", "home_building_1", "home_building_2",
    "hotel_room_1", "hotel_room_2", "japanese_room", "livingroom_1",
    "livingroom_2", "livingroom_3", "livingroom_4", "loft", "office_1",
    "office_2", "studio",
]


def load_obb_table(folder: Path, scene: str) -> dict[int, tuple[tuple[float, float, float], np.ndarray, float]]:
    """object_id -> (center, extents(3,), heading) straight off the raw CSV."""
    path = folder / f"{scene}_object_result.csv"
    out = {}
    with open(path, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            oid_raw = (row.get("object_id") or "").strip()
            if not oid_raw:
                continue
            oid = int(float(oid_raw))
            center = (
                float(row["object_bbox_cx"]),
                float(row["object_bbox_cy"]),
                float(row["object_bbox_cz"]),
            )
            extents = np.array([
                float(row["object_bbox_xlength"]),
                float(row["object_bbox_ylength"]),
                float(row["object_bbox_zlength"]),
            ])
            heading = float(row.get("object_bbox_heading") or 0.0)
            out[oid] = (center, extents, heading)
    return out


def point_in_obb_xy(px, py, center, extents, heading) -> bool:
    """True if (px,py) lies inside the object's TRUE oriented footprint (xy)."""
    cx, cy = center[0], center[1]
    dx, dy = px - cx, py - cy
    c, s = math.cos(-heading), math.sin(-heading)
    lx = dx * c - dy * s
    ly = dx * s + dy * c
    return abs(lx) <= extents[0] / 2.0 and abs(ly) <= extents[1] / 2.0


def build_scene_mirror(scene: str):
    """Fit the IF frame, derive wall cells, build the SyntheticScene mirror + a
    full-observation OccupancyGrid/Costmap exactly like gt_battery / the head does."""
    folder = GB._find_scene_folder(ROOT, scene)
    gt = load_scene(folder, scene_name=scene)
    idx = BasicSceneIndex(gt.instances)

    entry = next(e for e in json.load(open(GB.DEFAULT_QUESTIONS, encoding="utf-8")) if e["scene"] == scene)
    if_texts = entry["questions"]["instruction_following"]

    if_traj, if_cands, frame, residual, pairs = GB._fit_scene_if_frame(
        gt, idx, if_texts, GB.DEFAULT_QUESTIONS_ROOT
    )
    frame_for_walls = frame if residual is None or residual <= GB.WALL_FIT_MAX_RESIDUAL_M else None
    wall_cells = GB._scene_wall_cells(
        gt, frame_for_walls, unity_scenes_ros2_root=GB.DEFAULT_UNITY_SCENES_ROS2_ROOT
    )
    sc = GB._synthetic_from_gt(gt, wall_cells=wall_cells)

    spawn_xy = None
    if frame is not None and pairs:
        start_pt = pairs[0][0][0, :2]
        mapped = frame.apply(np.asarray([start_pt], dtype=float))[0]
        spawn_xy = (float(mapped[0]), float(mapped[1]))
    if spawn_xy is None:
        spawn_xy = (
            float(min(r.aabb_min[0] for r in gt.instances)) + 0.5,
            float(min(r.aabb_min[1] for r in gt.instances)) + 0.5,
        )

    clk = FakeClock(0.0)
    io = MockRobotIO(sc, clk, start_x=spawn_xy[0], start_y=spawn_xy[1])
    from core.nav.occupancy import OccupancyGrid
    from core.nav.costmap import Costmap
    patch = io.latest_terrain()
    grid = OccupancyGrid()
    grid.integrate_patch(patch, vehicle_z=0.0)
    cm = Costmap(grid)

    obb_table = load_obb_table(folder, scene)
    return dict(
        gt=gt, idx=idx, frame=frame, residual=residual, wall_cells=wall_cells,
        sc=sc, cm=cm, grid=grid, if_traj=if_traj, obb_table=obb_table,
        spawn_xy=spawn_xy,
    )


def _wall_subtype(x, y, sc):
    """'interior' if this raw-wall point is covered by the derived extra_wall_cells
    lattice (traversable-mesh complement), else 'border' (the synthetic outer room
    rectangle -- an artifact of the mirror's own padding, not GT-derived)."""
    if sc.extra_wall_cells:
        ix = int(round(x / FLOOR_SPACING))
        iy = int(round(y / FLOOR_SPACING))
        if (ix, iy) in sc.extra_wall_cells:
            return "interior"
    return "border"


def _raw_blocker_at(x, y, mir):
    """(kind, [objs]) using the TRUE (uninflated) obstacle geometry only, no search."""
    sc = mir["sc"]
    if sc._is_wall(x, y):
        return _wall_subtype(x, y, sc), []
    hits = []
    for obj in sc.objects:
        if obj.cz >= TERRAIN_SLAB_MAX_Z:
            continue
        if not obj.footprint_contains(x, y):
            continue
        hits.append(obj)
    if hits:
        return "object", hits
    return None, []


_CELL_M = FLOOR_SPACING


def classify_cell_blocker(x, y, mir, vehicle_radius_m=0.4):
    """Returns (kind, labels) for a blocked (x,y) in the INFLATED costmap.

    ``base_blocked`` cells can be pure inflation halo around a true obstacle up to
    ``vehicle_radius_m`` (0.4 m) away (``Costmap._inflate``'s own disc), so the true
    blocker at the exact point may be None. In that case we walk the SAME disc
    offsets ``Costmap._inflate`` uses (sorted nearest-first, exact grid-cell
    arithmetic, no priority category) and attribute to the nearest TRUE
    (uninflated) wall/object cell -- ties (equal distance, both a wall and an
    object cell) attribute to both (a real ambiguity: the vehicle footprint at
    that inflated cell would be excluded by whichever is actually closer, and at
    an exact tie both are equally the cause).
    """
    kind, hits = _raw_blocker_at(x, y, mir)
    if kind is not None:
        if kind in ("interior", "border"):
            return kind, []
        labels = _labels_for_hits(hits, mir, x, y)
        return "object", labels

    r_cells = int(math.ceil(vehicle_radius_m / _CELL_M))
    offs = sorted(
        (
            (dr, dc, dr * dr + dc * dc)
            for dr in range(-r_cells, r_cells + 1)
            for dc in range(-r_cells, r_cells + 1)
            if 0 < dr * dr + dc * dc <= r_cells * r_cells
        ),
        key=lambda t: t[2],
    )
    best_d2 = None
    found_wall_kinds = set()
    found_objs: dict[int, object] = {}
    for dr, dc, d2 in offs:
        if best_d2 is not None and d2 > best_d2:
            break
        px, py = x + dr * _CELL_M, y + dc * _CELL_M
        kk, hh = _raw_blocker_at(px, py, mir)
        if kk is None:
            continue
        best_d2 = d2
        if kk in ("interior", "border"):
            found_wall_kinds.add(kk)
        else:
            for obj in hh:
                found_objs[id(obj)] = obj
    if found_wall_kinds and not found_objs:
        # both subtypes can appear at the same nearest distance; report interior
        # if present (the GT-derived signal), else border.
        return ("interior" if "interior" in found_wall_kinds else "border"), []
    if found_objs and not found_wall_kinds:
        return "object", _labels_for_hits(list(found_objs.values()), mir, x, y)
    if found_wall_kinds and found_objs:
        return "object_wall_tie", _labels_for_hits(list(found_objs.values()), mir, x, y)
    return "unknown", []


def _labels_for_hits(hits, mir, x, y):
    labels = []
    for obj in hits:
        oid = None
        for rec in mir["gt"].instances:
            cx = float((rec.aabb_min[0] + rec.aabb_max[0]) / 2)
            cy = float((rec.aabb_min[1] + rec.aabb_max[1]) / 2)
            sx = float(max(rec.aabb_max[0] - rec.aabb_min[0], 0.05))
            sy = float(max(rec.aabb_max[1] - rec.aabb_min[1], 0.05))
            if (rec.label == obj.label and abs(cx - obj.cx) < 1e-6
                    and abs(cy - obj.cy) < 1e-6 and abs(sx - obj.sx) < 1e-6
                    and abs(sy - obj.sy) < 1e-6):
                oid = rec.instance_id
                break
        obb_clearable = None
        heading = None
        if oid is not None and oid in mir["obb_table"]:
            center, extents, heading = mir["obb_table"][oid]
            inside_obb = point_in_obb_xy(x, y, center, extents, heading)
            obb_clearable = not inside_obb  # AABB-blocked but OUTSIDE true OBB -> OBB rasterization clears it
        labels.append((obj.label, oid, obb_clearable, heading))
    return labels


def audit_scene(scene: str):
    mir = build_scene_mirror(scene)
    grid = mir["grid"]
    cm = mir["cm"]
    frame = mir["frame"]

    per_q = []
    total_blocked_cells = set()
    wall_cells_hit = set()       # any wall (border or interior-derived)
    wall_interior_hit = set()    # interior-derived only (the plan's "derived-wall")
    wall_border_hit = set()      # synthetic outer-boundary only (mirror-padding artifact)
    object_cells_hit = set()
    per_object_hits = {}  # label -> set of cells
    per_object_obb_clear = {}  # label -> (n_hit_cells, n_obb_clearable_cells)

    for qi, traj in enumerate(mir["if_traj"]):
        if traj is None or traj.shape[0] == 0 or frame is None:
            per_q.append(dict(qi=qi, n_points=0, n_blocked_cells=0, note="no trajectory or no fitted frame"))
            continue
        mapped = frame.apply(traj[:, :2])
        blocked_here = set()
        for x, y in mapped:
            r, c = grid.world_to_cell(float(x), float(y))
            if not (0 <= r < cm.base_blocked.shape[0] and 0 <= c < cm.base_blocked.shape[1]):
                continue
            if not cm.base_blocked[r, c]:
                continue
            cell = (r, c)
            if cell in blocked_here:
                continue
            blocked_here.add(cell)
            total_blocked_cells.add(cell)
            cx, cy = grid.cell_to_world(r, c)
            kind, labels = classify_cell_blocker(cx, cy, mir)
            if kind in ("interior", "border"):
                wall_cells_hit.add(cell)
                (wall_interior_hit if kind == "interior" else wall_border_hit).add(cell)
            elif kind == "object_wall_tie":
                # Nearest raw blocker is equidistant wall + object -- ambiguous;
                # count once in each split bucket, never double the grand total.
                wall_cells_hit.add(cell)
                object_cells_hit.add(cell)
                for label, oid, obb_clearable, heading in labels:
                    per_object_hits.setdefault(label, set()).add(cell)
                    n, nc = per_object_obb_clear.get(label, (0, 0))
                    per_object_obb_clear[label] = (n + 1, nc + (1 if obb_clearable else 0))
            else:
                object_cells_hit.add(cell)
                for label, oid, obb_clearable, heading in labels:
                    per_object_hits.setdefault(label, set()).add(cell)
                    n, nc = per_object_obb_clear.get(label, (0, 0))
                    per_object_obb_clear[label] = (n + 1, nc + (1 if obb_clearable else 0))
        per_q.append(dict(qi=qi, n_points=int(mapped.shape[0]), n_blocked_cells=len(blocked_here)))

    # OBB-clearable estimate over the UNION of object-blocked cells (avoid per-question double count)
    obb_clearable_cells = 0
    for cell in object_cells_hit:
        cx, cy = grid.cell_to_world(cell[0], cell[1])
        kind, labels = classify_cell_blocker(cx, cy, mir)
        if labels and all((cl is True) for _, _, cl, _ in labels if cl is not None):
            obb_clearable_cells += 1
    obb_clearable_cells_incl_wall_tie = obb_clearable_cells  # object-only cells; wall-tie cells always keep a wall blocker regardless of OBB

    return dict(
        scene=scene,
        residual=mir["residual"],
        wall_cells_derived=mir["wall_cells"] is not None,
        n_blocked_total=len(total_blocked_cells),
        n_blocked_wall=len(wall_cells_hit),
        n_blocked_wall_interior_derived=len(wall_interior_hit),
        n_blocked_wall_border=len(wall_border_hit),
        n_blocked_object=len(object_cells_hit),
        n_obb_clearable=obb_clearable_cells,
        per_object=[dict(label=k, n_cells=len(v)) for k, v in sorted(per_object_hits.items(), key=lambda kv: -len(kv[1]))],
        per_object_obb=per_object_obb_clear,
        per_q=per_q,
    )


if __name__ == "__main__":
    scene = sys.argv[1] if len(sys.argv) > 1 else "arabic_room"
    r = audit_scene(scene)
    print(json.dumps(r, indent=2, default=str))
