"""Offline dev tool: multi-scene validation of the overhead-clearance tunables.

Background (redteam H13 / SYS-F12, `docs/calibration.md` nav table + boxed
caveat): the five overhead tunables in ``core.nav.occupancy.OverheadConfig``
(``overhead_min``, ``overhead_max``, ``min_points_per_cell``,
``vehicle_sensor_height``) plus ``OVERHEAD_SCAN_MAX_PTS`` were fitted to a
single real-robot bag (jingfan) not present on this machine. This tool replays
recorded *sim* bags from other scenes through the same occupancy-grid wiring
the live heads use, then checks the resulting OVERHEAD flags against each
scene's ground-truth object list and traversable-area point cloud.

Replay wiring mirrors the call sites in ``core/heads/instruction.py`` and
``core/heads/explore_step.py``: ``integrate_patch`` for each non-extended
``/terrain_map`` patch (terrain first, so ``ground_z`` is populated before the
overhead pass reads it), ``integrate_scan_overhead_decimated`` for each
``/registered_scan`` using the *latest* odom's z as ``vehicle_z``, and
``mark_pose`` per odom sample -- but at a sane cadence (default 1 Hz of bag
time), not every odom message (odom arrives at 100-200 Hz; marking that often
buys nothing and is pure overhead, see F13).

Sensitivity variants (deliverable item 5: band shifted by the measured
sensor-height delta; ``min_points_per_cell=2``) are NOT separate bag re-reads.
The decimation stride and per-cell ground_z bookkeeping in
``integrate_scan_overhead_decimated``/``integrate_scan_overhead`` are cheap
numpy ops on already-decoded points; the expensive part of a replay is
decoding the mcap (image frames dwarf everything else, hence they are excluded
from the topic map below). So one bag pass drives three independent
``OccupancyGrid`` instances (default / shifted / minpts2) in lock-step, fed
identical terrain patches and scan points, differing only in the
``OverheadConfig`` passed to the overhead call -- equivalent to three full
replays without paying for three mcap decodes.

CLI::

    python -m tools.overhead_validation run <bag_dir> <scene_dir> \\
        --out <out.json> [--summary-md <summary.md>] [--mark-cadence 1.0]

Pure stdlib + numpy + rosbags (via BagSource). Offline dev tool -- not part of
the scored pipeline; never imported by ``core/``.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

# Ensure <repo>/src is importable (also done by the package __init__ on -m import).
import tools  # noqa: F401  (side effect: sys.path bootstrap)

from core.interfaces import LidarScan, OdomState, TerrainPatch
from core.nav.occupancy import (
    DEFAULT_OVERHEAD_CONFIG,
    OVERHEAD_SCAN_MAX_PTS,
    OccupancyGrid,
    OverheadConfig,
    integrate_scan_overhead_decimated,
)
from core.replay.bag_reader import (
    TOPIC_ODOM,
    TOPIC_SCAN,
    TOPIC_TERRAIN,
    BagSource,
    odom_to_state,
    pointcloud_to_lidar,
    pointcloud_to_terrain,
)

#: Only decode the topics the overhead replay actually needs -- skipping
#: /camera/image (the dominant contributor to bag size) is what keeps a full
#: replay to the ~2-3 min ballpark instead of tens of minutes.
_REPLAY_TOPIC_MAP = {
    TOPIC_ODOM: odom_to_state,
    TOPIC_TERRAIN: pointcloud_to_terrain,
    TOPIC_SCAN: pointcloud_to_lidar,
}
#: Pass 1 (median vehicle_z) needs odom only -- narrower still.
_ODOM_ONLY_TOPIC_MAP = {TOPIC_ODOM: odom_to_state}

_KNOWN_OVERHANG_KEYWORDS = ("table", "desk", "shelf", "cabinet")
_KNOWN_OVERHANG_TOP_MARGIN_M = 0.4
_FALSE_FLAG_INFLATE_CELLS = 1
_FALSE_FLAG_TOP_N = 10


# --------------------------------------------------------------------------- scene GT

@dataclass(frozen=True)
class SceneObject:
    """One row of ``object_list.txt``: a yaw-rotated AABB in the map frame."""

    id: int
    x: float
    y: float
    z: float
    dx: float
    dy: float
    dz: float
    yaw: float
    name: str

    @property
    def top_z(self) -> float:
        return self.z + self.dz / 2.0

    @property
    def bottom_z(self) -> float:
        return self.z - self.dz / 2.0


_OBJECT_LINE_RE = re.compile(
    r"^\s*(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+\"(.*)\"\s*$"
)


def parse_object_list(path) -> list[SceneObject]:
    """Parse ``object_list.txt``: ``id x y z dx dy dz yaw "name"`` per line.

    ``x/y/z`` = box centre (map frame), ``dx/dy/dz`` = full extents (size, not
    half-extents), ``yaw`` = radians about z. Blank lines are skipped.
    """
    objs: list[SceneObject] = []
    text = Path(path).read_text(encoding="utf-8")
    for lineno, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        m = _OBJECT_LINE_RE.match(line)
        if not m:
            raise ValueError(f"{path}:{lineno}: unparseable object_list row: {line!r}")
        oid, x, y, z, dx, dy, dz, yaw, name = m.groups()
        objs.append(
            SceneObject(
                id=int(float(oid)),
                x=float(x),
                y=float(y),
                z=float(z),
                dx=float(dx),
                dy=float(dy),
                dz=float(dz),
                yaw=float(yaw),
                name=name,
            )
        )
    return objs


def read_ply_xyz(path) -> np.ndarray:
    """Minimal ASCII-PLY xyz reader (no external PLY dependency).

    Reads the header only far enough to find ``element vertex N`` and
    ``end_header``, then parses ``N`` whitespace-separated rows, keeping just
    the first three columns (x, y, z) -- extra per-vertex properties, if any,
    are ignored. Raises on a binary PLY (unsupported; the scene fixtures here
    are all ``format ascii``).
    """
    path = Path(path)
    n_vertex = None
    with path.open("r", encoding="ascii", errors="strict") as fh:
        header_lines = []
        for line in fh:
            header_lines.append(line)
            stripped = line.strip()
            if stripped.startswith("format") and "ascii" not in stripped:
                raise ValueError(f"{path}: only ASCII PLY is supported, got {stripped!r}")
            if stripped.startswith("element vertex"):
                n_vertex = int(stripped.split()[-1])
            if stripped == "end_header":
                break
        if n_vertex is None:
            raise ValueError(f"{path}: no 'element vertex N' in PLY header")
        pts = np.empty((n_vertex, 3), dtype=np.float64)
        for i in range(n_vertex):
            row = fh.readline()
            if not row:
                raise ValueError(f"{path}: truncated PLY body, expected {n_vertex} vertices")
            parts = row.split()
            pts[i, 0] = float(parts[0])
            pts[i, 1] = float(parts[1])
            pts[i, 2] = float(parts[2])
    return pts


def floor_z_from_traversable(points: np.ndarray) -> float:
    """Scene floor_z = median z of the traversable-area point cloud."""
    if points.size == 0:
        raise ValueError("traversable_area.ply has no points")
    return float(np.median(points[:, 2]))


def object_overlaps_band(
    obj: SceneObject, floor_z: float, band_min: float, band_max: float
) -> bool:
    """True if ``obj``'s vertical extent intersects [floor_z+band_min, floor_z+band_max]."""
    lo = floor_z + band_min
    hi = floor_z + band_max
    return obj.bottom_z <= hi and obj.top_z >= lo


def known_overhang_objects(
    objects: list[SceneObject],
    floor_z: float,
    *,
    keywords: tuple[str, ...] = _KNOWN_OVERHANG_KEYWORDS,
    top_margin: float = _KNOWN_OVERHANG_TOP_MARGIN_M,
    band_max: float = DEFAULT_OVERHEAD_CONFIG.overhead_max,
) -> list[SceneObject]:
    """Named overhang furniture the layer is expected to catch.

    Name matches a keyword, top is well above the floor (``top_margin``), AND
    the underside sits at/below the clearance-band top (``floor_z + band_max``).
    The last condition is issue #37 hypothesis (a): an object mounted entirely
    ABOVE the band (``bottom_z > floor_z + band_max``) is out of the overhead
    layer's watched range by design -- the layer never claims to catch it, so
    counting it as a "known overhang" over-includes correctly-unflagged objects
    as misses. ``band_max`` defaults to the live ``OverheadConfig.overhead_max``
    so this classifier tracks the real band without a hardcoded duplicate.
    """
    out = []
    for obj in objects:
        name_l = obj.name.lower()
        if not any(kw in name_l for kw in keywords):
            continue
        if obj.top_z <= floor_z + top_margin:
            continue
        if obj.bottom_z > floor_z + band_max:
            continue
        out.append(obj)
    return out


def rotated_rect_cells(
    cx: float,
    cy: float,
    dx: float,
    dy: float,
    yaw: float,
    cell_m: float,
    origin_x: float,
    origin_y: float,
    *,
    inflate_cells: int = 0,
) -> set[tuple[int, int]]:
    """Rasterize a yaw-rotated footprint rectangle (full extents dx, dy) to grid cells.

    ``(row, col)`` cell identity matches ``OccupancyGrid.world_to_cell`` for the
    same ``(cell_m, origin_x, origin_y)`` -- callers pass a live grid's origin
    to get directly-comparable cell ids. Optionally inflates the half-extents
    by ``inflate_cells`` cells before the point-in-rect test (used for the
    false-flag GT footprint per the validation spec).
    """
    half_x = dx / 2.0 + inflate_cells * cell_m
    half_y = dy / 2.0 + inflate_cells * cell_m
    corner_radius = math.hypot(half_x, half_y)
    min_x, max_x = cx - corner_radius, cx + corner_radius
    min_y, max_y = cy - corner_radius, cy + corner_radius
    col_lo = int(math.floor((min_x - origin_x) / cell_m))
    col_hi = int(math.floor((max_x - origin_x) / cell_m))
    row_lo = int(math.floor((min_y - origin_y) / cell_m))
    row_hi = int(math.floor((max_y - origin_y) / cell_m))

    cos_t = math.cos(-yaw)
    sin_t = math.sin(-yaw)
    cells: set[tuple[int, int]] = set()
    for row in range(row_lo, row_hi + 1):
        wy = origin_y + (row + 0.5) * cell_m
        dy_w = wy - cy
        for col in range(col_lo, col_hi + 1):
            wx = origin_x + (col + 0.5) * cell_m
            dx_w = wx - cx
            lx = dx_w * cos_t - dy_w * sin_t
            ly = dx_w * sin_t + dy_w * cos_t
            if abs(lx) <= half_x and abs(ly) <= half_y:
                cells.add((row, col))
    return cells


def rasterize_points_to_cells(
    points: np.ndarray, cell_m: float, origin_x: float, origin_y: float
) -> set[tuple[int, int]]:
    """Map (N, >=2) xy points to the (row, col) cell lattice used by ``OccupancyGrid``."""
    if points.size == 0:
        return set()
    cols = np.floor((points[:, 0] - origin_x) / cell_m).astype(np.int64)
    rows = np.floor((points[:, 1] - origin_y) / cell_m).astype(np.int64)
    return set(zip(rows.tolist(), cols.tolist()))


# --------------------------------------------------------------------------- replay

_VARIANT_NAMES = ("default", "shifted", "minpts2")


def _quick_median_vehicle_z(bag_path) -> float:
    """Pass 1: odom-only scan of the bag to get median vehicle_z (cheap; no clouds)."""
    zs: list[float] = []
    for rec in BagSource(bag_path, topic_map=_ODOM_ONLY_TOPIC_MAP):
        if isinstance(rec.msg, OdomState):
            zs.append(float(rec.msg.z))
    if not zs:
        raise ValueError(f"{bag_path}: no /state_estimation messages found")
    return float(statistics.median(zs))


@dataclass
class ReplayResult:
    grids: dict[str, OccupancyGrid]
    cfgs: dict[str, OverheadConfig]
    n_odom: int
    n_terrain: int
    n_scan: int
    n_pose_marks: int
    median_vehicle_z: float
    bag_t_span: tuple[float, float] | None
    mean_scan_hz: float | None


def run_replay(
    bag_path,
    *,
    floor_z: float,
    mark_cadence_s: float = 1.0,
) -> ReplayResult:
    """Replay one bag through three lock-step ``OverheadConfig`` variants.

    ``default`` = ``DEFAULT_OVERHEAD_CONFIG``; ``shifted`` corrects
    ``vehicle_sensor_height`` to this bag's actual measured
    ``median(vehicle_z) - floor_z`` (item 1/5: the effective fallback-ground
    shift); ``minpts2`` lowers ``min_points_per_cell`` to 2. All three share
    identical terrain patches, scan points, and pose marks -- only the
    overhead-flagging config differs.
    """
    median_vz = _quick_median_vehicle_z(bag_path)
    measured_sensor_height = median_vz - floor_z

    cfgs: dict[str, OverheadConfig] = {
        "default": DEFAULT_OVERHEAD_CONFIG,
        # "shifted"/"minpts2" isolate their one hypothesis: pin the runtime
        # ground-offset estimator off (issue #36) so they still test the
        # constant-only sensor-height / min-points hypotheses untouched by the
        # now-default estimator (which "default" exercises instead).
        "shifted": replace(
            DEFAULT_OVERHEAD_CONFIG,
            vehicle_sensor_height=measured_sensor_height,
            use_ground_offset_estimator=False,
        ),
        "minpts2": replace(
            DEFAULT_OVERHEAD_CONFIG, min_points_per_cell=2, use_ground_offset_estimator=False
        ),
    }
    grids: dict[str, OccupancyGrid] = {name: OccupancyGrid() for name in _VARIANT_NAMES}

    latest_odom: OdomState | None = None
    last_mark_t: float | None = None
    n_odom = n_terrain = n_scan = n_pose_marks = 0
    t_min: float | None = None
    t_max: float | None = None
    scan_ts: list[float] = []

    for rec in BagSource(bag_path, topic_map=_REPLAY_TOPIC_MAP):
        t_min = rec.t if t_min is None else min(t_min, rec.t)
        t_max = rec.t if t_max is None else max(t_max, rec.t)

        if isinstance(rec.msg, OdomState):
            latest_odom = rec.msg
            n_odom += 1
            if last_mark_t is None or (rec.t - last_mark_t) >= mark_cadence_s:
                for grid in grids.values():
                    grid.mark_pose(latest_odom.x, latest_odom.y)
                last_mark_t = rec.t
                n_pose_marks += 1
        elif isinstance(rec.msg, TerrainPatch):
            if rec.msg.extended:
                continue  # live wiring uses the non-extended /terrain_map only
            n_terrain += 1
            vehicle_z = float(latest_odom.z) if latest_odom is not None else None
            for grid in grids.values():
                grid.integrate_patch(rec.msg, vehicle_z=vehicle_z)
        elif isinstance(rec.msg, LidarScan):
            n_scan += 1
            scan_ts.append(rec.t)
            vehicle_z = float(latest_odom.z) if latest_odom is not None else 0.0
            for name, grid in grids.items():
                integrate_scan_overhead_decimated(
                    grid, rec.msg, vehicle_z, cfg=cfgs[name], max_pts=OVERHEAD_SCAN_MAX_PTS
                )

    bag_span = (t_min, t_max) if t_min is not None and t_max is not None else None
    mean_scan_hz = None
    if len(scan_ts) >= 2:
        dur = scan_ts[-1] - scan_ts[0]
        if dur > 0:
            mean_scan_hz = (len(scan_ts) - 1) / dur

    return ReplayResult(
        grids=grids,
        cfgs=cfgs,
        n_odom=n_odom,
        n_terrain=n_terrain,
        n_scan=n_scan,
        n_pose_marks=n_pose_marks,
        median_vehicle_z=median_vz,
        bag_t_span=bag_span,
        mean_scan_hz=mean_scan_hz,
    )


# --------------------------------------------------------------------------- metrics


def _grid_flagged_cells(grid: OccupancyGrid, *, observed_only: bool = True) -> set[tuple[int, int]]:
    mask = grid.overhead
    if observed_only:
        mask = mask & grid.observed
    rows, cols = np.nonzero(mask)
    return set(zip(rows.tolist(), cols.tolist()))


def compute_metrics(
    replay: ReplayResult,
    *,
    scene_objects: list[SceneObject],
    traversable_pts: np.ndarray,
    floor_z: float,
    overhead_min: float = DEFAULT_OVERHEAD_CONFIG.overhead_min,
    overhead_max: float = DEFAULT_OVERHEAD_CONFIG.overhead_max,
) -> dict:
    default_grid = replay.grids["default"]
    cell_m = default_grid.cell_m
    origin_x, origin_y = default_grid.origin_x, default_grid.origin_y

    # --- item 1: sensor height -------------------------------------------------
    configured_height = DEFAULT_OVERHEAD_CONFIG.vehicle_sensor_height
    measured_height = replay.median_vehicle_z - floor_z
    delta = measured_height - configured_height

    fallback_mask = np.isnan(default_grid.ground_z) & default_grid.overhead & default_grid.observed
    terrain_mask = (~np.isnan(default_grid.ground_z)) & default_grid.overhead & default_grid.observed
    n_fallback = int(fallback_mask.sum())
    n_terrain_ground = int(terrain_mask.sum())

    estimated_offset = default_grid.ground_offset_estimate
    sensor_height = {
        "configured_vehicle_sensor_height_m": configured_height,
        "median_vehicle_z_m": replay.median_vehicle_z,
        "floor_z_m": floor_z,
        "measured_sensor_height_m": measured_height,
        "delta_m": delta,
        "note": (
            "delta = measured - configured; a positive delta means the real "
            "sensor sits higher than assumed, so the vehicle_z-0.60 fallback "
            "UNDER-estimates floor_z (over-estimates height-above-floor) by "
            "about this much for cells that never saw terrain"
        ),
        "flagged_cells_fallback_ground_z_nan": n_fallback,
        "flagged_cells_terrain_ground_z": n_terrain_ground,
        # issue #36 fix: the "default" grid's own runtime ground-offset estimate
        # (median vehicle_z - min terrain ground z), used for its fallback cells
        # once warmed -- should land close to measured_sensor_height_m, not
        # configured_vehicle_sensor_height_m.
        "estimated_ground_offset_m": estimated_offset,
    }

    # --- item 2: overhead cell counts ------------------------------------------
    observed = default_grid.observed
    n_observed = int(observed.sum())
    flagged = default_grid.overhead & observed
    n_flagged = int(flagged.sum())
    overhead_cells = {
        "observed_cells": n_observed,
        "flagged_cells": n_flagged,
        "fraction": (n_flagged / n_observed) if n_observed else None,
    }

    # --- item 3: false-flag rate -------------------------------------------------
    open_floor = rasterize_points_to_cells(traversable_pts, cell_m, origin_x, origin_y)
    gt_footprint: set[tuple[int, int]] = set()
    overhang_objs_for_gt = [
        o for o in scene_objects if object_overlaps_band(o, floor_z, overhead_min, overhead_max)
    ]
    for obj in overhang_objs_for_gt:
        gt_footprint |= rotated_rect_cells(
            obj.x, obj.y, obj.dx, obj.dy, obj.yaw, cell_m, origin_x, origin_y,
            inflate_cells=_FALSE_FLAG_INFLATE_CELLS,
        )

    flagged_cells = _grid_flagged_cells(default_grid, observed_only=True)
    false_flag_cells = (flagged_cells & open_floor) - gt_footprint

    examples = []
    if false_flag_cells:
        scored = []
        for (row, col) in false_flag_cells:
            wx, wy = default_grid.cell_to_world(row, col)
            best_name, best_dist = None, math.inf
            for obj in scene_objects:
                d = math.hypot(wx - obj.x, wy - obj.y)
                if d < best_dist:
                    best_dist, best_name = d, obj.name
            scored.append((best_dist, row, col, wx, wy, best_name))
        scored.sort(key=lambda t: t[0])
        for dist, row, col, wx, wy, name in scored[:_FALSE_FLAG_TOP_N]:
            examples.append(
                {
                    "row": row,
                    "col": col,
                    "x": wx,
                    "y": wy,
                    "nearest_object": name,
                    "nearest_object_dist_m": dist,
                }
            )

    false_flags = {
        "gt_overhang_object_count": len(overhang_objs_for_gt),
        "open_floor_cells": len(open_floor),
        "flagged_cells": len(flagged_cells),
        "false_flag_cells": len(false_flag_cells),
        "rate": (len(false_flag_cells) / len(flagged_cells)) if flagged_cells else None,
        "examples": examples,
    }

    # --- item 4: known-overhang hit rate -----------------------------------------
    overhangs = known_overhang_objects(scene_objects, floor_z, band_max=overhead_max)
    per_object = []
    total_obs = total_flag = 0
    for obj in overhangs:
        footprint = rotated_rect_cells(
            obj.x, obj.y, obj.dx, obj.dy, obj.yaw, cell_m, origin_x, origin_y, inflate_cells=0
        )
        h, w = default_grid.shape
        obs_cells = [
            (r, c) for (r, c) in footprint if 0 <= r < h and 0 <= c < w and observed[r, c]
        ]
        flag_cells = [rc for rc in obs_cells if default_grid.overhead[rc[0], rc[1]]]
        frac = (len(flag_cells) / len(obs_cells)) if obs_cells else None
        per_object.append(
            {
                "id": obj.id,
                "name": obj.name,
                "top_z": obj.top_z,
                "footprint_cells": len(footprint),
                "observed_cells": len(obs_cells),
                "flagged_cells": len(flag_cells),
                "hit_rate": frac,
            }
        )
        total_obs += len(obs_cells)
        total_flag += len(flag_cells)

    known_overhang_hit_rate = {
        "objects": per_object,
        "aggregate_observed_cells": total_obs,
        "aggregate_flagged_cells": total_flag,
        "aggregate_hit_rate": (total_flag / total_obs) if total_obs else None,
    }

    # --- item 5: sensitivity ------------------------------------------------------
    sensitivity = {}
    for name in _VARIANT_NAMES:
        g = replay.grids[name]
        n = int((g.overhead & g.observed).sum())
        sensitivity[name] = {"flagged_cells": n, "cfg": _cfg_to_dict(replay.cfgs[name])}
    base = sensitivity["default"]["flagged_cells"]
    for name in ("shifted", "minpts2"):
        sensitivity[name]["delta_vs_default"] = sensitivity[name]["flagged_cells"] - base

    return {
        "sensor_height": sensor_height,
        "overhead_cells": overhead_cells,
        "false_flags": false_flags,
        "known_overhang_hit_rate": known_overhang_hit_rate,
        "sensitivity": sensitivity,
    }


def _cfg_to_dict(cfg: OverheadConfig) -> dict:
    return {
        "overhead_min": cfg.overhead_min,
        "overhead_max": cfg.overhead_max,
        "min_points_per_cell": cfg.min_points_per_cell,
        "vehicle_sensor_height": cfg.vehicle_sensor_height,
    }


# --------------------------------------------------------------------------- report


def _summary_row(bag_name: str, scene_name: str, metrics: dict, replay: ReplayResult) -> str:
    sh = metrics["sensor_height"]
    oc = metrics["overhead_cells"]
    ff = metrics["false_flags"]
    ko = metrics["known_overhang_hit_rate"]
    sens = metrics["sensitivity"]
    degraded = " (degraded-rate)" if (replay.mean_scan_hz or 0) < 2.0 else ""

    def pct(x):
        return f"{x * 100:.1f}%" if x is not None else "n/a"

    return (
        f"| {bag_name}{degraded} | {scene_name} | "
        f"{sh['measured_sensor_height_m']:.2f} ({sh['delta_m']:+.2f}) | "
        f"{oc['flagged_cells']}/{oc['observed_cells']} ({pct(oc['fraction'])}) | "
        f"{ff['false_flag_cells']}/{ff['flagged_cells']} ({pct(ff['rate'])}) | "
        f"{pct(ko['aggregate_hit_rate'])} | "
        f"{sens['shifted']['delta_vs_default']:+d} / {sens['minpts2']['delta_vs_default']:+d} |"
    )


_SUMMARY_HEADER = (
    "| Bag | Scene | Sensor height m (delta) | Overhead cells (flagged/observed) | "
    "False flags (n/flagged) | Known-overhang hit rate | Sensitivity delta "
    "(shifted/minpts2) |\n"
    "|---|---|---|---|---|---|---|\n"
)


def _append_summary_row(summary_md: Path, bag_name: str, scene_name: str, metrics: dict, replay: ReplayResult) -> None:
    row = _summary_row(bag_name, scene_name, metrics, replay) + "\n"
    if not summary_md.exists():
        summary_md.write_text(
            "# Overhead-clearance multi-scene validation -- cross-scene table\n\n"
            + _SUMMARY_HEADER
            + row,
            encoding="utf-8",
        )
    else:
        with summary_md.open("a", encoding="utf-8") as fh:
            fh.write(row)


# --------------------------------------------------------------------------- CLI


def run_one(
    bag_dir,
    scene_dir,
    *,
    out_path,
    summary_md=None,
    mark_cadence_s: float = 1.0,
) -> dict:
    bag_dir = Path(bag_dir)
    scene_dir = Path(scene_dir)
    obj_list_path = scene_dir / "object_list.txt"
    ply_path = scene_dir / "traversable_area.ply"

    objects = parse_object_list(obj_list_path)
    traversable_pts = read_ply_xyz(ply_path)
    floor_z = floor_z_from_traversable(traversable_pts)

    t0 = time.perf_counter()
    replay = run_replay(bag_dir, floor_z=floor_z, mark_cadence_s=mark_cadence_s)
    dt = time.perf_counter() - t0

    metrics = compute_metrics(
        replay,
        scene_objects=objects,
        traversable_pts=traversable_pts,
        floor_z=floor_z,
    )

    result = {
        "bag": str(bag_dir),
        "scene": str(scene_dir),
        "runtime_s": dt,
        "replay": {
            "n_odom": replay.n_odom,
            "n_terrain": replay.n_terrain,
            "n_scan": replay.n_scan,
            "n_pose_marks": replay.n_pose_marks,
            "mark_cadence_s": mark_cadence_s,
            "bag_t_span_s": replay.bag_t_span,
            "mean_scan_hz": replay.mean_scan_hz,
            "grid_shape": list(replay.grids["default"].shape),
            "cell_m": replay.grids["default"].cell_m,
        },
        "n_scene_objects": len(objects),
        **metrics,
    }

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")

    if summary_md is not None:
        _append_summary_row(Path(summary_md), bag_dir.name, scene_dir.name, metrics, replay)

    return result


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tools.overhead_validation", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="replay one bag and validate against scene GT")
    r.add_argument("bag_dir", help="bag directory (contains *.mcap + metadata.yaml)")
    r.add_argument("scene_dir", help="scene directory (contains object_list.txt, traversable_area.ply)")
    r.add_argument("--out", required=True, help="output JSON path")
    r.add_argument("--summary-md", default=None, help="append a row to this summary markdown")
    r.add_argument("--mark-cadence", type=float, default=1.0, help="mark_pose cadence, s of bag time")
    return p


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    if args.cmd == "run":
        result = run_one(
            args.bag_dir,
            args.scene_dir,
            out_path=args.out,
            summary_md=args.summary_md,
            mark_cadence_s=args.mark_cadence,
        )
        sh = result["sensor_height"]
        oc = result["overhead_cells"]
        ff = result["false_flags"]
        ko = result["known_overhang_hit_rate"]
        print(
            f"wrote {args.out}\n"
            f"  runtime            : {result['runtime_s']:.1f} s\n"
            f"  odom/terrain/scan  : {result['replay']['n_odom']}/"
            f"{result['replay']['n_terrain']}/{result['replay']['n_scan']}\n"
            f"  sensor height      : measured {sh['measured_sensor_height_m']:.2f} m "
            f"(configured {sh['configured_vehicle_sensor_height_m']:.2f} m, "
            f"delta {sh['delta_m']:+.2f} m)\n"
            f"  overhead cells     : {oc['flagged_cells']}/{oc['observed_cells']}\n"
            f"  false-flag rate    : {ff['false_flag_cells']}/{ff['flagged_cells']}\n"
            f"  known-overhang hit : {ko['aggregate_hit_rate']}\n"
        )
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
