"""Incremental occupancy grid built from /terrain_map(_ext) XYZI clouds (map frame).

Units: metres. Cells are square (CELL_M). Classification uses the terrain
intensity channel (obstacle height above local ground, docs/upstream_notes.md
gotcha 9): intensity < FREE_MAX -> FREE, else OBSTACLE. Per cell we keep the
max intensity ever seen so an obstacle observation is never overwritten by a
later free reading in the same cell.

The grid grows lazily as patches arrive in a fixed world-frame lattice: a cell's
(row, col) is derived from floor(coord / CELL_M) offset by the grid origin, and
the backing array is reallocated (padded) whenever a patch falls outside current
bounds. Cell centres therefore stay on a stable world lattice across growth.

An `observed` mask marks cells within OBSERVE_RADIUS_M of any past vehicle pose
(the lidar footprint) — this is what frontier math treats as "seen", distinct
from FREE (a cell can be observed-but-obstacle or observed-but-free).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, replace

import numpy as np

from core.interfaces import LidarScan, TerrainPatch

# --------------------------------------------------------------------------- tunables
CELL_M: float = 0.10  # grid resolution, metres/cell
#: issue #90 -- snap epsilon added before every world->cell floor so that
#: coordinates sitting on (or one float32 ULP below) an exact cell boundary
#: floor consistently, instead of a stray-low bit flipping the result down by
#: one cell. `TerrainPatch.points` is float32 by contract (matches the real
#: ROS terrain message dtype); float32(n * CELL_M) rounds DOWN by one ULP for
#: many n (e.g. float32(0.7) == 0.699999988...), so an unguarded
#: `floor((x - origin) / cell_m)` collides two distinct lattice
#: rows/columns into one cell and leaves the other permanently UNKNOWN (see
#: docs/upstream_notes.md and the issue for the full repro). 1e-6 m is far
#: below both float32/float64 rounding noise at these coordinate magnitudes
#: and the smallest real-world coordinate difference we'd ever want to treat
#: as "a different cell", so it cannot merge genuinely distinct cells.
LATTICE_EPS_M: float = 1e-6
FREE_MAX: float = 0.15  # intensity < this -> FREE (matches TerrainPatch.FREE_MAX)
OBSERVE_RADIUS_M: float = 8.0  # lidar footprint radius for the observed mask
GROW_PAD_CELLS: int = 8  # extra ring of cells added when the grid must grow

#: issue #83 -- half-footprint of the mecanum platform: mark_pose carves every cell
#: within this radius of the vehicle FREE (a disc, not a single cell), so consecutive
#: tick poses at >= 0.1 m/tick spacing overlap into one connected trail instead of
#: disconnected singletons (the diagnosed root cause of the reachable-pocket collapse
#: -- see docs/upstream_notes.md and reports/issue83_live_captures/). Deliberately a
#: SEPARATE constant from core.nav.costmap.VEHICLE_RADIUS_M (0.4 m): that one is an
#: obstacle-INFLATION radius ("half footprint + margin" per its own docstring), tuned
#: for planning safety margin, not the vehicle's true physical half-footprint used
#: here for occupancy carving; keeping them distinct also avoids occupancy.py
#: importing costmap.py (costmap.py already imports occupancy.py -- a reverse import
#: would be circular).
VEHICLE_FOOTPRINT_RADIUS_M: float = 0.25

#: issue #36 -- min terrain patches (each contributing a vehicle_z sample) before
#: the runtime ground-offset estimator is trusted; kept small so warm-up is fast.
GROUND_OFFSET_WARMUP_PATCHES: int = 5

#: env override: pins the fallback vehicle-height-above-ground to a fixed value,
#: bypassing both the constant default and the runtime estimator (e.g. a rig
#: whose exact mount height is known ahead of time). Unset by default.
VEHICLE_SENSOR_HEIGHT_ENV_VAR: str = "VLA_OVERHEAD_VEHICLE_SENSOR_HEIGHT_M"


def _lattice_floor(coord, origin: float, cell_m: float):
    """floor((coord - origin) / cell_m), snapped by LATTICE_EPS_M (issue #90).

    ``coord`` may be a Python float, a numpy scalar, or an ndarray; ``np.floor``
    dispatches correctly on all three. Always computed in float64 (callers are
    responsible for widening float32 inputs first) -- widening alone does not
    fix the defect since the rounding already happened in float32, so this
    also adds a small epsilon before flooring to snap coordinates that are
    exactly on (or one float32 ULP below) a cell boundary to the intended
    cell instead of the one below it.
    """
    return np.floor((coord - origin) / cell_m + LATTICE_EPS_M)


def _env_vehicle_sensor_height_override() -> float | None:
    raw = os.environ.get(VEHICLE_SENSOR_HEIGHT_ENV_VAR)
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None

# --------------------------------------------------------------------------- states
UNKNOWN: int = 0
FREE: int = 1
OBSTACLE: int = 2


# --------------------------------------------------------------------------- overhead cfg
@dataclass(frozen=True)
class OverheadConfig:
    """Tunables for the overhead-clearance layer (see docs/calibration.md).

    Motivation: terrainAnalysis.cpp filters /registered_scan to a thin slab
    (maxRelZ=0.2 m above the vehicle) before publishing /terrain_map, so
    overhanging surfaces — bar-height tabletops, shelves, stool seats roughly
    0.25-1.2 m off the floor — are ABSENT from the terrain cloud: cells directly
    under a table read FREE. The full /registered_scan DOES carry those overhang
    points. ``integrate_scan_overhead`` re-reads the raw scan and flags any cell
    whose scan points fall in the clearance band above the local ground as
    OVERHEAD, so the planner can refuse to route the vehicle under furniture the
    base terrain stack cannot see.
    """

    overhead_min: float = 0.25  # m above local ground: band lower edge (skip near-ground)
    overhead_max: float = 1.20  # m above local ground: band upper edge (skip tall walls/ceiling)
    min_points_per_cell: int = 3  # cell needs >= this many in-band points to flag (noise reject)
    #: fallback local-ground estimate when a cell has no terrain-derived ground z
    #: AND the runtime estimator (below) has not warmed up yet:
    #: ground_z ~= vehicle_z - vehicle_sensor_height. The registered-scan / terrain
    #: clouds are map-frame, with the vehicle sensor origin ~0.6 m above the floor
    #: (jingfan fixtures: vehicle z ~= 0.0, ground/free terrain z ~= -0.6). This is
    #: the jingfan real-robot rig height; the sim sensor mount sits ~0.15-0.18 m
    #: higher (issue #36) -- ``use_ground_offset_estimator`` corrects for that
    #: automatically once warmed, so this constant only matters pre-warm-up.
    vehicle_sensor_height: float = 0.60
    #: when True (default), once >= GROUND_OFFSET_WARMUP_PATCHES terrain patches
    #: have contributed a vehicle_z sample, the per-grid runtime ground-offset
    #: estimate (``OccupancyGrid.ground_offset_estimate``) is used for the
    #: fallback instead of the constant above -- see issue #36. Set False to pin
    #: the fallback to ``vehicle_sensor_height`` unconditionally (e.g. to isolate
    #: the constant-only hypothesis in a validation run).
    use_ground_offset_estimator: bool = True


DEFAULT_OVERHEAD_CONFIG = OverheadConfig()

#: max scan points fed to the overhead layer per tick (decimation for the per-tick
#: wiring path — the raw /registered_scan is ~10-60k points; deterministic stride).
OVERHEAD_SCAN_MAX_PTS: int = 12000


@dataclass
class OccupancyGrid:
    """Growable world-frame occupancy grid.

    `state[r, c]` in {UNKNOWN, FREE, OBSTACLE}; `intensity[r, c]` = max terrain
    intensity seen in the cell; `observed[r, c]` = within lidar radius of a pose.
    `origin_x/origin_y` are the world coords of the lattice point at cell (0, 0)'s
    lower corner; cell (r, c) covers [origin + c*CELL, origin + (c+1)*CELL] in x
    and [origin_y + r*CELL, ...] in y.
    """

    cell_m: float = CELL_M
    free_max: float = FREE_MAX
    observe_radius_m: float = OBSERVE_RADIUS_M
    origin_x: float = 0.0
    origin_y: float = 0.0
    state: np.ndarray | None = None
    intensity: np.ndarray | None = None
    observed: np.ndarray | None = None
    #: OVERHEAD flag — a cell has a scan return in the clearance band above local
    #: ground (furniture the terrain slab filtered out). Kept SEPARATE from `state`
    #: so terrain-derived FREE/OBSTACLE classification stays pure (spec §1).
    overhead: np.ndarray | None = None
    #: per-cell terrain-derived ground surface z (map frame), max-lag from the
    #: lowest terrain point seen; NaN until a terrain point lands in the cell.
    ground_z: np.ndarray | None = None
    #: issue #36 runtime ground-offset estimator state: vehicle_z observed at
    #: each terrain-patch integrate call that also supplied a vehicle_z, and the
    #: running min terrain point z seen so far (proxy for the scene floor).
    #: ``ground_offset_estimate`` derives median(vehicle_z) - min ground z from
    #: these once warmed up. Deliberately plain Python (not grid arrays): O(one
    #: scalar/append) per patch, not per cell.
    _gz_offset_vehicle_zs: list[float] = field(default_factory=list, repr=False, compare=False)
    _gz_offset_min_ground_z: float = field(default=float("inf"), repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.state is None:
            # Start empty; first patch/pose seeds real bounds.
            self.state = np.zeros((1, 1), dtype=np.int8)
            self.intensity = np.full((1, 1), -np.inf, dtype=np.float32)
            self.observed = np.zeros((1, 1), dtype=bool)
            self.overhead = np.zeros((1, 1), dtype=bool)
            self.ground_z = np.full((1, 1), np.nan, dtype=np.float32)

    # ------------------------------------------------------------- indexing
    @property
    def shape(self) -> tuple[int, int]:
        return self.state.shape  # type: ignore[union-attr]

    def world_to_cell(self, x: float, y: float) -> tuple[int, int]:
        """(x, y) metres -> (row, col) integer index (may be out of current bounds)."""
        col = int(_lattice_floor(x, self.origin_x, self.cell_m))
        row = int(_lattice_floor(y, self.origin_y, self.cell_m))
        return row, col

    def cell_to_world(self, row: int, col: int) -> tuple[float, float]:
        """(row, col) -> world (x, y) metres at the cell CENTRE."""
        x = self.origin_x + (col + 0.5) * self.cell_m
        y = self.origin_y + (row + 0.5) * self.cell_m
        return x, y

    def in_bounds(self, row: int, col: int) -> bool:
        h, w = self.shape
        return 0 <= row < h and 0 <= col < w

    # ------------------------------------------------------------- growth
    def _ensure_bounds(self, rows: np.ndarray, cols: np.ndarray) -> None:
        """Grow the backing arrays so every (row, col) in the given index arrays fits.

        Because cell (0,0) is anchored to (origin_x, origin_y), a negative index
        means we must shift the origin down/left and pad the top/left of the arrays.
        """
        if rows.size == 0:
            return
        h, w = self.shape
        min_r, max_r = int(rows.min()), int(rows.max())
        min_c, max_c = int(cols.min()), int(cols.max())

        pad_top = max(0, -min_r + (GROW_PAD_CELLS if min_r < 0 else 0))
        pad_left = max(0, -min_c + (GROW_PAD_CELLS if min_c < 0 else 0))
        pad_bottom = max(0, (max_r - (h - 1)) + (GROW_PAD_CELLS if max_r >= h else 0))
        pad_right = max(0, (max_c - (w - 1)) + (GROW_PAD_CELLS if max_c >= w else 0))

        if not (pad_top or pad_left or pad_bottom or pad_right):
            return

        new_h = h + pad_top + pad_bottom
        new_w = w + pad_left + pad_right
        new_state = np.zeros((new_h, new_w), dtype=np.int8)
        new_int = np.full((new_h, new_w), -np.inf, dtype=np.float32)
        new_obs = np.zeros((new_h, new_w), dtype=bool)
        new_over = np.zeros((new_h, new_w), dtype=bool)
        new_gz = np.full((new_h, new_w), np.nan, dtype=np.float32)

        new_state[pad_top : pad_top + h, pad_left : pad_left + w] = self.state
        new_int[pad_top : pad_top + h, pad_left : pad_left + w] = self.intensity
        new_obs[pad_top : pad_top + h, pad_left : pad_left + w] = self.observed
        new_over[pad_top : pad_top + h, pad_left : pad_left + w] = self.overhead
        new_gz[pad_top : pad_top + h, pad_left : pad_left + w] = self.ground_z

        self.state = new_state
        self.intensity = new_int
        self.observed = new_obs
        self.overhead = new_over
        self.ground_z = new_gz
        # Shifting the top/left corner moves the origin (cell (0,0)) outward.
        self.origin_x -= pad_left * self.cell_m
        self.origin_y -= pad_top * self.cell_m

    # ------------------------------------------------------------- ingest
    def integrate_patch(self, patch: TerrainPatch, vehicle_z: float | None = None) -> None:
        """Fold one terrain cloud (N,4 XYZI) into the grid, keeping max intensity/cell.

        ``vehicle_z`` (optional): the vehicle's map-frame z at the time this patch
        was captured. When supplied, it feeds the runtime ground-offset estimator
        (issue #36) used by ``integrate_scan_overhead``'s fallback -- callers that
        have odom available alongside the patch should pass it; omitting it just
        means this patch doesn't contribute a warm-up sample (estimator stays
        unwarmed longer, falls back to ``OverheadConfig.vehicle_sensor_height``).
        """
        pts = np.asarray(patch.points, dtype=np.float64)  # float64 keeps cell
        # indexing consistent with world_to_cell (Python float); float32 rounds
        # differently at exact cell boundaries.
        if pts.size == 0:
            return
        xs, ys, zs, inten = pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3]
        if vehicle_z is not None:
            self._gz_offset_vehicle_zs.append(float(vehicle_z))
            self._gz_offset_min_ground_z = min(self._gz_offset_min_ground_z, float(zs.min()))
        cols = _lattice_floor(xs, self.origin_x, self.cell_m).astype(np.int64)
        rows = _lattice_floor(ys, self.origin_y, self.cell_m).astype(np.int64)
        self._ensure_bounds(rows, cols)
        # Recompute indices post-growth (origin may have shifted).
        cols = _lattice_floor(xs, self.origin_x, self.cell_m).astype(np.int64)
        rows = _lattice_floor(ys, self.origin_y, self.cell_m).astype(np.int64)

        # Reduce to max intensity per unique cell before writing (order-independent).
        flat = rows * self.shape[1] + cols
        order = np.argsort(flat, kind="stable")
        flat_s, inten_s, z_s = flat[order], inten[order], zs[order]
        uniq, starts = np.unique(flat_s, return_index=True)
        seg_max = np.maximum.reduceat(inten_s, starts)
        # Local ground surface z per cell = the LOWEST terrain point in the cell
        # (min z), so the overhead band is measured off the floor, not off a
        # low obstacle. Kept as a running min across patches.
        seg_gz = np.minimum.reduceat(z_s, starts)
        u_rows = (uniq // self.shape[1]).astype(np.int64)
        u_cols = (uniq % self.shape[1]).astype(np.int64)

        prev = self.intensity[u_rows, u_cols]
        new_int = np.maximum(prev, seg_max)
        self.intensity[u_rows, u_cols] = new_int
        classified = np.where(new_int < self.free_max, FREE, OBSTACLE).astype(np.int8)
        self.state[u_rows, u_cols] = classified

        prev_gz = self.ground_z[u_rows, u_cols]
        merged_gz = np.where(np.isnan(prev_gz), seg_gz, np.minimum(prev_gz, seg_gz))
        self.ground_z[u_rows, u_cols] = merged_gz.astype(np.float32)

    @property
    def ground_offset_estimate(self) -> float | None:
        """Runtime vehicle-height-above-ground estimate (issue #36), or None pre-warm-up.

        ``median(vehicle_z sampled at each contributing integrate_patch call) -
        min(terrain point z seen so far)`` -- a robust (median-based), deterministic,
        O(1)-per-patch running estimate. Reproduces ~0.60 m on a jingfan-like rig
        (vehicle z~=0, floor~=-0.6) and ~0.75-0.78 m on the sim bags (see
        reports/overhead_validation_2026-07-18/summary.md finding 1) without any
        scene-specific tuning. None until >= GROUND_OFFSET_WARMUP_PATCHES patches
        have contributed a vehicle_z sample -- callers fall back to the configured
        constant until then.
        """
        if len(self._gz_offset_vehicle_zs) < GROUND_OFFSET_WARMUP_PATCHES:
            return None
        return float(np.median(self._gz_offset_vehicle_zs)) - self._gz_offset_min_ground_z

    def _fallback_ground_z(self, vehicle_z: float, cfg: OverheadConfig) -> float:
        """Local-ground fallback for cells with no terrain-derived ``ground_z`` yet.

        Precedence: env override (if set) > runtime estimator (if warmed and
        enabled) > the configured constant. See issue #36.
        """
        override = _env_vehicle_sensor_height_override()
        if override is not None:
            return float(vehicle_z) - override
        if cfg.use_ground_offset_estimator:
            estimate = self.ground_offset_estimate
            if estimate is not None:
                return float(vehicle_z) - estimate
        return float(vehicle_z) - float(cfg.vehicle_sensor_height)

    def integrate_scan_overhead(
        self,
        scan: LidarScan,
        vehicle_z: float,
        cfg: OverheadConfig = DEFAULT_OVERHEAD_CONFIG,
    ) -> None:
        """Flag cells with an overhang the terrain slab can't see (spec §1, §5 gate).

        For each scan point (map-frame xyz from /registered_scan) whose height above
        the LOCAL GROUND estimate falls in the band [cfg.overhead_min, cfg.overhead_max],
        we accumulate a per-cell count; a cell with >= cfg.min_points_per_cell such
        points is set OVERHEAD. This is a separate boolean layer — it never touches
        the terrain-derived FREE/OBSTACLE `state`.

        Local ground per cell: the terrain-derived `ground_z` if a terrain point has
        landed in the cell; otherwise `_fallback_ground_z` (vehicle_z minus the
        warmed-up runtime ground-offset estimate, or `vehicle_sensor_height` before
        warm-up -- see issue #36). Overhead points over UNKNOWN floor (no terrain
        yet) therefore still flag — the table is caught before the floor beneath it
        is ever classified.
        """
        pts = np.asarray(scan.points, dtype=np.float64)
        if pts.size == 0:
            return
        xs, ys, zs = pts[:, 0], pts[:, 1], pts[:, 2]
        cols = _lattice_floor(xs, self.origin_x, self.cell_m).astype(np.int64)
        rows = _lattice_floor(ys, self.origin_y, self.cell_m).astype(np.int64)
        self._ensure_bounds(rows, cols)
        cols = _lattice_floor(xs, self.origin_x, self.cell_m).astype(np.int64)
        rows = _lattice_floor(ys, self.origin_y, self.cell_m).astype(np.int64)

        # Per-point local ground: terrain ground_z where known, else the fallback
        # (env override > runtime estimator once warmed > configured constant).
        fallback_gz = self._fallback_ground_z(vehicle_z, cfg)
        cell_gz = self.ground_z[rows, cols]
        gz = np.where(np.isnan(cell_gz), fallback_gz, cell_gz)
        height = zs - gz
        in_band = (height >= cfg.overhead_min) & (height <= cfg.overhead_max)
        if not in_band.any():
            return

        br, bc = rows[in_band], cols[in_band]
        flat = br * self.shape[1] + bc
        uniq, counts = np.unique(flat, return_counts=True)
        keep = counts >= cfg.min_points_per_cell
        if not keep.any():
            return
        u = uniq[keep]
        u_rows = (u // self.shape[1]).astype(np.int64)
        u_cols = (u % self.shape[1]).astype(np.int64)
        self.overhead[u_rows, u_cols] = True

    def mark_pose(self, x: float, y: float) -> None:
        """Carve a vehicle-footprint disc FREE and flag all cells within lidar radius observed.

        Issue #83: the vehicle physically occupies a disc of radius
        ``VEHICLE_FOOTPRINT_RADIUS_M`` around (x, y), not a single point, so every
        cell whose centre lies within that disc is traversable by construction and
        is carved FREE (intensity clamped <= 0, same as the previous single-cell
        carve). At >= 0.1 m/tick pose spacing a single-cell-per-tick carve leaves a
        trail of disconnected singletons (BFS-reachable pocket stuck at 1 cell,
        the live-capture root cause); overlapping footprint discs from consecutive
        ticks connect into one contiguous trail instead.
        """
        # Ensure the observe disc (>= footprint disc: OBSERVE_RADIUS_M >
        # VEHICLE_FOOTPRINT_RADIUS_M) fits in the grid.
        r_cells = int(np.ceil(self.observe_radius_m / self.cell_m))
        cr, cc = self.world_to_cell(x, y)
        corner_rows = np.array([cr - r_cells, cr + r_cells], dtype=np.int64)
        corner_cols = np.array([cc - r_cells, cc + r_cells], dtype=np.int64)
        self._ensure_bounds(corner_rows, corner_cols)
        cr, cc = self.world_to_cell(x, y)

        h, w = self.shape
        rr = np.arange(max(0, cr - r_cells), min(h, cr + r_cells + 1))
        ccg = np.arange(max(0, cc - r_cells), min(w, cc + r_cells + 1))
        if rr.size == 0 or ccg.size == 0:
            return
        gr, gc = np.meshgrid(rr, ccg, indexing="ij")
        wx = self.origin_x + (gc + 0.5) * self.cell_m
        wy = self.origin_y + (gr + 0.5) * self.cell_m
        d2 = (wx - x) ** 2 + (wy - y) ** 2
        within = d2 <= self.observe_radius_m**2
        self.observed[gr[within], gc[within]] = True

        # Vehicle footprint is always traversable: carve every cell within the
        # footprint disc FREE (not just the centre cell -- see docstring above).
        carve = d2 <= VEHICLE_FOOTPRINT_RADIUS_M**2
        carve_rows, carve_cols = gr[carve], gc[carve]
        if carve_rows.size:
            self.state[carve_rows, carve_cols] = FREE
            # Keep intensity coherent (below free cutoff) so re-classification stays FREE.
            self.intensity[carve_rows, carve_cols] = np.minimum(
                self.intensity[carve_rows, carve_cols], 0.0
            )

    # ------------------------------------------------------------- queries
    def is_free(self, row: int, col: int) -> bool:
        return self.in_bounds(row, col) and self.state[row, col] == FREE

    def is_obstacle(self, row: int, col: int) -> bool:
        return self.in_bounds(row, col) and self.state[row, col] == OBSTACLE

    def is_unknown(self, row: int, col: int) -> bool:
        return (not self.in_bounds(row, col)) or self.state[row, col] == UNKNOWN

    def is_overhead(self, row: int, col: int) -> bool:
        """True if the cell carries an overhang in the clearance band (furniture)."""
        return self.in_bounds(row, col) and bool(self.overhead[row, col])


def integrate_scan_overhead_decimated(
    grid: OccupancyGrid,
    scan: LidarScan,
    vehicle_z: float,
    cfg: OverheadConfig = DEFAULT_OVERHEAD_CONFIG,
    max_pts: int = OVERHEAD_SCAN_MAX_PTS,
) -> None:
    """Per-tick wiring helper: decimate + bounds-clip the scan, then fold overhead.

    Keeps the per-tick cost bounded (spec §3): a deterministic stride decimation to
    ``max_pts`` and a pre-clip to points already within the grid's current world
    bounds (a table well outside the mapped area is skipped — the grid should not
    grow just to flag a far overhang). Points are only clipped, never reordered, so
    the result stays deterministic.
    """
    pts = np.asarray(scan.points, dtype=np.float64)
    if pts.size == 0:
        return
    # Clip to current grid world bounds so we neither grow for far points nor pay
    # for them. Bounds are the outer edges of the current lattice.
    h, w = grid.shape
    min_x = grid.origin_x
    min_y = grid.origin_y
    max_x = grid.origin_x + w * grid.cell_m
    max_y = grid.origin_y + h * grid.cell_m
    xs, ys = pts[:, 0], pts[:, 1]
    inside = (xs >= min_x) & (xs < max_x) & (ys >= min_y) & (ys < max_y)
    pts = pts[inside]
    if pts.shape[0] == 0:
        return
    stride = 1
    if pts.shape[0] > max_pts:
        stride = int(np.ceil(pts.shape[0] / max_pts))
        pts = pts[::stride]
    # Stride-consistent noise gate (redteam H13 / SYS-F12 decimation-vs-min_points):
    # the ``min_points_per_cell`` gate counts points AFTER this stride decimation, so
    # a sparse-but-real overhang edge that just clears the gate at full density (say
    # exactly 3 points in a cell) drops below it after a ~5x stride (≈0-1 survive) and
    # silently stops flagging. Scale the effective threshold DOWN by the applied
    # stride — ``ceil(min_points / stride)`` with a floor of 1 — so the decimated
    # count is judged in stride-corrected units and the same physical edge flags at
    # any stride. Chosen over "count in stride-corrected units" (multiplying survivor
    # counts back up) because scaling the threshold keeps ``integrate_scan_overhead``
    # counting real, deterministic survivors — no reconstructed fractional counts —
    # and floor-1 preserves the "at least one real return" noise floor.
    if stride > 1 and cfg.min_points_per_cell > 1:
        eff_min = max(1, int(np.ceil(cfg.min_points_per_cell / stride)))
        if eff_min != cfg.min_points_per_cell:
            cfg = replace(cfg, min_points_per_cell=eff_min)
    grid.integrate_scan_overhead(LidarScan(t=scan.t, points=pts), vehicle_z, cfg)
