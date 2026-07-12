"""Incremental colored voxel map — the live twin of the offline colored-cloud tool.

``tools/colored_cloud.py`` accumulates a *recorded* session in one batch pass and
writes a PLY. :class:`ColoredVoxelMap` does the same voxel-mean reconstruction
**incrementally**, one ``(scan, pano)`` pair at a time, so the ROS adapter can grow a
colored map per tick and republish it to RViz while the robot drives (T6). It reuses
the shared forward projection (:mod:`core.perception.pano_projection`) so the online
and offline color/geometry are the same math — pinned by the equivalence test in
``src/tests/perception/test_colored_map.py``.

Storage is four parallel arrays sorted by packed-int64 voxel key (keys, position-sum,
color-sum, count) — no dict. Each :meth:`ingest` vectorizes its own scan with
``np.unique`` + ``np.add.at`` (no per-point Python loop), then merges the per-scan
partials into the persistent store with ``np.searchsorted`` for already-seen voxels
and a single concatenate + argsort for newly-seen ones (no per-voxel Python loop
either). :meth:`to_arrays` divides the running sums to per-voxel mean position
(float32) and mean color (uint8) — matching Open3D / PCL ``voxel_down_sample`` and the
offline tool exactly.

Voxel-key packing (correctness traps, mirrors ``tools/voxel.py``):

* A per-map ``origin`` is fixed once at the first ingest as ``first_scan.min - MARGIN``
  so later scans with smaller coordinates never produce a negative voxel index that
  would underflow the packed key. ``MARGIN`` (200 m) is generous: 21 bits/axis at
  0.05 m spans ~100 km, far beyond any indoor scene, so the offset costs nothing.
* Voxel indices are packed into a signed int64 as ``(ix << 42) | (iy << 21) | iz``
  with 21 bits/axis (``3 * 21 = 63 < 64``).

Gray policy matches the offline tool: out-of-VFOV / range-gated points still
contribute their geometry, tinted :data:`~core.perception.pano_projection.GRAY`,
unless ``keep_uncolored`` is false.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.interfaces import LidarScan, PanoFrame
from core.perception.pano_projection import (
    DEFAULT_MIN_RANGE,
    project_points_to_pano,
    sample_colors,
)

#: Bits per axis in the packed int64 voxel key (3 * 21 = 63 < 64). Mirrors tools/voxel.py.
_BITS_PER_AXIS = 21
_AXIS_MAX = 1 << _BITS_PER_AXIS

#: Metres subtracted from the first scan's min corner to fix a non-negative origin, so
#: later scans with smaller coordinates never underflow the packed voxel key.
DEFAULT_ORIGIN_MARGIN_M: float = 200.0

#: Initial capacity of the growing per-voxel arrays (amortized doubling on overflow).
_INITIAL_CAPACITY = 1 << 14  # 16384 voxels


@dataclass(frozen=True)
class ColoredMapConfig:
    """Tunables for :class:`ColoredVoxelMap` (module-level defaults)."""

    voxel_m: float = 0.05
    min_range: float = DEFAULT_MIN_RANGE
    keep_uncolored: bool = True  # gray out-of-VFOV/range points instead of dropping them
    origin_margin_m: float = DEFAULT_ORIGIN_MARGIN_M


@dataclass(frozen=True)
class IngestStats:
    """Summary of one :meth:`ColoredVoxelMap.ingest` call."""

    points_in: int  # points that survived the (optional) uncolored drop and were binned
    points_colored: int  # of those, how many landed on the panorama
    points_gray: int  # of those, how many got the gray fill


class ColoredVoxelMap:
    """Incremental per-voxel mean colored map (running position/color means).

    Feed it ``(scan, pano)`` pairs with :meth:`ingest`; read the current cloud with
    :meth:`to_arrays`. ``pano.odom`` supplies the apex/yaw for the forward projection
    (same as the offline tool). The map is monotonic — voxels are only added or their
    means refined; nothing is ever evicted.
    """

    def __init__(
        self,
        config: ColoredMapConfig | None = None,
        *,
        origin: np.ndarray | None = None,
    ) -> None:
        self.config = config or ColoredMapConfig()
        # An explicit origin pins the voxel grid (used by the equivalence test to align
        # with the offline tool's global-min grid); otherwise it is fixed at first ingest.
        self._origin: np.ndarray | None = (
            None if origin is None else np.asarray(origin, dtype=np.float64).reshape(3)
        )
        # Persistent store kept as arrays sorted by packed voxel key (no dict in the
        # hot path): self._keys[:self._n] is strictly increasing and parallel to
        # self._pos_sum / self._col_sum / self._count.
        cap = _INITIAL_CAPACITY
        self._keys = np.zeros(cap, dtype=np.int64)
        self._pos_sum = np.zeros((cap, 3), dtype=np.float64)
        self._col_sum = np.zeros((cap, 3), dtype=np.float64)
        self._count = np.zeros(cap, dtype=np.int64)
        self._n = 0  # number of occupied voxels (rows in use)

    # ------------------------------------------------------------------ properties

    @property
    def n_voxels(self) -> int:
        """Number of occupied voxels currently in the map."""
        return self._n

    # ------------------------------------------------------------------ ingest

    def ingest(self, scan: LidarScan, pano: PanoFrame) -> IngestStats:
        """Accumulate one scan, colored by ``pano``, into the running voxel means.

        Points are forward-projected onto ``pano`` via ``pano.odom`` (apex/yaw) and
        sampled nearest-neighbour; out-of-VFOV / range-gated points keep their geometry
        tinted gray unless ``config.keep_uncolored`` is false. Returns per-call stats.
        """
        cfg = self.config
        pts = np.asarray(scan.points, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] != 3 or len(pts) == 0:
            return IngestStats(0, 0, 0)

        rows, cols, valid = project_points_to_pano(
            pts, pano.odom, min_range=cfg.min_range
        )
        colors = sample_colors(pano.image, rows, cols, valid)

        if not cfg.keep_uncolored:
            pts = pts[valid]
            colors = colors[valid]
            valid = valid[valid]
        if len(pts) == 0:
            return IngestStats(0, 0, 0)

        n_colored = int(valid.sum())
        self._binned_merge(pts, colors)
        return IngestStats(
            points_in=len(pts),
            points_colored=n_colored,
            points_gray=len(pts) - n_colored,
        )

    # ------------------------------------------------------------------ read-out

    def to_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(xyz float32 (V, 3), rgb uint8 (V, 3))`` per-voxel mean cloud.

        ``xyz`` is the mean position and ``rgb`` the mean color of every occupied
        voxel (matching Open3D / PCL ``voxel_down_sample`` and the offline tool). The
        row order is the voxel-insertion order; it is not otherwise sorted.
        """
        if self._n == 0:
            return (
                np.zeros((0, 3), dtype=np.float32),
                np.zeros((0, 3), dtype=np.uint8),
            )
        counts = self._count[: self._n].astype(np.float64)[:, None]
        xyz = (self._pos_sum[: self._n] / counts).astype(np.float32)
        rgb = np.clip(np.round(self._col_sum[: self._n] / counts), 0, 255).astype(np.uint8)
        return xyz, rgb

    # ------------------------------------------------------------------ internals

    def _ensure_origin(self, pts: np.ndarray) -> None:
        if self._origin is None:
            self._origin = pts.astype(np.float64).min(axis=0) - self.config.origin_margin_m

    def _keys_for(self, pts: np.ndarray) -> np.ndarray:
        """Packed int64 voxel keys for ``pts`` against the fixed origin grid."""
        idx = np.floor((pts.astype(np.float64) - self._origin[None, :]) / self.config.voxel_m)
        idx = idx.astype(np.int64)
        if idx.min() < 0 or idx.max() >= _AXIS_MAX:
            raise ValueError(
                f"voxel index out of the {_AXIS_MAX}-cell packing range on an axis; "
                "point fell outside the origin margin (increase origin_margin_m or voxel_m)"
            )
        return (
            (idx[:, 0] << (2 * _BITS_PER_AXIS))
            | (idx[:, 1] << _BITS_PER_AXIS)
            | idx[:, 2]
        )

    def _grow_to(self, needed: int) -> None:
        cap = len(self._count)
        if needed <= cap:
            return
        while cap < needed:
            cap *= 2
        new_keys = np.zeros(cap, dtype=np.int64)
        new_keys[: self._n] = self._keys[: self._n]
        self._keys = new_keys
        self._pos_sum = np.resize(self._pos_sum, (cap, 3))
        self._pos_sum[self._n :] = 0.0
        self._col_sum = np.resize(self._col_sum, (cap, 3))
        self._col_sum[self._n :] = 0.0
        new_count = np.zeros(cap, dtype=np.int64)
        new_count[: self._n] = self._count[: self._n]
        self._count = new_count

    def _binned_merge(self, pts: np.ndarray, colors: np.ndarray) -> None:
        """Vectorized per-scan reduce (np.add.at on unique voxels), then vectorized merge
        into the sorted-key store (no per-voxel Python loop)."""
        self._ensure_origin(pts)
        keys = self._keys_for(pts)

        # Reduce this scan to its unique voxels first (no per-point Python loop): inverse
        # indices scatter-add positions/colors/counts into per-unique-voxel accumulators.
        new_keys, inverse = np.unique(keys, return_inverse=True)
        m = len(new_keys)
        pos_part = np.zeros((m, 3), dtype=np.float64)
        col_part = np.zeros((m, 3), dtype=np.float64)
        cnt_part = np.zeros(m, dtype=np.int64)
        np.add.at(pos_part, inverse, pts.astype(np.float64))
        np.add.at(col_part, inverse, colors.astype(np.float64))
        np.add.at(cnt_part, inverse, 1)

        # Vectorized merge into the persistent sorted-key store.
        store_keys = self._keys[: self._n]
        if self._n > 0:
            idx = np.searchsorted(store_keys, new_keys)
            clipped = np.minimum(idx, self._n - 1)
            found = (idx < self._n) & (store_keys[clipped] == new_keys)
        else:
            idx = np.zeros(m, dtype=np.intp)
            found = np.zeros(m, dtype=bool)

        if np.any(found):
            rows = idx[found]
            # new_keys are unique per ingest, so fancy-index += is correct here (no
            # repeated row indices within this update).
            self._pos_sum[rows] += pos_part[found]
            self._col_sum[rows] += col_part[found]
            self._count[rows] += cnt_part[found]

        missing = ~found
        n_missing = int(np.count_nonzero(missing))
        if n_missing:
            old_n = self._n
            self._grow_to(old_n + n_missing)
            # Concatenate the new (never-seen) voxels onto the store, then re-sort the
            # whole store by key once (cheap: V is ~1e5, argsort is milliseconds).
            all_keys = np.concatenate([self._keys[:old_n], new_keys[missing]])
            all_pos = np.concatenate([self._pos_sum[:old_n], pos_part[missing]], axis=0)
            all_col = np.concatenate([self._col_sum[:old_n], col_part[missing]], axis=0)
            all_cnt = np.concatenate([self._count[:old_n], cnt_part[missing]])

            order = np.argsort(all_keys)
            new_n = old_n + n_missing
            self._keys[:new_n] = all_keys[order]
            self._pos_sum[:new_n] = all_pos[order]
            self._col_sum[:new_n] = all_col[order]
            self._count[:new_n] = all_cnt[order]
            self._n = new_n
