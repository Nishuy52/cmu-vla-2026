"""Lidar fusion: lift a 2D :class:`~core.perception.detector.Detection` to a 3D
centroid + point set using the registered lidar scan.

Approach (pure numpy, deterministic):

1. **Frustum from the bbox.** The detection's four bbox corners are cast through the
   tiling inverse (:func:`core.perception.tiling.tile_pixel_to_map_ray`) to map-frame
   ``(bearing, elevation)`` rays. Their angular span (handling the ±pi azimuth wrap)
   defines the detection's viewing frustum, padded slightly.
2. **Angular gate.** Each lidar point (map frame) is converted to a bearing/elevation
   *from the vehicle position* (the camera's optical centre, approximated by the odom
   xyz). Points whose bearing/elevation fall inside the padded frustum are candidates.
3. **Depth clustering.** Candidate points are binned by ray-distance; the nearest
   dense cluster (1D histogram / simple range-DBSCAN) is taken as the object surface —
   this rejects background points that share the frustum but sit far behind.
4. **Accept / reject.** If the winning cluster has < ``min_points`` (default 5) points,
   the detection is rejected (``None``). Otherwise return the cluster centroid and the
   cluster's point set.

The vehicle position is the frustum apex; lidar arrives already in the ``map`` frame
(``/registered_scan``), so we only need odom to place the apex and (via the ray helper)
to orient bearings.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.interfaces import LidarScan, OdomState
from core.perception.detector import Detection
from core.perception.tiling import (
    DEFAULT_N_TILES,
    DEFAULT_TILE_HFOV,
    DEFAULT_TILE_VFOV,
    tile_pixel_to_map_ray,
    wrap_pi,
)


@dataclass(frozen=True)
class FusionConfig:
    """Tunable fusion constants. Non-spec values flagged in the task report."""

    min_points: int = 5           # reject a detection below this many cluster points
    angular_pad: float = np.deg2rad(3.0)  # frustum padding beyond the bbox rays
    depth_bin: float = 0.25       # m; range-histogram bin width for clustering
    max_range: float = 15.0       # m; ignore points beyond this (lidar/terrain scale)
    cluster_gap: float = 0.5      # m; a range gap wider than this splits clusters


DEFAULT_FUSION_CONFIG = FusionConfig()


@dataclass(frozen=True)
class Fused3D:
    """Result of fusing one detection with lidar: a 3D centroid + its point set."""

    centroid: np.ndarray       # (3,) map-frame
    points: np.ndarray         # (M, 3) map-frame cluster points
    n_points: int
    range_m: float             # centroid distance from the vehicle apex


def _bbox_map_frustum(
    det: Detection, yaw: float, n_tiles: int, hfov: float, vfov: float,
) -> tuple[float, float, float, float]:
    """Map-frame bearing/elevation bounds of the detection bbox.

    Returns ``(bearing_lo, bearing_span, el_lo, el_hi)`` where bearings are handled
    on the circle: ``bearing_lo`` is one edge and ``bearing_span`` (>=0) the angular
    width, so membership is ``wrap_pi(theta - bearing_lo) in [0, bearing_span]``.
    """
    x0, y0, x1, y1 = det.bbox_xyxy
    corners = [(x0, y0), (x1, y0), (x0, y1), (x1, y1)]
    bearings = []
    elevations = []
    for u, v in corners:
        b, e = tile_pixel_to_map_ray(det.tile_id, u, v, yaw, n_tiles, hfov, vfov)
        bearings.append(b)
        elevations.append(e)
    bearings = np.array(bearings)
    elevations = np.array(elevations)

    # Smallest arc covering all corner bearings (robust to the +/-pi wrap): try each
    # corner as the low edge, measure the span to the farthest other corner CCW.
    best_lo = bearings[0]
    best_span = 2.0 * np.pi
    for lo in bearings:
        d = np.mod(bearings - lo, 2.0 * np.pi)
        span = float(d.max())
        if span < best_span:
            best_span = span
            best_lo = float(lo)
    return best_lo, best_span, float(elevations.min()), float(elevations.max())


def _points_in_frustum(
    points: np.ndarray,
    apex: np.ndarray,
    bearing_lo: float,
    bearing_span: float,
    el_lo: float,
    el_hi: float,
    pad: float,
    max_range: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Select ``points`` (N, 3) whose ray from ``apex`` lies inside the padded frustum.

    Returns ``(mask, ranges)`` where ``ranges`` is the per-point distance from apex.
    """
    if len(points) == 0:
        return np.zeros(0, dtype=bool), np.zeros(0)
    rel = points - apex[None, :]
    dx, dy, dz = rel[:, 0], rel[:, 1], rel[:, 2]
    horiz = np.hypot(dx, dy)
    ranges = np.sqrt(horiz ** 2 + dz ** 2)
    bearing = np.arctan2(dy, dx)
    elevation = np.arctan2(dz, horiz)

    # bearing membership on the circle: shift so the padded interval is
    # [0, span + 2*pad] starting at bearing_lo - pad.
    off = np.mod(bearing - (bearing_lo - pad), 2.0 * np.pi)
    in_bearing = off <= (bearing_span + 2.0 * pad)
    in_el = (elevation >= el_lo - pad) & (elevation <= el_hi + pad)
    in_range = (ranges > 1e-6) & (ranges <= max_range)
    mask = in_bearing & in_el & in_range
    return mask, ranges


def _nearest_cluster(ranges: np.ndarray, cfg: FusionConfig) -> np.ndarray:
    """Indices of the nearest dense range cluster among candidate ``ranges``.

    Sorts by range and splits wherever consecutive points jump by more than
    ``cluster_gap``; returns the members of the first (nearest) cluster.
    """
    order = np.argsort(ranges)
    sorted_r = ranges[order]
    # split points where the gap to the previous sorted range exceeds cluster_gap
    gaps = np.diff(sorted_r)
    split_at = np.nonzero(gaps > cfg.cluster_gap)[0]
    end = split_at[0] + 1 if len(split_at) else len(sorted_r)
    return order[:end]


def fuse_detection(
    det: Detection,
    scan: LidarScan,
    odom: OdomState,
    cfg: FusionConfig = DEFAULT_FUSION_CONFIG,
    *,
    n_tiles: int = DEFAULT_N_TILES,
    hfov: float = DEFAULT_TILE_HFOV,
    vfov: float = DEFAULT_TILE_VFOV,
) -> Fused3D | None:
    """Lift one detection to 3D using the map-frame lidar scan.

    Returns a :class:`Fused3D`, or ``None`` if no lidar cluster with at least
    ``cfg.min_points`` points falls inside the detection's frustum.
    """
    points = np.asarray(scan.points, dtype=float).reshape(-1, 3)
    apex = np.array([odom.x, odom.y, odom.z], dtype=float)

    bearing_lo, bearing_span, el_lo, el_hi = _bbox_map_frustum(
        det, odom.yaw, n_tiles, hfov, vfov
    )
    mask, ranges = _points_in_frustum(
        points, apex, bearing_lo, bearing_span, el_lo, el_hi,
        cfg.angular_pad, cfg.max_range,
    )
    cand_idx = np.nonzero(mask)[0]
    if len(cand_idx) < cfg.min_points:
        return None

    cand_ranges = ranges[cand_idx]
    local = _nearest_cluster(cand_ranges, cfg)
    cluster_idx = cand_idx[local]
    if len(cluster_idx) < cfg.min_points:
        return None

    cluster_pts = points[cluster_idx]
    centroid = cluster_pts.mean(axis=0)
    range_m = float(np.linalg.norm(centroid - apex))
    return Fused3D(
        centroid=centroid,
        points=cluster_pts,
        n_points=len(cluster_idx),
        range_m=range_m,
    )
