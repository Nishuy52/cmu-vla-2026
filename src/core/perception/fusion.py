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
3. **Lateral (3D) clustering.** The old design accepted the *whole* range-nearest
   slab of the cone as the object — i.e. the full cone cross-section, which is why
   live boxes were 1.1-4.5x inflated versus GT and why two same-class objects sharing
   one cone (e.g. two monitors) collapsed into a single instance. Instead, candidate
   points are grouped into connected components by 3D proximity (radius-based
   union-find, ``cfg.cluster_radius``) — this captures depth AND lateral separation in
   one pass, so a background wall behind the object *and* a second object sitting
   beside it both fall into their own components rather than being absorbed into the
   winning slab.
4. **Component selection.** Among components with >= ``min_points`` points, the one
   whose points sit closest (in mean angular deviation) to the detection bbox's centre
   ray is kept — not merely the nearest in range — since the 2D detector's box is
   centred on the object it saw, so the real object's points should cluster near that
   ray more tightly than incidental background/clutter caught at the cone's padded
   edges.
5. **Accept / reject.** If the winning component has < ``min_points`` (default 5)
   points, the detection is rejected (``None``). Otherwise return the component's
   centroid and its point set.

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
    cluster_gap: float = 0.5      # m; a range gap wider than this splits clusters (legacy;
                                   # retained for the pre-lateral-split behaviour some
                                   # callers may still rely on, no longer used by
                                   # fuse_detection itself)
    cluster_radius: float = 0.35  # m; 3D union-find radius for lateral (post-frustum)
                                   # point clustering. Chosen between two failure modes:
                                   # - too large re-creates the old cone-slab bug (two
                                   #   nearby objects re-merge into one component);
                                   # - too small shatters a single continuous surface at
                                   #   long range, where lidar angular resolution spreads
                                   #   returns further apart than at close range.
                                   # 0.35 m sits below typical office inter-object gaps
                                   # (chairs/monitors/tables are usually >0.4-0.5 m apart)
                                   # while staying above typical intra-object point
                                   # spacing at <= max_range=15 m for this lidar's density.
                                   # Known limitation: sparse far-range surfaces can still
                                   # exceed this spacing and over-segment (see report).


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Select ``points`` (N, 3) whose ray from ``apex`` lies inside the padded frustum.

    Returns ``(mask, ranges, bearing, elevation)`` — the latter three are per-point
    (full-length, not just the masked subset) so callers can reuse them for the
    downstream lateral-clustering / centre-ray selection without recomputing.
    """
    if len(points) == 0:
        z = np.zeros(0)
        return np.zeros(0, dtype=bool), z, z, z
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
    return mask, ranges, bearing, elevation


def _lateral_components(points: np.ndarray, radius: float) -> np.ndarray:
    """Group ``points`` (N, 3) into connected components by 3D proximity.

    Deterministic radius-based union-find: two points are linked (same component)
    iff their Euclidean distance is <= ``radius``. Pure numpy, O(N^2) pairwise
    distances — fine for the frustum-gated candidate counts this runs on (tens to
    low hundreds of points after the angular/range gate), and simpler/more
    predictable than a voxel-hash scheme.

    Returns an ``(N,)`` int array of component labels (0..K-1, arbitrary order).
    """
    n = len(points)
    if n == 0:
        return np.zeros(0, dtype=int)
    parent = np.arange(n)

    def find(i: int) -> int:
        root = i
        while parent[root] != root:
            root = parent[root]
        while parent[i] != root:
            parent[i], i = root, parent[i]
        return root

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    diff = points[:, None, :] - points[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=-1))
    ii, jj = np.nonzero(np.triu(dist <= radius, k=1))
    for i, j in zip(ii.tolist(), jj.tolist()):
        union(i, j)

    roots = np.array([find(i) for i in range(n)])
    # relabel roots to compact 0..K-1 for a tidy return value
    _, labels = np.unique(roots, return_inverse=True)
    return labels


def _select_component(
    cand_pts: np.ndarray,
    cand_ranges: np.ndarray,
    labels: np.ndarray,
    apex: np.ndarray,
    centre_bearing: float,
    centre_elevation: float,
    min_points: int,
) -> np.ndarray | None:
    """Pick the component whose points sit closest to the detection's centre ray.

    "Closest" is mean angular deviation (bearing + elevation, combined) from
    ``(centre_bearing, centre_elevation)`` — direction-based, not range-based, so a
    background wall caught at the padded cone edge loses to the real object even
    when the wall happens to be nearer. Ties (to float precision) fall back to
    nearer range. Components below ``min_points`` are ignored. Returns local
    indices (into ``cand_pts``) of the winning component, or ``None`` if no
    component clears ``min_points``.
    """
    rel = cand_pts - apex[None, :]
    dx, dy, dz = rel[:, 0], rel[:, 1], rel[:, 2]
    horiz = np.hypot(dx, dy)
    bearing = np.arctan2(dy, dx)
    elevation = np.arctan2(dz, horiz)
    dbearing = np.abs(wrap_pi(bearing - centre_bearing))
    delevation = np.abs(elevation - centre_elevation)
    deviation = np.sqrt(dbearing ** 2 + delevation ** 2)

    best_idx = None
    best_key = None
    for label in np.unique(labels):
        idx = np.nonzero(labels == label)[0]
        if len(idx) < min_points:
            continue
        key = (round(float(deviation[idx].mean()), 6), float(cand_ranges[idx].mean()))
        if best_key is None or key < best_key:
            best_key = key
            best_idx = idx
    return best_idx


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
    mask, ranges, _bearing, _elevation = _points_in_frustum(
        points, apex, bearing_lo, bearing_span, el_lo, el_hi,
        cfg.angular_pad, cfg.max_range,
    )
    cand_idx = np.nonzero(mask)[0]
    if len(cand_idx) < cfg.min_points:
        return None

    cand_pts = points[cand_idx]
    cand_ranges = ranges[cand_idx]

    labels = _lateral_components(cand_pts, cfg.cluster_radius)
    centre_bearing = wrap_pi(bearing_lo + bearing_span / 2.0)
    centre_elevation = (el_lo + el_hi) / 2.0
    local = _select_component(
        cand_pts, cand_ranges, labels, apex, centre_bearing, centre_elevation, cfg.min_points,
    )
    if local is None:
        return None
    cluster_idx = cand_idx[local]

    cluster_pts = points[cluster_idx]
    centroid = cluster_pts.mean(axis=0)
    range_m = float(np.linalg.norm(centroid - apex))
    return Fused3D(
        centroid=centroid,
        points=cluster_pts,
        n_points=len(cluster_idx),
        range_m=range_m,
    )
