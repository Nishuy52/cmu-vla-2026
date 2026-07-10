"""Low-level axis-aligned-box and 2D segment geometry (map frame, metres).

Pure numpy, deterministic, no I/O. All boxes are the 3D AABBs of an
``InstanceRecord`` (``aabb_min`` / ``aabb_max``); "footprint" is the XY
projection, the height axis is Z. Every function is a total, side-effect-free
metric primitive used by ``toolbox.py``.
"""
from __future__ import annotations

import numpy as np

# Numerical slack for boundary comparisons (metres); keeps "touching" cases
# stable against float round-off without widening any real tolerance.
EPS: float = 1e-9


def _as3(v: np.ndarray) -> np.ndarray:
    """Return a (3,) float64 copy of a coordinate vector."""
    a = np.asarray(v, dtype=float).reshape(-1)
    if a.shape[0] < 3:
        a = np.concatenate([a, np.zeros(3 - a.shape[0])])
    return a[:3]


def footprint_min(box_min: np.ndarray, box_max: np.ndarray) -> np.ndarray:
    """XY lower corner of an AABB footprint, metres."""
    return _as3(box_min)[:2]


def footprint_max(box_min: np.ndarray, box_max: np.ndarray) -> np.ndarray:
    """XY upper corner of an AABB footprint, metres."""
    return _as3(box_max)[:2]


def footprint_diagonal(box_min: np.ndarray, box_max: np.ndarray) -> float:
    """Length of the XY footprint diagonal of an AABB, metres."""
    lo = footprint_min(box_min, box_max)
    hi = footprint_max(box_min, box_max)
    return float(np.hypot(*(hi - lo)))


def footprint_half_width(box_min: np.ndarray, box_max: np.ndarray) -> float:
    """Half of the larger XY footprint extent, metres (capsule radius unit)."""
    lo = footprint_min(box_min, box_max)
    hi = footprint_max(box_min, box_max)
    return float(np.max(hi - lo) / 2.0)


def interval_overlap(a_lo: float, a_hi: float, b_lo: float, b_hi: float) -> float:
    """Signed overlap of two 1D intervals; negative = gap, positive = overlap."""
    return float(min(a_hi, b_hi) - max(a_lo, b_lo))


def footprint_overlap_area(
    a_min: np.ndarray, a_max: np.ndarray, b_min: np.ndarray, b_max: np.ndarray
) -> float:
    """Area of XY-footprint intersection of two AABBs, m^2 (0 if disjoint)."""
    a0, a1 = footprint_min(a_min, a_max), footprint_max(a_min, a_max)
    b0, b1 = footprint_min(b_min, b_max), footprint_max(b_min, b_max)
    dx = interval_overlap(a0[0], a1[0], b0[0], b1[0])
    dy = interval_overlap(a0[1], a1[1], b0[1], b1[1])
    if dx <= 0.0 or dy <= 0.0:
        return 0.0
    return float(dx * dy)


def footprints_overlap(
    a_min: np.ndarray, a_max: np.ndarray, b_min: np.ndarray, b_max: np.ndarray
) -> bool:
    """True if the two AABB footprints share positive XY area."""
    return footprint_overlap_area(a_min, a_max, b_min, b_max) > EPS


def aabb_gap(
    a_min: np.ndarray, a_max: np.ndarray, b_min: np.ndarray, b_max: np.ndarray
) -> float:
    """Closest 3D distance between two AABB surfaces, metres (0 if overlapping)."""
    a0, a1 = _as3(a_min), _as3(a_max)
    b0, b1 = _as3(b_min), _as3(b_max)
    d = np.maximum.reduce([b0 - a1, a0 - b1, np.zeros(3)])
    return float(np.linalg.norm(d))


def aabb_gap_2d(
    a_min: np.ndarray, a_max: np.ndarray, b_min: np.ndarray, b_max: np.ndarray
) -> float:
    """Closest XY distance between two AABB footprints, metres (0 if overlapping)."""
    a0, a1 = footprint_min(a_min, a_max), footprint_max(a_min, a_max)
    b0, b1 = footprint_min(b_min, b_max), footprint_max(b_min, b_max)
    d = np.maximum.reduce([b0 - a1, a0 - b1, np.zeros(2)])
    return float(np.linalg.norm(d))


def point_to_segment_2d(
    p: np.ndarray, s0: np.ndarray, s1: np.ndarray
) -> tuple[float, float]:
    """Distance from XY point p to segment s0-s1, plus clamped parameter t in [0,1].

    Returns (distance_m, t) where the closest point is s0 + t*(s1-s0).
    """
    p2 = _as3(p)[:2]
    a = _as3(s0)[:2]
    b = _as3(s1)[:2]
    ab = b - a
    denom = float(ab @ ab)
    if denom <= EPS:  # degenerate segment: both anchors coincide
        return float(np.linalg.norm(p2 - a)), 0.0
    t = float((p2 - a) @ ab / denom)
    t = min(1.0, max(0.0, t))
    closest = a + t * ab
    return float(np.linalg.norm(p2 - closest)), t


def _orient(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """2D cross product (b-a) x (c-a); sign gives orientation of a->b->c."""
    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def _on_segment(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> bool:
    """True if collinear point c lies within the bounding box of segment a-b."""
    return (
        min(a[0], b[0]) - EPS <= c[0] <= max(a[0], b[0]) + EPS
        and min(a[1], b[1]) - EPS <= c[1] <= max(a[1], b[1]) + EPS
    )


def segments_intersect_2d(
    p1: np.ndarray, p2: np.ndarray, q1: np.ndarray, q2: np.ndarray
) -> bool:
    """True if XY segments p1-p2 and q1-q2 properly cross or touch (inclusive).

    Handles the collinear-overlap and shared-endpoint cases (both count as an
    intersection), which is what a corridor-gate threading test requires.
    """
    p1, p2 = _as3(p1)[:2], _as3(p2)[:2]
    q1, q2 = _as3(q1)[:2], _as3(q2)[:2]
    d1 = _orient(q1, q2, p1)
    d2 = _orient(q1, q2, p2)
    d3 = _orient(p1, p2, q1)
    d4 = _orient(p1, p2, q2)
    if ((d1 > EPS and d2 < -EPS) or (d1 < -EPS and d2 > EPS)) and (
        (d3 > EPS and d4 < -EPS) or (d3 < -EPS and d4 > EPS)
    ):
        return True
    if abs(d1) <= EPS and _on_segment(q1, q2, p1):
        return True
    if abs(d2) <= EPS and _on_segment(q1, q2, p2):
        return True
    if abs(d3) <= EPS and _on_segment(p1, p2, q1):
        return True
    if abs(d4) <= EPS and _on_segment(p1, p2, q2):
        return True
    return False


def aabb_face_points_2d(
    a_min: np.ndarray, a_max: np.ndarray, b_min: np.ndarray, b_max: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Closest points on two AABB footprints (XY), one per box.

    For each axis, if the intervals overlap the shared coordinate is the mid of
    the overlap on both; otherwise each box contributes its facing edge. Used to
    build the corridor gate between the two anchors' nearest faces.
    """
    a0, a1 = footprint_min(a_min, a_max), footprint_max(a_min, a_max)
    b0, b1 = footprint_min(b_min, b_max), footprint_max(b_min, b_max)
    pa = np.zeros(2)
    pb = np.zeros(2)
    for i in range(2):
        ov = interval_overlap(a0[i], a1[i], b0[i], b1[i])
        if ov > 0.0:  # overlapping on this axis: share the overlap midpoint
            mid = (max(a0[i], b0[i]) + min(a1[i], b1[i])) / 2.0
            pa[i] = pb[i] = mid
        elif a1[i] <= b0[i]:  # a is on the low side of this axis
            pa[i] = a1[i]
            pb[i] = b0[i]
        else:  # a is on the high side
            pa[i] = a0[i]
            pb[i] = b1[i]
    return pa, pb
