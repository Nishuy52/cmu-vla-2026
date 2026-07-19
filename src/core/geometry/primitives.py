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


def footprint_area(box_min: np.ndarray, box_max: np.ndarray) -> float:
    """Area of an AABB's XY footprint, m^2."""
    lo = footprint_min(box_min, box_max)
    hi = footprint_max(box_min, box_max)
    return float(max(hi[0] - lo[0], 0.0) * max(hi[1] - lo[1], 0.0))


def footprint_iom(
    a_min: np.ndarray, a_max: np.ndarray, b_min: np.ndarray, b_max: np.ndarray
) -> float:
    """Footprint intersection-over-min: overlap area / min(footprint areas), in [0,1].

    This is the VLA-3D generation-code support gate (``on``/``below``): the
    intersection is normalised by the SMALLER of the two footprints, so a small
    object fully covered by a large supporter scores 1.0 regardless of the
    supporter's size (unlike intersection-over-target, which never sees the
    anchor's extent).
    """
    inter = footprint_overlap_area(a_min, a_max, b_min, b_max)
    if inter <= 0.0:
        return 0.0
    denom = min(footprint_area(a_min, a_max), footprint_area(b_min, b_max))
    if denom <= EPS:
        return 0.0
    return float(inter / denom)


def largest_face_area(box_min: np.ndarray, box_max: np.ndarray) -> float:
    """Area of the AABB's largest axis-aligned face, m^2 (VLA-3D size basis).

    Size qualifiers ("small"/"big") in the generation spec rank by largest-face
    area, not volume. For extents (dx, dy, dz) the three face areas are dx*dy,
    dx*dz, dy*dz; the largest is returned.
    """
    lo = _as3(box_min)
    hi = _as3(box_max)
    d = np.clip(hi - lo, 0.0, None)
    faces = (d[0] * d[1], d[0] * d[2], d[1] * d[2])
    return float(max(faces))


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


def point_in_footprint_2d(pt: np.ndarray, box_min: np.ndarray, box_max: np.ndarray) -> bool:
    """True if ``pt`` (XY) lies within the AABB's XY footprint (inclusive of the edge)."""
    lo, hi = footprint_min(box_min, box_max), footprint_max(box_min, box_max)
    p = np.asarray(pt, dtype=float).reshape(-1)[:2]
    return bool(lo[0] - EPS <= p[0] <= hi[0] + EPS and lo[1] - EPS <= p[1] <= hi[1] + EPS)


def usable_gate_point(
    p0: np.ndarray,
    p1: np.ndarray,
    midpoint: np.ndarray,
    is_blocked,
    *,
    slide_step_frac: float = 0.1,
    max_slide_frac: float = 0.45,
) -> np.ndarray:
    """The usable crossing point of a (possibly partially-blocked) gate segment.

    Issue #63: a genuine object at the raw gate midpoint (e.g. hotel_room_2's
    duplicate GT "bed frame" instance sitting at the anchor-to-anchor gate line —
    a REAL obstruction, not an anchor's own footprint) is not a usable via-point:
    any route threading the gate has to detour around it, so a via-point placed
    inside it demands geometry no route can satisfy. Slides the crossing point
    away from ``midpoint``, ALONG the gate's own p0<->p1 axis (never off the
    verified anchor-to-anchor line — a real doorway), testing outward in
    ``slide_step_frac``-length increments alternating both directions (nearest
    candidates first), until ``is_blocked`` clears or the bounded sweep
    (``max_slide_frac`` of the gate's length either side, never reaching an
    anchor) is exhausted.

    Deliberately caller-agnostic about what "blocked" means: ``is_blocked`` is a
    ``(point_xy) -> bool`` predicate the caller supplies. This is the ONE shared
    helper both `core.geometry.toolbox.corridor_gate` (scoring, blocked = a third
    GT instance's raw footprint) and `core.nav.planner.plan_through` (live
    planning, blocked = a genuine solid obstacle cell in the costmap) call, so
    the two surfaces can never disagree about where a blocked gate's usable
    crossing is (the #51 scoring/planning-mismatch class of bug).

    Returns ``midpoint`` unchanged when it is already clear, when the segment is
    degenerate (zero length), or when nothing along the bounded sweep clears —
    callers then fall back to the pre-#63 exact-midpoint behaviour (a genuinely
    sealed gate is left for the caller's own existing recovery, never silently
    misplaced past the verified anchor extent).
    """
    p0 = np.asarray(p0, dtype=float).reshape(-1)[:2]
    p1 = np.asarray(p1, dtype=float).reshape(-1)[:2]
    midpoint = np.asarray(midpoint, dtype=float).reshape(-1)[:2]
    if not is_blocked(midpoint):
        return midpoint
    axis = p1 - p0
    length = float(np.hypot(*axis))
    if length < EPS:
        return midpoint
    unit = axis / length
    max_slide_m = max_slide_frac * length
    step_m = max(slide_step_frac * length, 1e-6)
    d = step_m
    while d <= max_slide_m + EPS:
        for sign in (1.0, -1.0):
            cand = midpoint + unit * (sign * d)
            if not is_blocked(cand):
                return cand
        d += step_m
    return midpoint
