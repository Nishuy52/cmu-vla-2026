"""issue #63: the shared usable_gate_point / point_in_footprint_2d primitives."""
from __future__ import annotations

import numpy as np

from core.geometry import primitives as P


def test_point_in_footprint_2d_inside_and_outside():
    box_min = np.array([-1.0, -1.0, 0.0])
    box_max = np.array([1.0, 1.0, 1.0])
    assert P.point_in_footprint_2d(np.array([0.0, 0.0]), box_min, box_max)
    assert P.point_in_footprint_2d(np.array([1.0, 1.0]), box_min, box_max)  # edge inclusive
    assert not P.point_in_footprint_2d(np.array([1.5, 0.0]), box_min, box_max)


def test_usable_gate_point_no_op_when_midpoint_already_clear():
    p0, p1 = np.array([-1.0, 0.0]), np.array([1.0, 0.0])
    mid = (p0 + p1) / 2.0
    out = P.usable_gate_point(p0, p1, mid, lambda pt: False)
    assert np.allclose(out, mid)


def test_usable_gate_point_slides_along_axis_off_a_blocker():
    p0, p1 = np.array([-1.0, 0.0]), np.array([1.0, 0.0])
    mid = np.array([0.0, 0.0])

    def blocked(pt):
        return -0.15 <= pt[0] <= 0.15  # a narrow blocker straddling the midpoint

    out = P.usable_gate_point(p0, p1, mid, blocked)
    assert not blocked(out)
    # Stays on the gate's own axis (y unchanged) and within the segment.
    assert abs(float(out[1])) < 1e-9
    assert -1.0 <= float(out[0]) <= 1.0


def test_usable_gate_point_falls_back_when_nothing_clears():
    p0, p1 = np.array([-1.0, 0.0]), np.array([1.0, 0.0])
    mid = np.array([0.0, 0.0])
    out = P.usable_gate_point(p0, p1, mid, lambda pt: True)  # everything blocked
    assert np.allclose(out, mid)


def test_usable_gate_point_degenerate_segment_returns_midpoint():
    p0 = p1 = np.array([2.0, 3.0])
    out = P.usable_gate_point(p0, p1, p0, lambda pt: True)
    assert np.allclose(out, p0)


# --------------------------------------------------------------------------- issue #69 (D1)


def test_centroid_axis_face_points_matches_axis_case():
    # Two boxes centred on the same y, separated along x: the centroid-axis
    # construction must agree with the plain axis-aligned face projection here.
    c0, c1 = np.array([-2.0, 0.0]), np.array([2.0, 0.0])
    b0_min, b0_max = np.array([-3.0, -1.0]), np.array([-1.0, 1.0])
    b1_min, b1_max = np.array([1.0, -1.0]), np.array([3.0, 1.0])
    p0, p1 = P.centroid_axis_face_points_2d(c0, b0_min, b0_max, c1, b1_min, b1_max)
    assert np.allclose(p0, [-1.0, 0.0])
    assert np.allclose(p1, [1.0, 0.0])


def test_centroid_axis_face_points_nondegenerate_when_footprints_touch():
    # A big and a small box whose footprints already touch/overlap in a thin sliver
    # (the livingroom_1 sofa/round-table shape): the old per-axis-overlap projection
    # collapses to a single point (zero width); the centroid axis must still produce
    # two DISTINCT points (a genuine, non-degenerate "between" segment).
    c0, c1 = np.array([-1.646, -2.170]), np.array([-0.316, -2.433])
    b0_min, b0_max = np.array([-2.588, -3.704]), np.array([-0.704, -0.636])
    b1_min, b1_max = np.array([-0.717, -2.833]), np.array([0.084, -2.032])
    p0, p1 = P.centroid_axis_face_points_2d(c0, b0_min, b0_max, c1, b1_min, b1_max)
    assert np.linalg.norm(p1 - p0) > 0.01
    # Both face points stay ON the respective anchor's own footprint boundary.
    assert P.point_in_footprint_2d(p0, b0_min, b0_max)
    assert P.point_in_footprint_2d(p1, b1_min, b1_max)


def test_centroid_axis_face_points_coincident_centroids_returns_both_centroids():
    c = np.array([1.0, 1.0])
    b_min, b_max = np.array([0.0, 0.0]), np.array([2.0, 2.0])
    p0, p1 = P.centroid_axis_face_points_2d(c, b_min, b_max, c, b_min, b_max)
    assert np.allclose(p0, c)
    assert np.allclose(p1, c)
