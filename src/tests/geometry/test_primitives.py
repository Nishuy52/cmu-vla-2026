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
