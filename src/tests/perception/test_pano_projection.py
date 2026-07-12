"""Forward-projection tests for core.perception.pano_projection.

Ported from the T5 tools suite (tools/tests/test_projection.py) so the coverage lives
with the code after the projection moved into core/. Pins the round-trip against tiling's
backward mapping, the VFOV gate, azimuth wrap, the min-range gate, and gray color fill.
"""
from __future__ import annotations

import numpy as np
import pytest

from core.interfaces import OdomState
from core.perception import tiling
from core.perception.pano_projection import (
    GRAY,
    project_points_to_pano,
    sample_colors,
)


def _point_along_ray(apex, bearing, elevation, r):
    """A map-frame point at range ``r`` along ``(bearing, elevation)`` from ``apex``."""
    dx = r * np.cos(elevation) * np.cos(bearing)
    dy = r * np.cos(elevation) * np.sin(bearing)
    dz = r * np.sin(elevation)
    return np.array(apex, dtype=float) + np.array([dx, dy, dz])


@pytest.mark.parametrize("yaw", [0.0, 0.9, -1.7, 3.0, -3.0])
def test_roundtrip_recovers_bearing_elevation(yaw):
    """Project points on a grid of rays, then invert the resulting pixel back
    through tiling to recover the original bearing/elevation."""
    apex = (1.5, -2.0, 0.7)
    odom = OdomState(t=0.0, x=apex[0], y=apex[1], z=apex[2], yaw=yaw)

    azis = np.linspace(-np.pi + 0.2, np.pi - 0.2, 13)
    els = np.linspace(np.deg2rad(-55), np.deg2rad(55), 7)

    for az_cam in azis:
        for el in els:
            bearing = tiling.wrap_pi(az_cam + yaw)
            pt = _point_along_ray(apex, bearing, el, 4.0)[None, :]
            rows, cols, valid = project_points_to_pano(pt, odom, min_range=0.75)
            assert valid[0], (az_cam, el)

            rec_az = tiling.column_to_azimuth(cols[0])
            rec_el = tiling.row_to_elevation(rows[0])
            rec_bearing, rec_el2 = tiling.camera_ray_to_map(float(rec_az), float(rec_el), yaw)

            d_bear = abs(float(tiling.wrap_pi(rec_bearing - bearing)))
            assert d_bear < 2.0 * np.pi / tiling.PANO_WIDTH + 1e-6
            assert abs(rec_el2 - el) < tiling.PANO_VFOV / tiling.PANO_HEIGHT + 1e-6


@pytest.mark.parametrize("yaw", [0.0, 0.5, -2.5, 3.1, -3.1])
def test_vectorized_azimuth_matches_scalar(yaw):
    """The vectorized bearing->camera-azimuth math must equal scalar map_ray_to_camera."""
    bearings = np.array([-np.pi + 0.01, -1.0, 0.0, 1.0, np.pi - 0.01, np.pi])
    vec = tiling.wrap_pi(bearings - yaw)
    for b, v in zip(bearings, vec):
        scal_az, _ = tiling.map_ray_to_camera(float(b), 0.0, yaw)
        assert abs(float(tiling.wrap_pi(v - scal_az))) < 1e-9


def test_vfov_gate_never_clamps():
    """Points above/below the +-60 deg band are invalid, not clamped to an edge row."""
    odom = OdomState(t=0.0, x=0.0, y=0.0, z=0.0, yaw=0.0)
    apex = (0.0, 0.0, 0.0)
    for el in [np.deg2rad(75), np.deg2rad(-75), np.deg2rad(89)]:
        pt = _point_along_ray(apex, 0.3, el, 5.0)[None, :]
        _, _, valid = project_points_to_pano(pt, odom, min_range=0.75)
        assert not valid[0], f"elevation {np.rad2deg(el):.0f} deg should be out of VFOV"

    pt = _point_along_ray(apex, 0.3, np.deg2rad(59), 5.0)[None, :]
    _, _, valid = project_points_to_pano(pt, odom, min_range=0.75)
    assert valid[0]


def test_azimuth_wrap_at_seam():
    """A camera azimuth just left/right of column 0 lands at col 0 / col 1919."""
    odom = OdomState(t=0.0, x=0.0, y=0.0, z=0.0, yaw=0.0)
    apex = (0.0, 0.0, 0.0)
    eps = 2.0 * np.pi / tiling.PANO_WIDTH * 0.25

    for az_cam, want_col in [(np.pi - eps, 0), (-np.pi + eps, tiling.PANO_WIDTH - 1)]:
        bearing = tiling.wrap_pi(az_cam)
        pt = _point_along_ray(apex, bearing, 0.0, 5.0)[None, :]
        _, cols, valid = project_points_to_pano(pt, odom, min_range=0.75)
        assert valid[0]
        assert cols[0] == want_col, (np.rad2deg(az_cam), cols[0], want_col)


def test_min_range_gate():
    """Points within min-range (and near-zero range) are invalid; beyond it, valid."""
    odom = OdomState(t=0.0, x=0.0, y=0.0, z=0.0, yaw=0.0)
    apex = (0.0, 0.0, 0.0)
    near = _point_along_ray(apex, 0.3, 0.0, 0.5)[None, :]
    far = _point_along_ray(apex, 0.3, 0.0, 2.0)[None, :]
    at_apex = np.array([[0.0, 0.0, 0.0]])

    _, _, v_near = project_points_to_pano(near, odom, min_range=0.75)
    _, _, v_far = project_points_to_pano(far, odom, min_range=0.75)
    _, _, v_apex = project_points_to_pano(at_apex, odom, min_range=0.75)
    assert not v_near[0]
    assert v_far[0]
    assert not v_apex[0]


def test_empty_points():
    odom = OdomState(t=0.0, x=0.0, y=0.0, z=0.0, yaw=0.0)
    rows, cols, valid = project_points_to_pano(np.zeros((0, 3)), odom)
    assert rows.shape == (0,) and cols.shape == (0,) and valid.shape == (0,)


def test_sample_colors_gray_for_invalid():
    """Valid pixels sample the image; invalid pixels get the gray fill."""
    img = np.zeros((tiling.PANO_HEIGHT, tiling.PANO_WIDTH, 3), dtype=np.uint8)
    img[:] = (10, 200, 40)
    rows = np.array([5, 0], dtype=np.int64)
    cols = np.array([7, 0], dtype=np.int64)
    valid = np.array([True, False])
    out = sample_colors(img, rows, cols, valid)
    np.testing.assert_array_equal(out[0], (10, 200, 40))
    np.testing.assert_array_equal(out[1], GRAY)
