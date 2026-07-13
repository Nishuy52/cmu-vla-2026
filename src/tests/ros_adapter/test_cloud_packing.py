"""Byte-level tests for the pure PointCloud2 packing (Windows-testable, no ROS).

Verifies point_step, field offsets/datatype, and a round-trip decode of a few known
points/colors via np.frombuffer — the layout the rclpy node stuffs into the message.
"""
from __future__ import annotations

import numpy as np

from ros_adapter.cloud_packing import (
    FIELDS,
    FLOAT32,
    POINT_STEP,
    pack_colored_cloud,
    pack_rgb_float,
)


def test_fields_and_point_step():
    assert POINT_STEP == 16
    names = [f[0] for f in FIELDS]
    offsets = [f[1] for f in FIELDS]
    dtypes = [f[2] for f in FIELDS]
    counts = [f[3] for f in FIELDS]
    assert names == ["x", "y", "z", "rgb"]
    assert offsets == [0, 4, 8, 12]
    assert dtypes == [FLOAT32, FLOAT32, FLOAT32, FLOAT32]
    assert counts == [1, 1, 1, 1]


def test_buffer_size_matches_point_step():
    xyz = np.zeros((7, 3), dtype=np.float32)
    rgb = np.zeros((7, 3), dtype=np.uint8)
    data, fields, step = pack_colored_cloud(xyz, rgb)
    assert step == 16
    assert len(data) == 16 * 7
    assert fields == FIELDS


def test_roundtrip_known_points_and_colors():
    xyz = np.array(
        [[1.0, 2.0, 3.0], [-4.5, 0.25, 100.0], [0.0, 0.0, 0.0]], dtype=np.float32
    )
    rgb = np.array([[255, 0, 0], [0, 128, 64], [10, 20, 30]], dtype=np.uint8)
    data, _, step = pack_colored_cloud(xyz, rgb)

    # Decode xyz as three leading float32 per point.
    xyz_dt = np.dtype({"names": ["x", "y", "z"], "formats": ["<f4"] * 3,
                       "offsets": [0, 4, 8], "itemsize": step})
    dec = np.frombuffer(data, dtype=xyz_dt)
    np.testing.assert_array_equal(dec["x"], xyz[:, 0])
    np.testing.assert_array_equal(dec["y"], xyz[:, 1])
    np.testing.assert_array_equal(dec["z"], xyz[:, 2])

    # Decode the rgb float field back to uint32 bits and unpack R,G,B.
    rgb_f = np.frombuffer(data, dtype=np.dtype({"names": ["rgb"], "formats": ["<f4"],
                                                "offsets": [12], "itemsize": step}))["rgb"]
    packed = rgb_f.copy().view(np.uint32)
    r = (packed >> 16) & 0xFF
    g = (packed >> 8) & 0xFF
    b = packed & 0xFF
    np.testing.assert_array_equal(r, rgb[:, 0])
    np.testing.assert_array_equal(g, rgb[:, 1])
    np.testing.assert_array_equal(b, rgb[:, 2])


def test_pack_rgb_float_bit_layout():
    """rgb float field carries (R<<16)|(G<<8)|B in its uint32 bit pattern."""
    rgb = np.array([[0x12, 0x34, 0x56]], dtype=np.uint8)
    f = pack_rgb_float(rgb)
    assert f.dtype == np.float32
    packed = f.copy().view(np.uint32)[0]
    assert int(packed) == 0x123456


def test_empty_cloud():
    data, fields, step = pack_colored_cloud(
        np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8)
    )
    assert data == b"" and step == 16 and fields == FIELDS
