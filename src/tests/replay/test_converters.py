"""Converter correctness through the real rosbags serialise->deserialise path."""
from __future__ import annotations

import math

import numpy as np
import pytest

from core.interfaces import LidarScan, OdomState, PanoFrame, Question, TerrainPatch
from core.replay.bag_reader import (
    TOPIC_CAMERA,
    TOPIC_ODOM,
    TOPIC_QUESTION,
    TOPIC_SCAN,
    TOPIC_TERRAIN,
    TOPIC_TERRAIN_EXT,
    BagSource,
)
from tests.replay.conftest import (
    make_image_msg,
    make_odom_msg,
    make_pointcloud_msg,
    make_string_msg,
)

IMG = "sensor_msgs/msg/Image"
PC2 = "sensor_msgs/msg/PointCloud2"
ODO = "nav_msgs/msg/Odometry"
STR = "std_msgs/msg/String"

# A deliberately NONSTANDARD point layout: 32-byte stride, x/y/z/intensity at scattered
# offsets (not the contiguous 0/4/8/12 packing) to prove offsets are read, not assumed.
NONSTD_LAYOUT = {
    "point_step": 32,
    "fields": [("x", 4), ("y", 12), ("z", 20), ("intensity", 28)],
}
STD_LAYOUT = {
    "point_step": 16,
    "fields": [("x", 0), ("y", 4), ("z", 8), ("intensity", 12)],
}


def _one(bag_factory, typestore, topic, msgtype, msg):
    bag = bag_factory([(topic, msgtype, msg, 5_000_000_000)])
    recs = list(BagSource(bag))
    assert len(recs) == 1
    return recs[0].msg


def test_string_to_question(bag_factory, typestore):
    msg = make_string_msg(typestore, "How many sofas are below a window?")
    q = _one(bag_factory, typestore, TOPIC_QUESTION, STR, msg)
    assert isinstance(q, Question)
    assert q.text == "How many sofas are below a window?"
    assert q.t_received == pytest.approx(5.0)


def test_image_rgb8_roundtrip(bag_factory, typestore):
    rgb = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)
    msg = make_image_msg(typestore, rgb, encoding="rgb8")
    pano = _one(bag_factory, typestore, TOPIC_CAMERA, IMG, msg)
    assert isinstance(pano, PanoFrame)
    assert pano.image.shape == (4, 6, 3)
    assert pano.image.dtype == np.uint8
    assert np.array_equal(pano.image, rgb)


def test_image_bgr8_swapped_to_rgb(bag_factory, typestore):
    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    rgb[..., 0] = 10  # R
    rgb[..., 1] = 20  # G
    rgb[..., 2] = 30  # B
    msg = make_image_msg(typestore, rgb, encoding="bgr8")
    pano = _one(bag_factory, typestore, TOPIC_CAMERA, IMG, msg)
    # bgr8 on the wire must come back in RGB order matching the original rgb array.
    assert np.array_equal(pano.image, rgb)


def test_image_padded_step(bag_factory, typestore):
    rgb = np.arange(3 * 5 * 3, dtype=np.uint8).reshape(3, 5, 3)
    msg = make_image_msg(typestore, rgb, encoding="rgb8", step=5 * 3 + 8)  # padded rows
    pano = _one(bag_factory, typestore, TOPIC_CAMERA, IMG, msg)
    assert np.array_equal(pano.image, rgb)


def test_scan_standard_layout(bag_factory, typestore):
    xyzi = np.array([[1, 2, 3, 0.1], [4, 5, 6, 0.2]], dtype=np.float32)
    msg = make_pointcloud_msg(typestore, xyzi, STD_LAYOUT)
    scan = _one(bag_factory, typestore, TOPIC_SCAN, PC2, msg)
    assert isinstance(scan, LidarScan)
    assert scan.points.shape == (2, 3)
    assert scan.points.dtype == np.float32
    assert np.allclose(scan.points, xyzi[:, :3])


def test_scan_nonstandard_offsets(bag_factory, typestore):
    """PointField offset robustness: scattered offsets must decode identically."""
    xyzi = np.array([[7, 8, 9, 0.3], [-1, -2, -3, 0.4]], dtype=np.float32)
    msg = make_pointcloud_msg(typestore, xyzi, NONSTD_LAYOUT)
    scan = _one(bag_factory, typestore, TOPIC_SCAN, PC2, msg)
    assert np.allclose(scan.points, xyzi[:, :3])


def test_terrain_carries_intensity_nonstandard(bag_factory, typestore):
    xyzi = np.array([[0.5, 0.5, 0.0, 0.0], [1.0, 1.0, 0.2, 0.25]], dtype=np.float32)
    msg = make_pointcloud_msg(typestore, xyzi, NONSTD_LAYOUT)
    terr = _one(bag_factory, typestore, TOPIC_TERRAIN, PC2, msg)
    assert isinstance(terr, TerrainPatch)
    assert terr.points.shape == (2, 4)
    assert terr.extended is False
    assert np.allclose(terr.points, xyzi)


def test_terrain_ext_flag(bag_factory, typestore):
    xyzi = np.array([[0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    msg = make_pointcloud_msg(typestore, xyzi, STD_LAYOUT)
    terr = _one(bag_factory, typestore, TOPIC_TERRAIN_EXT, PC2, msg)
    assert isinstance(terr, TerrainPatch)
    assert terr.extended is True


@pytest.mark.parametrize("yaw", [0.0, math.pi / 2, -math.pi / 3, 2.7, -3.0])
def test_odom_yaw_extraction(bag_factory, typestore, yaw):
    msg = make_odom_msg(typestore, x=1.5, y=-2.5, z=0.3, yaw=yaw)
    odom = _one(bag_factory, typestore, TOPIC_ODOM, ODO, msg)
    assert isinstance(odom, OdomState)
    assert odom.x == pytest.approx(1.5)
    assert odom.y == pytest.approx(-2.5)
    assert odom.z == pytest.approx(0.3)
    # yaw wraps: compare via angle difference
    dyaw = math.atan2(math.sin(odom.yaw - yaw), math.cos(odom.yaw - yaw))
    assert abs(dyaw) < 1e-5


def test_header_stamp_used_for_time(bag_factory, typestore):
    # header stamp 3.5s but bag time 9s -> converter prefers header stamp for sensor t.
    msg = make_odom_msg(typestore, 0, 0, 0, 0.0, sec=3, nsec=500_000_000)
    bag = bag_factory([(TOPIC_ODOM, ODO, msg, 9_000_000_000)])
    odom = list(BagSource(bag))[0].msg
    assert odom.t == pytest.approx(3.5)


def test_unregistered_topic_skipped(bag_factory, typestore):
    msg = make_string_msg(typestore, "hi")
    bag = bag_factory([("/some/other_topic", STR, msg, 1_000_000_000)])
    assert list(BagSource(bag)) == []


def test_custom_converter_override(bag_factory, typestore):
    msg = make_string_msg(typestore, "boom")
    bag = bag_factory([(TOPIC_QUESTION, STR, msg, 1_000_000_000)])
    src = BagSource(bag, topic_map={TOPIC_QUESTION: lambda m, ns: ("custom", m.data)})
    recs = list(src)
    assert recs[0].msg == ("custom", "boom")


def test_frames_attaches_nearest_odom(bag_factory, typestore):
    """PanoFrame.odom should be the newest odom at or before the frame time."""
    odom = make_odom_msg(typestore, x=3.0, y=4.0, z=0.0, yaw=0.5, sec=1)
    rgb = np.zeros((2, 2, 3), dtype=np.uint8)
    img = make_image_msg(typestore, rgb, encoding="rgb8", sec=2)
    bag = bag_factory([
        (TOPIC_ODOM, ODO, odom, 1_000_000_000),
        (TOPIC_CAMERA, IMG, img, 2_000_000_000),
    ])
    frames = list(BagSource(bag).frames())
    panos = [r.msg for r in frames if isinstance(r.msg, PanoFrame)]
    assert len(panos) == 1
    assert panos[0].odom.x == pytest.approx(3.0)
    assert panos[0].odom.y == pytest.approx(4.0)
