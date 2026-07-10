"""Synthetic-bag builders for the replay tests.

Uses rosbags' WRITER to create tiny real ROS 2 bags in ``tmp_path`` with known values, so
converter/store/fixture behaviour is exercised through the same serialisation path a real
bag would take. No ROS install, no network, fully deterministic.
"""
from __future__ import annotations

import math
import struct
from pathlib import Path

import numpy as np
import pytest


def _ts():
    from rosbags.typesys import Stores, get_typestore

    return get_typestore(Stores.ROS2_JAZZY)


def _header(ts, frame_id, sec, nsec):
    Header = ts.types["std_msgs/msg/Header"]
    Time = ts.types["builtin_interfaces/msg/Time"]
    return Header(stamp=Time(sec=int(sec), nanosec=int(nsec)), frame_id=frame_id)


def make_image_msg(ts, rgb: np.ndarray, encoding="rgb8", sec=1, nsec=0, step=None):
    """rgb is (H, W, 3) uint8 in RGB order; encoded per `encoding`."""
    Image = ts.types["sensor_msgs/msg/Image"]
    h, w, _ = rgb.shape
    if encoding == "bgr8":
        data = rgb[:, :, ::-1]
    else:
        data = rgb
    data = np.ascontiguousarray(data, dtype=np.uint8)
    row_bytes = w * 3
    if step is None or step == row_bytes:
        buf = data.reshape(-1)
        step_v = row_bytes
    else:  # padded rows
        step_v = step
        padded = np.zeros((h, step), dtype=np.uint8)
        padded[:, :row_bytes] = data.reshape(h, row_bytes)
        buf = padded.reshape(-1)
    return Image(
        header=_header(ts, "camera", sec, nsec),
        height=h, width=w, encoding=encoding,
        is_bigendian=0, step=step_v, data=buf,
    )


def make_pointcloud_msg(ts, xyzi, layout, frame_id="map", sec=1, nsec=0):
    """Build a PointCloud2 with fields at caller-specified offsets.

    xyzi: (N, k) float array. layout: list of (name, offset, point_step) OR a dict
    ``{'point_step': int, 'fields': [(name, offset)]}`` — offsets are BYTE offsets and
    need NOT be the standard contiguous XYZI packing.
    """
    PC2 = ts.types["sensor_msgs/msg/PointCloud2"]
    PF = ts.types["sensor_msgs/msg/PointField"]
    point_step = layout["point_step"]
    fields = layout["fields"]  # list of (name, offset)
    n = len(xyzi)
    buf = np.zeros(n * point_step, dtype=np.uint8)
    pf_list = []
    for col, (name, offset) in enumerate(fields):
        pf_list.append(PF(name=name, offset=offset, datatype=7, count=1))  # 7 = FLOAT32
        for i in range(n):
            struct.pack_into("<f", buf, i * point_step + offset, float(xyzi[i, col]))
    return PC2(
        header=_header(ts, frame_id, sec, nsec),
        height=1, width=n, fields=pf_list,
        is_bigendian=False, point_step=point_step, row_step=point_step * n,
        data=buf, is_dense=True,
    )


def make_odom_msg(ts, x, y, z, yaw, frame_id="map", sec=1, nsec=0):
    Odom = ts.types["nav_msgs/msg/Odometry"]
    PWC = ts.types["geometry_msgs/msg/PoseWithCovariance"]
    TWC = ts.types["geometry_msgs/msg/TwistWithCovariance"]
    Pose = ts.types["geometry_msgs/msg/Pose"]
    Twist = ts.types["geometry_msgs/msg/Twist"]
    Point = ts.types["geometry_msgs/msg/Point"]
    Quat = ts.types["geometry_msgs/msg/Quaternion"]
    Vec3 = ts.types["geometry_msgs/msg/Vector3"]
    qz = math.sin(yaw / 2.0)
    qw = math.cos(yaw / 2.0)
    pose = Pose(
        position=Point(x=float(x), y=float(y), z=float(z)),
        orientation=Quat(x=0.0, y=0.0, z=qz, w=qw),
    )
    twist = Twist(linear=Vec3(x=0.0, y=0.0, z=0.0), angular=Vec3(x=0.0, y=0.0, z=0.0))
    cov = np.zeros(36, dtype=np.float64)
    return Odom(
        header=_header(ts, frame_id, sec, nsec),
        child_frame_id="sensor",
        pose=PWC(pose=pose, covariance=cov),
        twist=TWC(twist=twist, covariance=cov.copy()),
    )


def make_string_msg(ts, text):
    String = ts.types["std_msgs/msg/String"]
    return String(data=text)


def write_bag(path: Path, records):
    """records: list of (topic, msgtype, msg, timestamp_ns). Writes a rosbag2 dir at path."""
    from rosbags.rosbag2 import Writer

    ts = _ts()
    with Writer(path, version=9) as w:
        conns = {}
        for topic, msgtype, _msg, _tns in records:
            if topic not in conns:
                conns[topic] = w.add_connection(topic, msgtype, typestore=ts)
        for topic, msgtype, msg, tns in records:
            w.write(conns[topic], int(tns), ts.serialize_cdr(msg, msgtype))
    return path


@pytest.fixture
def typestore():
    return _ts()


@pytest.fixture
def bag_factory(tmp_path, typestore):
    """Return a callable(records, name) -> bag path. records use helper msgs above."""
    ts = typestore
    counter = {"n": 0}

    def _make(records, name=None):
        counter["n"] += 1
        p = tmp_path / (name or f"bag_{counter['n']}")
        write_bag(p, records)
        return p

    return _make
