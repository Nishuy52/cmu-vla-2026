"""ReplayRobotIO / ReplayClock / MessageStore time semantics."""
from __future__ import annotations

import numpy as np
import pytest

from core.interfaces import IntAnswer, MarkerBox, RobotIO, WaypointCmd
from core.replay.bag_reader import (
    TOPIC_CAMERA,
    TOPIC_ODOM,
    TOPIC_QUESTION,
    TOPIC_SCAN,
    TOPIC_TERRAIN,
    TOPIC_TERRAIN_EXT,
)
from core.replay.replay_io import (
    CH_ODOM,
    CH_SCAN,
    MessageStore,
    ReplayClock,
    ReplayRobotIO,
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
STD = {"point_step": 16, "fields": [("x", 0), ("y", 4), ("z", 8), ("intensity", 12)]}

_ROBOTIO_METHODS = (
    "question", "latest_pano", "latest_scan", "latest_terrain", "latest_odom",
    "publish_waypoint", "publish_marker", "publish_int", "clock",
)


def _multiframe_bag(bag_factory, typestore):
    ts = typestore
    xyzi = np.array([[1, 1, 1, 0.0]], dtype=np.float32)
    recs = [
        (TOPIC_QUESTION, STR, make_string_msg(ts, "Find the sofa"), 1_000_000_000),
        (TOPIC_ODOM, ODO, make_odom_msg(ts, 0, 0, 0, 0.0, sec=1), 1_000_000_000),
        (TOPIC_ODOM, ODO, make_odom_msg(ts, 1, 0, 0, 0.0, sec=2), 2_000_000_000),
        (TOPIC_ODOM, ODO, make_odom_msg(ts, 2, 0, 0, 0.0, sec=3), 3_000_000_000),
        (TOPIC_SCAN, PC2, make_pointcloud_msg(ts, xyzi, STD, sec=2), 2_000_000_000),
        (TOPIC_TERRAIN, PC2, make_pointcloud_msg(ts, xyzi, STD, sec=2), 2_000_000_000),
        (TOPIC_TERRAIN_EXT, PC2, make_pointcloud_msg(ts, xyzi, STD, sec=3), 3_000_000_000),
        (TOPIC_CAMERA, IMG, make_image_msg(ts, np.zeros((2, 2, 3), np.uint8), sec=3), 3_000_000_000),
    ]
    return bag_factory(recs)


def test_satisfies_robotio_protocol(bag_factory, typestore):
    io = ReplayRobotIO.from_bag(_multiframe_bag(bag_factory, typestore))
    for name in _ROBOTIO_METHODS:
        assert callable(getattr(io, name, None)), f"missing {name}"
    assert set(_ROBOTIO_METHODS).issubset(dir(RobotIO))


def test_none_before_first_message(bag_factory, typestore):
    io = ReplayRobotIO.from_bag(_multiframe_bag(bag_factory, typestore))
    io.clock().set(0.0)  # before any message at t=1
    assert io.question() is None
    assert io.latest_odom() is None
    assert io.latest_scan() is None
    assert io.latest_pano() is None


def test_latest_respects_clock(bag_factory, typestore):
    io = ReplayRobotIO.from_bag(_multiframe_bag(bag_factory, typestore))
    io.clock().set(1.0)
    assert io.latest_odom().x == pytest.approx(0.0)
    assert io.latest_scan() is None  # first scan is at t=2
    io.clock().set(2.5)
    assert io.latest_odom().x == pytest.approx(1.0)  # newest <= 2.5 is the t=2 odom
    assert io.latest_scan() is not None
    io.clock().set(10.0)
    assert io.latest_odom().x == pytest.approx(2.0)  # newest overall


def test_terrain_extended_channel(bag_factory, typestore):
    io = ReplayRobotIO.from_bag(_multiframe_bag(bag_factory, typestore))
    io.clock().set(2.0)
    assert io.latest_terrain(extended=False) is not None
    assert io.latest_terrain(extended=True) is None  # ext is at t=3
    io.clock().set(3.0)
    assert io.latest_terrain(extended=True).extended is True


def test_clock_step_walks_schedule(bag_factory, typestore):
    io = ReplayRobotIO.from_bag(_multiframe_bag(bag_factory, typestore))
    clock = io.clock()
    clock.set(io.store.all_times()[0])
    assert clock.now() == pytest.approx(1.0)
    io.tick()
    assert clock.now() == pytest.approx(2.0)
    io.tick()
    assert clock.now() == pytest.approx(3.0)
    io.tick()  # exhausted -> stays put
    assert clock.now() == pytest.approx(3.0)
    assert clock.exhausted


def test_publish_sinks(bag_factory, typestore):
    io = ReplayRobotIO.from_bag(_multiframe_bag(bag_factory, typestore))
    io.publish_waypoint(WaypointCmd(1.0, 2.0))
    io.publish_marker(MarkerBox(0, 0, 0, 1, 1, 1, label="sofa"))
    io.publish_int(IntAnswer(3))
    assert io.waypoints == [WaypointCmd(1.0, 2.0)]
    assert io.markers[0].label == "sofa"
    assert io.ints == [IntAnswer(3)]


def test_question_latched_and_republish_tolerant(bag_factory, typestore):
    ts = typestore
    recs = [
        (TOPIC_QUESTION, STR, make_string_msg(ts, "Find the sofa"), 1_000_000_000),
        (TOPIC_QUESTION, STR, make_string_msg(ts, "Find the sofa"), 2_000_000_000),
        (TOPIC_QUESTION, STR, make_string_msg(ts, "Find the sofa"), 3_000_000_000),
    ]
    io = ReplayRobotIO.from_bag(bag_factory(recs))
    io.clock().set(5.0)
    assert io.question().text == "Find the sofa"


def test_messagestore_latest_bisect_empty():
    store = MessageStore()
    assert store.latest(CH_ODOM, 5.0) is None
    store.add(CH_ODOM, 1.0, "a")
    store.add(CH_ODOM, 3.0, "b")
    store.finalize()
    assert store.latest(CH_ODOM, 0.5) is None
    assert store.latest(CH_ODOM, 1.0) == "a"
    assert store.latest(CH_ODOM, 2.9) == "a"
    assert store.latest(CH_ODOM, 3.0) == "b"
    assert store.latest(CH_ODOM, 99.0) == "b"


def test_messagestore_all_times_sorted_unique():
    store = MessageStore()
    store.add(CH_ODOM, 3.0, "a")
    store.add(CH_SCAN, 1.0, "b")
    store.add(CH_ODOM, 1.0, "c")
    assert store.all_times() == [1.0, 3.0]


def test_replayclock_empty_schedule():
    clock = ReplayClock([])
    assert clock.now() == 0.0
    assert clock.step() == 0.0
    assert clock.exhausted
