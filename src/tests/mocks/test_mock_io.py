"""MockRobotIO conformance to the RobotIO protocol + publish-sink behaviour."""
from __future__ import annotations

import numpy as np

from core.interfaces import (
    IntAnswer,
    LidarScan,
    MarkerBox,
    OdomState,
    PanoFrame,
    Question,
    RobotIO,
    TerrainPatch,
    WaypointCmd,
)
from core.mocks.mock_io import PANO_SHAPE, FakeClock, MockRobotIO
from core.mocks.synthetic_scene import SyntheticScene

# The RobotIO protocol methods MockRobotIO must structurally implement.
_ROBOTIO_METHODS = (
    "question",
    "latest_pano",
    "latest_scan",
    "latest_terrain",
    "latest_odom",
    "publish_waypoint",
    "publish_marker",
    "publish_int",
    "clock",
)


def _io(seed=0, n=3):
    scene = SyntheticScene(seed)
    scene.populate_default(n)
    return MockRobotIO(scene, FakeClock())


def test_satisfies_robotio_protocol():
    # RobotIO is not @runtime_checkable, so assert structural conformance by
    # checking every protocol member is a callable on the mock.
    io = _io()
    for name in _ROBOTIO_METHODS:
        assert callable(getattr(io, name, None)), f"missing RobotIO method: {name}"
    # membership names come from the frozen contract, not a private list
    assert set(_ROBOTIO_METHODS).issubset(dir(RobotIO))


def test_question_none_before_receipt_then_latched():
    io = _io()
    assert io.question() is None
    q = io.set_question("How many sofas are below a window?")
    assert isinstance(q, Question)
    assert io.question() is not None
    assert io.question().text == "How many sofas are below a window?"


def test_latest_pano_shape_and_black():
    io = _io()
    pano = io.latest_pano()
    assert isinstance(pano, PanoFrame)
    assert pano.image.shape == PANO_SHAPE
    assert pano.image.dtype == np.uint8
    assert int(pano.image.max()) == 0  # black
    assert isinstance(pano.odom, OdomState)


def test_latest_scan_points_from_instances():
    io = _io(seed=3, n=4)
    scan = io.latest_scan()
    assert isinstance(scan, LidarScan)
    assert scan.points.shape[1] == 3
    assert len(scan.points) > 0  # object-surface samples present
    assert scan.points.dtype == np.float32


def test_latest_scan_empty_scene():
    io = MockRobotIO(SyntheticScene(0), FakeClock())  # no objects
    scan = io.latest_scan()
    assert scan.points.shape == (0, 3)


def test_latest_terrain_default_and_extended():
    io = _io()
    near = io.latest_terrain()
    far = io.latest_terrain(extended=True)
    assert isinstance(near, TerrainPatch)
    assert near.extended is False
    assert far.extended is True


def test_odom_reflects_set_pose_and_clock():
    clock = FakeClock(start=5.0)
    io = MockRobotIO(SyntheticScene(0), clock)
    io.set_pose(1.5, 2.5, yaw=0.3)
    odom = io.latest_odom()
    assert (odom.x, odom.y, odom.yaw) == (1.5, 2.5, 0.3)
    assert odom.t == 5.0
    clock.advance(2.0)
    assert io.latest_odom().t == 7.0


def test_publish_sinks_record():
    io = _io()
    io.publish_waypoint(WaypointCmd(1.0, 2.0))
    io.publish_marker(MarkerBox(0, 0, 0, 1, 1, 1, label="sofa"))
    io.publish_int(IntAnswer(3))
    assert io.waypoints == [WaypointCmd(1.0, 2.0)]
    assert io.markers[0].label == "sofa"
    assert io.ints == [IntAnswer(3)]


def test_clock_getter_returns_injected_clock():
    clock = FakeClock()
    io = MockRobotIO(SyntheticScene(0), clock)
    assert io.clock() is clock
    assert io.clock().now() == 0.0
