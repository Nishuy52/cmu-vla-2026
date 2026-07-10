"""The FULL QuestionController runs against ReplayRobotIO backed by a bag.

This is the harness's reason to exist: recorded data drives the real FSM with no ROS.
Stubs stand in for the parse/explore/verify callables (those have their own suites); the
point here is that ReplayRobotIO satisfies everything the controller touches per tick and
that a numerical answer flows out to the publish sink.
"""
from __future__ import annotations

import numpy as np

from core.fsm.controller import QuestionController, WorldView
from core.interfaces import IntAnswer, QType
from core.plan_schema import Plan, TargetSpec
from core.replay.bag_reader import (
    TOPIC_CAMERA,
    TOPIC_ODOM,
    TOPIC_QUESTION,
    TOPIC_SCAN,
    TOPIC_TERRAIN,
)
from core.replay.replay_io import ReplayRobotIO
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


class _FakeScene:
    def all_instances(self):
        return []

    def by_label(self, noun):
        return []


def _bag(bag_factory, typestore):
    ts = typestore
    xyzi = np.array([[1.0, 1.0, 1.0, 0.0]], dtype=np.float32)
    recs = [
        (TOPIC_QUESTION, STR, make_string_msg(ts, "how many chairs are there"), 1_000_000_000),
        (TOPIC_ODOM, ODO, make_odom_msg(ts, 0, 0, 0, 0.0, sec=1), 1_000_000_000),
        (TOPIC_SCAN, PC2, make_pointcloud_msg(ts, xyzi, STD, sec=1), 1_000_000_000),
        (TOPIC_TERRAIN, PC2, make_pointcloud_msg(ts, xyzi, STD, sec=1), 1_000_000_000),
        (TOPIC_CAMERA, IMG, make_image_msg(ts, np.zeros((2, 2, 3), np.uint8), sec=1), 1_000_000_000),
    ]
    return bag_factory(recs)


def test_full_controller_ticks_over_replay(bag_factory, typestore):
    io = ReplayRobotIO.from_bag(_bag(bag_factory, typestore))
    io.clock().set(2.0)  # after every message

    world = WorldView(scene=_FakeScene())

    def parse(q):
        return Plan(qtype=QType.NUMERICAL, question_raw=q.text, target=TargetSpec(noun="chair"))

    def explore(io_, plan, w):
        # exercise the getters the way a real explorer would
        io_.latest_pano()
        io_.latest_scan()
        io_.latest_terrain(extended=False)
        io_.latest_odom()

    def verify(io_, plan, w):
        return IntAnswer(4)

    def probe(io_):
        return world

    ctrl = QuestionController(parse=parse, explore=explore, verify=verify, probe=probe)

    # Drive the FSM to completion; the watchdog floor guarantees termination.
    for _ in range(200):
        ctrl.tick(io)
        io.clock().set(io.clock().now() + 5.0)

    # A numerical answer was published through the replay IO sink.
    assert len(io.ints) >= 1
    assert isinstance(io.ints[0], IntAnswer)
