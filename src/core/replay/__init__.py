"""Replay harness: ROS 2 bag files -> core dataclasses -> reusable fixtures.

Pure Python (the ``rosbags`` pip package); no ROS installation required. Lets the full
:class:`core.fsm.QuestionController` run against recorded data offline, and distils large
bags into compact, machine-portable ``.npz`` fixtures.

Public surface:
    - :class:`~core.replay.bag_reader.BagSource` + the converter registry
    - :class:`~core.replay.replay_io.ReplayRobotIO`, :class:`~core.replay.replay_io.ReplayClock`
    - :func:`~core.replay.fixtures.extract_fixtures`, :func:`~core.replay.fixtures.load_fixtures`
"""
from __future__ import annotations

from core.replay.bag_reader import (
    BagSource,
    DEFAULT_TOPIC_MAP,
    image_to_pano,
    odom_to_state,
    pointcloud_to_lidar,
    pointcloud_to_terrain,
    string_to_question,
)
from core.replay.replay_io import MessageStore, ReplayClock, ReplayRobotIO
from core.replay.fixtures import extract_fixtures, load_fixtures

__all__ = [
    "BagSource",
    "DEFAULT_TOPIC_MAP",
    "image_to_pano",
    "odom_to_state",
    "pointcloud_to_lidar",
    "pointcloud_to_terrain",
    "string_to_question",
    "MessageStore",
    "ReplayClock",
    "ReplayRobotIO",
    "extract_fixtures",
    "load_fixtures",
]
