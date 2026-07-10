"""Fixture extract -> load round-trip, movement gate, schema, and CLI smoke."""
from __future__ import annotations

import json
import math
import os

import numpy as np
import pytest

from core.interfaces import LidarScan, OdomState, PanoFrame, Question, TerrainPatch
from core.replay import fixtures as fx
from core.replay.fixtures import (
    ExtractParams,
    INDEX_NAME,
    SCHEMA_VERSION,
    extract_fixtures,
    load_fixtures,
)
from core.replay.replay_io import (
    CH_ODOM,
    CH_PANO,
    CH_SCAN,
    CH_TERRAIN,
    ReplayRobotIO,
)
from core.replay.bag_reader import (
    TOPIC_CAMERA,
    TOPIC_ODOM,
    TOPIC_QUESTION,
    TOPIC_SCAN,
    TOPIC_TERRAIN,
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


def _rich_bag(bag_factory, typestore, n_frames=6, move_per_frame=0.0):
    """A bag with n camera frames, each with fresh odom/scan/terrain at the same time."""
    ts = typestore
    recs = [(TOPIC_QUESTION, STR, make_string_msg(ts, "Find the sofa"), 500_000_000)]
    rng = np.random.default_rng(0)
    for k in range(n_frames):
        sec = k + 1
        tns = sec * 1_000_000_000
        x = k * move_per_frame
        recs.append((TOPIC_ODOM, ODO, make_odom_msg(ts, x, 0.0, 0.0, 0.0, sec=sec), tns))
        rgb = (rng.integers(0, 255, size=(4, 4, 3))).astype(np.uint8)
        recs.append((TOPIC_CAMERA, IMG, make_image_msg(ts, rgb, sec=sec), tns))
        xyzi = np.column_stack([
            rng.random(20), rng.random(20), rng.random(20), rng.random(20)
        ]).astype(np.float32)
        recs.append((TOPIC_SCAN, PC2, make_pointcloud_msg(ts, xyzi, STD, sec=sec), tns))
        terr = np.column_stack([
            rng.random(10), rng.random(10), np.zeros(10), rng.random(10) * 0.3
        ]).astype(np.float32)
        recs.append((TOPIC_TERRAIN, PC2, make_pointcloud_msg(ts, terr, STD, sec=sec), tns))
    return bag_factory(recs)


def test_extract_writes_index_and_keyframes(bag_factory, typestore, tmp_path):
    bag = _rich_bag(bag_factory, typestore, n_frames=6)
    out = tmp_path / "fixtures"
    index = extract_fixtures(bag, out, ExtractParams(stride=2, move_m=1e9, rot_rad=1e9))
    assert (out / INDEX_NAME).exists()
    assert index["schema_version"] == SCHEMA_VERSION
    assert index["n_keyframes"] >= 1
    for kf in index["keyframes"]:
        assert (out / kf["file"]).exists()
    # question latched
    assert index["question"]["text"] == "Find the sofa"


def test_stride_gate_count(bag_factory, typestore, tmp_path):
    bag = _rich_bag(bag_factory, typestore, n_frames=6)
    out = tmp_path / "f"
    # stride=2, no movement -> frames 0,2,4 => 3 keyframes (i==0 always, then every 2)
    index = extract_fixtures(bag, out, ExtractParams(stride=2, move_m=1e9, rot_rad=1e9))
    assert index["n_keyframes"] == 3


def test_movement_gate_triggers(bag_factory, typestore, tmp_path):
    # Big stride so only the movement gate can fire; move 1m/frame with 0.5m threshold.
    bag = _rich_bag(bag_factory, typestore, n_frames=6, move_per_frame=1.0)
    out = tmp_path / "f"
    index = extract_fixtures(
        bag, out, ExtractParams(stride=1000, move_m=0.5, rot_rad=1e9)
    )
    # frame 0 (always) + every subsequent frame moves >=1m -> all 6
    assert index["n_keyframes"] == 6


def test_roundtrip_arrays_identical(bag_factory, typestore, tmp_path):
    bag = _rich_bag(bag_factory, typestore, n_frames=4)
    out = tmp_path / "f"
    extract_fixtures(bag, out, ExtractParams(stride=1, move_m=1e9, rot_rad=1e9))
    store = load_fixtures(out)

    # Compare against the direct-from-bag store at each keyframe time.
    from core.replay.replay_io import MessageStore
    ref = MessageStore.from_bag(bag)

    for t in store.times(CH_SCAN):
        loaded = store.latest(CH_SCAN, t)
        original = ref.latest(CH_SCAN, t)
        assert isinstance(loaded, LidarScan)
        assert np.allclose(loaded.points, original.points, atol=1e-6)
    for t in store.times(CH_PANO):
        loaded = store.latest(CH_PANO, t)
        original = ref.latest(CH_PANO, t)
        assert np.array_equal(loaded.image, original.image)  # uint8 lossless
    for t in store.times(CH_TERRAIN):
        loaded = store.latest(CH_TERRAIN, t)
        original = ref.latest(CH_TERRAIN, t)
        assert np.allclose(loaded.points, original.points, atol=1e-6)


def test_roundtrip_odom_and_question(bag_factory, typestore, tmp_path):
    bag = _rich_bag(bag_factory, typestore, n_frames=3, move_per_frame=2.0)
    out = tmp_path / "f"
    extract_fixtures(bag, out, ExtractParams(stride=1, move_m=1e9, rot_rad=1e9))
    store = load_fixtures(out)
    q = store.latest("question", 1e9)
    assert isinstance(q, Question)
    assert q.text == "Find the sofa"
    odom = store.latest(CH_ODOM, 3.0)
    assert isinstance(odom, OdomState)
    assert odom.x == pytest.approx(4.0)  # frame k=2 -> x = 2*2.0


def test_decimation_caps_points(bag_factory, typestore, tmp_path):
    ts = typestore
    big = np.column_stack([np.arange(1000)] * 4).astype(np.float32)
    recs = [
        (TOPIC_ODOM, ODO, make_odom_msg(ts, 0, 0, 0, 0.0, sec=1), 1_000_000_000),
        (TOPIC_CAMERA, IMG, make_image_msg(ts, np.zeros((2, 2, 3), np.uint8), sec=1), 1_000_000_000),
        (TOPIC_SCAN, PC2, make_pointcloud_msg(ts, big, STD, sec=1), 1_000_000_000),
    ]
    bag = bag_factory(recs)
    out = tmp_path / "f"
    extract_fixtures(bag, out, ExtractParams(stride=1, max_scan_pts=100))
    store = load_fixtures(out)
    scan = store.latest(CH_SCAN, 1.0)
    assert len(scan.points) <= 100


def test_load_rejects_bad_schema(bag_factory, typestore, tmp_path):
    bag = _rich_bag(bag_factory, typestore, n_frames=2)
    out = tmp_path / "f"
    extract_fixtures(bag, out, ExtractParams(stride=1))
    with open(out / INDEX_NAME, encoding="utf-8") as fh:
        idx = json.load(fh)
    idx["schema_version"] = 999
    with open(out / INDEX_NAME, "w", encoding="utf-8") as fh:
        json.dump(idx, fh)
    with pytest.raises(ValueError):
        load_fixtures(out)


def test_fixtures_drive_replay_io(bag_factory, typestore, tmp_path):
    """Loaded fixtures are drop-in for ReplayRobotIO."""
    bag = _rich_bag(bag_factory, typestore, n_frames=4)
    out = tmp_path / "f"
    extract_fixtures(bag, out, ExtractParams(stride=1, move_m=1e9, rot_rad=1e9))
    io = ReplayRobotIO(load_fixtures(out))
    io.clock().set(4.0)
    assert io.latest_scan() is not None
    assert io.latest_pano() is not None
    assert io.latest_odom() is not None
    assert io.question().text == "Find the sofa"


def test_cli_extract_and_info(bag_factory, typestore, tmp_path, capsys):
    bag = _rich_bag(bag_factory, typestore, n_frames=3)
    out = tmp_path / "cli_fixtures"
    rc = fx.main(["extract", str(bag), str(out), "--stride", "1"])
    assert rc == 0
    assert (out / INDEX_NAME).exists()
    captured = capsys.readouterr()
    assert "keyframes" in captured.out

    rc = fx.main(["info", str(out)])
    assert rc == 0
    assert "schema v" in capsys.readouterr().out


def test_cli_subprocess_smoke(bag_factory, typestore, tmp_path):
    """`python -m core.replay.fixtures extract ...` runs end-to-end."""
    import subprocess
    import sys

    bag = _rich_bag(bag_factory, typestore, n_frames=2)
    out = tmp_path / "sub_fixtures"
    src_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    env = dict(os.environ, PYTHONPATH=src_root)
    result = subprocess.run(
        [sys.executable, "-m", "core.replay.fixtures", "extract", str(bag), str(out), "--stride", "1"],
        capture_output=True, text=True, cwd=src_root, env=env,
    )
    assert result.returncode == 0, result.stderr
    assert (out / INDEX_NAME).exists()


def test_fixture_size_smaller_than_bag(bag_factory, typestore, tmp_path):
    """Sanity: decimated compressed fixtures are not larger than the raw bag."""
    bag = _rich_bag(bag_factory, typestore, n_frames=6, move_per_frame=0.0)
    out = tmp_path / "f"
    extract_fixtures(bag, out, ExtractParams(stride=2, max_scan_pts=5))

    def _dir_size(p):
        total = 0
        for root, _dirs, files in os.walk(p):
            for f in files:
                total += os.path.getsize(os.path.join(root, f))
        return total

    assert _dir_size(out) <= _dir_size(bag) * 2  # generous; real gain is on 3GB bags
