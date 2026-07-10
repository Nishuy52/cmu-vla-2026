"""run_question with --detections: scripted 2D boxes + real scans -> 3D instances.

A tiny synthetic replay store (a handful of keyframes, each with pano+scan+odom) plus a
synthetic labels JSON. The scripted detections are placed at known azimuths and a matching
lidar cluster is dropped along each box's ray, so fusion produces a predictable number of
instances the heads then answer over. Also pins the CLI --detections/--fixtures validation.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from core.interfaces import (
    LidarScan,
    OdomState,
    PanoFrame,
    Question,
    QType,
    TerrainPatch,
    IntAnswer,
)
from core.perception import tiling as T
from core.perception.scripted import pano_bbox_to_detection
from core.replay.replay_io import (
    CH_ODOM,
    CH_PANO,
    CH_QUESTION,
    CH_SCAN,
    CH_TERRAIN,
    CH_TERRAIN_EXT,
    MessageStore,
    ReplayRobotIO,
)
from core.runner.single import run_question

_N_KF = 6  # keyframes in the synthetic bag


def _cluster_along_box(box, tile_id, yaw, rng, r=3.0, n=50):
    """A lidar cluster (map frame) at range r along the box centre's map ray."""
    det = pano_bbox_to_detection("x", box, 0.9, T.tile_specs())
    u = 0.5 * (det.bbox_xyxy[0] + det.bbox_xyxy[2])
    v = 0.5 * (det.bbox_xyxy[1] + det.bbox_xyxy[3])
    b, e = T.tile_pixel_to_map_ray(det.tile_id, u, v, yaw)
    center = np.array([r * np.cos(b), r * np.sin(b), r * np.sin(e)])
    return (center + rng.normal(0.0, 0.04, (n, 3))).astype(np.float32)


def _col_for_az(az_deg):
    return float(T.azimuth_to_column(np.deg2rad(az_deg)))


def _boxes_two_stools():
    """Two stool boxes at distinct front azimuths (both tile 0), well separated in 3D."""
    b1 = (_col_for_az(-25.0) - 20, 300, _col_for_az(-25.0) + 20, 360)
    b2 = (_col_for_az(25.0) - 20, 300, _col_for_az(25.0) + 20, 360)
    return [b1, b2]


def _build_store_and_labels(tmp_path, question, boxes, label="stool"):
    """A ~6-keyframe store; keyframe 0 carries the labelled boxes + a matching scan.

    Returns (store, labels_path). The scan holds one cluster per box so each box fuses to
    one 3D instance; later keyframes repeat the same pano/scan so a keyframe-change still
    only re-observes the same instances (they merge, not multiply).
    """
    rng = np.random.default_rng(1)
    clusters = [_cluster_along_box(b, 0, 0.0, rng) for b in boxes]
    scan_pts = np.vstack(clusters).astype(np.float32)
    # a couple of stray far points so the scene is not purely the clusters
    scan_pts = np.vstack([scan_pts, np.array([[12.0, 12.0, 0.0]], np.float32)])

    pano_img = np.zeros((T.PANO_HEIGHT, T.PANO_WIDTH, 3), dtype=np.uint8)

    store = MessageStore()
    store.add(CH_QUESTION, 1.0, Question(text=question, t_received=1.0))
    for i in range(_N_KF):
        t = float(i + 1)
        odom = OdomState(t=t, x=0.0, y=0.0, z=0.0, yaw=0.0)
        store.add(CH_ODOM, t, odom)
        store.add(CH_SCAN, t, LidarScan(t=t, points=scan_pts))
        store.add(CH_TERRAIN, t, TerrainPatch(t=t, points=np.zeros((4, 4), np.float32), extended=False))
        store.add(CH_TERRAIN_EXT, t, TerrainPatch(t=t, points=np.zeros((4, 4), np.float32), extended=True))
        store.add(CH_PANO, t, PanoFrame(t=t, image=pano_img, odom=odom))
    store.finalize()

    # Label ONLY keyframe 0 (position 0 in sorted pano order) with the boxes.
    labels = {
        "format": "pano_xyxy",
        "keyframes": {
            "0": [
                {"label": label, "attributes": ["white"], "bbox": list(b), "score": 0.9}
                for b in boxes
            ]
        },
    }
    p = tmp_path / "labels.json"
    p.write_text(json.dumps(labels))
    return store, str(p)


def _io(store):
    io = ReplayRobotIO(store)
    io.raw_clock().set(store.times(CH_PANO)[0])
    return io


# --------------------------------------------------------------- wiring


def test_scripted_detections_produce_tracked_instances(tmp_path):
    store, labels = _build_store_and_labels(tmp_path, "How many stools are in the room?", _boxes_two_stools())
    r = run_question(
        "How many stools are in the room?", _io(store),
        detections_path=labels, tick_hz=5.0, budget_scale=0.05,
    )
    # Two distinct scripted stools fuse into two 3D instances in the pipeline's scene index.
    assert r.instances_tracked == 2


def test_count_question_answers_scripted_count(tmp_path):
    store, labels = _build_store_and_labels(tmp_path, "How many stools are in the room?", _boxes_two_stools())
    r = run_question(
        "How many stools are in the room?", _io(store),
        detections_path=labels, tick_hz=5.0, budget_scale=0.05,
    )
    assert r.qtype is QType.NUMERICAL
    assert isinstance(r.answer, IntAnswer)
    assert r.answer.value == 2  # the scripted count, grounded on the synthetic scan
    assert r.floor_used is False  # answered from a real head verify, not the floor


def test_instances_are_the_index_the_heads_resolve(tmp_path):
    """A single scripted stool -> count of 1 (proves the pipeline index feeds the heads)."""
    one = [( _col_for_az(0.0) - 20, 300, _col_for_az(0.0) + 20, 360)]
    store, labels = _build_store_and_labels(tmp_path, "How many stools are in the room?", one)
    r = run_question(
        "How many stools are in the room?", _io(store),
        detections_path=labels, tick_hz=5.0, budget_scale=0.05,
    )
    assert r.instances_tracked == 1
    assert isinstance(r.answer, IntAnswer)
    assert r.answer.value == 1


def test_repeated_keyframes_merge_not_multiply(tmp_path):
    """The same stool re-observed across keyframes merges (n_obs up), never doubles the
    instance count — the tracker/scene-index association is exercised end-to-end."""
    store, labels = _build_store_and_labels(tmp_path, "How many stools are in the room?", _boxes_two_stools())
    # Also label a later keyframe with the SAME boxes so it re-observes the same objects.
    with open(labels) as fh:
        data = json.load(fh)
    data["keyframes"]["2"] = data["keyframes"]["0"]
    with open(labels, "w") as fh:
        json.dump(data, fh)
    r = run_question(
        "How many stools are in the room?", _io(store),
        detections_path=labels, tick_hz=5.0, budget_scale=0.05,
    )
    assert r.instances_tracked == 2  # re-observation merged, not duplicated


def test_no_detections_path_leaves_instances_zero(tmp_path):
    """Without --detections the pipeline is not built; instances_tracked stays 0."""
    store, _ = _build_store_and_labels(tmp_path, "How many stools are in the room?", _boxes_two_stools())
    r = run_question(
        "How many stools are in the room?", _io(store),
        tick_hz=5.0, budget_scale=0.05,
    )
    assert r.instances_tracked == 0


def test_detections_path_requires_replay_io():
    """A non-replay io (no pano/scan stream) with detections_path errors cleanly."""
    from core.mocks.mock_io import FakeClock, MockRobotIO
    from core.runner.scenegen import build_scene_for

    sc, _ = build_scene_for("_t_", {"numerical": ["How many stools are in the room?"]}, seed=0)
    io = MockRobotIO(sc, FakeClock(0.0))
    with pytest.raises(ValueError):
        run_question(
            "How many stools are in the room?", io,
            detections_path="whatever.json",
        )
