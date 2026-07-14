"""Tracker association + PerceptionPipeline glue over scripted multi-frame runs."""
from __future__ import annotations

import numpy as np
import pytest

from core.interfaces import LidarScan, OdomState, PanoFrame
from core.mocks.mock_io import MockRobotIO
from core.mocks.synthetic_scene import SyntheticScene
from core.perception import tiling as T
from core.perception.detector import Detection, FakeDetector
from core.perception.fusion import fuse_detection
from core.perception.scene_index import BasicSceneIndex
from core.perception.tracker import (
    KeyframeConfig,
    PerceptionPipeline,
    TrackerConfig,
    associate,
    canonical_for_match,
    labels_compatible,
)


def _front_det(label="sofa", score=0.9, tile_id=0):
    spec = T.tile_specs()[tile_id]
    bb = (spec.cx - 60, spec.cy - 120, spec.cx + 60, spec.cy + 60)
    return Detection(tile_id=tile_id, bbox_xyxy=bb, label=label, score=score)


def _box_cloud(cx, cy, cz, half=0.2, n=40, seed=0):
    rng = np.random.default_rng(seed)
    return np.column_stack([
        cx + rng.uniform(-half, half, n),
        cy + rng.uniform(-half, half, n),
        cz + rng.uniform(-half, half, n),
    ]).astype(np.float32)


def _odom(x=0.0, y=0.0, z=0.0, yaw=0.0):
    return OdomState(t=0.0, x=x, y=y, z=z, yaw=yaw)


def _fuse(det, cloud, odom):
    return fuse_detection(det, LidarScan(t=0.0, points=cloud), odom)


# ------------------------------------------------------------------ label compat


def test_labels_compatible_via_aliases():
    assert labels_compatible("tv", "television")
    assert labels_compatible("television", "tv")
    assert labels_compatible("fridge", "refrigerator")
    assert not labels_compatible("sofa", "chair")


def test_canonical_for_match_folds_aliases():
    assert canonical_for_match("television") == canonical_for_match("tv")
    assert canonical_for_match("fridge") == canonical_for_match("refrigerator")


# ------------------------------------------------------------------ associate()


def test_associate_creates_new_instance():
    idx = BasicSceneIndex()
    det = _front_det()
    fused = _fuse(det, _box_cloud(3.0, 0.0, 0.5), _odom())
    touched = associate([(det, fused)], idx)
    assert len(touched) == 1
    assert len(idx.all_instances()) == 1
    assert idx.all_instances()[0].label == "sofa"


def test_associate_matches_close_same_label():
    idx = BasicSceneIndex()
    det = _front_det()
    associate([(det, _fuse(det, _box_cloud(3.0, 0.0, 0.5, seed=1), _odom()))], idx)
    # second observation, ~0.1 m away -> matches, n_obs grows, no new instance
    fused2 = _fuse(det, _box_cloud(3.1, 0.05, 0.5, seed=2), _odom())
    touched = associate([(det, fused2)], idx)
    assert len(idx.all_instances()) == 1
    assert idx.all_instances()[0].n_obs == 2
    assert touched == [idx.all_instances()[0].instance_id]


def test_associate_new_when_beyond_gate():
    idx = BasicSceneIndex()
    det = _front_det()
    associate([(det, _fuse(det, _box_cloud(3.0, 0.0, 0.5), _odom()))], idx)
    # far detection (well beyond 0.75 m gate) -> new instance
    far_det = _front_det()
    fused = _fuse(far_det, _box_cloud(6.0, 0.0, 0.5, seed=3), _odom())
    associate([(far_det, fused)], idx)
    assert len(idx.all_instances()) == 2


def test_associate_new_when_label_incompatible():
    idx = BasicSceneIndex()
    det = _front_det("sofa")
    associate([(det, _fuse(det, _box_cloud(3.0, 0.0, 0.5), _odom()))], idx)
    # same location, different (incompatible) label -> separate instance
    chair = _front_det("chair")
    fused = _fuse(chair, _box_cloud(3.0, 0.0, 0.5, seed=4), _odom())
    associate([(chair, fused)], idx)
    assert len(idx.all_instances()) == 2


def test_associate_matches_across_alias_labels():
    idx = BasicSceneIndex()
    tv = _front_det("tv")
    associate([(tv, _fuse(tv, _box_cloud(3.0, 0.0, 0.5, seed=1), _odom()))], idx)
    tele = _front_det("television")
    fused = _fuse(tele, _box_cloud(3.05, 0.0, 0.5, seed=2), _odom())
    associate([(tele, fused)], idx)
    assert len(idx.all_instances()) == 1
    assert idx.all_instances()[0].n_obs == 2


def test_associate_greedy_nearest_pairing():
    """Two instances + two detections match to their nearest counterpart, not crossed."""
    idx = BasicSceneIndex()
    d = _front_det()
    associate([(d, _fuse(d, _box_cloud(3.0, 0.0, 0.5, seed=1), _odom()))], idx)
    associate([(d, _fuse(d, _box_cloud(3.0, 1.0, 0.5, seed=2), _odom()))], idx)
    assert len(idx.all_instances()) == 2
    f_a = _fuse(d, _box_cloud(3.05, 0.02, 0.5, seed=3), _odom())
    f_b = _fuse(d, _box_cloud(3.02, 0.98, 0.5, seed=4), _odom())
    associate([(d, f_a), (d, f_b)], idx)
    # still two instances, each got a 2nd obs (no double-matching)
    assert len(idx.all_instances()) == 2
    assert all(r.n_obs == 2 for r in idx.all_instances())


# ------------------------------------------------------------------ pipeline: 3 frames


@pytest.mark.slow
def test_pipeline_three_frames_grows_n_obs():
    """Same object seen across 3 scripted frames -> one instance, n_obs == 3."""
    det = _front_det()
    detector = FakeDetector(script=[[det], [det], [det]])
    pipe = PerceptionPipeline(
        detector, keyframe_cfg=KeyframeConfig(every_k=1),
    )
    ids = []
    for f in range(3):
        cloud = _box_cloud(3.0, 0.0, 0.5, seed=f)
        pano = PanoFrame(
            t=float(f),
            image=np.zeros((T.PANO_HEIGHT, T.PANO_WIDTH, 3), dtype=np.uint8),
            odom=_odom(x=0.1 * f),  # small motion, well under merge distance
        )
        scan = LidarScan(t=float(f), points=cloud)
        ids.append(pipe.process(pano, scan))
    assert len(pipe.index.all_instances()) == 1
    assert pipe.index.all_instances()[0].n_obs == 3
    assert all(len(i) == 1 for i in ids)


def test_pipeline_two_objects_two_instances():
    front = _front_det("sofa")
    rear_spec = T.tile_specs()[2]
    rear = Detection(2, (rear_spec.cx - 60, rear_spec.cy - 60,
                         rear_spec.cx + 60, rear_spec.cy + 60), "chair", 0.8)
    detector = FakeDetector([front, rear])
    pipe = PerceptionPipeline(detector, keyframe_cfg=KeyframeConfig(every_k=1))
    # front object at +x, rear object at -x
    cloud = np.vstack([
        _box_cloud(3.0, 0.0, 0.5, seed=1),
        _box_cloud(-3.0, 0.0, 0.0, seed=2),
    ]).astype(np.float32)
    pano = PanoFrame(t=0.0, image=np.zeros((T.PANO_HEIGHT, T.PANO_WIDTH, 3), np.uint8),
                     odom=_odom())
    pipe.process(pano, LidarScan(t=0.0, points=cloud))
    labels = sorted(r.label for r in pipe.index.all_instances())
    assert labels == ["chair", "sofa"]


# ------------------------------------------------------------------ keyframe gate


def test_keyframe_gate_skips_static_frames():
    det = _front_det()
    detector = FakeDetector([det])
    pipe = PerceptionPipeline(
        detector,
        keyframe_cfg=KeyframeConfig(every_k=10 ** 9, min_translation=0.5,
                                    min_rotation=np.deg2rad(30)),
    )
    cloud = _box_cloud(3.0, 0.0, 0.5)
    scan = LidarScan(t=0.0, points=cloud)
    img = np.zeros((T.PANO_HEIGHT, T.PANO_WIDTH, 3), dtype=np.uint8)
    # frame 0 is always a keyframe
    r0 = pipe.process(PanoFrame(0.0, img, _odom(0, 0)), scan)
    # frame 1 barely moved -> skipped
    r1 = pipe.process(PanoFrame(1.0, img, _odom(0.1, 0)), scan)
    assert len(r0) == 1
    assert r1 == []


def test_keyframe_gate_triggers_on_translation():
    det = _front_det()
    pipe = PerceptionPipeline(
        FakeDetector([det]),
        keyframe_cfg=KeyframeConfig(every_k=10 ** 9, min_translation=0.5,
                                    min_rotation=np.deg2rad(30)),
    )
    scan = LidarScan(t=0.0, points=_box_cloud(3.0, 0.0, 0.5))
    img = np.zeros((T.PANO_HEIGHT, T.PANO_WIDTH, 3), dtype=np.uint8)
    pipe.process(PanoFrame(0.0, img, _odom(0, 0)), scan)
    r = pipe.process(PanoFrame(1.0, img, _odom(1.0, 0)), scan)  # moved 1 m
    assert len(r) == 1


def test_keyframe_gate_triggers_on_rotation():
    det = _front_det()
    pipe = PerceptionPipeline(
        FakeDetector([det]),
        keyframe_cfg=KeyframeConfig(every_k=10 ** 9, min_translation=0.5,
                                    min_rotation=np.deg2rad(30)),
    )
    img = np.zeros((T.PANO_HEIGHT, T.PANO_WIDTH, 3), dtype=np.uint8)
    # frame 0: object straight ahead (map +x), vehicle yaw 0 -> fuses
    scan0 = LidarScan(t=0.0, points=_box_cloud(3.0, 0.0, 0.5, seed=1))
    pipe.process(PanoFrame(0.0, img, _odom(0, 0, yaw=0.0)), scan0)
    # frame 1: vehicle turned 45deg; put an object along the new heading (bearing 45deg)
    c = 3.0 / np.sqrt(2.0)
    scan1 = LidarScan(t=0.0, points=_box_cloud(c, c, 0.5, seed=2))
    r = pipe.process(PanoFrame(1.0, img, _odom(0, 0, yaw=np.deg2rad(45))), scan1)
    # rotation crossed the gate -> the frame was processed and the new object fused
    assert len(r) == 1


# ------------------------------------------------------------------ MockRobotIO smoke


def test_pipeline_smoke_on_mock_robot_io():
    """End-to-end smoke: MockRobotIO pano+scan frames through the pipeline."""
    scene = SyntheticScene(seed=2)
    scene.populate_default(3)
    io = MockRobotIO(scene, start_x=2.5, start_y=2.5)
    # scripted detection in each tile so at least one object is likely in-frustum
    dets = [_front_det("sofa", tile_id=t) for t in range(4)]
    pipe = PerceptionPipeline(FakeDetector(dets),
                              keyframe_cfg=KeyframeConfig(every_k=1))
    pano = io.latest_pano()
    scan = io.latest_scan()
    ids = pipe.process(pano, scan)
    # must not raise; returns a list; index reflects any fused instances
    assert isinstance(ids, list)
    assert len(pipe.index.all_instances()) == len(set(
        r.instance_id for r in pipe.index.all_instances()
    ))


def test_pipeline_empty_detections_noop():
    pipe = PerceptionPipeline(FakeDetector([]), keyframe_cfg=KeyframeConfig(every_k=1))
    img = np.zeros((T.PANO_HEIGHT, T.PANO_WIDTH, 3), dtype=np.uint8)
    scan = LidarScan(t=0.0, points=_box_cloud(3.0, 0.0, 0.5))
    ids = pipe.process(PanoFrame(0.0, img, _odom()), scan)
    assert ids == []
    assert pipe.index.all_instances() == []
