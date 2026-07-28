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


def test_associate_merges_across_label_variant_not_in_scene_index_synonym_map():
    """Issue #84 item 3: 'refridgerator' is only bridged to 'refrigerator' through
    core.parsing.vocab.NOUN_ALIASES (consulted by the tracker's canonical_for_match) --
    NOT through scene_index's own narrower _SYNONYM_MAP. Before the #89/#84 merge_into
    fix, associate() would correctly gate this pair as label-compatible (labels_compatible
    checks pass) but then hand off to index.add(), whose independent
    _find_merge_target re-derives label equality from scene_index.normalize_label ALONE
    -- missing the NOUN_ALIASES bridge -- so it disagreed and spawned a duplicate
    n_obs=1 ghost labelled 'refridgerator' instead of fusing into the existing
    'refrigerator' instance. Regression guard: must fuse into ONE instance."""
    idx = BasicSceneIndex()
    fridge = _front_det("refrigerator")
    associate([(fridge, _fuse(fridge, _box_cloud(3.0, 0.0, 0.5, seed=1), _odom()))], idx)
    variant = _front_det("refridgerator")
    fused = _fuse(variant, _box_cloud(3.05, 0.0, 0.5, seed=2), _odom())
    associate([(variant, fused)], idx)
    assert len(idx.all_instances()) == 1
    assert idx.all_instances()[0].n_obs == 2


def test_associate_merges_repeated_sightings_despite_jitter_defeating_iou():
    """Issue #89 repro: the tracker's own centroid-distance gate (0.75 m) can — and,
    under live pose jitter, routinely does — match a detection to an instance whose
    single-frame AABBs do NOT overlap (small objects' per-frame boxes are tiny, so even
    a sub-gate jitter easily drops IoU to 0). Before the merge_into fix, associate()'s
    own matched decision was silently overridden by index.add()'s independent, much
    stricter IoU>0.3 re-check, splitting one real object into a new instance on every
    jittered keyframe. Simulates 5 "sightings" of the same physical chair, each a tiny
    tight point cluster (so AABBs from consecutive sightings never overlap) but all
    within the association centroid gate of each other."""
    idx = BasicSceneIndex()
    centers = [(3.00, 0.00), (3.05, 0.30), (3.10, -0.25), (2.95, 0.35), (3.02, -0.30)]
    for i, (cx, cy) in enumerate(centers):
        det = _front_det("chair")
        # half=0.02 -> box width 0.04 m; consecutive centers are >0.2 m apart, so
        # consecutive AABBs never overlap (IoU == 0) even though every centroid stays
        # well inside the 0.75 m association gate of the very first sighting.
        fused = _fuse(det, _box_cloud(cx, cy, 0.5, half=0.02, seed=i), _odom())
        associate([(det, fused)], idx)
    assert len(idx.all_instances()) == 1  # one physical object, not 5 ghosts
    assert idx.all_instances()[0].n_obs == len(centers)


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


# ------------------------------------------------------------------ #84/#89 instance dump


def test_pipeline_dumps_instances_periodically_when_env_set(monkeypatch, tmp_path):
    from core.perception.scene_index import ENV_INSTANCE_DUMP_PATH

    out = tmp_path / "instances.jsonl"
    monkeypatch.setenv(ENV_INSTANCE_DUMP_PATH, str(out))
    det = _front_det()
    pipe = PerceptionPipeline(FakeDetector([det]), keyframe_cfg=KeyframeConfig(every_k=1))
    img = np.zeros((T.PANO_HEIGHT, T.PANO_WIDTH, 3), dtype=np.uint8)
    scan = LidarScan(t=0.0, points=_box_cloud(3.0, 0.0, 0.5))
    pipe.process(PanoFrame(0.0, img, _odom(0, 0)), scan)
    assert out.exists()
    assert len(out.read_text().strip().splitlines()) == 1


def test_pipeline_does_not_dump_without_env_var(monkeypatch, tmp_path):
    from core.perception.scene_index import ENV_INSTANCE_DUMP_PATH

    monkeypatch.delenv(ENV_INSTANCE_DUMP_PATH, raising=False)
    det = _front_det()
    pipe = PerceptionPipeline(FakeDetector([det]), keyframe_cfg=KeyframeConfig(every_k=1))
    img = np.zeros((T.PANO_HEIGHT, T.PANO_WIDTH, 3), dtype=np.uint8)
    scan = LidarScan(t=0.0, points=_box_cloud(3.0, 0.0, 0.5))
    pipe.process(PanoFrame(0.0, img, _odom(0, 0)), scan)
    assert list(tmp_path.iterdir()) == []


def test_pipeline_dump_is_throttled_by_interval(monkeypatch, tmp_path):
    from core.perception.scene_index import (
        ENV_INSTANCE_DUMP_INTERVAL_S,
        ENV_INSTANCE_DUMP_PATH,
    )

    out = tmp_path / "instances.jsonl"
    monkeypatch.setenv(ENV_INSTANCE_DUMP_PATH, str(out))
    monkeypatch.setenv(ENV_INSTANCE_DUMP_INTERVAL_S, "10.0")
    det = _front_det()
    pipe = PerceptionPipeline(FakeDetector([det]), keyframe_cfg=KeyframeConfig(every_k=1))
    img = np.zeros((T.PANO_HEIGHT, T.PANO_WIDTH, 3), dtype=np.uint8)
    scan = LidarScan(t=0.0, points=_box_cloud(3.0, 0.0, 0.5))
    pipe.process(PanoFrame(0.0, img, _odom(0, 0)), scan)
    pipe.process(PanoFrame(1.0, img, _odom(0.6, 0)), scan)  # t=1.0, well under 10s interval
    assert len(out.read_text().strip().splitlines()) == 1  # second tick throttled
    pipe.process(PanoFrame(11.0, img, _odom(1.2, 0)), scan)  # t=11.0, interval elapsed
    assert len(out.read_text().strip().splitlines()) == 2


def test_pipeline_empty_detections_noop():
    pipe = PerceptionPipeline(FakeDetector([]), keyframe_cfg=KeyframeConfig(every_k=1))
    img = np.zeros((T.PANO_HEIGHT, T.PANO_WIDTH, 3), dtype=np.uint8)
    scan = LidarScan(t=0.0, points=_box_cloud(3.0, 0.0, 0.5))
    ids = pipe.process(PanoFrame(0.0, img, _odom()), scan)
    assert ids == []
    assert pipe.index.all_instances() == []


# ------------------------------------------------------------------ #84 raw detection dump


def test_pipeline_dumps_raw_detections_when_env_set(monkeypatch, tmp_path):
    import json

    from core.perception.detector import ENV_RAW_DETECTION_DUMP_PATH, GATE_ACCEPTED

    out = tmp_path / "raw.jsonl"
    monkeypatch.setenv(ENV_RAW_DETECTION_DUMP_PATH, str(out))
    det = _front_det("sofa")
    pipe = PerceptionPipeline(FakeDetector([det]), keyframe_cfg=KeyframeConfig(every_k=1))
    img = np.zeros((T.PANO_HEIGHT, T.PANO_WIDTH, 3), dtype=np.uint8)
    scan = LidarScan(t=0.0, points=_box_cloud(3.0, 0.0, 0.5))
    pipe.process(PanoFrame(0.0, img, _odom()), scan)
    assert out.exists()
    lines = out.read_text().strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["total_detections"] == 1
    d = record["detections"][0]
    assert d["label"] == "sofa"
    assert d["gate"] == GATE_ACCEPTED
    assert d["instance_id"] == pipe.index.all_instances()[0].instance_id


def test_pipeline_raw_dump_records_gated_detection(monkeypatch, tmp_path):
    """A detection whose bbox has no lidar cluster nearby is gated (fusion rejects it,
    min_points floor) and never reaches the scene index, but still shows up in the raw
    dump with GATE_NO_LIDAR_CLUSTER and instance_id None."""
    import json

    from core.perception.detector import ENV_RAW_DETECTION_DUMP_PATH, GATE_NO_LIDAR_CLUSTER

    out = tmp_path / "raw.jsonl"
    monkeypatch.setenv(ENV_RAW_DETECTION_DUMP_PATH, str(out))
    det = _front_det("window")  # scripted detection with NO lidar cloud anywhere nearby
    pipe = PerceptionPipeline(FakeDetector([det]), keyframe_cfg=KeyframeConfig(every_k=1))
    img = np.zeros((T.PANO_HEIGHT, T.PANO_WIDTH, 3), dtype=np.uint8)
    # cloud far away from the detection's frustum -> fuse_detection returns None
    scan = LidarScan(t=0.0, points=_box_cloud(-30.0, -30.0, -5.0))
    pipe.process(PanoFrame(0.0, img, _odom()), scan)
    assert pipe.index.all_instances() == []  # never entered the scene index
    record = json.loads(out.read_text().strip())
    assert record["total_detections"] == 1
    d = record["detections"][0]
    assert d["label"] == "window"
    assert d["gate"] == GATE_NO_LIDAR_CLUSTER
    assert d["instance_id"] is None
    assert record["by_class"]["window"] == {"total": 1, "accepted": 0, "gated": 1}


def test_pipeline_does_not_dump_raw_detections_without_env_var(monkeypatch, tmp_path):
    from core.perception.detector import ENV_RAW_DETECTION_DUMP_PATH

    monkeypatch.delenv(ENV_RAW_DETECTION_DUMP_PATH, raising=False)
    det = _front_det("sofa")
    pipe = PerceptionPipeline(FakeDetector([det]), keyframe_cfg=KeyframeConfig(every_k=1))
    img = np.zeros((T.PANO_HEIGHT, T.PANO_WIDTH, 3), dtype=np.uint8)
    scan = LidarScan(t=0.0, points=_box_cloud(3.0, 0.0, 0.5))
    pipe.process(PanoFrame(0.0, img, _odom()), scan)
    assert list(tmp_path.iterdir()) == []


# ------------------------------------------------------------------ #89 label folding


def test_labels_compatible_folds_subphrase_fragments():
    assert labels_compatible("potted", "potted plant")
    assert labels_compatible("plant", "potted plant")
    assert labels_compatible("door door", "door")
    assert labels_compatible("door door frame", "door")
    assert labels_compatible("screen projector screen", "projector screen")


def test_labels_compatible_guards_against_over_merge():
    assert not labels_compatible("door", "floor")
    assert not labels_compatible("chair", "table")
    assert not labels_compatible("potted", "plant")  # neither is a subset of the other


def test_associate_folds_subphrase_fragment_into_established_instance():
    """Issue #89 repro: 'potted plant' seen first, then a bare 'potted' fragment of the
    SAME object nearby must fuse into it rather than spawning a 2nd instance."""
    idx = BasicSceneIndex()
    plant = _front_det("potted plant")
    associate([(plant, _fuse(plant, _box_cloud(3.0, 0.0, 0.5, seed=1), _odom()))], idx)
    fragment = _front_det("potted")
    fused = _fuse(fragment, _box_cloud(3.05, 0.0, 0.5, seed=2), _odom())
    associate([(fragment, fused)], idx)
    assert len(idx.all_instances()) == 1
    assert idx.all_instances()[0].n_obs == 2
    assert idx.all_instances()[0].label == "potted plant"  # established label kept


def test_associate_folds_duplicated_token_fragment_into_established_instance():
    """'door door' (duplicate-token GDINO phrase-decode artifact) must fuse into an
    established 'door' instance rather than spawning a ghost."""
    idx = BasicSceneIndex()
    door = _front_det("door")
    associate([(door, _fuse(door, _box_cloud(3.0, 0.0, 0.5, seed=1), _odom()))], idx)
    dup = _front_det("door door")
    fused = _fuse(dup, _box_cloud(3.05, 0.0, 0.5, seed=2), _odom())
    associate([(dup, fused)], idx)
    assert len(idx.all_instances()) == 1
    assert idx.all_instances()[0].n_obs == 2


def test_associate_does_not_fold_door_and_floor():
    """Guard: 'door' and 'floor' share no tokens and must never fold/merge even when
    co-located."""
    idx = BasicSceneIndex()
    door = _front_det("door")
    associate([(door, _fuse(door, _box_cloud(3.0, 0.0, 0.5, seed=1), _odom()))], idx)
    floor = _front_det("floor")
    fused = _fuse(floor, _box_cloud(3.02, 0.0, 0.5, seed=2), _odom())
    associate([(floor, fused)], idx)
    assert len(idx.all_instances()) == 2


def test_associate_does_not_fold_chair_and_table():
    idx = BasicSceneIndex()
    chair = _front_det("chair")
    associate([(chair, _fuse(chair, _box_cloud(3.0, 0.0, 0.5, seed=1), _odom()))], idx)
    table = _front_det("table")
    fused = _fuse(table, _box_cloud(3.02, 0.0, 0.5, seed=2), _odom())
    associate([(table, fused)], idx)
    assert len(idx.all_instances()) == 2


# ------------------------------------------------------------ degenerate AABB (#125)


def _flat_cloud(cx, cy, cz, half=0.2, n=40, seed=0):
    """A cluster that is exactly flat on z (a front-shell-only sighting)."""
    rng = np.random.default_rng(seed)
    return np.column_stack([
        cx + rng.uniform(-half, half, n),
        cy + rng.uniform(-half, half, n),
        np.full(n, cz),
    ]).astype(np.float32)


def test_new_instance_with_flat_cloud_gets_floored_z_extent():
    idx = BasicSceneIndex()
    chair = _front_det("chair")
    fused = _fuse(chair, _flat_cloud(3.0, 0.0, 0.5, seed=1), _odom())
    associate([(chair, fused)], idx)
    rec = idx.all_instances()[0]
    assert rec.aabb_max[2] - rec.aabb_min[2] > 0.0
    # centre (z) of the floored box is preserved at the flat cloud's z.
    assert abs((rec.aabb_min[2] + rec.aabb_max[2]) / 2.0 - 0.5) < 1e-6


def test_merged_instance_stays_floored_when_both_sightings_flat():
    idx = BasicSceneIndex()
    chair = _front_det("chair")
    fused1 = _fuse(chair, _flat_cloud(3.0, 0.0, 0.5, seed=1), _odom())
    associate([(chair, fused1)], idx)
    fused2 = _fuse(chair, _flat_cloud(3.02, 0.0, 0.5, seed=2), _odom())
    associate([(chair, fused2)], idx)
    rec = idx.all_instances()[0]
    assert rec.n_obs == 2
    assert rec.aabb_max[2] - rec.aabb_min[2] > 0.0
