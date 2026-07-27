"""Issues #94 (under-segmentation) and #89 (over-segmentation) through the real
``associate()`` path -- see LOG.md / the task brief for the live-run numbers these
mirror (112 accepted monitor detections of 6 physical monitors fused to n_obs=110
centroid z=1.86 vs GT 1.06; 141/78 chairs against single-digit truth).

Both are symptoms of the SAME wrong criterion: one fixed-radius centroid gate,
checked only against the pre-batch instance snapshot (:func:`core.perception.tracker.
associate`, pre-fix). The fix (see that module's docstrings) makes the gate
per-class (:func:`core.perception.tracker._assoc_gate`, scaled off the
data-derived dimension prior), folds a batch sequentially so same-frame duplicate
detections of one physical object can find each other, and vetoes any match whose
resulting box would grossly exceed the class's typical size
(:func:`core.perception.tracker._match_plausible`).

The det/cloud helpers below build a bbox around an ARBITRARY map-frame target point
(not just along-range as the existing ``test_tracker.py`` fixtures do), because #94's
row of monitors needs bearing offsets the fixed ``_front_det`` bbox cannot reach.
"""
from __future__ import annotations

import numpy as np
import pytest

from core.interfaces import LidarScan, OdomState
from core.perception import tiling as T
from core.perception.detector import Detection
from core.perception.fusion import fuse_detection
from core.perception.scene_index import BasicSceneIndex
from core.perception.tracker import TrackerConfig, associate

# --------------------------------------------------------------------------- helpers


def _det_for_point(x, y, z, od, label, score=0.9, tile_id=0, half_px=45):
    """Build a Detection whose bbox is centred on map point (x, y, z) as seen from
    ``od`` -- the inverse of the fusion frustum maths, so (unlike the fixed
    ``_front_det`` bbox in test_tracker.py) it can target any bearing, not just
    straight ahead."""
    apex = np.array([od.x, od.y, od.z])
    dx, dy, dz = x - apex[0], y - apex[1], z - apex[2]
    horiz = np.hypot(dx, dy)
    bearing = np.arctan2(dy, dx)
    elevation = np.arctan2(dz, horiz)
    az, el = T.map_ray_to_camera(bearing, elevation, od.yaw)
    spec = T.tile_specs()[tile_id]
    local_az = az - spec.yaw_center
    xcam = -np.tan(local_az)
    ycam = -np.tan(el) * np.hypot(xcam, 1.0)
    u = xcam * spec.focal_x + spec.cx
    v = ycam * spec.focal_y + spec.cy
    bb = (u - half_px, v - half_px, u + half_px, v + half_px)
    return Detection(tile_id=tile_id, bbox_xyxy=bb, label=label, score=score)


def _monitor_cloud(cx, cy, cz, half=0.08, n=40, seed=0):
    rng = np.random.default_rng(seed)
    return np.column_stack([
        cx + rng.uniform(-half, half, n),
        cy + rng.uniform(-half, half, n),
        cz + rng.uniform(-half, half, n),
    ]).astype(np.float32)


def _chair_cloud(cx, cy, cz, n=60, seed=0):
    """Realistic chair-shaped cluster (footprint/height per the dimension prior:
    typ_ext == (0.583, 0.652, 1.047)), not an isotropic cube -- an isotropic box lets
    horizontal growth hide under the class's (much larger) height threshold and
    defeats the extent veto for the wrong reason."""
    rng = np.random.default_rng(seed)
    return np.column_stack([
        cx + rng.uniform(-0.29, 0.29, n),
        cy + rng.uniform(-0.326, 0.326, n),
        cz + rng.uniform(-0.05, 0.5, n),
    ]).astype(np.float32)


def _odom(x=0.0, y=0.0, z=0.0, yaw=0.0):
    return OdomState(t=0.0, x=x, y=y, z=z, yaw=yaw)


def _fuse(det, cloud, od):
    return fuse_detection(det, LidarScan(t=0.0, points=cloud), od)


# --------------------------------------------------------------------------- #94: under-segmentation


def test_issue_94_monitor_row_stays_six_instances_post_fix():
    """6 physical monitors at realistic 0.5 m desk spacing, each seen 3x across
    keyframes -> must stay 6 tracked instances (not fused toward 1 like the live
    112-detection/n_obs=110 report). Pre-fix this collapsed to 3 instances (see
    test_issue_94_monitor_row_PRE_FIX_repro_numbers below for the recorded number)."""
    od = _odom()
    idx = BasicSceneIndex()
    monitor_y = [i * 0.5 for i in range(6)]
    for rep in range(3):
        for i, y in enumerate(monitor_y):
            det = _det_for_point(3.0, y, 1.0, od, "monitor")
            cloud = _monitor_cloud(3.0, y, 1.0, seed=rep * 10 + i)
            fused = _fuse(det, cloud, od)
            assert fused is not None
            associate([(det, fused)], idx)
    insts = idx.all_instances()
    assert len(insts) == 6
    assert all(r.n_obs == 3 for r in insts)


def test_issue_94_monitor_row_PRE_FIX_repro_numbers():
    """Pins the PRE-FIX failure number so the regression is documented even though
    the behaviour it names no longer reproduces post-fix: with the flat 0.75 m
    centroid-only gate and no extent veto, this exact scenario collapsed 6 physical
    monitors to 3 tracked instances (n_obs=6 each). Reproduced here with the OLD
    criterion reimplemented locally (not by calling the fixed associate()) so this
    test keeps documenting the pre-fix numbers regardless of future tracker changes."""
    from core.perception.tracker import _fused_to_record, labels_compatible

    def pre_fix_associate(fused_dets, index, gate=0.75):
        existing = index.all_instances()
        pairs = []
        for di, (det, fused) in enumerate(fused_dets):
            for ej, inst in enumerate(existing):
                if not labels_compatible(det.label, inst.label):
                    continue
                dist = float(np.linalg.norm(fused.centroid - inst.centroid))
                if dist <= gate:
                    pairs.append((dist, di, ej))
        pairs.sort(key=lambda p: p[0])
        matched, used = {}, set()
        for dist, di, ej in pairs:
            if di in matched or ej in used:
                continue
            matched[di] = existing[ej]
            used.add(ej)
        touched = []
        for di, (det, fused) in enumerate(fused_dets):
            if di in matched:
                target = matched[di]
                rec = _fused_to_record(det, fused, instance_id=target.instance_id)
                survivor = index.merge_into(target.instance_id, rec)
            else:
                new_id = index.next_id()
                rec = _fused_to_record(det, fused, instance_id=new_id)
                survivor = index.add(rec)
            touched.append(survivor.instance_id)
        return touched

    od = _odom()
    idx = BasicSceneIndex()
    monitor_y = [i * 0.5 for i in range(6)]
    for rep in range(3):
        for i, y in enumerate(monitor_y):
            det = _det_for_point(3.0, y, 1.0, od, "monitor")
            cloud = _monitor_cloud(3.0, y, 1.0, seed=rep * 10 + i)
            fused = _fuse(det, cloud, od)
            pre_fix_associate([(det, fused)], idx)
    insts = idx.all_instances()
    assert len(insts) == 3  # 6 physical monitors collapsed to 3 -- the #94 defect
    assert all(r.n_obs == 6 for r in insts)


# --------------------------------------------------------------------------- #89: over-segmentation


def test_issue_89_same_frame_duplicate_detections_fold_to_one_instance():
    """One physical chair, detected TWICE in the SAME keyframe (e.g. an
    overlapping-tile duplicate proposal) with tiny, mutually-disjoint per-frame AABBs
    (IoU == 0, same shape as the existing jitter regression) but well within the
    tracker's centroid gate of each other -- must fold to ONE instance. Pre-fix this
    spawned 2 (see test_issue_89_same_frame_duplicates_PRE_FIX_repro_numbers): the
    snapshot-only pairing never let the second detection see the instance the first
    one just created in the SAME associate() call."""
    idx = BasicSceneIndex()
    od = _odom()

    def front_det(tile_id=0):
        spec = T.tile_specs()[tile_id]
        bb = (spec.cx - 60, spec.cy - 120, spec.cx + 60, spec.cy + 60)
        return Detection(tile_id=tile_id, bbox_xyxy=bb, label="chair", score=0.9)

    def tiny_cloud(cx, cy, cz, seed):
        rng = np.random.default_rng(seed)
        return np.column_stack([
            cx + rng.uniform(-0.02, 0.02, 40),
            cy + rng.uniform(-0.02, 0.02, 40),
            cz + rng.uniform(-0.02, 0.02, 40),
        ]).astype(np.float32)

    det1, det2 = front_det(), front_det()
    f1 = _fuse(det1, tiny_cloud(3.00, 0.00, 0.5, seed=1), od)
    f2 = _fuse(det2, tiny_cloud(3.30, 0.00, 0.5, seed=2), od)
    touched = associate([(det1, f1), (det2, f2)], idx)
    assert len(idx.all_instances()) == 1
    assert idx.all_instances()[0].n_obs == 2
    assert touched[0] == touched[1]


def test_issue_89_same_frame_duplicates_PRE_FIX_repro_numbers():
    """Pins the PRE-FIX number for the same-frame-duplicate mechanism: the OLD
    associate() (candidate pairs built only against the pre-batch snapshot) spawned
    2 instances for these 2 same-keyframe detections of one physical object."""
    from core.perception.tracker import _fused_to_record, labels_compatible

    def pre_fix_associate(fused_dets, index, gate=0.75):
        existing = index.all_instances()
        pairs = []
        for di, (det, fused) in enumerate(fused_dets):
            for ej, inst in enumerate(existing):
                if not labels_compatible(det.label, inst.label):
                    continue
                dist = float(np.linalg.norm(fused.centroid - inst.centroid))
                if dist <= gate:
                    pairs.append((dist, di, ej))
        pairs.sort(key=lambda p: p[0])
        matched, used = {}, set()
        for dist, di, ej in pairs:
            if di in matched or ej in used:
                continue
            matched[di] = existing[ej]
            used.add(ej)
        touched = []
        for di, (det, fused) in enumerate(fused_dets):
            if di in matched:
                target = matched[di]
                rec = _fused_to_record(det, fused, instance_id=target.instance_id)
                survivor = index.merge_into(target.instance_id, rec)
            else:
                new_id = index.next_id()
                rec = _fused_to_record(det, fused, instance_id=new_id)
                survivor = index.add(rec)
            touched.append(survivor.instance_id)
        return touched

    idx = BasicSceneIndex()
    od = _odom()

    def front_det(tile_id=0):
        spec = T.tile_specs()[tile_id]
        bb = (spec.cx - 60, spec.cy - 120, spec.cx + 60, spec.cy + 60)
        return Detection(tile_id=tile_id, bbox_xyxy=bb, label="chair", score=0.9)

    def tiny_cloud(cx, cy, cz, seed):
        rng = np.random.default_rng(seed)
        return np.column_stack([
            cx + rng.uniform(-0.02, 0.02, 40),
            cy + rng.uniform(-0.02, 0.02, 40),
            cz + rng.uniform(-0.02, 0.02, 40),
        ]).astype(np.float32)

    det1, det2 = front_det(), front_det()
    f1 = _fuse(det1, tiny_cloud(3.00, 0.00, 0.5, seed=1), od)
    f2 = _fuse(det2, tiny_cloud(3.30, 0.00, 0.5, seed=2), od)
    pre_fix_associate([(det1, f1), (det2, f2)], idx)
    assert len(idx.all_instances()) == 2  # the #89 defect: one object, two ghosts


# --------------------------------------------------------------------------- do-not-over-correct


def test_genuinely_single_object_still_merges_across_keyframes():
    """A genuinely single physical chair, observed 5 times across separate keyframes
    under realistic pose jitter, must still merge to ONE instance post-fix -- the
    existing regression this task must not break (mirrors
    test_associate_merges_repeated_sightings_despite_jitter_defeating_iou)."""
    idx = BasicSceneIndex()

    def front_det(tile_id=0):
        spec = T.tile_specs()[tile_id]
        bb = (spec.cx - 60, spec.cy - 120, spec.cx + 60, spec.cy + 60)
        return Detection(tile_id=tile_id, bbox_xyxy=bb, label="chair", score=0.9)

    def tiny_cloud(cx, cy, cz, seed):
        rng = np.random.default_rng(seed)
        return np.column_stack([
            cx + rng.uniform(-0.02, 0.02, 40),
            cy + rng.uniform(-0.02, 0.02, 40),
            cz + rng.uniform(-0.02, 0.02, 40),
        ]).astype(np.float32)

    centers = [(3.00, 0.00), (3.05, 0.30), (3.10, -0.25), (2.95, 0.35), (3.02, -0.30)]
    for i, (cx, cy) in enumerate(centers):
        det = front_det()
        fused = _fuse(det, tiny_cloud(cx, cy, 0.5, seed=i), _odom())
        associate([(det, fused)], idx)
    assert len(idx.all_instances()) == 1
    assert idx.all_instances()[0].n_obs == len(centers)


def test_genuinely_distinct_dense_chairs_mostly_stay_separate_post_fix():
    """4 physical chairs around a small table at realistic (if tight) 0.8 m spacing,
    all first seen in ONE keyframe together -- must stay 4 distinct instances. This
    is the scenario the extent veto exists to protect: without it, the sequential
    same-batch folding (needed for the #89 fix above) would readily fuse distinct
    nearby chairs the moment they share a keyframe."""
    od = _odom()
    idx = BasicSceneIndex()
    spacing = 0.8
    positions = [(3.0, 0.0), (3.0 + spacing, 0.0), (3.0, spacing), (3.0 + spacing, spacing)]
    fused_dets = []
    for i, (x, y) in enumerate(positions):
        det = _det_for_point(x, y, 0.0, od, "chair")
        cloud = _chair_cloud(x, y, 0.0, seed=i)
        fused = _fuse(det, cloud, od)
        assert fused is not None
        fused_dets.append((det, fused))
    associate(fused_dets, idx)
    assert len(idx.all_instances()) == 4


@pytest.mark.parametrize("factor", [0.5, 1.0, 2.0])
def test_extent_veto_factor_is_the_only_new_tunable_beyond_defaults(factor):
    """Sanity: TrackerConfig accepts extent_veto_factor overrides without raising
    (guards the dataclass wiring, not a behavioural claim about any specific value)."""
    cfg = TrackerConfig(extent_veto_factor=factor)
    assert cfg.extent_veto_factor == factor
