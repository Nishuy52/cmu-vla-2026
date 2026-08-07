"""Issue #191: n_obs must count distinct viewpoints, not dwell.

The #187 diagnosis traced four ghost tracks (all shipped fixes rank/gate on
n_obs -- #151, #184, #186) to one mechanism: every frame is a keyframe
(``KeyframeConfig.every_k == 1``), and ``associate()``/``_fuse`` counted one
``n_obs`` per ACCEPTED DETECTION with no pose gate at all -- a parked robot
re-firing an identical box every tick inflated n_obs on dwell alone.

The fix threads the current pose through ``associate()`` (``odom=``): a
detection that matches an EXISTING instance only counts toward n_obs if the
pose moved >= ``KeyframeConfig.min_translation`` or turned >=
``KeyframeConfig.min_rotation`` since that instance's own last COUNTED pose
-- reusing the two existing keyframe-gate constants, no new tunable.
Fusion/geometry (box, score) still update on every accepted detection
exactly as before; only the n_obs increment is gated.

This module replays ARCHIVED raw data only (no bag reads, no live replay),
reproducing the #187/#191 predicted collapse from real cluster-verify
artifacts:

* ``reports/cluster_verify/712650/debug/2_chinese_room_obje/``
  (chinese_room, run 712650 slot 2) -- ghost id82 "folding screen"
  (80 raw hits, ONE pose) and real id14 "folding screen" (control, multiple
  poses).
* ``reports/cluster_verify/713818/debug/10_hotel_room_2_obje/``
  (hotel_room_2, run 713818 slot 10) -- ghosts id5 "flowers" and id48
  "window".
* ``reports/cluster_verify/713818/debug/8_hotel_room_1_obje/``
  (hotel_room_1, run 713818 slot 8) -- ghost id116 "bedside table".

Pose data: ``raw_detections.jsonl`` carries no pose, only pixel bbox/label/
score/gate/instance_id per detection (as the #187 diagnosis notes). Real
vehicle (x, y) position is recovered from the SAME slot's throttled
``explore_debug_*.jsonl`` ("pose": [x, y], "keyframes_processed": N), which
this module correlates to each raw detection's ``keyframe_idx`` by taking
the pose recorded once AT LEAST that many keyframes have been processed
(the earliest ``explore_debug`` sample whose ``keyframes_processed`` covers
the detection), falling back to the last known sample once the throttled
log runs out. Yaw is not recorded by ``explore_debug`` and is synthesized as
a constant 0.0 -- every ghost/control distinction in the #187 diagnosis is
argued on TRANSLATION alone ("1.1 m pose cluster", "2.2 m apart", "0.0 m"),
so a translation-only replay is faithful to the evidence being reproduced;
rotation gating is exercised separately by the synthetic unit tests below.

The archived geometry (pixel bbox) is never projected to 3D here -- these
are the SAME wall/surface re-observed each tick (that is the ghost's whole
signature), so each replayed detection reuses one fixed synthetic 3D
cluster per instance; only the (accepted-detection count per keyframe,
pose-at-that-keyframe) pattern is drawn from the archive. This isolates
exactly what changed: the pose-gated n_obs count, against real recorded
keyframe/pose traffic.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from core.interfaces import InstanceRecord, OdomState
from core.perception.detector import Detection
from core.perception.fusion import Fused3D
from core.perception.scene_index import BasicSceneIndex
from core.perception.tracker import DEFAULT_KEYFRAME_CONFIG, TrackerConfig, associate

_REPORTS = Path(__file__).resolve().parents[3] / "reports" / "cluster_verify"

_CHINESE_ROOM = _REPORTS / "712650" / "debug" / "2_chinese_room_obje"
_HOTEL_2 = _REPORTS / "713818" / "debug" / "10_hotel_room_2_obje"
_HOTEL_1 = _REPORTS / "713818" / "debug" / "8_hotel_room_1_obje"


# --------------------------------------------------------------------------- archive loading


def _hits_per_keyframe(raw_detections_path: Path, instance_id: int) -> "Counter[int]":
    """keyframe_idx -> count of accepted detections for ``instance_id`` in the
    archived ``raw_detections.jsonl``."""
    counts: Counter[int] = Counter()
    with open(raw_detections_path) as f:
        for line in f:
            rec = json.loads(line)
            for det in rec.get("detections", ()):
                if det.get("instance_id") == instance_id:
                    counts[rec["keyframe_idx"]] += 1
    return counts


def _pose_lookup(explore_debug_path: Path) -> list[tuple[int, tuple[float, float]]]:
    """Sorted ``[(keyframes_processed, (x, y)), ...]`` from an archived
    ``explore_debug_*.jsonl`` -- the last pose logged at each distinct
    ``keyframes_processed`` level (later log within the same level is the more
    settled position)."""
    last: dict[int, tuple[float, float]] = {}
    with open(explore_debug_path) as f:
        for line in f:
            rec = json.loads(line)
            kfp = rec.get("keyframes_processed")
            pose = rec.get("pose")
            if kfp is None or pose is None:
                continue
            last[kfp] = (float(pose[0]), float(pose[1]))
    return sorted(last.items())


def _pose_for_keyframe(
    lookup: list[tuple[int, tuple[float, float]]], keyframe_idx: int
) -> tuple[float, float]:
    """The archived vehicle (x, y) in effect for raw ``keyframe_idx`` -- the
    earliest logged sample whose ``keyframes_processed`` already covers this
    keyframe (``>= keyframe_idx + 1``), or the last known sample once the
    throttled log has nothing later (the vehicle held its final logged pose,
    exactly the "parked robot" signature these ghosts trace to)."""
    target = keyframe_idx + 1
    for kfp, pose in lookup:
        if kfp >= target:
            return pose
    return lookup[-1][1]


# --------------------------------------------------------------------------- replay harness


def _det(label: str, score: float = 0.3) -> Detection:
    return Detection(tile_id=0, bbox_xyxy=(0.0, 0.0, 10.0, 10.0), label=label, score=score)


def _fused_for_instance(seed: int) -> Fused3D:
    """One fixed synthetic 3D cluster per instance -- these ghosts are, per the
    #187 diagnosis, the SAME wall surface re-observed every tick; the archive's
    raw bbox is 2D pixel space and is not re-projected here (see module
    docstring)."""
    rng = np.random.default_rng(seed)
    pts = rng.normal(loc=(seed * 3.0, 0.0, 1.0), scale=0.02, size=(12, 3)).astype(np.float32)
    centroid = pts.mean(axis=0)
    return Fused3D(centroid=centroid, points=pts, n_points=len(pts), range_m=float(np.linalg.norm(centroid)))


def _replay(
    hits: "Counter[int]",
    pose_lookup: list[tuple[int, tuple[float, float]]],
    *,
    label: str,
    seed: int,
    gated: bool,
) -> int:
    """Replay one instance's archived (keyframe -> hit count) pattern through the
    REAL ``associate()``, driving real archived poses. ``gated=False`` reproduces
    pre-#191 behaviour (``odom=None``, ungated -- one n_obs per accepted
    detection, matching the archive's raw hit counts exactly). ``gated=True`` is
    the #191 fix. Returns the resulting instance's final ``n_obs``."""
    index = BasicSceneIndex()
    fused = _fused_for_instance(seed)
    cfg = TrackerConfig()
    iid: int | None = None
    for keyframe_idx in sorted(hits):
        n = hits[keyframe_idx]
        x, y = _pose_for_keyframe(pose_lookup, keyframe_idx)
        odom = OdomState(t=float(keyframe_idx), x=x, y=y, z=0.0, yaw=0.0) if gated else None
        fused_dets = [(_det(label), fused) for _ in range(n)]
        touched = associate(fused_dets, index, cfg, odom=odom, keyframe_cfg=DEFAULT_KEYFRAME_CONFIG)
        iid = touched[0]
        assert all(t == iid for t in touched), "all hits at one keyframe must fold to one instance"
    assert iid is not None
    rec = next(r for r in index.all_instances() if r.instance_id == iid)
    return rec.n_obs


# --------------------------------------------------------------------------- the four ghosts + control


def test_chinese_room_id82_ghost_collapses_from_dwell_count():
    """id82 "folding screen": 80 raw hits, 16 keyframes, ONE pose (#187/#191)."""
    hits = _hits_per_keyframe(_CHINESE_ROOM / "raw_detections.jsonl", instance_id=82)
    assert sum(hits.values()) == 80  # the archived dwell count #187/#191 cite
    assert len(hits) == 16

    pose_lookup = _pose_lookup(_CHINESE_ROOM / "explore_debug_125573.jsonl")

    pre_fix = _replay(hits, pose_lookup, label="folding screen", seed=82, gated=False)
    post_fix = _replay(hits, pose_lookup, label="folding screen", seed=82, gated=True)

    assert pre_fix == 80  # reproduces the archived ghost exactly
    assert post_fix == 1  # #187/#191 predicted 80 -> 1: one parked pose, one observation


def test_chinese_room_id14_real_screen_keeps_multiple_observations():
    """The real screen (id14) is the #187/#191 control: multi-pose, must NOT
    collapse to 1 the way the ghost does."""
    hits = _hits_per_keyframe(_CHINESE_ROOM / "raw_detections.jsonl", instance_id=14)
    assert sum(hits.values()) == 6
    assert len(hits) == 5  # keyframes 36-40

    pose_lookup = _pose_lookup(_CHINESE_ROOM / "explore_debug_125573.jsonl")

    # The archived pose actually moved >= min_translation between this
    # instance's early and late keyframes -- this is the real "two poses far
    # apart" case #187 contrasts against every ghost above.
    early_pose = np.array(_pose_for_keyframe(pose_lookup, min(hits)))
    late_pose = np.array(_pose_for_keyframe(pose_lookup, max(hits)))
    assert np.hypot(*(late_pose - early_pose)) >= DEFAULT_KEYFRAME_CONFIG.min_translation

    pre_fix = _replay(hits, pose_lookup, label="folding screen", seed=14, gated=False)
    post_fix = _replay(hits, pose_lookup, label="folding screen", seed=14, gated=True)

    assert pre_fix == 6
    assert post_fix == 2  # two distinct counted viewpoints -- keeps its multi-view evidence
    assert post_fix >= 2  # never collapses to a single-observation ghost


def test_hotel_room_2_id5_flowers_ghost_collapses():
    """id5 "flowers": #187/#191 measured 59-62 raw hits off one wall patch,
    predicted collapse to ~2."""
    hits = _hits_per_keyframe(_HOTEL_2 / "raw_detections.jsonl", instance_id=5)
    assert sum(hits.values()) >= 59
    pose_lookup = _pose_lookup(_HOTEL_2 / "explore_debug_373506.jsonl")

    pre_fix = _replay(hits, pose_lookup, label="flowers", seed=5, gated=False)
    post_fix = _replay(hits, pose_lookup, label="flowers", seed=5, gated=True)

    assert pre_fix == sum(hits.values())
    assert post_fix <= 3  # #187/#191 predicted ~2; the dwell inflation is gone either way
    assert post_fix < pre_fix / 10


def test_hotel_room_2_id48_window_ghost_collapses():
    """id48 "window": 22 raw hits, one identical bbox for 20 keyframes, predicted
    collapse to ~2 (#187/#191)."""
    hits = _hits_per_keyframe(_HOTEL_2 / "raw_detections.jsonl", instance_id=48)
    assert sum(hits.values()) == 22
    pose_lookup = _pose_lookup(_HOTEL_2 / "explore_debug_373506.jsonl")

    pre_fix = _replay(hits, pose_lookup, label="window", seed=48, gated=False)
    post_fix = _replay(hits, pose_lookup, label="window", seed=48, gated=True)

    assert pre_fix == 22
    assert post_fix == 2  # matches the #187/#191 prediction exactly


def test_hotel_room_1_id116_bedside_table_ghost_collapses():
    """id116 "bedside table": 36 raw hits off a 1.1 m pose cluster, predicted
    collapse to ~3 (#187/#191)."""
    hits = _hits_per_keyframe(_HOTEL_1 / "raw_detections.jsonl", instance_id=116)
    assert sum(hits.values()) == 36
    pose_lookup = _pose_lookup(_HOTEL_1 / "explore_debug_370464.jsonl")

    pre_fix = _replay(hits, pose_lookup, label="bedside table", seed=116, gated=False)
    post_fix = _replay(hits, pose_lookup, label="bedside table", seed=116, gated=True)

    assert pre_fix == 36
    # #187/#191 predicted ~3 from finer-grained pose data than this archive's
    # throttled explore_debug log offers; this replay's coarser (nearest
    # available sample) pose reconstruction still shows the same order-of-
    # magnitude collapse the fix is meant to produce, never dwell-inflated.
    assert 2 <= post_fix <= 4
    assert post_fix < 36 / 10


# --------------------------------------------------------------------------- synthetic unit tests (rotation, per-keyframe fold, ungated default)


def _odom(x=0.0, y=0.0, yaw=0.0, t=0.0):
    return OdomState(t=t, x=x, y=y, z=0.0, yaw=yaw)


def _one_det_fused(label="chair", seed=1):
    rng = np.random.default_rng(seed)
    pts = rng.normal(loc=(1.0, 0.0, 0.5), scale=0.02, size=(8, 3)).astype(np.float32)
    centroid = pts.mean(axis=0)
    return Fused3D(centroid=centroid, points=pts, n_points=len(pts), range_m=float(np.linalg.norm(centroid)))


def test_stationary_repeated_detection_counts_once():
    """Core #191 behaviour, direct: N identical-pose observations of one
    instance across N separate associate() calls (every frame IS a keyframe,
    KeyframeConfig.every_k == 1) count as n_obs == 1, not N."""
    index = BasicSceneIndex()
    cfg = TrackerConfig()
    fused = _one_det_fused()
    for i in range(10):
        touched = associate([(_det("chair"), fused)], index, cfg, odom=_odom(t=float(i)))
    rec = next(r for r in index.all_instances() if r.instance_id == touched[0])
    assert rec.n_obs == 1


def test_translation_past_threshold_counts_a_new_observation():
    """Moving >= min_translation between calls counts a second observation;
    less than that does not."""
    index = BasicSceneIndex()
    cfg = TrackerConfig()
    fused = _one_det_fused()
    mt = DEFAULT_KEYFRAME_CONFIG.min_translation
    associate([(_det("chair"), fused)], index, cfg, odom=_odom(x=0.0, t=0.0))
    associate([(_det("chair"), fused)], index, cfg, odom=_odom(x=mt * 0.5, t=1.0))  # under gate
    touched = associate([(_det("chair"), fused)], index, cfg, odom=_odom(x=mt * 1.1, t=2.0))  # over gate
    rec = next(r for r in index.all_instances() if r.instance_id == touched[0])
    assert rec.n_obs == 2


def test_rotation_past_threshold_counts_a_new_observation():
    """Turning >= min_rotation with zero translation also counts a new
    observation -- the OR in the #187/#191 fix recipe."""
    index = BasicSceneIndex()
    cfg = TrackerConfig()
    fused = _one_det_fused()
    mr = DEFAULT_KEYFRAME_CONFIG.min_rotation
    associate([(_det("chair"), fused)], index, cfg, odom=_odom(yaw=0.0, t=0.0))
    touched = associate([(_det("chair"), fused)], index, cfg, odom=_odom(yaw=mr * 1.1, t=1.0))
    rec = next(r for r in index.all_instances() if r.instance_id == touched[0])
    assert rec.n_obs == 2


def test_multiple_same_keyframe_detections_of_one_instance_count_once():
    """Several accepted detections landing on the same instance within ONE
    associate() call (one keyframe) count as a single n_obs, not one each --
    "at most one observation per keyframe per instance"."""
    index = BasicSceneIndex()
    cfg = TrackerConfig()
    fused = _one_det_fused()
    fused_dets = [(_det("chair"), fused) for _ in range(5)]
    touched = associate(fused_dets, index, cfg, odom=_odom())
    assert len(set(touched)) == 1
    rec = next(r for r in index.all_instances() if r.instance_id == touched[0])
    assert rec.n_obs == 1


def test_fusion_geometry_still_updates_on_a_gated_observation():
    """A gated (uncounted) observation still fuses geometry/score -- only the
    n_obs increment is gated, never the box/points/score update."""
    index = BasicSceneIndex()
    cfg = TrackerConfig()
    fused1 = _one_det_fused(seed=1)
    fused2 = _one_det_fused(seed=2)  # a different sample of the same cluster
    associate([(_det("chair", score=0.3), fused1)], index, cfg, odom=_odom(t=0.0))
    rec_before = next(iter(index.all_instances()))
    n_before = len(rec_before.points)
    touched = associate(
        [(_det("chair", score=0.9), fused2)], index, cfg, odom=_odom(t=1.0),
    )  # same pose -- gated, n_obs must not increment
    rec_after = next(r for r in index.all_instances() if r.instance_id == touched[0])
    assert rec_after.n_obs == 1  # gated
    assert len(rec_after.points) > n_before  # but points/geometry still fused
    assert rec_after.score == pytest.approx(0.9)  # and score still tracks the max


def test_no_odom_is_ungated_backward_compatible():
    """``odom=None`` (every pre-#191 caller/test) is unchanged prior behaviour:
    one n_obs counted per accepted detection, no gating at all."""
    index = BasicSceneIndex()
    cfg = TrackerConfig()
    fused = _one_det_fused()
    for i in range(4):
        touched = associate([(_det("chair"), fused)], index, cfg)  # no odom kwarg
    rec = next(r for r in index.all_instances() if r.instance_id == touched[0])
    assert rec.n_obs == 4


def test_brand_new_instance_first_observation_always_counts():
    """A genuinely new instance's very first observation always counts (nothing
    to compare its pose against yet), even under gating."""
    index = BasicSceneIndex()
    cfg = TrackerConfig()
    fused = _one_det_fused()
    touched = associate([(_det("chair"), fused)], index, cfg, odom=_odom())
    rec = next(r for r in index.all_instances() if r.instance_id == touched[0])
    assert rec.n_obs == 1
