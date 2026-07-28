"""Issue #112: ``associate()``'s same-batch folding must be order-independent.

The #94/#89 fix (see :mod:`core.perception.tracker`) made same-batch folding
sequential against a growing pool, which restored correctness for same-frame
duplicates but reintroduced an old defect: the result now depends on the
iteration order of detections WITHIN one batch. Permuting an identical set of
detections can change the final instance count (2 vs 3 for the same physical
scene -- see the issue body for the live repro numbers).

This test reproduces the defect directly (several permutations of one batch of
four closely-spaced monitor detections must all fold to the SAME instance set)
and is the acceptance criterion for the fix: it must go from failing to passing
without any test-only special-casing of order.
"""
from __future__ import annotations

import itertools

import numpy as np
import pytest

from core.interfaces import OdomState, LidarScan
from core.perception import tiling as T
from core.perception.detector import Detection
from core.perception.fusion import fuse_detection
from core.perception.scene_index import BasicSceneIndex
from core.perception.tracker import associate


def _det_for_point(x, y, z, od, label, score=0.9, tile_id=0, half_px=45):
    """Build a Detection whose bbox is centred on map point (x, y, z) as seen from
    ``od`` (inverse of the fusion frustum maths -- mirrors the helper in
    test_tracker_issue_94_89.py)."""
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


def _odom(x=0.0, y=0.0, z=0.0, yaw=0.0):
    return OdomState(t=0.0, x=x, y=y, z=z, yaw=yaw)


def _fuse(det, cloud, od):
    return fuse_detection(det, LidarScan(t=0.0, points=cloud), od)


def _build_batch():
    """One batch of 4 monitor detections at tight 0.3 m spacing -- narrow enough
    for the extent veto's margin to be order-sensitive pre-fix (mirrors the
    issue's exact repro geometry)."""
    od = _odom()
    positions = [(3.0, i * 0.3, 1.0) for i in range(4)]
    fused_dets = []
    for i, (x, y, z) in enumerate(positions):
        det = _det_for_point(x, y, z, od, "monitor")
        cloud = _monitor_cloud(x, y, z, seed=i)
        fused = _fuse(det, cloud, od)
        assert fused is not None
        fused_dets.append((det, fused))
    return fused_dets


def _instance_signature(index: BasicSceneIndex):
    """A permutation-of-input-independent description of the resulting instance
    set: sorted (label, rounded centroid, rounded extent) tuples. Deliberately
    excludes instance_id (an arrival-order/minting artifact -- #111/#113) and
    n_obs ordering."""
    sigs = []
    for rec in index.all_instances():
        centroid = tuple(round(float(c), 3) for c in rec.centroid)
        extent = tuple(round(float(v), 3) for v in (rec.aabb_max - rec.aabb_min))
        sigs.append((rec.label, centroid, extent, rec.n_obs))
    return sorted(sigs)


def _run_permutation(order):
    idx = BasicSceneIndex()
    batch = _build_batch()
    permuted = [batch[i] for i in order]
    associate(permuted, idx)
    return len(idx.all_instances()), _instance_signature(idx)


PERMUTATIONS = [
    (0, 1, 2, 3),
    (3, 2, 1, 0),
    (1, 0, 3, 2),
    (0, 2, 1, 3),
]


def test_same_batch_fold_is_permutation_invariant():
    results = {perm: _run_permutation(perm) for perm in PERMUTATIONS}
    counts = {perm: r[0] for perm, r in results.items()}
    sigs = {perm: r[1] for perm, r in results.items()}

    baseline_count = counts[PERMUTATIONS[0]]
    baseline_sig = sigs[PERMUTATIONS[0]]

    mismatches = {
        perm: (c, sigs[perm])
        for perm, c in counts.items()
        if c != baseline_count or sigs[perm] != baseline_sig
    }
    assert not mismatches, (
        f"associate() is order-dependent: baseline (order {PERMUTATIONS[0]}) gave "
        f"count={baseline_count} sig={baseline_sig}; mismatching permutations: {mismatches}"
    )


@pytest.mark.parametrize("order", list(itertools.permutations(range(4))))
def test_all_24_permutations_agree_with_identity_order(order):
    """Exhaustive: every one of the 4! orderings of the same 4-detection batch
    must fold to the identical instance set as the identity order."""
    identity_count, identity_sig = _run_permutation((0, 1, 2, 3))
    count, sig = _run_permutation(order)
    assert count == identity_count
    assert sig == identity_sig
