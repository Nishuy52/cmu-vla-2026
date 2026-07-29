"""Issue #153: the #94 extent veto minting duplicate instances at near-zero
centroid distance.

``_match_plausible`` (see :mod:`core.perception.tracker`) vetoes a match once the
union of the candidate instance's accumulated AABB and the incoming detection's
cluster AABB exceeds ``extent_veto_factor`` x the class's typical extent. For a
class whose thinnest typical axis is very tight (a television's depth, ~0.06 m
median), ordinary viewpoint-to-viewpoint depth noise on a REAL re-observation of
the SAME physical object is enough to blow the union past the threshold on its
own -- and the veto fires "however close the centroids landed", so it blocks the
match even at millimetre-to-centimetre centroid separation. ``associate()`` then
falls to its else-branch, mints a brand-new instance at (near-)the same spot, and
repeats forever: this is the mechanism behind the live sweep's byte-identical
duplicate clusters (48 duplicate televisions at one centroid, #153's issue body).

The fix (:data:`core.perception.tracker.TrackerConfig.extent_veto_min_sep`) floors
the veto: below that centroid separation, the match is accepted regardless of
extent, because a match landing that close is by construction the same object.
"""
from __future__ import annotations

import numpy as np

from core.interfaces import LidarScan, OdomState
from core.perception import tiling as T
from core.perception.detector import Detection
from core.perception.fusion import fuse_detection
from core.perception.scene_index import BasicSceneIndex
from core.perception.tracker import TrackerConfig, _fused_to_record, associate, labels_compatible


def _tv_det(tile_id=0):
    spec = T.tile_specs()[tile_id]
    bb = (spec.cx - 80, spec.cy - 100, spec.cx + 80, spec.cy + 100)
    return Detection(tile_id=tile_id, bbox_xyxy=bb, label="television", score=0.9)


def _det_for_point(x, y, z, od, label, score=0.9, tile_id=0, half_px=80):
    """Build a Detection whose bbox is centred on map point (x, y, z) as seen from
    ``od`` -- inverse of the fusion frustum maths (mirrors the helper in
    test_tracker_issue_94_89.py), needed so a detection actually TARGETS an
    off-centre point instead of reusing the same forward-facing bbox for objects
    at different bearings."""
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


def _tv_cloud(depth_centre, y_centre=0.0, z_centre=0.735, n=60, seed=0):
    """A thin, wide, tall planar cluster shaped like a real TV's typical extent
    (sorted (thin, mid, long) == (0.06, 0.791, 1.262), :mod:`dimension_priors`):
    depth (x) barely thicker than a front-shell sighting, width (y) and height (z)
    close to the class median."""
    rng = np.random.default_rng(seed)
    return np.column_stack([
        depth_centre + rng.uniform(-0.02, 0.02, n),
        y_centre + rng.uniform(-0.6, 0.6, n),
        z_centre + rng.uniform(-0.4, 0.4, n),
    ]).astype(np.float32)


def _odom():
    return OdomState(t=0.0, x=0.0, y=0.0, z=0.0, yaw=0.0)


def _fuse(det, cloud, od):
    return fuse_detection(det, LidarScan(t=0.0, points=cloud), od)


def test_repeat_sighting_with_depth_jitter_stays_one_instance_post_fix():
    """The SAME physical television, detected twice with mild point-cloud depth
    jitter (mimicking live depth noise: the front face's apparent depth wobbles
    ~0.05 m between sightings while width/height stay put) -- must produce exactly
    ONE instance. Pre-#153-fix this minted a second instance at the same spot
    (see test_repeat_sighting_PRE_FIX_mints_duplicate below): the union's depth
    extent (0.09 m) exceeds extent_veto_factor(1.3) x the class's typical depth
    (0.06 m -> 0.078 m ceiling), and the veto fired despite the two centroids
    landing only ~0.05 m apart."""
    idx = BasicSceneIndex()
    od = _odom()
    det1 = _tv_det()
    f1 = _fuse(det1, _tv_cloud(2.60, seed=1), od)
    assert f1 is not None
    associate([(det1, f1)], idx)
    assert len(idx.all_instances()) == 1

    det2 = _tv_det()
    f2 = _fuse(det2, _tv_cloud(2.65, seed=2), od)
    assert f2 is not None
    touched = associate([(det2, f2)], idx)

    insts = idx.all_instances()
    assert len(insts) == 1, (
        f"expected 1 instance (repeat sighting of the same TV), got {len(insts)} -- "
        "the #153 extent-veto-at-zero-distance defect"
    )
    assert insts[0].n_obs == 2
    assert touched == [insts[0].instance_id]


def test_repeat_sighting_PRE_FIX_mints_duplicate():
    """Pins the PRE-FIX failure number so the regression is documented even though
    the behaviour it names no longer reproduces post-fix: with extent_veto_min_sep
    disabled (0.0 -- the pre-#153 unconditional veto), the exact same repeat
    sighting above mints a SECOND instance at the same centroid instead of fusing
    into the first."""
    idx = BasicSceneIndex()
    od = _odom()
    pre_fix_cfg = TrackerConfig(extent_veto_min_sep=0.0)

    det1 = _tv_det()
    f1 = _fuse(det1, _tv_cloud(2.60, seed=1), od)
    associate([(det1, f1)], idx, cfg=pre_fix_cfg)
    assert len(idx.all_instances()) == 1

    det2 = _tv_det()
    f2 = _fuse(det2, _tv_cloud(2.65, seed=2), od)
    associate([(det2, f2)], idx, cfg=pre_fix_cfg)

    insts = idx.all_instances()
    assert len(insts) == 2  # the #153 defect: one physical TV, two instances
    assert all(r.n_obs == 1 for r in insts)
    # byte-identical-ish centroids -- this is the "however close" signature from
    # the issue, not two objects that are actually apart.
    dist = float(np.linalg.norm(insts[0].centroid - insts[1].centroid))
    assert dist < 0.1


def test_genuinely_distinct_same_class_objects_well_outside_centroid_gate_still_split():
    """#94 must-not-regress: two genuinely distinct televisions at a realistic
    (if tight) 1.0 m spacing, both first seen in the SAME keyframe, must stay TWO
    instances -- extent_veto_min_sep (0.2 m) is well below this spacing, so the
    floor added for #153 never masks a real distinct-object case.

    Renamed from ...at_realistic_spacing_still_split (#161): 1.0 m spacing is well
    OUTSIDE the television centroid gate (0.745 m, from dimension_priors' sorted
    typ_ext (0.06, 0.791, 1.262)), so the candidate pair is excluded by the
    per-class centroid gate and `_match_plausible` (the extent veto) is never even
    consulted here -- this test exercises the centroid gate, not the veto, despite
    its old name. Coverage for the veto actually being decisive (the 0.44-0.75 m
    band where centroid distance clears the gate but the extent veto still fires)
    now lives in test_tracker_issue_94_89.py's
    test_issue_161_extent_veto_decisive_band_two_tvs_still_split /
    test_issue_161_extent_veto_disabled_same_tvs_collapse_to_one pair."""
    idx = BasicSceneIndex()
    od = _odom()

    det_a = _det_for_point(2.60, 0.0, 0.735, od, "television")
    det_b = _det_for_point(2.60, 1.0, 0.735, od, "television")
    f_a = _fuse(det_a, _tv_cloud(2.60, y_centre=0.0, seed=1), od)
    f_b = _fuse(det_b, _tv_cloud(2.60, y_centre=1.0, seed=2), od)
    assert f_a is not None and f_b is not None
    assert labels_compatible(det_a.label, det_b.label)

    associate([(det_a, f_a), (det_b, f_b)], idx)
    assert len(idx.all_instances()) == 2
