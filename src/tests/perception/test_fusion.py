"""Lidar fusion: synthetic box -> centroid, frustum/min_points rejection, wrap seam."""
from __future__ import annotations

import numpy as np

from core.interfaces import LidarScan, OdomState
from core.perception import tiling as T
from core.perception.detector import Detection
from core.perception.fusion import (
    DEFAULT_FUSION_CONFIG,
    FusionConfig,
    _bbox_map_frustum,
    fuse_detection,
    robust_core_mask,
)


def _front_detection(tile_id=0):
    spec = T.tile_specs()[tile_id]
    bb = (spec.cx - 60, spec.cy - 120, spec.cx + 60, spec.cy + 60)
    return Detection(tile_id=tile_id, bbox_xyxy=bb, label="sofa", score=0.9)


def _lamp_detection(tile_id=0):
    spec = T.tile_specs()[tile_id]
    # a tall-ish box so its elevation span comfortably covers both ranges' rays
    bb = (spec.cx - 40, spec.cy - 100, spec.cx + 40, spec.cy + 20)
    return Detection(tile_id=tile_id, bbox_xyxy=bb, label="lamp", score=0.9)


def _ray_cloud(apex, bearing, elevation, range_m, half=0.12, n=30, seed=0):
    """A compact point cluster centred on the ray (bearing, elevation, range_m)
    from ``apex`` -- lets a test place two candidate clusters at the SAME angular
    position but different depths, exactly the ceiling-luminaire-vs-table-lamp
    ambiguity issue #199 fixes."""
    rng = np.random.default_rng(seed)
    cx = apex[0] + range_m * np.cos(elevation) * np.cos(bearing)
    cy = apex[1] + range_m * np.cos(elevation) * np.sin(bearing)
    cz = apex[2] + range_m * np.sin(elevation)
    pts = np.column_stack([
        cx + rng.uniform(-half, half, n),
        cy + rng.uniform(-half, half, n),
        cz + rng.uniform(-half, half, n),
    ])
    return pts.astype(np.float32)


def _box_cloud(cx, cy, cz, half=0.2, n=40, seed=0):
    rng = np.random.default_rng(seed)
    pts = np.column_stack([
        cx + rng.uniform(-half, half, n),
        cy + rng.uniform(-half, half, n),
        cz + rng.uniform(-half, half, n),
    ])
    return pts.astype(np.float32)


def _odom(x=0.0, y=0.0, z=0.0, yaw=0.0):
    return OdomState(t=0.0, x=x, y=y, z=z, yaw=yaw)


# ------------------------------------------------------------------ centroid


def test_box_of_points_yields_centroid_near_truth():
    det = _front_detection()
    cloud = _box_cloud(3.0, 0.0, 0.5)
    scan = LidarScan(t=0.0, points=cloud)
    fused = fuse_detection(det, scan, _odom())
    assert fused is not None
    assert np.allclose(fused.centroid, [3.0, 0.0, 0.5], atol=0.15)
    assert fused.n_points == 40
    assert abs(fused.range_m - 3.041) < 0.2


def test_centroid_within_a_cell_of_true_center():
    det = _front_detection()
    true_c = np.array([4.0, 0.0, 0.6])
    scan = LidarScan(t=0.0, points=_box_cloud(*true_c, half=0.25, n=60, seed=3))
    fused = fuse_detection(det, scan, _odom())
    assert fused is not None
    assert np.linalg.norm(fused.centroid - true_c) < 0.2  # within a ~0.2 m cell


def test_all_in_frustum_points_retained():
    det = _front_detection()
    scan = LidarScan(t=0.0, points=_box_cloud(3.0, 0.0, 0.5, n=50, seed=9))
    fused = fuse_detection(det, scan, _odom())
    assert fused is not None
    assert fused.points.shape == (50, 3)


# ------------------------------------------------------------------ frustum reject


def test_off_angle_points_rejected():
    det = _front_detection()  # looks along +x
    # points to the vehicle's right (bearing -90deg) are outside the front frustum
    off = _box_cloud(0.0, -3.0, 0.5, n=40, seed=2)
    fused = fuse_detection(det, LidarScan(t=0.0, points=off), _odom())
    assert fused is None


def test_points_behind_front_tile_rejected():
    det = _front_detection()
    behind = _box_cloud(-3.0, 0.0, 0.5, n=40, seed=5)
    assert fuse_detection(det, LidarScan(t=0.0, points=behind), _odom()) is None


def test_out_of_range_points_rejected():
    det = _front_detection()
    far = _box_cloud(30.0, 0.0, 0.5, n=40, seed=7)  # beyond default max_range 15
    assert fuse_detection(det, LidarScan(t=0.0, points=far), _odom()) is None


# ------------------------------------------------------------------ min_points


def test_min_points_rejection():
    det = _front_detection()
    few = np.array([[3, 0, 0.5], [3.05, 0.02, 0.5], [2.95, -0.02, 0.5]], dtype=np.float32)
    assert fuse_detection(det, LidarScan(t=0.0, points=few), _odom()) is None


def test_min_points_boundary_accept():
    det = _front_detection()
    cloud = _box_cloud(3.0, 0.0, 0.5, half=0.1, n=5, seed=1)
    fused = fuse_detection(det, LidarScan(t=0.0, points=cloud), _odom())
    assert fused is not None and fused.n_points == 5


def test_custom_min_points():
    det = _front_detection()
    cloud = _box_cloud(3.0, 0.0, 0.5, n=8, seed=1)
    cfg = FusionConfig(min_points=20)
    assert fuse_detection(det, LidarScan(t=0.0, points=cloud), _odom(), cfg) is None


# ------------------------------------------------------------------ depth cluster


def test_nearest_cluster_selected_over_background():
    det = _front_detection()
    near = _box_cloud(3.0, 0.0, 0.5, half=0.15, n=30, seed=1)
    far = _box_cloud(8.0, 0.0, 0.5, half=0.15, n=30, seed=2)  # same bearing, behind
    scan = LidarScan(t=0.0, points=np.vstack([near, far]).astype(np.float32))
    fused = fuse_detection(det, scan, _odom())
    assert fused is not None
    # centroid near the front cluster, not pulled toward 8 m
    assert fused.centroid[0] < 4.0
    assert fused.n_points == 30


# ------------------------------------------------------------------ wrap seam


def test_wrap_seam_object_behind_vehicle():
    """Object directly behind (bearing ~+-pi) fuses through the azimuth wrap."""
    spec = T.tile_specs()[2]  # rear tile, yaw_center 180
    bb = (spec.cx - 60, spec.cy - 60, spec.cx + 60, spec.cy + 60)
    det = Detection(tile_id=2, bbox_xyxy=bb, label="chair", score=0.8)
    behind = _box_cloud(-3.0, 0.0, 0.0, half=0.15, n=40, seed=4)
    fused = fuse_detection(det, LidarScan(t=0.0, points=behind), _odom())
    assert fused is not None
    assert np.allclose(fused.centroid[:2], [-3.0, 0.0], atol=0.15)


def test_wrap_seam_with_vehicle_yaw():
    """With the vehicle turned, an object straddling the map +-pi seam still fuses."""
    # vehicle yaw 180deg, object straight ahead in body frame -> at map bearing +-pi
    det = _front_detection()  # front tile, camera az 0
    yaw = np.pi
    # ahead of a pi-yawed vehicle is -x in the map
    cloud = _box_cloud(-3.0, 0.0, 0.5, half=0.15, n=40, seed=6)
    fused = fuse_detection(det, LidarScan(t=0.0, points=cloud), _odom(yaw=yaw))
    assert fused is not None
    assert np.allclose(fused.centroid[:2], [-3.0, 0.0], atol=0.15)


# ------------------------------------------------------------------ empty / apex


def test_empty_scan_returns_none():
    det = _front_detection()
    empty = LidarScan(t=0.0, points=np.empty((0, 3), dtype=np.float32))
    assert fuse_detection(det, empty, _odom()) is None


# ------------------------------------------------------------------ lateral (3D) clustering


def test_two_lateral_objects_yield_separate_component_not_merged():
    """Two same-class objects at the same range but different lateral offsets, both
    inside one detection's frustum, must NOT collapse into one merged box (the
    under-segmentation bug: 112 monitor detections -> 1 tracked instance)."""
    det = _front_detection()  # wide bbox: cy +/- 120/60 px -> a generous frustum
    left = _box_cloud(3.0, -0.5, 0.5, half=0.15, n=30, seed=11)
    right = _box_cloud(3.0, 0.5, 0.5, half=0.15, n=30, seed=12)
    scan = LidarScan(t=0.0, points=np.vstack([left, right]).astype(np.float32))
    fused = fuse_detection(det, scan, _odom())
    assert fused is not None
    # the winning component must be ONE of the two objects, not a merge of both:
    # a merged cluster would have points spanning y across both centres (~1.0 m
    # apart, half=0.15 -> span up to ~1.3 m); a single object's points span at
    # most ~0.3 m in y.
    y_span = fused.points[:, 1].max() - fused.points[:, 1].min()
    assert y_span < 0.5
    assert fused.n_points == 30


def test_single_object_not_shattered_by_lateral_clustering():
    """A single object's own point cloud (contiguous within the default
    cluster_radius) must remain one component, not fragment into several
    below-min_points pieces that get rejected."""
    det = _front_detection()
    cloud = _box_cloud(3.0, 0.0, 0.5, half=0.2, n=60, seed=13)
    fused = fuse_detection(det, LidarScan(t=0.0, points=cloud), _odom())
    assert fused is not None
    assert fused.n_points == 60  # nothing shattered off


def test_lateral_clustering_tightens_box_vs_old_cone_slab():
    """On a synthetic cone with two laterally-spread point groups at the same range
    (what the old algorithm accepted whole, since range clustering alone can't split
    same-range points), the new fused box must be tighter than the old cone-slab
    box would have been."""
    det = _front_detection()
    true_obj = _box_cloud(3.0, 0.0, 0.5, half=0.15, n=30, seed=14)
    # a second, laterally-offset cluster at (near-identical range) -- old algorithm's
    # range-only clustering would keep both (same range bucket), inflating the AABB.
    decoy = _box_cloud(3.02, 1.2, 0.5, half=0.15, n=30, seed=15)
    scan = LidarScan(t=0.0, points=np.vstack([true_obj, decoy]).astype(np.float32))
    fused = fuse_detection(det, scan, _odom())
    assert fused is not None
    new_span_y = fused.points[:, 1].max() - fused.points[:, 1].min()

    old_all = np.vstack([true_obj, decoy])
    old_span_y = old_all[:, 1].max() - old_all[:, 1].min()
    assert new_span_y < old_span_y
    assert new_span_y < 0.5  # tight to a single object, not the merged slab (~1.5 m)


def test_custom_cluster_radius_can_merge_or_split():
    """A larger cluster_radius bridges a gap that a smaller one splits; both objects
    still resolvable via the config knob (documents the tunable's effect). Sampled
    dense enough (n=40 per side in a small cube) that a moderate tight radius does
    not itself fragment a single side by chance."""
    det = _front_detection()
    left = _box_cloud(3.0, -0.5, 0.5, half=0.1, n=40, seed=16)
    right = _box_cloud(3.0, 0.5, 0.5, half=0.1, n=40, seed=17)
    scan = LidarScan(t=0.0, points=np.vstack([left, right]).astype(np.float32))

    tight_cfg = FusionConfig(cluster_radius=0.15)  # < the ~0.8 m inter-object gap
    fused_tight = fuse_detection(det, scan, _odom(), tight_cfg)
    assert fused_tight is not None
    assert fused_tight.n_points <= 45  # one side only, not a cross-object merge

    loose_cfg = FusionConfig(cluster_radius=2.0)  # bridges the inter-object gap
    fused_loose = fuse_detection(det, scan, _odom(), loose_cfg)
    assert fused_loose is not None
    assert fused_loose.n_points == 80  # bridged: both sides merge into one component


def test_apex_offset_by_odom_position():
    """Frustum apex follows the vehicle: object at map (5,2) seen from (2,2)."""
    det = _front_detection()  # front tile, yaw 0 -> looks along +x
    cloud = _box_cloud(5.0, 2.0, 0.5, half=0.15, n=40, seed=8)
    fused = fuse_detection(det, LidarScan(t=0.0, points=cloud), _odom(x=2.0, y=2.0))
    assert fused is not None
    assert np.allclose(fused.centroid[:2], [5.0, 2.0], atol=0.2)


# ------------------------------------------------------ depth plausibility (#199)


def _lamp_frustum():
    """Centre bearing/elevation + angular span of :func:`_lamp_detection`'s frustum
    (vehicle at the map origin, yaw 0), for placing on-ray synthetic clusters."""
    det = _lamp_detection()
    bearing_lo, span, el_lo, el_hi = _bbox_map_frustum(
        det, 0.0, T.DEFAULT_N_TILES, T.DEFAULT_TILE_HFOV, T.DEFAULT_TILE_VFOV
    )
    centre_bearing = bearing_lo + span / 2.0
    centre_elevation = (el_lo + el_hi) / 2.0
    angular_span = max(span, el_hi - el_lo)
    return det, centre_bearing, centre_elevation, angular_span


def test_ceiling_luminaire_component_rejected_for_wrong_depth():
    """Issue #199: two components share ONE detection's frustum -- a near, correct
    'lamp' cluster (table-height range) sitting slightly off the box's centre ray,
    and a far cluster sitting exactly ON the centre ray at a range that implies a
    physically implausible size for the class (the ceiling-luminaire mechanism:
    same bbox, wrong depth slab). Pure angular-deviation voting (pre-#199) would
    pick the far, angularly-dead-centre cluster; the depth-plausibility gate must
    reject it and keep the near, correctly-sized one even though it deviates more
    from the centre ray."""
    det, centre_bearing, centre_elevation, angular_span = _lamp_frustum()
    apex = np.zeros(3)

    near_range = 3.0  # implied size 3.0 * angular_span ~= 1.79 m -- plausible for lamp (typ long axis 0.863 m)
    far_range = 9.0  # implied size ~= 5.37 m -- far past 2.5x the typical long axis

    near_true = _ray_cloud(
        apex, centre_bearing + 0.05, centre_elevation, near_range, half=0.1, n=25, seed=21,
    )
    far_decoy = _ray_cloud(
        apex, centre_bearing, centre_elevation, far_range, half=0.1, n=25, seed=22,
    )
    scan = LidarScan(t=0.0, points=np.vstack([near_true, far_decoy]).astype(np.float32))

    fused = fuse_detection(det, scan, _odom())
    assert fused is not None
    assert fused.range_m < 5.0  # picked the near cluster, not the far one
    assert abs(fused.centroid[2] - near_true[:, 2].mean()) < 0.3

    # Disabling the depth gate (an enormous factor) reproduces the pre-#199 bug --
    # the far, angularly-central cluster wins on deviation alone. This documents
    # the mechanism the gate closes, the same way test_lateral_clustering_tightens_
    # box_vs_old_cone_slab documents its own old-vs-new comparison.
    no_gate_cfg = FusionConfig(depth_size_factor=1e9)
    fused_no_gate = fuse_detection(det, scan, _odom(), no_gate_cfg)
    assert fused_no_gate is not None
    assert fused_no_gate.range_m > 7.0  # the old bug: picked the far decoy


def test_stray_chained_points_do_not_disqualify_the_real_component():
    """Post-review regression (issue #199): a verifier reproduced the depth gate
    disqualifying a REAL object component over a small minority of stray points
    chained in by the lateral union-find at an implausible range, handing the win
    to a uniform, worse-angled but fully plausible decoy. Reproduces that exact
    shape directly: a tight 20-point real cluster, a realistic continuous chain
    (each hop < cluster_radius, so it genuinely unions into ONE component the way
    fusion.py's own docstring warns a sparse far-range surface can) stepping out
    to 2 individually depth-implausible points, versus a uniform, worse-angled,
    fully depth-plausible ghost far enough away to stay its own component.

    The old (pre-review) gate used the component's MEAN range -- close to a
    unanimity test in practice. Passing ``depth_plausible_frac=1.0`` (literal
    unanimity) reproduces the verifier's failure directly against this module's
    own code: the real component's minority of implausible points disqualifies
    it, so the ghost wins on being the only 'plausible' candidate. The shipped
    default (0.5, a plain majority) must instead keep the real, better-angled
    component -- its actual plausible fraction here is 28/30 (~0.93), so this
    demonstrates a wide margin, not a threshold tuned to one case."""
    det, centre_bearing, centre_elevation, _ = _lamp_frustum()
    apex = np.zeros(3)

    core = _ray_cloud(apex, centre_bearing, centre_elevation, 1.2, half=0.05, n=20, seed=11)
    # a continuous chain, each hop 0.3 m (< cluster_radius=0.35) so it unions with
    # the core into one component; tail 2 of 10 steps clear the ~3.615 m depth
    # threshold for 'lamp' at this bbox's angular span, the rest stay under it.
    chain_ranges = np.arange(1.35, 4.2, 0.3)
    chain = np.vstack([
        _ray_cloud(apex, centre_bearing, centre_elevation, r, half=0.02, n=1, seed=100 + i)
        for i, r in enumerate(chain_ranges)
    ])
    # uniform, worse-angled ghost -- individually depth-plausible at 2.2 m, but far
    # enough off both bearing and elevation to stay a separate component and to
    # lose to the real object on angular deviation whenever the real object is
    # actually in the candidate pool.
    ghost = _ray_cloud(
        apex, centre_bearing + 0.15, centre_elevation + 0.15, 2.2, half=0.08, n=20, seed=50,
    )
    scan = LidarScan(t=0.0, points=np.vstack([core, chain, ghost]).astype(np.float32))

    unanimity_cfg = FusionConfig(depth_plausible_frac=1.0)  # the pre-review bug
    fused_unanimity = fuse_detection(det, scan, _odom(), unanimity_cfg)
    assert fused_unanimity is not None
    assert fused_unanimity.range_m > 2.0  # reproduces the verifier's failure: picks the ghost

    fused_default = fuse_detection(det, scan, _odom(), DEFAULT_FUSION_CONFIG)
    assert fused_default is not None
    assert fused_default.range_m < 2.0  # majority vote (0.5): keeps the real, near component
    assert abs(fused_default.centroid[2] - core[:, 2].mean()) < 0.3


def test_depth_gate_fails_open_with_no_class_prior():
    """A class absent from dimension_priors must not be second-guessed by the
    depth gate -- the far cluster, exactly on the centre ray, still wins on
    angular deviation exactly as before #199."""
    det, centre_bearing, centre_elevation, angular_span = _lamp_frustum()
    det = Detection(tile_id=det.tile_id, bbox_xyxy=det.bbox_xyxy, label="not-a-real-class", score=0.9)
    apex = np.zeros(3)

    near_true = _ray_cloud(apex, centre_bearing + 0.05, centre_elevation, 3.0, half=0.1, n=25, seed=23)
    far_decoy = _ray_cloud(apex, centre_bearing, centre_elevation, 9.0, half=0.1, n=25, seed=24)
    scan = LidarScan(t=0.0, points=np.vstack([near_true, far_decoy]).astype(np.float32))

    fused = fuse_detection(det, scan, _odom())
    assert fused is not None
    assert fused.range_m > 7.0  # no prior -> gate fails open, deviation alone decides


# ------------------------------------------------------ robust outlier core (#199)


def test_robust_core_mask_leaves_clean_uniform_cluster_untouched():
    """A clean, bounded uniform cluster (this module's own fixture shape) must
    never be trimmed by the MAD-normalized core filter -- unlike a percentile-rank
    trim, which always clips some fixed fraction of points regardless of whether
    they are genuine outliers."""
    cloud = _box_cloud(3.0, 0.0, 0.5, half=0.2, n=200, seed=99)
    mask = robust_core_mask(cloud, k=DEFAULT_FUSION_CONFIG.outlier_k, min_mad=DEFAULT_FUSION_CONFIG.outlier_min_mad)
    assert mask.all()


def test_robust_core_mask_drops_a_genuine_outlier():
    cloud = _box_cloud(3.0, 0.0, 0.5, half=0.15, n=30, seed=98)
    outlier = np.array([[3.0, 0.0, 4.0]], dtype=np.float32)  # 3.5 m off in z alone
    pts = np.vstack([cloud, outlier])
    mask = robust_core_mask(pts, k=DEFAULT_FUSION_CONFIG.outlier_k, min_mad=DEFAULT_FUSION_CONFIG.outlier_min_mad)
    assert mask.sum() == 30
    assert not mask[-1]


def test_fuse_detection_drops_outlier_and_recentres_on_synthetic_bug_geometry():
    """A component picked up a few stray background points via the lateral
    union-find (issue #199's other failure mode: extent/centroid quality, not
    identity) -- the returned centroid and point set must reflect the clean
    core, not be pulled/inflated by the stray points."""
    det = _front_detection()
    core = _box_cloud(3.0, 0.0, 0.5, half=0.15, n=40, seed=25)
    # a couple of straggler points within cluster_radius chain distance of the
    # core (so they join the SAME lateral component) but well outside the core's
    # own spread -- the MAD test catches these; a percentile-rank trim's fixed 2%
    # cutoff might not, at this sample size.
    stragglers = np.array(
        [[3.6, 0.55, 0.5], [3.55, -0.5, 0.9]], dtype=np.float32,
    )
    scan = LidarScan(t=0.0, points=np.vstack([core, stragglers]).astype(np.float32))

    loose_cfg = FusionConfig(cluster_radius=0.6)  # bridges core+stragglers into one component
    fused = fuse_detection(det, scan, _odom(), loose_cfg)
    assert fused is not None
    assert fused.n_points == 40  # the 2 stragglers were dropped by the core filter
    assert np.allclose(fused.centroid, [3.0, 0.0, 0.5], atol=0.1)
    # extent (proxy for downstream AABB volume, see tracker._fused_to_record) is
    # tight to the core, not stretched toward the stragglers
    assert fused.points[:, 1].max() - fused.points[:, 1].min() < 0.5
