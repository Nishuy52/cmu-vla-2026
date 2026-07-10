"""Lidar fusion: synthetic box -> centroid, frustum/min_points rejection, wrap seam."""
from __future__ import annotations

import numpy as np

from core.interfaces import LidarScan, OdomState
from core.perception import tiling as T
from core.perception.detector import Detection
from core.perception.fusion import (
    DEFAULT_FUSION_CONFIG,
    FusionConfig,
    fuse_detection,
)


def _front_detection(tile_id=0):
    spec = T.tile_specs()[tile_id]
    bb = (spec.cx - 60, spec.cy - 120, spec.cx + 60, spec.cy + 60)
    return Detection(tile_id=tile_id, bbox_xyxy=bb, label="sofa", score=0.9)


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


def test_apex_offset_by_odom_position():
    """Frustum apex follows the vehicle: object at map (5,2) seen from (2,2)."""
    det = _front_detection()  # front tile, yaw 0 -> looks along +x
    cloud = _box_cloud(5.0, 2.0, 0.5, half=0.15, n=40, seed=8)
    fused = fuse_detection(det, LidarScan(t=0.0, points=cloud), _odom(x=2.0, y=2.0))
    assert fused is not None
    assert np.allclose(fused.centroid[:2], [5.0, 2.0], atol=0.2)
