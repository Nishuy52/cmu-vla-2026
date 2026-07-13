"""ColoredVoxelMap: ingest math, running means, gates, negative coords, and the
load-bearing equivalence invariant vs the T5 offline batch tool on jingfan keyframes.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

from core.interfaces import LidarScan, OdomState, PanoFrame
from core.perception import tiling
from core.perception.colored_map import (
    ColoredMapConfig,
    ColoredVoxelMap,
)
from core.perception.pano_projection import GRAY, project_points_to_pano, sample_colors
from core.replay.fixtures import load_fixtures
from core.replay.replay_io import CH_PANO, CH_SCAN

# --- repo-root path so the equivalence test can import the T5 batch downsample --------
_REPO = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
_FIXTURE_DIR = os.path.join(_REPO, "data", "fixtures", "jingfan")


def _point_along_ray(apex, bearing, elevation, r):
    dx = r * np.cos(elevation) * np.cos(bearing)
    dy = r * np.cos(elevation) * np.sin(bearing)
    dz = r * np.sin(elevation)
    return np.array(apex, dtype=float) + np.array([dx, dy, dz])


def _solid_pano(t, odom, rgb):
    img = np.zeros((tiling.PANO_HEIGHT, tiling.PANO_WIDTH, 3), dtype=np.uint8)
    img[:] = rgb
    return PanoFrame(t=t, image=img, odom=odom)


# --------------------------------------------------------------------------- unit tests


def test_running_mean_position_and_color():
    """Two points in one voxel average to the voxel mean; a third is its own voxel."""
    odom = OdomState(t=0.0, x=0.0, y=0.0, z=0.0, yaw=0.0)
    apex = (0.0, 0.0, 0.0)
    # Three colored points well inside the VFOV band and beyond min-range.
    p0 = _point_along_ray(apex, 0.3, np.deg2rad(5), 3.0)
    p1 = p0 + np.array([0.02, 0.0, 0.0])  # same 0.05 voxel as p0
    p2 = _point_along_ray(apex, -0.3, np.deg2rad(-5), 4.0)
    pts = np.array([p0, p1, p2], dtype=np.float32)
    pano = _solid_pano(0.0, odom, (100, 150, 200))

    m = ColoredVoxelMap(ColoredMapConfig(voxel_m=0.05, min_range=0.75))
    stats = m.ingest(LidarScan(t=0.0, points=pts), pano)
    assert stats.points_in == 3 and stats.points_gray == 0
    assert m.n_voxels == 2

    xyz, rgb = m.to_arrays()
    order = np.argsort(xyz[:, 0])
    xyz, rgb = xyz[order], rgb[order]
    # The merged voxel mean position is the average of p0/p1 (same cell).
    merged = pts[:2].mean(axis=0)
    # Which output row is the merged one: the cell containing p0.
    d = np.linalg.norm(xyz - merged[None, :], axis=1)
    mi = int(np.argmin(d))
    np.testing.assert_allclose(xyz[mi], merged, atol=1e-4)
    np.testing.assert_array_equal(rgb[mi], (100, 150, 200))


def test_multi_ingest_refines_same_voxel():
    """A voxel observed across two ingests carries the mean over BOTH scans' points."""
    odom = OdomState(t=0.0, x=0.0, y=0.0, z=0.0, yaw=0.0)
    p = _point_along_ray((0.0, 0.0, 0.0), 0.3, 0.0, 3.0).astype(np.float32)
    m = ColoredVoxelMap(ColoredMapConfig(voxel_m=0.05, min_range=0.75))
    m.ingest(LidarScan(t=0.0, points=p[None, :]), _solid_pano(0.0, odom, (0, 0, 0)))
    # Second point in the same voxel, different color -> mean color.
    p2 = (p + np.array([0.01, 0.0, 0.0], dtype=np.float32))
    m.ingest(LidarScan(t=1.0, points=p2[None, :]), _solid_pano(1.0, odom, (200, 100, 40)))
    assert m.n_voxels == 1
    xyz, rgb = m.to_arrays()
    np.testing.assert_allclose(xyz[0], (p.astype(np.float64) + p2) / 2.0, atol=1e-4)
    np.testing.assert_array_equal(rgb[0], (100, 50, 20))  # round((0+200)/2)=100 etc.


def test_gray_policy_keep_vs_drop():
    """Out-of-VFOV points contribute gray by default; keep_uncolored=False drops them."""
    odom = OdomState(t=0.0, x=0.0, y=0.0, z=0.0, yaw=0.0)
    in_band = _point_along_ray((0.0, 0.0, 0.0), 0.3, np.deg2rad(10), 3.0)
    out_band = _point_along_ray((0.0, 0.0, 0.0), 0.3, np.deg2rad(80), 3.0)
    pts = np.array([in_band, out_band], dtype=np.float32)
    pano = _solid_pano(0.0, odom, (10, 20, 30))

    keep = ColoredVoxelMap(ColoredMapConfig(voxel_m=0.05, keep_uncolored=True))
    st = keep.ingest(LidarScan(t=0.0, points=pts), pano)
    assert st.points_in == 2 and st.points_gray == 1
    _, rgb = keep.to_arrays()
    assert any(tuple(c) == tuple(GRAY) for c in rgb)

    drop = ColoredVoxelMap(ColoredMapConfig(voxel_m=0.05, keep_uncolored=False))
    st2 = drop.ingest(LidarScan(t=0.0, points=pts), pano)
    assert st2.points_in == 1 and drop.n_voxels == 1


def test_negative_coordinates_pack_correctly():
    """Points at negative map coords bin correctly under the origin-margin offset."""
    odom = OdomState(t=0.0, x=-10.0, y=-10.0, z=-1.0, yaw=0.0)
    apex = (-10.0, -10.0, -1.0)
    p0 = _point_along_ray(apex, 0.3, 0.0, 3.0)
    p1 = p0 + np.array([0.02, 0.0, 0.0])
    pts = np.array([p0, p1], dtype=np.float32)
    pano = _solid_pano(0.0, odom, (50, 60, 70))
    m = ColoredVoxelMap(ColoredMapConfig(voxel_m=0.05, min_range=0.75))
    m.ingest(LidarScan(t=0.0, points=pts), pano)
    assert m.n_voxels == 1  # both in one voxel, negative coords handled
    xyz, _ = m.to_arrays()
    np.testing.assert_allclose(xyz[0], pts.mean(axis=0), atol=1e-4)


def test_empty_and_readout():
    m = ColoredVoxelMap()
    xyz, rgb = m.to_arrays()
    assert xyz.shape == (0, 3) and rgb.shape == (0, 3)
    assert xyz.dtype == np.float32 and rgb.dtype == np.uint8
    st = m.ingest(LidarScan(t=0.0, points=np.zeros((0, 3), np.float32)),
                  _solid_pano(0.0, OdomState(0, 0, 0, 0, 0), (0, 0, 0)))
    assert st.points_in == 0 and m.n_voxels == 0


def test_capacity_growth_beyond_initial():
    """Ingesting more unique voxels than the initial capacity grows the arrays correctly."""
    from core.perception import colored_map as cm

    odom = OdomState(t=0.0, x=0.0, y=0.0, z=0.0, yaw=0.0)
    # Spread points along x so each lands in its own voxel; exceed a small capacity.
    n = 200
    xs = np.arange(n) * 0.5 + 2.0
    pts = np.stack([xs, np.zeros(n), np.zeros(n)], axis=1).astype(np.float32)
    pano = _solid_pano(0.0, odom, (5, 5, 5))
    orig_cap = cm._INITIAL_CAPACITY
    try:
        cm._INITIAL_CAPACITY = 16  # force several doublings
        m = ColoredVoxelMap(ColoredMapConfig(voxel_m=0.05, min_range=0.75))
        m.ingest(LidarScan(t=0.0, points=pts), pano)
    finally:
        cm._INITIAL_CAPACITY = orig_cap
    assert m.n_voxels == n
    xyz, _ = m.to_arrays()
    assert len(xyz) == n
    np.testing.assert_allclose(np.sort(xyz[:, 0]), xs, atol=1e-3)


# --------------------------------------------------------------------------- equivalence


def _batch_reference(per_frame, voxel, min_range):
    """The T5 offline batch path: concatenate all colored points, then voxel_downsample."""
    from tools.voxel import voxel_downsample  # T5 code (repo root on sys.path above)

    all_pts, all_cols = [], []
    for scan, pano in per_frame:
        pts = np.asarray(scan.points, dtype=np.float32)
        rows, cols, valid = project_points_to_pano(pts, pano.odom, min_range=min_range)
        colors = sample_colors(pano.image, rows, cols, valid)
        all_pts.append(pts)
        all_cols.append(colors)
    P = np.concatenate(all_pts, axis=0)
    C = np.concatenate(all_cols, axis=0)
    return voxel_downsample(P, C, voxel), P


def _voxel_key_set(xyz, voxel, origin):
    idx = np.floor((xyz.astype(np.float64) - origin[None, :]) / voxel).astype(np.int64)
    return {tuple(r) for r in idx}


def _by_voxel(xyz, rgb, voxel, origin):
    out = {}
    idx = np.floor((xyz.astype(np.float64) - origin[None, :]) / voxel).astype(np.int64)
    for k, x, c in zip(map(tuple, idx), xyz, rgb):
        out[k] = (x, c)
    return out


@pytest.mark.skipif(
    not os.path.isdir(_FIXTURE_DIR), reason="jingfan fixtures not present"
)
@pytest.mark.slow
def test_online_equals_offline_on_jingfan_keyframes():
    """Feeding N (scan, nearest-pano) keyframes through ColoredVoxelMap must reproduce the
    T5 batch tool's voxel set + per-voxel mean position/color, within float tolerance.

    Grids are aligned by handing the online map the same origin the batch tool uses
    (the global min of all points); with the grid aligned the accumulation math is the
    invariant under test, and it must agree exactly modulo float32 rounding.
    """
    store = load_fixtures(_FIXTURE_DIR)
    scan_entries = store.channels[CH_SCAN]
    pano_entries = store.channels[CH_PANO]
    pano_t = np.array([e[0] for e in pano_entries])

    # First 10 keyframes that carry a scan; pair each with its nearest-in-time pano
    # (which for these co-captured fixtures is the same-keyframe pano, exactly as the
    # offline tool's _nearest_pano resolves).
    per_frame = []
    for t, scan in scan_entries:
        if not isinstance(scan, LidarScan) or scan.points is None or len(scan.points) == 0:
            continue
        idx = int(np.argmin(np.abs(pano_t - t)))
        pano = pano_entries[idx][1]
        if isinstance(pano, PanoFrame):
            per_frame.append((scan, pano))
        if len(per_frame) >= 10:
            break
    assert len(per_frame) >= 8, "need a handful of scan+pano keyframes for equivalence"

    voxel, min_range = 0.05, 0.75
    (batch_xyz, batch_rgb), P = _batch_reference(per_frame, voxel, min_range)

    # Align the online grid to the batch grid (batch origin = global min of all points).
    origin = P.astype(np.float64).min(axis=0)
    m = ColoredVoxelMap(
        ColoredMapConfig(voxel_m=voxel, min_range=min_range), origin=origin
    )
    for scan, pano in per_frame:
        m.ingest(scan, pano)
    online_xyz, online_rgb = m.to_arrays()

    # Same voxel set.
    assert len(online_xyz) == len(batch_xyz)
    assert _voxel_key_set(online_xyz, voxel, origin) == _voxel_key_set(
        batch_xyz, voxel, origin
    )

    # Same per-voxel mean position (float32) and mean color (uint8), keyed by voxel.
    b = _by_voxel(batch_xyz, batch_rgb, voxel, origin)
    o = _by_voxel(online_xyz, online_rgb, voxel, origin)
    max_pos_err = 0.0
    for k, (bx, bc) in b.items():
        ox, oc = o[k]
        max_pos_err = max(max_pos_err, float(np.max(np.abs(ox - bx))))
        # Color mean rounding can differ by at most 1 LSB between summation orders.
        assert np.all(np.abs(oc.astype(int) - bc.astype(int)) <= 1), (k, oc, bc)
    assert max_pos_err < 1e-3, max_pos_err
