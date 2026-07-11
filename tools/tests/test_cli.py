"""End-to-end CLI smoke test: build a tiny synthetic session in-memory, run
extract main() to a temp PLY, and assert the points got the expected colors."""
from __future__ import annotations

import numpy as np

from core.interfaces import LidarScan, OdomState, PanoFrame
from core.perception import tiling
from core.replay.replay_io import CH_PANO, CH_SCAN, MessageStore
from tools import colored_cloud
from tools.colored_cloud import accumulate_cloud, main
from tools.tests.test_ply_io import _parse_ply
from tools.tests.test_projection import _point_along_ray


def _solid_pano(t, odom, rgb):
    img = np.zeros((tiling.PANO_HEIGHT, tiling.PANO_WIDTH, 3), dtype=np.uint8)
    img[:] = rgb
    return PanoFrame(t=t, image=img, odom=odom)


def _build_session():
    """Two frames, each a solid-color pano + a scan of points that project into it."""
    store = MessageStore()
    specs = [
        (0.0, OdomState(t=0.0, x=0.0, y=0.0, z=0.0, yaw=0.0), (200, 50, 50)),
        (1.0, OdomState(t=1.0, x=5.0, y=5.0, z=0.0, yaw=1.0), (50, 200, 50)),
    ]
    expected = []
    for t, odom, rgb in specs:
        store.add(CH_PANO, t, _solid_pano(t, odom, rgb))
        apex = (odom.x, odom.y, odom.z)
        # A handful of points well inside the VFOV band and beyond min-range.
        pts = np.array(
            [
                _point_along_ray(apex, odom.yaw + 0.5, np.deg2rad(10), 3.0),
                _point_along_ray(apex, odom.yaw - 0.5, np.deg2rad(-10), 4.0),
                _point_along_ray(apex, odom.yaw + 2.0, np.deg2rad(0), 5.0),
            ],
            dtype=np.float32,
        )
        store.add(CH_SCAN, t, LidarScan(t=t, points=pts))
        expected.append(rgb)
    return store.finalize(), expected


def test_accumulate_colors_match_solid_panos():
    store, expected = _build_session()
    # Disable voxel so every input point survives for a direct color check.
    pts, cols, stats = accumulate_cloud(store, voxel=0.0, min_range=0.75)
    assert stats.scans_used == 2
    assert stats.points_gray == 0  # all points placed inside the VFOV band
    present = {tuple(c) for c in np.unique(cols, axis=0)}
    for rgb in expected:
        assert rgb in present, (rgb, present)


def test_extract_main_writes_ply(tmp_path, monkeypatch):
    store, expected = _build_session()
    # Route load_store to our synthetic store regardless of the source arg.
    monkeypatch.setattr(colored_cloud, "load_store", lambda _p: store)

    out = tmp_path / "synth.ply"
    rc = main(["extract", "ignored", str(out), "--voxel", "0.0"])
    assert rc == 0
    assert out.exists()

    rp, rcols = _parse_ply(out)
    assert len(rp) == 6  # 3 points x 2 scans
    present = {tuple(c) for c in np.unique(rcols, axis=0)}
    for rgb in expected:
        assert rgb in present


def test_drop_uncolored(tmp_path):
    """Out-of-VFOV points are dropped under --drop-uncolored; kept gray otherwise."""
    store = MessageStore()
    odom = OdomState(t=0.0, x=0.0, y=0.0, z=0.0, yaw=0.0)
    store.add(CH_PANO, 0.0, _solid_pano(0.0, odom, (10, 20, 30)))
    apex = (0.0, 0.0, 0.0)
    pts = np.array(
        [
            _point_along_ray(apex, 0.3, np.deg2rad(10), 3.0),   # in band
            _point_along_ray(apex, 0.3, np.deg2rad(80), 3.0),   # out of VFOV
        ],
        dtype=np.float32,
    )
    store.add(CH_SCAN, 0.0, LidarScan(t=0.0, points=pts))
    store.finalize()

    _, cols_keep, st_keep = accumulate_cloud(store, voxel=0.0, drop_uncolored=False)
    assert st_keep.points_in == 2 and st_keep.points_gray == 1
    assert any(tuple(c) == (128, 128, 128) for c in cols_keep)

    pts_drop, _, st_drop = accumulate_cloud(store, voxel=0.0, drop_uncolored=True)
    assert st_drop.points_in == 1 and len(pts_drop) == 1
