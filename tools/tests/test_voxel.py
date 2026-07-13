"""Voxel-downsample tests: hand-computed means, negative coords, passthrough."""
from __future__ import annotations

import numpy as np

from tools.voxel import voxel_downsample


def test_exact_means_tiny_case():
    """Two points in one voxel average; a third in its own voxel stays put."""
    pts = np.array(
        [
            [0.01, 0.01, 0.01],  # voxel A
            [0.03, 0.03, 0.03],  # voxel A (same 0.1 cell)
            [0.55, 0.55, 0.55],  # voxel B
        ],
        dtype=np.float32,
    )
    cols = np.array(
        [
            [10, 20, 30],
            [30, 40, 50],
            [200, 100, 0],
        ],
        dtype=np.uint8,
    )
    out_pts, out_cols = voxel_downsample(pts, cols, voxel=0.1)
    assert len(out_pts) == 2

    # Sort output by x for deterministic comparison.
    order = np.argsort(out_pts[:, 0])
    out_pts = out_pts[order]
    out_cols = out_cols[order]

    np.testing.assert_allclose(out_pts[0], [0.02, 0.02, 0.02], atol=1e-6)
    np.testing.assert_allclose(out_pts[1], [0.55, 0.55, 0.55], atol=1e-5)
    # Mean color of voxel A: (20, 30, 40).
    np.testing.assert_array_equal(out_cols[0], [20, 30, 40])
    np.testing.assert_array_equal(out_cols[1], [200, 100, 0])


def test_negative_coordinates():
    """Negative coords must not truncate toward zero: points straddling 0 that are
    within one voxel collapse to a single voxel."""
    pts = np.array(
        [
            [-0.02, -0.02, -0.02],
            [-0.04, -0.04, -0.04],
        ],
        dtype=np.float32,
    )
    cols = np.array([[100, 100, 100], [200, 200, 200]], dtype=np.uint8)
    out_pts, out_cols = voxel_downsample(pts, cols, voxel=0.1)
    assert len(out_pts) == 1
    np.testing.assert_allclose(out_pts[0], [-0.03, -0.03, -0.03], atol=1e-6)
    np.testing.assert_array_equal(out_cols[0], [150, 150, 150])


def test_negative_split_into_two_voxels():
    """Two points more than a voxel apart across zero stay in separate voxels."""
    pts = np.array([[-0.15, 0.0, 0.0], [0.15, 0.0, 0.0]], dtype=np.float32)
    cols = np.array([[10, 10, 10], [20, 20, 20]], dtype=np.uint8)
    out_pts, _ = voxel_downsample(pts, cols, voxel=0.1)
    assert len(out_pts) == 2


def test_passthrough_voxel_zero():
    pts = np.array([[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]], dtype=np.float32)
    cols = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8)
    out_pts, out_cols = voxel_downsample(pts, cols, voxel=0.0)
    assert len(out_pts) == 2  # no reduction
    np.testing.assert_array_equal(out_pts, pts)
    np.testing.assert_array_equal(out_cols, cols)


def test_empty():
    out_pts, out_cols = voxel_downsample(
        np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8), voxel=0.05
    )
    assert out_pts.shape == (0, 3)
    assert out_cols.shape == (0, 3)
    assert out_pts.dtype == np.float32
    assert out_cols.dtype == np.uint8
