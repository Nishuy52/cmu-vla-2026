"""Packed-key numpy voxel downsampling for colored point clouds.

One occupied voxel -> one output point at the per-voxel **mean** position and
**mean** color (matches Open3D / PCL ``voxel_down_sample`` semantics). No Python
dict loop: voxel indices are packed into an int64 key, reduced with
``np.unique`` + ``np.bincount`` weighted sums.

Two correctness traps handled explicitly:

* The global min is subtracted before ``floor`` so negative coordinates do not
  truncate toward zero into the wrong voxel.
* Packing uses 21 bits/axis (``[0, 2**21)`` per axis) which covers a
  ~100 km span at 5 cm voxels — far beyond any indoor scene — inside a signed
  int64 without overflow.
"""
from __future__ import annotations

import numpy as np

#: Bits per axis in the packed int64 voxel key (3 * 21 = 63 < 64).
_BITS_PER_AXIS = 21
_AXIS_MAX = 1 << _BITS_PER_AXIS


def voxel_downsample(
    points: np.ndarray, colors: np.ndarray, voxel: float
) -> tuple[np.ndarray, np.ndarray]:
    """Downsample ``(points, colors)`` onto a ``voxel``-metre grid.

    ``points`` is (N, 3) float, ``colors`` is (N, 3) uint8. Returns
    ``(points_out, colors_out)`` with one row per occupied voxel: the mean
    position (float32) and mean color (uint8). ``voxel <= 0`` is a passthrough
    (returns contiguous float32/uint8 copies unchanged).
    """
    points = np.ascontiguousarray(points, dtype=np.float64)
    colors = np.ascontiguousarray(colors).astype(np.float64)
    if len(points) == 0:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.uint8),
        )
    if voxel is None or voxel <= 0.0:
        return (
            points.astype(np.float32, copy=False),
            np.clip(colors, 0, 255).astype(np.uint8),
        )

    origin = points.min(axis=0)
    idx = np.floor((points - origin) / voxel).astype(np.int64)  # (N, 3), >= 0
    if idx.max() >= _AXIS_MAX:
        raise ValueError(
            f"voxel grid exceeds {_AXIS_MAX} cells on an axis; increase voxel size "
            f"(span {(points.max(axis=0) - origin).max():.1f} m at {voxel} m voxels)"
        )

    keys = (
        (idx[:, 0].astype(np.int64) << (2 * _BITS_PER_AXIS))
        | (idx[:, 1].astype(np.int64) << _BITS_PER_AXIS)
        | idx[:, 2].astype(np.int64)
    )

    _, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
    n_vox = len(counts)
    counts = counts.astype(np.float64)

    pts_out = np.empty((n_vox, 3), dtype=np.float64)
    col_out = np.empty((n_vox, 3), dtype=np.float64)
    for c in range(3):
        pts_out[:, c] = np.bincount(inverse, weights=points[:, c], minlength=n_vox) / counts
        col_out[:, c] = np.bincount(inverse, weights=colors[:, c], minlength=n_vox) / counts

    return (
        pts_out.astype(np.float32),
        np.clip(np.round(col_out), 0, 255).astype(np.uint8),
    )
