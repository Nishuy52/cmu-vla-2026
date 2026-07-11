"""Zero-dependency PLY writer for colored point clouds.

Two formats, both hand-written so the only dependency is numpy:

* ``binary_little_endian`` (default) — a numpy structured array
  ``[x,y,z: <f4] + [red,green,blue: u1]`` dumped via ``.tobytes()`` behind a
  short ASCII header. 3-5x smaller and roughly an order of magnitude faster to
  write/parse than ASCII (~15 B/pt vs 45-60 B/pt).
* ``ascii`` — the PLY 1.0 ASCII variant, for eyeballing the file directly.

Colors are written as ``uchar red/green/blue`` (0-255), never float 0-1: float
color is the classic "cloud renders black/red" interop bug in CloudCompare /
MeshLab / Open3D.
"""
from __future__ import annotations

import numpy as np

#: Structured dtype of one binary_little_endian PLY vertex (matches the header).
VERTEX_DTYPE = np.dtype(
    [
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]
)


def _header(n: int, fmt: str) -> str:
    """Return the ASCII PLY header for ``n`` vertices in ``fmt`` (ends with newline)."""
    return (
        "ply\n"
        f"format {fmt} 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )


def _as_vertex_array(points: np.ndarray, colors: np.ndarray) -> np.ndarray:
    """Pack ``points`` (N, 3) float + ``colors`` (N, 3) uint8 into :data:`VERTEX_DTYPE`."""
    points = np.ascontiguousarray(points, dtype=np.float32)
    colors = np.ascontiguousarray(np.clip(colors, 0, 255)).astype(np.uint8)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must be (N, 3), got {points.shape}")
    if colors.shape != points.shape:
        raise ValueError(f"colors must match points shape {points.shape}, got {colors.shape}")
    verts = np.empty(len(points), dtype=VERTEX_DTYPE)
    verts["x"] = points[:, 0]
    verts["y"] = points[:, 1]
    verts["z"] = points[:, 2]
    verts["red"] = colors[:, 0]
    verts["green"] = colors[:, 1]
    verts["blue"] = colors[:, 2]
    return verts


def write_ply(path, points: np.ndarray, colors: np.ndarray, *, ascii: bool = False) -> int:
    """Write a colored point cloud to ``path`` as PLY. Returns the byte count written.

    ``points`` is (N, 3) float, ``colors`` is (N, 3) uint8 RGB. Binary
    little-endian by default; ``ascii=True`` writes the ASCII 1.0 variant.
    """
    verts = _as_vertex_array(points, colors)
    n = len(verts)
    if ascii:
        header = _header(n, "ascii").encode("ascii")
        # Interleave float xyz with integer rgb per row; keep it simple and explicit.
        lines = [
            f"{v['x']:.6f} {v['y']:.6f} {v['z']:.6f} "
            f"{int(v['red'])} {int(v['green'])} {int(v['blue'])}\n"
            for v in verts
        ]
        body = "".join(lines).encode("ascii")
        with open(path, "wb") as fh:
            fh.write(header)
            fh.write(body)
        return len(header) + len(body)

    header = _header(n, "binary_little_endian").encode("ascii")
    body = verts.tobytes()
    with open(path, "wb") as fh:
        fh.write(header)
        fh.write(body)
    return len(header) + len(body)
