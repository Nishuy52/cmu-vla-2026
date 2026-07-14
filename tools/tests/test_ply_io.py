"""PLY writer tests: binary + ASCII round-trip with a tiny manual parser."""
from __future__ import annotations

import numpy as np

from tools.ply_io import VERTEX_DTYPE, write_ply


def _parse_ply(path):
    """Minimal PLY reader for the exact format write_ply emits. Returns (pts, cols)."""
    with open(path, "rb") as fh:
        raw = fh.read()
    end = raw.index(b"end_header\n") + len(b"end_header\n")
    header = raw[:end].decode("ascii")
    body = raw[end:]

    n = None
    fmt = None
    for line in header.splitlines():
        if line.startswith("format "):
            fmt = line.split()[1]
        elif line.startswith("element vertex "):
            n = int(line.split()[2])
    assert n is not None and fmt is not None

    if fmt == "binary_little_endian":
        verts = np.frombuffer(body, dtype=VERTEX_DTYPE, count=n)
        pts = np.stack([verts["x"], verts["y"], verts["z"]], axis=1)
        cols = np.stack([verts["red"], verts["green"], verts["blue"]], axis=1)
        return pts.astype(np.float32), cols.astype(np.uint8)

    # ascii
    vals = [line.split() for line in body.decode("ascii").splitlines() if line.strip()]
    assert len(vals) == n
    arr = np.array(vals, dtype=float)
    return arr[:, :3].astype(np.float32), arr[:, 3:6].astype(np.uint8)


def _sample():
    pts = np.array(
        [[1.0, 2.0, 3.0], [-4.5, 0.25, 100.0], [0.0, 0.0, 0.0]], dtype=np.float32
    )
    cols = np.array([[255, 0, 0], [0, 128, 64], [10, 20, 30]], dtype=np.uint8)
    return pts, cols


def test_binary_roundtrip(tmp_path):
    pts, cols = _sample()
    out = tmp_path / "cloud.ply"
    nbytes = write_ply(out, pts, cols, ascii=False)
    assert nbytes == out.stat().st_size

    rp, rc = _parse_ply(out)
    np.testing.assert_allclose(rp, pts, atol=1e-6)
    np.testing.assert_array_equal(rc, cols)

    # Header must declare uchar color (the float-color interop trap).
    header = out.read_bytes()[:200].decode("ascii", "ignore")
    assert "property uchar red" in header
    assert "binary_little_endian" in header


def test_ascii_roundtrip(tmp_path):
    pts, cols = _sample()
    out = tmp_path / "cloud_ascii.ply"
    write_ply(out, pts, cols, ascii=True)
    rp, rc = _parse_ply(out)
    np.testing.assert_allclose(rp, pts, atol=1e-5)
    np.testing.assert_array_equal(rc, cols)
    assert "format ascii" in out.read_bytes()[:100].decode("ascii", "ignore")


def test_empty_cloud(tmp_path):
    out = tmp_path / "empty.ply"
    write_ply(out, np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8))
    rp, rc = _parse_ply(out)
    assert len(rp) == 0 and len(rc) == 0
