"""Offline colored dense point-cloud reconstruction (developer debug tool).

Accumulate the raw lidar points of a *recorded* session and tint each by the
panorama pixel it forward-projects onto, then voxel-downsample and write a
colored PLY a human opens in CloudCompare / MeshLab / Open3D. The repo otherwise
keeps only a sparse instance map (``core/perception/scene_index.py``) and a 2D
occupancy grid (``core/nav/occupancy.py``, height discarded); neither lets a
developer look at what the robot reconstructed geometrically. A colored dense
cloud makes fusion / tracking / calibration bugs (a 2x-too-wide AABB, a mis-tuned
sign convention, a parallax offset) diagnosable by eye.

This is diagnostics, NOT the scored pipeline: it does not run under the 600 s
question budget, the checkpoint ledger, or the relaunch-per-question constraint
(those govern the live ``core/`` answer path only). It lives outside
``src/ai_module`` so it is obviously excluded from the eventual challenge fork.

The forward mapping (map-frame point -> panorama pixel) is the mirror image of
``fusion._points_in_frustum`` (apex-relative ``arctan2`` bearing/elevation)
composed with the calibrated ``tiling`` functions — no sign conventions are
re-derived here, they are pulled from ``tiling.AZIMUTH_SIGN`` /
``COLUMN0_YAW_OFFSET`` / ``ELEVATION_SIGN`` via the tiling functions.

Known v1 limitations (documented, acceptable for a debug tool):

* **No occlusion handling.** A lidar point behind a nearer surface still
  forward-projects to some pixel and picks up whatever color is there, so it can
  be tinted with the front surface's color. A per-scan z-buffer (keep only the
  nearest point per pixel) is the planned fix; multi-view accumulation
  self-corrects much of it in practice (a point occluded in one pano is seen
  directly in another). Misregistration deliberately stays visible as ghosting —
  that is the diagnostic signal, which is why coloring is single-source with no
  blending.
* **No motion deskew.** Each scan is colored with a single pose; fast rotation
  smears the near field.
* **Single nearest-in-time panorama per scan.** No co-visibility fusion across
  frames.

CLI::

    python -m tools.colored_cloud extract <bag_or_fixture_dir> <out.ply> \
        [--voxel 0.05] [--min-range 0.75] [--stride 1] [--ascii] [--drop-uncolored]
    python -m tools.colored_cloud info <bag_or_fixture_dir>
"""
from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass

import numpy as np

# Ensure <repo>/src is importable (also done by the package __init__ on -m import).
import tools  # noqa: F401  (side effect: sys.path bootstrap)

from core.interfaces import LidarScan, OdomState, PanoFrame  # noqa: F401  (OdomState re-exported)
from core.perception.pano_projection import (
    DEFAULT_MIN_RANGE,
    GRAY,
    _NEAR_ZERO_RANGE,  # noqa: F401  (re-exported for the T5 test pins)
    project_points_to_pano,
    sample_colors,
)
from core.replay.fixtures import INDEX_NAME, load_fixtures
from core.replay.replay_io import CH_PANO, CH_SCAN, MessageStore

from tools.ply_io import write_ply
from tools.voxel import voxel_downsample

# The map-point -> panorama-pixel projection and nearest-neighbour color sampling now
# live in core.perception.pano_projection (shared with the live colored voxel map).
# They are re-imported above so this module's public API (project_points_to_pano,
# sample_colors, GRAY, DEFAULT_MIN_RANGE) is unchanged for the T5 test pins.

#: Default voxel edge for the offline downsample (CLI flag, not a calibration entry).
DEFAULT_VOXEL = 0.05  # m


# --------------------------------------------------------------------------- accumulation


@dataclass(frozen=True)
class ExtractStats:
    """Summary of one :func:`accumulate_cloud` run (printed by the CLI)."""

    scans_used: int
    points_in: int
    points_colored: int
    points_gray: int
    voxels_out: int


def _nearest_pano(store: MessageStore, t: float) -> PanoFrame | None:
    """Panorama nearest in absolute time to ``t`` (None if the channel is empty).

    Nearest-in-absolute-time (not latest-<=) is deliberate: a scan is best colored
    by the closest view in either time direction, and for these fixtures pano and
    scan are bundled at the same keyframe timestamp so it resolves to the exact
    co-captured frame.
    """
    entries = store.channels.get(CH_PANO)
    if not entries:
        return None
    times = np.array([e[0] for e in entries], dtype=np.float64)
    idx = int(np.argmin(np.abs(times - t)))
    msg = entries[idx][1]
    return msg if isinstance(msg, PanoFrame) else None


def accumulate_cloud(
    store: MessageStore,
    *,
    voxel: float = DEFAULT_VOXEL,
    min_range: float = DEFAULT_MIN_RANGE,
    stride: int = 1,
    drop_uncolored: bool = False,
) -> tuple[np.ndarray, np.ndarray, ExtractStats]:
    """Accumulate a colored cloud from every (strided) scan in ``store``.

    For each :class:`LidarScan` in time order, the panorama nearest in time
    supplies both the pose (its attached :class:`PanoFrame.odom` is the apex/yaw)
    and the colors. Uncolored (out-of-VFOV / range-gated) points keep their
    geometry tinted :data:`GRAY` unless ``drop_uncolored``. Returns
    ``(points, colors, stats)`` with the cloud already voxel-downsampled.
    """
    scan_entries = store.channels.get(CH_SCAN) or []
    stride = max(1, int(stride))

    pts_chunks: list[np.ndarray] = []
    col_chunks: list[np.ndarray] = []
    scans_used = 0
    points_in = 0
    points_colored = 0

    for i in range(0, len(scan_entries), stride):
        t, msg = scan_entries[i]
        if not isinstance(msg, LidarScan) or msg.points is None or len(msg.points) == 0:
            continue
        pano = _nearest_pano(store, t)
        if pano is None:
            continue
        odom = pano.odom
        pts = np.asarray(msg.points, dtype=np.float32)

        rows, cols, valid = project_points_to_pano(pts, odom, min_range=min_range)
        colors = sample_colors(pano.image, rows, cols, valid)

        if drop_uncolored:
            pts = pts[valid]
            colors = colors[valid]
        if len(pts) == 0:
            continue

        pts_chunks.append(pts)
        col_chunks.append(colors)
        scans_used += 1
        points_in += len(pts)
        points_colored += int(valid.sum()) if not drop_uncolored else len(pts)

    if pts_chunks:
        all_pts = np.concatenate(pts_chunks, axis=0)
        all_cols = np.concatenate(col_chunks, axis=0)
    else:
        all_pts = np.zeros((0, 3), dtype=np.float32)
        all_cols = np.zeros((0, 3), dtype=np.uint8)

    out_pts, out_cols = voxel_downsample(all_pts, all_cols, voxel)

    stats = ExtractStats(
        scans_used=scans_used,
        points_in=points_in,
        points_colored=points_colored,
        points_gray=points_in - points_colored,
        voxels_out=len(out_pts),
    )
    return out_pts, out_cols, stats


# --------------------------------------------------------------------------- source loading


def _is_fixture_dir(path: str) -> bool:
    """A fixture dir is a directory containing an ``index.json`` (see fixtures.py)."""
    return os.path.isdir(path) and os.path.exists(os.path.join(path, INDEX_NAME))


def load_store(path) -> MessageStore:
    """Build a :class:`MessageStore` from a fixture directory or a bag path.

    Detection: a directory holding ``index.json`` is fixtures
    (:func:`core.replay.fixtures.load_fixtures`); anything else is treated as a
    bag (:meth:`MessageStore.from_bag`).
    """
    path = str(path)
    if _is_fixture_dir(path):
        return load_fixtures(path)
    return MessageStore.from_bag(path)


# --------------------------------------------------------------------------- CLI


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tools.colored_cloud", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    ex = sub.add_parser("extract", help="reconstruct a colored PLY from a session")
    ex.add_argument("source", help="fixture directory (index.json) or bag path")
    ex.add_argument("out", help="output .ply path")
    ex.add_argument("--voxel", type=float, default=DEFAULT_VOXEL,
                    help="voxel edge (m); 0 disables downsampling")
    ex.add_argument("--min-range", type=float, default=DEFAULT_MIN_RANGE,
                    help="drop coloring for points closer than this (m)")
    ex.add_argument("--stride", type=int, default=1,
                    help="use every Nth scan")
    ex.add_argument("--ascii", action="store_true",
                    help="write ASCII PLY instead of binary_little_endian")
    ex.add_argument("--drop-uncolored", action="store_true",
                    help="discard out-of-VFOV / range-gated points instead of graying them")

    info = sub.add_parser("info", help="print a source summary (scan/pano counts)")
    info.add_argument("source", help="fixture directory or bag path")
    return p


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)

    if args.cmd == "extract":
        t0 = time.perf_counter()
        store = load_store(args.source)
        pts, cols, stats = accumulate_cloud(
            store,
            voxel=args.voxel,
            min_range=args.min_range,
            stride=args.stride,
            drop_uncolored=args.drop_uncolored,
        )
        out_dir = os.path.dirname(os.path.abspath(args.out))
        os.makedirs(out_dir, exist_ok=True)
        nbytes = write_ply(args.out, pts, cols, ascii=args.ascii)
        dt = time.perf_counter() - t0

        gray_frac = (stats.points_gray / stats.points_in) if stats.points_in else 0.0
        print(
            f"wrote {args.out} ({nbytes / 1e6:.1f} MB, "
            f"{'ascii' if args.ascii else 'binary'})\n"
            f"  scans used   : {stats.scans_used}\n"
            f"  points in    : {stats.points_in}\n"
            f"  colored/gray : {stats.points_colored} / {stats.points_gray} "
            f"({gray_frac * 100:.1f}% gray)\n"
            f"  voxels out   : {stats.voxels_out}\n"
            f"  runtime      : {dt:.1f} s"
        )
        return 0

    if args.cmd == "info":
        store = load_store(args.source)
        n_scan = len(store.channels.get(CH_SCAN) or [])
        n_pano = len(store.channels.get(CH_PANO) or [])
        print(f"{args.source}: {n_scan} scans, {n_pano} panoramas")
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
