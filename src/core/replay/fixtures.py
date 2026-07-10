"""Distillation path: a large bag -> compact, portable ``.npz`` keyframe fixtures.

A 3 GB bag distils to a few hundred MB of fixtures that move freely between machines
(the real bag may live on a remote Linux box; fixtures can be extracted on the cluster and
only the fixtures transferred). Each *keyframe* bundles the newest pano/scan/terrain/odom
at that bag time into one compressed ``.npz``; a schema-versioned ``index.json`` records
the keyframe order, timestamps, and any latched question.

Keyframe selection is a movement gate: a new keyframe is emitted when EITHER ``stride``
camera frames have elapsed, OR the vehicle has moved ``move_m`` metres / rotated
``rot_rad`` radians since the last keyframe (whichever fires first). Scans are decimated to
cap fixture size.

CLI::

    python -m core.replay.fixtures extract <bag> <out_dir> [--stride N] [--move-m M]
                                                           [--rot-deg D] [--max-scan-pts K]
    python -m core.replay.fixtures info <fixture_dir>
"""
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass

import numpy as np

from core.interfaces import (
    LidarScan,
    OdomState,
    PanoFrame,
    Question,
    TerrainPatch,
)
from core.replay.replay_io import (
    CH_ODOM,
    CH_PANO,
    CH_QUESTION,
    CH_SCAN,
    CH_TERRAIN,
    CH_TERRAIN_EXT,
    MessageStore,
)

SCHEMA_VERSION = 1
INDEX_NAME = "index.json"


@dataclass(frozen=True)
class ExtractParams:
    """Keyframe / movement-gate + decimation tunables."""

    stride: int = 5  # emit at least every `stride` camera frames
    move_m: float = 0.5  # or after this much translation (m)
    rot_rad: float = math.radians(30.0)  # or this much rotation (rad)
    max_scan_pts: int = 60_000  # decimate scans above this many points
    max_terrain_pts: int = 40_000  # decimate terrain clouds above this


def _decimate(pts: np.ndarray, cap: int) -> np.ndarray:
    """Uniform-stride decimation to at most ``cap`` rows (deterministic)."""
    n = len(pts)
    if cap <= 0 or n <= cap:
        return np.ascontiguousarray(pts)
    step = int(math.ceil(n / cap))
    return np.ascontiguousarray(pts[::step][:cap])


def _moved(a: OdomState, b: OdomState, move_m: float, rot_rad: float) -> bool:
    d = math.hypot(a.x - b.x, a.y - b.y)
    dyaw = abs(math.atan2(math.sin(a.yaw - b.yaw), math.cos(a.yaw - b.yaw)))
    return d >= move_m or dyaw >= rot_rad


def extract_fixtures(bag_path, out_dir, params: ExtractParams | None = None) -> dict:
    """Extract keyframe fixtures from ``bag_path`` into ``out_dir``.

    Returns the written index dict. Keyframes are chosen by the movement gate in
    :class:`ExtractParams`; each is written as ``keyframe_<NNNN>.npz`` holding the
    newest ``pano`` (uint8), ``scan`` (float32, decimated), ``terrain``/``terrain_ext``
    (float32), and ``odom`` (float32 [t,x,y,z,yaw]) available at that time.
    """
    params = params or ExtractParams()
    out_dir = str(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    store = MessageStore.from_bag(bag_path)

    # The camera stream drives keyframe cadence; if absent, fall back to the scan stream.
    driver = CH_PANO if store.times(CH_PANO) else CH_SCAN
    driver_times = store.times(driver)

    # First non-empty question is latched (mirrors the 1 Hz republish / latch behaviour).
    question_text: str | None = None
    question_t: float | None = None
    for _, msg in store.channels.get(CH_QUESTION, []):
        if isinstance(msg, Question) and msg.text:
            question_text = msg.text
            question_t = float(msg.t_received)
            break

    keyframes: list[dict] = []
    last_odom: OdomState | None = None
    frames_since = 0

    for i, t in enumerate(driver_times):
        odom = store.latest(CH_ODOM, t)
        frames_since += 1
        gate = i == 0 or frames_since >= params.stride
        if not gate and last_odom is not None and isinstance(odom, OdomState):
            gate = _moved(odom, last_odom, params.move_m, params.rot_rad)
        if not gate:
            continue
        frames_since = 0
        if isinstance(odom, OdomState):
            last_odom = odom

        idx = len(keyframes)
        arrays: dict[str, np.ndarray] = {}

        pano = store.latest(CH_PANO, t)
        if isinstance(pano, PanoFrame):
            arrays["pano"] = pano.image.astype(np.uint8, copy=False)
        scan = store.latest(CH_SCAN, t)
        if isinstance(scan, LidarScan):
            arrays["scan"] = _decimate(scan.points, params.max_scan_pts).astype(
                np.float32, copy=False
            )
        terr = store.latest(CH_TERRAIN, t)
        if isinstance(terr, TerrainPatch):
            arrays["terrain"] = _decimate(terr.points, params.max_terrain_pts).astype(
                np.float32, copy=False
            )
        terr_ext = store.latest(CH_TERRAIN_EXT, t)
        if isinstance(terr_ext, TerrainPatch):
            arrays["terrain_ext"] = _decimate(
                terr_ext.points, params.max_terrain_pts
            ).astype(np.float32, copy=False)
        if isinstance(odom, OdomState):
            arrays["odom"] = np.array(
                [odom.t, odom.x, odom.y, odom.z, odom.yaw], dtype=np.float32
            )

        fname = f"keyframe_{idx:04d}.npz"
        np.savez_compressed(os.path.join(out_dir, fname), **arrays)
        keyframes.append(
            {"file": fname, "t": float(t), "channels": sorted(arrays.keys())}
        )

    index = {
        "schema_version": SCHEMA_VERSION,
        "source": os.path.basename(str(bag_path)),
        "n_keyframes": len(keyframes),
        "params": {
            "stride": params.stride,
            "move_m": params.move_m,
            "rot_rad": params.rot_rad,
            "max_scan_pts": params.max_scan_pts,
            "max_terrain_pts": params.max_terrain_pts,
        },
        "question": (
            {"text": question_text, "t_received": question_t}
            if question_text is not None
            else None
        ),
        "keyframes": keyframes,
    }
    with open(os.path.join(out_dir, INDEX_NAME), "w", encoding="utf-8") as fh:
        json.dump(index, fh, indent=2)
    return index


def load_fixtures(fixture_dir) -> MessageStore:
    """Load a fixture directory into the same time-indexed store ReplayRobotIO uses.

    Reconstructs :class:`PanoFrame` / :class:`LidarScan` / :class:`TerrainPatch` /
    :class:`OdomState` per keyframe and the latched :class:`Question`, so the returned
    store is drop-in for :class:`~core.replay.replay_io.ReplayRobotIO`.
    """
    fixture_dir = str(fixture_dir)
    with open(os.path.join(fixture_dir, INDEX_NAME), encoding="utf-8") as fh:
        index = json.load(fh)
    if index.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported fixture schema_version {index.get('schema_version')} "
            f"(expected {SCHEMA_VERSION})"
        )

    store = MessageStore()
    q = index.get("question")
    if q:
        store.add(
            CH_QUESTION,
            float(q["t_received"]),
            Question(text=q["text"], t_received=float(q["t_received"])),
        )

    for kf in index["keyframes"]:
        t = float(kf["t"])
        with np.load(os.path.join(fixture_dir, kf["file"])) as data:
            odom = None
            if "odom" in data:
                o = data["odom"]
                odom = OdomState(
                    t=float(o[0]), x=float(o[1]), y=float(o[2]),
                    z=float(o[3]), yaw=float(o[4]),
                )
                store.add(CH_ODOM, t, odom)
            if "pano" in data:
                placeholder = odom or OdomState(t=t, x=0.0, y=0.0, z=0.0, yaw=0.0)
                store.add(
                    CH_PANO, t,
                    PanoFrame(t=t, image=data["pano"].astype(np.uint8), odom=placeholder),
                )
            if "scan" in data:
                store.add(
                    CH_SCAN, t, LidarScan(t=t, points=data["scan"].astype(np.float32))
                )
            if "terrain" in data:
                store.add(
                    CH_TERRAIN, t,
                    TerrainPatch(t=t, points=data["terrain"].astype(np.float32), extended=False),
                )
            if "terrain_ext" in data:
                store.add(
                    CH_TERRAIN_EXT, t,
                    TerrainPatch(t=t, points=data["terrain_ext"].astype(np.float32), extended=True),
                )
    return store.finalize()


# --------------------------------------------------------------------------- CLI


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="core.replay.fixtures", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    ex = sub.add_parser("extract", help="distil a bag into keyframe fixtures")
    ex.add_argument("bag", help="path to the ROS 2 bag directory")
    ex.add_argument("out", help="output fixture directory")
    ex.add_argument("--stride", type=int, default=ExtractParams.stride)
    ex.add_argument("--move-m", type=float, default=ExtractParams.move_m)
    ex.add_argument("--rot-deg", type=float, default=30.0)
    ex.add_argument("--max-scan-pts", type=int, default=ExtractParams.max_scan_pts)
    ex.add_argument("--max-terrain-pts", type=int, default=ExtractParams.max_terrain_pts)

    info = sub.add_parser("info", help="print a fixture index summary")
    info.add_argument("dir", help="fixture directory")
    return p


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    if args.cmd == "extract":
        params = ExtractParams(
            stride=args.stride,
            move_m=args.move_m,
            rot_rad=math.radians(args.rot_deg),
            max_scan_pts=args.max_scan_pts,
            max_terrain_pts=args.max_terrain_pts,
        )
        index = extract_fixtures(args.bag, args.out, params)
        print(
            f"wrote {index['n_keyframes']} keyframes to {args.out} "
            f"(schema v{index['schema_version']})"
        )
        return 0
    if args.cmd == "info":
        with open(os.path.join(args.dir, INDEX_NAME), encoding="utf-8") as fh:
            index = json.load(fh)
        print(
            f"schema v{index['schema_version']} | source {index['source']} | "
            f"{index['n_keyframes']} keyframes | question={index['question']}"
        )
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
