"""Gate-4 grounding probe for GitHub issue #42.

Replays data/sim_bags/japanese_room_q1 panoramas through the SAME tiling the live
adapter uses (core.perception.tiling.project_tiles, n_tiles=4, hfov=90deg,
vfov=120deg) and runs the REAL GroundingDinoDetector on GPU with the SAME prompt
the live run would have had for "Find the teapot on the table." (reconstructed via
the actual prompt-building code path: parse_regex -> _plan_nouns -> build_gdino_prompt
with the standing vocab, exactly as core.heads.factory.HeadState.bind does).

Logs every raw detection (frame index, bag time, tile_id, label, score, bbox,
approximate map-frame bearing of the detection's bbox centre ray) to a JSONL file
in this directory. Runs with a LOW box_threshold/text_threshold so the log captures
everything a threshold-sensitivity analysis might want; the offline analysis step
re-applies higher thresholds by simple score filtering (no re-inference needed).

Usage (host venv, GPU):
    GDINO_CONFIG_PATH=... GDINO_CHECKPOINT_PATH=... HF_HOME=... \\
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \\
    PYTHONPATH=<repo>/src /home/jason/vla_host/venv/bin/python probe.py \\
        --bag <repo>/data/sim_bags/japanese_room_q1 --stride 5 \\
        --out detections.jsonl
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def build_prompt() -> tuple[str, list[str]]:
    from core.parsing.regex_tier import parse_regex
    from core.heads.explore_step import _plan_nouns
    from core.heads.factory import _STANDING_VOCAB_NOUNS
    from core.perception.detector import build_gdino_prompt

    plan = parse_regex("Find the teapot on the table.")
    nouns = _plan_nouns(plan)
    prompt = build_gdino_prompt(nouns, [])  # SHORT caption: question nouns only
    return prompt, nouns


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--out", required=True)
    ap.add_argument("--box-threshold", type=float, default=0.05)
    ap.add_argument("--text-threshold", type=float, default=0.15)
    ap.add_argument("--max-frames", type=int, default=0, help="0 = no cap")
    args = ap.parse_args()

    from core.replay.bag_reader import BagSource
    from core.perception.tiling import (
        DEFAULT_N_TILES, DEFAULT_TILE_HFOV, DEFAULT_TILE_VFOV, project_tiles,
        tile_pixel_to_map_ray,
    )
    from core.perception.detector import GroundingDinoDetector
    from core.interfaces import PanoFrame

    prompt, nouns = build_prompt()
    print(f"[probe] plan nouns: {nouns}")
    print(f"[probe] prompt length: {len(prompt)} chars, "
          f"{prompt.count(' . ') + 1} phrases")

    det = GroundingDinoDetector(
        box_threshold=args.box_threshold,
        text_threshold=args.text_threshold,
    )
    det.prompt = prompt  # same seam refresh_prompt() would use

    out_path = Path(args.out)
    n_written = 0
    t_infer_total = 0.0
    frame_idx = -1
    kept_frames = 0
    t0 = time.time()
    with out_path.open("w") as fh:
        for rec in BagSource(args.bag).frames():
            pano = rec.msg
            if not isinstance(pano, PanoFrame):
                continue
            frame_idx += 1
            if frame_idx % args.stride != 0:
                continue
            if args.max_frames and kept_frames >= args.max_frames:
                break
            kept_frames += 1

            odom = pano.odom
            yaw = float(getattr(odom, "yaw", 0.0)) if odom is not None else 0.0
            ox = float(getattr(odom, "x", 0.0)) if odom is not None else 0.0
            oy = float(getattr(odom, "y", 0.0)) if odom is not None else 0.0

            tiles = project_tiles(pano.image, DEFAULT_N_TILES, DEFAULT_TILE_HFOV, DEFAULT_TILE_VFOV)
            t_a = time.time()
            per_tile = det(tiles)
            t_infer_total += time.time() - t_a

            for tile_id, dets in enumerate(per_tile):
                for d in dets:
                    u, v = d.center_xy
                    bearing, elev = tile_pixel_to_map_ray(
                        tile_id, u, v, yaw,
                        n_tiles=DEFAULT_N_TILES, hfov=DEFAULT_TILE_HFOV, vfov=DEFAULT_TILE_VFOV,
                    )
                    row = {
                        "frame_idx": frame_idx,
                        "t": rec.t,
                        "odom_x": ox,
                        "odom_y": oy,
                        "odom_yaw": yaw,
                        "tile_id": tile_id,
                        "label": d.label,
                        "score": d.score,
                        "bbox_xyxy": list(d.bbox_xyxy),
                        "map_bearing_deg": float(np.degrees(bearing)),
                        "map_elev_deg": float(np.degrees(elev)),
                    }
                    fh.write(json.dumps(row) + "\n")
                    n_written += 1

            if kept_frames % 20 == 0:
                elapsed = time.time() - t0
                print(f"[probe] processed {kept_frames} frames (bag frame {frame_idx}), "
                      f"{n_written} dets so far, {elapsed:.1f}s elapsed, "
                      f"infer {t_infer_total:.1f}s")

    elapsed = time.time() - t0
    print(f"[probe] DONE: {kept_frames} frames, {n_written} detections written to {out_path}")
    print(f"[probe] total wall time {elapsed:.1f}s, pure inference time {t_infer_total:.1f}s")


if __name__ == "__main__":
    main()
