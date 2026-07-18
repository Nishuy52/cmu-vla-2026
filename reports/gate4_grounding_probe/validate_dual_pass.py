"""Offline validation of the issue #42 dual-pass detector fix.

Replays data/sim_bags/japanese_room_q1 through the SAME code path the live adapter
uses end to end: core.heads.factory.HeadState.bind's refresh_prompt() call (which now
refreshes BOTH GroundingDinoDetector.prompt — the full question+vocab caption — and
GroundingDinoDetector.question_prompt — the short question-noun-only caption), then
PerceptionPipeline's tiling -> detector -> ... seam, calling det(tiles) exactly as
core.perception.tracker.PerceptionPipeline.process does. This exercises the NEW
dual-pass __call__ dispatch (question pass every tick at GDINO_QUESTION_BOX_THRESHOLD,
vocab pass every GDINO_VOCAB_PASS_CADENCE-th tick at the original 0.35 threshold) with
the real SwinB weights, not a synthetic short caption like probe_short.py used for the
diagnosis.

Usage (host venv, GPU):
    GDINO_CONFIG_PATH=... GDINO_CHECKPOINT_PATH=... HF_HOME=... \\
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \\
    PYTHONPATH=<repo>/src /home/jason/vla_host/venv/bin/python validate_dual_pass.py \\
        --bag <repo>/data/sim_bags/japanese_room_q1 --stride 10 \\
        --out detections_dual_pass.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

TARGET_NOUN = "teapot"
TEAPOT_GT = (-0.6369435471906074, 2.1241969218693484)  # (x, y); z omitted, bearing is 2D
GT_ANGLE_TOLERANCE_DEG = 15.0
MIN_HITS_REQUIRED = 10


def build_detector():
    """Build the detector exactly as core.heads.factory.HeadState.bind does: construct
    empty, then refresh_prompt() from the parsed question's nouns + the standing vocab
    — the shared seam that must now update BOTH captions (issue #42)."""
    from core.parsing.regex_tier import parse_regex
    from core.heads.explore_step import _plan_nouns
    from core.heads.factory import _STANDING_VOCAB_NOUNS
    from core.perception.detector import GroundingDinoDetector, refresh_prompt

    plan = parse_regex("Find the teapot on the table.")
    nouns = _plan_nouns(plan)
    det = GroundingDinoDetector()  # boot-time construction: no question latched yet
    new_prompt = refresh_prompt(det, nouns, _STANDING_VOCAB_NOUNS)
    assert new_prompt == det.prompt
    return det, nouns


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-frames", type=int, default=0, help="0 = no cap")
    args = ap.parse_args()

    from core.replay.bag_reader import BagSource
    from core.perception.tiling import (
        DEFAULT_N_TILES, DEFAULT_TILE_HFOV, DEFAULT_TILE_VFOV, project_tiles,
        tile_pixel_to_map_ray,
    )
    from core.interfaces import PanoFrame

    det, nouns = build_detector()
    print(f"[validate] plan nouns: {nouns}")
    print(f"[validate] full (vocab) prompt: {len(det.prompt)} chars, "
          f"{det.prompt.count(' . ') + 1} phrases")
    print(f"[validate] short (question) prompt: {det.question_prompt!r}")
    print(f"[validate] question_box_threshold={det.question_box_threshold} "
          f"vocab box_threshold={det.box_threshold} vocab_pass_cadence={det.vocab_pass_cadence}")

    out_path = Path(args.out)
    n_written = 0
    n_ticks = 0
    teapot_hits_within_tol = 0
    teapot_total = 0
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
            per_tile = det(tiles)  # the real dual-pass __call__ (issue #42)
            t_infer_total += time.time() - t_a
            n_ticks += 1

            gt_bearing = math.degrees(math.atan2(TEAPOT_GT[1] - oy, TEAPOT_GT[0] - ox))

            for tile_id, dets in enumerate(per_tile):
                for d in dets:
                    u, v = d.center_xy
                    bearing, elev = tile_pixel_to_map_ray(
                        tile_id, u, v, yaw,
                        n_tiles=DEFAULT_N_TILES, hfov=DEFAULT_TILE_HFOV, vfov=DEFAULT_TILE_VFOV,
                    )
                    bearing_deg = float(np.degrees(bearing))
                    ang_diff = abs((bearing_deg - gt_bearing + 180) % 360 - 180)
                    label = d.label.strip().lower()
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
                        "map_bearing_deg": bearing_deg,
                        "gt_bearing_deg": gt_bearing,
                        "angle_diff_deg": ang_diff,
                    }
                    fh.write(json.dumps(row) + "\n")
                    n_written += 1

                    if label == TARGET_NOUN:
                        teapot_total += 1
                        if (
                            ang_diff <= GT_ANGLE_TOLERANCE_DEG
                            and d.score >= det.question_box_threshold
                        ):
                            teapot_hits_within_tol += 1

            if kept_frames % 10 == 0:
                elapsed = time.time() - t0
                print(f"[validate] processed {kept_frames} frames (bag frame {frame_idx}), "
                      f"{n_written} dets so far, {teapot_hits_within_tol} teapot hits within "
                      f"{GT_ANGLE_TOLERANCE_DEG} deg, {elapsed:.1f}s elapsed, "
                      f"infer {t_infer_total:.1f}s")

    elapsed = time.time() - t0
    print(f"[validate] DONE: {kept_frames} frames ({n_ticks} detector ticks), "
          f"{n_written} detections written to {out_path}")
    print(f"[validate] total wall time {elapsed:.1f}s, pure inference time {t_infer_total:.1f}s")
    print(f"[validate] teapot-labeled detections (any score/bearing): {teapot_total}")
    print(f"[validate] teapot detections >= question_box_threshold "
          f"({det.question_box_threshold}) AND within {GT_ANGLE_TOLERANCE_DEG} deg of GT "
          f"bearing: {teapot_hits_within_tol}")
    verdict = "PASS" if teapot_hits_within_tol >= MIN_HITS_REQUIRED else "FAIL"
    print(f"[validate] VERDICT ({MIN_HITS_REQUIRED} required): {verdict}")


if __name__ == "__main__":
    main()
