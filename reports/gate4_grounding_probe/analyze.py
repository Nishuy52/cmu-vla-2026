"""Offline analysis of detections.jsonl for gate4 grounding probe (issue #42).

No re-inference: post-filters the raw low-threshold (box_threshold=0.05,
text_threshold=0.15) detections logged by probe.py.

Key subtlety: GroundingDinoDetector._decode_batch_item falls back to the FULL
prompt string as `label` when phrase-from-posmap decoding yields an empty phrase
(common for low-confidence/noisy boxes at this low text_threshold). Those
fallback rows are NOT "teapot" detections even though the string starts with
"teapot . table . ..." — they must be excluded from the teapot-labeled count and
handled as their own bucket.
"""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict

IN_PATH = "detections.jsonl"
TARGET_NOUN = "teapot"

MARKER = (2.26, -2.41, 1.48)
TEAPOT_GT = (-0.6369435471906074, 2.1241969218693484, 0.3412485122680664)
NEAREST_COLUMN_GT = (2.3634583379964687, -0.9925032593513767, 1.3290833234786987)  # id 49


def is_fallback(label: str) -> bool:
    # The fallback IS the full built prompt: many phrases joined by " . ", starting
    # with "teapot . table .". A real decoded phrase is short (<=~4 words typically).
    return label.count(" . ") > 3


def main() -> None:
    total = 0
    fallback = 0
    label_counter: Counter[str] = Counter()
    teapot_rows: list[dict] = []
    score_by_threshold = Counter()
    thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

    # bearing-from-marker bucket: for each detection, compute bearing from the
    # DETECTING FRAME'S odom position to the wrong-marker point, and to the GT
    # teapot point; if the detection's own map_bearing_deg is within 12deg of
    # that bearing, count its label (excluding fallback) as "near marker" / "near GT".
    near_marker_labels: Counter[str] = Counter()
    near_gt_labels: Counter[str] = Counter()
    near_marker_teapot_scores: list[float] = []

    frames_seen: set[int] = set()

    with open(IN_PATH) as fh:
        for line in fh:
            row = json.loads(line)
            total += 1
            frames_seen.add(row["frame_idx"])
            label = row["label"]
            fb = is_fallback(label)
            if fb:
                fallback += 1
            else:
                label_counter[label.strip().lower()] += 1

            if not fb and label.strip().lower() == TARGET_NOUN:
                teapot_rows.append(row)
                for th in thresholds:
                    if row["score"] >= th:
                        score_by_threshold[th] += 1

            # geolocation buckets (skip fallback rows; too generic to be meaningful)
            if not fb:
                ox, oy = row["odom_x"], row["odom_y"]
                det_bearing = row["map_bearing_deg"]

                def bearing_to(pt):
                    dx, dy = pt[0] - ox, pt[1] - oy
                    return math.degrees(math.atan2(dy, dx))

                b_marker = bearing_to(MARKER)
                b_gt = bearing_to(TEAPOT_GT)

                def ang_diff(a, b):
                    d = (a - b + 180) % 360 - 180
                    return abs(d)

                if ang_diff(det_bearing, b_marker) <= 12.0:
                    near_marker_labels[label.strip().lower()] += 1
                    if label.strip().lower() == TARGET_NOUN:
                        near_marker_teapot_scores.append(row["score"])
                if ang_diff(det_bearing, b_gt) <= 12.0:
                    near_gt_labels[label.strip().lower()] += 1

    print(f"=== summary ===")
    print(f"total detection rows: {total}")
    print(f"frames covered: {len(frames_seen)}")
    print(f"fallback (whole-prompt, undecoded-phrase) rows: {fallback} "
          f"({100*fallback/total:.1f}%)")
    print(f"non-fallback (decoded phrase) rows: {total - fallback}")
    print()

    print(f"=== top 40 decoded labels (non-fallback) ===")
    for lbl, c in label_counter.most_common(40):
        print(f"  {c:8d}  {lbl!r}")
    print()

    print(f"=== '{TARGET_NOUN}'-labeled detections (exact decoded phrase == '{TARGET_NOUN}') ===")
    print(f"count: {len(teapot_rows)}")
    if teapot_rows:
        scores = sorted(r["score"] for r in teapot_rows)
        print(f"score min={scores[0]:.3f} max={scores[-1]:.3f} "
              f"mean={sum(scores)/len(scores):.3f} median={scores[len(scores)//2]:.3f}")
        print("threshold sensitivity (count of teapot dets with score >= T):")
        for th in thresholds:
            print(f"  T={th:.2f}: {score_by_threshold[th]}")
        print()
        print("sample teapot rows (up to 20), frame/tile/odom/bearing/score:")
        for r in teapot_rows[:20]:
            print(f"  frame={r['frame_idx']:4d} tile={r['tile_id']} score={r['score']:.3f} "
                  f"odom=({r['odom_x']:.2f},{r['odom_y']:.2f}) yaw={r['odom_yaw']:.2f} "
                  f"bearing={r['map_bearing_deg']:.1f}deg elev={r['map_elev_deg']:.1f}deg")
    else:
        print("  (none at any score, box_threshold=0.05)")
    print()

    print("=== labels seen when the detecting tile's ray points toward the WRONG MARKER "
          "(2.26,-2.41,1.48), within 12deg bearing, from that frame's odom ===")
    for lbl, c in near_marker_labels.most_common(25):
        print(f"  {c:8d}  {lbl!r}")
    print(f"  ('teapot' scores in this bucket: "
          f"{sorted(near_marker_teapot_scores) if near_marker_teapot_scores else 'NONE'})")
    print()

    print("=== labels seen when the detecting tile's ray points toward the GT teapot "
          "(-0.64,2.12,0.34), within 12deg bearing, from that frame's odom ===")
    for lbl, c in near_gt_labels.most_common(25):
        print(f"  {c:8d}  {lbl!r}")


if __name__ == "__main__":
    main()
