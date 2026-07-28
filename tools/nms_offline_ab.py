"""Offline A/B of cross-tile NMS (#131) against banked raw-proposal replays.

Replays the per-keyframe raw detections harvested during live cluster runs
(``reports/cluster_verify/*/debug/**/raw_detections.jsonl``, schema: one JSON object
per keyframe with a ``detections`` array of
``{tile_id, label, score, bbox_xyxy, gate, instance_id}``) through
:func:`core.perception.detector.suppress_cross_tile_duplicates` and reports, before vs
after: total proposal count, and per-class max-per-keyframe proposal counts for a
configurable set of "pathological" (should collapse) and "legit multi-instance"
(should NOT collapse) classes.

Usage (from the worktree root, with ``.venv`` symlinked in)::

    .venv/bin/python -m tools.nms_offline_ab --split tuning
    .venv/bin/python -m tools.nms_offline_ab --split holdout
    .venv/bin/python -m tools.nms_offline_ab --split all --iou 0.75

Scene-level holdout split (docs/calibration.md "Generalization protocol" rule 1):
half the 15 training scenes are picked ONCE below as the tuning set (used to choose
``CROSS_TILE_NMS_IOU_THRESHOLD``); the other half is never used for tuning and is
reported separately.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

_REPO_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

from core.perception.detector import (  # noqa: E402
    CROSS_TILE_NMS_IOU_THRESHOLD,
    Detection,
    suppress_cross_tile_duplicates,
)

REPORTS_ROOT = Path(__file__).resolve().parent.parent / "reports" / "cluster_verify"

#: Scene-level holdout split, fixed once (alphabetical bisection of the 15 training
#: scenes) — never adjusted after looking at the A/B numbers.
TUNING_SCENES: tuple[str, ...] = (
    "arabic_room",
    "chinese_room",
    "home_building_1",
    "home_building_2",
    "hotel_room_1",
    "hotel_room_2",
    "japanese_room",
    "livingroom_1",
)
HOLDOUT_SCENES: tuple[str, ...] = (
    "livingroom_2",
    "livingroom_3",
    "livingroom_4",
    "loft",
    "office_1",
    "office_2",
    "studio",
)

#: Pathological (over-proposing) class per scene, from the #131 issue body.
PATHOLOGICAL: tuple[tuple[str, str], ...] = (
    ("office_1", "window"),
    ("hotel_room_2", "window"),
    ("livingroom_1", "table"),
    ("livingroom_2", "shelf"),
    ("home_building_2", "lamp"),
)

#: Legitimately multi-instance classes that must NOT collapse to a single survivor.
LEGIT_MULTI: tuple[tuple[str, str], ...] = (
    ("chinese_room", "chair"),
    ("hotel_room_1", "pillow"),
)


def _scene_name(slot_dir: Path) -> str:
    """``<idx>_<scene>_<suffix>`` -> ``<scene>`` (suffix is inst/nume/obje/...)."""
    name = slot_dir.name
    parts = name.split("_")
    if parts and parts[0].isdigit():
        parts = parts[1:]
    if parts:
        parts = parts[:-1]  # drop the trailing question-type suffix
    return "_".join(parts)


def _iter_slots(scenes: set[str] | None):
    for slot_dir in sorted(REPORTS_ROOT.glob("*/debug/*")):
        f = slot_dir / "raw_detections.jsonl"
        if not f.is_file():
            continue
        scene = _scene_name(slot_dir)
        if scenes is not None and scene not in scenes:
            continue
        yield scene, f
    # a few job dirs bank raw_detections.jsonl directly under debug/ (single-scene run,
    # no per-scene subdir) — recover the scene name from the sibling captures/ dir.
    for f in sorted(REPORTS_ROOT.glob("*/debug/raw_detections.jsonl")):
        job_dir = f.parent.parent
        captures = job_dir / "captures"
        scene = None
        if captures.is_dir():
            scene_dirs = [d.name for d in captures.iterdir() if d.is_dir()]
            if len(scene_dirs) == 1:
                scene = scene_dirs[0]
        if scene is None:
            continue
        if scenes is not None and scene not in scenes:
            continue
        yield scene, f


def _detections_by_tile(raw_dets: list[dict]) -> list[list[Detection]]:
    n_tiles = 4
    by_tile: list[list[Detection]] = [[] for _ in range(n_tiles)]
    for d in raw_dets:
        tid = int(d["tile_id"])
        if tid >= len(by_tile):
            by_tile.extend([] for _ in range(tid - len(by_tile) + 1))
        by_tile[tid].append(
            Detection(
                tile_id=tid,
                bbox_xyxy=tuple(d["bbox_xyxy"]),
                label=str(d["label"]),
                score=float(d["score"]),
            )
        )
    return by_tile


def run(scenes: set[str] | None, iou_threshold: float) -> dict:
    total_before = 0
    total_after = 0
    max_per_kf: dict[tuple[str, str], tuple[int, int]] = defaultdict(lambda: (0, 0))
    n_keyframes = 0
    n_slots = 0
    seen_slots: set[Path] = set()

    for scene, f in _iter_slots(scenes):
        if f in seen_slots:
            continue
        seen_slots.add(f)
        n_slots += 1
        with f.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                dets = rec.get("detections", [])
                if not dets:
                    continue
                n_keyframes += 1
                by_tile = _detections_by_tile(dets)
                after = suppress_cross_tile_duplicates(by_tile, iou_threshold=iou_threshold)
                before_count = sum(len(t) for t in by_tile)
                after_count = sum(len(t) for t in after)
                total_before += before_count
                total_after += after_count

                before_by_label = Counter(d.label for tile in by_tile for d in tile)
                after_by_label = Counter(d.label for tile in after for d in tile)
                for label, cnt in before_by_label.items():
                    key = (scene, label)
                    b, a = max_per_kf[key]
                    max_per_kf[key] = (max(b, cnt), max(a, after_by_label.get(label, 0)))

    return {
        "n_slots": n_slots,
        "n_keyframes": n_keyframes,
        "total_before": total_before,
        "total_after": total_after,
        "max_per_kf": dict(max_per_kf),
    }


def _print_report(label: str, result: dict) -> None:
    print(f"=== {label} ===")
    print(f"slots: {result['n_slots']}  keyframes-with-detections: {result['n_keyframes']}")
    print(f"total proposals: {result['total_before']} -> {result['total_after']}"
          f"  ({result['total_before'] - result['total_after']} removed,"
          f" {100.0 * (1 - result['total_after'] / max(result['total_before'], 1)):.1f}%)")
    print("-- pathological (should collapse) --")
    for scene, cls in PATHOLOGICAL:
        b, a = result["max_per_kf"].get((scene, cls), (None, None))
        print(f"  {scene:20s} {cls:10s} max/keyframe: {b} -> {a}")
    print("-- legit multi-instance (must NOT collapse to 1) --")
    for scene, cls in LEGIT_MULTI:
        b, a = result["max_per_kf"].get((scene, cls), (None, None))
        print(f"  {scene:20s} {cls:10s} max/keyframe: {b} -> {a}")
    print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["tuning", "holdout", "all"], default="all")
    ap.add_argument("--iou", type=float, default=CROSS_TILE_NMS_IOU_THRESHOLD)
    args = ap.parse_args()

    if args.split == "tuning":
        scenes = set(TUNING_SCENES)
    elif args.split == "holdout":
        scenes = set(HOLDOUT_SCENES)
    else:
        scenes = None

    result = run(scenes, args.iou)
    _print_report(f"split={args.split} iou={args.iou}", result)


if __name__ == "__main__":
    main()
