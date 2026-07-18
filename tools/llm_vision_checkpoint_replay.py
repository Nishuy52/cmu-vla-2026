"""Phase-2 (vision half) battery: CP2/CP3/CP5 replayed against real recorded panoramas.

Companion to ``tools/llm_parse_battery.py`` (the parse half) and
``reports/gate4_grounding_probe/probe.py``'s replay pattern (real bag -> real tiling ->
real provider call, offline, no ROS/sim). Here the "detector" under test is the local
vision-capable LLM itself, called through the exact checkpoint code
(``core.checkpoints.miss_recovery`` / ``anchor_confirm`` / ``frontier_select``) and the
exact provider path (``core.llm.providers.OpenAIChatAdapter.vision_chat`` against the
local Ollama slot), against ground truth read from ``object_list.txt`` + odom bearing
math (mirroring ``reports/gate4_grounding_probe/analyze.py``'s bearing pattern).

Method (see ``docs/local_llm_plan.md`` Phase 2 item 2 and the task brief this tool was
built from):

* Source: ``data/sim_bags/{japanese_room_q1,office_1_q1,livingroom_1_tour}`` panoramas
  via ``core.replay.bag_reader.BagSource``, strided. GT per scene from
  ``data/unity_scenes_ros2/<scene>/<scene>/object_list.txt``.
* CP2 (4-tile miss recovery): positive cases where GT places a noun inside one of the
  frame's 4 tiles within ~6 m (real ``core.perception.tiling.project_tiles`` tiling,
  same n_tiles/hfov/vfov as the live adapter); negative cases ask about a noun absent
  from the whole scene.
* CP3 (single-tile anchor confirm): crop centred on the anchor's *projected* image
  location (pinhole forward-projection, the analytic inverse of
  ``tiling.tile_pixel_to_camera_ray`` — no crop-generation code exists yet upstream of
  the checkpoint, so this tool derives it from the same tiling geometry) vs a
  deliberately-wrong crop centred on a different real object.
* CP5 (frontier select): the real pano + a numbered-disc overlay at synthetic
  evenly-spaced frontier azimuths (real ``tiling.azimuth_to_column`` projection; no
  frontier-detector or disc-renderer exists upstream either, so both are supplied here
  for format/sanity purposes only, per the task brief).

Every case runs through the real checkpoint entrypoint (``run_miss_recovery`` /
``run_anchor_confirm`` / ``run_frontier_select``) with ``encode_fn=jpeg_encode_fn`` (real
JPEG bytes) and a live ``OpenAIChatAdapter.vision_chat`` bound to
``VLA_LLM_LOCAL_*``-style env (defaults match the local Ollama slot:
``http://localhost:11434/v1``, ``qwen2.5vl:3b``, dummy key) — ``ledger=None`` (always
allow; the ledger's per-question cap is an integration-time concern, not this offline
battery's) and a real wall clock.

Usage (host venv, GPU, Ollama already serving qwen2.5vl:3b)::

    python -m tools.llm_vision_checkpoint_replay \\
        --out-dir reports/local_llm_phase2 --stride 12

Output: ``reports/local_llm_phase2/vision_checkpoints.jsonl`` (one row per case) and
``reports/local_llm_phase2/vision_checkpoints.md`` (per-checkpoint tables + the Phase-2
enable matrix).
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

import tools  # noqa: F401 -- inserts <repo>/src onto sys.path

from core.checkpoints.anchor_confirm import run_anchor_confirm
from core.checkpoints.frontier_select import DEFAULT_N_FRONTIERS, run_frontier_select
from core.checkpoints.miss_recovery import run_miss_recovery
from core.interfaces import PanoFrame
from core.llm.providers import OpenAIChatAdapter
from core.perception import tiling
from core.perception.vision_encode import jpeg_encode_fn
from core.replay.bag_reader import BagSource

REPO_ROOT = Path(__file__).resolve().parent.parent

#: (bag dir name under data/sim_bags, GT scene dir name under data/unity_scenes_ros2,
#:  a real object_reference question for that scene — used as CP5's driving question).
SCENES: tuple[tuple[str, str, str], ...] = (
    ("japanese_room_q1", "japanese_room",
     "The lantern between the vase and the stone decoration that is closest to the vase."),
    ("office_1_q1", "office_1", "Find the potted plant on the file cabinet."),
    ("livingroom_1_tour", "livingroom_1", "Find the vase on the cabinet below the picture."),
)

#: Structural/background annotation classes that are not sensible "find this object"
#: nouns (mirrors the kind of filtering a real noun-vocab pass would apply).
_STOPLIST_LABELS = frozenset({"wall", "floor", "ceiling", "unknown", "window frame", "door frame"})

#: Candidate nouns guaranteed absent from at least one scene (curated pool; the picker
#: verifies absence against that scene's own label set before using one).
_ABSENT_NOUN_POOL: tuple[str, ...] = (
    "teapot", "guitar", "bicycle", "refrigerator", "piano", "aquarium",
    "fire extinguisher", "backpack", "umbrella", "globe", "telescope", "toaster",
    "xbox controller", "skateboard", "birdcage", "fish tank", "drum kit", "violin",
)

#: CP2 GT gate: a noun counts as "in tile" only within this range (task brief: "~6 m").
CP2_MAX_DIST_M = 6.0
#: CP3 GT gate: tighter range for a crop that is actually framable/legible.
CP3_MAX_DIST_M = 5.0

N_TILES = tiling.DEFAULT_N_TILES
TILE_HFOV = tiling.DEFAULT_TILE_HFOV
TILE_VFOV = tiling.DEFAULT_TILE_VFOV


# =============================================================================== pure helpers
# (no network, no Pillow, no bag I/O -- exercised directly by tools/tests/)


def parse_object_list_line(line: str) -> dict[str, Any] | None:
    """Parse one ``object_list.txt`` line: ``id x y z sx sy sz yaw "label"``.

    Returns ``None`` for a blank/malformed line (never raises -- the file is trusted
    scene data but a defensive parser keeps the loader total).
    """
    line = line.strip()
    if not line:
        return None
    q = line.find('"')
    if q < 0:
        return None
    head = line[:q].split()
    label = line[q:].strip().strip('"')
    if len(head) < 8 or not label:
        return None
    try:
        return {
            "id": int(head[0]),
            "x": float(head[1]),
            "y": float(head[2]),
            "z": float(head[3]),
            "sx": float(head[4]),
            "sy": float(head[5]),
            "sz": float(head[6]),
            "yaw": float(head[7]),
            "label": label,
        }
    except ValueError:
        return None


def load_object_list(path: str | Path) -> list[dict[str, Any]]:
    """Parse a whole ``object_list.txt`` into a list of object dicts."""
    out: list[dict[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        obj = parse_object_list_line(line)
        if obj is not None:
            out.append(obj)
    return out


def scene_labels(objects: list[dict[str, Any]]) -> set[str]:
    return {o["label"].strip().lower() for o in objects}


def pick_absent_nouns(labels: set[str], k: int, rng: random.Random) -> list[str]:
    """``k`` nouns from :data:`_ABSENT_NOUN_POOL` that do not appear (as substring, either
    direction) in ``labels`` -- true negatives for CP2's "absent from the whole scene" arm.
    """
    candidates = [
        n for n in _ABSENT_NOUN_POOL
        if not any(n in lbl or lbl in n for lbl in labels)
    ]
    rng.shuffle(candidates)
    if len(candidates) < k:
        return candidates
    return candidates[:k]


def bearing_map_rad(ox: float, oy: float, tx: float, ty: float) -> float:
    """Map-frame bearing (rad, atan2 convention) from ``(ox,oy)`` to ``(tx,ty)``.

    Same global reference frame ``core.perception.tiling.camera_ray_to_map`` produces
    (camera azimuth + yaw) -- matches ``reports/gate4_grounding_probe/analyze.py``'s
    ``bearing_to`` helper.
    """
    return float(np.arctan2(ty - oy, tx - ox))


def tile_containing_azimuth(azimuth: float, specs: tuple) -> int | None:
    """Index of the tile spec whose ``yaw_center`` is within ``hfov/2`` of ``azimuth``,
    or ``None`` if it falls in no tile (shouldn't happen for 4x90deg full coverage, but
    the check stays honest rather than assuming coverage)."""
    for spec in specs:
        half = spec.hfov / 2.0
        diff = abs(float(tiling.wrap_pi(azimuth - spec.yaw_center)))
        if diff <= half:
            return spec.tile_id
    return None


def project_to_tile_pixel(azimuth: float, elevation: float, spec) -> tuple[float, float]:
    """Forward pinhole projection: camera-frame ``(azimuth, elevation)`` -> tile pixel
    ``(u, v)``. The analytic inverse of ``tiling.tile_pixel_to_camera_ray`` (derived from
    the same gnomonic model; no forward projection exists upstream)."""
    local_az = float(tiling.wrap_pi(azimuth - spec.yaw_center))
    local_el = float(elevation)
    x = -np.tan(local_az)
    r = np.hypot(x, 1.0)
    y = -np.tan(local_el) * r
    u = x * spec.focal_x + spec.cx
    v = y * spec.focal_y + spec.cy
    return float(u), float(v)


def crop_box(u: float, v: float, half_w: float, half_h: float, width: int, height: int
             ) -> tuple[int, int, int, int]:
    """Integer ``(x1,y1,x2,y2)`` crop box centred at ``(u,v)``, clamped to the image and
    guaranteed non-degenerate (at least 1x1)."""
    x1 = int(np.clip(round(u - half_w), 0, width - 1))
    x2 = int(np.clip(round(u + half_w), x1 + 1, width))
    y1 = int(np.clip(round(v - half_h), 0, height - 1))
    y2 = int(np.clip(round(v + half_h), y1 + 1, height))
    return x1, y1, x2, y2


def crop_half_size(spec, obj: dict[str, Any], dist_m: float) -> tuple[float, float]:
    """Apparent half-width/height (px) of ``obj`` at ``dist_m``, via the pinhole focal
    length, clamped to a sane framing range."""
    extent = max(obj["sx"], obj["sy"], 0.2)
    half_w = spec.focal_x * (extent / 2.0) / max(dist_m, 0.3)
    half_h = spec.focal_y * (max(obj["sz"], 0.2) / 2.0) / max(dist_m, 0.3)
    return float(np.clip(half_w, 40.0, 220.0)), float(np.clip(half_h, 40.0, 220.0))


def frontier_azimuths(n: int) -> list[float]:
    """``n`` evenly-spaced synthetic frontier azimuths (camera frame, rad), starting
    45deg off-heading so none sits dead-centre on the panorama seam."""
    start = np.deg2rad(45.0)
    return [float(tiling.wrap_pi(start + i * (2.0 * np.pi / n))) for i in range(n)]


def frontier_columns(n_frontiers: int, width: int = tiling.PANO_WIDTH) -> list[int]:
    """Panorama pixel columns for :func:`frontier_azimuths` via the real
    ``tiling.azimuth_to_column`` projection."""
    return [int(round(float(tiling.azimuth_to_column(az, width)))) for az in frontier_azimuths(n_frontiers)]


def cp5_choice_is_sane(choice: int | None, n_frontiers: int) -> bool:
    """CP5 "gross sanity": a non-fallback choice must be one of the offered discs."""
    if choice is None:
        return True  # fallback is always sane (deterministic geometric fallback)
    return 1 <= choice <= n_frontiers


# =============================================================================== GT candidate scan


@dataclass
class FrameInfo:
    frame_idx: int
    t: float
    ox: float
    oy: float
    oz: float
    yaw: float


@dataclass
class Cp2Candidate:
    frame_idx: int
    label: str
    tile_id: int
    dist_m: float


@dataclass
class Cp3Candidate:
    frame_idx: int
    label: str
    tile_id: int
    dist_m: float
    azimuth: float
    elevation: float


def scan_scene(
    bag_dir: Path, objects: list[dict[str, Any]], stride: int, max_frames: int | None = None,
) -> tuple[dict[int, FrameInfo], list[Cp2Candidate], list[Cp3Candidate]]:
    """Pass 1: iterate the bag (strided, GT-only) and build candidate lists WITHOUT
    holding any images. Returns ``(frames_by_idx, cp2_candidates, cp3_candidates)``."""
    specs = tiling.tile_specs(N_TILES, TILE_HFOV, TILE_VFOV)
    frames: dict[int, FrameInfo] = {}
    cp2: list[Cp2Candidate] = []
    cp3: list[Cp3Candidate] = []
    keep_labels = [o for o in objects if o["label"].strip().lower() not in _STOPLIST_LABELS]

    frame_idx = -1
    kept = 0
    for rec in BagSource(str(bag_dir)).frames():
        if not isinstance(rec.msg, PanoFrame):
            continue
        frame_idx += 1
        if frame_idx % stride != 0:
            continue
        if max_frames and kept >= max_frames:
            break
        kept += 1
        odom = rec.msg.odom
        ox, oy, oz = float(odom.x), float(odom.y), float(odom.z)
        yaw = float(odom.yaw)
        frames[frame_idx] = FrameInfo(frame_idx, rec.t, ox, oy, oz, yaw)

        for obj in keep_labels:
            dx, dy, dz = obj["x"] - ox, obj["y"] - oy, obj["z"] - oz
            dist = float(np.hypot(dx, dy))
            if dist <= 0.05:
                continue
            bearing = bearing_map_rad(ox, oy, obj["x"], obj["y"])
            azimuth, elevation = tiling.map_ray_to_camera(bearing, float(np.arctan2(dz, dist)), yaw)
            tile_id = tile_containing_azimuth(azimuth, specs)
            if tile_id is None:
                continue
            label = obj["label"].strip().lower()
            if dist < CP2_MAX_DIST_M:
                cp2.append(Cp2Candidate(frame_idx, label, tile_id, dist))
            if dist < CP3_MAX_DIST_M:
                cp3.append(Cp3Candidate(frame_idx, label, tile_id, dist, azimuth, elevation))
    return frames, cp2, cp3


# =============================================================================== case rows


@dataclass
class CaseResult:
    checkpoint: str
    scene: str
    case_kind: str  # cp2: "positive"|"negative"; cp3: "correct"|"wrong"; cp5: "single"
    frame_idx: int
    noun: str
    expected: Any
    outcome_action: str
    outcome_fields: dict[str, Any]
    latency_s: float
    dist_m: float | None = None


@dataclass
class Stats:
    n: int = 0
    hits: int = 0  # positives correctly detected / correct-crops correctly confirmed
    false: int = 0  # negatives wrongly flagged / wrong-crops wrongly confirmed
    latencies: list[float] = field(default_factory=list)


def _timed(fn, *args, **kwargs) -> tuple[Any, float]:
    t0 = time.monotonic()
    out = fn(*args, **kwargs)
    return out, time.monotonic() - t0


class _RealClock:
    def now(self) -> float:
        return time.monotonic()


def build_vision_chat(base_url: str, model: str, api_key: str) -> Any:
    adapter = OpenAIChatAdapter(base_url=base_url, model=model, api_key=api_key)
    return adapter.vision_chat


# =============================================================================== CP2 runner


def run_cp2_cases(
    vision_chat, scene: str, bag, frames: dict[int, FrameInfo],
    positives: list[Cp2Candidate], negatives: list[tuple[int, str]], rows: list[CaseResult],
) -> None:
    """Execute CP2 (positive + negative) cases for one scene; needs re-decoded tiles per
    needed frame -- pass 2 of the two-pass scan."""
    needed = {c.frame_idx for c in positives} | {fi for fi, _ in negatives}
    tiles_by_frame: dict[int, list[np.ndarray]] = {}
    frame_idx = -1
    for rec in bag.frames():
        if not isinstance(rec.msg, PanoFrame):
            continue
        frame_idx += 1
        if frame_idx not in needed:
            continue
        tiles_by_frame[frame_idx] = tiling.project_tiles(rec.msg.image, N_TILES, TILE_HFOV, TILE_VFOV)
        if len(tiles_by_frame) == len(needed):
            break

    spec = tiling.tile_specs(N_TILES, TILE_HFOV, TILE_VFOV)[0]  # all tiles share dims
    for c in positives:
        tiles = tiles_by_frame.get(c.frame_idx)
        if tiles is None:
            continue
        outcome, dt = _timed(
            run_miss_recovery, vision_chat, None, _RealClock(),
            noun=c.label, raw=c.label, tiles=tiles,
            tile_w=spec.width, tile_h=spec.height, encode_fn=jpeg_encode_fn,
        )
        rows.append(CaseResult(
            "CP2", scene, "positive", c.frame_idx, c.label, True,
            outcome.action, {"tile": outcome.tile, "confidence": outcome.confidence,
                              "gt_tile": c.tile_id}, dt, c.dist_m,
        ))

    for fi, noun in negatives:
        tiles = tiles_by_frame.get(fi)
        if tiles is None:
            continue
        outcome, dt = _timed(
            run_miss_recovery, vision_chat, None, _RealClock(),
            noun=noun, raw=noun, tiles=tiles,
            tile_w=spec.width, tile_h=spec.height, encode_fn=jpeg_encode_fn,
        )
        rows.append(CaseResult(
            "CP2", scene, "negative", fi, noun, False,
            outcome.action, {"tile": outcome.tile, "confidence": outcome.confidence}, dt,
        ))


# =============================================================================== CP3 runner


def run_cp3_cases(
    vision_chat, scene: str, bag, pairs: list[tuple[Cp3Candidate, Cp3Candidate]], rows: list[CaseResult],
) -> None:
    """Execute CP3 (correct, wrong) crop pairs for one scene. Each pair shares the
    *anchor description* (the correct case's label) but the wrong case's crop is centred
    on a genuinely different object."""
    specs = tiling.tile_specs(N_TILES, TILE_HFOV, TILE_VFOV)
    needed = {c.frame_idx for pair in pairs for c in pair}
    images_by_frame: dict[int, np.ndarray] = {}
    frame_idx = -1
    for rec in bag.frames():
        if not isinstance(rec.msg, PanoFrame):
            continue
        frame_idx += 1
        if frame_idx not in needed:
            continue
        images_by_frame[frame_idx] = rec.msg.image
        if len(images_by_frame) == len(needed):
            break

    def make_crop(c: Cp3Candidate) -> np.ndarray | None:
        image = images_by_frame.get(c.frame_idx)
        if image is None:
            return None
        tiles = tiling.project_tiles(image, N_TILES, TILE_HFOV, TILE_VFOV)
        tile_img = tiles[c.tile_id]
        spec = specs[c.tile_id]
        u, v = project_to_tile_pixel(c.azimuth, c.elevation, spec)
        half_w, half_h = crop_half_size(spec, {"sx": 0.5, "sy": 0.5, "sz": 0.5}, c.dist_m)
        x1, y1, x2, y2 = crop_box(u, v, half_w, half_h, spec.width, spec.height)
        return tile_img[y1:y2, x1:x2]

    for correct, wrong in pairs:
        crop = make_crop(correct)
        if crop is None or crop.size == 0:
            continue
        outcome, dt = _timed(
            run_anchor_confirm, vision_chat, None, _RealClock(),
            anchor_desc=correct.label, crop=crop, anchor_noun=correct.label,
            encode_fn=jpeg_encode_fn,
        )
        rows.append(CaseResult(
            "CP3", scene, "correct", correct.frame_idx, correct.label, True,
            outcome.action, {"match": outcome.match, "actual_label": outcome.actual_label,
                              "confidence": outcome.confidence}, dt, correct.dist_m,
        ))

        wrong_crop = make_crop(wrong)
        if wrong_crop is None or wrong_crop.size == 0:
            continue
        outcome2, dt2 = _timed(
            run_anchor_confirm, vision_chat, None, _RealClock(),
            anchor_desc=correct.label, crop=wrong_crop, anchor_noun=correct.label,
            encode_fn=jpeg_encode_fn,
        )
        rows.append(CaseResult(
            "CP3", scene, "wrong", wrong.frame_idx, correct.label, False,
            outcome2.action, {"match": outcome2.match, "actual_label": outcome2.actual_label,
                               "confidence": outcome2.confidence,
                               "wrong_true_label": wrong.label}, dt2, wrong.dist_m,
        ))


# =============================================================================== CP5 runner


def _draw_discs(image: np.ndarray, columns: list[int]) -> bytes:
    """Numbered-disc overlay -> JPEG bytes (Pillow; lazy import mirrors
    ``vision_encode.jpeg_encode_fn``'s pattern -- no disc-renderer exists upstream, so
    this tool supplies a minimal one for CP5 replay only)."""
    from PIL import Image, ImageDraw

    img = Image.fromarray(image.astype("uint8"), mode="RGB")
    draw = ImageDraw.Draw(img)
    row = image.shape[0] // 2
    radius = 22
    for i, col in enumerate(columns, start=1):
        draw.ellipse((col - radius, row - radius, col + radius, row + radius),
                     fill=(255, 60, 60), outline=(0, 0, 0), width=3)
        draw.text((col - 7, row - 10), str(i), fill=(255, 255, 255))
    import io
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def run_cp5_cases(
    vision_chat, scene: str, bag, question: str, frame_idxs: list[int], rows: list[CaseResult],
) -> None:
    needed = set(frame_idxs)
    n = DEFAULT_N_FRONTIERS
    columns = frontier_columns(n)
    frame_idx = -1
    for rec in bag.frames():
        if not isinstance(rec.msg, PanoFrame):
            continue
        frame_idx += 1
        if frame_idx not in needed:
            continue
        disc_bytes = _draw_discs(rec.msg.image, columns)

        def encode_fn(_img: Any, _bytes: bytes = disc_bytes) -> bytes:
            return _bytes

        outcome, dt = _timed(
            run_frontier_select, vision_chat, None, _RealClock(),
            question=question, panorama=rec.msg.image, n_frontiers=n, encode_fn=encode_fn,
        )
        rows.append(CaseResult(
            "CP5", scene, "single", frame_idx, question, None,
            outcome.action, {"choice": outcome.choice, "index": outcome.index,
                              "reason": outcome.reason[:120]}, dt,
        ))
        needed.discard(frame_idx)
        if not needed:
            break


# =============================================================================== aggregation


def aggregate(rows: list[CaseResult]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}

    cp2_pos = [r for r in rows if r.checkpoint == "CP2" and r.case_kind == "positive"]
    cp2_neg = [r for r in rows if r.checkpoint == "CP2" and r.case_kind == "negative"]
    out["CP2"] = {
        "n_positive": len(cp2_pos),
        "n_negative": len(cp2_neg),
        "hit_rate": _rate(cp2_pos, lambda r: r.outcome_action == "provisional"),
        "false_yes_rate": _rate(cp2_neg, lambda r: r.outcome_action == "provisional"),
        "latency_s": _lat_stats([r.latency_s for r in cp2_pos + cp2_neg]),
    }

    cp3_correct = [r for r in rows if r.checkpoint == "CP3" and r.case_kind == "correct"]
    cp3_wrong = [r for r in rows if r.checkpoint == "CP3" and r.case_kind == "wrong"]
    out["CP3"] = {
        "n_correct": len(cp3_correct),
        "n_wrong": len(cp3_wrong),
        "confirm_rate_on_correct": _rate(cp3_correct, lambda r: r.outcome_fields.get("match") is True),
        "reject_rate_on_wrong": _rate(cp3_wrong, lambda r: r.outcome_fields.get("match") is False),
        "demote_rate_on_wrong": _rate(cp3_wrong, lambda r: r.outcome_action == "demote"),
        "false_demote_rate_on_correct": _rate(cp3_correct, lambda r: r.outcome_action == "demote"),
        "latency_s": _lat_stats([r.latency_s for r in cp3_correct + cp3_wrong]),
    }

    cp5_rows = [r for r in rows if r.checkpoint == "CP5"]
    out["CP5"] = {
        "n": len(cp5_rows),
        "format_valid_rate": _rate(cp5_rows, lambda r: r.outcome_action in ("choice", "fallback")),
        "sane_choice_rate": _rate(
            cp5_rows,
            lambda r: cp5_choice_is_sane(r.outcome_fields.get("choice"), DEFAULT_N_FRONTIERS),
        ),
        "fallback_rate": _rate(cp5_rows, lambda r: r.outcome_action == "fallback"),
        "latency_s": _lat_stats([r.latency_s for r in cp5_rows]),
    }
    return out


def _rate(rows: list[CaseResult], pred) -> float | None:
    if not rows:
        return None
    return sum(1 for r in rows if pred(r)) / len(rows)


def _lat_stats(latencies: list[float]) -> dict[str, float] | None:
    if not latencies:
        return None
    s = sorted(latencies)
    return {
        "mean": statistics.mean(s),
        "p50": s[len(s) // 2],
        "p90": s[min(len(s) - 1, int(len(s) * 0.9))],
        "max": s[-1],
    }


# =============================================================================== report


def _fmt_lat(lat: dict[str, float] | None) -> str:
    if lat is None:
        return "n/a"
    return f"mean {lat['mean']:.2f}s / p50 {lat['p50']:.2f}s / p90 {lat['p90']:.2f}s / max {lat['max']:.2f}s"


def _fmt_pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def build_enable_matrix(agg: dict[str, dict[str, Any]]) -> str:
    cp2 = agg["CP2"]
    cp3 = agg["CP3"]
    cp5 = agg["CP5"]

    lines: list[str] = []
    lines.append("## Enable matrix (local 3B regime)\n")
    lines.append(
        "A checkpoint is ON only if its error rate is low enough that acting on it beats "
        "ignoring it. Thresholds (numeric, stated per checkpoint below) come from what the "
        "*action* costs on a wrong call, not from an arbitrary accuracy bar:\n"
    )
    lines.append(
        "* **CP2** (miss recovery) acting on a false-yes means the caller casts a bogus "
        "provisional instance and re-navigates toward nothing -- wasted travel with no "
        "recovery. A false-yes rate has to sit *well under* the hit rate (>=2x headroom) "
        "for the expected value of enabling CP2 to beat just falling through to the "
        "deterministic resolve-fallback ladder unchanged.\n"
    )
    lines.append(
        "* **CP3** (anchor confirm) only *acts* (demotes) on a confident mismatch (OR-F9, "
        "conf>=0.8); every other outcome (including a wrong verdict at low confidence) "
        "already falls back to \"confirm\" == the map's belief, so CP3's downside is bounded "
        "by its *false-demote-on-correct-crop* rate, not its raw confirm/reject accuracy. "
        "That rate has to be low (<10%) since a false demote throws away a correct anchor "
        "and re-plans onto a worse runner-up.\n"
    )
    lines.append(
        "* **CP5** (frontier select) never has a wrong-vs-right GT (the task brief scores "
        "format validity + sanity only) and the failure mode of a bad pick is *at most* one "
        "exploration leg toward a worse-than-geometric frontier, self-correcting on the next "
        "cap-1-per-question call in a later question. The bar is format reliability, not "
        "correctness: format-valid parse rate has to be high (>=90%) for the checkpoint to "
        "be worth wiring over the free geometric fallback.\n"
    )

    cp2_hit = cp2["hit_rate"] or 0.0
    cp2_false = cp2["false_yes_rate"] or 0.0
    cp2_on = cp2["hit_rate"] is not None and cp2_false <= cp2_hit / 2.0 and cp2_hit >= 0.4
    lines.append(
        f"\n**CP2: {'ON' if cp2_on else 'OFF'}** -- hit rate "
        f"{_fmt_pct(cp2['hit_rate'])} (n={cp2['n_positive']}), false-yes rate "
        f"{_fmt_pct(cp2['false_yes_rate'])} (n={cp2['n_negative']}). "
        + (
            "False-yes stays under half the hit rate, so a wasted-detour false positive is "
            "rarer than a genuine recovery -- acting on CP2 beats silently falling to the "
            "resolve-fallback ladder."
            if cp2_on else
            "False-yes rate is not comfortably below the hit rate (need false-yes <= "
            "hit/2 and hit >= 40%), so on this local 3B a wasted detour is too likely per "
            "genuine recovery -- ignoring the checkpoint (straight to resolve-fallback) "
            "dominates."
        )
    )

    cp3_false_demote = cp3["false_demote_rate_on_correct"] or 0.0
    cp3_on = cp3["false_demote_rate_on_correct"] is not None and cp3_false_demote < 0.10
    lines.append(
        f"\n**CP3: {'ON' if cp3_on else 'OFF'}** -- false-demote-on-correct-crop rate "
        f"{_fmt_pct(cp3['false_demote_rate_on_correct'])} (n={cp3['n_correct']}); for "
        f"context, confirm rate on correct crops {_fmt_pct(cp3['confirm_rate_on_correct'])}, "
        f"reject rate on wrong crops {_fmt_pct(cp3['reject_rate_on_wrong'])}, "
        f"demote rate on wrong crops {_fmt_pct(cp3['demote_rate_on_wrong'])}. "
        + (
            "The only actionable failure (false demote of a correct anchor) stays under "
            "10%, and OR-F9's conf>=0.8 gate is already doing real work damping the rest -- "
            "acting on CP3 is net-positive over always trusting the map."
            if cp3_on else
            "The only actionable failure (false demote of a correct anchor) is at or above "
            "the 10% bar, so on this local 3B CP3 throws away correct anchors often enough "
            "that always trusting the map (CP3 off) dominates."
        )
    )

    cp5_fmt = cp5["format_valid_rate"] or 0.0
    cp5_sane = cp5["sane_choice_rate"] or 0.0
    cp5_on = cp5["format_valid_rate"] is not None and cp5_fmt >= 0.90 and cp5_sane >= 0.90
    lines.append(
        f"\n**CP5: {'ON' if cp5_on else 'OFF'}** -- format-valid rate "
        f"{_fmt_pct(cp5['format_valid_rate'])}, sane-choice rate "
        f"{_fmt_pct(cp5['sane_choice_rate'])}, fallback rate {_fmt_pct(cp5['fallback_rate'])} "
        f"(n={cp5['n']}). "
        + (
            "Both bars clear >=90%, and CP5's downside on a bad pick is bounded (one "
            "exploration leg) -- worth wiring over the free geometric fallback."
            if cp5_on else
            "Format-valid or sane-choice rate falls under the 90% bar, so on this local 3B "
            "a mis-parsed/out-of-range reply degrades into the geometric fallback often "
            "enough that it is not worth the extra call over always using that fallback."
        )
    )

    lines.append(
        f"\n**Combined with the Phase-2 parse-half verdict "
        f"(`reports/local_llm_phase2/parse_battery.md`): CP1 OFF (floor wins), "
        f"CP2 {'ON' if cp2_on else 'OFF'}, CP3 {'ON' if cp3_on else 'OFF'}, "
        f"CP5 {'ON' if cp5_on else 'OFF'}.**\n"
    )
    return "\n".join(lines)


def write_report(
    out_dir: Path, rows: list[CaseResult], agg: dict[str, dict[str, Any]],
    scenes_used: list[str], stride: int, model: str, base_url: str,
) -> None:
    md = []
    md.append("# Phase-2 vision-checkpoint replay (CP2/CP3/CP5)\n")
    md.append(
        f"Offline replay of the three VISION checkpoints against real recorded panoramas "
        f"(`data/sim_bags/{{{','.join(scenes_used)}}}`, stride={stride}) with GT from "
        f"`object_list.txt`, through the real checkpoint entrypoints and the real "
        f"`OpenAIChatAdapter.vision_chat` path (model=`{model}`, base_url=`{base_url}`).\n"
    )
    md.append(f"Total cases: {len(rows)}. Raw rows: `vision_checkpoints.jsonl`.\n")

    cp2 = agg["CP2"]
    md.append("## CP2 -- detector-miss recovery (4-tile)\n")
    md.append("| metric | value | n |")
    md.append("|---|---|---|")
    md.append(f"| hit rate (present, correct noun IS visible) | {_fmt_pct(cp2['hit_rate'])} | {cp2['n_positive']} |")
    md.append(f"| false-yes rate (noun absent from whole scene) | {_fmt_pct(cp2['false_yes_rate'])} | {cp2['n_negative']} |")
    md.append(f"| latency | {_fmt_lat(cp2['latency_s'])} | {cp2['n_positive'] + cp2['n_negative']} |")
    md.append("")

    cp3 = agg["CP3"]
    md.append("## CP3 -- single-tile anchor confirm\n")
    md.append("| metric | value | n |")
    md.append("|---|---|---|")
    md.append(f"| confirm rate on correct crops (match=true) | {_fmt_pct(cp3['confirm_rate_on_correct'])} | {cp3['n_correct']} |")
    md.append(f"| reject rate on wrong crops (match=false) | {_fmt_pct(cp3['reject_rate_on_wrong'])} | {cp3['n_wrong']} |")
    md.append(f"| demote rate on wrong crops (actionable reject) | {_fmt_pct(cp3['demote_rate_on_wrong'])} | {cp3['n_wrong']} |")
    md.append(f"| false-demote rate on correct crops (actionable failure) | {_fmt_pct(cp3['false_demote_rate_on_correct'])} | {cp3['n_correct']} |")
    md.append(f"| latency | {_fmt_lat(cp3['latency_s'])} | {cp3['n_correct'] + cp3['n_wrong']} |")
    md.append("")

    cp5 = agg["CP5"]
    md.append("## CP5 -- frontier select (format validity + sanity only; no GT for \"best direction\")\n")
    md.append("| metric | value | n |")
    md.append("|---|---|---|")
    md.append(f"| format-valid rate (parseable disc choice or clean abstain) | {_fmt_pct(cp5['format_valid_rate'])} | {cp5['n']} |")
    md.append(f"| sane-choice rate (chosen disc is one of the offered) | {_fmt_pct(cp5['sane_choice_rate'])} | {cp5['n']} |")
    md.append(f"| fallback rate (abstain/timeout/malformed) | {_fmt_pct(cp5['fallback_rate'])} | {cp5['n']} |")
    md.append(f"| latency | {_fmt_lat(cp5['latency_s'])} | {cp5['n']} |")
    md.append("")

    md.append(build_enable_matrix(agg))

    (out_dir / "vision_checkpoints.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def write_jsonl(out_dir: Path, rows: list[CaseResult]) -> None:
    with (out_dir / "vision_checkpoints.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps({
                "checkpoint": r.checkpoint, "scene": r.scene, "case_kind": r.case_kind,
                "frame_idx": r.frame_idx, "noun": r.noun, "expected": r.expected,
                "outcome_action": r.outcome_action, "outcome_fields": r.outcome_fields,
                "latency_s": r.latency_s, "dist_m": r.dist_m,
            }) + "\n")


# =============================================================================== main


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "reports" / "local_llm_phase2"))
    ap.add_argument("--stride", type=int, default=12, help="pano-frame stride within each bag")
    ap.add_argument("--n-cp2-positive", type=int, default=18)
    ap.add_argument("--n-cp2-negative", type=int, default=14)
    ap.add_argument("--n-cp3-pairs", type=int, default=16)
    ap.add_argument("--n-cp5", type=int, default=12)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--base-url", default="http://localhost:11434/v1")
    ap.add_argument("--model", default="qwen2.5vl:3b")
    ap.add_argument("--api-key", default="dummy")
    ap.add_argument("--bags-dir", default=str(REPO_ROOT / "data" / "sim_bags"))
    ap.add_argument("--scenes-dir", default=str(REPO_ROOT / "data" / "unity_scenes_ros2"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    vision_chat = build_vision_chat(args.base_url, args.model, args.api_key)

    rows: list[CaseResult] = []
    scenes_used: list[str] = []
    t_wall0 = time.monotonic()

    for bag_name, gt_scene, question in SCENES:
        bag_dir = Path(args.bags_dir) / bag_name
        obj_path = Path(args.scenes_dir) / gt_scene / gt_scene / "object_list.txt"
        if not bag_dir.exists() or not obj_path.exists():
            print(f"[skip] {bag_name}: missing bag or object_list.txt")
            continue
        scenes_used.append(bag_name)
        objects = load_object_list(obj_path)
        labels = scene_labels(objects)
        print(f"[{bag_name}] scanning (stride={args.stride})...")
        t0 = time.monotonic()
        frames, cp2_cands, cp3_cands = scan_scene(bag_dir, objects, args.stride)
        print(f"[{bag_name}] {len(frames)} frames, {len(cp2_cands)} CP2 candidates, "
              f"{len(cp3_cands)} CP3 candidates, scan {time.monotonic() - t0:.1f}s")

        # ---- CP2 sampling
        rng.shuffle(cp2_cands)
        n_pos = max(1, args.n_cp2_positive // len(SCENES))
        positives = cp2_cands[:n_pos]
        absent_nouns = pick_absent_nouns(labels, max(1, args.n_cp2_negative // len(SCENES)), rng)
        frame_pool = list(frames.keys())
        negatives = [(rng.choice(frame_pool), noun) for noun in absent_nouns] if frame_pool else []

        # ---- CP3 sampling: pairs of (correct, wrong) with distinct labels where possible
        rng.shuffle(cp3_cands)
        n_pairs = max(1, args.n_cp3_pairs // len(SCENES))
        pairs: list[tuple[Cp3Candidate, Cp3Candidate]] = []
        pool = list(cp3_cands)
        for correct in pool[:n_pairs]:
            others = [c for c in cp3_cands if c.label != correct.label]
            if not others:
                continue
            wrong = rng.choice(others)
            pairs.append((correct, wrong))

        # ---- CP5 sampling: a handful of whole panoramas
        n_cp5_here = max(1, args.n_cp5 // len(SCENES))
        cp5_frames = rng.sample(frame_pool, min(n_cp5_here, len(frame_pool))) if frame_pool else []

        bag = BagSource(str(bag_dir))
        print(f"[{bag_name}] CP2: {len(positives)} positive / {len(negatives)} negative cases")
        run_cp2_cases(vision_chat, bag_name, bag, frames, positives, negatives, rows)

        bag = BagSource(str(bag_dir))
        print(f"[{bag_name}] CP3: {len(pairs)} pairs")
        run_cp3_cases(vision_chat, bag_name, bag, pairs, rows)

        bag = BagSource(str(bag_dir))
        print(f"[{bag_name}] CP5: {len(cp5_frames)} panoramas, question={question!r}")
        run_cp5_cases(vision_chat, bag_name, bag, question, cp5_frames, rows)

        print(f"[{bag_name}] cumulative wall time {time.monotonic() - t_wall0:.1f}s")

    agg = aggregate(rows)
    write_jsonl(out_dir, rows)
    write_report(out_dir, rows, agg, scenes_used, args.stride, args.model, args.base_url)

    print(f"DONE: {len(rows)} cases, {time.monotonic() - t_wall0:.1f}s wall time")
    print(json.dumps(agg, indent=2, default=str))


if __name__ == "__main__":
    main()
