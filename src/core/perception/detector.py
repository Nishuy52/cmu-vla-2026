"""Open-vocabulary 2D detection over the panorama tiles.

A :data:`DetectorFn` takes the list of pinhole tiles from
:func:`core.perception.tiling.project_tiles` and returns, per tile, a list of
:class:`Detection` boxes. The real Phase-2 detector is a GroundingDINO-class
open-vocab model prompted with the question nouns + vocab nouns
(``docs/architecture.md`` §6). To keep ``core/`` fully offline-testable, torch and
groundingdino are imported *lazily* inside :class:`GroundingDinoDetector` — importing
this module never touches them.

Tests drive the pipeline with :class:`FakeDetector`, which replays scripted
detections deterministically.
"""
from __future__ import annotations

import contextlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Protocol, Sequence

import numpy as np

from core.perception.tiling import (
    DEFAULT_N_TILES,
    DEFAULT_TILE_HFOV,
    DEFAULT_TILE_VFOV,
    tile_pixel_to_camera_ray,
    wrap_pi,
)
from core.perception.vocab import prioritize_vocab_nouns

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Detection:
    """One 2D detection in a single tile's pixel frame.

    ``bbox_xyxy`` is ``(x0, y0, x1, y1)`` in tile pixels (x right, y down). ``label``
    is the open-vocab noun the prompt matched (canonicalised downstream against the
    vocab). ``score`` is detector confidence in [0, 1]. ``mask`` is an optional
    (H, W) bool array for mask-tightened fusion (below the cut-line; usually None).
    """

    tile_id: int
    bbox_xyxy: tuple[float, float, float, float]
    label: str
    score: float
    mask: np.ndarray | None = None

    @property
    def center_xy(self) -> tuple[float, float]:
        """Pixel centre of the bbox (used for the depth/centroid ray)."""
        x0, y0, x1, y1 = self.bbox_xyxy
        return (0.5 * (x0 + x1), 0.5 * (y0 + y1))

    @property
    def foot_xy(self) -> tuple[float, float]:
        """Bottom-centre of the bbox (object's ground contact ray)."""
        x0, y0, x1, y1 = self.bbox_xyxy
        return (0.5 * (x0 + x1), float(y1))


#: The detector seam: tiles in -> per-tile detection lists out.
DetectorFn = Callable[[Sequence[np.ndarray]], list[list[Detection]]]


class DetectorProtocol(Protocol):
    """Structural type for anything usable as a :data:`DetectorFn`."""

    def __call__(self, tiles: Sequence[np.ndarray]) -> list[list[Detection]]: ...


# --------------------------------------------------------------------------- cross-tile NMS

# Issue #131: the question pass and the (cadenced) vocab pass are unioned per tick with
# no dedupe (see the ``merged[i].extend(dets)`` sites in
# :meth:`GroundingDinoDetector.__call__`). tiling.py's module docstring and
# ``tiling.DEFAULT_SEAM_OVERLAP`` describe the 4 gnomonic tiles as overlapping ~10 deg by
# construction, but at the actual shipped defaults (``DEFAULT_TILE_HFOV`` 90 deg,
# ``DEFAULT_N_TILES`` 4 -> 90 deg spacing) the nominal per-tile overlap is exactly zero
# (verified: two boxes at each tile's own edge pixel map to touching, non-overlapping
# angular ranges) — ``DEFAULT_SEAM_OVERLAP`` is presently unused/aspirational, a
# pre-existing inconsistency out of scope here (flagged separately). What the banked
# data actually shows duplicating is mostly the SAME tile: the question pass and the
# vocab pass both re-detect one physical object and each contributes its own box. That
# still produces the same downstream harm — measured at 37.5 accepted detections per
# surviving tracked instance across the banked live slots (see the #131 issue body),
# with per-keyframe worst cases like 18-23 proposals of one label in a scene that has a
# handful of that object at most. Each redundant box is then lifted independently by
# ``fusion.fuse_detection()``, so the object ends up with several different
# centroids/AABBs — which is what defeats the association gate (#128) and the extent
# veto (#130) downstream. This suppresses same-label duplicates *before* fusion ever
# sees them, comparing boxes in a shared angular frame (rather than tile pixel space) so
# the same mechanism also catches a genuine cross-seam duplicate if the tiling geometry
# is ever widened to actually overlap (``hfov`` is a parameter here for exactly that).
#
# Angular IoU threshold: 0.75. Geometric reasoning, not a fit to these 15 scenes
# (docs/calibration.md "Generalization protocol" forbids the latter) —
#   * A true duplicate is the SAME physical point cluster detected twice (two passes in
#     one tile, or — geometry permitting — two overlapping tiles) through gnomonic
#     reprojections that agree to within sub-degree residual reprojection error
#     (``tile_pixel_to_camera_ray`` is an exact analytic inverse of the forward
#     projection used to build the tiles), so a genuine duplicate's angular box matches
#     itself almost exactly: empirically a 0.9-1.0 IoU spike in the banked data's
#     same-label box-pair histogram (see the A/B below), comfortably clear of 0.75.
#   * A pair of genuinely distinct same-label objects at different depths along a
#     similar bearing (the chair-in-front-of-a-sofa case the issue calls out) differs in
#     *apparent angular size*, not just position: angular width/height both scale
#     ~1/depth, so a box for an object at 1.3x the depth of another already has ~1.3x
#     smaller angular extent in EACH axis, i.e. ~1/1.3^2 =~ 0.59 IoU even if their
#     centres coincide exactly (the closest case for two distinct objects) — a wide
#     margin below 0.75. Depth differences smaller than that (near-tangent objects that
#     really do almost coincide angularly) are exactly the case fusion's own depth
#     disambiguation (frustum/point-cluster split) exists to handle post-hoc, not this
#     seam.
#   * 0.75 therefore sits in the gap between "true duplicate" (>0.85) and "smallest
#     depth-distinguishable pair" (<0.6): conservative by construction, erring toward
#     under-suppressing (letting fusion/association see a residual duplicate, which they
#     already tolerate) rather than over-suppressing (destroying a real second instance,
#     which nothing downstream can recover).
CROSS_TILE_NMS_IOU_THRESHOLD: float = 0.75


def _detection_angular_box(
    tile_id: int,
    bbox_xyxy: tuple[float, float, float, float],
    n_tiles: int,
    hfov: float,
    vfov: float,
) -> tuple[float, float, float, float]:
    """A detection's pixel bbox -> an angular ``(az_min, az_max, el_min, el_max)`` box.

    Projects all 4 corners (not just 2) through :func:`tile_pixel_to_camera_ray`: azimuth
    is monotonic in the pixel column alone, but elevation depends on both column and row
    in the gnomonic inverse, so the row extremes alone are not guaranteed to bound it.
    Azimuths are unwrapped relative to the first corner (via the signed shortest
    difference) so a box that straddles the +-pi seam still yields a proper ``min <
    max`` interval instead of a spuriously huge one.
    """
    x0, y0, x1, y1 = bbox_xyxy
    corners = ((x0, y0), (x0, y1), (x1, y0), (x1, y1))
    ref_az: float | None = None
    azs: list[float] = []
    els: list[float] = []
    for u, v in corners:
        az, el = tile_pixel_to_camera_ray(tile_id, u, v, n_tiles, hfov, vfov)
        if ref_az is None:
            ref_az = az
        else:
            az = ref_az + float(wrap_pi(az - ref_az))
        azs.append(az)
        els.append(el)
    return min(azs), max(azs), min(els), max(els)


def _angular_box_iou(
    a: tuple[float, float, float, float], b: tuple[float, float, float, float],
) -> float:
    """IoU of two ``(az_min, az_max, el_min, el_max)`` angular boxes.

    Treats ``(azimuth, elevation)`` as a flat 2D plane (the same small-angle
    approximation ``fusion.py``'s own angular gating already relies on) and handles the
    azimuth wraparound by shifting ``b`` by whichever multiple of 2*pi brings its centre
    closest to ``a``'s centre before intersecting — valid because both boxes are always
    far narrower than 2*pi (at most one tile's ``hfov``, 90 deg).
    """
    a_az0, a_az1, a_el0, a_el1 = a
    b_az0, b_az1, b_el0, b_el1 = b
    a_center = 0.5 * (a_az0 + a_az1)
    b_center = 0.5 * (b_az0 + b_az1)
    shift = round((a_center - b_center) / (2.0 * np.pi)) * 2.0 * np.pi
    b_az0 += shift
    b_az1 += shift
    inter_az = max(0.0, min(a_az1, b_az1) - max(a_az0, b_az0))
    inter_el = max(0.0, min(a_el1, b_el1) - max(a_el0, b_el0))
    inter = inter_az * inter_el
    area_a = (a_az1 - a_az0) * (a_el1 - a_el0)
    area_b = (b_az1 - b_az0) * (b_el1 - b_el0)
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def suppress_cross_tile_duplicates(
    detections_by_tile: Sequence[Sequence[Detection]],
    *,
    n_tiles: int | None = None,
    hfov: float = DEFAULT_TILE_HFOV,
    vfov: float = DEFAULT_TILE_VFOV,
    iou_threshold: float = CROSS_TILE_NMS_IOU_THRESHOLD,
) -> list[list[Detection]]:
    """Class-wise NMS over one tick's detections, in a shared angular frame (#131).

    ``detections_by_tile`` is the per-tile structure the detector seam already uses
    (``detections_by_tile[tile_id]`` -> that tile's :class:`Detection` list, e.g. the
    ``merged`` union of the question + vocab passes). Boxes are compared in
    ``(azimuth, elevation)`` space (via :func:`_detection_angular_box`) rather than tile
    pixel space, so a duplicate spanning two neighbouring (differently-projected) tiles
    is still recognised as one object. Only same-``label`` boxes ever compete; the
    highest-``score`` box in each cluster survives, its tile assignment unchanged.
    Returns a same-shape ``list[list[Detection]]`` (one list per input tile) containing
    only the surviving detections, in their original per-tile order.
    """
    resolved_n_tiles = n_tiles if n_tiles is not None else (len(detections_by_tile) or DEFAULT_N_TILES)
    # (tile_id, index-within-tile, Detection) so survivors can be re-bucketed by tile
    # afterward without disturbing each tile's original relative order.
    flat: list[tuple[int, int, Detection]] = []
    for tile_id, dets in enumerate(detections_by_tile):
        for idx, det in enumerate(dets):
            flat.append((tile_id, idx, det))
    if len(flat) <= 1:
        return [list(dets) for dets in detections_by_tile]

    boxes = [
        _detection_angular_box(tile_id, det.bbox_xyxy, resolved_n_tiles, hfov, vfov)
        for tile_id, _idx, det in flat
    ]
    by_label: dict[str, list[int]] = {}
    for i, (_tile_id, _idx, det) in enumerate(flat):
        by_label.setdefault(det.label, []).append(i)

    keep = [False] * len(flat)
    for indices in by_label.values():
        # Highest score first — the box kept for a duplicate cluster is always its
        # best-scored member.
        ordered = sorted(indices, key=lambda i: flat[i][2].score, reverse=True)
        suppressed = [False] * len(ordered)
        for a_pos, i in enumerate(ordered):
            if suppressed[a_pos]:
                continue
            keep[i] = True
            for b_pos in range(a_pos + 1, len(ordered)):
                if suppressed[b_pos]:
                    continue
                j = ordered[b_pos]
                if _angular_box_iou(boxes[i], boxes[j]) >= iou_threshold:
                    suppressed[b_pos] = True

    survivors: list[list[Detection]] = [[] for _ in detections_by_tile]
    for i, (tile_id, _idx, det) in enumerate(flat):
        if keep[i]:
            survivors[tile_id].append(det)
    return survivors


# --------------------------------------------------------------------------- fake detector


class FakeDetector:
    """Deterministic scripted detector for offline tests.

    Construct with either a flat list of :class:`Detection` (routed to their own
    ``tile_id``) or a per-call script (a list of frames, each a list of detections)
    replayed one frame per call. With a script it advances through frames on each
    call and repeats the last frame once exhausted; without one it returns the same
    fixed detections every call.
    """

    def __init__(
        self,
        detections: Sequence[Detection] | None = None,
        *,
        script: Sequence[Sequence[Detection]] | None = None,
        n_tiles: int = 4,
    ) -> None:
        self.n_tiles = int(n_tiles)
        self._fixed = list(detections or [])
        self._script = [list(f) for f in script] if script is not None else None
        self._call = 0
        # Mirrors GroundingDinoDetector.prompt (issue #34): a plain string attribute so
        # FakeDetector can stand in for the real detector in prompt-refresh tests
        # (:func:`refresh_prompt`) without changing its scripted-replay behaviour, which
        # never actually consults ``self.prompt`` — the script/fixed detections play back
        # regardless, matching its documented "deterministic scripted detector" contract.
        self.prompt: str = ""
        # Mirrors GroundingDinoDetector.question_prompt (issue #42 dual-pass detection):
        # same rationale as ``self.prompt`` above — kept in sync by :func:`refresh_prompt`
        # but never consulted by the scripted replay itself.
        self.question_prompt: str = ""

    def __call__(self, tiles: Sequence[np.ndarray]) -> list[list[Detection]]:
        n = len(tiles)
        if self._script is not None:
            idx = min(self._call, len(self._script) - 1)
            frame = self._script[idx] if self._script else []
            self._call += 1
        else:
            frame = self._fixed
        out: list[list[Detection]] = [[] for _ in range(n)]
        for det in frame:
            if 0 <= det.tile_id < n:
                out[det.tile_id].append(det)
        return out


# --------------------------------------------------------------------------- gdino stub


#: Expected model + prompt format for the Phase-2 GroundingDINO integration.
GDINO_MODEL_ID: str = "IDEA-Research/grounding-dino-base"
GDINO_PROMPT_TEMPLATE: str = "{nouns}"  # ". "-joined noun phrases, trailing " ."

#: The exact pip installs the real path needs (Phase 2 only; never at test time).
GDINO_REQUIRED_INSTALLS: tuple[str, ...] = (
    "torch",
    "torchvision",
    "groundingdino-py",
)

# --------------------------------------------------------------------------- gdino env

# Gate 4: every knob below is overridable via env var (set in the Docker image / compose
# env_file, never in code) so a deploy-time change never needs a core/ edit. Constructor
# args always win over the env var, which wins over the module default.

#: Override :data:`GDINO_MODEL_ID` at deploy time (same name, read as an env var).
ENV_GDINO_MODEL_ID = "GDINO_MODEL_ID"

#: "fp16" | "fp32" | "auto" (default). "auto" -> fp16 on CUDA (8 GB dev box / the 10-14 GB
#: eval 4090 both want half precision headroom, architecture doc §6), fp32 on CPU (halved
#: math on CPU is usually slower, not faster, and some CPU kernels don't support it).
ENV_GDINO_PRECISION = "GDINO_PRECISION"
DEFAULT_GDINO_PRECISION = "auto"

#: "cuda" | "cpu"; unset -> torch.cuda.is_available().
ENV_GDINO_DEVICE = "GDINO_DEVICE"

#: Local filesystem paths to the model config .py / checkpoint .pth. Per the
#: offline-capable rule (docs/ubuntu_setup.md §7), both are baked into the Docker image
#: at build time and never fetched over the network at inference time.
ENV_GDINO_CONFIG_PATH = "GDINO_CONFIG_PATH"
ENV_GDINO_CHECKPOINT_PATH = "GDINO_CHECKPOINT_PATH"

# --------------------------------------------------------------------------- load backoff

# Issue #39: a permanently-failing model load (bad weights, OOM, missing config) used to
# re-run the full load_model cascade every perception tick (5 Hz), pegging adapter CPU and
# starving other subscribers. Failed loads now back off exponentially instead.

#: First retry delay after a load failure, doubling each consecutive failure.
GDINO_BACKOFF_BASE_S: float = 1.0
#: Once degraded (see :data:`GDINO_BACKOFF_DEGRADE_N`), every retry waits this long.
GDINO_BACKOFF_CAP_S: float = 60.0
#: Consecutive failures after which the detector stops climbing the backoff curve and
#: settles at the capped retry interval (logged once as a single ERROR transition).
GDINO_BACKOFF_DEGRADE_N: int = 5

# --------------------------------------------------------------------------- dual-caption detection

# Issue #42: the live 117-phrase caption (question nouns + full 114-noun standing vocab)
# decodes ZERO 'teapot' at any threshold — caption dilution, not a threshold problem (the
# gate-4 grounding probe, reports/gate4_grounding_probe/). A short "teapot . table ."
# caption decodes teapot 194/211 keyframes and peaks at score 0.292, below the 0.35 vocab-
# pass box threshold. So every detection tick now runs a SHORT question-noun-only caption
# pass at its own (lower) box threshold for target recall, and the existing full
# question+vocab caption pass — which feeds scene-index breadth (anchor/other-object
# instances), not target recall — runs at a reduced cadence instead of every tick.

#: Box threshold for the short question-noun-only caption pass. Lower than
#: :attr:`GroundingDinoDetector.box_threshold`'s 0.35 default because a 2-phrase prompt
#: carries far less token-position score dilution than the full vocab caption (probe:
#: peak short-caption teapot score 0.292, 19/19 detections within 6.4 deg of GT at 0.25).
ENV_GDINO_QUESTION_BOX_THRESHOLD = "GDINO_QUESTION_BOX_THRESHOLD"
DEFAULT_GDINO_QUESTION_BOX_THRESHOLD: float = 0.25

#: Run the full question+vocab caption pass every Nth detection tick (1 == every tick,
#: matching the pre-#42 behaviour). The question-noun pass still runs every tick.
ENV_GDINO_VOCAB_PASS_CADENCE = "GDINO_VOCAB_PASS_CADENCE"
DEFAULT_GDINO_VOCAB_PASS_CADENCE: int = 3

# --------------------------------------------------------------------------- answer eligibility

# Issue #43(a): DEFAULT_GDINO_QUESTION_BOX_THRESHOLD (0.25) above is a RECALL floor — low
# enough that a weakly-scored, barely-observed instance still enters the scene index and
# feeds exploration/frontier scoring. That is deliberately permissive; it must not also be
# the bar an instance clears to WIN an answer. A question-pass-derived instance (i.e. one
# resolved for the question's own target noun — the noun the short question-noun caption
# pass grounds on) is ANSWER-eligible only once it has been independently re-observed
# (n_obs >= the min below) AND its peak detector confidence clears a higher score floor —
# both stricter than the 0.25 recall floor, which stays untouched for index/exploration.

#: Minimum distinct-keyframe observation count for a target-noun instance to be
#: ANSWER-eligible (win the object-ref ranking head, or populate object-ref floor rung 3).
ENV_GDINO_ANSWER_MIN_OBS = "GDINO_ANSWER_MIN_OBS"
DEFAULT_GDINO_ANSWER_MIN_OBS: int = 2

#: Minimum peak detector confidence (``InstanceRecord.score``, max over observations) for
#: answer eligibility. Above the 0.25 recall floor by design (issue #43(a)).
ENV_GDINO_ANSWER_MIN_SCORE = "GDINO_ANSWER_MIN_SCORE"
DEFAULT_GDINO_ANSWER_MIN_SCORE: float = 0.30


def answer_min_obs() -> int:
    """Resolve the answer-eligibility min-observation floor (env override > default)."""
    try:
        return int(os.environ.get(ENV_GDINO_ANSWER_MIN_OBS, DEFAULT_GDINO_ANSWER_MIN_OBS))
    except (TypeError, ValueError):
        return DEFAULT_GDINO_ANSWER_MIN_OBS


def answer_min_score() -> float:
    """Resolve the answer-eligibility min peak-score floor (env override > default)."""
    try:
        return float(os.environ.get(ENV_GDINO_ANSWER_MIN_SCORE, DEFAULT_GDINO_ANSWER_MIN_SCORE))
    except (TypeError, ValueError):
        return DEFAULT_GDINO_ANSWER_MIN_SCORE


#: :func:`answer_eligibility_reason`'s possible return values, for consumers that want
#: to branch on the reason rather than pattern-match the string (issue #84 gate
#: observability: this gate's rejections used to be a silent boolean with no visibility
#: anywhere — offline the GT mocks are born n_obs=3 (``core.runner.gt_battery``), well
#: clear of both floors, so the gate is never actually exercised by the battery; the only
#: place it can silently reject something is live, where it was previously invisible).
ELIGIBLE = "eligible"
INELIGIBLE_N_OBS = "n_obs_below_floor"
INELIGIBLE_SCORE = "score_below_floor"
INELIGIBLE_BOTH = "n_obs_and_score_below_floor"
INELIGIBLE_MALFORMED = "malformed"


def answer_eligibility_reason(record: object) -> str:
    """Diagnostic breakdown of :func:`is_answer_eligible`'s verdict for ``record``.

    Returns one of :data:`ELIGIBLE`, :data:`INELIGIBLE_N_OBS`, :data:`INELIGIBLE_SCORE`,
    :data:`INELIGIBLE_BOTH`, or :data:`INELIGIBLE_MALFORMED` (missing ``.n_obs``/``.score``).
    Pure/side-effect-free — callers (logging here, the instance-index dump in
    :mod:`core.perception.scene_index`, tests) decide what to do with the reason.
    """
    n_obs = getattr(record, "n_obs", None)
    score = getattr(record, "score", None)
    if n_obs is None or score is None:
        return INELIGIBLE_MALFORMED
    obs_ok = n_obs >= answer_min_obs()
    score_ok = score >= answer_min_score()
    if obs_ok and score_ok:
        return ELIGIBLE
    if not obs_ok and not score_ok:
        return INELIGIBLE_BOTH
    return INELIGIBLE_N_OBS if not obs_ok else INELIGIBLE_SCORE


def is_answer_eligible(record: object) -> bool:
    """True iff ``record`` (an :class:`~core.interfaces.InstanceRecord`) clears both the
    answer-eligibility observation and peak-score floors (issue #43a).

    Duck-typed on ``.n_obs`` / ``.score`` (avoids importing ``core.interfaces`` from this
    module purely for a type hint); a record missing either attribute is treated as
    ineligible rather than raising, so a malformed/partial record never wins an answer.

    Issue #84 gate observability: a rejection is logged at DEBUG (not silently dropped)
    with the :func:`answer_eligibility_reason` breakdown and the record's id/label when
    available, so a live run's debug log (or anything capturing this logger) shows WHY an
    instance lost an answer — the offline battery's GT mocks (n_obs=3) never clear this
    path since they always pass, so this was previously unexercisable outside a live run.
    """
    reason = answer_eligibility_reason(record)
    if reason == ELIGIBLE:
        return True
    _LOGGER.debug(
        "answer-eligibility gate rejected instance id=%r label=%r n_obs=%r score=%r: %s",
        getattr(record, "instance_id", None), getattr(record, "label", None),
        getattr(record, "n_obs", None), getattr(record, "score", None), reason,
    )
    return False


# --------------------------------------------------------------------------- raw detection dump
#
# Issue #84: dump_instance_index (core.perception.scene_index) shows what SURVIVED into
# the scene index; it cannot tell "an anchor class the question needs was NEVER proposed
# by the detector at all this run" apart from "it was proposed and then dropped by a
# downstream gate" (office_1: 2/4 question-anchor classes -- projector screen, window --
# were never indexed, and there was no way to tell which of those two this was without
# re-running with extra logging). This is the missing pre-gate view: every raw
# :class:`Detection` the :data:`DetectorFn` returned this keyframe, tagged with whether
# fusion (the only point in the pipeline that can drop a detection outright -- see
# :mod:`core.perception.fusion`'s ``min_points`` cluster-size floor) accepted it and,
# if so, which instance id it landed in via the tracker's association.
#
# SCOPE NOTE: this module's own DetectorFn implementations (:class:`GroundingDinoDetector`
# in particular) already apply the model's box_threshold internally when decoding raw
# logits into Detection objects (see ``_decode_batch_item``) -- a candidate box scoring
# below that threshold never becomes a Detection at all, so this dump cannot recover a
# sub-box-threshold raw score. What it CAN split apart is "the detector emitted this
# label at least once, above its own box threshold, this run" (proposed) vs "downstream
# fusion/tracking gated every instance of it out" (proposed-but-gated) -- which is
# exactly the split #84 needs, given the DetectorFn seam's contract.
#
# Opt-in only, same contract as VLA_INSTANCE_DUMP_PATH: unset -> a single os.environ.get
# per keyframe and nothing else runs; any dump failure is swallowed so diagnostics never
# break the run they are observing.

#: Path to append JSONL raw-detection records to. Unset (default) -> dump_raw_detections
#: is a no-op (single os.environ.get, no I/O). Debug-only; never affects the scored path.
ENV_RAW_DETECTION_DUMP_PATH: str = "VLA_RAW_DETECTION_DUMP_PATH"

#: :func:`dump_raw_detections`'s per-detection ``gate`` values.
GATE_ACCEPTED = "accepted"                    # fused with a lidar cluster and indexed
GATE_NO_LIDAR_CLUSTER = "no_lidar_cluster"    # fusion rejected: < FusionConfig.min_points


def dump_raw_detections(
    records: Sequence[tuple[Detection, str, int | None]],
    *,
    keyframe_idx: int,
    tag: str = "raw_detections",
) -> None:
    """Append one JSONL record of this keyframe's RAW (pre-eligibility-gate) detector
    output, if :data:`ENV_RAW_DETECTION_DUMP_PATH` is set. No-op (no I/O at all) when
    unset.

    ``records`` is one ``(detection, gate, instance_id)`` triple per raw
    :class:`Detection` the detector returned this keyframe, where ``gate`` is
    :data:`GATE_ACCEPTED` (fusion produced a lidar cluster and the tracker filed it
    into ``instance_id``) or :data:`GATE_NO_LIDAR_CLUSTER` (fusion's ``min_points``
    floor rejected it before it ever reached the tracker/scene index -- ``instance_id``
    is ``None`` in that case).

    Record shape: ``wall_time``, ``tag``, ``keyframe_idx``, ``total_detections``,
    ``by_class`` (label -> ``{"total", "accepted", "gated"}`` counts), and
    ``detections`` -- one entry per raw detection with ``tile_id``/``label``/``score``/
    ``bbox_xyxy``/``gate``/``instance_id``.

    Any failure (bad path, unwritable dir, etc.) is swallowed -- diagnostics must never
    break the run they are observing, matching
    :func:`core.perception.scene_index.dump_instance_index`.
    """
    path = os.environ.get(ENV_RAW_DETECTION_DUMP_PATH)
    if not path:
        return
    try:
        by_class: dict[str, dict[str, int]] = {}
        detections_out: list[dict] = []
        for det, gate, instance_id in records:
            stats = by_class.setdefault(det.label, {"total": 0, "accepted": 0, "gated": 0})
            stats["total"] += 1
            stats["accepted" if gate == GATE_ACCEPTED else "gated"] += 1
            detections_out.append(
                {
                    "tile_id": int(det.tile_id),
                    "label": det.label,
                    "score": round(float(det.score), 4),
                    "bbox_xyxy": [round(float(c), 2) for c in det.bbox_xyxy],
                    "gate": gate,
                    "instance_id": int(instance_id) if instance_id is not None else None,
                }
            )
        record = {
            "wall_time": time.time(),
            "tag": tag,
            "keyframe_idx": keyframe_idx,
            "total_detections": len(detections_out),
            "by_class": by_class,
            "detections": detections_out,
        }
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "a") as fh:
            fh.write(json.dumps(record) + "\n")
    except Exception:  # noqa: BLE001 - diagnostics must never break the run
        pass


#: Issue #105: GroundingDINO's own BERT-class text encoder truncates its input at this
#: many wordpiece tokens (``max_text_len`` in the exact config this project loads,
#: ``GroundingDINO_SwinB_cfg.py``); anything past it is silently dropped by the
#: tokenizer, never by anything in this codebase, which is why the loss was invisible
#: for the whole project's history. The un-budgeted prompt (116 standing-vocab nouns,
#: alphabetically sorted per :data:`core.heads.factory._STANDING_VOCAB_NOUNS`) measures
#: 293 real ``bert-base-uncased`` tokens -- 37 over this limit -- which silently
#: amputates the alphabet's tail (``trash can`` through ``window``) from every
#: detection pass, on every scene, forever.
DEFAULT_GDINO_MAX_TEXT_TOKENS: int = 256


def _heuristic_token_estimate(text: str) -> int:
    """Dependency-free, conservative upper bound on BERT-style wordpiece token count.

    ``src/`` has no hard dependency on ``transformers`` (the fast test tier must keep
    working without it), so this cannot call the real tokenizer -- it approximates it
    from word length alone, calibrated against a real ``bert-base-uncased`` encoding of
    the full 116-noun standing vocab (measured: 293 tokens, see
    :data:`DEFAULT_GDINO_MAX_TEXT_TOKENS`). Wordpiece splits roughly track word length
    in that measurement (max observed: 1 token per <=4 chars, 2 per <=7, 3 per <=11, 4
    beyond), so this buckets on the same breakpoints -- deliberately loose enough that
    it never underestimated a single word in that measurement (calibration pinned by
    ``test_default_token_estimate_never_underestimates_measured_vocab``). ``+2`` covers
    the CLS/SEP special tokens BERT-class tokenizers wrap every sequence in.

    Deliberately a plain word-length heuristic, not a smarter approximation: the
    failure mode this exists to prevent is silent overflow, so the estimate must stay
    on the safe (over-count) side even for vocabulary this measurement never saw --
    trading some caption capacity for that guarantee is the right side to err on.

    This is the fallback tier only (see :func:`_default_token_estimate`): the deployed
    environment has the real tokenizer available (``groundingdino-py`` pulls in
    ``transformers`` transitively, ``docker/ai_module_fork/docker/Dockerfile``), where
    exact counting keeps far more of the standing vocab than this conservative
    approximation would.
    """
    if not text:
        return 0
    total = 2
    for word in text.split():
        length = len(word)
        if length <= 4:
            total += 1
        elif length <= 7:
            total += 2
        elif length <= 11:
            total += 3
        else:
            total += 4
    return total


#: Cache for the real-tokenizer probe (issue #105 follow-up): ``None`` before the first
#: probe, a callable ``str -> int`` once a real tokenizer loads, or the sentinel
#: :data:`_TOKENIZER_UNAVAILABLE` if the probe failed once and should not be retried
#: (retrying a broken/missing install on every detection tick would waste cycles on the
#: hot path for no benefit -- the environment does not change mid-run).
_TOKENIZER_UNAVAILABLE = object()
_cached_real_tokenizer: object | None = None


def _probe_real_tokenizer() -> Callable[[str], int] | None:
    """Load and cache a real BERT tokenizer's exact-count function, once.

    Returns ``None`` (every call, cheaply) if no real tokenizer is importable/loadable
    -- ``transformers`` absent (the ``src/`` fast test tier) or any other failure
    constructing it. Never raises: this sits ahead of a detection-path call
    (:func:`_default_token_estimate`), so a broken install must degrade to the
    heuristic, not take the pipeline down with it (same discipline as
    :func:`dump_raw_detections`'s diagnostics-must-never-break-the-run guard above).

    Logs exactly once, at whichever outcome the first call resolves to, so a live run's
    logs say once and for all whether that run budgeted the GDINO prompt exactly (real
    tokenizer) or conservatively (heuristic fallback) -- previously undiscoverable from
    a bag.
    """
    global _cached_real_tokenizer
    if _cached_real_tokenizer is None:
        try:
            from transformers import AutoTokenizer  # lazy: optional dependency

            tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

            def _count(text: str) -> int:
                return len(tokenizer(text)["input_ids"])

            _cached_real_tokenizer = _count
            _LOGGER.info(
                "build_gdino_prompt: real bert-base-uncased tokenizer loaded -- "
                "budgeting the GDINO prompt with exact token counts."
            )
        except Exception:  # noqa: BLE001 - any failure degrades to the heuristic
            _cached_real_tokenizer = _TOKENIZER_UNAVAILABLE
            _LOGGER.info(
                "build_gdino_prompt: no real tokenizer available (transformers not "
                "installed, or failed to load) -- budgeting the GDINO prompt with the "
                "conservative word-length heuristic instead."
            )
    if _cached_real_tokenizer is _TOKENIZER_UNAVAILABLE:
        return None
    return _cached_real_tokenizer  # type: ignore[return-value]


def _default_token_estimate(text: str) -> int:
    """Exact real-tokenizer count when available, else the conservative heuristic.

    Issue #105 follow-up: the heuristic in :func:`_heuristic_token_estimate` is
    calibrated to never under-estimate, which necessarily means it over-estimates --
    measured at 421 estimated vs. 293 real tokens for the full standing vocab, which
    left only 69/116 nouns fitting the 256-token budget (vs. 104/116 that actually
    reach the model pre-budget-cut, since only the alphabetic tail past 256 real
    tokens silently truncated). The real tokenizer is available in the deployed
    environment (``groundingdino-py`` -> ``transformers`` transitively), so this uses
    it whenever it loads successfully, falling back to the heuristic only where it
    cannot (the dependency-free ``src/`` fast test tier, or if the install is somehow
    broken) -- never letting tokenizer construction/failure propagate into the
    detection path.
    """
    if not text:
        return 0
    real_counter = _probe_real_tokenizer()
    if real_counter is not None:
        try:
            return real_counter(text)
        except Exception:  # noqa: BLE001 - degrade to heuristic, never raise
            _LOGGER.warning(
                "build_gdino_prompt: real tokenizer call failed on this prompt -- "
                "falling back to the heuristic estimate for it.",
            )
    return _heuristic_token_estimate(text)


def build_gdino_prompt(
    question_nouns: Sequence[str],
    vocab_nouns: Sequence[str],
    *,
    max_tokens: int = DEFAULT_GDINO_MAX_TEXT_TOKENS,
    token_estimator: Callable[[str], int] = _default_token_estimate,
    dropped_out: list[str] | None = None,
) -> str:
    """Build the GroundingDINO text prompt from question + vocab nouns, token-budgeted.

    GroundingDINO expects a lowercase, ``.``-separated list of noun phrases with a
    trailing separator, e.g. ``"sofa . window . potted plant ."``. Question nouns come
    first (query-relevant recall priority), then any extra vocab nouns, both
    de-duplicated preserving order.

    Issue #105: the caption is silently truncated by GroundingDINO's text encoder past
    ``max_tokens`` wordpiece tokens (default 256, its real ``max_text_len``) -- so this
    budgets the caption itself rather than letting that happen invisibly downstream.
    Priority order matters: question nouns are never dropped for vocab nouns. Vocab
    nouns are appended in the order given (the caller's priority order, e.g.
    alphabetical) until the next one would push the estimated token count over budget;
    everything from that point on is the dropped tail, never cherry-picked out of
    order, so the drop is a predictable, describable set rather than a shuffled one.

    ``token_estimator`` defaults to :func:`_default_token_estimate`, a dependency-free
    heuristic that over-estimates rather than under-estimates (calibrated against a
    real 293-token ``bert-base-uncased`` encoding of the full standing vocab -- see
    that function's docstring); pass the real tokenizer's ``len(tokenizer(text)[...])``
    here for an exact count where ``transformers`` is available.

    Dropping is never silent (the entire point of issue #105 is that it used to be):
    every drop logs a warning naming the dropped nouns, and if the caller passes a
    ``dropped_out`` list it is extended with them (in drop order) so the caller can
    surface or assert on it without parsing logs. If the question nouns alone already
    estimate over budget, they are still kept in full -- silently trimming the query's
    own target/anchor nouns would be worse than a caption GroundingDINO itself
    truncates -- but that condition is logged as an error, never swallowed.

    Chunking the standing vocab across successive keyframes (so the full vocab gets
    covered over time instead of the same tail being permanently dropped) was
    considered and deliberately left undone here: this function is pure and stateless
    (nouns in, prompt out) and callers (:mod:`core.heads.factory`, out of this change's
    ownership) do not thread any per-keyframe rotation state through it today. Bolting
    module-level rotation state onto a pure function to fake statefulness would be a
    half-measure that changes detection recall in a way that needs its own design and
    test coverage, not a token-budget bug fix; the dropped-tail is now at least
    visible and reportable (this fix), which is the prerequisite for a future
    keyframe-rotation change to be designed against.
    """
    seen: set[str] = set()
    question_ordered: list[str] = []
    for noun in question_nouns:
        n = noun.strip().lower()
        if n and n not in seen:
            seen.add(n)
            question_ordered.append(n)
    vocab_ordered: list[str] = []
    for noun in vocab_nouns:
        n = noun.strip().lower()
        if n and n not in seen:
            seen.add(n)
            vocab_ordered.append(n)

    def render(nouns: list[str]) -> str:
        return " . ".join(nouns) + " ." if nouns else ""

    if not question_ordered and not vocab_ordered:
        return ""

    kept = list(question_ordered)
    question_tokens = token_estimator(render(kept)) if kept else 0
    if question_tokens > max_tokens:
        _LOGGER.error(
            "build_gdino_prompt: question nouns alone estimate %d tokens, over the "
            "%d-token budget (keeping all %d anyway -- never drop question nouns): %s",
            question_tokens, max_tokens, len(kept), kept,
        )

    dropped: list[str] = []
    fits = True
    for noun in vocab_ordered:
        if fits:
            trial_tokens = token_estimator(render(kept + [noun]))
            if trial_tokens <= max_tokens:
                kept.append(noun)
                continue
            fits = False
        dropped.append(noun)

    if dropped:
        _LOGGER.warning(
            "build_gdino_prompt: dropping %d/%d vocab noun(s) to stay within the "
            "%d-token budget: %s",
            len(dropped), len(vocab_ordered), max_tokens, dropped,
        )
        if dropped_out is not None:
            dropped_out.extend(dropped)

    return render(kept)


def _eviction_safe_vocab_order(
    question_nouns: Sequence[str],
    full_only_nouns: Sequence[str],
    vocab_nouns: Sequence[str],
    *,
    max_tokens: int = DEFAULT_GDINO_MAX_TEXT_TOKENS,
    token_estimator: Callable[[str], int] = _default_token_estimate,
) -> list[str]:
    """Reorder ``vocab_nouns`` so disambiguator promotion can never evict a survivor.

    Issue #91 follow-up: the original fix (:func:`core.perception.vocab.
    prioritize_vocab_nouns` applied to the WHOLE standing-vocab list) reordered every
    noun, not just the already-dropped tail -- so pulling :data:`core.perception.vocab.
    DISAMBIGUATOR_PRIORITY_NOUNS` to the front could, and measurably did, push nouns
    that used to survive the token-budget cut (:func:`build_gdino_prompt`) past it
    instead. Verified against the live 116-noun standing vocab under the heuristic
    estimator (the deployed fallback when ``transformers`` is unavailable): the plain
    whole-list reorder evicts ``microwave``, ``mirror`` and ``monitor`` -- none of
    which are in the #91 evidence's overflow tail -- to make room, which is a worse
    trade than the one #91 was fixing (``monitor`` alone is issue #94's single
    highest-frequency detected class; ``mirror`` is an arabic_room terminal-leg anchor).

    This fixes that by computing the SAME budget cut :func:`build_gdino_prompt` would
    apply to the unreordered ``vocab_nouns`` first (a dry run), splitting the result
    into the nouns that already survive it and the tail that doesn't, and reordering
    ONLY the tail (:func:`core.perception.vocab.prioritize_vocab_nouns`) before
    re-appending it after the untouched survivors. Every survivor keeps its exact
    pre-reorder cumulative token position, so the real budget cut (run again on this
    returned order, downstream in :func:`refresh_prompt`) reproduces the identical
    survivor set -- reordering the tail can rescue a disambiguator noun into the
    leftover headroom past the survivors, but it can never cost a survivor its spot,
    because nothing about the survivors' order or count changed.

    Reserving budget rather than reordering the survivors was chosen over an explicit
    hand-maintained "protected noun" allowlist (the other option on the table):
    protecting *every* current survivor is strictly safer than protecting a
    hand-picked subset (a future standing-vocab edit can't quietly reintroduce this
    same bug for some OTHER noun the allowlist didn't anticipate), and it needs no
    maintenance as the standing vocab or budget changes -- the dry run recomputes the
    survivor/tail split from whatever the caller passes, every time.

    Measured budget arithmetic (heuristic estimator, current 116-noun standing vocab,
    no question/full-only nouns latched): the unreordered vocab already sits at 253
    heuristic tokens against the 256 budget -- 3 tokens of headroom. The cheapest
    dropped disambiguator noun (``wall decal``) costs 4 incremental tokens once
    rendered into the caption (its own wordpieces plus the ``" . "`` separator); the
    other overflowing one (``pyramid candle holder``) costs 7. Neither fits in 3
    tokens of headroom, so under this estimator NEITHER is actually rescued by this
    fix -- the tail reorder still happens (harmless), but the dry-run survivor
    prefix consumes the entire budget before either gets a turn. This is a real,
    reported finding, not a bug in this function: the #91 promotion's gain was
    entirely paid for by the microwave/mirror/monitor eviction it is this function's
    job to prevent, and there is no free lunch of spare heuristic-estimator budget to
    grant it instead. The real ``bert-base-uncased`` tokenizer (the deployed
    environment's actual path once ``transformers`` is installed) counts differently
    and may have different headroom; this fix does not assume either way; it is safe
    under whichever estimator ``build_gdino_prompt`` is actually called with.
    """
    baseline_dropped: list[str] = []
    build_gdino_prompt(
        question_nouns,
        list(full_only_nouns) + list(vocab_nouns),
        max_tokens=max_tokens,
        token_estimator=token_estimator,
        dropped_out=baseline_dropped,
    )
    dropped_set = set(baseline_dropped)
    survived = [n for n in vocab_nouns if n not in dropped_set]
    tail = [n for n in vocab_nouns if n in dropped_set]
    return survived + prioritize_vocab_nouns(tail)


#: Issue #145: the composed GDINO prompt was logged nowhere -- not in job stdout, not
#: in any debug artifact -- so a live sweep could not tell "the noun was never queried
#: because the prompt hit the 256-token cap" (this issue) apart from "the noun was
#: queried and the detector missed it" (#91) or "the noun was detected under a merged
#: label" (#154). Path to append JSONL prompt-diagnostic records to. Unset (default) ->
#: :func:`dump_prompt_diagnostics` is a no-op (single ``os.environ.get``, no I/O),
#: matching every other opt-in debug dump in this codebase (``core.perception.
#: scene_index.dump_instance_index``'s ``VLA_INSTANCE_DUMP_PATH``, ``core.heads.factory.
#: dump_plan``'s ``VLA_PLAN_DUMP_PATH``).
ENV_PROMPT_DUMP_PATH: str = "VLA_PROMPT_DUMP_PATH"


def dump_prompt_diagnostics(
    *,
    tag: str,
    question_prompt: str,
    vocab_prompt: str,
    dropped_vocab_nouns: Sequence[str],
) -> None:
    """Append one JSONL record of the composed GDINO prompt(s) + what got dropped, if
    :data:`ENV_PROMPT_DUMP_PATH` is set. No-op (no I/O at all) when unset.

    Called once per :func:`refresh_prompt` invocation -- i.e. once at boot priming
    (``tag="boot_prime"``, empty question) and once per question the moment it latches
    (``tag="question_latch"``, see :meth:`core.heads.factory.HeadState.bind`) -- never
    per keyframe, so this is not a hot-path cost (:func:`refresh_prompt` itself already
    documents that same non-hot-path guarantee for :func:`_eviction_safe_vocab_order`).

    Record shape: ``wall_time``, ``tag``, ``question_prompt`` (the short question-noun-
    only caption, or ``""`` if the detector has no such tier), ``vocab_prompt`` (the
    full question+vocab caption actually assigned to ``detector.prompt``),
    ``dropped_vocab_nouns`` (the exact tail :func:`build_gdino_prompt` cut to stay
    within the token budget, in drop order -- empty when nothing was dropped) and
    ``n_dropped_vocab_nouns``. This is deliberately the whole answer to "was a given
    noun ever asked for": a noun in ``dropped_vocab_nouns`` was never queried (this
    issue's mechanism); a noun that appears in neither prompt string was never in the
    vocabulary to begin with; anything else was queried, so its absence from a
    detection log is a detector-recall or merged-label question (#91/#154), not this
    one.

    Any failure (bad path, unwritable dir, etc.) is swallowed -- diagnostics must never
    break the run they are observing, matching ``dump_instance_index``/``dump_plan``.
    """
    path = os.environ.get(ENV_PROMPT_DUMP_PATH)
    if not path:
        return
    try:
        record = {
            "wall_time": time.time(),
            "tag": tag,
            "question_prompt": question_prompt,
            "vocab_prompt": vocab_prompt,
            "dropped_vocab_nouns": list(dropped_vocab_nouns),
            "n_dropped_vocab_nouns": len(dropped_vocab_nouns),
        }
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "a") as fh:
            fh.write(json.dumps(record) + "\n")
    except Exception:  # noqa: BLE001 - diagnostics must never break the run
        pass


def refresh_prompt(
    detector: object | None,
    question_nouns: Sequence[str],
    vocab_nouns: Sequence[str],
    *,
    full_only_nouns: Sequence[str] = (),
) -> str | None:
    """Rebuild ``detector.prompt`` from the latched question's nouns + standing vocab.

    Issue #34: ``GroundingDinoDetector`` is constructed at boot with empty
    ``question_nouns``/``vocab_nouns`` (no question latched yet), so its ``prompt`` stays
    ``""`` and every ``__call__`` short-circuits to zero detections forever, even after a
    question latches and the target/anchor nouns become known. This is the shared refresh
    seam: called once the question is parsed (see
    :meth:`core.heads.factory.HeadState.bind`), it rebuilds and reassigns ``detector.prompt``
    in place via :func:`build_gdino_prompt` so the SAME live detector instance that
    :class:`~core.perception.tracker.PerceptionPipeline` calls every keyframe starts
    grounding on the very next call.

    Structural, not nominal: any object exposing a settable ``.prompt`` string attribute
    (:class:`GroundingDinoDetector`, :class:`FakeDetector`) is refreshed; ``None`` or a
    detector with no ``.prompt`` attribute (a plain function, a
    :class:`~core.perception.scripted.ScriptedPanoDetector`, which replays labels keyed by
    keyframe index and has no text-prompt concept) is left untouched — this is a no-op, not
    an error, so it is always safe to call unconditionally on every plan latch.

    Issue #42 dual-pass detection: if the detector also exposes a settable
    ``.question_prompt`` (both :class:`GroundingDinoDetector` and :class:`FakeDetector`
    do), that is refreshed too, from the question nouns alone (no vocab nouns, and no
    ``full_only_nouns`` either — see below) — the short caption the question-noun pass
    grounds on. A detector with ``.prompt`` but no ``.question_prompt`` still gets its
    full prompt refreshed as before.

    Issue #108: ``full_only_nouns`` is the third category ``_plan_nouns`` (issue #95's
    docstring) had nowhere correct to send — nouns that must reach the full
    question+vocab caption (breadth: the detector needs to be asked to look for them so
    downstream logic, e.g. avoidance-region scoring, has something to find) but must NOT
    inflate the short question-noun-only caption, whose lower box threshold
    (``ENV_GDINO_QUESTION_BOX_THRESHOLD``) is calibrated specifically on that caption
    staying tiny (a 117-phrase caption decodes zero ``teapot`` at any threshold; a
    2-phrase caption decodes it in 194/211 keyframes — detector.py's own probe). The
    canonical caller is ``plan.avoid`` anchors (via ``core.plan_walk.iter_avoid_anchors``):
    avoid anchors are breadth, not the target being grounded.

    Priority in the full caption is ``question_nouns`` > ``full_only_nouns`` >
    ``vocab_nouns`` — plan-relevant avoid nouns are placed ahead of the generic standing
    vocab so issue #105's token budget (the full caption sits at exactly its 256-token
    limit with zero headroom) drops standing-vocab nouns before it ever drops a
    plan-relevant one. Achieved by simply prepending ``full_only_nouns`` to the
    ``vocab_nouns`` list handed to :func:`build_gdino_prompt` for the full caption only
    — that function's own de-dup (question nouns win ties) and budget-in-order behaviour
    do the rest; nothing about :func:`build_gdino_prompt` itself needed to change.

    Issue #91: within the ``vocab_nouns`` tier itself, :func:`_eviction_safe_vocab_order`
    reorders it before it reaches :func:`build_gdino_prompt` — the standing vocab is a
    plain alphabetical list (``core.heads.factory._STANDING_VOCAB_NOUNS``), so a
    token-budget cut always amputates the SAME alphabetic span regardless of which
    scene or question is live. A narrow, evidence-named set of relational-disambiguator
    classes (:data:`core.perception.vocab.DISAMBIGUATOR_PRIORITY_NOUNS` — see that
    constant's docstring for why it is narrow and which classes it deliberately leaves
    out) is pulled toward the front of whatever's LEFT of the vocab tier after the
    token-budget cut it would otherwise take, so a degraded budget (e.g. the real
    tokenizer unavailable, falling back to :func:`_heuristic_token_estimate`'s far more
    pessimistic count) has a chance to cut a different span without ever cutting a span
    that used to survive. :func:`_eviction_safe_vocab_order` is a pure reorder — every
    vocab noun the caller passed is still eligible for the caption, exactly once;
    nothing is filtered or invented — but unlike a whole-list reorder (the original,
    reverted #91 approach — see that function's docstring for the measured regression),
    it never moves a noun that already survives the budget cut out of its surviving
    position, so promoting a disambiguator noun can only ever spend LEFTOVER headroom,
    never a survivor's own spot. Nouns that are the CURRENT question's own target/anchor
    already reach ``question_nouns`` (never dropped) via
    :func:`core.heads.explore_step._plan_nouns`; this only helps the standing-vocab
    tier, i.e. scene-index breadth for nouns the live question does not itself mention.

    Thread-safety: this performs exactly one attribute assignment (``detector.prompt =
    ...``), which is atomic under the GIL — safe to call from whichever thread latches the
    plan (the adapter's 5 Hz tick timer) even though a different thread's subscription
    callback may concurrently be feeding the same detector through
    ``PerceptionPipeline.process`` — the reader either sees the old prompt or the fully-built
    new one, never a partial write.

    Returns the new prompt string, or ``None`` if ``detector`` has no ``.prompt`` to refresh.
    """
    if detector is None or not hasattr(detector, "prompt"):
        return None
    dropped: list[str] = []
    prompt = build_gdino_prompt(
        question_nouns,
        list(full_only_nouns)
        + _eviction_safe_vocab_order(question_nouns, full_only_nouns, vocab_nouns),
        dropped_out=dropped,
    )
    detector.prompt = prompt
    question_prompt = ""
    if hasattr(detector, "question_prompt"):
        question_prompt = build_gdino_prompt(question_nouns, ())
        detector.question_prompt = question_prompt
    dump_prompt_diagnostics(
        tag="question_latch" if question_nouns else "boot_prime",
        question_prompt=question_prompt,
        vocab_prompt=prompt,
        dropped_vocab_nouns=dropped,
    )
    return prompt


#: ImageNet normalisation GroundingDINO's own preprocessing uses (torchvision convention).
_IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
_IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)


def _norm_cxcywh_to_tile_xyxy(
    cx: float, cy: float, w: float, h: float, tile_w: int, tile_h: int,
) -> tuple[float, float, float, float]:
    """GroundingDINO box ``(cx, cy, w, h)`` normalised to [0, 1] -> tile-pixel xyxy.

    GroundingDINO (and the underlying DETR-style decoder) reports boxes as fractions of
    the *input tensor's* height/width, regardless of any resize applied before inference
    — so converting back to the tile's native pixel space is just multiplying by the
    tile's own (unresized) width/height (:class:`~core.perception.tiling.TileSpec`'s
    ``width``/``height``), no inverse-resize math needed. Pure function (no torch), so
    this piece of the real-inference math is unit-testable without the model.

    The raw affine result is clamped to the tile bounds ``[0, tile_w] x [0, tile_h]``: a
    real GDINO box centred near an edge (or, degenerately, entirely outside ``[0, 1]``
    normalised space) otherwise yields negative or beyond-frame coordinates that
    downstream pixel indexing (mask crops, tile-array slicing) cannot safely use.
    Clamping each edge independently before ordering can, for a box that already
    straddles a bound asymmetrically, leave ``x1 < x0`` or ``y1 < y0``; the two are
    swapped back into order afterward so the result is always a well-formed (possibly
    zero-area, never inverted) box — consistent with how
    :mod:`core.perception.fusion` already tolerates a zero-area/degenerate bbox (its
    frustum padding, ``FusionConfig.angular_pad``, guarantees a non-empty angular gate
    regardless of bbox span, so no additional degenerate-box special-casing is needed
    downstream of this clamp).
    """
    x0 = (cx - w / 2.0) * tile_w
    y0 = (cy - h / 2.0) * tile_h
    x1 = (cx + w / 2.0) * tile_w
    y1 = (cy + h / 2.0) * tile_h
    x0 = min(max(x0, 0.0), float(tile_w))
    x1 = min(max(x1, 0.0), float(tile_w))
    y0 = min(max(y0, 0.0), float(tile_h))
    y1 = min(max(y1, 0.0), float(tile_h))
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    return float(x0), float(y0), float(x1), float(y1)


class GroundingDinoDetector:
    """Real open-vocab GroundingDINO-class detector (Gate 4).

    torch / groundingdino are imported lazily on first real ``__call__`` so that merely
    constructing or importing this class — or calling it with an empty prompt (no
    question latched yet) — stays dependency-free. Calling it with a non-empty prompt
    without those installs raises a clear :class:`ImportError` listing exactly what to
    pip install (see :data:`GDINO_REQUIRED_INSTALLS`); this exact error is depended on by
    ``tests/perception/test_detector.py``.

    The prompt is question nouns + vocab nouns joined via :func:`build_gdino_prompt`;
    the model is :data:`GDINO_MODEL_ID` (overridable via the ``GDINO_MODEL_ID`` env var
    or the ``model_id`` constructor arg — constructor > env > module default).

    Issue #42 dual-pass detection: every call also grounds a SHORT question-noun-only
    caption (``self.question_prompt``, built without vocab nouns) at its own, lower
    ``question_box_threshold`` — this is the pass that actually recalls small/rare
    targets like a teapot, which the full ~117-phrase question+vocab caption dilutes to
    zero detections at any threshold. The full caption keeps running too, but only every
    ``vocab_pass_cadence``-th tick (it feeds scene-index breadth — other-object/anchor
    instances — not target recall, so it doesn't need every-tick cadence). Per-tile
    detections from both passes that ran this tick are unioned, then deduplicated
    class-wise across ALL tiles by :func:`suppress_cross_tile_duplicates` (issue #131) —
    the tile seam overlap and the union of two passes both produce same-object repeats
    that used to reach fusion/association untouched. A tick where only the question pass
    runs costs one forward pass, not two.

    Deploy-time knobs (env var, all optional — see the ``ENV_GDINO_*`` constants above):
    ``GDINO_MODEL_ID``, ``GDINO_PRECISION`` (``fp16``/``fp32``/``auto``), ``GDINO_DEVICE``
    (``cuda``/``cpu``), ``GDINO_CONFIG_PATH``, ``GDINO_CHECKPOINT_PATH``. Per the
    offline-capable rule (docs/ubuntu_setup.md §7), the config + checkpoint are files
    baked into the Docker image at build time — this class never fetches anything over
    the network at inference time; a missing path raises a clear, actionable error.

    Precision defaults to half (fp16) on CUDA — the dev box is an 8 GB laptop GPU, the
    eval box targets 10-14 GB peak on an RTX 4090 (architecture §6) — and full (fp32) on
    CPU. Override with ``precision=`` / ``GDINO_PRECISION`` if a specific deploy needs
    otherwise (e.g. a variant that scores poorly in fp16).

    The 4 gnomonic tiles :func:`~core.perception.tiling.project_tiles` produces for one
    frame are batched into a single forward pass when the loaded model supports a
    multi-image batch with per-item captions (the common case for the raw
    ``groundingdino-py`` model object). If that batched path raises for any reason (a
    model/tokenizer variant that does not support it, mismatched tile shapes, etc.) this
    falls back to the public single-image ``groundingdino.util.inference.predict`` call
    once per tile — slower but always correct; a warning is logged once.
    """

    def __init__(
        self,
        question_nouns: Sequence[str] = (),
        vocab_nouns: Sequence[str] = (),
        *,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
        question_box_threshold: float | None = None,
        vocab_pass_cadence: int | None = None,
        model_id: str | None = None,
        precision: str | None = None,
        device: str | None = None,
        config_path: str | None = None,
        checkpoint_path: str | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        # Full question+vocab caption (the pre-#42 prompt; feeds scene-index breadth,
        # runs at ``vocab_pass_cadence``) and the short question-noun-only caption (#42
        # dual-pass; runs every tick, at its own lower box threshold for target recall).
        self.prompt = build_gdino_prompt(question_nouns, vocab_nouns)
        self.question_prompt = build_gdino_prompt(question_nouns, ())
        self.box_threshold = float(box_threshold)
        self.text_threshold = float(text_threshold)
        # constructor arg > env var > module default (same precedence as every other
        # ENV_GDINO_* knob below).
        self.question_box_threshold = float(
            question_box_threshold
            if question_box_threshold is not None
            else os.environ.get(
                ENV_GDINO_QUESTION_BOX_THRESHOLD, DEFAULT_GDINO_QUESTION_BOX_THRESHOLD
            )
        )
        self.vocab_pass_cadence = int(
            vocab_pass_cadence
            if vocab_pass_cadence is not None
            else os.environ.get(ENV_GDINO_VOCAB_PASS_CADENCE, DEFAULT_GDINO_VOCAB_PASS_CADENCE)
        )
        # Detection-tick counter driving the vocab-pass cadence (issue #42); incremented
        # once per non-empty-tiles __call__, regardless of cooldown/pass outcome.
        self._tick: int = 0
        # constructor arg > env var > module default (docstring-documented precedence).
        self.model_id = model_id or os.environ.get(ENV_GDINO_MODEL_ID) or GDINO_MODEL_ID
        self.precision = (
            precision or os.environ.get(ENV_GDINO_PRECISION) or DEFAULT_GDINO_PRECISION
        ).strip().lower()
        self._device_pref = (device or os.environ.get(ENV_GDINO_DEVICE) or "").strip().lower() or None
        self._config_path = config_path or os.environ.get(ENV_GDINO_CONFIG_PATH)
        self._checkpoint_path = checkpoint_path or os.environ.get(ENV_GDINO_CHECKPOINT_PATH)
        # Populated by _ensure_model() on first real (non-empty-prompt) call; cached
        # thereafter so weights are loaded exactly once per process.
        self._model = None
        self._resolved_device: str | None = None
        self._resolved_half: bool | None = None
        self._warned_batch_fallback = False
        # Load-failure backoff state (issue #39): incremented on every failed
        # _ensure_model attempt, reset on success. ``_clock`` is injectable (default
        # time.monotonic) so tests can drive the schedule with a fake clock.
        self._clock = clock
        self._consecutive_load_failures = 0
        self._next_retry_at: float = 0.0

    def _lazy_import(self):
        try:
            import torch  # noqa: F401
            from groundingdino.util.inference import load_model, predict  # noqa: F401
        except ImportError as exc:  # pragma: no cover - exercised via stub test
            installs = " ".join(GDINO_REQUIRED_INSTALLS)
            raise ImportError(
                "GroundingDinoDetector requires the Phase-2 model dependencies, which "
                "are not installed. Install them on the Ubuntu box with:\n"
                f"    pip install {installs}\n"
                "and download the GroundingDINO weights for model "
                f"'{self.model_id}'. See docs/ubuntu_setup.md section 7. The offline "
                "test path uses FakeDetector instead."
            ) from exc
        return torch, load_model, predict

    # ------------------------------------------------------------------ device/precision

    def _resolve_device(self, torch) -> str:
        if self._device_pref in ("cuda", "cpu"):
            return self._device_pref
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _resolve_half(self, device: str) -> bool:
        if self.precision == "fp16":
            return True
        if self.precision == "fp32":
            return False
        if self.precision != "auto":
            _LOGGER.warning(
                "GroundingDinoDetector: unrecognised precision %r (expected "
                "fp16|fp32|auto); treating as 'auto'.", self.precision,
            )
        return device == "cuda"  # auto: half on CUDA (VRAM headroom), full on CPU

    # ------------------------------------------------------------------ weight paths

    def _resolve_config_path(self) -> str:
        if self._config_path:
            return self._config_path
        # Fall back to the config shipped inside the groundingdino-py package itself
        # (its standard SwinT_OGC config) when no explicit path was baked/configured.
        try:
            import groundingdino

            pkg_dir = os.path.dirname(groundingdino.__file__)
            candidate = os.path.join(pkg_dir, "config", "GroundingDINO_SwinT_OGC.py")
            if os.path.exists(candidate):
                return candidate
        except Exception:  # pragma: no cover - defensive; falls through to the raise below
            pass
        raise RuntimeError(
            "GroundingDinoDetector needs a model config path. Set the config_path= "
            f"constructor arg or the {ENV_GDINO_CONFIG_PATH} env var to the .py config "
            "baked into the image alongside the checkpoint (docs/ubuntu_setup.md §7)."
        )

    def _resolve_checkpoint_path(self) -> str:
        if self._checkpoint_path:
            return self._checkpoint_path
        raise RuntimeError(
            "GroundingDinoDetector needs the pre-baked model weights. Set the "
            f"checkpoint_path= constructor arg or the {ENV_GDINO_CHECKPOINT_PATH} env "
            f"var to the .pth for model '{self.model_id}'. Per the offline-capable rule "
            "(docs/ubuntu_setup.md §7) weights are baked into the Docker image at build "
            "time and never fetched over the network at inference time."
        )

    def _ensure_model(self, torch, load_model_fn):
        if self._model is not None:
            return self._model
        device = self._resolve_device(torch)
        half = self._resolve_half(device)
        config_path = self._resolve_config_path()
        checkpoint_path = self._resolve_checkpoint_path()
        model = load_model_fn(config_path, checkpoint_path, device=device)
        # groundingdino-py's load_model(..., device=device) does NOT actually move the
        # returned model: it sets args.device and loads the checkpoint with
        # map_location="cpu", so parameters stay on CPU regardless of `device`. Move
        # explicitly before applying precision (see issue #41).
        model = model.to(device)
        model.eval()
        # fp16 is applied via torch.autocast at forward time (_forward_ctx), NOT by
        # hard-.half()ing the weights: GroundingDINO's internals mix float32 buffers
        # into the graph, and a halved model fails with "expected scalar type Float
        # but found Half" on both forward paths (issue #41).
        self._model = model
        self._resolved_device = device
        self._resolved_half = half
        _LOGGER.info(
            "GroundingDinoDetector: loaded %s on %s (%s precision).",
            self.model_id, device, "fp16-autocast" if half else "fp32",
        )
        return model

    # ------------------------------------------------------------------ load backoff

    def _in_cooldown(self) -> bool:
        return self._consecutive_load_failures > 0 and self._clock() < self._next_retry_at

    def _record_load_failure(self, exc: Exception) -> None:
        """Record one failed load attempt and schedule the next retry (issue #39).

        Delay doubles each consecutive failure (``GDINO_BACKOFF_BASE_S`` * 2**(n-1)) until
        ``GDINO_BACKOFF_DEGRADE_N`` consecutive failures, at which point the detector is
        considered degraded and every subsequent retry waits the full
        ``GDINO_BACKOFF_CAP_S`` — logged once as a single ERROR transition, not re-logged
        on every attempt thereafter (this fires at most once per 5 Hz tick per failure, not
        per tick: cooldown short-circuits every call in between).
        """
        self._consecutive_load_failures += 1
        n = self._consecutive_load_failures
        if n >= GDINO_BACKOFF_DEGRADE_N:
            delay = GDINO_BACKOFF_CAP_S
        else:
            delay = min(GDINO_BACKOFF_BASE_S * (2 ** (n - 1)), GDINO_BACKOFF_CAP_S)
        self._next_retry_at = self._clock() + delay
        if n == GDINO_BACKOFF_DEGRADE_N:
            _LOGGER.error(
                "GroundingDinoDetector: model load failed %d consecutive times (%s); "
                "degraded to long-cooldown retries (%.0fs).",
                n, exc, GDINO_BACKOFF_CAP_S,
            )
        elif n < GDINO_BACKOFF_DEGRADE_N:
            _LOGGER.warning(
                "GroundingDinoDetector: model load failed (%s); retrying in %.0fs.",
                exc, delay,
            )
        # n > GDINO_BACKOFF_DEGRADE_N: already-degraded transition was logged once above;
        # stay silent on further attempts so a stuck GPU doesn't spam the log forever.

    def _reset_load_failures(self) -> None:
        self._consecutive_load_failures = 0
        self._next_retry_at = 0.0

    def _forward_ctx(self, torch):
        """Mixed-precision context for forward passes: fp16 autocast on CUDA when the
        resolved precision is half, no-op otherwise (weights stay fp32 — see #41)."""
        if self._resolved_half and self._resolved_device == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        return contextlib.nullcontext()

    # ------------------------------------------------------------------ preprocessing

    def _to_tensor(self, tile: np.ndarray, torch, device: str, dtype):
        """Tile (H, W, 3) uint8 RGB -> normalised (3, H, W) tensor, no resize (see
        :func:`_norm_cxcywh_to_tile_xyxy`: boxes are decoded back at the tile's native
        pixel size, so we never need to invert a resize)."""
        arr = np.asarray(tile)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        arr = np.ascontiguousarray(arr[..., :3])
        t = torch.from_numpy(arr).to(device=device, dtype=torch.float32)
        t = t.permute(2, 0, 1) / 255.0
        mean = torch.tensor(_IMAGENET_MEAN, device=device, dtype=torch.float32).view(3, 1, 1)
        std = torch.tensor(_IMAGENET_STD, device=device, dtype=torch.float32).view(3, 1, 1)
        t = (t - mean) / std
        return t.to(dtype=dtype)

    # ------------------------------------------------------------------ inference

    def __call__(self, tiles: Sequence[np.ndarray]) -> list[list[Detection]]:
        if not tiles:
            return []
        # Issue #42: decide which caption pass(es) fire on this tick BEFORE the lazy
        # import, same short-circuit shape as the old single-prompt empty check — a
        # detector with no question latched yet (both prompts empty) still needs torch
        # present for nothing.
        tick = self._tick
        self._tick += 1
        run_question = bool(self.question_prompt)
        cadence = max(self.vocab_pass_cadence, 1)
        run_vocab = bool(self.prompt) and tick % cadence == 0
        if not run_question and not run_vocab:
            return [[] for _ in tiles]
        if self._in_cooldown():
            # Issue #39: a load that already failed is backing off — no-op cleanly
            # (mirrors the empty-prompt short-circuit above) instead of re-running the
            # full import/load cascade every tick until the cooldown elapses.
            return [[] for _ in tiles]
        # Lazy-import gate: raises the clear install error when torch/groundingdino are
        # absent (a deploy misconfiguration, not the transient/repeated failure this
        # backoff targets) — exercised with a non-empty prompt above so this always
        # actually runs inference when there is something to look for.
        torch, load_model_fn, predict_fn = self._lazy_import()
        try:
            model = self._ensure_model(torch, load_model_fn)
        except Exception as exc:  # noqa: BLE001 - any model-load failure backs off (issue #39)
            self._record_load_failure(exc)
            return [[] for _ in tiles]
        self._reset_load_failures()
        device = self._resolved_device
        # Inputs stay fp32; _forward_ctx's autocast downcasts per-op where safe (#41).
        dtype = torch.float32
        merged: list[list[Detection]] = [[] for _ in tiles]
        # Question-noun pass first (every tick, target recall) — a tick where the vocab
        # pass is not due costs exactly this one forward, not two.
        if run_question:
            per_tile = self._dispatch_pass(
                torch, model, predict_fn, tiles, device, dtype,
                self.question_prompt, self.question_box_threshold,
            )
            for i, dets in enumerate(per_tile):
                merged[i].extend(dets)
        # Full question+vocab pass (only every ``vocab_pass_cadence``-th tick) — union
        # its detections into the same per-tile lists.
        if run_vocab:
            per_tile = self._dispatch_pass(
                torch, model, predict_fn, tiles, device, dtype,
                self.prompt, self.box_threshold,
            )
            for i, dets in enumerate(per_tile):
                merged[i].extend(dets)
        # Issue #131: dedupe the union in a shared angular frame before returning it —
        # see :func:`suppress_cross_tile_duplicates`'s module docstring for the geometry
        # behind the threshold and why the old "no dedupe, fusion tolerates it" design
        # note above no longer holds.
        return suppress_cross_tile_duplicates(merged, n_tiles=len(tiles))

    def run_caption_pass(
        self,
        tiles: Sequence[np.ndarray],
        caption: str,
        box_threshold: float,
        text_threshold: float | None = None,
    ) -> list[list[Detection]]:
        """Run exactly one caption pass over ``tiles`` and return the per-tile detections.

        This is the entry point ``tools/cluster/gdino_server.py`` (the remote GDINO offload
        server, issue #86) calls: it owns the loaded model on the cluster GPU and serves one
        caption pass per request, decoupled from this detector's own tick-driven dual-pass
        scheduling in :meth:`__call__` (question pass + cadenced vocab pass, backoff, tick
        counter) — none of that applies here.

        Empty ``tiles`` or an empty/whitespace-only ``caption`` short-circuits to
        ``[[] for _ in tiles]`` with no import/load, mirroring :meth:`__call__`'s own
        empty-prompt short circuit. Otherwise this lazily imports torch/groundingdino,
        ensures the model is loaded, and delegates to :meth:`_dispatch_pass` for the actual
        forward pass. Unlike :meth:`__call__`, a load failure here RAISES rather than
        recording a backoff failure: the server wants a loud, immediate error on a bad
        request, not the tick-driven cooldown behaviour of issue #39 (and the backoff
        counters are left untouched by this method either way).

        ``text_threshold``, if given, temporarily overrides ``self.text_threshold`` for the
        duration of this one pass (restored in a ``finally``, including on exception) — the
        server's request may specify its own text threshold per call.
        """
        if not tiles or not caption or not caption.strip():
            return [[] for _ in tiles]
        torch, load_model_fn, predict_fn = self._lazy_import()
        model = self._ensure_model(torch, load_model_fn)
        saved_text_threshold = self.text_threshold
        if text_threshold is not None:
            self.text_threshold = float(text_threshold)
        try:
            return self._dispatch_pass(
                torch, model, predict_fn, tiles, self._resolved_device, torch.float32,
                caption, float(box_threshold),
            )
        finally:
            self.text_threshold = saved_text_threshold

    def _dispatch_pass(
        self, torch, model, predict_fn, tiles: Sequence[np.ndarray], device, dtype,
        prompt: str, box_threshold: float,
    ) -> list[list[Detection]]:
        """Run one caption pass (batched, falling back to per-tile) through the shared
        ``_call_batched``/``_call_per_tile`` dispatch, which both read ``self.prompt``/
        ``self.box_threshold`` — temporarily swapped to this pass's values so those two
        methods (and their exact call signature, depended on by the issue #39 backoff
        tests' ``det._call_batched = lambda torch, model, tiles, device, dtype: ...``
        monkeypatch) stay untouched by the dual-pass change."""
        saved_prompt, saved_threshold = self.prompt, self.box_threshold
        self.prompt, self.box_threshold = prompt, box_threshold
        try:
            try:
                return self._call_batched(torch, model, tiles, device, dtype)
            except Exception as exc:
                if not self._warned_batch_fallback:
                    _LOGGER.warning(
                        "GroundingDinoDetector: batched tile forward failed (%s); falling "
                        "back to one predict() call per tile (slower, always correct).", exc,
                    )
                    self._warned_batch_fallback = True
                return self._call_per_tile(torch, model, predict_fn, tiles, device, dtype)
        finally:
            self.prompt, self.box_threshold = saved_prompt, saved_threshold

    def _decode_batch_item(
        self, tile_id: int, tile: np.ndarray, item_logits, item_boxes,
    ) -> list[Detection]:
        """Decode one batch item's (nq, ntok) logits + (nq, 4) boxes into Detections."""
        from groundingdino.util.utils import get_phrases_from_posmap

        tile_h, tile_w = tile.shape[0], tile.shape[1]
        model = self._model
        tokenizer = model.tokenizer
        tokenized = tokenizer(self.prompt)
        max_logits, _ = item_logits.max(dim=1)
        keep = (max_logits > self.box_threshold).nonzero().flatten().tolist()
        dets: list[Detection] = []
        for i in keep:
            posmap = item_logits[i] > self.text_threshold
            phrase = get_phrases_from_posmap(posmap, tokenized, tokenizer).replace(".", "").strip()
            cx, cy, bw, bh = (float(v) for v in item_boxes[i])
            bbox = _norm_cxcywh_to_tile_xyxy(cx, cy, bw, bh, tile_w, tile_h)
            dets.append(
                Detection(
                    tile_id=tile_id, bbox_xyxy=bbox, label=phrase or self.prompt,
                    score=float(max_logits[i]),
                )
            )
        return dets

    def _call_batched(self, torch, model, tiles: Sequence[np.ndarray], device, dtype) -> list[list[Detection]]:
        """Single forward pass over all tiles (batched) — the fast path.

        Requires every tile the same pixel size (true within one frame: all 4 gnomonic
        tiles share ``project_tiles``' output dims) and the loaded model to accept a
        batch tensor + one caption per item, matching the raw ``groundingdino-py`` model
        object's ``forward(samples, captions=[...])`` signature.
        """
        tensors = [self._to_tensor(t, torch, device, dtype) for t in tiles]
        shapes = {tuple(t.shape) for t in tensors}
        if len(shapes) != 1:
            raise ValueError("tiles have mismatched shapes; batching requires a uniform tile size")
        batch = torch.stack(tensors, dim=0)
        captions = [self.prompt] * len(tiles)
        with torch.no_grad(), self._forward_ctx(torch):
            outputs = model(batch, captions=captions)
        logits = outputs["pred_logits"].sigmoid()  # (B, nq, ntok)
        boxes = outputs["pred_boxes"]              # (B, nq, 4) cxcywh in [0, 1]
        per_tile: list[list[Detection]] = []
        for b, tile in enumerate(tiles):
            per_tile.append(self._decode_batch_item(b, tile, logits[b], boxes[b]))
        return per_tile

    def _call_per_tile(
        self, torch, model, predict_fn, tiles: Sequence[np.ndarray], device, dtype,
    ) -> list[list[Detection]]:
        """Fallback: one ``groundingdino.util.inference.predict`` call per tile."""
        per_tile: list[list[Detection]] = []
        for b, tile in enumerate(tiles):
            tile_h, tile_w = tile.shape[0], tile.shape[1]
            image = self._to_tensor(tile, torch, device, dtype)
            with self._forward_ctx(torch):
                boxes, scores, phrases = predict_fn(
                    model=model, image=image, caption=self.prompt,
                    box_threshold=self.box_threshold, text_threshold=self.text_threshold,
                    device=device,
                )
            dets: list[Detection] = []
            for (cx, cy, bw, bh), score, phrase in zip(
                boxes.tolist(), scores.tolist(), phrases,
            ):
                bbox = _norm_cxcywh_to_tile_xyxy(cx, cy, bw, bh, tile_w, tile_h)
                dets.append(
                    Detection(
                        tile_id=b, bbox_xyxy=bbox, label=phrase or self.prompt,
                        score=float(score),
                    )
                )
            per_tile.append(dets)
        return per_tile
