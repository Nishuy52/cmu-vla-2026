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
import logging
import os
from dataclasses import dataclass, field
from typing import Callable, Protocol, Sequence

import numpy as np

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


def build_gdino_prompt(question_nouns: Sequence[str], vocab_nouns: Sequence[str]) -> str:
    """Build the GroundingDINO text prompt from question + vocab nouns.

    GroundingDINO expects a lowercase, ``.``-separated list of noun phrases with a
    trailing separator, e.g. ``"sofa . window . potted plant ."``. Question nouns
    come first (query-relevant recall priority), then any extra vocab nouns, both
    de-duplicated preserving order.
    """
    seen: set[str] = set()
    ordered: list[str] = []
    for noun in list(question_nouns) + list(vocab_nouns):
        n = noun.strip().lower()
        if n and n not in seen:
            seen.add(n)
            ordered.append(n)
    if not ordered:
        return ""
    return " . ".join(ordered) + " ."


def refresh_prompt(
    detector: object | None,
    question_nouns: Sequence[str],
    vocab_nouns: Sequence[str],
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
    prompt = build_gdino_prompt(question_nouns, vocab_nouns)
    detector.prompt = prompt
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
        model_id: str | None = None,
        precision: str | None = None,
        device: str | None = None,
        config_path: str | None = None,
        checkpoint_path: str | None = None,
    ) -> None:
        self.prompt = build_gdino_prompt(question_nouns, vocab_nouns)
        self.box_threshold = float(box_threshold)
        self.text_threshold = float(text_threshold)
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
        if not self.prompt:
            # No question nouns yet (e.g. freshly booted, question not latched) — nothing
            # to ground. Short-circuits BEFORE the lazy import, so a detector constructed
            # before the first question never needs torch present to no-op harmlessly.
            return [[] for _ in tiles]
        # Lazy-import gate: raises the clear install error when torch/groundingdino are
        # absent. Exercised with a non-empty prompt above so this always actually runs
        # inference when there is something to look for.
        torch, load_model_fn, predict_fn = self._lazy_import()
        model = self._ensure_model(torch, load_model_fn)
        device = self._resolved_device
        # Inputs stay fp32; _forward_ctx's autocast downcasts per-op where safe (#41).
        dtype = torch.float32
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
