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

from dataclasses import dataclass, field
from typing import Callable, Protocol, Sequence

import numpy as np


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


class GroundingDinoDetector:
    """STUB open-vocab detector — real weights land in Phase 2.

    torch / groundingdino are imported lazily on first ``__call__`` so that merely
    constructing or importing this class stays dependency-free. Calling it without
    those installs raises a clear :class:`ImportError` listing exactly what to pip
    install (see :data:`GDINO_REQUIRED_INSTALLS`).

    The intended prompt is question nouns + vocab nouns joined via
    :func:`build_gdino_prompt`; the model is :data:`GDINO_MODEL_ID`.
    """

    def __init__(
        self,
        question_nouns: Sequence[str] = (),
        vocab_nouns: Sequence[str] = (),
        *,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
        model_id: str = GDINO_MODEL_ID,
    ) -> None:
        self.prompt = build_gdino_prompt(question_nouns, vocab_nouns)
        self.box_threshold = float(box_threshold)
        self.text_threshold = float(text_threshold)
        self.model_id = model_id
        self._model = None

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

    def __call__(self, tiles: Sequence[np.ndarray]) -> list[list[Detection]]:
        # Lazy-import gate: raises the clear install error when torch is absent.
        self._lazy_import()
        raise NotImplementedError(  # pragma: no cover - real inference is Phase 2
            "GroundingDinoDetector inference is implemented in Phase 2 once weights "
            "are baked into the Docker image (docs/ubuntu_setup.md section 7)."
        )
