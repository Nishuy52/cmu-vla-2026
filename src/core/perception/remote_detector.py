"""Dev-only remote GDINO offload seam: keeps the ROS stack local, ships tiles to a cluster
GPU for inference over HTTP instead of loading GroundingDINO on this box.

Issue #86: the laptop GPU has an EC power-wedge that fires at GroundingDINO model load,
killing live sim runs. :class:`RemoteDetector` is an opt-in, dev-only alternative to
:class:`~core.perception.detector.GroundingDinoDetector` — construct it in place of the
real detector and every tile still gets grounded, but the forward pass runs on a remote
GPU reached through an SSH tunnel (``http://127.0.0.1:8765`` by default) instead of on this
machine. It is NEVER the submission path: the scored image never talks to a network
detector, and this class exists purely to keep dev iteration moving on a wedge-prone box.

Wire contract (the other end is ``tools/cluster/gdino_server.py``, which wraps
:meth:`core.perception.detector.GroundingDinoDetector.run_caption_pass`): one HTTP POST per
caption pass, JSON body::

    {"tiles": [<base64 jpeg>, ...], "caption": <str>, "box_threshold": <float>,
     "text_threshold": <float>}

to ``<url>/detect``, JSON response::

    {"per_tile": [[{"bbox_xyxy": [x0, y0, x1, y1], "label": <str>, "score": <float>}, ...],
                   ...]}

with ``bbox_xyxy`` in tile pixels and ``len(per_tile) == len(tiles)``.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import time
import urllib.request
from typing import Callable, Sequence

import numpy as np

from .detector import (
    Detection,
    DEFAULT_GDINO_QUESTION_BOX_THRESHOLD,
    DEFAULT_GDINO_VOCAB_PASS_CADENCE,
    ENV_GDINO_QUESTION_BOX_THRESHOLD,
    ENV_GDINO_VOCAB_PASS_CADENCE,
    GDINO_BACKOFF_BASE_S,
    GDINO_BACKOFF_CAP_S,
    GDINO_BACKOFF_DEGRADE_N,
    suppress_cross_tile_duplicates,
)

_LOGGER = logging.getLogger(__name__)

#: Cluster GDINO server URL (dev-only offload, issue #86); e.g. via an SSH tunnel to a
#: cluster GPU node. No module default — construction fails loudly if unset (see __init__).
ENV_REMOTE_DETECTOR_URL = "VLA_REMOTE_DETECTOR_URL"

#: Per-request HTTP timeout in seconds.
ENV_REMOTE_DETECTOR_TIMEOUT_S = "VLA_REMOTE_DETECTOR_TIMEOUT_S"
DEFAULT_REMOTE_DETECTOR_TIMEOUT_S: float = 10.0


class RemoteDetector:
    """Dev-only GDINO offload over HTTP — same :class:`~core.perception.detector.
    DetectorProtocol` seam as :class:`~core.perception.detector.GroundingDinoDetector`, so
    it drops straight into :class:`~core.perception.tracker.PerceptionPipeline` and
    :func:`~core.perception.detector.refresh_prompt` unchanged.

    Mirrors ``GroundingDinoDetector.__call__``'s dual-pass tick scheduling exactly (question
    pass every tick + a cadenced vocab pass) so behaviour is otherwise identical to the local
    detector — only the forward pass itself moves off-box. Never the submission path: this
    is opt-in via ``VLA_DETECTOR=remote`` + ``VLA_REMOTE_DETECTOR_URL`` for local dev only.
    """

    def __init__(
        self,
        url: str | None = None,
        *,
        timeout_s: float | None = None,
        box_threshold: float = 0.35,
        text_threshold: float = 0.25,
        question_box_threshold: float | None = None,
        vocab_pass_cadence: int | None = None,
        encode_fn: Callable[[np.ndarray], bytes] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.url = url or os.environ.get(ENV_REMOTE_DETECTOR_URL)
        if not self.url:
            raise RuntimeError(
                "RemoteDetector needs a server URL. Set the url= constructor arg or the "
                f"{ENV_REMOTE_DETECTOR_URL} env var to the tunnelled gdino_server.py "
                "address (e.g. http://127.0.0.1:8765) — this is a dev-only offload mode "
                "(issue #86) and a missing URL must fail loudly at construction, not "
                "silently degrade like the submission-path detector."
            )
        self.timeout_s = float(
            timeout_s
            if timeout_s is not None
            else os.environ.get(ENV_REMOTE_DETECTOR_TIMEOUT_S, DEFAULT_REMOTE_DETECTOR_TIMEOUT_S)
        )
        self.box_threshold = float(box_threshold)
        self.text_threshold = float(text_threshold)
        # constructor arg > env var > module default (same precedence as GroundingDinoDetector).
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
        # Settable str attrs so core.perception.detector.refresh_prompt works unchanged.
        self.prompt: str = ""
        self.question_prompt: str = ""
        if encode_fn is None:
            from .vision_encode import jpeg_encode_fn

            encode_fn = jpeg_encode_fn
        self._encode_fn = encode_fn
        self._tick: int = 0
        # Backoff state (mirrors issue #39's GroundingDinoDetector._record_load_failure).
        self._clock = clock
        self._consecutive_failures = 0
        self._next_retry_at: float = 0.0

    # ------------------------------------------------------------------ backoff

    def _in_cooldown(self) -> bool:
        return self._consecutive_failures > 0 and self._clock() < self._next_retry_at

    def _record_failure(self, exc: Exception) -> None:
        self._consecutive_failures += 1
        n = self._consecutive_failures
        if n >= GDINO_BACKOFF_DEGRADE_N:
            delay = GDINO_BACKOFF_CAP_S
        else:
            delay = min(GDINO_BACKOFF_BASE_S * (2 ** (n - 1)), GDINO_BACKOFF_CAP_S)
        self._next_retry_at = self._clock() + delay
        if n == GDINO_BACKOFF_DEGRADE_N:
            _LOGGER.error(
                "RemoteDetector: request to %s failed %d consecutive times (%s); degraded "
                "to long-cooldown retries (%.0fs).",
                self.url, n, exc, GDINO_BACKOFF_CAP_S,
            )
        elif n < GDINO_BACKOFF_DEGRADE_N:
            _LOGGER.warning(
                "RemoteDetector: request to %s failed (%s); retrying in %.0fs.",
                self.url, exc, delay,
            )
        # n > GDINO_BACKOFF_DEGRADE_N: already-degraded transition was logged once above;
        # stay silent on further attempts so a stuck server doesn't spam the log forever.

    def _reset_failures(self) -> None:
        self._consecutive_failures = 0
        self._next_retry_at = 0.0

    # ------------------------------------------------------------------ transport

    def _post(self, tiles_b64: list[str], caption: str, box_threshold: float) -> list[list[Detection]]:
        body = json.dumps(
            {
                "tiles": tiles_b64,
                "caption": caption,
                "box_threshold": box_threshold,
                "text_threshold": self.text_threshold,
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            self.url.rstrip("/") + "/detect",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            payload = json.loads(resp.read())
        per_tile_raw = payload["per_tile"]
        if len(per_tile_raw) != len(tiles_b64):
            raise ValueError(
                f"RemoteDetector: server returned {len(per_tile_raw)} per-tile results for "
                f"{len(tiles_b64)} tiles."
            )
        per_tile: list[list[Detection]] = []
        for tile_id, dets_raw in enumerate(per_tile_raw):
            dets: list[Detection] = []
            for d in dets_raw:
                x0, y0, x1, y1 = d["bbox_xyxy"]
                dets.append(
                    Detection(
                        tile_id=tile_id,
                        bbox_xyxy=(float(x0), float(y0), float(x1), float(y1)),
                        label=str(d["label"]),
                        score=float(d["score"]),
                    )
                )
            per_tile.append(dets)
        return per_tile

    # ------------------------------------------------------------------ inference

    def __call__(self, tiles: Sequence[np.ndarray]) -> list[list[Detection]]:
        if not tiles:
            return []
        # Mirrors GroundingDinoDetector.__call__'s tick scheduling exactly (issue #42).
        tick = self._tick
        self._tick += 1
        run_question = bool(self.question_prompt)
        cadence = max(self.vocab_pass_cadence, 1)
        run_vocab = bool(self.prompt) and tick % cadence == 0
        if not run_question and not run_vocab:
            return [[] for _ in tiles]
        if self._in_cooldown():
            return [[] for _ in tiles]
        try:
            tiles_b64 = [
                base64.b64encode(self._encode_fn(t)).decode("ascii") for t in tiles
            ]
            merged: list[list[Detection]] = [[] for _ in tiles]
            if run_question:
                per_tile = self._post(tiles_b64, self.question_prompt, self.question_box_threshold)
                for i, dets in enumerate(per_tile):
                    merged[i].extend(dets)
            if run_vocab:
                per_tile = self._post(tiles_b64, self.prompt, self.box_threshold)
                for i, dets in enumerate(per_tile):
                    merged[i].extend(dets)
        except Exception as exc:  # noqa: BLE001 - any failure anywhere backs off, never raises
            self._record_failure(exc)
            return [[] for _ in tiles]
        self._reset_failures()
        # Issue #131 (ported for path consistency, #134): dedupe the question+vocab union
        # in a shared angular frame before returning it, exactly as
        # GroundingDinoDetector.__call__ does — same helper, same threshold source, same
        # n_tiles=len(tiles) semantics (this path has no separate tile-count source; the
        # request tiles list IS the tile count here too).
        return suppress_cross_tile_duplicates(merged, n_tiles=len(tiles))


__all__ = [
    "ENV_REMOTE_DETECTOR_URL",
    "ENV_REMOTE_DETECTOR_TIMEOUT_S",
    "DEFAULT_REMOTE_DETECTOR_TIMEOUT_S",
    "RemoteDetector",
]
