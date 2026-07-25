"""Dev-only GroundingDINO offload server (issue #86) — NEVER the submission path.

Runs on a cluster GPU node (see ``tools/cluster/servers.sbatch``) and serves single
GroundingDINO caption passes over HTTP so a laptop with a GPU power-wedge (#86) can keep
iterating without loading the model locally. The scored pipeline never talks to this
server; it is reached only through an SSH tunnel a developer starts by hand
(``tools/cluster/tunnel.sh``) and ``core.perception.remote_detector.RemoteDetector``, which
is itself opt-in via ``VLA_DETECTOR=remote``.

Wire contract (mirrored by ``RemoteDetector``):

    GET  /health -> 200 {"status": "ok", "model": <str>, "device": <str>,
                          "precision": <str>, "uptime_s": <float>}
    POST /detect body {"tiles": [<base64 jpeg>, ...], "caption": <str>,
                        "box_threshold": <float>, "text_threshold": <float, optional>}
         -> 200 {"per_tile": [[{"bbox_xyxy": [x0,y0,x1,y1], "label": <str>,
                                 "score": <float>}, ...], ...]}
         400 {"error": <str>} on a malformed request, 500 {"error": <str>} on an
         inference failure. Unknown paths -> 404.

Exactly ONE caption pass runs per ``/detect`` request (one call to
``GroundingDinoDetector.run_caption_pass``): the client (``RemoteDetector``) owns
dual-pass (question-noun + cadenced vocab) scheduling, this server is a pure single-pass
primitive with no scheduling state of its own.

All torch/numpy/PIL/``core.perception`` imports are LAZY (deferred into ``main``/the
request-handling paths) so ``python tools/cluster/gdino_server.py --help`` runs on a box
with none of those installed. The JSON-validation / base64 / response-shaping helpers
below are pure stdlib and importable + unit-testable without torch.
"""
from __future__ import annotations

import argparse
import base64
import binascii
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

#: One black RGB tile used to force the model to actually load at startup (a load
#: failure crashes the process loudly here, not on the first real request). The tile must
#: be realistically sized: GroundingDINO's decoder runs ``torch.topk(..., 900)`` over the
#: flattened feature proposals, and a tiny image (e.g. 32x32) yields fewer than 900
#: proposals -> "RuntimeError: selected index k out of range" during warmup. 480x640
#: matches the gnomonic tile size the client actually sends.
_WARMUP_TILE_H = 480
_WARMUP_TILE_W = 640
_WARMUP_CAPTION = "warmup ."


class BadRequest(ValueError):
    """A malformed ``/detect`` payload — maps to an HTTP 400."""


# --------------------------------------------------------------------- pure helpers
# No torch/numpy/PIL here: these are exercised directly by tools/tests without either
# installed.


def validate_detect_payload(payload: object) -> dict:
    """Check a decoded JSON ``/detect`` body has the required shape.

    Returns ``payload`` unchanged on success (for chaining); raises :class:`BadRequest`
    with a human-readable message on anything malformed: not an object, a missing
    required field, or ``tiles`` not a list.
    """
    if not isinstance(payload, dict):
        raise BadRequest("request body must be a JSON object")
    required = ("tiles", "caption", "box_threshold")
    missing = [f for f in required if f not in payload]
    if missing:
        raise BadRequest(f"missing field(s): {', '.join(missing)}")
    if not isinstance(payload["tiles"], list):
        raise BadRequest("'tiles' must be a list of base64-encoded JPEG strings")
    if not isinstance(payload["caption"], str):
        raise BadRequest("'caption' must be a string")
    return payload


def decode_b64_tiles(tiles_b64: list) -> list[bytes]:
    """Decode each entry of ``tiles_b64`` (base64-encoded JPEG strings) to raw bytes.

    Raises :class:`BadRequest` naming the offending index on a non-string entry or
    invalid base64 — never lets a single bad tile crash the request handler with an
    opaque traceback.
    """
    out: list[bytes] = []
    for i, t in enumerate(tiles_b64):
        if not isinstance(t, str):
            raise BadRequest(f"tiles[{i}] must be a base64-encoded string")
        try:
            out.append(base64.b64decode(t, validate=True))
        except (binascii.Error, ValueError) as exc:
            raise BadRequest(f"tiles[{i}] is not valid base64: {exc}") from exc
    return out


def response_body(per_tile) -> dict:
    """Shape a ``list[list[Detection]]`` (as returned by ``run_caption_pass``) into the
    exact wire JSON the client (``RemoteDetector``) expects: tile pixel coords straight
    from ``Detection.bbox_xyxy``, no rescaling."""
    return {
        "per_tile": [
            [
                {
                    "bbox_xyxy": [float(v) for v in det.bbox_xyxy],
                    "label": det.label,
                    "score": float(det.score),
                }
                for det in tile_dets
            ]
            for tile_dets in per_tile
        ]
    }


def error_body(message: object) -> dict:
    """Shape an error message into the ``{"error": ...}`` wire body."""
    return {"error": str(message)}


def decode_jpeg_tile(data: bytes):
    """Base64-decoded JPEG bytes -> (H, W, 3) uint8 RGB numpy array (lazy PIL import)."""
    import io

    import numpy as np
    from PIL import Image

    with Image.open(io.BytesIO(data)) as img:
        return np.asarray(img.convert("RGB"))


# --------------------------------------------------------------------- HTTP handler


def _make_handler_class(detector, lock: threading.Lock, model_id: str, device: str,
                         precision: str, start_time: float):
    """Build a ``BaseHTTPRequestHandler`` subclass closed over the loaded ``detector`` and
    server metadata (stdlib's handler classes take no constructor args of their own, so
    the closure is how the handler reaches the one already-loaded model)."""

    class GdinoHandler(BaseHTTPRequestHandler):
        server_version = "gdino-offload/1"

        def _send_json(self, status: int, payload: dict) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802 - stdlib handler naming
            if self.path != "/health":
                self._send_json(404, error_body(f"unknown path: {self.path}"))
                return
            self._send_json(200, {
                "status": "ok",
                "model": model_id,
                "device": device,
                "precision": precision,
                "uptime_s": time.monotonic() - start_time,
            })

        def do_POST(self) -> None:  # noqa: N802 - stdlib handler naming
            if self.path != "/detect":
                self._send_json(404, error_body(f"unknown path: {self.path}"))
                return
            try:
                length = int(self.headers.get("Content-Length", 0) or 0)
                raw = self.rfile.read(length) if length > 0 else b""
                payload = validate_detect_payload(json.loads(raw or b"{}"))
                tile_bytes = decode_b64_tiles(payload["tiles"])
                tiles = [decode_jpeg_tile(b) for b in tile_bytes]
                caption = payload["caption"]
                box_threshold = float(payload["box_threshold"])
                raw_text_threshold = payload.get("text_threshold")
                text_threshold = (
                    float(raw_text_threshold) if raw_text_threshold is not None else None
                )
            except BadRequest as exc:
                self._send_json(400, error_body(exc))
                return
            except Exception as exc:  # noqa: BLE001 - any other malformed body -> 400
                self._send_json(400, error_body(exc))
                return

            try:
                t0 = time.monotonic()
                with lock:  # one global lock: GPU inference is serialised
                    per_tile = detector.run_caption_pass(
                        tiles, caption, box_threshold, text_threshold
                    )
                dt = time.monotonic() - t0
                print(
                    f"[gdino_server] /detect tiles={len(tiles)} caption={caption!r} "
                    f"box_threshold={box_threshold} took={dt:.3f}s",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001 - inference failure -> 500, never crash
                print(f"[gdino_server] /detect FAILED: {exc}", flush=True)
                self._send_json(500, error_body(exc))
                return
            self._send_json(200, response_body(per_tile))

        def log_message(self, fmt: str, *args) -> None:  # noqa: N802 - stdlib override
            print(f"[gdino_server] {self.address_string()} - {fmt % args}", flush=True)

    return GdinoHandler


# --------------------------------------------------------------------- CLI / serve


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Dev-only GroundingDINO offload HTTP server (issue #86). Never the "
            "submission path — start on demand via tools/cluster/servers.sh, tear down "
            "with the same script when done."
        )
    )
    parser.add_argument("--port", type=int, default=8765, help="listen port (default: 8765)")
    parser.add_argument("--host", default="0.0.0.0", help="listen host (default: 0.0.0.0)")
    parser.add_argument(
        "--config", default=None,
        help="GroundingDINO model config .py path (default: GDINO_CONFIG_PATH env var)",
    )
    parser.add_argument(
        "--checkpoint", default=None,
        help="GroundingDINO checkpoint .pth path (default: GDINO_CHECKPOINT_PATH env var)",
    )
    parser.add_argument(
        "--precision", default=os.environ.get("GDINO_PRECISION", "fp32"),
        help="fp16|fp32|auto (default: GDINO_PRECISION env var, else fp32 — the fp32 "
             "default matches the 24GB cluster card, unlike the 8GB-laptop fp16 default)",
    )
    parser.add_argument("--device", default="cuda", help="cuda|cpu (default: cuda)")
    return parser


def _warmup(detector) -> None:
    """Force the model to load right now via one tiny caption pass, so a bad
    config/checkpoint/weights path crashes the process loudly at startup instead of on
    the first real request."""
    import numpy as np

    tile = np.zeros((_WARMUP_TILE_H, _WARMUP_TILE_W, 3), dtype="uint8")
    detector.run_caption_pass([tile], _WARMUP_CAPTION, 0.35)


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)

    # Lazy: only imported once we're actually about to load the model, so --help (and
    # any static inspection/tests of the helpers above) never needs torch/core installed.
    from core.perception.detector import GroundingDinoDetector

    print(
        f"[gdino_server] loading model (config={args.config or 'env/default'}, "
        f"checkpoint={args.checkpoint or 'env/default'}, precision={args.precision}, "
        f"device={args.device})...",
        flush=True,
    )
    detector = GroundingDinoDetector(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        precision=args.precision,
        device=args.device,
    )
    t0 = time.monotonic()
    _warmup(detector)
    warmup_s = time.monotonic() - t0

    resolved_device = detector._resolved_device or args.device
    resolved_precision = "fp16-autocast" if detector._resolved_half else "fp32"
    print(
        f"[gdino_server] model={detector.model_id} device={resolved_device} "
        f"precision={resolved_precision} loaded in {warmup_s:.2f}s",
        flush=True,
    )

    lock = threading.Lock()
    handler_cls = _make_handler_class(
        detector, lock, detector.model_id, resolved_device, resolved_precision,
        time.monotonic(),
    )
    httpd = ThreadingHTTPServer((args.host, args.port), handler_cls)
    print(f"[gdino_server] listening on {args.host}:{args.port}", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("[gdino_server] shutting down", flush=True)
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main(sys.argv[1:])
