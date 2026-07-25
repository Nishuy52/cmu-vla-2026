"""Torch-free tests for tools/cluster/gdino_server.py: CLI smoke test (subprocess, no
torch needed) + the pure JSON-validation / response-shaping helpers."""
from __future__ import annotations

import base64
import subprocess
import sys
from pathlib import Path

import pytest

from core.perception.detector import Detection
from tools.cluster import gdino_server as server
from tools.cluster.gdino_server import BadRequest

_SERVER_PATH = Path(__file__).resolve().parents[2] / "tools" / "cluster" / "gdino_server.py"


# --------------------------------------------------------------------------- CLI smoke


def test_help_exits_zero_without_torch():
    result = subprocess.run(
        [sys.executable, str(_SERVER_PATH), "--help"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0
    assert "--port" in result.stdout
    assert "--precision" in result.stdout


# --------------------------------------------------------------------------- validate_detect_payload


def test_validate_detect_payload_ok():
    payload = {"tiles": [], "caption": "sofa .", "box_threshold": 0.35}
    assert server.validate_detect_payload(payload) is payload


def test_validate_detect_payload_not_an_object():
    with pytest.raises(BadRequest):
        server.validate_detect_payload(["tiles", "caption"])


def test_validate_detect_payload_missing_fields():
    with pytest.raises(BadRequest, match="missing field"):
        server.validate_detect_payload({"tiles": []})


def test_validate_detect_payload_tiles_not_a_list():
    with pytest.raises(BadRequest, match="'tiles'"):
        server.validate_detect_payload(
            {"tiles": "not-a-list", "caption": "sofa .", "box_threshold": 0.35}
        )


def test_validate_detect_payload_caption_not_a_string():
    with pytest.raises(BadRequest, match="'caption'"):
        server.validate_detect_payload(
            {"tiles": [], "caption": 123, "box_threshold": 0.35}
        )


# --------------------------------------------------------------------------- decode_b64_tiles


def test_decode_b64_tiles_roundtrip():
    raw = [b"hello", b"world"]
    encoded = [base64.b64encode(b).decode("ascii") for b in raw]
    assert server.decode_b64_tiles(encoded) == raw


def test_decode_b64_tiles_non_string_entry():
    with pytest.raises(BadRequest, match=r"tiles\[1\]"):
        server.decode_b64_tiles(["aGVsbG8=", 42])


def test_decode_b64_tiles_bad_base64():
    with pytest.raises(BadRequest, match=r"tiles\[0\]"):
        server.decode_b64_tiles(["not valid base64 !!!"])


# --------------------------------------------------------------------------- decode_jpeg_tile


def test_decode_jpeg_tile_roundtrip():
    pytest.importorskip("PIL")
    import io

    import numpy as np
    from PIL import Image

    arr = np.zeros((4, 4, 3), dtype="uint8")
    arr[:, :, 0] = 200
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG")
    decoded = server.decode_jpeg_tile(buf.getvalue())
    assert decoded.shape == (4, 4, 3)
    # JPEG is lossy; just check it round-tripped into the right ballpark, not exact.
    assert decoded[..., 0].mean() > 150


# --------------------------------------------------------------------------- response_body / error_body


def test_response_body_shapes_detections():
    per_tile = [
        [
            Detection(tile_id=0, bbox_xyxy=(1.0, 2.0, 3.0, 4.0), label="sofa", score=0.9),
            Detection(tile_id=0, bbox_xyxy=(5.0, 6.0, 7.0, 8.0), label="chair", score=0.4),
        ],
        [],
    ]
    assert server.response_body(per_tile) == {
        "per_tile": [
            [
                {"bbox_xyxy": [1.0, 2.0, 3.0, 4.0], "label": "sofa", "score": 0.9},
                {"bbox_xyxy": [5.0, 6.0, 7.0, 8.0], "label": "chair", "score": 0.4},
            ],
            [],
        ]
    }


def test_response_body_empty():
    assert server.response_body([]) == {"per_tile": []}


def test_error_body():
    assert server.error_body("bad request") == {"error": "bad request"}
    assert server.error_body(ValueError("boom")) == {"error": "boom"}
