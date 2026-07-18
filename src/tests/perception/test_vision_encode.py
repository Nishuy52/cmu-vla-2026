"""Real JPEG encode_fn (Gate 4 item 2): jpeg_encode_fn, FakeEncoder, resolve_encode_fn.

Pillow is present in this project's venv (checked at authoring time), so the real
encoder path is exercised directly here, not just its lazy-import guard. The lazy-import
error path (Pillow absent) is still covered via the same builtins.__import__ monkeypatch
trick core.perception.detector's stub test uses, so the guard is verified even though
Pillow happens to be installed in this environment.
"""
from __future__ import annotations

import builtins
import io

import numpy as np
import pytest

from core.perception.vision_encode import (
    DEFAULT_JPEG_QUALITY,
    ENV_JPEG_QUALITY,
    FakeEncoder,
    JPEG_REQUIRED_INSTALLS,
    _to_uint8,
    has_pillow,
    jpeg_encode_fn,
    resolve_encode_fn,
)


def _rgb(h=20, w=30):
    rng = np.random.default_rng(0)
    return (rng.random((h, w, 3)) * 255).astype(np.uint8)


# ------------------------------------------------------------------ _to_uint8


def test_to_uint8_passes_through_uint8():
    arr = np.array([[0, 255], [10, 200]], dtype=np.uint8)
    out = _to_uint8(arr)
    assert out.dtype == np.uint8
    assert np.array_equal(out, arr)


def test_to_uint8_scales_zero_one_floats():
    arr = np.array([0.0, 0.5, 1.0])
    out = _to_uint8(arr)
    assert out.tolist() == [0, 127, 255]


def test_to_uint8_clips_out_of_range():
    arr = np.array([-10.0, 300.0])
    out = _to_uint8(arr)
    assert out.tolist() == [0, 255]


# ------------------------------------------------------------------ jpeg_encode_fn (real path)


def test_jpeg_encode_fn_bytes_pass_through():
    raw = b"already-encoded"
    assert jpeg_encode_fn(raw) == raw


def test_jpeg_encode_fn_rgb_uint8_produces_valid_jpeg():
    b = jpeg_encode_fn(_rgb())
    assert isinstance(b, bytes) and len(b) > 0
    assert b[:2] == b"\xff\xd8"  # JPEG SOI marker
    assert b[-2:] == b"\xff\xd9"  # JPEG EOI marker


def test_jpeg_encode_fn_grayscale():
    arr = np.zeros((10, 10), dtype=np.uint8)
    b = jpeg_encode_fn(arr)
    assert b[:2] == b"\xff\xd8"


def test_jpeg_encode_fn_rgba_drops_alpha():
    arr = np.zeros((10, 10, 4), dtype=np.uint8)
    b = jpeg_encode_fn(arr)  # must not raise (JPEG has no alpha channel)
    assert b[:2] == b"\xff\xd8"


def test_jpeg_encode_fn_float_zero_one_input():
    arr = np.random.default_rng(1).random((8, 8, 3))
    b = jpeg_encode_fn(arr)
    assert b[:2] == b"\xff\xd8"


def test_jpeg_encode_fn_unsupported_shape_raises():
    with pytest.raises(ValueError):
        jpeg_encode_fn(np.zeros((4, 4, 5), dtype=np.uint8))


def test_jpeg_encode_fn_quality_arg_changes_size():
    img = _rgb(64, 64)
    lo = jpeg_encode_fn(img, quality=5)
    hi = jpeg_encode_fn(img, quality=95)
    assert len(lo) < len(hi)  # lower quality -> smaller file, same content


def test_jpeg_encode_fn_quality_env_var(monkeypatch):
    monkeypatch.setenv(ENV_JPEG_QUALITY, "5")
    img = _rgb(64, 64)
    from_env = jpeg_encode_fn(img)
    explicit_default = jpeg_encode_fn(img, quality=DEFAULT_JPEG_QUALITY)
    assert len(from_env) < len(explicit_default)


# ------------------------------------------------------------------ lazy-import guard


def test_jpeg_encode_fn_raises_clear_install_error_without_pillow(monkeypatch):
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "PIL" or name.startswith("PIL."):
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError) as exc:
        jpeg_encode_fn(_rgb())
    msg = str(exc.value)
    for pkg in JPEG_REQUIRED_INSTALLS:
        assert pkg in msg
    assert "pip install" in msg


# ------------------------------------------------------------------ FakeEncoder


def test_fake_encoder_deterministic_and_distinct_from_npy_magic():
    fe = FakeEncoder()
    img = _rgb()
    b1 = fe(img)
    b2 = fe(img)
    assert b1 == b2  # deterministic
    assert len(b1) > 0
    assert not b1.startswith(b"\x93NUMPY")  # distinct from default_encode_fn's .npy magic
    assert b1.startswith(FakeEncoder.MAGIC)


def test_fake_encoder_bytes_pass_through():
    raw = b"already-bytes"
    assert FakeEncoder()(raw) == raw


def test_fake_encoder_distinguishes_different_images():
    fe = FakeEncoder()
    a = fe(np.zeros((4, 4, 3), dtype=np.uint8))
    b = fe(np.ones((4, 4, 3), dtype=np.uint8))
    assert a != b


# ------------------------------------------------------------------ resolve_encode_fn / has_pillow


def test_has_pillow_true_in_this_venv():
    # Documents the venv fact this module's docstring relies on (pillow IS installed here).
    assert has_pillow() is True


def test_resolve_encode_fn_returns_jpeg_when_pillow_present():
    assert resolve_encode_fn() is jpeg_encode_fn


class _RecordingLogger:
    def __init__(self) -> None:
        self.warnings: list[str] = []

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)


def test_resolve_encode_fn_falls_back_and_warns_without_pillow(monkeypatch):
    monkeypatch.setattr(
        "core.perception.vision_encode.has_pillow", lambda: False
    )
    logger = _RecordingLogger()
    fn = resolve_encode_fn(logger)
    assert fn is None
    assert len(logger.warnings) == 1
    assert "Pillow" in logger.warnings[0]


def test_resolve_encode_fn_never_raises_without_logger(monkeypatch):
    monkeypatch.setattr(
        "core.perception.vision_encode.has_pillow", lambda: False
    )
    assert resolve_encode_fn(None) is None  # no logger given -> still no raise
