"""Real JPEG image encoding for the vision checkpoints (CP2/CP3/CP5) — Gate 4 item 2.

``core.checkpoints._vision.default_encode_fn`` (unmodified here — ``core/checkpoints/**``
is out of scope for this change, see the ownership note in the Gate-4 task) emits raw
``.npy`` bytes so the offline test suite never needs an image library. Pillow *is*
available in this project's venv in practice, but is imported *lazily* here — mirroring
``core.perception.detector``'s lazy-torch pattern — so importing this module never
requires Pillow to be installed, and the offline/core test suite stays exactly as
fast/dependency-free as before.

Wire :func:`jpeg_encode_fn` into a checkpoint builder's ``cfg={"encode_fn": jpeg_encode_fn}``
(see ``ros_adapter.adapter_node._build_checkpoint_seams``, the one place in this repo
allowed to reach into ``core.checkpoints``' builder ``cfg`` dicts) wherever a real vision
API call needs real JPEG bytes instead of ``.npy``. :func:`resolve_encode_fn` is the
single seam that picks the real encoder when Pillow is present and gracefully falls back
(returning ``None``, so the checkpoint builder's own ``.npy`` default applies) when it is
not — never raising, so a missing Pillow degrades rather than crashes the boot.
:class:`FakeEncoder` is a deterministic, Pillow-free stand-in for tests that want to
assert "a real-image-shaped encoder was wired" without requiring Pillow.
"""
from __future__ import annotations

import io
import os
from typing import Any

import numpy as np

#: JPEG quality (1-95); overridable via env so ops can trade bandwidth vs fidelity
#: without a code change.
ENV_JPEG_QUALITY = "VLA_JPEG_QUALITY"
DEFAULT_JPEG_QUALITY = 85

#: The exact pip install the real path needs (mirrors detector.GDINO_REQUIRED_INSTALLS).
JPEG_REQUIRED_INSTALLS: tuple[str, ...] = ("pillow",)


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    """Best-effort float/other -> uint8 conversion for JPEG encoding.

    Arrays already scaled to [0, 1] (a common convention for float image tensors) are
    scaled up to [0, 255] first; everything else is simply clipped/cast. Deterministic
    and pure numpy (no torch/PIL needed), so unit-testable on its own.
    """
    a = np.asarray(arr, dtype=float)
    if a.size and 0.0 <= a.min() and a.max() <= 1.0:
        a = a * 255.0
    return np.clip(a, 0, 255).astype(np.uint8)


def jpeg_encode_fn(image: Any, *, quality: int | None = None) -> bytes:
    """Encode an image (numpy array) to real JPEG bytes via Pillow (lazy import).

    * ``bytes``/``bytearray`` pass through unchanged (already-encoded input).
    * grayscale ``(H, W)``, RGB ``(H, W, 3)``, RGBA ``(H, W, 4)`` arrays are accepted;
      non-uint8 dtypes go through :func:`_to_uint8` first.
    * ``quality`` defaults to the ``VLA_JPEG_QUALITY`` env var, else
      :data:`DEFAULT_JPEG_QUALITY`.

    Raises a clear :class:`ImportError` (naming ``pip install pillow``) if Pillow is
    absent, so a misconfigured environment fails loudly rather than silently emitting a
    different byte format a real vision API would reject. Production code should not
    call this directly without checking :func:`resolve_encode_fn` first — that seam
    degrades gracefully; this function does not.
    """
    if isinstance(image, (bytes, bytearray)):
        return bytes(image)
    try:
        from PIL import Image
    except ImportError as exc:
        installs = " ".join(JPEG_REQUIRED_INSTALLS)
        raise ImportError(
            "jpeg_encode_fn requires Pillow, which is not installed. Install it with:\n"
            f"    pip install {installs}\n"
            "The offline/test path should inject core.checkpoints._vision.default_encode_fn "
            "(raw .npy bytes) or vision_encode.FakeEncoder instead — no image lib needed."
        ) from exc

    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        arr = _to_uint8(arr)
    if arr.ndim == 2:
        mode = "L"
    elif arr.ndim == 3 and arr.shape[-1] == 3:
        mode = "RGB"
    elif arr.ndim == 3 and arr.shape[-1] == 4:
        mode = "RGBA"
    else:
        raise ValueError(f"jpeg_encode_fn: unsupported image shape {arr.shape!r}")

    pil_img = Image.fromarray(arr, mode=mode)
    if mode == "RGBA":
        pil_img = pil_img.convert("RGB")  # JPEG has no alpha channel
    q = int(quality if quality is not None else os.environ.get(ENV_JPEG_QUALITY, DEFAULT_JPEG_QUALITY))
    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=q)
    return buf.getvalue()


class FakeEncoder:
    """Deterministic offline stand-in for :func:`jpeg_encode_fn` — no Pillow needed.

    Emits a distinctive, stable, non-empty byte string per call (a fixed magic prefix +
    the array's shape + a slice of its raw bytes) so tests can assert "some real-image
    encoder distinct from the raw-.npy default was wired" without requiring Pillow to be
    installed. NOT a real image format — tests only; production always uses
    :func:`jpeg_encode_fn` via :func:`resolve_encode_fn`.
    """

    #: `.npy`'s magic is ``\x93NUMPY`` (see core.checkpoints._vision); a different,
    #: equally-distinctive prefix makes the two trivially distinguishable in tests.
    MAGIC = b"FAKEJPEG"

    def __call__(self, image: Any) -> bytes:
        if isinstance(image, (bytes, bytearray)):
            return bytes(image)
        arr = np.asarray(image)
        return self.MAGIC + repr(arr.shape).encode("utf-8") + arr.tobytes()[:64]


def has_pillow() -> bool:
    """True iff Pillow is importable — used to pick a real vs fallback encoder at boot."""
    try:
        import PIL  # noqa: F401
    except ImportError:
        return False
    return True


def resolve_encode_fn(logger: Any = None):
    """Pick the best available vision encode_fn at boot: real JPEG if Pillow is present,
    else ``None`` (the caller / checkpoint builder falls back to its own default raw
    ``.npy`` encoder).

    Never raises — a missing Pillow is a soft degrade (a loud warning if a ``logger``
    with a ``.warn`` method is given), not a boot failure. The adapter's H14 boot assert
    (``ros_adapter.adapter_node._assert_encoder_provider_consistency``) separately shouts
    if a network LLM/VLM provider is configured with no real image encoder available.
    """
    if has_pillow():
        return jpeg_encode_fn
    if logger is not None:
        logger.warn(
            "Pillow not installed: vision checkpoints (CP2/CP3/CP5) will use the "
            "offline default .npy image encoder, which a real vision API rejects. "
            "Install with: pip install %s" % " ".join(JPEG_REQUIRED_INSTALLS)
        )
    return None


__all__ = [
    "ENV_JPEG_QUALITY",
    "DEFAULT_JPEG_QUALITY",
    "JPEG_REQUIRED_INSTALLS",
    "jpeg_encode_fn",
    "FakeEncoder",
    "has_pillow",
    "resolve_encode_fn",
]
