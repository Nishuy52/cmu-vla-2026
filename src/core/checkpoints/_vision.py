"""Injectable image encoding for the vision checkpoints (CP2/CP3/CP5).

The design specifies JPEG-encoded crops/tiles, but cv2/PIL are not project dependencies
(and adding one violates "no new hard deps"). So the vision checkpoints take an injectable
``encode_fn(image) -> bytes`` and the DEFAULT is a dependency-free encoder that serialises
the raw numpy array via ``numpy.save`` into ``.npy`` bytes — fully deterministic and ideal
for offline fixture tests. Real JPEG encoding is a Phase-2 wiring step: production bind
passes an ``encode_fn`` that runs cv2/PIL, no checkpoint code changes.

The ``.npy`` bytes carry a distinctive magic prefix (``\\x93NUMPY``) so a provider adapter
could sniff them; for tests we only care that the bytes are stable and non-empty.
"""
from __future__ import annotations

import io
from typing import Any

import numpy as np


def default_encode_fn(image: Any) -> bytes:
    """Encode an image (numpy array or bytes) to raw ``.npy`` bytes. Deterministic.

    * ``bytes`` pass through unchanged (already-encoded input).
    * anything array-like is coerced via ``np.asarray`` and written with ``np.save``.
    """
    if isinstance(image, (bytes, bytearray)):
        return bytes(image)
    buf = io.BytesIO()
    np.save(buf, np.asarray(image), allow_pickle=False)
    return buf.getvalue()
