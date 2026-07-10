"""Golden input->output fixtures for the five checkpoints (the 46-parse-goldens pattern).

Each ``*.json`` file holds a list of golden cases. A case is a dict:

    {
      "name": str,                 # unique case id
      "input": {...},              # the assembled prompt inputs (snapshot)
      "stub_replies": [str, ...],  # scripted LocalStub replies, in order (1st + repair)
      "timeout": bool,             # optional: simulate a hard timeout (no usable reply)
      "expected": {...}            # the expected structured outcome fields
    }

:func:`load` reads one checkpoint's goldens; :func:`all_goldens` reads every file. Tests
round-trip these against the real checkpoint runners with scripted stubs.
"""
from __future__ import annotations

import json
import os
from typing import Any

_DIR = os.path.dirname(os.path.abspath(__file__))


def load(checkpoint: str) -> list[dict[str, Any]]:
    """Load the golden cases for one checkpoint (e.g. ``"verification"``)."""
    path = os.path.join(_DIR, f"{checkpoint}.json")
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def all_goldens() -> dict[str, list[dict[str, Any]]]:
    """Load every checkpoint's goldens keyed by checkpoint name."""
    out: dict[str, list[dict[str, Any]]] = {}
    for fn in os.listdir(_DIR):
        if fn.endswith(".json"):
            out[fn[:-5]] = load(fn[:-5])
    return out
