"""Developer debug tooling that lives *outside* ``src/ai_module``.

Everything under ``tools/`` is diagnostic-only and never runs in the scored
per-question pipeline, so it is deliberately excluded from what gets copied into
the eventual challenge fork (only ``ai_module/`` may be modified — root
``CLAUDE.md``). It is free to import from ``core/`` (the pure-Python, numpy-only
world model) but ``core/`` must never import back.

Importing this package inserts ``<repo>/src`` onto ``sys.path`` so ``core`` is
importable when the tools are run as ``python -m tools.<mod>`` from the repo
root, with no ``PYTHONPATH`` fiddling.
"""
from __future__ import annotations

import os
import sys

# <repo>/tools/__init__.py -> <repo>/src
_SRC = os.path.normpath(os.path.join(os.path.dirname(__file__), os.pardir, "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
