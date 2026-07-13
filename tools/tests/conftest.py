"""Path bootstrap for the tools test suite.

Runs before collection so ``import core.*`` and ``import tools.*`` both resolve
when the suite is invoked as ``python -m pytest tools`` from the repo root — no
``PYTHONPATH`` needed. Inserts ``<repo>`` (for ``tools``) and ``<repo>/src``
(for ``core``) onto ``sys.path``.
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(__file__)
_REPO = os.path.normpath(os.path.join(_HERE, os.pardir, os.pardir))
_SRC = os.path.join(_REPO, "src")

for _p in (_REPO, _SRC):
    if _p not in sys.path:
        sys.path.insert(0, _p)
