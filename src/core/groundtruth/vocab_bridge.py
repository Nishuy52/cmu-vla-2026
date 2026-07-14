"""Backwards-compatible re-export of the object-class vocabulary bridge.

The bridge map moved to the neutral shared location :mod:`core.perception.vocab`
so the *live* matcher (:class:`core.perception.scene_index.BasicSceneIndex`) can
consult it without the perception layer importing from ``groundtruth/``. The scorer
(:mod:`core.groundtruth.scoring`) still imports these names from here; this shim
keeps those imports — and the public API — unchanged.

See :mod:`core.perception.vocab` for the full documentation, the honest-none
discipline, and the extendable :data:`VOCAB_BRIDGE` constant.
"""
from __future__ import annotations

from core.perception.vocab import (
    VOCAB_BRIDGE,
    bridge_synonyms,
    bridged_agree,
)

__all__ = [
    "VOCAB_BRIDGE",
    "bridge_synonyms",
    "bridged_agree",
]
