"""Hand-built synthetic InstanceRecords + a tiny SceneIndex for geometry tests.

Deliberately does NOT import core.mocks (may not exist yet). Boxes are given by
centre + full extents; centroid defaults to the AABB centre. All map frame, metres.
"""
from __future__ import annotations

import difflib
from typing import Sequence

import numpy as np

from core.interfaces import InstanceRecord


def rec(
    instance_id: int,
    label: str,
    center: Sequence[float],
    size: Sequence[float] = (1.0, 1.0, 1.0),
    *,
    n_obs: int = 3,
    score: float = 0.9,
    centroid: Sequence[float] | None = None,
    caption: str = "",
    aliases: tuple[str, ...] = (),
) -> InstanceRecord:
    """Build an InstanceRecord from centre + full extents (metres)."""
    c = np.asarray(center, dtype=float)
    s = np.asarray(size, dtype=float)
    half = s / 2.0
    cen = np.asarray(centroid, dtype=float) if centroid is not None else c.copy()
    return InstanceRecord(
        instance_id=instance_id,
        label=label,
        score=score,
        n_obs=n_obs,
        centroid=cen,
        aabb_min=c - half,
        aabb_max=c + half,
        caption=caption,
        aliases=aliases,
    )


class FakeIndex:
    """Minimal SceneIndex: typo-tolerant by_label over a fixed record list."""

    def __init__(self, records: Sequence[InstanceRecord]):
        self._records = list(records)

    def all_instances(self) -> Sequence[InstanceRecord]:
        return list(self._records)

    def by_label(self, noun: str) -> Sequence[InstanceRecord]:
        n = noun.strip().lower()
        exact = [
            r
            for r in self._records
            if r.label.lower() == n or n in {a.lower() for a in r.aliases}
        ]
        if exact:
            return exact
        # typo tolerance: close-ratio match against labels + aliases
        out = []
        for r in self._records:
            cands = [r.label.lower(), *(a.lower() for a in r.aliases)]
            if any(difflib.SequenceMatcher(None, n, c).ratio() >= 0.8 for c in cands):
                out.append(r)
        return out
