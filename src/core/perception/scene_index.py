"""BasicSceneIndex — a :class:`~core.interfaces.SceneIndex` over InstanceRecords.

Provides typo/plural/synonym-tolerant label lookup (used by the toolbox and answer
heads) plus the add/merge API that real perception uses to fuse cross-frame
detections: same-label instances overlapping in 3D (IoU > MERGE_IOU) are fused —
points concatenated, AABB recomputed as the per-axis 2nd/98th-percentile trimmed
box, ``n_obs`` incremented, ``score`` kept as the max.

Label matching ladder (first hit wins):
  1. exact canonical (lowercase-singular) match
  2. synonym-table equivalence (fridge<->refrigerator, sofa<->couch, ...)
  3. edit-distance <= 2 against any canonical label (typo tolerance;
     'refridgerator' -> 'refrigerator')
"""
from __future__ import annotations

import numpy as np

from core.interfaces import InstanceRecord

MERGE_IOU: float = 0.3  # 3D IoU threshold for fusing same-label instances
TYPO_MAX_DIST: int = 2  # max Levenshtein distance for typo-tolerant match
TRIM_LO_PCT: float = 2.0
TRIM_HI_PCT: float = 98.0

# Canonical synonym groups. Every member maps to the group's canonical head
# (the first element). Lookup normalises a query to its canonical head, then
# matches instances whose label shares the same head.
_SYNONYM_GROUPS: tuple[tuple[str, ...], ...] = (
    ("refrigerator", "fridge"),
    ("sofa", "couch"),
    ("television", "tv"),
    ("picture", "photo"),
)


def _build_synonym_map() -> dict[str, str]:
    m: dict[str, str] = {}
    for group in _SYNONYM_GROUPS:
        head = group[0]
        for member in group:
            m[member] = head
    return m


_SYNONYM_MAP = _build_synonym_map()


def singularize(noun: str) -> str:
    """Cheap English plural stripping to a singular canonical form.

    Handles the common regular patterns present in the challenge vocabulary
    (…ies -> …y, …ses/…xes/…zes/…ches/…shes -> drop 'es', trailing 's').
    """
    w = noun.strip().lower()
    if len(w) <= 3 or not w.endswith("s"):
        return w
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith(("ches", "shes", "sses", "xes", "zes")):
        return w[:-2]
    if w.endswith("ss"):
        return w
    return w[:-1]


def normalize_label(noun: str) -> str:
    """Canonicalise a noun: lowercase, singular, mapped through the synonym table."""
    base = singularize(noun)
    return _SYNONYM_MAP.get(base, base)


def _levenshtein(a: str, b: str) -> int:
    """Standard Levenshtein edit distance (iterative two-row DP)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[-1]


def _aabb_iou_3d(min_a, max_a, min_b, max_b) -> float:
    """Axis-aligned 3D IoU of two boxes given as (3,) min/max corners."""
    lo = np.maximum(min_a, min_b)
    hi = np.minimum(max_a, max_b)
    inter_dims = np.clip(hi - lo, 0.0, None)
    inter = float(np.prod(inter_dims))
    if inter <= 0.0:
        return 0.0
    vol_a = float(np.prod(np.clip(max_a - min_a, 0.0, None)))
    vol_b = float(np.prod(np.clip(max_b - min_b, 0.0, None)))
    union = vol_a + vol_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def _trimmed_aabb(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-axis 2nd/98th-percentile trimmed AABB of an (M, 3) cloud."""
    lo = np.percentile(points, TRIM_LO_PCT, axis=0)
    hi = np.percentile(points, TRIM_HI_PCT, axis=0)
    return lo.astype(float), hi.astype(float)


class BasicSceneIndex:
    """Mutable in-memory scene index. Implements the SceneIndex protocol."""

    def __init__(self, instances: list[InstanceRecord] | None = None) -> None:
        self._instances: list[InstanceRecord] = list(instances or [])
        self._next_id = 1 + max(
            (r.instance_id for r in self._instances), default=-1
        )

    # ------------------------------------------------------------- read protocol

    def all_instances(self):
        return list(self._instances)

    def by_label(self, noun: str):
        """Typo/plural/synonym-tolerant lookup; returns matching instances."""
        query = normalize_label(noun)
        exact: list[InstanceRecord] = []
        syn: list[InstanceRecord] = []
        typo: list[InstanceRecord] = []
        for rec in self._instances:
            canon = normalize_label(rec.label)
            if canon == query:
                exact.append(rec)
                continue
            # synonym already folded into normalize_label; typo tolerance next.
            # match against both the record canonical and its declared aliases.
            candidates = [canon] + [normalize_label(a) for a in rec.aliases]
            if any(c == query for c in candidates):
                syn.append(rec)
            elif any(_levenshtein(c, query) <= TYPO_MAX_DIST for c in candidates):
                typo.append(rec)
        # exact first, then synonym, then typo — de-duplicated by identity order
        return exact + syn + typo

    # -------------------------------------------------------------- write / merge

    def add(self, rec: InstanceRecord) -> InstanceRecord:
        """Add an observation, fusing into an existing same-label instance when
        their 3D AABB IoU exceeds MERGE_IOU. Returns the surviving record."""
        target = self._find_merge_target(rec)
        if target is None:
            if rec.instance_id in (r.instance_id for r in self._instances):
                rec = InstanceRecord(
                    instance_id=self._next_id,
                    label=rec.label,
                    score=rec.score,
                    n_obs=rec.n_obs,
                    centroid=rec.centroid,
                    aabb_min=rec.aabb_min,
                    aabb_max=rec.aabb_max,
                    points=rec.points,
                    caption=rec.caption,
                    aliases=rec.aliases,
                )
            self._next_id = max(self._next_id, rec.instance_id + 1)
            self._instances.append(rec)
            return rec
        self._fuse(target, rec)
        return target

    def _find_merge_target(self, rec: InstanceRecord) -> InstanceRecord | None:
        q = normalize_label(rec.label)
        best: InstanceRecord | None = None
        best_iou = MERGE_IOU
        for existing in self._instances:
            if normalize_label(existing.label) != q:
                continue
            iou = _aabb_iou_3d(
                existing.aabb_min, existing.aabb_max, rec.aabb_min, rec.aabb_max
            )
            if iou > best_iou:
                best_iou = iou
                best = existing
        return best

    def _fuse(self, target: InstanceRecord, other: InstanceRecord) -> None:
        """Fuse ``other`` into ``target`` in place: concat points, recompute the
        trimmed AABB and centroid, bump n_obs, keep max score."""
        clouds = [p for p in (target.points, other.points) if p is not None and len(p)]
        if clouds:
            fused = np.vstack(clouds).astype(float)
            target.points = fused
            lo, hi = _trimmed_aabb(fused)
            target.aabb_min = lo
            target.aabb_max = hi
            target.centroid = (lo + hi) / 2.0
        else:
            # no points to trim with — union the boxes as a fallback
            target.aabb_min = np.minimum(target.aabb_min, other.aabb_min)
            target.aabb_max = np.maximum(target.aabb_max, other.aabb_max)
            target.centroid = (target.aabb_min + target.aabb_max) / 2.0
        target.n_obs += other.n_obs
        target.score = max(target.score, other.score)
