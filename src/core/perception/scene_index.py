"""BasicSceneIndex — a :class:`~core.interfaces.SceneIndex` over InstanceRecords.

Provides typo/plural/synonym-tolerant label lookup (used by the toolbox and answer
heads) plus the add/merge API that real perception uses to fuse cross-frame
detections: same-label instances overlapping in 3D (IoU > MERGE_IOU) are fused —
points concatenated, AABB recomputed as the per-axis 2nd/98th-percentile trimmed
box, ``n_obs`` incremented, ``score`` kept as the max.

Label matching ladder (higher tier wins; :meth:`by_label` returns higher tiers
first, and :meth:`by_label_tiered` surfaces which tier each hit came from):
  1. exact     — canonical (lowercase-singular) match
  2. synonym   — synonym-table / vocab-bridge equivalence (fridge<->refrigerator,
                 sofa<->couch, bedside table<->night stand, ...); matches on the
                 record canonical or its declared aliases (exact string only)
  3. head-noun — the query's head noun matches a candidate's head-noun-headed label
                 ("X table" matches "table"-headed labels, "beer bottle"->"bottle")
  4. typo      — edit-distance fuzzy match, length-scaled tolerance, returned ONLY
                 when tiers 1-3 are all empty; aliases NEVER participate here
                 ('refridgerator' -> 'refrigerator')
"""
from __future__ import annotations

import threading

import numpy as np

from core.interfaces import InstanceRecord, MarkerBox, MatchTier

# Re-exported for backward compatibility: this module used to define MatchTier
# itself; it now lives on core.interfaces (see SceneIndex.by_label_tiered, #24) so
# the Protocol can name it without a core -> perception import cycle.
__all__ = ["BasicSceneIndex", "MatchTier", "normalize_label", "singularize"]

MERGE_IOU: float = 0.3  # 3D IoU threshold for fusing same-label instances
TYPO_MAX_DIST: int = 2  # max Levenshtein distance for the longest length band
TRIM_LO_PCT: float = 2.0
TRIM_HI_PCT: float = 98.0


def _typo_budget(query: str, candidate: str) -> int:
    """Length-scaled Levenshtein budget for a fuzzy match; 0 disables fuzzy.

    No fuzzy match when either side is shorter than 5 chars (short words collide
    too readily: 'door'/'floor', 'tap'/'cup', 'bag'/'bed'). Edit distance <= 1 for
    length 5-7, <= 2 only for length 8+. The band is set by the SHORTER of the two
    strings, so a long candidate cannot buy tolerance a short query never earns.
    """
    n = min(len(query), len(candidate))
    if n < 5:
        return 0
    if n <= 7:
        return 1
    return 2

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
    """Mutable in-memory scene index. Implements the SceneIndex protocol.

    Thread safety (issue #88): a live run now feeds this index from a dedicated
    perception worker thread (core.perception.async_pipeline.AsyncPerceptionWorker)
    while the tick thread's heads concurrently read it (by_label/all_instances/...)
    to resolve answers. An internal RLock (``self._lock``) guards every read and
    write method so index mutation (add/remove/_fuse) and iteration (by_label_tiered,
    all_instances) are never interleaved on the same underlying list — without it, a
    reader iterating ``self._instances`` while the worker thread deletes from it can
    raise ``RuntimeError: list changed size during iteration``. Callers (nav/heads)
    need no changes: the lock lives entirely inside this class.
    """

    def __init__(self, instances: list[InstanceRecord] | None = None) -> None:
        self._lock = threading.RLock()
        self._instances: list[InstanceRecord] = list(instances or [])
        self._next_id = 1 + max(
            (r.instance_id for r in self._instances), default=-1
        )

    # ------------------------------------------------------------- read protocol

    def all_instances(self):
        with self._lock:
            return list(self._instances)

    def remove(self, instance_id: int) -> bool:
        """Drop an instance by id; return True if one was removed.

        Used by track decay (H15a) to prune one-frame ghosts. No-op (returns False)
        if the id is absent. Never mutates ``_next_id`` — freed ids are not recycled,
        so a pruned ghost's id cannot be silently reused by a later real object.
        """
        with self._lock:
            for i, rec in enumerate(self._instances):
                if rec.instance_id == instance_id:
                    del self._instances[i]
                    return True
            return False

    def marker_for(self, record: InstanceRecord) -> MarkerBox:
        """Prior-clamped marker for a record — the H12 marker seam.

        The trimmed AABB an instance carries is a single-viewpoint *under*-box; the
        raw ``record.to_marker()`` would publish it verbatim and shed IoU against the
        GT over-hull (red-team OR-F6). This routes the record through the per-class
        dimension prior (:mod:`core.perception.dimension_priors`): clamp every axis to
        the class-min, inflate the least-observed axis toward class-typical only when
        the instance signals under-observation, centre preserved. A GT-perfect / well-
        observed box is returned identical to ``record.to_marker()``.

        Marker-path owners (``heads/object_ref.py``, ``fsm/floors.py``) should publish
        ``index.marker_for(rec)`` in place of ``rec.to_marker()`` — the seam lives here
        so the clamp is applied wherever the scene index is in scope. Deferred import
        avoids a load cycle (dimension_priors imports normalize_label from this module).
        """
        from core.perception.dimension_priors import clamp_record_marker

        return clamp_record_marker(record)

    def next_id(self) -> int:
        """The instance_id the next fresh (non-merged) instance would receive.

        Read-only peek used by the tracker to mint ids for unmatched detections;
        ``add`` still owns id assignment and will reassign on collision.
        """
        with self._lock:
            return self._next_id

    def by_label(self, noun: str):
        """Typo/plural/synonym-tolerant lookup; returns matching instances, best-first.

        Tiers are tried exact -> synonym -> head-noun -> typo (see the module
        docstring). The typo tier is a short-circuit: it contributes ONLY when the
        exact, synonym and head-noun tiers are all empty, so an exact match is never
        polluted by fuzzy cousins.
        """
        return [rec for rec, _ in self.by_label_tiered(noun)]

    def by_label_tiered(self, noun: str) -> list[tuple[InstanceRecord, MatchTier]]:
        """Like :meth:`by_label` but pairs each hit with its :class:`MatchTier`.

        Lets consumers (e.g. ``resolve``/audit) prefer stronger-tier candidates before
        superlative ranking without breaking the flat-list return type of
        :meth:`by_label`.
        """
        # Deferred import: perception.vocab imports normalize_label from this module,
        # so we consult it at call time to avoid a module-load cycle.
        from core.perception.vocab import bridge_synonyms, head_noun

        query = normalize_label(noun)
        query_head = head_noun(query)
        syns = bridge_synonyms(query)  # normalised object-bridge equivalents

        exact: list[InstanceRecord] = []
        syn: list[InstanceRecord] = []
        head: list[InstanceRecord] = []
        typo: list[InstanceRecord] = []
        with self._lock:
            instances_snapshot = list(self._instances)
        for rec in instances_snapshot:
            canon = normalize_label(rec.label)
            if canon == query:
                exact.append(rec)
                continue
            # synonym tier: the vocab bridge, plus the record's declared aliases —
            # but aliases are matched by EXACT string only, never fuzzily (colour
            # aliases like 'yellow' are edit-distance 2 from 'pillow').
            alias_canons = [normalize_label(a) for a in rec.aliases]
            if canon in syns or query in bridge_synonyms(canon) or query in alias_canons:
                syn.append(rec)
                continue
            # head-noun tier: modifier-stripped class-word match, either direction
            # ("beer bottle" query vs "bottle" label, or "table" query vs "X table").
            if head_noun(canon) == query_head:
                head.append(rec)
                continue
            # typo tier candidate (length-gated); aliases excluded on purpose.
            budget = _typo_budget(query, canon)
            if budget and _levenshtein(canon, query) <= budget:
                typo.append(rec)

        tiered: list[tuple[InstanceRecord, MatchTier]] = []
        tiered += [(r, MatchTier.EXACT) for r in exact]
        tiered += [(r, MatchTier.SYNONYM) for r in syn]
        tiered += [(r, MatchTier.HEAD_NOUN) for r in head]
        # typo short-circuit: only when every stronger tier is empty.
        if not tiered:
            tiered += [(r, MatchTier.TYPO) for r in typo]
        return tiered

    # -------------------------------------------------------------- write / merge

    def add(self, rec: InstanceRecord) -> InstanceRecord:
        """Add an observation, fusing into an existing same-label instance when
        their 3D AABB IoU exceeds MERGE_IOU. Returns the surviving record."""
        with self._lock:
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
        # Called only from within add()'s locked section (RLock: reentrant).
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
