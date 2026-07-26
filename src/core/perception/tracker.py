"""Cross-frame association + the glued perception pipeline.

New 3D detections (from :mod:`core.perception.fusion`) are matched to existing
:class:`~core.interfaces.InstanceRecord`s by:

* **label compatibility** — canonicalised through the scene index's
  :func:`~core.perception.scene_index.normalize_label` *and* the vocab's
  :data:`~core.parsing.vocab.NOUN_ALIASES` table (so ``television`` associates with a
  ``tv`` instance, ``fridge`` with ``refrigerator``), and
* **centroid proximity** — 3D distance below ``gate`` (default 0.75 m).

Matching is greedy nearest: candidate (detection, instance) pairs are considered in
ascending distance and each side consumed at most once. Matched detections are fed
to :meth:`~core.perception.scene_index.BasicSceneIndex.add`, which reuses the existing
IoU-gated merge machinery (points concatenated, trimmed AABB recomputed, ``n_obs``
incremented, max ``score`` kept). Unmatched detections become new instances.

:class:`PerceptionPipeline` glues tiling -> detector -> fusion -> tracker behind a
keyframe gate so we only pay for perception every K-th frame or after enough vehicle
motion (translation / rotation), mirroring the geometry toolbox's ``Thresholds``
pattern.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

from core.interfaces import InstanceRecord, LidarScan, OdomState, PanoFrame
from core.parsing.vocab import NOUN_ALIASES
from core.perception.detector import (
    Detection,
    DetectorFn,
    ENV_RAW_DETECTION_DUMP_PATH,
    GATE_ACCEPTED,
    GATE_NO_LIDAR_CLUSTER,
    dump_raw_detections,
)
from core.perception.fusion import (
    DEFAULT_FUSION_CONFIG,
    Fused3D,
    FusionConfig,
    fuse_detection,
)
from core.perception.scene_index import (
    BasicSceneIndex,
    DEFAULT_INSTANCE_DUMP_INTERVAL_S,
    ENV_INSTANCE_DUMP_INTERVAL_S,
    ENV_INSTANCE_DUMP_PATH,
    dump_instance_index,
    labels_foldable,
    normalize_label,
)
from core.perception.tiling import (
    DEFAULT_N_TILES,
    DEFAULT_TILE_HFOV,
    DEFAULT_TILE_VFOV,
    project_tiles,
)

# --------------------------------------------------------------------------- label compat

#: Reverse of NOUN_ALIASES: every alias surface form -> its canonical noun. Combined
#: with normalize_label this lets 'television' match a 'tv' instance and vice-versa.
_ALIAS_TO_CANON: dict[str, str] = {}
for _canon, _aliases in NOUN_ALIASES.items():
    for _a in _aliases:
        _ALIAS_TO_CANON[normalize_label(_a)] = normalize_label(_canon)
    _ALIAS_TO_CANON[normalize_label(_canon)] = normalize_label(_canon)


def canonical_for_match(label: str) -> str:
    """Fully canonicalise a label for association: scene-index normalise then fold
    through NOUN_ALIASES (so 'television' and 'tv' collapse to one key)."""
    base = normalize_label(label)
    return _ALIAS_TO_CANON.get(base, base)


def labels_compatible(a: str, b: str) -> bool:
    """True when two labels denote the same canonical noun (alias/synonym-aware), OR
    are a subphrase/duplicated-token fold of one another (issue #89: GDINO phrase
    decode fragments -- "potted"/"plant" vs "potted plant", "door door" vs "door" --
    see :func:`~core.perception.scene_index.labels_foldable`). Association is already
    centroid-gated (the caller only considers pairs within ``cfg.gate``), so folding
    here only ever widens which CO-LOCATED detection can join an existing track; it
    never on its own decides two spatially-unrelated detections are the same object."""
    return canonical_for_match(a) == canonical_for_match(b) or labels_foldable(a, b)


# --------------------------------------------------------------------------- config


@dataclass(frozen=True)
class TrackerConfig:
    """Association gate. Non-spec value flagged in the task report."""

    gate: float = 0.75  # m; max centroid distance for a match
    # H15(a) track decay: an instance still at n_obs==1 that has not been re-observed
    # within this many keyframes of first sighting is a one-frame ghost and is pruned.
    # Confirmed tracks (n_obs>=2) are NEVER decayed. 0 disables decay.
    decay_k: int = 5


DEFAULT_TRACKER_CONFIG = TrackerConfig()


@dataclass(frozen=True)
class KeyframeConfig:
    """Keyframe gate tunables (geometry-Thresholds style, one dataclass).

    Process a frame when it is at least ``every_k`` frames since the last keyframe,
    OR the vehicle has moved ``min_translation`` metres, OR turned ``min_rotation``
    radians, since the last processed frame.
    """

    every_k: int = 1
    min_translation: float = 0.5           # m
    min_rotation: float = np.deg2rad(30.0)  # rad


DEFAULT_KEYFRAME_CONFIG = KeyframeConfig()


# --------------------------------------------------------------------------- tracker


def _fused_to_record(det: Detection, fused: Fused3D, instance_id: int) -> InstanceRecord:
    """Build an InstanceRecord from a fused detection (AABB from its point set).

    The scene index recomputes a trimmed AABB on merge; here we seed a raw
    min/max box + centroid from the cluster so a brand-new instance is well-formed.
    """
    pts = fused.points
    aabb_min = pts.min(axis=0).astype(float)
    aabb_max = pts.max(axis=0).astype(float)
    canon = normalize_label(det.label)
    aliases = NOUN_ALIASES.get(canon, ())
    return InstanceRecord(
        instance_id=instance_id,
        label=canon,
        score=float(det.score),
        n_obs=1,
        centroid=fused.centroid.astype(float),
        aabb_min=aabb_min,
        aabb_max=aabb_max,
        points=pts.astype(float),
        aliases=tuple(aliases),
    )


def associate(
    fused_dets: list[tuple[Detection, Fused3D]],
    index: BasicSceneIndex,
    cfg: TrackerConfig = DEFAULT_TRACKER_CONFIG,
) -> list[int]:
    """Greedy-nearest associate fused detections to instances; update the index.

    Returns the list of instance_ids touched (matched-into or newly created), in the
    order the detections were processed.
    """
    existing = index.all_instances()

    # Build all label-compatible (det_i, inst_j, distance) candidate pairs under gate.
    pairs: list[tuple[float, int, int]] = []
    for di, (det, fused) in enumerate(fused_dets):
        for ej, inst in enumerate(existing):
            if not labels_compatible(det.label, inst.label):
                continue
            dist = float(np.linalg.norm(fused.centroid - inst.centroid))
            if dist <= cfg.gate:
                pairs.append((dist, di, ej))
    pairs.sort(key=lambda p: p[0])

    matched_det: dict[int, InstanceRecord] = {}
    used_inst: set[int] = set()
    for dist, di, ej in pairs:
        if di in matched_det or ej in used_inst:
            continue
        matched_det[di] = existing[ej]
        used_inst.add(ej)

    touched: list[int] = []
    for di, (det, fused) in enumerate(fused_dets):
        if di in matched_det:
            target = matched_det[di]
            rec = _fused_to_record(det, fused, instance_id=target.instance_id)
            # Issue #89/#84: trust THIS association's own match decision (alias-bridged
            # label compatibility + centroid gate, both already checked above) and fuse
            # directly into `target` — do NOT hand off to index.add(), whose independent
            # label/IoU re-derivation can (and under live pose jitter routinely does)
            # disagree with this decision and silently mint a duplicate instance instead
            # of fusing (see BasicSceneIndex.merge_into's docstring for the full story).
            survivor = index.merge_into(target.instance_id, rec)
            touched.append(survivor.instance_id)
        else:
            new_id = index.next_id()
            rec = _fused_to_record(det, fused, instance_id=new_id)
            survivor = index.add(rec)
            touched.append(survivor.instance_id)
    return touched


def decay_singletons(
    index: BasicSceneIndex,
    first_seen: dict[int, int],
    keyframe_idx: int,
    decay_k: int,
) -> list[int]:
    """Prune one-frame ghosts (H15a): remove n_obs==1 instances not re-observed in time.

    An instance whose ``n_obs`` is still 1 ``decay_k`` keyframes after it was first
    seen never got a second observation — a spurious single-frame detection. It is
    removed from the index. Instances with ``n_obs >= 2`` are confirmed tracks and are
    NEVER decayed, regardless of age. ``first_seen`` maps instance_id -> the keyframe
    index at which it was minted; stale entries for removed/absent ids are cleaned up.

    Returns the list of pruned instance_ids. ``decay_k <= 0`` disables decay.
    """
    if decay_k <= 0:
        return []
    live = {r.instance_id: r for r in index.all_instances()}
    pruned: list[int] = []
    for iid in list(first_seen):
        rec = live.get(iid)
        if rec is None:
            del first_seen[iid]  # already gone (merged away / previously pruned)
            continue
        if rec.n_obs >= 2:
            del first_seen[iid]  # confirmed — stop tracking its age, never decays
            continue
        if keyframe_idx - first_seen[iid] >= decay_k:
            if index.remove(iid):
                pruned.append(iid)
            del first_seen[iid]
    return pruned


# --------------------------------------------------------------------------- pipeline


class PerceptionPipeline:
    """Glue: tiling -> detector -> fusion -> tracker, behind a keyframe gate.

    ``process(pano, scan)`` returns the list of instance_ids updated on that frame,
    or an empty list when the keyframe gate skips the frame. Instances accumulate in
    ``self.index`` (a :class:`BasicSceneIndex`).
    """

    def __init__(
        self,
        detector: DetectorFn,
        *,
        index: BasicSceneIndex | None = None,
        fusion_cfg: FusionConfig = DEFAULT_FUSION_CONFIG,
        tracker_cfg: TrackerConfig = DEFAULT_TRACKER_CONFIG,
        keyframe_cfg: KeyframeConfig = DEFAULT_KEYFRAME_CONFIG,
        n_tiles: int = DEFAULT_N_TILES,
        hfov: float = DEFAULT_TILE_HFOV,
        vfov: float = DEFAULT_TILE_VFOV,
    ) -> None:
        self.detector = detector
        self.index = index if index is not None else BasicSceneIndex()
        self.fusion_cfg = fusion_cfg
        self.tracker_cfg = tracker_cfg
        self.keyframe_cfg = keyframe_cfg
        self.n_tiles = n_tiles
        self.hfov = hfov
        self.vfov = vfov
        self._frame_count = 0
        self._last_keyframe: OdomState | None = None
        self._keyframe_idx = 0                    # keyframes processed
        self._det_kf_idx = 0                      # DETECTION-BEARING keyframes (H15a decay clock)
        self._first_seen: dict[int, int] = {}     # instance_id -> det-keyframe it was minted
        # Issues #84/#89 instrumentation: question-clock time (the PanoFrame's own `t`,
        # not wall-clock) of the last periodic instance-index dump (None -> unconditionally
        # dumps once VLA_INSTANCE_DUMP_PATH is set). Keeping this off frame time rather
        # than wall-clock stays deterministic/testable and matches
        # core.heads.explore_debug's throttle style.
        self._last_dump_t: float | None = None

    def _is_keyframe(self, odom: OdomState) -> bool:
        kf = self.keyframe_cfg
        if self._last_keyframe is None:
            return True
        dt_frames = self._frame_count - self._last_frame_idx
        moved = np.hypot(odom.x - self._last_keyframe.x, odom.y - self._last_keyframe.y)
        turned = abs(float(np.arctan2(
            np.sin(odom.yaw - self._last_keyframe.yaw),
            np.cos(odom.yaw - self._last_keyframe.yaw),
        )))
        return (
            dt_frames >= kf.every_k
            or moved >= kf.min_translation
            or turned >= kf.min_rotation
        )

    def process(self, pano: PanoFrame, scan: LidarScan) -> list[int]:
        """Run one frame through the pipeline; return updated instance_ids."""
        odom = pano.odom
        idx_before = self._frame_count
        self._frame_count += 1
        if not self._is_keyframe(odom):
            return []
        self._last_keyframe = odom
        self._last_frame_idx = idx_before

        tiles = project_tiles(pano.image, self.n_tiles, self.hfov, self.vfov)
        per_tile = self.detector(tiles)

        # Issue #84: opt-in raw pre-gate dump. The env lookup is the only cost paid
        # when unset (matches _maybe_dump_instances' cost contract); raw_records stays
        # empty and dump_raw_detections() below no-ops on its own env check anyway, but
        # skipping the list-building here too avoids paying for it at all when off.
        dump_raw = bool(os.environ.get(ENV_RAW_DETECTION_DUMP_PATH))
        raw_records: list[list] = []  # [Detection, gate, instance_id] triples (mutable placeholder)

        fused_dets: list[tuple[Detection, Fused3D]] = []
        for tile_dets in per_tile:
            for det in tile_dets:
                fused = fuse_detection(
                    det, scan, odom, self.fusion_cfg,
                    n_tiles=self.n_tiles, hfov=self.hfov, vfov=self.vfov,
                )
                if fused is not None:
                    fused_dets.append((det, fused))
                    if dump_raw:
                        raw_records.append([det, GATE_ACCEPTED, None])
                elif dump_raw:
                    raw_records.append([det, GATE_NO_LIDAR_CLUSTER, None])

        touched = associate(fused_dets, self.index, self.tracker_cfg)

        if dump_raw:
            # Back-fill the instance id each accepted detection landed in: associate()
            # returns `touched` in fused_dets order, and raw_records' GATE_ACCEPTED
            # entries were appended in that exact same order above.
            fused_i = 0
            for rec in raw_records:
                if rec[1] == GATE_ACCEPTED:
                    rec[2] = touched[fused_i]
                    fused_i += 1
            dump_raw_detections(
                [(d, g, iid) for d, g, iid in raw_records],
                keyframe_idx=self._keyframe_idx,
            )

        # H15a: record first sighting for any newly minted instance, then decay
        # one-frame ghosts that never got a second look. The decay clock counts
        # DETECTION-BEARING keyframes only: a keyframe on which the detector saw
        # nothing at all is no evidence against a singleton (cold start, occlusion,
        # scripted single-keyframe labels) — pruning requires decay_k keyframes on
        # which the detector demonstrably produced detections yet never re-observed
        # this instance.
        for iid in touched:
            self._first_seen.setdefault(iid, self._det_kf_idx)
        if fused_dets:
            self._det_kf_idx += 1
        decay_singletons(
            self.index, self._first_seen, self._det_kf_idx, self.tracker_cfg.decay_k
        )
        self._keyframe_idx += 1
        self._maybe_dump_instances(pano.t)
        return touched

    def _maybe_dump_instances(self, t: float) -> None:
        """Issues #84/#89: throttled periodic instance-index dump (opt-in via
        ``VLA_INSTANCE_DUMP_PATH``; no-op — not even the env lookup's cost matters,
        this is one dict-get per keyframe — when unset)."""
        if not os.environ.get(ENV_INSTANCE_DUMP_PATH):
            return
        try:
            interval = float(
                os.environ.get(ENV_INSTANCE_DUMP_INTERVAL_S, DEFAULT_INSTANCE_DUMP_INTERVAL_S)
            )
        except (TypeError, ValueError):
            interval = DEFAULT_INSTANCE_DUMP_INTERVAL_S
        if self._last_dump_t is not None and (t - self._last_dump_t) < interval:
            return
        self._last_dump_t = t
        dump_instance_index(self.index, tag="periodic", keyframes_processed=self._keyframe_idx)

    # convenience for tests / callers
    @property
    def frame_count(self) -> int:
        return self._frame_count

    # track last keyframe frame index for the every_k gate
    _last_frame_idx: int = -(10 ** 9)
