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

from dataclasses import dataclass, field

import numpy as np

from core.interfaces import InstanceRecord, LidarScan, OdomState, PanoFrame
from core.parsing.vocab import NOUN_ALIASES
from core.perception.detector import Detection, DetectorFn
from core.perception.fusion import (
    DEFAULT_FUSION_CONFIG,
    Fused3D,
    FusionConfig,
    fuse_detection,
)
from core.perception.scene_index import BasicSceneIndex, normalize_label
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
    """True when two labels denote the same canonical noun (alias/synonym-aware)."""
    return canonical_for_match(a) == canonical_for_match(b)


# --------------------------------------------------------------------------- config


@dataclass(frozen=True)
class TrackerConfig:
    """Association gate. Non-spec value flagged in the task report."""

    gate: float = 0.75  # m; max centroid distance for a match


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
            # add() finds the same-label IoU-overlapping target and fuses in place.
            survivor = index.add(rec)
            touched.append(survivor.instance_id)
        else:
            new_id = index.next_id()
            rec = _fused_to_record(det, fused, instance_id=new_id)
            survivor = index.add(rec)
            touched.append(survivor.instance_id)
    return touched


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

        fused_dets: list[tuple[Detection, Fused3D]] = []
        for tile_dets in per_tile:
            for det in tile_dets:
                fused = fuse_detection(
                    det, scan, odom, self.fusion_cfg,
                    n_tiles=self.n_tiles, hfov=self.hfov, vfov=self.vfov,
                )
                if fused is not None:
                    fused_dets.append((det, fused))

        return associate(fused_dets, self.index, self.tracker_cfg)

    # convenience for tests / callers
    @property
    def frame_count(self) -> int:
        return self._frame_count

    # track last keyframe frame index for the every_k gate
    _last_frame_idx: int = -(10 ** 9)
