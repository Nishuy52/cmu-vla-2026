"""Scripted (hand-labelled) 2D detections replayed onto the tile-space pipeline.

The real Phase-2 detector is a GroundingDINO-class open-vocab model
(:class:`~core.perception.detector.GroundingDinoDetector`); it runs on the pinhole
*tiles* :func:`~core.perception.tiling.project_tiles` carves from the panorama and
returns :class:`~core.perception.detector.Detection` boxes in *tile* pixel space.

For grounding on REAL data without weights, we replay hand-labelled boxes that were
drawn on the raw 1920x640 equirectangular panorama (``data/fixtures/jingfan_labels.json``
schema: ``keyframes[str(kf_index)] -> [{label, attributes, bbox: [x0,y0,x1,y1], score}]``,
pano-pixel xyxy). :class:`ScriptedPanoDetector` converts those pano boxes into the exact
tile-space :class:`Detection` objects the pipeline expects, so fusion sees real lidar
against real (if approximate) 2D boxes.

Conversion, per box (reusing the tiling module's projection — never reimplementing trig):

1. **Pick the tile.** The bbox centre column -> camera azimuth via
   :func:`~core.perception.tiling.column_to_azimuth`; choose the tile whose optical axis
   (``TileSpec.yaw_center``) is angularly nearest that azimuth (the tile whose 90deg HFOV
   contains it). A box straddling a tile seam is therefore assigned by its centre.
2. **Project the corners.** Each pano corner (col,row) -> camera ``(azimuth, elevation)``
   ray (``column_to_azimuth`` / ``row_to_elevation``), then the inverse of that tile's
   gnomonic projection (using the tile's own ``focal_x``/``focal_y``/``cx``/``cy`` from
   :class:`~core.perception.tiling.TileSpec`) places the ray at a tile pixel. This is the
   exact inverse of :func:`~core.perception.tiling.tile_pixel_to_camera_ray`.

The corner pixels are re-min/maxed into an axis-aligned tile bbox (the gnomonic map is
monotonic within a tile but can mildly rotate a box near the edges, so we take the
enclosing axis-aligned box). Boxes are clipped to the tile bounds.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from core.perception.detector import Detection
from core.perception.tiling import (
    DEFAULT_N_TILES,
    DEFAULT_TILE_HFOV,
    DEFAULT_TILE_VFOV,
    TileSpec,
    column_to_azimuth,
    row_to_elevation,
    tile_specs,
    wrap_pi,
)


def camera_ray_to_tile_pixel(spec: TileSpec, azimuth: float, elevation: float) -> tuple[float, float]:
    """Camera-frame ``(azimuth, elevation)`` ray -> tile pixel ``(u, v)`` for ``spec``.

    Exact analytic inverse of :func:`~core.perception.tiling.tile_pixel_to_camera_ray`
    (which does ``x=(u-cx)/fx``, ``local_az=atan2(-x, 1)``,
    ``local_el=atan2(-y, hypot(x,1))``). We invert those, reusing the tile's own focal
    lengths / principal point from :class:`~core.perception.tiling.TileSpec` — no fresh
    trig conventions, just the algebraic inverse of the module's forward gnomonic map.
    """
    local_az = float(wrap_pi(azimuth - spec.yaw_center))
    # local_az = atan2(-x, z) with z = 1  ->  x = -tan(local_az)
    x = -np.tan(local_az)
    # local_el = atan2(-y, hypot(x, z))  ->  y = -tan(local_el) * hypot(x, 1)
    y = -np.tan(elevation) * np.hypot(x, 1.0)
    u = x * spec.focal_x + spec.cx
    v = y * spec.focal_y + spec.cy
    return float(u), float(v)


def _select_tile(center_col: float, specs: tuple[TileSpec, ...]) -> TileSpec:
    """Tile whose optical axis is angularly nearest the column's camera azimuth."""
    az = float(column_to_azimuth(center_col))
    return min(specs, key=lambda s: abs(float(wrap_pi(az - s.yaw_center))))


def pano_bbox_to_detection(
    label: str,
    bbox_xyxy: tuple[float, float, float, float],
    score: float,
    specs: tuple[TileSpec, ...],
) -> Detection:
    """Convert one pano-pixel xyxy box to a tile-space :class:`Detection`.

    The tile is chosen by the box centre (so seam-spanning boxes go to the tile holding
    the centre); the four corners are gnomonically projected into that tile and the
    enclosing axis-aligned box (clipped to tile bounds) becomes ``bbox_xyxy``.
    """
    x0, y0, x1, y1 = (float(v) for v in bbox_xyxy)
    center_col = 0.5 * (x0 + x1)
    spec = _select_tile(center_col, specs)

    us: list[float] = []
    vs: list[float] = []
    for col, row in ((x0, y0), (x1, y0), (x0, y1), (x1, y1)):
        az = float(column_to_azimuth(col))
        el = float(row_to_elevation(row))
        u, v = camera_ray_to_tile_pixel(spec, az, el)
        us.append(u)
        vs.append(v)

    tu0 = float(np.clip(min(us), 0.0, spec.width - 1))
    tu1 = float(np.clip(max(us), 0.0, spec.width - 1))
    tv0 = float(np.clip(min(vs), 0.0, spec.height - 1))
    tv1 = float(np.clip(max(vs), 0.0, spec.height - 1))
    return Detection(
        tile_id=spec.tile_id,
        bbox_xyxy=(tu0, tv0, tu1, tv1),
        label=str(label),
        score=float(score),
    )


@dataclass(frozen=True)
class _LabelSet:
    """Parsed labels file: keyframe index -> raw pano-box dicts."""

    keyframes: dict[int, list[dict]]


class ScriptedPanoDetector:
    """Replays hand-labelled pano boxes as tile-space detections, one keyframe per call.

    Load from the ``jingfan_labels.json`` schema (via :meth:`from_json` or the ``path``
    ctor arg). :meth:`detections_for` returns the converted :class:`Detection` list for a
    given keyframe index; :attr:`keyframes` exposes the labelled keyframe->box mapping so
    the runner can feed the right frame.

    As a :data:`~core.perception.detector.DetectorFn` the instance is *stateful*: the
    runner sets :attr:`current_keyframe` before each keyframe's ``pipeline.process`` call,
    and ``__call__(tiles)`` returns that keyframe's per-tile detections (empty lists for
    an unlabelled keyframe). This keeps the pipeline seam unchanged — it still calls the
    detector with tiles and gets per-tile detection lists back.
    """

    def __init__(
        self,
        labels: _LabelSet,
        *,
        n_tiles: int = DEFAULT_N_TILES,
        hfov: float = DEFAULT_TILE_HFOV,
        vfov: float = DEFAULT_TILE_VFOV,
    ) -> None:
        self._labels = labels
        self.n_tiles = int(n_tiles)
        self._specs = tile_specs(n_tiles, hfov, vfov)
        self.current_keyframe: int | None = None

    # ------------------------------------------------------------------ loading

    @classmethod
    def from_json(cls, path, **kwargs) -> "ScriptedPanoDetector":
        with open(str(path), encoding="utf-8") as fh:
            raw = json.load(fh)
        fmt = raw.get("format")
        if fmt not in (None, "pano_xyxy"):
            raise ValueError(f"unsupported labels format {fmt!r} (expected 'pano_xyxy')")
        kfs: dict[int, list[dict]] = {}
        for key, boxes in raw.get("keyframes", {}).items():
            kfs[int(key)] = list(boxes)
        return cls(_LabelSet(keyframes=kfs), **kwargs)

    # --------------------------------------------------------------- accessors

    @property
    def keyframes(self) -> dict[int, list[Detection]]:
        """Full keyframe-index -> converted tile-space detections mapping."""
        return {kf: self.detections_for(kf) for kf in self._labels.keyframes}

    @property
    def labelled_indices(self) -> tuple[int, ...]:
        """Sorted keyframe indices that carry labels."""
        return tuple(sorted(self._labels.keyframes))

    def detections_for(self, keyframe: int) -> list[Detection]:
        """Converted tile-space detections for ``keyframe`` (``[]`` if unlabelled)."""
        boxes = self._labels.keyframes.get(int(keyframe))
        if not boxes:
            return []
        out: list[Detection] = []
        for box in boxes:
            out.append(
                pano_bbox_to_detection(
                    box["label"], tuple(box["bbox"]), box.get("score", 1.0), self._specs
                )
            )
        return out

    # --------------------------------------------------------------- detector seam

    def per_tile(self, keyframe: int) -> list[list[Detection]]:
        """Detections for ``keyframe`` routed into per-tile lists (pipeline shape)."""
        out: list[list[Detection]] = [[] for _ in range(self.n_tiles)]
        for det in self.detections_for(keyframe):
            if 0 <= det.tile_id < self.n_tiles:
                out[det.tile_id].append(det)
        return out

    def __call__(self, tiles) -> list[list[Detection]]:
        n = len(tiles)
        if self.current_keyframe is None:
            return [[] for _ in range(n)]
        dets = self.detections_for(self.current_keyframe)
        out: list[list[Detection]] = [[] for _ in range(n)]
        for det in dets:
            if 0 <= det.tile_id < n:
                out[det.tile_id].append(det)
        return out
