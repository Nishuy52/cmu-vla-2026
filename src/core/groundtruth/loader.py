"""Load a VLA-3D scene folder into the core world model.

A VLA-3D "Unity scene" (e.g. ``loft``) ships a fixed per-scene file set (see
``docs/vla3d_notes.md`` §2). This loader consumes the three that carry the
geometry + semantics the answer heads need:

* ``<scene>_object_result.csv`` — one row per object: ``raw_label`` (the free-text
  label that matches challenge question nouns verbatim), an *oriented* bounding box
  (center + xyz extents + heading yaw), and up to three dominant colours.
* ``<scene>_region_result.csv`` — one row per region (room): id, label, AABB.
* ``<scene>_scene_graph.json`` — the object->region membership map (used for the
  ``in`` predicate — "the chair in the bedroom").

Each object row becomes an :class:`~core.interfaces.InstanceRecord`:

* ``label``   = ``raw_label`` (verbatim, lowercased) — the notes confirm challenge
  question nouns match ``raw_label`` exactly, not the coarser NYU labels.
* ``aabb_min/aabb_max`` = the **AABB-of-OBB approximation** (see :func:`obb_to_aabb`):
  the object's oriented box is rotated by its heading, then the axis-aligned min/max
  over the 8 rotated corners is taken. This is a strict *over*-approximation — a box
  rotated 45 deg yields an AABB up to ~sqrt(2) larger per axis. It is the honest
  cost of the core pipeline being axis-aligned-only (:class:`MarkerBox` is an AABB);
  documented here so IoU scores are read with that inflation in mind.
* ``n_obs`` = 3 — ground truth is *fully observed*, so every instance clears the
  ``n_obs >= 3`` confidence gate the heads apply.
* ``points`` = None — no per-object point cloud is loaded (the AABB is exact enough
  for the toolbox predicates; loading the ~2 GB point cloud is out of scope).
* ``caption`` = the dominant colour names + a coarse size token, so the toolbox's
  keyword attribute matcher (``_attrs_match`` scans ``label + caption + aliases``)
  can satisfy "the **blue** chair" / "the **big** table" style attributes.

No network, no heavy deps: CSV via :mod:`csv` (stdlib), JSON via :mod:`json`.
"""
from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

from core.interfaces import ColorBin, InstanceRecord

# The 15 canonical VLA-3D colour names are already human words; we attach them
# verbatim. A very coarse size bucket is derived from AABB volume so size
# attributes ("big"/"small"/"large") in questions can match.
_SIZE_LARGE_VOL = 0.5  # m^3 AABB volume at/above which an object reads "big"/"large"
_SIZE_SMALL_VOL = 0.02  # m^3 AABB volume at/below which an object reads "small"


@dataclass
class GTRegion:
    """One VLA-3D region (room) within a scene."""

    region_id: int
    label: str  # e.g. "bedroom", "livingroom", "corridor"
    aabb_min: np.ndarray  # (3,) float
    aabb_max: np.ndarray  # (3,) float
    object_ids: list[int] = field(default_factory=list)

    def to_instance(self, instance_id: int) -> InstanceRecord:
        """Represent the region itself as an InstanceRecord (label == room name).

        Lets ``in``-predicate resolution treat "the chair in the bedroom" with the
        room as a resolvable anchor, since the toolbox's ``in_`` predicate takes two
        InstanceRecords.
        """
        centroid = (self.aabb_min + self.aabb_max) / 2.0
        return InstanceRecord(
            instance_id=instance_id,
            label=self.label,
            score=1.0,
            n_obs=3,
            centroid=centroid,
            aabb_min=self.aabb_min.copy(),
            aabb_max=self.aabb_max.copy(),
            points=None,
            caption="region",
        )


@dataclass
class GTScene:
    """A fully-observed ground-truth scene loaded from a VLA-3D folder."""

    scene_name: str
    instances: list[InstanceRecord]
    regions: list[GTRegion]
    #: object_id -> region_id (from the scene graph / object CSV region column).
    object_region: dict[int, int] = field(default_factory=dict)

    def region_of(self, object_id: int) -> GTRegion | None:
        rid = self.object_region.get(object_id)
        if rid is None:
            return None
        for r in self.regions:
            if r.region_id == rid:
                return r
        return None


# --------------------------------------------------------------------------- OBB -> AABB


def obb_to_aabb(
    center: Iterable[float],
    extents: Iterable[float],
    heading: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Axis-aligned bounding box of an oriented box (AABB-of-OBB approximation).

    The oriented box is centred at ``center`` with full extents ``extents``
    (x/y/z lengths) and yaw ``heading`` (radians, rotation about +Z). We build the
    8 corners in the box's local frame, rotate them by ``heading`` about Z, translate
    to ``center``, and take the per-axis min/max.

    This is an *over*-approximation: for a heading that is not a multiple of 90 deg
    the resulting AABB is strictly larger than the true footprint (worst case near
    45 deg, footprint diagonal grows by ~sqrt(2)). Z is unaffected (rotation is about
    Z only). Callers scoring IoU against this box should treat small IoU deficits on
    rotated objects as partly this approximation, not pipeline error.
    """
    c = np.asarray(list(center), dtype=float).reshape(3)
    e = np.asarray(list(extents), dtype=float).reshape(3)
    half = e / 2.0
    # 8 local corners (±half per axis)
    signs = np.array(
        [
            [sx, sy, sz]
            for sx in (-1.0, 1.0)
            for sy in (-1.0, 1.0)
            for sz in (-1.0, 1.0)
        ]
    )
    local = signs * half  # (8, 3)
    cos_h, sin_h = math.cos(heading), math.sin(heading)
    rot = np.array(
        [
            [cos_h, -sin_h, 0.0],
            [sin_h, cos_h, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    world = local @ rot.T + c  # (8, 3)
    return world.min(axis=0), world.max(axis=0)


# --------------------------------------------------------------------------- colour/size


def _dominant_colors(row: dict[str, str]) -> list[str]:
    """The up-to-3 dominant colour scheme names for an object row ('_' filtered)."""
    out: list[str] = []
    for i in (1, 2, 3):
        name = (row.get(f"object_color_scheme{i}") or "").strip()
        if name and name != "_" and name.lower() != "n/a":
            out.append(name.lower())
    return out


def _color_bins(row: dict[str, str]) -> tuple[ColorBin, ...]:
    """The up-to-3 dominant colour bins as (scheme name, raw RGB, fraction).

    Reads the paired ``object_color_r/g/b{i}`` + ``object_color_scheme{i}`` +
    ``object_color_scheme_percentage{i}`` columns. Carries the raw RGB and fraction
    the scheme name alone loses, so colour matching can apply luminance / dominance
    salience (issues #11/#12). A bin is included only when it has a scheme name AND
    parseable RGB (kept in lock-step with :func:`_dominant_colors`)."""
    bins: list[ColorBin] = []
    for i in (1, 2, 3):
        name = (row.get(f"object_color_scheme{i}") or "").strip().lower()
        if not name or name == "_" or name == "n/a":
            continue
        try:
            rgb = (
                int(float(row[f"object_color_r{i}"])),
                int(float(row[f"object_color_g{i}"])),
                int(float(row[f"object_color_b{i}"])),
            )
        except (KeyError, TypeError, ValueError):
            continue
        try:
            frac = float(row.get(f"object_color_scheme_percentage{i}") or 0.0)
        except (TypeError, ValueError):
            frac = 0.0
        bins.append(ColorBin(name=name, rgb=rgb, fraction=frac))
    return tuple(bins)


def _size_token(extents: np.ndarray) -> str:
    """Coarse size word from AABB volume, so 'big'/'small' attributes can match."""
    vol = float(np.prod(np.clip(extents, 0.0, None)))
    if vol >= _SIZE_LARGE_VOL:
        return "big large"
    if vol <= _SIZE_SMALL_VOL:
        return "small"
    return "medium"


def _build_caption(colors: list[str], extents: np.ndarray) -> str:
    """Caption text scanned by the toolbox attribute matcher (colours + size)."""
    parts = list(colors)
    parts.append(_size_token(extents))
    return " ".join(parts)


# --------------------------------------------------------------------------- CSV parse


def parse_object_csv(path: os.PathLike | str) -> list[InstanceRecord]:
    """Parse a ``<scene>_object_result.csv`` into InstanceRecords (AABB-of-OBB).

    ``instance_id`` == the object's own ``object_id`` (kept stable so callers can
    cross-reference the scene graph / referential statements by index).
    """
    recs: list[InstanceRecord] = []
    with open(path, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            # Blank/comment lines occasionally appear between the header and data
            # in the sample; skip any row missing the id.
            oid_raw = (row.get("object_id") or "").strip()
            if not oid_raw:
                continue
            oid = int(float(oid_raw))
            label = (row.get("raw_label") or "").strip().lower()
            if not label:
                continue
            center = (
                float(row["object_bbox_cx"]),
                float(row["object_bbox_cy"]),
                float(row["object_bbox_cz"]),
            )
            extents = np.array(
                [
                    float(row["object_bbox_xlength"]),
                    float(row["object_bbox_ylength"]),
                    float(row["object_bbox_zlength"]),
                ]
            )
            heading = float(row.get("object_bbox_heading") or 0.0)
            amin, amax = obb_to_aabb(center, extents, heading)
            colors = _dominant_colors(row)
            caption = _build_caption(colors, extents)
            centroid = (amin + amax) / 2.0
            recs.append(
                InstanceRecord(
                    instance_id=oid,
                    label=label,
                    score=1.0,
                    n_obs=3,  # ground truth == fully observed
                    centroid=centroid,
                    aabb_min=amin,
                    aabb_max=amax,
                    points=None,
                    caption=caption,
                    aliases=tuple(colors),
                    color_bins=_color_bins(row),
                )
            )
    return recs


def parse_region_csv(path: os.PathLike | str) -> list[GTRegion]:
    """Parse ``<scene>_region_result.csv`` into GTRegions (AABB-of-OBB per region)."""
    regions: list[GTRegion] = []
    with open(path, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rid_raw = (row.get("region_id") or "").strip()
            if not rid_raw:
                continue
            rid = int(float(rid_raw))
            label = (row.get("region_label") or "").strip().lower()
            center = (
                float(row["region_bbox_cx"]),
                float(row["region_bbox_cy"]),
                float(row["region_bbox_cz"]),
            )
            extents = np.array(
                [
                    float(row["region_bbox_xlength"]),
                    float(row["region_bbox_ylength"]),
                    float(row["region_bbox_zlength"]),
                ]
            )
            heading = float(row.get("region_bbox_heading") or 0.0)
            amin, amax = obb_to_aabb(center, extents, heading)
            regions.append(GTRegion(region_id=rid, label=label, aabb_min=amin, aabb_max=amax))
    return regions


def _object_region_map(
    object_csv: os.PathLike | str,
    scene_graph: os.PathLike | str | None,
) -> dict[int, int]:
    """object_id -> region_id. Prefer the scene graph; fall back to the CSV column."""
    mapping: dict[int, int] = {}
    if scene_graph is not None and Path(scene_graph).exists():
        with open(scene_graph, encoding="utf-8") as fh:
            data = json.load(fh)
        for rid_str, region in (data.get("regions") or {}).items():
            try:
                rid = int(rid_str)
            except (TypeError, ValueError):
                rid = int(region.get("region_id", rid_str))
            for obj in region.get("objects", []):
                try:
                    oid = int(obj["object_id"])
                except (KeyError, TypeError, ValueError):
                    continue
                mapping[oid] = rid
    if mapping:
        return mapping
    # Fallback: the object CSV carries a region_id column.
    with open(object_csv, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            oid_raw = (row.get("object_id") or "").strip()
            rid_raw = (row.get("region_id") or "").strip()
            if not oid_raw or not rid_raw:
                continue
            mapping[int(float(oid_raw))] = int(float(rid_raw))
    return mapping


# --------------------------------------------------------------------------- top level


def _scene_files(folder: os.PathLike | str) -> dict[str, Path]:
    """Locate the per-scene files in a VLA-3D scene folder by suffix."""
    fdir = Path(folder)
    found: dict[str, Path] = {}
    for p in fdir.iterdir():
        name = p.name.lower()
        if name.endswith("_object_result.csv"):
            found["object_csv"] = p
        elif name.endswith("_region_result.csv"):
            found["region_csv"] = p
        elif name.endswith("_scene_graph.json"):
            found["scene_graph"] = p
        elif name.endswith("_referential_statements.json"):
            found["referential"] = p
    return found


def load_scene(folder: os.PathLike | str, scene_name: str | None = None) -> GTScene:
    """Load a VLA-3D scene folder into a :class:`GTScene`.

    ``folder`` is the per-scene directory (e.g. ``.../Unity/loft``). The scene name
    defaults to the folder basename. Requires at least the object CSV; region CSV and
    scene graph are used when present (region map degrades gracefully to the object
    CSV's ``region_id`` column).
    """
    fdir = Path(folder)
    if not fdir.is_dir():
        raise NotADirectoryError(f"not a scene folder: {fdir}")
    name = scene_name or fdir.name
    files = _scene_files(fdir)
    if "object_csv" not in files:
        raise FileNotFoundError(f"no *_object_result.csv in {fdir}")

    instances = parse_object_csv(files["object_csv"])
    regions = parse_region_csv(files["region_csv"]) if "region_csv" in files else []
    object_region = _object_region_map(files["object_csv"], files.get("scene_graph"))

    # Attach object ids to their regions for convenience.
    by_rid = {r.region_id: r for r in regions}
    for oid, rid in object_region.items():
        r = by_rid.get(rid)
        if r is not None:
            r.object_ids.append(oid)

    return GTScene(
        scene_name=name,
        instances=instances,
        regions=regions,
        object_region=object_region,
    )
