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
  question nouns match ``raw_label`` exactly, not the coarser NYU labels. A small,
  scene-and-object-id-scoped table (:data:`_SCENE_LABEL_CORRECTIONS`) additionally
  grants specific mislabelled objects an extra ``aliases`` entry when geometry
  unambiguously contradicts ``raw_label`` (issue #110, generalised in #144 via
  ``tools.gt_label_geometry_validator``) — ``label`` itself is never overwritten and
  no CSV row is touched.
* ``aabb_min/aabb_max`` = the **AABB-of-OBB approximation** (see :func:`obb_to_aabb`):
  the object's oriented box is rotated by its heading, then the axis-aligned min/max
  over the 8 rotated corners is taken. This is a strict *over*-approximation — a box
  rotated 45 deg yields an AABB up to ~sqrt(2) larger per axis. It is the honest
  cost of the core pipeline being axis-aligned-only (:class:`MarkerBox` is an AABB);
  documented here so IoU scores are read with that inflation in mind.
* ``obb_center/obb_extents/obb_heading`` = the ORIGINAL oriented box, additive fields
  carried alongside the AABB approximation (issue #77, Pre-Stage 1a). Every existing
  scoring/toolbox geometry predicate keeps reading ``aabb_min/aabb_max`` unchanged
  (frozen semantics); only the mirror-costmap stamping path
  (``core.runner.gt_battery._synthetic_from_gt``) reads these three, to rasterize the
  true rotated footprint instead of its AABB hull.
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
import logging
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np

from core.interfaces import ColorBin, InstanceRecord

_LOGGER = logging.getLogger(__name__)

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


def _color_slots(
    row: dict[str, str], *, source: str, line: int
) -> list[tuple[str, tuple[int, int, int], float]]:
    """The up-to-3 dominant-colour slots for an object row, as (name, raw RGB,
    fraction) — the single source of truth for BOTH :func:`_dominant_colors`
    (aliases/caption) and :func:`_color_bins` (:class:`ColorBin` list), so the two
    can never desync (issue #27).

    Reads the paired ``object_color_scheme{i}`` + ``object_color_r/g/b{i}`` +
    ``object_color_scheme_percentage{i}`` columns for ``i`` in 1..3.

    A slot with no scheme name (blank / ``_`` / ``n/a``) is simply ABSENT — not
    corrupt — and is skipped quietly. A slot WITH a scheme name but an unparseable
    RGB, or a *present-but-malformed* percentage, is CORRUPT: the whole slot (name +
    RGB + percentage together) is dropped from BOTH the alias list and the
    ColorBin list, with a loud warning naming the source file, CSV line, and slot
    index. Previously the RGB failure silently desynced the two structures (the name
    kept surviving as an alias while the bin vanished — a colour then unmatchable via
    either the alias or the bin path), and a malformed percentage silently collapsed
    to ``fraction=0.0`` ('0% of object', not 'unknown'), which can never clear
    ``colour_dominance_floor`` and so silently defeated colour salience (#11/#12).

    A genuinely MISSING percentage field (column absent or blank) is not malformed —
    it legitimately defaults to ``0.0``, unchanged from prior behaviour.
    """
    out: list[tuple[str, tuple[int, int, int], float]] = []
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
        except (KeyError, TypeError, ValueError) as exc:
            _LOGGER.warning(
                "%s:%d: corrupt colour slot %d (scheme=%r): unparseable RGB (%s) — "
                "dropping this slot from both aliases and color_bins to avoid desync",
                source, line, i, name, exc,
            )
            continue
        raw_pct = row.get(f"object_color_scheme_percentage{i}")
        if raw_pct is None or not str(raw_pct).strip():
            frac = 0.0  # legitimately absent, not malformed
        else:
            try:
                frac = float(raw_pct)
            except (TypeError, ValueError) as exc:
                _LOGGER.warning(
                    "%s:%d: corrupt colour slot %d (scheme=%r): unparseable "
                    "percentage %r (%s) — dropping this slot from both aliases and "
                    "color_bins instead of silently collapsing to 0.0 (which would "
                    "defeat colour-dominance salience)",
                    source, line, i, name, raw_pct, exc,
                )
                continue
        out.append((name, rgb, frac))
    return out


def _dominant_colors(
    row: dict[str, str], *, source: str = "<unknown>", line: int = 0
) -> list[str]:
    """The up-to-3 dominant colour scheme names for an object row (corrupt slots
    excluded — see :func:`_color_slots`)."""
    return [name for name, _rgb, _frac in _color_slots(row, source=source, line=line)]


def _color_bins(
    row: dict[str, str], *, source: str = "<unknown>", line: int = 0
) -> tuple[ColorBin, ...]:
    """The up-to-3 dominant colour bins as (scheme name, raw RGB, fraction).

    Carries the raw RGB and fraction the scheme name alone loses, so colour matching
    can apply luminance / dominance salience (issues #11/#12). Stays in lock-step
    with :func:`_dominant_colors` by construction — both draw from the same
    :func:`_color_slots` pass (issue #27)."""
    return tuple(
        ColorBin(name=name, rgb=rgb, fraction=frac)
        for name, rgb, frac in _color_slots(row, source=source, line=line)
    )


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
    source = str(path)
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
            obb_center = np.asarray(center, dtype=float)
            # One shared pass over the colour slots (issue #27): computing colors and
            # bins separately from independent row scans (the pre-fix shape) is what
            # let a corrupt slot survive in one structure while vanishing from the
            # other. line_num counts the header, so this is the 1-based physical CSV
            # line of the current data row.
            slots = _color_slots(row, source=source, line=reader.line_num)
            colors = [name for name, _rgb, _frac in slots]
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
                    color_bins=tuple(
                        ColorBin(name=name, rgb=rgb, fraction=frac)
                        for name, rgb, frac in slots
                    ),
                    obb_center=obb_center,
                    obb_extents=extents,
                    obb_heading=heading,
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


# --------------------------------------------------------------------------- scene-scoped label corrections

#: Ground-truth ANNOTATION errors where an object's geometry unambiguously
#: contradicts its ``raw_label``, corrected without editing the frozen VLA-3D CSV
#: (the GT is the scoring substrate; hand-editing it would silently change every
#: historical number). Each entry keys on the SPECIFIC ``(scene_name, object_id)``
#: and adds the corrected noun as an extra alias on that one InstanceRecord —
#: ``raw_label``/``label`` stay verbatim, and ``BasicSceneIndex.by_label_tiered``
#: matches ``aliases`` by exact string only (never fuzzily), so this only ever adds
#: one more way to reach the one object named, never a class-wide bridge.
#:
#: Issue #110: livingroom_1 object_id 91 carries ``raw_label == "tv remote"`` (VLA-3D
#: ``nyu_label`` "remote control") but its OBB is a 1.697 x 0.019 x 0.976 m panel
#: centred at z=1.385m, at essentially the same (x, y) as the tv cabinet (object_id
#: 103, z 0.28-0.56m) — a wall-mounted television, not a remote: no handheld remote is
#: 1.7 m wide or mounted 1.4 m up. The SAME scene also has object_id 72, also labelled
#: ``"tv remote"``, whose OBB (0.204 x 0.051 x 0.019 m, centre z=0.28m — tabletop
#: height, remote-sized) genuinely IS a remote. That pair is exactly why the fix
#: cannot be a class-wide ``"tv remote" -> "tv"`` bridge in
#: :mod:`core.perception.vocab` (VOCAB_BRIDGE): a blanket bridge would make a "tv"
#: query in this scene return BOTH objects (or, matched the other way, could make a
#: "remote" query second-guess a genuine remote elsewhere). Keying on
#: ``(scene_name, object_id)`` is the narrowest correction that resolves the
#: mislabelled panel to "tv" while leaving object_id 72 exactly as annotated.
#: Issue #144: the class of defect behind #110, found by
#: :mod:`tools.gt_label_geometry_validator` (an offline scan that flags any raw-label
#: instance whose true OBB volume — never the AABB-of-OBB, which is a deliberate
#: over-approximation for rotated boxes — sits >=12x or <=1/12x its class's
#: :mod:`core.perception.dimension_priors` typical volume, restricted to labels with
#: >=3 instances pooled across all 15 scenes so there is a "family" to be out of).
#: Every entry below was individually verified against its raw geometry (position,
#: footprint, height, and — for the repeated case — cross-scene consistency) before
#: being added; the validator only proposes candidates, it never auto-applies one.
#:
#: livingroom_2 object_id 53, raw_label "clock": a 1.038 x 0.062 x 1.062 m panel
#: centred at z=2.09m (spanning ~1.56-2.62m, near the ceiling) on the same wall run as
#: the scene's genuine "tv" (object_ids 8/9) and "painting" (object_id 73) instances.
#: A 1 m square "clock" mounted near-ceiling is implausible to read as a clock; its
#: geometry (thin near-square wall panel) fits the scene's own "mirror" class far
#: better (volume ~2.2x the dimension_priors mirror-typical vs. ~13.5x the
#: clock-typical) -- corrected to "mirror".
#:
#: studio object_id 46, raw_label "vase": 0.296 x 0.384 x 2.493 m, floor-to-near-
#: ceiling height. Volume is 21x the vase-typical (an impossible vase) but sits almost
#: exactly on the dimension_priors "column" typical on every axis (thin/mid/long all
#: within ~15% of the column median) -- corrected to "column".
#:
#: studio object_id 6, raw_label "hanger": 0.457 x 0.438 x 1.731 m, a furniture-scale
#: footprint and shoulder height. A coat/garment "hanger" is a small, paper-thin,
#: handheld item -- physically incompatible with this box regardless of prior (no
#: dimension_priors entry exists for "hanger" at all, all VLA-3D instances of that
#: label being genuinely small). The geometry instead fits a floor coat rack: close
#: to the dimension_priors "rack" typical on the mid axis, same order of magnitude on
#: thin/long -- corrected to "rack".
#:
#: home_building_1 object_ids {91, 160, 195, 198, 206, 216, 256, 276, 308, 371, 423},
#: raw_label "lamp": ELEVEN instances, every one 0.766 x 0.766 x 4.32 m (matching to
#: three decimal places), gray, centred at z~2.15m (floor-to-ceiling), scattered
#: across four different regions of a large multi-room building. No real floor lamp
#: is 4.3 m tall (more than twice the tallest real floor lamps); the volume is ~24x
#: the lamp-typical. The near-exact size/colour repetition across many rooms plus a
#: floor-to-ceiling span instead matches a single reused structural-column asset:
#: volume sits within ~4x the dimension_priors "column" typical (vs. ~24x for
#: "lamp") on every one of the eleven -- corrected to "column".
#:
#: livingroom_2 object_id 81, raw_label "lamp": 0.597 x 1.856 x 2.100 m -- a lamp
#: base 1.86 m across is not a lamp. Its dimensions fit the dimension_priors
#: "bookcase" typical closely (volume ~1.5x typical vs. ~22x the lamp-typical, long
#: axis within 6% of the bookcase median) -- corrected to "bookcase".
#:
#: Other tools.gt_label_geometry_validator hits were reviewed and left uncorrected as
#: plausible-but-unusual, not mislabels (see issue #144's resolution comment for the
#: full per-candidate table): structural classes (wall/window/door/floor/curtain) vary
#: legitimately with room shell geometry; "book"/"books" outliers are annotation
#: boxes drawn around a shelf/stack of books, which does not make "book" a WRONG class
#: (unlike #110/the entries above, no alternate class fits better); "tray" (home_
#: building_1, object 252) is thin/flat exactly like a tray, just an oversized one (no
#: better-fitting class); "phone" extremes are known annotation-size noise (#144's own
#: heuristic scan note); "picture"/"pillow"/"flowers"/"fireplace"/small "vase"/"table"
#: outliers are within the range of real-world size variation for those classes.
_SCENE_LABEL_CORRECTIONS: dict[tuple[str, int], tuple[str, ...]] = {
    ("livingroom_1", 91): ("tv",),
    ("livingroom_2", 53): ("mirror",),
    ("studio", 46): ("column",),
    ("studio", 6): ("rack",),
    ("home_building_1", 91): ("column",),
    ("home_building_1", 160): ("column",),
    ("home_building_1", 195): ("column",),
    ("home_building_1", 198): ("column",),
    ("home_building_1", 206): ("column",),
    ("home_building_1", 216): ("column",),
    ("home_building_1", 256): ("column",),
    ("home_building_1", 276): ("column",),
    ("home_building_1", 308): ("column",),
    ("home_building_1", 371): ("column",),
    ("home_building_1", 423): ("column",),
    ("livingroom_2", 81): ("bookcase",),
}


def _apply_scene_label_corrections(scene_name: str, instances: list[InstanceRecord]) -> None:
    """Append any :data:`_SCENE_LABEL_CORRECTIONS` aliases for ``scene_name`` in place.

    Only touches instances whose ``(scene_name, instance_id)`` has an entry; every
    other instance's ``aliases`` is left exactly as :func:`parse_object_csv` built it.
    """
    for rec in instances:
        extra = _SCENE_LABEL_CORRECTIONS.get((scene_name, rec.instance_id))
        if not extra:
            continue
        rec.aliases = tuple(rec.aliases) + tuple(a for a in extra if a not in rec.aliases)


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
    _apply_scene_label_corrections(name, instances)
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
