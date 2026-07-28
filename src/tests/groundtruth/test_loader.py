"""Loader tests: real loft sample + hand-built OBB->AABB rotation correctness."""
from __future__ import annotations

import csv
import math

import numpy as np
import pytest

from core.groundtruth.loader import load_scene, obb_to_aabb, parse_object_csv
from core.interfaces import InstanceRecord
from core.perception.scene_index import BasicSceneIndex

from tests.groundtruth.conftest import (
    FULL_UNITY_ROOT,
    requires_full_unity,
    requires_loft,
    LOFT_DIR,
)


# --------------------------------------------------------------------------- OBB -> AABB


def test_obb_to_aabb_axis_aligned_zero_heading():
    """Heading 0 -> AABB is exactly center +/- half-extents."""
    amin, amax = obb_to_aabb(center=(1.0, 2.0, 3.0), extents=(2.0, 4.0, 6.0), heading=0.0)
    assert np.allclose(amin, [0.0, 0.0, 0.0])
    assert np.allclose(amax, [2.0, 4.0, 6.0])


def test_obb_to_aabb_90deg_swaps_xy_extents():
    """A 90 deg yaw swaps the x/y footprint extents; z unchanged."""
    amin, amax = obb_to_aabb(center=(0.0, 0.0, 0.0), extents=(2.0, 4.0, 6.0),
                             heading=math.pi / 2)
    ext = amax - amin
    assert np.allclose(ext, [4.0, 2.0, 6.0], atol=1e-9)  # x<->y swapped, z same


def test_obb_to_aabb_45deg_inflates_footprint():
    """A 45 deg yaw inflates the footprint by ~diagonal (over-approximation)."""
    amin, amax = obb_to_aabb(center=(0.0, 0.0, 0.0), extents=(2.0, 2.0, 1.0),
                             heading=math.pi / 4)
    ext = amax - amin
    # square of side 2 rotated 45deg -> AABB side = 2*sqrt(2)
    assert ext[0] == pytest.approx(2 * math.sqrt(2), abs=1e-9)
    assert ext[1] == pytest.approx(2 * math.sqrt(2), abs=1e-9)
    assert ext[2] == pytest.approx(1.0, abs=1e-9)  # z untouched by yaw


def test_obb_to_aabb_center_preserved():
    """The AABB is symmetric about the OBB center for any heading."""
    c = (5.0, -3.0, 1.5)
    amin, amax = obb_to_aabb(center=c, extents=(1.0, 3.0, 2.0), heading=0.9)
    mid = (amin + amax) / 2.0
    assert np.allclose(mid, c, atol=1e-9)


# -------------------------------------------------------------- #77 additive OBB fields


def test_parse_object_csv_carries_obb_fields(tmp_path):
    """Issue #77 Pre-Stage 1a: parse_object_csv additionally carries the ORIGINAL
    oriented box (center/extents/heading) on InstanceRecord, alongside the
    unchanged AABB-of-OBB approximation."""
    row = _base_row(
        object_bbox_cx="2.5", object_bbox_cy="-1.5", object_bbox_cz="0.3",
        object_bbox_xlength="2.0", object_bbox_ylength="0.6", object_bbox_zlength="0.7",
        object_bbox_heading="0.7853981633974483",  # pi/4
    )
    path = _write_object_csv(tmp_path, [row])
    rec = parse_object_csv(path)[0]
    assert rec.obb_heading == pytest.approx(math.pi / 4)
    assert np.allclose(rec.obb_center, [2.5, -1.5, 0.3])
    assert np.allclose(rec.obb_extents, [2.0, 0.6, 0.7])
    # aabb_min/aabb_max stay the (unchanged) over-approximation, computed by
    # obb_to_aabb the same way it always was.
    amin, amax = obb_to_aabb(rec.obb_center, rec.obb_extents, rec.obb_heading)
    assert np.allclose(rec.aabb_min, amin)
    assert np.allclose(rec.aabb_max, amax)


def test_parse_object_csv_default_heading_is_zero(tmp_path):
    row = _base_row()  # object_bbox_heading defaults to "0.0" in _base_row
    path = _write_object_csv(tmp_path, [row])
    rec = parse_object_csv(path)[0]
    assert rec.obb_heading == 0.0


def test_instance_record_obb_fields_default_none_zero():
    """A producer that never sets the additive OBB fields (mocks/perception) keeps
    the safe defaults — no OBB info, callers fall back to the AABB."""
    rec = InstanceRecord(
        instance_id=0, label="chair", score=1.0, n_obs=3,
        centroid=np.zeros(3), aabb_min=np.zeros(3), aabb_max=np.ones(3),
    )
    assert rec.obb_center is None
    assert rec.obb_extents is None
    assert rec.obb_heading == 0.0


# --------------------------------------------------------------------------- real loft


@requires_loft
def test_instance_count_matches_csv_rows():
    """One InstanceRecord per data row of the object CSV."""
    scene = load_scene(LOFT_DIR)
    with open(LOFT_DIR / "loft_object_result.csv", encoding="utf-8", newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if (r.get("object_id") or "").strip()]
    assert len(scene.instances) == len(rows)
    assert len(scene.instances) == 115  # 116 lines - 1 header


@requires_loft
def test_all_records_are_instance_records_fully_observed():
    scene = load_scene(LOFT_DIR)
    for rec in scene.instances:
        assert isinstance(rec, InstanceRecord)
        assert rec.n_obs == 3  # ground truth == fully observed
        assert rec.points is None
        assert rec.label == rec.label.lower()


@requires_loft
def test_known_object_aabb_contains_its_center():
    """Every loaded AABB must contain its own centroid (well-formed box)."""
    scene = load_scene(LOFT_DIR)
    for rec in scene.instances:
        assert np.all(rec.centroid >= rec.aabb_min - 1e-9)
        assert np.all(rec.centroid <= rec.aabb_max + 1e-9)
        assert np.all(rec.aabb_max >= rec.aabb_min)


@requires_loft
def test_raw_label_used_not_nyu():
    """raw_label ('sphere decoration', 'door frame') is the label, not NYU classes."""
    scene = load_scene(LOFT_DIR)
    labels = {r.label for r in scene.instances}
    # raw_label specifics that NYU40 would collapse to 'otherprop'/'otherstructure'
    assert "sphere decoration" in labels
    assert "door frame" in labels
    assert "dvd" in labels


@requires_loft
def test_colors_land_in_attributes():
    """Dominant colour names reach caption + aliases so attribute matching works."""
    scene = load_scene(LOFT_DIR)
    # object 0 in the sample is a gray 'dvd'
    dvd = next(r for r in scene.instances if r.instance_id == 0)
    assert dvd.label == "dvd"
    assert "gray" in dvd.caption
    assert "gray" in dvd.aliases


@requires_full_unity
def test_color_bins_carry_raw_rgb_and_fraction():
    """Colour bins carry raw RGB + fraction, not just scheme names (issues #11/#12)."""
    scene = load_scene(FULL_UNITY_ROOT / "loft")
    by_id = {r.instance_id: r for r in scene.instances}
    # pillow 89 is a dark-slate-gray pillow: scheme 'gray' but RGB (47,79,79)
    p89 = by_id[89]
    assert p89.color_bins, "color_bins must be populated from the CSV"
    b0 = p89.color_bins[0]
    assert b0.name == "gray"
    assert b0.rgb == (47, 79, 79)
    assert b0.fraction == pytest.approx(1.0, abs=1e-6)
    # bin scheme names stay in lock-step with the alias list (no regression there)
    assert tuple(b.name for b in p89.color_bins) == p89.aliases
    # a lighter gray pillow shares the scheme name but differs in raw RGB + luma
    p54 = by_id[54]
    assert p54.color_bins[0].rgb == (112, 128, 144)
    assert p89.color_bins[0].luma < p54.color_bins[0].luma


@requires_full_unity
def test_multi_bin_object_keeps_all_bins_with_fractions():
    """A multi-colour object exposes each bin's fraction (dominant first)."""
    scene = load_scene(FULL_UNITY_ROOT / "loft")
    by_id = {r.instance_id: r for r in scene.instances}
    # loft chair 10: gray (47,79,79) 0.79 + black (0,0,0) 0.21
    chair = by_id[10]
    assert [b.name for b in chair.color_bins] == ["gray", "black"]
    assert chair.color_bins[0].fraction > chair.color_bins[1].fraction
    assert chair.color_bins[1].rgb == (0, 0, 0)


@requires_loft
def test_regions_loaded_and_membership_mapped():
    scene = load_scene(LOFT_DIR)
    labels = {r.label for r in scene.regions}
    assert labels == {"bedroom", "corridor", "livingroom"}
    # object->region map is populated (scene graph covers most objects)
    assert len(scene.object_region) > 50
    # every mapped region id is a real region
    rids = {r.region_id for r in scene.regions}
    assert set(scene.object_region.values()) <= rids


@requires_loft
def test_parse_object_csv_direct_matches_load_scene():
    """parse_object_csv is the same instance list load_scene wraps."""
    direct = parse_object_csv(LOFT_DIR / "loft_object_result.csv")
    scene = load_scene(LOFT_DIR)
    assert len(direct) == len(scene.instances)
    assert [r.instance_id for r in direct] == [r.instance_id for r in scene.instances]


# --------------------------------------------------------------------------- #27:
# corrupt colour slots must not desync aliases vs color_bins

_OBJECT_CSV_COLUMNS = [
    "object_id",
    "raw_label",
    "object_bbox_cx",
    "object_bbox_cy",
    "object_bbox_cz",
    "object_bbox_xlength",
    "object_bbox_ylength",
    "object_bbox_zlength",
    "object_bbox_heading",
]


def _write_object_csv(tmp_path, rows: list[dict[str, str]]):
    """Write a minimal synthetic ``*_object_result.csv`` and return its path.

    ``rows`` may carry extra colour columns (``object_color_*{1,2,3}``) beyond the
    base geometry columns; the header is the union across all rows so a row that
    omits a column (e.g. a missing percentage) round-trips as a genuinely blank
    field, not a present-but-empty one — distinguishing "absent" from "malformed".
    """
    fieldnames = list(_OBJECT_CSV_COLUMNS)
    for row in rows:
        for k in row:
            if k not in fieldnames:
                fieldnames.append(k)
    path = tmp_path / "synthetic_object_result.csv"
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def _base_row(**overrides) -> dict[str, str]:
    row = {
        "object_id": "0",
        "raw_label": "pillow",
        "object_bbox_cx": "0.0",
        "object_bbox_cy": "0.0",
        "object_bbox_cz": "0.0",
        "object_bbox_xlength": "0.3",
        "object_bbox_ylength": "0.3",
        "object_bbox_zlength": "0.3",
        "object_bbox_heading": "0.0",
    }
    row.update(overrides)
    return row


def test_corrupt_rgb_drops_slot_from_both_aliases_and_bins(tmp_path, caplog):
    """A scheme name with unparseable RGB must not survive as an alias while its
    ColorBin silently vanishes (issue #27a): the whole slot is dropped from both."""
    row = _base_row(
        object_color_scheme1="gray",
        object_color_r1="not-a-number",
        object_color_g1="79",
        object_color_b1="79",
        object_color_scheme_percentage1="0.95",
    )
    path = _write_object_csv(tmp_path, [row])
    with caplog.at_level("WARNING", logger="core.groundtruth.loader"):
        recs = parse_object_csv(path)
    assert len(recs) == 1
    rec = recs[0]
    assert "gray" not in rec.aliases  # would desync if it survived here alone
    assert rec.color_bins == ()
    assert "corrupt colour slot 1" in caplog.text
    assert str(path) in caplog.text
    assert "gray" in caplog.text


def test_malformed_percentage_treated_as_invalid_not_zero(tmp_path, caplog):
    """A percentage that fails float() must be INVALID (whole slot skipped, issue
    #27b) — not silently collapsed to fraction=0.0, which would defeat colour
    dominance salience (#11/#12) while still looking like legitimate zero data."""
    row = _base_row(
        object_color_scheme1="gray",
        object_color_r1="47",
        object_color_g1="79",
        object_color_b1="79",
        object_color_scheme_percentage1="not-a-percentage",
    )
    path = _write_object_csv(tmp_path, [row])
    with caplog.at_level("WARNING", logger="core.groundtruth.loader"):
        recs = parse_object_csv(path)
    rec = recs[0]
    assert rec.color_bins == ()
    assert "gray" not in rec.aliases
    assert "corrupt colour slot 1" in caplog.text
    assert "percentage" in caplog.text


def test_missing_percentage_still_defaults_to_zero(tmp_path):
    """A genuinely ABSENT percentage (no column value at all) is not malformed —
    it must keep defaulting to fraction=0.0, unlike a present-but-unparseable one."""
    row = _base_row(
        object_color_scheme1="gray",
        object_color_r1="47",
        object_color_g1="79",
        object_color_b1="79",
        # no object_color_scheme_percentage1 key at all
    )
    path = _write_object_csv(tmp_path, [row])
    recs = parse_object_csv(path)
    rec = recs[0]
    assert [b.name for b in rec.color_bins] == ["gray"]
    assert rec.color_bins[0].fraction == 0.0
    assert "gray" in rec.aliases


def test_second_slot_corruption_does_not_affect_first_slot(tmp_path, caplog):
    """Corruption in one colour slot must not drop a sibling slot's valid data."""
    row = _base_row(
        object_color_scheme1="gray",
        object_color_r1="47",
        object_color_g1="79",
        object_color_b1="79",
        object_color_scheme_percentage1="0.7",
        object_color_scheme2="black",
        object_color_r2="bad",
        object_color_g2="0",
        object_color_b2="0",
        object_color_scheme_percentage2="0.3",
    )
    path = _write_object_csv(tmp_path, [row])
    with caplog.at_level("WARNING", logger="core.groundtruth.loader"):
        recs = parse_object_csv(path)
    rec = recs[0]
    assert [b.name for b in rec.color_bins] == ["gray"]
    assert rec.aliases == ("gray",)
    assert "corrupt colour slot 2" in caplog.text


def test_aliases_and_color_bins_always_stay_in_lock_step(tmp_path):
    """Regression pin for #27a: whatever survives, aliases and color_bins names
    must match 1:1 in order — the very invariant the bug broke."""
    rows = [
        _base_row(
            object_id="0",
            object_color_scheme1="gray",
            object_color_r1="47",
            object_color_g1="79",
            object_color_b1="79",
            object_color_scheme_percentage1="0.95",
        ),
        _base_row(
            object_id="1",
            object_color_scheme1="maroon",
            object_color_r1="oops",
            object_color_g1="0",
            object_color_b1="0",
            object_color_scheme_percentage1="0.5",
        ),
    ]
    path = _write_object_csv(tmp_path, rows)
    recs = parse_object_csv(path)
    for rec in recs:
        assert tuple(b.name for b in rec.color_bins) == rec.aliases


# --------------------------------------------------------------------------- #110 scene-scoped label correction


def test_scene_label_correction_tv_resolves_mislabelled_panel_only(tmp_path):
    """Issue #110: livingroom_1 object 91 is annotated ``raw_label == "tv remote"``
    but its OBB (1.697 x 0.019 x 0.976 m, centre z=1.385m) is a wall-mounted TV
    panel, not a remote. The scene-scoped correction must resolve a "tv" query to
    THAT object only, while object 72 — genuinely a remote, same raw_label, same
    scene — keeps resolving under "tv remote" and must NOT come back for "tv"."""
    scene_dir = tmp_path / "livingroom_1"
    scene_dir.mkdir()
    rows = [
        _base_row(
            object_id="72", raw_label="tv remote",
            object_bbox_cx="-0.2126", object_bbox_cy="-2.3795", object_bbox_cz="0.2802",
            object_bbox_xlength="0.2039", object_bbox_ylength="0.0511",
            object_bbox_zlength="0.0193",
        ),
        _base_row(
            object_id="91", raw_label="tv remote",
            object_bbox_cx="2.1030", object_bbox_cy="-2.5280", object_bbox_cz="1.3849",
            object_bbox_xlength="1.6971", object_bbox_ylength="0.0193",
            object_bbox_zlength="0.9764",
        ),
    ]
    path = _write_object_csv(scene_dir, rows)
    # _write_object_csv always names the file "synthetic_object_result.csv"; the
    # loader matches by suffix regardless of scene-name prefix, so this still loads.
    assert path.name.endswith("_object_result.csv")

    scene = load_scene(scene_dir)
    assert scene.scene_name == "livingroom_1"  # the correction table key
    by_id = {r.instance_id: r for r in scene.instances}
    assert by_id[91].label == "tv remote"  # raw_label is NEVER overwritten
    assert by_id[72].label == "tv remote"
    assert by_id[91].aliases == ("tv",)
    assert by_id[72].aliases == ()  # untouched: correction is (scene, object_id)-scoped

    idx = BasicSceneIndex(scene.instances)
    tv_hits = {r.instance_id for r in idx.by_label("tv")}
    assert tv_hits == {91}, "tv must resolve to the mislabelled panel, and ONLY it"

    remote_hits = {r.instance_id for r in idx.by_label("tv remote")}
    assert remote_hits == {72, 91}, "both objects keep resolving under their real raw_label"
