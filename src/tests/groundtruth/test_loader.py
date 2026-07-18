"""Loader tests: real loft sample + hand-built OBB->AABB rotation correctness."""
from __future__ import annotations

import csv
import math

import numpy as np
import pytest

from core.groundtruth.loader import load_scene, obb_to_aabb, parse_object_csv
from core.interfaces import InstanceRecord

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
