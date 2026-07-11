"""Loader tests: real loft sample + hand-built OBB->AABB rotation correctness."""
from __future__ import annotations

import csv
import math

import numpy as np
import pytest

from core.groundtruth.loader import load_scene, obb_to_aabb, parse_object_csv
from core.interfaces import InstanceRecord

from tests.groundtruth.conftest import requires_loft, LOFT_DIR


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
