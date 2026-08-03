"""Tests for tools/gt_label_geometry_validator.py (issue #144)."""
from __future__ import annotations

import csv

import numpy as np
import pytest

from tools.gt_label_geometry_validator import scan_all


_HEADER = [
    "object_id", "region_id", "raw_label", "nyu_id", "nyu40_id", "nyu_label",
    "nyu40_label", "object_bbox_cx", "object_bbox_cy", "object_bbox_cz",
    "object_bbox_xlength", "object_bbox_ylength", "object_bbox_zlength",
    "object_bbox_heading", "object_front_heading",
]


def _row(object_id, label, x, y, z, heading="0.0", region_id="0"):
    row = {h: "" for h in _HEADER}
    row.update(
        object_id=str(object_id),
        region_id=region_id,
        raw_label=label,
        nyu_label=label,
        nyu40_label=label,
        object_bbox_cx="0.0",
        object_bbox_cy="0.0",
        object_bbox_cz="0.0",
        object_bbox_xlength=str(x),
        object_bbox_ylength=str(y),
        object_bbox_zlength=str(z),
        object_bbox_heading=heading,
    )
    return row


def _write_scene(tmp_path, scene_name, rows):
    scene_dir = tmp_path / scene_name
    scene_dir.mkdir()
    path = scene_dir / f"{scene_name}_object_result.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_HEADER)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return scene_dir


def test_implausible_class_volume_is_flagged(tmp_path):
    """A 'tv' with an absurd 3m-cube footprint, in a scene with a same-label family
    of >=3, must be flagged."""
    rows = [_row(0, "tv", 3.0, 3.0, 3.0)]
    # pad the family with plausible same-label instances so min_n clears.
    rows += [_row(i, "tv", 0.8, 0.1, 0.6) for i in range(1, 4)]
    _write_scene(tmp_path, "scene_a", rows)

    flags = scan_all(tmp_path, factor=12.0, min_n=3)
    hit = {(f.scene, f.object_id) for f in flags}
    assert ("scene_a", 0) in hit


def test_plausible_instance_not_flagged(tmp_path):
    """A near-typical instance of a class with an established prior must never be
    flagged, however small the family."""
    rows = [_row(i, "chair", 0.55, 0.6, 0.9) for i in range(4)]
    _write_scene(tmp_path, "scene_b", rows)

    flags = scan_all(tmp_path, factor=12.0, min_n=3)
    assert flags == []


def test_below_min_n_family_is_never_flagged(tmp_path):
    """A label with fewer than min_n instances has no family to be out of -- the
    documented bounding note -- even if its one instance is wildly implausible."""
    rows = [_row(0, "tv", 3.0, 3.0, 3.0), _row(1, "tv", 3.0, 3.0, 3.0)]
    _write_scene(tmp_path, "scene_c", rows)

    flags = scan_all(tmp_path, factor=12.0, min_n=3)
    assert flags == []


def test_class_with_no_dimension_prior_is_skipped(tmp_path):
    """A raw_label that folds onto no dimension_priors class at all (e.g. 'hanger',
    per #144) cannot be geometrically checked and must never be flagged, however
    implausible its extents look by eye."""
    rows = [_row(i, "hanger", 0.45, 0.44, 1.73) for i in range(4)]
    _write_scene(tmp_path, "scene_d", rows)

    flags = scan_all(tmp_path, factor=12.0, min_n=3)
    assert flags == []


def test_uses_true_obb_extents_not_inflated_aabb(tmp_path):
    """A rotated (45deg) box that is a genuinely typical instance of its class must
    NOT be flagged merely because the AABB-of-OBB over-approximation inflates its
    apparent footprint -- the validator must compare against obb_extents, the true
    box dims, not the AABB."""
    # A typical chair-sized box (~0.55 x 0.6 x 0.9), rotated 45deg. Its AABB footprint
    # is inflated by ~sqrt(2), which would read as notably larger than a chair if
    # the validator (wrongly) used the AABB instead of the true OBB dims.
    rows = [_row(i, "chair", 0.55, 0.6, 0.9, heading="0.7853981633974483") for i in range(4)]
    _write_scene(tmp_path, "scene_e", rows)

    flags = scan_all(tmp_path, factor=12.0, min_n=3)
    assert flags == []


def test_global_pooling_across_scenes_for_min_n(tmp_path):
    """A label with only 1 instance per scene but >=3 pooled across scenes still
    clears min_n and gets checked -- #144's own scan pooled across all 15 scenes,
    not per-scene."""
    for i, scene in enumerate(("scene_f1", "scene_f2", "scene_f3")):
        _write_scene(tmp_path, scene, [_row(0, "tv", 3.0, 3.0, 3.0)])

    flags = scan_all(tmp_path, factor=12.0, min_n=3)
    hits = {(f.scene, f.object_id) for f in flags}
    assert hits == {("scene_f1", 0), ("scene_f2", 0), ("scene_f3", 0)}
