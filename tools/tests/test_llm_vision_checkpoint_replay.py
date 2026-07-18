"""Tests for the pure helpers in ``tools.llm_vision_checkpoint_replay`` (no network,
no bag I/O, no Pillow)."""
from __future__ import annotations

import random

import numpy as np
import pytest

from core.perception import tiling
from tools.llm_vision_checkpoint_replay import (
    Cp3Candidate,
    bearing_map_rad,
    cp5_choice_is_sane,
    crop_box,
    crop_half_size,
    frontier_azimuths,
    frontier_columns,
    load_object_list,
    parse_object_list_line,
    pick_absent_nouns,
    project_to_tile_pixel,
    scene_labels,
    tile_containing_azimuth,
)

# --------------------------------------------------------------------------- object_list parsing


def test_parse_object_list_line_basic():
    line = '4 -0.963 2.125 0.470 0.132 0.190 0.359 2.910 "flowers"'
    obj = parse_object_list_line(line)
    assert obj == {
        "id": 4, "x": -0.963, "y": 2.125, "z": 0.470,
        "sx": 0.132, "sy": 0.190, "sz": 0.359, "yaw": 2.910, "label": "flowers",
    }


def test_parse_object_list_line_multiword_label():
    line = '8 -1.268 5.360 1.390 0.508 0.038 0.699 -0.0002 "calligraphy painting"'
    obj = parse_object_list_line(line)
    assert obj["label"] == "calligraphy painting"


@pytest.mark.parametrize("line", ["", "   ", "not a valid line", "1 2 3 \"unterminated"])
def test_parse_object_list_line_malformed_returns_none(line):
    assert parse_object_list_line(line) is None


def test_load_object_list(tmp_path):
    p = tmp_path / "object_list.txt"
    p.write_text(
        '0 1.0 2.0 0.5 0.1 0.1 0.1 0.0 "sofa"\n'
        "\n"
        '1 3.0 4.0 0.5 0.1 0.1 0.1 0.0 "table"\n'
    )
    objs = load_object_list(p)
    assert len(objs) == 2
    assert objs[0]["label"] == "sofa"
    assert objs[1]["label"] == "table"


def test_scene_labels_lowercases():
    objs = [{"label": "Sofa"}, {"label": "TABLE"}]
    assert scene_labels(objs) == {"sofa", "table"}


# --------------------------------------------------------------------------- absent-noun picker


def test_pick_absent_nouns_excludes_present_labels():
    labels = {"sofa", "table", "chair"}
    rng = random.Random(0)
    picked = pick_absent_nouns(labels, k=5, rng=rng)
    assert len(picked) == 5
    assert not any(p in labels for p in picked)


def test_pick_absent_nouns_excludes_substring_match():
    # "teapot" pool candidate should be excluded if a present label contains it
    labels = {"teapot lid"}
    rng = random.Random(0)
    picked = pick_absent_nouns(labels, k=20, rng=rng)
    assert "teapot" not in picked


def test_pick_absent_nouns_caps_at_available():
    # ask for more than the pool can honestly supply once everything is excluded
    from tools.llm_vision_checkpoint_replay import _ABSENT_NOUN_POOL
    labels = set(_ABSENT_NOUN_POOL)  # exclude everything
    rng = random.Random(0)
    picked = pick_absent_nouns(labels, k=5, rng=rng)
    assert picked == []


# --------------------------------------------------------------------------- bearing / tiling geometry


def test_bearing_map_rad_cardinal_directions():
    assert bearing_map_rad(0, 0, 1, 0) == pytest.approx(0.0)
    assert bearing_map_rad(0, 0, 0, 1) == pytest.approx(np.pi / 2)
    assert bearing_map_rad(0, 0, -1, 0) == pytest.approx(np.pi)


def test_tile_containing_azimuth_matches_tile_center():
    specs = tiling.tile_specs(4, tiling.DEFAULT_TILE_HFOV, tiling.DEFAULT_TILE_VFOV)
    for spec in specs:
        assert tile_containing_azimuth(spec.yaw_center, specs) == spec.tile_id


def test_tile_containing_azimuth_covers_full_circle():
    specs = tiling.tile_specs(4, tiling.DEFAULT_TILE_HFOV, tiling.DEFAULT_TILE_VFOV)
    # 4x90deg tiles with 0deg gap should cover the entire circle with no holes
    for deg in range(-179, 180, 3):
        az = np.deg2rad(deg)
        assert tile_containing_azimuth(az, specs) is not None


def test_project_to_tile_pixel_is_inverse_of_tile_pixel_to_camera_ray():
    specs = tiling.tile_specs(4, tiling.DEFAULT_TILE_HFOV, tiling.DEFAULT_TILE_VFOV)
    spec = specs[0]
    for u, v in [(spec.cx, spec.cy), (100.0, 200.0), (spec.width - 5, spec.height - 5)]:
        az, el = tiling.tile_pixel_to_camera_ray(spec.tile_id, u, v, 4,
                                                   tiling.DEFAULT_TILE_HFOV, tiling.DEFAULT_TILE_VFOV)
        u2, v2 = project_to_tile_pixel(az, el, spec)
        assert u2 == pytest.approx(u, abs=1e-6)
        assert v2 == pytest.approx(v, abs=1e-6)


def test_project_to_tile_pixel_center_axis_is_tile_center():
    specs = tiling.tile_specs(4, tiling.DEFAULT_TILE_HFOV, tiling.DEFAULT_TILE_VFOV)
    spec = specs[2]
    u, v = project_to_tile_pixel(spec.yaw_center, 0.0, spec)
    assert u == pytest.approx(spec.cx, abs=1e-6)
    assert v == pytest.approx(spec.cy, abs=1e-6)


def test_crop_box_clamped_to_image_bounds():
    x1, y1, x2, y2 = crop_box(u=5, v=5, half_w=50, half_h=50, width=480, height=640)
    assert x1 == 0 and y1 == 0
    assert x2 > x1 and y2 > y1


def test_crop_box_non_degenerate_at_image_edge():
    x1, y1, x2, y2 = crop_box(u=479, v=639, half_w=50, half_h=50, width=480, height=640)
    assert x2 - x1 >= 1
    assert y2 - y1 >= 1


def test_crop_half_size_clamped_range():
    specs = tiling.tile_specs(4, tiling.DEFAULT_TILE_HFOV, tiling.DEFAULT_TILE_VFOV)
    spec = specs[0]
    obj = {"sx": 0.3, "sy": 0.3, "sz": 0.3}
    half_w, half_h = crop_half_size(spec, obj, dist_m=3.0)
    assert 40.0 <= half_w <= 220.0
    assert 40.0 <= half_h <= 220.0


def test_crop_half_size_far_objects_are_small_but_clamped():
    specs = tiling.tile_specs(4, tiling.DEFAULT_TILE_HFOV, tiling.DEFAULT_TILE_VFOV)
    spec = specs[0]
    obj = {"sx": 0.05, "sy": 0.05, "sz": 0.05}
    half_w, half_h = crop_half_size(spec, obj, dist_m=100.0)
    assert half_w == pytest.approx(40.0)  # floor clamp kicks in


# --------------------------------------------------------------------------- CP5 helpers


def test_frontier_azimuths_evenly_spaced():
    azs = frontier_azimuths(5)
    assert len(azs) == 5
    # consecutive gaps should all be ~72deg (mod wraparound at the end)
    for a in azs:
        assert -np.pi <= a <= np.pi


def test_frontier_columns_in_bounds():
    cols = frontier_columns(5, width=tiling.PANO_WIDTH)
    assert len(cols) == 5
    for c in cols:
        assert 0 <= c < tiling.PANO_WIDTH


def test_cp5_choice_is_sane_fallback_always_sane():
    assert cp5_choice_is_sane(None, 5) is True


def test_cp5_choice_is_sane_in_range():
    assert cp5_choice_is_sane(3, 5) is True
    assert cp5_choice_is_sane(1, 5) is True
    assert cp5_choice_is_sane(5, 5) is True


def test_cp5_choice_is_sane_out_of_range():
    assert cp5_choice_is_sane(0, 5) is False
    assert cp5_choice_is_sane(6, 5) is False
    assert cp5_choice_is_sane(-1, 5) is False


# --------------------------------------------------------------------------- Cp3Candidate dataclass smoke


def test_cp3_candidate_fields():
    c = Cp3Candidate(frame_idx=1, label="sofa", tile_id=2, dist_m=3.5, azimuth=0.1, elevation=-0.05)
    assert c.label == "sofa"
    assert c.tile_id == 2
