"""Tests for the GT rasterization / object-filtering helpers in
``tools.overhead_validation`` (rotated-rect rasterize, object_list.txt parsing,
band-overlap and known-overhang filtering). Pure-Python/numpy, no bag I/O."""
from __future__ import annotations

import math

import numpy as np
import pytest

from tools.overhead_validation import (
    SceneObject,
    floor_z_from_traversable,
    known_overhang_objects,
    object_overlaps_band,
    parse_object_list,
    rasterize_points_to_cells,
    read_ply_xyz,
    rotated_rect_cells,
)

CELL_M = 0.10


# --------------------------------------------------------------------------- rotated_rect_cells

def test_axis_aligned_square_covers_expected_cell_count():
    """A 1x1 m square centred at the origin, no rotation, tiles exactly to cells."""
    cells = rotated_rect_cells(0.0, 0.0, 1.0, 1.0, 0.0, CELL_M, 0.0, 0.0)
    # 1.0 / 0.10 = 10 cells per side -> 100 cells, centred on (0,0).
    assert len(cells) == 100
    rows = {r for r, c in cells}
    cols = {c for r, c in cells}
    assert rows == set(range(-5, 5))
    assert cols == set(range(-5, 5))


def test_45_degree_rotation_produces_diamond_smaller_than_bbox():
    """A rotated square's rasterized footprint is strictly smaller than its AABB."""
    cells_rot = rotated_rect_cells(0.0, 0.0, 1.0, 1.0, math.pi / 4, CELL_M, 0.0, 0.0)
    cells_flat = rotated_rect_cells(0.0, 0.0, 1.0, 1.0, 0.0, CELL_M, 0.0, 0.0)
    # The rotated footprint's bounding box is bigger (diagonal ~1.41 m) but the
    # actual rasterized cell count inside the rotated square is less than the
    # bbox area because corners of the bbox are outside the diamond.
    assert len(cells_rot) < len(cells_flat) * 2
    assert len(cells_rot) > 0


def test_inflate_cells_grows_the_footprint():
    # 0.46 m (not grid-aligned) avoids a boundary tie between the base and
    # inflated footprints at this cell size.
    base = rotated_rect_cells(0.0, 0.0, 0.46, 0.46, 0.0, CELL_M, 0.0, 0.0, inflate_cells=0)
    inflated = rotated_rect_cells(0.0, 0.0, 0.46, 0.46, 0.0, CELL_M, 0.0, 0.0, inflate_cells=1)
    assert inflated.issuperset(base)
    assert len(inflated) > len(base)


def test_offset_origin_shifts_cell_ids_consistently():
    """Cell ids are relative to (origin_x, origin_y), matching OccupancyGrid.world_to_cell."""
    cells_o0 = rotated_rect_cells(1.0, 1.0, 0.2, 0.2, 0.0, CELL_M, 0.0, 0.0)
    cells_o1 = rotated_rect_cells(1.0, 1.0, 0.2, 0.2, 0.0, CELL_M, 0.5, 0.5)
    # Same world rectangle, origin shifted by 0.5 m = 5 cells -> ids shift by 5.
    shifted_back = {(r + 5, c + 5) for r, c in cells_o1}
    assert shifted_back == cells_o0


def test_rotated_rect_matches_point_in_rect_ground_truth():
    """Cross-check the rasterizer against a brute-force point-in-rotated-rect test."""
    cx, cy, dx, dy, yaw = 2.3, -1.1, 1.6, 0.8, 0.7
    cells = rotated_rect_cells(cx, cy, dx, dy, yaw, CELL_M, 0.0, 0.0)
    cos_t, sin_t = math.cos(-yaw), math.sin(-yaw)
    half_x, half_y = dx / 2.0, dy / 2.0
    for row, col in cells:
        wx = (col + 0.5) * CELL_M
        wy = (row + 0.5) * CELL_M
        lx = (wx - cx) * cos_t - (wy - cy) * sin_t
        ly = (wx - cx) * sin_t + (wy - cy) * cos_t
        assert abs(lx) <= half_x + 1e-9
        assert abs(ly) <= half_y + 1e-9
    # And a cell far outside is never included.
    assert (int((cy + 10) / CELL_M), int((cx + 10) / CELL_M)) not in cells


# --------------------------------------------------------------------------- rasterize_points_to_cells

def test_rasterize_points_to_cells_matches_world_to_cell_convention():
    pts = np.array([[0.05, 0.05, 0.0], [0.15, 0.25, 0.0], [-0.05, -0.05, 0.0]])
    cells = rasterize_points_to_cells(pts, CELL_M, 0.0, 0.0)
    assert cells == {(0, 0), (2, 1), (-1, -1)}


def test_rasterize_empty_points_returns_empty_set():
    assert rasterize_points_to_cells(np.zeros((0, 3)), CELL_M, 0.0, 0.0) == set()


# --------------------------------------------------------------------------- object_overlaps_band

def test_object_overlaps_band_true_when_intersecting():
    # floor_z=0.0, band [0.25, 1.20]; table top at 0.75, bottom at 0.70 -> inside band.
    obj = SceneObject(0, 0, 0, 0.725, 1.0, 1.0, 0.05, 0.0, "table")
    assert object_overlaps_band(obj, floor_z=0.0, band_min=0.25, band_max=1.20)


def test_object_overlaps_band_false_when_below_band():
    # A rug flush on the floor: [−0.01, 0.01], entirely below band_min=0.25.
    obj = SceneObject(0, 0, 0, 0.0, 1.0, 1.0, 0.02, 0.0, "rug")
    assert not object_overlaps_band(obj, floor_z=0.0, band_min=0.25, band_max=1.20)


def test_object_overlaps_band_false_when_above_band():
    # A ceiling lamp at z=2.1 with small extent, band tops out at 1.20.
    obj = SceneObject(0, 0, 0, 2.1, 0.3, 0.3, 0.2, 0.0, "lamp")
    assert not object_overlaps_band(obj, floor_z=0.0, band_min=0.25, band_max=1.20)


def test_object_overlaps_band_true_when_spanning_whole_band():
    # A tall bookshelf from floor to ceiling spans (and exceeds) the band entirely.
    obj = SceneObject(0, 0, 0, 1.0, 0.4, 0.4, 2.0, 0.0, "bookshelf")
    assert object_overlaps_band(obj, floor_z=0.0, band_min=0.25, band_max=1.20)


# --------------------------------------------------------------------------- known_overhang_objects

def test_known_overhang_objects_filters_by_keyword_and_height():
    floor_z = -0.6
    objs = [
        SceneObject(0, 0, 0, floor_z + 0.75, 1.0, 1.0, 0.05, 0.0, "coffee table"),  # top ~1.1 above floor
        SceneObject(1, 1, 1, floor_z + 0.1, 0.3, 0.3, 0.1, 0.0, "shelf"),  # top only 0.15 above floor
        SceneObject(2, 2, 2, floor_z + 0.9, 0.5, 0.5, 0.05, 0.0, "chair"),  # not a keyword match
        SceneObject(3, 3, 3, floor_z + 1.0, 1.2, 0.4, 0.03, 0.0, "wall cabinet"),  # keyword + high top
    ]
    result = known_overhang_objects(objs, floor_z)
    names = {o.name for o in result}
    assert names == {"coffee table", "wall cabinet"}


def test_known_overhang_objects_empty_when_none_match():
    floor_z = 0.0
    objs = [SceneObject(0, 0, 0, 0.5, 0.3, 0.3, 0.05, 0.0, "chair")]
    assert known_overhang_objects(objs, floor_z) == []


# --------------------------------------------------------------------------- parse_object_list

def test_parse_object_list_handles_quoted_multiword_names(tmp_path):
    p = tmp_path / "object_list.txt"
    p.write_text(
        "0 -2.365999728904278 1.3159997519016502 1.1918222904205322 "
        "1.4365473 0.5382916 2.3886433 0.00028257512371903667 \"shelf\"\n"
        "2 1.8870365670998213 -1.6414096229035477 1.0647079944610596 "
        "0.4928776 0.19388354 0.49120265 -0.08527218125696875 \"computer monitor\"\n"
        "\n"  # blank line should be skipped
    )
    objs = parse_object_list(p)
    assert len(objs) == 2
    assert objs[0].id == 0
    assert objs[0].name == "shelf"
    assert objs[1].name == "computer monitor"
    assert objs[1].id == 2
    assert objs[1].x == pytest.approx(1.8870365670998213)


def test_parse_object_list_rejects_malformed_row(tmp_path):
    p = tmp_path / "object_list.txt"
    p.write_text("this is not a valid row\n")
    with pytest.raises(ValueError):
        parse_object_list(p)


# --------------------------------------------------------------------------- ply reading

def test_read_ply_xyz_and_floor_z(tmp_path):
    p = tmp_path / "traversable_area.ply"
    p.write_text(
        "ply\n"
        "format ascii 1.0\n"
        "element vertex 5\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "end_header\n"
        "0.0 0.0 -0.60\n"
        "1.0 0.0 -0.61\n"
        "0.0 1.0 -0.59\n"
        "1.0 1.0 -0.60\n"
        "2.0 2.0 -0.62\n"
    )
    pts = read_ply_xyz(p)
    assert pts.shape == (5, 3)
    floor_z = floor_z_from_traversable(pts)
    assert floor_z == pytest.approx(-0.60)


def test_read_ply_xyz_rejects_binary_format(tmp_path):
    p = tmp_path / "bad.ply"
    p.write_text(
        "ply\nformat binary_little_endian 1.0\nelement vertex 1\n"
        "property float x\nproperty float y\nproperty float z\nend_header\n"
    )
    with pytest.raises(ValueError):
        read_ply_xyz(p)
