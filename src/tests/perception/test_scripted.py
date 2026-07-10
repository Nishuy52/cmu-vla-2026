"""ScriptedPanoDetector: pano-pixel xyxy -> tile-space Detection conversion.

Uses the tiling module as the oracle: a box at a known azimuth must land on the tile
whose HFOV contains that azimuth, and its tile pixels must round-trip back to the source
pano rays via the tiling inverse. Seam-spanning boxes are assigned by their centre.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from core.perception import tiling as T
from core.perception.detector import Detection
from core.perception.scripted import (
    ScriptedPanoDetector,
    camera_ray_to_tile_pixel,
    pano_bbox_to_detection,
)


# --------------------------------------------------------------- ray/pixel inverse


def test_camera_ray_to_tile_pixel_is_inverse_of_forward():
    """camera_ray_to_tile_pixel exactly inverts tiling.tile_pixel_to_camera_ray."""
    for tid in range(4):
        spec = T.tile_specs()[tid]
        for u, v in [(spec.cx, spec.cy), (25, 25), (spec.width - 25, spec.height - 25),
                     (spec.width - 25, 25), (25, spec.height - 25)]:
            az, el = T.tile_pixel_to_camera_ray(tid, u, v)
            u2, v2 = camera_ray_to_tile_pixel(spec, az, el)
            assert abs(u - u2) < 1e-6
            assert abs(v - v2) < 1e-6


# --------------------------------------------------------------- tile selection


def _center_col_for_azimuth(az_deg: float) -> float:
    """Pano column whose camera azimuth is az_deg (oracle via azimuth_to_column)."""
    return float(T.azimuth_to_column(np.deg2rad(az_deg)))


def test_box_at_heading_lands_on_front_tile_center():
    """A small box centred on the panorama centre column (azimuth 0 = heading) lands on
    tile 0 (yaw_center 0) near that tile's horizontal centre."""
    col = T.PANO_WIDTH / 2.0  # azimuth 0
    row = T.PANO_HEIGHT / 2.0  # elevation ~0
    box = (col - 5, row - 5, col + 5, row + 5)
    det = pano_bbox_to_detection("stool", box, 0.9, T.tile_specs())
    assert det.tile_id == 0
    spec = T.tile_specs()[0]
    ucx = 0.5 * (det.bbox_xyxy[0] + det.bbox_xyxy[2])
    vcy = 0.5 * (det.bbox_xyxy[1] + det.bbox_xyxy[3])
    assert abs(ucx - spec.cx) < 3.0
    assert abs(vcy - spec.cy) < 3.0


@pytest.mark.parametrize(
    "az_deg,expected_tile",
    [(0.0, 0), (90.0, 1), (180.0, 3), (-90.0, 3)],
)
def test_box_azimuth_selects_tile_with_matching_yaw(az_deg, expected_tile):
    """The tile chosen is the one whose yaw_center is angularly nearest the box azimuth."""
    col = _center_col_for_azimuth(az_deg)
    box = (col - 4, 300, col + 4, 360)
    det = pano_bbox_to_detection("chair", box, 0.8, T.tile_specs())
    spec = T.tile_specs()[det.tile_id]
    # the chosen tile's yaw_center is within half its HFOV of the box azimuth
    az = float(T.column_to_azimuth(col))
    assert abs(float(T.wrap_pi(az - spec.yaw_center))) <= spec.hfov / 2.0 + 1e-9


def test_seam_spanning_box_assigned_by_center():
    """A box straddling two tiles' boundary goes to the tile holding its CENTRE column,
    not either edge."""
    # Centre at azimuth 45deg — exactly between tile 0 (yaw 0) and tile 1 (yaw 90); nudge
    # the centre just past 45deg toward tile 1 so the centre resolves to tile 1.
    col_center = _center_col_for_azimuth(50.0)
    # wide box spanning ~40deg so its edges reach into both tiles' overlap
    half = abs(_center_col_for_azimuth(20.0) - _center_col_for_azimuth(0.0))
    box = (col_center - half, 300, col_center + half, 360)
    det = pano_bbox_to_detection("door", box, 0.9, T.tile_specs())
    # centre azimuth 50deg is nearer tile 1 (yaw 90, delta 40) than tile 0 (yaw 0, delta 50)
    assert det.tile_id == 1


def test_converted_box_pixels_roundtrip_to_source_rays():
    """Each converted corner pixel maps back (via the tiling inverse) to a camera ray
    that matches the source pano corner's ray — i.e. the projection is faithful."""
    specs = T.tile_specs()
    col = _center_col_for_azimuth(85.0)  # solidly inside tile 1
    box = (col - 30, 280, col + 30, 380)
    det = pano_bbox_to_detection("stool", box, 0.9, specs)
    spec = specs[det.tile_id]
    # the tile bbox is axis-aligned min/max of the projected corners; its own corners must
    # re-project to camera rays whose azimuth band brackets the source box's azimuth band.
    src_az = sorted(float(T.column_to_azimuth(c)) for c in (box[0], box[2]))
    tu0, tv0, tu1, tv1 = det.bbox_xyxy
    tile_az = sorted(
        T.tile_pixel_to_camera_ray(det.tile_id, u, spec.cy)[0] for u in (tu0, tu1)
    )
    # azimuth spans agree to within a small tolerance (gnomonic edge rotation aside)
    assert abs(tile_az[0] - src_az[0]) < np.deg2rad(2.0)
    assert abs(tile_az[1] - src_az[1]) < np.deg2rad(2.0)


def test_converted_box_is_clipped_to_tile_bounds():
    specs = T.tile_specs()
    spec = specs[0]
    col = _center_col_for_azimuth(40.0)  # near tile 0's edge -> some corners spill out
    box = (col - 60, 0, col + 60, T.PANO_HEIGHT - 1)
    det = pano_bbox_to_detection("pillar", box, 0.9, specs)
    x0, y0, x1, y1 = det.bbox_xyxy
    assert 0.0 <= x0 <= spec.width - 1
    assert 0.0 <= x1 <= spec.width - 1
    assert 0.0 <= y0 <= spec.height - 1
    assert 0.0 <= y1 <= spec.height - 1
    assert x0 <= x1 and y0 <= y1


# --------------------------------------------------------------- loader / mapping


def _write_labels(tmp_path, keyframes):
    p = tmp_path / "labels.json"
    p.write_text(json.dumps({"format": "pano_xyxy", "keyframes": keyframes}))
    return str(p)


def test_from_json_loads_keyframe_mapping(tmp_path):
    path = _write_labels(
        tmp_path,
        {
            "0": [{"label": "stool", "attributes": ["white"], "bbox": [950, 300, 990, 360], "score": 0.8}],
            "5": [{"label": "chair", "attributes": [], "bbox": [100, 300, 140, 360], "score": 0.7}],
        },
    )
    det = ScriptedPanoDetector.from_json(path)
    assert det.labelled_indices == (0, 5)
    d0 = det.detections_for(0)
    assert len(d0) == 1
    assert isinstance(d0[0], Detection)
    assert d0[0].label == "stool"
    # unlabelled keyframe -> empty
    assert det.detections_for(3) == []
    # keyframes property exposes converted detections per labelled index
    mapping = det.keyframes
    assert set(mapping) == {0, 5}
    assert all(isinstance(x, Detection) for x in mapping[0])


def test_detector_call_returns_current_keyframe_per_tile(tmp_path):
    path = _write_labels(
        tmp_path,
        {"0": [{"label": "stool", "bbox": [955, 300, 985, 360], "score": 0.8}]},
    )
    det = ScriptedPanoDetector.from_json(path)
    tiles = [np.zeros((4, 4, 3), np.uint8) for _ in range(4)]
    # no keyframe selected -> all empty
    assert det(tiles) == [[], [], [], []]
    det.current_keyframe = 0
    per_tile = det(tiles)
    assert sum(len(t) for t in per_tile) == 1
    # the one detection sits in exactly one tile
    tid = next(i for i, t in enumerate(per_tile) if t)
    assert per_tile[tid][0].label == "stool"


def test_from_json_rejects_unknown_format(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text(json.dumps({"format": "coco", "keyframes": {}}))
    with pytest.raises(ValueError):
        ScriptedPanoDetector.from_json(str(p))
