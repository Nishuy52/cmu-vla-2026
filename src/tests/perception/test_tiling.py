"""Gnomonic tiling: equirect<->ray round-trips, seam continuity, FOV coverage."""
from __future__ import annotations

import numpy as np

from core.perception import tiling as T


# --------------------------------------------------------------- spec / coverage


def test_four_tiles_cover_360_with_overlap():
    specs = T.tile_specs()
    assert len(specs) == 4
    centers = sorted(np.rad2deg(s.yaw_center) for s in specs)
    # centres every 90deg (wrapped): -90, 0, 90, 180
    assert np.allclose(centers, [-90.0, 0.0, 90.0, 180.0])
    # each tile is 90deg HFOV; spacing 90deg -> 0deg gap but the grids overlap because
    # a 90deg-HFOV pinhole tile's corners reach past +-45deg. Verify HFOV/VFOV values.
    for s in specs:
        assert np.isclose(np.rad2deg(s.hfov), 90.0)
        assert np.isclose(np.rad2deg(s.vfov), 120.0)


def test_tile_grid_spans_expected_fov():
    g = T.tile_grids()[0]
    assert np.rad2deg(g.azimuth.max()) > 40.0
    assert np.rad2deg(g.azimuth.min()) < -40.0
    assert np.rad2deg(g.elevation.max()) > 55.0
    assert np.rad2deg(g.elevation.min()) < -55.0


def test_tile_focal_matches_fov():
    s = T.tile_specs()[0]
    # f = (w/2)/tan(hfov/2); for 90deg hfov, tan(45)=1 -> f = w/2
    assert np.isclose(s.focal_x, s.width / 2.0)


# --------------------------------------------------------------- equirect round trip


def test_column_azimuth_round_trip():
    cols = np.array([0, 480, 960, 1440, 1919], dtype=float)
    az = T.column_to_azimuth(cols)
    back = T.azimuth_to_column(az)
    # columns recovered modulo width
    assert np.allclose(np.mod(back, T.PANO_WIDTH), np.mod(cols, T.PANO_WIDTH), atol=1e-6)


def test_row_elevation_round_trip():
    rows = np.array([0, 160, 320, 480, 639], dtype=float)
    el = T.row_to_elevation(rows)
    back = T.elevation_to_row(el)
    assert np.allclose(back, rows, atol=1e-6)


def test_center_column_looks_along_heading():
    # per PanoFrame docstring the image centre column looks along the vehicle heading
    az = T.column_to_azimuth(T.PANO_WIDTH / 2.0)
    assert abs(float(az)) < 1e-6


def test_column0_at_yaw_plus_pi():
    az = T.column_to_azimuth(0.0)
    assert np.isclose(abs(float(az)), np.pi, atol=1e-6)


def test_equator_row_is_zero_elevation():
    # pixel-centre convention: elevation ~0 near the middle two rows
    el_lo = T.row_to_elevation(T.PANO_HEIGHT / 2 - 1)
    el_hi = T.row_to_elevation(T.PANO_HEIGHT / 2)
    assert el_lo > 0 > el_hi
    assert abs(float(el_lo) + float(el_hi)) < 1e-6  # symmetric about horizon


# --------------------------------------------------------------- pixel->ray->pixel


def test_tile_pixel_to_ray_round_trip_via_equirect():
    """tile pixel -> camera ray -> panorama pixel -> back to camera ray must agree."""
    for tid in range(4):
        spec = T.tile_specs()[tid]
        for u, v in [(spec.cx, spec.cy), (20, 20), (spec.width - 20, spec.height - 20),
                     (spec.width - 20, 20)]:
            az, el = T.tile_pixel_to_camera_ray(tid, u, v)
            col = T.azimuth_to_column(az)
            row = T.elevation_to_row(el)
            az2 = T.column_to_azimuth(col)
            el2 = T.row_to_elevation(row)
            # azimuth compared on the circle
            daz = float(T.wrap_pi(az - az2))
            assert abs(daz) < 1e-6
            assert abs(el - el2) < 1e-6


def test_tile_center_ray_matches_tile_yaw():
    for tid in range(4):
        spec = T.tile_specs()[tid]
        az, el = T.tile_pixel_to_camera_ray(tid, spec.cx, spec.cy)
        assert abs(float(T.wrap_pi(az - spec.yaw_center))) < 1e-6
        assert abs(el) < 1e-6


# --------------------------------------------------------------- seam continuity


def test_seam_continuity_across_pi():
    """A ray just past +pi wraps to -pi continuously in column space."""
    eps = np.deg2rad(0.5)
    col_a = T.azimuth_to_column(np.pi - eps)
    col_b = T.azimuth_to_column(-np.pi + eps)
    # both near column 0 / width seam; their wrapped distance is small
    d = abs(float(col_a) - float(col_b))
    d = min(d, T.PANO_WIDTH - d)
    # 0.5deg either side of the seam ~= 2.67 columns each -> ~5.3 total; must stay small
    assert d < 6.0  # continuous across the seam (no big jump)


def test_wrap_pi_bounds():
    vals = np.array([-3 * np.pi, -np.pi, 0.0, np.pi, 3 * np.pi, 5.0])
    w = T.wrap_pi(vals)
    assert np.all(w > -np.pi - 1e-9)
    assert np.all(w <= np.pi + 1e-9)


# --------------------------------------------------------------- map-frame rays


def test_camera_ray_to_map_adds_yaw():
    yaw = np.deg2rad(30.0)
    b, e = T.camera_ray_to_map(np.deg2rad(10.0), np.deg2rad(5.0), yaw)
    assert np.isclose(np.rad2deg(b), 40.0)
    assert np.isclose(np.rad2deg(e), 5.0)


def test_map_ray_round_trip():
    yaw = np.deg2rad(123.0)
    b, e = T.camera_ray_to_map(np.deg2rad(20.0), np.deg2rad(-8.0), yaw)
    az, el = T.map_ray_to_camera(b, e, yaw)
    assert abs(float(T.wrap_pi(az - np.deg2rad(20.0)))) < 1e-9
    assert np.isclose(el, np.deg2rad(-8.0))


def test_map_ray_wraps_across_seam_with_yaw():
    # camera az +170deg with yaw +20deg -> map bearing +190 -> wraps to -170
    b, _ = T.camera_ray_to_map(np.deg2rad(170.0), 0.0, np.deg2rad(20.0))
    assert np.isclose(np.rad2deg(b), -170.0)


def test_tile_pixel_to_map_ray_uses_yaw():
    spec = T.tile_specs()[0]
    yaw = np.deg2rad(45.0)
    b, e = T.tile_pixel_to_map_ray(0, spec.cx, spec.cy, yaw)
    assert np.isclose(np.rad2deg(b), 45.0, atol=1e-3)


# --------------------------------------------------------------- projection


def test_project_tiles_shapes_and_dtype():
    img = np.zeros((T.PANO_HEIGHT, T.PANO_WIDTH, 3), dtype=np.uint8)
    tiles = T.project_tiles(img)
    assert len(tiles) == 4
    for tile in tiles:
        assert tile.dtype == np.uint8
        assert tile.ndim == 3 and tile.shape[2] == 3


def test_project_tiles_samples_correct_azimuth_band():
    """A bright vertical stripe at the panorama centre column lands in the front tile
    (tile 0, yaw 0) near its centre, not in the rear tile."""
    img = np.zeros((T.PANO_HEIGHT, T.PANO_WIDTH, 3), dtype=np.uint8)
    img[:, T.PANO_WIDTH // 2 - 3: T.PANO_WIDTH // 2 + 3, :] = 255  # centre = heading
    tiles = T.project_tiles(img)
    front = tiles[0].astype(float).mean(axis=(0, 2))  # column-wise brightness
    # brightest column of the front tile is near its centre
    assert abs(int(np.argmax(front)) - tiles[0].shape[1] // 2) < 10
    # rear tile (tile 2) sees essentially nothing from the front stripe
    assert tiles[2].max() < 10


def test_grids_are_cached():
    a = T.tile_grids()
    b = T.tile_grids()
    assert a is b  # lru_cache returns the identical object


def test_deterministic_projection():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (T.PANO_HEIGHT, T.PANO_WIDTH, 3), dtype=np.uint8)
    t1 = T.project_tiles(img)
    t2 = T.project_tiles(img)
    for a, b in zip(t1, t2):
        assert np.array_equal(a, b)
