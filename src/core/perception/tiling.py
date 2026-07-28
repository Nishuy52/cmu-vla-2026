"""Gnomonic (pinhole) tiling of the 1920x640 equirectangular panorama.

The challenge camera is a 360deg-H / 120deg-V equirectangular strip
(``docs/upstream_notes.md`` gotcha 7; :class:`core.interfaces.PanoFrame`). Open-vocab
detectors expect ordinary pinhole frames, so we reproject the strip into four
90deg-HFOV gnomonic tiles (``docs/architecture.md`` §6 "Panorama handling"). Each
tile shares the panorama's full 120deg VFOV. With the default 4 tiles at 90deg
HFOV each, tile spacing is 360/4 = 90deg, equal to the HFOV itself, so adjacent
tiles exactly touch at their edges with 0deg overlap (verify: 360deg / 4 tiles =
90deg spacing vs 90deg hfov -> overlap = hfov - spacing = 0deg). A thin object
straddling a seam can therefore be split across two tiles with nothing to
re-join it (see :func:`tile_specs`).

Geometry, all pure numpy and deterministic:

* **Equirectangular <-> ray.** Per the ``PanoFrame`` docstring, image column 0 is at
  vehicle ``yaw + pi`` and azimuth *decreases* left-to-right, so the centre column
  looks along the vehicle heading. We expose that as a signed convention constant
  (:data:`AZIMUTH_SIGN`) and a yaw-offset constant (:data:`COLUMN0_YAW_OFFSET`) so
  Phase-2 sim calibration can flip either in one place if the real image disagrees.
* **Camera-frame ray.** ``(azimuth, elevation)`` where azimuth is measured about the
  camera +Z (up) axis, 0 straight ahead (along heading), positive CCW; elevation is
  measured from the horizon, positive up. Rows run top->bottom so elevation *decreases*
  with row index (:data:`ELEVATION_SIGN`).
* **Map-frame ray.** Add the vehicle yaw from :class:`core.interfaces.OdomState` to the
  camera azimuth to get a map-frame bearing; elevation is unchanged (the camera is
  assumed level — Phase-2 calibration territory, flagged in the report).

Remap grids are precomputed once per (resolution, n_tiles, hfov, vfov) and cached, so
tiling a stream of frames costs one gather per frame.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

# --------------------------------------------------------------------------- panorama spec

#: Native panorama resolution (H, W) — 1920x640 equirectangular strip.
PANO_HEIGHT: int = 640
PANO_WIDTH: int = 1920

#: Angular field of view of the full strip (radians): 360deg H, 120deg V.
PANO_HFOV: float = 2.0 * np.pi          # 360 deg
PANO_VFOV: float = np.deg2rad(120.0)    # 120 deg

# --------------------------------------------------------------------------- conventions
# These four constants encode the sign/offset conventions in the PanoFrame docstring.
# Phase-2 sim calibration flips them here (one place) if the real image disagrees.

#: Azimuth decreases left-to-right across the panorama columns (PanoFrame docstring).
#: +1 would mean azimuth increases with column index; the contract says it decreases.
AZIMUTH_SIGN: float = -1.0

#: Camera-frame azimuth (rad) at image column 0. The docstring puts column 0 at
#: vehicle ``yaw + pi``; measured in the camera frame (0 = along heading) that is +pi.
COLUMN0_YAW_OFFSET: float = np.pi

#: Elevation decreases top-to-bottom (row 0 = top of the strip = highest elevation).
ELEVATION_SIGN: float = -1.0

#: Default tiling: four 90deg-HFOV pinhole tiles evenly spaced around the 360deg strip.
DEFAULT_N_TILES: int = 4
DEFAULT_TILE_HFOV: float = np.deg2rad(90.0)
DEFAULT_TILE_VFOV: float = PANO_VFOV          # tiles share the full 120deg VFOV
# NOTE: there is no seam overlap at the defaults. spacing = 360/4 = 90deg, equal to
# the 90deg tile HFOV, so overlap = hfov - spacing = 0deg (tiles exactly touch, they
# do not overlap). A previous DEFAULT_SEAM_OVERLAP = 10deg constant here was never
# wired into tile_specs() and misled a root-cause investigation (issue #131); it has
# been removed rather than fixed because seam overlap is not actually implemented.


# --------------------------------------------------------------------------- equirect rays


def column_to_azimuth(col: np.ndarray | float, width: int = PANO_WIDTH) -> np.ndarray:
    """Camera-frame azimuth (rad) for panorama column index(es).

    Column 0 -> :data:`COLUMN0_YAW_OFFSET`; azimuth then advances by
    :data:`AZIMUTH_SIGN` * (2pi / width) per column, wrapped to (-pi, pi].
    """
    col = np.asarray(col, dtype=float)
    az = COLUMN0_YAW_OFFSET + AZIMUTH_SIGN * (PANO_HFOV * col / width)
    return wrap_pi(az)


def row_to_elevation(row: np.ndarray | float, height: int = PANO_HEIGHT,
                     vfov: float = PANO_VFOV) -> np.ndarray:
    """Camera-frame elevation (rad) for panorama row index(es).

    Row 0 is the top of the strip (+vfov/2); the bottom row is -vfov/2. Uses
    pixel-centre sampling so the mapping is symmetric about the horizon.
    """
    row = np.asarray(row, dtype=float)
    # fractional position of the pixel centre in [0, 1], top->bottom
    frac = (row + 0.5) / height
    return ELEVATION_SIGN * (frac - 0.5) * vfov


def azimuth_to_column(az: np.ndarray | float, width: int = PANO_WIDTH) -> np.ndarray:
    """Inverse of :func:`column_to_azimuth`: azimuth (rad) -> fractional column."""
    az = np.asarray(az, dtype=float)
    col = (wrap_pi(az) - COLUMN0_YAW_OFFSET) / (AZIMUTH_SIGN * PANO_HFOV / width)
    return np.mod(col, width)


def elevation_to_row(el: np.ndarray | float, height: int = PANO_HEIGHT,
                     vfov: float = PANO_VFOV) -> np.ndarray:
    """Inverse of :func:`row_to_elevation`: elevation (rad) -> fractional row."""
    el = np.asarray(el, dtype=float)
    frac = el / (ELEVATION_SIGN * vfov) + 0.5
    return frac * height - 0.5


def wrap_pi(a: np.ndarray | float) -> np.ndarray:
    """Wrap angle(s) to the half-open interval (-pi, pi]."""
    a = np.asarray(a, dtype=float)
    return -np.mod(-a + np.pi, 2.0 * np.pi) + np.pi


def camera_ray_to_map(azimuth: float, elevation: float, yaw: float) -> tuple[float, float]:
    """Camera-frame ``(azimuth, elevation)`` -> map-frame ``(bearing, elevation)``.

    Map bearing = camera azimuth + vehicle yaw (both measured CCW from +x/heading),
    wrapped to (-pi, pi]. Elevation is unchanged (camera assumed level; Phase-2
    calibration owns any tilt correction).
    """
    return float(wrap_pi(azimuth + yaw)), float(elevation)


def map_ray_to_camera(bearing: float, elevation: float, yaw: float) -> tuple[float, float]:
    """Inverse of :func:`camera_ray_to_map`: map bearing -> camera azimuth."""
    return float(wrap_pi(bearing - yaw)), float(elevation)


# --------------------------------------------------------------------------- tile spec


@dataclass(frozen=True)
class TileSpec:
    """Immutable description of one pinhole tile carved from the panorama.

    ``yaw_center`` is the camera-frame azimuth (rad) the tile's optical axis points
    along. ``width``/``height`` are the output pixel dims; ``hfov``/``vfov`` its
    angular field of view; ``focal_x``/``focal_y`` the pinhole focal lengths (px)
    implied by that fov (``f = (dim/2) / tan(fov/2)``).
    """

    tile_id: int
    yaw_center: float
    width: int
    height: int
    hfov: float
    vfov: float

    @property
    def focal_x(self) -> float:
        return (self.width / 2.0) / np.tan(self.hfov / 2.0)

    @property
    def focal_y(self) -> float:
        return (self.height / 2.0) / np.tan(self.vfov / 2.0)

    @property
    def cx(self) -> float:
        return (self.width - 1) / 2.0

    @property
    def cy(self) -> float:
        return (self.height - 1) / 2.0


@dataclass(frozen=True)
class TileGrid:
    """A tile's precomputed equirectangular sample grid + its :class:`TileSpec`.

    ``map_x``/``map_y`` are (H, W) float arrays of panorama pixel coordinates to
    sample for each output tile pixel (bilinear-ready). ``azimuth``/``elevation``
    are the (H, W) camera-frame rays of each output pixel — the inverse mapping
    fusion casts through.
    """

    spec: TileSpec
    map_x: np.ndarray  # (H, W) source column in the panorama
    map_y: np.ndarray  # (H, W) source row in the panorama
    azimuth: np.ndarray  # (H, W) camera-frame azimuth per tile pixel
    elevation: np.ndarray  # (H, W) camera-frame elevation per tile pixel


# --------------------------------------------------------------------------- tile geometry


def tile_specs(
    n_tiles: int = DEFAULT_N_TILES,
    hfov: float = DEFAULT_TILE_HFOV,
    vfov: float = DEFAULT_TILE_VFOV,
    tile_width: int | None = None,
    tile_height: int | None = None,
) -> tuple[TileSpec, ...]:
    """Return the :class:`TileSpec` for each of ``n_tiles`` tiles around the strip.

    Tiles are centred evenly every ``2pi / n_tiles`` starting at the panorama
    centre column (camera azimuth 0 = along heading). Overlap between neighbours is
    ``hfov - 2pi/n_tiles``, which is 0deg for the 4x90deg default (360/4 = 90deg
    spacing == 90deg hfov, so adjacent tiles exactly touch rather than overlap).
    Default pixel dims keep the tile's horizontal angular resolution close to the
    source panorama.
    """
    spacing = 2.0 * np.pi / n_tiles
    if tile_width is None:
        # match source horizontal resolution: PANO_WIDTH px span 2pi, tile spans hfov
        tile_width = int(round(PANO_WIDTH * hfov / (2.0 * np.pi)))
    if tile_height is None:
        tile_height = int(round(PANO_HEIGHT * vfov / PANO_VFOV))
    specs: list[TileSpec] = []
    for i in range(n_tiles):
        specs.append(
            TileSpec(
                tile_id=i,
                yaw_center=wrap_pi(i * spacing),
                width=tile_width,
                height=tile_height,
                hfov=hfov,
                vfov=vfov,
            )
        )
    return tuple(specs)


def _build_tile_grid(spec: TileSpec) -> TileGrid:
    """Precompute the equirectangular sample grid + camera rays for one tile.

    Gnomonic (rectilinear) projection: for each output pixel we form the pinhole
    ray direction, convert it to ``(azimuth, elevation)``, and map that back to the
    source panorama pixel via the equirect inverse. Deterministic; called once per
    spec and cached by :func:`tile_grids`.
    """
    us = np.arange(spec.width, dtype=float)
    vs = np.arange(spec.height, dtype=float)
    uu, vv = np.meshgrid(us, vs)  # (H, W)

    # Pinhole ray in the tile's local frame: +z forward, +x right, +y down.
    x = (uu - spec.cx) / spec.focal_x
    y = (vv - spec.cy) / spec.focal_y
    z = np.ones_like(x)

    # Local azimuth (about the tile axis) and elevation of each ray.
    # azimuth increases to the LEFT in camera convention (CCW), so negate x.
    local_az = np.arctan2(-x, z)
    local_el = np.arctan2(-y, np.hypot(x, z))

    azimuth = wrap_pi(spec.yaw_center + local_az)
    elevation = local_el

    map_x = azimuth_to_column(azimuth, PANO_WIDTH)
    map_y = elevation_to_row(elevation, PANO_HEIGHT, PANO_VFOV)

    return TileGrid(
        spec=spec,
        map_x=map_x.astype(float),
        map_y=map_y.astype(float),
        azimuth=azimuth.astype(float),
        elevation=elevation.astype(float),
    )


@lru_cache(maxsize=8)
def _tile_grids_cached(
    n_tiles: int, hfov: float, vfov: float,
    tile_width: int | None, tile_height: int | None,
) -> tuple[TileGrid, ...]:
    specs = tile_specs(n_tiles, hfov, vfov, tile_width, tile_height)
    return tuple(_build_tile_grid(s) for s in specs)


def tile_grids(
    n_tiles: int = DEFAULT_N_TILES,
    hfov: float = DEFAULT_TILE_HFOV,
    vfov: float = DEFAULT_TILE_VFOV,
    tile_width: int | None = None,
    tile_height: int | None = None,
) -> tuple[TileGrid, ...]:
    """Precomputed remap grids for the tiling, cached per parameter set."""
    return _tile_grids_cached(n_tiles, hfov, vfov, tile_width, tile_height)


# --------------------------------------------------------------------------- resampling


def _sample_bilinear(image: np.ndarray, map_x: np.ndarray, map_y: np.ndarray) -> np.ndarray:
    """Bilinearly sample ``image`` (H, W[, C]) at fractional coords, wrapping x.

    x wraps around the 360deg seam (columns are cyclic); y is clamped to the strip.
    """
    h, w = image.shape[:2]
    x0 = np.floor(map_x).astype(int)
    y0 = np.floor(map_y).astype(int)
    x1 = x0 + 1
    y1 = y0 + 1
    wx = map_x - x0
    wy = map_y - y0

    x0m = np.mod(x0, w)
    x1m = np.mod(x1, w)
    y0c = np.clip(y0, 0, h - 1)
    y1c = np.clip(y1, 0, h - 1)

    def gather(yy, xx):
        v = image[yy, xx]
        return v.astype(float)

    if image.ndim == 3:
        wx = wx[..., None]
        wy = wy[..., None]
    top = gather(y0c, x0m) * (1 - wx) + gather(y0c, x1m) * wx
    bot = gather(y1c, x0m) * (1 - wx) + gather(y1c, x1m) * wx
    out = top * (1 - wy) + bot * wy
    return out


def project_tiles(
    image: np.ndarray,
    n_tiles: int = DEFAULT_N_TILES,
    hfov: float = DEFAULT_TILE_HFOV,
    vfov: float = DEFAULT_TILE_VFOV,
) -> list[np.ndarray]:
    """Reproject an equirectangular ``image`` (H, W, C) into ``n_tiles`` pinhole tiles.

    Returns a list of uint8 tile images (matching the input dtype). Grids are
    precomputed/cached, so this is one bilinear gather per tile per frame.
    """
    if image.shape[0] != PANO_HEIGHT or image.shape[1] != PANO_WIDTH:
        # allow non-native resolutions but rebuild grids for that resolution
        grids = _tile_grids_for_resolution(image.shape[0], image.shape[1],
                                           n_tiles, hfov, vfov)
    else:
        grids = tile_grids(n_tiles, hfov, vfov)
    tiles: list[np.ndarray] = []
    for g in grids:
        sampled = _sample_bilinear(image, g.map_x, g.map_y)
        tiles.append(sampled.astype(image.dtype))
    return tiles


@lru_cache(maxsize=8)
def _tile_grids_for_resolution(
    height: int, width: int, n_tiles: int, hfov: float, vfov: float,
) -> tuple[TileGrid, ...]:
    """Grids for a non-native panorama resolution (grids scale to width/height)."""
    specs = tile_specs(n_tiles, hfov, vfov)
    grids: list[TileGrid] = []
    for spec in specs:
        us = np.arange(spec.width, dtype=float)
        vs = np.arange(spec.height, dtype=float)
        uu, vv = np.meshgrid(us, vs)
        x = (uu - spec.cx) / spec.focal_x
        y = (vv - spec.cy) / spec.focal_y
        z = np.ones_like(x)
        local_az = np.arctan2(-x, z)
        local_el = np.arctan2(-y, np.hypot(x, z))
        azimuth = wrap_pi(spec.yaw_center + local_az)
        elevation = local_el
        map_x = azimuth_to_column(azimuth, width)
        map_y = elevation_to_row(elevation, height, vfov)
        grids.append(TileGrid(spec, map_x, map_y, azimuth, elevation))
    return tuple(grids)


# --------------------------------------------------------------------------- inverse mapping


def tile_pixel_to_camera_ray(
    tile_id: int,
    u: float,
    v: float,
    n_tiles: int = DEFAULT_N_TILES,
    hfov: float = DEFAULT_TILE_HFOV,
    vfov: float = DEFAULT_TILE_VFOV,
) -> tuple[float, float]:
    """Tile pixel ``(u, v)`` -> camera-frame ``(azimuth, elevation)`` ray (rad).

    Analytic gnomonic inverse (no grid lookup, so ``u``/``v`` may be fractional and
    off-grid — e.g. a detection bbox centre or foot point).
    """
    spec = tile_specs(n_tiles, hfov, vfov)[tile_id]
    x = (float(u) - spec.cx) / spec.focal_x
    y = (float(v) - spec.cy) / spec.focal_y
    z = 1.0
    local_az = np.arctan2(-x, z)
    local_el = np.arctan2(-y, np.hypot(x, z))
    azimuth = float(wrap_pi(spec.yaw_center + local_az))
    elevation = float(local_el)
    return azimuth, elevation


def tile_pixel_to_map_ray(
    tile_id: int,
    u: float,
    v: float,
    yaw: float,
    n_tiles: int = DEFAULT_N_TILES,
    hfov: float = DEFAULT_TILE_HFOV,
    vfov: float = DEFAULT_TILE_VFOV,
) -> tuple[float, float]:
    """Tile pixel -> map-frame ``(bearing, elevation)`` ray given the vehicle ``yaw``."""
    az, el = tile_pixel_to_camera_ray(tile_id, u, v, n_tiles, hfov, vfov)
    return camera_ray_to_map(az, el, yaw)
