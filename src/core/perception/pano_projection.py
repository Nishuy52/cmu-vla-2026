"""Vectorized forward projection of map-frame points onto the native panorama.

Single source of truth for the map-point -> panorama-pixel mapping used by both the
offline colored-cloud tool (``tools/colored_cloud.py``) and the live colored voxel
map (``core/perception/colored_map.py``). The forward mapping is the mirror image of
``fusion._points_in_frustum`` (apex-relative ``arctan2`` bearing/elevation) composed
with the calibrated :mod:`core.perception.tiling` functions — no sign conventions are
re-derived here, they are pulled from ``tiling`` (``AZIMUTH_SIGN`` / ``COLUMN0_YAW_OFFSET``
/ ``ELEVATION_SIGN``) via the tiling functions.

Gates (identical to the T5 batch tool it was extracted from):

* **VFOV band.** The panorama covers only +-60 deg elevation; a point projecting to a
  row outside ``[0, PANO_HEIGHT)`` is invalid and never clamped to an edge row.
* **Azimuth wrap.** Columns are cyclic (``col % PANO_WIDTH``); rows are not.
* **Min range.** Points closer than ``min_range`` (or within numerical zero of the
  apex) are invalid.

Colors are sampled nearest-neighbour with a gray fill for invalid points; no bilinear
interpolation, so misregistration stays visible as crisp ghosting (the diagnostic signal).
"""
from __future__ import annotations

import numpy as np

from core.interfaces import OdomState
from core.perception import tiling

#: Color given to points that fall outside the panorama (out of VFOV / range-gated).
GRAY = np.array([128, 128, 128], dtype=np.uint8)

#: Default forward-projection tunables (not calibration-ledger entries).
DEFAULT_MIN_RANGE: float = 0.75  # m — kills the near-field parallax miscoloring tail
_NEAR_ZERO_RANGE: float = 1e-6  # m — degenerate apex-coincident points


def project_points_to_pano(
    points: np.ndarray,
    odom: OdomState,
    *,
    min_range: float = DEFAULT_MIN_RANGE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Forward-project map-frame ``points`` (N, 3) onto the native panorama.

    Returns ``(rows, cols, valid_mask)`` — integer row/column indices into the
    1920x640 equirect strip and a boolean mask of which points landed on it.

    Apex is ``(odom.x, odom.y, odom.z)``; bearing/elevation are the apex-relative
    ``arctan2`` of ``fusion._points_in_frustum``. The map bearing is converted to a
    camera azimuth exactly as the scalar :func:`tiling.map_ray_to_camera` does
    (``wrap_pi(bearing - yaw)``) — vectorized here because that helper is scalar —
    then to fractional pixels via the array-capable
    :func:`tiling.azimuth_to_column` / :func:`tiling.elevation_to_row`.

    Columns wrap (``col % PANO_WIDTH``); rows do **not** — the panorama covers only
    +-60 deg elevation, so any point projecting outside ``[0, PANO_HEIGHT)`` is
    invalid and never clamped. Points closer than ``min_range`` (or within numerical
    zero of the apex) are also invalid.
    """
    points = np.asarray(points, dtype=np.float64)
    n = len(points)
    rows = np.zeros(n, dtype=np.int64)
    cols = np.zeros(n, dtype=np.int64)
    if n == 0:
        return rows, cols, np.zeros(0, dtype=bool)

    apex = np.array([odom.x, odom.y, odom.z], dtype=np.float64)
    rel = points - apex[None, :]
    dx, dy, dz = rel[:, 0], rel[:, 1], rel[:, 2]
    horiz = np.hypot(dx, dy)
    ranges = np.hypot(horiz, dz)
    bearing = np.arctan2(dy, dx)
    elevation = np.arctan2(dz, horiz)

    # Map bearing -> camera azimuth, vectorized twin of scalar tiling.map_ray_to_camera.
    azimuth = tiling.wrap_pi(bearing - odom.yaw)

    col_f = tiling.azimuth_to_column(azimuth)  # already modulo width -> [0, W)
    row_f = tiling.elevation_to_row(elevation)  # may fall outside [0, H)

    cols_all = np.mod(np.floor(col_f).astype(np.int64), tiling.PANO_WIDTH)
    rows_all = np.floor(row_f).astype(np.int64)

    in_vfov = (rows_all >= 0) & (rows_all < tiling.PANO_HEIGHT)
    in_range = ranges >= max(min_range, _NEAR_ZERO_RANGE)
    valid = in_vfov & in_range

    rows[valid] = rows_all[valid]
    cols[valid] = cols_all[valid]
    return rows, cols, valid


def sample_colors(
    image: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    """Nearest-neighbour RGB sample of ``image`` (H, W, 3) at ``(rows, cols)``.

    Invalid points get :data:`GRAY`. No bilinear interpolation — the geometric
    error budget is tens of pixels near-field, and single-source sampling keeps
    misregistration visible as crisp ghosting.
    """
    out = np.broadcast_to(GRAY, (len(rows), 3)).copy()
    if valid.any():
        vr = rows[valid]
        vc = cols[valid]
        out[valid] = image[vr, vc, :3].astype(np.uint8)
    return out
