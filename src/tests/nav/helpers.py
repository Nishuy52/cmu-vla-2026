"""Hand-built terrain fixtures for nav tests.

Deliberately does NOT import core.mocks (that scaffold is concurrent/owned
elsewhere). We build TerrainPatch clouds directly from tiny ASCII maps so tests
are self-contained and readable.
"""
from __future__ import annotations

import numpy as np

from core.interfaces import TerrainPatch
from core.nav.occupancy import CELL_M, OccupancyGrid


def patch_from_ascii(
    rows: list[str],
    *,
    cell_m: float = CELL_M,
    origin_x: float = 0.0,
    origin_y: float = 0.0,
    free_h: float = 0.02,
    obs_h: float = 0.5,
    extended: bool = False,
    t: float = 0.0,
) -> TerrainPatch:
    """Build a TerrainPatch from an ASCII map.

    Characters: '.' = free (intensity free_h), '#' = obstacle (intensity obs_h),
    ' ' = no point emitted (stays UNKNOWN). rows[0] is the TOP row; we map top row
    to the HIGHEST y so the picture matches map-frame orientation (y up).

    One terrain point is emitted at each cell centre for '.' and '#'.
    """
    h = len(rows)
    pts: list[tuple[float, float, float, float]] = []
    for i, line in enumerate(rows):
        # Top row (i=0) -> highest y. row-index in grid terms = (h-1-i).
        gy = h - 1 - i
        y = origin_y + (gy + 0.5) * cell_m
        for j, ch in enumerate(line):
            x = origin_x + (j + 0.5) * cell_m
            if ch == ".":
                pts.append((x, y, 0.0, free_h))
            elif ch == "#":
                pts.append((x, y, 0.0, obs_h))
            # ' ' -> skip (unknown)
    arr = np.array(pts, dtype=np.float32) if pts else np.zeros((0, 4), np.float32)
    return TerrainPatch(t=t, points=arr, extended=extended)


def grid_from_ascii(rows: list[str], **kw) -> OccupancyGrid:
    """Build an OccupancyGrid pre-filled from one ASCII patch (origin at 0,0)."""
    cell_m = kw.pop("cell_m", CELL_M)
    grid = OccupancyGrid(cell_m=cell_m)
    grid.integrate_patch(patch_from_ascii(rows, cell_m=cell_m, **kw))
    return grid


def make_points(coords_inten: list[tuple[float, float, float]], t: float = 0.0) -> TerrainPatch:
    """TerrainPatch from explicit [(x, y, intensity), ...] (z=0)."""
    arr = np.array([(x, y, 0.0, i) for (x, y, i) in coords_inten], dtype=np.float32)
    return TerrainPatch(t=t, points=arr, extended=False)
