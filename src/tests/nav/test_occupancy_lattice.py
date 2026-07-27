"""Regression test for issue #90: float32 lattice coords colliding in world_to_cell.

`TerrainPatch.points` is float32 by contract. `float32(n * 0.1)` rounds DOWN by
one ULP for many `n` (e.g. `float32(0.7) == 0.699999988079071`), so an
unguarded `floor((x - origin) / cell_m)` used to collide two distinct lattice
columns/rows into the same occupancy cell, leaving the other permanently
UNKNOWN and fragmenting FREE space into a checkerboard.

A first attempt at this fix (a fixed 1e-6 "cells" epsilon) was itself
insufficient -- it left 2 collisions per axis (200/2601 cells, 7.7%) on the
empty-room repro below, because the true float32 rounding error scales with
coordinate magnitude and inversely with cell_m, and a fixed constant doesn't
track either. The tests below pin the CURRENT magnitude-scaled-tolerance fix
against that concrete regression (not just a loose upper bound), against
float64-promotion arithmetic explicitly (so a numpy version change or a
future float64 terrain dtype can't silently reopen the bug), against exact
cell-boundary determinism (including negative coordinates), and across
several cell_m values.
"""
from __future__ import annotations

import numpy as np
import pytest

from core.interfaces import TerrainPatch
from core.mocks.synthetic_scene import SyntheticScene
from core.nav.occupancy import FREE, OBSTACLE, UNKNOWN, OccupancyGrid, _lattice_floor
from tests.nav.helpers import make_points


def test_exact_decimal_float32_lattice_columns_do_not_collide():
    """Adjacent exact-decimal x's (float32) must land in distinct, consecutive cells."""
    grid = OccupancyGrid(cell_m=0.1)
    # n=7,9,13,14,17 are confirmed pathological: float32(n * 0.1) rounds down
    # by one ULP, so floor((x - 0) / 0.1) used to yield n - 1 instead of n.
    ns = [6, 7, 8, 9, 13, 14, 17, 18]
    xs = np.array([float(np.float32(n * 0.1)) for n in ns], dtype=np.float32)
    pts = np.stack(
        [xs, np.zeros_like(xs), np.zeros_like(xs), np.full_like(xs, 0.02)], axis=1
    ).astype(np.float32)
    from core.interfaces import TerrainPatch

    grid.integrate_patch(TerrainPatch(t=0.0, points=pts, extended=False))

    cols = [grid.world_to_cell(float(x), 0.0)[1] for x in xs]
    # Each distinct sampled x must map to its own distinct column...
    assert len(set(cols)) == len(ns), cols
    # ...and specifically to the intended column index n (not n - 1).
    assert cols == ns, cols


def test_world_to_cell_matches_cell_to_world_round_trip_on_exact_lattice():
    """world_to_cell(cell_to_world(r, c)) must be the identity for every cell,
    even when the world coordinate is first narrowed through float32 (as real
    TerrainPatch points are)."""
    grid = OccupancyGrid(cell_m=0.1)
    for n in range(0, 30):
        x, y = grid.cell_to_world(0, n)
        x32, y32 = float(np.float32(x)), float(np.float32(y))
        row, col = grid.world_to_cell(x32, y32)
        assert col == n, (n, col, x, x32)


def test_synthetic_terrain_lattice_does_not_fragment_free_space():
    """End-to-end repro from issue #90: integrating the full synthetic terrain
    patch into a fresh grid must not leave a checkerboard of spurious UNKNOWN
    cells inside the room bounds -- contiguous FREE space stays contiguous."""
    sc = SyntheticScene(0)
    sc.populate_default(2)
    grid = OccupancyGrid(cell_m=0.1)
    grid.integrate_patch(sc.terrain_patch())

    r0, c0 = grid.world_to_cell(0.0, 0.0)
    r1, c1 = grid.world_to_cell(5.0, 5.0)
    sub = grid.state[r0 : r1 + 1, c0 : c1 + 1]
    total = sub.size
    unknown_frac = (sub == UNKNOWN).sum() / total

    # Before the fix this was ~47% (1232/2601) in a checkerboard pattern purely
    # from float32 rounding. Any remaining UNKNOWN here is real scene structure
    # (e.g. columns/objects the terrain slab didn't sample), not a lattice
    # collision artefact -- bound it well below the old pathological fraction.
    assert unknown_frac < 0.10, unknown_frac

    # The issue's core symptom: fragmenting FREE space into many disconnected
    # components purely from coordinate rounding. Assert the FREE cells form
    # one dominant connected component (4-connectivity BFS) rather than a
    # checkerboard of small islands.
    free_mask = sub == FREE
    visited = np.zeros_like(free_mask)
    components = []
    h, w = free_mask.shape
    for sr in range(h):
        for sc_ in range(w):
            if free_mask[sr, sc_] and not visited[sr, sc_]:
                stack = [(sr, sc_)]
                visited[sr, sc_] = True
                size = 0
                while stack:
                    r, c = stack.pop()
                    size += 1
                    for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                        if (
                            0 <= nr < h
                            and 0 <= nc < w
                            and free_mask[nr, nc]
                            and not visited[nr, nc]
                        ):
                            visited[nr, nc] = True
                            stack.append((nr, nc))
                components.append(size)

    total_free = free_mask.sum()
    largest = max(components)
    # Before the fix: 221 components, largest = 25/1065 (~2%) -- a checkerboard
    # shattering FREE space into tiny islands. After the fix: 9 components,
    # largest = 1571/1986 (~79%) -- the remainder is real room/furniture
    # structure, not lattice-collision fragmentation. Bound well above the
    # pathological regime but below 100% (real structure legitimately splits
    # some floor area, e.g. behind furniture).
    assert len(components) < 30, (len(components), sorted(components)[-10:])
    assert largest / total_free > 0.7, (largest, total_free, sorted(components)[-10:])


def test_hand_built_lattice_free_region_stays_contiguous():
    """Small, self-contained repro not depending on core.mocks: a solid FREE
    slab sampled at exact-decimal float32 coordinates must integrate into a
    fully-connected FREE block with no spurious UNKNOWN holes."""
    grid = OccupancyGrid(cell_m=0.1)
    coords = []
    for i in range(20):
        for j in range(20):
            x = float(np.float32(i * 0.1 + 0.05))
            y = float(np.float32(j * 0.1 + 0.05))
            coords.append((x, y, 0.02))
    grid.integrate_patch(make_points(coords))

    for i in range(20):
        for j in range(20):
            x = float(np.float32(i * 0.1 + 0.05))
            y = float(np.float32(j * 0.1 + 0.05))
            assert grid.is_free(*grid.world_to_cell(x, y)), (i, j)


def test_empty_room_has_zero_unknown_lattice_collisions():
    """Done-criterion 1: a furniture-free scene has nothing to occlude the
    floor, so integrating its full terrain patch must leave ZERO UNKNOWN
    cells in the room interior -- any UNKNOWN here is pure lattice-collision
    loss, not real scene structure. This regressed at 200/2601 (7.7%) under
    the previous fixed-epsilon fix; must be exactly 0 now."""
    sc = SyntheticScene(0)
    sc.populate_default(0)
    grid = OccupancyGrid(cell_m=0.1)
    grid.integrate_patch(sc.terrain_patch())

    r0, c0 = grid.world_to_cell(0.0, 0.0)
    r1, c1 = grid.world_to_cell(5.0, 5.0)
    sub = grid.state[r0 : r1 + 1, c0 : c1 + 1]
    unknown = int((sub == UNKNOWN).sum())
    assert unknown == 0, (unknown, sub.size)


# Coordinates (n, cell_m) pairs where float32(n * cell_m) is known to round
# DOWN by enough to have collided under the old fixed 1e-6 epsilon (see the
# module docstring); n=41 and n=46 at cell_m=0.1 are the two that produced
# the 200/2601 empty-room regression above.
_KNOWN_PATHOLOGICAL_CASES = [
    (0.1, [6, 7, 8, 9, 13, 14, 17, 18, 41, 42, 46, 47]),
    (0.05, list(range(1, 100))),
    (0.25, list(range(1, 100))),
]


@pytest.mark.parametrize("cell_m,ns", _KNOWN_PATHOLOGICAL_CASES)
def test_lattice_floor_correct_under_float64_promotion(cell_m, ns):
    """Done-criterion 2: pin correctness under float64 PROMOTION arithmetic
    explicitly (the numpy 1.x behaviour, and the behaviour a future
    float64-dtype terrain cloud would also hit) -- not just under numpy
    2.x's NEP-50 weak promotion, which happens to keep float32/python-float
    division in float32 and can mask most collisions by luck. Regardless of
    which promotion rule produced the float32 storage error, `_lattice_floor`
    must resolve every n back to its intended index once explicitly widened
    to float64 the way numpy 1.x's cross-dtype promotion used to."""
    for n in ns:
        x32 = np.float32(n * cell_m)
        # Explicit float64-promotion path: widen the float32-rounded value to
        # float64 BEFORE calling the shared helper -- exactly what numpy 1.x's
        # float32-array / python-float division used to do internally.
        x64 = np.float64(x32)
        col = int(_lattice_floor(x64, 0.0, cell_m))
        assert col == n, (cell_m, n, float(x32), float(x64), col)


@pytest.mark.parametrize("cell_m", [0.05, 0.1, 0.25])
def test_lattice_floor_boundary_determinism(cell_m):
    """Done-criterion 3: a point exactly on a cell boundary lands in exactly
    one cell, deterministically (matches the documented convention: cell c
    covers [origin + c*cell_m, origin + (c+1)*cell_m) -- the boundary itself
    belongs to the upper cell c, not c-1), and repeated calls agree."""
    for n in range(-5, 25):
        boundary = n * cell_m
        col_a = int(_lattice_floor(boundary, 0.0, cell_m))
        col_b = int(_lattice_floor(boundary, 0.0, cell_m))
        assert col_a == col_b == n, (cell_m, n, boundary, col_a, col_b)

        # A hair below the boundary must still land in cell n - 1 (the
        # tolerance must not swallow a full cell's worth of genuinely
        # distinct coordinate).
        just_below = boundary - cell_m * 0.4
        assert int(_lattice_floor(just_below, 0.0, cell_m)) == n - 1, (
            cell_m,
            n,
            just_below,
        )

        # A hair above must land in cell n (not skip ahead to n + 1).
        just_above = boundary + cell_m * 0.4
        assert int(_lattice_floor(just_above, 0.0, cell_m)) == n, (cell_m, n, just_above)


def test_lattice_floor_negative_coordinates_no_off_by_one():
    """Done-criterion 3: negative coordinates and the grid origin itself
    (row/col 0) must not off-by-one -- floor() and round-to-nearest disagree
    below zero, so this is checked explicitly rather than assumed to
    generalize from the positive-side tests above."""
    cell_m = 0.1
    cases = [
        (0.0, 0),  # exact origin -> cell 0, not -1
        (-1e-8, 0),  # noise-level negative near zero must still snap to cell 0
        (-0.05, -1),  # halfway into the cell below origin
        (-0.1, -1),  # exact boundary -> belongs to the upper (less-negative) cell
        (-0.1005, -2),  # well past the boundary (beyond the tolerance) -> next cell down
        (-1.0, -10),
    ]
    for x, expected in cases:
        got = int(_lattice_floor(x, 0.0, cell_m))
        assert got == expected, (x, expected, got)


@pytest.mark.parametrize("cell_m", [0.05, 0.1, 0.25])
def test_lattice_floor_sweep_no_collisions_across_cell_sizes(cell_m):
    """Done-criterion 4: sweep n=1..199 float32-decimal coordinates at
    several cell_m values (not just the 0.1 m lattice the issue originally
    measured) and confirm every one maps to a distinct, correct column under
    the CURRENT (float32-storage) arithmetic path."""
    xs = np.array([float(np.float32(n * cell_m)) for n in range(1, 200)], dtype=np.float32)
    grid = OccupancyGrid(cell_m=cell_m)
    pts = np.stack(
        [xs, np.zeros_like(xs), np.zeros_like(xs), np.full_like(xs, 0.02)], axis=1
    ).astype(np.float32)
    grid.integrate_patch(TerrainPatch(t=0.0, points=pts, extended=False))
    cols = [grid.world_to_cell(float(x), 0.0)[1] for x in xs]
    assert cols == list(range(1, 200)), (cell_m, cols)
