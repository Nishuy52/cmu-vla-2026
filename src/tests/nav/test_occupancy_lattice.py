"""Regression test for issue #90: float32 lattice coords colliding in world_to_cell.

`TerrainPatch.points` is float32 by contract. `float32(n * 0.1)` rounds DOWN by
one ULP for many `n` (e.g. `float32(0.7) == 0.699999988079071`), so an
unguarded `floor((x - origin) / cell_m)` used to collide two distinct lattice
columns/rows into the same occupancy cell, leaving the other permanently
UNKNOWN and fragmenting FREE space into a checkerboard.
"""
from __future__ import annotations

import numpy as np

from core.mocks.synthetic_scene import SyntheticScene
from core.nav.occupancy import FREE, OBSTACLE, UNKNOWN, OccupancyGrid
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
