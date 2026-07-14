"""Equivalence property tests for the vectorised frontier hot loop (H14 / F13).

WHAT CHANGED
------------
`core.nav.frontiers._bfs_distances` was rewritten from a per-cell python deque
BFS to a vectorised numpy boolean wavefront (bbox-cropped). `_cluster` and the
`detect_frontiers` scoring loop are UNCHANGED (measured not to be the bottleneck;
see tests/nav/bench_frontiers.py). These tests pin that the rewrite is behaviour-
preserving on randomised grids, per the H14 acceptance criterion:

  same frontiers (set equality on cluster cells) + same reachability verdicts +
  score-relevant fields (size, path_distance, affinity, score) within tolerance.

`_reference_bfs_distances` below is a verbatim copy of the ORIGINAL python-deque
implementation, kept only as the oracle for these tests. It is marked for
deletion once the vectorised path has baked in the sim (H14 hygiene item).

The BFS is unweighted 8-connected (king-move), so the wavefront distances are
bit-identical to the deque BFS, not merely close — the equality tests assert
exact array equality (with np.inf preserved), which is a stronger guarantee than
H14 requires and catches any off-by-one / neighbourhood / snapping regression.
"""
from __future__ import annotations

from collections import deque

import numpy as np
import pytest

from core.nav import frontiers as F
from core.nav.frontiers import Frontier, detect_frontiers, frontier_mask
from core.nav.occupancy import FREE, OBSTACLE, UNKNOWN, OccupancyGrid

_NEIGH8 = F._NEIGH8


# --------------------------------------------------------------------------- oracle
def _reference_bfs_distances(grid: OccupancyGrid, start: tuple[int, int]) -> np.ndarray:
    """VERBATIM copy of the original per-cell python-deque BFS (pre-vectorisation).

    Oracle only — DELETE when the vectorised `_bfs_distances` has baked in the sim.
    """
    h, w = grid.shape
    dist = np.full((h, w), np.inf, dtype=np.float32)
    sr, sc = start
    if not (0 <= sr < h and 0 <= sc < w):
        return dist
    free = grid.state == FREE
    if not free[sr, sc]:
        frs = np.argwhere(free)
        if frs.size == 0:
            return dist
        d2 = (frs[:, 0] - sr) ** 2 + (frs[:, 1] - sc) ** 2
        sr, sc = map(int, frs[int(np.argmin(d2))])
    dist[sr, sc] = 0.0
    dq = deque([(sr, sc)])
    while dq:
        r, c = dq.popleft()
        base = dist[r, c]
        for dr, dc in _NEIGH8:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and free[nr, nc] and np.isinf(dist[nr, nc]):
                dist[nr, nc] = base + 1.0
                dq.append((nr, nc))
    return dist


def _reference_detect_frontiers(grid, vehicle_xy, affinity=None, **kw):
    """`detect_frontiers` with the BFS swapped for the oracle — the full-pipeline
    reference for the equivalence test. Mirrors production so only the BFS differs."""
    orig = F._bfs_distances
    F._bfs_distances = _reference_bfs_distances
    try:
        return detect_frontiers(grid, vehicle_xy, affinity, **kw)
    finally:
        F._bfs_distances = orig


# --------------------------------------------------------------------------- grids
def _random_grid(rng: np.random.Generator, n: int, free_frac: float) -> OccupancyGrid:
    """Random n x n grid with ~`free_frac` FREE, some OBSTACLE, rest UNKNOWN.

    Deliberately NOT room-structured: random noise maximally stresses the wavefront
    (many disconnected FREE pockets, ragged frontiers, unreachable clusters) so the
    reachability/inf handling and clustering are exercised hard.
    """
    r = rng.random((n, n))
    state = np.full((n, n), UNKNOWN, dtype=np.int8)
    state[r < free_frac] = FREE
    state[(r >= free_frac) & (r < free_frac + 0.15)] = OBSTACLE
    grid = OccupancyGrid(cell_m=0.10)
    grid.state = state
    grid.intensity = np.where(state == OBSTACLE, 0.5, 0.0).astype(np.float32)
    grid.observed = state != UNKNOWN
    grid.overhead = np.zeros((n, n), dtype=bool)
    grid.ground_z = np.full((n, n), np.nan, dtype=np.float32)
    return grid


# --------------------------------------------------------------------------- tests
@pytest.mark.parametrize("seed", range(25))
@pytest.mark.parametrize("free_frac", [0.15, 0.35, 0.55])
def test_bfs_distances_bit_identical(seed: int, free_frac: float) -> None:
    """Vectorised BFS == oracle BFS, exactly (incl. np.inf), over random grids and
    random start cells — including starts that fall on non-FREE cells (snapping)."""
    rng = np.random.default_rng(seed * 100 + int(free_frac * 100))
    n = int(rng.integers(20, 60))
    grid = _random_grid(rng, n, free_frac)
    # A mix of FREE, non-FREE, and out-of-bounds starts.
    for _ in range(4):
        sr = int(rng.integers(-2, n + 2))
        sc = int(rng.integers(-2, n + 2))
        got = F._bfs_distances(grid, (sr, sc))
        want = _reference_bfs_distances(grid, (sr, sc))
        # Exact equality with inf preserved (both are float32 king-move counts).
        assert np.array_equal(got, want, equal_nan=False)
        assert np.array_equal(np.isinf(got), np.isinf(want))


def test_bfs_empty_free_all_inf() -> None:
    """No FREE cells anywhere -> all-inf, both implementations."""
    rng = np.random.default_rng(1)
    grid = _random_grid(rng, 15, 0.0)  # free_frac 0 -> no FREE
    grid.state[:] = UNKNOWN
    got = F._bfs_distances(grid, (5, 5))
    assert np.all(np.isinf(got))
    assert np.array_equal(np.isinf(got), np.isinf(_reference_bfs_distances(grid, (5, 5))))


def test_bfs_single_free_cell() -> None:
    """One isolated FREE cell: distance 0 there, inf elsewhere."""
    grid = _random_grid(np.random.default_rng(2), 10, 0.0)
    grid.state[:] = UNKNOWN
    grid.state[4, 4] = FREE
    got = F._bfs_distances(grid, (4, 4))
    want = _reference_bfs_distances(grid, (4, 4))
    assert got[4, 4] == 0.0
    assert np.isinf(got[0, 0])
    assert np.array_equal(got, want)


def _frontier_key(f: Frontier) -> tuple:
    return (f.row, f.col)


@pytest.mark.parametrize("seed", range(20))
@pytest.mark.parametrize("free_frac", [0.25, 0.45])
def test_detect_frontiers_equivalent(seed: int, free_frac: float) -> None:
    """Full `detect_frontiers` equivalence: same set of frontier clusters, same
    reachability verdicts, and score-relevant fields within float tolerance vs the
    oracle-BFS reference — the H14 acceptance criterion."""
    rng = np.random.default_rng(7000 + seed * 13 + int(free_frac * 100))
    n = int(rng.integers(24, 70))
    grid = _random_grid(rng, n, free_frac)

    # An injected non-trivial affinity (depends on xy) so the affinity/score fields
    # are actually exercised, not just the geometric path.
    def aff(xy: tuple[float, float]) -> float:
        return float(np.sin(xy[0]) + 0.5 * np.cos(xy[1]))

    vx, vy = grid.cell_to_world(int(rng.integers(0, n)), int(rng.integers(0, n)))

    got = detect_frontiers(grid, (vx, vy), affinity=aff, min_cluster_size=3)
    want = _reference_detect_frontiers(grid, (vx, vy), affinity=aff, min_cluster_size=3)

    # Same number of surviving clusters.
    assert len(got) == len(want)

    # Set equality on the identifying (row, col) centroid cell — the clustering is
    # unchanged so centroids must match exactly, cluster-for-cluster.
    got_by_key = {_frontier_key(f): f for f in got}
    want_by_key = {_frontier_key(f): f for f in want}
    assert set(got_by_key) == set(want_by_key)

    for key, gf in got_by_key.items():
        wf = want_by_key[key]
        assert gf.size == wf.size, key  # cluster cell-set size identical
        assert gf.xy == wf.xy, key
        # Reachability verdict identical (both finite, or both the 1e6 unreachable
        # sentinel).
        g_reach = gf.path_distance < 1e6
        w_reach = wf.path_distance < 1e6
        assert g_reach == w_reach, key
        assert gf.path_distance == pytest.approx(wf.path_distance, abs=1e-4), key
        assert gf.affinity == pytest.approx(wf.affinity, abs=1e-6), key
        assert gf.score == pytest.approx(wf.score, rel=1e-6, abs=1e-4), key

    # Ranking (best-first) identical.
    assert [_frontier_key(f) for f in got] == [_frontier_key(f) for f in want]


def test_detect_frontiers_equivalent_no_affinity() -> None:
    """Equivalence with the default (None) affinity path on a structured-ish grid."""
    rng = np.random.default_rng(999)
    grid = _random_grid(rng, 50, 0.4)
    vx, vy = grid.cell_to_world(25, 25)
    got = detect_frontiers(grid, (vx, vy))
    want = _reference_detect_frontiers(grid, (vx, vy))
    assert [(f.row, f.col, f.size, round(f.score, 5)) for f in got] == \
           [(f.row, f.col, f.size, round(f.score, 5)) for f in want]
