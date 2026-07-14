"""Frontier extraction and scoring over an OccupancyGrid.

A frontier cell is a FREE cell 4-adjacent to at least one UNKNOWN cell — the
boundary of explored space. Frontier cells are clustered by 8-connectivity;
clusters below MIN_CLUSTER_SIZE are discarded as noise. Each surviving cluster
yields a centroid (snapped to the nearest FREE cell in the cluster) scored:

    score = w_size * size
          - w_dist * path_distance          (grid-cell BFS distance from vehicle)
          + w_affinity * affinity(cell_xy)   (question-noun detector-affinity hook)

`affinity` is an injected callable (x, y) -> float; default returns 0 so scoring
is purely geometric (VLFM-style semantic bias is layered in by the caller). We
return frontier centroids ranked best-first.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Callable

import numpy as np

from core.nav.occupancy import FREE, UNKNOWN, OccupancyGrid

# --------------------------------------------------------------------------- tunables
MIN_CLUSTER_SIZE: int = 5  # frontier clusters smaller than this are noise
W_SIZE: float = 1.0  # reward larger frontiers (more unknown to reveal)
W_DIST: float = 0.5  # penalise distance (cells) so we exploit nearby frontiers first
W_AFFINITY: float = 4.0  # weight on the injected semantic affinity term

_NEIGH4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
_NEIGH8 = _NEIGH4 + ((-1, -1), (-1, 1), (1, -1), (1, 1))


@dataclass(frozen=True)
class Frontier:
    """A scored frontier cluster. `xy` is the world-frame centroid (metres)."""

    row: int
    col: int
    xy: tuple[float, float]
    size: int
    path_distance: float
    affinity: float
    score: float


def frontier_mask(grid: OccupancyGrid) -> np.ndarray:
    """Boolean mask: FREE cells 4-adjacent to an UNKNOWN cell."""
    state = grid.state
    free = state == FREE
    unknown = state == UNKNOWN
    # A cell is a frontier if FREE and any 4-neighbour is UNKNOWN. Pad-shift the
    # unknown mask in each direction and OR together.
    adj_unknown = np.zeros_like(unknown)
    adj_unknown[1:, :] |= unknown[:-1, :]
    adj_unknown[:-1, :] |= unknown[1:, :]
    adj_unknown[:, 1:] |= unknown[:, :-1]
    adj_unknown[:, :-1] |= unknown[:, 1:]
    # Grid edges border implicit UNKNOWN (out-of-bounds == unknown): a FREE cell on
    # the array border is also a frontier.
    border = np.zeros_like(unknown)
    border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
    return free & (adj_unknown | border)


def _cluster(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    """8-connected connected components of a boolean mask."""
    h, w = mask.shape
    seen = np.zeros_like(mask)
    clusters: list[list[tuple[int, int]]] = []
    for r0 in range(h):
        for c0 in range(w):
            if not mask[r0, c0] or seen[r0, c0]:
                continue
            comp: list[tuple[int, int]] = []
            dq = deque([(r0, c0)])
            seen[r0, c0] = True
            while dq:
                r, c = dq.popleft()
                comp.append((r, c))
                for dr, dc in _NEIGH8:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        dq.append((nr, nc))
            clusters.append(comp)
    return clusters


def _bfs_distances(grid: OccupancyGrid, start: tuple[int, int]) -> np.ndarray:
    """Grid-cell BFS distance from `start` over FREE cells (8-connected, in cells).

    Unreachable cells stay np.inf. Cheap enough at 0.10 m resolution for the frontier
    counts we see; exact planning uses A* on the costmap, not this.

    Implementation: a vectorised boolean wavefront (numpy-only; scipy is not a
    declared dependency). On an unweighted 8-connected (king-move) grid the BFS
    distance of a cell equals the number of dilation steps to first reach it from
    the start, so each expansion ring is one 3x3 boolean dilation over the FREE
    mask; we stamp the step index the first time each cell is entered. This is
    equivalent to the previous per-cell python deque BFS (identical +1.0 per
    king-move, identical np.inf for unreachable, identical nearest-FREE start
    snapping) but replaces the O(free cells) python loop with O(geodesic diameter)
    vectorised array passes — the frontier hot loop's dominant cost (H14 / F13:
    ~340 ms -> ~15-26 ms on a 200x200 grid).

    The wavefront is cropped to the FREE bounding box: every reachable cell is
    FREE, so the wavefront can never leave that box; on a partially-explored map
    (FREE a compact blob near the vehicle) each pass then touches the observed
    region, not the whole arena. On a fully-explored grid the box is the array and
    the crop is a no-op.
    """
    h, w = grid.shape
    dist = np.full((h, w), np.inf, dtype=np.float32)
    sr, sc = start
    if not (0 <= sr < h and 0 <= sc < w):
        return dist
    free = grid.state == FREE
    if not free[sr, sc]:
        # Snap start to nearest FREE cell so a vehicle sitting on a just-carved cell
        # still yields finite distances. Same nearest-by-squared-euclidean tie-break
        # (first in argwhere/argmin order) as the previous implementation.
        frs = np.argwhere(free)
        if frs.size == 0:
            return dist
        d2 = (frs[:, 0] - sr) ** 2 + (frs[:, 1] - sc) ** 2
        sr, sc = map(int, frs[int(np.argmin(d2))])

    fr_rows, fr_cols = np.nonzero(free)
    r0, r1 = int(fr_rows.min()), int(fr_rows.max()) + 1
    c0, c1 = int(fr_cols.min()), int(fr_cols.max()) + 1
    sub_free = free[r0:r1, c0:c1]
    sh, sw = sub_free.shape

    visited = np.zeros((sh, sw), dtype=bool)
    visited[sr - r0, sc - c0] = True
    sub_dist = np.full((sh, sw), np.inf, dtype=np.float32)
    sub_dist[sr - r0, sc - c0] = 0.0
    wave = visited.copy()  # cells entered on the previous step
    step = 0.0
    while wave.any():
        step += 1.0
        # Dilate the current wavefront by one king-move; keep only new FREE cells.
        nxt = np.zeros((sh, sw), dtype=bool)
        nxt[1:, :] |= wave[:-1, :]
        nxt[:-1, :] |= wave[1:, :]
        nxt[:, 1:] |= wave[:, :-1]
        nxt[:, :-1] |= wave[:, 1:]
        nxt[1:, 1:] |= wave[:-1, :-1]
        nxt[1:, :-1] |= wave[:-1, 1:]
        nxt[:-1, 1:] |= wave[1:, :-1]
        nxt[:-1, :-1] |= wave[1:, 1:]
        nxt &= sub_free & ~visited
        if not nxt.any():
            break
        sub_dist[nxt] = step
        visited |= nxt
        wave = nxt
    dist[r0:r1, c0:c1] = sub_dist
    return dist


def detect_frontiers(
    grid: OccupancyGrid,
    vehicle_xy: tuple[float, float],
    affinity: Callable[[tuple[float, float]], float] | None = None,
    *,
    min_cluster_size: int = MIN_CLUSTER_SIZE,
    w_size: float = W_SIZE,
    w_dist: float = W_DIST,
    w_affinity: float = W_AFFINITY,
) -> list[Frontier]:
    """Return frontier clusters ranked best-first (highest score first).

    vehicle_xy: current vehicle position (metres, map frame) — the BFS distance root.
    affinity: (x, y) -> float semantic bias; None == constant 0.
    """
    if affinity is None:
        affinity = lambda _xy: 0.0  # noqa: E731

    mask = frontier_mask(grid)
    clusters = [c for c in _cluster(mask) if len(c) >= min_cluster_size]
    if not clusters:
        return []

    dist = _bfs_distances(grid, grid.world_to_cell(*vehicle_xy))

    out: list[Frontier] = []
    for comp in clusters:
        rows = np.array([p[0] for p in comp])
        cols = np.array([p[1] for p in comp])
        # Geometric centroid, snapped to the nearest actual cluster cell.
        mr, mc = float(rows.mean()), float(cols.mean())
        k = int(np.argmin((rows - mr) ** 2 + (cols - mc) ** 2))
        cr, cc = int(rows[k]), int(cols[k])
        xy = grid.cell_to_world(cr, cc)

        # Path distance = min BFS distance over the cluster (nearest reachable point).
        cd = dist[rows, cols]
        pd = float(np.min(cd)) if np.any(np.isfinite(cd)) else float("inf")
        if not np.isfinite(pd):
            # Unreachable cluster: heavily deprioritise but still surface it.
            pd = 1e6
        aff = float(affinity(xy))
        score = w_size * len(comp) - w_dist * pd + w_affinity * aff
        out.append(
            Frontier(
                row=cr,
                col=cc,
                xy=xy,
                size=len(comp),
                path_distance=pd,
                affinity=aff,
                score=score,
            )
        )

    out.sort(key=lambda f: f.score, reverse=True)
    return out
