"""A* planner over a Costmap, with via-segment (corridor) threading support.

A* is 8-connected: orthogonal step cost 1, diagonal sqrt(2). UNKNOWN cells are
traversable at UNKNOWN_COST_MULT (3x) so exploration can route through unexplored
space while still preferring known-FREE routes. Hard-blocked cells (obstacle,
inflation, capsule) are impassable.

Via-segment support (`plan_through`) handles ordered instruction-following legs.
A CORRIDOR_BETWEEN leg must physically cross its gate segment (the gap between two
anchors), not merely pass near a midpoint (architecture §1 row 3 threading check).
We insert the gate midpoint as a waypoint, then VERIFY the path crosses the gate
segment; if it doesn't, we recompute on a pinch-corridor local costmap that stamps
OBSTACLE everywhere outside a narrow corridor around the gate within a 3 m disc, so
A* is forced through the gap.
"""
from __future__ import annotations

import heapq
import math

import numpy as np

from core.nav.costmap import Costmap

# --------------------------------------------------------------------------- tunables
UNKNOWN_COST_MULT: float = 3.0  # traversing an UNKNOWN cell costs 3x a FREE cell
PINCH_DISC_M: float = 3.0  # radius of the local pinch overlay around a gate
PINCH_CORRIDOR_HALF_W_M: float = 0.5  # half-width of the forced corridor through the gate

_DIAG = math.sqrt(2.0)
_STEPS = (
    (-1, 0, 1.0),
    (1, 0, 1.0),
    (0, -1, 1.0),
    (0, 1, 1.0),
    (-1, -1, _DIAG),
    (-1, 1, _DIAG),
    (1, -1, _DIAG),
    (1, 1, _DIAG),
)


def _heuristic(r: int, c: int, gr: int, gc: int) -> float:
    """Octile distance heuristic (admissible for 8-connected unit/diag costs)."""
    dr, dc = abs(r - gr), abs(c - gc)
    return (dr + dc) + (_DIAG - 2.0) * min(dr, dc)


def astar(
    costmap: Costmap, start_xy: tuple[float, float], goal_xy: tuple[float, float]
) -> list[tuple[float, float]] | None:
    """A* from start to goal (world metres). Returns a list of world waypoints, or None.

    UNKNOWN cells cost UNKNOWN_COST_MULT; hard-blocked cells are impassable.
    """
    grid = costmap.grid
    h, w = grid.shape
    sr, sc = grid.world_to_cell(*start_xy)
    gr, gc = grid.world_to_cell(*goal_xy)

    sr, sc = _snap_passable(costmap, sr, sc)
    gr, gc = _snap_passable(costmap, gr, gc)
    if sr is None or gr is None:
        return None

    open_heap: list[tuple[float, int, int, int]] = []
    heapq.heappush(open_heap, (0.0, 0, sr, sc))
    g_cost = np.full((h, w), np.inf, dtype=np.float64)
    g_cost[sr, sc] = 0.0
    came: dict[tuple[int, int], tuple[int, int]] = {}
    closed = np.zeros((h, w), dtype=bool)
    counter = 1

    while open_heap:
        _, _, r, c = heapq.heappop(open_heap)
        if closed[r, c]:
            continue
        closed[r, c] = True
        if (r, c) == (gr, gc):
            return _reconstruct(grid, came, (sr, sc), (gr, gc))
        for dr, dc, base in _STEPS:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w) or closed[nr, nc]:
                continue
            if costmap.blocked(nr, nc):
                continue
            step = base
            if costmap.is_unknown(nr, nc):
                step *= UNKNOWN_COST_MULT
            ng = g_cost[r, c] + step
            if ng < g_cost[nr, nc]:
                g_cost[nr, nc] = ng
                came[(nr, nc)] = (r, c)
                f = ng + _heuristic(nr, nc, gr, gc)
                heapq.heappush(open_heap, (f, counter, nr, nc))
                counter += 1
    return None


def _snap_passable(costmap: Costmap, r: int, c: int):
    grid = costmap.grid
    if grid.in_bounds(r, c) and costmap.passable(r, c):
        return r, c
    nr, nc = costmap._nearest_passable_cell(r, c)
    return nr, nc


def _reconstruct(grid, came, start, goal) -> list[tuple[float, float]]:
    path_cells = [goal]
    cur = goal
    while cur != start:
        cur = came[cur]
        path_cells.append(cur)
    path_cells.reverse()
    return [grid.cell_to_world(r, c) for (r, c) in path_cells]


# --------------------------------------------------------------------------- threading
def segments_cross(
    a0: tuple[float, float], a1: tuple[float, float],
    b0: tuple[float, float], b1: tuple[float, float],
) -> bool:
    """True if segment a0-a1 properly intersects segment b0-b1 (2D)."""
    def orient(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    d1 = orient(b0, b1, a0)
    d2 = orient(b0, b1, a1)
    d3 = orient(a0, a1, b0)
    d4 = orient(a0, a1, b1)
    if ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0)):
        return True
    # Collinear/touching endpoints: treat exact touches on the gate as a crossing.
    def on_seg(p, q, r):
        return (
            min(p[0], r[0]) - 1e-9 <= q[0] <= max(p[0], r[0]) + 1e-9
            and min(p[1], r[1]) - 1e-9 <= q[1] <= max(p[1], r[1]) + 1e-9
        )
    for d, (p, q, r) in (
        (d1, (b0, a0, b1)),
        (d2, (b0, a1, b1)),
        (d3, (a0, b0, a1)),
        (d4, (a0, b1, a1)),
    ):
        if abs(d) < 1e-9 and on_seg(p, q, r):
            return True
    return False


def path_crosses_gate(
    path: list[tuple[float, float]], gate: tuple[tuple[float, float], tuple[float, float]]
) -> bool:
    """True if any path edge crosses the gate segment."""
    g0, g1 = gate
    for p, q in zip(path[:-1], path[1:]):
        if segments_cross(p, q, g0, g1):
            return True
    return False


def _pinch_costmap(
    costmap: Costmap, gate: tuple[tuple[float, float], tuple[float, float]]
) -> Costmap:
    """Overlay: within PINCH_DISC_M of the gate, block everything outside a narrow
    corridor around the gate segment, forcing A* through the gap."""
    g0, g1 = gate
    gmx, gmy = (g0[0] + g1[0]) / 2.0, (g0[1] + g1[1]) / 2.0
    pinch = costmap.clone()
    grid = costmap.grid
    h, w = grid.shape
    rr, cc = np.mgrid[0:h, 0:w]
    cx = grid.origin_x + (cc + 0.5) * grid.cell_m
    cy = grid.origin_y + (rr + 0.5) * grid.cell_m
    from core.nav.costmap import _point_segment_dist

    in_disc = (cx - gmx) ** 2 + (cy - gmy) ** 2 <= PINCH_DISC_M**2
    dist_to_gate = _point_segment_dist(cx, cy, g0[0], g0[1], g1[0], g1[1])
    outside_corridor = dist_to_gate > PINCH_CORRIDOR_HALF_W_M
    pinch.capsule_blocked = pinch.capsule_blocked | (in_disc & outside_corridor)
    return pinch


def plan_through(
    costmap: Costmap,
    start_xy: tuple[float, float],
    legs: list[tuple[str, object]],
) -> list[tuple[float, float]] | None:
    """Plan an ordered multi-leg route.

    legs: ordered [(kind, geometry)] where
      - ("goto", (x, y))                          -> plan to a point
      - ("via_near", (x, y))                      -> waypoint at the point
      - ("corridor_between", ((x0,y0),(x1,y1)))   -> thread the gate segment; the
        planned path MUST cross the gate (verified; recomputed on a pinch corridor
        if the direct plan misses the gap).

    Returns the concatenated world-frame path, or None if any leg is unreachable.
    """
    full: list[tuple[float, float]] = [start_xy]
    cur = start_xy
    for kind, geom in legs:
        if kind == "corridor_between":
            gate = geom  # type: ignore[assignment]
            g0, g1 = gate  # type: ignore[misc]
            mid = ((g0[0] + g1[0]) / 2.0, (g0[1] + g1[1]) / 2.0)
            seg = astar(costmap, cur, mid)
            if seg is None:
                return None
            if not path_crosses_gate(seg, gate):  # type: ignore[arg-type]
                pinch = _pinch_costmap(costmap, gate)  # type: ignore[arg-type]
                seg = astar(pinch, cur, mid)
                if seg is None or not path_crosses_gate(seg, gate):  # type: ignore[arg-type]
                    return None
            full.extend(seg[1:])
            cur = full[-1]
        else:  # goto / via_near
            pt = geom  # type: ignore[assignment]
            seg = astar(costmap, cur, pt)  # type: ignore[arg-type]
            if seg is None:
                return None
            full.extend(seg[1:])
            cur = full[-1]
    return full
