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
    costmap: Costmap,
    start_xy: tuple[float, float],
    goal_xy: tuple[float, float],
    *,
    unknown_cost_mult: float = UNKNOWN_COST_MULT,
) -> list[tuple[float, float]] | None:
    """A* from start to goal (world metres). Returns a list of world waypoints, or None.

    UNKNOWN cells cost ``unknown_cost_mult`` (default ``UNKNOWN_COST_MULT``, the calibration
    seam for ``nav.unknown_cost_mult``); hard-blocked cells are impassable.
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
                step *= unknown_cost_mult
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
    costmap: Costmap,
    gate: tuple[tuple[float, float], tuple[float, float]],
    *,
    pinch_disc_m: float = PINCH_DISC_M,
    pinch_corridor_half_w_m: float = PINCH_CORRIDOR_HALF_W_M,
) -> Costmap:
    """Overlay: within ``pinch_disc_m`` of the gate, block everything outside a narrow
    corridor (``pinch_corridor_half_w_m`` half-width) around the gate segment, forcing
    A* through the gap. Defaults are the calibration seams for ``nav.pinch_disc_m`` /
    ``nav.pinch_corridor_half_w_m``."""
    g0, g1 = gate
    gmx, gmy = (g0[0] + g1[0]) / 2.0, (g0[1] + g1[1]) / 2.0
    pinch = costmap.clone()
    grid = costmap.grid
    h, w = grid.shape
    rr, cc = np.mgrid[0:h, 0:w]
    cx = grid.origin_x + (cc + 0.5) * grid.cell_m
    cy = grid.origin_y + (rr + 0.5) * grid.cell_m
    from core.nav.costmap import _point_segment_dist

    in_disc = (cx - gmx) ** 2 + (cy - gmy) ** 2 <= pinch_disc_m**2
    dist_to_gate = _point_segment_dist(cx, cy, g0[0], g0[1], g1[0], g1[1])
    outside_corridor = dist_to_gate > pinch_corridor_half_w_m
    pinch.capsule_blocked = pinch.capsule_blocked | (in_disc & outside_corridor)
    return pinch


def free_space_via_point(
    costmap: Costmap,
    anchor_xy: tuple[float, float],
    start_xy: tuple[float, float],
    near_thresh_m: float,
    *,
    ring_tol_m: float = 0.3,
) -> tuple[float, float] | None:
    """A "path near the anchor" waypoint placed by free-space gradient (IF-F7).

    Replaces the fixed ``anchor + (+1.2, 0)`` offset, which ignored walls, approach
    direction and object size and could land inside a wall or on the unreachable side.

    Candidate cells are those at ~``near_thresh_m`` from ``anchor_xy`` (a ring of
    half-width ``ring_tol_m``) that are passable and reachable from ``start_xy``. Among
    them we pick the one with maximum clearance (distance to the nearest blocked cell) —
    the most comfortably-open passable cell on the robot's reachable side of the anchor.
    Returns its world centre, or None if no reachable near-cell exists (the caller then
    falls back to the anchor centroid projection).
    """
    grid = costmap.grid
    h, w = grid.shape
    ar, ac = grid.world_to_cell(*anchor_xy)
    near_cells = int(round(near_thresh_m / grid.cell_m))
    tol_cells = max(1, int(round(ring_tol_m / grid.cell_m)))
    lo = max(0, near_cells - tol_cells)
    hi = near_cells + tol_cells

    reachable = _reachable_mask(costmap, start_xy)
    if reachable is None:
        return None
    clearance = _clearance_field(costmap)

    best: tuple[float, float] | None = None
    best_clear = -1.0
    lo2, hi2 = lo * lo, hi * hi
    for dr in range(-hi, hi + 1):
        for dc in range(-hi, hi + 1):
            d2 = dr * dr + dc * dc
            if d2 < lo2 or d2 > hi2:
                continue
            r, c = ar + dr, ac + dc
            if not (0 <= r < h and 0 <= c < w) or not reachable[r, c]:
                continue
            cl = float(clearance[r, c])
            if cl > best_clear:
                best_clear = cl
                best = grid.cell_to_world(r, c)
    return best


def _reachable_mask(costmap: Costmap, start_xy: tuple[float, float]):
    """Boolean (h, w) mask of passable cells reachable from ``start_xy`` (8-connected BFS).

    Returns None if the start cannot be snapped to any passable cell.
    """
    grid = costmap.grid
    h, w = grid.shape
    sr, sc = grid.world_to_cell(*start_xy)
    sr, sc = _snap_passable(costmap, sr, sc)
    if sr is None:
        return None
    from collections import deque

    seen = np.zeros((h, w), dtype=bool)
    seen[sr, sc] = True
    dq = deque([(sr, sc)])
    steps = ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1))
    while dq:
        r, c = dq.popleft()
        for dr, dc in steps:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and costmap.passable(nr, nc):
                seen[nr, nc] = True
                dq.append((nr, nc))
    return seen


def _clearance_field(costmap: Costmap) -> np.ndarray:
    """Per-cell distance (in cells) to the nearest blocked cell — a clearance heuristic.

    Multi-source BFS from every blocked cell. Passable cells far from any obstacle score
    high; cells hugging a wall score low. Used to prefer the most-open via candidate (IF-F7).
    """
    grid = costmap.grid
    h, w = grid.shape
    from collections import deque

    dist = np.full((h, w), np.inf, dtype=np.float64)
    blocked = costmap.base_blocked | costmap.capsule_blocked
    brs, bcs = np.nonzero(blocked)
    dq: deque[tuple[int, int]] = deque()
    for r, c in zip(brs.tolist(), bcs.tolist()):
        dist[r, c] = 0.0
        dq.append((r, c))
    steps = ((-1, 0), (1, 0), (0, -1), (0, 1))
    while dq:
        r, c = dq.popleft()
        base = dist[r, c]
        for dr, dc in steps:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and dist[nr, nc] > base + 1.0:
                dist[nr, nc] = base + 1.0
                dq.append((nr, nc))
    if not np.isfinite(dist).any():
        dist[:] = float(max(h, w))  # all-free costmap: every cell maximally clear
    else:
        dist[~np.isfinite(dist)] = 0.0
    return dist


def plan_through(
    costmap: Costmap,
    start_xy: tuple[float, float],
    legs: list[tuple[str, object]],
    *,
    unknown_cost_mult: float = UNKNOWN_COST_MULT,
    pinch_disc_m: float = PINCH_DISC_M,
    pinch_corridor_half_w_m: float = PINCH_CORRIDOR_HALF_W_M,
) -> list[tuple[float, float]] | None:
    """Plan an ordered multi-leg route.

    legs: ordered [(kind, geometry)] where
      - ("goto", (x, y))                          -> plan to a point
      - ("via_near", (x, y))                      -> waypoint at the point
      - ("corridor_between", ((x0,y0),(x1,y1)))   -> thread the gate segment; the
        planned path MUST cross the gate (verified; recomputed on a pinch corridor
        if the direct plan misses the gap).

    ``unknown_cost_mult`` / ``pinch_disc_m`` / ``pinch_corridor_half_w_m`` are the
    calibration seams for ``nav.unknown_cost_mult`` / ``nav.pinch_disc_m`` /
    ``nav.pinch_corridor_half_w_m``; defaults reproduce today's behaviour. The pinch
    overlay is engaged ONLY as a fallback for a ``corridor_between`` leg whose direct
    A* plan misses the gate — open (non-corridor) legs never see it.

    Returns the concatenated world-frame path, or None if any leg is unreachable.
    """
    full: list[tuple[float, float]] = [start_xy]
    cur = start_xy
    for kind, geom in legs:
        if kind == "corridor_between":
            gate = geom  # type: ignore[assignment]
            g0, g1 = gate  # type: ignore[misc]
            mid = ((g0[0] + g1[0]) / 2.0, (g0[1] + g1[1]) / 2.0)
            seg = astar(costmap, cur, mid, unknown_cost_mult=unknown_cost_mult)
            if seg is None:
                return None
            if not path_crosses_gate(seg, gate):  # type: ignore[arg-type]
                pinch = _pinch_costmap(
                    costmap,
                    gate,  # type: ignore[arg-type]
                    pinch_disc_m=pinch_disc_m,
                    pinch_corridor_half_w_m=pinch_corridor_half_w_m,
                )
                seg = astar(pinch, cur, mid, unknown_cost_mult=unknown_cost_mult)
                if seg is None or not path_crosses_gate(seg, gate):  # type: ignore[arg-type]
                    return None
            full.extend(seg[1:])
            cur = full[-1]
        else:  # goto / via_near
            pt = geom  # type: ignore[assignment]
            seg = astar(costmap, cur, pt, unknown_cost_mult=unknown_cost_mult)  # type: ignore[arg-type]
            if seg is None:
                return None
            full.extend(seg[1:])
            cur = full[-1]
    return full
