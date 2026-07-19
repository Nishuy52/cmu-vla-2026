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

from core.geometry.primitives import usable_gate_point
from core.nav.costmap import Costmap, _point_segment_dist

# --------------------------------------------------------------------------- tunables
UNKNOWN_COST_MULT: float = 3.0  # traversing an UNKNOWN cell costs 3x a FREE cell
PINCH_DISC_M: float = 3.0  # radius of the local pinch overlay around a gate
PINCH_CORRIDOR_HALF_W_M: float = 0.5  # half-width of the forced corridor through the gate
# issue #54: cap on the start-seal relaxation rounds `plan_through` runs when the pinch
# overlay's default disc doesn't fully clear the vehicle's own local inflation halo (a
# halo wider than `pinch_disc_m` can leave `cur` still boxed in even with the start-disc
# exemption). Each round widens the disc by PINCH_RELAX_GROWTH and retries; almost every
# real geometry converges in round 1. Bounded so a pathological halo whose edge keeps
# landing just past the exemption every round can never relax->replan->re-seal forever —
# past the cap, `plan_through` falls back to the pre-#54 behaviour (leg unreachable).
MAX_PINCH_RELAX_ROUNDS: int = 4
PINCH_RELAX_GROWTH: float = 1.75  # disc growth multiplier applied each relax round

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
    gr0, gc0 = grid.world_to_cell(*goal_xy)

    sr, sc = _snap_passable(costmap, sr, sc)
    gr, gc = _snap_passable(costmap, gr0, gc0)
    if sr is None or gr is None:
        return None
    # The requested goal snapped to its OWN cell unchanged (it was already passable) ->
    # the exact continuous goal_xy lies inside the searched-for cell, so the vehicle's
    # true stopping point is goal_xy itself, not that cell's centre. Only substitute
    # when the snap did not move (a moved snap means goal_xy was blocked, and the
    # nearest-passable-cell centre IS the principled least-bad stand-in, same as
    # everywhere else this snap is used) so this never invents a point outside the
    # cell A* actually searched to.
    exact_goal_xy = goal_xy if (gr, gc) == (gr0, gc0) else None

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
            path = _reconstruct(grid, came, (sr, sc), (gr, gc))
            if exact_goal_xy is not None:
                path[-1] = (float(exact_goal_xy[0]), float(exact_goal_xy[1]))
            return path
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
    start_xy: tuple[float, float] | None = None,
) -> Costmap:
    """Overlay: within ``pinch_disc_m`` of the gate, block everything outside a narrow
    corridor (``pinch_corridor_half_w_m`` half-width) around the gate segment, forcing
    A* through the gap. Defaults are the calibration seams for ``nav.pinch_disc_m`` /
    ``nav.pinch_corridor_half_w_m``.

    Also CLEARS inflation-only cells (``Costmap.inflation_only_mask``) inside the
    forced corridor band (issue #51): ``gate`` is GT-derived (the two anchors' closest
    AABB faces — a verified real doorway), but a physical gap narrower than
    ``2 * vehicle_radius_m`` gets sealed end-to-end by the vehicle's own inflation
    margin from BOTH anchors regardless of how the corridor is forced, since the
    overlay above only ever ADDS blocking outside the band — it never had a way to
    open one up. Clearing is scoped tightly (inflation halo only, band only) so a real
    solid obstacle footprint is never driven through, only the safety margin around it,
    and only across the verified gate span.

    ``start_xy`` (issue #54): the corridor band is centred on the GATE, with no regard
    for where the vehicle currently is. When ``start_xy`` sits within ``pinch_disc_m``
    of the gate but off the forced corridor line, the overlay above would newly stamp
    the vehicle's OWN cell impassable (``outside_corridor`` has no notion of "already
    occupied") — the forced-corridor A* is then DOA before it can even leave, sealing a
    route that was reachable pre-overlay. Symmetric with the gate-side treatment: a
    same-radius (``pinch_disc_m``) disc around ``start_xy`` is exempted from the
    overlay's ADDED blocking (never removes a pre-existing real obstacle — the vehicle
    is already sitting in that space, unharmed, so its own neighbourhood can never
    legitimately need MORE blocking than the base costmap already has), and inflation-
    only cells inside that disc are cleared too (same discipline as the gate-side
    clearing above — only the safety margin, never a real footprint)."""
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
    inside_corridor = dist_to_gate <= pinch_corridor_half_w_m
    outside_corridor = ~inside_corridor
    in_start_disc = None
    if start_xy is not None:
        in_start_disc = (cx - start_xy[0]) ** 2 + (cy - start_xy[1]) ** 2 <= pinch_disc_m**2
        outside_corridor = outside_corridor & ~in_start_disc
    pinch.capsule_blocked = pinch.capsule_blocked | (in_disc & outside_corridor)
    clear = in_disc & inside_corridor & pinch.inflation_only_mask()
    if in_start_disc is not None:
        clear = clear | (in_start_disc & pinch.inflation_only_mask())
    if clear.any():
        pinch.base_blocked = pinch.base_blocked & ~clear
    return pinch


def _nudge_past_gate(
    cur: tuple[float, float],
    mid: tuple[float, float],
    cell_m: float,
    pinch_corridor_half_w_m: float,
) -> tuple[float, float]:
    """A point just past the gate midpoint, continuing ``cur``'s approach (issue #54).

    Nudges ``eps`` past ``mid``, CONTINUING the cur->mid direction (rather than the
    gate's own normal — a corridor can be oriented parallel to travel, e.g. a narrow
    gap threaded lengthwise, where a normal-based nudge would push straight into the
    flanking obstacle), so a genuinely threaded path's final approach lands past the
    line it was already heading toward. ``eps`` is kept small (two grid cells, capped
    at the forced corridor's half-width) so the nudged target stays inside the pinch
    corridor band. Falls back to the raw midpoint if ``cur`` is (numerically) already
    at the gate.
    """
    dx, dy = mid[0] - cur[0], mid[1] - cur[1]
    dlen = math.hypot(dx, dy)
    if dlen < 1e-9:
        return mid
    eps = min(2.0 * cell_m, pinch_corridor_half_w_m)
    return (mid[0] + dx / dlen * eps, mid[1] + dy / dlen * eps)


def _raw_obstacle_blocked_xy(costmap: Costmap, pt) -> bool:
    """True if ``pt`` (world XY) lands on a genuine solid obstacle cell.

    issue #63: tests ``raw_blocked`` (the UNINFLATED obstacle footprint), never
    ``base_blocked``/``blocked()`` — a cell blocked only by the vehicle's own
    inflation margin is not a real obstruction the gate needs to route around
    (same discipline `_pinch_costmap`'s inflation-only clearing already applies).
    Off-grid points read as clear (never force a slide off the mapped area).
    """
    r, c = costmap.grid.world_to_cell(float(pt[0]), float(pt[1]))
    if not costmap.grid.in_bounds(r, c):
        return False
    rh, rw = costmap.raw_blocked.shape
    if not (0 <= r < rh and 0 <= c < rw):
        return False
    return bool(costmap.raw_blocked[r, c])


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
            raw_mid = ((g0[0] + g1[0]) / 2.0, (g0[1] + g1[1]) / 2.0)
            # issue #63: a genuine solid obstacle sitting at the raw gate midpoint
            # (e.g. hotel_room_2's duplicate GT "bed frame" instance for the same
            # physical bed as an anchor) is not a usable via-point — slide the
            # mandatory via-point along the gate's own verified axis to the nearest
            # clear point via the SAME `usable_gate_point` helper the scoring
            # geometry's `corridor_gate` uses, so the two can never disagree about
            # where a blocked gate's usable crossing is (the #51 mismatch class).
            # A no-op (returns `raw_mid` unchanged) whenever the raw midpoint is
            # already clear, which is every gate today's tests exercise.
            nudged = usable_gate_point(
                np.asarray(g0, dtype=float),
                np.asarray(g1, dtype=float),
                np.asarray(raw_mid, dtype=float),
                lambda pt: _raw_obstacle_blocked_xy(costmap, pt),
            )
            mid = (float(nudged[0]), float(nudged[1]))
            seg = astar(costmap, cur, mid, unknown_cost_mult=unknown_cost_mult)
            # issue #51: the pinch fallback used to engage ONLY when the direct A*
            # succeeded but missed the gate (`seg is not None`); when the gate midpoint
            # was itself unreachable in the un-pinched costmap (`seg is None`) — e.g. its
            # cell sits inside the vehicle's own inflation halo from the two anchor
            # objects — the leg gave up immediately without ever trying the pinch
            # overlay that exists precisely to force a path through a tight verified
            # gate. Try the pinch fallback in BOTH cases; only report the leg
            # unreachable once the forced-corridor attempt has also failed.
            direct_seg_was_none = seg is None
            if seg is None or not path_crosses_gate(seg, gate):  # type: ignore[arg-type]
                # issue #54: the pinch overlay's forced corridor band is centred on the
                # GATE only, with no regard for the vehicle's OWN position — when `cur`
                # sits within `pinch_disc_m` of the gate but off the forced corridor
                # line, the overlay would newly stamp the vehicle's own cell impassable
                # (`outside_corridor` has no notion of "already occupied"), so the
                # forced-corridor A* is DOA before it can leave. `start_xy` exempts a
                # same-radius disc around `cur` from the overlay's ADDED blocking
                # (never removes a pre-existing real obstacle — the vehicle is already
                # sitting there, unharmed) and clears inflation-only cells inside it
                # (same discipline as the existing gate-side clearing, #51). Safe on
                # every corridor attempt: it only ever widens what stays passable near
                # the vehicle, never narrows it.
                target = mid
                if direct_seg_was_none:
                    # The direct (un-pinched) attempt found NO route to the gate at all
                    # — this leg has never threaded this gate before, so there is no
                    # already-driven progress to protect. Target a point nudged just
                    # past the midpoint, continuing `cur`'s approach direction, rather
                    # than the exact midpoint: A*-ing straight to `mid` can land its
                    # snapped goal cell fractionally on the SAME side of the gate line
                    # `cur` approached from (a grid-quantization near-miss), so the
                    # mandatory `path_crosses_gate` check below fails even though the
                    # corridor was genuinely threaded.
                    target = _nudge_past_gate(cur, mid, costmap.cell_m, pinch_corridor_half_w_m)
                else:
                    # issue #64: the direct attempt DID reach the gate (just without
                    # crossing it) — the case #54's guard normally leaves un-nudged,
                    # because `cur` may legitimately already be on the far side of a
                    # gate this route threaded earlier before a route rebuild (nudging
                    # there forces a spurious detour BACK through an already-threaded
                    # gate: the real #54 regression). But this same shape also covers a
                    # genuine grid-quantization near-miss: the goal-cell snap
                    # (`_snap_passable`) can land the direct segment's last vertex up to
                    # half a grid cell short of the exact gate line, on the SAME side
                    # `cur` approached from, so `path_crosses_gate` fails even though the
                    # corridor was, in every practical sense, threaded. Distinguish the
                    # two with a tight, two-part engagement condition rather than
                    # broadening nudging to every case-B miss (which is exactly what
                    # broke the #54 guard previously):
                    #   1. the miss itself must be smaller than the quantization
                    #      mechanism can produce (one full grid cell — double the
                    #      half-cell bound `_snap_passable` can introduce, so real
                    #      near-misses always qualify with margin) — an
                    #      already-threaded-gate miss from a genuinely different route
                    #      shape stays much farther out than that;
                    #   2. `cur` itself must NOT already be sitting within that same
                    #      one-cell band of the gate at the START of this leg — an
                    #      already-threaded gate typically resumes with `cur` planted
                    #      right at (or just past) the line it crossed earlier, whereas a
                    #      fresh quantization near-miss approaches from farther away and
                    #      only the direct segment's SNAPPED END lands close.
                    # Both conditions must hold; either failing leaves `target = mid`
                    # (the pre-#64, #54-safe behaviour) unchanged.
                    miss_dist = float(
                        _point_segment_dist(seg[-1][0], seg[-1][1], g0[0], g0[1], g1[0], g1[1])
                    )
                    cur_dist = float(_point_segment_dist(cur[0], cur[1], g0[0], g0[1], g1[0], g1[1]))
                    if miss_dist < costmap.cell_m and cur_dist >= costmap.cell_m:
                        target = _nudge_past_gate(cur, mid, costmap.cell_m, pinch_corridor_half_w_m)
                # issue #54: bounded start-seal relaxation. Round 0 uses the caller's
                # own pinch_disc_m (today's behaviour, unchanged for the common case);
                # if the start-disc exemption still doesn't free a route (the local
                # inflation halo around `cur` is wider than the disc), widen the disc
                # and retry. Capped at MAX_PINCH_RELAX_ROUNDS rounds — never an
                # unbounded relax->replan->re-seal cycle. If no round threads the gate,
                # fall back to the pre-#54 behaviour: leg unreachable.
                pinch_seg = None
                relax_disc_m = pinch_disc_m
                for _relax_round in range(MAX_PINCH_RELAX_ROUNDS):
                    pinch = _pinch_costmap(
                        costmap,
                        gate,  # type: ignore[arg-type]
                        pinch_disc_m=relax_disc_m,
                        pinch_corridor_half_w_m=pinch_corridor_half_w_m,
                        start_xy=cur,
                    )
                    attempt = astar(pinch, cur, target, unknown_cost_mult=unknown_cost_mult)
                    if attempt is not None and path_crosses_gate(attempt, gate):  # type: ignore[arg-type]
                        pinch_seg = attempt
                        break
                    relax_disc_m *= PINCH_RELAX_GROWTH
                seg = pinch_seg
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
