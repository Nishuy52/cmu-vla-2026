"""Per-question Costmap: an OccupancyGrid view + inflation + HARD avoid stamps.

Architecture §1 rows 3/4 (non-negotiable): avoid-capsules are stamped as
OBSTACLE-forever for the lifetime of the question — a capsule NEVER shrinks or
relaxes. If a goal is unreachable with the capsules in place we do NOT relax them
(that would reintroduce the very violation we prevent, adjudication row 4);
instead `nearest_reachable_point(goal)` returns the least-bad legal cell and the
caller drives there and answers from there.

The costmap holds its own derived obstacle mask so stamping/inflation never
mutates the shared OccupancyGrid (which other questions and the live map use).
UNKNOWN remains traversable (at a penalty applied by the planner) so exploration
can route through unexplored space; only FREE/UNKNOWN cells are candidate.

Overhead-clearance handling (redteam H13 / SYS-F12). Cells the overhead layer
flagged (furniture the terrain slab filtered out — bar tables/shelves the base
stack reads as FREE floor) are, by DEFAULT, treated as SOFT high-cost obstacles:
they stay traversable but the planner pays a steep multiplier to cross them, so a
route only goes under furniture when there is no clear alternative. This replaces
the previous hard-block + inflation, which sealed doorways whenever an overhead
cell was a false positive next to the only door (route unreachable ⇒ points
quietly lost via `_recover_path`). A false overhead flag now makes a route
EXPENSIVE, not IMPOSSIBLE, while a genuinely-blocked under-furniture route (no
alternative) is still strongly avoided.

The soft treatment is consumed by A* through the SAME per-cell cost seam A*
already reads for UNKNOWN cells (`is_unknown()` ⇒ `planner.UNKNOWN_COST_MULT`):
a soft-overhead cell reports as cost-bearing so A* pays the penalty to cross it
but is never blocked. This needs no planner change and preserves the Costmap API.

EXCEPTION — corridor threading. Inside the pinch-corridor overlay the planner
builds to force A* through a physical gate, a soft under-furniture cell must NOT
count as a legal gate: a pinch corridor threaded UNDER a table is not a real
doorway. `clone()` (the only entry the pinch overlay uses) therefore HARDENS the
overhead layer — the clone hard-blocks overhead cells so the forced-corridor plan
cannot thread under furniture. `overhead_hard(True)` at construction requests the
same hardening explicitly (legacy hard-block behaviour), and `allow_overhead=True`
still ignores overhead entirely (terrain-only, sweepable opt-out).
"""
from __future__ import annotations

from collections import deque

import numpy as np

from core.nav.occupancy import FREE, OBSTACLE, UNKNOWN, OccupancyGrid

# --------------------------------------------------------------------------- tunables
VEHICLE_RADIUS_M: float = 0.4  # inflation radius (half footprint + margin)
#: Issue #207 -- the base stack's own waypoint-snap clearance rule, confirmed from
#: upstream source: ``waypointConverter.cpp`` (the `waypointAdj` branch, ~line 210-233)
#: only accepts a traversable-area candidate point as a valid snap target when its
#: planar (x, y) distance to EVERY point in the terrain `obstacleArea` cloud is
#: `>= obstacleDisThre` (squared-distance compare: `disX3*disX3 + disY3*disY3 <
#: obstacleDisThre * obstacleDisThre` disqualifies the candidate). The launch file
#: (`waypoint_converter.launch`) sets `obstacleDisThre = 0.75`. Our own planning
#: inflation (`VEHICLE_RADIUS_M = 0.4`) is 0.35 m short of this: we route and publish
#: crumbs the base then refuses to snap onto, and it freezes (#205/#207). This constant
#: is a PUBLISH-time clearance floor, deliberately kept separate from
#: `VEHICLE_RADIUS_M` (the A*-planning inflation): raising the planning inflation to
#: 0.75 m globally would make any corridor/doorway narrower than 1.5 m clear
#: unplannable everywhere along a route, not just at the final published crumb (see
#: `Costmap.obstacle_clearance_m`/`nearest_clear_point` docstrings and
#: ``breadcrumbs.ensure_base_clearance``, which apply this floor only to the point
#: actually handed to the base).
BASE_OBSTACLE_CLEARANCE_M: float = 0.75
#: Search window (issue #207) ``nearest_clear_point`` scans around a rejected crumb
#: for a nearby alternative that clears ``BASE_OBSTACLE_CLEARANCE_M``. Kept small
#: (well under the base's own 5.0 m ``searchDisThre``) so a nudge stays LOCAL to the
#: rejected point (never substitutes a distant, unrelated part of the map) — the
#: caller (``breadcrumbs.ensure_base_clearance``) also re-checks line-of-sight from
#: the vehicle pose before adopting a nudged point.
LOOKAHEAD_CLEARANCE_SEARCH_M: float = 1.5
#: Cost multiplier A* pays to cross a SOFT-overhead cell, relative to a FREE cell.
#: The applied per-cell penalty on the A* path is `planner.UNKNOWN_COST_MULT` (the
#: only per-cell cost seam A* reads without a planner change); this constant records
#: the INTENDED soft-overhead weight and is the sweep handle for a Phase-2 planner
#: seam. It is set >= UNKNOWN_COST_MULT so soft overhead is at least as discouraged
#: as unexplored space — steep enough that a clear alternative always wins, low
#: enough that a false positive never makes a route unreachable.
OVERHEAD_SOFT_COST_MULT: float = 4.0

# Blocked-cell codes in the costmap's own mask.
_PASSABLE = 0  # FREE or UNKNOWN, traversable (UNKNOWN at planner penalty)
_BLOCKED = 1  # OBSTACLE, inflated obstacle, or capsule stamp — hard, never passable


class Costmap:
    """A question-scoped planning surface over a snapshot of an OccupancyGrid."""

    def __init__(
        self,
        grid: OccupancyGrid,
        vehicle_radius_m: float = VEHICLE_RADIUS_M,
        allow_overhead: bool = False,
        overhead_hard: bool = False,
        overhead_soft_cost_mult: float = OVERHEAD_SOFT_COST_MULT,
    ):
        self.grid = grid
        self.vehicle_radius_m = vehicle_radius_m
        self.allow_overhead = allow_overhead
        #: overhead_hard=True restores the legacy hard-block + inflation for overhead
        #: cells (used by the pinch-corridor clone so a gate can't thread UNDER
        #: furniture); default False makes overhead a SOFT high-cost layer (see
        #: module docstring). allow_overhead=True wins over both (terrain-only).
        self.overhead_hard = overhead_hard
        self.overhead_soft_cost_mult = overhead_soft_cost_mult
        self.cell_m = grid.cell_m
        h, w = grid.shape
        # base_blocked: OBSTACLE + inflation. capsule_blocked: hard avoid stamps.
        # Overhead-flagged cells (furniture the terrain slab filtered out — bar
        # tables/shelves the base stack reads as FREE floor):
        #   * allow_overhead      -> ignored entirely (terrain-only, sweepable opt-out);
        #   * overhead_hard       -> hard obstacle + inflation (legacy / corridor threading);
        #   * default (soft)      -> a separate SOFT layer, inflated to keep the footprint
        #     clear but traversable at a steep planner cost (never blocks a route).
        has_overhead = grid.overhead is not None and not allow_overhead
        blocked_seed = grid.state == OBSTACLE
        if has_overhead and overhead_hard:
            blocked_seed = blocked_seed | grid.overhead
        #: UNINFLATED obstacle footprint (the raw seed before ``_inflate``). Lets a
        #: consumer (the pinch-corridor overlay) tell a real solid obstacle cell
        #: apart from an inflation-halo cell that is only blocked because it sits
        #: within ``vehicle_radius_m`` of one (see ``inflation_only_mask``).
        self.raw_blocked = blocked_seed
        self.base_blocked = self._inflate(blocked_seed, vehicle_radius_m)
        # Soft-overhead mask (inflated like an obstacle so the vehicle FOOTPRINT is
        # discouraged near furniture, not just the centre cell) — empty unless the
        # overhead layer is soft-active.
        if has_overhead and not overhead_hard:
            self.overhead_soft = self._inflate(grid.overhead, vehicle_radius_m) & ~self.base_blocked
        else:
            self.overhead_soft = np.zeros((h, w), dtype=bool)
        self.capsule_blocked = np.zeros((h, w), dtype=bool)
        # UNKNOWN mask travels for the planner's UNKNOWN penalty.
        self.unknown = grid.state == UNKNOWN

    # ------------------------------------------------------------- inflation
    def _inflate(self, obstacle: np.ndarray, radius_m: float) -> np.ndarray:
        """Dilate an obstacle mask by a disc of `radius_m` (metres)."""
        r_cells = int(np.ceil(radius_m / self.cell_m))
        if r_cells <= 0 or not obstacle.any():
            return obstacle.copy()
        h, w = obstacle.shape
        out = obstacle.copy()
        # Disc offsets.
        offs = [
            (dr, dc)
            for dr in range(-r_cells, r_cells + 1)
            for dc in range(-r_cells, r_cells + 1)
            if dr * dr + dc * dc <= r_cells * r_cells
        ]
        rs, cs = np.nonzero(obstacle)
        for dr, dc in offs:
            nr = rs + dr
            nc = cs + dc
            ok = (nr >= 0) & (nr < h) & (nc >= 0) & (nc < w)
            out[nr[ok], nc[ok]] = True
        return out

    # ------------------------------------------------------------- capsule stamps
    def stamp_capsule(
        self, seg: tuple[tuple[float, float], tuple[float, float]], radius_m: float
    ) -> None:
        """Mark every cell within `radius_m` of segment `seg` as hard OBSTACLE-forever.

        seg = ((x0, y0), (x1, y1)) in metres. Cumulative: capsules only ever grow.
        """
        (x0, y0), (x1, y1) = seg
        h, w = self.grid.shape
        # Cell centres.
        rr, cc = np.mgrid[0:h, 0:w]
        cx = self.grid.origin_x + (cc + 0.5) * self.cell_m
        cy = self.grid.origin_y + (rr + 0.5) * self.cell_m
        d = _point_segment_dist(cx, cy, x0, y0, x1, y1)
        # Inflate the capsule by the vehicle radius too, so the *footprint* stays out.
        self.capsule_blocked |= d <= (radius_m + self.vehicle_radius_m)

    # ------------------------------------------------------------- queries
    def blocked(self, row: int, col: int) -> bool:
        """True if the cell is hard-blocked (obstacle, inflation, or capsule)."""
        if not self.grid.in_bounds(row, col):
            return True
        # ``base_blocked`` is snapshotted at construction; the underlying grid can GROW
        # afterwards (terrain integration), so grid.in_bounds may admit a cell beyond the
        # snapshot. Treat any cell outside the snapshot extent as blocked (unmapped at
        # build time) rather than IndexError. Cheap guard, covers every blocked() caller
        # incl. the reachable-mask flood.
        bh, bw = self.base_blocked.shape
        if not (0 <= row < bh and 0 <= col < bw):
            return True
        return bool(self.base_blocked[row, col] or self.capsule_blocked[row, col])

    def passable(self, row: int, col: int) -> bool:
        return not self.blocked(row, col)

    def is_soft_overhead(self, row: int, col: int) -> bool:
        """True if the cell is a SOFT-overhead cell (traversable at a steep cost).

        Empty when overhead is hardened (`overhead_hard`/clone), ignored
        (`allow_overhead`), or the grid carries no overhead layer.
        """
        return self.grid.in_bounds(row, col) and bool(self.overhead_soft[row, col])

    def is_unknown(self, row: int, col: int) -> bool:
        """Cost-bearing-traversable query, consumed by A* to apply its penalty.

        Returns True for genuinely-UNKNOWN cells AND for SOFT-overhead cells: both
        are passable but should cost more than FREE space. A* multiplies its step
        cost by `planner.UNKNOWN_COST_MULT` for any cell that answers True here — the
        single per-cell cost seam A* reads — so folding soft overhead in here makes
        the planner route AROUND under-furniture cells when a clear alternative
        exists, without a planner change and without ever blocking a route
        (`OVERHEAD_SOFT_COST_MULT` records the intended weight; see module docstring).
        """
        if not self.grid.in_bounds(row, col):
            return False
        return bool(self.unknown[row, col] or self.overhead_soft[row, col])

    def clone(self) -> "Costmap":
        """Shallow-copy for a local pinch-corridor overlay (planner use).

        HARDENS the soft-overhead layer: the pinch overlay forces A* through a
        physical gate, and a gate threaded UNDER furniture is not a legal doorway
        (redteam H13). So in the clone, soft-overhead cells fold into the hard
        `base_blocked` mask — the forced corridor cannot route under a table. The
        base costmap keeps its soft treatment; only corridor-threading hardens.
        """
        cm = Costmap.__new__(Costmap)
        cm.grid = self.grid
        cm.vehicle_radius_m = self.vehicle_radius_m
        cm.allow_overhead = self.allow_overhead
        cm.overhead_hard = True  # corridor threading treats overhead as hard
        cm.overhead_soft_cost_mult = self.overhead_soft_cost_mult
        cm.cell_m = self.cell_m
        # Fold the soft-overhead cells into the hard mask for the corridor plan.
        if self.overhead_soft.any():
            cm.base_blocked = self.base_blocked | self.overhead_soft
            # raw_blocked stays consistent with base_blocked: fold the RAW (uninflated)
            # overhead seed in too, so inflation_only_mask still only ever reports true
            # inflation halo, never a real (overhead) obstacle cell.
            cm.raw_blocked = self.raw_blocked | (
                self.grid.overhead if self.grid.overhead is not None
                else np.zeros(self.raw_blocked.shape, dtype=bool)
            )
        else:
            cm.base_blocked = self.base_blocked  # shared read-only
            cm.raw_blocked = self.raw_blocked
        cm.overhead_soft = np.zeros(self.overhead_soft.shape, dtype=bool)
        cm.capsule_blocked = self.capsule_blocked.copy()
        cm.unknown = self.unknown
        return cm

    def inflation_only_mask(self) -> np.ndarray:
        """Cells blocked ONLY by vehicle-radius inflation, never by a real obstacle
        footprint (``base_blocked & ~raw_blocked``). Lets a consumer relax the
        inflation margin (e.g. the pinch-corridor overlay squeezing a verified GT
        gate) without ever letting the vehicle drive through solid geometry."""
        return self.base_blocked & ~self.raw_blocked

    # ------------------------------------------------------------- recovery
    def reachable_mask(self, start_xy: tuple[float, float]) -> np.ndarray | None:
        """Boolean (h, w) mask of cells reachable from ``start`` over passable cells.

        Memoised per start CELL for this costmap instance: a grounding pass projects
        many leg goals against the same costmap+pose, so re-flooding the whole grid per
        call is an O(cells × legs × ticks) blow-up (the instruction head calls this per
        leg per tick). The costmap is rebuilt whenever the map changes, which discards
        the cache, so it can never go stale. Returns None if the map is fully impassable.
        """
        sr, sc = self.grid.world_to_cell(*start_xy)
        h, w = self.grid.shape
        if not (0 <= sr < h and 0 <= sc < w) or self.blocked(sr, sc):
            snapped = self._nearest_passable_cell(sr, sc)
            if snapped[0] is None:
                return None
            sr, sc = snapped
        key = (sr, sc)
        cached = getattr(self, "_reach_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        seen = np.zeros((h, w), dtype=bool)
        seen[sr, sc] = True
        dq = deque([(sr, sc)])
        while dq:
            r, c = dq.popleft()
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and self.passable(nr, nc):
                    seen[nr, nc] = True
                    dq.append((nr, nc))
        self._reach_cache = (key, seen)
        return seen

    def nearest_reachable_point(
        self, goal_xy: tuple[float, float], start_xy: tuple[float, float]
    ) -> tuple[float, float]:
        """Nearest cell reachable from `start` (BFS over passable cells) to `goal`.

        Deliberate least-bad choice when the goal is unreachable with capsules in
        place (adjudication row 4): capsules are NEVER relaxed here. Returns the
        world-frame centre of the reachable cell closest (Euclidean) to the goal.
        """
        seen = self.reachable_mask(start_xy)
        if seen is None:
            return start_xy
        reachable = np.argwhere(seen)
        gr, gc = self.grid.world_to_cell(*goal_xy)
        d2 = (reachable[:, 0] - gr) ** 2 + (reachable[:, 1] - gc) ** 2
        br, bc = reachable[int(np.argmin(d2))]
        return self.grid.cell_to_world(int(br), int(bc))

    # ------------------------------------------------------------- issue #207 clearance
    def obstacle_clearance_m(
        self, x: float, y: float, search_radius_m: float = 2.0
    ) -> float:
        """Planar distance (m) from ``(x, y)`` to the nearest RAW (uninflated)
        obstacle cell -- our best proxy for the base stack's own ``obstacleArea``
        clearance check (see ``BASE_OBSTACLE_CLEARANCE_M``). Bounded local-window
        search (not a whole-grid transform): cheap enough to call per candidate
        crumb. Returns ``search_radius_m`` (i.e. "at least this clear") when no
        raw-obstacle cell falls inside the search window.
        """
        r0, c0 = self.grid.world_to_cell(x, y)
        r_cells = int(np.ceil(search_radius_m / self.cell_m))
        h, w = self.raw_blocked.shape
        r_lo, r_hi = max(0, r0 - r_cells), min(h, r0 + r_cells + 1)
        c_lo, c_hi = max(0, c0 - r_cells), min(w, c0 + r_cells + 1)
        if r_lo >= r_hi or c_lo >= c_hi:
            return search_radius_m
        window = self.raw_blocked[r_lo:r_hi, c_lo:c_hi]
        if not window.any():
            return search_radius_m
        rr, cc = np.nonzero(window)
        dr = (rr + r_lo) - r0
        dc = (cc + c_lo) - c0
        d2 = dr * dr + dc * dc
        return float(np.sqrt(d2.min()) * self.cell_m)

    def nearest_clear_point(
        self,
        x: float,
        y: float,
        min_clearance_m: float = BASE_OBSTACLE_CLEARANCE_M,
        search_radius_m: float = LOOKAHEAD_CLEARANCE_SEARCH_M,
    ) -> tuple[float, float] | None:
        """Nearest PASSABLE cell to ``(x, y)`` within ``search_radius_m`` whose
        ``obstacle_clearance_m`` is ``>= min_clearance_m`` (mirrors the base's own
        "search nearby traversable points, keep the closest one far enough from
        every obstacle" rule -- see ``BASE_OBSTACLE_CLEARANCE_M``). Returns ``None``
        if no such cell exists in the window (a genuinely tight passage narrower
        than ``2 * min_clearance_m`` -- e.g. sub-1.5 m doorway -- has no interior
        point that clears 0.75 m from both sides; the caller then keeps the
        original point rather than diverge onto an unrelated part of the map).
        """
        r0, c0 = self.grid.world_to_cell(x, y)
        h, w = self.base_blocked.shape
        cand_r_cells = int(np.ceil(search_radius_m / self.cell_m))
        cr_lo, cr_hi = max(0, r0 - cand_r_cells), min(h, r0 + cand_r_cells + 1)
        cc_lo, cc_hi = max(0, c0 - cand_r_cells), min(w, c0 + cand_r_cells + 1)
        if cr_lo >= cr_hi or cc_lo >= cc_hi:
            return None
        cand_blocked = (
            self.base_blocked[cr_lo:cr_hi, cc_lo:cc_hi]
            | self.capsule_blocked[cr_lo:cr_hi, cc_lo:cc_hi]
        )
        cand_r, cand_c = np.nonzero(~cand_blocked)
        if cand_r.size == 0:
            return None
        cand_r = cand_r + cr_lo
        cand_c = cand_c + cc_lo

        # Obstacle search window: wide enough that every candidate's full
        # min_clearance_m neighbourhood is covered.
        obs_r_cells = cand_r_cells + int(np.ceil(min_clearance_m / self.cell_m)) + 1
        or_lo, or_hi = max(0, r0 - obs_r_cells), min(h, r0 + obs_r_cells + 1)
        oc_lo, oc_hi = max(0, c0 - obs_r_cells), min(w, c0 + obs_r_cells + 1)
        obs_r, obs_c = np.nonzero(self.raw_blocked[or_lo:or_hi, oc_lo:oc_hi])
        min_clear_cells2 = (min_clearance_m / self.cell_m) ** 2
        if obs_r.size == 0:
            ok_r, ok_c = cand_r, cand_c
        else:
            obs_r = obs_r + or_lo
            obs_c = obs_c + oc_lo
            dr = cand_r[:, None] - obs_r[None, :]
            dc = cand_c[:, None] - obs_c[None, :]
            min_obs_d2 = (dr * dr + dc * dc).min(axis=1)
            ok = min_obs_d2 >= min_clear_cells2
            if not ok.any():
                return None
            ok_r, ok_c = cand_r[ok], cand_c[ok]
        d2_to_query = (ok_r - r0) ** 2 + (ok_c - c0) ** 2
        i = int(np.argmin(d2_to_query))
        return self.grid.cell_to_world(int(ok_r[i]), int(ok_c[i]))

    def _nearest_passable_cell(self, r0: int, c0: int):
        h, w = self.grid.shape
        pass_cells = np.argwhere(~(self.base_blocked | self.capsule_blocked))
        if pass_cells.size == 0:
            return None, None
        d2 = (pass_cells[:, 0] - r0) ** 2 + (pass_cells[:, 1] - c0) ** 2
        r, c = pass_cells[int(np.argmin(d2))]
        return int(r), int(c)


def _point_segment_dist(px, py, x0, y0, x1, y1):
    """Vectorised distance from points (px, py) to segment (x0,y0)-(x1,y1)."""
    dx, dy = x1 - x0, y1 - y0
    seg_len2 = dx * dx + dy * dy
    if seg_len2 == 0.0:
        return np.hypot(px - x0, py - y0)
    t = ((px - x0) * dx + (py - y0) * dy) / seg_len2
    t = np.clip(t, 0.0, 1.0)
    projx = x0 + t * dx
    projy = y0 + t * dy
    return np.hypot(px - projx, py - projy)
