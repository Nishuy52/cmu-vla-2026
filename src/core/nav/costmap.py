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
"""
from __future__ import annotations

from collections import deque

import numpy as np

from core.nav.occupancy import FREE, OBSTACLE, UNKNOWN, OccupancyGrid

# --------------------------------------------------------------------------- tunables
VEHICLE_RADIUS_M: float = 0.4  # inflation radius (half footprint + margin)

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
    ):
        self.grid = grid
        self.vehicle_radius_m = vehicle_radius_m
        self.allow_overhead = allow_overhead
        self.cell_m = grid.cell_m
        h, w = grid.shape
        # base_blocked: OBSTACLE + inflation. capsule_blocked: hard avoid stamps.
        # Overhead-flagged cells (furniture the terrain slab filtered out — bar
        # tables/shelves the base stack reads as FREE floor) are treated as
        # obstacles and inflated the same way, UNLESS allow_overhead is set (the
        # explicit, sweepable opt-out that restores terrain-only behaviour).
        blocked_seed = grid.state == OBSTACLE
        if not allow_overhead and grid.overhead is not None:
            blocked_seed = blocked_seed | grid.overhead
        self.base_blocked = self._inflate(blocked_seed, vehicle_radius_m)
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
        return bool(self.base_blocked[row, col] or self.capsule_blocked[row, col])

    def passable(self, row: int, col: int) -> bool:
        return not self.blocked(row, col)

    def is_unknown(self, row: int, col: int) -> bool:
        return self.grid.in_bounds(row, col) and bool(self.unknown[row, col])

    def clone(self) -> "Costmap":
        """Shallow-copy for a local pinch-corridor overlay (planner use)."""
        cm = Costmap.__new__(Costmap)
        cm.grid = self.grid
        cm.vehicle_radius_m = self.vehicle_radius_m
        cm.allow_overhead = self.allow_overhead
        cm.cell_m = self.cell_m
        cm.base_blocked = self.base_blocked  # shared read-only
        cm.capsule_blocked = self.capsule_blocked.copy()
        cm.unknown = self.unknown
        return cm

    # ------------------------------------------------------------- recovery
    def nearest_reachable_point(
        self, goal_xy: tuple[float, float], start_xy: tuple[float, float]
    ) -> tuple[float, float]:
        """Nearest cell reachable from `start` (BFS over passable cells) to `goal`.

        Deliberate least-bad choice when the goal is unreachable with capsules in
        place (adjudication row 4): capsules are NEVER relaxed here. Returns the
        world-frame centre of the reachable cell closest (Euclidean) to the goal.
        """
        h, w = self.grid.shape
        sr, sc = self.grid.world_to_cell(*start_xy)
        # Snap start into a passable cell if needed.
        if not (0 <= sr < h and 0 <= sc < w) or self.blocked(sr, sc):
            sr, sc = self._nearest_passable_cell(sr, sc)
            if sr is None:
                return start_xy

        seen = np.zeros((h, w), dtype=bool)
        seen[sr, sc] = True
        dq = deque([(sr, sc)])
        reachable: list[tuple[int, int]] = [(sr, sc)]
        while dq:
            r, c = dq.popleft()
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and self.passable(nr, nc):
                    seen[nr, nc] = True
                    dq.append((nr, nc))
                    reachable.append((nr, nc))

        gx, gy = goal_xy
        best = min(
            reachable,
            key=lambda rc: (self.grid.cell_to_world(*rc)[0] - gx) ** 2
            + (self.grid.cell_to_world(*rc)[1] - gy) ** 2,
        )
        return self.grid.cell_to_world(*best)

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
