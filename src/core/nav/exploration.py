"""Exploration policy loop glue: (grid, frontiers, affinity, budget) -> next move.

Ties the nav primitives into a policy the FSM steps:
  - First ~SWEEP_S seconds: an orientation-sweep primitive. The vehicle CANNOT
    rotate in place via /way_point_with_heading (heading is ignored this year,
    docs/upstream_notes.md gotcha 4), so a literal in-place spin is impossible.
    Instead we emit a small 4-point diamond of side ~SWEEP_SIDE_M around the start
    pose: driving the short loop sweeps the 360 panorama + seeds the occupancy grid
    from every yaw, achieving the same panoramic seeding a spin would. (Design choice
    per the task spec / architecture §5 step 1.)
  - After the sweep: pick the best-scored frontier centroid above MIN_FRONTIER_SCORE
    and hand back its position as the next goal. If no frontier clears the bar (or
    the map is coverage-saturated), report EXPLORATION_COMPLETE.

This module decides WHERE to go next; the planner/breadcrumb layer decides HOW to
get there. Deterministic: all timing comes from the injected `t`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

from core.interfaces import QType, WaypointCmd
from core.nav.frontiers import Frontier, W_SIZE, detect_frontiers
from core.nav.occupancy import FREE, OccupancyGrid

# --------------------------------------------------------------------------- tunables
SWEEP_S: float = 60.0  # duration of the opening orientation sweep
SWEEP_SIDE_M: float = 1.0  # side length of the sweep diamond around the start pose
MIN_FRONTIER_SCORE: float = 0.0  # frontiers below this score aren't worth pursuing
COVERAGE_SATURATED_FREE_FRAC: float = 0.0  # reserved; frontier absence is the real gate

# #150/#103: for NUMERICAL, UNSEEN AREA is the objective -- an under-count is an instance
# that was never in view, not a duplicate or a mislabel (#150's live evidence: the merged
# NMS stack cut over-counting ~82%, leaving coverage as the entire remaining error). The
# default frontier score (`nav.frontiers.detect_frontiers`) is
# `w_size * size - w_dist * path_distance + w_affinity * affinity`: a large-but-far
# frontier bordering a genuinely big unexplored pocket can lose to a small-but-near or
# high-affinity frontier, so the policy keeps re-covering already-seen ground instead of
# committing to the pocket that would reveal more instances (the #103 failure mode: the
# resolved/target region sits outside what was actually explored). Boosting the size term
# for NUMERICAL only shifts that tradeoff toward "go where there is more unseen boundary
# to reveal" without touching OBJECT_REFERENCE/INSTRUCTION_FOLLOWING, whose objective is
# grounding a specific referent rather than exhaustive coverage, or the neutral W_SIZE
# default other callers (e.g. explore_debug diagnostics) still get with qtype=None.
#
# 2.0x is a first, conservative cut: enough to let a frontier roughly twice the boundary
# size of a rival outweigh the ~1-cell-per-2-score-unit W_DIST penalty (W_SIZE=1.0,
# W_DIST=0.5 in nav.frontiers) over a few extra metres of travel, without letting size
# swamp the W_AFFINITY=4.0 semantic term entirely. Flagged as thin evidence per the
# generalization protocol (docs/calibration.md) -- not fit to a held-out sample, just a
# principled starting point pending a live A/B; revisit if #150-style coverage evidence
# comes back with a live before/after on this specific weight.
NUMERICAL_W_SIZE_MULT: float = 2.0


class ExplorationStatus(str, Enum):
    SWEEPING = "sweeping"
    FRONTIER = "frontier"
    COMPLETE = "complete"


@dataclass
class ExplorationDecision:
    status: ExplorationStatus
    waypoint: WaypointCmd | None = None
    frontier: Frontier | None = None


def sweep_waypoints(start_xy: tuple[float, float], side_m: float = SWEEP_SIDE_M):
    """The 4-point diamond around the start pose used as the orientation sweep.

    Diamond (rotated square) vertices at +/-side/2 along x and y — driving the loop
    exposes the panorama from four headings without needing in-place rotation.
    """
    x, y = start_xy
    h = side_m / 2.0
    return [
        WaypointCmd(x + h, y),
        WaypointCmd(x, y + h),
        WaypointCmd(x - h, y),
        WaypointCmd(x, y - h),
    ]


@dataclass
class ExplorationPolicy:
    """Stateful exploration policy the FSM steps once per decision cycle."""

    start_xy: tuple[float, float]
    affinity: Callable[[tuple[float, float]], float] | None = None
    sweep_s: float = SWEEP_S
    sweep_side_m: float = SWEEP_SIDE_M
    min_frontier_score: float = MIN_FRONTIER_SCORE
    t0: float | None = None  # question-clock origin; if None, anchored on first step
    #: #150 -- when QType.NUMERICAL, detect_frontiers is called with a boosted w_size
    #: (NUMERICAL_W_SIZE_MULT) so unseen-area coverage outweighs distance/affinity more
    #: than it does for the other qtypes. None (default) == unchanged geometric scoring.
    qtype: QType | None = None
    _sweep: list[WaypointCmd] = field(default_factory=list)
    _sweep_idx: int = 0

    def __post_init__(self) -> None:
        self._sweep = sweep_waypoints(self.start_xy, self.sweep_side_m)

    def _elapsed(self, t: float) -> float:
        # Anchor the clock the first time we're stepped so `sweep_s` measures wall
        # time since exploration began, regardless of the absolute question clock.
        if self.t0 is None:
            self.t0 = t
        return t - self.t0

    def step(
        self,
        grid: OccupancyGrid,
        vehicle_xy: tuple[float, float],
        t: float,
        *,
        budget_state: dict | None = None,
    ) -> ExplorationDecision:
        """Return the next exploration move given the current map and clock.

        budget_state: optional dict (e.g. {'force_frontier': True}) letting the FSM
        cut the sweep short under clock pressure. `EXPLORATION_COMPLETE` == no frontier
        above the score bar (coverage saturated).
        """
        elapsed = self._elapsed(t)
        force_frontier = bool(budget_state and budget_state.get("force_frontier"))

        # --- opening orientation sweep ---
        if elapsed < self.sweep_s and not force_frontier and self._sweep_idx < len(self._sweep):
            wp = self._sweep[self._sweep_idx]
            # Advance to next diamond vertex once we're near the current one.
            if _near(vehicle_xy, (wp.x, wp.y), tol=0.4):
                self._sweep_idx += 1
                if self._sweep_idx < len(self._sweep):
                    wp = self._sweep[self._sweep_idx]
            if self._sweep_idx < len(self._sweep):
                return ExplorationDecision(ExplorationStatus.SWEEPING, waypoint=wp)

        # --- frontier pursuit ---
        w_size = NUMERICAL_W_SIZE_MULT * W_SIZE if self.qtype is QType.NUMERICAL else W_SIZE
        frontiers = detect_frontiers(grid, vehicle_xy, self.affinity, w_size=w_size)
        for f in frontiers:
            if f.score >= self.min_frontier_score:
                return ExplorationDecision(
                    ExplorationStatus.FRONTIER,
                    waypoint=WaypointCmd(f.xy[0], f.xy[1]),
                    frontier=f,
                )

        # --- nothing left worth exploring ---
        return ExplorationDecision(ExplorationStatus.COMPLETE)


def _near(a: tuple[float, float], b: tuple[float, float], tol: float) -> bool:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5 <= tol
