"""Exploration head — one deterministic exploration/execution step per tick.

Architecture §5 step 1 + row 2:

* **Orientation sweep first.** :class:`core.nav.ExplorationPolicy` emits the opening
  4-point diamond sweep (an in-place spin is impossible — heading is ignored this year),
  seeding the occupancy grid and panorama from every yaw.
* **Question-conditioned frontier exploration** thereafter: the frontier score is biased
  by a detector-affinity term for the plan's nouns. ``affinity_fn`` is injected
  (noun-list -> ((x,y) -> float)); the default is uniform (constant 0), i.e. purely
  geometric frontier scoring.
* **Instruction-following delegates** to the :class:`InstructionHead`'s interleaved
  explore-execute logic once its legs exist (travel doubles as exploration, row 10):
  while a later leg's anchor is ungrounded we still drive the current leg, and the
  frontier affinity is biased toward the ungrounded noun.

This head OWNS the occupancy grid for NUMERICAL / OBJECT_REFERENCE questions (built from
terrain each tick) and publishes exploration waypoints via ``io.publish_waypoint``.

Units: `map` frame, metres. Deterministic: timing from the odom timestamp.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

from core.interfaces import QType, RobotIO, WaypointCmd
from core.nav.exploration import ExplorationPolicy, ExplorationStatus
from core.nav.occupancy import OccupancyGrid
from core.plan_schema import Plan

from core.heads.instruction import InstructionHead

# noun-list -> ((x, y) -> affinity float); default uniform (constant 0).
AffinityFactory = Callable[[Sequence[str]], Callable[[tuple[float, float]], float]]


def uniform_affinity(_nouns: Sequence[str]):
    """Default detector-affinity: constant 0 (purely geometric frontier scoring)."""
    return lambda _xy: 0.0


@dataclass
class ExploreHead:
    """Steps exploration/execution once per tick; delegates IF to the InstructionHead."""

    plan: Plan | None = None
    affinity_fn: AffinityFactory = uniform_affinity
    instruction: InstructionHead | None = None

    grid: OccupancyGrid = field(default_factory=OccupancyGrid)
    _policy: ExplorationPolicy | None = None
    last_status: ExplorationStatus | None = None

    # ------------------------------------------------------------------ per-tick
    def advance(self, io: RobotIO, scene) -> None:
        """One exploration/execution step for this tick."""
        if self.plan is not None and self.plan.qtype is QType.INSTRUCTION_FOLLOWING:
            if self.instruction is not None:
                self.instruction.advance(io, scene)
            return
        self._explore(io)

    def _explore(self, io: RobotIO) -> None:
        odom = io.latest_odom()
        pose = (float(odom.x), float(odom.y)) if odom is not None else (0.0, 0.0)
        t = float(odom.t) if odom is not None else 0.0
        patch = io.latest_terrain(extended=False)
        if patch is not None:
            self.grid.integrate_patch(patch)
        self.grid.mark_pose(pose[0], pose[1])

        if self._policy is None:
            self._policy = ExplorationPolicy(start_xy=pose, affinity=self._affinity())
        decision = self._policy.step(self.grid, pose, t)
        self.last_status = decision.status
        if decision.waypoint is not None:
            io.publish_waypoint(decision.waypoint)

    def _affinity(self):
        nouns = _plan_nouns(self.plan)
        try:
            return self.affinity_fn(nouns)
        except Exception:
            return uniform_affinity(nouns)


def _plan_nouns(plan: Plan | None) -> list[str]:
    """All nouns referenced by a plan (target + clauses, or route anchors)."""
    if plan is None:
        return []
    nouns: list[str] = []
    if plan.target is not None:
        nouns.append(plan.target.noun)
        for cl in plan.target.clauses:
            nouns.extend(a.noun for a in cl.anchors)
    for leg in plan.route:
        nouns.extend(a.noun for a in leg.anchors)
    return [n for n in nouns if n]
