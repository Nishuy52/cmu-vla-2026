"""Instruction-following head — the 70.6%-of-points machinery (architecture §4).

Owns the deepest per-question pipeline:

1. **Ground anchors.** Each ``RouteLeg``'s anchors are resolved via
   :func:`core.geometry.toolbox.resolve` (typo/synonym tolerant, allocentric).
   ``ungrounded_subgoals`` counts legs whose anchors are not yet grounded with the
   observation floor — the FSM's IF early-answer gate reads it (never bank early with a
   gap).
2. **Leg geometry.** goto -> anchor centroid projected to free space; via_near -> a
   point near the anchor; corridor_between -> :func:`toolbox.corridor_gate`.
3. **Avoid capsules.** Stamped ONCE per question into a cloned :class:`Costmap`
   (hard, never relaxed — architecture §1 rows 3/4).
4. **Route.** :func:`core.nav.plan_through` over the costmap -> a
   :class:`BreadcrumbFollower` streaming near-vehicle waypoints at 5 Hz.
5. **Interleaved explore-execute** (architecture row 10): while a later leg's anchor is
   still ungrounded, bias exploration affinity toward its noun but KEEP driving the
   current leg. On arrival at each leg we mark progress (the anchor-confirmation
   checkpoint is an injected callable, stubbed in tests).

The final answer is the drive itself: :meth:`terminal_waypoint` returns the
``WaypointCmd`` at the terminal goal, which is what the FSM publishes for IF (the FSM's
``_dispatch`` routes a WaypointCmd to ``publish_waypoint``).

Units: `map` frame, metres. Deterministic: all timing via the odom timestamp passed to
``advance``.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from core.interfaces import RobotIO, WaypointCmd
from core.geometry import toolbox as TB
from core.geometry.toolbox import DEFAULT_THRESHOLDS, Thresholds
from core.nav.breadcrumbs import BreadcrumbFollower
from core.nav.costmap import Costmap
from core.nav.occupancy import OccupancyGrid
from core.nav.planner import astar, plan_through
from core.plan_schema import Anchor, LegKind, Plan, RouteLeg, TargetSpec

# Flight-recorder-visible seam: recovery events are logged here so a violation-free
# least-bad choice is auditable (architecture §1 row 4 — "never a silent geometry edit").
_LOG = logging.getLogger("core.heads.instruction")

VIA_NEAR_OFFSET_M: float = 1.2  # how far "near" the via anchor to place the waypoint
ARRIVAL_TOL_M: float = 0.8  # within this of a leg goal -> arrived (mark progress)
MIN_GROUND_OBS: int = 3  # per architecture: grounded == confirmed with >= 3 obs


# anchor-confirmation checkpoint (checkpoint 3): (plan, leg_index, anchor_summary) -> bool
# stubbed in tests; True == "confirmed on arrival".
from typing import Callable

AnchorConfirmFn = Callable[[Plan, int, str], bool]


@dataclass
class _GroundedLeg:
    """A route leg with its resolved geometry."""

    kind: LegKind
    grounded: bool
    geom: object | None  # (x,y) for goto/via_near; ((x0,y0),(x1,y1)) for corridor
    nouns: tuple[str, ...]


@dataclass
class InstructionHead:
    """Plans and drives an instruction-following route; steps once per tick."""

    plan: Plan | None = None
    thresholds: Thresholds = DEFAULT_THRESHOLDS
    anchor_confirm: AnchorConfirmFn | None = None

    grid: OccupancyGrid = field(default_factory=OccupancyGrid)
    _costmap: Costmap | None = None
    _follower: BreadcrumbFollower | None = None
    _legs: list[_GroundedLeg] = field(default_factory=list)
    _leg_progress: int = 0  # index of the current (not-yet-arrived) leg
    _confirmed: set[int] = field(default_factory=set)
    _terminal_xy: tuple[float, float] | None = None
    _last_wp: WaypointCmd | None = None

    # ------------------------------------------------------------------ per-tick
    def advance(self, io: RobotIO, scene) -> None:
        """One deterministic step: refresh the map, (re)ground legs, drive the route.

        ``scene`` is the live SceneIndex the factory injected (the anchors are resolved
        against it); ``io`` supplies terrain/odom and the waypoint sink.
        """
        if self.plan is None or not self.plan.route:
            return
        odom = io.latest_odom()
        pose = (float(odom.x), float(odom.y)) if odom is not None else (0.0, 0.0)
        t = float(odom.t) if odom is not None else 0.0
        self._ingest_terrain(io, pose)
        self._ground_legs(scene)

        if self._follower is None:
            self._build_route(pose, scene)
        self._drive(io, pose, t)

    # ------------------------------------------------------------------ map
    def _ingest_terrain(self, io: RobotIO, pose: tuple[float, float]) -> None:
        patch = io.latest_terrain(extended=False)
        if patch is not None:
            self.grid.integrate_patch(patch)
        self.grid.mark_pose(pose[0], pose[1])

    # ------------------------------------------------------------------ grounding
    def _ground_legs(self, scene) -> None:
        legs: list[_GroundedLeg] = []
        for leg in self.plan.route:
            legs.append(self._ground_one(leg, scene))
        self._legs = legs

    def _ground_one(self, leg: RouteLeg, scene) -> _GroundedLeg:
        nouns = tuple(a.noun for a in leg.anchors)
        if scene is None:
            return _GroundedLeg(leg.kind, False, None, nouns)
        recs = [self._resolve_anchor(a, scene) for a in leg.anchors]
        if any(r is None for r in recs):
            return _GroundedLeg(leg.kind, False, None, nouns)
        grounded = all(r.n_obs >= MIN_GROUND_OBS for r in recs)
        if leg.kind is LegKind.CORRIDOR_BETWEEN:
            gate = TB.corridor_gate(recs[0], recs[1])
            geom = (
                (float(gate.p0[0]), float(gate.p0[1])),
                (float(gate.p1[0]), float(gate.p1[1])),
            )
        elif leg.kind is LegKind.VIA_NEAR:
            geom = self._via_point(recs[0])
        else:  # GOTO
            geom = self._goto_point(recs[0])
        return _GroundedLeg(leg.kind, grounded, geom, nouns)

    def _resolve_anchor(self, anchor: Anchor, scene):
        """Resolve one anchor to its best InstanceRecord, honouring attributes."""
        spec = TargetSpec(noun=anchor.noun, raw=anchor.raw, attributes=list(anchor.attributes))
        res = TB.resolve(spec, scene, self.thresholds)
        return res.candidates_ranked[0] if res.candidates_ranked else None

    def _goto_point(self, rec) -> tuple[float, float]:
        """Anchor centroid projected to the nearest free cell (drivable goal)."""
        c = TB.P._as3(rec.centroid)
        return self._project_free((float(c[0]), float(c[1])))

    def _via_point(self, rec) -> tuple[float, float]:
        """A point ~VIA_NEAR_OFFSET_M from the anchor centroid, projected to free space."""
        c = TB.P._as3(rec.centroid)
        return self._project_free((float(c[0]) + VIA_NEAR_OFFSET_M, float(c[1])))

    def _project_free(self, xy: tuple[float, float]) -> tuple[float, float]:
        """Nudge a point off an obstacle onto the nearest passable cell (if a costmap exists)."""
        cm = self._costmap
        if cm is None:
            return xy
        r, c = self.grid.world_to_cell(*xy)
        if cm.passable(r, c):
            return xy
        nr, nc = cm._nearest_passable_cell(r, c)
        if nr is None:
            return xy
        return self.grid.cell_to_world(nr, nc)

    # ------------------------------------------------------------------ routing
    def _build_route(self, start_xy: tuple[float, float], scene) -> None:
        """Stamp avoids ONCE, plan the ordered legs, wrap in a BreadcrumbFollower.

        Requires the first leg grounded (a start point to plan from) and all legs to
        have geometry; retried each tick until legs firm up.
        """
        if not self._legs or any(l.geom is None for l in self._legs):
            return
        self._costmap = Costmap(self.grid)
        self._stamp_avoids(scene)
        # geometry may need re-projection now the costmap exists.
        self._ground_legs(scene)
        if any(l.geom is None for l in self._legs):
            return
        legs = [self._leg_tuple(l) for l in self._legs]
        path = plan_through(self._costmap, start_xy, legs)
        if path is None:
            # Unreachable with the hard capsules in place: drive to the nearest legal
            # point to the terminal goal and answer from there (architecture row 4).
            # The recovery path MUST be planned through the costmap — never a raw
            # straight segment, which could cut through a hard capsule (the very
            # violation the capsule exists to prevent).
            path = self._recover_path(start_xy)
        self._terminal_xy = path[-1]
        self._follower = BreadcrumbFollower(path=path, costmap=self._costmap)

    def _recover_path(self, start_xy: tuple[float, float]) -> list[tuple[float, float]]:
        """Least-bad legal path when the full route is unreachable (architecture §1 row 4).

        ``nearest_reachable_point`` returns the passable cell closest to the terminal
        goal, found by BFS over passable cells *from start* — so it is astar-reachable
        by construction. We still plan the segment with A* over the stamped costmap so
        the path is capsule-validated, never a raw segment. If A* nonetheless fails
        (theoretically impossible given the BFS reachability guarantee), we remain in
        place (emit the start cell) and log a flight-recorder-visible event rather than
        ever hand a raw, unvalidated segment to the follower.
        """
        goal = self._legs[-1].geom
        goal_xy = goal[1] if isinstance(goal[0], tuple) else goal  # corridor->2nd pt
        legal = self._costmap.nearest_reachable_point(goal_xy, start_xy)
        path = astar(self._costmap, start_xy, legal)
        if path is None:
            _LOG.warning(
                "IF recovery: A* to nearest reachable point %s from %s failed despite "
                "BFS reachability; remaining in place at start.",
                legal,
                start_xy,
            )
            # Snap start onto its own cell centre so the emitted path is a valid,
            # costmap-consistent single vertex (never a raw two-point segment).
            sr, sc = self.grid.world_to_cell(*start_xy)
            here = self.grid.cell_to_world(sr, sc)
            return [here]
        _LOG.info(
            "IF recovery: route unreachable with hard capsules; driving to nearest "
            "legal point %s (%d waypoints) and answering there.",
            path[-1],
            len(path),
        )
        return path

    def _leg_tuple(self, leg: _GroundedLeg):
        if leg.kind is LegKind.CORRIDOR_BETWEEN:
            return ("corridor_between", leg.geom)
        if leg.kind is LegKind.VIA_NEAR:
            return ("via_near", leg.geom)
        return ("goto", leg.geom)

    def _stamp_avoids(self, scene) -> None:
        """Stamp every AvoidSpec's capsule hard into the costmap once (never relaxed)."""
        if scene is None or self._costmap is None:
            return
        for spec in self.plan.avoid:
            try:
                cap = TB.avoid_capsule(spec, scene, self.thresholds)
            except ValueError:
                continue  # anchor unresolvable: no capsule to stamp yet
            seg = (
                (float(cap.a[0]), float(cap.a[1])),
                (float(cap.b[0]), float(cap.b[1])),
            )
            self._costmap.stamp_capsule(seg, cap.radius)

    # ------------------------------------------------------------------ drive
    def _drive(self, io: RobotIO, pose: tuple[float, float], t: float) -> None:
        if self._follower is None:
            return
        wp = self._follower.advance(pose, t)
        self._mark_arrivals(pose)
        if wp is None:
            # route exhausted: hold at the terminal goal.
            if self._terminal_xy is not None:
                wp = WaypointCmd(float(self._terminal_xy[0]), float(self._terminal_xy[1]))
        if wp is not None:
            io.publish_waypoint(wp)
            self._last_wp = wp

    def _mark_arrivals(self, pose: tuple[float, float]) -> None:
        """Confirm each leg goal as the vehicle reaches it (anchor-confirmation checkpoint)."""
        for i, leg in enumerate(self._legs):
            if i in self._confirmed or leg.geom is None:
                continue
            goal = leg.geom[1] if isinstance(leg.geom[0], tuple) else leg.geom
            if _dist(pose, goal) <= ARRIVAL_TOL_M:
                self._confirm_leg(i, leg)

    def _confirm_leg(self, i: int, leg: _GroundedLeg) -> None:
        self._confirmed.add(i)
        self._leg_progress = max(self._leg_progress, i + 1)
        if self.anchor_confirm is not None:
            try:
                self.anchor_confirm(self.plan, i, f"leg {i} nouns={leg.nouns}")
            except Exception:
                pass

    # ------------------------------------------------------------------ read-out
    def ungrounded_subgoals(self) -> int:
        """Count of route legs not yet grounded with the observation floor."""
        if not self._legs:
            # No grounding attempted yet: every leg is ungrounded.
            return len(self.plan.route) if self.plan and self.plan.route else 0
        return sum(1 for l in self._legs if not l.grounded)

    def first_anchor_pt(self):
        """(x, y) of the best-grounded first leg goal, for the floor. None if unknown."""
        for leg in self._legs:
            if leg.geom is not None:
                g = leg.geom
                return g[1] if isinstance(g[0], tuple) else g
        return None

    def next_noun_affinity_target(self) -> str | None:
        """The noun of the earliest still-ungrounded leg (interleaved explore bias)."""
        for leg in self._legs:
            if not leg.grounded and leg.nouns:
                return leg.nouns[0]
        return None

    def terminal_waypoint(self) -> WaypointCmd | None:
        """The WaypointCmd at the terminal goal — the FSM's IF 'answer'."""
        if self._terminal_xy is not None:
            return WaypointCmd(float(self._terminal_xy[0]), float(self._terminal_xy[1]))
        return self._last_wp


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5
