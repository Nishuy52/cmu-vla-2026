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

import inspect
import logging
from dataclasses import dataclass, field

import numpy as np

from core.interfaces import RobotIO, WaypointCmd
from core.geometry import toolbox as TB
from core.geometry.toolbox import DEFAULT_THRESHOLDS, Thresholds
from core.nav.breadcrumbs import BreadcrumbFollower
from core.nav.costmap import Costmap
from core.nav.occupancy import OccupancyGrid, integrate_scan_overhead_decimated
from core.nav.planner import astar, free_space_via_point, plan_through
from core.plan_schema import Anchor, LegKind, Plan, Pred, RouteLeg, TargetSpec

# Flight-recorder-visible seam: recovery events are logged here so a violation-free
# least-bad choice is auditable (architecture §1 row 4 — "never a silent geometry edit").
_LOG = logging.getLogger("core.heads.instruction")

VIA_NEAR_OFFSET_M: float = 1.2  # fallback "near" offset if free-space placement fails
ARRIVAL_TOL_M: float = 0.8  # within this of a leg goal -> arrived (mark progress)
#: Standoff clearance (m) beyond a "near" anchor's footprint edge for the VIA_NEAR via
#: placement. A "near" via should sit just clear of the object (about a vehicle radius),
#: NOT a full 1.2 m away: the rubric credits a leg only when the driven pose comes within
#: ARRIVAL_TOL_M of the anchor centroid, so a via placed footprint+1.2 m out overshoots
#: the tolerance for every compact anchor and the ordered leg scores 0 (T11: the dominant
#: VIA_NEAR "legs=0/N" rows). Kept below ARRIVAL_TOL_M so a compact anchor's via lands
#: inside the credited band while still clearing the footprint.
VIA_NEAR_CLEARANCE_M: float = 0.45
MIN_GROUND_OBS: int = 3  # per architecture: grounded == confirmed with >= 3 obs

# Issue #33 — route-prefix commitment floor: a leg backed by only a SINGLE observation
# (n_obs == 1) is too weak to drive on. Such a leg may still be PLANNED (geometry
# computed, feeding explore affinity/probing) but is withheld from the COMMITTED route
# — the driven prefix stops just short of it — until either it gathers a second
# observation (n_obs >= MIN_COMMIT_OBS) or the forced-assembly time-pressure gate
# (core.fsm.budget.BudgetState.forced_assembly, T-90) is reached, at which point
# late-stage expected points favor acting on the single-obs leg anyway.
MIN_COMMIT_OBS: int = 2

# H11 / IF-F8 / SYS-F10: cap re-plans per question so a pathological stall/violation loop
# cannot burn the whole budget replanning every tick. Each replan trigger (stall, no-LOS,
# follower-exhausted-not-arrived, capsule tripwire) counts against this.
MAX_REPLANS_PER_QUESTION: int = 3

# H4c: fraction of the explore budget past which a PROVISIONAL terminal grounding
# (one resolved via a relaxation rung — drop_relation/category_only/drop_disambiguator)
# is committed anyway. Below this, we keep exploring for the missing disambiguator/anchor
# rather than banking a possibly-wrong terminal (IF-F3). With no budget signal injected
# (budget_frac is None) there is no pressure to withhold, so a provisional terminal
# commits immediately (today's behaviour + the drive still banks the grounded prefix).
PROVISIONAL_COMMIT_FRAC: float = 0.85

# Relaxation steps that make a terminal grounding PROVISIONAL (the resolve audit exposes
# these): the terminal instance was picked only after a filter was dropped.
_PROVISIONAL_STEPS = frozenset(
    {"drop_relation", "category_only", "drop_disambiguator", "relax_attributes"}
)


from typing import Any, Callable

# Legacy anchor-confirmation seam (checkpoint 3): (plan, leg_index, anchor_summary) -> bool
# stubbed in tests; True == "confirmed on arrival". Preserved for backward compatibility.
LegacyAnchorConfirmFn = Callable[[Plan, int, str], bool]

# Rich CP3 seam (design doc §CP3, vision contract). Called with the anchor's description
# string and a zero-arg projected-crop supplier; returns an AnchorConfirmOutcome-shaped
# object (duck-typed: ``.action`` in {"confirm","demote"}, plus ``.confidence``). On
# ``demote`` the head demotes the leg's grounding and re-plans it to the runner-up.
#     run(anchor_desc: str, crop: Any) -> AnchorConfirmOutcome
AnchorConfirmFn = Callable[..., Any]


@dataclass
class _GroundedLeg:
    """A route leg with its resolved geometry."""

    kind: LegKind
    grounded: bool
    geom: object | None  # (x,y) for goto/via_near; ((x0,y0),(x1,y1)) for corridor
    nouns: tuple[str, ...]
    record: object | None = None  # best InstanceRecord for the primary anchor (CP3)
    runner_up: object | None = None  # runner-up InstanceRecord for demote-and-replan (CP3)
    #: H4c — True when this leg's grounding used a relaxation rung (drop_relation /
    #: category_only / drop_disambiguator / relax_attributes). A provisional TERMINAL
    #: leg is withheld from the committed route while explore budget remains.
    provisional: bool = False
    #: Issue #33 — the minimum ``n_obs`` across this leg's resolved anchor record(s).
    #: A leg with fewer than MIN_COMMIT_OBS observations is PLANNED (geom present) but
    #: withheld from the COMMITTED route prefix (see ``_committable_prefix_len``) absent
    #: forced-assembly time pressure. A very large default keeps an ungrounded leg
    #: (geom is None, min_n_obs never set) from spuriously gating a shorter prefix.
    min_n_obs: int = 1 << 30


@dataclass
class InstructionHead:
    """Plans and drives an instruction-following route; steps once per tick."""

    plan: Plan | None = None
    thresholds: Thresholds = DEFAULT_THRESHOLDS
    anchor_confirm: AnchorConfirmFn | None = None
    #: H4c — zero-arg callable -> explored-budget fraction [0,1]. Feeds the provisional
    #: terminal commit gate: below PROVISIONAL_COMMIT_FRAC a relaxation-audited terminal
    #: is withheld from the route (keep exploring for the disambiguator). None == no
    #: pressure signal, so a provisional terminal commits immediately.
    budget_frac: Callable[[], float] | None = None
    #: Issue #33 — zero-arg callable -> True once the T-90 forced-assembly gate
    #: (core.fsm.budget.BudgetState.forced_assembly) has been reached. Feeds the
    #: single-observation route-prefix commit gate: below MIN_COMMIT_OBS a leg is
    #: withheld from the committed prefix until either it gathers a second observation
    #: or this hook reports the forced-assembly gate reached. None (unconfigured, or a
    #: broken/raising hook) == not yet forced, so the strict n_obs >= MIN_COMMIT_OBS
    #: floor applies (this is a NEW correctness gate, not a legacy behaviour to
    #: preserve — unlike ``budget_frac``'s None-means-commit-immediately default).
    forced_assembly: Callable[[], bool] | None = None

    grid: OccupancyGrid = field(default_factory=OccupancyGrid)
    _costmap: Costmap | None = None
    _follower: BreadcrumbFollower | None = None
    _legs: list[_GroundedLeg] = field(default_factory=list)
    _driven_prefix: int = 0  # number of leading legs the committed route currently covers
    _leg_progress: int = 0  # index of the current (not-yet-arrived) leg
    _confirmed: set[int] = field(default_factory=set)
    _terminal_xy: tuple[float, float] | None = None
    _last_wp: WaypointCmd | None = None
    _pose: tuple[float, float] = (0.0, 0.0)  # last-known pose, for free-space via placement
    _demoted: set[int] = field(default_factory=set)  # anchors CP3 confidently rejected
    _legacy_confirm: bool = False  # anchor_confirm matches the old bool seam
    _scene: object | None = None  # live scene retained for CP3 re-resolve
    #: H11 — the avoid Capsules currently stamped into the costmap (toolbox.Capsule
    #: objects), retained so the per-tick runtime tripwire can test the pose/next-crumb
    #: against them via ``toolbox.capsule_violated`` (read-only) without re-deriving them.
    _stamped_capsules: list[object] = field(default_factory=list)
    #: H11 (IF-F8/SYS-F10) — count of re-plans this question + a flight-recorder-visible
    #: log of why each fired. Capped at MAX_REPLANS_PER_QUESTION.
    _replans: int = 0
    _replan_events: list[str] = field(default_factory=list)
    #: H11 capsule tripwire edge-detect: True once the pose has been observed inside a
    #: stamped capsule, so we re-plan on ENTRY (clear->violated) rather than every tick a
    #: legitimately-engulfed start stays inside (the recovery path already drives it out).
    _in_capsule: bool = False

    def __post_init__(self) -> None:
        # Backward compat: the old ``(plan, leg_index, summary) -> bool`` seam is detected
        # by signature and called the legacy way in _confirm_leg.
        if self.anchor_confirm is not None and _is_legacy_anchor_seam(self.anchor_confirm):
            self._legacy_confirm = True

    # ------------------------------------------------------------------ per-tick
    def advance(self, io: RobotIO, scene) -> bool:
        """One deterministic step: refresh the map, (re)ground legs, drive the route.

        ``scene`` is the live SceneIndex the factory injected (the anchors are resolved
        against it); ``io`` supplies terrain/odom and the waypoint sink.

        Returns ``True`` iff a waypoint was published this tick. ``False`` lets the
        caller (``ExploreHead``) fall through to frontier exploration so an ungrounded
        or not-yet-grounded route never leaves the robot parked (H3 / IF-F1 / SYS-F3).
        """
        if self.plan is None or not self.plan.route:
            return False
        self._scene = scene
        odom = io.latest_odom()
        pose = (float(odom.x), float(odom.y)) if odom is not None else (0.0, 0.0)
        t = float(odom.t) if odom is not None else 0.0
        self._pose = pose
        self._ingest_terrain(io, pose)
        self._ground_legs(scene)

        # (Re)build/extend the committed route to cover the longest grounded prefix.
        # A later leg grounding after the first build extends the drive (H3c); until any
        # prefix is grounded, no follower exists and we emit nothing (caller explores).
        self._maybe_build_or_extend_route(pose, scene)
        # H11 runtime avoid tripwire (IF-F5): a stamped capsule the driven pose is entering
        # forces a re-plan away (bounded by the replan cap). Checked before the drive so the
        # crumb emitted this tick reflects any re-plan.
        self._capsule_tripwire(pose, scene)
        return self._drive(io, pose, t)

    # ------------------------------------------------------------------ map
    def _ingest_terrain(self, io: RobotIO, pose: tuple[float, float]) -> None:
        patch = io.latest_terrain(extended=False)
        if patch is not None:
            # vehicle_z feeds the runtime ground-offset estimator (issue #36) used
            # by the overhead fallback below.
            self.grid.integrate_patch(patch, vehicle_z=_vehicle_z(io))
        # Overhead-clearance layer: fold the raw /registered_scan through the grid so
        # overhangs the terrain slab filtered out (bar tables the base stack reads as
        # FREE floor) get flagged. Terrain first so per-cell ground_z is available.
        scan = io.latest_scan() if hasattr(io, "latest_scan") else None
        if scan is not None:
            vehicle_z = _vehicle_z(io)
            integrate_scan_overhead_decimated(self.grid, scan, vehicle_z)
        self.grid.mark_pose(pose[0], pose[1])

    # ------------------------------------------------------------------ grounding
    def _ground_legs(self, scene) -> None:
        legs: list[_GroundedLeg] = []
        prev_xy: tuple[float, float] | None = None
        for leg in self.plan.route:
            gl = self._ground_one(leg, scene, prev_xy)
            legs.append(gl)
            if gl.geom is not None:
                g = gl.geom
                prev_xy = g[1] if isinstance(g[0], tuple) else g
        self._legs = legs

    def _ground_one(
        self, leg: RouteLeg, scene, prev_xy: tuple[float, float] | None = None
    ) -> _GroundedLeg:
        nouns = tuple(a.noun for a in leg.anchors)
        if scene is None:
            return _GroundedLeg(leg.kind, False, None, nouns)
        recs, provisional = self._resolve_leg_anchors(leg, scene, prev_xy)
        if any(r is None for r in recs):
            # IF-F6: a corridor leg whose two anchors share a noun but has < 2 distinct
            # instances lands here (recs[1] is None) and stays ungrounded — which now
            # correctly feeds the H3 explore path, not a zero-width-gate recovery beeline.
            return _GroundedLeg(leg.kind, False, None, nouns, provisional=provisional)
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
        runner_up = self._resolve_anchor(leg.anchors[0], scene, prev_xy)[1]
        min_n_obs = min(r.n_obs for r in recs)
        return _GroundedLeg(
            leg.kind, grounded, geom, nouns, record=recs[0], runner_up=runner_up,
            provisional=provisional, min_n_obs=min_n_obs,
        )

    def _resolve_leg_anchors(
        self, leg: RouteLeg, scene, prev_xy: tuple[float, float] | None = None
    ):
        """Resolve every anchor of a leg to a best InstanceRecord, enforcing DISTINCT
        instances across the leg's anchors (IF-F6).

        Returns ``(recs, provisional)`` where ``recs`` is one record (or None) per
        anchor in order, and ``provisional`` is True iff ANY anchor's resolve leaned on
        a relaxation rung (H4c). "the two X" / "between the two X" duplicate the anchor
        (regex_tier ``_split_pair``); resolving both independently yields the SAME
        top-ranked instance -> a zero-width gate -> whole-route recovery collapse. Here
        the second anchor of a shared-noun pair takes the next distinct ranked instance;
        if fewer than 2 exist the leg stays ungrounded.
        """
        recs: list[object | None] = []
        used: set[int] = set()
        provisional = False
        for anchor in leg.anchors:
            best, prov = self._resolve_anchor_distinct(anchor, scene, used, prev_xy)
            provisional = provisional or prov
            recs.append(best)
            if best is not None:
                used.add(getattr(best, "instance_id", -1))
        return recs, provisional

    def _resolve_anchor_distinct(
        self, anchor: Anchor, scene, used: set[int],
        prev_xy: tuple[float, float] | None = None,
    ):
        """Best ranked InstanceRecord for ``anchor`` NOT already claimed by an earlier
        anchor of the same leg (``used``); plus whether the resolve was relaxation-audited.
        """
        ranked, provisional = self._ranked_anchor(anchor, scene, prev_xy)
        for c in ranked:
            if getattr(c, "instance_id", -1) not in used:
                return c, provisional
        return None, provisional

    def _ranked_anchor(
        self, anchor: Anchor, scene, prev_xy: tuple[float, float] | None = None
    ):
        """(ranked survivors minus demoted, provisional?) for one anchor spec.

        ``provisional`` mirrors H4c: True when the resolve audit trail contains a
        relaxation rung (a filter was dropped to land on these candidates).

        Salience tie-break (IF-F3): the toolbox ranks survivors, but among ambiguous
        equals it falls back to instance-id order — an arbitrary detection-order pick.
        When a previous leg is grounded we re-order the survivors by proximity to it
        (nearest-to-previous-leg), a defensible ordered-route salience: the instance the
        robot would reach next by continuing the drive wins, never a detection-id accident.
        Chosen over largest-instance because IF routes are ordered and spatially local, so
        "nearest to where I just was" matches the instruction's intent far more often.
        """
        spec = TargetSpec(
            noun=anchor.noun,
            raw=anchor.raw,
            attributes=list(anchor.attributes),
            # A nested disambiguator on a route anchor ("the lamp near the bench") is a
            # constraint the toolbox must evaluate — carry it as a target clause so any
            # relaxation (drop_relation/category_only) shows up in the audit and marks the
            # grounding provisional (H4c). Without this the clause is invisible and a
            # relaxed terminal looks clean.
            clauses=[anchor.disambiguator] if anchor.disambiguator is not None else [],
        )
        res = TB.resolve(spec, scene, self.thresholds)
        ranked = [c for c in res.candidates_ranked if c.instance_id not in self._demoted]
        provisional = any(r.step in _PROVISIONAL_STEPS for r in res.audit)
        # Only re-order when the toolbox left an instance-id-only tie (no superlative
        # margin distinguishing the top survivors) and we have a previous leg to anchor on.
        if prev_xy is not None and len(ranked) > 1 and not _has_superlative(anchor):
            px, py = prev_xy
            ranked = sorted(
                ranked,
                key=lambda c: (
                    (float(TB.P._as3(c.centroid)[0]) - px) ** 2
                    + (float(TB.P._as3(c.centroid)[1]) - py) ** 2,
                    getattr(c, "instance_id", 0),
                ),
            )
        return ranked, provisional

    def _resolve_anchor(
        self, anchor: Anchor, scene, prev_xy: tuple[float, float] | None = None
    ):
        """Resolve one anchor to (best, runner_up) InstanceRecords, honouring attributes.

        The runner-up is used by the CP3 demote-and-replan path. Demoted anchors (CP3
        confidently rejected on arrival) are skipped so re-resolution lands on the
        runner-up.
        """
        ranked, _ = self._ranked_anchor(anchor, scene, prev_xy)
        if not ranked:
            return (None, None)
        return (ranked[0], ranked[1] if len(ranked) > 1 else None)

    def _goto_point(self, rec) -> tuple[float, float]:
        """Anchor centroid projected to the nearest free cell reachable from the pose.

        Projecting to the nearest *passable* cell is not enough: that cell can sit in a
        pocket disconnected from the drivable free-space component (a goal wedged between
        an obstacle and a wall), so ``plan_through``'s A* to it fails and the whole leg is
        skipped — the vehicle beelines to the terminal and the ordered leg scores 0 (T11
        sig-1). When a costmap + pose exist we therefore snap to the nearest cell REACHABLE
        from the current pose (``nearest_reachable_point`` BFS), guaranteeing the leg goal
        is a point the drive can actually arrive at."""
        c = TB.P._as3(rec.centroid)
        anchor_xy = (float(c[0]), float(c[1]))
        cm = self._costmap
        if cm is None:
            return self._project_free(anchor_xy)
        # Use the costmap's per-pose reachable mask (memoised, so a whole grounding pass
        # over many legs floods once, not once-per-leg-per-tick): if the anchor cell is
        # directly reachable use it, else snap to the nearest reachable cell.
        seen = cm.reachable_mask(self._pose)
        if seen is not None:
            r, col = self.grid.world_to_cell(*anchor_xy)
            if 0 <= r < seen.shape[0] and 0 <= col < seen.shape[1] and seen[r, col]:
                return anchor_xy
        return cm.nearest_reachable_point(anchor_xy, self._pose)

    def _via_point(self, rec) -> tuple[float, float]:
        """A "path near the anchor" waypoint placed by free-space gradient (IF-F7).

        Once a costmap exists, ``free_space_via_point`` picks the max-clearance passable
        cell at ~``near_thresh`` from the anchor that is reachable from the current pose —
        so the via lands on the robot's side of a wall-adjacent anchor, not inside a wall
        and not on the unreachable side (the old fixed ``+1.2 m +x`` offset's failure
        mode). Before a costmap exists (first grounding pass, pre-route), we fall back to
        the fixed offset projection.
        """
        c = TB.P._as3(rec.centroid)
        anchor_xy = (float(c[0]), float(c[1]))
        cm = self._costmap
        if cm is not None:
            via = free_space_via_point(cm, anchor_xy, self._pose, self._near_thresh(rec))
            if via is not None:
                return via
        return self._project_free((anchor_xy[0] + VIA_NEAR_OFFSET_M, anchor_xy[1]))

    def _near_thresh(self, rec) -> float:
        """The proximity threshold for a "near" via, scaled to the anchor's footprint.

        Half the anchor's footprint diagonal (so the via clears the object's shell) plus a
        vehicle-radius clearance — NOT a fixed 1.2 m standoff, which overshoots the rubric's
        arrival band for every compact anchor (T11). For a compact anchor (half-diag small)
        this lands the via within ``ARRIVAL_TOL_M`` of the centroid, exactly where "near"
        is credited; for a large anchor the half-diag term dominates and the via sits just
        outside its shell (no in-footprint via), the best reachable "near" the geometry
        allows.
        """
        try:
            ext = rec.aabb_max - rec.aabb_min
            half_diag = 0.5 * float((float(ext[0]) ** 2 + float(ext[1]) ** 2) ** 0.5)
        except Exception:  # noqa: BLE001 — a malformed rec falls back to the fixed offset
            half_diag = 0.0
        return half_diag + VIA_NEAR_CLEARANCE_M

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
    def _grounded_prefix_len(self) -> int:
        """Length of the longest LEADING run of legs that already have geometry.

        Partial-credit drive (H3c): a route with legs 1-2 grounded and leg 3 not yields
        prefix length 2 — we build and drive that prefix now, banking the ordered legs,
        and extend the route when leg 3 later grounds. A provisional TERMINAL leg (H4c)
        is withheld from the prefix while budget remains (see ``_committable_prefix_len``).
        """
        n = 0
        for leg in self._legs:
            if leg.geom is None:
                break
            n += 1
        return n

    def _committable_prefix_len(self) -> int:
        """The grounded-prefix length we will actually commit to the route this tick.

        Two independent withhold gates apply, in order:

        1. Issue #33 — single-observation floor: the prefix is first truncated at the
           earliest leg backed by fewer than MIN_COMMIT_OBS observations (n_obs == 1
           legs are PLANNED — geometry already computed by ``_ground_legs`` — but not
           COMMITTED), unless the T-90 forced-assembly gate has been reached.
        2. H4c — provisional terminal: if the (obs-gated) prefix reaches the FINAL route
           leg and that terminal was resolved via a relaxation rung, hold it back (drive
           only the legs before it) until budget pressure forces the commit, so
           exploration can still find the missing disambiguator/anchor. Non-terminal
           provisional legs are NOT withheld — partial credit on ordered early legs is
           banked regardless.
        """
        n = self._obs_gated_prefix_len(self._grounded_prefix_len())
        if n == 0 or n < len(self._legs):
            return n  # terminal not yet in the prefix; nothing to withhold
        terminal = self._legs[-1]
        if terminal.provisional and not self._commit_forced():
            return n - 1  # withhold the provisional terminal; drive the rest
        return n

    def _obs_gated_prefix_len(self, n: int) -> int:
        """Issue #33: truncate a geometry-grounded prefix of length ``n`` at the first
        leg backed by fewer than MIN_COMMIT_OBS observations.

        A leg with n_obs == 1 stays PLANNED (its geometry is already in ``self._legs``,
        feeding explore affinity / the WorldView probe) but is withheld from the
        COMMITTED route — the driven prefix stops just short of it — until it gathers a
        second observation, EXCEPT once the T-90 forced-assembly gate is reached: late-
        stage time pressure favors acting on the single-obs leg over stalling further.
        """
        if self._forced_assembly_reached():
            return n
        for i in range(n):
            if self._legs[i].min_n_obs < MIN_COMMIT_OBS:
                return i
        return n

    def _forced_assembly_reached(self) -> bool:
        """True once the T-90 forced-assembly gate (issue #33) has been reached.

        No injected hook (``forced_assembly is None``), or a hook that raises, means NOT
        yet forced — this is a new correctness floor (single obs is too weak to commit
        on), not legacy behaviour to preserve, so the unconfigured default is strict."""
        if self.forced_assembly is None:
            return False
        try:
            return bool(self.forced_assembly())
        except Exception:  # noqa: BLE001 — a broken signal must not strand the route
            return False

    def _commit_forced(self) -> bool:
        """True once budget pressure forces committing even a provisional terminal.

        No injected budget signal (``budget_frac is None``) == no pressure to withhold,
        so a provisional terminal commits immediately (preserves today's single-tick
        behaviour and still banks the drive)."""
        if self.budget_frac is None:
            return True
        try:
            return float(self.budget_frac()) >= PROVISIONAL_COMMIT_FRAC
        except Exception:  # noqa: BLE001 — a broken signal must not strand the route
            return True

    def _maybe_build_or_extend_route(self, start_xy: tuple[float, float], scene) -> None:
        """Build the committed route over the grounded prefix, or extend it as later
        legs ground. Rebuilds from ``start_xy`` whenever the committable prefix grows
        (H3c partial-route drive); a shrinking/steady prefix leaves the follower intact.
        """
        want = self._committable_prefix_len()
        if want == 0:
            return  # nothing grounded yet — caller (ExploreHead) explores this tick
        # IF-F5: an unresolvable avoid anchor keeps the WHOLE route uncommitted while
        # budget remains (feeding the explore fallthrough) — committing now would drive a
        # route the forbidden capsule can't yet be stamped into. Once we already have a
        # follower we keep driving it (re-stamp happens on rebuild); this gate only blocks
        # the FIRST commit.
        if (
            self._follower is None
            and not self._avoids_all_resolvable(scene)
            and not self._commit_forced()
        ):
            return
        if self._follower is not None and want <= self._driven_prefix:
            return  # already driving a route covering (at least) this prefix
        self._build_route(start_xy, scene, want)

    def _build_route(self, start_xy: tuple[float, float], scene, prefix_len: int) -> None:
        """Stamp avoids ONCE, plan the leading ``prefix_len`` ordered legs, wrap in a
        BreadcrumbFollower.

        Plans only the grounded prefix (partial-route drive, H3c); the route is rebuilt
        with a longer prefix as later legs ground. ``_stamp_avoids`` is idempotent over
        specs, so a rebuild re-stamps any avoid anchors that have since grounded.
        """
        prefix = self._legs[:prefix_len]
        if not prefix or any(l.geom is None for l in prefix):
            return
        self._costmap = Costmap(self.grid)
        self._stamp_avoids(scene)
        # geometry may need re-projection now the costmap exists.
        self._ground_legs(scene)
        prefix = self._legs[:prefix_len]
        if any(l.geom is None for l in prefix):
            return
        legs = [self._leg_tuple(l) for l in prefix]
        path = plan_through(self._costmap, start_xy, legs)
        if path is None:
            # Unreachable with the hard capsules in place: drive to the nearest legal
            # point to the terminal goal and answer from there (architecture row 4).
            # The recovery path MUST be planned through the costmap — never a raw
            # straight segment, which could cut through a hard capsule (the very
            # violation the capsule exists to prevent).
            path = self._recover_path(start_xy, prefix)
        self._terminal_xy = path[-1]
        self._follower = BreadcrumbFollower(path=path, costmap=self._costmap)
        self._driven_prefix = prefix_len

    def _recover_path(
        self, start_xy: tuple[float, float], prefix: list[_GroundedLeg] | None = None
    ) -> list[tuple[float, float]]:
        """Least-bad legal path when the (prefix) route is unreachable (architecture §1 row 4).

        ``nearest_reachable_point`` returns the passable cell closest to the goal (the
        prefix's last grounded leg), found by BFS over passable cells *from start* — so
        it is astar-reachable by construction. We still plan the segment with A* over the
        stamped costmap so the path is capsule-validated, never a raw segment. If A*
        nonetheless fails (theoretically impossible given the BFS reachability guarantee),
        we remain in place (emit the start cell) and log a flight-recorder-visible event
        rather than ever hand a raw, unvalidated segment to the follower.
        """
        legs = prefix if prefix is not None else self._legs
        goal = legs[-1].geom
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
        """Stamp every resolvable AvoidSpec's capsule hard into the costmap (never relaxed).

        Idempotent over specs, so a re-plan / re-ground re-stamps any avoid anchor that
        has since grounded (IF-F5). The stamped Capsules are retained in
        ``_stamped_capsules`` for the per-tick runtime tripwire (``capsule_violated``).
        """
        if scene is None or self._costmap is None:
            return
        self._stamped_capsules = []
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
            self._stamped_capsules.append(cap)

    def _avoids_all_resolvable(self, scene) -> bool:
        """True iff every AvoidSpec's anchors currently resolve (IF-F5).

        Avoid grounding is a route-commit precondition of the SAME rank as leg grounding:
        an unresolvable avoid anchor (the "TV" not yet detected) keeps the route
        uncommitted while budget remains, so the robot keeps exploring toward the avoid
        noun instead of committing a route that would legally cut through the (unstamped)
        forbidden corridor. Once budget pressure forces the commit we proceed with whatever
        capsules resolved (never strand).
        """
        if scene is None:
            return True
        for spec in self.plan.avoid:
            try:
                TB.avoid_capsule(spec, scene, self.thresholds)
            except ValueError:
                return False
        return True

    def avoid_nouns(self) -> list[str]:
        """All distinct nouns of avoid specs (frontier-affinity bias, IF-F5).

        Exposed so exploration biases toward avoid anchors too — an ungrounded avoid
        anchor is as much a reason to keep exploring as an ungrounded leg anchor.
        """
        out: list[str] = []
        if self.plan is None:
            return out
        for spec in self.plan.avoid:
            anchors: list[Anchor] = []
            if spec.between is not None:
                anchors.extend(spec.between)
            if spec.near is not None:
                anchors.append(spec.near)
            for a in anchors:
                if a.noun and a.noun not in out:
                    out.append(a.noun)
        return out

    # ------------------------------------------------------------------ replanning
    def _can_replan(self) -> bool:
        return self._replans < MAX_REPLANS_PER_QUESTION

    def _replan(self, reason: str, scene) -> bool:
        """Rebuild the costmap from the current grid snapshot, re-stamp avoids, and
        re-plan the committed prefix from the CURRENT pose (H11 / IF-F8 / SYS-F10).

        Bounded by ``MAX_REPLANS_PER_QUESTION``; each fire is recorded to
        ``_replan_events`` (flight-recorder-visible) with its reason. Returns True iff a
        re-plan was actually performed.
        """
        if not self._can_replan() or self._follower is None:
            return False
        self._replans += 1
        self._replan_events.append(
            f"replan #{self._replans} @pose={self._pose} reason={reason}"
        )
        _LOG.info(
            "IF replan #%d (reason=%s): rebuilding costmap + re-planning from pose %s.",
            self._replans,
            reason,
            self._pose,
        )
        prefix_len = max(self._driven_prefix, 1)
        self._follower = None
        self._driven_prefix = 0
        self._in_capsule = False  # recomputed against the new plan next tick
        self._build_route(self._pose, scene, prefix_len)
        return self._follower is not None

    def _capsule_tripwire(self, pose: tuple[float, float], scene) -> None:
        """H11 / IF-F5 runtime tripwire: if the current pose (or next crumb) is entering a
        stamped avoid capsule, re-plan away.

        Edge-triggered on ENTRY (clear -> violated) so a legitimately-engulfed start — the
        vehicle begins inside the capsule by construction and the recovery path already
        drives it out — does not force a re-plan every tick. Counts against the replan cap.
        """
        if self._follower is None or not self._stamped_capsules:
            return
        probe = [list(pose)]
        nxt = self._follower.current(pose)
        if nxt is not None:
            probe.append([float(nxt.x), float(nxt.y)])
        traj = np.asarray(probe, dtype=float)
        violated = any(TB.capsule_violated(traj, cap)[0] for cap in self._stamped_capsules)
        if violated and not self._in_capsule:
            self._in_capsule = True
            self._replan("capsule_tripwire", scene)
        elif not violated:
            self._in_capsule = False

    # ------------------------------------------------------------------ drive
    def _drive(self, io: RobotIO, pose: tuple[float, float], t: float) -> bool:
        """Emit the next crumb (or hold at the terminal). Returns True iff a waypoint
        was published this tick — the emit-signal ExploreHead reads to decide whether to
        fall through to frontier exploration (H3).

        H11 (IF-F8/SYS-F10): consume the follower's ``replan_flag`` — on a stall or a
        no-LOS crumb we re-plan from the current pose (bounded) rather than republish a
        wedged crumb forever. A follower that exhausts its path while the vehicle has NOT
        reached the terminal is also a re-plan trigger.

        H11 (IF-F4): the answer IS the drive. While the route is unfinished we keep
        emitting breadcrumbs; the raw terminal coordinate is published only once the
        follower is exhausted AND the vehicle is within reach — never a distant beeline
        that abandons ordered/avoid compliance for the trajectory tail.
        """
        if self._follower is None:
            return False
        wp = self._follower.advance(pose, t)
        self._mark_arrivals(pose)

        # A CP3 demote inside _mark_arrivals can invalidate the route (follower
        # reset to None). The crumb from the pre-demote route is stale — drop it;
        # the rebuilt route emits from the next tick (explore covers this one).
        if self._follower is None:
            return False

        # Consume the stall / no-LOS replan flag (IF-F8/SYS-F10).
        if self._follower.replan_flag and self._can_replan():
            if self._replan("stall_or_no_los", self._scene) and self._follower is not None:
                wp = self._follower.advance(pose, t)

        if wp is None:
            # Follower exhausted its path. If the vehicle actually reached the terminal,
            # hold the raw terminal coordinate (the drive is complete). If it has NOT
            # arrived, the path ran out short — re-plan from here toward the terminal
            # (IF-F8) rather than teleport a distant terminal waypoint (IF-F4).
            reached = self._terminal_xy is not None and (
                _dist(pose, self._terminal_xy) <= ARRIVAL_TOL_M
            )
            if not reached and self._can_replan():
                if self._replan("follower_exhausted_not_arrived", self._scene):
                    wp = self._follower.advance(pose, t) if self._follower else None
            if wp is None and self._terminal_xy is not None:
                wp = WaypointCmd(float(self._terminal_xy[0]), float(self._terminal_xy[1]))
        if wp is not None:
            io.publish_waypoint(wp)
            self._last_wp = wp
            return True
        return False

    def replan_events(self) -> list[str]:
        """Flight-recorder-visible log of the re-plans this question triggered (H11)."""
        return list(self._replan_events)

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
        if self.anchor_confirm is None:
            return
        if self._legacy_confirm:
            try:
                self.anchor_confirm(self.plan, i, f"leg {i} nouns={leg.nouns}")
            except Exception:
                pass
            return
        self._rich_confirm_leg(i, leg)

    def _rich_confirm_leg(self, i: int, leg: _GroundedLeg) -> None:
        """CP3 vision confirmation: on a confident mismatch, demote the anchor instance
        and re-plan this leg to the runner-up (design doc §CP3). Never blocks the drive —
        a demote with no runner-up leaves the map's belief standing.
        """
        # CONTRACT: anchor_desc must stay the BARE class noun — the CP3 seam's
        # anchor_noun (vocab-bridge same-class test) defaults to it. If this is
        # ever enriched to a fuller description, pass anchor_noun=leg.nouns[0]
        # explicitly or the bridge's synonym check silently degrades.
        anchor_desc = leg.nouns[0] if leg.nouns else "object"
        try:
            outcome = self.anchor_confirm(anchor_desc, self._project_crop(leg))
        except Exception:  # noqa: BLE001 — a dark/broken checkpoint trusts the map
            return
        action = getattr(outcome, "action", "confirm")
        if action != "demote":
            return
        # Confident mismatch: demote this anchor instance and re-plan to the runner-up.
        rec = leg.record
        runner_up = leg.runner_up
        if rec is not None:
            self._demoted.add(getattr(rec, "instance_id", -1))
        _LOG.info(
            "CP3 anchor mismatch on leg %d (%s); demoting instance %s -> runner-up %s.",
            i,
            anchor_desc,
            getattr(rec, "instance_id", None),
            getattr(runner_up, "instance_id", None),
        )
        if runner_up is not None and self._scene is not None:
            # Re-ground + re-route so the drive continues toward the corrected anchor.
            self._confirmed.discard(i)
            self._leg_progress = min(self._leg_progress, i)
            self._follower = None
            self._driven_prefix = 0
            self._ground_legs(self._scene)

    def _project_crop(self, leg: _GroundedLeg):
        """Supply the anchor's projected image crop for CP3.

        Real projection (project the anchor centroid to the tile via
        ``tiling.map_ray_to_camera`` inverse) is integration wiring; here we hand the seam
        the anchor's InstanceRecord as a stand-in crop token so stubs can key off it.
        """
        return leg.record

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
        # No grounding attempted yet (empty scene at t=0): the plan's route still names
        # the anchors we must explore toward, so bias toward the first leg's noun (H3b).
        if not self._legs and self.plan is not None and self.plan.route:
            for leg in self.plan.route:
                if leg.anchors and leg.anchors[0].noun:
                    return leg.anchors[0].noun
        return None

    def ungrounded_nouns(self) -> list[str]:
        """All distinct nouns of not-yet-grounded legs (frontier affinity bias, H3b).

        Before any grounding attempt (empty ``_legs``) this is every route anchor noun,
        so exploration is biased from the very first tick against an empty scene.
        """
        out: list[str] = []
        if not self._legs:
            if self.plan is not None:
                for leg in self.plan.route:
                    for a in leg.anchors:
                        if a.noun and a.noun not in out:
                            out.append(a.noun)
            return out
        for leg in self._legs:
            if leg.grounded:
                continue
            for n in leg.nouns:
                if n and n not in out:
                    out.append(n)
        return out

    def drive_complete(self) -> bool:
        """True once the IF drive has nothing left to do (IF-F4 continue-drive).

        Read-only signal the FSM's DRIVE_OUT state polls to decide when to stop ticking the
        head and go DONE. The drive is complete only when the committed route covers the
        WHOLE plan (the driven prefix reaches the final leg and every leg is grounded) AND
        EITHER:

          * the terminal is known and the vehicle is within ``ARRIVAL_TOL_M`` of it — the
            full route was driven to its end; or
          * the follower has exhausted its path AND cannot re-plan any further (the replan
            cap is spent) — no more progress is possible.

        Crucially, arriving at a PARTIAL prefix's terminal is NOT completion: while a later
        leg is still ungrounded (the terminal anchor has not appeared yet), the committed
        follower only covers the grounded prefix, so its terminal is an intermediate leg
        goal — DRIVE_OUT must keep ticking (the explore fall-through hunts the missing
        anchor, extending the route when it appears), and the watchdog floor remains the
        hard backstop if the anchor never appears. Before any follower is built (nothing
        grounded yet) the drive is likewise NOT complete. This is a pure read of existing
        head state; it publishes nothing and mutates nothing.

        Note (head-property addition, IF-F4): this property is added specifically so the FSM
        can distinguish "arrived / holding the terminal" from "still streaming crumbs" —
        ``advance``/``_drive`` return ``True`` in BOTH cases (a waypoint was published), so
        the emit bool alone cannot gate DRIVE_OUT termination.
        """
        follower = self._follower
        if follower is None:
            return False
        # The committed route must cover the entire plan — the driven prefix reaches the
        # final leg and no leg remains ungrounded — or "arrival" is only at an intermediate
        # leg goal while a later anchor is still being hunted.
        n_legs = len(self.plan.route) if self.plan and self.plan.route else 0
        route_covers_plan = (
            n_legs > 0
            and self._driven_prefix >= n_legs
            and self.ungrounded_subgoals() == 0
        )
        if not route_covers_plan:
            return False
        if self._terminal_xy is not None and _dist(self._pose, self._terminal_xy) <= ARRIVAL_TOL_M:
            return True
        # Follower exhausted (no next crumb) and no replan budget left to extend it.
        if follower.current(self._pose) is None and not self._can_replan():
            return True
        return False

    def terminal_waypoint(self) -> WaypointCmd | None:
        """The WaypointCmd the FSM publishes as the IF 'answer'.

        H11 (IF-F4): the answer IS the drive. When the FSM forces answer assembly we must
        NOT abandon breadcrumb guidance for one distant terminal coordinate (that strands
        the vehicle and drops ordered/avoid compliance for the trajectory tail). While the
        route is still unfinished — the follower has a next crumb and the vehicle is not
        yet within reach of the terminal — we return that NEXT CRUMB, so the FSM's answer
        path keeps the vehicle threading the planned route. Only once the follower is
        exhausted (or the vehicle is already within reach of the terminal) do we publish
        the raw terminal coordinate.
        """
        follower = self._follower
        if (
            follower is not None
            and self._terminal_xy is not None
            and _dist(self._pose, self._terminal_xy) > ARRIVAL_TOL_M
        ):
            crumb = follower.current(self._pose)
            if crumb is not None:
                return crumb
        if self._terminal_xy is not None:
            return WaypointCmd(float(self._terminal_xy[0]), float(self._terminal_xy[1]))
        return self._last_wp


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


_SUPERLATIVE_PREDS = frozenset({Pred.CLOSEST_TO, Pred.FARTHEST_FROM})


def _has_superlative(anchor: Anchor) -> bool:
    """True if the anchor carries a superlative disambiguator (closest_to/farthest_from).

    The toolbox already ranks those survivors by the superlative metric, so the
    nearest-to-previous-leg salience tie-break must not override it.
    """
    disamb = anchor.disambiguator
    return disamb is not None and disamb.pred in _SUPERLATIVE_PREDS


def _vehicle_z(io: RobotIO) -> float:
    """Latest vehicle z (map frame) for the overhead local-ground fallback; 0.0 if none."""
    odom = io.latest_odom()
    return float(odom.z) if odom is not None else 0.0


def _is_legacy_anchor_seam(fn: Callable) -> bool:
    """True if ``fn`` matches the legacy ``(plan, leg_index, summary) -> bool`` seam.

    Heuristic (design: "inspect.signature"): exactly 3 positional-capable params and none
    named ``crop`` / ``anchor_desc`` (the rich vision seam's keywords). Unintrospectable
    or ``*args`` callables are treated as rich (safer to pass the vision contract).
    """
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    params = list(sig.parameters.values())
    if any(p.kind is inspect.Parameter.VAR_POSITIONAL for p in params):
        return False
    if any(p.name in ("crop", "anchor_desc") for p in params):
        return False
    positional = [
        p for p in params
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    return len(positional) == 3
