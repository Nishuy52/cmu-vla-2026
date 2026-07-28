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
import json
import logging
import math
import os
import time
from dataclasses import dataclass, field

import numpy as np

from core.interfaces import RobotIO, WaypointCmd
from core.geometry import toolbox as TB
from core.geometry.toolbox import DEFAULT_THRESHOLDS, Thresholds
from core.groundtruth.arrival import NOMINAL_ARRIVAL_TOL_M
from core.nav.breadcrumbs import BreadcrumbFollower
from core.nav.costmap import VEHICLE_RADIUS_M, Costmap
from core.nav.occupancy import OccupancyGrid, integrate_scan_overhead_decimated
from core.nav import planner as _planner
from core.nav.planner import astar, free_space_via_point, plan_through
from core.plan_schema import Anchor, LegKind, Plan, Pred, RouteLeg, TargetSpec

# Flight-recorder-visible seam: recovery events are logged here so a violation-free
# least-bad choice is auditable (architecture §1 row 4 — "never a silent geometry edit").
_LOG = logging.getLogger("core.heads.instruction")

VIA_NEAR_OFFSET_M: float = 1.2  # fallback "near" offset if free-space placement fails

#: Within this of a leg goal -> arrived (mark progress).
#:
#: DERIVED, not asserted (issue #70): equals
#: :data:`core.groundtruth.arrival.NOMINAL_ARRIVAL_TOL_M` --
#: :func:`core.groundtruth.arrival.derived_arrival_tol_m` evaluated at that
#: module's documented NOMINAL p95 fit residual. This constant is
#: INTENTIONALLY COUPLED to :data:`core.groundtruth.scoring.LEG_ARRIVAL_TOL_M`
#: (both derive from the same function; the battery scorer additionally
#: recomputes a LIVE per-run value from its own scenes' fit residuals, which
#: this head cannot do -- see the docstring of
#: ``core.groundtruth.arrival`` for why the head freezes at a nominal residual
#: instead of a live one). Kept as a plain float constant (not re-derived at
#: import time from a live measurement) so the head stays ROS-free and
#: dependency-light: it depends only on the small, pure-Python/numpy
#: ``core.groundtruth.arrival`` module for this one constant, never on the
#: GT-loading/scoring machinery itself.
ARRIVAL_TOL_M: float = NOMINAL_ARRIVAL_TOL_M
#: Standoff clearance (m) beyond a "near" anchor's footprint edge for the VIA_NEAR via
#: placement. A "near" via should sit just clear of the object (about a vehicle radius),
#: NOT a full 1.2 m away: the rubric credits a leg only when the driven pose comes within
#: ARRIVAL_TOL_M of the anchor centroid, so a via placed footprint+1.2 m out overshoots
#: the tolerance for every compact anchor and the ordered leg scores 0 (T11: the dominant
#: VIA_NEAR "legs=0/N" rows). Kept below ARRIVAL_TOL_M so a compact anchor's via lands
#: inside the credited band while still clearing the footprint.
VIA_NEAR_CLEARANCE_M: float = 0.45
#: Issue #126 — margin (m) the terminal-leg standoff (``InstructionHead._standoff_push``)
#: keeps between the pushed goal and ARRIVAL_TOL_M. Landing exactly ON the tolerance
#: boundary is fragile (float/grid-quantization noise can tip a "just reached" leg over
#: to "just missed"), so the push is capped at ARRIVAL_TOL_M minus this margin, not at
#: ARRIVAL_TOL_M itself.
STANDOFF_ARRIVAL_MARGIN_M: float = 0.15
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

# --------------------------------------------------------------------------- instrumentation
#
# Issue #98: when an anchor's relational disambiguator can't be evaluated (its class was
# never perceived), the resolve() fallback ladder silently drops it and the head falls back
# to a same-label salience tie-break (see the reorder block in _ranked_anchor below) -- a
# real score and an arbitrary tie-break over N same-label candidates are indistinguishable
# in every other artifact. This dump makes the distinction visible: per resolved leg, which
# relaxation rung(s) the resolve audit recorded and the size of the candidate pool the head
# fell back over. Opt-in only, same contract as core.perception.scene_index.dump_instance_index
# (issues #84/#89): unset env var -> zero I/O, no behaviour change; any failure swallowed.

#: Path to append JSONL leg-relaxation records to. Unset (default) -> no-op.
ENV_LEG_RELAX_DUMP_PATH: str = "VLA_LEG_RELAX_DUMP_PATH"


@dataclass(frozen=True)
class AnchorAudit:
    """Diagnostic-only trail for one anchor resolve, additive to the scored path.

    steps: the resolve() fallback rungs taken (empty == resolved with no relaxation).
    candidate_count: len(res.candidates_ranked) -- the pool size the anchor was ranked
                      (and, on a relaxed resolve, fell back) over.
    tie_break_group_size: size of the same-label group actually re-sorted by the
                           salience tie-break in _ranked_anchor, or None when that
                           reorder never ran (no tie, or a distinguishing superlative).
    """

    steps: tuple[str, ...] = ()
    candidate_count: int = 0
    tie_break_group_size: int | None = None


def dump_leg_relaxations(plan: "Plan | None", legs: list["_GroundedLeg"]) -> None:
    """Append one JSONL record of each leg's anchor relaxation audit, if
    :data:`ENV_LEG_RELAX_DUMP_PATH` is set. No-op (no I/O at all) when unset.

    Issue #98: makes a leg grounded only via a dropped disambiguator + same-label
    tie-break visible as such, instead of looking identical to a genuinely grounded
    leg in every other artifact.
    """
    path = os.environ.get(ENV_LEG_RELAX_DUMP_PATH)
    if not path:
        return
    try:
        record = {
            "wall_time": time.time(),
            "question_raw": plan.question_raw if plan is not None else None,
            "legs": [
                {
                    "index": i,
                    "kind": leg.kind.value,
                    "nouns": list(leg.nouns),
                    "grounded": leg.grounded,
                    "provisional": leg.provisional,
                    "anchors": [
                        {
                            "relax_steps": list(a.steps),
                            "candidate_count": a.candidate_count,
                            "tie_break_group_size": a.tie_break_group_size,
                        }
                        for a in leg.anchor_audits
                    ],
                }
                for i, leg in enumerate(legs)
            ],
        }
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "a") as fh:
            fh.write(json.dumps(record) + "\n")
    except Exception:  # noqa: BLE001 - diagnostics must never break the run
        pass


# Pre-grounding-movement plan, Stage 3 (docs/proposals/pre_grounding_movement_plan.md
# §1(3), decision 3) — goal-credibility withhold gate. A GOTO/VIA_NEAR leg's resolved
# goal (``_GroundedLeg.goal_clamp_m``) can sit far from its anchor's actual centroid
# when the anchor lives in a pocket the reachable-mask BFS has to clamp away from (an
# unreachable/pinched anchor cell snaps to the nearest reachable cell instead, see
# ``_goto_point``) or, for VIA_NEAR, when no compliant "near" placement is reachable.
# Driving to a goal that is farther than ARRIVAL_TOL_M from the anchor centroid can
# never earn scoring credit for that leg (the rubric credits arrival only within
# ARRIVAL_TOL_M of the centroid — the same rationale documented above for
# VIA_NEAR_CLEARANCE_M). ARRIVAL_TOL_M itself is therefore the credibility bar:
# parameter-free (no new tunable), derived from the existing constant, exactly as
# decision 3 calls for. A clamp within tolerance is exactly as good as landing on the
# anchor for scoring purposes; a clamp beyond it cannot score, so committing it isn't
# worth stopping exploration for.
GOAL_CLAMP_CREDIBILITY_BAR_M: float = ARRIVAL_TOL_M

# Stage 3 — cap on re-ground "improvement" rebuilds per question (see
# ``_maybe_reground_rebuild``), separate from MAX_REPLANS_PER_QUESTION so improvement
# rebuilds triggered by better geometry can never starve a stall/no-LOS/capsule replan
# of its own budget.
MAX_REGROUND_REBUILDS: int = 2


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
    #: Stage 3 (pre-grounding movement plan §1(3)) — straight-line distance from this
    #: leg's resolved goal (``geom``) to its primary anchor's centroid, recorded by
    #: ``_goto_point``/``_via_point`` via ``_ground_one``. ``None`` for CORRIDOR_BETWEEN
    #: legs (no single goal point to clamp) and for any leg whose geometry never
    #: resolved. Feeds Gate 3 in ``_committable_prefix_len`` (withhold an uncreditable
    #: clamp) and the re-ground improvement trigger in ``_maybe_reground_rebuild``.
    goal_clamp_m: float | None = None
    #: Issue #98 — per-anchor resolve() relaxation audit (diagnostic-only; read by
    #: dump_leg_relaxations, never by any scoring/geometry logic). One entry per
    #: this leg's anchors, in anchor order.
    anchor_audits: tuple["AnchorAudit", ...] = ()


@dataclass
class InstructionHead:
    """Plans and drives an instruction-following route; steps once per tick."""

    plan: Plan | None = None
    thresholds: Thresholds = DEFAULT_THRESHOLDS
    #: Calibration seams for ``nav.unknown_cost_mult`` / ``nav.pinch_disc_m`` /
    #: ``nav.pinch_corridor_half_w_m`` (planner.py module constants). Defaults
    #: reproduce today's planner behaviour; a sweep overrides these on construction.
    unknown_cost_mult: float = _planner.UNKNOWN_COST_MULT
    pinch_disc_m: float = _planner.PINCH_DISC_M
    pinch_corridor_half_w_m: float = _planner.PINCH_CORRIDOR_HALF_W_M
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
    #: Stage 3 — ``goal_clamp_m`` per committed leg index, snapshotted at the moment its
    #: route was (re)built (``_snapshot_committed_clamp``). Compared against the current
    #: ``self._legs`` clamp on every tick to detect material improvement
    #: (``_maybe_reground_rebuild``).
    _committed_clamp_m: dict[int, float] = field(default_factory=dict)
    #: Stage 3 — count of re-ground improvement rebuilds this question, capped at
    #: MAX_REGROUND_REBUILDS (separate from ``_replans``/MAX_REPLANS_PER_QUESTION).
    _reground_rebuilds: int = 0

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
        # Stage 3 (pre-grounding movement plan §1(3)): re-grounding an already-committed
        # leg from the current pose can reveal a materially better goal than the one
        # driven on — rebuild (guarded) to adopt it before considering prefix growth.
        self._maybe_reground_rebuild(scene)

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
        prev_kind: LegKind | None = None
        prev_gate: tuple[tuple[float, float], tuple[float, float]] | None = None
        n_legs = len(self.plan.route)
        for i, leg in enumerate(self.plan.route):
            gl = self._ground_one(
                leg, scene, prev_xy, prev_kind, prev_gate, is_terminal=(i == n_legs - 1)
            )
            legs.append(gl)
            if gl.geom is not None:
                g = gl.geom
                prev_xy = g[1] if isinstance(g[0], tuple) else g
                prev_gate = g if (gl.kind is LegKind.CORRIDOR_BETWEEN and isinstance(g[0], tuple)) else None
            else:
                prev_gate = None
            prev_kind = gl.kind
        self._legs = legs
        # Issue #98: diagnostic-only, no-op unless VLA_LEG_RELAX_DUMP_PATH is set.
        dump_leg_relaxations(self.plan, legs)

    def _ground_one(
        self, leg: RouteLeg, scene, prev_xy: tuple[float, float] | None = None,
        prev_kind: LegKind | None = None,
        prev_gate: tuple[tuple[float, float], tuple[float, float]] | None = None,
        is_terminal: bool = False,
    ) -> _GroundedLeg:
        nouns = tuple(a.noun for a in leg.anchors)
        if scene is None:
            return _GroundedLeg(leg.kind, False, None, nouns)
        recs, provisional, audits = self._resolve_leg_anchors(leg, scene, prev_xy)
        if any(r is None for r in recs):
            # IF-F6: a corridor leg whose two anchors share a noun but has < 2 distinct
            # instances lands here (recs[1] is None) and stays ungrounded — which now
            # correctly feeds the H3 explore path, not a zero-width-gate recovery beeline.
            return _GroundedLeg(
                leg.kind, False, None, nouns, provisional=provisional, anchor_audits=tuple(audits)
            )
        grounded = all(r.n_obs >= MIN_GROUND_OBS for r in recs)
        if leg.kind is LegKind.CORRIDOR_BETWEEN:
            gate = TB.corridor_gate(recs[0], recs[1], scene)
            geom = (
                (float(gate.p0[0]), float(gate.p0[1])),
                (float(gate.p1[0]), float(gate.p1[1])),
            )
        elif leg.kind is LegKind.VIA_NEAR:
            geom = self._via_point(recs[0])
        else:  # GOTO
            geom = self._goto_point(
                recs[0],
                prev_gate=prev_gate if prev_kind is LegKind.CORRIDOR_BETWEEN else None,
                is_terminal=is_terminal,
            )
        runner_up = self._resolve_anchor(leg.anchors[0], scene, prev_xy)[1]
        min_n_obs = min(r.n_obs for r in recs)
        goal_clamp_m = None
        if leg.kind is not LegKind.CORRIDOR_BETWEEN:
            anchor_c = TB.P._as3(recs[0].centroid)
            goal_clamp_m = math.hypot(
                geom[0] - float(anchor_c[0]), geom[1] - float(anchor_c[1])
            )
        return _GroundedLeg(
            leg.kind, grounded, geom, nouns, record=recs[0], runner_up=runner_up,
            provisional=provisional, min_n_obs=min_n_obs, goal_clamp_m=goal_clamp_m,
            anchor_audits=tuple(audits),
        )

    def _resolve_leg_anchors(
        self, leg: RouteLeg, scene, prev_xy: tuple[float, float] | None = None
    ):
        """Resolve every anchor of a leg to a best InstanceRecord, enforcing DISTINCT
        instances across the leg's anchors (IF-F6).

        Returns ``(recs, provisional, audits)`` where ``recs`` is one record (or None)
        per anchor in order, ``provisional`` is True iff ANY anchor's resolve leaned on
        a relaxation rung (H4c), and ``audits`` is one :class:`AnchorAudit` per anchor
        (issue #98, diagnostic-only). "the two X" / "between the two X" duplicate the
        anchor (regex_tier ``_split_pair``); resolving both independently yields the SAME
        top-ranked instance -> a zero-width gate -> whole-route recovery collapse. Here
        the second anchor of a shared-noun pair takes the next distinct ranked instance;
        if fewer than 2 exist the leg stays ungrounded.
        """
        recs: list[object | None] = []
        used: set[int] = set()
        provisional = False
        audits: list[AnchorAudit] = []
        for anchor in leg.anchors:
            best, prov, audit = self._resolve_anchor_distinct(anchor, scene, used, prev_xy)
            provisional = provisional or prov
            recs.append(best)
            audits.append(audit)
            if best is not None:
                used.add(getattr(best, "instance_id", -1))
        return recs, provisional, audits

    def _resolve_anchor_distinct(
        self, anchor: Anchor, scene, used: set[int],
        prev_xy: tuple[float, float] | None = None,
    ):
        """Best ranked InstanceRecord for ``anchor`` NOT already claimed by an earlier
        anchor of the same leg (``used``); plus whether the resolve was relaxation-audited
        and its :class:`AnchorAudit` (issue #98, diagnostic-only).
        """
        ranked, provisional, audit = self._ranked_anchor(anchor, scene, prev_xy)
        for c in ranked:
            if getattr(c, "instance_id", -1) not in used:
                return c, provisional, audit
        return None, provisional, audit

    def _ranked_anchor(
        self, anchor: Anchor, scene, prev_xy: tuple[float, float] | None = None
    ):
        """(ranked survivors minus demoted, provisional?, AnchorAudit) for one anchor spec.

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
        # Issue #98: diagnostic-only audit trail, never read by any scoring/geometry
        # logic below -- tie_break_group_size is filled in only if the same-label
        # salience reorder actually runs (end of this method).
        tie_break_group_size: int | None = None

        # Issue #73 (post-#71 parity audit, home_building_1 asymmetry): once the
        # resolve fallback ladder has dropped every relation clause (category_only,
        # the noisiest rung -- issue #59), the remaining soft best-effort ranking can
        # rank a loosely/generically matched label ahead of a survivor whose label is
        # an EXACT match for the anchor noun (e.g. a bare "table" outranking an
        # actual "dining table"). ``gt_battery._if_rubric_geometry`` already carries
        # this correction (issue #71 RUBRIC-WRONG fix); port the same preference here
        # so the head converges on the same instance from its own state, independent
        # of the rubric's. Same guard as the rubric: never runs for a superlative
        # anchor (the toolbox already ranks those by a real margin).
        if not _has_superlative(anchor) and any(r.step == "category_only" for r in res.audit):
            norm = anchor.noun.strip().lower()
            exact = [c for c in ranked if c.label.strip().lower() == norm]
            if exact:
                rest = [c for c in ranked if c.label.strip().lower() != norm]
                ranked = exact + rest

        # Only re-order when the toolbox left a genuine tie (no superlative margin,
        # no clause evidence distinguishing the top survivors).
        #
        # Issue #71 (resolve-outcome parity audit): reordering the FULL survivor pool by
        # raw distance discarded whatever label discrimination the toolbox's own ranking
        # already encoded whenever survivors carried DIFFERENT labels (e.g. an exact "tv
        # cabinet" match losing to an alias-matched "sink cabinet", or an exact "crystal
        # ball decoration" losing to a "dice decoration", purely for sitting closer to the
        # previous leg) — a genuine product bug, not the intended "instance-id-only tie"
        # the docstring above promises. Restrict the reorder to the group of survivors
        # sharing the toolbox's OWN top-ranked label: a same-label group is the only case
        # where "arbitrary detection-order pick" is actually true; a different-label
        # survivor was placed where it is by real (tier/clause) evidence the salience
        # signal has no business overriding.
        # Issue #75 (post-73 parity audit, home_building_2/office_2 residual): the
        # same-label restriction above still unconditionally re-sorted the WHOLE
        # same-label group by raw distance, even when resolve()'s own ranking already
        # carried real discriminating evidence within that group -- e.g. a `between()`
        # clause the survivors matched with different soft margins, or the #73
        # category-only relaxed-relation score. Gate the reorder: only run it when the
        # group is a genuine tie on resolve()'s own evidence (every survivor's summed
        # clause score over ``res.pass_matrix`` is equal, within float tolerance) --
        # the same "no discriminating evidence" bar the toolbox's own tie-breaks
        # apply. Mirrors ``core.runner.gt_battery._if_rubric_geometry``'s
        # ``_resolve_anchor_rec`` symmetrically (the #71/#73/#75 convergence rule).
        #
        # Issue #107: two things changed here from the #75 version.
        #
        # 1. The tie is no longer gated on ``prev_xy is not None``. A first-leg (or
        #    otherwise prior-less) anchor used to skip this reorder entirely and fall
        #    straight through to resolve()'s own list order, which for an unevaluable
        #    disambiguator is instance_id-ascending -- an annotation/detection-order
        #    artifact with no relation to the world (verified: permuting instance_id
        #    while leaving geometry fixed changed the winning object). Ordering must
        #    be invariant under renumbering, so a tied group is now ALWAYS re-ordered
        #    by a declared, world-grounded prior -- route continuity when there is a
        #    previous leg to anchor on, declared salience when there is not (see
        #    ``_declared_salience_key``) -- never left to fall through to list order.
        # 2. The sort key's final tie-break used to end in ``instance_id`` -- exactly
        #    the same renumbering-sensitive artifact this reorder exists to escape,
        #    just demoted to last place instead of first. It is replaced with
        #    ``_quantized_position``: a property of the *world* (the candidate's own
        #    centroid, rounded to a stable precision) that is reproducible across
        #    both renumbering and repeated runs.
        if len(ranked) > 1 and not _has_superlative(anchor):
            top_label = ranked[0].label
            same = [c for c in ranked if c.label == top_label]
            if len(same) > 1 and _same_label_group_is_tied(
                same, anchor.disambiguator, scene, self.thresholds
            ):
                tie_break_group_size = len(same)  # issue #98: fell back over this many
                rest = [c for c in ranked if c.label != top_label]
                if prev_xy is not None:
                    px, py = prev_xy
                    same = sorted(
                        same,
                        key=lambda c: (
                            (float(TB.P._as3(c.centroid)[0]) - px) ** 2
                            + (float(TB.P._as3(c.centroid)[1]) - py) ** 2,
                            _declared_salience_key(c),
                            _quantized_position(c),
                        ),
                    )
                else:
                    same = sorted(
                        same,
                        key=lambda c: (_declared_salience_key(c), _quantized_position(c)),
                    )
                ranked = same + rest
        audit = AnchorAudit(
            steps=tuple(r.step for r in res.audit),
            candidate_count=len(res.candidates_ranked),
            tie_break_group_size=tie_break_group_size,
        )
        return ranked, provisional, audit

    def _resolve_anchor(
        self, anchor: Anchor, scene, prev_xy: tuple[float, float] | None = None
    ):
        """Resolve one anchor to (best, runner_up) InstanceRecords, honouring attributes.

        The runner-up is used by the CP3 demote-and-replan path. Demoted anchors (CP3
        confidently rejected on arrival) are skipped so re-resolution lands on the
        runner-up.
        """
        ranked, _, _ = self._ranked_anchor(anchor, scene, prev_xy)
        if not ranked:
            return (None, None)
        return (ranked[0], ranked[1] if len(ranked) > 1 else None)

    def _goto_point(
        self, rec,
        prev_gate: tuple[tuple[float, float], tuple[float, float]] | None = None,
        is_terminal: bool = False,
    ) -> tuple[float, float]:
        """GOTO leg goal: the anchor projection, pulled off the anchor by a standoff
        clearance when it would otherwise land too close for the stock local planner
        to accept (issue #126).

        ``_goto_point_raw`` places the goal AT (or nearest-reachable to) the anchor
        centroid, which is correct for scoring (closer to the centroid is always
        better within ``ARRIVAL_TOL_M``) but can land within the planner's obstacle
        clearance requirement for a compact/close anchor — the planner then refuses
        the final approach and issues a zero-``cmd_vel`` stop. That matters ONLY for
        the TERMINAL leg: it's the one goal the FSM republishes and the vehicle
        actually settles/holds at as the IF "answer" (``terminal_waypoint`` docstring
        -- "the answer IS the drive"), which is exactly where 15/15 live runs wedge,
        stationary for the trajectory tail. A non-terminal GOTO leg is a breadcrumb
        the follower threads through en route to the NEXT leg, never a point the
        vehicle stops and holds at, so it was never exposed to this failure mode --
        standing it off too would only add scoring-geometry drift no live run
        exhibits. Only apply the standoff when the raw goal's own clearance is below
        the planner's floor (``VIA_NEAR_CLEARANCE_M``) — an already-clear raw goal is
        left untouched, so this never moves a goal that didn't need moving.

        Push RADIALLY OUTWARD from the anchor along ``raw``'s own direction
        (``_standoff_push``), not a fresh max-clearance ring search (the mechanism
        ``_via_point`` uses for VIA_NEAR): ``_goto_point_raw`` already picked the
        reachable/approach side of the anchor, and a ring search free to land on
        ANY max-clearance cell around the anchor can jump to a different side of a
        multi-sided object, moving the goal much further from where the vehicle
        would actually arrive than the clearance fix requires. Reusing that
        already-chosen direction keeps the standoff to the minimum deviation that
        clears the floor. ``_standoff_push`` itself clamps the push distance so the
        result never leaves ``ARRIVAL_TOL_M`` of the anchor centroid -- for a large
        anchor (a bed, a sofa) the clearance floor and the arrival tolerance can
        genuinely conflict, and trading a short-of-goal wedge for an
        outside-tolerance miss is not a fix (both are disqualifying, and the
        tolerance miss is unrecoverable while the wedge at least banks partial
        credit)."""
        raw = self._goto_point_raw(rec, prev_gate)
        cm = self._costmap
        if not is_terminal or cm is None or self._clearance_m(raw) >= VIA_NEAR_CLEARANCE_M:
            return raw
        c = TB.P._as3(rec.centroid)
        anchor_xy = (float(c[0]), float(c[1]))
        return self._standoff_push(anchor_xy, raw, self._near_thresh(rec))

    def _standoff_push(
        self, anchor_xy: tuple[float, float], raw: tuple[float, float], near_thresh_m: float,
    ) -> tuple[float, float]:
        """Move ``raw`` out along the anchor->raw direction, nudged onto the nearest
        PASSABLE cell, capped so the result never leaves ``ARRIVAL_TOL_M`` (less
        ``STANDOFF_ARRIVAL_MARGIN_M``) of the anchor centroid.

        ``near_thresh_m`` (the VIA_NEAR-style footprint+clearance target) can EXCEED
        the arrival tolerance for a large anchor -- a bed, a sofa, a dining table --
        where clearing the planner's obstacle floor and staying inside the rubric's
        scored band genuinely conflict. Trading a short-of-goal wedge for an
        outside-tolerance miss is not a fix (issue #126 verification), so the push
        distance is clamped to ``max_dist`` FIRST, before anything else: the goal
        may end up under-clearing the planner's floor for a large enough anchor, but
        it never leaves the credited band. If ``raw`` itself is already at/beyond
        that bound (an anchor so large its raw goal already skirts the tolerance),
        there is no room left to push at all -- ``raw`` is returned untouched, on
        the same "don't make it worse" principle.

        Deliberately a bounded LOCAL nudge, not ``nearest_reachable_point``'s full
        connectivity BFS: if the direct radial candidate sits in a pocket the BFS
        has to route far around (a wall pinch, a disconnected alcove), that BFS
        happily returns some distant reachable cell that clears the ring test but
        can be a metre-plus from the anchor in a direction nothing to do with the
        approach -- worse than the wedge this exists to fix. So the push is
        accepted only when: the nudge onto a passable cell stayed local (within
        ``near_thresh_m`` of where we aimed), the FINAL nudged point is still within
        ``max_dist`` of the anchor (the local nudge itself could in principle push
        past the clamp even though the pre-nudge candidate didn't), and the result
        is still reachable from the current pose (``_goto_point_raw``'s own
        guarantee); anything else falls back to ``raw`` rather than risk a wild jump
        or a tolerance miss.

        Degenerate direction (``raw`` coincides with the anchor, e.g. a floor-level
        anchor with no footprint yet) falls back to ``raw`` unchanged rather than
        picking an arbitrary direction."""
        dx, dy = raw[0] - anchor_xy[0], raw[1] - anchor_xy[1]
        dist = math.hypot(dx, dy)
        if dist < 1e-6:
            return raw
        max_dist = ARRIVAL_TOL_M - STANDOFF_ARRIVAL_MARGIN_M
        if dist >= max_dist:
            return raw  # raw already at/beyond the clamp -- no room to push further
        ux, uy = dx / dist, dy / dist
        target_dist = min(max(dist, near_thresh_m), max_dist)
        candidate = (anchor_xy[0] + ux * target_dist, anchor_xy[1] + uy * target_dist)
        projected = self._project_free(candidate)
        if math.hypot(projected[0] - candidate[0], projected[1] - candidate[1]) > near_thresh_m:
            return raw
        if math.hypot(projected[0] - anchor_xy[0], projected[1] - anchor_xy[1]) > max_dist:
            return raw
        cm = self._costmap
        seen = cm.reachable_mask(self._pose)
        if seen is None:
            return raw
        r, c = self.grid.world_to_cell(*projected)
        if not (0 <= r < seen.shape[0] and 0 <= c < seen.shape[1] and seen[r, c]):
            return raw
        return projected

    def _clearance_m(self, xy: tuple[float, float]) -> float:
        """Distance (m) from ``xy`` to the nearest blocked cell, via the same
        clearance field ``free_space_via_point`` uses to rank via candidates (IF-F7).
        Returns 0.0 (i.e. "not clear") for a point outside the costmap's grid."""
        cm = self._costmap
        field = _planner._clearance_field(cm)
        r, c = self.grid.world_to_cell(*xy)
        h, w = field.shape
        if not (0 <= r < h and 0 <= c < w):
            return 0.0
        return float(field[r, c]) * self.grid.cell_m

    def _goto_point_raw(
        self, rec,
        prev_gate: tuple[tuple[float, float], tuple[float, float]] | None = None,
    ) -> tuple[float, float]:
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
        best = cm.nearest_reachable_point(anchor_xy, self._pose)
        if prev_gate is not None:
            best = self._goto_point_pinch_relax(cm, anchor_xy, prev_gate, best)
        return best

    def _goto_point_pinch_relax(self, cm, anchor_xy, gate, best):
        """When a GOTO leg immediately follows a CORRIDOR_BETWEEN leg, the plain
        reachable-mask BFS (which floods the base, non-pinch costmap) sees only the
        near-side pocket of the just-threaded gate (#78) — the same asymmetry
        ``plan_through`` already resolves for ROUTING via ``_pinch_costmap``, but
        goal SELECTION never saw. Retry ``nearest_reachable_point`` through the SAME
        gate, over the SAME relax-round schedule ``plan_through`` itself uses (#54),
        and keep whichever candidate (plain or any relaxed round) lands CLOSER to the
        anchor's own centroid — never farther, so this can only improve the resolved
        goal, matching the generalization protocol (no new free parameter: reuses
        ``plan_through``'s own ``PINCH_DISC_M``/``MAX_PINCH_RELAX_ROUNDS``/
        ``PINCH_RELAX_GROWTH``/``PINCH_CORRIDOR_HALF_W_M`` constants and
        ``_pinch_costmap`` helper verbatim). Confirmed (#79) this schedule genuinely
        does NOT help every such leg — some rooms are disconnected by real geometry
        beyond the gate — so this is a best-effort widen, not a guaranteed fix.

        ``cm`` can be transiently stale relative to ``self.grid`` mid-tick (built
        before this tick's terrain ingestion grew the live grid; ``_build_route``
        reconciles this later in the same tick) — a pre-existing property never
        exercised before because ``reachable_mask``'s BFS only ever touches cells it
        actually walks to. ``_pinch_costmap`` vectorizes over the FULL grid shape
        unconditionally, so a stale ``cm`` would raise on the shape mismatch; skip
        the relax rather than risk it (falls back to the plain BFS result, same as
        before this leg had a corridor predecessor)."""
        if cm.grid.shape != cm.capsule_blocked.shape:
            return best
        # Guard (found via #77c integration testing): naively taking whichever
        # relaxed candidate lands closest to the anchor can move THIS leg's own
        # goal to a point plan_through's `_gate_crossing_extension` can no longer
        # confirm reachable for the gate leg immediately before it — trading that
        # corridor leg's own (already-earned) threading/credit for a goal move that
        # doesn't even flip THIS leg (still short of its own tolerance either way).
        # Re-run the SAME lookahead the corridor leg's own extension will use
        # (`_gate_extension_keeps_route_planned`, already shared across both call
        # sites in planner.py) against the gate's own usable point as a stand-in
        # `leg_start`/extension origin — reject a candidate that would fail it,
        # keeping the previous (narrower but non-regressing) result instead.
        g0, g1 = gate
        raw_mid = ((g0[0] + g1[0]) / 2.0, (g0[1] + g1[1]) / 2.0)
        seg_end = _planner.usable_gate_point(
            np.asarray(g0, dtype=float), np.asarray(g1, dtype=float),
            np.asarray(raw_mid, dtype=float),
            lambda pt: _planner._raw_obstacle_blocked_xy(cm, pt),
        )
        seg_end_xy = (float(seg_end[0]), float(seg_end[1]))

        def _keeps_corridor_threaded(candidate: tuple[float, float]) -> bool:
            ext = _planner._gate_crossing_extension(
                cm, self._pose, gate, seg_end_xy, ("goto", candidate),
                unknown_cost_mult=self.unknown_cost_mult,
                pinch_disc_m=self.pinch_disc_m,
                pinch_corridor_half_w_m=self.pinch_corridor_half_w_m,
            )
            return ext is not None

        best_d = math.hypot(best[0] - anchor_xy[0], best[1] - anchor_xy[1])
        relax_disc_m = _planner.PINCH_DISC_M
        for _round in range(_planner.MAX_PINCH_RELAX_ROUNDS):
            pinch = _planner._pinch_costmap(
                cm, gate, pinch_disc_m=relax_disc_m,
                pinch_corridor_half_w_m=_planner.PINCH_CORRIDOR_HALF_W_M,
                start_xy=self._pose,
            )
            cand = pinch.nearest_reachable_point(anchor_xy, self._pose)
            d = math.hypot(cand[0] - anchor_xy[0], cand[1] - anchor_xy[1])
            if d < best_d and _keeps_corridor_threaded(cand):
                best, best_d = cand, d
            relax_disc_m *= _planner.PINCH_RELAX_GROWTH
        return best

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

        Half the anchor's footprint diagonal (so the via clears the object's shell) plus
        the costmap's own inflation radius (so the push target is measured from the same
        blocked boundary ``_clearance_field`` measures against, not the raw un-inflated
        footprint) plus a vehicle-radius clearance — NOT a fixed 1.2 m standoff, which
        overshoots the rubric's arrival band for every compact anchor (T11).

        Issue #132: ``Costmap.base_blocked`` (what ``_clearance_field`` measures distance
        to) is the anchor's raw footprint already dilated by ``Costmap.vehicle_radius_m``
        (``Costmap._inflate``) — the blocked boundary sits ``vehicle_radius_m`` FARTHER
        from the centroid than ``half_diag`` alone. Omitting that inflation radius here
        meant a push aimed at "``VIA_NEAR_CLEARANCE_M`` past the boundary" only ever
        travelled "``VIA_NEAR_CLEARANCE_M`` past the raw footprint edge", landing short of
        the actual blocked boundary by one inflation radius — reproduced on the real
        terminal-leg anchor footprints (arabic jar, trash can, potted plant, mirror):
        goal clearance measured 0.2-0.3 m against a 0.45 m target. Falls back to
        ``VEHICLE_RADIUS_M`` (the costmap default) when no costmap exists yet (pre-route
        grounding pass), so the threshold never depends on ungrounded state.

        For a compact anchor (half-diag small) this lands the via within
        ``ARRIVAL_TOL_M`` of the centroid, exactly where "near" is credited; for a large
        anchor the half-diag term dominates and the via sits just outside its shell (no
        in-footprint via), the best reachable "near" the geometry allows.
        """
        try:
            ext = rec.aabb_max - rec.aabb_min
            half_diag = 0.5 * float((float(ext[0]) ** 2 + float(ext[1]) ** 2) ** 0.5)
        except Exception:  # noqa: BLE001 — a malformed rec falls back to the fixed offset
            half_diag = 0.0
        cm = self._costmap
        inflation_m = cm.vehicle_radius_m if cm is not None else VEHICLE_RADIUS_M
        return half_diag + inflation_m + VIA_NEAR_CLEARANCE_M

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

        Three independent withhold gates apply, in order:

        1. Issue #33 — single-observation floor: the prefix is first truncated at the
           earliest leg backed by fewer than MIN_COMMIT_OBS observations (n_obs == 1
           legs are PLANNED — geometry already computed by ``_ground_legs`` — but not
           COMMITTED), unless the T-90 forced-assembly gate has been reached.
        2. Stage 3 (pre-grounding movement plan §1(3)) — goal-credibility floor: the
           obs-gated prefix is further truncated at the earliest GOTO/VIA_NEAR leg whose
           resolved goal was clamped more than GOAL_CLAMP_CREDIBILITY_BAR_M away from its
           anchor centroid (an uncreditable clamp), unless budget/forced-assembly
           pressure has already forced the commit — the identical override pattern as
           H4c/#33 (see ``_clamp_gated_prefix_len``).
        3. H4c — provisional terminal: if the (gated) prefix reaches the FINAL route
           leg and that terminal was resolved via a relaxation rung, hold it back (drive
           only the legs before it) until budget pressure forces the commit, so
           exploration can still find the missing disambiguator/anchor. Non-terminal
           provisional legs are NOT withheld — partial credit on ordered early legs is
           banked regardless.
        """
        n = self._obs_gated_prefix_len(self._grounded_prefix_len())
        n = self._clamp_gated_prefix_len(n)
        if n == 0 or n < len(self._legs):
            return n  # terminal not yet in the prefix; nothing to withhold
        terminal = self._legs[-1]
        if terminal.provisional and not self._commit_forced():
            return n - 1  # withhold the provisional terminal; drive the rest
        return n

    def _clamp_gated_prefix_len(self, n: int) -> int:
        """Stage 3 Gate 3: truncate a geometry/obs-gated prefix of length ``n`` at the
        first GOTO/VIA_NEAR leg whose ``goal_clamp_m`` exceeds
        GOAL_CLAMP_CREDIBILITY_BAR_M — driving to that clamped goal can never earn
        scoring credit, so it is PLANNED (geometry stays in ``self._legs``, feeding
        explore affinity) but withheld from the COMMITTED prefix.

        Uses the SAME override pair as H4c/#33 — ``_commit_forced()`` (budget pressure;
        defaults True with no ``budget_frac`` hook) or ``_forced_assembly_reached()``
        (T-90 time pressure; defaults False with no ``forced_assembly`` hook) — so with
        both hooks unconfigured (``None``), ``_commit_forced()`` is True and this gate
        never withholds: all current behaviour is byte-preserved by default.
        CORRIDOR_BETWEEN legs have no single goal point (``goal_clamp_m`` stays
        ``None``) and are never gated here.
        """
        if self._commit_forced() or self._forced_assembly_reached():
            return n
        for i in range(n):
            clamp = self._legs[i].goal_clamp_m
            if clamp is not None and clamp > GOAL_CLAMP_CREDIBILITY_BAR_M:
                return i
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
        if self._follower is None and self._costmap is None:
            # Stage 3 Gate 3 (goal-credibility withhold) needs a reachable-mask costmap
            # to know a leg's true ``goal_clamp_m`` — before any route has ever been
            # built there is none yet, so the grounding pass ``advance`` already ran
            # this tick used the pre-costmap ``_project_free`` fallback instead of the
            # real BFS-reachable clamp. Probe one now so the very first commit decision
            # sees the SAME geometry the driven route itself will use. Idempotent
            # (mirrors ``_stamp_avoids``) — ``_build_route``/``_stamp_ground_plan``
            # redoes this exact stamp+ground once more before actually adopting a
            # route, so this changes no final geometry, only what Gate 3 sees.
            self._refresh_costmap_and_geometry(scene)
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

    def _refresh_costmap_and_geometry(self, scene) -> None:
        """Rebuild the costmap from the current grid, re-stamp avoids, and re-ground
        every leg against it. ``_stamp_avoids`` is idempotent over specs (re-stamps any
        avoid anchor that has since grounded); re-grounding is likewise safe to run
        speculatively. Shared by ``_stamp_ground_plan`` (the actual build/rebuild path)
        and ``_maybe_build_or_extend_route``'s Gate 3 probe (see its docstring)."""
        self._costmap = Costmap(self.grid)
        self._stamp_avoids(scene)
        self._ground_legs(scene)

    def _stamp_ground_plan(
        self, start_xy: tuple[float, float], scene, prefix_len: int
    ) -> tuple[list[tuple[float, float]], list[int]] | None:
        """Stamp avoids fresh, re-ground (geometry may need re-projection now the
        costmap exists), and A*-plan the leading ``prefix_len`` ordered legs.

        Returns ``(path, leg_bounds)``, or ``None`` if the prefix lacks geometry or
        ``plan_through`` cannot reach it. This always mutates ``self._costmap``/
        ``self._legs``, but never touches ``self._follower``/``self._driven_prefix``:
        callers decide whether/how to adopt the routing result (``_build_route`` always
        adopts, falling back to ``_recover_path`` on ``None``; the Stage 3 guarded
        re-ground rebuild adopts ONLY on success).
        """
        self._refresh_costmap_and_geometry(scene)
        prefix = self._legs[:prefix_len]
        if any(l.geom is None for l in prefix):
            return None
        legs = [self._leg_tuple(l) for l in prefix]
        path, leg_bounds = plan_through(
            self._costmap,
            start_xy,
            legs,
            unknown_cost_mult=self.unknown_cost_mult,
            pinch_disc_m=self.pinch_disc_m,
            pinch_corridor_half_w_m=self.pinch_corridor_half_w_m,
            record_leg_bounds=True,
        )
        if path is None:
            return None
        return path, leg_bounds

    def _build_route(self, start_xy: tuple[float, float], scene, prefix_len: int) -> None:
        """Stamp avoids ONCE, plan the leading ``prefix_len`` ordered legs, wrap in a
        BreadcrumbFollower.

        Plans only the grounded prefix (partial-route drive, H3c); the route is rebuilt
        with a longer prefix as later legs ground.
        """
        prefix = self._legs[:prefix_len]
        if not prefix or any(l.geom is None for l in prefix):
            return
        result = self._stamp_ground_plan(start_xy, scene, prefix_len)
        if result is not None:
            path, leg_bounds = result
        else:
            # Unreachable with the hard capsules in place: drive to the nearest legal
            # point to the terminal goal and answer from there (architecture row 4).
            # The recovery path MUST be planned through the costmap — never a raw
            # straight segment, which could cut through a hard capsule (the very
            # violation the capsule exists to prevent).
            prefix = self._legs[:prefix_len]
            if any(l.geom is None for l in prefix):
                return
            path = self._recover_path(start_xy, prefix)
            # Issue #74: the recovery path collapses the ordered legs into a single
            # best-effort segment to the nearest legal point — it no longer threads
            # each leg's own goal, so there is nothing for leg-boundary hard-stops to
            # protect (and capping crumb selection at a leg-goal index that isn't
            # actually ON this path would just wedge the follower for no benefit).
            leg_bounds = []
        self._terminal_xy = path[-1]
        self._follower = BreadcrumbFollower(
            path=path, costmap=self._costmap, leg_goal_indices=leg_bounds
        )
        self._driven_prefix = prefix_len
        self._snapshot_committed_clamp(prefix_len)

    def _snapshot_committed_clamp(self, prefix_len: int) -> None:
        """Stage 3 (§1(3)) — record ``goal_clamp_m`` for each committed leg at the
        moment its route is (re)built, so a later re-ground can be compared against the
        geometry that was actually driven on to detect material improvement (see
        ``_maybe_reground_rebuild``)."""
        self._committed_clamp_m = {
            i: self._legs[i].goal_clamp_m
            for i in range(prefix_len)
            if self._legs[i].goal_clamp_m is not None
        }

    def _try_reground_rebuild(self, start_xy: tuple[float, float], scene, prefix_len: int) -> bool:
        """Guarded rebuild for the Stage 3 re-ground improvement trigger: attempt
        ``_stamp_ground_plan`` and adopt the new follower ONLY if ``plan_through``
        succeeds. Never falls back to ``_recover_path`` — swapping an already-working,
        already-threaded route for a least-bad beeline over an improvement signal would
        cost more (earned threading, #77c) than the improvement is worth. Returns True
        iff the rebuild was adopted."""
        result = self._stamp_ground_plan(start_xy, scene, prefix_len)
        if result is None:
            return False
        path, leg_bounds = result
        self._terminal_xy = path[-1]
        self._follower = BreadcrumbFollower(
            path=path, costmap=self._costmap, leg_goal_indices=leg_bounds
        )
        self._driven_prefix = prefix_len
        self._snapshot_committed_clamp(prefix_len)
        return True

    def _maybe_reground_rebuild(self, scene) -> None:
        """Stage 3 (pre-grounding movement plan §1(3)) — while driving, re-grounding an
        already-committed leg from the CURRENT (moved) pose can materially improve a
        previously-clamped goal (more of the map is reachable/known now than when the
        route was built). Detect that and rebuild the committed route to adopt it.

        Eligible legs: GOTO/VIA_NEAR (``goal_clamp_m`` applies; corridor legs are
        excluded — no single goal point and no clamp), index >= ``_leg_progress`` (never
        touch a leg the drive has already passed), and not in ``_confirmed`` (never
        rewrite an arrived leg, mirrors ``_mark_arrivals``). "Material" mirrors Gate 3:
        improvement must exceed GOAL_CLAMP_CREDIBILITY_BAR_M (parameter-free, same bar).
        Bounded by MAX_REGROUND_REBUILDS (separate from MAX_REPLANS_PER_QUESTION, so
        stall/no-LOS/capsule replans are never starved by improvement rebuilds). Adoption
        is guarded (``_try_reground_rebuild``): never swaps into ``_recover_path``.
        Follower progress resets exactly like ``_replan`` (a fresh ``BreadcrumbFollower``
        over the rebuilt path).
        """
        if self._follower is None or self._reground_rebuilds >= MAX_REGROUND_REBUILDS:
            return
        prefix_len = self._driven_prefix
        if prefix_len == 0:
            return
        trigger_i = None
        for i in range(self._leg_progress, prefix_len):
            if i in self._confirmed:
                continue
            leg = self._legs[i]
            if leg.kind not in (LegKind.GOTO, LegKind.VIA_NEAR):
                continue
            if leg.geom is None or leg.goal_clamp_m is None:
                continue
            old = self._committed_clamp_m.get(i)
            if old is None:
                continue
            if old - leg.goal_clamp_m > GOAL_CLAMP_CREDIBILITY_BAR_M:
                trigger_i = i
                break
        if trigger_i is None:
            return
        self._reground_rebuilds += 1
        self._replan_events.append(
            f"reground_rebuild #{self._reground_rebuilds} @pose={self._pose} leg={trigger_i}"
        )
        _LOG.info(
            "IF reground rebuild #%d: leg %d goal_clamp_m improved; re-planning from "
            "pose %s.",
            self._reground_rebuilds,
            trigger_i,
            self._pose,
        )
        if self._try_reground_rebuild(self._pose, scene, prefix_len):
            self._in_capsule = False  # recomputed against the new plan next tick
        # else: the improvement isn't reachable through the hard capsules — keep driving
        # the existing (already-working) follower unchanged; never strand on a failed
        # guarded rebuild.

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
        path = astar(
            self._costmap, start_xy, legal, unknown_cost_mult=self.unknown_cost_mult
        )
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


_SALIENCE_TIE_EPS = 1e-9  # float-equality tolerance for the issue #75 clause-score tie check

#: Issue #107 -- decimal places for the quantised-position final tie-break.
#: VLA-3D geometry is metres; 3 dp is millimetre precision, well below sensor/
#: annotation noise, so two DISTINCT physical objects essentially never collide
#: here while float jitter within the "same" object always rounds away.
_POS_QUANT_PREC = 3


def _quantized_position(c) -> tuple[float, float]:
    """Deterministic final tie-break key (issue #107): the candidate's own (x, y)
    centroid, quantised to :data:`_POS_QUANT_PREC` decimal places.

    This replaces ``instance_id`` as the last-resort tie-break in the same-label
    salience reorder. ``instance_id`` is a detection/annotation-order artifact --
    renumbering a scene's instances (permuting IDs, geometry unchanged) changes it
    for a candidate without changing anything about the candidate itself, so using
    it as a sort key makes the final pick depend on numbering rather than on the
    world. Quantised position is a property of the object: it is identical for a
    given physical instance under any renumbering and identical across repeated
    runs (unlike raw float centroids, which can jitter at the ULP level between
    otherwise-identical resolves).
    """
    x = float(TB.P._as3(c.centroid)[0])
    y = float(TB.P._as3(c.centroid)[1])
    return (round(x, _POS_QUANT_PREC), round(y, _POS_QUANT_PREC))


def _declared_salience_key(c) -> tuple[int, float, float]:
    """Sort key for the same-label salience reorder when there is no previous leg
    to anchor route-continuity on (issue #107, first leg / standalone reference).

    Ascending on this tuple ranks the most trustworthy/prominent candidate first.
    Every component is a property of how the object was actually perceived or how
    large it physically is -- never list position, never ``instance_id``:

      1. ``-n_obs``  -- candidates seen in more distinct keyframes are more
                        reliably localized (the same signal ``MIN_COMMIT_OBS``
                        already uses elsewhere to gate confident commits).
      2. ``-score``  -- higher max detector confidence.
      3. ``-volume`` -- larger physical footprint (AABB extents product) is a
                        more prominent, more likely-intended reference object
                        when nothing else distinguishes the group.

    Chosen over inheriting resolve()'s list order (which is what silently fell
    through to instance_id order before this fix) precisely because these three
    are grounded in the perceived world and are therefore invariant under
    instance_id renumbering, unlike list/detection order.
    """
    ext = c.extents
    volume = float(ext[0]) * float(ext[1]) * float(ext[2])
    return (-int(c.n_obs), -float(c.score), -volume)


def _same_label_group_is_tied(same, clause, idx, th) -> bool:
    """True iff ``same`` (a same-labeled survivor group from ``resolve()``'s own
    ranking) carries no discriminating evidence from the anchor's own disambiguator
    clause -- every member's ``PredResult.score`` AND ``PredResult.margin`` for that
    clause are equal within float tolerance (or the anchor carries no disambiguator
    at all, which is the pre-#75 genuine-tie case). Margin is checked alongside
    score because a hard clause gate (e.g. ``on()``'s upper-z-band FAIL) can
    quantise every survivor's soft ``score`` to the same 0.0 while ``margin``
    (the continuous slack ``PredResult`` carries specifically "for ranking/audit")
    still separates a near-miss from a clear miss -- office_2's folder/cabinet leg
    (issue #75) is exactly this: every survivor's ``on()`` score is 0.0, but the
    margin cleanly splits the group the toolbox's own soft ranking would have
    preferred from the one raw distance-to-previous-leg alone would wrongly favour.

    Mirrors ``core.runner.gt_battery._same_label_group_is_tied`` symmetrically
    (issue #75, the #71/#73 convergence rule). ``clause`` is the SAME clause
    ``resolve()`` itself would score for this survivor pool, whether resolve
    returned it via a hard pass, the issue #59 ``_relaxed_relation_order`` soft
    fallback (category_only rung -- whose score ``ResolveResult.pass_matrix`` does
    NOT carry, since ``hard_clauses`` is emptied before the pass matrix is built),
    or a plain tier tie-break. Re-evaluating it here independently (never
    re-deriving resolve()'s filter/relaxation ladder -- just probing its scoring
    primitive, same pattern as the exact-label correction above) is the only way
    to see that evidence from outside ``resolve()``. A same-label group is
    already tier-tied by construction (same label), so this clause-score check is
    the remaining discriminator.

    Issue #107 -- non-finite margins, explicitly: ``_eval_clause`` returns
    ``margin=float("-inf")`` when the clause's own anchor class was never
    perceived (``"anchor not found"``). Comparing two such margins with
    ``abs(a - b) <= eps`` computes ``abs(-inf - -inf) == nan``, and ``nan`` compares
    False against everything -- so the old code silently landed on "not tied" by
    accident whenever the whole group shared that sentinel, never on purpose. That
    is backwards: a clause nobody in the group could evaluate is exactly ZERO
    discriminating evidence, i.e. the group is MORE tied than a normal float-equal
    tie, not less. This is handled as its own explicit branch below rather than
    relying on IEEE-754 arithmetic to fall out the right way; a future refactor
    that clamps the sentinel to a finite value (removing the ``nan``) must not be
    able to silently flip this verdict, because the branch no longer depends on
    what the arithmetic happens to do with it.
    """
    if clause is None:
        return True
    results = [TB._eval_clause(c, clause, idx, th) for c in same]
    if not results:
        return True
    s0, m0 = results[0].score, results[0].margin
    m0_finite = math.isfinite(m0)
    for r in results:
        if abs(r.score - s0) > _SALIENCE_TIE_EPS:
            return False
        r_finite = math.isfinite(r.margin)
        if r_finite != m0_finite:
            # Issue #107: one candidate carries a real (finite) margin and another
            # carries the non-finite "anchor not found" sentinel -- that mismatch
            # IS discriminating evidence (one of them evaluated the clause and the
            # other genuinely didn't), so the group is deliberately NOT tied.
            return False
        if r_finite and abs(r.margin - m0) > _SALIENCE_TIE_EPS:
            return False
        # else: both non-finite (issue #107) -- every candidate hit the same
        # "clause unevaluable" sentinel, which is the most-tied case there is.
        # Deliberately treated as tied (continue) rather than compared by value.
    return True


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
