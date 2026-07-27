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
from typing import Any, Callable, Sequence

from core.interfaces import QType, RobotIO, WaypointCmd
from core.fsm.floors import _anchor_nouns
from core.heads import explore_debug
from core.nav.exploration import ExplorationPolicy, ExplorationStatus
from core.nav.frontiers import detect_frontiers
from core.nav.occupancy import OccupancyGrid, integrate_scan_overhead_decimated
from core.perception.detector import is_answer_eligible
from core.plan_schema import Plan
from core.plan_walk import iter_avoid_anchors, iter_route_anchors, iter_target_anchors

from core.heads.instruction import InstructionHead

# noun-list -> ((x, y) -> affinity float); default uniform (constant 0).
AffinityFactory = Callable[[Sequence[str]], Callable[[tuple[float, float]], float]]

# CP2 detector-miss recovery seam (design doc §CP2). Called when a plan-critical noun has 0
# instances after >=60% of the explore budget is spent (once per question; the checkpoint
# module's ledger enforces the hard cap). Returns a MissRecoveryOutcome-shaped object
# (duck-typed: ``.action`` in {"provisional","absent"}, ``.tile``, ``.bbox_hint``,
# ``.confidence``, ``.n_obs``, ``.score``).
#     run(noun: str, raw: str, tiles: Sequence) -> MissRecoveryOutcome
MissRecoveryFn = Callable[..., Any]

# CP2 fusion hook: cast a recovery outcome into a provisional instance placed in the world.
# Default places it at the frontier-direction centroid with n_obs=1, score=confidence*0.5.
#     fuse_hint(noun, xy, outcome) -> provisional-instance token (opaque to the head)
FuseHintFn = Callable[..., Any]

# CP5 frontier-selection seam (design doc §CP5). Fires only when the scene proves
# multi-room (>=2 disconnected explored regions OR explored area over threshold) and the
# seam is non-None; called with the question + the top-5 geometric frontier candidates.
# Returns a FrontierOutcome-shaped object (duck-typed: ``.action`` in {"choice","fallback"}
# and ``.index`` a 0-indexed position into the frontier list). Fallback == the geometric
# top frontier.
#     frontier_selector(question: str, frontiers: list[Frontier]) -> FrontierOutcome
FrontierSelectFn = Callable[..., Any]

#: CP5 scene-scale triggers (design §CP5): multi-room evidence.
MULTI_REGION_TRIGGER: int = 2  # >= this many disconnected explored regions
EXPLORED_AREA_TRIGGER_M2: float = 60.0  # OR explored FREE area over this (m^2)
CP5_TOP_FRONTIERS: int = 5

#: Design "Risk note" default provisional score factor (mirrors miss_recovery module).
PROVISIONAL_SCORE_FACTOR: float = 0.5
#: Fraction of the explore budget that must be spent before CP2 may fire (design §CP2).
COVERAGE_TRIGGER_FRAC: float = 0.60

#: SYS-F9: never publish a raw frontier centroid further than this along the straight
#: line toward it (gotcha 14: a distant waypoint strands the vehicle at a dead end). The
#: clamp keeps the goal near the vehicle, exactly like the breadcrumb lookahead.
FRONTIER_GOAL_CLAMP_M: float = 2.5
#: SYS-F9: path_distance sentinel the frontier scorer stamps on an UNREACHABLE cluster
#: (frontiers.py assigns pd=1e6). Such frontiers are dropped rather than commanded.
UNREACHABLE_PD: float = 1e6

#: SYS-F11: clear a CP2 provisional waypoint after this long without arrival (design §CP2).
PROVISIONAL_TIMEOUT_S: float = 45.0
#: SYS-F11: arrival tolerance for clearing a provisional waypoint on reach.
PROVISIONAL_ARRIVAL_TOL_M: float = 0.8

#: H14: minimum interval between frontier-detection refreshes (1 Hz). The frontier scorer
#: (detect_frontiers) is the per-tick cost centre; nothing about frontier goals needs 5 Hz.
FRONTIER_THROTTLE_S: float = 1.0


def uniform_affinity(_nouns: Sequence[str]):
    """Default detector-affinity: constant 0 (purely geometric frontier scoring)."""
    return lambda _xy: 0.0


@dataclass
class ExploreHead:
    """Steps exploration/execution once per tick; delegates IF to the InstructionHead."""

    plan: Plan | None = None
    affinity_fn: AffinityFactory = uniform_affinity
    instruction: InstructionHead | None = None

    #: CP2 detector-miss recovery seam; None == today's behaviour (no recovery).
    miss_recoverer: MissRecoveryFn | None = None
    #: fusion hook for CP2 provisional instances; None == the default centroid placement.
    fuse_hint: FuseHintFn | None = None
    #: zero-arg callable -> explored-budget fraction [0,1]; feeds the >=60% CP2 trigger.
    #: None == 0.0 (CP2 never fires — safe default).
    budget_frac: Callable[[], float] | None = None
    #: tile supplier for CP2 (zero-arg -> list of tiles); None == an empty tile list.
    tiles_fn: Callable[[], Sequence[Any]] | None = None

    #: CP5 frontier-selection seam; None == today's behaviour (geometric top frontier).
    frontier_selector: FrontierSelectFn | None = None
    #: explored-area trigger for CP5 (m^2); overridable per config.
    explored_area_trigger_m2: float = EXPLORED_AREA_TRIGGER_M2

    grid: OccupancyGrid = field(default_factory=OccupancyGrid)
    _policy: ExplorationPolicy | None = None
    last_status: ExplorationStatus | None = None
    _cp2_fired: bool = False  # once-per-question head guard (ledger caps the hard limit)
    provisional: Any | None = None  # last CP2 provisional instance (navigation bias target)
    provisional_xy: tuple[float, float] | None = None
    #: SYS-F11 — the noun the CP2 provisional stands in for, and the odom time it was set,
    #: so the latch can be cleared on arrival / timeout / the noun appearing in the scene.
    _provisional_noun: str | None = None
    _provisional_t0: float | None = None
    #: H14 — 1 Hz frontier-detection throttle. The frontier phase re-detects at most once
    #: per FRONTIER_THROTTLE_S, reusing the last decision between refreshes (nothing about
    #: frontiers needs 5 Hz; detect_frontiers is the tick's cost centre). The opening sweep
    #: is NOT throttled (it needs 5 Hz waypoint advancement and does no frontier work).
    _last_frontier_decision: Any | None = None
    _last_frontier_t: float | None = None
    #: Issue #43b — the most recent scene index, refreshed every ``_explore`` tick.
    #: ``_affinity_nouns`` reads it to check target-noun eligibility (no live SceneIndex
    #: reaches ``_affinity()``'s callers otherwise: policy construction happens once, at
    #: the first tick, so this must be current by then).
    _scene: Any | None = None

    # ------------------------------------------------------------------ per-tick
    def advance(self, io: RobotIO, scene) -> None:
        """One exploration/execution step for this tick.

        For INSTRUCTION_FOLLOWING we delegate to the instruction head FIRST; if it did
        not emit a waypoint this tick (no grounded prefix yet, or the route is
        ungrounded) we FALL THROUGH to frontier exploration so the robot always moves
        and can observe the ungrounded anchors (H3 / IF-F1 / SYS-F3 — the deadlock:
        the IF head waited for grounding, grounding waited for motion, motion never
        came). Exploration is noun-affinity-biased toward the still-ungrounded nouns.
        """
        if self.plan is not None and self.plan.qtype is QType.INSTRUCTION_FOLLOWING:
            emitted = False
            if self.instruction is not None:
                emitted = bool(self.instruction.advance(io, scene))
            if emitted:
                return
            # No waypoint from the route this tick — explore to ground the remaining
            # anchors instead of parking.
            self._explore(io, scene)
            return
        self._explore(io, scene)

    def _explore(self, io: RobotIO, scene=None) -> None:
        self._scene = scene  # issue #43b: keep _affinity_nouns' eligibility check current
        odom = io.latest_odom()
        # SYS-F9: never construct or run the exploration policy against a pre-odom
        # (0,0)/t=0 anchor — that seeds the sweep at the origin and, once the real odom
        # header stamp jumps in, skips the whole orientation sweep. Publish nothing this
        # tick and wait for the first real pose.
        if odom is None:
            return
        pose = (float(odom.x), float(odom.y))
        t = float(odom.t)
        patch = io.latest_terrain(extended=False)
        if patch is not None:
            # vehicle_z feeds the runtime ground-offset estimator (issue #36) used
            # by the overhead fallback below.
            self.grid.integrate_patch(patch, vehicle_z=float(odom.z))
        # Overhead-clearance layer: flag overhangs (bar tables/shelves) the terrain
        # slab dropped, so exploration/planning won't route under furniture the base
        # stack reads as FREE floor. Terrain first so per-cell ground_z is set.
        scan = io.latest_scan() if hasattr(io, "latest_scan") else None
        if scan is not None:
            odom_z = float(odom.z)
            integrate_scan_overhead_decimated(self.grid, scan, odom_z)
        self.grid.mark_pose(pose[0], pose[1])

        self._maybe_recover_miss(io, scene, pose, t)
        # SYS-F11: a stale provisional latch must not replace exploration forever — clear
        # it on arrival / timeout / the noun appearing in the scene, then resume frontier
        # flow this same tick.
        self._maybe_clear_provisional(scene, pose, t)

        if self._policy is None:
            self._policy = ExplorationPolicy(start_xy=pose, affinity=self._affinity())
        decision = self._stepped_decision(pose, t)
        self.last_status = decision.status
        # issue #83: diagnostic-only, no-op unless VLA_EXPLORE_DEBUG_DIR is set (see
        # core.heads.explore_debug docstring) — reports what was actually decided below.
        published: tuple[float, float] | None = None
        try:
            # A live CP2 provisional biases navigation: prefer driving toward it over the
            # geometric frontier while it stands. (Not distance-clamped: unlike a raw far
            # frontier centroid, the provisional is a fixed placed target the vehicle
            # converges on tick over tick — SYS-F9's stranding concern is the far
            # *frontier* goal.)
            if self.provisional_xy is not None:
                io.publish_waypoint(WaypointCmd(self.provisional_xy[0], self.provisional_xy[1]))
                published = self.provisional_xy
                return
            # CP5: on a frontier decision in a multi-room scene, let the seam pick among the
            # top-5 frontiers (fallback = the geometric top the policy already chose).
            cp5_wp = self._maybe_cp5_frontier(pose, decision)
            if cp5_wp is not None:
                clamped = self._clamp_goal(pose, (cp5_wp.x, cp5_wp.y))
                io.publish_waypoint(clamped)
                published = (clamped.x, clamped.y)
            elif decision.waypoint is not None:
                # SYS-F9: drop an UNREACHABLE frontier goal (pd sentinel) — commanding it
                # just strands the vehicle; skipping lets the next tick surface a reachable
                # one. Only frontier decisions carry a Frontier to test; sweep waypoints
                # always go.
                if (
                    decision.status is ExplorationStatus.FRONTIER
                    and _frontier_unreachable(decision.frontier)
                ):
                    return
                clamped = self._clamp_goal(pose, (decision.waypoint.x, decision.waypoint.y))
                io.publish_waypoint(clamped)
                published = (clamped.x, clamped.y)
        finally:
            explore_debug.maybe_dump(self, self.grid, pose, t, decision.status.value, published)

    # ------------------------------------------------------------------ H14 throttle
    def _stepped_decision(self, pose: tuple[float, float], t: float):
        """Step the exploration policy with a 1 Hz frontier-detection throttle (H14).

        ``ExplorationPolicy.step`` runs ``detect_frontiers`` (the tick's cost centre) on
        every call once past the opening sweep. Nothing about frontier goals needs 5 Hz,
        so during the FRONTIER phase we re-step (and thus re-detect) at most once per
        ``FRONTIER_THROTTLE_S``, reusing the last FRONTIER decision between refreshes. A
        re-plan/avoid-stamp event does not reach here (those live in the IF head); the
        throttle is time-based and self-refreshes each second. The sweep phase is never
        throttled — it needs per-tick waypoint advancement and does no frontier work.
        """
        cached = self._last_frontier_decision
        if (
            cached is not None
            and self._last_frontier_t is not None
            and (t - self._last_frontier_t) < FRONTIER_THROTTLE_S
        ):
            return cached
        decision = self._policy.step(self.grid, pose, t)
        if decision.status is ExplorationStatus.FRONTIER:
            self._last_frontier_decision = decision
            self._last_frontier_t = t
        else:
            # Sweep / complete: no frontier work to throttle; drop any stale cache.
            self._last_frontier_decision = None
            self._last_frontier_t = None
        return decision

    # ------------------------------------------------------------------ SYS-F9 clamp
    def _clamp_goal(
        self, pose: tuple[float, float], goal: tuple[float, float]
    ) -> WaypointCmd:
        """Clamp a raw goal to <= FRONTIER_GOAL_CLAMP_M along the line toward it (SYS-F9).

        A far frontier centroid published raw can strand the vehicle at a dead end
        (gotcha 14). We keep the goal near the vehicle — direction preserved, distance
        bounded — so the base stack drives toward the frontier without a long beeline.
        """
        gx, gy = float(goal[0]), float(goal[1])
        dx, dy = gx - pose[0], gy - pose[1]
        d = (dx * dx + dy * dy) ** 0.5
        if d <= FRONTIER_GOAL_CLAMP_M or d < 1e-9:
            return WaypointCmd(gx, gy)
        s = FRONTIER_GOAL_CLAMP_M / d
        return WaypointCmd(pose[0] + dx * s, pose[1] + dy * s)

    # ------------------------------------------------------------------ CP5
    def _maybe_cp5_frontier(self, pose: tuple[float, float], decision) -> WaypointCmd | None:
        """CP5 frontier selection. Returns the chosen frontier waypoint, or None to defer
        to the policy's geometric choice (fallback / not triggered)."""
        if self.frontier_selector is None:
            return None
        if decision.status is not ExplorationStatus.FRONTIER:
            return None
        if not self._scene_is_multi_room(pose):
            return None
        frontiers = [
            f
            for f in detect_frontiers(self.grid, pose, self._affinity())
            if not _frontier_unreachable(f)
        ][:CP5_TOP_FRONTIERS]
        if not frontiers:
            return None
        question = getattr(self.plan, "question_raw", "") or ""
        try:
            outcome = self.frontier_selector(question, frontiers)
        except Exception:  # noqa: BLE001 — a dark checkpoint falls back to the geometric top
            return None
        if getattr(outcome, "action", "fallback") != "choice":
            return None
        idx = getattr(outcome, "index", None)
        if not isinstance(idx, int) or not (0 <= idx < len(frontiers)):
            return None
        f = frontiers[idx]
        return WaypointCmd(f.xy[0], f.xy[1])

    def _scene_is_multi_room(self, pose: tuple[float, float]) -> bool:
        """Scene-scale trigger: >=2 disconnected explored regions OR explored area over
        the configured threshold (design §CP5)."""
        regions = _explored_regions(self.grid)
        if regions >= MULTI_REGION_TRIGGER:
            return True
        return _explored_area_m2(self.grid) > self.explored_area_trigger_m2

    # ------------------------------------------------------------------ CP2
    def _maybe_recover_miss(
        self, io: RobotIO, scene, pose: tuple[float, float], t: float
    ) -> None:
        """Fire CP2 once when a plan-critical noun has 0 instances past the coverage gate."""
        if self.miss_recoverer is None or self._cp2_fired or scene is None:
            return
        if self._explored_frac() < COVERAGE_TRIGGER_FRAC:
            return
        noun, raw = self._missing_noun(scene)
        if noun is None:
            return
        self._cp2_fired = True  # head guard: attempt once (the ledger caps the hard limit)
        tiles = list(self.tiles_fn()) if self.tiles_fn is not None else []
        try:
            outcome = self.miss_recoverer(noun, raw, tiles)
        except Exception:  # noqa: BLE001 — a dark checkpoint falls through to resolve ladder
            return
        if getattr(outcome, "action", "absent") != "provisional":
            return
        xy = self._frontier_centroid(pose)
        self.provisional = self._fuse(noun, xy, outcome)
        self.provisional_xy = xy
        self._provisional_noun = noun
        self._provisional_t0 = t

    def _maybe_clear_provisional(
        self, scene, pose: tuple[float, float], t: float
    ) -> None:
        """Clear the CP2 provisional latch and resume frontier flow (SYS-F11).

        The latch is a navigation bias, not a permanent replacement for exploration.
        Clear it — keeping the once-per-question CP2 guard so it never re-fires — on any
        of: (a) arrival within tolerance, (b) ~PROVISIONAL_TIMEOUT_S elapsed since it was
        set, (c) the stood-in noun appearing in the scene index. Not cleared on the tick it
        was set (its t0 == t) unless the vehicle is already at it.
        """
        if self.provisional_xy is None:
            return
        reason: str | None = None
        just_set = self._provisional_t0 is not None and t <= self._provisional_t0
        if _near(pose, self.provisional_xy, PROVISIONAL_ARRIVAL_TOL_M):
            reason = "arrival"
        elif (
            not just_set
            and self._provisional_t0 is not None
            and (t - self._provisional_t0) >= PROVISIONAL_TIMEOUT_S
        ):
            reason = "timeout"
        elif not just_set and self._provisional_noun is not None and scene is not None:
            try:
                found = scene.by_label(self._provisional_noun)
            except Exception:  # noqa: BLE001
                found = None
            if found:
                reason = "noun_observed"
        if reason is None:
            return
        self.provisional = None
        self.provisional_xy = None
        self._provisional_noun = None
        self._provisional_t0 = None

    def _missing_noun(self, scene) -> tuple[str | None, str]:
        """The first plan-critical noun with 0 instances in the scene (else (None, ''))."""
        for noun in _plan_nouns(self.plan):
            try:
                found = scene.by_label(noun)
            except Exception:  # noqa: BLE001
                found = None
            if not found:
                raw = ""
                if self.plan is not None and self.plan.target is not None:
                    raw = self.plan.target.raw or ""
                return noun, raw or noun
        return None, ""

    def _explored_frac(self) -> float:
        if self.budget_frac is None:
            return 0.0
        try:
            return float(self.budget_frac())
        except Exception:  # noqa: BLE001
            return 0.0

    def _frontier_centroid(self, pose: tuple[float, float]) -> tuple[float, float]:
        """Direction centroid for the CP2 provisional placement (SYS-F9 aware).

        Prefer the top REACHABLE frontier; if none is reachable yet (early ticks where the
        BFS-connected FREE region is still small) fall back to the top-scored frontier so
        the provisional still biases toward unexplored space rather than pinning the
        vehicle at its own pose. The published waypoint is distance-clamped downstream
        (``_clamp_goal``), so a far centroid never becomes a stranding beeline.
        """
        frontiers = detect_frontiers(self.grid, pose, self._affinity())
        for f in frontiers:
            if not _frontier_unreachable(f):
                return f.xy
        if frontiers:
            return frontiers[0].xy
        return pose

    def _fuse(self, noun: str, xy: tuple[float, float], outcome: Any) -> Any:
        """Cast a CP2 outcome into a provisional instance via the injected fuse hook.

        Default: a lightweight :class:`_ProvisionalInstance` at ``xy`` with n_obs=1 and
        score=confidence*0.5 (documented: real fusion binding — casting bbox_hint through
        the detection fusion path — is integration work).
        """
        if self.fuse_hint is not None:
            return self.fuse_hint(noun, xy, outcome)
        conf = float(getattr(outcome, "confidence", 0.0))
        score = float(getattr(outcome, "score", conf * PROVISIONAL_SCORE_FACTOR))
        n_obs = int(getattr(outcome, "n_obs", 1))
        return _ProvisionalInstance(noun=noun, xy=xy, n_obs=n_obs, score=score)

    def _affinity(self):
        nouns = self._affinity_nouns()
        try:
            return self.affinity_fn(nouns)
        except Exception:
            return uniform_affinity(nouns)

    def _affinity_nouns(self) -> list[str]:
        """Nouns the frontier scorer should bias toward.

        For INSTRUCTION_FOLLOWING (issue #43c): ordered leg-seeking, mirroring the
        OBJECT_REFERENCE mechanism below. Walk the plan's route legs in order; the FIRST
        leg whose anchor noun has NO answer-eligible scene instance is the current
        exploration focus (``_first_ungrounded_leg_noun``) — bias the frontier score
        toward that noun alone (plus the plan nouns as a fallback tail). Ordered
        discipline: leg k leads before k+1; once leg k's noun becomes answer-eligible the
        focus advances to k+1 (architecture §4 row 10, H3b). All legs' anchor nouns
        already answer-eligible (or an empty route): no injection, plan nouns unchanged.

        For OBJECT_REFERENCE (issue #43b): when the target noun has NO answer-eligible
        instances (see ``core.perception.detector.is_answer_eligible`` — the target is
        either absent or only weakly-scored/under-observed question-pass hits), bias the
        frontier score toward the target's ANCHOR noun(s) instead (e.g. 'table' in 'the
        teapot on the table') — steering exploration toward the grounded anchor, from
        which a re-observed/rescored target instance is most likely to be found. If the
        target IS already eligible, this is a no-op (target nouns as before).

        For all other qtypes this is exactly the plan's nouns as before.
        """
        base = _plan_nouns(self.plan)
        if self.plan is not None and self.plan.qtype is QType.OBJECT_REFERENCE:
            return self._object_ref_affinity_nouns(base)
        if self.plan is not None and self.plan.qtype is QType.INSTRUCTION_FOLLOWING:
            return self._instruction_affinity_nouns(base)
        return base

    def _instruction_affinity_nouns(self, base: list[str]) -> list[str]:
        """Issue #43c: INSTRUCTION_FOLLOWING affinity — ordered leg anchor-seeking.

        The first route leg (in plan order) whose anchor noun lacks an answer-eligible
        scene instance leads the affinity list; every other leg (grounded or not) is left
        alone this tick, so the bias stays on one anchor at a time (ordered discipline).
        All legs already answer-eligible: fall through to the plan nouns unchanged.
        """
        focus = self._first_ungrounded_leg_noun()
        if focus is None:
            return base
        biased: list[str] = [focus]
        for n in base:  # keep the remaining plan nouns as a fallback tail
            if n not in biased:
                biased.append(n)
        return biased

    def _first_ungrounded_leg_noun(self) -> str | None:
        """The anchor noun of the earliest route leg with no answer-eligible instance.

        None if the plan has no route, or every leg's anchor noun is already
        answer-eligible (see ``_target_is_answer_eligible``, reused generically here).
        """
        if self.plan is None:
            return None
        for leg in self.plan.route:
            noun = leg.anchors[0].noun if leg.anchors else None
            if noun and not self._target_is_answer_eligible(noun):
                return noun
        return None

    def _object_ref_affinity_nouns(self, base: list[str]) -> list[str]:
        """Issue #43b: OBJECT_REFERENCE affinity — anchor-seeking when target-starved.

        No anchor nouns, no scene, or the target already answer-eligible: fall through
        to the plan nouns unchanged (today's purely-geometric-plus-target behaviour).
        """
        target_noun = getattr(self.plan.target, "noun", None) if self.plan.target else None
        anchors = _anchor_nouns(self.plan)
        if not anchors or self._target_is_answer_eligible(target_noun):
            return base
        biased: list[str] = [n for n in anchors if n]
        for n in base:  # target noun (+ any others) stays as a fallback tail
            if n not in biased:
                biased.append(n)
        return biased

    def _target_is_answer_eligible(self, target_noun: str | None) -> bool:
        """True iff the scene has >=1 answer-eligible instance of the target noun.

        No scene yet (pre-first-tick) or no target noun: treated as "not yet known to be
        starved" — returns True so affinity stays on the target noun rather than jumping
        to the anchor before perception has had a chance to ground anything at all.
        """
        if self._scene is None or not target_noun:
            return True
        try:
            candidates = self._scene.by_label(target_noun)
        except Exception:  # noqa: BLE001 — a broken scene lookup must not crash exploration
            return True
        return any(is_answer_eligible(c) for c in candidates)


def _frontier_unreachable(f) -> bool:
    """True if a frontier is the UNREACHABLE sentinel (SYS-F9).

    ``detect_frontiers`` stamps ``path_distance = UNREACHABLE_PD`` (1e6) on a cluster with
    no finite BFS path from the vehicle. Commanding such a goal strands the robot, so it
    is dropped. None (no cluster) is treated as unreachable.
    """
    if f is None:
        return True
    return getattr(f, "path_distance", 0.0) >= UNREACHABLE_PD


def _near(a: tuple[float, float], b: tuple[float, float], tol: float) -> bool:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5 <= tol


def _explored_area_m2(grid: OccupancyGrid) -> float:
    """Total FREE explored area in square metres."""
    from core.nav.occupancy import FREE

    state = grid.state
    if state is None:
        return 0.0
    n_free = int((state == FREE).sum())
    return n_free * (grid.cell_m * grid.cell_m)


def _explored_regions(grid: OccupancyGrid) -> int:
    """Count of disconnected FREE explored regions (8-connected components).

    A multi-room scene shows up as >=2 FREE blobs separated by UNKNOWN/OBSTACLE (design
    §CP5). Deterministic BFS over the FREE mask.
    """
    from collections import deque

    from core.nav.occupancy import FREE

    import numpy as np

    state = grid.state
    if state is None:
        return 0
    free = state == FREE
    h, w = free.shape
    seen = np.zeros_like(free)
    neigh = ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1))
    regions = 0
    for r0 in range(h):
        for c0 in range(w):
            if not free[r0, c0] or seen[r0, c0]:
                continue
            regions += 1
            dq = deque([(r0, c0)])
            seen[r0, c0] = True
            while dq:
                r, c = dq.popleft()
                for dr, dc in neigh:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and free[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        dq.append((nr, nc))
    return regions


@dataclass(frozen=True)
class _ProvisionalInstance:
    """A CP2 provisional instance: a low-confidence, single-observation navigation target.

    Deliberately carries ``n_obs=1`` so it can NEVER on its own satisfy the >=3-obs
    early-answer gate (design §CP2 "Risk note"). Real fusion binding (casting the VLM's
    bbox_hint through the detection fusion path into a tracked InstanceRecord) is
    integration work — this stand-in only biases navigation.
    """

    noun: str
    xy: tuple[float, float]
    n_obs: int = 1
    score: float = 0.0


def _plan_nouns(plan: Plan | None) -> list[str]:
    """All nouns referenced by a plan (target + clauses, or route anchors), recursing
    through every anchor's nested ``disambiguator`` chain (issue #95) — a noun that
    appears only as a disambiguator (e.g. "the potted plant closest to the pyramid
    candle holder") must still reach the detector prompt / affinity / missing-noun
    logic downstream of this function, or the robot never even asks the detector to
    look for it.

    Deliberately excludes ``plan.avoid`` anchors (use :func:`_plan_avoid_nouns` for
    those). This return value feeds BOTH GroundingDINO caption passes
    (``core.perception.detector``) as ``refresh_prompt``'s ``question_nouns`` argument:
    the short question-noun-only pass ("target recall") and, via ``question_nouns``'
    priority slot, the full question+vocab pass ("scene-index breadth"). The short
    pass's box threshold is calibrated against it staying small
    (``ENV_GDINO_QUESTION_BOX_THRESHOLD`` = 0.25, below the 0.35 vocab-pass default,
    specifically because a ~2-phrase prompt carries far less token-position score
    dilution — detector.py's own probe found a 117-phrase caption decodes ZERO
    'teapot' at any threshold where a 2-phrase caption decodes it in 194/211
    keyframes), so only nouns that belong in the SHORT pass go here. Avoid anchors are
    breadth, not the target being grounded — they route separately (issue #108) via
    :func:`_plan_avoid_nouns` into ``refresh_prompt``'s ``full_only_nouns`` argument,
    which joins the full caption only.
    """
    if plan is None:
        return []
    nouns: list[str] = []
    if plan.target is not None:
        nouns.append(plan.target.noun)
        nouns.extend(a.noun for a in iter_target_anchors(plan.target))
    nouns.extend(a.noun for a in iter_route_anchors(plan.route))
    return [n for n in nouns if n]


def _plan_avoid_nouns(plan: Plan | None) -> list[str]:
    """Nouns referenced by a plan's ``avoid`` anchors (``between``/``near``), recursing
    through nested disambiguators (:func:`core.plan_walk.iter_avoid_anchors`).

    Issue #108: avoid anchors are objects the robot must locate so navigation can
    penalise their region — breadth, not the target being grounded — so they belong in
    the full question+vocab GroundingDINO caption only, never the short
    question-noun-only caption (see :func:`_plan_nouns`'s docstring for why the short
    pass must stay tiny). Feed this return value to ``refresh_prompt``'s
    ``full_only_nouns`` argument, never to its ``question_nouns`` argument.
    """
    if plan is None:
        return []
    return [n for n in (a.noun for a in iter_avoid_anchors(plan.avoid)) if n]
