"""Breadcrumb emitter: a planned path -> a stream of near-vehicle WaypointCmds.

The autonomy stack warns that a distant waypoint can strand the vehicle at a dead
end (docs/upstream_notes.md gotcha 14); waypoints should stay <= ~2.5 m ahead
(interfaces.WaypointCmd docstring). So instead of publishing the whole path, we
emit the FARTHEST path point that is both <= LOOKAHEAD_M ahead AND in clear
line-of-sight (all intermediate cells FREE/passable). The FSM calls `advance(pose)`
at 5 Hz; we step to the next crumb when the vehicle is within REACH_M of the
current one (or the /way_point_reached-equivalent signal fires).

Stall detection: if the pose moves < STALL_MOVE_M over STALL_WINDOW_S seconds, we
raise a replan flag so the FSM can recompute (e.g. a snapped waypoint stuck at a
wall). Time is injected via the pose timestamps (deterministic, no wall clock).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from core.interfaces import WaypointCmd
from core.nav.costmap import Costmap

# --------------------------------------------------------------------------- tunables
LOOKAHEAD_M: float = 2.5  # farthest a crumb may sit ahead of the vehicle
REACH_M: float = 0.8  # advance to next crumb within this distance of current
STALL_MOVE_M: float = 0.3  # movement below this over the window == stalled
STALL_WINDOW_S: float = 10.0  # stall observation window
#: Issue #74 — the dwell tolerance a leg-goal waypoint-of-record (see
#: ``BreadcrumbFollower.leg_goal_indices``) must be reached within before the follower
#: may advance past it.
#:
#: Issue #77 — was ``core.groundtruth.arrival.NOMINAL_ARRIVAL_TOL_M`` (~1.75 m, the
#: rubric's own frame-fit-derived credit tolerance), on the theory that a crumb this
#: follower refuses to skip should never be a NEW, stricter bar than the rubric
#: already applies. That theory holds for whether the leg gets CREDIT, but the dwell
#: gate does a second, unrelated job: it is what allows crumb selection to stop
#: aiming at the leg goal and move on. Releasing that gate from ~1.75 m out — nearly
#: double a CORRIDOR_BETWEEN leg's gate width in the observed cases — let the crumb
#: scan jump to a farther, still-LOS-clear point beyond the goal before the vehicle
#: had ever physically closed on it, so a corridor leg could get scored "arrived" at
#: the gate midpoint (rubric-satisfied) while the driven trajectory swung past without
#: ever crossing the gate segment (threading is a literal-crossing geometric check,
#: unaffected by arrival tolerance) — and non-corridor legs lost the last ~1 m of
#: approach that would have closed several near-miss excesses.
#:
#: ``REACH_M`` is the right constant instead: it is this SAME follower's own
#: established "close enough to a path point to consider it reached" radius, already
#: governing the ordinary within/overshoot progress-index advance a few lines below.
#: Reusing it here makes a leg-goal waypoint-of-record use the identical physical-
#: arrival criterion the follower already applies to every OTHER path point, rather
#: than importing a second, much looser tolerance concept (GT frame-fit residual) that
#: has nothing to do with the follower's own steering precision. This can only make
#: the follower drive closer before releasing a leg goal, never farther — it does not
#: change what counts as CREDIT (that stays keyed to the rubric's own tolerance,
#: untouched here); it only changes when the crumb scan stops being capped at the
#: goal.
LEG_GOAL_ARRIVAL_TOL_M: float = REACH_M


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def _mask_blocked(costmap: Costmap, r: int, c: int) -> bool:
    """Blocked test bounded by the costmap's OWN stamped mask shape.

    ``Costmap.blocked`` gates its array lookup on ``grid.in_bounds``, but the grid can
    grow after the costmap's masks were snapshotted (the vehicle drives into newly
    integrated terrain). A cell outside the snapshotted mask was never validated by
    stamping/inflation, so for line-of-sight purposes we conservatively treat it as
    blocked rather than index past the mask (which would raise) or trust an
    unvalidated cell. Inside the mask we defer to the costmap's own blocked test.
    """
    h, w = costmap.base_blocked.shape
    if not (0 <= r < h and 0 <= c < w):
        return True
    return costmap.blocked(r, c)


def line_of_sight(
    costmap: Costmap, a: tuple[float, float], b: tuple[float, float]
) -> bool:
    """True if the straight segment a->b crosses only passable cells (Bresenham-ish)."""
    grid = costmap.grid
    ar, ac = grid.world_to_cell(*a)
    br, bc = grid.world_to_cell(*b)
    dr = abs(br - ar)
    dc = abs(bc - ac)
    sr = 1 if ar < br else -1
    sc = 1 if ac < bc else -1
    err = dc - dr
    r, c = ar, ac
    while True:
        if _mask_blocked(costmap, r, c):
            return False
        if (r, c) == (br, bc):
            return True
        e2 = 2 * err
        if e2 > -dr:
            err -= dr
            c += sc
        if e2 < dc:
            err += dc
            r += sr


@dataclass
class BreadcrumbFollower:
    """Turns a planned path into an advancing WaypointCmd stream the FSM polls."""

    path: list[tuple[float, float]]
    costmap: Costmap
    lookahead_m: float = LOOKAHEAD_M
    reach_m: float = REACH_M
    stall_move_m: float = STALL_MOVE_M
    stall_window_s: float = STALL_WINDOW_S
    _idx: int = 0  # index into path of the last point we've committed to passing
    _hist: list[tuple[float, float, float]] = field(default_factory=list)  # (t, x, y)
    replan_flag: bool = False
    #: issue #62 — path index of the crumb ``_select_crumb`` most recently handed
    #: back. The LOS scan below can jump this crumb many indices ahead of ``_idx``
    #: (any point with a clear line-of-sight from the pose is fair game, regardless
    #: of how far along the path it sits) — a normal, intended shortcut on an open
    #: stretch. But ``_idx`` itself only advances via the local within/overshoot
    #: check against ``path[_idx]``/``path[_idx+1]`` in ``advance``. On a path that
    #: loops back near itself (e.g. after threading a corridor gate and doubling
    #: back through the same room on the way to a later leg), the vehicle can end
    #: up sitting exactly at the FAR crumb while ``_idx`` is still parked at an
    #: earlier, now spatially-unrelated stretch of the path whose neighbouring
    #: points never come within ``reach_m``/overshoot range of the actual pose —
    #: ``_idx`` then never advances again and the follower wedges forever (T-62
    #: sig-1). Tracking the last-selected crumb's own index lets ``advance`` fast-
    #: forward ``_idx`` past it once the vehicle demonstrably reaches it, instead of
    #: only ever creeping one index at a time.
    _last_crumb_idx: int = -1
    #: Cumulative path arc length (metres) up to and including ``path[k]``, index-
    #: matched to ``path``. Lazily built (see ``_arc_len``) the first time a lookahead
    #: bound is needed, then cached — ``path`` is never mutated in place after
    #: construction (a rebuilt/replanned route gets a NEW ``BreadcrumbFollower``), so
    #: a one-time build is always valid for the object's lifetime.
    _cum_dist: list[float] | None = field(default=None, repr=False, compare=False)
    #: Issue #74 — ascending ``path`` indices of each ordered leg's own goal (the
    #: "waypoint-of-record" ``plan_through(..., record_leg_bounds=True)`` recorded for
    #: that leg; see its docstring). Empty for a follower whose caller doesn't thread
    #: leg boundaries (e.g. every existing test, and any single-leg/no-leg route) —
    #: the follower then behaves exactly as before this issue. Not mutated after
    #: construction; ``_next_leg_goal_ptr`` tracks progress through it instead.
    leg_goal_indices: list[int] = field(default_factory=list)
    #: Dwell tolerance for ``leg_goal_indices`` — see ``LEG_GOAL_ARRIVAL_TOL_M``.
    leg_goal_tol_m: float = LEG_GOAL_ARRIVAL_TOL_M
    #: Index into ``leg_goal_indices`` of the next leg goal not yet dwelled-at. Once it
    #: reaches ``len(leg_goal_indices)`` every leg boundary has been visited and the
    #: follower is unconstrained (matches pre-#74 behaviour) for the remainder of the
    #: path (typically just the terminal tail, if any).
    _next_leg_goal_ptr: int = 0

    def _leg_ceiling(self) -> int | None:
        """``path`` index of the next un-dwelled leg-goal waypoint-of-record, or None.

        Issue #74 — while this is not None, neither crumb selection nor progress-index
        advancement may go past it: it is a HARD STOP, not just an ordinary path point,
        because the calling head (``core.heads.instruction.InstructionHead``) grades
        this leg on the driven trajectory actually dwelling near it, and a flat
        polyline follower has no other notion of "this point is the required stop for
        leg N" (issue's root-cause framing) to protect it from a lookahead shortcut or
        the #62 fast-forward jumping straight past it.
        """
        if self._next_leg_goal_ptr < len(self.leg_goal_indices):
            return self.leg_goal_indices[self._next_leg_goal_ptr]
        return None

    def _advance_leg_goals(self, pose: tuple[float, float]) -> None:
        """Pop every pending leg goal the pose has now dwelled within tolerance of.

        A ``while`` (not a single ``if``) because two ordered legs can share the same
        path index (e.g. a leg whose resolved goal coincides with the previous leg's,
        or a corridor leg whose gate-crossing segment is zero-length) — both must clear
        in the same tick the pose reaches that shared point, or the second would wedge
        the follower at a point already physically visited.
        """
        ceiling = self._leg_ceiling()
        while (
            ceiling is not None
            and ceiling < len(self.path)
            and _dist(pose, self.path[ceiling]) <= self.leg_goal_tol_m
        ):
            self._next_leg_goal_ptr += 1
            ceiling = self._leg_ceiling()

    def _arc_len(self) -> list[float]:
        """``_cum_dist``, building it on first use."""
        cum = self._cum_dist
        if cum is None or len(cum) != len(self.path):
            cum = [0.0] * len(self.path)
            for k in range(1, len(self.path)):
                cum[k] = cum[k - 1] + _dist(self.path[k - 1], self.path[k])
            self._cum_dist = cum
        return cum

    # ------------------------------------------------------------- crumb selection
    def _select_crumb(self, pose: tuple[float, float]) -> WaypointCmd | None:
        """Farthest path point <= lookahead AHEAD ON THE PATH with clear line-of-sight.

        "Ahead" is measured as arc length travelled ALONG the path from ``_idx`` to the
        candidate index, not the candidate's straight-line (Euclidean) distance from the
        current pose. On a path that runs out to a leg's own goal and then doubles back
        near its own earlier ground (e.g. returning past the start on the way to a later
        corridor/leg — the same self-proximity shape issue #62 already names for
        ``_idx`` tracking), a point many indices — and many real metres of travel —
        ahead can sit Euclidean-CLOSE to the current pose merely because the path folds
        back near it. Bounding by Euclidean pose distance alone then treats that distant,
        not-yet-visited point as "nearby enough to shortcut to", skipping the entire
        out-and-back detour (and whatever leg goal sits at its tip) even though it was
        never actually driven. Arc length is the literal distance the vehicle would have
        to travel along the planned route to reach the candidate, which is what
        "how far ahead" the docstring's LOOKAHEAD_M bound was always meant to cap — LOS
        shortcuts across a genuinely open, non-looping stretch are unaffected (arc length
        stays close to Euclidean distance there, since the path itself does not fold).
        """
        if self._idx >= len(self.path):
            return None
        cum = self._arc_len()
        # ``_idx`` can sit AHEAD of the actual pose (the within/overshoot advance in
        # ``advance`` consumes path points close to a not-yet-moved pose — legitimate,
        # see that method's docstring). Anchor the arc-length budget on the pose's own
        # remaining distance to ``path[_idx]`` plus arc length from there, not on
        # ``cum[_idx]`` alone, so a lagging pose is never granted MORE lookahead than a
        # pose that is truly sitting at ``path[_idx]`` would get.
        base = cum[self._idx] - _dist(pose, self.path[self._idx])
        chosen: tuple[float, float] | None = None
        chosen_idx: int | None = None
        # Issue #74 — never select a crumb past an un-dwelled leg-goal
        # waypoint-of-record. Without this cap the farthest-within-lookahead scan
        # below is exactly the mechanism the issue names: a leg goal that sits only
        # 1-2 lookahead-windows off the direct polyline (or at a short leg with little
        # remaining arc length) gets skipped in favour of a farther, LOS-clear point
        # past it, because the scan has no notion that the intervening point is a
        # mandatory stop rather than an ordinary path vertex. Capping the scan's own
        # END at the ceiling (inclusive — the leg goal itself must stay selectable)
        # makes the farthest-within-lookahead logic below select AT MOST the ceiling,
        # never past it, with no other change to how "farthest" is chosen.
        ceiling = self._leg_ceiling()
        scan_end = len(self.path) if ceiling is None else min(len(self.path), ceiling + 1)
        # Scan forward from current progress index; keep the farthest LOS-clear point
        # within lookahead. Stop early once a point exceeds lookahead (path is ordered).
        for j in range(self._idx, scan_end):
            pt = self.path[j]
            if cum[j] - base > self.lookahead_m:
                # Beyond lookahead — but keep the last good one; break to bound cost.
                if chosen is not None:
                    break
                # Nothing within lookahead yet: fall back to the nearest forward point
                # even if slightly beyond, so we still make progress.
                if line_of_sight(self.costmap, pose, pt):
                    chosen, chosen_idx = pt, j
                break
            if line_of_sight(self.costmap, pose, pt):
                chosen, chosen_idx = pt, j
        if chosen_idx is not None:
            self._last_crumb_idx = chosen_idx
        if chosen is None:
            # No LOS-clear crumb within lookahead. Never hand back a raw point whose
            # straight segment from the pose crosses a blocked cell — that is exactly the
            # unvalidated-segment failure this follower must not produce. Prefer the
            # nearest forward path point that IS reachable per line_of_sight (scanning
            # the whole remaining path, not just within lookahead); only if none is
            # reachable do we surface the next path vertex WITH a replan flag so the FSM
            # recomputes rather than silently driving through geometry.
            chosen, nearest_idx = self._nearest_reachable_point(pose, scan_end)
            if chosen is None:
                self.replan_flag = True
                chosen, nearest_idx = self.path[self._idx], self._idx
            self._last_crumb_idx = nearest_idx
        return WaypointCmd(x=float(chosen[0]), y=float(chosen[1]))

    def _nearest_reachable_point(
        self, pose: tuple[float, float], scan_end: int | None = None
    ) -> tuple[tuple[float, float] | None, int]:
        """Nearest forward path point in clear line-of-sight from ``pose`` (or (None, -1)).

        ``scan_end`` (issue #74): bounds the scan at an un-dwelled leg-goal ceiling,
        same discipline as ``_select_crumb``'s primary scan — defaults to the whole
        remaining path for callers outside this leg-boundary-aware fallback.
        """
        best: tuple[float, float] | None = None
        best_idx = -1
        best_d = float("inf")
        end = len(self.path) if scan_end is None else scan_end
        for j in range(self._idx, end):
            pt = self.path[j]
            if not line_of_sight(self.costmap, pose, pt):
                continue
            d = _dist(pose, pt)
            if d < best_d:
                best_d, best, best_idx = d, pt, j
        return best, best_idx

    # ------------------------------------------------------------- public API
    def current(self, pose: tuple[float, float]) -> WaypointCmd | None:
        """The crumb to publish for this pose (does not advance)."""
        return self._select_crumb(pose)

    def advance(
        self, pose: tuple[float, float], t: float, reached_signal: bool = False
    ) -> WaypointCmd | None:
        """Called by the FSM at 5 Hz. Advances progress, updates stall flag, returns
        the next crumb to publish (or None when the path is exhausted).

        pose: (x, y) metres. t: timestamp (s). reached_signal: True mirrors the
        /way_point_reached-equivalent from the stack.
        """
        self._record(pose, t)
        # Issue #74 — mark any pending leg-goal waypoint(s)-of-record the pose has now
        # dwelled within tolerance of BEFORE computing this tick's ceiling, so a leg
        # reached this very tick immediately lifts the hard stop below rather than
        # costing an extra tick of wedging at (or just past) it.
        self._advance_leg_goals(pose)
        ceiling = self._leg_ceiling()
        # Advance the progress index past any path point we're within reach of, OR
        # that we've clearly overshot (the following point is nearer than this one, so
        # the vehicle has moved past it). Also honour the reached signal.
        while self._idx < len(self.path):
            # Issue #74 — hard stop: `_idx` may not creep past an un-dwelled leg-goal
            # waypoint-of-record via the ordinary within/overshoot/signalled checks
            # below (mechanism 2 in the issue: grid resolution far finer than
            # `reach_m` lets a single stationary tick consume many indices at once,
            # which can itself pre-consume a short leg's approach before it is ever
            # actually reached).
            if ceiling is not None and self._idx >= ceiling:
                break
            within = _dist(pose, self.path[self._idx]) <= self.reach_m
            overshot = (
                self._idx + 1 < len(self.path)
                and _dist(pose, self.path[self._idx + 1]) < _dist(pose, self.path[self._idx])
            )
            signalled = reached_signal and self._idx == self._progress_target(pose)
            if within or overshot or signalled:
                self._idx += 1
            else:
                break
        # issue #62: fast-forward past a far-ahead LOS-selected crumb the vehicle has
        # actually reached, even when the local within/overshoot check above never
        # catches up. `_select_crumb` can hand back a crumb many indices ahead of
        # `_idx` (any point with clear line-of-sight qualifies, regardless of index
        # gap) — normal on an open stretch. On a path that loops back near itself
        # (e.g. after threading a corridor gate en route to a later leg), `path[_idx]`
        # and its immediate neighbours can end up on an earlier, now spatially-
        # unrelated stretch the vehicle no longer approaches, so within/overshoot
        # against THOSE points never fires again and `_idx` wedges forever. Safe: only
        # ever jumps ahead to a crumb the follower itself already selected and
        # LOS-verified, never past unverified path points.
        # Issue #74: `_select_crumb`'s own scan already never selects
        # `_last_crumb_idx` past a pending ceiling (see its `scan_end` cap), so this
        # fast-forward is automatically leg-boundary-safe too — no separate ceiling
        # check needed here.
        if (
            0 <= self._last_crumb_idx < len(self.path)
            and self._last_crumb_idx >= self._idx
            and _dist(pose, self.path[self._last_crumb_idx]) <= self.reach_m
        ):
            self._idx = self._last_crumb_idx + 1
        if self._idx >= len(self.path):
            return None
        return self._select_crumb(pose)

    def _progress_target(self, pose: tuple[float, float]) -> int:
        """Index of the crumb the reached_signal refers to (nearest ahead)."""
        best = self._idx
        best_d = float("inf")
        for j in range(self._idx, len(self.path)):
            d = _dist(pose, self.path[j])
            if d < best_d:
                best_d, best = d, j
        return best

    def at_goal(self, pose: tuple[float, float]) -> bool:
        return self._idx >= len(self.path) or (
            len(self.path) > 0 and _dist(pose, self.path[-1]) <= self.reach_m
        )

    # ------------------------------------------------------------- stall
    def _record(self, pose: tuple[float, float], t: float) -> None:
        self._hist.append((t, pose[0], pose[1]))
        # Drop history older than the stall window.
        cutoff = t - self.stall_window_s
        self._hist = [h for h in self._hist if h[0] >= cutoff]
        self.replan_flag = self._is_stalled(t)

    def _is_stalled(self, t: float) -> bool:
        """Moved < stall_move_m over the full stall window -> stalled."""
        if not self._hist:
            return False
        t0 = self._hist[0][0]
        if t - t0 < self.stall_window_s:
            return False  # not enough history yet to judge
        xs = [h[1] for h in self._hist]
        ys = [h[2] for h in self._hist]
        span = ((max(xs) - min(xs)) ** 2 + (max(ys) - min(ys)) ** 2) ** 0.5
        return span < self.stall_move_m
