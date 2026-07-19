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

    # ------------------------------------------------------------- crumb selection
    def _select_crumb(self, pose: tuple[float, float]) -> WaypointCmd | None:
        """Farthest path point <= lookahead ahead of pose with clear line-of-sight."""
        if self._idx >= len(self.path):
            return None
        chosen: tuple[float, float] | None = None
        chosen_idx: int | None = None
        # Scan forward from current progress index; keep the farthest LOS-clear point
        # within lookahead. Stop early once a point exceeds lookahead (path is ordered).
        for j in range(self._idx, len(self.path)):
            pt = self.path[j]
            if _dist(pose, pt) > self.lookahead_m:
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
            chosen, nearest_idx = self._nearest_reachable_point(pose)
            if chosen is None:
                self.replan_flag = True
                chosen, nearest_idx = self.path[self._idx], self._idx
            self._last_crumb_idx = nearest_idx
        return WaypointCmd(x=float(chosen[0]), y=float(chosen[1]))

    def _nearest_reachable_point(
        self, pose: tuple[float, float]
    ) -> tuple[tuple[float, float] | None, int]:
        """Nearest forward path point in clear line-of-sight from ``pose`` (or (None, -1))."""
        best: tuple[float, float] | None = None
        best_idx = -1
        best_d = float("inf")
        for j in range(self._idx, len(self.path)):
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
        # Advance the progress index past any path point we're within reach of, OR
        # that we've clearly overshot (the following point is nearer than this one, so
        # the vehicle has moved past it). Also honour the reached signal.
        while self._idx < len(self.path):
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
