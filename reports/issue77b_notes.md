# Issue #77b — threading pose-convergence lead (dry, reverted)

Branch `if77b-gate-convergence`, worktree
`.claude/worktrees/if77b-gate-convergence`. Baseline:
`reports/gt_battery_if77_fix1_dwelltol` (credit 0.6000, headline 0.5389,
threading violations 7) — the merged state after #77's Fix 1
(`nav: derive leg-goal dwell release from the follower's own reach radius
(#77)`), which is what `main` currently ships. No `gt_battery_main_post77a`
existed to use as a separate baseline; this IS that state.

## Pose-level trace (concrete finding)

Instrumented `core.runner.gt_battery._drive_if_trajectory`'s own loop body
(scripts: `reports/issue77b_notes_work/trace_gate.py`,
`trace_gate2.py` — ephemeral debug tools, not part of the test suite) for
`hotel_room_2`'s "Go between the bench and the bed and stop at the lamp
closest to the fireplace" question (leg0 = `corridor_between`).

Dumped the driven poses and the gate segment directly:

```
gate p0=(0.1961, 1.4122)  p1=(-1.0616, 1.4640)
driven poses near the gate (perp = perpendicular distance to the gate LINE,
side = signed distance, same sign = same side of the line):
  (0.3555,-0.6062)  perp=2.0101  side=+2.5303
  (0.3077,-0.3608)  perp=1.7668  side=+2.2241
  (0.2600,-0.1154)  perp=1.5236  side=+1.9180
  (0.2285, 0.1327)  perp=1.2771  side=+1.6077
  (0.1751, 0.3769)  perp=1.0353  side=+1.3032
  (0.1397, 0.6244)  perp=0.7895  side=+0.9938
  (0.0469, 0.8565)  perp=0.5613  side=+0.7066   <- closest approach
  (0.1500, 0.4500)  perp=0.9633  side=+1.2126   <- final pose (retreats!)
```

Every driven pose sits on the SAME side of the gate line (`side` never
changes sign) — the trajectory never crosses. Closest approach is 0.56 m,
short of the line, then the trajectory retreats to 0.96 m on its final tick.

**Root cause, traced to `core.nav.breadcrumbs.BreadcrumbFollower`'s crumb
target, not the driving/controller loop.** The follower is aiming the whole
time at `path[21]` = `(-0.1812, 1.4277)` — this leg's `leg_goal_indices`
waypoint-of-record, which `core.nav.planner.plan_through` sets to the A*
path's own snapped endpoint at the gate's `usable_gate_point` midpoint. That
point sits essentially ON the gate line by construction (it IS the gate's
own via-point). At tick 6, the pose comes within `REACH_M` (0.8 m, per
#77 Fix 1) of `path[21]` — dwell releases — but in the SAME tick, `path[22]`
(the very next leg's own goal-of-record) is ALSO within `REACH_M` of the
SAME pose (0.42 m, since it happens to sit close to earlier, already-visited
ground — see "second finding" below), so BOTH leg goals release together,
`_idx` jumps past `len(path)`, and the follower returns `None` — no further
crumb is ever selected past the gate line, so the follower physically never
had a target beyond it. This is a stricter, more concrete version of what
#77's Fix 2 attempt already suspected ("driven trajectory's own poses never
get close enough... to cross a zero-width gate line") — it's not a
convergence-precision limit in some downstream controller stepping loop;
`_drive_if_trajectory` (`core/runner/gt_battery.py`) IS the pose-emission
loop, and it steps in constant `_DRIVE_STEP_M` increments directly toward
whatever `BreadcrumbFollower.advance` hands it. The gap is entirely in what
target `BreadcrumbFollower`/`plan_through` ever hands it: a point ON the
line, never a point PAST it.

**Second finding (a separate, pre-existing bug, not fixed here — flagged
for a future issue).** `path[22]` (leg1's own `goto` waypoint-of-record, "the
lamp closest to the fireplace") resolves to `(0.15, 0.45)` — identical to
an EARLIER point on the same route (`path[11]`), nowhere near the leg's own
scored rubric goal `(1.1106, 1.0142)` (`_if_rubric_geometry`'s
independently-computed goal for the same leg). The planner-side goal
resolution for this `goto` leg (`core.heads.instruction._goto_point`) landed
on a spatially wrong point — worth its own issue (not chased here: out of
this issue's scope, and touching `_goto_point`'s resolution risks the
frozen-adjacent `resolve()` boundary without a clear win in the remaining
budget).

## Fix attempted (dry, reverted)

**Hypothesis.** If a corridor leg's waypoint-of-record sits strictly on the
gate's FAR side by more than `BreadcrumbFollower.REACH_M` (0.8 m), then
ordinary proximity-based dwell release (the existing, already-landed Fix 1
mechanism — no new release logic needed) can only fire once the pose is
itself past the line, which geometrically guarantees the driven trajectory
crossed it en route (the follower drives straight kinematic steps toward
its target).

**Implementation.** `core/nav/planner.py`:
- `GATE_CROSSING_MARGINS_M = (REACH_M + 0.5, REACH_M + 0.2)` — imported
  `REACH_M` from `core.nav.breadcrumbs` (no cycle: breadcrumbs imports
  nothing from planner).
- `_raw_segment_clear` — a raw-obstacle (uninflated) line-of-sight test,
  reusing the same discipline as the existing `_raw_obstacle_blocked_xy`
  (the vehicle's own inflation halo routinely boxes in the cells right
  around a just-threaded gate — the same reason `_pinch_costmap`'s
  start-disc exemption exists — so gating this short hop on the INFLATED
  costmap would reject genuinely open floor).
- `_gate_crossing_extension` — for each corridor leg, once its mandatory
  gate-crossing A* segment lands on/at the line, tries a short straight hop
  along the gate's own NORMAL (oriented away from the leg's own approach
  side), at each candidate margin, accepting the first with clear raw floor.
- `_gate_extension_keeps_route_planned` — a one-leg lookahead safety guard:
  reject the extension if committing to the far-side point would strand the
  VERY NEXT leg (one bounded extra `astar` probe from the candidate point to
  the next leg's own target). Necessary because only `corridor_between` legs
  get the pinch-overlay gate-forcing fallback; an ordinary `goto` leg
  immediately after has no such affordance, so if it needs to re-cross the
  same gate, pushing further across first turns an otherwise-fine route into
  a dead end.
- Wired into `plan_through`'s corridor branch, after the existing crossing
  verification, before `leg_bounds.append(...)`.

Fast tier: green (`pytest` from `src/`, no failures) with the change present.

**Measurement — dry, zero effect, zero regression.** Integrated battery
(`reports/gt_battery_if77b_fix1_gateextend`): credit/headline/threading
violations byte-identical to baseline (`if_rubric=0.539`,
`if_thread_viol=7`). Diffed EVERY question's `leg_outcomes` between baseline
and this run: **0 diffs across all 75 questions** — not a single leg's
outcome changed anywhere in the battery, for or against.

**Why it measured exactly zero.** Traced why the extension never actually
committed for any of the 3 target legs (`hotel_room_1` leg1, `hotel_room_2`
leg0, `studio` leg1): in every case, the safety guard correctly rejected it.
Direct check for `hotel_room_2`: with the extension committed, the very next
(`goto`) leg's own target became unreachable via plain `astar`
(`P.astar(costmap, far_pt, next_goal) -> None`) — but reachable through a
`_pinch_costmap` overlay THROUGH THE SAME GATE
(`P.astar(pinch, far_pt, next_goal) -> <21-point path>`), confirming the
blocker is exactly "the next leg needs to re-cross the same gate, which only
`corridor_between` legs get help crossing." All 3 corridor legs in this
dwell-class cluster are FURNITURE-DEFINED virtual gates inside a single open
room (e.g. "between the bench and the bed", "between the TV and the bed") —
not real wall-separated doorways — and each is immediately followed by a
`goto` leg whose own resolved target sits back on the gate's near side
(confirmed for `hotel_room_1` leg2 and `studio` leg2 too: their goals'
`side` sign matches the approach side, not the far side). Making this fix
land for real would require also giving an ordinary `goto` leg
pinch-overlay gate-forcing help when its target requires re-crossing a gate
the route just threaded — a materially bigger structural change to
`plan_through`'s non-corridor branch, out of reach in the remaining budget
and risk profile for this session.

**Reverted** (`git checkout -- src/core/nav/planner.py`; fast tier
re-verified green after revert, diff clean). Counts as 1 dry fix (not the
2-consecutive stop condition) but combined with the session time budget,
stopping Phase 2 here.

## Per-leg ledger (unchanged by this session — informational)

| scene | leg | before | after | note |
|---|---|---|---|---|
| hotel_room_1 | leg1 (corridor_between) | threaded=False | threaded=False | next leg (goto, near-side goal) blocks the extension |
| hotel_room_2 | leg0 (corridor_between) | threaded=False | threaded=False | confirmed: next leg's goal only reachable via a same-gate pinch overlay, which goto legs don't get |
| studio | leg1 (corridor_between) | threaded=False | threaded=False | same shape; next leg's own goal sits on the near side |

## Integrated numbers

Baseline (`reports/gt_battery_if77_fix1_dwelltol`) == this session's final
state (change reverted): credit 0.6000, headline 0.5389, threading
violations 7, avoid violations 0. No change either direction.

## Files

- `src/core/nav/planner.py` — reverted to baseline (no net change).
- `reports/issue77b_notes_work/trace_gate.py`,
  `reports/issue77b_notes_work/trace_gate2.py` — ephemeral diagnostic
  scripts (pose-level trace tooling), kept for reference; not part of the
  scored pipeline or test suite.
- `reports/gt_battery_if77b_fix1_gateextend/` — the dry attempt's
  integrated battery run (kept as evidence the diff was truly zero).

## Tests

`pytest` from `src/` (fast tier): green, both with the dry change present
and after the revert.

## Deferred / next steps

1. **The real threading fix needs `goto`-leg gate-crossing help.** Concrete
   shape now known: extend `plan_through`'s non-corridor (`goto`/`via_near`)
   branch to retry via a `_pinch_costmap` overlay (through any gate the
   route has already threaded in this call) when its plain `astar` fails —
   then the already-implemented `_gate_crossing_extension` from this
   session's dry attempt could be reinstated and would very likely land the
   3 threading violations (confirmed reachable via pinch for at least the
   `hotel_room_2` case, directly demonstrated above). Scope: bigger than a
   single-constant swap; touches the shared tail of `plan_through`'s
   non-corridor branch, which every leg kind uses — needs care that it can't
   regress the many legs that never need it.
2. **`hotel_room_2` leg1's `goto` goal resolution bug** (separate from
   threading): `_goto_point`'s route-embedded resolution for "the lamp
   closest to the fireplace" (`core/heads/instruction.py`) lands at
   `(0.15, 0.45)`, coincident with an unrelated earlier path point, versus
   the rubric's own independently-computed goal `(1.1106, 1.0142)` for the
   SAME leg. Worth its own issue + trace — flagged, not filed as a GitHub
   issue this session (time budget), but should be before further chasing
   `hotel_room_2`-adjacent legs.
3. Everything else from #77's own "Deferred" section (the 22-leg broad
   "(e) other" residual) is untouched by this session.
