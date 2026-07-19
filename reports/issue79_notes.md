# Issue #79 — goto/via_near pinch-overlay gate-crossing fallback

Branch `if79-goto-pinch`, worktree `.claude/worktrees/if79-goto-pinch`.
Baseline: `reports/gt_battery_if77_fix1_dwelltol` (credit 0.6000, headline
0.5389, threading violations 7) — the merged state after #77b (dry, reverted;
`main` unchanged since).

## Design

Two pieces, both needed together (per 77b's "Deferred / next steps" §1):

1. **`plan_through`'s non-corridor (`goto`/`via_near`) branch gets a pinch
   fallback.** `plan_through` now tracks `last_gate` — the raw endpoints of
   the most recently threaded `corridor_between` leg's gate, persisting
   across subsequent legs in the same call. When a `goto`/`via_near` leg's
   direct `astar` returns `None` AND `last_gate` is set, it retries once via
   `_pinch_costmap` through `last_gate`. Engages ONLY when the direct route
   is wholly unreachable (constraint 1 — never a preference), and only
   commits if `_gate_extension_keeps_route_planned` (the reused #77b
   lookahead guard) confirms the leg immediately AFTER this one stays
   reachable from the candidate goal — either directly or via the same
   gate's own pinch overlay.

2. **Reinstated `_gate_crossing_extension`** (the #77b dry attempt, verbatim
   design, reconstructed from `reports/issue77b_notes.md` since the dry
   change was fully `git checkout --`-reverted and left no commit history).
   Every `corridor_between` leg's own mandatory gate-crossing segment lands
   ON the gate line (`usable_gate_point`) by construction; once verified
   threaded, a short straight hop (`GATE_CROSSING_MARGINS_M` =
   `(REACH_M+0.5, REACH_M+0.2)`, widest first) along the gate's own normal —
   oriented away from the leg's approach side — pushes the leg's
   waypoint-of-record strictly past the line, so `BreadcrumbFollower`'s
   existing proximity-based dwell release can only fire once the driven pose
   is itself past the line (this is the actual pose-level mechanism the
   77b trace diagnosed — root cause, not merely a symptom). Gated by
   `_raw_segment_clear` (raw/uninflated obstacle test, same discipline as
   `_raw_obstacle_blocked_xy`/#63) AND `_gate_extension_keeps_route_planned`
   for the leg immediately after. This is why piece (2) alone (77b's dry
   attempt) measured zero effect: the guard always rejected it because the
   following `goto` leg had no pinch help of its own. Piece (1) is what lets
   the guard start passing.

Both pieces reuse the SAME `_gate_extension_keeps_route_planned` helper
(one function, two call sites) — the crossing point for the guard's own
lookahead (when the next leg is itself `corridor_between`) is always
`usable_gate_point`, matching constraint 2. `core.nav.breadcrumbs.REACH_M`
is imported into `planner.py` for the crossing-margin scale (no import
cycle — `breadcrumbs` imports nothing from `planner`, confirmed via grep
before importing).

`core/nav/breadcrumbs.py` was NOT touched (frozen per brief).

## Per-scene regression table

Diffed EVERY question's `leg_outcomes` and `rubric_score` between baseline
(`reports/gt_battery_if77_fix1_dwelltol`) and this fix's full battery run.
**4 questions changed, all improved, zero regressions** across all 75
questions / 15 scenes:

| scene | question (truncated) | leg | before | after | rubric before→after |
|---|---|---|---|---|---|
| hotel_room_2 | "Go between the bench and the bed and stop at the lamp…" | leg0 (`corridor_between`) | `threaded=False` | `threaded=True` | 0.500 → 1.000 |
| hotel_room_1 | "First, go near the bedside table…take the path between the TV and the bed…" | leg1 (`corridor_between`) | `threaded=False` | `threaded=True` | 0.333 → 0.667 |
| studio | "First, go to the vase…take the path between the couch and the table…" | leg1 (`corridor_between`) | `threaded=False` | `threaded=True` | 0.333 → 0.667 |
| home_building_2 | "Take the path between the sofa and the coffee table…" | leg0 (`corridor_between`) | `threaded=False`, `reached_in_order=False` | `threaded=True`, `reached_in_order=True` | 0.000 → 0.333 |

`home_building_2` was NOT one of the 3 target legs from 77b's trace — an
unanticipated spillover win from the same mechanism (its leg0 is also a
`corridor_between` leg whose next leg's direct route needed a same-gate
re-cross). All 4 target/spillover legs are `corridor_between` legs (piece 2,
the reinstated extension) whose OWN threading flips; piece 1 (the goto pinch
fallback) is what unblocked the `_gate_extension_keeps_route_planned` guard
for all 4 — confirmed by re-running the guard's underlying astar/pinch
checks manually for `hotel_room_2` (documented in 77b's notes: direct
`astar(costmap, far_pt, next_goal)` is `None`, `astar(pinch, far_pt,
next_goal)` is a 21-point path).

No other leg in any of the other 71 questions changed `leg_outcomes` or
`rubric_score` (full diff script in the session, not committed — see
"Tests" below for the reproducible check).

## Integrated numbers

| | baseline (`gt_battery_if77_fix1_dwelltol`) | this fix (`reports/gt_battery_if79_goto_pinch`) |
|---|---|---|
| mean_rubric_score (headline) | 0.5389 | 0.5889 |
| mean_ordered_leg_credit | 0.6000 | 0.6111 |
| total_threading_violations | 7 | 3 |
| total_avoid_violations | 0 | 0 |

Success criteria: threading violations dropped below 7 (3 < 7, target met
— all 3 of 77b's target legs plus 1 spillover); credit did NOT regress below
0.6000 (0.6111, actually improved). Both satisfied.

hotel_room_2 direct evidence, isolated (`--scenes hotel_room_2`): baseline
`if_rubric=0.750 if_thread_viol=1`; fixed `if_rubric=1.000 if_thread_viol=0`
— confirms the exact case cited in the issue.

## Files

- `src/core/nav/planner.py` — `last_gate` tracking + goto/via_near pinch
  fallback in `plan_through`'s non-corridor branch; reinstated
  `_gate_crossing_extension` (+ `_raw_segment_clear`,
  `GATE_CROSSING_MARGINS_M`) wired into the `corridor_between` branch;
  reused `_gate_extension_keeps_route_planned` lookahead guard (new
  function, both call sites share it); `REACH_M` imported from
  `core.nav.breadcrumbs`.

## Tests

Fast tier (`pytest` from `src/`, no `-m`): green, including the #54 nudge
guard (`tests/nav/test_planner.py::test_plan_through_nudge_only_engages_when_direct_route_is_wholly_unreachable`)
unmodified and passing.

Full gate (`timeout 1800 <venv>/bin/python -m pytest -m "" -q` from `src/`):
green — exit code 0, all dots/skips, zero failures.

## #78 blast-radius spillover investigation (2 dry attempts, both refuted)

Mid-session, the coordinator flagged #78's stale/wrong-goal diagnosis
(`reports/issue78_notes.md`, merged to `main` separately) as a same-root-cause,
6-leg blast radius and asked this session to also extend `_goto_point`'s
(`src/core/heads/instruction.py`) reachability BFS through the previous
corridor leg's gate, so goal SELECTION and ROUTING share one notion of
reachability. Investigated with two bounded, evidence-based dry attempts
(probe scripts in `reports/issue79_notes_work/`, not part of the test suite)
before this note; **neither closes the gap — both refuted with numbers, no
source change landed for #78's mechanism.**

**Attempt 1 — relax `_goto_point`'s BFS via `_pinch_costmap` through the
leg's `prev_gate`, seeded at the grounding-time pose (spawn), mirroring
`plan_through`'s own `#54` relax-round schedule.** This is exactly what
#78's own diagnosis already tried once at a single fixed disc and reported
"still short by 1.18 m" — reproduced here across the FULL `MAX_PINCH_RELAX_ROUNDS`
schedule for `hotel_room_2` leg1 (`probe_goto_pinch.py`): pocket grows
524 → 1040 → 1456 → 1465 → 1465 cells (of 22,902), anchor cell never
enters the reachable set at any round, and `nearest_reachable_point` on the
relaxed pinch costmap is actually WORSE (1.178 m from the rubric goal,
vs. 1.114 m for the plain unrelaxed BFS). Confirms #78's finding stands
under the full relax schedule, not just its single-disc probe.

**Attempt 2 — seed the BFS from a point already past the gate line**
(`usable_gate_point` + `_nudge_past_gate`, the same construction
`_gate_crossing_extension` uses), instead of from spawn. Hypothesis: the
real asymmetry is that `_goto_point` resolves ALL leg goals up front from
the vehicle's spawn pose, while `plan_through`'s ACTUAL per-tick execution
re-plans from the vehicle's EVOLVING pose as it physically drives — so a
point only reachable after the vehicle has already crossed the gate can
never show up in a single BFS seeded at spawn, gate-pinch or not.
Reproduced: `reachable_mask` seeded at the nudged-past-gate point returns
the SAME 524-cell pocket as spawn's plain BFS — the nudge lands back in the
same connected component. Confirms the room is discononnected by REAL
(uninflated) geometry at this gate, not merely the vehicle's inflation halo
(matches #78's own `raw_blocked`-relaxed BFS attempt, which only grew to
1,472/28,614 cells and still excluded the lamp). Cross-checked against the
ACTUAL production driven trajectory (this session's own fixed `plan_through`,
`trace_gate2.py` re-run): the real drive's closest approach to the rubric
goal `(1.1106, 1.0142)` is `(0.0589, 1.1112)` — still ~1.05 m short in X —
confirming the driven trajectory genuinely never reaches that side of the
room even with #79's own fix in place; leg1 only scores `reached_in_order`
today because the arrival tolerance (1.7463 m) happens to be wider than the
miss, not because the goal or the route is actually correct.

**Conclusion.** The mechanism is NOT "one more gate to pinch through" —
something in the room's real (uninflated) geometry beyond the immediate
gate vicinity keeps that pocket disconnected even under the full pinch
relax schedule, and it is invisible to any single-shot, grounding-time BFS
because the real drive only ever gets there (if at all) through many ticks
of incremental re-planning from an evolving pose, a fundamentally different
computation than `_goto_point`'s one-shot resolve. Chasing this further
would mean either duplicating `plan_through`'s own per-tick replanning logic
into grounding-time goal resolution (a materially bigger, riskier change
than #79's scope), or reworking WHEN goal resolution happens relative to
routing — out of reach in this session's remaining budget and outside #79's
frozen/owned-file boundary.

**Recommendation:** keep `#78` open as its own issue (do not fold into
`#79`) — the blast-radius table in its notes remains accurate evidence, but
the fix it needs is a different, larger mechanism than "extend the
non-corridor branch's pinch fallback" (#79's actual, delivered, verified
scope). Checked which of the 6 blast-radius legs currently cost real credit
in this session's fixed battery: `studio` leg2, `hotel_room_1` leg2, and
`arabic_room` leg2 still score `reached_in_order=False` (their
`dist_to_rubric` exceeds `tol_used`); `hotel_room_2` leg1 and `livingroom_1`
leg2 already score `reached_in_order=True` today (their miss happens to sit
inside `tol_used`, so #78's bug costs them nothing currently);
`home_building_2` leg1 is a `pass_by` leg (different arrival semantics,
not a simple distance-to-goal comparison) and was not further
investigated. No regression risk either way: no `src/` change landed for
this investigation, so it cannot have touched #79's own verified numbers.

Files (evidence only, not part of the scored pipeline or test suite):
`reports/issue79_notes_work/probe_goto_pinch.py`.

## Deferred / next steps

- The separate `_goto_point` goal-resolution bug flagged in 77b's notes
  (`hotel_room_2` leg1's own `goto` target landing at a coincident,
  spatially wrong point in the FULL `_run_instruction_head` driven-sim
  path) is untouched — out of this issue's scope, does not affect the
  `plan_through`-level fix measured here (the battery scores
  `plan_through`'s own leg outcomes independent of that downstream bug).
