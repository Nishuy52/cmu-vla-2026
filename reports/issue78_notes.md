# Issue #78 — stale/spatially-wrong goto goal (diagnosis, handoff to #79)

Branch `if78-stale-goal`, worktree `.claude/worktrees/if78-stale-goal`.
Baseline: `main` (post-#77b merge, `f240887`) — no `gt_battery_main_post78`
run needed; no source change lands from this session (diagnosis-only,
handoff).

## Repro

`core.runner.gt_battery._run_instruction_head` for `hotel_room_2` question 0
("Go between the bench and the bed and stop at the lamp closest to the
fireplace"), then inspect `head._legs[1].geom` vs.
`core.runner.gt_battery._if_rubric_geometry(...)`'s `leg_goals[1]`. Script:
`reports/issue78_notes_work/trace_goto.py` (ephemeral, not part of the test
suite).

Confirmed exactly as filed: `head._legs[1].geom == (0.15, 0.45)`, numerically
identical to `follower.path[11]` (an EARLIER point on the leg0 corridor
approach, visited well before the gate). The rubric's independently-computed
goal for the same leg is `(1.1106, 1.0142)` — 1.11 m away, on the far side of
the leg0 gate (bench/bed virtual corridor).

## Root cause — traced to the source, NOT a cache or frame bug

Not a stale cache (no caching bug — `Costmap.reachable_mask`'s memoisation is
keyed per start CELL per costmap instance and is correctly invalidated:
verified it recomputes fresh for a different start cell). Not a frame bug
(the (0.15, 0.45) point is in the same map frame as everything else — it
just happens to be a real, previously-visited point in that frame). The
"stale" appearance is a coincidental ARTIFACT of a real reachability
mismatch:

1. `InstructionHead._goto_point` (`src/core/heads/instruction.py:438`)
   resolves a GOTO leg's floor target via `Costmap.reachable_mask` /
   `Costmap.nearest_reachable_point` (`src/core/nav/costmap.py:238,271`): an
   8-connected BFS flood over the PLAIN (fully-inflated, non-pinch)
   `passable()` predicate from the grounding pose, then argmin-nearest (in
   grid-cell distance) to the anchor.
2. For `hotel_room_2`'s route, leg0 is a `CORRIDOR_BETWEEN` leg (a
   furniture-defined virtual gate between the bench and the bed, not a real
   wall doorway). Per #77b's own trace, the driven trajectory (and the
   route's OWN `follower.path`) never actually crosses this gate — every
   path point up to and including `path[21]` (leg0's waypoint-of-record)
   sits ON or before the gate line. `plan_through` only got that far via its
   `_pinch_costmap` overlay (`src/core/nav/planner.py:187`), which is a
   LOCAL, temporary relaxation (clears inflation-only cells in a narrow band
   around the gate, scoped to that one A* call) — it is never folded back
   into `head._costmap`, the base costmap `_goto_point` reads.
3. Measured directly (`reachable_mask(pose)` on `head._costmap`): only 524
   of 22,902 passable cells in the room are reachable from the spawn pose
   under the plain (non-pinch) BFS — a small pocket entirely on the near
   side of the leg0 gate. The lamp's anchor cell, the leg0 gate's own p1
   cell, and the rubric's independently-computed goal cell are ALL outside
   this pocket (confirmed `seen[r,c] == False` for all three).
4. `nearest_reachable_point` therefore falls back to "least-bad point in the
   near-side pocket nearest (by raw grid distance) to the far-side lamp" —
   and the true nearest boundary point of that pocket to the lamp happens to
   numerically coincide with `path[11]` (both sit at the pocket's edge
   closest, in Euclidean terms, to the unreachable far-side goal). Not a
   bug in the argmin/tie-break itself — a faithful "nearest reachable" answer
   to a fundamentally wrong reachability set.

**Tried and confirmed does NOT fix it within `Costmap`/`instruction.py`
alone:**
- Using `prev_xy` (leg0's own gate endpoint, already threaded through
  `_ground_legs` for anchor salience but never used as the reachability BFS
  seed) instead of the raw pose as the BFS start: no change — the gate
  endpoint IS on the near-side/gate line itself (not past it), so it snaps
  into the same pocket.
- Relaxing the plain BFS to ignore inflation entirely (`raw_blocked` instead
  of `base_blocked`): grows the reachable pocket only to 1,472 of 28,614
  raw-passable cells — still excludes the lamp/rubric-goal area. Real
  (uninflated) obstacle geometry, not just the vehicle's inflation margin,
  fragments this room at the gate.
- Explicitly reusing `core.nav.planner._pinch_costmap` (read-only, NOT
  modified) with the leg0 gate + spawn pose to build a locally-relaxed
  reachable set: grows the pocket to 1,040 cells but STILL does not reach
  the lamp/rubric-goal area (`nearest_reachable_point` on the pinch-relaxed
  costmap returns `(0.55, 2.05)`, 1.18 m from the rubric goal — no better).
  This is the planner's OWN best attempt at reachability and it still comes
  up short for this specific leg pairing.

That last result is the same failure #77b's dry `_gate_crossing_extension`
attempt already hit and diagnosed: **for `hotel_room_2` leg1 specifically,
reaching the true goal requires re-crossing (or further navigating past) the
SAME gate a SECOND time in a way only `plan_through`'s corridor branch gets
pinch-overlay help for — an ordinary `goto` leg gets none.** #77b's notes
call this out explicitly as deferred future work item #1: *"extend
`plan_through`'s non-corridor (`goto`/`via_near`) branch to retry via a
`_pinch_costmap` overlay ... when its plain `astar` fails."* That deferred
item is now filed as **issue #79** (`planner: goto/via_near legs need
pinch-overlay gate-crossing when the target requires re-crossing a threaded
gate`), already open, already scoped to exactly this mechanism, already
assigned to the concurrent #79 lane, and already flagged by #79's own body
as touching `plan_through`'s SHARED non-corridor tail (every leg kind uses
it) — the frozen surface for this session.

**Conclusion: the mechanism lives in `core/nav/planner.py`
(`plan_through`'s non-corridor branch + `_pinch_costmap`), which is frozen
for this session and owned by #79. Per the task brief, stopping after
diagnosis rather than attempting a workaround fix in `costmap.py`/
`instruction.py` — every workaround tried above either fails to actually
fix the leg (still short of the true goal) or would duplicate planner-owned
pinch semantics (`PINCH_DISC_M`/`PINCH_CORRIDOR_HALF_W_M` calibration seams,
`_pinch_costmap`) into a second module, risking drift from the real fix
#79 will land.**

## Blast radius

NOT a single-leg issue. Same mechanism (a `goto` leg immediately following
a `corridor_between` leg whose gate the driven route/base costmap never
actually crosses) reproduces across **6 of 59 `goto` legs, in 6 distinct
scene/question pairs** — scanned every IF question's every `goto` leg's
resolved geom vs. `_if_rubric_geometry`'s independently-computed goal
(`reports/issue78_notes_work/scan_all_goto2.py`, full dump in
`goto_scan2.json`):

| scene | q | leg | head goal | rubric goal | dist (m) | BFS pocket / room |
|---|---|---|---|---|---|---|
| hotel_room_2 | 0 | 1 | (0.15, 0.45) | (1.1106, 1.0142) | 1.114 | 2.3% |
| arabic_room | 1 | 2 | (3.75, 0.85) | (2.8206, -0.661) | 1.774 | 2.3% |
| hotel_room_1 | 1 | 2 | (-0.85, -1.35) | (-2.5335, -3.0735) | 2.409 | 2.3% |
| livingroom_1 | 1 | 2 | (-3.25, -6.55) | (-2.2339, -5.5438) | 1.430 | 30.0% |
| home_building_2 | 1 | 1 | (2.45, 9.65) | (2.074, 11.119) | 1.516 | 36.8% |
| studio | 1 | 2 | (1.45, -1.15) | (3.299, -3.9285) | 3.338 | 4.3% |

All 6 are `goto`/`via_near` legs immediately after a `corridor_between`
leg — exactly #79's described mechanism. Every one of #77b's 3 originally
identified corridor legs (`hotel_room_1` leg1, `hotel_room_2` leg0, `studio`
leg1) has its FOLLOWING leg in this table, plus 3 more corridor-adjacent
legs #77b's narrower dwell-cluster analysis didn't examine
(`arabic_room`, `home_building_2`, `livingroom_1`).

The other 31 flagged (`dist_to_rubric > 1.0 m`) `goto` legs in the full
59-leg scan (`goto_scan.json`/`goto_scan2.json`) do NOT share this
mechanism — most are `prev_leg_kind is None` (the route's FIRST leg) with a
tiny BFS pocket from spawn (e.g. `loft` q0 leg0, 0.44% of the room
reachable) — a pre-existing, already-tracked `structurally_unreachable`
classification (see `reports/issue77_notes_work/legs.json`'s `bucket`
field), not a "stale point" artifact, and out of #78's scope. A few others
reflect the rubric's `_nearest_free_goal` and the head's `_goto_point` using
genuinely different, legitimate projection strategies (footprint-push vs.
pose-reachability-BFS) that can diverge on a large/awkwardly-shaped anchor
even with no gate involved — also not this bug.

## Fix / handoff

No fix lands from this session. `core/nav/planner.py` (`plan_through`'s
non-corridor branch + `_pinch_costmap`) is frozen here and is exactly what
issue #79 already tracks, already scoped correctly (its body independently
arrived at the same "goto legs get no pinch-overlay fallback" mechanism from
the threading-violation angle). Landing #79 should fix (or at minimum
measurably improve) all 6 legs in the blast-radius table above, since it is
the SAME leg-adjacency-to-gate pattern in both directions (threading
violation on the corridor leg AND wrong-goal on the following goto leg are
two symptoms of the same missing pinch-fallback).

**Recommendation for the #79 lane:** when validating the #79 fix, also
re-run this session's leg-goal comparison
(`reports/issue78_notes_work/scan_all_goto2.py`) against the 6-row table
above — a real fix should shrink `dist_to_rubric` toward 0 for these legs (or
correctness-over-score: land the goal ON the far side, even if still short
of the exact rubric point), not just close the threading-violation count.
Suggest closing #78 as a duplicate-root-cause of #79 once #79 lands and this
table is re-verified, rather than tracking it as a separate fix.

## Battery

No integrated battery run — no fix landed, so no `gt_battery_post78`
artifact. `pytest` fast tier: green (no source touched; ran to confirm no
accidental drift while diagnosing).

## Files

- `reports/issue78_notes_work/trace_goto.py` — single-leg repro/trace
  (ephemeral, not part of the test suite).
- `reports/issue78_notes_work/scan_all_goto.py`,
  `scan_all_goto2.py` — blast-radius scan across every IF question's `goto`
  legs (ephemeral, not part of the test suite).
- `reports/issue78_notes_work/goto_scan.json`,
  `goto_scan2.json` — full scan dumps (evidence for the blast-radius table).
- No `src/` files modified (diagnosis-only session; `core/heads/instruction.py`
  and `core/nav/costmap.py` were read and probed interactively but no
  workaround was landed there, per the reasoning above).

## Tests

`pytest` from `src/` (fast tier): green — unaffected, since no source
change was made this session.

## Stop condition

Stopping after diagnosis per the task brief's planner.py-frozen rule: the
mechanism traces conclusively into `core/nav/planner.py`'s non-corridor
branch / `_pinch_costmap`, which is owned by the concurrent #79 lane and
already tracks this exact fix.
