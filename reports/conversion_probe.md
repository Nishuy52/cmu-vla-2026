# IF ordered-leg conversion probe (issue #71, post-#70 yardstick)

Baseline: `reports/gt_battery_post70` (already the current merged-main battery,
regenerated is unnecessary — it is HEAD's own evidence, commit 5d47cac).
`ordered_leg_credit=0.4389, headline=0.3778, threading_violations=7,
avoid_violations=0` — matches the numbers cited in the task brief exactly, so
this is used as-is instead of a redundant rerun.

Every leg NOT `reached_in_order` (39 of 72 total instruction-following legs
across 30 questions / 15 scenes) is classified per-leg-segment from the
battery's own `leg_probe` (`min_dist_driven_to_goal_m`,
`dist_goal_to_gt_traj_m`) against each leg's actual applied tolerance
(`tol_used`, which already varies 1.75–3.10 m per leg under the #70
STOP/PASS-BY split):

* **reached_in_order** (33/72) — baseline credit.
* **cascade_victim** (1/72) — `reached=True` (some driven pose is within
  tolerance of the goal) but `reached_in_order=False` (that pose came before
  the ordered cursor, i.e. an earlier leg's failure to advance the cursor
  left this leg's own arrival un-creditable in order).
* **arrival_blocked** (29/72) — never reached (`dist_driven > tol`), but the
  RESOLVED GOAL is itself within tolerance of the GT reference path
  (`dist_gt <= tol`) — the goal is plausible, our own driven trajectory
  simply never gets there.
* **structurally_unreachable** (9/72) — never reached, and the goal itself
  sits beyond tolerance of the GT reference path (`dist_gt > tol`) — even a
  perfect drive to our own goal would not have counted, because the goal
  we resolved is not where the GT demonstrator went.

Full per-leg data: `reports/conversion_probe.json`.

## Headline finding: cascade effects are negligible

Only **1 of 39** non-in-order legs is a true cascade victim (livingroom_1 leg
1: reached the goal, just too early relative to the ordered cursor). Because
`score_instruction_rubric`'s cursor only ADVANCES on an in-order arrival and
never advances on a miss, a failed leg makes every LATER leg's search window
*more* permissive (unchanged left edge), not less — so one leg's failure does
not lock out downstream legs the way "cascade" implies elsewhere in this
codebase's failure taxonomy. **Ranking root-cause fixes by "total legs
unlocked including cascade effects" collapses to ranking by bucket size
directly** — there is no cascade multiplier to chase here. This is itself the
useful phase-1 result: conversion is capped almost entirely by many
independent per-leg misses, not by a small number of route-derailing legs.

## Structurally unreachable (9 legs)

| scene | leg | kind | tol(m) | dist_driven(m) | dist_gt(m) |
|---|---|---|---|---|---|
| livingroom_1 | 0 | goto | 2.07 | 2.11 | 2.12 |
| livingroom_1 | 1 | corridor_between | 1.75 | 2.52 | 1.91 |
| studio | 0 | goto | 1.86 | 2.17 | 2.17 |
| arabic_room | 1 | corridor_between | 1.75 | 4.82 | 3.85 |
| arabic_room | 0 | goto | 2.16 | 7.45 | 5.67 |
| arabic_room | 0 | goto | 2.51 | 6.65 | 6.34 |
| home_building_1 | 1 | corridor_between | 1.75 | 6.84 | 7.99 |
| home_building_1 | 0 | goto | 2.24 | 11.89 | 13.01 |
| home_building_1 | 0 | goto | 3.10 | 4.01 | 14.82 |

Two legs (livingroom_1 leg0, studio leg0) sit only marginally over the
ceiling (excess 0.04–0.31 m) — resolver near-misses, not a systemic pattern.
The rest (arabic_room, home_building_1) are far over (2–12 m excess) —
consistent with wrong-instance resolution or a genuinely bad frame fit,
not a tolerance/geometry nit. `home_building_1`/`arabic_room` are also the
two scenes carrying the "interior walls unavailable / boundary-only costmap"
note, but that note affects PLANNING, not the goal-resolution distance
comparison here, so it is not the direct cause of these two scenes' structural
misses.

## Arrival-blocked (29 legs) — the dominant bucket

Sorted by how far the driven path fell short (`dist_driven - tol`):
`reports/conversion_probe.json` (`bucket=="arrival_blocked"`) has the full
sorted table; excess ranges from 0.002 m (a hair's-breadth miss,
`arabic_room` leg1) to 5.5 m (`loft`, a goal on the far side of the scene the
drive never approaches). 5 legs miss by under 0.1 m (near-misses); the
remaining 24 miss by 0.1–5.5 m, spread across essentially every scene (13 of
15 scenes have at least one arrival-blocked leg) — this is NOT concentrated
in one or two scenes or one failure signature; it is a broad, low-amplitude
navigation-quality gap.

## Root-cause investigation and TWO fixes attempted, both reverted

Given the flat cascade profile, the natural next question was: is
`arrival_blocked` a genuine navigation-quality gap, or a scoring-geometry
ARTIFACT (comparing the driven trajectory against a goal computed by a
DIFFERENT method than the one the driven trajectory was actually chasing)?
`core.runner.gt_battery._if_rubric_geometry` (the rubric's own goal
resolution, via `_nearest_free_goal`'s AABB edge-push) and
`core.heads.instruction.InstructionHead._goto_point`/`_via_point` (the real
navigation head's goal resolution, via `Costmap.nearest_reachable_point`
BFS) are two INDEPENDENT implementations of "the point a navigator stops at
for this leg" — a genuine risk of exactly this kind of measurement mismatch.

**Fix attempt 1 — substitute the head's own driven-toward goal for
GOTO/VIA_NEAR legs it grounded** (`_driven_leg_goal_overrides`, reading
`head._legs[i].geom` straight off the SAME head object that produced the
driven trajectory). Integrated battery: credit 0.4389 → **0.789**, headline
0.3778 → **0.706**. REJECTED and reverted despite the huge apparent gain: a
diff of old vs. new `leg_goals` showed many legs' goal moved by **4–12
metres** (e.g. `home_building_2` leg 12.05 m, `arabic_room` leg 6.39 m) — far
beyond what a same-instance projection refinement could produce. This proves
the head's anchor resolution frequently lands on a **different instance**
than the rubric's independent resolve (see fix 2's finding below), not just a
different reachable point on the same instance — substituting the goal wholesale
makes the rubric partly self-referential (scored against wherever the
pipeline's own planner decided to go, not an independent check of whether
that decision was any good). This is the exact "measures determinism, not
truth" trap the codebase already flags for the `pipeline_gt` metric, so it
does not ship even though the integrated numbers improved.

**Fix attempt 2 — mirror ONLY the head's anchor-selection tie-break**
(`InstructionHead._ranked_anchor`'s salience reorder: when the toolbox
resolve leaves an unbroken tie, the head re-ranks survivors by proximity to
the previous leg's own point; `_if_rubric_geometry` previously always took
the toolbox's raw top candidate). This keeps `_nearest_free_goal`'s own
independent projection untouched — only the anchor INSTANCE selection is
made consistent with the real pipeline. Integrated battery: credit 0.4389 →
**0.439** (unchanged), headline 0.3778 → **0.367** (down — one MORE
threading violation, 7 → 8). REJECTED and reverted: neutral-to-negative on
the integrated numbers, confirming that tie-break divergence is not a
material contributor to `arrival_blocked` in this battery (the large
goal-position deltas seen in fix 1 are not explained by simple tie-breaking;
they trace to genuinely different resolve outcomes the tie-break mirror does
not reproduce, most likely candidate-set differences from `self.thresholds`
plumbing or ordering the two call sites do not otherwise share and which
would need a deeper resolver-parity audit to pin down safely).

Per the mission's stop condition, two consecutive ranked fixes failing to
improve the integrated numbers on legitimate grounds is where this session
stops chasing `arrival_blocked` fixes. **Working conclusion: the dominant
remaining gap is a genuine navigation/planning shortfall (the driven route
does not get close enough to a plausible goal), not a scoring-geometry
artifact** — fix 2's near-zero effect after removing the tie-break variable
is the load-bearing evidence for this. A real fix needs planner-quality work
(route completeness/precision near a goal), which is out of safe reach for a
quick, principled, non-tuning change in the remaining budget; it is a
legitimate next-session target, not resolved here.

## Issue #67 ruling: OBSOLETE (YES)

#67's original bug: `hotel_room_1`'s GOTO leg had a raw (pre-push) goal
already within the OLD fixed 0.8 m tolerance of GT, but
`_nearest_free_goal`'s obstacle push moved it far enough that the leg then
missed. Under the NEW derived, per-leg tolerance (now 1.75–2.5+ m depending
on PASS-BY footprint widening, not a fixed 0.8 m), checking all of
`hotel_room_1`'s legs directly:

| leg | tol(m) | dist_gt (goal to GT path, m) | verdict |
|---|---|---|---|
| leg0 (Q1, bedside table) | 2.34 | 1.24 | plausible, well inside tol |
| leg1 (Q1, chair) | 1.75 | 0.52 | plausible, reached_in_order=True |
| leg0 (Q2, bedside table) | 2.34 | 1.11 | plausible, well inside tol |
| leg2 (Q2, picture) | 1.75 | 0.82 | plausible, arrival_blocked (drive-side) |

Every `hotel_room_1` leg's resolved (possibly pushed) goal sits comfortably
inside the new, much wider tolerance of the GT reference path — the specific
"push moved a plausible goal outside tolerance" failure mode #67 diagnosed
cannot reproduce here: the new tolerance band (1.75–2.5+ m) dwarfs
`_nearest_free_goal`'s bounded push distance (max 2.5 m, typically well
under a metre for a single-obstacle push) by enough margin that a push can
no longer be the deciding factor for these legs. The actual remaining
`hotel_room_1` failures are `arrival_blocked` (driven path doesn't reach a
goal that IS plausible) — a drive-side gap, not a goal-placement one.
**Ruling: #67 is OBSOLETE under the #70 yardstick** — no legs currently show
the raw-goal-reachable-but-pushed-goal-unreachable signature; the tolerance
widening subsumes it. Recommend closing #67 with this evidence.

## Phase 3 — threading (7 violations)

Cross-referencing threading verdicts against the same `leg_probe` distances:

| scene | leg | threaded | reached_in_order | dist_driven(m) | tol(m) |
|---|---|---|---|---|---|
| arabic_room | 1 | False | False | 4.82 | 1.75 |
| home_building_1 | 1 | False | False | 6.84 | 1.75 |
| home_building_2 | 0 | False | False | 2.16 | 1.75 |
| hotel_room_1 | 1 | False | **True** | 0.63 | 1.75 |
| hotel_room_2 | 0 | False | **True** | 1.15 | 1.99 |
| livingroom_1 | 1 | False | False | 2.52 | 1.75 |
| studio | 1 | False | False | 1.80 | 1.75 |

**Only 2 of the 7 threading violations are actually attempted corridors**
(`hotel_room_1` leg1, `hotel_room_2` leg0): the driven path gets close enough
to be scored ARRIVED at the gate midpoint (0.6–1.2 m, well inside tolerance)
but never literally crosses the gate segment. The other 5 corridors are not
approached at all (1.8–6.8 m away) — those are the same `arrival_blocked` /
`structurally_unreachable` legs already covered above; threading is moot for
them.

For the 2 attempted-but-unthreaded corridors, traced `core/nav/planner.py`'s
`#64` nudge (`plan_through`'s corridor_between branch, ~line 434–518): the
nudge only engages when `miss_dist < costmap.cell_m` (0.1 m) AND
`cur_dist >= costmap.cell_m` — a deliberately tight quantization-only guard
(the #54 regression the comment documents). Both attempted corridors miss by
0.6–1.2 m, six to twelve times the guard's 0.1 m band, so **the #64 nudge
correctly does not engage here — this is not a quantization near-miss, it is
a genuine "arrived near the midpoint without crossing between the two
anchors" case** (the vehicle got to the gate area from one side and stopped
short of actually passing between the anchors). No code defect traced in
`#64`'s guard itself. `hotel_room_2`'s `#63` `usable_gate_point` is called
consistently at both the scoring site (`_if_rubric_geometry`'s
`corridor_gate`) and the planning site (`plan_through`'s
`usable_gate_point` call at line 434) — both read off the SAME `Gate`
construction inputs (the two resolved anchor records), so no split-brain
gate definition was found for this case. **No threading fix identified or
attempted this session** — the corridors that are actually reached are
blocked by the same underlying drive-precision gap identified above (the
vehicle stops near the gate rather than continuing through it), not by a
traceable threading-logic bug.

## Final state

No code changes ship this session — both attempted fixes were reverted after
measurement (`fix1`: illegitimate self-referential improvement; `fix2`:
neutral/negative). Integrated numbers are UNCHANGED from the `reports/gt_battery_post70`
baseline: `ordered_leg_credit=0.4389`, `headline=0.3778`, `threading_violations=7`,
`avoid_violations=0`. Stop condition (b) — two consecutive ranked fixes failing to
improve the integrated numbers — reached before condition (a) (headline >= 0.55) or
(c) (90 min wall clock).

Evidence retained for the next session: `reports/gt_battery_fix1_headgoal/` (the
rejected head-goal-substitution run) and `reports/gt_battery_fix2_salience/` (the
rejected salience-tie-break run) — both are diagnostic artifacts, not shipped
results; neither corresponds to code currently in the tree.

## Recommended next steps (not attempted this session)

1. Planner-quality work on `arrival_blocked`'s drive-precision gap (29 legs,
   broad across 13/15 scenes) — likely needs route/goal-approach refinement
   in `core/nav`, not a scoring change. Any threshold touched here MUST go
   through the generalization protocol per `docs/calibration.md`.
2. Resolver-parity audit between `_if_rubric_geometry`'s `resolve()` call and
   `InstructionHead._ranked_anchor` beyond the tie-break mirrored in fix 2 —
   fix 1's 4–12 m goal deltas are unexplained by the tie-break alone and
   point at a real resolve-outcome divergence between the two call sites
   worth root-causing (NOT by substituting one goal for the other, as fix 1
   did) before it is trusted as a scoring input.
3. Close issue #67 with this session's ruling (OBSOLETE — see above).
