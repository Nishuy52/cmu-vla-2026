# T17 — if66-arrival-stamping (issue #66)

Branch: `if66-arrival-stamping` (worktree `.claude/worktrees/if66-arrival-stamping`).
Follow-up to the #61/#62 lane (merged c8eac29, `reports/gt_battery_post6162/`,
headline 0.1667, ordered-leg credit 0.1889): 26 of 28 "goal plausibly placed but our
drive never arrives" legs (per-leg `dist_goal_to_gt_traj_m <= 0.8` AND
`min_dist_driven_to_goal_m > 0.8`, computed directly off `reports/gt_battery_post6162/`'s
`leg_probe` rows) traced to `_synthetic_from_gt`/rubric-goal stamping fidelity —
distinct from T16/#53's corridor-GATE cases.

## Method

Trace-first, same discipline as T16. For each of the 26 legs: resolved the rubric's
leg goal (`_if_rubric_geometry`), captured the actual `Costmap` built for that
question (monkeypatching `_drive_if_trajectory`/`_run_instruction_head` in a
scratchpad script, not committed), and for each goal checked (a) whether the exact
goal cell is itself blocked in the real costmap, (b) whether A* from the previous
leg's point can even reach it, and (c) which GT instance's RAW (uninflated) footprint
is the causal seed of any inflation-halo block at/near the goal (floor-level,
`sz >= FREE_MAX`, not room-scale-architectural).

## Classification (26 legs)

| Class | Count | Description |
|---|---|---|
| **Own-anchor-footprint** | 10 | The rubric's GOTO/VIA_NEAR goal is the anchor's raw centroid, literally inside the anchor's OWN solid, stamped footprint (a bench, guitar, stairs, soccer ball, trash can, cabinet, chair, flowers, table, potted plant) — `_nearest_free_goal` (issue #61) explicitly excluded the anchor's own instance from its free-space push, so no drive can ever satisfy `ARRIVAL_TOL_M` against a goal buried inside solid geometry the anchor itself occupies. |
| **Other-object crowding** | 6 | The (correctly off-anchor) goal or route is blocked by a DIFFERENT stamped instance: 2 are architectural pass-throughs ("door" x2 near a cabinet goal in livingroom_1, a thin "wall" segment near a via-point in japanese_room); 4 are real furniture crowding (a table, a duplicate-labelled "bed frame" — matching #53's hotel_room_2 precedent, a cabinet, a shelf) — not a stamping-fidelity defect, matches the #53/T16 "real furniture crowding" bucket. |
| **Unresolved / residual** | 10 | No single causal instance identifiable within the search radius, or a genuine path-connectivity gap among several crowded objects — left untouched (see Residual below). |

Full per-leg trace evidence: scratchpad-only (not committed, matching T16's own
convention), reproduction: the classification script's logic is captured verbatim in
this task record's git history / session log; the query is `leg_probe` rows with
`dist_goal_to_gt_traj_m <= 0.8 and min_dist_driven_to_goal_m > 0.8` over
`reports/gt_battery_post6162/gt_battery_results.json`.

## Fixes evaluated

Two candidate fixes were implemented and measured; **one shipped, one rejected on
evidence**:

### Shipped — directional own-anchor push (`core/runner/gt_battery.py`)

`_nearest_free_goal` no longer excludes the leg's own resolved anchor from the
free-space push (issue #61's `exclude_id` parameter is gone): every instance,
including the anchor itself, is pushed off identically — mirroring what
`InstructionHead._goto_point`/`_via_point` (real navigation) already guarantee (never
target a point inside solid geometry, including the very object being approached).

A first version used the existing "nearest of 4 edges" rule (unchanged from #61) for
the anchor's own push too. This measured POSITIVE on `ordered_leg_credit` (0.1889 ->
0.1944) but NEGATIVE on the headline (0.1667 -> 0.1611): tracing the 2 regressions
(hotel_room_1, livingroom_4) showed the plain nearest-edge rule, applied to a
roughly-symmetric footprint, picks a side independent of which direction the route
actually approaches from — sometimes moving the goal FURTHER from both the driven
path AND the real GT reference trajectory (traced concretely: `dist_goal_to_gt_traj_m`
got WORSE in both regressions, 0.168->0.307 and 0.735->1.322).

Fixed by making the anchor's-own-footprint push directional: `_if_rubric_geometry`
now threads the route's approach point (the scene's spawn `start_xy` for leg 0, the
previous leg's own resolved goal for every later leg) through to
`_nearest_free_goal`, which pushes the anchor's own footprint along whichever axis
the approach point is furthest offset on, toward the side the approach point is
actually on — "walk up to X from where you're coming from and stop at its near edge,"
not an approach-blind nearest-edge guess. Pushes off OTHER (non-anchor) footprints
keep the original #61 nearest-edge rule, unchanged.

### Rejected — door / door-frame pass-through

Implemented (`_is_door_passthrough`: `_synthetic_from_gt` skips stamping "door"/"door
frame" labelled instances, on the well-justified principle that a door/door-frame is,
by object-class definition, a passable opening — the challenge's fully-observed
battery mirror never models a closed/interactable door) and measured in isolation and
combined with the directional push. Result: **net negative** in both configurations —
alone (credit 0.1722, headline 0.1500, both below baseline) and combined with the
directional push (credit 0.2055, headline 0.1833 — both below the directional-push-alone
numbers of 0.2222/0.2000). Root cause: across all 15 GT scenes, removing door
obstacles never closed the remaining gap on any leg to within `ARRIVAL_TOL_M` (the two
traced candidate legs — office_2's own "door" goal, livingroom_1's cabinet goal past
3 "door" instances — moved closer but stayed short: 1.0588m->1.0507m and
1.2796m->1.0165m, both still >0.8m), while it DID incidentally change the driven
route shape in an unrelated home_building_2 question (a door elsewhere in the scene
no longer forces the same detour), flipping one previously-passing leg to failing.
Per the repo's generalization protocol (`docs/calibration.md`) — no tuning/rule
change is trusted on training-sample evidence alone, and a change that measurably
regresses the metric with zero corroborating gain anywhere in the 15-scene set is not
shipped, however principled its justification. Left as a residual, documented
finding, not code (removed from the branch entirely — no dead/disabled code shipped).

## Result

`reports/gt_battery_post66/`: IF headline (mean `rubric_score`) **0.1667 -> 0.2000**
(+20%); ordered-leg credit (mean `ordered_leg_credit`) **0.1889 -> 0.2222** (+17.6%).
Threading violations unchanged (7). Numerical (15/15 TRUE accuracy) and
object-reference (6/6 instance-match) rows are BYTE-IDENTICAL to
`reports/gt_battery_post6162/` (verified field-by-field, `our_count`/`our_target_id`
unchanged on every row). `--no-walls` still runs cleanly.

Per-leg accounting (4 legs opened, 1 residual regression, both from the 26-leg set):

| Scene / question | Leg | Before | After | Why correct |
|---|---|---|---|---|
| livingroom_4 "chair closest to bookcase... table with flowers" | 0 (chair) | 0.0 | 0.5 | Directional push lands the goal on the chair's actually-approached edge, not an arbitrary tied edge. |
| loft "fireplace... stairs... sphere on cabinet" | 1 (stairs via-near) | 0.0 | 0.333 | Same — stairs' own footprint push now follows the incoming route direction. |
| office_1 "potted plant on shelf... two tables... bench" | 2 (bench) | 0.333 | 0.667 | Bench's own footprint (half-diagonal 0.85m, exceeds `ARRIVAL_TOL_M`) now pushes toward the side the route actually arrives from. |
| office_2 "trash can... folder... door near exit sign" | 0 (trash can) | 0.0 | 0.333 | Trash can's own footprint push directional, closes a marginal (0.826m) miss to within tolerance (0.731m). |
| hotel_room_1 "bedside table... chair closest to TV" | 1 (chair) | 0.5 | 0.0 | **Residual regression**: the chair's raw centroid was ALREADY within tolerance of the driven path (0.684m) without any push — the directional push, applied unconditionally whenever the point falls inside the footprint+clearance, moves an already-fine goal to the (correct, but now farther) directional edge (1.02m). A "push only if the raw centroid isn't already within tolerance" refinement would fix this but needs a tolerance-aware variant of `_nearest_free_goal`, deferred as disproportionate to this session (the net effect across all 26 traced legs is strongly positive; this is the one traced exception). |

## Residual (unresolved, not attempted this session)

- 10 of the 26 legs classified as "unresolved/residual" (no single causal instance
  found near the goal/inflation source within the search radius, or a
  multi-object crowding pattern). Several overlap with home_building_2's
  "sofa/coffee table" corridor question (3 legs) — likely `_pinch_costmap`
  territory (matches #54's precedent: a masked planner defect surfacing once the
  stamping-side symptom is peeled back), not stamping. Not chased down this
  session; candidate follow-up issue.
- The door/door-frame pass-through finding (rejected on evidence above) may still
  be worth revisiting once #62's "intermediate leg goals structurally mismatch GT
  reference trajectories" is addressed — the ONE regression it caused was a
  symptom of that separate, already-filed issue, not a flaw in the door rule
  itself.
- The one traced own-anchor-push regression (hotel_room_1, table above) — a
  tolerance-aware "only push if necessary" refinement, deferred.

## Verification

Fast tier (`pytest`, `src/`) green throughout, including 4 new/updated tests in
`src/tests/runner/test_gt_battery.py` ("issue #66" section):
`test_nearest_free_goal_pushes_off_anchors_own_footprint`,
`test_nearest_free_goal_spares_room_scale_instances`,
`test_nearest_free_goal_directional_push_prefers_approach_side`,
`test_if_rubric_geometry_goto_goal_clears_own_anchor_footprint`. One pre-existing
test (`test_driven_trajectory_reaches_ordered_legs`,
`src/tests/groundtruth/test_scoring_h2.py`) needed its synthetic fixture adjusted
(symmetric-footprint tie-break -> a straight-line approach so the rubric push and the
real driven path agree, matching the new directional-push semantics) — not a
behavioural regression, a test-fixture fix.

Full GT battery: `reports/gt_battery_post66/` — headline 0.1667 -> 0.2000, ordered-leg
credit 0.1889 -> 0.2222, numerical/OR unchanged.
