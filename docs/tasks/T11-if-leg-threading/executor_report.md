# T11 IF-leg-threading — executor report

Branch: `fix/if-leg-threading`. Diagnosis + fix of the IF intermediate-leg
threading defect (top remaining pipeline item, ~36 pts). This report is the
self-contained hand-off; the dated working notes are in `task.md`.

## Root cause per failure signature

**Common upstream cause (harness):** the GT→costmap mirror
(`gt_battery._synthetic_from_gt` → `SyntheticScene.place_box`) stamped every
VLA-3D object as a solid AABB box with base on the floor, discarding the true
`z_min`. VLA-3D AABBs over-approximate footprints AND include tabletop /
wall-mounted objects (a plant *on* a cabinet at z≈0.83, a vase *on* a shelf).
Stamped floor-to-top with vehicle-radius inflation (0.4 m) this (a) sealed leg
goals into enclosed pockets and (b) collapsed between-object gates to zero
width. The real terrain stack does NOT do this (terrainAnalysis.cpp filters
`/registered_scan` to a ~0.2 m slab, so overhangs are absent from
`/terrain_map`).

1. **Sig-1 — ordered-leg arrival failure (dominant), `legs=0/N`
   thread_viol=0 — PIPELINE (+ harness).** `instruction._goto_point`
   projected the anchor centroid to the *nearest passable* cell, which can be
   an isolated pocket disconnected from the drivable free-space component
   (office_2 plant-on-cabinet: goal in a 6-cell alcove sealed by the cabinet
   cluster + a `glass wall`; A* from spawn cannot reach it — flood-fill: 2623
   reachable cells, goal not among them). `plan_through`'s A* to that leg then
   fails → the leg is dropped → the vehicle beelines to the terminal → the
   ordered leg scores 0. The pocket sealing itself is the harness cause above.

2. **Sig-2 — corridor-gate misses (9 rows) — HARNESS (mirror fidelity).**
   `corridor_gate` places the gate on the two anchors' facing AABB edges;
   when anchors are close (bench beside bed) the whole gate segment + a
   ~1.7 m disc around it is solid obstacle after inflation, OR an intervening
   floor object fills the gap (hotel_room_2 "between the bench and the bed" is
   filled by a separate `bed frame` AABB). Every sampled gate cell is blocked
   → physically unthreadable in the mirror; no planner can cross. This is
   mirror over-approximation, not a planner defect.

3. **Sig-3 — empty motion / poses=0 — PIPELINE.** studio q4 leg 0 "the vases
   on the cabinet below the TV" never grounded: `vocab.singularize("vases")`
   returned `"vas"` (the "-es" sibilant rule misfired on an e-final stem), so
   the anchor matched no `vase` instance → route never committed → follower
   None → zero driven poses. The two poses=4002 (watchdog) rows are a
   driven-sim harness artifact (a wedged follower padding to the cap).

**Secondary finding (scorer, NOT changed — H2):** for several legs the scorer
(`_if_rubric_geometry`, plain `resolve`) picks a different instance than the
head (ordered-salience re-ranking) — e.g. hotel_room_2 terminal. Reported,
not fixed; the scorer is the trusted instrument.

## Achievable ceiling (why the headline is modest, not a defect)

Instrumented every IF leg goal:
- **Our reachable ceiling:** only 13/71 leg goals had any pose-reachable
  passable cell within 0.8 m of the raw centroid (15/71 after the mirror
  z-fix).
- **GT-reference ceiling:** the ground-truth trajectories themselves reach
  only **30/72** leg goals within 0.8 m of the raw centroid. This number is now
  a REPRODUCIBLE committed artifact: `reports/gt_battery_postT11_2026-07-14/
  gt_leg_ceiling.json` (per-question/per-leg goal xy + GT min distance) and
  `gt_leg_ceiling.md`, regenerable with
  `python -m core.runner.gt_leg_ceiling --groundtruth <UNITY_ROOT> --out …`
  from `src/` (the artifact records the exact command + data root; the 15 GT
  scenes are the git-ignored `data/vla3d/Unity` fixtures, which is why a
  fresh-context checkout without them cannot reproduce it). Re-run confirmed
  30/72 (ratio 0.4167).

So the headline ceiling is bounded by (a) the scorer measuring arrival to the
raw *centroid* at 0.8 m — legitimately >0.8 m from any path for large-object
anchors (a scorer property the GT hits too, NOT a defect) and (b) mirror
over-approximation sealing floor the real terrain leaves open (the gap between
our 15 and the GT's 30 — a harness limit, not scored-path code; the real
challenge runs on real terrain).

## Changes

PIPELINE (scored `core/`, commit `e7ce1bb`):
- `core/heads/instruction.py` `_goto_point`: snap leg goal to nearest cell
  REACHABLE from the pose (`Costmap.nearest_reachable_point` BFS), single BFS,
  centroid-projection fallback with no costmap.
- `core/heads/instruction.py` `_near_thresh` / new `VIA_NEAR_CLEARANCE_M`
  (0.45 m): VIA_NEAR standoff = footprint half-diag + vehicle-radius clearance
  (was fixed 1.2 m), so a compact "near X" via lands inside the 0.8 m arrival
  band.
- `core/parsing/vocab.py` `singularize`: "-es" strip only for x/z/ch/sh/ss
  stems, so e-final stems drop just "s" ("vases"→"vase").

HARNESS (battery mirror, commit `e7ce1bb`, reported separately):
- `core/mocks/synthetic_scene.py`: `GTObject.cz` base height; `terrain_patch`
  treats a base ≥ `TERRAIN_SLAB_MAX_Z` (0.25 m) object as a free-floor
  overhang. `core/runner/gt_battery.py` `_synthetic_from_gt` passes true
  `z_min`.
- `core/runner/gt_battery.py` `_drive_if_trajectory`: net-progress stall guard
  (`_DRIVE_STALL_EPS_M`/`_DRIVE_STALL_TICKS`). **What it does / does not catch
  (corrected):** it breaks a *translational* wedge — a follower whose pose
  stops moving (< 0.05 m over 40 ticks). It does **not** catch a follower that
  keeps *moving* but never *arrives* at the terminal; that still pads to
  `_DRIVE_MAX_TICKS`. So `poses=4002/4001` rows persist in postT11 (both
  hotel_room_1 IF rows, studio q4) — the "poses=4002 artifact removed" wording
  in the earlier draft was WRONG. No score impact (arrival credit is
  independent of trailing padded poses), so the score-safety property holds.
- `core/runner/gt_battery.py`: set `rec.our_n_waypoints = rub.driven_n_poses`
  in the IF rubric path — a PRE-EXISTING full-gate failure
  (`test_score_scene_if_produces_two_numbers`, `our_n_waypoints is None`)
  independent of T11: the wave rebuilt IF scoring onto the rubric proxy and
  left this diagnostic field unset. Fixed in `commit 9e29366`.

Perf fix (commit `9e29366`, `core/nav/costmap.py`): the reachable-goal
projection above naively re-flooded the whole grid per leg per tick, a severe
regression — `test_instruction_following_happy_path_publishes_waypoint` blew
its 30 s wall budget at **242 s** (5 full-gate tests failed; the fast tier
missed it because they are `slow`-marked). Fix: `Costmap.reachable_mask`
memoises the BFS per start CELL on the costmap instance (rebuilt whenever the
map changes, so never stale), and `nearest_reachable_point` / `_goto_point`
consume it — one flood per grounding pass, not one per leg. Also hardened
`Costmap.blocked` to bound-check the `base_blocked` snapshot (the growable grid
can outrun it), which the new flood exposed as an `IndexError`.

Regression tests:
- `tests/heads/test_if_leg_reachability.py` (new) — reachable leg goal.
- `tests/parsing/test_vocab.py` — e-final plural singularisation.
- `tests/mocks/test_synthetic_scene.py` — elevated object = overhang; floor
  object still blocks.

## Before / after (battery)

`reports/gt_battery_postwave_2026-07-14/` → `reports/gt_battery_postT11_2026-07-14/`,
same 15 GT scenes / 75 questions.

| Metric | Before | After |
|---|---|---|
| IF rubric-proxy (headline) | 0.061 | **0.100** |
| IF ordered-leg credit | 0.094 | **0.122** |
| IF threading violations | 9 | **8** |
| IF avoid violations | 0 | 0 |
| Numerical independent agreement | 56% | 56% (unchanged) |
| Object-reference mean 3D IoU | 0.875 | 0.875 (unchanged) |

**Per-question movers (re-derived from the two report JSONs — corrected).**
Exactly THREE IF rows changed rubric/ordered-leg between postwave and postT11
(scene :: verbatim question):

| scene | question | rubric | ordered-leg | legs |
|---|---|---|---|---|
| hotel_room_1 | "Go to the bedside table closest to the window and stop at the chair closest to the TV." | 0.000→**0.500** | 0.000→0.500 | 0/2→1/2 (GAIN) |
| office_2 | "First, go to the trash can near the cabinet, then go to the folder on the cabinet closest to the whiteboard, and finally, to the door near the exit sign." | 0.000→**0.667** | 0.000→0.667 | 0/3→2/3 (GAIN) |
| hotel_room_1 | "First, go near the bedside table closest to the bench, then take the path between the TV and the bed to the picture closest to the TV." | 0.000→0.000 | **0.333→0.000** | 1/3→0/3 (REGRESSION) |

The earlier draft's "office_2 q4 0.00→0.50" and "office_2 q5 0.00→0.33" DO NOT
EXIST in the reports — that attribution was wrong; the real movers are the two
GAIN rows above. studio q4 ("vases…") now MOVES (poses 0→4001, grounding fixed)
but scored 0.000→0.000, so it did NOT contribute to the headline gain. The
+0.039 net is two gains (+0.5, +0.667) minus one regression (−0.333), over 30
IF rows.

**Disclosed per-question regression (watch-item).** hotel_room_1 "First, go
near the bedside table closest to the bench…" lost its first ordered leg
(0.333→0.000). Mechanism: this leg is a **GOTO** ("go near X" is parsed GOTO,
not VIA_NEAR — verified via `parse_regex`), so the cause is the `_goto_point`
reachable-projection, NOT the `_near_thresh` shrink: snapping the leg goal to
the nearest *pose-reachable* cell moved it far enough from the raw centroid
(scorer measures to the centroid) that the driven pose no longer passes within
0.8 m, whereas the old nearest-*passable* projection happened to stay close
enough for arrival credit. Net IF still +0.039, but this is a real trade-off:
**watch GOTO/near legs whose anchor sits in a pocket — the reachable standoff
can overshoot the scorer's centroid tolerance and lose ordered credit.** (The
`_near_thresh` VIA_NEAR shrink remains a separate watch-item for genuine
"take the path near X" VIA_NEAR legs, none of which moved in this battery.)

No numerical / OR regression (toplines identical; verified 0 per-question field
diffs).

Remaining gap: bounded by the reproduced 30/72 GT-reference ceiling
(`gt_leg_ceiling.json`/`.md`) — the 8 residual gate violations are
mirror-unthreadable gates (intervening floor objects / over-approximated
AABBs), and many `legs=0/N` legs are anchors sealed by neighbouring furniture
AABBs (reachable by the GT path, boxed in our mirror). Harness-fidelity limits
(need real walls/terrain), not scored-path bugs.

## Test results

- Fast tier (`python -m pytest`): `1035 passed, 5 skipped, 55 deselected`
  (was 1030 pre-T11; +5 new regression tests; deselected 54→55 as the new IF
  battery slow test joined the slow set).
- Full gate (`python -m pytest -m "" -n auto`): the FIRST full-gate run (on the
  original commit `e7ce1bb`) FAILED 5 tests — root cause the per-tick reflood
  perf regression (242 s vs 30 s wall budget) described above, plus one
  pre-existing `our_n_waypoints` gap. After the perf/gate fix, the 5 formerly
  failing tests all pass individually
  (`test_score_scene_if_produces_two_numbers`,
  `test_capsule_engulfs_start_and_gate_recovery_never_violates`,
  `test_build_route_fallback_drive_never_violates_capsule`,
  `test_instruction_following_happy_path_publishes_waypoint`,
  `test_simulated_clock_keeps_wall_far_below_sim_time`), and the fast tier +
  `tests/{nav,heads,parsing,mocks}` (385 passed) are green. A fresh full
  `-n auto` gate is delegated to the verifier (not re-run here per the
  no-heavy-suites directive).

## Unresolved / flagged

- Scorer vs pipeline instance disagreement on some legs (secondary finding) —
  not fixed (scorer is the trusted instrument, H2). If pursued later, align by
  making the head's leg resolution match `_if_rubric_geometry`'s plain
  `resolve`, or vice-versa — a design choice, not a bug.
- Mirror fidelity (solid over-approximated AABBs, no interior walls) is the
  dominant remaining ceiling-limiter. A faithful mirror (per-object footprint
  inset, real walls) or scoring on real sim terrain would raise the ceiling;
  out of scope for T11 (harness, not scored path).
- GOTO reachable-standoff trade-off (watch-item): snapping a GOTO leg goal to
  the nearest pose-reachable cell can overshoot the scorer's 0.8 m centroid
  tolerance for a pocketed anchor (cost hotel_room_1 q1 leg 0, −0.333). Net IF
  still positive; revisit if a later scene shows it dominating.
- Never-arriving-but-moving follower: `poses=4002/4001` watchdog-padded drives
  persist (the stall guard only catches translational wedges). A latent
  driven-sim inefficiency (runtime, and a sign some routes loop without
  arriving); no score impact. A "no arrival-progress over N ticks" guard would
  catch it — deferred.
