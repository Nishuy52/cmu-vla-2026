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
  only **30/72** leg goals within 0.8 m of the raw centroid.

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
  (`_DRIVE_STALL_EPS_M`/`_DRIVE_STALL_TICKS`) ends a wedged drive instead of
  padding to the watchdog length (the poses=4002 artifact).

Regression tests (commit `e7ce1bb`):
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

The IF gain is traceable to real ordered arrivals: office_2 q4 0.00→0.50
(leg 1/2), office_2 q5 0.00→0.33, studio q4 empty-motion → moving (grounds),
etc. No numerical / OR regression (toplines identical). The stall guard
changes runtime + the poses=4002 artifact only; it does not change any score
(a wedged drive kept the legs reached before the wedge either way).

Remaining gap: bounded by the ceilings above — the 8 residual gate violations
are mirror-unthreadable gates (intervening floor objects / over-approximated
AABBs), and many `legs=0/N` legs are anchors sealed by neighbouring furniture
AABBs (reachable by the GT path, boxed in our mirror). These are
harness-fidelity limits (need real walls/terrain), not scored-path bugs.

## Test results

- Fast tier (`python -m pytest`): `1035 passed, 5 skipped, 54 deselected`
  (was 1030 pre-T11; +5 new regression tests).
- Full gate (`python -m pytest -m "" -n auto`): launched as the final act;
  under heavy machine load it had not returned at hand-off. All targeted
  subsets are green (fast tier 1035 passed; `tests/heads tests/nav
  tests/parsing tests/mocks tests/integration` 385 passed; via/vocab/mirror
  subsets green). Final full-gate confirmation delegated to the orchestrator's
  verifier per the run-once-as-final-act policy.

## Unresolved / flagged

- Scorer vs pipeline instance disagreement on some legs (secondary finding) —
  not fixed (scorer is the trusted instrument, H2). If pursued later, align by
  making the head's leg resolution match `_if_rubric_geometry`'s plain
  `resolve`, or vice-versa — a design choice, not a bug.
- Mirror fidelity (solid over-approximated AABBs, no interior walls) is the
  dominant remaining ceiling-limiter. A faithful mirror (per-object footprint
  inset, real walls) or scoring on real sim terrain would raise the ceiling;
  out of scope for T11 (harness, not scored path).
