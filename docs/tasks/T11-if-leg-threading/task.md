# T11 — if-leg-threading

Diagnose and fix IF intermediate-leg threading — the top remaining
pipeline item (36 pts). Branch: `fix/if-leg-threading` (off main at
`b0c2518`, post-hardening-wave).

## Intent

Post-wave GT battery (`reports/gt_battery_postwave_2026-07-14/`) holds
the IF rubric-proxy flat at **0.061** (ordered-leg credit 0.094, 9
threading violations). The closed-loop driven-sim fix proved this is
genuine pipeline behavior, not a harness artifact. The CV sweep is
deliberately HELD until this lands (flat IF term = no gradient for the
×6-weighted objective). Backlog row: `docs/redteam/hardening_backlog.md`
"NEW: IF intermediate-leg threading".

## Failure signatures (from the post-wave per-question table)

1. **Ordered-leg arrival failure (dominant):** most `inst` rows score
   `legs=0/N` with `thread_viol=0` — driven trajectories never pass
   within 0.8 m of intermediate leg goals, even when terminal Fréchet
   is decent (office_2: Fréchet 1.25 m / cover1m 97% yet rubric 0.00).
2. **Corridor-gate misses:** 9 rows "trajectory never crossed the gate
   segment" on between/take-the-path legs.
3. **Motion artifacts:** studio "driven trajectory empty — pipeline
   produced no motion"; several rows with only 2–30 poses; two rows
   pinned at poses=4002 (watchdog-length).

Suspects named in the backlog: `plan_through` leg-goal chaining,
corridor-gate arrival handling, anchor-centroid reachability (leg goals
at object centroids may be unreachable within 0.8 m for large objects).

## Constraints (binding)

- Fix the PIPELINE, not the scorer. The rubric oracles (ordered per-leg
  arrival, threading_check, capsule_violated) are the trusted measuring
  instrument from H2 — they only change if a defect is proven in the
  scorer itself, documented separately.
- If a harness (driven-sim) defect is found, fix it, but report it
  separately from pipeline fixes.
- Scored-path changes must stay within the ai_module design and the
  test-time I/O contract (`docs/challenge_brief.md`).

## Acceptance criteria

1. Root cause(s) documented per failure signature in this record.
2. IF battery slice rerun on the same 15 scenes: mean rubric-proxy
   materially above 0.061 with the improvement traced to ordered-leg
   arrivals / gate crossings (not scorer changes); remaining gap
   explained. Report committed under `reports/`.
3. Numerical + OR toplines unchanged (56% / 0.875) — no regression from
   nav/FSM edits.
4. Fast tier green; full gate green except the known environmental
   non-passes; new regression tests pin the fixed behaviors.
5. Backlog ledger row updated in `docs/redteam/hardening_backlog.md`
   (threading item + sweep HELD status if now unblocked).

## Todo

- [x] Diagnosis (executor) — see notes + executor_report.md
- [x] Fix + tests (executor)
- [x] Battery rerun + report (reports/gt_battery_postT11_2026-07-14/)
- [x] Verifier pass (REFUTED narrowly — see verification.md; corrections below)
- [x] Post-verification corrections + perf/gate fix
- [ ] LOG one-liner, commit, PR

## Notes

- 2026-07-14: task opened; branch cut from main `b0c2518`.

### 2026-07-14 — Diagnosis (root cause per signature)

Reproduced on office_2 q4 (sig-1), hotel_room_2 q4 (sig-2, gate), and
studio q4 (sig-3, empty motion) via `_run_instruction_head` /
`_drive_if_trajectory` diagnostics over the GT mirror scenes.

**Common upstream cause — the GT->costmap mirror stamps every object as a
solid, floor-to-nothing AABB box** (`gt_battery._synthetic_from_gt` ->
`SyntheticScene.place_box`). VLA-3D AABBs over-approximate footprints and
include *tabletop* objects (a plant "on a cabinet", a "vase on a
cabinet") whose XY footprint sits on top of furniture — floor the robot
can actually stand beside is stamped OBSTACLE. With vehicle-radius
inflation (0.4 m) this (a) walls off leg goals into enclosed pockets and
(b) collapses between-object gates to zero width.

1. **Sig-1 (dominant, legs=0/N, thread_viol=0) — leg goal placed on
   unreachable floor.** `instruction._goto_point` projects the anchor
   centroid to the *nearest passable cell*, but "nearest passable" can be
   an isolated pocket disconnected from the drivable free-space component
   (office_2: plant-on-cabinet goal lands in a 6-cell alcove sealed by the
   cabinet cluster + a `glass wall` object; A* from spawn cannot reach it,
   confirmed by flood-fill: 2623 reachable cells, goal cell not among
   them). `plan_through`'s A* to that leg then fails -> recovery path
   drives straight to the terminal, skipping the intermediate leg
   entirely. The scorer independently uses the raw centroid
   (`_if_rubric_geometry`), also unreachable, so legs=0. **Pipeline
   defect: leg-goal projection must land on a cell reachable from the
   current pose, not merely any passable cell.**

2. **Sig-2 (9 gate violations) — between-object gate has zero passable
   width.** `corridor_gate` places the gate on the two anchors' facing
   AABB edges; when the anchors are close (bench beside bed) the whole
   gate segment + a ~1.7 m disc around it is solid obstacle after
   inflation (every sampled gate cell blocked). `plan_through`'s pinch
   fallback also fails (no free cell in the corridor), so the planned path
   routes around and never crosses -> thread_viol. **Pipeline/geometry
   defect: gate placement / mirror stamping leaves no threadable gap.**

3. **Sig-3 (empty motion / poses 0) — plural leg anchor fails to
   ground.** studio q4 leg 0 "the vases on the cabinet below the TV"
   (plural "vases") never resolves -> `_GroundedLeg.geom=None` -> route
   never commits -> follower None -> zero driven poses. **Pipeline
   defect: plural/collective leg anchor grounding.**

Secondary finding (scorer vs pipeline terminal disagreement, hotel_room_2):
`_if_rubric_geometry` resolves "lamp closest to the fireplace" to a
different instance (0.885, 1.42) than the head's grounded terminal
(0.15, 3.05). Noted; the scorer path uses a plain `resolve` while the head
uses ordered-salience re-ranking. Not the dominant driver — flagged for the
report, not changing the scorer (H2).

### 2026-07-14 — Achievable-ceiling measurement (why the headline stays modest)

Instrumented every IF leg goal two ways:
- **Reachable ceiling (our costmap):** only **13/71** leg goals have ANY
  passable cell reachable from spawn within 0.8 m of the raw centroid
  (15/71 after the mirror z-fix below). A perfect planner on this mirror
  tops out ≈0.18.
- **GT-reference ceiling:** the ground-truth trajectories themselves reach
  only **30/72** leg goals within 0.8 m of the raw centroid. Even the
  canonical correct path scores ≈0.42 ordered-leg credit.

Conclusion: the headline ceiling is set by (a) the scorer measuring arrival
to the raw *centroid* at 0.8 m — legitimately >0.8 m from any path for
large-object anchors (a scorer property, NOT a defect; GT hits it too), and
(b) the GT→costmap mirror stamping over-approximated **solid AABBs for every
floor object**, sealing drivable floor the real terrain leaves open (the
gap between our 15 and the GT's 30). (b) is a HARNESS-fidelity limit, not a
scored-path pipeline bug (the real challenge runs on real terrain, not this
mirror). Reported separately.

### 2026-07-14 — Fixes

PIPELINE (scored-path, `core/`):
- `heads/instruction._goto_point`: snap the leg goal to the nearest cell
  REACHABLE from the pose (`nearest_reachable_point` BFS), not merely the
  nearest passable cell — an isolated pocket made `plan_through` A* fail and
  the leg was dropped (sig-1). Single-BFS; falls back to centroid projection
  with no costmap. Regression: `tests/heads/test_if_leg_reachability.py`.
- `heads/instruction._near_thresh`: VIA_NEAR standoff = footprint half-diag +
  `VIA_NEAR_CLEARANCE_M` (0.45 m, ~a vehicle radius) instead of the fixed
  1.2 m offset, so a compact "near X" via lands inside the rubric's 0.8 m
  arrival band instead of overshooting it.
- `parsing/vocab.singularize`: "-es" plural of an e-final stem drops only the
  "s" ("vases"->"vase", not "vas"), so plural/collective leg anchors ground
  (sig-3 studio empty-motion). Regression in `tests/parsing/test_vocab.py`.

HARNESS (battery mirror, reported separately):
- `mocks/synthetic_scene`: `GTObject` gains a base height `cz`; `terrain_patch`
  treats an object whose base sits at/above `TERRAIN_SLAB_MAX_Z` (0.25 m) as an
  OVERHANG (free floor under/beside it), mirroring the real terrain slab
  filter. `runner/gt_battery._synthetic_from_gt` passes the true `z_min`.
  Previously every AABB (incl. tabletop plants/vases and the ceiling) was
  stamped floor-to-top, sealing floor near most anchors and collapsing
  between-object gates (sig-1/2). Regressions in `tests/mocks/`.

STILL HARNESS-BOUND (not fixed — mirror fidelity, needs real walls/terrain):
- 8 remaining gate violations: between-object gates blocked by an intervening
  floor object (hotel_room_2 "between bench and bed" is filled by a separate
  "bed frame" AABB) or by over-approximated AABBs — physically unthreadable in
  the mirror; no planner can cross them.
- Many GOTO/near legs whose anchor is sealed by neighbouring furniture AABBs
  (livingroom_4 chair reachable by the GT path but boxed in our mirror).

### 2026-07-14 — Post-verification corrections (verifier REFUTED narrowly)

Verifier `verification.md` confirmed C2,C3,C4,C5-safety,C7,C8 and REFUTED
three record claims. Corrections landed:

1. **Per-question movers fixed.** The report's "office_2 q4 0.00→0.50 / q5
   0.00→0.33" were fabricated (no such rows). Re-derived from the two report
   JSONs — the three IF rows that actually changed:
   - hotel_room_1 "Go to the bedside table closest to the window…" 0.000→0.500
     (legs 0/2→1/2, GAIN)
   - office_2 "First, go to the trash can near the cabinet…" 0.000→0.667
     (legs 0/3→2/3, GAIN)
   - hotel_room_1 "First, go near the bedside table closest to the bench…"
     ordered-leg 0.333→0.000 (legs 1/3→0/3, PER-QUESTION REGRESSION) — now
     disclosed. Mechanism: this leg is GOTO ("go near X" parses GOTO, not
     VIA_NEAR — verified), so the `_goto_point` reachable-projection moved the
     goal to a farther reachable cell past the scorer's 0.8 m centroid band;
     NOT the `_near_thresh` shrink. Watch-item for pocketed GOTO/near legs.
2. **Stall-guard claim corrected.** It catches TRANSLATIONAL wedges only; a
   moving-but-never-arriving follower still pads to the watchdog — postT11
   still shows poses=4002/4001 rows. No score impact. Report reworded.
3. **30/72 ceiling now reproducible.** New `core/runner/gt_leg_ceiling.py`
   emits per-question leg goals + GT-reference min distances;
   `reports/gt_battery_postT11_2026-07-14/gt_leg_ceiling.{json,md}` committed
   with the regenerate command + data root (git-ignored `data/vla3d/Unity`).
   Re-run confirmed **30/72** (0.4167) — original number stands.

Also found + fixed a REGRESSION the fast tier missed (all 3 refuted items were
report-only; this one is code): the original `e7ce1bb` reachable-goal
projection re-flooded the whole grid per leg per tick — the first full
`-n auto` gate FAILED 5 tests (perf: 242 s vs 30 s wall budget; + one
pre-existing `our_n_waypoints` gap). Fixed via `Costmap.reachable_mask`
(memoised per start cell) + a `Costmap.blocked` bound-check for the growable
grid + setting `our_n_waypoints` in the IF path. All 5 formerly-failing tests
pass; fast tier 1035 passed.
