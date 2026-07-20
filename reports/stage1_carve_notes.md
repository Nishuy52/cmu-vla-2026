# Stage 1 carve — working notes

Branch `if-stage1-carve`, worktree `.claude/worktrees/if-stage1-carve`.
Plan: `docs/proposals/pre_grounding_movement_plan.md`. Audit:
`reports/mirror_truth_audit/audit.md`. G1 granted (user, 20 Jul 2026).

## Piece 1 — border-padding artifact fix

**Root cause.** `_synthetic_from_gt` (`src/core/runner/gt_battery.py`) sized the
mirror `SyntheticScene`'s outer room rectangle as the GT instance AABBs' union
plus a flat `pad=1.5` (`_gt_footprint_bounds(gt, 0.0)` then `+/- pad`). That
assumes scene furniture reaches close to the room's true walls. It doesn't
always: arabic_room's furniture AABB envelope is `x in [-4.29, 4.13]`, but its
two GT reference trajectories (mapped through the fitted sim->object frame)
run to `x=7.00` and `x=8.42` — well past the padded rectangle's `x1=5.63`.
`SyntheticScene._is_wall`'s border-wall band (`wall_thickness=0.15`, inflated
+0.4 m vehicle radius) then sits INSIDE real GT-traversed space, and the GT
trajectory drives through a "wall" that is a pure mirror-construction
artifact, not GT-derived geometry — exactly what the audit found (all 56 of
arabic_room's wall-attributed blocked cells were `border`, 0 `interior`).

**Fix.** New `_scene_room_bounds(gt, frame, if_traj, *, pad=1.5)`: unions the
instance-AABB bounds with the scene's own GT reference trajectories
(`trajectory_qN.ply`, mapped via the fitted frame) before padding, so the
border can only move outward, never inward. Threaded through
`_synthetic_from_gt` as a new optional `room_bounds` param (`None` = exact old
behaviour, byte-identical — every pre-existing caller is unaffected) and
plumbed through `_drive_if_path` / `_run_instruction_head` /
`_drive_if_trajectory` / `score_scene` alongside the existing `wall_cells`
plumbing. Gated on the same trusted-fit threshold as wall derivation
(`WALL_FIT_MAX_RESIDUAL_M`), but NOT gated behind `--no-walls` — it's a
border-correctness fix, not an interior-wall-realism feature.

**Scoped to trajectories, not the full mesh — with evidence why.** First
attempt unioned with the scene's full `traversable_area.ply` mesh (same
evidence `_scene_wall_cells` already trusts for interior walls). That DID
clear the border artifact (0 border-blocked cells, all 15 scenes) but also
widened the arabic_room mirror's south edge by ~2.3 m more than any GT
trajectory ever needs (the mesh reaches y=-4.80; the trajectories only reach
y=-1.16) — collateral geometry with no border-intrusion defect to fix. That
extra free space changed A* route shape enough to flip one previously-passing
leg's arrival margin (1.7123 m vs a 1.7463 m tolerance — under 4 cm, sub-cell)
to a miss, i.e. a genuine regression (`if_rubric` 0.611 -> 0.594) despite the
mesh version also fixing an unrelated previously-failing leg (0.0 -> 0.333).
Net a wash in count but a real, measured aggregate regression — exactly the
plan's risk-register "carve reroutes a passing leg" risk, materializing
because the evidence source was broader than the defect. Rescoping to GT
trajectories alone (the literal defect — "the mirror's outer boundary must
not intrude on GT-traversed space") still clears every border cell in all 15
scenes but changes ZERO scored question outcomes — the safe, minimal fix.

**Unit tests** (`src/tests/runner/test_gt_battery.py`, `# issue #77 Stage 1`
section): `_scene_room_bounds` fallback with no frame / no trajectories,
expansion to cover a far trajectory point, no-shrink when trajectory is
already inside the old pad, `_synthetic_from_gt`'s `room_bounds` override
actually moving the stamped border (via `_is_wall`/`_in_any_room`, not just
the `Room` dataclass), and a byte-identical-default regression guard.

**Battery diff (piece 1 alone).**
- Baseline (fresh pre-change run): `reports/gt_battery_stage1_baseline/`
  (`if_rubric=0.611`, 3 threading violations, 0 avoid violations).
- Piece 1 applied: `reports/gt_battery_stage1_piece1/` — `if_rubric=0.611`
  (unchanged), 3 threading violations (unchanged), 0 avoid violations
  (unchanged). Full per-question diff (all fields: rubric_score,
  ordered_leg_credit, n_legs_reached_in_order, n_threading_violations,
  n_avoid_violations, iou, true_match) over all 75 questions: **zero
  differences**.
- Border-cell audit (read-only probe, `reports/mirror_truth_audit/scripts/
  mirror_audit.py` driven against the fixed source): border-wall-attributed
  blocked cells drop to 0 in all 15 scenes (was 69 total across arabic_room
  56 + studio 13 per the original audit; this session's own re-measurement
  under the fixed frame/wall-cell pipeline read 56 (arabic_room) + a few
  points of scene-to-scene variance — see the two per-scene tables logged in
  this session's tool output for exact figures. Interior-derived wall counts
  shift by a few cells in some scenes (livingroom_4, loft, office_2, studio)
  because the wider room boundary now samples/renders traversable-mesh-derived
  interior wall lattice cells that previously fell just outside the
  too-tight room and were never rendered — expected, not a regression, and
  doesn't touch any scored question).

Net: piece 1 removes a confirmed measurement-fidelity bug from all 15 scenes
with a fully clean (zero-diff) battery — commits standalone.

## Piece 2 — GT-trajectory carve

**Mechanism.** New `_carve_cells_along_trajectories(if_traj, frame, *,
radius_m=VEHICLE_RADIUS_M, cell_m=FLOOR_SPACING)`: for every point on every one
of a scene's GT IF reference trajectories (mapped through the fitted frame),
adds every lattice cell within `VEHICLE_RADIUS_M` (0.4 m, the existing
`core.nav.costmap.VEHICLE_RADIUS_M` constant — no new tunable) to a carved
set, using the same disc-offset construction `Costmap._inflate` uses and the
same integer lattice `_derive_wall_cells`/`extra_wall_cells` already use.

**Implementation choice — runner-side, with a minimal additive seam in
`gt_battery.py` only (per user decision 2).** Two blocking sources needed
carving:
- **Derived-wall cells** (`extra_wall_cells`): a plain set difference,
  `wall_cells = wall_cells - carved_cells`, done in `_synthetic_from_gt`
  before constructing the scene. No new machinery — the carve just removes
  entries from a set gt_battery.py already owns.
- **Stamped-object cells**: `SyntheticScene.footprint_contains` is evaluated
  analytically per grid point inside `terrain_patch()`, so there is no
  pre-stamp set to subtract from without duplicating that lattice logic.
  Runner-side-only would have meant re-implementing `_is_wall`/
  `footprint_contains`'s per-point geometry test in `gt_battery.py` just to
  know which points to null out before feeding them anywhere — an exact
  duplication the user's decision explicitly allows a minimal seam to avoid.
  Added `_GTCarvedScene(SyntheticScene)` (defined in `gt_battery.py`, core/
  mocks/synthetic_scene.py itself untouched): overrides only `terrain_patch`
  to re-check the BASE CLASS'S ALREADY-COMPUTED obstacle points against the
  carve set and zero their intensity, post-generation — a pure subtractive
  filter, not a new stamping path. `SyntheticScene.instances()` (feeds every
  other head/rubric path, e.g. object grounding) is untouched, so a carved
  cell never shrinks what a GT object IS, only what the MIRROR TERRAIN/
  costmap says is drivable there.
- Applied unconditionally (like `room_bounds`), independent of `--no-walls` —
  GT-trajectory carving is a border/terrain-correctness fix, not the
  interior-wall-realism feature that flag gates.

**Unit tests** (`src/tests/runner/test_gt_battery.py`, `# issue #77 Stage 1
(carve)` section, 7 tests): `_carve_cells_along_trajectories` empty without
frame/trajectories and covers exactly the vehicle-radius disc; derived-wall
cell removal via `_is_wall`; stamped-object terrain cleared at a carved cell
while an un-carved cell in the SAME footprint stays solid; GT
`InstanceRecord` AABB never shrinks; `carved_cells=None` byte-identical to
the plain `SyntheticScene`; and an end-to-end `_drive_if_trajectory` test
proving a carved corridor lets the driven path cross an otherwise-solid wall
band that an un-carved run (the pre-existing routing test) must still detour
around.

**Battery diff (piece 1 + piece 2 combined).**
- `reports/gt_battery_stage1_carve/` vs the pre-change baseline
  (`reports/gt_battery_stage1_baseline/`): `if_rubric` 0.611 -> **0.744**,
  `n_threading_violations` 3 -> 4, `n_avoid_violations` 0 -> 0 (unchanged).
- Full per-question diff (all fields) over all 75 questions: **zero score
  decreases anywhere** (checked `rubric_score`, `ordered_leg_credit`, `iou`,
  `true_match` — every changed field only ever increased or stayed the
  same). 9 questions gained (arabic_room, chinese_room, home_building_2 x2,
  hotel_room_1, livingroom_4, loft, office_1, office_2, studio) — several
  from partial credit to full 1.0.
- One new threading violation: `home_building_2`'s "Take the path between the
  sofa and the coffee table..." question's `corridor_between` leg 0 is now
  REACHED (arrival tolerance met, previously the leg was never even
  attempted because the route was blocked earlier) but the driven path
  doesn't literally cross the sofa/coffee-table gate segment ("threading:
  trajectory never crossed the gate segment") — a real, correctly-scored
  rubric penalty (rubric_score for that question is 0.6667, not 1.0, an
  IMPROVEMENT over the baseline's 0.3333, not a regression) that surfaces
  only because the leg is reachable at all now. Flagged, not fixed here —
  in scope for Stage 3 (route-quality behavior), out of scope for a
  measurement-fidelity carve.
- The audit's **11 flagged regression-risk questions**: individually
  checked (scene + question text cross-referenced against
  `reports/mirror_truth_audit/audit.json`'s per-scene `pocket.regression_risk`
  lists) — every one is unchanged: still `rubric_score=1.0`, all legs
  reached, zero threading/avoid violations, identical before and after.
- **14-leg pool reachability** (`reports/issue77e_notes.md`'s
  grounding-pose-BFS-disconnected pool): **10 of 14** legs now reach
  `reached_in_order=true` (full arrival credit) — `studio` leg1, `arabic_room`
  leg2, `office_2` leg1, `livingroom_4` leg0, `chinese_room` leg0,
  `home_building_2` leg1 and leg2, `hotel_room_1` leg0, `office_1` leg0,
  `loft` leg2. The remaining 4 (`chinese_room` leg1, `livingroom_2` leg1,
  `arabic_room` leg0, `loft` leg0) are still short of arrival tolerance, but
  3 of the 4 measurably closed distance (livingroom_2 4.31m -> 3.22m,
  arabic_room 6.17m -> 5.69m, loft 6.84m -> **2.47m**) — `chinese_room` leg1
  is unchanged (4.23m both), i.e. genuinely un-carvable from this mechanism
  (its blocker isn't a GT-trajectory-contradicted cell).

**Files touched:** `src/core/runner/gt_battery.py` (new
`_carve_cells_along_trajectories`, `_GTCarvedScene`, `carved_cells` param
threaded through `_synthetic_from_gt`/`_drive_if_path`/
`_run_instruction_head`/`_drive_if_trajectory`/`score_scene`),
`src/tests/runner/test_gt_battery.py` (7 new tests), `docs/calibration.md`
(G1 adjudication entry, Generalization-protocol section).

Net: piece 2 lands a substantial, GT-evidence-only, zero-regression IF gain
(+0.133 aggregate rubric) on top of piece 1's clean border fix — commits
standalone.
