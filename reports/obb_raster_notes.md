# Issue #77 Pre-Stage 1a — OBB rasterization (rotated-rectangle stamp)

Per `docs/proposals/pre_grounding_movement_plan.md` §0/§5 (user decision 1,
20 Jul 2026): rasterize the ORIENTED GT box in the mirror costmap stamping
path instead of its AABB hull. No new thresholds; runner-side per decision 2.

## Where OBB data flows

- `core/groundtruth/loader.py::parse_object_csv` already computes
  `obb_to_aabb(center, extents, heading)` for `aabb_min/aabb_max` (the frozen
  scoring-side approximation). This session adds three ADDITIVE fields on
  `InstanceRecord` (`core/interfaces.py`) — `obb_center`, `obb_extents`,
  `obb_heading` — populated by the loader alongside the existing AABB fields.
  Every other `InstanceRecord` producer (mocks, perception/tracker,
  scene_index) leaves them at their defaults (`None`/`None`/`0.0`), which
  reads as "no OBB info" downstream.
- `core/runner/gt_battery.py::_synthetic_from_gt` (the mirror stamping path)
  now reads `rec.obb_heading`/`obb_center`/`obb_extents` when heading is
  non-zero, using the OBB's own centre/local extents (not the AABB hull's)
  for the stamped box, and passes `heading` through to `sc.place_box`.
- `core/mocks/synthetic_scene.py::GTObject` gains a `heading` field (default
  0.0) and `footprint_contains` does an exact rotated-rectangle point test
  (rotate the query point into the box's local frame, compare to half-extents)
  instead of an axis-aligned one. `terrain_patch` (the actual per-cell
  rasterization loop) already calls `footprint_contains` per grid cell, so no
  separate scan-fill algorithm was needed — the existing per-cell membership
  test IS the rasterizer once it can do a rotated test.
- `place_box` gained an optional `heading` kwarg (default 0.0), so every
  existing caller (mocks `populate_default`, other `_synthetic_from_gt`
  callers with heading-less `InstanceRecord`s) is unaffected.

Frozen-surface note: `scoring.py`/`arrival.py`/toolbox geometry (corridor
gates, footprint predicates) all keep reading `aabb_min/aabb_max` unchanged —
this fix touches ONLY the stamping path (`gt_battery.py` + `synthetic_scene.py`
+ the additive loader fields). **Asymmetry flagged for follow-up
adjudication**: the rubric's own geometry (corridor_gate, in/on/near
predicates in the toolbox) still reasons over the AABB-of-OBB hull for every
rotated object, while the mirror costmap the SAME battery uses to plan/drive
now reasons over the true footprint. This is intentional per the user's scope
decision (stamping-side only) but is a genuine geometric asymmetry between
"what blocks the vehicle" and "what counts as inside/near/on" for rotated
objects — worth a dedicated follow-up decision on whether toolbox geometry
should eventually get the same treatment.

## Regression guard (heading == 0)

`GTObject.footprint_contains` takes the `lx, ly = dx, dy` branch with no trig
call at all when `heading` is falsy — bit-for-bit identical to the pre-fix
axis-aligned test. `_synthetic_from_gt` only overrides `cx/cy/sx/sy` off the
OBB's own center/extents when `heading` is truthy; heading==0 objects use the
exact same AABB-derived center/size computation as before. Verified by
`test_synthetic_from_gt_zero_heading_is_byte_identical_to_aabb_stamp` (full
object list + `terrain_patch()` byte-for-byte `np.array_equal`) and
`test_footprint_contains_zero_heading_is_byte_identical_to_axis_aligned`
(exhaustive grid sweep).

## Unit tests added

- `src/tests/mocks/test_synthetic_scene.py`: `footprint_contains` zero-heading
  regression guard, 90deg exact swap, 45deg strict-subset-of-hull property,
  the real arabic_room sofa fixture (object_id 56, heading -3.1328664 rad),
  `place_box` heading plumbing, `terrain_patch` end-to-end rotated-vs-hull cell
  count.
- `src/tests/groundtruth/test_loader.py`: `parse_object_csv` carries
  `obb_center/obb_extents/obb_heading`; default heading 0.0; a plain
  `InstanceRecord` (no OBB info) defaults to `None`/`None`/`0.0`.
- `src/tests/runner/test_gt_battery.py`: `_synthetic_from_gt` zero-heading
  byte-identical guard (through `terrain_patch()`), rotated-object strict cell
  reduction via the real `_synthetic_from_gt` path, and the exact arabic_room
  sofa fixture through the same path.

All new + existing tests pass (`pytest tests/mocks/test_synthetic_scene.py
tests/groundtruth/test_loader.py tests/runner/test_gt_battery.py`, and the
fast tier `pytest` from `src/`). Full gate (`pytest -m ""`) run and reported
below.

## Battery comparison (full run, `--out ../reports/gt_battery_obb_raster`)

Same command as `reports/gt_battery_main_post77c`
(`--groundtruth ../data/vla3d/Unity`). Aggregate:

| metric | post77c (AABB hull) | obb_raster (this fix) |
|---|---|---|
| numerical true_accuracy | 1.000 | 1.000 (unchanged) |
| object_reference mean_iou | 1.000 | 1.000 (unchanged) |
| IF mean_rubric_score (headline) | 0.6111 | 0.6111 (unchanged) |
| IF mean_ordered_leg_credit | 0.6333 | 0.6333 (unchanged) |
| IF total_threading_violations | 3 | 3 (unchanged) |
| IF total_avoid_violations | 0 | 0 (unchanged) |
| IF mean_frechet_m (diag, secondary) | 3.5734 | 3.5926 (+0.019) |
| IF mean_coverage_1m (diag, secondary) | 0.5724 | 0.5788 (+0.006, improved) |

**Zero score regressions, zero score changes at the headline/leg level in
this run.** Full per-question diff (`rubric_score`, `ordered_leg_credit`,
`leg_outcomes`, `pipeline_answer`, `true_answer`) confirms every scored field
identical between the two runs. `leg_probe.min_dist_driven_to_goal_m` shifted
by a few cm to ~1.4 m on 15 legs across 10 scenes (the driven path's shape
changed because obstacle footprints shrank), always the SAME resolved goal
and SAME candidate instance — no leg's `reached_in_order` status flipped
either direction. The two small secondary diagnostics that did move both
moved the "expected" direction (path shape now tracks the reference
trajectory slightly more closely), consistent with the costmap having
strictly less spurious blockage.

**Honest interpretation:** this run shows the fix does what it claims
(shrinks stamped footprints, changes driven paths, zero regressions) but does
not, on its own, flip any scored leg in the current 75-question battery. This
matches §0's own framing — the OBB fix targets the mirror's over-coverage
defect specifically, not the (separate, larger) grounding-pose-BFS-pocket
mechanism the plan defers to Stage 1 / pre-grounding movement.

## 14-leg pool reachability re-check (issue77e cluster)

`reports/issue77_notes_work/obb_raster_pool_check.py` reran issue77e's
"anchor cell reachable from grounding pose?" diagnostic (the
`trace_leg2.py` pattern) for all 14 legs in the issue77e fixable pool under
this fix. **Result: 0/14 became reachable** (all still `reachable=False`,
matching issue77e's original all-14 "no").

This is expected, not a null result: issue77e's own investigation (falsifying
the wall-derivation-dilation lever) already attributed this specific 14-leg
cluster to "density of stamped furniture AABBs collectively sealing off large
sub-regions from a SINGLE FIXED grounding-time BFS origin" — a
single-pose-BFS architectural limitation, not specifically rotated-object
over-coverage. Shrinking footprints (this fix) grows the reachable pocket
(see below) but doesn't reconnect these particular anchors from the fixed
spawn pose; only progressive/pre-grounding re-flood (deferred Stage 1)
targets that mechanism.

## GT-trajectory-contradiction re-probe (arabic_room q1 leg0, the §0 pocket)

`reports/issue77_notes_work/obb_raster_pocket_probe.py` reruns §0's own BFS-
pocket probe (arabic_room hookah-question leg0) for OLD (AABB hull; every
instance's `obb_heading` zeroed so `_synthetic_from_gt` takes its pre-fix
axis-aligned branch — the faithful old formula, not merely a heading-zeroed
copy of the new one) vs NEW (this fix):

| | pocket (BFS-reachable cells from spawn) | nearest reachable to anchor |
|---|---|---|
| OLD (AABB hull) | 536 | 7.326 m |
| NEW (OBB rasterized) | 558 (+22, +4.1%) | 7.326 m (unchanged) |

The reachable pocket measurably GREW (strictly fewer cells now blocked, as
designed) but the growth didn't land on the side nearest this leg's anchor —
the pocket is still 7.33 m short (leg tolerance ~2.5 m), so this one leg
stays blocked. Consistent with §0's own implication: "the spawn pocket is
sealed by mirror-fidelity defects... no amount of movement can escape it" —
here refined to: the OBB fix demonstrably reduces one contributing defect
(rotated-object over-coverage) without being sufficient alone to reopen this
specific pocket. The deferred GT-trajectory-carve (original Stage 1) is
NOT shown unnecessary by this session's evidence — if anything this
strengthens the case that it (or pre-grounding movement) is still needed for
this class, contrary to the plan's speculative "may be largely unnecessary"
framing. Left to the maintainer for the Stage 1 gate decision.

A separate direct point-membership probe against the GT reference
trajectory's own sampled points (`obb_raster_gt_traj_probe.py`) found ZERO
difference in blocked-cell count for this leg's trajectory sample (24 -> 24,
`sofa: 11, wall: 7, carpet: 6` both runs) once the OLD stamp was reproduced
correctly (an earlier draft of this probe had a bug reusing the NEW code's
already-heading-swapped extents for the "OLD" comparison, which produced an
inflated 45 spurious blocked-cell count for OLD — corrected before reporting
here). The §0 planning-session number ("102 mirror-blocked cells... sofa 74")
was not reproduced by either probe in this session; it likely used a denser
trajectory interpolation or covered more than this one leg/question — not
re-derived here (out of this pre-stage's bounded scope), but the pocket-BFS
number above (536 -> 558, §0's own stated diagnostic) DOES reproduce exactly
on the OLD side, cross-checking the OLD-scene reconstruction is faithful.

## Integrated numbers

See table above — full battery `mean_rubric_score`=0.6111,
`mean_ordered_leg_credit`=0.6333, `total_threading_violations`=3,
`total_avoid_violations`=0, all unchanged from `gt_battery_main_post77c`.

## Files

- `src/core/interfaces.py` — additive `obb_center/obb_extents/obb_heading`
  fields on `InstanceRecord`.
- `src/core/groundtruth/loader.py` — `parse_object_csv` populates the three
  additive fields (module docstring updated).
- `src/core/mocks/synthetic_scene.py` — `GTObject.heading`, rotated
  `footprint_contains`, `place_box(..., heading=0.0)`.
- `src/core/runner/gt_battery.py` — `_synthetic_from_gt` reads
  `rec.obb_heading/obb_center/obb_extents` and passes heading through to
  `sc.place_box`.
- `src/tests/mocks/test_synthetic_scene.py`,
  `src/tests/groundtruth/test_loader.py`,
  `src/tests/runner/test_gt_battery.py` — new tests (see above).
- `reports/issue77_notes_work/obb_raster_pool_check.py`,
  `obb_raster_pocket_probe.py`, `obb_raster_gt_traj_probe.py` — this
  session's probe scripts.
- `reports/gt_battery_obb_raster/` — full battery output.

## Tests

- `pytest tests/mocks/test_synthetic_scene.py tests/groundtruth/test_loader.py
  tests/runner/test_gt_battery.py` — pass (all new + pre-existing).
- Fast tier `pytest` (default markers) from `src/` — pass (43.6 s).
- Full gate `pytest -m ""` from `src/` — pass, exit code 0 (5m15s).
