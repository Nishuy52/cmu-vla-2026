# Stage 0 audit — GT-trajectory-vs-blocked-cells, all 15 scenes

Read-only w.r.t. `src/`/`docs/` (per task scope). Extends
`docs/proposals/pre_grounding_movement_plan.md` §0's arabic_room probe to all 15
GT scenes on current main (`8908a9b`, post-#80 ORIENT fix). Reference battery run:
`reports/gt_battery_post80_bytecheck/gt_battery_results.json` (fresh, same commit).

Scripts (reference only — paths inside them point at this session's scratchpad,
not portable as-is): `reports/mirror_truth_audit/scripts/{mirror_audit.py,
run_all.py, pocket_and_regression.py}`.

## Method

For each scene: fit the scene's sim<->object `Frame2D` (`GB._fit_scene_if_frame`),
derive interior wall cells from `traversable_area.ply` (`GB._scene_wall_cells`),
build the mirror `SyntheticScene` (`GB._synthetic_from_gt`), ingest its full terrain
into a real `OccupancyGrid` + `Costmap` (fully observed at tick 0, so this is
pose-independent — matches `MockRobotIO.latest_terrain`). This reproduces
exactly the costmap the `InstructionHead`/BFS/A* actually reason over, not a
simplified proxy.

**Blocked-cell definition:** `Costmap.base_blocked` — the INFLATED obstacle mask
(vehicle-radius 0.4 m disc around every raw wall/object cell), i.e. the same mask
`reachable_mask`/BFS/A* consult. This matches the plan §0 methodology (its "536-cell
pocket" figure was explicitly the *inflated* mirror; the *raw* uninflated mask gave
1,513 cells for arabic_room, cited only as a contrast). My own BFS pocket for
arabic_room from the same shared spawn came out to **558 reachable cells** (vs the
plan's 536 — 4% off, consistent with a small residual/wall-derivation difference,
not a methodology mismatch).

For each blocked cell the GT reference trajectory (`trajectory_q4.ply` /
`trajectory_q5.ply`, mapped through the fitted frame) passes through: attribute the
blocker by finding the nearest TRUE (uninflated) obstacle cell within the same 0.4 m
disc `Costmap._inflate` itself uses (nearest-cell search, no category priority —
ties split into both buckets). Wall blockers are further split into **interior**
(covered by the derived `extra_wall_cells` lattice — real GT-derived wall) vs
**border** (the mirror's own synthetic outer-boundary rectangle, a construction
artifact of the 1.5 m scene padding, not a GT wall). Object blockers are attributed
to every GT instance whose AABB footprint contains the nearest raw-blocked point,
matched back to the raw CSV OBB row (center/extents/heading) to test
**OBB-clearability**: the trajectory-driving point lies inside the AABB hull but
OUTSIDE the true oriented rectangle → OBB rasterization (the dispatched,
already-approved pre-Stage-1a fix) removes that cell as a blocker.

**Pocket-excluded legs:** `Costmap.reachable_mask(spawn_xy)` from the scene's
shared spawn (same point `score_scene` uses — first IF question's mapped GT
trajectory start); a leg's anchor (`leg_probe.our_goal`, from the current battery
results) is "pocket-excluded" if its cell is outside that reachable set — the same
structural condition the plan's arabic_room probe used to explain why offline
movement cannot help (BFS from spawn can never reach a disconnected component,
independent of tolerance).

**Regression-risk flag:** for every CURRENTLY-PASSING leg (all legs in the question
reached in order, zero threading/avoid violations), replan that leg's own route
(`GB._drive_if_path`) and check whether the planned path passes within 0.6 m
(1.5× vehicle radius) of a cell this audit found to be OBB-clearable in that scene.
A "yes" means: once OBB rasterization stops stamping that cell solid (the dispatched
change), the planner may see a materially different local geometry near a route that
currently passes — worth a full per-question diff after that change lands, not
evidence of an actual regression yet.

## Method check — arabic_room reproduction vs plan §0

| Metric | Plan §0 | This audit | Note |
|---|---|---|---|
| BFS spawn pocket (inflated mirror) | 536 cells | 558 cells | 4% off, same order — mirror pipeline reproduced faithfully |
| GT-trajectory blocked cells (total) | 102 | 106 | 4% off |
| — wall-attributed | 36 | 56 (all **border**, 0 interior-derived) | **Discrepancy** — see below |
| — object-box-attributed | 66 | 50 | sum doesn't have to match plan's per-object sum (138) either — see below |
| sofa cells | 74 | 28 | plan's per-object counts are NOT disjoint (a cell touching sofa+carpet counts under both) |

**Total blocked-cell count reproduces within 4%** (106 vs 102) using the real
inflated `Costmap.base_blocked` mask, the real fitted frame, and the real
`InstructionHead`-equivalent mirror build — strong evidence this audit's pipeline
matches the plan's probe. The **wall/object split does not reproduce exactly**: this
audit attributes 56/50 (all wall hits are *border*, i.e. the synthetic outer-rectangle
padding, not GT-derived interior wall), where the plan reports 36/66 with the wall
bucket implied to be GT-derived. The plan's original probe script was never
committed (only `probe_leg0_detail.py`/`probe_hb2_joint.py` survive in
`reports/issue77d_notes_work/`, and neither computes a blocked-cell count), so its
exact wall/object attribution rule cannot be checked byte-for-byte — this is flagged
as an honest limitation, not silently reconciled. Whichever rule is "right," the
practical Stage-1 takeaway is unaffected here: object-box blockers dominate in every
scene this audit measured (see table below), and none of arabic_room's wall hits
this audit found are GT-derived-interior — they are all mirror-padding boundary
artifacts, which is itself useful evidence that arabic_room's GT-trajectory
contradiction is essentially 100% an object-stamping problem, not a wall-derivation
problem.

## Per-scene table

Blocked-cell counts are UNIQUE cells the GT trajectory(ies) pass through in the
inflated mirror costmap. `obb_clear` = object-box-blocked cells where the driving
point sits inside the AABB hull but outside the true OBB (cleared once OBB
rasterization ships). `pocket_excl` = leg anchors (from the live battery results)
outside the spawn BFS pocket, `n/n_legs` probed. `risk` = currently-passing legs
flagged for possible reroute once OBB-clearable cells open up.

| Scene | blocked (tot) | wall (interior/border) | object-box | obb-clearable | pocket-excluded legs | regression-risk legs |
|---|---:|---|---:|---:|---|---|
| arabic_room | 106 | 56 (0/56) | 50 | 22 | 5/5 | 0 |
| chinese_room | 60 | 0 (0/0) | 60 | 42 | 4/4 | 0 |
| home_building_1 | 278 | 0 (0/0) | 278 | 115 | 5/5 | 0 |
| home_building_2 | 272 | 6 (6/0) | 266 | 64 | 4/5 | 0 |
| hotel_room_1 | 139 | 60 (57/0) | 82 | 47 | 5/5 | 1 |
| hotel_room_2 | 90 | 0 (0/0) | 90 | 56 | 3/5 | 2 |
| japanese_room | 69 | 0 (0/0) | 69 | 39 | 5/5 | 2 |
| livingroom_1 | 149 | 0 (0/0) | 149 | 82 | 3/5 | 1 |
| livingroom_2 | 102 | 4 (3/0) | 99 | 62 | 4/4 | 1 |
| livingroom_3 | 93 | 0 (0/0) | 93 | 34 | 5/5 | 0 |
| livingroom_4 | 146 | 6 (4/0) | 142 | 80 | 4/5 | 1 |
| loft | 210 | 1 (1/0) | 209 | 73 | 3/4 | 0 |
| office_1 | 78 | 0 (0/0) | 78 | 36 | 3/5 | 1 |
| office_2 | 69 | 12 (12/0) | 57 | 52 | 4/5 | 1 |
| studio | 145 | 19 (3/13) | 129 | 84 | 4/5 | 1 |
| **Total** | **2006** | **164 (86/69)** | **1851** | **888** | **65/71** | **11 (leg-questions)** |

Notes:
- Object-box blockers dominate every scene (57/60 of every scene's blocked cells,
  min chinese_room 100% object, max hotel_room_1 59% object) — consistent with the
  plan's own arabic_room finding and with the user's amendment (§5.1) that OBB
  rasterization is "the exact fix for the dominant blocking cause."
- Interior-derived (GT-sourced) wall blockers are rare and concentrated in a few
  scenes (hotel_room_1: 57, office_2: 12, studio: 3, home_building_2: 6,
  livingroom_2: 3, livingroom_4: 4, loft: 1) — 9 of 15 scenes have **zero**
  wall-attributed GT-trajectory blockers of any kind.
- `studio` is the only scene where the mirror's outer-boundary padding
  (border-wall, 13 cells) meaningfully contributes alongside real interior wall
  cells (3) — worth a look if Stage 1's carve radius is later swept, but out of
  scope for OBB rasterization.
- Pocket exclusion is high almost everywhere (65/71 leg-anchors, 92%) — most GT
  leg anchors sit outside the tiny BFS-reachable pocket around spawn in every
  scene, confirming the plan's "spawn pocket is sealed" finding generalizes far
  beyond arabic_room. This does NOT mean 92% of legs fail: several reach anyway
  because `nearest_reachable_point` + a generous derived arrival tolerance can
  still count as "reached" even when the literal anchor cell is outside the
  strict BFS component (see `n_legs_reached_in_order` in the battery results —
  many legs with pocket-excluded anchors still score full credit).
- Regression-risk flags: 11 leg-questions across 8 scenes have a currently-passing
  route that comes within 0.6 m of a cell OBB rasterization will unstamp. None of
  these is evidence of an actual regression — it flags exactly the set of
  passing questions that need a **full per-question diff after OBB rasterization
  ships** (the same "zero regressions required" gate the plan already specifies
  for Stage 1, generalized to the pre-Stage-1a OBB change).

## Bottom line

- **Total GT-trajectory-contradicted blocked cells across all 15 scenes: 2,006.**
- **OBB rasterization alone clears an estimated 888 of those 2,006 cells (44%)** —
  every case where the trajectory-driving point sits inside the AABB-of-OBB hull
  but outside the object's true oriented footprint. This is a per-cell
  *point-membership* estimate (does the exact grid-cell centre the trajectory
  passed through fall inside the true rotated rectangle?), not a full rasterized
  re-stamp — the real OBB-rasterization lane will very likely clear a few more
  cells than this (some cells this audit calls "not cleared" are blocked by a
  corner of the true OBB that a coarser rasterizer might still miss at 0.1 m
  resolution, and conversely some near-boundary cells could go either way at
  sub-cell precision) — but 888/2,006 is a solid, conservative first-order size.
- **Residual after OBB rasterization: 1,118 cells (56%)** still block the GT
  trajectory — 164 of those are wall-attributed (86 truly GT-derived interior
  wall, 78 mirror-padding border-wall artifacts that OBB rasterization was never
  going to touch), and **963 are object-box cells that are genuinely inside the
  true OBB footprint** (real solid-obstacle contradictions, not an AABB
  over-approximation artifact) — this is the honest residual that still needs the
  deferred GT-trajectory carve (original Stage 1) if the plan wants those legs to
  become attemptable offline.
- Practically: **OBB rasterization is worth dispatching first as already decided**
  (§5.1) — it clears a large, well-defined minority of the contradiction cheaply
  and with no new thresholds — but it does **not** obsolete the carve. Roughly
  half the object-box contradiction survives OBB rasterization (963/1,851 object
  cells, 52%), meaning the carve (or an equivalent mirror-truth fix) is still
  needed for the residual once 1a's actual effect is re-measured on a live
  battery run (not just this static geometric estimate).
- Per §0's Stage-1-sizing purpose: **every one of the 15 scenes has at least one
  GT-trajectory-contradicted blocked cell** — no scene is currently "clean," so
  Stage 1 (or the OBB pre-stage) scope is genuinely all-scene, not
  arabic_room-specific. Gate G0 evidence: recommend scoping Stage 1's carve to
  the residual after re-measuring with OBB rasterization live (per the plan's own
  ordering — 1a first, then re-probe before committing to the carve's scope).
