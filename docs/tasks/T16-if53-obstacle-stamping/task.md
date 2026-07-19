# T16 — if53-obstacle-stamping (issue #53)

Branch: `if53-stamping` (worktree `.claude/worktrees/if53-stamping`). Follow-up to
T15/#51, on the fixB baseline (`reports/gt_battery_fixB_2026-07-19/`, IF headline
0.150, 7 threading violations).

## Method

Trace-first, same discipline as T15. Extended the T15 ad-hoc trace approach into a
reusable script (not committed — scratchpad-only, see below) that, for each of the 7
failing legs:

1. Resolves the corridor's two anchors exactly as `_if_rubric_geometry` does (distinct
   instances), computes the gate via `corridor_gate`.
2. Lists every OTHER GT instance whose raw AABB contains the gate midpoint, tagging
   each floor-level (`cz < TERRAIN_SLAB_MAX_Z`) vs overhang, and whether its stamped
   height clears `FREE_MAX` (0.15 m) — the actual costmap-blocking threshold.
3. For legs with no exact-midpoint hit, an instrumented `core.nav.planner.astar` /
   `path_crosses_gate` / `_pinch_costmap` wrapper (monkeypatched in a scratch script)
   traces the real driven route.

## Per-leg classification (7 failing legs, wall-honest fixB baseline)

| Scene | Leg | Shape | Root cause |
|---|---|---|---|
| home_building_1 | dining table / picture | **(a)** | GT id 25 `"wall"` (sx=21.6 sy=18.8, frac 0.40/0.33 of a 54x57 m room) and GT id 82 `"unknown"` (frac 0.68/0.71) both sit AT the gate midpoint, floor-level, `sz` 3.66 m / 2.63 m — solid floor-to-ceiling blocks from a room-scale AABB aggregate. |
| home_building_2 | sofa / coffee table | **(a)** | GT id 126 `"wall"` (x:[-10.07,6.07] y:[-1.84,14.05], frac 0.54/0.61 of a 30x26 m room) — the EXACT instance named in #53's issue text — plus GT id 167 `"unknown"` (frac 0.99/1.00) and GT id 132 `"floor"` (frac 0.52/0.69, `sz`=0.20 m, just over `FREE_MAX`) all sit at the gate midpoint. |
| studio | couch / table | **(a)** | GT id 31 `"wall"` (frac 0.96/0.31 of a 7.8x8.4 m room, `sz`=2.88 m) sits astride the route between the couch and table — not exactly ON the gate midpoint, but blocking the approach; removing it materially changes the driven path (Fréchet 3.65 m -> lower, see below). |
| hotel_room_2 | bench / bed | **(b)** | GT id 44 `"bed frame"` (a REAL furniture item, floor-level, frac 0.38/0.28 — short on the y axis, correctly spared by the room-scale rule) sits at the gate midpoint. VLA-3D records the same physical bed as two separate instances ("bed" id 57, the corridor anchor, and "bed frame" id 44) — the anchor's own AABB face doesn't cover the frame's footprint, so the gate lands on the frame. Genuinely a third real object, not an architectural over-stamp — candidate fix #3 from #53 (validate/nudge the midpoint) territory, NOT implemented this session (see below). |
| arabic_room | two columns | other | No floor-level blocker at the gate midpoint (`floor` id 46 stamps but `sz`=0.03 m, well under `FREE_MAX`). Matches T15's own note: fixed at the SCORING level (distinct-instance gate), still failing for "a genuinely different reason — a real navigation/reachability defect," unrelated to stamping. Unaffected by this fix. |
| hotel_room_1 | TV / bed | other | No floor-level blocker at the exact midpoint; a real chair (id 15) sits within 0.5 m. Matches #53's own note: "crowded by a desk cabinet + chair whose RAW footprints already overlap the TV's — no amount of inflation relaxation can open a gap that doesn't exist in the raw geometry." Real furniture crowding, not an architectural stamping defect. Unaffected. |
| livingroom_1 | sofa / round tables | other | Gate width = 0.000 (degenerate — the top-ranked "round table" candidate's AABB touches the sofa's). Matches #53's own note: a bare-noun corridor tie needing a "prefer the pairing that forms a real gate" heuristic (candidate fix #2), not a stamping defect. Unaffected. |

So of the 7: **3 (home_building_1, home_building_2, studio) are shape (a)**, fixed by
this task's stamping change; **1 (hotel_room_2) is shape (b)**, traced but NOT fixed
(see below); **3 (arabic_room, hotel_room_1, livingroom_1) are pre-existing, distinct
defects already correctly identified as out-of-scope by #51/#53's own text**, left
untouched.

## Fix implemented — shape (a): room-scale architectural AABB stamping

`core/runner/gt_battery.py`: `_synthetic_from_gt` no longer stamps a GT instance as a
solid box when `_is_architectural_room_scale_aabb` returns true — floor-level
(`cz < TERRAIN_SLAB_MAX_Z`, the same cutoff the terrain mirror already uses for
overhang/floor-obstacle) AND its footprint spans more than
`ARCHITECTURAL_AABB_ROOM_FRACTION` (0.3) of the SCENE'S OWN room footprint in BOTH x
and y. Purely geometric, self-referential per scene (a fraction of THAT scene's own
bounds, never a fixed metres constant) — no label list, no per-scene tuning.

Swept over all 15 GT scenes to check false positives (generalization protocol,
`docs/calibration.md`): every floor-level instance clearing the bar in BOTH axes is
labelled `"wall"`, `"floor"`, or `"unknown"` — home_building_1 id 25/82,
home_building_2 id 126/132/167, hotel_room_2 id 89, japanese_room id 12,
livingroom_2 id 10, studio id 31. **Zero real furniture instance across all 15
scenes clears it** — the closest real furniture item (hotel_room_2's bed frame,
id 44) sits at 0.38/0.28, short on the y axis. Scenes hotel_room_2/japanese_room/
livingroom_2 weren't in the failing-7 list but carry the same defect (a room-scale
"wall" that would otherwise falsely seal floor area); their IF questions didn't
route through those cells, so the fix is neutral for them here but correct for any
future leg that does.

Skipped instances rely on: (1) the traversable-mesh-derived `wall_cells` (IF-F2,
#52 — already wired) as the interior-wall source when a scene fit is available, and
(2) the room's own outer boundary (`SyntheticScene._is_wall`) for the exterior —
avoiding the double-representation #53 called out (a wall as BOTH a raw solid AABB
AND mesh-derived cells).

New tests (`src/tests/runner/test_gt_battery.py`, "issue #53" section): direct unit
tests of `_is_architectural_room_scale_aabb` (detects the traced home_building_2
shape; spares the traced hotel_room_2 bed-frame shape; requires BOTH axes, so a real
thin wall panel is untouched; spares an elevated slab/ceiling), plus
`_synthetic_from_gt` integration tests (a room-scale "wall" is not stamped and the
corridor gap under it reads free; a genuinely thin wall panel is still stamped).

## Result — IF headline unchanged, but for a newly-diagnosed reason

`reports/gt_battery_if53_2026-07-19/`: **IF headline unchanged at 0.150, threading
violations unchanged at 7.** Numerical 15/15 and object-reference 6/6 unchanged
(aggregate JSON diff: only the `instruction_following` block differs, and only its
secondary Fréchet/coverage diagnostics — `mean_frechet_m_aligned_diag` 6.764 m ->
4.225 m, `mean_coverage_1m_aligned_diag` 38% -> 47% — the ordered-leg-credit rubric
fields are byte-identical).

Tracing WHY the 3 shape-(a) legs still fail post-fix (instrumented `astar` /
`path_crosses_gate` / `_pinch_costmap` wrapper) found the gates are no longer sealed
by a stamped obstacle — the fix works as intended — but a DIFFERENT, previously
MASKED defect in `core.nav.planner._pinch_costmap` now becomes the operative
blocker for all three:

* home_building_2: the un-pinched direct A* to the gate midpoint fails outright (the
  inflation-only seal `_pinch_costmap` exists to relax); the pinch retry ALSO fails,
  because the pinch overlay's "block everything outside the 1 m corridor band within
  3 m of the gate" rule has no exemption for the vehicle's OWN start position — the
  leg's start (0.75, -0.15) sits ~2.7 m from the gate (inside the 3 m pinch disc,
  off the corridor line), so the pinch overlay newly blocks the start cell itself
  (`passable` True -> False across the pinch clone) and the forced-corridor A* is
  DOA.
* home_building_1: the direct A* DOES reach the gate's cell (31 pts) but the
  polyline never crosses the gate SEGMENT transversally (`path_crosses_gate ->
  False`) — it arrives at the far side without threading through; the pinch retry
  hits the same start-cell self-block as above.
* studio: same near-miss-then-fail-to-cross pattern; the pinch retry's start cell is
  outside the 3 m disc this time (~3.26 m), but the forced straight corridor band
  still finds no path — the band intersects real furniture the un-pinched route was
  detouring around.

Filed as **issue #54** (`gh issue create`) with the full per-leg trace evidence and
candidate fix directions — a `core/nav/planner.py` change, materially different
surface from this task's GT-mirror-stamping fix, and risky to attempt in the same
pass without dedicated per-leg regression coverage (could reopen #51's already-fixed
legs).

## Deferred — shape (b): hotel_room_2's third-real-object gate

Traced (GT id 44 `"bed frame"` sits at the gate midpoint, a real furniture item, not
an architectural over-stamp) but the candidate fix (#53's direction 3: validate the
gate midpoint isn't inside a third object's raw footprint, nudge along the gate axis
if it is) was NOT implemented this session:

* It needs a new call-site-shared helper (mirroring the anchor-distinctness fix's
  discipline of keeping `_if_rubric_geometry` and `InstructionHead`'s real
  resolution in lockstep — a scoring-only nudge that the real planner doesn't also
  apply would reopen the #51 scoring/planning mismatch class of bug), a bounded
  nudge-along-segment design, and dedicated regression tests proving it doesn't
  perturb the 23 currently-passing IF legs.
* Given the wall-clock guard (numerical/OR must not move; already spent this
  session's budget tracing the pinch-overlay defect above), this is left for a
  follow-up task rather than force-fit here — matching T15's own precedent
  (deferring the harder pinch/gate-selection work rather than rushing it into scope).

## Verification

Fast tier (`pytest`, `src/`) green throughout. Full gate (`pytest -m ""`) run before
commit. New tests: `src/tests/runner/test_gt_battery.py` "issue #53" section (6
tests) — `test_architectural_room_scale_aabb_detects_wide_flat_object`,
`test_architectural_room_scale_aabb_spares_real_furniture`,
`test_architectural_room_scale_aabb_requires_both_axes`,
`test_architectural_room_scale_aabb_spares_elevated_slab`,
`test_synthetic_from_gt_skips_room_scale_architectural_object`,
`test_synthetic_from_gt_still_stamps_real_thin_wall`.

Full GT battery: `reports/gt_battery_if53_2026-07-19/` — headline 0.150 (unchanged),
7 threading violations (unchanged), numerical 15/15 (unchanged), OR 6/6 (unchanged),
IF Fréchet/coverage secondary diagnostics improved (6.764 m -> 4.225 m,
38% -> 47%).

Not closing #53: the stamping-fidelity root cause it named IS fixed (verified
per-leg), but the headline metric it was filed to move does not move, because
fixing it unmasked #54. Commented on #53 with this finding instead of a false
`Fixes #53`.
