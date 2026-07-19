# T15 — if2-corridor-threading (issues #52 + #51)

Branch: `if2-corridor` (worktree `.claude/worktrees/if2-corridor`). Two fixes on the
wall-honest battery (`reports/gt_battery_walls_2026-07-19/`, IF headline 0.100, 9
threading violations), folded into one battery cycle per #51's note.

## FIX A (#52) — wall-derivation frame-fit gate

`core/runner/gt_battery.py`: added `WALL_FIT_MAX_RESIDUAL_M = 0.8`, a TIGHTER gate
than the scoring alignment gate (`S._ALIGN_RESIDUAL_GATE_M = 1.0`) for wall
derivation specifically. A scene whose sim->object fit residual clears 1.0 m but
sits in the 0.8-1.0 m band is still trusted for Frechet/coverage scoring but no
longer for hard wall-cell rasterization (a binary decision that can wall off a real
object location on a marginal fit).

Registered in `docs/calibration.md` (new "groundtruth/runner harness constants"
section — these live outside `core.calibration.Calibration` since they gate the
battery's OWN measurement fidelity, never the shipped pipeline). Scene-holdout note:
the 0.8 m value came from ONE scene (livingroom_1, 0.926 m) but ALSO reclassifies two
scenes not used to pick it — home_building_1 (0.879 m) and hotel_room_2 (0.975 m) —
while every other scene sits clearly below the gate (max 0.692 m, livingroom_4).

Result: livingroom_1's flipped leg reverts (0.00 -> 0.50), IF headline 0.100 -> 0.117
(`reports/gt_battery_fixA_2026-07-19/`). This became the new baseline for FIX B.
Numerical 15/15, OR 6/6 unchanged. Tests: `test_wall_derivation_declines_just_above_residual_gate`
/ `test_wall_derivation_proceeds_just_below_residual_gate` (`src/tests/runner/test_gt_battery.py`).

## FIX B (#51) — corridor threading violations

Issue premise: "ALL 8 threading violations are corridor legs, one geometry/threading
defect accounts for every violation." Traced 3 failing corridor questions per-leg
BEFORE fixing (`reports/if2_corridor_trace_2026-07-19/trace.log` + ad-hoc scripts,
see LOG.md). Finding: the premise was **overturned** — tracing surfaced at least
THREE distinct defects, not one:

1. **Rubric-scoring anchor-distinctness mismatch.** `_if_rubric_geometry` (scoring)
   resolved a corridor leg's two anchors INDEPENDENTLY, unlike
   `InstructionHead._resolve_leg_anchors` (real planning), which enforces DISTINCT
   instances within a leg (`used` set). For a duplicate-noun corridor ("between the
   two columns"), the rubric collapsed both anchors onto the SAME top-ranked
   instance -> a zero-width gate the correctly-driven real path (which threads the
   TRUE, distinct-instance gate) can never register as "crossed." Fixed by mirroring
   the same distinctness enforcement in `_if_rubric_geometry`
   (`core/runner/gt_battery.py`). Affected: arabic_room, office_1.

2. **Resolver tier-priority loss.** `resolve()`'s final tie-break, when no clause or
   superlative disambiguates a MODIFIED-noun query, fell straight to `_stable_by_id`
   (raw instance id) — discarding the match-TIER information the index already
   computed. A query like "coffee table" could rank an unrelated "dressing table"
   (lower instance id, HEAD_NOUN-tier cousin) ahead of the EXACT "coffee table"
   match. This was not just a scoring artifact — it's a real navigation defect: the
   REAL driven route also targeted the wrong object (verified: InstructionHead's own
   corridor gate matched the identically-wrong geometry). Fixed with a new
   `_tier_priority_order` helper (`core/geometry/toolbox.py`), reusing the
   EXACT/SYNONYM/HEAD_NOUN/TYPO tier ladder `_match_anchor_noun` already established
   for anchor resolution (#13), with the same bare-noun exemption (#21) so a bare
   "table" superlative/count pool is untouched. Affected: home_building_1,
   home_building_2, hotel_room_2 (tv-cabinet leg).

3. **Pinch fallback never engaged / never opened a real gap.** Two planner defects
   in `core/nav/planner.py`:
   a. `plan_through`'s corridor branch tried the pinch overlay ONLY when the direct
      A* to the gate midpoint SUCCEEDED but missed the gate — when the midpoint was
      unreachable outright in the un-pinched costmap (`astar` returns `None`), the
      leg gave up immediately without ever trying the forced corridor. Fixed: the
      pinch fallback is now attempted whenever the direct plan either fails outright
      or reaches-but-misses.
   b. `_pinch_costmap` only ever ADDED blocking (outside the forced corridor); it
      never had a way to OPEN a real gate narrower than `2 * vehicle_radius_m` —
      such a gate gets sealed end-to-end by the vehicle's own inflation margin from
      BOTH anchor objects regardless of how tightly the corridor is forced. Added
      `Costmap.raw_blocked` / `Costmap.inflation_only_mask()` (`core/nav/costmap.py`)
      so the pinch overlay can clear INFLATION-ONLY cells (never a raw obstacle
      footprint) inside the forced corridor band — squeezing a verified GT gate
      without ever letting the vehicle drive through solid geometry.

## Result

`reports/gt_battery_fixB_2026-07-19/`: IF headline 0.117 -> 0.150 (rubric),
threading violations 9 -> 7 (target in the task brief was 0-2; NOT reached — see
below). Numerical 15/15 and OR 6/6 unchanged throughout (verified after every fix).

arabic_room's "column" leg (defect 1) is fixed at the SCORING level: the rubric now
scores against the SAME real, distinct-instance 1.71 m gate the planner already
targeted (verified: `_if_rubric_geometry`'s gate now matches `InstructionHead`'s
exactly). The leg is STILL a violation post-fix, but now for a genuinely different
reason — a real navigation/reachability defect around that gate (see below), not a
scoring artifact. Genuinely newly-passing: **office_1** (defect 1) and
**hotel_room_2's tv-cabinet leg** (defect 2).

## Remaining 7 violations — root cause NOT fixed in this task

Deeper tracing on the still-failing legs (arabic_room, home_building_1,
home_building_2, hotel_room_1, hotel_room_2's other leg, livingroom_1, studio) found
a FOURTH, distinct class of defect, out of this issue's stated scope (which named
overlay geometry / midpoint selection / the crossing test / silent fallback-keeping
— not this): the computed gate midpoint (from `aabb_face_points_2d`, the two
anchors' closest AABB faces) sometimes lands INSIDE a THIRD object's real footprint,
or on the far side of a real "wall"-labeled GT instance whose AABB spans a large
fraction of the room (VLA-3D sometimes records a room's perimeter walls as one
oversized bounding box, which `_synthetic_from_gt` stamps as one solid obstacle
covering the interior it should leave hollow) — genuinely sealing the gate with NO
inflation-only slack to relax. Examples traced: home_building_2's "sofa/coffee
table" gate sits under a room-spanning "wall" AABB (id 126, x:[-10.07,6.07]
y:[-1.84,14.05]); hotel_room_1's "TV/bed" gate is crowded by a desk cabinet + chair
whose RAW (uninflated) footprints already overlap; livingroom_1's "sofa/round
table(s)" corridor resolves to a genuinely touching-AABB pair (bare-noun tie, tier
fix doesn't apply) with a second, better-separated candidate available but unpicked.

This is a materially different, harder problem (obstacle-stamping fidelity for
architectural GT labels, or a smarter gate-selection/validation step) — filed as
#53 rather than force-fit into this task's scope or falsely close #51 (left open
with a status comment instead). See LOG.md and #53 for detail.

## Verification

Fast tier + full gate (`pytest -m ""`) green throughout. New tests:
`src/tests/geometry/test_resolve.py` (tier-priority fix, 3 tests + bare-noun
exemption), `src/tests/runner/test_gt_battery.py` (rubric distinctness, 2 tests),
`src/tests/nav/test_planner.py` (pinch inflation-clearing + total-miss fallback, 3
tests) — all confirmed to FAIL on the pre-fix code (reverted planner.py and reran to
verify the two planner regression tests actually catch the regression).
