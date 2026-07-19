# IF-F2 wall realism — analysis (2026-07-19)

Baseline: `reports/gt_battery_postsweep_2026-07-19/` (walls off; the old boundary-only
mirror costmap — object obstacles + outer boundary, no interior walls).
This run: `reports/gt_battery_walls_2026-07-19/` (walls on, the new default), same
`--groundtruth data/vla3d/Unity`, same questions/answers.

## What changed

Interior walls are now derived per-scene from `data/unity_scenes_ros2/<scene>/<scene>/
traversable_area.ply` (a floor-traversability point mesh, in the same sim/trajectory
frame as `trajectory_qN.ply`) instead of being absent. The mesh is mapped into the
object (VLA-3D) frame via the scene's already-computed sim->object `Frame2D` fit (the
same fit the IF trajectory scoring uses), rasterized to the mirror costmap's 0.1 m grid,
and any cell inside the scene's outer boundary that the mesh does not cover — after a
0.2 m dilation to bridge point-cloud sampling gaps — becomes an OBSTACLE. Walls are only
derived when the scene's frame fit is itself trustworthy (residual <= the 1.0 m
alignment gate that already gates the Frechet/coverage secondary diagnostics) — a scene
we don't trust to score path shape against isn't a frame we should trust to place walls
with either.

## Topline delta

| Metric | Old (no walls) | New (walls on) | Delta |
|---|---|---|---|
| IF headline (mean rubric-proxy) | 0.117 (3.5/30) | 0.100 (3.0/30) | **-0.017** |
| IF ordered-leg credit (mean) | — | 0.122 | (not in old topline line; comparable order) |
| IF threading violations (total) | 8 | 9 | +1 |
| IF avoid violations (total) | 0 | 0 | 0 |
| Numerical TRUE accuracy | 15/15 | 15/15 | **0 (unchanged)** |
| Object-reference instance-match | 6/6 | 6/6 | **0 (unchanged)** |
| Object-reference IoU | 1.000 | 1.000 | 0 (unchanged) |

Numerical and object-reference are untouched by this change (walls only feed the IF
mirror costmap) — both stayed exactly 15/15 and 6/6 as required.

IF moved down (less honest-looking) by 0.017 mean / 0.5 total over 30 questions — a
single question flipped from partial credit to zero. That is the whole delta; every
other question's `rubric_score` is bit-identical between the two runs (poses/Frechet/
coverage — SECONDARY diagnostics only — do shift on many rows because the DRIVEN path
now routes around interior walls, but those never carried headline credit).

## The one flipped question

**livingroom_1**, *"Go to the potted plant closest to the pyramid candle holder and stop
at the vase between the TV and the door."* — `rubric_score` 0.5 -> 0.0
(`n_legs_reached_in_order` 1 -> 0).

Root cause: this scene's sim->object frame fit has residual 0.926 m — under the 1.0 m
alignment gate (so it's used), but close to it (a marginal fit). Mapping
`traversable_area.ply` through that noisy transform lands the mesh points measurably
off from where the GT objects actually sit. Leg 0's goal — the potted-plant anchor at
object-frame `(1.782, -4.299)` — ends up landing inside a cell the (mis-projected) mesh
doesn't cover, so it now reads OBSTACLE even though the real floor there is driveable.
The driven trajectory (which does not itself avoid obstacles — it just follows the
planned breadcrumbs; see `_drive_if_trajectory`) is planned by the head's A* against
this costmap and can no longer route a sensible path to a goal cell that is now inside
a wall; the drive stalls almost immediately (`driven_n_poses` 4002 -> 10 — previously
the vehicle burned the full watchdog window nominally credited a leg reached from a long
wander, which is itself not a strong signal of geometric correctness).

This is exactly the known, documented failure mode of the approach — see the
`_WALL_DILATE_CELLS` comment in `core/runner/gt_battery.py`: the 0.2 m dilation bridges
point-cloud sampling gaps but is deliberately **not** sized to also absorb frame-fit
residual (which can run up to the 1.0 m gate), so a scene with a noisy-but-passing fit
can still wall off a real object/floor location near the fit's error radius. It is a
genuine cost of deriving walls from a mesh in a different frame that has to be
rigid-fit rather than a bug in the rasterization itself (verified separately with
synthetic doorway/solid-wall fixtures — `test_derive_wall_cells_blocks_solid_wall_keeps_
doorway_open` and friends in `tests/runner/test_gt_battery.py` — the algorithm correctly
keeps a doorway open and blocks a solid wall band when the input points are
well-aligned).

`--no-walls` is the escape for a scene whose wall placement looks wrong; `walls=False`
programmatically. No other IF question's headline changed; the other pose-count/
Frechet/coverage churn across the report reflects the driven path legitimately routing
around interior walls it previously could not see, which is the intended effect of this
change.

## livingroom_3 (frame-fit-unfittable scene)

livingroom_3's sim->object frame fit has residual 1.81 m (both the default top-candidate
fit AND the meth-F11 exhaustive candidate-pairing fallback fail the 1.0 m gate — its two
GT trajectory terminal endpoints are 1.20 m apart in the sim frame, but every
resolvable pillow-endpoint <-> bowl-endpoint pairing is >= 3.27 m apart, a
frame-independent distance contradiction, not a resolver mis-rank). Confirmed by
re-running the fit search directly (`_fit_if_frame_over_candidates` returns `None`) —
this matches the pre-existing `_DATA_UNFITTABLE_IF_SCENES["livingroom_3"]` record
exactly, so it is left as-is: a documented GT-data defect, not a code logic bug.

Because the frame is untrustworthy, this run's wall derivation now explicitly declines
to use it (guarded on the same 1.0 m alignment gate) — livingroom_3's two IF rows fall
back to the old boundary-only costmap and carry an `interior walls unavailable (frame
fit unaligned (residual 1.81 m > 1.0 m gate))` note. Its rubric scores (both 0.00) and
`frame_aligned=false` status are unchanged from the walls-off baseline. No fix was
attempted beyond this guard — the underlying data contradiction is outside this
worktree's scope (IF1) and is already a documented, honest exclusion rather than a
silent drop.

## Interior wall coverage vs a claimed doorway example

The brief for this task named a specific validation point — japanese_room's doorway at
`x in [-0.9, 0.2], y = -4.8` — that was expected to remain passable. That coordinate
does not fall inside japanese_room's object-frame footprint at all (GT AABB y-range is
`[-2.11, 6.53]`, padded to `[-3.61, 8.03]`; `y = -4.8` is outside even the padded mirror
costmap, so it is simply never in the wall-cell set — not evidence either way).
japanese_room's own `region_result.csv` lists a **single** region (`livingroom`) — the
whole scene is one open room with no room-to-room doorway to validate against; its six
"door"/"wardrobe door" objects are closet/alcove doors set into the room's own walls
(confirmed: several read OBSTACLE post-wall-derivation, one — the exterior "entrance
door" — reads passable). That is plausibly correct (the space behind a closet door is
not real floor for a ground robot) rather than a doorway wrongly sealed, but it could
not be independently confirmed against the stated coordinate since that coordinate does
not correspond to any position in this scene's data. The doorway-preservation property
itself IS verified — with synthetic fixtures that have a real solid-wall-with-gap
geometry (`test_derive_wall_cells_blocks_solid_wall_keeps_doorway_open`), and with an
end-to-end planner test that a real interior wall band forces the driven path around it
(`test_drive_if_trajectory_routes_around_interior_wall`).

## Verdict

IF moved **down** by a small, fully-explained amount (one question, one marginal-fit
scene) — the honest direction is whichever the data gives; here it is a small drop,
attributable to frame-fit noise rather than the rasterization logic. Numerical and
object-reference are unaffected as required. No regressions found in the full fast or
full (`-m ""`) test tiers.
