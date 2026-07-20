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

TBD, see below for progress once started.
