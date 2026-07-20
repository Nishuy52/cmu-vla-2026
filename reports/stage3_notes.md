# Stage 3 implementation notes (pre-grounding movement plan, issue #77)

Branch `if-stage3-withhold`, worktree `.claude/worktrees/if-stage3-withhold`.

## Scope
Implement Stage 3 of `docs/proposals/pre_grounding_movement_plan.md` in
`src/core/heads/instruction.py` only: `_GroundedLeg.goal_clamp_m`, Gate 3 in
`_committable_prefix_len`, and the guarded re-ground improvement rebuild.

## Design decisions taken
- **Credibility bar (decision 3):** `GOAL_CLAMP_CREDIBILITY_BAR_M = ARRIVAL_TOL_M`
  exactly. Parameter-free per the plan; also matches the plan's own evidence table
  (arabic_room "nearest reachable 2.50 m (leg tol 2.51)" — the bar IS the leg
  tolerance). A clamp within tolerance is exactly as good as landing on the anchor
  for scoring purposes (rubric credits arrival within `ARRIVAL_TOL_M`); beyond it,
  driving there can never score.
- **goal_clamp_m** is the straight-line distance from a GOTO/VIA_NEAR leg's resolved
  `geom` to its primary anchor's centroid, computed in `_ground_one` right after
  `_goto_point`/`_via_point` return. `None` for CORRIDOR_BETWEEN (no single goal
  point) and for any leg whose geometry never resolved.
- **Gate 3 override pattern:** identical to H4c/#33 — `_commit_forced()` (budget
  pressure, defaults True with no `budget_frac` hook) OR `_forced_assembly_reached()`
  (T-90 pressure, defaults False with no `forced_assembly` hook) skip the gate
  entirely. With both hooks unconfigured, `_commit_forced()` is True by default, so
  Gate 3 never withholds — byte-identical to pre-Stage-3 behaviour.
- **First-tick costmap ordering bug found + fixed during test-writing:**
  `goal_clamp_m` depends on the reachable-mask BFS, which only exists once a costmap
  has been stamped — but the very first `_ground_legs` call each tick (inside
  `advance`) runs BEFORE any costmap exists, so it falls back to the coarse
  `_project_free` nudge (no BFS clamp). On the FIRST-ever route build this let a
  badly-clamped leg slip past Gate 3 (checked on stale, artificially-small clamp
  data) before the real costmap-aware re-ground (inside `_build_route`) computed the
  true clamp — one tick too late to gate on. Fixed by extracting
  `_refresh_costmap_and_geometry` (costmap + stamp_avoids + re-ground, the shared
  first half of `_stamp_ground_plan`) and having `_maybe_build_or_extend_route` run
  it once as a probe before computing `_committable_prefix_len()` whenever no route
  has ever been built yet (`_follower is None and _costmap is None`). Idempotent —
  `_build_route`/`_stamp_ground_plan` redo the exact same stamp+ground once more
  before actually adopting a route, so no final geometry changes, only what Gate 3
  sees. Caught by test (a) (`test_clamped_leg_withheld_while_budget_available`) before
  it ever reached a committed report.
- **Re-ground improvement rebuild is unconditional on hooks** (no budget/forced-
  assembly gating) per the plan — it's a pure geometry-improvement correction, not a
  withhold decision. Confirmed safe for the offline battery: `_drive_if_trajectory`
  stops re-ticking the head once `_driven_prefix >= n_legs` (gt_battery.py:507), and
  the battery's synthetic scenes are fully observed at tick 0 with a stationary build
  pose, so `goal_clamp_m` cannot change between the (at most 1-2) build ticks — the
  trigger is a structural no-op offline until Stage 4 wires live hooks.
- **Refactor:** `_build_route` split into `_refresh_costmap_and_geometry` (stamp +
  ground) → `_stamp_ground_plan` (+ plan_through, returns `(path, leg_bounds) | None`,
  never touches `_follower`/`_driven_prefix`) → `_build_route` (adopts, falling back
  to `_recover_path` on `None`, same as before) / `_try_reground_rebuild` (adopts
  ONLY on success, never falls back to recovery). `_snapshot_committed_clamp` records
  `goal_clamp_m` per committed leg index at every real adopt (both paths).

## Tests added
- `tests/heads/test_instruction_stage3_withhold.py` — plan tests (a) (withhold +
  forced-assembly/budget-pressure commit + no-hooks byte-preservation), (c)
  (reground rebuild fires/caps/skips confirmed legs/never downgrades to recovery),
  (d) (corridor-adjacent goto rebuild declines rather than costing earned threading).
- `tests/heads/test_if_explore_ungrounded.py` — plan test (b): explore fall-through
  engages while a clamped leg is withheld.
- (e) is the full pre-existing suite run unmodified (byte-preservation proof).

## Verification log
- Fast suite (`pytest` from `src/`) green throughout, including after the
  first-tick-ordering fix.
- Full gate (`pytest -m ""`) and battery byte-check: see final report / commit
  messages for results.
