# T14 — IF-F2 wall realism (interior walls from traversable_area.ply)

Branch: `if1-walls` (worktree). Goal: the offline gt_battery's mirror costmap for IF
planning/scoring had object obstacles + outer boundary but NO interior walls (see
`reports/gt_battery_postsweep_2026-07-19/gt_battery_report.md` "Wall realism (IF-F2)"),
so planned routes could cut through walls and the IF rubric-proxy (0.117 mean) was
partly measurement artifact.

## What was done

1. Extended `core.mocks.synthetic_scene.SyntheticScene` with `extra_wall_cells` —
   optional interior-wall cells (on the `round(coord / FLOOR_SPACING)` lattice) checked
   alongside the existing border/doorway wall logic in `_is_wall`.
2. `core/runner/gt_battery.py`: added `_derive_wall_cells` (pure rasterization: traversable
   XY points -> covered-cell grid -> dilate 2 cells (0.2 m) -> complement inside the
   scene's outer boundary = wall cells) and `_scene_wall_cells` (loads
   `data/unity_scenes_ros2/<scene>/<scene>/traversable_area.ply`, maps it into the
   object frame via the scene's already-fitted sim->object `Frame2D`, guarded on that
   fit's residual passing the same 1.0 m alignment gate used for the Frechet/coverage
   diagnostics — an untrustworthy frame isn't used to place walls either). Threaded
   `wall_cells` through `_synthetic_from_gt` / `_drive_if_path` / `_run_instruction_head`
   / `_drive_if_trajectory` / `score_scene` / `run_gt_battery` / `main`. `walls: bool =
   True` default; `--no-walls` CLI flag reproduces the old boundary-only costmap
   unconditionally. Per-question report rows note `interior walls unavailable (...)`
   when a scene falls back (no frame / unaligned frame / no ply).
3. livingroom_3 (the one scene whose IF frame fit stays unaligned) was investigated per
   the task brief: re-ran the fit search directly, confirmed the pre-existing
   `_DATA_UNFITTABLE_IF_SCENES["livingroom_3"]` record exactly (endpoints 1.20 m apart in
   sim frame vs every candidate pairing >= 3.27 m apart in object frame — a
   frame-independent contradiction, not a resolver mis-rank or a code defect). Left as
   documented; wall derivation now explicitly declines to use its untrustworthy frame
   too (residual 1.81 m > 1.0 m gate).
4. Full battery re-run with walls on: `reports/gt_battery_walls_2026-07-19/` (+ short
   `wall_realism_analysis.md` in that dir). IF headline moved 0.117 -> 0.100 (one
   question flipped, traced to a marginal-fit scene — see the analysis file). Numerical
   stayed 15/15, object-reference stayed 6/6 IoU=1.000.
5. Unit tests added (`src/tests/runner/test_gt_battery.py`, "IF-F2 walls" section):
   rasterization keeps a doorway open / blocks a solid wall band, fully-covered room
   has zero wall cells, `SyntheticScene.extra_wall_cells` reaches `terrain_patch`,
   an end-to-end driven-trajectory test proves a wall band forces the planner around
   it, `_scene_wall_cells` returns `None` without a frame / without a ply file, the
   `walls=False` / `--no-walls` escape is wired through `score_scene` and `main`.

## Acceptance

- Fast tier (`pytest` from `src/`) green.
- Full gate (`pytest -m ""` from `src/`) green.
- Numerical 15/15, object-reference 6/6 unchanged with walls on.
- Doorway-preservation property verified with synthetic fixtures (real-scene doorway
  coordinate given in the task brief did not correspond to any position in
  japanese_room's data — see the analysis file for why — so it could not be used
  directly; the underlying rasterization property is verified instead).
