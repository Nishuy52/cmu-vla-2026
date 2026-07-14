# cvsweep IF-instrument endogeneity fix (meth-F1) — implementation note

Implemented by executor #2 on 2026-07-14/15; the agent's session ended
after its test run completed but before it could commit, so this note
was finalized by the orchestrator from the agent's interim report and
the on-disk test log (orchestration.md restart-proofing rules).

## Change (src/core/runner/cvsweep.py)

- `_score_if` now calls `GB._if_rubric_geometry(text, gt, idx)` and
  `GB._terminal_goal_centroid(text, idx)` — gt_battery's
  fixed-default-threshold implementations — instead of local
  threshold-aware duplicates. The IF rubric's reference geometry (leg
  goals, corridor gates, avoid capsules) and the scene-alignment
  terminal goals are now identical for every candidate config; swept
  `thresholds` reach only the pipeline-under-test
  (`InstructionHead(thresholds=...)` in `_drive_if_trajectory`).
- The local `_terminal_goal_centroid(text, idx, thresholds)` and
  `_if_rubric_geometry(text, gt, idx, thresholds)` near-duplicates
  were DELETED (delegation over frozen copies — one source of truth).
- `_IF_ALIGN_RESIDUAL_GATE_M` now aliases
  `scoring._ALIGN_RESIDUAL_GATE_M` instead of a second hardcoded 1.0
  (single-sourced instrument constant).
- Module + `_score_if` docstrings state the invariant explicitly:
  instrument = frozen, subject = swept.

## Numerical/OR paths audited (no change needed)

`score_numerical` / `score_object_reference` pass `thresholds` only to
OUR resolver (the subject). Their yardsticks —
`_gt_target_from_referential`, `_independent_count`,
`_scene_graph_count`, `_annotation_coverage` — take no thresholds
argument at all. No endogeneity there.

## Test (src/tests/runner/test_cvsweep.py)

`test_if_scoring_uses_frozen_rubric_geometry_across_configs`:
1. `GB._if_rubric_geometry` / `GB._terminal_goal_centroid` output is
   identical regardless of caller-side config;
2. source-pins that `cvsweep._score_if` calls the frozen
   (no-thresholds-arg) forms;
3. end-to-end: `score_scene` under two divergent threshold configs
   must produce identical `if_available` / `if_excluded` (the rubric
   denominator).

Verified passing in isolation before the full-file run. Full-file run
(`python -m pytest tests/runner/test_cvsweep.py -m "" -q`, 30 tests):
completed 100% with no failure markers (log:
scratchpad `cvsweep_full.log`; summary line lost to the session cut —
re-covered by the full gate run after this commit).
