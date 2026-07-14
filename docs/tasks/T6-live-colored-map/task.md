# T6 — live-colored-map

Live colored 3D voxel map + RViz "robot inside the map" debug view.

## Intent

The user wants a live 3D map the robot localises itself within. Localization
is already provided by the challenge base stack (`/state_estimation` pose,
`/registered_scan` map-frame lidar) — this task builds the two missing
pieces:

1. **Incremental colored voxel map** (`core/perception/colored_map.py`):
   the T5 projection/voxel math restructured as a per-tick accumulator
   (ingest one scan + nearest pano at a time, running per-voxel means)
   instead of an offline batch pass. Pure numpy, Windows-testable over the
   replay harness.
2. **RViz integration** (`ros_adapter`, Phase-2 draft): a debug-gated
   `PointCloud2` publisher of the growing colored map on
   `/debug/colored_cloud` + RViz config so the robot's pose (odometry
   display) moves through the colored reconstruction live.

## Scope boundary (binding)

Debug/visualization layer ONLY. The scored answer path stays on the sparse
instance map + 2D costmap per the architecture v1.0 adjudication. Feeding
colored-reconstruction views to a VLM checkpoint remains a separate,
later decision (sanctioned follow-on of T5 once quality is eyeballed).

## Design decisions

- Projection logic moves to `core/perception/pano_projection.py` (single
  source of truth); `tools/colored_cloud.py` delegates to it — its CLI,
  flags, and tests keep working (T5 tests pin the behavior).
- `ColoredVoxelMap`: packed-int64 voxel keys → growing arrays of position/
  color running sums + counts; per-voxel mean position and mean color;
  same gates as T5 (VFOV, min-range 0.75 m, gray for uncolored).
- Offline == online equivalence is a tested invariant: batch tool output
  and sequentially-fed map must agree on the same fixtures/params.
- PointCloud2 packing (xyz + packed-rgb float32 field — RViz's expected
  layout) written as a pure-numpy function with no ROS imports so it is
  unit-testable on Windows; the rclpy node wrapper is thin and carries
  the same "confirm on Ubuntu" inline flags as the rest of `ros_adapter/`.
- Publisher throttled (param, default one publish per ~2 s) so RViz stays
  responsive; map frame `map`.

## Acceptance criteria

1. `ColoredVoxelMap` unit-tested (ingest math, running means, gates,
   negative coords); equivalence test vs the T5 batch tool on jingfan
   keyframes passes.
2. `tools/` suite still green unchanged (behavior pinned through the
   refactor); main `src/` suite collection unaffected by new tests only.
3. PointCloud2 packing function unit-tested on Windows (bytes/fields/
   point_step verified against the PointCloud2 spec).
4. `ros_adapter` publisher + RViz display additions in place, debug-gated,
   Ubuntu-flagged; `sim_verification.md` §2.8 updated with the how-to-see-it
   steps.
5. No install-affecting changes (no new deps) — `ubuntu_setup.md` untouched.

## Todo

- [x] Implement (executor delegation)
- [x] Verifier pass
- [x] Docs + PR (stacked on `tool/colored-cloud`, PR #2)

## Notes

- 2026-07-11: task started on branch `feat/live-colored-map` stacked on
  `tool/colored-cloud` (T5, PR #2). Working tree still carries live
  concurrent suite-tiering edits (`src/pyproject.toml`,
  `src/core/mocks/mock_io.py`, `src/tests/integration/*`,
  `src/tests/integration/_scaled.py`) — T6 commits exclude them.
- 2026-07-12: implementation landed across two executor runs. The first
  died at a session usage limit with the core code and tests complete
  (`core/perception/colored_map.py`, `core/perception/pano_projection.py`,
  `ros_adapter/cloud_packing.py`, the debug-gated colored-cloud publisher
  in `ros_adapter/adapter_node.py`). This run finished the mechanical
  leftovers: the `ColoredMap` RViz `PointCloud2` display on
  `/debug/colored_cloud` in `ai_module_debug.rviz` (odometry display for
  `/state_estimation` was already present, no change needed there), the
  `sim_verification.md` §2.8 write-up, and the missing
  `src/tests/ros_adapter/__init__.py`. Targeted suites green: `tools` 25
  passed; new perception+ros_adapter tests 30 passed, 1 skipped (the
  rclpy import test skips on Windows).
- 2026-07-12: fresh-context verifier — 8/9 claims CONFIRMED, incl. the
  offline≡online equivalence independently reproduced on keyframe
  slices the test never uses (exact agreement; the tools batch path is
  an independent implementation, so this cross-checks two separately
  written accumulators). Claim 9 PARTIALLY REFUTED: `_binned_merge`'s
  per-voxel Python loop cost ~0.4 s/frame at ~114 K voxels — too slow
  for 5 Hz live ingest.
- 2026-07-12: merge vectorized (sorted-key store + `np.searchsorted`;
  dict removed from the hot path): 30-keyframe probe 12.87 s → 0.913 s;
  last-5-frame ingests 0.038–0.045 s (budget 0.2 s). Suites re-run
  green after the fix (tools 25; perception+ros_adapter 31 + 1 skip).
