# Issue #88 — move GDINO forward off the tick thread

## Problem
`_maybe_process_perception()` was called inline from `AdapterNode._on_tick()`
(the 5 Hz rclpy timer). `PerceptionPipeline.process()` calls the detector
synchronously, and a live GroundingDINO forward takes ~1s -> the entire timer
callback blocks for ~1s -> waypoint publication and subscription servicing
stall for that whole second. Measured: ~176 waypoints/210s explore (~0.8 Hz)
vs ~1050/210s (5 Hz) with the detector off.

## Design chosen
- New module `src/core/perception/async_pipeline.py`:
  `AsyncPerceptionWorker` — one dedicated thread per pipeline, `submit(pano,
  scan)` is non-blocking and latest-frame-wins (overwrites any not-yet-picked
  -up pending frame; no backlog). The worker thread is the one that calls
  `pipeline.process()`.
  - Fully rclpy-free and independently unit-testable (this box has no rclpy
    installed at all — Windows-testable and Ubuntu-testable the same way).
- `src/core/perception/scene_index.py`: `BasicSceneIndex` gained an internal
  `threading.RLock` guarding every read (`all_instances`, `by_label_tiered`,
  `next_id`) and write (`add`, `remove`, `_fuse`) method. This is the actual
  thread-safety of the hand-off: the worker thread mutates the index while
  the tick thread's heads concurrently read it to resolve answers. No caller
  changes needed anywhere else (nav/heads untouched, as required).
- `src/ros_adapter/adapter_node.py`:
  - `_maybe_process_perception()` now calls `self._perception_worker.submit()`
    instead of `self._perception.process()` directly (when threaded).
  - New env var `VLA_PERCEPTION_SYNC` (default off) forces the old inline
    synchronous call — debug-only escape hatch, never for live/eval.
  - `AsyncPerceptionWorker` constructed once at boot alongside
    `self._perception`, stopped cleanly in a new `destroy_node()` override.
- Offline/battery determinism: `core.runner.single` (the battery/offline
  path) never imports `ros_adapter.adapter_node` and calls
  `PerceptionPipeline.process()` directly itself — it is untouched by this
  change and stays byte-deterministic with no env gating needed on that side.

## Tests added
`src/tests/perception/test_async_pipeline.py` (6 tests, all pass):
- `test_submit_never_blocks_on_a_slow_forward` — submit() returns in <0.1s
  even while the worker is mid a 0.5s forward.
- `test_tick_cadence_survives_a_slow_detector_forward` — simulates the 5 Hz
  executor tick loop for 1.5s against a 0.5s fake forward; asserts tick count
  stays within 60% of the ideal 5 Hz*duration (nowhere near the ~0.8 Hz
  collapse the issue reports), and that `processed_count < submitted_count`
  (latest-frame-wins dropping stale frames, not backlogging).
- `test_synchronous_baseline_reproduces_the_stall_for_contrast` — confirms
  the *inline* call (pre-fix behaviour) DOES collapse cadence to
  roughly one tick per forward, so the harness would actually have caught
  the regression.
- `test_latest_frame_wins_drops_stale_pending_frame` — four rapid submits
  while the worker is busy on frame 0; the worker only ever sees frame 0 and
  the final submitted frame (2 calls out of 4 submits).
- `test_concurrent_submit_and_stop_do_not_raise` — hand-off race: four
  threads hammering submit() while stop() races in; no exceptions, no
  deadlock.
- `test_scene_index_concurrent_read_write_does_not_raise` — writer thread
  doing add/remove on `BasicSceneIndex` concurrently with two reader threads
  calling `all_instances`/`by_label` for 0.3s; asserts no
  `RuntimeError: list changed size during iteration` (the actual bug the
  RLock fixes).

## Test evidence
```
$ python3 -m pytest tests/perception/test_async_pipeline.py -v
6 passed in 5.08s
```
Fast tier (`pytest` from `src/`): 1362 passed, 66 skipped, 56 deselected,
39 errors (pre-existing — confirmed via `git stash -u` on the unmodified
baseline: identical 1356 passed/1 failed/39 errors count before this change's
tests are added; the 39 errors are `ModuleNotFoundError` on
`tests/replay/*` and `tests/parsing/test_regex_full_set.py`, unrelated to
this work — likely a missing `rosbag2_py`/parsing fixture dependency in this
worktree's venv). The pre-existing failure
(`tests/runner/test_gt_battery.py::test_synthetic_scene_extra_wall_cells_mark_terrain_obstacle`)
is also present on the unmodified baseline — not introduced by this change.

## Ownership notes / handoff
- Touched: `src/ros_adapter/adapter_node.py` (dispatch only — `_maybe_process_
  perception`, constructor wiring, `destroy_node`), `src/core/perception/
  scene_index.py` (lock only), new `src/core/perception/async_pipeline.py`.
- NOT touched: `_on_terrain*`, anything occupancy/costmap, `src/core/nav/**`,
  `src/core/heads/factory.py`, `core/perception/detector.py` prompt/gate
  logic — confirmed untouched (grep verified, diff reviewed).
- `docs/ubuntu_setup.md` updated with the new `VLA_PERCEPTION_SYNC` env var
  (perception dispatch section, near the cluster offload table) per the
  standing "keep ubuntu_setup.md current" rule.
- Nothing further needed for cluster integration beyond the new
  (optional, default-off) `VLA_PERCEPTION_SYNC` env var — no other env
  changes required; the threaded path is on by default the moment
  `VLA_DETECTOR` selects a real detector.

## Status: DONE
Code + tests committed. See commit log for hashes.
