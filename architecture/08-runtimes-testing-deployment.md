# Runtimes, testing, and deployment

Three ways to drive `QuestionController` (synthetic, bag
replay, live ROS 2), how the container packages the ROS
adapter, and how the test suite plus the two battery harnesses
verify all of it.

## ELI10

The same robot brain runs in three bodies: a fake body made of
placed boxes (for instant offline tests), a recorded body made
from a real sensor recording played back tick by tick (for
checking real-world numbers without a robot), and a real body
plugged into the actual ROS 2 stack (drafted, not yet run - no
Ubuntu machine to test it on). All three feed the identical
`RobotIO` seam, so the reasoning code never knows which one it
is talking to.

## Runtime surfaces

### Synthetic (`MockRobotIO` + `scenegen`)

[src/core/mocks/mock_io.py](../src/core/mocks/mock_io.py)
implements `RobotIO` over a
[src/core/mocks/synthetic_scene.py](../src/core/mocks/synthetic_scene.py)
`SyntheticScene`: a manually-advanced `FakeClock`, odometry
fixed at `(start_x, start_y)` (default `0.5, 0.5`), a black
`(640, 1920, 3)` placeholder pano (`PANO_SHAPE`), and a lidar
scan built by stacking every scene instance's `points`. Publish
calls append to `waypoints` / `markers` / `ints` lists so tests
assert directly on them.

[src/core/runner/scenegen.py](../src/core/runner/scenegen.py)
builds a scene from a question set with no ground truth
involved: `nouns_in_text` extracts every vocabulary noun
left-to-right (longest n-gram first), `build_scene_for` places
one box per noun on a seeded grid (`step=0.7`, `margin=0.6`,
`cols=8`), and doubles any noun that is a numerical-count target
so counting is non-degenerate. Placement is a pure function of
`(scene_name, questions, seed)` - fully deterministic.

### Bag replay (`ReplayRobotIO` + fixtures)

[src/core/replay/bag_reader.py](../src/core/replay/bag_reader.py)
`BagSource` reads a ROS 2 bag via `rosbags` (no ROS install
needed) and converts the five ingest topics
(`/camera/image`, `/registered_scan`, `/terrain_map`,
`/terrain_map_ext`, `/state_estimation`) plus
`/challenge_question` into the core dataclasses. Point clouds
are decoded from each message's own `PointField` offsets, not
an assumed layout.

[src/core/replay/replay_io.py](../src/core/replay/replay_io.py)
`ReplayRobotIO` wraps a time-indexed `MessageStore`
(bisect-based latest-`<=`-now lookup per channel) and a
`ReplayClock`. Once the bag's recorded schedule is exhausted the
clock free-runs at `FREE_RUN_DT_S = 0.2` (5 Hz) over the frozen
final messages, so the FSM's 510 s / 570 s gates still fire
past end-of-bag instead of stalling - this was a real defect,
fixed after the first real-data run (LOG.md, session 8
overnight). `end_of_data` reports whether ticking has moved past
the last scheduled timestamp.

[src/core/replay/fixtures.py](../src/core/replay/fixtures.py)
distils a multi-GB bag into portable `.npz` keyframes: a new
keyframe is cut every `stride=5` camera frames OR after
`move_m=0.5` m translation OR `rot_rad=30 deg` rotation,
whichever fires first; scans/terrain are decimated to
`max_scan_pts=60_000` / `max_terrain_pts=40_000`. The real
jingfan bag distilled 3.16 GB -> 569 MB across 246 keyframes with
zero converter failures (LOG.md, session 8).

Scripted detections: when the runner's `--detections` flag
points at a labels JSON (`jingfan_labels.json` schema), the
runner drives a `PerceptionPipeline` fed by a
`ScriptedPanoDetector`
([src/core/runner/single.py](../src/core/runner/single.py)
`_ScriptedPerception`, around line 186) instead of leaving the
scene index empty. It maps each new pano's timestamp to a
keyframe index by its bisect position in the sorted pano-time
list, fuses the scripted 2D boxes against the real lidar scan
each tick, and hands the resulting live `BasicSceneIndex` to the
heads - the only replay path with real 3D instances. See
[05-perception.md](05-perception.md) for the fusion pipeline
itself. `--detections` requires `--fixtures` (checked in
`core/runner/__main__.py`); the runner raises `ValueError` if
given without a replay `io`.

### ROS 2 adapter (`src/ros_adapter/`)

[src/ros_adapter/adapter_node.py](../src/ros_adapter/adapter_node.py)
is a full `rclpy` `Node` (`AdapterNode`) implementing `RobotIO`
by latching the newest converted message per topic under a
`threading.Lock`, reusing the exact converters from
`core/replay/bag_reader.py` (same message-decoding path proven
against the real bag), and ticking `QuestionController.tick(self)`
at `TICK_HZ = 5.0` via an `rclpy` timer. It subscribes the six
allowed topics with `SENSOR_DATA` QoS (best-effort) except
`/challenge_question`, which uses RELIABLE + TRANSIENT_LOCAL so
a late-joining node still gets the latched question, and
publishes the three answer topics (`/way_point_with_heading`,
`/selected_object_marker`, `/numerical_response`). An optional
`debug_viz` launch arg (default `false`) adds a 1 Hz republish
of the tracked instance map and a waypoint breadcrumb for RVIZ -
gated so the eval path allocates nothing extra when it is off.

Marked **UNTESTED DRAFT** in its own docstring: `rclpy` is
unavailable on the Windows dev box, so this module has never
been imported, let alone run. Two things are explicitly flagged
in-file as unverified-on-Ubuntu:

- The scene index the controller resolves against is a
  permanently empty `BasicSceneIndex([])` (`adapter_node.py`
  around line 228) - Phase 2/3 work is to wire the live
  perception pipeline in here; until then the node only ever
  produces floor answers.
- `core` + the `ament_python` `vla_ai_module` package
  coexisting inside one colcon workspace is untested (Dockerfile
  comment, around the `colcon build` step).

The only executable test today is
[src/tests/ros_adapter/test_importable.py](../src/tests/ros_adapter/test_importable.py):
it `compile()`s `adapter_node.py` and both launch files (catches
syntax errors without importing `rclpy`), YAML-parses the debug
RVIZ config and checks all ten debug/eval topics appear in it,
and only actually *imports* the node via
`pytest.importorskip("rclpy", ...)` - which skips everywhere
except a real ROS box. See
[09-gaps-and-risks.md](09-gaps-and-risks.md) for the unverified
list in full.

## How to run each

```bash
# synthetic (default) - deterministic, offline, no fixtures needed
python -m core.runner "How many chairs are near the table?"

# bag replay via distilled fixtures, with scripted-detection grounding
python -m core.runner "..." --fixtures data/fixtures/jingfan \
    --detections data/fixtures/jingfan_labels.json

# extract fixtures from a raw bag first
python -m core.replay.fixtures extract <bag_dir> <out_dir> --stride 5 --move-m 0.5

# battery: all training questions over synthetic scenes (structural health only)
python -m core.runner.battery

# battery v2: real accuracy against downloaded VLA-3D ground truth
python -m core.runner.battery --groundtruth <unity_root>
# (equivalently: python -m core.runner.gt_battery --groundtruth <unity_root>)

# k-fold calibration sweep
python -m core.runner.cvsweep --groundtruth <unity_root> --n-samples 60
```

Other `single.py` flags worth knowing: `--seed` (synthetic-scene
seed), `--tick-hz` (default 1 Hz from the CLI, vs the adapter's
5 Hz - phase gates are time-based so this only changes wall
time, not structure), `--budget-scale` (compresses the FSM's
510 s / 570 s gates for short bags, e.g. `0.2` puts them at
~102 s / 114 s of bag time - implemented as a `_ScaledClock`
wrapper in `single.py` around line 124, since the gate constants
in `core.fsm` have no injection seam), and `--verbose` (dumps
the flight log).

## Container

[docker/ai_module/Dockerfile](../docker/ai_module/Dockerfile)
builds `FROM zhangjicmu/ubuntu24_ros:ai_module` (the upstream
base image, tag taken verbatim from the challenge's
`docker/compose.yml`), sets
`RMW_IMPLEMENTATION=rmw_cyclonedds_cpp` (required - both
containers must agree on the DDS implementation or topic
discovery silently fails), copies `src/` to `/opt/vla/src` on
`PYTHONPATH`, `pip install`s `numpy`, `rosbags`, `pytest`, then
`colcon build --packages-select vla_ai_module` inside
`/opt/vla/ws` and launches
`ros2 launch vla_ai_module ai_module.launch.py`. The
`torch`/`groundingdino` perception layer is commented out,
gated on a weights decision (`docs/ubuntu_setup.md` §7). LLM
keys are never baked in - they arrive at run time via
`env_file` / environment variables read by
`core/llm/config.py`.

This Dockerfile is marked **UNTESTED DRAFT** in its own header
comment - authored on Windows with no Docker/ROS available to
build or run it; every step is "confirm on Ubuntu."
[docker/ai_module/README.md](../docker/ai_module/README.md)
documents the build/tag/push flow and how to point the
upstream `compose.yml` `ai_module` service at this image instead
of the dummy `dummy_vlm` node.

## Test architecture

737 tests collected (`python -m pytest --collect-only -q` from
`src/`, summed per-file counts, verified 11 Jul 2026 - no
collection errors). LOG.md's own counts track this same suite
growing across the session: 605 -> 642 -> ~711 as perception,
checkpoints, and the overhead-clearance layer landed; 737
reflects the cvsweep tests and later additions since the last
LOG.md count.

Layout mirrors `core/` one-to-one under
[src/tests/](../src/tests/):
`fsm/`, `geometry/`, `groundtruth/`, `heads/`, `llm/`, `mocks/`,
`nav/`, `parsing/`, `perception/`, `replay/`, `runner/`,
`ros_adapter/`, plus `integration/` for cross-module drives and
top-level `test_calibration.py` / `test_plan_schema.py`. No
`@pytest.mark.slow` tiering exists yet - `pyproject.toml`'s
`[tool.pytest.ini_options]` sets only `pythonpath = ["."]`,
`testpaths = ["tests"]`, `addopts = "-q"`; splitting fast/slow
tiers is an open backlog item (LOG.md, session 9 close, item 2).

Determinism rules, enforced by construction rather than lint:
every runtime clock is either a manually-advanced `FakeClock`
(mocks) or a `ReplayClock` walking a fixed recorded schedule
(replay) - nothing reads the wall clock inside `core` logic.
Scene placement (`scenegen.build_scene_for`) and the sweep's
fold assignment / random search
(`core/runner/cvsweep.py` `make_folds`, `run_cv_sweep`) are both
keyed off an explicit `seed` argument; no bare `random`/`numpy`
RNG call runs unseeded in a code path the tests exercise. The
one wall-clock read that does exist -
`run_question`'s `max_wall_s` guard
([src/core/runner/single.py](../src/core/runner/single.py)
around line 327) - is a hang-safety net, not a source of
behavioral variance: the simulated clock, not wall time, drives
every FSM transition.

## Evaluation

### GT battery (`core.runner.gt_battery`)

Scores real accuracy against the downloaded VLA-3D Unity scene
subset - a fully-observed ground truth (`n_obs=3` on every
`InstanceRecord`, so the confidence gate always clears) loaded
by
[src/core/groundtruth/loader.py](../src/core/groundtruth/loader.py)
`load_scene` from each scene's `_object_result.csv` (oriented
boxes, AABB-of-OBB'd - a documented over-approximation for
rotated objects),
`_region_result.csv`, and `_scene_graph.json`.

Per-type scoring
([src/core/groundtruth/scoring.py](../src/core/groundtruth/scoring.py)):
NUMERICAL is exact-match count, with the primary count computed
by running our own resolver over GT geometry - the module
docstring is explicit that this validates *pipeline
self-consistency*, not absolute correctness, and reports an
independent second opinion (from `_referential_statements.json`
or the scene graph) alongside it rather than treating the two as
interchangeable. OBJECT_REFERENCE is 3D IoU between our answer
box and the GT target's AABB, with the target itself flagged
`"ambiguous"` rather than guessed when the question can't be
matched to an annotation. INSTRUCTION_FOLLOWING reports discrete
Fréchet distance plus fraction of the GT path within
`PROXIMITY_M = 1.0` m of ours - "never a single fake composite"
per the module docstring.

Reports land in `reports/gt_battery_<date>/` as
`gt_battery_report.md` + `gt_battery_results.json`; committed
examples include
[reports/gt_battery_full_2026-07-11/](../reports/gt_battery_full_2026-07-11/)
(all 15 training scenes, 75 questions) and two smaller
`gt_battery_loft*` runs. The full-15-scene run recorded in
LOG.md (session 9): numerical pipeline-exact 100% but
independent agreement only 15-27% (an over-counting signal,
flagged as the top calibration target); object-reference
scoreable on 6/30 questions; instruction-following mean Fréchet
6.1 m, 30% coverage@1m on 12/15 aligned scenes (3 scenes
unaligned by the frame-fit residual gate).
`core.runner.battery --groundtruth <dir>` is a thin CLI
delegation into this module (`battery.py` around line 319); the
plain `core.runner.battery` with no `--groundtruth` runs the
separate structural-health path (no ground truth, scores
answered/type-correct/grounded rates against synthetic scenes -
see the module docstring for the distinction) and writes to
`reports/battery_<date>/`, e.g.
[reports/battery_2026-07-11/](../reports/battery_2026-07-11/).

### k-fold cvsweep (`core.runner.cvsweep`)

[src/core/runner/cvsweep.py](../src/core/runner/cvsweep.py)
is code-committed (`git ls-files` confirms it and
`src/tests/runner/test_cvsweep.py`, 20 tests, are tracked) but
has **no committed results** as of this writing. It runs
5-fold leave-3-scenes-out cross-validation over 11 geometry
thresholds plus the numerical `counting.min_obs` gate,
random-searching each fold's train scenes and freezing the
winner on the held-out 3, scored by the same points-weighted
objective the challenge uses (numerical agreement x1, OR IoU
x2, IF coverage x6). A per-`(config, scene)` cache
(`SceneEvaluator`) means every fold that shares a scene reuses
its evaluation rather than re-driving it.

Per
[docs/cvsweep_rerun_brief.md](../docs/cvsweep_rerun_brief.md),
the first run was killed after ~55 CPU-minutes (contending with
other work on the same Windows machine; a process-suspend did
not hold) with `reports/cvsweep_run.*` left at 0 bytes - no
`reports/cvsweep_<date>/` directory exists yet. The brief
documents the rerun command
(`python -m core.runner.cvsweep --n-samples 60 --out ...`) and
says to delete itself once results are committed and adopted;
until that happens, treat the sweep's recommended thresholds as
**not yet produced** rather than pending review.

## References

- Runner CLI entry points:
  [src/core/runner/__main__.py](../src/core/runner/__main__.py),
  [src/core/runner/single.py](../src/core/runner/single.py)
- Synthetic runtime:
  [src/core/mocks/mock_io.py](../src/core/mocks/mock_io.py),
  [src/core/runner/scenegen.py](../src/core/runner/scenegen.py)
- Replay runtime:
  [src/core/replay/bag_reader.py](../src/core/replay/bag_reader.py),
  [src/core/replay/replay_io.py](../src/core/replay/replay_io.py),
  [src/core/replay/fixtures.py](../src/core/replay/fixtures.py)
- ROS adapter:
  [src/ros_adapter/adapter_node.py](../src/ros_adapter/adapter_node.py),
  [src/tests/ros_adapter/test_importable.py](../src/tests/ros_adapter/test_importable.py)
- Container:
  [docker/ai_module/Dockerfile](../docker/ai_module/Dockerfile),
  [docker/ai_module/README.md](../docker/ai_module/README.md)
- Ground-truth evaluation:
  [src/core/groundtruth/loader.py](../src/core/groundtruth/loader.py),
  [src/core/groundtruth/scoring.py](../src/core/groundtruth/scoring.py),
  [src/core/runner/gt_battery.py](../src/core/runner/gt_battery.py),
  [src/core/runner/cvsweep.py](../src/core/runner/cvsweep.py)
- Rerun status:
  [docs/cvsweep_rerun_brief.md](../docs/cvsweep_rerun_brief.md)
- Deployment posture and known-good checks:
  [docs/sim_verification.md](../docs/sim_verification.md),
  [docs/phase2_playbook.md](../docs/phase2_playbook.md)
- Sibling docs: [05-perception.md](05-perception.md),
  [09-gaps-and-risks.md](09-gaps-and-risks.md)
- Tests: `src/tests/` (mirrors `core/` layout); run with
  `python -m pytest` from `src/`
