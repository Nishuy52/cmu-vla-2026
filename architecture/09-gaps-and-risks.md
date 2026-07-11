# 09 - Gaps and risks

Where the built system diverges from its design intent, what is
wired only for offline runs, and where a contributor should
change things. Every claim below was checked against current
source; leads that did not hold are marked so.

## ELI10

The car has a good brain for practicing in a video game (the
mock and replay runs), but the seat it sits in on the real robot
still has its eyes unplugged: on the actual ROS stack it answers
against an empty world and leans on its always-legal backup
answers. A few wires are also crossed - it decides the question
type twice and sometimes trusts the first, weaker guess - and a
couple of warning lights (a "I'm stuck, replan" flag) are wired
in but nothing is listening on the other end.

## Confirmed divergences

### D1 - The ROS answer path resolves against an empty scene

The live ROS node builds the heads against an empty index and
never populates it. In `RosAdapterNode.__init__` the scene is
`self._scene_index = BasicSceneIndex([])`
([../src/ros_adapter/adapter_node.py](../src/ros_adapter/adapter_node.py)
around line 228), and `_on_tick` passes exactly that object into
`build_callables(self._scene_index)` (around line 293). The six
sensor callbacks (`_on_image`, `_on_scan`, `_on_terrain`,
`_on_odom`) latch their converted messages, but nothing casts
pano + scan through a `PerceptionPipeline` into the index - there
is no `PerceptionPipeline`, no detector, and no `pipeline.process`
call anywhere in the adapter node.

Impact on a scored run: on the real robot today every answer
comes from the floor path in
[../src/core/fsm/floors.py](../src/core/fsm/floors.py) -
`IntAnswer(MODAL_COUNT)` = 2 for NUMERICAL (line 31), a
`1x1x1` marker at the origin for OBJECT_REFERENCE (empty scene,
so branch 5 at line 175), and a `WaypointCmd(0.0, 0.0)` for
INSTRUCTION_FOLLOWING (line 194). Navigation still runs - the
`ExploreHead` reads terrain/odom/scan straight off the io and
publishes real exploration waypoints - but the map it answers
against stays empty, so scored accuracy is floor-only. The
offline runners do NOT share this gap: `run_question` grounds a
real `PerceptionPipeline` when given `detections_path`
([../src/core/runner/single.py](../src/core/runner/single.py),
`_ScriptedPerception` around line 186), and mock/GT runs inject a
synthetic scene.

Fix direction: instantiate a `PerceptionPipeline` in the adapter,
step it from the sensor latches each tick (mirror
`_ScriptedPerception.maybe_process`), and hand its
`BasicSceneIndex` to `build_callables` instead of the empty one.
The detector behind it is the second half of this gap (see U1).
See [05-perception.md](05-perception.md) and
[08-runtimes-testing-deployment.md](08-runtimes-testing-deployment.md).

### D2 - Question type is decided twice and never reconciled

The controller classifies the question at intake with a cheap
regex heuristic and never revisits it after the parse lands.
`_intake` sets `self.qtype = _infer_qtype(q)`
([../src/core/fsm/controller.py](../src/core/fsm/controller.py)
around line 149) and builds the budget from it,
`BudgetState(io.clock(), self.qtype)` (line 150). `_tick_parsing`
later stores the parsed `Plan` in `self.plan` but leaves
`self.qtype` untouched (lines 193-201). The heads, meanwhile,
specialise on `plan.qtype` inside `HeadState.bind`
([../src/core/heads/factory.py](../src/core/heads/factory.py)
lines 82-95) and `_final_answer` dispatches on `plan.qtype`
(lines 191-202). There is no reconciliation step.

On disagreement the two authorities split: the head produces the
answer type from `plan.qtype`, but the budget, the early-answer
gate, and the floor all follow the intake guess. `explore_budget`
reads `self.qtype` (210/240/270 s per type,
[../src/core/interfaces.py](../src/core/interfaces.py) line 196),
`_early_answer_ready` branches on `self.qtype`
(controller lines 249-260), and both the watchdog and the
answer-state floor call `self.floors.get(self.qtype)`
(controller lines 228, 284). The dangerous case: if the head
yields nothing and the intake type is wrong, the floor publishes
the wrong answer type on the wrong topic (e.g. an `IntAnswer`
for a question the plan calls OBJECT_REFERENCE), which scores
zero. Even when the head answers, a wrong intake type mis-sizes
the explore budget and can open the NUMERICAL stability gate on
an instruction-following question.

Fix direction: after `_tick_parsing` latches a plan, reconcile
`self.qtype` (and re-seed the budget's `qtype`) from
`plan.qtype`, or make the floor/gate read `plan.qtype` when a
plan exists. Tests would live beside the FSM lifecycle suite.
See [02-core-loop.md](02-core-loop.md),
[03-time-budgeting.md](03-time-budgeting.md), and
[04-parsing-and-checkpoints.md](04-parsing-and-checkpoints.md).

### D3 - The orientation window is dead time before the sweep

`ORIENT` waits; it drives nothing. `_tick_orient` only checks the
clock and transitions once the window closes:
`if not self.budget.in_orientation: self._to(EXPLORE_EXECUTE, ...)`
([../src/core/fsm/controller.py](../src/core/fsm/controller.py)
lines 203-206). It publishes no waypoint and seeds no grid
(grid integration lives in `ExploreHead._explore`, only reached
in `EXPLORE_EXECUTE`). The window is `ORIENTATION_S = 60.0`
([../src/core/fsm/budget.py](../src/core/fsm/budget.py) line 20).
Then, once exploration begins, the `ExplorationPolicy` runs its
OWN opening sweep for `SWEEP_S = 60.0`
([../src/core/nav/exploration.py](../src/core/nav/exploration.py)
line 29), whose clock is anchored on its first step (`t0` in
`_elapsed`), i.e. after the orientation window.

Impact: roughly the first 60 s of a 600 s budget is spent with
the vehicle stationary and the map unseeded, and the actual
panoramic seeding (the diamond sweep) pays a second 60 s after
it - about 20% of the budget consumed before frontier pursuit,
half of it idle. The design intent (an in-place seeding sweep
during ORIENT) is not what the code does.

Fix direction: either drive the diamond sweep from `_tick_orient`
so ORIENT does the seeding, or delete the separate
`ExplorationPolicy` sweep and let ORIENT fall straight through to
frontier pursuit. See
[07-navigation-and-exploration.md](07-navigation-and-exploration.md).

### D4 - General exploration waypoints bypass A* and breadcrumbs

Only the instruction-following path routes through the costmap.
For NUMERICAL and OBJECT_REFERENCE, `ExploreHead._explore`
publishes the raw sweep/frontier point directly -
`io.publish_waypoint(decision.waypoint)`
([../src/core/heads/explore_step.py](../src/core/heads/explore_step.py)
lines 140, 146, 148) - the coordinate straight out of
`ExplorationPolicy.step`. It never calls `astar`, `plan_through`,
or a `BreadcrumbFollower`. Those exist and are wired, but only
inside `InstructionHead` (`plan_through` + `BreadcrumbFollower`
at [../src/core/heads/instruction.py](../src/core/heads/instruction.py)
around lines 223-232). `exploration.py`'s own docstring says the
planner/breadcrumb layer "decides HOW to get there" (line 15),
but for two of three question types nothing does.

Impact: exploration waypoints can point across an obstacle or
under overhead furniture the occupancy grid marks blocked; the
base autonomy stack must absorb that. Lower risk than D1/D2
because the upstream local planner still avoids collisions, but
it forfeits the overhead-clearance and line-of-sight guarantees
the IF path gets.

Fix direction: route `decision.waypoint` through
`plan_through`/`BreadcrumbFollower` before publishing, reusing
the IF machinery. See
[07-navigation-and-exploration.md](07-navigation-and-exploration.md).

### D5 - Signals set but never consumed

Three signals are written and never read outside tests:

- `BreadcrumbFollower.replan_flag`
  ([../src/core/nav/breadcrumbs.py](../src/core/nav/breadcrumbs.py)
  set at lines 121 and 194) is meant to tell the FSM the vehicle
  stalled or a waypoint is unreachable. `InstructionHead._drive`
  reads the follower's returned waypoint but not the flag
  ([../src/core/heads/instruction.py](../src/core/heads/instruction.py)
  around line 295); the only readers are
  `tests/nav/test_breadcrumbs.py`. Stall recovery is therefore
  inert - a wedged vehicle keeps re-publishing the same
  unreachable point.
- `ExploreHead.last_status` (set at
  [../src/core/heads/explore_step.py](../src/core/heads/explore_step.py)
  line 136) and the `ExplorationStatus.COMPLETE` return from
  `ExplorationPolicy.step` are read only in tests. Exploration
  never short-circuits when coverage is saturated; it runs until
  the explore budget or early-answer gate ends it.
- `ExplorationPolicy.step`'s `budget_state={'force_frontier': ...}`
  seam ([../src/core/nav/exploration.py](../src/core/nav/exploration.py)
  lines 92-102) is never passed by any caller, so the FSM cannot
  cut the sweep short under clock pressure.
- `ObjectRefHead.remaining_s`
  ([../src/core/heads/object_ref.py](../src/core/heads/object_ref.py)
  line 74, default `None`) is the seam CP4 reads to enforce its
  90 s re-resolve rule (`RE_RESOLVE_MIN_REMAINING_S = 90.0`,
  [../src/core/checkpoints/verification.py](../src/core/checkpoints/verification.py)).
  Neither live call site feeds it from `BudgetState.remaining()`:
  `run_question` (single.py around line 315) and `_on_tick`
  (adapter_node.py around line 293) both omit `remaining_s=`, so
  CP4 always sees `None` (treated as +inf) and would never decline
  a re-resolve on a low clock.

Fix direction: consume `replan_flag` in `InstructionHead._drive`
(force a re-plan / nudge) and let the controller end
`EXPLORE_EXECUTE` on `ExplorationStatus.COMPLETE`.

### D6 - Only 2 of 6 checkpoint caps run through the shared ledger

`CHECKPOINT_MAX` defines six caps
([../src/core/fsm/budget.py](../src/core/fsm/budget.py) lines
97-104), but only `parse` and `verification` are actually gated
by `ledger.allow(...)` - both in `QuestionController`
([../src/core/fsm/controller.py](../src/core/fsm/controller.py)
around lines 194 and 218). The other four are enforced elsewhere,
or not at all:

- `anchor_confirm` (cap 3) fires once per route leg via a local
  `_confirmed` set in `InstructionHead`
  ([../src/core/heads/instruction.py](../src/core/heads/instruction.py)
  around line 314), not the ledger - a plan with more than 3 legs
  is not capped at 3.
- `miss_recovery` (cap 1) fires once per question via a local
  `_cp2_fired` flag in `ExploreHead`
  ([../src/core/heads/explore_step.py](../src/core/heads/explore_step.py)
  around line 185), not the ledger.
- `frontier_select` (cap 1) has no once-only guard at all:
  `ExploreHead._maybe_cp5_frontier`
  ([../src/core/heads/explore_step.py](../src/core/heads/explore_step.py)
  around line 151) is re-evaluated every multi-room
  `EXPLORE_EXECUTE` tick, and never consults
  `ledger.allow("frontier_select")`.
- `self_consistency` (cap 2) has no implementing module and zero
  call sites anywhere in `core/`; the cap is defined and pinned by
  `test_ledger_caps_from_architecture` but nothing spends it.

Impact: the `LEDGER_RESERVE_S = 45.0` floor reserve only protects
the two checkpoints wired through the ledger; a CP5 storm in a
multi-room scene is bounded by wall time, not by cap 1. Fix
direction: route the head-level checkpoints through
`ledger.allow(...)`/`ledger.record(...)` instead of local flags,
or delete the unused caps. See
[03-time-budgeting.md](03-time-budgeting.md) and
[04-parsing-and-checkpoints.md](04-parsing-and-checkpoints.md).

### D7 - The live parse path is regex-only

`build_callables`'s default `parse` is `_default_parse`, which
calls `core.parsing.regex_tier.parse_regex` directly and
**bypasses `core.parsing.ladder.parse` entirely**
([../src/core/heads/factory.py](../src/core/heads/factory.py)
around lines 157 and 228-233). No live entry point overrides it
with a ladder-backed function: the ROS adapter calls
`build_callables(self._scene_index)` with no keyword args
([../src/ros_adapter/adapter_node.py](../src/ros_adapter/adapter_node.py)
around line 293), and `run_question` forwards only `llm_verify`
and `anchor_confirm` into `build_callables`, never `parse`
([../src/core/runner/single.py](../src/core/runner/single.py)
around lines 315-319). The CLI/battery/cvsweep runners never build
`chat_fns` at all.

Impact: every run through any real (non-test) entry point parses
with the deterministic regex tier; the LLM parse tiers
(`api`/`api2`/`local`) and the whole `ladder.parse` repair loop
are reachable only from tests. The regex tier is total and
correctly typed on all 75 training questions, so this is a
capability ceiling (novel phrasing degrades to a best-effort
Plan), not a crash risk. Fix direction: thread a
`load_config`/`build_chat_fns`-backed `parse=` into
`build_callables` at the CLI and adapter entry points. See
[04-parsing-and-checkpoints.md](04-parsing-and-checkpoints.md).

### D8 - Tracking is greedy nearest-neighbour, not ByteTrack

`docs/architecture.md` §6 specifies "ByteTrack-style association
in NumPy". The implementation is simpler: `associate()`
([../src/core/perception/tracker.py](../src/core/perception/tracker.py))
does greedy nearest-neighbour bipartite matching on centroid
distance (`TrackerConfig.gate = 0.75` m), and the actual dedup
authority is 3D AABB IoU inside `BasicSceneIndex.add()`
(`MERGE_IOU = 0.3`, strict `>`,
[../src/core/perception/scene_index.py](../src/core/perception/scene_index.py)
line 21) - not a ByteTrack-style track state machine. Both
`FusionConfig` and `TrackerConfig` self-flag their constants as
non-spec engineering defaults.

Impact: adequate on the one real bag and the test suite, but
double-count behaviour under occlusion/re-entry is untuned against
the challenge spec, and feeds the counting over-count risk below.
Fix direction: either adopt real ByteTrack track states or update
the design doc to record the greedy+IoU choice as intended. See
[05-perception.md](05-perception.md).

### Confirmed correct (not a gap)

Object-reference final dispatch is marker-only, as the contract
requires: `_final_answer` returns `ObjectRefHead.verify()`, whose
only return is a `MarkerBox` or None
([../src/core/heads/object_ref.py](../src/core/heads/object_ref.py)
lines 111-127). No navigation waypoint is emitted for an OR
answer. This was a checked lead, and it holds.

## Unwired or placeholder subsystems

| # | Subsystem | State today | Evidence | Blocks |
|---|---|---|---|---|
| U1 | Real detector (GroundingDINO) | Stub; `__call__` raises `NotImplementedError` after a lazy torch import guard | [../src/core/perception/detector.py](../src/core/perception/detector.py) line 182 | Real perception on ROS (feeds D1) |
| U2 | ROS scene index / perception fusion | Empty `BasicSceneIndex([])`, no pipeline in the node | [../src/ros_adapter/adapter_node.py](../src/ros_adapter/adapter_node.py) line 228 | All real answers (D1) |
| U3 | Rich checkpoint seams (CP2 miss-recovery, CP3 anchor-confirm, CP4 verifier, CP5 frontier-select) | Injection points exist; all default `None` == deterministic-only behaviour | [../src/core/heads/factory.py](../src/core/heads/factory.py) lines 59-69, 132-141 | LLM-assisted resolution |
| U4 | CP2 provisional-instance fusion | Default `_ProvisionalInstance` biases nav only; never becomes a tracked `InstanceRecord` (`n_obs=1`, cannot satisfy the >=3-obs gate) | [../src/core/heads/explore_step.py](../src/core/heads/explore_step.py) lines 235-247, 305-318 | Recovered-object answers |
| U5 | ros_adapter node | Untested draft; parse-guard test only `compile()`s it, never imports | [../src/ros_adapter/adapter_node.py](../src/ros_adapter/adapter_node.py) header; `tests/ros_adapter/test_importable.py` | Any on-robot run |
| U6 | Docker perception layer | torch/torchvision/groundingdino install commented out pending the weights decision | [../docker/ai_module/Dockerfile](../docker/ai_module/Dockerfile) (last lines) | Container with real detector |
| U7 | Calibration wiring | `Calibration` dataclass mirrors the constants but most fields have no setter; injection is "Phase-2 wiring TODO" | [../src/core/calibration.py](../src/core/calibration.py) lines 10-30 | Sweeping budget/nav/fusion params (only geometry `Thresholds` + `min_obs` sweep today) |

Declared-but-never-read constants (dead config, harmless but
misleading if trusted as live knobs):

- `COVERAGE_SATURATED_FREE_FRAC` in `exploration.py` (line 32) -
  frontier absence, not this fraction, is the real completion gate.
- `FusionConfig.depth_bin` (0.25 m,
  [../src/core/perception/fusion.py](../src/core/perception/fusion.py)
  line 47) - the docstring calls the clustering a
  "1D histogram / range-DBSCAN", but the real algorithm is the
  0.5 m gap-split in `_nearest_cluster`; `depth_bin` is never read.
- `Thresholds.above_gap_max` (3.0 m,
  [../src/core/geometry/toolbox.py](../src/core/geometry/toolbox.py)
  line 47) - declared on `Thresholds` but never referenced in the
  `above()`/`under()` bodies, which impose no vertical-gap ceiling.

## Calibration and accuracy risks

From the first ground-truth battery (15 scenes, 75 questions;
report and per-question JSON under
[../reports/gt_battery_full_2026-07-11/](../reports/gt_battery_full_2026-07-11/)):

- Counting over-counts. Pipeline-exact reproduction is 100%, but
  independent agreement with the annotations is only 15-27% - the
  counter systematically sees more instances than the ground
  truth. Top calibration target.
- Object-reference is mostly unscoreable. Only 6 of 30 OR
  questions scored (4 perfect IoU, 2 wrong-instance); the other
  24 are vocabulary-drift non-matches (our label vocabulary does
  not line up with the annotation vocabulary). This is a scoring
  ceiling, not just a tuning knob.
- Three instruction-following scenes are unaligned. Mean Fréchet
  distance is 6.1 m with 30% coverage@1m across the 12 of 15
  aligned scenes; `home_building_2`, `hotel_room_2`, and
  `livingroom_3` are unaligned, with goal disambiguation the
  suspected cause.

These numbers come from the offline GT battery with a fully
observed synthetic scene, so they measure the geometry/counting
logic, not the (still unwired) real perception stack. Treat them
as an upper bound on the logic and a floor on the total system
error once D1/U1 land.

## Unverifiable until Ubuntu

Nothing below can be confirmed on the Windows dev box; each needs
the native ROS/sim stack.

- Adapter import and the 5 Hz drive. `adapter_node.py` imports
  every ROS symbol at module top and is only `compile()`-checked
  offline; whether it imports and ticks under rclpy is untested.
- Sim topic QoS / discovery. Both containers must agree on
  CycloneDDS (`RMW_IMPLEMENTATION=rmw_cyclonedds_cpp` in the
  Dockerfile) or topic discovery silently fails; the reliable-QoS
  question latch is likewise unconfirmed against the live
  publisher.
- Real detector inference. `GroundingDinoDetector.__call__` past
  the import guard is Phase-2 code that has never run (U1).
- Docker image build with the perception layer (U6) and the
  `ros2 run vla_ai_module adapter_node` entry point resolving
  `core` on `PYTHONPATH` (noted as "confirm on Ubuntu" in the
  package setup).

## Where to change what

| To change... | Edit | Tests |
|---|---|---|
| Real perception on ROS (populate the scene index) | `ros_adapter/adapter_node.py` (`__init__` scene, `_on_tick`); wire a `PerceptionPipeline` | new `tests/ros_adapter/*` + `tests/perception/test_tracker.py` |
| The open-vocab detector | `core/perception/detector.py` (`GroundingDinoDetector.__call__`) | `tests/perception/test_detector.py` (stub path) |
| Question-type reconciliation | `core/fsm/controller.py` (`_tick_parsing` / `_intake`) | `tests/fsm/` lifecycle suite |
| Budget gates / floors timing | `core/interfaces.py` constants + `core/fsm/budget.py` | `tests/fsm/test_budget.py` |
| Orientation vs sweep (D3) | `core/fsm/controller.py` `_tick_orient` and/or `core/nav/exploration.py` `SWEEP_S` | `tests/nav/test_exploration.py`, `tests/fsm/` |
| Route exploration through A* (D4) | `core/heads/explore_step.py` `_explore` (call `plan_through`) | `tests/heads/test_explore_step.py`, `tests/nav/test_planner.py` |
| Consume the stall/replan flag (D5) | `core/heads/instruction.py` `_drive` (read `follower.replan_flag`) | `tests/nav/test_breadcrumbs.py`, `tests/heads/test_instruction.py` |
| Object-reference ranking / verify | `core/heads/object_ref.py`; thresholds in `core/geometry/toolbox.py` | `tests/heads/test_object_ref.py`, `tests/geometry/` |
| Counting over-count (calibration) | `core/geometry/toolbox.py` thresholds, numerical `min_obs` | `src/core/runner/cvsweep.py` sweep, `tests/heads/test_numerical.py` |
| Floor / degraded answers | `core/fsm/floors.py` | `tests/fsm/test_floors.py` |

## Design rationale

The wide gap between offline behaviour (grounded, scored) and the
ROS path (floor-only) is deliberate sequencing, not oversight:
`core/` is fully offline-testable with mocks and replays, and the
single ROS-dependent module is quarantined behind a one-way
`ros_adapter -> core` dependency so the test suite never needs
ROS. The cost is that the most load-bearing integration - feeding
a real scene index in D1/U1/U2 - is exactly the part that cannot
be exercised until the Ubuntu box exists.

## References

Entry points and modules
- [../src/core/runner/single.py](../src/core/runner/single.py) -
  live offline composition path (`run_question`).
- [../src/ros_adapter/adapter_node.py](../src/ros_adapter/adapter_node.py) -
  the ROS drive loop (empty scene today).
- [../src/core/fsm/controller.py](../src/core/fsm/controller.py) -
  lifecycle FSM, intake qtype, watchdog/floor dispatch.
- [../src/core/heads/factory.py](../src/core/heads/factory.py) -
  head wiring, plan-qtype authority, checkpoint seams.
- [../src/core/heads/explore_step.py](../src/core/heads/explore_step.py),
  [../src/core/nav/exploration.py](../src/core/nav/exploration.py) -
  exploration + raw-waypoint publication.
- [../src/core/nav/breadcrumbs.py](../src/core/nav/breadcrumbs.py) -
  `replan_flag` (unconsumed).
- [../src/core/perception/detector.py](../src/core/perception/detector.py) -
  detector stub.
- [../src/core/fsm/floors.py](../src/core/fsm/floors.py) -
  degraded answers.

Tests
- `tests/ros_adapter/test_importable.py` - parse-guard only.
- `tests/nav/test_breadcrumbs.py`,
  `tests/heads/test_explore_step.py` - where the unconsumed
  signals are asserted.

Reports
- [../reports/gt_battery_full_2026-07-11/](../reports/gt_battery_full_2026-07-11/) -
  GT battery report + per-question JSON (calibration numbers).

Sibling docs
- [02-core-loop.md](02-core-loop.md),
  [03-time-budgeting.md](03-time-budgeting.md),
  [04-parsing-and-checkpoints.md](04-parsing-and-checkpoints.md),
  [05-perception.md](05-perception.md),
  [06-answer-heads.md](06-answer-heads.md),
  [07-navigation-and-exploration.md](07-navigation-and-exploration.md),
  [08-runtimes-testing-deployment.md](08-runtimes-testing-deployment.md).
