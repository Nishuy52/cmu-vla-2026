# src/ — module map & conventions

Implementation of `docs/architecture.md` v1.0. Pure-Python core (`core/`), OS-independent,
developed and tested on Windows against mocks/replays; the ROS 2 adapter (`ros_adapter/`) is
written later and only ever *executed* on Ubuntu.

## Layout & ownership (one module = one owner; do not edit outside your module)

| Path | Contents | Status |
|---|---|---|
| `core/interfaces.py` | Frozen dataclasses + `RobotIO` protocol — THE contract. Change only via architecture review | ✅ frozen v1 |
| `core/plan_schema.py` | Typed plan DSL: dataclasses + JSON (de)serialisation + validation | parse-ladder task |
| `core/parsing/` | Checkpoint-1 parse ladder: prompt templates, API/local/regex tiers, fixtures for all 75 training questions | parse-ladder task |
| `core/geometry/` | Spatial toolbox: deterministic predicates, corridor/avoid geometry, threading check | toolbox task |
| `core/nav/` | Occupancy grid from terrain XYZI, frontier extraction/scoring, A* costmap (hard avoid stamps, via-segments), breadcrumb emission | nav task |
| `core/fsm/` | Question lifecycle FSM, 600 s budget gates, watchdog T−30 s answer floor, checkpoint call ledger | fsm task |
| `core/perception/` | Instance map + tracking. v1: interfaces + mock detector only (real models at integration) | scaffold task |
| `core/mocks/` | `MockRobotIO`, synthetic scene generator (rooms + boxed objects + terrain), replay stubs | scaffold task |
| `tests/` | pytest; mirrors module layout (`tests/geometry/…`). Every module ships with its tests | all |
| `ros_adapter/` | rclpy node mapping topics ↔ `RobotIO` (~300 lines) | Phase 2 (Ubuntu) |

## Conventions

- Python 3.12, `numpy` only in the hot path; no ROS imports anywhere under `core/`.
- All frames are the `map` frame; units metres/radians/seconds. `Pose2D.theta` is always published 0.
- Every public function: type hints + a one-line docstring stating units/frames.
- Determinism: no wall-clock or RNG in `core/` logic except via injected `Clock`/seeded generators.
- Tests must run offline (`pytest -q` from `src/`), no network, no GPU.
- Do not commit from module tasks; integration commits happen at review.

## Test tiers

The suite is split into a **fast** default tier and a **slow** tier of full-controller
sims and multi-scene batteries (marked `@pytest.mark.slow`):

- **While iterating** (default): `pytest` — runs the fast tier only (`-m "not slow"` is
  baked into `addopts`). ~40 s, no test over ~2 s.
- **At milestones** (the full gate — run before committing a milestone / opening a PR):
  `pytest -m ""` runs everything (~6 min serial). With the optional `dev` extra installed
  (`pip install -e .[dev]`, adds `pytest-xdist`), `pytest -m "" -n auto` runs it in
  parallel (~2.5 min). `-n` is deliberately **not** in `addopts` so single-test debugging
  stays serial and readable; add it explicitly for the full gate.

The slow tier is where simulated FSM ticking or batteries dominate runtime. Several
integration cases compress the FSM's budget/watchdog gates via a scaled budget clock
(`tests/integration/_scaled.py`, mirroring `core.runner.single._ScaledClock`): only the
FSM's time-budget sees amplified time, so structural assertions (states, answers, driven
geometry) are unchanged while the tick count collapses ~20x. Cases that genuinely need
real-scale timing stay unscaled.
