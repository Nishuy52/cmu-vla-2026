# T9 — Test-suite tiering

## Intent

The `src/` pytest suite grew to ~6 min because a handful of full-controller
simulations and multi-scene batteries dominate wall time (each drives the FSM's
600 s budget clock in sim-time at 5 Hz = thousands of ticks). That is too slow for
the tight edit→test loop. Split the suite into a **fast default tier** (run while
iterating) and a **slow milestone gate** (run before commits/PRs), speed up the slow
integration tests themselves via the existing budget-scale mechanism, and make the
full gate parallelisable with `pytest-xdist` — all without losing a single test or
weakening any assertion.

## Context

- `core/runner/single.py` already has the time-compression pattern: `_ScaledClock`
  amplifies elapsed time by `1/budget_scale` as the FSM's `BudgetState` reads it, so a
  gate at G s fires once the underlying clock has moved only `G*scale` s. It is installed
  via the `set_budget_clock` seam, which previously existed only on `ReplayRobotIO`.
- Integration cases (`tests/integration/**`) assert *structure* — published states,
  answer objects, driven-trajectory geometry, seam-call counts — never absolute wall or
  sim timing, so sim-time can be compressed without touching assertions. The follower
  steps a fixed distance per tick, so trajectory geometry is scale-invariant.
- `core/fsm` (where the 510/570 s gate constants live) is off-limits and has no injection
  seam, so scaling is done by wrapping the clock, not by injecting scaled constants.

## Acceptance criteria

- [x] `slow` marker registered in `src/pyproject.toml`; `-m "not slow"` in `addopts` so
      plain `pytest` runs the fast tier. Full gate is `pytest -m ""`.
- [x] Every test whose runtime is dominated by simulated ticking / batteries and measures
      > ~2 s is marked `@pytest.mark.slow` (measured via `--durations`). No unmarked test
      exceeds ~2 s.
- [x] Fast + slow counts sum to the current total (zero tests lost).
- [x] Slow integration tests sped up via budget-scale where assertions are structural; no
      assertion weakened; real-scale-timing cases left unscaled.
- [x] `pytest-xdist` added as an optional `dev` dependency; `-n` NOT in `addopts`; full
      gate documented as `pytest -m "" -n auto`.
- [x] `src/README.md` test-tier conventions + CLAUDE.md session-protocol steps 2/3 updated.
- [x] Fast tier green + timed; full gate green + timed with and without xdist (xdist-masked
      ordering-dependency check).

## Results (2026-07-12)

Counts: **737 total = 692 fast + 45 slow** (verified via `--collect-only`; sums, zero lost).

Timings:

| Run | Command | Wall | Result |
|---|---|---|---|
| Fast tier | `pytest` | ~38 s | 685 passed, 7 skipped |
| Full gate (serial) | `pytest -m ""` | ~374 s (6m13s) | 725 passed, 12 skipped |
| Full gate (parallel) | `pytest -m "" -n auto` | ~152 s (2m32s) | 725 passed, 12 skipped |

Serial and parallel full-gate runs have identical pass/skip counts → **no xdist-masked
ordering dependency**; suite is order-independent.

Tests marked slow: **45** (integration/* [3 files, module-level], runner/test_battery.py
[module-level], and per-test in runner/{test_single, test_single_scripted, test_gt_battery,
test_cvsweep, test_cli}, heads/{test_instruction, test_instruction_cp3,
test_instruction_recovery}, perception/{test_tracker, test_colored_map}). Count expands
past the ~30 function-defs because gt_battery/parsing cases are parametrized.

Integration tests converted to budget-scale (`scale=0.05`, ~20x fewer ticks), before→after
call time (unscaled baseline measured with `BUDGET_SCALE=1.0`; both scales pass, proving the
assertions are scale-invariant):

| Test | Before | After |
|---|---|---|
| test_capsule_engulfs_gate_only_full_trajectory_clear | 84.8 s | 3.6 s |
| test_capsule_engulfs_start_and_gate_recovery_never_violates | 53.7 s | 4.0 s |
| test_object_reference_completes_with_all_seams_active | 52.4 s | 1.2 s |
| test_instruction_following_threads_corridor_avoids_capsule_and_publishes_terminal | 44.5 s | 2.3 s |
| test_instruction_following_completes_with_all_seams_active | 43.2 s | 2.1 s |
| test_object_reference_publishes_correct_marker | 34.9 s | 1.1 s |
| test_absent_target_object_reference_falls_to_floor_never_silent | 27.4 s | 1.0 s |
| test_absent_target_numerical_answers_zero_never_silent | 22.3 s | 0.8 s |
| test_numerical_publishes_correct_int_before_watchdog | 0.15 s | 0.11 s |

Integration suite total ~363 s → ~16 s. The `test_numerical...` case already finished on a
head verify before any budget gate, so scaling barely moves it; its `< 570 s` watchdog
assertion was retargeted from raw ticked time to `ctrl.budget.elapsed()` (the FSM's own
scaled budget time — the quantity the 570 s gate is actually measured against), keeping the
"answered before watchdog" semantic honest under compression.

Production-code touch (justified test-support seam): `core/mocks/mock_io.py` gained
`set_budget_clock` / `raw_clock` and a budget-clock-aware `clock()` — an exact mirror of the
seam `core/replay/replay_io.py` already exposes. No behaviour change: the wrapper is inert
unless a test installs it; `latest_*` getters still stamp off the real clock.

Left slow, not converted: runner batteries (`test_battery.py`, `test_gt_battery.py`) and
`test_single.py` cases assert against their own `sim_elapsed` / `answered_before_watchdog`
report fields at real budget scale, and `test_instruction_recovery.py`'s drive test loops
motion (no FSM budget clock to scale). `test_fixtures.py` (replay) and the perception long
tail are all sub-2 s per test, so not marked — the residual ~38 s fast tier is broad
sub-2 s mass, not a few offenders. Reaching < 30 s would need either marking sub-2 s tests
(against the > 2 s rule) or `-n` in `addopts` (against the single-test-debug requirement),
so the fast tier is reported honestly at ~38 s.

## Notes

- 2026-07-12: implemented on branch `test/suite-tiering`. `upstream/` is git-ignored and
  absent from fresh worktrees; a directory junction into the primary checkout's `upstream/`
  restores `questions.json` + GT scenes so the full gate collects all 737 tests (not
  committed — junction is ignored).
