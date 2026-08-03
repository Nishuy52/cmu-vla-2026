"""run_question happy-path tests: one per qtype on a synthetic scene.

Each asserts the pipeline publishes a legal answer of the right type, in plausible
*simulated* time, and completes far faster in CPU time than its simulated budget (the
clock is simulated, so a full 600 s sim budget must still run in a few CPU-seconds).

The guard exists to catch a "sim runs in real time" regression, not to pin a tight
latency SLA. It is measured via ``time.process_time()`` (CPU time consumed by this
process) rather than wall-clock time: on a machine that routinely runs several
concurrent test/agent processes, wall time inflates with contention (observed up to
~59 s wall for a drive that costs ~4 s of actual CPU work — see #164), which produced
spurious failures unrelated to any real regression. CPU time is immune to that kind of
contention (a process not scheduled simply doesn't accrue CPU time), so it preserves the
real invariant — the sim must not burn CPU proportional to sim time — without the flake.
See ``_CPU_BUDGET_S`` below.
"""
from __future__ import annotations

import time

import pytest

from core.interfaces import IntAnswer, MarkerBox, QType, WaypointCmd
from core.mocks.mock_io import FakeClock, MockRobotIO
from core.runner.scenegen import build_scene_for
from core.runner.single import run_question

# Coarse tick: the FSM phase gates are time-based, so 1 Hz reproduces the same structural
# behaviour as 5 Hz with far less mock terrain/frontier work — keeping wall time tiny.
_HZ = 1.0

# CPU-time guard for the run_question-driving cases. The simulated clock means a full
# 600 s sim budget costs only a few CPU-seconds; this bound catches a regression that
# lets the sim run in real time (which would show as ~600 s), while staying loose enough
# to survive heavy contention from other concurrently-running processes on this machine —
# CPU time (time.process_time()) only accrues while this process is actually scheduled,
# so unlike wall time it does not inflate when other agents/tests are running in parallel.
# On an idle machine these drives take ~3-4 s of CPU; the bound is set at 20x below the
# sim budget rather than a tight SLA, giving generous headroom for slower hardware.
_CPU_BUDGET_S = 30.0


def _io_for(question: str, bucket: str) -> MockRobotIO:
    sc, _ = build_scene_for("t", {bucket: [question]})
    return MockRobotIO(sc, FakeClock(0.0))


def test_numerical_happy_path_publishes_int_before_watchdog():
    q = "How many chairs are near the table?"
    io = _io_for(q, "numerical")
    t0 = time.process_time()
    r = run_question(q, io, tick_hz=_HZ)
    cpu = time.process_time() - t0

    assert r.published is True
    assert isinstance(r.answer, IntAnswer)
    assert r.qtype in QType  # a real qtype was resolved
    assert 0.0 < r.elapsed_sim_s < 570.0  # answered before the watchdog floor
    assert cpu < _CPU_BUDGET_S


@pytest.mark.slow
def test_object_reference_happy_path_publishes_marker():
    q = "Find the vase on the table"
    io = _io_for(q, "object_reference")
    t0 = time.process_time()
    r = run_question(q, io, tick_hz=_HZ)
    cpu = time.process_time() - t0

    assert r.published is True
    assert isinstance(r.answer, MarkerBox)
    assert 0.0 < r.elapsed_sim_s <= 570.0
    assert cpu < _CPU_BUDGET_S


@pytest.mark.slow
def test_instruction_following_happy_path_publishes_waypoint():
    """Regression guard against the sim burning CPU time proportional to sim time.

    Measured via ``time.process_time()`` (this process's CPU time) rather than wall
    time: wall time inflates under concurrent-process contention on this machine
    (observed ~59 s wall vs a genuine ~4 s of CPU work under load — see #164) which
    produced spurious failures unrelated to any real regression. CPU time only accrues
    while this process is actually scheduled, so it is immune to that contention while
    still catching a real "sim runs in real time" bug. The 30 s budget is ~7x the
    idle-machine cost (~4 s), a generous margin for slower/loaded hardware.
    """
    q = "Go to the table and stop at the chair"
    io = _io_for(q, "instruction_following")
    t0 = time.process_time()
    r = run_question(q, io, tick_hz=_HZ)
    cpu = time.process_time() - t0

    assert r.published is True
    assert isinstance(r.answer, WaypointCmd)
    assert 0.0 < r.elapsed_sim_s <= 570.0
    assert cpu < _CPU_BUDGET_S


def test_flight_log_and_states_populated():
    q = "How many chairs are near the table?"
    io = _io_for(q, "numerical")
    r = run_question(q, io, tick_hz=_HZ)

    # the flight recording captured the run and the states span the normal lifecycle
    assert len(r.flight_log) > 0
    assert "done" in r.states_visited
    assert "explore_execute" in r.states_visited
    assert r.checkpoint_calls >= 1  # at least the parse checkpoint fired


@pytest.mark.slow
def test_run_question_is_deterministic():
    q = "Find the vase on the table"
    r1 = run_question(q, _io_for(q, "object_reference"), tick_hz=_HZ)
    r2 = run_question(q, _io_for(q, "object_reference"), tick_hz=_HZ)
    assert r1.elapsed_sim_s == r2.elapsed_sim_s
    assert type(r1.answer) is type(r2.answer)
    assert r1.states_visited == r2.states_visited


@pytest.mark.slow
def test_simulated_clock_keeps_wall_far_below_sim_time():
    """Regression guard for the class of bug the per-test ``cpu < _CPU_BUDGET_S`` bound
    protects against: the clock silently ceasing to be *simulated* and the drive burning
    real time proportional to sim time (which would show as ~hundreds of CPU-seconds
    for a run that advances ~hundreds of sim-seconds).

    Measured via the ratio ``elapsed_sim_s / cpu_s`` where ``cpu_s`` comes from
    ``time.process_time()`` wrapped around ``run_question`` in this test (not wall time
    via ``time.monotonic()``/``r.wall_s``). Wall time inflates under concurrent-process
    contention on this machine — observed wall of 58.76 s for a run whose actual CPU cost
    was a few seconds, collapsing the wall-based ratio from a comfortable margin to ~4.7
    and failing spuriously (see #164). CPU time only accrues while this process is
    scheduled, so contention from unrelated concurrent processes does not inflate it,
    while a genuine "sim runs in real time" regression still burns proportional CPU and
    still fails loudly. The >=10x ratio bound keeps a generous margin: on an idle machine
    the ratio is closer to ~50x, so even a partial regression trips this well before the
    literal real-time case.
    """
    q = "Go to the table and stop at the chair"
    t0 = time.process_time()
    r = run_question(q, _io_for(q, "instruction_following"), tick_hz=_HZ)
    cpu = time.process_time() - t0

    assert r.elapsed_sim_s > 100.0  # the drive really advanced the simulated budget
    assert cpu > 0.0
    # A simulated clock makes CPU time a tiny fraction of sim time. CPU time (unlike wall
    # time) is not inflated by other concurrently-running processes on this machine, so a
    # >=10x ratio has ample headroom while still catching a real-time-clock regression.
    assert r.elapsed_sim_s / cpu > 10.0


@pytest.mark.slow
def test_unknown_noun_completes_via_floor_never_silent():
    """Regression: a question with an unknown noun still completes via the floor."""
    q = "Find the wibblesprocket near the flibber"
    io = _io_for(q, "object_reference")
    r = run_question(q, io, tick_hz=_HZ)

    from core.runner.battery import _floor_used

    assert r.published is True  # answered_before_watchdog: a legal answer reached a sink
    assert isinstance(r.answer, MarkerBox)  # a legal type, never silence
    assert r.elapsed_sim_s < 570.0
    assert _floor_used(r.flight_log) is True  # floor_used True
