"""run_question happy-path tests: one per qtype on a synthetic scene.

Each asserts the pipeline publishes a legal answer of the right type, in plausible
*simulated* time, and completes far faster in wall time than its simulated budget (the
clock is simulated, so a full 600 s sim budget must still run in a few wall-seconds).

The wall-time guard exists to catch a "sim runs in real wall time" regression, not to
pin a tight latency SLA: at 20x below the 600 s sim budget it still fails loudly if the
clock ever stops being simulated, while tolerating the CPU contention of the full
``-n auto`` gate (14 workers on 14 cores, numpy threadpools oversubscribing) — under
which a legitimately ~3.5 s single-threaded drive can stretch past a 5 s bound. See
``_WALL_BUDGET_S`` below.
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

# Wall-time guard for the run_question-driving cases. The simulated clock means a full
# 600 s sim budget costs only a few wall-seconds; this bound catches a regression that
# lets the sim run in real wall time (which would show as ~600 s), while staying loose
# enough to survive the full `-n auto` gate's CPU contention. On an idle machine these
# drives take ~3–4 s single-threaded; under 14 oversubscribed workers that can climb
# past 5 s, so the bound is set at 20x below the sim budget rather than a tight 5 s SLA.
_WALL_BUDGET_S = 30.0


def _io_for(question: str, bucket: str) -> MockRobotIO:
    sc, _ = build_scene_for("t", {bucket: [question]})
    return MockRobotIO(sc, FakeClock(0.0))


def test_numerical_happy_path_publishes_int_before_watchdog():
    q = "How many chairs are near the table?"
    io = _io_for(q, "numerical")
    t0 = time.monotonic()
    r = run_question(q, io, tick_hz=_HZ)
    wall = time.monotonic() - t0

    assert r.published is True
    assert isinstance(r.answer, IntAnswer)
    assert r.qtype in QType  # a real qtype was resolved
    assert 0.0 < r.elapsed_sim_s < 570.0  # answered before the watchdog floor
    assert wall < _WALL_BUDGET_S


@pytest.mark.slow
def test_object_reference_happy_path_publishes_marker():
    q = "Find the vase on the table"
    io = _io_for(q, "object_reference")
    t0 = time.monotonic()
    r = run_question(q, io, tick_hz=_HZ)
    wall = time.monotonic() - t0

    assert r.published is True
    assert isinstance(r.answer, MarkerBox)
    assert 0.0 < r.elapsed_sim_s <= 570.0
    assert wall < _WALL_BUDGET_S


@pytest.mark.slow
def test_instruction_following_happy_path_publishes_waypoint():
    q = "Go to the table and stop at the chair"
    io = _io_for(q, "instruction_following")
    t0 = time.monotonic()
    r = run_question(q, io, tick_hz=_HZ)
    wall = time.monotonic() - t0

    assert r.published is True
    assert isinstance(r.answer, WaypointCmd)
    assert 0.0 < r.elapsed_sim_s <= 570.0
    assert wall < _WALL_BUDGET_S


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
    """Regression guard for the class of bug the per-test ``wall < _WALL_BUDGET_S`` bound
    protects against: the clock silently ceasing to be *simulated* and the drive burning
    real wall time proportional to sim time (which would show as ~hundreds of wall-seconds
    for a run that advances ~hundreds of sim-seconds).

    Unlike an absolute wall bound this asserts the *ratio* ``elapsed_sim_s / wall_s`` is
    large, so it stays true under the ``-n auto`` gate's CPU contention (contention inflates
    wall by a few seconds; the invariant has a ~50x margin) yet fails loudly the moment the
    sim starts running in real time. This is why the absolute per-test bound is intentionally
    loose (see module docstring) — this test carries the tight, contention-robust check.
    """
    q = "Go to the table and stop at the chair"
    r = run_question(q, _io_for(q, "instruction_following"), tick_hz=_HZ)

    assert r.elapsed_sim_s > 100.0  # the drive really advanced the simulated budget
    assert r.wall_s > 0.0
    # A simulated clock makes wall time a tiny fraction of sim time. Even under heavy
    # parallel-gate contention wall stays single-digit seconds vs >100 s of sim time, so a
    # >=10x ratio has ample headroom while still catching a real-time-clock regression.
    assert r.elapsed_sim_s / r.wall_s > 10.0


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
