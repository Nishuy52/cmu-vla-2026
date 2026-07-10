"""Free-run / end-of-data semantics for ReplayClock and ReplayRobotIO.

Once the recorded schedule is exhausted, tick() must keep advancing time at a fixed dt
(the frozen-world free-run) so the FSM's budget/watchdog gates eventually fire, while the
latest_* getters keep returning the final recorded messages. Regression cover for the
"bag ends before the 510/570 s gates" defect.
"""
from __future__ import annotations

import pytest

from core.interfaces import IntAnswer, LidarScan, OdomState
import numpy as np

from core.replay.replay_io import (
    CH_ODOM,
    CH_SCAN,
    FREE_RUN_DT_S,
    MessageStore,
    ReplayClock,
    ReplayRobotIO,
)


def _store_30s() -> MessageStore:
    """A tiny store whose messages span t=1..30 s (one odom + one scan per second)."""
    store = MessageStore()
    for sec in range(1, 31):
        store.add(CH_ODOM, float(sec), OdomState(t=float(sec), x=float(sec), y=0.0, z=0.0, yaw=0.0))
        store.add(
            CH_SCAN,
            float(sec),
            LidarScan(t=float(sec), points=np.zeros((1, 3), dtype=np.float32)),
        )
    return store.finalize()


def test_clock_free_runs_past_schedule():
    clock = ReplayClock([1.0, 2.0, 3.0])
    assert clock.now() == pytest.approx(1.0)
    clock.step(); clock.step()  # -> 3.0, on the last scheduled timestamp
    assert clock.now() == pytest.approx(3.0)
    assert clock.exhausted
    assert not clock.end_of_data
    # Now free-run: fixed-dt advance, end_of_data flips True.
    t = clock.step()
    assert t == pytest.approx(3.0 + FREE_RUN_DT_S)
    assert clock.end_of_data
    clock.step()
    assert clock.now() == pytest.approx(3.0 + 2 * FREE_RUN_DT_S)


def test_free_run_reaches_watchdog_horizon():
    """Free-running from a 30 s bag must be able to cross the 570 s watchdog floor."""
    clock = ReplayClock([float(s) for s in range(1, 31)])
    while clock.now() < 30.0:
        clock.step()
    steps = 0
    while clock.now() < 570.0 and steps < 100_000:
        clock.step()
        steps += 1
    assert clock.now() >= 570.0
    assert clock.end_of_data


def test_custom_free_run_dt():
    clock = ReplayClock([1.0, 2.0], free_run_dt=5.0)
    clock.step()  # -> 2.0 (last scheduled)
    clock.step()  # free-run
    assert clock.now() == pytest.approx(7.0)


def test_empty_schedule_stays_put_and_not_end_of_data():
    clock = ReplayClock([])
    assert clock.step() == 0.0
    assert clock.exhausted
    assert not clock.end_of_data  # nothing was ever scheduled -> not "past" any data


def test_getters_frozen_after_end_of_data():
    io = ReplayRobotIO(_store_30s())
    clock = io.raw_clock()
    clock.set(30.0)
    assert io.latest_odom().x == pytest.approx(30.0)
    assert not io.end_of_data
    # Free-run well past the last message; getters keep returning the final (frozen) world.
    for _ in range(50):
        io.tick()
    assert io.end_of_data
    assert io.clock().now() > 30.0
    assert io.latest_odom().x == pytest.approx(30.0)  # still the last recorded odom
    assert io.latest_scan() is not None


def test_end_of_data_property_on_io():
    io = ReplayRobotIO(_store_30s())
    io.raw_clock().set(io.store.all_times()[0])
    assert not io.end_of_data
    for _ in range(200):
        io.tick()
    assert io.end_of_data


def test_budget_clock_override_leaves_getters_on_raw_time():
    io = ReplayRobotIO(_store_30s())
    io.raw_clock().set(10.0)

    class _Double:
        def __init__(self, inner):
            self._inner = inner

        def now(self):
            return self._inner.now() * 2.0

    io.set_budget_clock(_Double(io.raw_clock()))
    assert io.clock().now() == pytest.approx(20.0)  # FSM-facing clock is scaled
    assert io.latest_odom().x == pytest.approx(10.0)  # getters still read raw time
    io.set_budget_clock(None)
    assert io.clock().now() == pytest.approx(10.0)
