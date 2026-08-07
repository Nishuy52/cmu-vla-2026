"""Frame-arrival watchdog tests (issue #174) — pure Python, no ROS/rclpy needed.

Mirrors test_cloud_packing.py's pattern: ros_adapter.frame_watchdog imports nothing
ROS-specific, so it is directly testable on the Windows dev box with a mock clock (no
real sleeping). Behaviour matches the split-tree evidence on the #174 thread: a dead-at-kf-2
loft slot has the worker thread alive but the camera source below Python has stopped
delivering frames — nothing currently logs that. This module fixes the "nothing logs" half.
"""
from __future__ import annotations

from ros_adapter.frame_watchdog import (
    FRAME_QUIET_REPEAT_S,
    FRAME_QUIET_WARN_S,
    FrameWatchdog,
)


class MockClock:
    """A monotonic-seconds clock the test advances explicitly — no real sleeping."""

    def __init__(self, t: float = 0.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def test_quiet_feed_warns_once_then_rate_limits_repeats():
    clock = MockClock()
    wd = FrameWatchdog(clock=clock)

    # Below threshold: silent.
    clock.advance(FRAME_QUIET_WARN_S - 1.0)
    assert wd.check(question_active=True) is None

    # Cross the threshold: exactly one warning fires.
    clock.advance(2.0)  # total age now > FRAME_QUIET_WARN_S
    msg = wd.check(question_active=True)
    assert msg is not None
    assert "quiet" in msg
    assert "#174" in msg

    # Same tick cadence, still well inside the repeat window: no second warning.
    clock.advance(1.0)
    assert wd.check(question_active=True) is None
    clock.advance(FRAME_QUIET_REPEAT_S - 2.0)
    assert wd.check(question_active=True) is None

    # Repeat window elapsed: exactly one more warning, not a flood.
    clock.advance(2.0)
    msg2 = wd.check(question_active=True)
    assert msg2 is not None


def test_quiet_feed_before_question_active_does_not_warn():
    clock = MockClock()
    wd = FrameWatchdog(clock=clock)

    clock.advance(FRAME_QUIET_WARN_S * 3)
    # No question yet — the watchdog must stay silent regardless of quiet duration.
    assert wd.check(question_active=False) is None
    # Once the question goes active, the accumulated quiet duration still triggers.
    assert wd.check(question_active=True) is not None


def test_steady_frames_never_warn():
    clock = MockClock()
    wd = FrameWatchdog(clock=clock)

    # Simulate a healthy 10 Hz-ish feed for well past the warn threshold, with the
    # watchdog polled every tick like the adapter's 5 Hz drive timer.
    ticks = int((FRAME_QUIET_WARN_S * 5) / 0.2)
    for _ in range(ticks):
        clock.advance(0.2)
        wd.on_frame()
        assert wd.check(question_active=True) is None


def test_last_frame_age_s_baselines_at_construction_with_no_frames():
    clock = MockClock()
    wd = FrameWatchdog(clock=clock)
    assert wd.last_frame_age_s() == 0.0
    clock.advance(5.0)
    assert wd.last_frame_age_s() == 5.0


def test_on_frame_resets_age():
    clock = MockClock()
    wd = FrameWatchdog(clock=clock)
    clock.advance(FRAME_QUIET_WARN_S - 0.1)
    wd.on_frame()
    assert wd.last_frame_age_s() == 0.0
    assert wd.check(question_active=True) is None
