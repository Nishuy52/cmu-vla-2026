"""Camera frame-arrival watchdog (issue #174) — pure Python, no ROS imports.

Split-tree evidence (issue #174, live sweeps 8 Aug 2026): the async-worker guards fixed
the Python-side deaths (a dead perception worker now logs a restart line), but one loft
death mode remains — the camera frame SOURCE goes quiet (Unity/camera stops publishing
keyframes) while the worker thread stays alive. Nothing below the worker ever logs, so
these runs die silently on the cluster and each one needs fresh forensics to diagnose.

This module is pure observability: it tracks the wall-time of the last received camera
frame and tells the caller (once, rate-limited) when the feed has gone quiet too long
while a question is active. It does NOT attempt recovery or resubscription — that needs
design evidence the #174 thread does not have yet.

Like ``ros_adapter.cloud_packing``, this file imports nothing ROS-specific, so it is
directly unit-testable on the Windows dev box with a mock clock (no real sleeping, no
rclpy). The adapter node (``ros_adapter.adapter_node``) is the only caller: it feeds
``on_frame()`` from the camera callback and ``check()`` from the 5 Hz drive timer.
"""
from __future__ import annotations

from typing import Callable

#: Default quiet-feed threshold (seconds) before the first loud warning fires.
FRAME_QUIET_WARN_S = 15.0

#: Minimum gap (seconds) between repeated warnings for the same ongoing quiet spell.
FRAME_QUIET_REPEAT_S = 30.0


class FrameWatchdog:
    """Tracks camera-frame arrival and rate-limits a "feed is quiet" warning.

    ``clock`` is a zero-arg callable returning monotonic seconds (``time.monotonic`` in
    production, a mock clock in tests) — the same injection pattern the rest of the
    adapter's Windows-testable helpers use.

    The watchdog baselines ``last_frame_age_s()`` at construction time, so a slot that
    never receives a single frame still ages out and warns (a camera that never starts
    is the same failure as one that stops).
    """

    def __init__(
        self,
        clock: Callable[[], float],
        warn_after_s: float = FRAME_QUIET_WARN_S,
        repeat_after_s: float = FRAME_QUIET_REPEAT_S,
    ) -> None:
        self._clock = clock
        self._warn_after_s = warn_after_s
        self._repeat_after_s = repeat_after_s
        self._last_frame_t: float = clock()
        self._last_warn_t: float | None = None

    def on_frame(self) -> None:
        """Call on every received camera frame (the adapter's image callback)."""
        self._last_frame_t = self._clock()

    def last_frame_age_s(self) -> float:
        """Seconds since the last received frame (since construction if there was none)."""
        return self._clock() - self._last_frame_t

    def check(self, question_active: bool) -> str | None:
        """Call from the periodic drive timer.

        Returns a ready-to-log warning message exactly when one should fire (the feed has
        been quiet for at least ``warn_after_s`` while a question is active, and either no
        warning has fired yet or the last one is older than ``repeat_after_s``). Returns
        None otherwise. Never logs itself — the caller owns the logger, keeping this class
        trivially testable without a fake logger.
        """
        if not question_active:
            return None
        age = self.last_frame_age_s()
        if age < self._warn_after_s:
            return None
        now = self._clock()
        if self._last_warn_t is not None and (now - self._last_warn_t) < self._repeat_after_s:
            return None
        self._last_warn_t = now
        return (
            "camera frame feed quiet for %.1fs (no /camera/image received) — "
            "worker alive, source below Python has stalled (issue #174)" % age
        )
