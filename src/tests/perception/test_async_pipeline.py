"""AsyncPerceptionWorker (issue #88): off-thread perception dispatch must not stall
the tick cadence, and the tracker/index hand-off must be race-free.

Context: on a live cluster run, GroundingDINO forwards take ~1s. Calling
PerceptionPipeline.process() inline from the rclpy 5 Hz tick timer blocked waypoint
publication + subscription servicing for that entire second (~176 waypoints per
210 s explore, ~0.8 Hz effective, vs ~1050 / 5 Hz with the detector off). These
tests simulate the executor's tick loop directly (no rclpy needed — the fast tier
runs on Windows too) with a fake ~0.5s detector forward standing in for GDINO.
"""
from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from core.interfaces import InstanceRecord, LidarScan, OdomState, PanoFrame
from core.perception.async_pipeline import AsyncPerceptionWorker
from core.perception.scene_index import BasicSceneIndex

TICK_HZ = 5.0
TICK_PERIOD_S = 1.0 / TICK_HZ


def _pano(t: float, x: float = 0.0) -> PanoFrame:
    # Content is irrelevant to these tests (the fake pipeline below never reads it) —
    # only the odom/t identity matters, so keep the image tiny.
    return PanoFrame(t=t, image=np.zeros((2, 2, 3), dtype=np.uint8), odom=OdomState(
        t=t, x=x, y=0.0, z=0.0, yaw=0.0,
    ))


def _scan(t: float) -> LidarScan:
    return LidarScan(t=t, points=np.zeros((1, 3), dtype=np.float32))


class _SlowFakePipeline:
    """Stands in for PerceptionPipeline with a detector whose forward takes ~0.5s.

    Records every ``process()`` call so tests can assert how many frames were
    actually run vs. how many ticks were submitted (latest-frame-wins drops the
    difference).
    """

    def __init__(self, forward_s: float = 0.5) -> None:
        self.forward_s = forward_s
        self.calls: list[float] = []  # pano.t of each processed frame
        self._lock = threading.Lock()

    def process(self, pano: PanoFrame, scan: LidarScan) -> list[int]:
        time.sleep(self.forward_s)
        with self._lock:
            self.calls.append(pano.t)
        return []


# --------------------------------------------------------------------- cadence


def test_submit_never_blocks_on_a_slow_forward():
    """submit() must return ~instantly even while the worker is mid-forward."""
    pipeline = _SlowFakePipeline(forward_s=0.5)
    worker = AsyncPerceptionWorker(pipeline)
    try:
        worker.submit(_pano(0.0), _scan(0.0))
        time.sleep(0.05)  # let the worker pick it up and start its 0.5s sleep
        t0 = time.monotonic()
        worker.submit(_pano(1.0), _scan(1.0))  # worker is busy — must not block
        elapsed = time.monotonic() - t0
        assert elapsed < 0.1, "submit() blocked on the in-flight forward (elapsed=%.3f)" % elapsed
    finally:
        worker.stop()


def test_tick_cadence_survives_a_slow_detector_forward():
    """Simulate the 5 Hz executor tick loop: cadence must hold despite ~0.5s forwards.

    This is the issue #88 regression check: ticks (submit calls) must land at the
    node's real cadence — the slow forward must show up only as dropped/stale
    frames on the perception side, never as a stalled tick loop.
    """
    pipeline = _SlowFakePipeline(forward_s=0.5)
    worker = AsyncPerceptionWorker(pipeline)
    tick_times: list[float] = []
    duration_s = 1.5
    try:
        start = time.monotonic()
        next_tick = start
        while time.monotonic() - start < duration_s:
            now = time.monotonic()
            tick_times.append(now)
            t = now - start
            worker.submit(_pano(t, x=t), _scan(t))  # x grows -> always a fresh keyframe id
            next_tick += TICK_PERIOD_S
            sleep_for = next_tick - time.monotonic()
            if sleep_for > 0:
                time.sleep(sleep_for)
        total_elapsed = time.monotonic() - start
    finally:
        worker.stop(timeout=3.0)

    expected_ticks = duration_s * TICK_HZ
    # Allow generous slack for scheduling jitter but this must be nowhere near the
    # ~0.8 Hz collapse the issue reports (that would cap ticks at ~duration/0.5).
    assert len(tick_times) >= expected_ticks * 0.6, (
        "tick cadence collapsed: got %d ticks in %.2fs (expected ~%.0f at %.1f Hz)"
        % (len(tick_times), total_elapsed, expected_ticks, TICK_HZ)
    )
    # The forward is slower than the tick period, so the worker cannot possibly have
    # kept up with every submitted frame — latest-frame-wins means processed_count
    # stays well below submitted_count (the pre-#88 synchronous bug is exactly the
    # inverse: ticks collapse to match the forward rate instead).
    assert worker.processed_count < worker.submitted_count
    assert worker.processed_count >= 1


def test_synchronous_baseline_reproduces_the_stall_for_contrast():
    """Sanity check for the test itself: calling process() inline (the pre-#88
    behaviour) DOES collapse cadence to roughly one tick per forward — confirming
    the harness would have caught the regression this fix addresses."""
    pipeline = _SlowFakePipeline(forward_s=0.2)
    tick_times: list[float] = []
    duration_s = 0.9
    start = time.monotonic()
    while time.monotonic() - start < duration_s:
        tick_times.append(time.monotonic())
        pipeline.process(_pano(time.monotonic() - start), _scan(0.0))  # inline, blocking
    # ~0.2s per forward over ~0.9s -> ~4-5 ticks, nowhere near 5 Hz * 0.9s = 4.5 expected
    # at full cadence but starved because the "tick" IS the forward here.
    assert len(pipeline.calls) == len(tick_times)
    assert len(tick_times) <= (duration_s / pipeline.forward_s) + 1


# --------------------------------------------------------------------- latest-frame-wins


def test_latest_frame_wins_drops_stale_pending_frame():
    pipeline = _SlowFakePipeline(forward_s=0.3)
    worker = AsyncPerceptionWorker(pipeline)
    try:
        worker.submit(_pano(0.0), _scan(0.0))
        time.sleep(0.02)  # ensure the worker has picked frame 0 up and is mid-forward
        # These should overwrite each other in the pending slot; only the last
        # (t=3.0) should ever be seen by the worker after frame 0 finishes.
        worker.submit(_pano(1.0), _scan(1.0))
        worker.submit(_pano(2.0), _scan(2.0))
        worker.submit(_pano(3.0), _scan(3.0))
        time.sleep(0.5)
    finally:
        worker.stop()
    assert pipeline.calls == [0.0, 3.0]


# --------------------------------------------------------------------- hand-off race


def test_concurrent_submit_and_stop_do_not_raise():
    """Hand-off race: hammer submit() from multiple threads while the worker drains
    and while stop() races in — must never raise or deadlock."""
    pipeline = _SlowFakePipeline(forward_s=0.01)
    worker = AsyncPerceptionWorker(pipeline)
    errors: list[Exception] = []

    def hammer(base: float) -> None:
        try:
            for i in range(200):
                worker.submit(_pano(base + i), _scan(base + i))
        except Exception as exc:  # pragma: no cover - failure path
            errors.append(exc)

    threads = [threading.Thread(target=hammer, args=(k * 1000.0,)) for k in range(4)]
    for th in threads:
        th.start()
    time.sleep(0.05)
    worker.stop(timeout=3.0)
    for th in threads:
        th.join(timeout=3.0)
        assert not th.is_alive()
    assert errors == []


def test_scene_index_concurrent_read_write_does_not_raise():
    """The other half of the hand-off: BasicSceneIndex must tolerate a writer thread
    (the perception worker, add/remove) racing readers (heads, by_label/all_instances)
    without raising 'list changed size during iteration'."""
    idx = BasicSceneIndex()
    stop = threading.Event()
    errors: list[Exception] = []

    def _rec(iid: int) -> InstanceRecord:
        c = np.array([float(iid % 7), 0.0, 0.0])
        return InstanceRecord(
            instance_id=iid, label="vase", score=0.9, n_obs=1,
            centroid=c, aabb_min=c - 0.05, aabb_max=c + 0.05,
        )

    def writer() -> None:
        try:
            iid = 0
            while not stop.is_set():
                idx.add(_rec(iid))
                if iid % 3 == 0:
                    idx.remove(iid - 1)
                iid += 1
        except Exception as exc:  # pragma: no cover - failure path
            errors.append(exc)

    def reader() -> None:
        try:
            while not stop.is_set():
                idx.all_instances()
                idx.by_label("vase")
        except Exception as exc:  # pragma: no cover - failure path
            errors.append(exc)

    threads = [
        threading.Thread(target=writer),
        threading.Thread(target=reader),
        threading.Thread(target=reader),
    ]
    for th in threads:
        th.start()
    time.sleep(0.3)
    stop.set()
    for th in threads:
        th.join(timeout=3.0)
        assert not th.is_alive()
    assert errors == []
