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
from core.perception import tiling as T
from core.perception.async_pipeline import AsyncPerceptionWorker
from core.perception.detector import Detection, FakeDetector
from core.perception.scene_index import BasicSceneIndex
from core.perception.tracker import KeyframeConfig, PerceptionPipeline

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


# --------------------------------------------------------------------- liveness (#174)


class _RecordingPipeline:
    """Records every processed frame's pano.t; never raises on its own."""

    def __init__(self) -> None:
        self.calls: list[float] = []
        self._lock = threading.Lock()

    def process(self, pano: PanoFrame, scan: LidarScan) -> list[int]:
        with self._lock:
            self.calls.append(pano.t)
        return []


def test_worker_survives_a_long_boot_gap_between_the_first_two_frames():
    """Issue #174: a slow SECOND keyframe at boot must not end the worker.

    Reproduces the measured discriminator from the loft investigation: a dead
    slot got keyframes 0 and 1 arriving 7.1s apart, then keyframe flow stopped
    forever; a surviving sibling slot got them 0.5s apart and kept going. The
    frame-wait loop (``Condition.wait()`` with no timeout) has no fixed bound to
    exceed, so this is a regression guard: an arbitrary gap between submissions
    must never leave frames submitted-but-never-processed.
    """
    pipeline = _RecordingPipeline()
    worker = AsyncPerceptionWorker(pipeline)
    try:
        worker.submit(_pano(0.0), _scan(0.0))
        time.sleep(0.1)  # let the worker pick up and finish frame 0
        # A long inter-keyframe gap at boot (bigger than the measured 7.1s death
        # gap would be, scaled down so the test stays fast) — during this whole
        # window the worker has nothing pending and is parked on cv.wait().
        time.sleep(1.5)
        # More frames now arrive, exactly like the surviving loft slot's later
        # 5-10s cadence resuming after the gap.
        worker.submit(_pano(1.0, x=1.0), _scan(1.0))
        time.sleep(0.1)
        worker.submit(_pano(2.0, x=2.0), _scan(2.0))
        time.sleep(0.1)
        worker.submit(_pano(3.0, x=3.0), _scan(3.0))
        time.sleep(0.2)
    finally:
        worker.stop(timeout=3.0)
    assert pipeline.calls == [0.0, 1.0, 2.0, 3.0], (
        "keyframe flow stopped after the boot gap instead of resuming: %r" % pipeline.calls
    )
    assert worker.is_alive is False  # cleanly stopped, not crashed
    assert worker.restart_count == 0  # no fault occurred — nothing to restart


def test_worker_crash_triggers_bounded_restart_and_counter_stays_readable(monkeypatch):
    """Issue #174: if the worker thread ends for ANY reason, ``submit`` detects it,
    logs loudly, and restarts it (bounded), and ``keyframes_processed`` never goes
    unreadable in the meantime.

    ``_run`` is monkeypatched at the class level so the very first spawned thread
    simulates a fatal, unguarded fault that ends the thread outright (standing in
    for whatever unforeseen fault the per-frame try/except doesn't cover — the
    restart supervisor is the backstop for those, not just for ordinary
    ``pipeline.process()`` exceptions, which never kill the thread at all). Every
    later (re)spawn runs the real loop.
    """
    pipeline = _RecordingPipeline()
    calls = {"n": 0}
    real_run = AsyncPerceptionWorker._run

    def flaky_then_real_run(self: AsyncPerceptionWorker) -> None:
        calls["n"] += 1
        if calls["n"] == 1:
            with self._cv:
                self._last_error = RuntimeError("simulated fatal worker fault")
            return  # thread ends immediately, as if it had crashed
        real_run(self)

    monkeypatch.setattr(AsyncPerceptionWorker, "_run", flaky_then_real_run)

    worker = AsyncPerceptionWorker(pipeline)
    try:
        # keyframes_processed must be a readable int even while the very first
        # (crashed) worker thread is dead and no restart has happened yet.
        assert worker.keyframes_processed == 0
        assert isinstance(worker.keyframes_processed, int)

        deadline = time.monotonic() + 3.0
        while worker.is_alive and time.monotonic() < deadline:
            time.sleep(0.01)
        assert not worker.is_alive, "simulated crash did not end the first thread"

        # submit() must notice the dead thread and restart it.
        worker.submit(_pano(0.0), _scan(0.0))
        time.sleep(0.3)
        assert worker.restart_count == 1
        assert worker.is_alive

        # keyframes_processed stays readable throughout — never None.
        assert isinstance(worker.keyframes_processed, int)

        worker.submit(_pano(1.0, x=1.0), _scan(1.0))
        time.sleep(0.2)
    finally:
        worker.stop(timeout=3.0)

    assert pipeline.calls == [0.0, 1.0], (
        "restarted worker did not resume processing frames: %r" % pipeline.calls
    )
    assert worker.restart_count == 1


# --------------------------------------------------------------------- issue #211 counters


def test_dropped_count_tracks_overwritten_pending_frames():
    """dropped_count counts frames overwritten in the pending slot before the worker
    ever started them -- the keep-latest policy's cost, made countable."""
    pipeline = _SlowFakePipeline(forward_s=0.3)
    worker = AsyncPerceptionWorker(pipeline)
    try:
        worker.submit(_pano(0.0), _scan(0.0))
        time.sleep(0.02)  # worker picks frame 0 up and is mid-forward
        assert worker.dropped_count == 0
        worker.submit(_pano(1.0), _scan(1.0))  # overwrites nothing pending yet -> queued
        worker.submit(_pano(2.0), _scan(2.0))  # overwrites frame 1 -> dropped
        worker.submit(_pano(3.0), _scan(3.0))  # overwrites frame 2 -> dropped
        time.sleep(0.5)
    finally:
        worker.stop()
    assert pipeline.calls == [0.0, 3.0]
    assert worker.dropped_count == 2
    # Every submitted frame is accounted for: processed, dropped, or (none, here,
    # since the worker drained to idle) still pending/in flight.
    assert worker.submitted_count == worker.processed_count + worker.dropped_count


class _BlockingPipeline:
    """process() blocks on an Event instead of sleeping — lets a test hold a forward
    open indefinitely without actually burning wall time."""

    def __init__(self) -> None:
        self.release = threading.Event()
        self.started = threading.Event()

    def process(self, pano: PanoFrame, scan: LidarScan) -> list[int]:
        self.started.set()
        self.release.wait()
        return []


def test_in_flight_age_s_grows_during_a_slow_forward_and_clears_when_idle():
    """in_flight_age_s is the direct backlog signal: it must be None while idle, grow
    monotonically while the worker is mid-forward (NOT reset by fresh submissions —
    those only touch the pending slot), and clear back to None once the forward
    completes."""
    clock = {"t": 0.0}
    pipeline = _BlockingPipeline()
    worker = AsyncPerceptionWorker(pipeline, clock=lambda: clock["t"])
    try:
        assert worker.in_flight_age_s is None  # idle: nothing submitted yet
        worker.submit(_pano(0.0), _scan(0.0))
        assert pipeline.started.wait(timeout=2.0), "worker never entered in-flight state"
        assert worker.in_flight_age_s is not None

        clock["t"] = 5.0
        assert worker.in_flight_age_s == pytest.approx(5.0)
        # A fresh submission (a new tick's frame arriving) must NOT reset the in-flight
        # clock -- the worker is still busy with the OLD forward; the new frame just
        # waits in the pending slot.
        worker.submit(_pano(1.0), _scan(1.0))
        clock["t"] = 9.0
        assert worker.in_flight_age_s == pytest.approx(9.0)
    finally:
        pipeline.release.set()
        worker.stop(timeout=2.0)


def test_in_flight_age_s_clears_once_the_forward_completes():
    pipeline = _SlowFakePipeline(forward_s=0.1)
    worker = AsyncPerceptionWorker(pipeline)
    try:
        worker.submit(_pano(0.0), _scan(0.0))
        time.sleep(0.02)
        assert worker.in_flight_age_s is not None
        time.sleep(0.3)  # forward_s=0.1 has long since completed
        assert worker.in_flight_age_s is None
    finally:
        worker.stop()


# --------------------------------------------------------------------- issue #211 non-blocking


def test_submit_stays_fast_even_under_a_deep_multi_frame_backlog():
    """Issue #211: whatever ELSE the caller's tick does after submit() (occupancy-grid
    ingestion, waypoint publication — neither of which lives in this module, both live
    in the qtype heads reading io.latest_terrain()/latest_scan() directly) must never
    wait on the detector. Proven here at the seam this module owns: submit() itself
    stays fast across many consecutive calls even while a single forward takes far
    longer than a realistic 780 s question budget could ever wait for one keyframe."""
    pipeline = _BlockingPipeline()  # never returns until released — models a stuck detector
    worker = AsyncPerceptionWorker(pipeline)
    try:
        worker.submit(_pano(0.0), _scan(0.0))
        assert pipeline.started.wait(timeout=2.0)
        for i in range(1, 51):  # 51 ticks at 5 Hz worth of submissions (~10s of tick time)
            t0 = time.monotonic()
            worker.submit(_pano(float(i)), _scan(float(i)))
            elapsed = time.monotonic() - t0
            assert elapsed < 0.05, (
                "submit() #%d blocked for %.3fs behind a stuck detector" % (i, elapsed)
            )
        assert worker.dropped_count == 49  # frames 1..49 overwritten; 50 still pending
    finally:
        pipeline.release.set()
        worker.stop(timeout=2.0)


def test_fast_detector_keeps_up_with_no_material_drop():
    """Issue #211 validation (d): with a FAST detector (faster than tick period),
    end-to-end behaviour is unchanged from today — the worker keeps pace, dropped_count
    stays at (or very near) zero, and processed_count tracks submitted_count closely."""
    pipeline = _SlowFakePipeline(forward_s=0.02)  # far faster than the 0.2s tick period
    worker = AsyncPerceptionWorker(pipeline)
    try:
        for i in range(20):
            worker.submit(_pano(float(i)), _scan(float(i)))
            time.sleep(TICK_PERIOD_S)
        time.sleep(0.2)  # drain the last frame
    finally:
        worker.stop(timeout=2.0)
    assert worker.submitted_count == 20
    assert worker.processed_count >= 18  # keeps up; at most a frame or two in flight/queued
    assert worker.dropped_count <= 1


class _DelayedDetectorFn:
    """Wraps a real ``FakeDetector`` with a configurable per-call ``time.sleep`` —
    stands in for a slow GDINO forward while still exercising the REAL
    ``PerceptionPipeline`` (tiling -> detect -> fuse -> associate -> index), unlike
    ``_SlowFakePipeline`` above (which fakes the whole pipeline)."""

    def __init__(self, detector: FakeDetector, delay_s: float) -> None:
        self._detector = detector
        self.delay_s = delay_s

    def __call__(self, tiles):
        time.sleep(self.delay_s)
        return self._detector(tiles)


def _pano_with_cluster(t: float) -> tuple[PanoFrame, LidarScan]:
    rng = np.random.default_rng(0)
    cloud = np.column_stack(
        [
            3.0 + rng.uniform(-0.2, 0.2, 40),
            0.0 + rng.uniform(-0.2, 0.2, 40),
            0.5 + rng.uniform(-0.2, 0.2, 40),
        ]
    ).astype(np.float32)
    pano = PanoFrame(
        t=t, image=np.zeros((T.PANO_HEIGHT, T.PANO_WIDTH, 3), dtype=np.uint8),
        odom=OdomState(t=t, x=0.0, y=0.0, z=0.0, yaw=0.0),
    )
    return pano, LidarScan(t=t, points=cloud)


def test_detections_still_integrate_once_a_backlogged_detector_catches_up():
    """Issue #211 validation (b): a lagging detector must not lose its work forever —
    once it finally catches up on a (dropped-down-to) latest frame, that frame's
    detections land in the SAME live scene index the heads already read from."""
    spec = T.tile_specs()[0]
    det = Detection(
        tile_id=0,
        bbox_xyxy=(spec.cx - 60, spec.cy - 120, spec.cx + 60, spec.cy + 60),
        label="sofa", score=0.9,
    )
    fake = FakeDetector([det])
    delayed = _DelayedDetectorFn(fake, delay_s=0.3)
    sc = BasicSceneIndex([])
    pipeline = PerceptionPipeline(delayed, index=sc, keyframe_cfg=KeyframeConfig(every_k=1))
    worker = AsyncPerceptionWorker(pipeline)
    try:
        # Flood the worker with frames far faster than it can process (simulates the
        # tick thread submitting every 200ms against a 300ms+ forward) -- most get
        # dropped (keep-latest), but the worker is never blocked doing this.
        for i in range(10):
            pano, scan = _pano_with_cluster(float(i))
            worker.submit(pano, scan)
            time.sleep(0.05)
        # Let the worker actually finish the (few) frames it managed to start.
        time.sleep(1.0)
    finally:
        worker.stop(timeout=2.0)
    assert sc.all_instances(), "the detector's completed work never reached the index"
    assert any(inst.label == "sofa" for inst in sc.all_instances())
    assert worker.dropped_count > 0  # the backlog was real, not incidental
    assert worker.processed_count >= 1


def test_worker_restart_is_bounded():
    """Issue #174: restarts stop after ``max_restarts`` — the worker does not spin
    forever, and ``keyframes_processed`` remains a readable int even once the
    restart budget is exhausted and the thread is left dead."""
    pipeline = _RecordingPipeline()

    def always_dies(self: AsyncPerceptionWorker) -> None:
        return  # every spawn "crashes" immediately

    worker = AsyncPerceptionWorker(pipeline, max_restarts=2)
    worker._run = always_dies.__get__(worker, AsyncPerceptionWorker)  # type: ignore[method-assign]
    # Re-spawn using the patched _run (the constructor already started a thread
    # bound to the real _run before we could patch the instance).
    worker._thread = worker._spawn_thread()

    deadline = time.monotonic() + 2.0
    while worker.is_alive and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not worker.is_alive

    for _ in range(5):  # far more submits than the restart budget allows
        worker.submit(_pano(0.0), _scan(0.0))
        deadline = time.monotonic() + 1.0
        while worker.is_alive and time.monotonic() < deadline:
            time.sleep(0.01)

    assert worker.restart_count == 2, "restart count exceeded max_restarts=2"
    assert isinstance(worker.keyframes_processed, int)
