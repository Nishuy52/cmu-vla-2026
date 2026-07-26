"""Off-thread dispatch for :meth:`PerceptionPipeline.process` (issue #88).

On a live cluster run, GroundingDINO forwards take ~1s. Feeding
:class:`~core.perception.tracker.PerceptionPipeline` synchronously from the rclpy
tick/callback chain (the pre-#88 behaviour) blocks waypoint publication and
subscription servicing for that entire second: the adapter's 5 Hz drive timer
collapses to roughly one tick per detector forward (~0.8 Hz measured on a 210 s
explore run, vs ~5 Hz with the detector off). :class:`AsyncPerceptionWorker` moves
the ``pipeline.process()`` call to a dedicated worker thread so the caller's
``submit()`` never blocks on a forward pass.

Latest-frame-wins: ``submit()`` overwrites any not-yet-picked-up pending frame
rather than queueing, so a slow forward drops stale frames instead of building a
backlog — the caller only ever needs the newest frame processed, not every frame
in order (a queued backlog of ~1s forwards would just re-introduce the same
cadence starvation one frame later).

Thread safety of the hand-off: ``PerceptionPipeline.process()`` mutates the shared
:class:`~core.perception.scene_index.BasicSceneIndex` (``pipeline.index``) on the
worker thread while the caller's own thread (heads resolving an answer) concurrently
reads that same index. ``BasicSceneIndex`` guards its own state with an internal
``RLock`` (see its docstring), so each individual index call from either thread is
atomic; this module only needs to serialize frame submission into the pipeline
itself (one writer thread, one pipeline instance), not index access.
"""
from __future__ import annotations

import threading
from typing import Callable

from core.interfaces import LidarScan, PanoFrame
from core.perception.tracker import PerceptionPipeline


class AsyncPerceptionWorker:
    """Runs one :class:`PerceptionPipeline` on a dedicated thread, latest-frame-wins.

    The caller's only entry point is :meth:`submit`; it never blocks on a forward
    pass — it latches the frame and returns immediately, replacing any not-yet-
    processed pending frame. A single background thread wakes on each submission,
    picks up the latest frame, and runs it through ``pipeline.process()``.
    """

    def __init__(
        self,
        pipeline: PerceptionPipeline,
        on_error: Callable[[Exception], None] | None = None,
    ) -> None:
        self._pipeline = pipeline
        self._on_error = on_error
        self._cv = threading.Condition()
        self._pending: tuple[PanoFrame, LidarScan] | None = None
        self._stopped = False
        self._processed_count = 0
        self._submitted_count = 0
        self._thread = threading.Thread(
            target=self._run, name="vla-perception-worker", daemon=True
        )
        self._thread.start()

    def submit(self, pano: PanoFrame, scan: LidarScan) -> None:
        """Non-blocking: latch ``(pano, scan)`` as the next frame to process.

        Overwrites any frame the worker has not yet picked up (drop-stale-frames
        policy, issue #88). No-op after :meth:`stop`.
        """
        with self._cv:
            if self._stopped:
                return
            self._pending = (pano, scan)
            self._submitted_count += 1
            self._cv.notify()

    def stop(self, timeout: float = 2.0) -> None:
        """Signal the worker to exit and join it. Safe to call more than once."""
        with self._cv:
            self._stopped = True
            self._cv.notify()
        self._thread.join(timeout=timeout)

    @property
    def processed_count(self) -> int:
        """Frames actually run through ``pipeline.process()`` so far (test/debug)."""
        with self._cv:
            return self._processed_count

    @property
    def submitted_count(self) -> int:
        """Frames handed to :meth:`submit` so far, including any dropped as stale."""
        with self._cv:
            return self._submitted_count

    @property
    def is_alive(self) -> bool:
        return self._thread.is_alive()

    def _run(self) -> None:
        while True:
            with self._cv:
                while self._pending is None and not self._stopped:
                    self._cv.wait()
                if self._pending is None and self._stopped:
                    return
                pano, scan = self._pending
                self._pending = None
            try:
                self._pipeline.process(pano, scan)
            except Exception as exc:  # a perception glitch must never kill the worker
                if self._on_error is not None:
                    self._on_error(exc)
            with self._cv:
                self._processed_count += 1
