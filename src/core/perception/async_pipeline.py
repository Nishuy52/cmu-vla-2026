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

Worker liveness (issue #174): the frame-wait loop (:meth:`_run`) blocks on
``Condition.wait()`` with NO timeout, so an arbitrarily long gap between two
submitted frames at boot (measured on a live loft run: keyframe 0 -> 1 arriving
7.1s apart on a slot whose keyframe flow then stopped forever, vs 0.5s apart on a
sibling slot that survived) never by itself ends the loop or the thread — the wait
simply keeps blocking until the next :meth:`submit`. The mechanism that DID kill a
worker was narrower: ``_run`` only guarded ``pipeline.process()`` with
``except Exception``; the ``on_error`` callback invocation lived *inside* that
except block but outside any guard of its own. On a live cluster run ``on_error``
calls into ``rclpy``'s logger, which can itself raise (the adapter logs show a
"benign teardown RCLError" alongside every dead loft slot); that second exception
then escaped ``_run`` entirely, silently ending the worker thread with no
traceback anywhere reachable in the captured log, and nothing detected the death
or ever picked up another frame — matching the observed signature exactly
(keyframe flow processes 1-4 frames then stops forever; ``submit()`` keeps being
called every tick but nobody is listening). Two fixes close this: (1) every path
inside :meth:`_run`, including the ``on_error`` callback itself, is now guarded —
nothing can escape and end the thread from a single frame's fault; (2)
:meth:`submit` is now a supervisor as well as a producer — it checks worker
liveness on every call and restarts a dead thread (bounded, see
``max_restarts``), logging loudly with the triggering exception, instead of
silently dropping every future frame.

Issue #211 "map ingestion must never wait on detection": the caller (the adapter's
5 Hz tick, see ``ros_adapter.adapter_node._maybe_process_perception``) only ever
calls :meth:`submit`, never anything that blocks on ``pipeline.process()`` -- so
whatever ELSE that same tick does (waypoint publication, occupancy-grid ingestion
from ``/terrain_map``, which lives entirely in the qtype heads and never touches
this module or ``PerceptionPipeline``) is unaffected by how slow the detector is.
This was already true structurally before #211 (traced and confirmed: occupancy
ingestion is not, and was never, gated by this worker's completion); #211 adds the
missing OBSERVABILITY half of that guarantee -- :attr:`dropped_count` and
:attr:`in_flight_age_s` make the keep-latest / bounded-lag policy legible instead of
only inferable, and the counter distinction below resolves #211's own "keyframes_
processed disagrees with raw_detections record counts" question: they were never
supposed to agree -- :attr:`keyframes_processed` (this module) / ``_keyframe_idx``
(:class:`~core.perception.tracker.PerceptionPipeline`) counts frames that
completed the full detect -> fuse -> associate pipeline (one increment per
keyframe-gated ``process()`` call); ``core.perception.detector.dump_raw_detections``
writes one JSONL row per RAW per-tile detection candidate within that SAME call
(often several per frame -- one tile can yield multiple boxes) -- so raw_detections
having more rows than keyframes_processed has increments is the expected shape of
two counters at different granularities, not evidence of a queue drop.
"""
from __future__ import annotations

import logging
import threading
import time
import traceback
from typing import Callable

from core.interfaces import LidarScan, PanoFrame
from core.perception.tracker import PerceptionPipeline

_LOGGER = logging.getLogger(__name__)

#: Bounded restart budget (issue #174): a worker thread that ends for any reason
#: (not just the guarded per-frame exception path — anything unforeseen in the
#: loop scaffolding itself) is restarted this many times before the pipeline
#: gives up and leaves the last-known state in place rather than restart forever.
DEFAULT_MAX_RESTARTS = 3


class AsyncPerceptionWorker:
    """Runs one :class:`PerceptionPipeline` on a dedicated thread, latest-frame-wins.

    The caller's only entry point is :meth:`submit`; it never blocks on a forward
    pass — it latches the frame and returns immediately, replacing any not-yet-
    processed pending frame. A single background thread wakes on each submission,
    picks up the latest frame, and runs it through ``pipeline.process()``.

    Issue #174: :meth:`submit` also supervises the worker thread — if it finds the
    thread dead, it logs loudly and restarts a fresh one (bounded by
    ``max_restarts``), so a worker fault degrades to "restarted, frame dropped"
    instead of "every future frame silently discarded forever".
    """

    def __init__(
        self,
        pipeline: PerceptionPipeline,
        on_error: Callable[[Exception], None] | None = None,
        max_restarts: int = DEFAULT_MAX_RESTARTS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._pipeline = pipeline
        self._on_error = on_error
        self._max_restarts = max_restarts
        self._clock = clock
        self._cv = threading.Condition()
        self._pending: tuple[PanoFrame, LidarScan] | None = None
        # Issue #211: wall time (this clock's units) the worker started the forward it
        # is CURRENTLY inside, or None while idle -- the direct answer to "how long has
        # detection been behind" (bounded-lag visibility for the keep-latest policy).
        # Deliberately NOT keyed off `_pending`: a new frame arrives (and overwrites
        # `_pending`) every tick regardless of backlog, so a pending-based timestamp
        # would reset every ~0.2s and hide a genuinely stuck forward.
        self._in_flight_since: float | None = None
        self._stopped = False
        self._processed_count = 0
        self._submitted_count = 0
        # Issue #211: frames overwritten by a fresher `submit()` before the worker ever
        # started them -- the keep-latest policy's cost, made countable instead of only
        # inferable from submitted_count - processed_count (which also includes any
        # frame still pending or in flight).
        self._dropped_count = 0
        self._restart_count = 0
        self._last_error: BaseException | None = None
        self._thread = self._spawn_thread()

    def _spawn_thread(self) -> threading.Thread:
        thread = threading.Thread(
            target=self._run, name="vla-perception-worker", daemon=True
        )
        thread.start()
        return thread

    def submit(self, pano: PanoFrame, scan: LidarScan) -> None:
        """Non-blocking: latch ``(pano, scan)`` as the next frame to process.

        Overwrites any frame the worker has not yet picked up (drop-stale-frames
        policy, issue #88). No-op after :meth:`stop`. Also supervises the worker
        thread (issue #174): if it died, restart it (bounded, see
        ``max_restarts``) before latching the new frame, instead of letting every
        future submit silently vanish into a dead thread.
        """
        with self._cv:
            if self._stopped:
                return
            self._restart_if_dead_locked()
            if self._pending is not None:
                # A still-unpicked-up frame is being overwritten: this is the keep-
                # latest policy actually paying its cost (issue #211 observability).
                self._dropped_count += 1
            self._pending = (pano, scan)
            self._submitted_count += 1
            self._cv.notify()

    def _restart_if_dead_locked(self) -> None:
        """Restart the worker thread if it ended. Caller must hold ``self._cv``."""
        if self._thread.is_alive():
            return
        if self._restart_count >= self._max_restarts:
            return  # budget exhausted — leave state as-is; counters stay readable
        self._restart_count += 1
        _LOGGER.error(
            "AsyncPerceptionWorker: worker thread %r ended unexpectedly "
            "(restart %d/%d); last error: %r",
            self._thread.name, self._restart_count, self._max_restarts,
            self._last_error,
        )
        self._thread = self._spawn_thread()

    def stop(self, timeout: float = 2.0) -> None:
        """Signal the worker to exit and join it. Safe to call more than once."""
        with self._cv:
            self._stopped = True
            self._cv.notify()
            thread = self._thread  # snapshot under lock — submit() may not restart
        thread.join(timeout=timeout)

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
    def dropped_count(self) -> int:
        """Issue #211: frames overwritten in the pending slot before the worker ever
        started them (the keep-latest policy's cost). ``submitted_count ==
        processed_count + dropped_count + (1 if a frame is pending or in flight else
        0)`` at any instant -- the three counters partition every submitted frame."""
        with self._cv:
            return self._dropped_count

    @property
    def in_flight_age_s(self) -> float | None:
        """Issue #211: seconds since the worker started the forward it is CURRENTLY
        running, or ``None`` while idle. This is the direct "how far behind is
        detection" signal -- unlike a pending-frame timestamp (which a new submission
        refreshes every tick regardless of backlog), this only changes when the worker
        actually finishes or starts a forward, so a stuck/slow detector shows up as a
        steadily growing value instead of being masked by tick cadence."""
        with self._cv:
            since = self._in_flight_since
        return None if since is None else max(0.0, self._clock() - since)

    @property
    def is_alive(self) -> bool:
        return self._thread.is_alive()

    @property
    def restart_count(self) -> int:
        """How many times :meth:`submit` has restarted a dead worker thread."""
        with self._cv:
            return self._restart_count

    @property
    def keyframes_processed(self) -> int:
        """The underlying pipeline's own keyframe counter (issue #174).

        Reads ``pipeline._keyframe_idx`` directly off the (still-live) pipeline
        object rather than anything the worker thread maintains, so it stays a
        readable ``int`` — never ``None`` — for as long as this
        ``AsyncPerceptionWorker`` (and the pipeline it wraps) exists, independent
        of whether the worker thread itself is currently alive, dead, or
        mid-restart. ``getattr`` with a default guards against a pipeline stub
        that has no such attribute at all (e.g. a test double).
        """
        return int(getattr(self._pipeline, "_keyframe_idx", 0))

    def _run(self) -> None:
        while True:
            with self._cv:
                while self._pending is None and not self._stopped:
                    self._cv.wait()
                if self._pending is None and self._stopped:
                    return
                pano, scan = self._pending
                self._pending = None
                self._in_flight_since = self._clock()
            try:
                self._pipeline.process(pano, scan)
            except Exception as exc:  # a perception glitch must never kill the worker
                self._last_error = exc
                _LOGGER.error(
                    "AsyncPerceptionWorker: pipeline.process() raised %r; frame "
                    "dropped, worker continues.\n%s",
                    exc, traceback.format_exc(),
                )
                self._safe_on_error(exc)
            with self._cv:
                self._processed_count += 1
                self._in_flight_since = None

    def _safe_on_error(self, exc: Exception) -> None:
        """Call the caller's ``on_error`` hook without letting IT kill the worker.

        Issue #174 root cause: the pre-fix ``_run`` called ``self._on_error(exc)``
        from inside the ``except`` block with no guard of its own. On a live
        cluster run that callback logs through ``rclpy``, which can itself raise
        (a "benign teardown RCLError" observed in the adapter log on every dead
        loft slot) — that second, unguarded exception then escaped ``_run``
        entirely and silently ended the thread. Guarding the callback call here
        closes that specific escape hatch; the bounded-restart supervisor in
        :meth:`submit` is the backstop for any other way the thread might end.
        """
        if self._on_error is None:
            return
        try:
            self._on_error(exc)
        except Exception:
            _LOGGER.error(
                "AsyncPerceptionWorker: on_error callback itself raised; "
                "suppressed so it cannot end the worker thread.\n%s",
                traceback.format_exc(),
            )
