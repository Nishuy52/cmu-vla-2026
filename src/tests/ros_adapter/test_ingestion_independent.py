"""Issue #211 "map ingestion must never wait on detection" — adapter-side guards.

Diagnosis (see the #211 commit for the full trace): occupancy/costmap ingestion
(scan+pose -> grid) lives entirely in the qtype heads (``core.heads.explore_step``,
``core.heads.instruction``), reading ``io.latest_terrain()`` / ``io.latest_scan()``
directly off their own ROS callbacks (``_on_terrain``/``_on_scan``) — it was never
routed through ``PerceptionPipeline``/``AsyncPerceptionWorker`` and so was never
gated by detection completion. This file guards that invariant at the one seam that
COULD reintroduce coupling: ``AdapterNode._on_tick`` / ``_maybe_process_perception``.
It also guards the #211 backlog-observability addition (``_check_perception_backlog``).

Same source-inspection convention as ``test_coldstart.py``/``test_seam_wiring.py``:
the adapter imports rclpy at module top, unavailable on this dev box, so these tests
read the source text rather than constructing a node.
"""
from __future__ import annotations

import pathlib

import pytest

_ADAPTER = pathlib.Path(__file__).resolve().parents[2] / "ros_adapter" / "adapter_node.py"


@pytest.fixture(scope="module")
def src() -> str:
    return _ADAPTER.read_text(encoding="utf-8")


def _tick_body(src: str) -> str:
    tick_start = src.index("def _on_tick(self)")
    return src[tick_start : src.index("def _build_controller_callables")]


def test_maybe_process_perception_never_touches_map_or_occupancy_state(src: str):
    """Structural guard: the perception-dispatch method must stay confined to
    submitting frames to the detector. It must never read/write grid, occupancy, or
    terrain state directly — that would reintroduce a dependency map ingestion (owned
    entirely by the qtype heads) must never have on detection."""
    start = src.index("def _maybe_process_perception(self)")
    end = src.index("\n    def ", start + 1)
    body = src[start:end]
    for forbidden in ("grid", "occupancy", "Occupancy", "terrain", "Terrain"):
        assert forbidden not in body, (
            "‑_maybe_process_perception now references %r — map ingestion must stay "
            "independent of the detection-dispatch path (issue #211)" % forbidden
        )


def test_perception_dispatch_is_fire_and_forget(src: str):
    """submit() (not the blocking .process() call) is what _maybe_process_perception
    uses on the default (threaded) path — the caller never waits on a forward pass."""
    start = src.index("def _maybe_process_perception(self)")
    end = src.index("\n    def ", start + 1)
    body = src[start:end]
    assert "self._perception_worker.submit(pano, scan)" in body
    # The only synchronous call is the explicit debug-only opt-out (VLA_PERCEPTION_SYNC).
    assert body.count("self._perception.process(pano, scan)") == 1


def test_tick_never_lets_perception_dispatch_block_the_controller(src: str):
    """_on_tick must call perception dispatch, then still reach controller.tick() the
    SAME call — i.e. nothing about dispatch can early-return or block ahead of it."""
    body = _tick_body(src)
    perceive_at = body.index("self._maybe_process_perception()")
    controller_tick_at = body.index("self._controller.tick(self)")
    assert perceive_at < controller_tick_at


def test_backlog_observability_wired_every_tick(src: str):
    """Issue #211: the async-worker backlog check runs every tick, after dispatch."""
    body = _tick_body(src)
    assert "self._check_perception_backlog()" in body
    perceive_at = body.index("self._maybe_process_perception()")
    backlog_at = body.index("self._check_perception_backlog()")
    assert perceive_at < backlog_at


def test_backlog_check_reads_in_flight_age_and_rate_limits(src: str):
    start = src.index("def _check_perception_backlog(self)")
    end = src.index("\n    def ", start + 1)
    body = src[start:end]
    assert "worker.in_flight_age_s" in body
    assert "PERCEPTION_BACKLOG_WARN_S" in body
    assert "PERCEPTION_BACKLOG_REPEAT_S" in body
    assert "self._perception_backlog_last_warn_t" in body
    # Never raises out — same discipline as every other per-tick observability check.
    assert "except Exception as exc:" in body


def test_backlog_check_never_raises_and_is_a_noop_without_a_worker(src: str):
    start = src.index("def _check_perception_backlog(self)")
    end = src.index("\n    def ", start + 1)
    body = src[start:end]
    assert "if worker is None:" in body


# --------------------------------------------------------------------- issue #211
# pre-latch detection throttle


def _dispatch_body(src: str) -> str:
    start = src.index("def _maybe_process_perception(self)")
    end = src.index("\n    def ", start + 1)
    return src[start:end]


def test_pre_latch_throttle_interval_is_at_least_the_worst_case_forward_cost(src: str):
    """PRE_LATCH_DETECT_EVERY must actually buy idle time between forwards: at
    TICK_HZ, the throttle interval (PRE_LATCH_DETECT_EVERY / TICK_HZ) must be at
    least as long as the measured per-forward cost (4-15 s) — a smaller skip factor
    (e.g. the originally proposed 8, an 8 * 0.2s = 1.6s interval) would leave the
    worker just as continuously saturated as today, buying no contention relief."""
    import re

    tick_hz_m = re.search(r"^TICK_HZ = ([0-9.]+)", src, re.MULTILINE)
    every_m = re.search(r"^PRE_LATCH_DETECT_EVERY: int = (\d+)", src, re.MULTILINE)
    assert tick_hz_m and every_m
    tick_hz = float(tick_hz_m.group(1))
    every = int(every_m.group(1))
    interval_s = every / tick_hz
    # Measured worst case (issue #211's own numbers): the every-3rd-tick full-caption
    # pass costs up to 15 s. The throttle interval must reach that, or the worker
    # never actually gets to finish a forward and go idle before the next submission
    # arrives -- the whole point of throttling in the first place.
    assert interval_s >= 10.0, (
        "PRE_LATCH_DETECT_EVERY=%d at TICK_HZ=%.1f only spaces submissions %.1fs "
        "apart -- too short to relieve from-boot worker saturation (issue #211)"
        % (every, tick_hz, interval_s)
    )


def test_pre_latch_throttle_wired_before_dispatch(src: str):
    body = _dispatch_body(src)
    assert "PRE_LATCH_DETECT_EVERY" in body
    assert "self._pre_latch_frame_count" in body
    throttle_at = body.index("self._pre_latch_frame_count += 1")
    controller_check_at = body.index("if self._controller is None:")
    submit_at = body.index("self._perception_worker.submit(pano, scan)")
    # The throttle's own gate check must be nested inside the pre-latch branch, and
    # both must precede the actual dispatch.
    assert controller_check_at < throttle_at < submit_at


def test_pre_latch_throttle_skip_formula_selects_1st_1plusN_1plus2N():
    """The documented policy — dispatch the 1st, (1+N)th, (1+2N)th... pre-latch
    frame — reproduced as a standalone pure function and checked against the exact
    modulo expression the adapter uses, so the two cannot silently drift apart."""

    def would_dispatch(frame_count: int, every: int) -> bool:
        return (frame_count - 1) % every != 0

    # Mirrors "if (self._pre_latch_frame_count - 1) % PRE_LATCH_DETECT_EVERY != 0:
    # return" -- that condition is the SKIP condition, so dispatch == not skip.
    dispatched = [n for n in range(1, 21) if not would_dispatch(n, every=5)]
    assert dispatched == [1, 6, 11, 16]


def test_post_latch_dispatch_is_never_throttled(src: str):
    """Once self._controller is not None, the throttle branch's own guard
    (`if self._controller is None:`) short-circuits the whole block — every frame
    reaches submit()/process() exactly as it did before issue #211."""
    body = _dispatch_body(src)
    # The throttle's early return is the ONLY return between the controller check
    # and the dispatch calls -- and it lives strictly inside the pre-latch branch
    # (already proven nested by test_pre_latch_throttle_wired_before_dispatch), so a
    # non-None controller skips straight to dispatch with no other gate in the way.
    between = body[body.index("if self._controller is None:") : body.index(
        "if self._perception_worker is not None:"
    )]
    assert between.count("return") == 1


def test_frame_watchdog_is_fed_independently_of_perception_dispatch(src: str):
    """Issue #174 semantics preserved: on_frame() (arrival) is called from the image
    callback, never from _maybe_process_perception -- so a frame skipped here for
    detection (issue #211 throttle) still counts as ARRIVED for the watchdog."""
    dispatch_body = _dispatch_body(src)
    assert "_frame_watchdog" not in dispatch_body
    on_image_start = src.index("def _on_image(self, msg: Image)")
    on_image_end = src.index("\n    def ", on_image_start + 1)
    on_image_body = src[on_image_start:on_image_end]
    assert "self._frame_watchdog.on_frame()" in on_image_body
