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
