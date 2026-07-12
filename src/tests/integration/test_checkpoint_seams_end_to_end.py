"""Full-controller integration with all four checkpoint seams stubbed active.

Runs real QuestionControllers wired by ``build_callables`` with scripted CP2/CP3/CP4/CP5
seam stubs (no network, no real checkpoint builders). Verifies the controller still
completes and publishes a legal answer with the seams live, and that the seams are
consulted on the relevant qtypes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest

from core.fsm.controller import QuestionController, State
from core.heads import build_callables
from core.interfaces import MarkerBox, WaypointCmd
from core.mocks.mock_io import FakeClock, MockRobotIO
from core.mocks.synthetic_scene import Room, SyntheticScene
from core.perception.scene_index import BasicSceneIndex

from ._scaled import BUDGET_SCALE, install_scaled_budget

# Full-controller sims with all checkpoint seams live: slow even after budget-scaling.
# Fast tier skips these; run them via `pytest -m ""`.
pytestmark = pytest.mark.slow

TICK_DT = 0.2


@dataclass
class _Verify:
    action: str
    winner: Any


@dataclass
class _Confirm:
    action: str
    confidence: float = 0.9


@dataclass
class _Miss:
    action: str
    confidence: float = 0.0


@dataclass
class _Frontier:
    action: str
    index: int | None = None


def _run(io, ctrl, clk, *, drive, max_t=600.0):
    # Compress the FSM's budget/watchdog gates (see tests/integration/_scaled.py); the seam
    # call counts and published answers these cases assert are unaffected by tick count.
    install_scaled_budget(io)
    max_t = max_t * BUDGET_SCALE
    while ctrl.state is not State.DONE and clk.now() < max_t:
        ctrl.tick(io)
        clk.advance(TICK_DT)
        if drive and io.waypoints:
            wp = io.waypoints[-1]
            od = io.latest_odom()
            dx, dy = wp.x - od.x, wp.y - od.y
            d = (dx * dx + dy * dy) ** 0.5
            s = min(0.1, d)
            if d > 1e-6:
                io.set_pose(od.x + dx / d * s, od.y + dy / d * s)
    ctrl.tick(io)


def _seams(counters):
    """A full set of scripted checkpoint seams, recording invocations in `counters`."""

    def verifier(*, question, winner, runner_up, winner_facts, notes, resolve_again):
        counters["cp4"] += 1
        return _Verify("keep", winner)  # confirm the deterministic winner

    def anchor_confirmer(anchor_desc, crop):
        counters["cp3"] += 1
        return _Confirm("confirm")

    def miss_recoverer(noun, raw, tiles):
        counters["cp2"] += 1
        return _Miss("absent")

    def frontier_selector(question, frontiers):
        counters["cp5"] += 1
        return _Frontier("fallback")

    return dict(
        verifier=verifier,
        anchor_confirmer=anchor_confirmer,
        miss_recoverer=miss_recoverer,
        frontier_selector=frontier_selector,
        remaining_s=lambda: 300.0,
        budget_frac=lambda: 0.9,
        tiles_fn=lambda: [],
    )


def test_object_reference_completes_with_all_seams_active():
    sc = SyntheticScene(0)
    sc.place_box("table", 1.0, 1.0, 0.6, 0.6, 0.5)
    sc.place_box("lamp", 1.5, 1.0, 0.2, 0.2, 0.3)
    sc.place_box("table", 4.0, 4.0, 0.6, 0.6, 0.5)
    idx = BasicSceneIndex(sc.instances())
    clk = FakeClock(0.0)
    io = MockRobotIO(sc, clk)
    io.set_question("find the table near the lamp")

    counters = {"cp2": 0, "cp3": 0, "cp4": 0, "cp5": 0}
    ctrl = QuestionController(**build_callables(idx, **_seams(counters)))
    _run(io, ctrl, clk, drive=False)

    assert ctrl.state is State.DONE
    assert len(io.markers) == 1
    m = io.markers[0]
    assert isinstance(m, MarkerBox)
    # CP4 verified the object-reference answer at least once (still the correct near table).
    assert counters["cp4"] >= 1
    assert (round(m.cx, 3), round(m.cy, 3)) == (1.0, 1.0)


def test_instruction_following_completes_with_all_seams_active():
    sc = SyntheticScene(0)
    sc.rooms = [Room(0.0, 0.0, 8.0, 6.0)]
    sc.place_box("sofa", 6.0, 3.0, 0.5, 0.5, 0.5)
    idx = BasicSceneIndex(sc.instances())
    clk = FakeClock(0.0)
    io = MockRobotIO(sc, clk, start_x=0.7, start_y=3.0)
    io.set_question("go to the sofa")

    counters = {"cp2": 0, "cp3": 0, "cp4": 0, "cp5": 0}
    ctrl = QuestionController(**build_callables(idx, **_seams(counters)))
    _run(io, ctrl, clk, drive=True)

    assert ctrl.state is State.DONE
    assert io.waypoints
    assert isinstance(io.waypoints[-1], WaypointCmd)
    # CP3 confirmed anchor arrival at least once on the driven route.
    assert counters["cp3"] >= 1
