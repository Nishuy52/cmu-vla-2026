"""run_question over a SHORT replay bag: the end-of-data / free-run regression suite.

Pins the defect found on the real 123 s jingfan bag: the bag ends long before the FSM's
510/570 s budget gates, so the runner must free-run the frozen world until the controller
publishes a real answer — never surfacing an exploration WaypointCmd as the answer, and
never ending the loop merely because the recorded data ran out.

Fixture: a synthetic ~30 s store (messages t=1..30 s) driven with a large budget-scale so a
run completes in a handful of wall-seconds while still exercising the free-run path (the
unscaled watchdog is 570 s; scaling compresses it onto the 30 s bag horizon).
"""
from __future__ import annotations

import numpy as np
import pytest

from core.interfaces import (
    IntAnswer,
    MarkerBox,
    QType,
    Question,
    WaypointCmd,
    WATCHDOG_FLOOR_S,
)
from core.replay.replay_io import (
    CH_ODOM,
    CH_PANO,
    CH_QUESTION,
    CH_SCAN,
    CH_TERRAIN,
    CH_TERRAIN_EXT,
    MessageStore,
    ReplayRobotIO,
)
from core.interfaces import LidarScan, OdomState, PanoFrame, TerrainPatch
from core.runner.single import run_question

_BAG_END_S = 30.0
# Compress the 570 s watchdog onto the ~30 s bag: 570 * 0.05 = 28.5 s < 30 s, so the gate is
# reached inside the recorded window; the explore budgets (>=210 s) scale below that too, so
# a run assembles and publishes without needing to free-run for minutes.
_SCALE = 0.05


def _short_store(question: str) -> MessageStore:
    """A ~30 s store with a latched question and every channel the FSM touches per tick."""
    store = MessageStore()
    store.add(CH_QUESTION, 1.0, Question(text=question, t_received=1.0))
    for sec in range(1, int(_BAG_END_S) + 1):
        t = float(sec)
        odom = OdomState(t=t, x=float(sec) * 0.1, y=0.0, z=0.0, yaw=0.0)
        store.add(CH_ODOM, t, odom)
        store.add(CH_SCAN, t, LidarScan(t=t, points=np.zeros((4, 3), dtype=np.float32)))
        store.add(
            CH_TERRAIN,
            t,
            TerrainPatch(t=t, points=np.zeros((4, 4), dtype=np.float32), extended=False),
        )
        store.add(
            CH_TERRAIN_EXT,
            t,
            TerrainPatch(t=t, points=np.zeros((4, 4), dtype=np.float32), extended=True),
        )
        store.add(
            CH_PANO,
            t,
            PanoFrame(t=t, image=np.zeros((4, 8, 3), dtype=np.uint8), odom=odom),
        )
    return store.finalize()


def _io_for(question: str) -> ReplayRobotIO:
    io = ReplayRobotIO(_short_store(question))
    io.raw_clock().set(1.0)  # start at the first message / question receipt
    return io


def _run(question: str):
    return run_question(question, _io_for(question), tick_hz=5.0, budget_scale=_SCALE)


# --------------------------------------------------------------------------- completion


def test_short_bag_completes_with_published_answer():
    r = _run("Find the chair closest to the table")
    assert r.published is True
    assert r.answer is not None
    assert "done" in r.states_visited


def test_end_of_data_recorded_near_bag_end():
    """The bag exhausts around t=30 s; end_of_data_at_sim_s reflects that (raw bag time)."""
    # Use a slower gate so the run keeps ticking past the bag end before it answers.
    io = _io_for("Find the chair closest to the table")
    r = run_question(
        "Find the chair closest to the table", io, tick_hz=5.0, budget_scale=0.2
    )
    assert r.end_of_data_at_sim_s is not None
    assert r.end_of_data_at_sim_s == pytest.approx(_BAG_END_S, abs=1.0)


def test_sim_end_reaches_watchdog_gate():
    """The FSM's perceived elapsed at completion is at/after a budget gate, not stalled."""
    r = _run("Find the chair closest to the table")
    # With scaling the FSM sees amplified time; completion is on a budget/watchdog gate,
    # never frozen at the ~30 s bag end.
    assert r.elapsed_sim_s > _BAG_END_S
    # It reached at least the forced-assembly horizon in the FSM's scaled clock.
    assert r.elapsed_sim_s >= 200.0


# ------------------------------------------------------------------ answer-type per qtype


def test_numerical_answer_type():
    r = _run("How many chairs are in the room")
    assert r.qtype is QType.NUMERICAL
    assert isinstance(r.answer, IntAnswer)


def test_object_reference_answer_type():
    r = _run("Find the chair closest to the table")
    assert r.qtype is QType.OBJECT_REFERENCE
    assert isinstance(r.answer, MarkerBox)


def test_instruction_following_answer_type():
    r = _run("Go to the table and stop at the chair")
    assert r.qtype is QType.INSTRUCTION_FOLLOWING
    assert isinstance(r.answer, WaypointCmd)


# ---------------------------------------------------- the misleading-waypoint-as-answer pin


def test_object_reference_never_reports_waypoint_as_answer():
    """The core defect: exploration WaypointCmds must never be surfaced as the OR answer."""
    io = _io_for("Find the chair closest to the table")
    r = run_question(
        "Find the chair closest to the table", io, tick_hz=5.0, budget_scale=_SCALE
    )
    # Exploration DID publish waypoints into the sink during the run...
    assert isinstance(r.answer, MarkerBox)  # ...yet the reported answer is the OR marker,
    assert not isinstance(r.answer, WaypointCmd)  # never a waypoint.


def test_floor_used_flag_set_when_answer_from_floor():
    """An empty scene forces the floor path; floor_used must report it."""
    r = _run("Find the chair closest to the table")
    assert r.floor_used is True
