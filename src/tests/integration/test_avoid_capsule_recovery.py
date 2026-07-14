"""Regression: the avoid capsule must never be violated even when it engulfs the
start and the corridor gate, forcing the unreachable-route recovery path.

Reproduces the HIGH-severity defect found in independent verification: with
``corridor_between(table, chair)`` + ``avoid near(vase)`` where the inflated vase
capsule swallows both the corridor gate midpoint and the start pose, ``plan_through``
returns None and ``_build_route`` used to fall back to a RAW straight segment
``[start, nearest_legal]`` that was never validated against the costmap. The
BreadcrumbFollower then handed those raw points straight through, and the vehicle
drove to within 0.025 m of the vase (capsule radius 1.45 m) — a hard capsule
violation that architecture §1 row 4 forbids ("a deliberate least-bad choice, never
a silent geometry edit").

The fix plans the recovery path through A* over the stamped costmap, so the driven
trajectory never enters the capsule. These tests assert capsule-clear on BOTH the
published waypoint stream and the odom trajectory, and that an answer is still
published (never silence).
"""
from __future__ import annotations

import numpy as np
import pytest

from core.fsm.controller import QuestionController, State
from core.geometry.toolbox import avoid_capsule, capsule_violated
from core.heads import build_callables
from core.interfaces import WaypointCmd
from core.mocks.mock_io import FakeClock, MockRobotIO
from core.mocks.synthetic_scene import Room, SyntheticScene
from core.perception.scene_index import BasicSceneIndex
from core.plan_schema import Anchor, AvoidSpec

from ._scaled import BUDGET_SCALE, install_scaled_budget

# Full-controller capsule-recovery sims: slow even after budget-scaling (A* recovery
# replans every tick). Fast tier skips these; run them via `pytest -m ""`.
pytestmark = pytest.mark.slow

TICK_DT = 0.2  # 5 Hz
DRIVE_STEP_M = 0.1


def _run(io: MockRobotIO, ctrl: QuestionController, clk: FakeClock, max_t: float = 600.0):
    """Tick the controller at 5 Hz, chasing the published waypoints. Return the odom
    trajectory as an (N, 2) array.

    The FSM budget/watchdog gates are compressed (see tests/integration/_scaled.py); the
    capsule-clearance geometry these regression cases assert is driven by ticks, not by
    sim-time, so it is unchanged."""
    install_scaled_budget(io)
    max_t = max_t * BUDGET_SCALE
    traj = [(io.latest_odom().x, io.latest_odom().y)]
    while ctrl.state is not State.DONE and clk.now() < max_t:
        ctrl.tick(io)
        clk.advance(TICK_DT)
        if io.waypoints:
            wp = io.waypoints[-1]
            od = io.latest_odom()
            dx, dy = wp.x - od.x, wp.y - od.y
            d = (dx * dx + dy * dy) ** 0.5
            s = min(DRIVE_STEP_M, d)
            if d > 1e-6:
                io.set_pose(od.x + dx / d * s, od.y + dy / d * s)
            traj.append((io.latest_odom().x, io.latest_odom().y))
    ctrl.tick(io)
    return np.array(traj)


def _capsule_engulfing_scene() -> tuple[SyntheticScene, BasicSceneIndex]:
    """table (3,2), chair (3,4) -> gate mid ~(3,3); vase at (2,3) whose capsule
    (~1.85 m inflated blocked disc, oracle radius 1.45 m) swallows gate mid AND the
    (0.7, 3) start; sofa goal at (8,3) beyond the wall of avoidance."""
    sc = SyntheticScene(0)
    sc.rooms = [Room(0.0, 0.0, 10.0, 6.0)]
    sc.place_box("table", 3.0, 2.0, 0.4, 0.4, 0.5)
    sc.place_box("chair", 3.0, 4.0, 0.4, 0.4, 0.5)
    sc.place_box("vase", 2.0, 3.0, 0.3, 0.3, 0.4)
    sc.place_box("sofa", 8.0, 3.0, 0.5, 0.5, 0.5)
    return sc, BasicSceneIndex(sc.instances())


def test_capsule_engulfs_start_and_gate_recovery_never_violates():
    sc, idx = _capsule_engulfing_scene()
    clk = FakeClock(0.0)
    io = MockRobotIO(sc, clk, start_x=0.7, start_y=3.0)
    io.set_question(
        "take the path between the table and the chair to the sofa, avoid the vase"
    )
    ctrl = QuestionController(**build_callables(idx))

    traj = _run(io, ctrl, clk)

    # Never silence: an answer (terminal waypoint) is still published.
    assert ctrl.state is State.DONE
    assert io.waypoints, "no waypoints published — silence is the only unforgivable failure"
    assert isinstance(io.waypoints[-1], WaypointCmd)

    cap = avoid_capsule(AvoidSpec(near=Anchor(noun="vase")), idx)

    # Published waypoint stream — fully controllable output — must never enter the capsule.
    wp_stream = np.array([[w.x, w.y] for w in io.waypoints])
    wp_violated, _ = capsule_violated(wp_stream, cap)
    assert wp_violated is False, "published waypoint stream violates the hard avoid capsule"

    # Odom trajectory: the START pose (0.7, 3.0) is itself inside the 1.45 m capsule by
    # construction (dist 1.3 m) — no planner can undo where the vehicle begins. What the
    # fix guarantees is that the vehicle drives OUT and never back in: from the first
    # vertex that clears the capsule onward, the odom trajectory is violation-free, and
    # it never drives DEEPER than the start (the old raw-segment bug drove to 0.025 m).
    d_to_vase = np.hypot(traj[:, 0] - 2.0, traj[:, 1] - 3.0)
    assert d_to_vase.min() >= d_to_vase[0] - 1e-6, (
        "vehicle drove deeper into the capsule than its engulfed start "
        f"(min {d_to_vase.min():.3f} m < start {d_to_vase[0]:.3f} m)"
    )
    outside = d_to_vase > cap.radius
    assert outside.any(), "vehicle never escaped the avoid capsule"
    first_clear = int(np.argmax(outside))
    post = traj[first_clear:]
    post_violated, pt = capsule_violated(post, cap)
    assert post_violated is False, (
        f"odom re-entered the avoid capsule after escaping (first at {pt})"
    )


def test_capsule_engulfs_gate_only_full_trajectory_clear():
    """Parametrized geometry: the capsule swallows the corridor gate but NOT the start.

    Here the start is legal, so the recovery path is fully controllable and the ENTIRE
    odom trajectory (start included) must be capsule-clear on both streams.
    """
    sc = SyntheticScene(0)
    sc.rooms = [Room(0.0, 0.0, 12.0, 6.0)]
    sc.place_box("table", 5.0, 2.0, 0.4, 0.4, 0.5)
    sc.place_box("chair", 5.0, 4.0, 0.4, 0.4, 0.5)
    sc.place_box("vase", 5.0, 3.0, 0.3, 0.3, 0.4)  # engulfs gate mid (5,3), not start
    sc.place_box("sofa", 10.0, 3.0, 0.5, 0.5, 0.5)
    idx = BasicSceneIndex(sc.instances())

    clk = FakeClock(0.0)
    io = MockRobotIO(sc, clk, start_x=0.7, start_y=1.0)  # well outside the capsule
    io.set_question(
        "take the path between the table and the chair to the sofa, avoid the vase"
    )
    ctrl = QuestionController(**build_callables(idx))

    traj = _run(io, ctrl, clk)

    assert ctrl.state is State.DONE
    assert io.waypoints
    cap = avoid_capsule(AvoidSpec(near=Anchor(noun="vase")), idx)

    # Start is clear, so the full odom trajectory must never touch the capsule.
    start_dist = np.hypot(traj[0, 0] - 5.0, traj[0, 1] - 3.0)
    assert start_dist > cap.radius, "test precondition: start must be outside the capsule"
    odom_violated, pt = capsule_violated(traj, cap)
    assert odom_violated is False, f"odom trajectory violates the avoid capsule at {pt}"

    wp_stream = np.array([[w.x, w.y] for w in io.waypoints])
    wp_violated, _ = capsule_violated(wp_stream, cap)
    assert wp_violated is False
