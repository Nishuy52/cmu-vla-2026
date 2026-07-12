"""Unit tests for InstructionHead's unreachable-route recovery (_build_route fallback).

The HIGH-severity defect: when ``plan_through`` returns None (the hard avoid capsules
make the route unreachable), ``_build_route`` used to build a RAW straight segment
``[start_xy, legal]`` that was never validated against the costmap and could cut
straight through a hard capsule. The fix routes the recovery through A* over the
stamped costmap; every path handed to the BreadcrumbFollower is astar/plan_through
output, never a raw segment.
"""
from __future__ import annotations

import logging

import numpy as np
import pytest

from core.geometry.toolbox import avoid_capsule, capsule_violated
from core.heads.instruction import InstructionHead
from core.interfaces import OdomState, WaypointCmd
from core.mocks.synthetic_scene import Room, SyntheticScene
from core.nav.breadcrumbs import line_of_sight
from core.perception.scene_index import BasicSceneIndex
from core.plan_schema import Anchor, AvoidSpec, LegKind, Plan, QType, RouteLeg


def _goto(noun: str) -> RouteLeg:
    return RouteLeg(kind=LegKind.GOTO, anchors=[Anchor(noun=noun)])


def _corridor(a: str, b: str) -> RouteLeg:
    return RouteLeg(kind=LegKind.CORRIDOR_BETWEEN, anchors=[Anchor(noun=a), Anchor(noun=b)])


def _if_plan(route, avoid) -> Plan:
    return Plan(qtype=QType.INSTRUCTION_FOLLOWING, question_raw="go", route=route, avoid=avoid)


class _DriveIO:
    """RobotIO stub: fixed terrain patch, kinematic pose chasing the last waypoint."""

    def __init__(self, sc: SyntheticScene, start=(0.7, 3.0), step=0.1):
        self._sc = sc
        self._x, self._y = start
        self._t = 0.0
        self._step = step
        self.waypoints: list[WaypointCmd] = []

    def latest_terrain(self, extended: bool = False):
        return self._sc.terrain_patch(extended=extended, t=self._t)

    def latest_odom(self):
        return OdomState(t=self._t, x=self._x, y=self._y, z=0.0, yaw=0.0)

    def publish_waypoint(self, wp: WaypointCmd):
        self.waypoints.append(wp)

    def tick_motion(self):
        if not self.waypoints:
            self._t += 0.2
            return
        wp = self.waypoints[-1]
        dx, dy = wp.x - self._x, wp.y - self._y
        d = (dx * dx + dy * dy) ** 0.5
        s = min(self._step, d)
        if d > 1e-6:
            self._x += dx / d * s
            self._y += dy / d * s
        self._t += 0.2

    @property
    def pose(self):
        return (self._x, self._y)


def _engulfing_scene() -> tuple[SyntheticScene, BasicSceneIndex]:
    sc = SyntheticScene(0)
    sc.rooms = [Room(0.0, 0.0, 10.0, 6.0)]
    sc.place_box("table", 3.0, 2.0, 0.4, 0.4, 0.5)
    sc.place_box("chair", 3.0, 4.0, 0.4, 0.4, 0.5)
    sc.place_box("vase", 2.0, 3.0, 0.3, 0.3, 0.4)  # capsule engulfs gate mid + start
    sc.place_box("sofa", 8.0, 3.0, 0.5, 0.5, 0.5)
    return sc, BasicSceneIndex(sc.instances())


@pytest.mark.slow
def test_build_route_fallback_uses_astar_not_raw_segment():
    """When the route is unreachable, the follower's path is an A*-planned polyline
    (many costmap-adjacent vertices), never a raw 2-point [start, legal] segment."""
    sc, idx = _engulfing_scene()
    head = InstructionHead(
        plan=_if_plan([_corridor("table", "chair"), _goto("sofa")],
                      [AvoidSpec(near=Anchor(noun="vase"))])
    )
    io = _DriveIO(sc)
    head.advance(io, idx)  # grounds legs, stamps capsule, builds recovery route

    assert head._follower is not None
    path = head._follower.path
    # The old bug produced exactly 2 points [start, legal]. A* over the stamped costmap
    # produces a stepwise polyline hugging passable cells.
    assert len(path) > 2, "recovery path is a raw straight segment, not an A* plan"

    # Every consecutive path edge is in clear line-of-sight over the stamped costmap
    # (i.e. crosses only passable cells) — the defining property a raw segment lacked.
    cm = head._costmap
    for p, q in zip(path[:-1], path[1:]):
        assert line_of_sight(cm, p, q), f"recovery path edge {p}->{q} crosses a blocked cell"


@pytest.mark.slow
def test_build_route_fallback_drive_never_violates_capsule():
    """Driving the recovery route never enters the hard avoid capsule (aside from the
    engulfed start, which no planner can undo)."""
    sc, idx = _engulfing_scene()
    head = InstructionHead(
        plan=_if_plan([_corridor("table", "chair"), _goto("sofa")],
                      [AvoidSpec(near=Anchor(noun="vase"))])
    )
    io = _DriveIO(sc)
    traj = [io.pose]
    for _ in range(4000):
        head.advance(io, idx)
        io.tick_motion()
        traj.append(io.pose)
        tw = head.terminal_waypoint()
        if tw is not None and ((io.pose[0] - tw.x) ** 2 + (io.pose[1] - tw.y) ** 2) ** 0.5 < 0.3:
            break
    traj = np.array(traj)

    cap = avoid_capsule(AvoidSpec(near=Anchor(noun="vase")), idx)
    d = np.hypot(traj[:, 0] - 2.0, traj[:, 1] - 3.0)
    # Never drives deeper than the engulfed start; the old bug reached ~0.025 m.
    assert d.min() >= d[0] - 1e-6
    # Waypoint stream fully clear.
    wp = np.array([[w.x, w.y] for w in io.waypoints])
    violated, _ = capsule_violated(wp, cap)
    assert violated is False


@pytest.mark.slow
def test_recovery_remains_in_place_when_astar_fails(caplog):
    """If A* to the nearest legal point ever fails (theoretically impossible given the
    BFS reachability guarantee), the head emits an in-place single-vertex path — never
    a raw unvalidated segment — and logs a flight-recorder-visible event."""
    sc, idx = _engulfing_scene()
    head = InstructionHead(
        plan=_if_plan([_goto("sofa")], [AvoidSpec(near=Anchor(noun="vase"))])
    )
    io = _DriveIO(sc)
    # Ground + build enough state so the costmap exists.
    head.advance(io, idx)
    # Force plan_through and astar to fail by monkeypatching the module functions the
    # head calls, then rebuild via the recovery helper directly.
    import core.heads.instruction as mod

    orig_astar = mod.astar
    mod.astar = lambda *a, **k: None
    try:
        with caplog.at_level(logging.WARNING, logger="core.heads.instruction"):
            start = io.pose
            path = head._recover_path(start)
    finally:
        mod.astar = orig_astar

    # Single in-place vertex snapped to the start cell centre — not a 2-point segment.
    assert len(path) == 1
    assert any("remaining in place" in r.message for r in caplog.records)
