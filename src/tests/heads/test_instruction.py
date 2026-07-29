"""InstructionHead: leg grounding, ungrounded tracking, corridor/avoid routing, drive."""
from __future__ import annotations

import numpy as np
import pytest

from core.geometry.toolbox import (
    avoid_capsule,
    capsule_violated,
    corridor_gate,
    threading_check,
)
from core.heads.instruction import InstructionHead
from core.interfaces import WaypointCmd
from core.mocks.mock_io import FakeClock, MockRobotIO
from core.mocks.synthetic_scene import SyntheticScene
from core.perception.scene_index import BasicSceneIndex
from core.plan_schema import Anchor, AvoidSpec, LegKind, RouteLeg
from tests.heads._helpers import inst, instruction_plan, scene


def _goto(noun: str) -> RouteLeg:
    return RouteLeg(kind=LegKind.GOTO, anchors=[Anchor(noun=noun)])


def _corridor(n1: str, n2: str) -> RouteLeg:
    return RouteLeg(kind=LegKind.CORRIDOR_BETWEEN, anchors=[Anchor(noun=n1), Anchor(noun=n2)])


def _via(noun: str) -> RouteLeg:
    return RouteLeg(kind=LegKind.VIA_NEAR, anchors=[Anchor(noun=noun)])


class _DriveIO:
    """Minimal RobotIO: fixed terrain patch, kinematic pose that chases the last waypoint."""

    def __init__(self, sc: SyntheticScene, start=(0.7, 3.0), step=0.1):
        self._sc = sc
        self._x, self._y = start
        self._t = 0.0
        self._step = step
        self.waypoints: list[WaypointCmd] = []

    def latest_terrain(self, extended: bool = False):
        return self._sc.terrain_patch(extended=extended, t=self._t)

    def latest_odom(self):
        from core.interfaces import OdomState

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


def _if_scene() -> tuple[SyntheticScene, BasicSceneIndex]:
    sc = SyntheticScene(0)
    sc.rooms = [type(sc.rooms[0])(0.0, 0.0, 8.0, 6.0)]
    sc.place_box("table", 3.0, 2.0, 0.4, 0.4, 0.5)
    sc.place_box("chair", 3.0, 4.0, 0.4, 0.4, 0.5)
    sc.place_box("sofa", 6.0, 3.0, 0.5, 0.5, 0.5)
    sc.place_box("vase", 3.0, 5.4, 0.3, 0.3, 0.4)
    return sc, BasicSceneIndex(sc.instances())


# --------------------------------------------------------------------------- grounding


def test_ungrounded_count_before_advance():
    head = InstructionHead(plan=instruction_plan([_goto("sofa"), _goto("chair")]))
    assert head.ungrounded_subgoals() == 2


def test_grounds_present_anchors():
    sc, idx = _if_scene()
    head = InstructionHead(plan=instruction_plan([_goto("sofa")]))
    io = _DriveIO(sc)
    head.advance(io, idx)
    assert head.ungrounded_subgoals() == 0


def test_ungrounded_when_anchor_absent():
    sc, idx = _if_scene()
    head = InstructionHead(plan=instruction_plan([_goto("unicorn")]))
    io = _DriveIO(sc)
    head.advance(io, idx)
    assert head.ungrounded_subgoals() == 1


def test_terminal_waypoint_none_when_anchor_absent():
    # Zero instance evidence for the (only) leg's anchor: no leg ever grounds geometry,
    # so no route/follower is ever built. terminal_waypoint() must withhold (None) rather
    # than fabricate a coordinate — the FSM floor's richer scene-centroid/origin fallback
    # (core/fsm/floors.py _instruction_following) answers instead. Mirrors the
    # NumericalHead empty-index withhold pattern (core/heads/numerical.py).
    sc, idx = _if_scene()
    head = InstructionHead(plan=instruction_plan([_goto("unicorn")]))
    io = _DriveIO(sc)
    head.advance(io, idx)
    assert head.ungrounded_subgoals() == 1
    assert head.terminal_waypoint() is None


def test_terminal_waypoint_none_on_empty_index():
    # Perception fully dark (empty scene index, e.g. not wired yet): every leg stays
    # ungrounded (no anchors resolve at all), so terminal_waypoint() must withhold rather
    # than emit a fabricated waypoint. The watchdog floor supplies a legal fallback.
    head = InstructionHead(plan=instruction_plan([_goto("sofa")]))
    io = _DriveIO(SyntheticScene(0))
    head.advance(io, scene())
    assert head.ungrounded_subgoals() == 1
    assert head.terminal_waypoint() is None


def test_ungrounded_when_low_nobs():
    sc = scene(inst(1, "sofa", n_obs=1, centroid=(2.0, 2.0, 0.0)))
    head = InstructionHead(plan=instruction_plan([_goto("sofa")]))
    io = _DriveIO(SyntheticScene(0))
    head.advance(io, sc)
    assert head.ungrounded_subgoals() == 1  # present but under-observed


# --------------------------------------------------------------- issue #33 commit gate
def test_single_obs_leg_planned_but_not_committed():
    """A leg backed by only n_obs == 1 is PLANNED (geometry computed) but withheld from
    the COMMITTED route while no forced-assembly pressure has been injected."""
    sc = scene(inst(1, "sofa", n_obs=1, centroid=(2.0, 2.0, 0.0)))
    head = InstructionHead(plan=instruction_plan([_goto("sofa")]))
    io = _DriveIO(SyntheticScene(0))
    head.advance(io, sc)
    assert head._legs[0].geom is not None  # PLANNED: geometry already resolved
    assert head._follower is None  # NOT COMMITTED: no route built to drive
    assert head.terminal_waypoint() is None


def test_two_obs_leg_commits_without_forced_assembly():
    """n_obs == 2 clears the MIN_COMMIT_OBS floor and commits immediately — no forced-
    assembly pressure needed (distinct from the stricter 3-obs 'grounded' report gate)."""
    sc = scene(inst(1, "sofa", n_obs=2, centroid=(2.0, 2.0, 0.0)))
    head = InstructionHead(plan=instruction_plan([_goto("sofa")]))
    io = _DriveIO(SyntheticScene(0))
    head.advance(io, sc)
    assert head._follower is not None  # COMMITTED at n_obs == 2
    assert head.terminal_waypoint() is not None


def test_single_obs_leg_commits_under_forced_assembly():
    """Once the T-90 forced-assembly gate is reached, even a single-observation leg
    commits — late-stage expected points favor acting over stalling further."""
    sc = scene(inst(1, "sofa", n_obs=1, centroid=(2.0, 2.0, 0.0)))
    head = InstructionHead(
        plan=instruction_plan([_goto("sofa")]),
        forced_assembly=lambda: True,
    )
    io = _DriveIO(SyntheticScene(0))
    head.advance(io, sc)
    assert head._follower is not None  # forced-assembly pressure commits it anyway
    assert head.terminal_waypoint() is not None


def test_committable_prefix_stops_before_single_obs_leg():
    """A multi-leg route with a well-observed first leg and a single-obs second leg
    commits only the leading (well-observed) prefix, not the whole grounded run."""
    sc = scene(
        inst(1, "table", n_obs=3, centroid=(3.0, 2.0, 0.0)),
        inst(2, "sofa", n_obs=1, centroid=(6.0, 3.0, 0.0)),
    )
    head = InstructionHead(plan=instruction_plan([_goto("table"), _goto("sofa")]))
    io = _DriveIO(SyntheticScene(0))
    head.advance(io, sc)
    # both legs PLANNED (geometry present)...
    assert head._legs[0].geom is not None
    assert head._legs[1].geom is not None
    # ...but only the first (n_obs=3) leg is COMMITTED.
    assert head._committable_prefix_len() == 1
    assert head._follower is not None


def test_forced_assembly_hook_exception_does_not_strand_route():
    """A broken forced_assembly hook must not strand the route: treated as not-forced,
    same as an unconfigured hook (the strict n_obs >= 2 floor still applies)."""
    def _boom() -> bool:
        raise RuntimeError("broken signal")

    sc = scene(inst(1, "sofa", n_obs=2, centroid=(2.0, 2.0, 0.0)))
    head = InstructionHead(
        plan=instruction_plan([_goto("sofa")]),
        forced_assembly=_boom,
    )
    io = _DriveIO(SyntheticScene(0))
    head.advance(io, sc)
    assert head._follower is not None  # n_obs == 2 already clears the floor regardless


def test_first_anchor_pt_reported():
    sc, idx = _if_scene()
    head = InstructionHead(plan=instruction_plan([_goto("sofa")]))
    io = _DriveIO(sc)
    head.advance(io, idx)
    pt = head.first_anchor_pt()
    assert pt is not None and len(pt) == 2


# --------------------------------------------------------------------------- driving


def _run(head: InstructionHead, io: _DriveIO, idx, max_ticks=4000):
    traj = [io.pose]
    for _ in range(max_ticks):
        head.advance(io, idx)
        io.tick_motion()
        traj.append(io.pose)
        if head.terminal_waypoint() is not None:
            tw = head.terminal_waypoint()
            if ((io.pose[0] - tw.x) ** 2 + (io.pose[1] - tw.y) ** 2) ** 0.5 < 0.3:
                break
    return np.array(traj)


@pytest.mark.slow
def test_drives_and_reaches_terminal():
    sc, idx = _if_scene()
    head = InstructionHead(plan=instruction_plan([_goto("sofa")]))
    io = _DriveIO(sc)
    traj = _run(head, io, idx)
    tw = head.terminal_waypoint()
    assert tw is not None
    assert io.waypoints  # published a stream
    d = ((traj[-1][0] - tw.x) ** 2 + (traj[-1][1] - tw.y) ** 2) ** 0.5
    assert d < 0.3


@pytest.mark.slow
def test_corridor_route_threads_gate_and_respects_avoid():
    sc, idx = _if_scene()
    head = InstructionHead(
        plan=instruction_plan(
            [_corridor("table", "chair"), _goto("sofa")],
            avoid=[AvoidSpec(near=Anchor(noun="vase"))],
        )
    )
    io = _DriveIO(sc, start=(0.7, 3.0))
    traj = _run(head, io, idx)
    gate = corridor_gate(idx.by_label("table")[0], idx.by_label("chair")[0])
    cap = avoid_capsule(AvoidSpec(near=Anchor(noun="vase")), idx)
    assert threading_check(traj, gate)[0] is True
    assert capsule_violated(traj, cap)[0] is False


def test_planner_seams_default_to_planner_module_constants():
    """InstructionHead's calibration seams (nav.unknown_cost_mult / nav.pinch_disc_m /
    nav.pinch_corridor_half_w_m) default to the live planner.py constants — an
    unconfigured head reproduces today's behaviour exactly."""
    from core.nav import planner as _planner

    head = InstructionHead(plan=instruction_plan([_goto("sofa")]))
    assert head.unknown_cost_mult == _planner.UNKNOWN_COST_MULT
    assert head.pinch_disc_m == _planner.PINCH_DISC_M
    assert head.pinch_corridor_half_w_m == _planner.PINCH_CORRIDOR_HALF_W_M


def test_corridor_route_threads_gate_with_overridden_planner_seams():
    """Overriding the injected planner constants (still at their default VALUES) must
    thread the gate identically to the unconfigured head — the seam is plumbed end to
    end through _build_route/plan_through without changing behaviour."""
    from core.nav import planner as _planner

    sc, idx = _if_scene()
    head = InstructionHead(
        plan=instruction_plan([_corridor("table", "chair"), _goto("sofa")]),
        unknown_cost_mult=_planner.UNKNOWN_COST_MULT,
        pinch_disc_m=_planner.PINCH_DISC_M,
        pinch_corridor_half_w_m=_planner.PINCH_CORRIDOR_HALF_W_M,
    )
    io = _DriveIO(sc, start=(0.7, 3.0))
    traj = _run(head, io, idx)
    gate = corridor_gate(idx.by_label("table")[0], idx.by_label("chair")[0])
    assert threading_check(traj, gate)[0] is True


def test_avoid_capsule_stamped_once():
    sc, idx = _if_scene()
    head = InstructionHead(
        plan=instruction_plan([_goto("sofa")], avoid=[AvoidSpec(near=Anchor(noun="vase"))])
    )
    io = _DriveIO(sc)
    head.advance(io, idx)  # builds route + stamps
    cm1 = head._costmap
    head.advance(io, idx)  # second tick must not rebuild
    assert head._costmap is cm1
    assert cm1.capsule_blocked.any()  # a capsule was stamped


def test_unresolvable_avoid_anchor_withholds_first_commit():
    """Issue #156: an avoid anchor with zero grounding instances (the "unicorn" never
    detected) must keep the WHOLE route uncommitted (IF-F5) — driving now would cut
    through a forbidden corridor that can't yet be stamped into the costmap — even
    though the main leg itself is well-grounded. Neither release hook (budget_frac,
    forced_assembly) is wired here, i.e. the live-run default: this is the dead-guard
    bug from #156 and FAILS without the fix."""
    sc = scene(inst(1, "sofa", n_obs=3, centroid=(2.0, 2.0, 0.0)))
    head = InstructionHead(
        plan=instruction_plan(
            [_goto("sofa")], avoid=[AvoidSpec(near=Anchor(noun="unicorn"))]
        )
    )
    io = _DriveIO(SyntheticScene(0))
    head.advance(io, sc)
    assert head._legs[0].geom is not None  # main leg IS grounded (PLANNED)
    assert head._follower is None  # route stays UNCOMMITTED: avoid anchor unresolved
    assert head.terminal_waypoint() is None


def test_unresolvable_avoid_anchor_commits_under_forced_assembly():
    """Once the T-90 forced-assembly gate is reached, IF-F5's withhold no longer applies
    (never strand a route forever) — the route commits with whatever avoid capsules
    resolved, same override used by the issue #33 single-obs floor."""
    sc = scene(inst(1, "sofa", n_obs=3, centroid=(2.0, 2.0, 0.0)))
    head = InstructionHead(
        plan=instruction_plan(
            [_goto("sofa")], avoid=[AvoidSpec(near=Anchor(noun="unicorn"))]
        ),
        forced_assembly=lambda: True,
    )
    io = _DriveIO(SyntheticScene(0))
    head.advance(io, sc)
    assert head._follower is not None  # forced-assembly pressure commits it anyway


def test_resolvable_avoid_anchors_still_commit_and_stamp():
    """Issue #156 regression guard: an avoid anchor that DOES ground must be completely
    unaffected by the IF-F5 fix — the route commits normally and its capsule is stamped
    into the costmap, same as before."""
    sc, idx = _if_scene()
    head = InstructionHead(
        plan=instruction_plan([_goto("sofa")], avoid=[AvoidSpec(near=Anchor(noun="vase"))])
    )
    io = _DriveIO(sc)
    head.advance(io, idx)
    assert head._follower is not None
    assert head._costmap.capsule_blocked.any()  # avoid capsule was stamped


def test_via_near_leg_drives():
    sc, idx = _if_scene()
    head = InstructionHead(plan=instruction_plan([_via("table"), _goto("sofa")]))
    io = _DriveIO(sc)
    head.advance(io, idx)
    assert head._follower is not None  # route built with a via leg


@pytest.mark.slow
def test_arrival_confirmation_checkpoint_fires():
    sc, idx = _if_scene()
    seen = []
    head = InstructionHead(
        plan=instruction_plan([_goto("sofa")]),
        anchor_confirm=lambda plan, i, summary: seen.append(i) or True,
    )
    io = _DriveIO(sc)
    _run(head, io, idx)
    assert 0 in seen  # arrival at leg 0 confirmed the anchor
