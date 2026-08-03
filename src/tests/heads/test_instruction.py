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
    the COMMITTED route while no forced-assembly pressure has been injected.

    Issue #159: the withhold blocks the DRIVE (no follower is built) but not the
    ANSWER — ``terminal_waypoint()`` still banks the grounded leg's own goal via the
    grounded-prefix fallback, since the geometry is already resolved regardless of the
    commit decision. Pre-#159 this asserted ``is None``; that was the exact defect.
    """
    sc = scene(inst(1, "sofa", n_obs=1, centroid=(2.0, 2.0, 0.0)))
    head = InstructionHead(plan=instruction_plan([_goto("sofa")]))
    io = _DriveIO(SyntheticScene(0))
    head.advance(io, sc)
    assert head._legs[0].geom is not None  # PLANNED: geometry already resolved
    assert head._follower is None  # NOT COMMITTED: no route built to drive
    tw = head.terminal_waypoint()
    assert tw is not None  # #159: grounded-prefix fallback banks the PLANNED leg's goal
    goal = head._legs[0].geom
    assert (tw.x, tw.y) == goal


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
    bug from #156 and FAILS without the fix.

    Issue #159 reconciliation: the withhold above (uncommitted route, no follower) is
    UNCHANGED — those two assertions still pin the #156 fix byte-for-byte. But
    withholding the DRIVE must not become withholding the ANSWER: verify/forced-
    assembly reading ``terminal_waypoint()`` must still get the grounded prefix's goal
    (the main leg banked), not the generic floor. Pre-#159 the third assertion here was
    ``is None`` — that was the exact "verify yielded nothing" defect #159 fixes.
    """
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
    tw = head.terminal_waypoint()
    assert tw is not None  # #159: prefix goal banked despite the avoid-withhold
    assert (tw.x, tw.y) == head._legs[0].geom


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


# --------------------------------------------------------------------------- issue #159
#
# "verify yielded nothing" — terminal_waypoint() must always bank the grounded
# PREFIX rather than fall through to None whenever at least one leg has grounded,
# regardless of why the route as a whole is not (yet) committed/driven. See the
# reconciliation comments on ``_maybe_build_or_extend_route`` (the DRIVE-side
# withhold, unchanged) and ``terminal_waypoint``/``_grounded_prefix_goal`` (the
# ANSWER-side fallback, new) in core/heads/instruction.py.


def test_prefix_answer_before_final_leg_grounds():
    """(a) A 3-leg route with legs 1-2 grounded and leg 3 still ungrounded (anchor
    never observed) yields leg-2's goal at verify time, not None. This is the plain
    H3c partial-route-drive path (already committed today) — pinned here as the #159
    baseline case the grounded-prefix fallback must never regress."""
    sc = scene(
        inst(1, "table", n_obs=3, centroid=(3.0, 2.0, 0.0)),
        inst(2, "chair", n_obs=3, centroid=(3.0, 4.0, 0.0)),
        # no "sofa" instance: leg 3's anchor never grounds
    )
    head = InstructionHead(
        plan=instruction_plan([_goto("table"), _goto("chair"), _goto("sofa")])
    )
    io = _DriveIO(SyntheticScene(0))
    head.advance(io, sc)
    assert head._legs[0].geom is not None
    assert head._legs[1].geom is not None
    assert head._legs[2].geom is None  # leg 3 never grounds
    assert head._committable_prefix_len() == 2
    assert head._follower is not None  # H3c: partial prefix already committed
    # The committed route's driving target is leg-2's goal (not leg-1's, not None) —
    # terminal_waypoint() itself streams breadcrumbs toward it (H11/IF-F4) rather than
    # the raw coordinate until the vehicle is within reach, so pin the actual target.
    assert head._terminal_xy == head._legs[1].geom
    tw = head.terminal_waypoint()
    assert tw is not None


def test_prefix_answer_survives_unresolvable_avoid_withhold():
    """(b) An unresolvable avoid anchor with no budget/forced-assembly pressure keeps
    the WHOLE route uncommitted (#156, unchanged — see
    test_unresolvable_avoid_anchor_withholds_first_commit for the full withhold pin),
    but terminal_waypoint() still returns the grounded-prefix goal via the new #159
    fallback rather than None, using ``_grounded_prefix_goal`` directly to pin the
    exact mechanism (not just the end-to-end observation)."""
    sc = scene(inst(1, "sofa", n_obs=3, centroid=(2.0, 2.0, 0.0)))
    head = InstructionHead(
        plan=instruction_plan(
            [_goto("sofa")], avoid=[AvoidSpec(near=Anchor(noun="unicorn"))]
        )
    )
    io = _DriveIO(SyntheticScene(0))
    head.advance(io, sc)
    assert head._follower is None  # #156: still uncommitted, unchanged
    assert head._grounded_prefix_goal() == head._legs[0].geom
    tw = head.terminal_waypoint()
    assert tw is not None
    assert (tw.x, tw.y) == head._legs[0].geom


def test_prefix_answer_none_when_nothing_grounded():
    """(c) When NOTHING has grounded at all — no leg has resolved geometry — the
    grounded-prefix fallback correctly returns None too: the floor genuinely is the
    best available answer, and the head must not fabricate a waypoint."""
    sc = scene()  # empty index: nothing can ground
    head = InstructionHead(
        plan=instruction_plan([_goto("unicorn")], avoid=[AvoidSpec(near=Anchor(noun="dragon"))])
    )
    io = _DriveIO(SyntheticScene(0))
    head.advance(io, sc)
    assert head._grounded_prefix_len() == 0
    assert head._grounded_prefix_goal() is None
    assert head._follower is None
    assert head.terminal_waypoint() is None
