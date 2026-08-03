"""Stage 3 of docs/proposals/pre_grounding_movement_plan.md (issue #77) — the
goal-credibility withhold gate (Gate 3 in ``_committable_prefix_len``) and the guarded
re-ground improvement rebuild (``_maybe_reground_rebuild``).

Covers plan tests (a), (c), (d): (b) lives in test_if_explore_ungrounded.py (the explore
fall-through family) and (e) is the unmodified pre-existing suite (byte-preservation).
"""
from __future__ import annotations

import numpy as np
import pytest

from core.geometry.toolbox import corridor_gate, threading_check
from core.heads.instruction import (
    ARRIVAL_TOL_M,
    GOAL_CLAMP_CREDIBILITY_BAR_M,
    MAX_REGROUND_REBUILDS,
    InstructionHead,
)
from core.interfaces import OdomState, WaypointCmd
from core.mocks.synthetic_scene import Room, SyntheticScene
from core.perception.scene_index import BasicSceneIndex
from core.plan_schema import Anchor, LegKind, RouteLeg
from tests.heads._helpers import instruction_plan


def _goto(noun: str) -> RouteLeg:
    return RouteLeg(kind=LegKind.GOTO, anchors=[Anchor(noun=noun)])


def _corridor(a: str, b: str) -> RouteLeg:
    return RouteLeg(kind=LegKind.CORRIDOR_BETWEEN, anchors=[Anchor(noun=a), Anchor(noun=b)])


class _IO:
    def __init__(self, sc: SyntheticScene, start=(1.0, 1.0)):
        self._sc = sc
        self._x, self._y = start
        self._t = 0.0
        self.waypoints: list[WaypointCmd] = []

    def latest_terrain(self, extended: bool = False):
        return self._sc.terrain_patch(extended=extended, t=self._t)

    def latest_odom(self):
        return OdomState(t=self._t, x=self._x, y=self._y, z=0.0, yaw=0.0)

    def publish_waypoint(self, wp: WaypointCmd):
        self.waypoints.append(wp)

    def advance_time(self, dt: float = 1.0) -> None:
        self._t += dt


def _sealed_pocket_scene() -> tuple[SyntheticScene, BasicSceneIndex]:
    """A 16x16 room with a corner pocket sealed except for a hairline sliver gap, and
    an anchor deep in the far corner of the pocket — the sliver's nearest-reachable
    snap lands ~5 m from the anchor centroid (>> ARRIVAL_TOL_M), an uncreditable clamp.
    """
    sc = SyntheticScene(0)
    sc.rooms = [Room(0.0, 0.0, 16.0, 16.0)]
    sc.place_box("wall", 12.85, 9.7, 6.3, 0.3, 1.0)  # seals pocket floor, x:[9.7,16]
    sc.place_box("wall", 9.7, 12.85, 0.3, 6.3, 1.0)  # seals pocket left wall, y:[9.7,16]
    sc.place_box("cabinet", 14.5, 14.5, 0.4, 0.4, 0.8)
    return sc, BasicSceneIndex(sc.instances())


def _clamped_leg(inst: InstructionHead) -> float:
    leg = inst._legs[0]
    assert leg.geom is not None
    assert leg.goal_clamp_m is not None
    return leg.goal_clamp_m


# --------------------------------------------------------------------------- (a)


def test_sealed_pocket_produces_uncreditable_clamp():
    """Sanity: the sealed-pocket fixture actually produces a clamp beyond the
    credibility bar (the fixture, not yet the gate, under test)."""
    sc, idx = _sealed_pocket_scene()
    inst = InstructionHead(plan=instruction_plan([_goto("cabinet")]))
    io = _IO(sc)
    inst.advance(io, idx)
    clamp = _clamped_leg(inst)
    assert clamp > GOAL_CLAMP_CREDIBILITY_BAR_M
    assert GOAL_CLAMP_CREDIBILITY_BAR_M == ARRIVAL_TOL_M  # decision 3: parameter-free


def test_clamped_leg_withheld_while_budget_available():
    """Gate 3: a GOTO leg whose resolved goal clamps beyond the credibility bar is
    PLANNED (geometry present) but withheld from the COMMITTED route while budget
    pressure has not forced the commit and forced-assembly has not been reached.

    Issue #159: the commit withhold above is unchanged (still uncommitted, no
    follower) — but ``terminal_waypoint()`` still banks the PLANNED leg's own
    (clamped) goal via the grounded-prefix fallback rather than returning None:
    withholding the DRIVE must not become withholding the ANSWER.
    """
    sc, idx = _sealed_pocket_scene()
    forced = {"v": False}
    inst = InstructionHead(
        plan=instruction_plan([_goto("cabinet")]),
        budget_frac=lambda: 0.10,  # early: below PROVISIONAL_COMMIT_FRAC
        forced_assembly=lambda: forced["v"],
    )
    io = _IO(sc)
    inst.advance(io, idx)

    assert _clamped_leg(inst) > GOAL_CLAMP_CREDIBILITY_BAR_M
    assert inst._committable_prefix_len() == 0
    assert inst._follower is None, "an uncreditable clamp must not be committed"
    tw = inst.terminal_waypoint()
    assert tw is not None  # #159: grounded-prefix fallback banks the clamped leg's goal
    assert (tw.x, tw.y) == inst._legs[0].geom


def test_clamped_leg_commits_once_forced_assembly_reached():
    """The same withheld leg commits once the T-90 forced-assembly gate fires —
    identical override pattern to #33 (worst case: today's behaviour, delayed)."""
    sc, idx = _sealed_pocket_scene()
    forced = {"v": False}
    inst = InstructionHead(
        plan=instruction_plan([_goto("cabinet")]),
        budget_frac=lambda: 0.10,
        forced_assembly=lambda: forced["v"],
    )
    io = _IO(sc)
    inst.advance(io, idx)
    assert inst._follower is None  # withheld first

    forced["v"] = True
    io.advance_time()
    inst.advance(io, idx)
    assert inst._follower is not None, "forced assembly must commit the clamped leg"
    assert inst._driven_prefix == 1
    assert inst.terminal_waypoint() is not None


def test_clamped_leg_commits_under_budget_pressure_alone():
    """Budget pressure crossing PROVISIONAL_COMMIT_FRAC also forces the commit, without
    needing forced_assembly (mirrors H4c's own override)."""
    sc, idx = _sealed_pocket_scene()
    inst = InstructionHead(
        plan=instruction_plan([_goto("cabinet")]),
        budget_frac=lambda: 0.99,  # late: over PROVISIONAL_COMMIT_FRAC
    )
    io = _IO(sc)
    inst.advance(io, idx)
    assert inst._follower is not None
    assert inst._driven_prefix == 1


def test_clamped_leg_commits_immediately_with_no_hooks_byte_preserved():
    """No injected hooks (both None) => Gate 3 never withholds — byte-identical to
    pre-Stage-3 behaviour, matching Gate 1 (#33) and Gate 2 (H4c)'s own defaults."""
    sc, idx = _sealed_pocket_scene()
    inst = InstructionHead(plan=instruction_plan([_goto("cabinet")]))  # no hooks
    io = _IO(sc)
    inst.advance(io, idx)
    assert _clamped_leg(inst) > GOAL_CLAMP_CREDIBILITY_BAR_M
    assert inst._follower is not None
    assert inst._driven_prefix == 1


# --------------------------------------------------------------------------- (c)


def _simple_committed_head() -> tuple[InstructionHead, _IO, BasicSceneIndex]:
    sc = SyntheticScene(0)
    sc.rooms = [Room(0.0, 0.0, 10.0, 6.0)]
    sc.place_box("sofa", 8.0, 3.0, 0.5, 0.5, 0.5)
    idx = BasicSceneIndex(sc.instances())
    inst = InstructionHead(plan=instruction_plan([_goto("sofa")]))
    io = _IO(sc, start=(1.0, 3.0))
    inst.advance(io, idx)
    assert inst._follower is not None
    return inst, io, idx


def test_reground_rebuild_fires_on_material_improvement_and_refreshes_snapshot():
    """A materially better goal_clamp_m than the committed snapshot triggers a guarded
    rebuild that succeeds (real, reachable geometry): a NEW follower is adopted and the
    committed snapshot is refreshed to the improved value."""
    inst, io, idx = _simple_committed_head()
    old_follower = inst._follower
    real_clamp = inst._legs[0].goal_clamp_m
    assert real_clamp is not None

    # Simulate "the snapshot at build time was worse" (the fixture's real geometry is
    # already reachable/clean — we control only the TRIGGER side, exercising the real
    # _stamp_ground_plan/plan_through adoption path with genuine geometry).
    inst._committed_clamp_m[0] = real_clamp + GOAL_CLAMP_CREDIBILITY_BAR_M + 1.0

    inst._maybe_reground_rebuild(idx)

    assert inst._reground_rebuilds == 1
    assert inst._follower is not old_follower, "a genuinely improved rebuild must adopt"
    assert inst._committed_clamp_m[0] == pytest.approx(real_clamp)


def test_reground_rebuild_capped_at_max_reground_rebuilds():
    """Repeated material 'improvement' signals stop firing once MAX_REGROUND_REBUILDS is
    spent — the cap is separate from MAX_REPLANS_PER_QUESTION."""
    inst, io, idx = _simple_committed_head()
    real_clamp = inst._legs[0].goal_clamp_m
    assert real_clamp is not None

    for _ in range(MAX_REGROUND_REBUILDS):
        inst._committed_clamp_m[0] = real_clamp + GOAL_CLAMP_CREDIBILITY_BAR_M + 1.0
        inst._maybe_reground_rebuild(idx)
    assert inst._reground_rebuilds == MAX_REGROUND_REBUILDS

    follower_at_cap = inst._follower
    inst._committed_clamp_m[0] = real_clamp + GOAL_CLAMP_CREDIBILITY_BAR_M + 1.0
    inst._maybe_reground_rebuild(idx)
    assert inst._reground_rebuilds == MAX_REGROUND_REBUILDS, "cap must hold"
    assert inst._follower is follower_at_cap, "no further rebuild once capped"


def test_reground_rebuild_never_fires_on_a_confirmed_leg():
    """A leg already in ``_confirmed`` (arrived) must never be reground-rebuilt, even
    with an inflated stale snapshot -- mirrors _mark_arrivals' own never-rewrite rule."""
    sc = SyntheticScene(0)
    sc.rooms = [Room(0.0, 0.0, 10.0, 6.0)]
    sc.place_box("table", 3.0, 3.0, 0.4, 0.4, 0.5)
    sc.place_box("sofa", 8.0, 3.0, 0.5, 0.5, 0.5)
    idx = BasicSceneIndex(sc.instances())
    inst = InstructionHead(plan=instruction_plan([_goto("table"), _goto("sofa")]))
    io = _IO(sc, start=(1.0, 3.0))
    inst.advance(io, idx)
    assert inst._follower is not None
    assert inst._driven_prefix == 2

    # Fake leg 0 as already arrived/confirmed (the state _mark_arrivals would set).
    inst._confirmed.add(0)
    inst._leg_progress = 1
    real_clamp0 = inst._legs[0].goal_clamp_m
    assert real_clamp0 is not None
    inst._committed_clamp_m[0] = real_clamp0 + GOAL_CLAMP_CREDIBILITY_BAR_M + 1.0
    old_follower = inst._follower

    inst._maybe_reground_rebuild(idx)

    assert inst._reground_rebuilds == 0, "a confirmed leg must never trigger a rebuild"
    assert inst._follower is old_follower


def test_reground_rebuild_never_downgrades_to_recovery(monkeypatch):
    """If the guarded rebuild's plan_through cannot re-thread the prefix, adoption is
    declined and the OLD (already-working) follower is kept -- never swapped for a
    _recover_path beeline (the whole point of the guard)."""
    inst, io, idx = _simple_committed_head()
    old_follower = inst._follower
    old_driven_prefix = inst._driven_prefix
    real_clamp = inst._legs[0].goal_clamp_m
    assert real_clamp is not None
    inst._committed_clamp_m[0] = real_clamp + GOAL_CLAMP_CREDIBILITY_BAR_M + 1.0

    monkeypatch.setattr("core.heads.instruction.plan_through", lambda *a, **k: (None, []))

    inst._maybe_reground_rebuild(idx)

    assert inst._reground_rebuilds == 1, "the attempt still counts against the cap"
    assert inst._follower is old_follower, "a failed guarded rebuild must not strand the drive"
    assert inst._driven_prefix == old_driven_prefix


# --------------------------------------------------------------------------- (d)


def _corridor_then_goto_scene() -> tuple[SyntheticScene, BasicSceneIndex]:
    sc = SyntheticScene(0)
    sc.rooms = [Room(0.0, 0.0, 8.0, 6.0)]
    sc.place_box("table", 3.0, 2.0, 0.4, 0.4, 0.5)
    sc.place_box("chair", 3.0, 4.0, 0.4, 0.4, 0.5)
    sc.place_box("sofa", 6.0, 3.0, 0.5, 0.5, 0.5)
    return sc, BasicSceneIndex(sc.instances())


def test_reground_rebuild_after_corridor_never_costs_earned_threading(monkeypatch):
    """A GOTO leg immediately after a CORRIDOR_BETWEEN leg: if the full-prefix replan the
    improvement trigger would attempt fails to re-thread the gate, the guarded rebuild
    declines and the OLD, already-threaded follower is kept (#77c risk class) — the
    corridor leg's own earned threading is never traded away for an unrelated goto
    improvement."""
    sc, idx = _corridor_then_goto_scene()
    inst = InstructionHead(plan=instruction_plan([_corridor("table", "chair"), _goto("sofa")]))
    io = _IO(sc, start=(0.7, 3.0))
    inst.advance(io, idx)
    assert inst._follower is not None
    assert inst._driven_prefix == 2

    gate = corridor_gate(idx.by_label("table")[0], idx.by_label("chair")[0])
    old_follower = inst._follower
    old_path = np.asarray(old_follower.path, dtype=float)
    assert threading_check(old_path, gate)[0] is True

    real_clamp = inst._legs[1].goal_clamp_m  # the goto("sofa") leg, index 1
    assert real_clamp is not None
    inst._committed_clamp_m[1] = real_clamp + GOAL_CLAMP_CREDIBILITY_BAR_M + 1.0

    # Simulate the corridor-threading rejection: the guarded rebuild's full-prefix
    # plan_through cannot re-thread the gate for the "improved" candidate.
    monkeypatch.setattr("core.heads.instruction.plan_through", lambda *a, **k: (None, []))

    inst._maybe_reground_rebuild(idx)

    assert inst._follower is old_follower, "old (threaded) follower must be kept"
    new_path = np.asarray(inst._follower.path, dtype=float)
    assert threading_check(new_path, gate)[0] is True
