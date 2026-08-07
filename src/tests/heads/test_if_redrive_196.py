"""Issue #196: the #183 re-drive cannot fire on the slots it was built for.

Root cause (#103 diagnosis, 8 Aug): a leg with ``grounded=False`` gives
``ungrounded_subgoals() > 0``, so ``_route_covers_plan()`` is False, so the redrive
gate (the OLD ``self._route_covers_plan()`` check in ``InstructionHead._drive``) never
opens -- 14 of 16 watchdog-parked slots in the archived diagnosis fit this shape. The
gate excluded its own use case: the slots with unreached legs are exactly the slots
whose legs are ungrounded.

The fix relaxes the REDRIVE gate specifically (``_redrive_worth_attempting``,
``drive_complete``'s own ``_route_covers_plan`` check is untouched) to fire whenever
at least one leg is COMMITTABLE (resolved past the #33 MIN_COMMIT_OBS floor -- this
is what admits a leg that is "grounded but below MIN_GROUND_OBS") and the route is
not already fully visited. ``_redrive`` itself now caps the rebuilt prefix at
``_committable_prefix_len()`` instead of forcing every plan leg, so a leg with no
geometry at all stays open while the resolvable legs get a bounded extra pass.
"""
from __future__ import annotations

import pytest

from core.heads.instruction import (
    MAX_REDRIVE_PASSES,
    MIN_COMMIT_OBS,
    MIN_GROUND_OBS,
    InstructionHead,
)
from core.mocks.synthetic_scene import Room, SyntheticScene
from core.perception.scene_index import BasicSceneIndex
from core.plan_schema import Anchor, LegKind, RouteLeg
from tests.heads._helpers import instruction_plan
from tests.heads.test_if_redrive_183 import _DeadFollower, _DriveIO, _stall_the_follower


def _goto(noun: str) -> RouteLeg:
    return RouteLeg(kind=LegKind.GOTO, anchors=[Anchor(noun=noun)])


def _two_leg_scene() -> SyntheticScene:
    sc = SyntheticScene(0)
    sc.rooms = [Room(0.0, 0.0, 10.0, 6.0)]
    sc.place_box("lamp", 2.0, 3.0, 0.4, 0.4, 0.5)
    sc.place_box("shelf", 8.0, 3.0, 0.4, 0.4, 0.5)
    return sc


def _index_with_n_obs(sc: SyntheticScene, **overrides: int) -> BasicSceneIndex:
    """A scene index identical to ``sc.instances()`` except the given labels' n_obs
    are overridden -- lets a test park a leg strictly between MIN_COMMIT_OBS and
    MIN_GROUND_OBS (resolvable/committable, but not fully grounded)."""
    recs = sc.instances()
    for r in recs:
        if r.label in overrides:
            r.n_obs = overrides[r.label]
    return BasicSceneIndex(recs)


@pytest.mark.slow
def test_196_redrive_fires_when_one_leg_is_committable_but_below_min_ground_obs():
    """#196: 'shelf' sits at MIN_COMMIT_OBS (committable -- has geometry, drivable)
    but below MIN_GROUND_OBS (ungrounded). Before the fix, this alone kept
    ``_route_covers_plan()`` False forever, so the redrive gate never opened even
    though the shelf leg is perfectly resolvable and reachable."""
    assert MIN_COMMIT_OBS < MIN_GROUND_OBS  # the gap this test exercises
    sc = _two_leg_scene()
    idx = _index_with_n_obs(sc, shelf=MIN_COMMIT_OBS)
    head = InstructionHead(plan=instruction_plan([_goto("lamp"), _goto("shelf")]))
    io = _DriveIO(sc)
    head.advance(io, idx)

    # The shelf leg resolved and committed, but sits below the grounded floor.
    assert head._legs[1].geom is not None
    assert head._legs[1].grounded is False
    assert head.ungrounded_subgoals() == 1
    assert head._route_covers_plan() is False  # the OLD gate: still closed

    # Yet the whole route (both legs) is already committable and driven.
    assert head._committable_prefix_len() == 2
    assert head._driven_prefix == 2
    assert head._redrive_worth_attempting() is True  # the NEW gate: open

    _stall_the_follower(head)
    assert head.all_legs_visited() is False
    published = head._drive(io, io.pose, 0.0)
    assert published is True
    assert head._redrives == 1, "the #196-relaxed gate did not fire a redrive"
    assert head.redrive_events(), "no redrive event recorded"
    assert not isinstance(head._follower, _DeadFollower)
    assert head._follower is not None


@pytest.mark.slow
def test_196_redrive_spends_budget_on_the_reachable_leg_while_the_unresolved_one_stays_open():
    """Three legs: 'lamp' reached, 'shelf' committable-but-ungrounded (reachable and
    unvisited), 'ghost' never resolves at all (no instance of that label exists in
    the scene). The relaxed gate must still fire a redrive that rebuilds the route
    over the RESOLVABLE prefix (lamp, shelf) -- it must not wait on 'ghost', and must
    not blow a redrive pass trying to force a route through it."""
    sc = _two_leg_scene()
    idx = _index_with_n_obs(sc, shelf=MIN_COMMIT_OBS)
    head = InstructionHead(
        plan=instruction_plan([_goto("lamp"), _goto("shelf"), _goto("ghost")])
    )
    io = _DriveIO(sc)
    head.advance(io, idx)

    assert head._legs[2].geom is None  # 'ghost' never resolved -- the true open leg
    assert head._committable_prefix_len() == 2  # lamp + shelf only
    assert head._redrive_worth_attempting() is True
    assert head.all_legs_visited() is False

    _stall_the_follower(head)
    fired = head._redrive("test_forced", idx)
    assert fired is True
    # The rebuilt route covers exactly the resolvable prefix -- 'ghost' stays open,
    # never forced into a route with no geometry for it.
    assert head._driven_prefix == 2
    assert head._legs[2].geom is None


def test_196_redrive_gate_stays_open_exactly_as_before_when_the_full_route_grounds():
    """The relaxed gate must not regress the ORIGINAL #183 case: once every leg is
    fully grounded and the driven route already covers the whole plan,
    ``_redrive_worth_attempting()`` agrees with the old ``_route_covers_plan()``."""
    sc = _two_leg_scene()
    idx = BasicSceneIndex(sc.instances())  # default n_obs=3 -- fully grounded
    head = InstructionHead(plan=instruction_plan([_goto("lamp"), _goto("shelf")]))
    io = _DriveIO(sc)
    head.advance(io, idx)
    assert head._route_covers_plan() is True
    assert head._redrive_worth_attempting() is True


def test_196_all_visited_route_still_exits_immediately():
    """#183/#196 invariant preserved: a route whose legs are ALL individually
    confirmed reached is never redrive-eligible, whatever the grounding state --
    an all-visited run must still end exactly as before."""
    sc = _two_leg_scene()
    idx = _index_with_n_obs(sc, shelf=MIN_COMMIT_OBS)  # shelf still ungrounded
    head = InstructionHead(plan=instruction_plan([_goto("lamp"), _goto("shelf")]))
    io = _DriveIO(sc)
    head.advance(io, idx)
    head._confirmed = {0, 1}  # both legs individually reached
    assert head.all_legs_visited() is True
    assert head._redrive_worth_attempting() is False


def test_196_redrive_worth_attempting_stays_closed_with_nothing_committable():
    """The true UNGROUNDED-PARK case (no leg has resolved ANY candidate) must still
    keep the gate closed -- there is nothing resolvable to spend a redrive pass on."""
    head = InstructionHead(plan=instruction_plan([_goto("nonexistent")]))
    sc = _two_leg_scene()
    idx = BasicSceneIndex(sc.instances())
    head.advance(_DriveIO(sc), idx)
    assert head._committable_prefix_len() == 0
    assert head._redrive_worth_attempting() is False


@pytest.mark.slow
def test_196_redrive_budget_stays_bounded_when_one_leg_never_resolves():
    """Even with the relaxed gate permanently open (one leg never resolves), the
    redrive pass count is still bounded by MAX_REDRIVE_PASSES -- the watchdog
    remains the real backstop, unchanged by #196."""
    sc = _two_leg_scene()
    idx = _index_with_n_obs(sc, shelf=MIN_COMMIT_OBS)
    head = InstructionHead(
        plan=instruction_plan([_goto("lamp"), _goto("shelf"), _goto("ghost")])
    )
    io = _DriveIO(sc)
    head.advance(io, idx)
    _stall_the_follower(head)

    for _ in range(MAX_REDRIVE_PASSES):
        head._follower = _DeadFollower()
        fired = head._redrive("test_forced", idx)
        assert fired is True
    assert head._can_redrive() is False
    head._follower = _DeadFollower()
    assert head._redrive("test_forced", idx) is False
    assert head._redrives == MAX_REDRIVE_PASSES
