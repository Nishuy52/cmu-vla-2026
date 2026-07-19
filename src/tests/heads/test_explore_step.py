"""ExploreHead: orientation sweep, frontier pursuit, affinity injection, IF delegation."""
from __future__ import annotations

from core.heads.explore_step import ExploreHead, _plan_nouns, uniform_affinity
from core.heads.instruction import InstructionHead
from core.interfaces import OdomState, WaypointCmd
from core.mocks.synthetic_scene import SyntheticScene
from core.perception.scene_index import BasicSceneIndex
from core.nav.exploration import ExplorationStatus
from core.plan_schema import Anchor, Clause, LegKind, Pred, RouteLeg
from tests.heads._helpers import inst, instruction_plan, numerical_plan, object_plan, scene


class _ExploreIO:
    def __init__(self, sc: SyntheticScene, start=(2.5, 2.5)):
        self._sc = sc
        self._x, self._y, self._t = start[0], start[1], 0.0
        self.waypoints: list[WaypointCmd] = []

    def latest_terrain(self, extended: bool = False):
        return self._sc.terrain_patch(extended=extended, t=self._t)

    def latest_odom(self):
        return OdomState(t=self._t, x=self._x, y=self._y, z=0.0, yaw=0.0)

    def publish_waypoint(self, wp: WaypointCmd):
        self.waypoints.append(wp)

    def advance_time(self, dt):
        self._t += dt


def test_sweep_first():
    sc = SyntheticScene(0)
    sc.populate_default(3)
    head = ExploreHead(plan=numerical_plan("chair"))
    io = _ExploreIO(sc)
    head.advance(io, BasicSceneIndex(sc.instances()))
    assert head.last_status is ExplorationStatus.SWEEPING
    assert io.waypoints  # the sweep emits a diamond waypoint


def test_frontier_after_sweep_window():
    sc = SyntheticScene(0)
    sc.populate_default(3)
    head = ExploreHead(plan=numerical_plan("chair"))
    io = _ExploreIO(sc)
    head.advance(io, BasicSceneIndex(sc.instances()))  # anchor sweep clock at t=0
    io.advance_time(70.0)  # past the 60 s sweep window
    head.advance(io, BasicSceneIndex(sc.instances()))
    assert head.last_status in (ExplorationStatus.FRONTIER, ExplorationStatus.COMPLETE)


def test_affinity_factory_invoked_with_plan_nouns():
    sc = SyntheticScene(0)
    sc.populate_default(2)
    captured = {}

    def factory(nouns):
        captured["nouns"] = list(nouns)
        return lambda xy: 1.0

    head = ExploreHead(plan=object_plan("chair"), affinity_fn=factory)
    io = _ExploreIO(sc)
    head.advance(io, BasicSceneIndex(sc.instances()))
    assert captured["nouns"] == ["chair"]


# --------------------------------------------------------------- issue #43b affinity
def test_object_ref_affinity_stays_on_target_when_eligible():
    """Target noun has an answer-eligible instance -> no anchor-seeking bias."""
    sc = scene(inst(1, "table", n_obs=3, score=0.9, centroid=(1.0, 1.0, 0.0)))
    captured = {}

    def factory(nouns):
        captured["nouns"] = list(nouns)
        return lambda xy: 1.0

    plan = object_plan("table", clauses=[Clause(pred=Pred.NEAR, anchors=[Anchor(noun="lamp")])])
    head = ExploreHead(plan=plan, affinity_fn=factory)
    io = _ExploreIO(SyntheticScene(0))
    head.advance(io, sc)
    assert captured["nouns"][0] == "table"


def test_object_ref_affinity_seeks_anchor_when_target_absent():
    """Target noun absent from the scene entirely -> bias toward the anchor noun."""
    sc = scene(inst(1, "lamp", n_obs=3, score=0.9, centroid=(1.0, 1.0, 0.0)))
    captured = {}

    def factory(nouns):
        captured["nouns"] = list(nouns)
        return lambda xy: 1.0

    plan = object_plan("teapot", clauses=[Clause(pred=Pred.NEAR, anchors=[Anchor(noun="lamp")])])
    head = ExploreHead(plan=plan, affinity_fn=factory)
    io = _ExploreIO(SyntheticScene(0))
    head.advance(io, sc)
    assert captured["nouns"][0] == "lamp"
    assert "teapot" in captured["nouns"]  # target kept as a fallback tail


def test_object_ref_affinity_seeks_anchor_when_target_ineligible():
    """Target present but under the answer-eligibility bar (low n_obs / low score) is
    treated as starved just like an absent target — bias toward the anchor."""
    sc = scene(
        inst(1, "teapot", n_obs=1, score=0.9, centroid=(1.0, 1.0, 0.0)),  # n_obs < 2
        inst(2, "lamp", n_obs=3, score=0.9, centroid=(2.0, 2.0, 0.0)),
    )
    captured = {}

    def factory(nouns):
        captured["nouns"] = list(nouns)
        return lambda xy: 1.0

    plan = object_plan("teapot", clauses=[Clause(pred=Pred.NEAR, anchors=[Anchor(noun="lamp")])])
    head = ExploreHead(plan=plan, affinity_fn=factory)
    io = _ExploreIO(SyntheticScene(0))
    head.advance(io, sc)
    assert captured["nouns"][0] == "lamp"


def test_object_ref_affinity_no_anchor_falls_back_to_target():
    """No anchor noun on the plan (bare 'the teapot') -> unchanged target-noun affinity,
    even though the target is starved (nothing to seek toward instead)."""
    sc = scene()
    captured = {}

    def factory(nouns):
        captured["nouns"] = list(nouns)
        return lambda xy: 1.0

    head = ExploreHead(plan=object_plan("teapot"), affinity_fn=factory)
    io = _ExploreIO(SyntheticScene(0))
    head.advance(io, sc)
    assert captured["nouns"] == ["teapot"]


def test_if_delegates_to_instruction_head():
    sc = SyntheticScene(0)
    sc.place_box("sofa", 2.5, 2.5, 0.4, 0.4, 0.5)
    idx = BasicSceneIndex(sc.instances())
    inst_head = InstructionHead(
        plan=instruction_plan([RouteLeg(kind=LegKind.GOTO, anchors=[Anchor(noun="sofa")])])
    )
    head = ExploreHead(
        plan=instruction_plan([RouteLeg(kind=LegKind.GOTO, anchors=[Anchor(noun="sofa")])]),
        instruction=inst_head,
    )
    io = _ExploreIO(sc, start=(0.7, 2.5))
    head.advance(io, idx)
    # delegation drove the IF head: it grounded the leg (no separate sweep waypoint owner)
    assert inst_head.ungrounded_subgoals() == 0


# --------------------------------------------------------------- issue #43c IF affinity
def test_if_affinity_focuses_first_ungrounded_leg():
    """First leg's anchor has no answer-eligible instance -> bias toward it, ahead of a
    later leg whose anchor IS already eligible."""
    sc = scene(inst(1, "sofa", n_obs=3, score=0.9, centroid=(2.0, 2.0, 0.0)))
    captured = {}

    def factory(nouns):
        captured["nouns"] = list(nouns)
        return lambda xy: 1.0

    plan = instruction_plan(
        [
            RouteLeg(kind=LegKind.GOTO, anchors=[Anchor(noun="lamp")]),
            RouteLeg(kind=LegKind.GOTO, anchors=[Anchor(noun="sofa")]),
        ]
    )
    head = ExploreHead(plan=plan, affinity_fn=factory)
    io = _ExploreIO(SyntheticScene(0))
    head.advance(io, sc)
    assert captured["nouns"][0] == "lamp"


def test_if_affinity_advances_when_leg_grounds():
    """Once the first leg's anchor becomes answer-eligible, focus advances to the next
    still-ungrounded leg's anchor noun."""
    sc = scene(
        inst(1, "lamp", n_obs=3, score=0.9, centroid=(1.0, 1.0, 0.0)),
        inst(2, "sofa", n_obs=1, score=0.9, centroid=(2.0, 2.0, 0.0)),  # n_obs < 2
    )
    captured = {}

    def factory(nouns):
        captured["nouns"] = list(nouns)
        return lambda xy: 1.0

    plan = instruction_plan(
        [
            RouteLeg(kind=LegKind.GOTO, anchors=[Anchor(noun="lamp")]),
            RouteLeg(kind=LegKind.GOTO, anchors=[Anchor(noun="sofa")]),
        ]
    )
    head = ExploreHead(plan=plan, affinity_fn=factory)
    io = _ExploreIO(SyntheticScene(0))
    head.advance(io, sc)
    assert captured["nouns"][0] == "sofa"


def test_if_affinity_no_injection_when_all_legs_grounded():
    """Every leg's anchor already answer-eligible -> no bias, plan nouns unchanged."""
    sc = scene(
        inst(1, "lamp", n_obs=3, score=0.9, centroid=(1.0, 1.0, 0.0)),
        inst(2, "sofa", n_obs=3, score=0.9, centroid=(2.0, 2.0, 0.0)),
    )
    captured = {}

    def factory(nouns):
        captured["nouns"] = list(nouns)
        return lambda xy: 1.0

    plan = instruction_plan(
        [
            RouteLeg(kind=LegKind.GOTO, anchors=[Anchor(noun="lamp")]),
            RouteLeg(kind=LegKind.GOTO, anchors=[Anchor(noun="sofa")]),
        ]
    )
    head = ExploreHead(plan=plan, affinity_fn=factory)
    io = _ExploreIO(SyntheticScene(0))
    head.advance(io, sc)
    assert captured["nouns"] == ["lamp", "sofa"]


def test_plan_nouns_extraction():
    from core.plan_schema import Clause, Pred, TargetSpec

    p = object_plan("table")
    p.target = TargetSpec(noun="table", clauses=[Clause(pred=Pred.NEAR, anchors=[Anchor(noun="lamp")])])
    assert set(_plan_nouns(p)) == {"table", "lamp"}


def test_uniform_affinity_is_zero():
    aff = uniform_affinity(["chair"])
    assert aff((1.0, 2.0)) == 0.0
