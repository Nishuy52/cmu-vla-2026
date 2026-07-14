"""IF explore-while-ungrounded + provisional terminal + distinct pair anchors.

Covers the hardening-backlog items H3 (IF-F1/SYS-F3 deadlock), H4c (IF-F3 provisional
terminal grounding + defensible salience), and IF-F6 (distinct "the two X" pair anchors).

The acceptance test IS the fix's definition (the SYS-F3 deadlock repro, inverted): an IF
question over an empty scene index must still publish waypoints within N ticks — the robot
explores to ground the anchors instead of parking.
"""
from __future__ import annotations

from core.heads.explore_step import ExploreHead
from core.heads.instruction import InstructionHead
from core.interfaces import OdomState, WaypointCmd
from core.mocks.synthetic_scene import Room, SyntheticScene
from core.perception.scene_index import BasicSceneIndex
from core.plan_schema import Anchor, LegKind, RouteLeg
from tests.heads._helpers import instruction_plan


def _goto(noun: str) -> RouteLeg:
    return RouteLeg(kind=LegKind.GOTO, anchors=[Anchor(noun=noun)])


def _corridor(n1: str, n2: str) -> RouteLeg:
    return RouteLeg(kind=LegKind.CORRIDOR_BETWEEN, anchors=[Anchor(noun=n1), Anchor(noun=n2)])


class _ExploreDriveIO:
    """RobotIO: terrain from the scene, kinematic pose chasing the last waypoint."""

    def __init__(self, sc: SyntheticScene, start=(1.0, 3.0), step=0.1):
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
        if self.waypoints:
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


def _big_room() -> SyntheticScene:
    sc = SyntheticScene(0)
    sc.rooms = [Room(0.0, 0.0, 8.0, 6.0)]
    return sc


# --------------------------------------------------------------------------- H3: SYS-F3


def test_if_empty_scene_still_publishes_waypoints_within_n_ticks():
    """The SYS-F3 acceptance test, inverted: IF question + empty scene index => the robot
    explores and publishes waypoints (the deadlock produced 0 in 30 ticks)."""
    sc = _big_room()
    idx = BasicSceneIndex([])  # empty: no anchor can ground from spawn
    route = [_goto("sofa")]
    inst = InstructionHead(plan=instruction_plan(route))
    head = ExploreHead(plan=instruction_plan(route), instruction=inst)
    io = _ExploreDriveIO(sc)

    for _ in range(30):
        head.advance(io, idx)
        io.tick_motion()

    assert io.waypoints, "IF over empty scene must still publish (explore), not deadlock"
    assert inst.ungrounded_subgoals() == 1  # still ungrounded, but the robot moved


def test_if_exploration_is_noun_affinity_biased_toward_ungrounded_noun():
    """The affinity factory must receive the ungrounded route noun first (H3b, arch §4
    row 10) so the frontier scorer biases toward where the missing anchor likely is."""
    sc = _big_room()
    idx = BasicSceneIndex([])
    captured: dict[str, list[str]] = {}

    def factory(nouns):
        captured["nouns"] = list(nouns)
        return lambda _xy: 0.0

    route = [_goto("sofa")]
    inst = InstructionHead(plan=instruction_plan(route))
    head = ExploreHead(plan=instruction_plan(route), instruction=inst, affinity_fn=factory)
    io = _ExploreDriveIO(sc)
    head.advance(io, idx)  # delegates to IF head (no emit), falls through to explore

    assert captured.get("nouns"), "affinity factory was never invoked on the explore path"
    assert captured["nouns"][0] == "sofa"  # the ungrounded noun leads the affinity bias


# --------------------------------------------------------------------------- H3c: prefix


def test_grounded_prefix_drives_while_later_leg_ungrounded():
    """Legs 1-2 grounded, leg 3 not => the grounded prefix route drives (partial credit
    is banked) while exploration continues; grounding leg 3 later extends the route."""
    sc = _big_room()
    sc.place_box("table", 3.0, 2.0, 0.4, 0.4, 0.5)
    sc.place_box("chair", 3.0, 4.0, 0.4, 0.4, 0.5)
    # "sofa" (terminal) initially absent.
    idx = BasicSceneIndex(sc.instances())
    route = [_goto("table"), _goto("chair"), _goto("sofa")]
    inst = InstructionHead(plan=instruction_plan(route))
    head = ExploreHead(plan=instruction_plan(route), instruction=inst)
    io = _ExploreDriveIO(sc)

    head.advance(io, idx)
    assert inst._follower is not None, "grounded prefix (legs 1-2) must build a route now"
    assert inst._driven_prefix == 2, "route should cover exactly the 2 grounded legs"
    assert io.waypoints, "prefix route drives a waypoint stream"

    # The terminal grounds later: the route extends to cover all three legs.
    sc.place_box("sofa", 6.0, 3.0, 0.5, 0.5, 0.5)
    idx2 = BasicSceneIndex(sc.instances())
    head.advance(io, idx2)
    assert inst._driven_prefix == 3, "route extends to the terminal once it grounds"


# --------------------------------------------------------------------------- H4c: provisional


def _relaxable_scene() -> tuple[SyntheticScene, BasicSceneIndex]:
    """A scene where the terminal 'lamp near the bench' resolves only by dropping the
    relation clause (no bench present), i.e. a provisional terminal grounding."""
    sc = _big_room()
    sc.place_box("table", 3.0, 2.0, 0.4, 0.4, 0.5)
    sc.place_box("lamp", 6.0, 3.0, 0.2, 0.2, 0.3)  # a lamp exists, but no bench
    return sc, BasicSceneIndex(sc.instances())


def _lamp_near_bench_leg() -> RouteLeg:
    from core.plan_schema import Clause, Pred

    bench = Anchor(noun="bench")
    lamp = Anchor(
        noun="lamp",
        disambiguator=Clause(pred=Pred.NEAR, anchors=[bench]),
    )
    return RouteLeg(kind=LegKind.GOTO, anchors=[lamp])


def test_provisional_terminal_withheld_while_budget_remains():
    """A relaxation-audited terminal is NOT committed to the route while explore budget
    remains (H4c): the head keeps exploring for the missing disambiguator instead."""
    sc, idx = _relaxable_scene()
    route = [_goto("table"), _lamp_near_bench_leg()]
    inst = InstructionHead(
        plan=instruction_plan(route),
        budget_frac=lambda: 0.10,  # early: pressure has not forced the commit
    )
    head = ExploreHead(plan=instruction_plan(route), instruction=inst)
    io = _ExploreDriveIO(sc)
    head.advance(io, idx)

    # The grounded, non-provisional leg 1 drives; the provisional terminal is withheld.
    assert inst._follower is not None
    assert inst._driven_prefix == 1, "provisional terminal must be withheld under budget"


def test_provisional_terminal_commits_under_budget_pressure():
    """Once budget pressure crosses the commit fraction, even a provisional terminal is
    committed (H4c) — banking a possibly-wrong terminal beats publishing nothing."""
    sc, idx = _relaxable_scene()
    route = [_goto("table"), _lamp_near_bench_leg()]
    inst = InstructionHead(
        plan=instruction_plan(route),
        budget_frac=lambda: 0.99,  # late: pressure forces the commit
    )
    head = ExploreHead(plan=instruction_plan(route), instruction=inst)
    io = _ExploreDriveIO(sc)
    head.advance(io, idx)

    assert inst._driven_prefix == 2, "budget pressure commits the provisional terminal"


def test_provisional_terminal_commits_when_no_budget_signal():
    """With no budget_frac injected there is no pressure to withhold, so a provisional
    terminal commits immediately (preserves today's single-tick behaviour)."""
    sc, idx = _relaxable_scene()
    route = [_goto("table"), _lamp_near_bench_leg()]
    inst = InstructionHead(plan=instruction_plan(route))  # budget_frac None
    head = ExploreHead(plan=instruction_plan(route), instruction=inst)
    io = _ExploreDriveIO(sc)
    head.advance(io, idx)

    assert inst._driven_prefix == 2


# --------------------------------------------------------------------------- IF-F6: pairs


def test_pair_anchor_two_columns_resolve_distinct_nonzero_gate():
    """'between the two columns' with 2 column instances => a nonzero-width gate (the two
    anchors resolve to DISTINCT instances), not a zero-width collapse to recovery."""
    from core.geometry.toolbox import corridor_gate

    sc = _big_room()
    sc.place_box("column", 2.0, 3.0, 0.4, 0.4, 2.0)
    sc.place_box("column", 5.0, 3.0, 0.4, 0.4, 2.0)
    sc.place_box("sofa", 7.0, 3.0, 0.5, 0.5, 0.5)
    idx = BasicSceneIndex(sc.instances())
    # regex _split_pair duplicates the anchor: two Anchor(noun="column").
    route = [_corridor("column", "column"), _goto("sofa")]
    inst = InstructionHead(plan=instruction_plan(route))
    io = _ExploreDriveIO(sc, start=(0.7, 3.0))
    inst.advance(io, idx)

    corridor = inst._legs[0]
    assert corridor.geom is not None, "distinct columns must ground the corridor leg"
    p0, p1 = corridor.geom
    width = ((p0[0] - p1[0]) ** 2 + (p0[1] - p1[1]) ** 2) ** 0.5
    assert width > 0.5, f"gate width collapsed to {width:.3f} (duplicate-instance bug)"


def test_pair_anchor_single_instance_stays_ungrounded_no_recovery():
    """'between the two columns' with only 1 column => the corridor leg stays ungrounded
    (fewer than 2 distinct instances) and feeds the H3 explore path — NOT a recovery
    beeline through a zero-width gate."""
    sc = _big_room()
    sc.place_box("column", 3.0, 3.0, 0.4, 0.4, 2.0)  # only one
    sc.place_box("sofa", 7.0, 3.0, 0.5, 0.5, 0.5)
    idx = BasicSceneIndex(sc.instances())
    route = [_corridor("column", "column"), _goto("sofa")]
    inst = InstructionHead(plan=instruction_plan(route))
    io = _ExploreDriveIO(sc, start=(0.7, 3.0))
    inst.advance(io, idx)

    corridor = inst._legs[0]
    assert corridor.geom is None, "one instance cannot form a distinct-pair corridor gate"
    assert not corridor.grounded
    # No committed route yet (leg 1 ungrounded) => no recovery beeline was built.
    assert inst._driven_prefix == 0
