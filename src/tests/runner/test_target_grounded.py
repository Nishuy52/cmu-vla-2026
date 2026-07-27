"""Fast unit tests for :func:`core.runner.battery._target_grounded` -- kept separate
from ``test_battery.py`` (module-marked ``slow``, full sim runs) so this stays in the
default fast tier."""
from __future__ import annotations

from core.interfaces import QType
from core.perception.scene_index import BasicSceneIndex
from core.plan_schema import Anchor, Clause, LegKind, Plan, Pred, RouteLeg
from core.runner.battery import _target_grounded
from tests.heads._helpers import inst


def test_target_grounded_recurses_into_route_anchor_disambiguator():
    """A route anchor with a blank own noun but a real noun in its ``disambiguator``
    (issue #95) must still be found groundable -- the ``lamp`` instance resolves via
    the nested anchor, not the blank top-level one."""
    idx = BasicSceneIndex([inst(1, "lamp")])
    plan = Plan(
        qtype=QType.INSTRUCTION_FOLLOWING,
        question_raw="",
        route=[
            RouteLeg(
                kind=LegKind.GOTO,
                anchors=[
                    Anchor(noun="", disambiguator=Clause(pred=Pred.NEAR, anchors=[Anchor(noun="lamp")]))
                ],
            )
        ],
    )
    assert _target_grounded(plan, idx) is True
