"""Hand-built instance / plan helpers for the heads unit tests."""
from __future__ import annotations

import numpy as np

from core.interfaces import InstanceRecord, QType
from core.perception.scene_index import BasicSceneIndex
from core.plan_schema import (
    Anchor,
    AvoidSpec,
    Clause,
    LegKind,
    Plan,
    Pred,
    RouteLeg,
    TargetSpec,
)


def inst(
    instance_id: int,
    label: str,
    *,
    n_obs: int = 3,
    centroid=(0.0, 0.0, 0.0),
    extent=(0.5, 0.5, 0.5),
    score: float = 0.9,
    aliases: tuple[str, ...] = (),
) -> InstanceRecord:
    """InstanceRecord with a symmetric AABB centred on ``centroid``."""
    c = np.array(centroid, dtype=float)
    half = np.array(extent, dtype=float) / 2.0
    return InstanceRecord(
        instance_id=instance_id,
        label=label,
        score=score,
        n_obs=n_obs,
        centroid=c,
        aabb_min=c - half,
        aabb_max=c + half,
        aliases=aliases,
    )


def scene(*records: InstanceRecord) -> BasicSceneIndex:
    return BasicSceneIndex(list(records))


def numerical_plan(noun: str, clauses: list[Clause] | None = None) -> Plan:
    return Plan(
        qtype=QType.NUMERICAL,
        question_raw=f"how many {noun}",
        target=TargetSpec(noun=noun, clauses=clauses or []),
    )


def object_plan(noun: str, clauses: list[Clause] | None = None) -> Plan:
    return Plan(
        qtype=QType.OBJECT_REFERENCE,
        question_raw=f"the {noun}",
        target=TargetSpec(noun=noun, clauses=clauses or []),
    )


def instruction_plan(route: list[RouteLeg], avoid: list[AvoidSpec] | None = None) -> Plan:
    return Plan(
        qtype=QType.INSTRUCTION_FOLLOWING,
        question_raw="go",
        route=route,
        avoid=avoid or [],
    )


def near_clause(noun: str) -> Clause:
    return Clause(pred=Pred.NEAR, anchors=[Anchor(noun=noun)])


def closest_clause(noun: str) -> Clause:
    return Clause(pred=Pred.CLOSEST_TO, anchors=[Anchor(noun=noun)])
