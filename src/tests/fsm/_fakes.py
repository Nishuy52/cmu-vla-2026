"""Deterministic test doubles for the FSM tests: FakeClock, FakeRobotIO, scene helpers."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from core.interfaces import (
    InstanceRecord,
    IntAnswer,
    MarkerBox,
    MatchTier,
    Question,
    WaypointCmd,
)


class FakeClock:
    """Manually advanced clock; now() returns whatever t was last set."""

    def __init__(self, t: float = 0.0) -> None:
        self.t = float(t)

    def now(self) -> float:
        return self.t

    def advance(self, dt: float) -> float:
        self.t += float(dt)
        return self.t

    def set(self, t: float) -> float:
        self.t = float(t)
        return self.t


class FakeScene:
    """Minimal SceneIndex: substring-tolerant by_label over a fixed instance list."""

    def __init__(self, instances: Sequence[InstanceRecord] = ()) -> None:
        self._inst = list(instances)

    def all_instances(self) -> Sequence[InstanceRecord]:
        return list(self._inst)

    def by_label(self, noun: str) -> Sequence[InstanceRecord]:
        n = (noun or "").lower()
        return [r for r in self._inst if n and (n in r.label.lower() or r.label.lower() in n)]

    def by_label_tiered(self, noun: str) -> list[tuple[InstanceRecord, MatchTier]]:
        """No real tier ladder here; every hit reported at MatchTier.EXACT (see
        tests/geometry/_helpers.FakeIndex.by_label_tiered for the same convention)."""
        return [(r, MatchTier.EXACT) for r in self.by_label(noun)]


class FakeRobotIO:
    """RobotIO double: scripted question, records every published answer."""

    def __init__(self, clock: FakeClock, question: Question | None = None) -> None:
        self._clock = clock
        self._question = question
        self.published_ints: list[IntAnswer] = []
        self.published_markers: list[MarkerBox] = []
        self.published_waypoints: list[WaypointCmd] = []

    # inputs
    def set_question(self, q: Question | None) -> None:
        self._question = q

    def question(self) -> Question | None:
        return self._question

    def latest_pano(self):
        return None

    def latest_scan(self):
        return None

    def latest_terrain(self, extended: bool = False):
        return None

    def latest_odom(self):
        return None

    # outputs
    def publish_waypoint(self, wp: WaypointCmd) -> None:
        self.published_waypoints.append(wp)

    def publish_marker(self, box: MarkerBox) -> None:
        self.published_markers.append(box)

    def publish_int(self, ans: IntAnswer) -> None:
        self.published_ints.append(ans)

    def clock(self) -> FakeClock:
        return self._clock

    # test convenience
    @property
    def publish_count(self) -> int:
        return (
            len(self.published_ints)
            + len(self.published_markers)
            + len(self.published_waypoints)
        )


def make_instance(
    instance_id: int,
    label: str,
    *,
    score: float = 0.9,
    n_obs: int = 3,
    centroid=(0.0, 0.0, 0.0),
    extent=(1.0, 1.0, 1.0),
) -> InstanceRecord:
    """Build an InstanceRecord with a symmetric AABB around centroid."""
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
    )
