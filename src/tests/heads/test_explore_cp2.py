"""CP2 detector-miss recovery seam on ExploreHead + the provisional-instance gate.

Stubs match the seam signature ``run(noun, raw, tiles) -> MissRecoveryOutcome`` (duck-typed
on ``.action`` in {"provisional","absent"}, ``.confidence``, ``.n_obs``, ``.score``); the
real CP2 builder is not called.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from core.heads.explore_step import ExploreHead, _ProvisionalInstance
from core.heads.numerical import NumericalHead
from core.interfaces import OdomState, WaypointCmd
from core.mocks.synthetic_scene import SyntheticScene
from core.perception.scene_index import BasicSceneIndex
from tests.heads._helpers import inst, numerical_plan, object_plan, scene


@dataclass
class _MR:
    action: str
    tile: int | None = 0
    bbox_hint: list | None = field(default=None)
    confidence: float = 0.8
    n_obs: int = 1
    score: float = 0.4


class _ExploreIO:
    def __init__(self, sc, start=(2.5, 2.5)):
        self._sc = sc
        self._x, self._y, self._t = start[0], start[1], 0.0
        self.waypoints = []

    def latest_terrain(self, extended=False):
        return self._sc.terrain_patch(extended=extended, t=self._t)

    def latest_odom(self):
        return OdomState(t=self._t, x=self._x, y=self._y, z=0.0, yaw=0.0)

    def publish_waypoint(self, wp):
        self.waypoints.append(wp)

    def advance_time(self, dt):
        self._t += dt


def _scene_without(noun_present="chair"):
    sc = SyntheticScene(0)
    sc.populate_default(2)
    return sc, BasicSceneIndex(sc.instances())


# --------------------------------------------------------------------------- trigger
def test_cp2_fires_when_noun_missing_past_coverage():
    sc, idx = _scene_without()
    calls = {"n": 0}

    def recoverer(noun, raw, tiles):
        calls["n"] += 1
        return _MR("provisional", confidence=0.8)

    head = ExploreHead(
        plan=object_plan("unicorn"),  # unicorn absent from the scene
        miss_recoverer=recoverer,
        budget_frac=lambda: 0.7,  # past the 0.60 coverage gate
    )
    head.advance(_ExploreIO(sc), idx)
    assert calls["n"] == 1
    assert isinstance(head.provisional, _ProvisionalInstance)
    assert head.provisional_xy is not None


def test_cp2_does_not_fire_before_coverage():
    sc, idx = _scene_without()
    calls = {"n": 0}
    head = ExploreHead(
        plan=object_plan("unicorn"),
        miss_recoverer=lambda n, r, t: calls.__setitem__("n", calls["n"] + 1) or _MR("absent"),
        budget_frac=lambda: 0.3,  # below the gate
    )
    head.advance(_ExploreIO(sc), idx)
    assert calls["n"] == 0


def test_cp2_does_not_fire_when_noun_present():
    sc, idx = _scene_without()
    calls = {"n": 0}
    head = ExploreHead(
        plan=object_plan("sofa"),  # sofa IS present in populate_default
        miss_recoverer=lambda n, r, t: calls.__setitem__("n", calls["n"] + 1) or _MR("absent"),
        budget_frac=lambda: 0.9,
    )
    head.advance(_ExploreIO(sc), idx)
    assert calls["n"] == 0


def test_cp2_fires_once_per_question():
    sc, idx = _scene_without()
    calls = {"n": 0}

    def recoverer(noun, raw, tiles):
        calls["n"] += 1
        return _MR("absent")

    head = ExploreHead(
        plan=object_plan("unicorn"), miss_recoverer=recoverer, budget_frac=lambda: 0.8
    )
    io = _ExploreIO(sc)
    for _ in range(5):
        head.advance(io, idx)
        io.advance_time(1.0)
    assert calls["n"] == 1  # head guard: attempted once


def test_cp2_absent_outcome_no_provisional():
    sc, idx = _scene_without()
    head = ExploreHead(
        plan=object_plan("unicorn"),
        miss_recoverer=lambda n, r, t: _MR("absent"),
        budget_frac=lambda: 0.8,
    )
    head.advance(_ExploreIO(sc), idx)
    assert head.provisional is None


def test_cp2_provisional_biases_navigation():
    sc, idx = _scene_without()
    head = ExploreHead(
        plan=object_plan("unicorn"),
        miss_recoverer=lambda n, r, t: _MR("provisional", confidence=0.8),
        budget_frac=lambda: 0.8,
    )
    io = _ExploreIO(sc)
    head.advance(io, idx)
    assert io.waypoints  # a waypoint toward the provisional was published
    last = io.waypoints[-1]
    assert (last.x, last.y) == head.provisional_xy


def test_cp2_custom_fuse_hint_used():
    sc, idx = _scene_without()
    marker = object()

    def fuse(noun, xy, outcome):
        return marker

    head = ExploreHead(
        plan=object_plan("unicorn"),
        miss_recoverer=lambda n, r, t: _MR("provisional"),
        fuse_hint=fuse,
        budget_frac=lambda: 0.8,
    )
    head.advance(_ExploreIO(sc), idx)
    assert head.provisional is marker


def test_cp2_exception_falls_through():
    sc, idx = _scene_without()

    def boom(n, r, t):
        raise RuntimeError("dark checkpoint")

    head = ExploreHead(
        plan=object_plan("unicorn"), miss_recoverer=boom, budget_frac=lambda: 0.8
    )
    head.advance(_ExploreIO(sc), idx)  # must not raise
    assert head.provisional is None


def test_cp2_default_off_without_budget_frac():
    sc, idx = _scene_without()
    calls = {"n": 0}
    head = ExploreHead(
        plan=object_plan("unicorn"),
        miss_recoverer=lambda n, r, t: calls.__setitem__("n", calls["n"] + 1) or _MR("absent"),
    )  # no budget_frac -> frac 0.0 -> never fires
    head.advance(_ExploreIO(sc), idx)
    assert calls["n"] == 0


# --------------------------------------------------------------------------- provisional gate
def test_provisional_instance_has_single_observation():
    p = _ProvisionalInstance(noun="unicorn", xy=(1.0, 2.0), score=0.4)
    assert p.n_obs == 1  # can NEVER satisfy the >=3-obs early-answer gate on its own


def test_provisional_never_satisfies_early_answer_gate():
    # A provisional (n_obs=1) instance can NEVER inflate a numerical answer (design §CP2
    # "Risk note"). The safety property now holds by EXCLUSION: answer-time observation
    # gating drops single-observation instances from the count entirely, so a provisional
    # ghost cannot on its own become a counted object. We verify the property directly —
    # adding the ghost leaves the count (and its min-contributor floor) unchanged versus a
    # scene without it.
    confident_only = scene(inst(1, "chair", n_obs=5, centroid=(0, 0, 0)))
    with_ghost = scene(
        inst(1, "chair", n_obs=5, centroid=(0, 0, 0)),
        inst(2, "chair", n_obs=1, centroid=(3, 0, 0)),  # provisional-like single-obs ghost
    )

    def _settle(sc):
        head = NumericalHead(plan=numerical_plan("chair"))
        for _ in range(5):  # let the count stabilise
            head.advance(sc)
        return head

    base = _settle(confident_only)
    ghosted = _settle(with_ghost)

    # The ghost is excluded: the counted answer is identical with and without it, and no
    # single-obs contributor is ever counted (min contributor stays >= 3).
    assert ghosted.answer().value == base.answer().value
    assert ghosted.signal().min_contrib_n_obs >= 3
