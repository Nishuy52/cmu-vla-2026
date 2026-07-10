"""CP5 frontier-selection seam on ExploreHead: scene-scale trigger + seam choice/fallback.

Stubs match the head seam ``frontier_selector(question, frontiers) -> FrontierOutcome``
(duck-typed on ``.action`` in {"choice","fallback"} and ``.index``); the real CP5 builder
is not called.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.heads.explore_step import (
    EXPLORED_AREA_TRIGGER_M2,
    ExploreHead,
    _explored_area_m2,
    _explored_regions,
)
from core.nav.exploration import ExplorationDecision, ExplorationStatus
from core.nav.frontiers import Frontier
from core.nav.occupancy import FREE, UNKNOWN, OccupancyGrid
from core.interfaces import WaypointCmd
from tests.heads._helpers import object_plan


@dataclass
class _FO:
    action: str
    choice: int | None = None
    index: int | None = None
    reason: str = ""


def _grid_two_regions() -> OccupancyGrid:
    """Two FREE blobs separated by a band of UNKNOWN -> 2 disconnected regions."""
    g = OccupancyGrid()
    state = np.full((5, 12), UNKNOWN, dtype=np.int8)
    state[1:4, 1:4] = FREE  # region A
    state[1:4, 8:11] = FREE  # region B (separated by UNKNOWN columns 4-7)
    g.state = state
    g.intensity = np.full(state.shape, -np.inf, dtype=np.float32)
    g.observed = np.zeros(state.shape, dtype=bool)
    return g


def _grid_one_region() -> OccupancyGrid:
    g = OccupancyGrid()
    state = np.full((5, 6), UNKNOWN, dtype=np.int8)
    state[1:4, 1:5] = FREE  # single blob
    g.state = state
    g.intensity = np.full(state.shape, -np.inf, dtype=np.float32)
    g.observed = np.zeros(state.shape, dtype=bool)
    return g


def _fake_frontiers(n=5):
    return [
        Frontier(row=i, col=i, xy=(float(i), 0.0), size=10, path_distance=1.0,
                 affinity=0.0, score=float(10 - i))
        for i in range(n)
    ]


# --------------------------------------------------------------------------- helpers
def test_explored_regions_counts_disconnected_blobs():
    assert _explored_regions(_grid_two_regions()) == 2
    assert _explored_regions(_grid_one_region()) == 1


def test_explored_area_m2():
    g = _grid_one_region()  # 3x4 = 12 FREE cells @ 0.10 m -> 12 * 0.01 = 0.12 m^2
    assert abs(_explored_area_m2(g) - 0.12) < 1e-9


# --------------------------------------------------------------------------- trigger
def test_multi_room_trigger_on_two_regions():
    head = ExploreHead(plan=object_plan("chair"), frontier_selector=lambda q, f: _FO("fallback"))
    head.grid = _grid_two_regions()
    assert head._scene_is_multi_room((0.0, 0.0)) is True


def test_no_trigger_on_single_small_region():
    head = ExploreHead(plan=object_plan("chair"), frontier_selector=lambda q, f: _FO("fallback"))
    head.grid = _grid_one_region()
    assert head._scene_is_multi_room((0.0, 0.0)) is False


def test_area_trigger_fires():
    head = ExploreHead(
        plan=object_plan("chair"),
        frontier_selector=lambda q, f: _FO("fallback"),
        explored_area_trigger_m2=0.05,  # tiny threshold -> single region area trips it
    )
    head.grid = _grid_one_region()
    assert head._scene_is_multi_room((0.0, 0.0)) is True


# --------------------------------------------------------------------------- seam selection
def _frontier_decision():
    return ExplorationDecision(ExplorationStatus.FRONTIER, waypoint=WaypointCmd(0.0, 0.0))


def test_cp5_choice_used(monkeypatch):
    import core.heads.explore_step as mod

    frontiers = _fake_frontiers()
    monkeypatch.setattr(mod, "detect_frontiers", lambda g, p, a=None: frontiers)
    head = ExploreHead(
        plan=object_plan("chair"),
        frontier_selector=lambda q, f: _FO("choice", choice=3, index=2),
    )
    head.grid = _grid_two_regions()
    wp = head._maybe_cp5_frontier((0.0, 0.0), _frontier_decision())
    assert wp is not None
    assert (wp.x, wp.y) == frontiers[2].xy  # the seam's 0-indexed pick


def test_cp5_fallback_defers_to_geometric(monkeypatch):
    import core.heads.explore_step as mod

    monkeypatch.setattr(mod, "detect_frontiers", lambda g, p, a=None: _fake_frontiers())
    head = ExploreHead(
        plan=object_plan("chair"), frontier_selector=lambda q, f: _FO("fallback")
    )
    head.grid = _grid_two_regions()
    assert head._maybe_cp5_frontier((0.0, 0.0), _frontier_decision()) is None


def test_cp5_not_triggered_when_single_room(monkeypatch):
    import core.heads.explore_step as mod

    monkeypatch.setattr(mod, "detect_frontiers", lambda g, p, a=None: _fake_frontiers())
    head = ExploreHead(
        plan=object_plan("chair"),
        frontier_selector=lambda q, f: _FO("choice", index=0),
    )
    head.grid = _grid_one_region()
    assert head._maybe_cp5_frontier((0.0, 0.0), _frontier_decision()) is None


def test_cp5_no_seam_returns_none():
    head = ExploreHead(plan=object_plan("chair"))  # frontier_selector None
    head.grid = _grid_two_regions()
    assert head._maybe_cp5_frontier((0.0, 0.0), _frontier_decision()) is None


def test_cp5_only_top5_frontiers_offered(monkeypatch):
    import core.heads.explore_step as mod

    frontiers = _fake_frontiers(8)
    monkeypatch.setattr(mod, "detect_frontiers", lambda g, p, a=None: frontiers)
    seen = {}

    def selector(q, f):
        seen["n"] = len(f)
        return _FO("choice", index=0)

    head = ExploreHead(plan=object_plan("chair"), frontier_selector=selector)
    head.grid = _grid_two_regions()
    head._maybe_cp5_frontier((0.0, 0.0), _frontier_decision())
    assert seen["n"] == 5  # top-5 only


def test_cp5_out_of_range_index_falls_back(monkeypatch):
    import core.heads.explore_step as mod

    monkeypatch.setattr(mod, "detect_frontiers", lambda g, p, a=None: _fake_frontiers())
    head = ExploreHead(
        plan=object_plan("chair"),
        frontier_selector=lambda q, f: _FO("choice", index=99),
    )
    head.grid = _grid_two_regions()
    assert head._maybe_cp5_frontier((0.0, 0.0), _frontier_decision()) is None


def test_cp5_exception_falls_back(monkeypatch):
    import core.heads.explore_step as mod

    monkeypatch.setattr(mod, "detect_frontiers", lambda g, p, a=None: _fake_frontiers())

    def boom(q, f):
        raise RuntimeError("dark checkpoint")

    head = ExploreHead(plan=object_plan("chair"), frontier_selector=boom)
    head.grid = _grid_two_regions()
    assert head._maybe_cp5_frontier((0.0, 0.0), _frontier_decision()) is None
