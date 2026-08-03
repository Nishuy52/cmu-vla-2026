"""Exploration policy: orientation sweep, frontier pursuit, EXPLORATION_COMPLETE."""
from __future__ import annotations

import pytest

import core.nav.exploration as exploration_module
from core.interfaces import QType
from core.nav.exploration import (
    NUMERICAL_W_SIZE_MULT,
    ExplorationPolicy,
    ExplorationStatus,
    sweep_waypoints,
)
from core.nav.frontiers import W_SIZE
from core.nav.occupancy import OccupancyGrid
from tests.nav.helpers import patch_from_ascii


def _free_block_grid():
    grid = OccupancyGrid(cell_m=0.1)
    grid.integrate_patch(patch_from_ascii(["." * 8 for _ in range(8)], cell_m=0.1))
    return grid


def test_sweep_is_four_point_diamond():
    wps = sweep_waypoints((1.0, 1.0), side_m=1.0)
    assert len(wps) == 4
    # Vertices at +/- side/2 along each axis.
    xs = sorted(w.x for w in wps)
    ys = sorted(w.y for w in wps)
    assert abs(xs[0] - 0.5) < 1e-6 and abs(xs[-1] - 1.5) < 1e-6
    assert abs(ys[0] - 0.5) < 1e-6 and abs(ys[-1] - 1.5) < 1e-6


def test_policy_sweeps_first():
    grid = _free_block_grid()
    pol = ExplorationPolicy(start_xy=(0.4, 0.4), sweep_s=60.0)
    dec = pol.step(grid, vehicle_xy=(0.4, 0.4), t=0.0)
    assert dec.status is ExplorationStatus.SWEEPING
    assert dec.waypoint is not None


def test_policy_advances_through_sweep_vertices():
    grid = _free_block_grid()
    pol = ExplorationPolicy(start_xy=(0.4, 0.4), sweep_s=60.0, sweep_side_m=1.0)
    # First vertex.
    d0 = pol.step(grid, (0.4, 0.4), t=0.0)
    first = (d0.waypoint.x, d0.waypoint.y)
    # Arrive at the first vertex -> policy should offer a different vertex.
    d1 = pol.step(grid, first, t=1.0)
    assert (d1.waypoint.x, d1.waypoint.y) != first


def test_policy_switches_to_frontier_after_sweep():
    grid = _free_block_grid()
    pol = ExplorationPolicy(start_xy=(0.4, 0.4), sweep_s=60.0, t0=0.0)
    dec = pol.step(grid, vehicle_xy=(0.4, 0.4), t=120.0)  # past sweep window
    assert dec.status is ExplorationStatus.FRONTIER
    assert dec.frontier is not None
    assert dec.waypoint is not None


def test_force_frontier_skips_sweep():
    grid = _free_block_grid()
    pol = ExplorationPolicy(start_xy=(0.4, 0.4), sweep_s=60.0)
    dec = pol.step(grid, (0.4, 0.4), t=0.0, budget_state={"force_frontier": True})
    assert dec.status is ExplorationStatus.FRONTIER


def test_exploration_complete_when_no_frontiers():
    # Fully enclosed FREE room -> no frontiers -> COMPLETE after sweep.
    grid = OccupancyGrid(cell_m=0.1)
    rows = [
        "#####",
        "#...#",
        "#...#",
        "#...#",
        "#####",
    ]
    grid.integrate_patch(patch_from_ascii(rows, cell_m=0.1))
    pol = ExplorationPolicy(start_xy=(0.25, 0.25), sweep_s=60.0, t0=0.0)
    dec = pol.step(grid, vehicle_xy=(0.25, 0.25), t=120.0)
    assert dec.status is ExplorationStatus.COMPLETE
    assert dec.waypoint is None


def test_min_frontier_score_gate():
    grid = _free_block_grid()
    # An absurdly high score bar -> nothing qualifies -> COMPLETE.
    pol = ExplorationPolicy(start_xy=(0.4, 0.4), sweep_s=0.0, min_frontier_score=1e9)
    dec = pol.step(grid, vehicle_xy=(0.4, 0.4), t=1.0)
    assert dec.status is ExplorationStatus.COMPLETE


# --------------------------------------------------------------------------- #150: NUMERICAL
# coverage weighting. For counting, UNSEEN AREA is the objective (#150/#103): NUMERICAL
# scores frontiers with a boosted size term so the policy favours a bigger unexplored
# pocket over one that only wins on distance/affinity. These tests pin the WIRING (the
# right w_size reaches detect_frontiers for the right qtype) rather than hand-building a
# specific room geometry, since the tradeoff itself lives in nav.frontiers' scoring
# formula (unowned here) and is exercised by that module's own tests.


def _spy_on_detect_frontiers(monkeypatch):
    """Patch exploration_module.detect_frontiers to record the w_size it was called
    with, while still delegating to the real implementation so `.step()` behaves
    normally."""
    captured: dict[str, float | None] = {"w_size": "NOT_CALLED"}
    real = exploration_module.detect_frontiers

    def spy(grid, vehicle_xy, affinity, **kwargs):
        captured["w_size"] = kwargs.get("w_size")
        return real(grid, vehicle_xy, affinity, **kwargs)

    monkeypatch.setattr(exploration_module, "detect_frontiers", spy)
    return captured


def test_numerical_boosts_frontier_size_weight(monkeypatch):
    grid = _free_block_grid()
    captured = _spy_on_detect_frontiers(monkeypatch)
    pol = ExplorationPolicy(start_xy=(0.4, 0.4), sweep_s=0.0, qtype=QType.NUMERICAL)
    dec = pol.step(grid, vehicle_xy=(0.4, 0.4), t=1.0)
    assert dec.status is ExplorationStatus.FRONTIER  # sanity: the spy didn't break scoring
    assert captured["w_size"] == pytest.approx(NUMERICAL_W_SIZE_MULT * W_SIZE)
    assert NUMERICAL_W_SIZE_MULT > 1.0  # it must actually be a boost


@pytest.mark.parametrize("qtype", [None, QType.OBJECT_REFERENCE, QType.INSTRUCTION_FOLLOWING])
def test_non_numerical_uses_default_frontier_size_weight(monkeypatch, qtype):
    grid = _free_block_grid()
    captured = _spy_on_detect_frontiers(monkeypatch)
    pol = ExplorationPolicy(start_xy=(0.4, 0.4), sweep_s=0.0, qtype=qtype)
    pol.step(grid, vehicle_xy=(0.4, 0.4), t=1.0)
    assert captured["w_size"] == pytest.approx(W_SIZE)
