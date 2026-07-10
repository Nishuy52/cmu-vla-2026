"""Exploration policy: orientation sweep, frontier pursuit, EXPLORATION_COMPLETE."""
from __future__ import annotations

from core.nav.exploration import (
    ExplorationPolicy,
    ExplorationStatus,
    sweep_waypoints,
)
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
