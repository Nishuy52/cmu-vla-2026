"""Tests for the shared derived-arrival-tolerance module (issue #70).

Covers: the derived-tolerance formula term-by-term, the "single vehicle-radius
term, not double-counted with inflation" resolution, and the head/scorer
coupling at the documented nominal residual.
"""
from __future__ import annotations

import math

import pytest

from core.groundtruth import arrival as A
from core.groundtruth.arrival import (
    DEFAULT_CELL_M,
    FIXED_SAFETY_MARGIN_M,
    NOMINAL_ARRIVAL_TOL_M,
    NOMINAL_FIT_RESIDUAL_P95_M,
    PASS_BY_NOMINAL_FIT_RESIDUAL_M,
    PASS_BY_NOMINAL_TOL_M,
    derived_arrival_tol_m,
)
from core.nav.costmap import VEHICLE_RADIUS_M


# --------------------------------------------------------------------------- formula terms


def test_derived_tol_sums_exactly_the_four_cited_terms():
    residual = 0.5
    expected = (
        VEHICLE_RADIUS_M
        + DEFAULT_CELL_M * math.sqrt(2.0) / 2.0
        + residual
        + FIXED_SAFETY_MARGIN_M
    )
    assert derived_arrival_tol_m(residual) == pytest.approx(expected)


def test_derived_tol_has_no_free_parameters_beyond_the_margin():
    # Every term besides the margin is either a cited constant or a caller-supplied
    # measurement -- zeroing the residual and overriding the constants to zero must
    # leave exactly the margin behind.
    tol = derived_arrival_tol_m(0.0, vehicle_radius_m=0.0, cell_m=0.0, margin_m=0.03)
    assert tol == pytest.approx(0.03)


def test_derived_tol_vehicle_radius_term_matches_costmap_constant():
    # The vehicle-radius term must be exactly the costmap's own constant -- no
    # independent/duplicated tuning of "vehicle size" inside this module.
    tol = derived_arrival_tol_m(0.0, cell_m=0.0, margin_m=0.0)
    assert tol == pytest.approx(VEHICLE_RADIUS_M)


def test_derived_tol_does_not_double_count_inflation_radius():
    """Issue #70 brief conflict, resolved: the brief asks for vehicle_radius AND a
    separate "costmap inflation" term, but Costmap._inflate's radius IS
    vehicle_radius_m (core/nav/costmap.py) -- one constant, two names. Adding it
    twice would double the vehicle-radius contribution; this module adds it once.
    Regression-guard: the vehicle-radius contribution to the total must equal
    VEHICLE_RADIUS_M exactly, not 2x it.
    """
    residual = 0.0
    tol = derived_arrival_tol_m(residual, cell_m=0.0, margin_m=0.0)
    assert tol == pytest.approx(VEHICLE_RADIUS_M)
    assert tol != pytest.approx(2 * VEHICLE_RADIUS_M)


def test_derived_tol_grid_diagonal_term_is_half_cell_diagonal():
    cell_m = 0.20
    tol = derived_arrival_tol_m(0.0, vehicle_radius_m=0.0, cell_m=cell_m, margin_m=0.0)
    assert tol == pytest.approx(cell_m * math.sqrt(2.0) / 2.0)


def test_derived_tol_grid_diagonal_uses_default_cell_m_when_unspecified():
    # DEFAULT_CELL_M (0.10 m) matches core.nav.occupancy.CELL_M and the battery's
    # synthetic-grid resolution -- verify the module's own default, not a re-import,
    # so a drift between the two stays visible as a failing test, not a silent skew.
    from core.nav.occupancy import CELL_M as OCCUPANCY_CELL_M

    assert DEFAULT_CELL_M == pytest.approx(OCCUPANCY_CELL_M)


def test_derived_tol_residual_term_is_additive_and_caller_supplied():
    base = derived_arrival_tol_m(0.3)
    bumped = derived_arrival_tol_m(0.3 + 0.4)
    assert bumped - base == pytest.approx(0.4)


def test_derived_tol_margin_is_small_relative_to_structural_terms():
    # The margin must not be able to paper over a genuine geometric miss -- it should
    # be a small fraction of the vehicle-radius term, the largest fixed structural term.
    assert FIXED_SAFETY_MARGIN_M < 0.25 * VEHICLE_RADIUS_M


def test_derived_tol_monotone_increasing_in_every_positive_term():
    lo = derived_arrival_tol_m(0.2, vehicle_radius_m=0.3, cell_m=0.05, margin_m=0.01)
    hi = derived_arrival_tol_m(0.9, vehicle_radius_m=0.5, cell_m=0.20, margin_m=0.05)
    assert hi > lo


# --------------------------------------------------------------------------- nominal / coupling


def test_nominal_arrival_tol_is_the_formula_at_the_nominal_residual():
    assert NOMINAL_ARRIVAL_TOL_M == pytest.approx(
        derived_arrival_tol_m(NOMINAL_FIT_RESIDUAL_P95_M)
    )


def test_nominal_fit_residual_is_a_plausible_p95_from_battery_provenance():
    # Sanity bound, not a tautology: the documented nominal must sit inside the
    # actual per-scene residual range recorded in reports/issue59_leg_ceiling.json
    # (0.1953-1.8103 m across the 15 battery scenes) -- guards against a typo'd
    # constant that no longer reflects any real measurement.
    assert 0.1953 <= NOMINAL_FIT_RESIDUAL_P95_M <= 1.8103


def test_scoring_and_head_agree_at_nominal_residual():
    """Coupling test (issue #70): scoring.py's LEG_ARRIVAL_TOL_M and
    instruction.py's ARRIVAL_TOL_M must be the SAME number -- both derived from
    this module at the same nominal residual, never independently tuned."""
    from core.groundtruth.scoring import LEG_ARRIVAL_TOL_M
    from core.heads.instruction import ARRIVAL_TOL_M

    assert LEG_ARRIVAL_TOL_M == pytest.approx(ARRIVAL_TOL_M)
    assert LEG_ARRIVAL_TOL_M == pytest.approx(NOMINAL_ARRIVAL_TOL_M)


def test_scoring_leg_arrival_tol_m_wrapper_matches_module_function():
    from core.groundtruth.scoring import leg_arrival_tol_m

    for residual in (0.0, 0.4, 1.5):
        assert leg_arrival_tol_m(residual) == pytest.approx(
            derived_arrival_tol_m(residual)
        )


def test_explore_step_provisional_tolerance_is_untouched():
    # issue #70 explicitly leaves PROVISIONAL_ARRIVAL_TOL_M alone (different purpose:
    # explore_step's provisional-goal-reached check, not the IF leg-arrival rubric).
    from core.heads.explore_step import PROVISIONAL_ARRIVAL_TOL_M

    assert PROVISIONAL_ARRIVAL_TOL_M == 0.8


# --------------------------------------------------------------------------- PASS-BY nominal (issue #100)


def test_pass_by_nominal_tol_is_the_formula_at_the_pass_by_nominal_residual():
    assert PASS_BY_NOMINAL_TOL_M == pytest.approx(
        derived_arrival_tol_m(PASS_BY_NOMINAL_FIT_RESIDUAL_M)
    )


def test_pass_by_nominal_residual_is_a_plausible_value_from_the_same_battery_provenance():
    # Same sanity bound as test_nominal_fit_residual_is_a_plausible_p95_from_battery_
    # provenance, same source snapshot (reports/issue59_leg_ceiling.json,
    # 0.1953-1.8103 m across the 15 battery scenes) -- guards against a typo'd
    # constant that no longer reflects any real measurement of that snapshot.
    assert 0.1953 <= PASS_BY_NOMINAL_FIT_RESIDUAL_M <= 1.8103


def test_pass_by_nominal_residual_is_the_median_not_the_p95():
    # The whole point of issue #100's fix: a DIFFERENT (smaller) aggregate of the
    # SAME underlying measurement, not a new/independently-tuned number.
    assert PASS_BY_NOMINAL_FIT_RESIDUAL_M < NOMINAL_FIT_RESIDUAL_P95_M


def test_pass_by_nominal_tol_is_tighter_than_the_stop_nominal_tol():
    # Direct regression guard for the issue #100 generosity: the PASS-BY base
    # tolerance (before the instance's own AABB half-diagonal is added) must be
    # strictly smaller than the STOP tolerance it used to reuse wholesale.
    assert PASS_BY_NOMINAL_TOL_M < NOMINAL_ARRIVAL_TOL_M


def test_pass_by_nominal_tol_still_shares_the_same_structural_terms():
    # Not a new formula -- same four cited terms, only the residual input differs.
    expected = (
        VEHICLE_RADIUS_M
        + DEFAULT_CELL_M * math.sqrt(2.0) / 2.0
        + PASS_BY_NOMINAL_FIT_RESIDUAL_M
        + FIXED_SAFETY_MARGIN_M
    )
    assert PASS_BY_NOMINAL_TOL_M == pytest.approx(expected)


def test_scoring_pass_by_arrival_tol_m_matches_pass_by_nominal():
    from core.groundtruth.scoring import PASS_BY_ARRIVAL_TOL_M

    assert PASS_BY_ARRIVAL_TOL_M == pytest.approx(PASS_BY_NOMINAL_TOL_M)
