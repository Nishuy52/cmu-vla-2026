"""Eval-time counting hygiene (H15 b/c, NUM-F8): obs gating + coverage-gated early fire."""
from __future__ import annotations

from core.heads.numerical import (
    COVERAGE_MIN_FRAC,
    STABLE_TICKS,
    NumericalHead,
)
from tests.heads._helpers import inst, numerical_plan, scene


def _advance(head, sc, n):
    for _ in range(n):
        head.advance(sc)


# --------------------------------------------------------------- H15(b) obs gating

def test_ghost_dropped_once_class_established():
    # one solid chair (n_obs=3) establishes the class; a one-frame ghost is dropped.
    sc = scene(inst(1, "chair", n_obs=3), inst(2, "chair", n_obs=1))
    head = NumericalHead(plan=numerical_plan("chair"))
    head.advance(sc)
    assert head.count == 1  # ghost excluded


def test_cold_start_all_singletons_still_count():
    # nothing established yet (peak n_obs < 3): count everything so cold starts aren't starved.
    sc = scene(inst(1, "chair", n_obs=1), inst(2, "chair", n_obs=1))
    head = NumericalHead(plan=numerical_plan("chair"))
    head.advance(sc)
    assert head.count == 2


def test_established_keeps_two_obs_instances():
    # once established, n_obs>=2 instances still count (only n_obs==1 ghosts drop).
    sc = scene(
        inst(1, "chair", n_obs=3),
        inst(2, "chair", n_obs=2),
        inst(3, "chair", n_obs=1),
    )
    head = NumericalHead(plan=numerical_plan("chair"))
    head.advance(sc)
    assert head.count == 2  # the two n_obs>=2, ghost dropped


def test_establish_threshold_is_per_queried_noun():
    # a different established noun does not gate the queried noun.
    sc = scene(
        inst(1, "table", n_obs=3),
        inst(2, "chair", n_obs=1),
        inst(3, "chair", n_obs=1),
    )
    head = NumericalHead(plan=numerical_plan("chair"))
    head.advance(sc)
    assert head.count == 2  # chairs are cold; table's n_obs=3 is irrelevant


def test_gt_battery_unaffected_all_nobs_3():
    # GT instances are n_obs=3; gating keeps them (established, all >= 2).
    sc = scene(inst(1, "chair", n_obs=3), inst(2, "chair", n_obs=3))
    head = NumericalHead(plan=numerical_plan("chair"))
    head.advance(sc)
    assert head.count == 2


# --------------------------------------------------------------- H15(c) coverage gate

def test_no_coverage_probe_is_stability_only():
    # default behaviour: stability alone fires early.
    sc = scene(inst(1, "chair"), inst(2, "chair"))
    head = NumericalHead(plan=numerical_plan("chair"))
    _advance(head, sc, STABLE_TICKS)
    assert head.signal().stable is True


def test_coverage_probe_blocks_early_fire_when_low():
    sc = scene(inst(1, "chair"), inst(2, "chair"))
    head = NumericalHead(
        plan=numerical_plan("chair"),
        coverage_frac=lambda: COVERAGE_MIN_FRAC - 0.1,  # under-covered
    )
    _advance(head, sc, STABLE_TICKS)
    # count is stable but coverage is too low -> no early fire.
    assert head.signal().winner_margin == 0.0
    assert head.signal().stable is False


def test_coverage_probe_allows_early_fire_when_high():
    sc = scene(inst(1, "chair"), inst(2, "chair"))
    head = NumericalHead(
        plan=numerical_plan("chair"),
        coverage_frac=lambda: 1.0,  # fully covered
    )
    _advance(head, sc, STABLE_TICKS)
    assert head.signal().stable is True


def test_coverage_gate_not_consulted_before_stability():
    # coverage probe must not matter until the count is stable (avoid firing on high
    # coverage but an unstable count).
    calls = {"n": 0}

    def probe():
        calls["n"] += 1
        return 1.0

    sc = scene(inst(1, "chair"), inst(2, "chair"))
    head = NumericalHead(plan=numerical_plan("chair"), coverage_frac=probe)
    head.advance(sc)  # 1 tick, not yet stable
    head.signal()
    assert calls["n"] == 0  # probe skipped while unstable
