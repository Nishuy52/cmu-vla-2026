"""Budget clock + call-ledger tests (FakeClock-driven)."""
from __future__ import annotations

import pytest

from core.fsm.budget import (
    BudgetState,
    CallLedger,
    DEFAULT_FORCED_ASSEMBLY_S,
    DEFAULT_WATCHDOG_FLOOR_S,
    LEDGER_RESERVE_S,
    ORIENTATION_S,
)
from core.interfaces import (
    EXPLORE_BUDGET_S,
    FORCED_ASSEMBLY_S,
    QUESTION_BUDGET_S,
    QType,
    WATCHDOG_FLOOR_S,
)
from tests.fsm._fakes import FakeClock


def test_t0_latches_once_and_is_idempotent():
    clk = FakeClock(100.0)
    b = BudgetState(clk, QType.NUMERICAL)
    assert not b.latched
    b.latch()  # latch at now()=100
    assert b.t0 == 100.0
    clk.set(200.0)
    b.latch(999.0)  # ignored — already latched
    assert b.t0 == 100.0
    assert b.elapsed() == 100.0


def test_latch_prefers_explicit_t0_on_first_call():
    clk = FakeClock(100.0)
    b = BudgetState(clk)
    b.latch(80.0)  # e.g. Question.t_received earlier than now()
    assert b.t0 == 80.0
    assert b.elapsed() == 20.0


def test_orientation_phase_boundary():
    clk = FakeClock(0.0)
    b = BudgetState(clk, QType.NUMERICAL)
    b.latch()
    assert b.in_orientation is True
    clk.set(ORIENTATION_S - 0.001)
    assert b.in_orientation is True
    clk.set(ORIENTATION_S)
    assert b.in_orientation is False  # < 60 s strictly


def test_remaining_and_forced_and_floor_boundaries():
    clk = FakeClock(0.0)
    b = BudgetState(clk, QType.NUMERICAL)
    b.latch()
    assert b.remaining() == pytest.approx(600.0)
    clk.set(509.9)
    assert b.forced_assembly is False
    clk.set(510.0)
    assert b.forced_assembly is True
    assert b.watchdog_floor is False
    clk.set(570.0)
    assert b.watchdog_floor is True
    clk.set(600.0)
    assert b.remaining() == pytest.approx(0.0)


@pytest.mark.parametrize(
    "qt", [QType.NUMERICAL, QType.OBJECT_REFERENCE, QType.INSTRUCTION_FOLLOWING]
)
def test_explore_budget_per_qtype(qt):
    b = BudgetState(FakeClock(0.0), qt)
    assert b.explore_budget() == EXPLORE_BUDGET_S[qt]


def test_past_explore_budget():
    clk = FakeClock(0.0)
    b = BudgetState(clk, QType.NUMERICAL)
    b.latch()
    clk.set(EXPLORE_BUDGET_S[QType.NUMERICAL] - 1)
    assert b.past_explore_budget() is False
    clk.set(EXPLORE_BUDGET_S[QType.NUMERICAL])
    assert b.past_explore_budget() is True


def test_budget_gate_defaults_are_interface_constants():
    """BudgetState's own defaults stay the neutral 510/570 interface constants."""
    b = BudgetState(FakeClock(0.0), QType.NUMERICAL)
    assert b.forced_assembly_s == FORCED_ASSEMBLY_S
    assert b.watchdog_floor_s == WATCHDOG_FLOOR_S


def test_controller_effective_gate_defaults_are_pulled_in():
    """The controller's effective defaults are earlier than the interface constants (SYS-F6)."""
    assert DEFAULT_FORCED_ASSEMBLY_S == 480.0
    assert DEFAULT_WATCHDOG_FLOOR_S == 540.0
    assert DEFAULT_FORCED_ASSEMBLY_S < FORCED_ASSEMBLY_S
    assert DEFAULT_WATCHDOG_FLOOR_S < WATCHDOG_FLOOR_S


def test_configurable_gates_move_the_boundaries():
    clk = FakeClock(0.0)
    b = BudgetState(
        clk,
        QType.NUMERICAL,
        forced_assembly_s=DEFAULT_FORCED_ASSEMBLY_S,
        watchdog_floor_s=DEFAULT_WATCHDOG_FLOOR_S,
    )
    b.latch()
    clk.set(479.9)
    assert b.forced_assembly is False
    clk.set(480.0)
    assert b.forced_assembly is True
    assert b.watchdog_floor is False
    clk.set(540.0)
    assert b.watchdog_floor is True


def test_gate_ordering_invariant_holds_for_defaults():
    """explore < forced_assembly < floor < total for both the neutral and pulled-in gates."""
    explore_max = max(EXPLORE_BUDGET_S.values())
    for fa, wf in (
        (FORCED_ASSEMBLY_S, WATCHDOG_FLOOR_S),
        (DEFAULT_FORCED_ASSEMBLY_S, DEFAULT_WATCHDOG_FLOOR_S),
    ):
        assert explore_max < fa < wf < QUESTION_BUDGET_S


def test_numerical_explore_budget_raised_and_still_orders_against_both_gate_pairs():
    """#150: NUMERICAL's soft explore budget was raised from 210 -> 450 s so a numerical
    question (which answers in place with no drive-out phase) is not cut off ~270 s early
    relative to the effective 480 s forced-assembly gate. It is now the qtype that sets
    `max(EXPLORE_BUDGET_S.values())`, and construction must still succeed against both the
    neutral (510/570) and effective/pulled-in (480/540) gate pairs -- i.e. the 30 s settle
    margin below DEFAULT_FORCED_ASSEMBLY_S is real headroom, not a rounding accident.
    """
    assert EXPLORE_BUDGET_S[QType.NUMERICAL] == 450.0
    assert EXPLORE_BUDGET_S[QType.NUMERICAL] == max(EXPLORE_BUDGET_S.values())
    assert DEFAULT_FORCED_ASSEMBLY_S - EXPLORE_BUDGET_S[QType.NUMERICAL] == pytest.approx(30.0)
    for fa, wf in (
        (FORCED_ASSEMBLY_S, WATCHDOG_FLOOR_S),
        (DEFAULT_FORCED_ASSEMBLY_S, DEFAULT_WATCHDOG_FLOOR_S),
    ):
        b = BudgetState(FakeClock(0.0), QType.NUMERICAL, forced_assembly_s=fa, watchdog_floor_s=wf)
        b.latch()
        assert b.explore_budget() == 450.0


@pytest.mark.parametrize(
    "fa, wf",
    [
        (570.0, 540.0),  # forced_assembly after floor
        (400.0, 540.0),  # forced_assembly below max explore (450, NUMERICAL — #150)
        (480.0, 700.0),  # floor past total budget
        (540.0, 540.0),  # equal (must be strict)
    ],
)
def test_gate_ordering_invariant_rejects_bad_gates(fa, wf):
    with pytest.raises(ValueError):
        BudgetState(FakeClock(0.0), QType.NUMERICAL, forced_assembly_s=fa, watchdog_floor_s=wf)


def test_phases_false_before_latch():
    b = BudgetState(FakeClock(300.0))
    assert b.in_orientation is False
    assert b.forced_assembly is False
    assert b.watchdog_floor is False
    assert b.elapsed() == 0.0


def test_ledger_caps_from_architecture():
    b = BudgetState(FakeClock(0.0), QType.NUMERICAL)
    b.latch()
    led = CallLedger(b)
    assert led.cap("parse") == 2
    assert led.cap("miss_recovery") == 1
    assert led.cap("anchor_confirm") == 3
    assert led.cap("verification") == 1
    assert led.cap("frontier_select") == 1
    assert led.cap("self_consistency") == 2


def test_ledger_exhaustion_blocks_checkpoint():
    b = BudgetState(FakeClock(0.0), QType.NUMERICAL)
    b.latch()
    led = CallLedger(b)
    assert led.allow("verification") is True
    led.record("verification", 0.1, "api")
    assert led.count("verification") == 1
    assert led.allow("verification") is False  # cap = 1


def test_ledger_reserve_blocks_when_time_low():
    clk = FakeClock(0.0)
    b = BudgetState(clk, QType.NUMERICAL)
    b.latch()
    led = CallLedger(b)
    # push remaining below the reserve floor
    clk.set(600.0 - LEDGER_RESERVE_S + 0.001)
    assert led.allow("parse") is False
    # even a fresh checkpoint with cap left is blocked
    assert led.allow("anchor_confirm") is False


def test_ledger_record_logs_and_counts():
    b = BudgetState(FakeClock(0.0), QType.NUMERICAL)
    b.latch()
    led = CallLedger(b)
    led.record("parse", 1.2, "local")
    led.record("parse", 0.8, "api")
    assert led.count("parse") == 2
    assert led.allow("parse") is False
    log = led.log()
    assert [r.tier for r in log] == ["local", "api"]
