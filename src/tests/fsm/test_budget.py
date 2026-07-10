"""Budget clock + call-ledger tests (FakeClock-driven)."""
from __future__ import annotations

import pytest

from core.fsm.budget import BudgetState, CallLedger, LEDGER_RESERVE_S, ORIENTATION_S
from core.interfaces import EXPLORE_BUDGET_S, QType
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
