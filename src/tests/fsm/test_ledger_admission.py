"""CallLedger admission bound (SYS-F8 / OR-F7): reserve + worst-case call headroom.

The tightened ``allow`` must refuse a discretionary checkpoint once there is not enough
time left for the floor reserve PLUS one worst-case call to run to its timeout — so a call
admitted near the reserve cannot run its full timeout + repair round and strand the
watchdog past the deadline.
"""
from __future__ import annotations

from core.fsm.budget import (
    BudgetState,
    CallLedger,
    DEFAULT_WORST_CASE_CALL_S,
    LEDGER_RESERVE_S,
)
from core.interfaces import QUESTION_BUDGET_S, QType
from tests.fsm._fakes import FakeClock


def _latched(clk: FakeClock) -> BudgetState:
    b = BudgetState(clk, QType.NUMERICAL)
    b.latch(0.0)
    return b


def test_default_worst_case_is_two_call_timeouts():
    # 2x the 20 s per-call timeout = one primary + one repair round.
    assert DEFAULT_WORST_CASE_CALL_S == 40.0


def test_admits_with_full_budget():
    clk = FakeClock(0.0)
    led = CallLedger(_latched(clk))
    assert led.allow("verification") is True


def test_refuses_once_below_reserve_plus_worst_case():
    clk = FakeClock(0.0)
    led = CallLedger(_latched(clk))
    bound = LEDGER_RESERVE_S + DEFAULT_WORST_CASE_CALL_S  # 45 + 40 = 85 s
    # remaining just BELOW the bound -> refuse (would strand the watchdog).
    clk.set(QUESTION_BUDGET_S - bound + 0.5)
    assert led.allow("verification") is False
    # remaining comfortably ABOVE the bound -> admit.
    clk.set(QUESTION_BUDGET_S - bound - 5.0)
    assert led.allow("verification") is True


def test_bound_is_tighter_than_the_flat_reserve():
    # A call that the OLD flat >=45 s reserve would have admitted (remaining=60 s) is now
    # refused, because 60 < 45 + 40. This is the SYS-F8 regression guard.
    clk = FakeClock(0.0)
    led = CallLedger(_latched(clk))
    clk.set(QUESTION_BUDGET_S - 60.0)  # 60 s remaining
    assert led.allow("verification") is False


def test_worst_case_is_configurable():
    clk = FakeClock(0.0)
    led = CallLedger(_latched(clk), worst_case_call_s=0.0)
    assert led.worst_case_call_s == 0.0
    # With zero worst-case headroom the bound collapses back to the flat reserve.
    clk.set(QUESTION_BUDGET_S - (LEDGER_RESERVE_S + 1.0))
    assert led.allow("verification") is True
    clk.set(QUESTION_BUDGET_S - (LEDGER_RESERVE_S - 0.5))
    assert led.allow("verification") is False
