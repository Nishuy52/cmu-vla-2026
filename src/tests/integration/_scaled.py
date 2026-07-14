"""Budget-scale helper for the end-to-end integration tests.

The integration cases below build a real :class:`~core.fsm.controller.QuestionController`
and tick it at 5 Hz until it latches an answer. Left unscaled, a case that only finishes
via a budget floor runs the FSM's full 510 s / 570 s gates in sim-time — thousands of
ticks of mock terrain/frontier recompute per case (tens of wall-seconds each).

These cases assert *structure* — published states, answer objects, driven-trajectory
geometry, seam-call counts — never absolute wall or sim timing. So we compress sim-time
exactly the way :class:`core.runner.single._ScaledClock` does for replay: a wrapper clock
amplifies elapsed time by ``1 / scale`` as the FSM's :class:`~core.fsm.budget.BudgetState`
reads it, so a gate at ``G`` seconds is crossed once the underlying (test-advanced) clock
has moved only ``G * scale`` seconds. At ``scale=0.05`` the 570 s watchdog fires after
~28.5 s of ticked time — ~570 fewer ticks than unscaled — while the follower still steps
``DRIVE_STEP_M`` per tick, so the trajectory geometry is byte-for-byte unchanged.

The wrapper is installed via the :meth:`set_budget_clock` seam (the same one
``ReplayRobotIO`` exposes and ``MockRobotIO`` now mirrors); the io's ``latest_*`` getters
keep stamping messages off the real clock, so nothing but the budget gates is scaled.
"""
from __future__ import annotations

from typing import Any

# Default compression for these structural cases. 0.05 keeps every gate ordering intact
# (explore budgets < 510 s forced assembly < 570 s watchdog all scale by the same factor)
# while collapsing the tick count ~20x.
BUDGET_SCALE = 0.05


class ScaledBudgetClock:
    """Amplifies elapsed time past construction by ``1 / scale`` (see module docstring)."""

    def __init__(self, inner: Any, scale: float = BUDGET_SCALE) -> None:
        self._inner = inner
        self._scale = float(scale)
        self._t0 = float(inner.now())

    def now(self) -> float:
        raw = float(self._inner.now())
        return self._t0 + (raw - self._t0) / self._scale


def install_scaled_budget(io: Any, scale: float = BUDGET_SCALE) -> ScaledBudgetClock:
    """Wrap the io's raw clock in a :class:`ScaledBudgetClock` and install it as the FSM's
    budget clock. Call before the first controller tick. Returns the wrapper so a test can
    read scaled elapsed time (e.g. to assert an answer landed before the scaled watchdog)."""
    raw = io.raw_clock() if hasattr(io, "raw_clock") else io.clock()
    scaled = ScaledBudgetClock(raw, scale)
    io.set_budget_clock(scaled)
    return scaled
