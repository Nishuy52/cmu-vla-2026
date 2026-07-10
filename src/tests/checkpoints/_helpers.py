"""Shared fakes for the checkpoint tests: clock, candidates, ledgers, timeout stub."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from core.fsm.budget import BudgetState, CallLedger
from core.interfaces import InstanceRecord
from core.llm.providers import LocalStub


class FakeClock:
    """Manually-advanced monotonic clock (Clock protocol)."""

    def __init__(self, start: float = 0.0) -> None:
        self._t = float(start)

    def now(self) -> float:
        return self._t

    def advance(self, dt: float) -> float:
        self._t += float(dt)
        return self._t


def cand(instance_id: int, label: str = "obj", centroid=(0.0, 0.0, 0.0)) -> InstanceRecord:
    """Minimal InstanceRecord for CP4 winner/runner-up summaries."""
    c = np.array(centroid, dtype=float)
    half = np.array([0.25, 0.25, 0.25])
    return InstanceRecord(
        instance_id=instance_id,
        label=label,
        score=0.9,
        n_obs=3,
        centroid=c,
        aabb_min=c - half,
        aabb_max=c + half,
    )


@dataclass
class Pred:
    """Stand-in for a toolbox PredResult row (has .passed and .explanation)."""

    passed: bool
    explanation: str


def make_ledger(remaining_s: float = 600.0, caps: dict | None = None) -> CallLedger:
    """A real CallLedger with a latched budget leaving ``remaining_s`` in the window."""
    clock = FakeClock(0.0)
    budget = BudgetState(clock)
    budget.latch(0.0)
    # remaining() = QUESTION_BUDGET_S - elapsed; advance the clock to set remaining.
    from core.interfaces import QUESTION_BUDGET_S

    clock.advance(QUESTION_BUDGET_S - remaining_s)
    return CallLedger(budget, caps=caps)


class ExhaustedLedger:
    """A ledger that always denies (cap exhausted) and records nothing was called."""

    def __init__(self) -> None:
        self.records: list = []

    def allow(self, name: str) -> bool:
        return False

    def record(self, name: str, duration: float, tier: str):
        self.records.append((name, tier))
        return None


class CountingLedger:
    """Duck-typed ledger that always allows and logs (name, tier) records."""

    def __init__(self) -> None:
        self.records: list = []

    def allow(self, name: str) -> bool:
        return True

    def record(self, name: str, duration: float, tier: str):
        self.records.append((name, tier))
        return None


class TimeoutStub:
    """A ChatFn/VisionChatFn simulating the per-call timeout.

    Raises ``TimeoutError`` directly (the same exception ``call_with_timeout`` raises when
    a real provider hangs past the deadline), so ``guarded_call`` takes its timeout branch
    — recording under the ``"timeout"`` tier and returning the deterministic fallback —
    without a multi-second real sleep. ``block_s`` is kept for API symmetry.
    """

    def __init__(self, block_s: float = 5.0) -> None:
        self.block_s = block_s
        self.calls = 0

    def _hang(self) -> str:
        self.calls += 1
        raise TimeoutError("simulated chat call timeout")

    def chat(self, messages) -> str:
        return self._hang()

    def vision_chat(self, messages, images) -> str:
        return self._hang()

    __call__ = chat


def text_stub(*replies: str) -> LocalStub:
    return LocalStub(list(replies))
