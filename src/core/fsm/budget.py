"""The 600 s question budget clock and the checkpoint call ledger.

Phase boundaries and per-checkpoint call caps come from architecture §3 and §5 and the
constants in core.interfaces. All timing is off the injected Clock (Clock.now(), seconds)
so the whole module is deterministic under a FakeClock.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from core.interfaces import (
    Clock,
    EXPLORE_BUDGET_S,
    FORCED_ASSEMBLY_S,
    QType,
    QUESTION_BUDGET_S,
    WATCHDOG_FLOOR_S,
)

ORIENTATION_S: float = 60.0  # in-place sweep window (architecture §5 step 1)

# Reserve below which no discretionary checkpoint may fire — leave room for the floor path.
LEDGER_RESERVE_S: float = 45.0


class BudgetState:
    """Tracks elapsed time since first-question receipt against the 600 s gates.

    t0 is latched exactly once (idempotent) at the first Question receipt; every phase
    query derives from now() - t0.
    """

    def __init__(self, clock: Clock, qtype: QType | None = None) -> None:
        self._clock = clock
        self._t0: float | None = None
        self.qtype = qtype

    # ------------------------------------------------------------------ latch
    def latch(self, t0: float | None = None) -> float:
        """Latch t0 at first question receipt (idempotent); returns the latched t0.

        t0 defaults to clock.now(); an explicit value (e.g. Question.t_received) wins on
        the first call and is ignored thereafter.
        """
        if self._t0 is None:
            self._t0 = float(t0) if t0 is not None else float(self._clock.now())
        return self._t0

    @property
    def latched(self) -> bool:
        return self._t0 is not None

    @property
    def t0(self) -> float | None:
        return self._t0

    def elapsed(self) -> float:
        """Seconds since t0; 0.0 before the latch."""
        if self._t0 is None:
            return 0.0
        return float(self._clock.now()) - self._t0

    def remaining(self) -> float:
        """Seconds left in the 600 s window (may go negative past the deadline)."""
        return QUESTION_BUDGET_S - self.elapsed()

    # ------------------------------------------------------------------ phases
    @property
    def in_orientation(self) -> bool:
        """True during the opening in-place sweep (< 60 s)."""
        return self.latched and self.elapsed() < ORIENTATION_S

    def explore_budget(self, qtype: QType | None = None) -> float:
        """Soft per-type exploration budget (s) before answer-path pressure."""
        qt = qtype if qtype is not None else self.qtype
        if qt is None:
            return max(EXPLORE_BUDGET_S.values())
        return EXPLORE_BUDGET_S[qt]

    def past_explore_budget(self, qtype: QType | None = None) -> bool:
        """True once elapsed has crossed the qtype's soft exploration budget."""
        return self.latched and self.elapsed() >= self.explore_budget(qtype)

    @property
    def forced_assembly(self) -> bool:
        """T-90: begin best-effort answer assembly (>= 510 s)."""
        return self.latched and self.elapsed() >= FORCED_ASSEMBLY_S

    @property
    def watchdog_floor(self) -> bool:
        """T-30: publish the floor answer unconditionally (>= 570 s)."""
        return self.latched and self.elapsed() >= WATCHDOG_FLOOR_S


# Per-checkpoint hard call caps (architecture §3). A checkpoint absent from this map is
# treated as uncapped-by-name but still gated by the time reserve.
CHECKPOINT_MAX: dict[str, int] = {
    "parse": 2,
    "miss_recovery": 1,
    "anchor_confirm": 3,
    "verification": 1,
    "frontier_select": 1,
    "self_consistency": 2,
}


@dataclass
class CallRecord:
    """One logged checkpoint invocation."""

    checkpoint: str
    duration: float
    tier: str


class CallLedger:
    """Counts checkpoint calls and enforces per-checkpoint caps + the time reserve.

    allow(checkpoint) is the gate the FSM consults before spending a call; record(...)
    is written after the call returns. The ledger holds a BudgetState so it can refuse
    discretionary calls once remaining() drops below the floor reserve.
    """

    def __init__(self, budget: BudgetState, caps: dict[str, int] | None = None) -> None:
        self._budget = budget
        self._caps = dict(CHECKPOINT_MAX if caps is None else caps)
        self._counts: dict[str, int] = {}
        self._log: list[CallRecord] = []

    def cap(self, checkpoint: str) -> int | None:
        """Max allowed calls for a checkpoint, or None if uncapped by name."""
        return self._caps.get(checkpoint)

    def count(self, checkpoint: str) -> int:
        return self._counts.get(checkpoint, 0)

    def allow(self, checkpoint: str) -> bool:
        """True iff another call is permitted: cap not exhausted AND time reserve intact.

        Returns False when the checkpoint's count has reached its cap, or when
        remaining() < LEDGER_RESERVE_S (reserve the tail for the watchdog floor path).
        """
        if self._budget.remaining() < LEDGER_RESERVE_S:
            return False
        cap = self._caps.get(checkpoint)
        if cap is not None and self.count(checkpoint) >= cap:
            return False
        return True

    def record(self, checkpoint: str, duration: float, tier: str) -> CallRecord:
        """Log a completed checkpoint call and bump its count. Always succeeds."""
        self._counts[checkpoint] = self._counts.get(checkpoint, 0) + 1
        rec = CallRecord(checkpoint=str(checkpoint), duration=float(duration), tier=str(tier))
        self._log.append(rec)
        return rec

    def log(self) -> list[CallRecord]:
        return list(self._log)
