"""Flight recorder: a bounded ring buffer of typed FSM events for post-run debugging.

One record per notable transition/decision; dumped on DONE. Times are seconds in the
question clock (Clock.now()); no wall-clock here.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterator


@dataclass(frozen=True)
class LogRecord:
    """A single flight-recorder entry.

    t: question-clock seconds; state: FSM state name at emission; event: short tag;
    detail: free-form human-readable context.
    """

    t: float
    state: str
    event: str
    detail: str = ""


class EventLog:
    """Fixed-capacity ring buffer of LogRecord (oldest dropped when full)."""

    def __init__(self, capacity: int = 512) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._buf: deque[LogRecord] = deque(maxlen=capacity)

    def record(self, t: float, state: str, event: str, detail: str = "") -> LogRecord:
        """Append an event; returns the stored record. Never raises on normal input."""
        rec = LogRecord(t=float(t), state=str(state), event=str(event), detail=str(detail))
        self._buf.append(rec)
        return rec

    def dump(self) -> list[LogRecord]:
        """Return all retained records oldest-first (the post-run debug snapshot)."""
        return list(self._buf)

    def __len__(self) -> int:
        return len(self._buf)

    def __iter__(self) -> Iterator[LogRecord]:
        return iter(self._buf)
