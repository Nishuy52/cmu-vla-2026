"""Shared call-runtime for the checkpoints: ledger gate + hard timeout + record.

Every checkpoint call obeys the same envelope (design doc "Shared rules"):

    if not ledger.allow(name): -> deterministic fallback, NO provider call
    run the provider call under a hard per-call timeout (VLA_LLM_CALL_TIMEOUT_S)
    ledger.record(name, duration, tier) after it returns (success or timeout)

:func:`guarded_call` centralises that envelope so each module only supplies its own
prompt assembly, decode, and fallback. Timing is off the injected clock so tests are
deterministic under a FakeClock; the wall-clock timeout uses the ladder's thread-based
:func:`core.llm.timeout.call_with_timeout` (signal-free, cross-platform).
"""
from __future__ import annotations

import os
from typing import Callable

from core.llm.timeout import DEFAULT_CALL_TIMEOUT_S, call_with_timeout


def timeout_s() -> float:
    """Per-call hard timeout from ``VLA_LLM_CALL_TIMEOUT_S`` (default 20 s)."""
    raw = os.environ.get("VLA_LLM_CALL_TIMEOUT_S")
    if raw:
        try:
            return float(raw)
        except ValueError:
            pass
    return DEFAULT_CALL_TIMEOUT_S


class Ledger:
    """Structural view of the CallLedger surface the checkpoints use (duck-typed)."""

    def allow(self, checkpoint: str) -> bool: ...  # pragma: no cover - protocol
    def record(self, checkpoint: str, duration: float, tier: str) -> object: ...  # pragma: no cover


def guarded_call(
    name: str,
    ledger,
    clock,
    call: Callable[[], str],
    *,
    tier: str = "checkpoint",
    on_timeout_s: float | None = None,
) -> str | None:
    """Run one ledger-gated, timeout-bounded provider call; return its raw reply or None.

    Returns ``None`` (and makes NO provider call) when ``ledger.allow(name)`` is False —
    the caller then applies its deterministic fallback. On timeout or any provider
    exception it records the (attempted) call for audit and returns ``None``. On success
    it records the real duration and returns the raw reply string.

    ``ledger`` and ``clock`` are duck-typed: a missing ``allow``/``record``/``now`` is
    tolerated (treated as "allowed", "0 duration") so partial stubs work in tests.
    """
    if _ledger_denies(ledger, name):
        return None
    limit = on_timeout_s if on_timeout_s is not None else timeout_s()
    t_start = _now(clock)
    try:
        raw = call_with_timeout(call, limit)
    except Exception:  # noqa: BLE001 — timeout OR dead provider -> deterministic fallback
        _record(ledger, name, _now(clock) - t_start, tier="timeout")
        return None
    _record(ledger, name, _now(clock) - t_start, tier=tier)
    return raw


def _ledger_denies(ledger, name: str) -> bool:
    allow = getattr(ledger, "allow", None)
    if allow is None:
        return False
    try:
        return not bool(allow(name))
    except Exception:  # noqa: BLE001 — a broken ledger must not block; treat as allowed
        return False


def _record(ledger, name: str, duration: float, tier: str) -> None:
    record = getattr(ledger, "record", None)
    if record is None:
        return
    try:
        record(name, float(duration), tier)
    except Exception:  # noqa: BLE001 — recording is best-effort audit, never fatal
        pass


def _now(clock) -> float:
    now = getattr(clock, "now", None)
    if now is None:
        return 0.0
    try:
        return float(now())
    except Exception:  # noqa: BLE001
        return 0.0
