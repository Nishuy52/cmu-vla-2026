"""Cross-platform per-call timeout wrapper for ChatFn calls.

The ladder (core.parsing.ladder) enforces a wall-clock cap *between* tiers via the
injected Clock, but a single hung provider call can still stall past that cap because
the cap is only checked when control returns to the ladder. This module gives every
adapter a hard per-call backstop: run the call on a daemon worker thread and abandon it
if it does not finish within ``timeout_s``.

Why threads, not ``signal.alarm``: ``signal`` only fires on the main thread and is a
no-op for ``SIGALRM`` on Windows — the eval/dev split (Windows now, Ubuntu later) rules
it out. A worker thread + result queue works identically on both platforms. The worker
is a daemon so a truly wedged call (e.g. a socket with no timeout) cannot keep the
process alive; we abandon the thread and raise, letting the ladder fall to the next tier.

On timeout we raise ``TimeoutError`` (a builtin ``OSError`` subclass). The ladder's
``_attempt`` wraps every ChatFn call in ``except Exception``, so a raised TimeoutError is
caught like any other dead-provider failure and the ladder advances — exactly the
"swallow-and-raise" behaviour the tier try/except expects.
"""
from __future__ import annotations

import functools
import queue
import threading
from typing import Any, Callable, TypeVar

from core.parsing.prompts import ChatFn

T = TypeVar("T")

#: Default per-call ceiling (seconds). Sits below the ladder's 45 s total cap so a single
#: wedged provider cannot consume the whole parse budget.
DEFAULT_CALL_TIMEOUT_S: float = 20.0


def call_with_timeout(fn: Callable[[], T], timeout_s: float) -> T:
    """Run ``fn()`` on a daemon worker thread, returning its result or raising.

    Raises ``TimeoutError`` if ``fn`` does not complete within ``timeout_s`` seconds.
    Re-raises any exception ``fn`` itself raised (preserving its type/traceback). The
    abandoned worker thread is a daemon and cannot block interpreter shutdown.
    """
    result: queue.Queue[tuple[bool, object]] = queue.Queue(maxsize=1)

    def _worker() -> None:
        try:
            result.put((True, fn()))
        except BaseException as exc:  # noqa: BLE001 — ferry any failure back to caller
            result.put((False, exc))

    worker = threading.Thread(target=_worker, daemon=True)
    worker.start()
    try:
        ok, payload = result.get(timeout=timeout_s)
    except queue.Empty:
        raise TimeoutError(
            f"chat call exceeded {timeout_s:g}s timeout"
        ) from None
    if ok:
        return payload  # type: ignore[return-value]
    raise payload  # type: ignore[misc]  # the exception fn raised


def with_timeout(fn: ChatFn, timeout_s: float = DEFAULT_CALL_TIMEOUT_S) -> ChatFn:
    """Wrap a ChatFn so each call is bounded by ``timeout_s`` seconds.

    The returned callable has the same ``ChatFn`` signature
    (``list[dict] -> str``); on timeout it raises ``TimeoutError`` which the ladder's
    per-tier try/except treats as a provider failure and falls through on.
    """

    def _wrapped(messages: list[dict[str, str]]) -> str:
        return call_with_timeout(lambda: fn(messages), timeout_s)

    return _wrapped


def wrap_call_timeout(fn: Callable[..., T], timeout_s: float = DEFAULT_CALL_TIMEOUT_S) -> Callable[..., T]:
    """Wrap an ARBITRARY-signature callable so each call is bounded by ``timeout_s``.

    Unlike :func:`with_timeout` (which is specific to the ``ChatFn`` ``list[dict] -> str``
    shape), this preserves any ``*args, **kwargs`` signature — so it can wrap the injected
    checkpoint seams (``llm_verify(plan, summary, matrix)``, ``anchor_confirm(...)``, the rich
    ``verifier``/``miss_recoverer``/``frontier_selector`` callables, and the parse fn) at the
    factory's injection boundary. This is the SYS-F8 requirement: enforce a hard per-call
    timeout on EVERY seam that can trigger a provider call inside ``build_callables`` itself,
    rather than trusting each call site to have wrapped its ChatFn. On timeout it raises
    ``TimeoutError`` (an ``OSError`` subclass); the heads/ladder treat that like any other
    dead-provider failure and fall through to their deterministic path.

    Wrapping is idempotent-safe to skip: pass ``None`` through untouched so an unconfigured
    seam stays ``None`` (offline determinism).
    """
    if fn is None:  # keep unconfigured seams None so offline tests stay deterministic
        return fn  # type: ignore[return-value]

    @functools.wraps(fn)
    def _wrapped(*args: Any, **kwargs: Any) -> T:
        return call_with_timeout(lambda: fn(*args, **kwargs), timeout_s)

    return _wrapped
