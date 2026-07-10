"""Timeout-wrapper tests: fast calls pass through, hung calls raise TimeoutError."""
from __future__ import annotations

import threading
import time

import pytest

from core.llm.timeout import call_with_timeout, with_timeout


def test_fast_call_returns_result():
    assert call_with_timeout(lambda: "ok", timeout_s=5.0) == "ok"


def test_hung_call_raises_timeout_error():
    started = threading.Event()

    def hang():
        started.set()
        time.sleep(10.0)  # far past the timeout; daemon thread is abandoned
        return "never"

    t0 = time.monotonic()
    with pytest.raises(TimeoutError):
        call_with_timeout(hang, timeout_s=0.1)
    # returns promptly at the timeout, not after the full sleep
    assert time.monotonic() - t0 < 5.0
    assert started.is_set()


def test_call_reraises_inner_exception_type():
    def boom():
        raise ValueError("inner failure")

    with pytest.raises(ValueError, match="inner failure"):
        call_with_timeout(boom, timeout_s=5.0)


def test_with_timeout_wraps_chatfn_and_forwards_messages():
    seen = []

    def fn(messages):
        seen.append(messages)
        return "reply"

    wrapped = with_timeout(fn, timeout_s=5.0)
    out = wrapped([{"role": "user", "content": "hi"}])
    assert out == "reply"
    assert seen == [[{"role": "user", "content": "hi"}]]


def test_with_timeout_cancels_hung_chatfn():
    def hung_fn(_messages):
        time.sleep(10.0)
        return "never"

    wrapped = with_timeout(hung_fn, timeout_s=0.1)
    with pytest.raises(TimeoutError):
        wrapped([{"role": "user", "content": "hi"}])
