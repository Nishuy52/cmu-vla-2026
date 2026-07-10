"""CP5 frontier selection: choice/out-of-range fallback, ledger, timeout, repair."""
from __future__ import annotations

import numpy as np

from core.checkpoints.frontier_select import build_frontier_select
from tests.checkpoints._helpers import (
    CountingLedger,
    ExhaustedLedger,
    FakeClock,
    TimeoutStub,
    make_ledger,
    text_stub,
)


def _pano():
    return np.zeros((8, 32), dtype=np.uint8)


def _run(stub, ledger=None, cfg=None, n=None):
    ledger = ledger if ledger is not None else make_ledger()
    run = build_frontier_select(stub, ledger, FakeClock(), cfg=cfg)
    return run("how many chairs", _pano(), n_frontiers=n)


def test_choice_maps_to_zero_index():
    out = _run(text_stub('{"choice": 2, "reason": "doorway"}'))
    assert out.action == "choice"
    assert out.choice == 2
    assert out.index == 1


def test_choice_boundary_five():
    out = _run(text_stub('{"choice": 5, "reason": "far room"}'))
    assert out.action == "choice"
    assert out.index == 4


def test_out_of_range_falls_back():
    out = _run(text_stub('{"choice": 9, "reason": "invalid"}'))
    assert out.action == "fallback"


def test_respects_custom_n_frontiers():
    # with only 3 frontiers, choice 4 is out of range -> fallback
    out = _run(text_stub('{"choice": 4, "reason": "x"}'), n=3)
    assert out.action == "fallback"
    out2 = _run(text_stub('{"choice": 3, "reason": "x"}'), n=3)
    assert out2.action == "choice" and out2.index == 2


def test_malformed_then_repair_choice():
    stub = text_stub("door number one", '{"choice": 1, "reason": "repaired"}')
    run = build_frontier_select(
        stub, make_ledger(), FakeClock(), cfg={"repair": lambda raw, errs: stub.chat([])}
    )
    out = run("q", _pano())
    assert out.action == "choice"
    assert out.index == 0


def test_malformed_no_repair_falls_back():
    out = _run(text_stub("no json"))
    assert out.action == "fallback"


def test_timeout_falls_back():
    ledger = CountingLedger()
    run = build_frontier_select(TimeoutStub(3.0), ledger, FakeClock())
    out = run("q", _pano())
    assert out.action == "fallback"
    assert any(tier == "timeout" for _, tier in ledger.records)


def test_ledger_exhausted_no_call():
    stub = text_stub('{"choice": 2, "reason": "x"}')
    run = build_frontier_select(stub, ExhaustedLedger(), FakeClock())
    out = run("q", _pano())
    assert out.action == "fallback"
    assert stub.calls == []
