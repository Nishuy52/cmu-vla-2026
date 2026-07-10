"""CP3 anchor confirmation: match/mismatch-demote/low-conf, ledger, timeout, repair."""
from __future__ import annotations

import numpy as np

from core.checkpoints.anchor_confirm import build_anchor_confirm, MISMATCH_MIN_CONFIDENCE
from tests.checkpoints._helpers import (
    CountingLedger,
    ExhaustedLedger,
    FakeClock,
    TimeoutStub,
    make_ledger,
    text_stub,
)


def _crop():
    return np.zeros((16, 16), dtype=np.uint8)


def _run(stub, ledger=None, cfg=None):
    ledger = ledger if ledger is not None else make_ledger()
    run = build_anchor_confirm(stub, ledger, FakeClock(), cfg=cfg)
    return run("wooden table", _crop())


def test_match_confirms():
    out = _run(text_stub('{"match": true, "actual_label": "table", "confidence": 0.95}'))
    assert out.action == "confirm"
    assert out.match is True


def test_mismatch_high_conf_demotes():
    out = _run(text_stub('{"match": false, "actual_label": "stool", "confidence": 0.85}'))
    assert out.action == "demote"
    assert out.actual_label == "stool"
    assert out.confidence >= MISMATCH_MIN_CONFIDENCE


def test_mismatch_low_conf_confirms():
    out = _run(text_stub('{"match": false, "actual_label": "lamp", "confidence": 0.4}'))
    assert out.action == "confirm"  # too weak to override the map (design rule)


def test_mismatch_at_threshold_demotes():
    out = _run(text_stub('{"match": false, "actual_label": "x", "confidence": 0.6}'))
    assert out.action == "demote"


def test_malformed_then_repair_match():
    stub = text_stub("it matches", '{"match": true, "actual_label": "sofa", "confidence": 0.9}')
    run = build_anchor_confirm(
        stub, make_ledger(), FakeClock(), cfg={"repair": lambda raw, errs: stub.chat([])}
    )
    out = run("red sofa", _crop())
    assert out.action == "confirm"
    assert out.match is True


def test_malformed_no_repair_confirms():
    out = _run(text_stub("no json"))
    assert out.action == "confirm"  # fallback = trust the map
    assert out.match is None


def test_timeout_confirms():
    ledger = CountingLedger()
    run = build_anchor_confirm(TimeoutStub(3.0), ledger, FakeClock())
    out = run("bookshelf", _crop())
    assert out.action == "confirm"
    assert out.match is None
    assert any(tier == "timeout" for _, tier in ledger.records)


def test_ledger_exhausted_no_call():
    stub = text_stub('{"match": false, "actual_label": "x", "confidence": 0.9}')
    run = build_anchor_confirm(stub, ExhaustedLedger(), FakeClock())
    out = run("chair", _crop())
    assert out.action == "confirm"  # never demotes without a real call
    assert stub.calls == []


def test_crop_encoded_and_passed():
    stub = text_stub('{"match": true, "actual_label": "t", "confidence": 0.9}')
    run = build_anchor_confirm(stub, make_ledger(), FakeClock())
    run("table", _crop())
    assert len(stub.image_calls) == 1
    assert len(stub.image_calls[0]) == 1
