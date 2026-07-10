"""CP2 detector-miss recovery: present/absent/hallucination-cap, ledger, timeout, repair."""
from __future__ import annotations

import numpy as np

from core.checkpoints.miss_recovery import build_miss_recovery, PROVISIONAL_MIN_POINTS
from tests.checkpoints._helpers import (
    CountingLedger,
    ExhaustedLedger,
    FakeClock,
    TimeoutStub,
    make_ledger,
    text_stub,
)


def _tiles(n=4):
    return [np.zeros((8, 8), dtype=np.uint8) for _ in range(n)]


def _run(stub, ledger=None, cfg=None):
    ledger = ledger if ledger is not None else make_ledger()
    run = build_miss_recovery(stub, ledger, FakeClock(), cfg=cfg)
    return run("extinguisher", "fire extinguisher", _tiles())


def test_present_yields_provisional():
    out = _run(text_stub('{"present": true, "tile": 2, "bbox_hint": [1,2,3,4], "confidence": 0.8}'))
    assert out.action == "provisional"
    assert out.tile == 2
    assert out.bbox_hint == [1.0, 2.0, 3.0, 4.0]
    assert out.n_obs == 1
    assert out.min_points == PROVISIONAL_MIN_POINTS


def test_hallucination_cap_score_below_early_gate():
    out = _run(text_stub('{"present": true, "tile": 0, "bbox_hint": [0,0,4,4], "confidence": 1.0}'))
    assert out.action == "provisional"
    assert out.n_obs == 1  # can never reach the >=3-obs early-answer gate alone
    assert out.score == 0.5  # confidence * 0.5


def test_absent_proceeds_to_fallback():
    out = _run(text_stub('{"present": false, "tile": null, "bbox_hint": null, "confidence": 0.9}'))
    assert out.action == "absent"
    assert out.confidence == 0.9


def test_present_without_bbox_treated_absent():
    out = _run(text_stub('{"present": true, "tile": 1, "bbox_hint": null, "confidence": 0.5}'))
    assert out.action == "absent"


def test_present_with_out_of_range_tile_treated_absent():
    out = _run(text_stub('{"present": true, "tile": 9, "bbox_hint": [1,1,2,2], "confidence": 0.5}'))
    assert out.action == "absent"


def test_malformed_then_repair_present():
    stub = text_stub(
        "present yes",
        '{"present": true, "tile": 3, "bbox_hint": [5,5,9,9], "confidence": 0.7}',
    )
    run = build_miss_recovery(
        stub, make_ledger(), FakeClock(), cfg={"repair": lambda raw, errs: stub.chat([])}
    )
    out = run("cabinet", "cabinet", _tiles())
    assert out.action == "provisional"
    assert out.tile == 3


def test_timeout_absent():
    ledger = CountingLedger()
    run = build_miss_recovery(TimeoutStub(3.0), ledger, FakeClock())
    out = run("printer", "printer", _tiles())
    assert out.action == "absent"
    assert any(tier == "timeout" for _, tier in ledger.records)


def test_ledger_exhausted_no_call():
    stub = text_stub('{"present": true, "tile": 0, "bbox_hint": [1,1,2,2], "confidence": 0.9}')
    run = build_miss_recovery(stub, ExhaustedLedger(), FakeClock())
    out = run("vase", "vase", _tiles())
    assert out.action == "absent"
    assert stub.calls == []


def test_images_encoded_and_passed():
    stub = text_stub('{"present": false, "tile": null, "bbox_hint": null, "confidence": 0.1}')
    run = build_miss_recovery(stub, make_ledger(), FakeClock())
    run("lamp", "lamp", _tiles(4))
    # LocalStub records image batches for vision calls
    assert len(stub.image_calls) == 1
    assert len(stub.image_calls[0]) == 4
    assert all(isinstance(b, (bytes, bytearray)) and b for b in stub.image_calls[0])
