"""CP4 pre-answer verification: confirm/runner_up/neither, re-resolve, fallbacks, seam."""
from __future__ import annotations

from core.checkpoints import verification as V
from core.checkpoints.verification import build_verifier, as_llm_verify_seam, VerificationOutcome
from tests.checkpoints._helpers import (
    CountingLedger,
    ExhaustedLedger,
    FakeClock,
    TimeoutStub,
    cand,
    make_ledger,
    text_stub,
)


def _run(stub, ledger=None, remaining=600.0, resolve_again=None, runner=None):
    ledger = ledger if ledger is not None else make_ledger()
    verifier = build_verifier(stub, ledger, FakeClock(), cfg={"remaining_s": lambda: remaining})
    return verifier(
        "the mug on the table",
        cand(7, "mug"),
        runner_up=runner,
        winner_facts=[],
        notes="",
        resolve_again=resolve_again,
    )


def test_confirm_keeps_winner():
    out = _run(text_stub('{"verdict": "confirm", "missed_constraint": null, "reason": "ok"}'))
    assert out.action == "keep"
    assert out.winner.instance_id == 7
    assert out.verdict == "confirm"


def test_runner_up_swaps_when_present():
    out = _run(
        text_stub('{"verdict": "runner_up", "missed_constraint": null, "reason": "r"}'),
        runner=cand(9, "mug"),
    )
    assert out.action == "runner_up"
    assert out.winner.instance_id == 9


def test_runner_up_keeps_when_no_runner_up():
    out = _run(
        text_stub('{"verdict": "runner_up", "missed_constraint": null, "reason": "r"}'),
        runner=None,
    )
    assert out.action == "keep"
    assert out.winner.instance_id == 7


def test_neither_reresolve_when_time_ge_90():
    out = _run(
        text_stub('{"verdict": "neither", "missed_constraint": "red", "reason": "r"}'),
        remaining=200.0,
        resolve_again=lambda c: cand(42, "mug"),
    )
    assert out.action == "re_resolve"
    assert out.winner.instance_id == 42
    assert out.missed_constraint == "red"


def test_neither_keeps_when_time_below_90():
    out = _run(
        text_stub('{"verdict": "neither", "missed_constraint": "red", "reason": "r"}'),
        remaining=30.0,
        resolve_again=lambda c: cand(42, "mug"),
    )
    assert out.action == "keep"
    assert out.winner.instance_id == 7


def test_neither_keeps_when_no_missed_constraint():
    out = _run(
        text_stub('{"verdict": "neither", "missed_constraint": null, "reason": "r"}'),
        remaining=200.0,
        resolve_again=lambda c: cand(42, "mug"),
    )
    assert out.action == "keep"
    assert out.winner.instance_id == 7


def test_neither_keeps_when_reresolve_returns_none():
    out = _run(
        text_stub('{"verdict": "neither", "missed_constraint": "red", "reason": "r"}'),
        remaining=200.0,
        resolve_again=lambda c: None,
    )
    assert out.action == "keep"
    assert out.winner.instance_id == 7


def test_malformed_then_one_repair_confirms():
    stub = text_stub(
        "sure: {verdict confirm}",
        '{"verdict": "confirm", "missed_constraint": null, "reason": "repaired"}',
    )
    # supply a repair callable via cfg that pulls the stub's next reply
    ledger = make_ledger()
    verifier = build_verifier(
        stub, ledger, FakeClock(), cfg={"repair": lambda raw, errs: stub.chat([])}
    )
    out = verifier("q", cand(3), runner_up=None, winner_facts=[], notes="")
    assert out.action == "keep"
    assert out.verdict == "confirm"


def test_malformed_no_repair_falls_back():
    out = _run(text_stub("not json at all"))
    assert out.action == "keep"
    assert out.verdict == "fallback"


def test_timeout_falls_back_to_winner():
    ledger = CountingLedger()
    verifier = build_verifier(TimeoutStub(block_s=3.0), ledger, FakeClock())
    out = verifier("q", cand(4), winner_facts=[], notes="")
    assert out.action == "keep"
    assert out.verdict == "fallback"
    # timeout is still recorded for audit, under the "timeout" tier
    assert any(tier == "timeout" for _, tier in ledger.records)


def test_ledger_exhausted_makes_no_call_and_falls_back():
    ledger = ExhaustedLedger()
    stub = text_stub('{"verdict": "runner_up", "missed_constraint": null, "reason": "x"}')
    verifier = build_verifier(stub, ledger, FakeClock())
    out = verifier("q", cand(7), runner_up=cand(9), winner_facts=[], notes="")
    assert out.action == "keep"  # deterministic fallback, no swap
    assert out.winner.instance_id == 7
    assert stub.calls == []  # provider never invoked
    assert ledger.records == []  # nothing recorded (no call made)


def test_reserve_denies_low_remaining():
    # make_ledger with 30 s remaining < LEDGER_RESERVE_S (45 s) -> allow() False
    ledger = make_ledger(remaining_s=30.0)
    stub = text_stub('{"verdict": "runner_up", "missed_constraint": null, "reason": "x"}')
    verifier = build_verifier(stub, ledger, FakeClock())
    out = verifier("q", cand(7), runner_up=cand(9), winner_facts=[], notes="")
    assert out.action == "keep"
    assert stub.calls == []


def test_fact_table_and_prompt_assembly():
    from tests.checkpoints._helpers import Pred

    rows = [Pred(True, "on: shelf ok"), Pred(False, "near: 2.1m too far")]
    table = V.fact_table_text(rows)
    assert "clause 0: PASS | on: shelf ok" in table
    assert "clause 1: FAIL | near: 2.1m too far" in table
    msgs = V.build_prompt("the mug", "#7 (mug)", table, "#9 (mug)", "definite article")
    body = msgs[-1]["content"]
    assert "the mug" in body and "#7 (mug)" in body and "#9 (mug)" in body
    assert "definite article" in body


def test_seam_confirm_returns_true_keep():
    ledger = make_ledger()
    seam = as_llm_verify_seam(
        text_stub('{"verdict": "confirm", "missed_constraint": null, "reason": "ok"}'),
        ledger,
        FakeClock(),
    )

    class P:
        question_raw = "the mug"
        notes = ""

    assert seam(P(), "#7 (mug)", "PASS: on shelf") is True


def test_seam_neither_keeps_winner_or_f4():
    """OR-F4: a hallucinated `neither` must KEEP the deterministic winner, not demote.

    The old seam mapped `neither` -> False (demote), which swapped to an unverified
    runner-up the model never saw — strictly worse than not verifying. The design
    fallback is keep-winner, so `neither` (non-actionable) returns True (keep).
    """
    ledger = make_ledger()
    seam = as_llm_verify_seam(
        text_stub('{"verdict": "neither", "missed_constraint": "closest_to folding screen", "reason": "r"}'),
        ledger,
        FakeClock(),
    )

    class P:
        question_raw = "the bowl on the table closest to the folding screen"
        notes = ""

    assert seam(P(), "#7 (bowl)", "PASS: on table") is True


def test_seam_runner_up_demotes_and_receives_real_runner_up():
    """OR-F4: an explicit `runner_up` verdict demotes; the real runner-up summary is
    passed into the prompt instead of the hardcoded (none)."""
    ledger = make_ledger()
    stub = text_stub('{"verdict": "runner_up", "missed_constraint": null, "reason": "r"}')
    seam = as_llm_verify_seam(stub, ledger, FakeClock())

    class P:
        question_raw = "the chair closest to the table"
        notes = ""

    assert seam(P(), "#7 (chair)", "FAIL: closest_to", "#9 (chair)") is False
    # the real runner-up one-liner reached the prompt (not the hardcoded placeholder)
    body = stub.calls[-1][-1]["content"]
    assert "#9 (chair)" in body  # real runner-up reached the prompt


def test_seam_falls_back_true_on_exhausted_ledger():
    seam = as_llm_verify_seam(text_stub('{"verdict": "neither"}'), ExhaustedLedger(), FakeClock())

    class P:
        question_raw = "q"
        notes = ""

    assert seam(P(), "s", "m") is True  # unavailable -> keep deterministic winner
