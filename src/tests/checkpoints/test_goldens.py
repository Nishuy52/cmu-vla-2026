"""Golden fixtures round-trip: every case -> real runner -> expected structured outcome."""
from __future__ import annotations

import numpy as np
import pytest

from core.checkpoints import fixtures
from core.checkpoints.anchor_confirm import build_anchor_confirm
from core.checkpoints.frontier_select import build_frontier_select
from core.checkpoints.miss_recovery import build_miss_recovery
from core.checkpoints.verification import build_verifier
from core.llm.providers import LocalStub
from tests.checkpoints._helpers import CountingLedger, FakeClock, TimeoutStub, cand, make_ledger


def _chat_fns(case):
    """A stub replaying the case's scripted replies, or a hanging stub for timeout cases."""
    if case.get("timeout"):
        return TimeoutStub(3.0)
    return LocalStub(list(case["stub_replies"]))


def _repair_cfg(stub, case):
    """When a case has 2 replies, the 2nd is the repair reply -> feed it via a repair cb."""
    if case.get("timeout") or len(case.get("stub_replies", [])) < 2:
        return {}
    return {"repair": lambda raw, errs: stub.chat([])}


def _ledger(case):
    return CountingLedger() if case.get("timeout") else make_ledger()


# --------------------------------------------------------------------------- CP4


@pytest.mark.parametrize("case", fixtures.load("verification"), ids=lambda c: c["name"])
def test_verification_goldens(case):
    stub = _chat_fns(case)
    ledger = _ledger(case)
    inp = case["input"]
    cfg = {"remaining_s": lambda: case.get("remaining_s", float("inf"))}
    cfg.update(_repair_cfg(stub, case))
    verifier = build_verifier(stub, ledger, FakeClock(), cfg=cfg)

    runner = None if "(none)" in inp["runner_up_summary"] else cand(_id(inp["runner_up_summary"]))
    resolve_again = None
    if "reresolve_id" in case:
        rid = case["reresolve_id"]
        resolve_again = lambda c, _rid=rid: cand(_rid)

    out = verifier(
        inp["question"],
        cand(_id(inp["winner_summary"]), _label(inp["winner_summary"])),
        runner_up=runner,
        winner_facts=[],
        notes=inp["notes"],
        resolve_again=resolve_again,
    )
    exp = case["expected"]
    assert out.action == exp["action"]
    assert out.verdict == exp["verdict"]
    assert out.winner.instance_id == exp["winner_id"]
    if "missed_constraint" in exp:
        assert out.missed_constraint == exp["missed_constraint"]


# --------------------------------------------------------------------------- CP2


@pytest.mark.parametrize("case", fixtures.load("miss_recovery"), ids=lambda c: c["name"])
def test_miss_recovery_goldens(case):
    stub = _chat_fns(case)
    ledger = _ledger(case)
    inp = case["input"]
    run = build_miss_recovery(stub, ledger, FakeClock(), cfg=_repair_cfg(stub, case))
    tiles = [np.zeros((4, 4), dtype=np.uint8) for _ in range(inp["n_tiles"])]
    out = run(inp["noun"], inp["raw"], tiles)
    exp = case["expected"]
    assert out.action == exp["action"]
    for key in ("tile", "confidence", "score", "n_obs", "min_points"):
        if key in exp:
            assert getattr(out, key) == exp[key], key


# --------------------------------------------------------------------------- CP3


@pytest.mark.parametrize("case", fixtures.load("anchor_confirm"), ids=lambda c: c["name"])
def test_anchor_confirm_goldens(case):
    stub = _chat_fns(case)
    ledger = _ledger(case)
    inp = case["input"]
    run = build_anchor_confirm(stub, ledger, FakeClock(), cfg=_repair_cfg(stub, case))
    out = run(inp["anchor_desc"], np.zeros((4, 4), dtype=np.uint8))
    exp = case["expected"]
    assert out.action == exp["action"]
    for key in ("match", "actual_label", "confidence"):
        if key in exp:
            assert getattr(out, key) == exp[key], key


# --------------------------------------------------------------------------- CP5


@pytest.mark.parametrize("case", fixtures.load("frontier_select"), ids=lambda c: c["name"])
def test_frontier_select_goldens(case):
    stub = _chat_fns(case)
    ledger = _ledger(case)
    inp = case["input"]
    run = build_frontier_select(stub, ledger, FakeClock(), cfg=_repair_cfg(stub, case))
    out = run(inp["question"], np.zeros((4, 8), dtype=np.uint8), n_frontiers=inp["n_frontiers"])
    exp = case["expected"]
    assert out.action == exp["action"]
    for key in ("choice", "index"):
        if key in exp:
            assert getattr(out, key) == exp[key], key


# --------------------------------------------------------------------------- helpers


def _id(summary: str) -> int:
    # summaries look like "#7 (mug) at (...)"
    return int(summary.split("#", 1)[1].split(" ", 1)[0])


def _label(summary: str) -> str:
    return summary.split("(", 1)[1].split(")", 1)[0]


def test_all_checkpoints_have_at_least_three_goldens():
    g = fixtures.all_goldens()
    assert set(g) >= {"verification", "miss_recovery", "anchor_confirm", "frontier_select"}
    for name, cases in g.items():
        assert len(cases) >= 3, name
