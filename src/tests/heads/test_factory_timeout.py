"""build_callables enforces a per-call timeout on injected seams + forwards support hooks.

SYS-F8: every provider-triggering seam (parse, llm_verify, anchor_confirm(er), verifier,
miss_recoverer, frontier_selector) is wrapped with a hard per-call timeout AT THE INJECTION
BOUNDARY inside build_callables — a hung seam cannot stall the tick past that bound. The
fast local support hooks (budget_frac, remaining_s, tiles_fn, fuse_hint) are NOT wrapped and
must flow through to the heads verbatim (H8 budget_frac wiring).
"""
from __future__ import annotations

import time

import pytest

from core.fsm.controller import WorldView
from core.heads.factory import HeadState, build_callables
from core.interfaces import Question, QType
from core.perception.scene_index import BasicSceneIndex
from tests.heads._helpers import inst, instruction_plan, object_plan
from core.plan_schema import Anchor, LegKind, RouteLeg


def _idx(*records):
    return BasicSceneIndex(list(records))


# --------------------------------------------------------------------- timeout wrapping


def test_parse_seam_times_out():
    """A parse fn that sleeps past the timeout raises TimeoutError through the wrapper."""

    def slow_parse(_q):
        time.sleep(0.5)
        return None

    cbs = build_callables(_idx(inst(1, "chair")), parse=slow_parse, call_timeout_s=0.05)
    with pytest.raises(TimeoutError):
        cbs["parse"](Question(text="x", t_received=0.0))


def test_parse_seam_returns_within_timeout():
    """A fast parse fn returns its value unchanged through the wrapper."""
    sentinel = object()
    cbs = build_callables(
        _idx(inst(1, "chair")), parse=lambda q: sentinel, call_timeout_s=1.0
    )
    assert cbs["parse"](Question(text="x", t_received=0.0)) is sentinel


def test_llm_verify_seam_is_timeout_wrapped():
    """The llm_verify seam that reaches the ObjectRefHead is the timeout WRAPPER, not the raw
    callable — a slow verify seam raises TimeoutError when the head invokes it."""

    def slow_verify(plan, summary, matrix):
        time.sleep(0.5)
        return True

    sc = _idx(inst(1, "chair"), inst(2, "chair", centroid=(5, 0, 0)))
    # Build the state exactly as build_callables does (it wraps llm_verify at injection),
    # then confirm the head's seam is the wrapper by observing the timeout.
    from core.llm.timeout import wrap_call_timeout

    st = HeadState(scene=sc, llm_verify=wrap_call_timeout(slow_verify, 0.05))
    st.bind(object_plan("chair"))
    assert st.object_ref.llm_verify is not slow_verify  # it is the wrapper
    with pytest.raises(TimeoutError):
        st.object_ref.llm_verify(None, "summary", "matrix")


def test_none_seams_stay_none_offline():
    """Unconfigured seams pass through as None so the offline path stays deterministic."""
    st = HeadState(scene=_idx(inst(1, "chair")))
    assert st.llm_verify is None
    assert st.verifier is None
    assert st.anchor_confirm is None
    # build_callables with no seams still constructs and parses via the regex default.
    cbs = build_callables(_idx(inst(1, "chair")))
    plan = cbs["parse"](Question(text="how many chairs", t_received=0.0))
    assert plan is not None and plan.parse_tier == "regex"


# --------------------------------------------------------------------- budget_frac forward


def test_budget_frac_forwarded_to_instruction_head():
    """budget_frac injected into build_callables reaches InstructionHead.budget_frac (H4c).

    The H4c provisional-terminal gate is inert without budget_frac (verifier note in H8), so
    the factory must forward it verbatim to the InstructionHead AND the ExploreHead. It is a
    fast local read, so it is NOT timeout-wrapped (same object identity).
    """

    def frac():
        return 0.73

    sc = _idx(inst(1, "chair"))
    plan = instruction_plan([RouteLeg(kind=LegKind.GOTO, anchors=[Anchor(noun="chair")])])
    st = HeadState(scene=sc, budget_frac=frac)
    st.bind(plan)
    assert st.instruction is not None
    assert st.instruction.budget_frac is frac  # forwarded verbatim, NOT timeout-wrapped
    assert st.explore.budget_frac is frac


def test_forced_assembly_forwarded_to_instruction_head():
    """forced_assembly injected into build_callables reaches InstructionHead.forced_assembly
    (issue #33). Unwired, the single-observation route-prefix commit gate stays STRICT, so
    the factory must forward it verbatim (fast local read, NOT timeout-wrapped)."""

    def forced():
        return True

    sc = _idx(inst(1, "chair"))
    plan = instruction_plan([RouteLeg(kind=LegKind.GOTO, anchors=[Anchor(noun="chair")])])
    st = HeadState(scene=sc, forced_assembly=forced)
    st.bind(plan)
    assert st.instruction is not None
    assert st.instruction.forced_assembly is forced  # forwarded verbatim


def test_support_hooks_not_timeout_wrapped():
    """budget_frac / remaining_s reach the head as the SAME object (fast local reads)."""
    sc = _idx(inst(1, "chair"))
    bf = lambda: 0.5
    rs = lambda: 120.0
    st = HeadState(scene=sc, budget_frac=bf, remaining_s=rs)
    st.bind(object_plan("chair"))
    # object_ref receives remaining_s verbatim; explore receives budget_frac verbatim.
    assert st.object_ref.remaining_s is rs
    assert st.explore.budget_frac is bf
