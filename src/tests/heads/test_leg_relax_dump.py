"""Issue #98: per-leg resolver relaxation audit (AnchorAudit / dump_leg_relaxations).

A leg grounded only via a dropped filter + same-label salience tie-break looks
identical to a genuinely-grounded leg in every other artifact -- these tests cover
the diagnostic-only trail that makes the two distinguishable, and the JSONL dump's
no-op-when-unset / append / bad-path contract (matching
core.perception.scene_index.dump_instance_index's tests)."""
from __future__ import annotations

import json

from core.heads.instruction import (
    ENV_LEG_RELAX_DUMP_PATH,
    AnchorAudit,
    InstructionHead,
    dump_leg_relaxations,
)
from core.plan_schema import Anchor, LegKind, RouteLeg
from tests.heads._helpers import inst, instruction_plan, scene


def _goto(noun: str, **kw) -> RouteLeg:
    return RouteLeg(kind=LegKind.GOTO, anchors=[Anchor(noun=noun, **kw)])


def _two_chair_scene():
    return scene(
        inst(1, "table", centroid=(0.0, 0.0, 0.0)),
        inst(2, "chair", centroid=(1.0, 0.0, 0.0)),
        inst(3, "chair", centroid=(5.0, 0.0, 0.0)),
    )


# --------------------------------------------------------------------------- AnchorAudit


def test_ranked_anchor_surfaces_relaxation_and_tie_break_size():
    """A never-matched attribute forces the relax_attributes rung; with the anchor's
    own disambiguator absent (the pre-#75 genuine-tie case), the same-label salience
    reorder runs over both survivors -- exactly the "arbitrary tie-break over N
    same-label candidates" issue #98 needs visible."""
    sc = _two_chair_scene()
    head = InstructionHead()
    anchor = Anchor(noun="chair", attributes=["red"])  # no chair has this attribute
    ranked, provisional, audit = head._ranked_anchor(anchor, sc, prev_xy=(0.0, 0.0))
    assert provisional is True
    assert audit.steps == ("relax_attributes",)
    assert audit.candidate_count == 2
    assert audit.tie_break_group_size == 2


def test_ranked_anchor_no_relaxation_when_clean_match():
    sc = _two_chair_scene()
    head = InstructionHead()
    anchor = Anchor(noun="table")
    ranked, provisional, audit = head._ranked_anchor(anchor, sc, prev_xy=None)
    assert provisional is False
    assert audit.steps == ()
    assert audit.tie_break_group_size is None


def test_grounded_leg_carries_one_audit_per_anchor():
    sc = _two_chair_scene()
    plan = instruction_plan(
        [_goto("table"), _goto("chair", attributes=["red"])]
    )
    head = InstructionHead(plan=plan)
    head._ground_legs(sc)
    assert len(head._legs) == 2
    assert len(head._legs[0].anchor_audits) == 1
    assert head._legs[0].anchor_audits[0].steps == ()
    leg2_audit = head._legs[1].anchor_audits[0]
    assert leg2_audit.steps == ("relax_attributes",)
    assert leg2_audit.tie_break_group_size == 2


# --------------------------------------------------------------------------- dump_leg_relaxations


def test_dump_is_noop_without_env_var(monkeypatch, tmp_path):
    monkeypatch.delenv(ENV_LEG_RELAX_DUMP_PATH, raising=False)
    sc = _two_chair_scene()
    plan = instruction_plan([_goto("table"), _goto("chair", attributes=["red"])])
    head = InstructionHead(plan=plan)
    head._ground_legs(sc)  # would dump if env var were set
    assert list(tmp_path.iterdir()) == []


def test_dump_writes_jsonl_record(monkeypatch, tmp_path):
    out = tmp_path / "leg_relax.jsonl"
    monkeypatch.setenv(ENV_LEG_RELAX_DUMP_PATH, str(out))
    sc = _two_chair_scene()
    plan = instruction_plan([_goto("table"), _goto("chair", attributes=["red"])])
    head = InstructionHead(plan=plan)
    head._ground_legs(sc)

    lines = out.read_text().strip().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["question_raw"] == plan.question_raw
    assert len(rec["legs"]) == 2
    leg0, leg1 = rec["legs"]
    assert leg0["kind"] == "goto"
    assert leg0["nouns"] == ["table"]
    assert leg0["anchors"][0]["relax_steps"] == []
    assert leg1["nouns"] == ["chair"]
    assert leg1["anchors"][0]["relax_steps"] == ["relax_attributes"]
    assert leg1["anchors"][0]["candidate_count"] == 2
    assert leg1["anchors"][0]["tie_break_group_size"] == 2
    # Issue #198: the pick itself (not just the resolve process) is recorded.
    assert leg0["picked_instance_id"] == 1
    assert leg0["goal_xy"] == [0.0, 0.0]
    assert leg1["picked_instance_id"] in (2, 3)
    assert leg1["goal_xy"] is not None


def test_dump_appends_across_calls(monkeypatch, tmp_path):
    out = tmp_path / "leg_relax.jsonl"
    monkeypatch.setenv(ENV_LEG_RELAX_DUMP_PATH, str(out))
    sc = _two_chair_scene()
    plan = instruction_plan([_goto("table")])
    head = InstructionHead(plan=plan)
    head._ground_legs(sc)
    head._ground_legs(sc)
    assert len(out.read_text().strip().splitlines()) == 2


def test_dump_never_raises_on_bad_path(monkeypatch):
    monkeypatch.setenv(ENV_LEG_RELAX_DUMP_PATH, "/dev/null/nonexistent/leg_relax.jsonl")
    sc = _two_chair_scene()
    plan = instruction_plan([_goto("table")])
    head = InstructionHead(plan=plan)
    head._ground_legs(sc)  # must not raise


def test_dump_helper_handles_none_plan_and_empty_legs(monkeypatch, tmp_path):
    out = tmp_path / "leg_relax.jsonl"
    monkeypatch.setenv(ENV_LEG_RELAX_DUMP_PATH, str(out))
    dump_leg_relaxations(None, [])
    rec = json.loads(out.read_text().strip())
    assert rec["question_raw"] is None
    assert rec["legs"] == []
