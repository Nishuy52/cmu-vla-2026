"""Issue #102: resolved-Plan dump at answer time (dump_plan / ENV_PLAN_DUMP_PATH).

Follows the no-op-when-unset / append / bad-path contract established by
core.perception.scene_index.dump_instance_index's own tests."""
from __future__ import annotations

import json

from core.heads.factory import ENV_PLAN_DUMP_PATH, HeadState, build_callables, dump_plan
from core.interfaces import QType
from core.mocks.mock_io import FakeClock, MockRobotIO
from core.mocks.synthetic_scene import SyntheticScene
from core.plan_schema import Anchor, Clause, Plan, Pred, TargetSpec
from core.fsm.controller import WorldView
from tests.heads._helpers import inst, near_clause, numerical_plan, object_plan, scene


def _state_for(plan: Plan, sc) -> HeadState:
    state = HeadState(scene=sc)
    state.bind(plan)
    return state


# --------------------------------------------------------------------------- no-op contract


def test_dump_plan_is_noop_without_env_var(monkeypatch, tmp_path):
    monkeypatch.delenv(ENV_PLAN_DUMP_PATH, raising=False)
    sc = scene(inst(1, "chair"))
    state = _state_for(numerical_plan("chair"), sc)
    dump_plan(state)
    assert list(tmp_path.iterdir()) == []


def test_dump_plan_is_noop_without_plan(monkeypatch, tmp_path):
    out = tmp_path / "plan.jsonl"
    monkeypatch.setenv(ENV_PLAN_DUMP_PATH, str(out))
    state = HeadState(scene=scene(inst(1, "chair")))  # never bound
    dump_plan(state)
    assert not out.exists()


def test_dump_plan_never_raises_on_bad_path(monkeypatch):
    monkeypatch.setenv(ENV_PLAN_DUMP_PATH, "/dev/null/nonexistent/plan.jsonl")
    sc = scene(inst(1, "chair"))
    state = _state_for(numerical_plan("chair"), sc)
    dump_plan(state)  # must not raise


def test_dump_plan_appends_across_calls(monkeypatch, tmp_path):
    out = tmp_path / "plan.jsonl"
    monkeypatch.setenv(ENV_PLAN_DUMP_PATH, str(out))
    sc = scene(inst(1, "chair"))
    state = _state_for(numerical_plan("chair"), sc)
    dump_plan(state)
    dump_plan(state)
    assert len(out.read_text().strip().splitlines()) == 2


# --------------------------------------------------------------------------- schema / recursion


def test_dump_plan_walks_nested_disambiguator():
    """The motivating recursion case: a Clause's Anchor carries its own nested
    disambiguator Clause ('the chair near the lamp on the table')."""
    inner = Clause(pred=Pred.ON, anchors=[Anchor(noun="table")])
    outer_anchor = Anchor(noun="lamp", disambiguator=inner)
    outer = Clause(pred=Pred.NEAR, anchors=[outer_anchor])
    plan = object_plan("chair", clauses=[outer])
    sc = scene(inst(1, "chair"), inst(2, "lamp"), inst(3, "table"))
    d = None
    from core.heads.factory import _plan_to_dict

    d = _plan_to_dict(plan)
    clause0 = d["target"]["clauses"][0]
    assert clause0["pred"] == "near"
    anchor0 = clause0["anchors"][0]
    assert anchor0["noun"] == "lamp"
    nested = anchor0["disambiguator"]
    assert nested is not None
    assert nested["pred"] == "on"
    assert nested["anchors"][0]["noun"] == "table"
    assert nested["anchors"][0]["disambiguator"] is None


def test_dump_plan_depth_cap_guards_pathological_nesting():
    """A chain of nested disambiguators deeper than _PLAN_DUMP_MAX_DEPTH terminates
    with a sentinel instead of recursing unbounded (issue #102's cyclic-nesting guard)."""
    from core.heads.factory import _PLAN_DUMP_MAX_DEPTH, _plan_to_dict

    clause = Clause(pred=Pred.NEAR, anchors=[Anchor(noun="leaf0")])
    for i in range(1, _PLAN_DUMP_MAX_DEPTH + 10):
        anchor = Anchor(noun=f"leaf{i}", disambiguator=clause)
        clause = Clause(pred=Pred.NEAR, anchors=[anchor])
    plan = object_plan("chair", clauses=[clause])
    d = _plan_to_dict(plan)  # must terminate, not stack-overflow / hang

    node = d["target"]["clauses"][0]
    depth = 0
    while True:
        anchor0 = node["anchors"][0]
        disamb = anchor0["disambiguator"]
        if disamb == "<max-depth-exceeded>":
            break
        assert disamb is not None
        node = disamb
        depth += 1
        assert depth < _PLAN_DUMP_MAX_DEPTH + 10  # sanity: loop must terminate


def test_dump_plan_writes_route_and_avoid_for_instruction_following():
    from core.plan_schema import AvoidSpec, LegKind, RouteLeg

    leg = RouteLeg(kind=LegKind.GOTO, anchors=[Anchor(noun="sofa")])
    avoid = AvoidSpec(near=Anchor(noun="vase"))
    plan = Plan(
        qtype=QType.INSTRUCTION_FOLLOWING,
        question_raw="go to the sofa avoiding the vase",
        route=[leg],
        avoid=[avoid],
    )
    from core.heads.factory import _plan_to_dict

    d = _plan_to_dict(plan)
    assert d["qtype"] == "instruction_following"
    assert d["route"][0]["kind"] == "goto"
    assert d["route"][0]["anchors"][0]["noun"] == "sofa"
    assert d["avoid"][0]["near"]["noun"] == "vase"
    assert d["avoid"][0]["between"] is None


# --------------------------------------------------------------------------- candidate counts


def test_dump_plan_numerical_aggregate_candidate_counts(monkeypatch, tmp_path):
    out = tmp_path / "plan.jsonl"
    monkeypatch.setenv(ENV_PLAN_DUMP_PATH, str(out))
    sc = scene(inst(1, "chair"), inst(2, "chair"), inst(3, "chair"))
    plan = numerical_plan("chair", clauses=[near_clause("table")])
    state = _state_for(plan, sc)
    state.numerical.advance(sc)  # populate .count
    dump_plan(state)

    rec = json.loads(out.read_text().strip())
    cand = rec["candidates"]
    assert cand["noun"] == "chair"
    assert cand["before_any_filter"] == 3
    # no table in scene -> the near(table) clause can never be satisfied -> 0
    assert cand["after_all_filters"] == 0
    assert cand["granularity"] == "whole_target_aggregate"


def test_dump_plan_object_reference_reports_relaxation_audit(monkeypatch, tmp_path):
    out = tmp_path / "plan.jsonl"
    monkeypatch.setenv(ENV_PLAN_DUMP_PATH, str(out))
    # near(table) clause: no table in scene -> resolve() relaxes down to category_only.
    sc = scene(inst(1, "chair"), inst(2, "chair"))
    plan = object_plan("chair", clauses=[near_clause("table")])
    state = _state_for(plan, sc)
    state.object_ref.advance(sc)
    dump_plan(state)

    rec = json.loads(out.read_text().strip())
    cand = rec["candidates"]
    assert cand["before_any_filter"] == 2
    assert cand["after_all_filters"] == 2  # category_only fell back to the whole class
    steps = [a["step"] for a in cand["relaxation_audit"]]
    assert "category_only" in steps


def test_dump_plan_instruction_following_has_no_candidates_block(monkeypatch, tmp_path):
    from core.plan_schema import LegKind, RouteLeg

    out = tmp_path / "plan.jsonl"
    monkeypatch.setenv(ENV_PLAN_DUMP_PATH, str(out))
    sc = scene(inst(1, "sofa"))
    plan = Plan(
        qtype=QType.INSTRUCTION_FOLLOWING,
        question_raw="go to the sofa",
        route=[RouteLeg(kind=LegKind.GOTO, anchors=[Anchor(noun="sofa")])],
    )
    state = _state_for(plan, sc)
    dump_plan(state)

    rec = json.loads(out.read_text().strip())
    assert "candidates" not in rec  # no target on an INSTRUCTION_FOLLOWING plan


# --------------------------------------------------------------------------- wired into verify()


def test_final_answer_dumps_plan_when_env_set(monkeypatch, tmp_path):
    out = tmp_path / "plan.jsonl"
    monkeypatch.setenv(ENV_PLAN_DUMP_PATH, str(out))
    sc = scene(inst(1, "chair"), inst(2, "chair"))
    cbs = build_callables(sc)
    io = MockRobotIO(SyntheticScene(0), FakeClock())
    ans = cbs["verify"](io, numerical_plan("chair"), WorldView(scene=sc))
    assert ans is not None
    rec = json.loads(out.read_text().strip())
    assert rec["tag"] == "answer_time"
    assert rec["plan"]["target"]["noun"] == "chair"


def test_final_answer_does_not_dump_plan_when_env_unset(monkeypatch, tmp_path):
    monkeypatch.delenv(ENV_PLAN_DUMP_PATH, raising=False)
    sc = scene(inst(1, "chair"), inst(2, "chair"))
    cbs = build_callables(sc)
    io = MockRobotIO(SyntheticScene(0), FakeClock())
    cbs["verify"](io, numerical_plan("chair"), WorldView(scene=sc))
    assert list(tmp_path.iterdir()) == []
