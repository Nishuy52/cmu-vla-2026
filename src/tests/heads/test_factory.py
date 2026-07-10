"""build_callables: shape, plan binding, WorldView assembly, default parse tier."""
from __future__ import annotations

from core.fsm.controller import QuestionController, WorldView
from core.heads.factory import HeadState, build_callables
from core.interfaces import IntAnswer, MarkerBox, Question, QType
from core.mocks.mock_io import FakeClock, MockRobotIO
from core.mocks.synthetic_scene import SyntheticScene
from core.perception.scene_index import BasicSceneIndex
from tests.heads._helpers import inst, numerical_plan, object_plan, scene


def _idx(*records):
    return BasicSceneIndex(list(records))


def test_returns_four_callables():
    cbs = build_callables(_idx(inst(1, "chair")))
    assert set(cbs) == {"parse", "explore", "verify", "probe"}
    # constructible into a controller
    QuestionController(**cbs)


def test_default_parse_is_regex_tier():
    cbs = build_callables(_idx(inst(1, "chair")))
    plan = cbs["parse"](Question(text="how many chairs are there", t_received=0.0))
    assert plan is not None
    assert plan.qtype is QType.NUMERICAL
    assert plan.parse_tier == "regex"


def test_injected_parse_overrides_default():
    marker = numerical_plan("chair")
    cbs = build_callables(_idx(inst(1, "chair")), parse=lambda q: marker)
    assert cbs["parse"](Question(text="x", t_received=0.0)) is marker


def test_explore_binds_plan_and_advances_numerical():
    sc = _idx(inst(1, "chair"), inst(2, "chair"))
    cbs = build_callables(sc)
    io = MockRobotIO(SyntheticScene(0), FakeClock())
    plan = numerical_plan("chair")
    cbs["explore"](io, plan, WorldView(scene=sc))
    wv = cbs["probe"](io)
    assert wv.partial.count == 2
    assert wv.scene is sc


def test_probe_assembles_stability_signal():
    sc = _idx(inst(1, "chair"), inst(2, "chair"))
    cbs = build_callables(sc)
    io = MockRobotIO(SyntheticScene(0), FakeClock())
    plan = numerical_plan("chair")
    for _ in range(5):  # let the stability run build up
        cbs["explore"](io, plan, WorldView(scene=sc))
    wv = cbs["probe"](io)
    assert wv.stability.stable is True


def test_verify_returns_intanswer_for_numerical():
    sc = _idx(inst(1, "chair"), inst(2, "chair"))
    cbs = build_callables(sc)
    io = MockRobotIO(SyntheticScene(0), FakeClock())
    ans = cbs["verify"](io, numerical_plan("chair"), WorldView(scene=sc))
    assert ans == IntAnswer(2)


def test_verify_returns_marker_for_object_reference():
    sc = _idx(inst(1, "table", centroid=(1.0, 1.0, 0.0)))
    cbs = build_callables(sc)
    io = MockRobotIO(SyntheticScene(0), FakeClock())
    ans = cbs["verify"](io, object_plan("table"), WorldView(scene=sc))
    assert isinstance(ans, MarkerBox)


def test_probe_reports_ungrounded_for_instruction():
    from core.plan_schema import Anchor, LegKind, RouteLeg
    from tests.heads._helpers import instruction_plan

    sc = _idx(inst(1, "table"))
    cbs = build_callables(sc)
    io = MockRobotIO(SyntheticScene(0), FakeClock())
    plan = instruction_plan([RouteLeg(kind=LegKind.GOTO, anchors=[Anchor(noun="unicorn")])])
    cbs["explore"](io, plan, WorldView(scene=sc))
    wv = cbs["probe"](io)
    assert wv.ungrounded_subgoals == 1


def test_headstate_binds_once():
    sc = _idx(inst(1, "chair"))
    st = HeadState(scene=sc)
    p1 = numerical_plan("chair")
    st.bind(p1)
    st.bind(object_plan("table"))  # second bind ignored
    assert st.plan is p1
    assert st.numerical is not None
    assert st.object_ref is None


def test_object_reference_llm_verify_wired():
    sc = _idx(inst(1, "chair"), inst(2, "chair", centroid=(5, 0, 0)))
    calls = {"n": 0}

    def verify(plan, summary, matrix):
        calls["n"] += 1
        return True

    cbs = build_callables(sc, llm_verify=verify)
    io = MockRobotIO(SyntheticScene(0), FakeClock())
    cbs["verify"](io, object_plan("chair"), WorldView(scene=sc))
    assert calls["n"] >= 1
