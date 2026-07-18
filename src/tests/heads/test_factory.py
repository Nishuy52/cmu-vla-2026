"""build_callables: shape, plan binding, WorldView assembly, default parse tier."""
from __future__ import annotations

from core.fsm.controller import QuestionController, WorldView
from core.heads.factory import HeadState, build_callables
from core.interfaces import IntAnswer, MarkerBox, Question, QType
from core.mocks.mock_io import FakeClock, MockRobotIO
from core.mocks.synthetic_scene import SyntheticScene
from core.perception.scene_index import BasicSceneIndex
from tests.heads._helpers import inst, near_clause, numerical_plan, object_plan, scene


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


def test_explore_withholds_count_on_empty_index():
    # Perception dark: zero tracked instances overall is absence of data, not an
    # observed zero. partial.count must stay None so the FSM floor's modal count wins
    # instead of a fabricated 0 (core/fsm/floors.py's designed fallback).
    sc = _idx()  # empty index
    cbs = build_callables(sc)
    io = MockRobotIO(SyntheticScene(0), FakeClock())
    plan = numerical_plan("chair")
    cbs["explore"](io, plan, WorldView(scene=sc))
    wv = cbs["probe"](io)
    assert wv.partial.count is None


def test_verify_returns_none_for_numerical_on_empty_index():
    # The verify callable's contract is "final publishable answer, or None" (controller.py)
    # — on an empty index, numerical must return None so the controller falls back to the
    # FloorAnswers modal count rather than publishing a fabricated 0.
    sc = _idx()  # empty index
    cbs = build_callables(sc)
    io = MockRobotIO(SyntheticScene(0), FakeClock())
    plan = numerical_plan("chair")
    cbs["explore"](io, plan, WorldView(scene=sc))
    ans = cbs["verify"](io, plan, WorldView(scene=sc))
    assert ans is None


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


# ------------------------------------------------------------------ detector prompt refresh (issue #34)
#
# GroundingDinoDetector is constructed at boot (adapter) with no question latched yet, so
# its prompt stays "" and every __call__ short-circuits to zero detections forever unless
# something rebuilds the prompt once the question's nouns are known. HeadState.bind is the
# one seam both the adapter (ros_adapter/adapter_node.py) and the offline replay path
# (core/runner/single.py -> build_callables) drive identically — these tests exercise that
# shared seam directly, standing in for both callers.


def test_headstate_bind_refreshes_detector_prompt_from_plan_nouns():
    from core.perception.detector import FakeDetector

    fake = FakeDetector()
    st = HeadState(scene=_idx(), detector=fake)
    assert fake.prompt == ""
    st.bind(numerical_plan("chair", [near_clause("sofa")]))
    # question nouns (target + clause anchors) come first, query-relevant recall priority.
    assert fake.prompt.startswith("chair . sofa .")
    # then the standing vocab (deduped: "sofa" isn't repeated) rides along at lower priority.
    assert "lamp" in fake.prompt


def test_headstate_bind_detector_prompt_refreshed_exactly_once():
    from core.perception.detector import FakeDetector

    fake = FakeDetector()
    st = HeadState(scene=_idx(), detector=fake)
    st.bind(numerical_plan("chair"))
    first_prompt = fake.prompt
    fake.prompt = "tampered"  # simulate something else touching it between binds
    st.bind(object_plan("table"))  # second bind is a no-op (plan already latched)
    assert fake.prompt == "tampered"  # NOT re-refreshed to "table ..."
    assert first_prompt.startswith("chair .")


def test_headstate_bind_with_no_detector_is_a_noop():
    st = HeadState(scene=_idx(), detector=None)
    st.bind(numerical_plan("chair"))  # must not raise
    assert st.plan is not None


def test_build_callables_wires_detector_through_explore():
    from core.perception.detector import FakeDetector

    fake = FakeDetector()
    sc = _idx(inst(1, "chair"))
    cbs = build_callables(sc, detector=fake)
    io = MockRobotIO(SyntheticScene(0), FakeClock())
    assert fake.prompt == ""
    cbs["explore"](io, numerical_plan("chair", [near_clause("window")]), WorldView(scene=sc))
    assert fake.prompt.startswith("chair . window .")


def test_build_callables_wires_detector_through_verify():
    from core.perception.detector import FakeDetector

    fake = FakeDetector()
    sc = _idx(inst(1, "table"))
    cbs = build_callables(sc, detector=fake)
    io = MockRobotIO(SyntheticScene(0), FakeClock())
    assert fake.prompt == ""
    cbs["verify"](io, object_plan("table"), WorldView(scene=sc))
    assert fake.prompt.startswith("table .")


def test_detector_prompt_refresh_feeds_the_live_perception_pipeline():
    """The exact detector instance a PerceptionPipeline calls every keyframe is the one
    whose prompt gets refreshed on plan latch — the fix targets the live shared object,
    not a disconnected copy. Mirrors both the adapter (detector shared with
    self._perception) and the replay path (a detector could equally back a
    _ScriptedPerception-style pipeline)."""
    import numpy as np

    from core.interfaces import LidarScan, OdomState, PanoFrame
    from core.perception import tiling as T
    from core.perception.detector import Detection, FakeDetector
    from core.perception.tracker import KeyframeConfig, PerceptionPipeline

    spec = T.tile_specs()[0]
    det = Detection(
        tile_id=0,
        bbox_xyxy=(spec.cx - 60, spec.cy - 120, spec.cx + 60, spec.cy + 60),
        label="sofa",
        score=0.9,
    )
    fake = FakeDetector([det])
    sc = _idx()
    pipe = PerceptionPipeline(fake, index=sc, keyframe_cfg=KeyframeConfig(every_k=1))

    cbs = build_callables(sc, detector=fake)
    io = MockRobotIO(SyntheticScene(0), FakeClock())
    # Latch the plan FIRST (as the FSM would tick explore before/alongside perception) —
    # this refreshes the SAME `fake` instance `pipe` is about to call.
    cbs["explore"](io, numerical_plan("sofa"), WorldView(scene=sc))
    assert fake.prompt.startswith("sofa .")

    rng = np.random.default_rng(0)
    cloud = np.column_stack([
        3.0 + rng.uniform(-0.2, 0.2, 40),
        0.0 + rng.uniform(-0.2, 0.2, 40),
        0.5 + rng.uniform(-0.2, 0.2, 40),
    ]).astype(np.float32)
    pano = PanoFrame(
        t=0.0,
        image=np.zeros((T.PANO_HEIGHT, T.PANO_WIDTH, 3), dtype=np.uint8),
        odom=OdomState(t=0.0, x=0.0, y=0.0, z=0.0, yaw=0.0),
    )
    pipe.process(pano, LidarScan(t=0.0, points=cloud))

    assert len(sc.all_instances()) == 1
    assert sc.all_instances()[0].label == "sofa"
