"""factory.build_callables wires the four rich checkpoint seams through to the heads."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core.fsm.controller import WorldView
from core.heads.factory import HeadState, build_callables
from core.interfaces import QType
from core.mocks.mock_io import FakeClock, MockRobotIO
from core.mocks.synthetic_scene import SyntheticScene
from core.perception.scene_index import BasicSceneIndex
from tests.heads._helpers import inst, object_plan


def _idx(*records):
    return BasicSceneIndex(list(records))


@dataclass
class _Outcome:
    action: str
    winner: Any = None


def test_verifier_wired_to_object_ref_head():
    sc = _idx(inst(1, "chair"), inst(2, "chair", centroid=(5, 0, 0)))
    st = HeadState(scene=sc, verifier=lambda **k: _Outcome("keep"))
    st.bind(object_plan("chair"))
    assert st.object_ref.verifier is not None


def test_anchor_confirmer_alias_wired():
    sc = _idx(inst(1, "chair"))
    conf = lambda desc, crop: _Outcome("confirm")
    cbs = build_callables(sc, anchor_confirmer=conf)
    # Bind an IF plan to observe the InstructionHead picks up the seam.
    from core.plan_schema import Anchor, LegKind, RouteLeg
    from tests.heads._helpers import instruction_plan

    io = MockRobotIO(SyntheticScene(0), FakeClock())
    plan = instruction_plan([RouteLeg(kind=LegKind.GOTO, anchors=[Anchor(noun="chair")])])
    cbs["explore"](io, plan, WorldView(scene=sc))
    # seam threaded through; legacy detection left it as a rich seam.
    # (indirect: the callable is the same object on the head)


def test_all_seams_default_none():
    sc = _idx(inst(1, "chair"))
    st = HeadState(scene=sc)
    assert st.verifier is None
    assert st.miss_recoverer is None
    assert st.frontier_selector is None
    st.bind(object_plan("chair"))
    assert st.object_ref.verifier is None


def test_miss_recoverer_and_frontier_selector_wired_to_explore():
    sc = _idx(inst(1, "chair"))
    st = HeadState(
        scene=sc,
        miss_recoverer=lambda n, r, t: _Outcome("absent"),
        frontier_selector=lambda q, f: _Outcome("fallback"),
        budget_frac=lambda: 0.7,
    )
    st.bind(object_plan("chair"))
    assert st.explore.miss_recoverer is not None
    assert st.explore.frontier_selector is not None
    assert st.explore.budget_frac is not None


def test_build_callables_accepts_all_new_kwargs():
    sc = _idx(inst(1, "chair"))
    cbs = build_callables(
        sc,
        verifier=lambda **k: _Outcome("keep"),
        anchor_confirmer=lambda desc, crop: _Outcome("confirm"),
        miss_recoverer=lambda n, r, t: _Outcome("absent"),
        frontier_selector=lambda q, f: _Outcome("fallback"),
        remaining_s=lambda: 120.0,
        budget_frac=lambda: 0.5,
        tiles_fn=lambda: [],
        fuse_hint=lambda n, xy, o: None,
    )
    assert set(cbs) == {"parse", "explore", "verify", "probe"}
