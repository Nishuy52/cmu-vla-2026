"""Issue #200 "detector cold start" — perception must observe the opening orientation
sweep, not just the post-latch portion of the run.

The mechanism (see the #118 8 Aug comment and #200's own body): every zero-detection
keyframe of a run lay in one unbroken prefix at the START of the run. The
PerceptionPipeline was already fed (pano, scan) pairs every tick, independent of the
QuestionController (see ``_maybe_process_perception``'s own docstring) — but the LIVE
detector instance it calls was constructed with an empty ``.prompt`` (see
``core.perception.detector.refresh_prompt``'s docstring: an empty prompt makes every
``__call__`` short-circuit to zero detections). Before this fix, the first refresh only
happened inside ``build_callables()`` (core/heads/factory.py), and
``_build_controller_callables`` only calls that once the QuestionController is built AT
QUESTION LATCH — so every keyframe ticked between node start and latch ran blind.

Two test styles, matching this suite's convention (test_seam_wiring.py):

* Source inspection (always runs, no rclpy needed): the adapter's ``__init__`` now primes
  the detector's prompt with the standing vocabulary immediately at construction, BEFORE
  the 5 Hz timer (and therefore before any tick, controller, or question) exists.
* Behavioural (always runs — core is rclpy-free): replays the adapter's exact new
  sequence — construct detector -> construct PerceptionPipeline -> prime boot vocab ->
  process frames with no question latched -> build_callables at "latch" -> bind a plan —
  entirely through core.heads.factory / core.perception.tracker, proving the four
  validation points end to end without needing rclpy.
"""
from __future__ import annotations

import pathlib

import numpy as np
import pytest

from core.fsm.controller import WorldView
from core.heads.factory import _STANDING_VOCAB_NOUNS, build_callables
from core.interfaces import LidarScan, OdomState, PanoFrame
from core.mocks.mock_io import FakeClock, MockRobotIO
from core.mocks.synthetic_scene import SyntheticScene
from core.perception import tiling as T
from core.perception.detector import Detection, FakeDetector, refresh_prompt
from core.perception.scene_index import BasicSceneIndex
from core.perception.tracker import KeyframeConfig, PerceptionPipeline
from tests.heads._helpers import numerical_plan

_ADAPTER = pathlib.Path(__file__).resolve().parents[2] / "ros_adapter" / "adapter_node.py"


@pytest.fixture(scope="module")
def src() -> str:
    return _ADAPTER.read_text(encoding="utf-8")


def _pano_with_sofa_cluster(t: float) -> tuple[PanoFrame, LidarScan]:
    """A synthetic (pano, scan) pair a keyframe gate accepts and fusion can cluster."""
    rng = np.random.default_rng(0)
    cloud = np.column_stack(
        [
            3.0 + rng.uniform(-0.2, 0.2, 40),
            0.0 + rng.uniform(-0.2, 0.2, 40),
            0.5 + rng.uniform(-0.2, 0.2, 40),
        ]
    ).astype(np.float32)
    pano = PanoFrame(
        t=t,
        image=np.zeros((T.PANO_HEIGHT, T.PANO_WIDTH, 3), dtype=np.uint8),
        odom=OdomState(t=t, x=0.0, y=0.0, z=0.0, yaw=0.0),
    )
    scan = LidarScan(t=t, points=cloud)
    return pano, scan


# ------------------------------------------------------------- source inspection (adapter)


def test_boot_vocab_priming_wired_in_init(src: str):
    # The fix: prime the SAME live detector instance the pipeline calls, with the standing
    # vocabulary, right where the pipeline is constructed — not deferred to build_callables
    # at question latch.
    assert "from core.heads.factory import _STANDING_VOCAB_NOUNS" in src
    assert "from core.perception.detector import GroundingDinoDetector, refresh_prompt" in src
    assert "refresh_prompt(self._detector, (), _STANDING_VOCAB_NOUNS)" in src


def test_boot_vocab_priming_happens_before_the_drive_timer(src: str):
    # The 5 Hz timer is what starts ticking (and therefore starts the controller-latch
    # path); the priming call must land in __init__ strictly before that timer is created,
    # so it has already run by the time the very first tick fires.
    prime_at = src.index("refresh_prompt(self._detector, (), _STANDING_VOCAB_NOUNS)")
    timer_at = src.index('self._timer = self.create_timer(1.0 / TICK_HZ, self._on_tick)')
    assert prime_at < timer_at


def test_perception_dispatch_precedes_controller_construction_in_tick(src: str):
    # Requirement 1/3: frames must be processed independent of (before) the controller/
    # question-latch check, every tick, from the first tick onward.
    tick_start = src.index("def _on_tick(self)")
    tick_body = src[tick_start : src.index("def _build_controller_callables")]
    perceive_at = tick_body.index("self._maybe_process_perception()")
    controller_check_at = tick_body.index("if self._controller is None:")
    assert perceive_at < controller_check_at


def test_scene_index_never_recreated_after_init(src: str):
    # Requirement 1 (second half): the early (pre-latch) index must persist into the
    # post-latch controller, not be thrown away. self._scene_index is assigned exactly
    # twice in __init__ (perception-on / perception-off branches) and nowhere else —
    # in particular never re-assigned around question latch or controller construction.
    init_start = src.index("def __init__(self) -> None:")
    init_end = src.index("# ------------------------------------------------------------------ callbacks")
    body = src[init_start:init_end]
    assert body.count("self._scene_index = ") == 2
    rest = src[init_end:]
    assert "self._scene_index = " not in rest


def test_detector_prompt_refresh_still_wired_at_latch(src: str):
    # Requirement 2: build_callables (called once, at controller construction / question
    # latch) must still receive the live detector so HeadState.bind can refresh it from
    # the question's nouns (+ #173 avoid nouns) the moment the plan latches — unchanged.
    assert src.count("detector=self._detector") >= 2


# ------------------------------------------------------------------- behavioural (core)


def test_boot_priming_grounds_detector_before_any_question():
    """Mirrors AdapterNode.__init__: prime the detector the instant the pipeline exists,
    with no question, no controller, nothing latched yet."""
    fake = FakeDetector()
    sc = BasicSceneIndex([])
    PerceptionPipeline(fake, index=sc)  # constructed at "node start", same as the adapter
    assert fake.prompt == ""

    refresh_prompt(fake, (), _STANDING_VOCAB_NOUNS)

    assert fake.prompt != ""
    assert "lamp" in fake.prompt  # a standing-vocab noun, unrelated to any question


def test_frames_before_a_question_are_processed_into_the_index():
    """Mirrors _maybe_process_perception being driven from the first tick, independent of
    self._controller / self.question() — i.e. exactly the opening orientation sweep."""
    spec = T.tile_specs()[0]
    det = Detection(
        tile_id=0,
        bbox_xyxy=(spec.cx - 60, spec.cy - 120, spec.cx + 60, spec.cy + 60),
        label="sofa",
        score=0.9,
    )
    fake = FakeDetector([det])
    sc = BasicSceneIndex([])
    pipe = PerceptionPipeline(fake, index=sc, keyframe_cfg=KeyframeConfig(every_k=1))

    # Boot-vocab priming (the fix), then frames arrive with NO question latched at all.
    refresh_prompt(fake, (), _STANDING_VOCAB_NOUNS)
    for i in range(3):
        pano, scan = _pano_with_sofa_cluster(float(i))
        pipe.process(pano, scan)

    assert sc.all_instances(), "sweep frames observed before latch must reach the index"
    assert any(inst.label == "sofa" for inst in sc.all_instances())


def test_index_survives_the_latch():
    """The pre-latch instances must still be present in the SAME index object
    build_callables (controller construction) is handed — the early index must not be
    thrown away when the question arrives."""
    spec = T.tile_specs()[0]
    det = Detection(
        tile_id=0,
        bbox_xyxy=(spec.cx - 60, spec.cy - 120, spec.cx + 60, spec.cy + 60),
        label="sofa",
        score=0.9,
    )
    fake = FakeDetector([det])
    sc = BasicSceneIndex([])
    pipe = PerceptionPipeline(fake, index=sc, keyframe_cfg=KeyframeConfig(every_k=1))
    refresh_prompt(fake, (), _STANDING_VOCAB_NOUNS)
    pano, scan = _pano_with_sofa_cluster(0.0)
    pipe.process(pano, scan)
    pre_latch_count = len(sc.all_instances())
    assert pre_latch_count > 0

    # "Question latches": the controller is built now, handing the SAME index through.
    cbs = build_callables(sc, detector=fake)

    assert len(sc.all_instances()) == pre_latch_count  # nothing discarded
    assert sc.all_instances()[0].label == "sofa"
    # And the pipeline keeps writing into that identical object after latch too.
    pano2, scan2 = _pano_with_sofa_cluster(1.0)
    pipe.process(pano2, scan2)
    assert sc is pipe.index  # identity preserved throughout


def test_prompt_refresh_still_fires_exactly_at_latch():
    """Requirement 2: after boot-vocab priming, the question latch must still refresh the
    prompt to the question's own nouns (#173: avoid nouns route into the short caption
    too), and later frames use it — unchanged from today's behaviour."""
    fake = FakeDetector()
    sc = BasicSceneIndex([])
    PerceptionPipeline(fake, index=sc)
    refresh_prompt(fake, (), _STANDING_VOCAB_NOUNS)
    primed_prompt = fake.prompt
    assert primed_prompt != ""
    assert not primed_prompt.startswith("chair .")

    cbs = build_callables(sc, detector=fake)
    # build_callables' own re-prime (unchanged) leaves the prompt on the boot vocabulary
    # until a plan actually latches.
    assert "chair" not in fake.prompt.split(" . ")[0]

    io = MockRobotIO(SyntheticScene(0), FakeClock())
    cbs["explore"](io, numerical_plan("chair"), WorldView(scene=sc))

    assert fake.prompt.startswith("chair .")  # question noun now leads the caption
    assert "lamp" in fake.prompt  # standing vocab still rides along at lower priority
