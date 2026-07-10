"""Run one question end-to-end through the full pipeline and report.

:func:`run_question` constructs the scene index + heads + controller, injects the
question into the io, then ticks the :class:`~core.fsm.controller.QuestionController`
on the io's *simulated* clock until it reaches DONE (or a wall-clock guard trips).
Because the clock is simulated, a question whose sim budget is the full 600 s completes
in far under a second of wall time.

Works with :class:`~core.mocks.mock_io.MockRobotIO` (synthetic scene) and
:class:`~core.replay.replay_io.ReplayRobotIO` (fixtures / bags). The scene index the
heads resolve against is derived from the io when it exposes a synthetic scene
(``io.scene``); otherwise an explicit ``scene_index`` may be passed, and failing that an
empty index is used (the FSM floor still guarantees a legal, never-silent answer).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import bisect

from core.fsm.controller import QuestionController, State
from core.fsm.events import LogRecord
from core.heads import build_callables
from core.interfaces import IntAnswer, MarkerBox, QType, SceneIndex, WaypointCmd
from core.perception.scene_index import BasicSceneIndex


@dataclass
class RunResult:
    """Structured outcome of a single :func:`run_question` drive.

    answer:            the FSM's *published* answer object (IntAnswer/MarkerBox/WaypointCmd)
                       for the qtype — never an exploration waypoint — or None if the run
                       ended without the controller latching an answer.
    qtype:             the qtype the controller resolved for the question.
    elapsed_sim_s:     simulated seconds elapsed on the injected clock at DONE.
    states_visited:    ordered list of distinct FSM states the run passed through.
    checkpoint_calls:  count of injected-checkpoint calls (parse/verify) logged by the FSM.
    flight_log:        the controller's dumped flight recording (list of LogRecord).
    published:         True iff exactly one legal answer reached a RobotIO sink via the FSM.
    end_of_data_at_sim_s: sim time at which the replay ran out of recorded messages and
                       began free-running the frozen world (None for live/synthetic runs
                       that never exhaust their schedule).
    floor_used:        True iff the published answer came from the floor / watchdog path
                       rather than a head verify (wired to the flight-recorder events).
    instances_tracked: number of 3D instances in the scene index at the end of the run.
                       0 for synthetic runs and for replays without ``detections_path``
                       (the heads then resolve against the io's own / an empty index);
                       for a scripted-detection replay this is the count of real
                       lidar-fused instances the PerceptionPipeline accumulated.
    """

    answer: Any | None
    qtype: QType | None
    elapsed_sim_s: float
    states_visited: list[str]
    checkpoint_calls: int
    flight_log: list[LogRecord]
    published: bool
    ticks: int = 0
    wall_s: float = 0.0
    end_of_data_at_sim_s: float | None = None
    floor_used: bool = False
    instances_tracked: int = 0


def _clock_advance(clock: Any, dt: float) -> None:
    """Advance a clock one tick, tolerating both FakeClock (advance) and ReplayClock (step)."""
    if hasattr(clock, "advance"):
        clock.advance(dt)
    elif hasattr(clock, "step"):
        clock.step()
    elif hasattr(clock, "set"):
        clock.set(clock.now() + dt)


def _inject_question(io: Any, text: str) -> None:
    """Latch the question onto the io the way the ROS 1 Hz republish would."""
    if hasattr(io, "set_question"):
        io.set_question(text)
        return
    # ReplayRobotIO / fixtures already carry a latched question on the store; if a text was
    # supplied and the store has none, seed one at the current clock time.
    if io.question() is None:
        from core.interfaces import Question

        store = getattr(io, "store", None)
        if store is not None:
            from core.replay.replay_io import CH_QUESTION

            now = io.clock().now()
            store.add(CH_QUESTION, now, Question(text=text, t_received=now))
            store.finalize()


def _derive_scene_index(io: Any, scene_index: SceneIndex | None) -> SceneIndex:
    """Pick the SceneIndex the heads resolve against: explicit > io.scene > empty."""
    if scene_index is not None:
        return scene_index
    scene = getattr(io, "scene", None)
    if scene is not None and hasattr(scene, "instances"):
        return BasicSceneIndex(scene.instances())
    return BasicSceneIndex([])


def _floor_used(flight_log: list[LogRecord]) -> bool:
    """True iff the published answer came from the floor / watchdog path, not a head verify.

    Reads the flight-recorder events (the cleanest available signal): the answer-state
    floor fallbacks log ``answer_from_floor``, and the watchdog floor publish logs a
    ``published`` event whose detail names the watchdog.
    """
    for rec in flight_log:
        ev = getattr(rec, "event", "")
        if ev == "answer_from_floor":
            return True
        if ev == "published" and "watchdog" in getattr(rec, "detail", ""):
            return True
    return False


class _ScaledClock:
    """Reads the true bag clock but reports elapsed time amplified by ``1 / budget_scale``.

    The three budget gates (per-type explore budgets, forced-assembly 510 s, watchdog
    floor 570 s) are module constants in ``core.fsm.budget`` / ``core.interfaces`` with no
    constructor seam — and ``core/fsm`` is off-limits here. So rather than inject scaled
    constants, we scale *time itself*: this wrapper amplifies elapsed sim-time by
    ``1 / budget_scale`` as the FSM sees it, so a gate at G seconds is crossed once the
    underlying (bag) clock has advanced only ``G * budget_scale`` seconds. E.g.
    ``budget_scale=0.2`` puts the 510/570 s gates at ~102/114 s of bag time — enough to
    fire against a short (~30–120 s) bag.

    Only elapsed time past the reference point (``t0`` at construction, i.e. current bag
    time when scaling is enabled) is amplified. This is the wrapper the io hands to the
    FSM via ``io.set_budget_clock``; the replay getters keep reading the true bag clock,
    so message lookups stay on real recorded time.
    """

    def __init__(self, inner: Any, scale: float) -> None:
        self._inner = inner
        self._scale = float(scale)
        self._t0 = float(inner.now())

    def now(self) -> float:
        raw = float(self._inner.now())
        return self._t0 + (raw - self._t0) / self._scale


def _published_answer(io: Any, published_ref: dict) -> tuple[Any | None, bool]:
    """Return the FSM's latched answer and whether one was published.

    ``published_ref`` is captured during the drive loop the moment the controller flips
    ``answer_published`` True — it holds the exact object the FSM dispatched to a sink, so
    an exploration waypoint written to ``io.waypoints`` is never mistaken for the answer.
    """
    if published_ref.get("published"):
        return published_ref.get("answer"), True
    return None, False


_SINK_NAMES = ("ints", "markers", "waypoints")


def _sink_lengths(io: Any) -> dict[str, int]:
    return {name: len(getattr(io, name, []) or []) for name in _SINK_NAMES}


def _capture_published(io: Any, before: dict[str, int]) -> Any | None:
    """Return the object the FSM just dispatched: the one new item across the answer sinks.

    ``before`` is the per-sink length snapshot taken *before* the tick that flipped
    ``answer_published`` True. Whichever sink grew holds the freshly published answer as
    its last element — this is how the FSM's answer is told apart from exploration
    waypoints (which grew in earlier ticks, before the flip).
    """
    for name in _SINK_NAMES:
        sink = getattr(io, name, None) or []
        if len(sink) > before.get(name, 0):
            return sink[-1]
    return None


class _ScriptedPerception:
    """Drives a :class:`~core.perception.tracker.PerceptionPipeline` over a replay.

    Owns a :class:`~core.perception.scripted.ScriptedPanoDetector` (loaded from a labels
    JSON) and the pipeline it feeds. :meth:`maybe_process` is called every tick with the
    io's current :class:`~core.interfaces.PanoFrame` + scan; it processes the frame only
    when a *new* pano appears (``PanoFrame.t`` changed), mapping that pano timestamp to a
    keyframe index by its position in the fixtures' sorted pano-time order (== index.json
    keyframe order). The pipeline's :class:`BasicSceneIndex` is the one the heads resolve.
    """

    def __init__(self, io: Any, detections_path: str) -> None:
        from core.perception.scripted import ScriptedPanoDetector
        from core.perception.tracker import PerceptionPipeline
        from core.replay.replay_io import CH_PANO, CH_SCAN

        self._detector = ScriptedPanoDetector.from_json(detections_path)
        self.index = BasicSceneIndex([])
        self.pipeline = PerceptionPipeline(self._detector, index=self.index)
        # Sorted pano timestamps == index.json keyframe order; a pano.t's bisect position
        # in this list is its keyframe index (the labels JSON keys those indices).
        store = getattr(io, "store", None)
        self._pano_times: list[float] = store.times(CH_PANO) if store is not None else []
        self._last_pano_t: float | None = None
        # Startup fallback: the lidar stream can begin a few seconds after the first pano
        # (the jingfan bag has no /registered_scan for its first ~9 s while the vehicle
        # sits at the origin). A labelled keyframe in that window has no `latest_scan()`
        # (strictly <= now), so fuse it against the earliest recorded scan instead of
        # dropping it — the vehicle is stationary there, so the nearest scan is valid.
        self._earliest_scan = None
        if store is not None:
            scan_times = store.times(CH_SCAN)
            if scan_times:
                self._earliest_scan = store.latest(CH_SCAN, scan_times[0])

    def _keyframe_for(self, t: float) -> int:
        idx = bisect.bisect_left(self._pano_times, t)
        if idx < len(self._pano_times) and self._pano_times[idx] == t:
            return idx
        # tolerate float drift: nearest scheduled pano time
        idx = min(idx, len(self._pano_times) - 1)
        if idx > 0 and abs(self._pano_times[idx - 1] - t) < abs(self._pano_times[idx] - t):
            idx -= 1
        return idx

    def maybe_process(self, io: Any) -> None:
        pano = io.latest_pano()
        if pano is None:
            return
        if self._last_pano_t is not None and pano.t == self._last_pano_t:
            return
        scan = io.latest_scan()
        if scan is None:
            # Pre-lidar startup window: fall back to the earliest recorded scan so a
            # labelled keyframe there still grounds (vehicle stationary at the origin).
            scan = self._earliest_scan
        if scan is None:
            return
        self._last_pano_t = pano.t
        kf = self._keyframe_for(pano.t)
        self._detector.current_keyframe = kf
        self.pipeline.process(pano, scan)


def run_question(
    question_text: str,
    io: Any,
    *,
    scene_index: SceneIndex | None = None,
    detections_path: str | None = None,
    chat_fns: dict[str, Callable] | None = None,
    max_wall_s: float = 600.0,
    tick_hz: float = 5.0,
    budget_scale: float = 1.0,
) -> RunResult:
    """Drive one question through the full pipeline on the io's simulated clock.

    The loop runs until the controller reaches DONE (its answer is published) or the
    wall-clock guard trips. Running out of recorded messages does NOT end the loop: a
    replay io free-runs its frozen world so the FSM's budget/watchdog gates still fire.

    question_text: the challenge question to answer.
    io:            a RobotIO (MockRobotIO or ReplayRobotIO); its clock is advanced here.
    scene_index:   override the resolved scene index (else derived from ``io.scene``).
    detections_path: path to a scripted-labels JSON (jingfan_labels.json schema). When set
                   (replay io only), a PerceptionPipeline fed by a ScriptedPanoDetector
                   grounds each new pano frame's boxes against the real lidar scan; its
                   scene index — not ``io.scene`` — is what the heads resolve, and its final
                   instance count is reported as ``RunResult.instances_tracked``.
    chat_fns:      optional injected LLM checkpoints ``{"llm_verify", "anchor_confirm"}``;
                   omitted / None keeps the deterministic offline path (regex parse only).
    max_wall_s:    hard wall-clock guard so a pathological loop can never hang the cockpit.
    tick_hz:       simulated tick rate; each tick advances the clock by 1/tick_hz seconds.
    budget_scale:  scales the FSM's three budget gates (explore budgets, 510 s forced
                   assembly, 570 s watchdog) for replay against short bags. 1.0 = unscaled;
                   0.2 puts the gates at ~102/114 s of bag time. Implemented with a wrapper
                   clock (see :class:`_ScaledClock`) because the gate constants live in
                   ``core.fsm`` with no injection seam; only applied when the io exposes the
                   ``set_budget_clock`` seam (ReplayRobotIO) and scale != 1.0.
    """
    chat_fns = chat_fns or {}
    dt = 1.0 / float(tick_hz)

    _inject_question(io, question_text)

    # Scripted-detection grounding: build a PerceptionPipeline over the replay and let the
    # heads resolve against its (live-mutated) scene index. Only valid with a replay io.
    perception: _ScriptedPerception | None = None
    if detections_path is not None:
        if not hasattr(io, "latest_pano") or getattr(io, "store", None) is None:
            raise ValueError(
                "detections_path requires a replay io (ReplayRobotIO / --fixtures); "
                "it has no pano/scan stream to ground against."
            )
        perception = _ScriptedPerception(io, detections_path)

    if perception is not None:
        idx: SceneIndex = perception.index
    else:
        idx = _derive_scene_index(io, scene_index)

    # The clock we ADVANCE each tick is always the io's real/underlying clock (so replay
    # message lookups and free-run stay correct). The clock the FSM READS may be a scaled
    # wrapper installed via set_budget_clock.
    raw_clock = io.raw_clock() if hasattr(io, "raw_clock") else io.clock()
    scaled = budget_scale != 1.0 and hasattr(io, "set_budget_clock")
    if scaled:
        io.set_budget_clock(_ScaledClock(raw_clock, budget_scale))

    callables = build_callables(
        idx,
        llm_verify=chat_fns.get("llm_verify"),
        anchor_confirm=chat_fns.get("anchor_confirm"),
    )
    ctrl = QuestionController(**callables)

    states_visited: list[str] = []
    last_state: State | None = None
    published_ref: dict = {"published": False, "answer": None}
    end_of_data_at_sim_s: float | None = None

    wall_start = time.monotonic()
    ticks = 0
    # Tick bound: with free-run at `dt` per tick, reaching the (scaled) 570 s watchdog from
    # t0 takes ~ (570 * budget_scale) / dt ticks; give generous headroom. The wall-clock
    # guard is the real safety net, this only prevents an unbounded spin if a gate misfires.
    horizon_s = 650.0 * budget_scale + 30.0 if scaled else 650.0
    # During replay free-run the clock only advances by its own FREE_RUN_DT_S per tick
    # (independent of `dt`), so budget the tick cap on the finer of the two granularities.
    from core.replay.replay_io import FREE_RUN_DT_S

    tick_granularity = min(dt, FREE_RUN_DT_S) if hasattr(io, "end_of_data") else dt
    max_ticks = int(horizon_s / tick_granularity) + 50
    while ctrl.state is not State.DONE and ticks < max_ticks:
        if ctrl.state is not last_state:
            states_visited.append(ctrl.state.value)
            last_state = ctrl.state
        if perception is not None:
            perception.maybe_process(io)
        before = _sink_lengths(io)
        was_published = ctrl.answer_published
        ctrl.tick(io)
        if not was_published and ctrl.answer_published and not published_ref["published"]:
            published_ref["answer"] = _capture_published(io, before)
            published_ref["published"] = True
        _clock_advance(raw_clock, dt)
        if end_of_data_at_sim_s is None and getattr(io, "end_of_data", False):
            end_of_data_at_sim_s = float(raw_clock.now())
        ticks += 1
        if time.monotonic() - wall_start > max_wall_s:
            break
    # Finalize: one more tick to dump the flight recording and catch a same-tick publish.
    if perception is not None:
        perception.maybe_process(io)
    before = _sink_lengths(io)
    was_published = ctrl.answer_published
    ctrl.tick(io)
    if not was_published and ctrl.answer_published and not published_ref["published"]:
        published_ref["answer"] = _capture_published(io, before)
        published_ref["published"] = True
    if ctrl.state is not last_state:
        states_visited.append(ctrl.state.value)

    wall_s = time.monotonic() - wall_start

    answer, published = _published_answer(io, published_ref)
    flight_log = ctrl.flight_recording()
    checkpoint_calls = ctrl.ledger.count("parse") + ctrl.ledger.count("verification") if ctrl.ledger else 0

    return RunResult(
        answer=answer,
        qtype=ctrl.qtype,
        elapsed_sim_s=ctrl.budget.elapsed() if ctrl.budget is not None else 0.0,
        states_visited=states_visited,
        checkpoint_calls=checkpoint_calls,
        flight_log=flight_log,
        published=published,
        ticks=ticks,
        wall_s=wall_s,
        end_of_data_at_sim_s=end_of_data_at_sim_s,
        floor_used=_floor_used(flight_log),
        instances_tracked=len(idx.all_instances()) if perception is not None else 0,
    )
