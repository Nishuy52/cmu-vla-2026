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

from core.fsm.controller import QuestionController, State
from core.fsm.events import LogRecord
from core.heads import build_callables
from core.interfaces import IntAnswer, MarkerBox, QType, SceneIndex, WaypointCmd
from core.perception.scene_index import BasicSceneIndex


@dataclass
class RunResult:
    """Structured outcome of a single :func:`run_question` drive.

    answer:            the published answer object (IntAnswer/MarkerBox/WaypointCmd) or None.
    qtype:             the qtype the controller resolved for the question.
    elapsed_sim_s:     simulated seconds elapsed on the injected clock at DONE.
    states_visited:    ordered list of distinct FSM states the run passed through.
    checkpoint_calls:  count of injected-checkpoint calls (parse/verify) logged by the FSM.
    flight_log:        the controller's dumped flight recording (list of LogRecord).
    published:         True iff exactly one legal answer reached a RobotIO sink.
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


def _published_answer(io: Any) -> tuple[Any | None, bool]:
    """Return the single published answer (last of any sink) and whether one exists."""
    for sink_name in ("ints", "markers", "waypoints"):
        sink = getattr(io, sink_name, None)
        if sink:
            return sink[-1], True
    return None, False


def run_question(
    question_text: str,
    io: Any,
    *,
    scene_index: SceneIndex | None = None,
    chat_fns: dict[str, Callable] | None = None,
    max_wall_s: float = 600.0,
    tick_hz: float = 5.0,
) -> RunResult:
    """Drive one question through the full pipeline on the io's simulated clock.

    question_text: the challenge question to answer.
    io:            a RobotIO (MockRobotIO or ReplayRobotIO); its clock is advanced here.
    scene_index:   override the resolved scene index (else derived from ``io.scene``).
    chat_fns:      optional injected LLM checkpoints ``{"llm_verify", "anchor_confirm"}``;
                   omitted / None keeps the deterministic offline path (regex parse only).
    max_wall_s:    hard wall-clock guard so a pathological loop can never hang the cockpit.
    tick_hz:       simulated tick rate; each tick advances the clock by 1/tick_hz seconds.
    """
    chat_fns = chat_fns or {}
    dt = 1.0 / float(tick_hz)

    _inject_question(io, question_text)
    idx = _derive_scene_index(io, scene_index)

    callables = build_callables(
        idx,
        llm_verify=chat_fns.get("llm_verify"),
        anchor_confirm=chat_fns.get("anchor_confirm"),
    )
    ctrl = QuestionController(**callables)
    clock = io.clock()

    states_visited: list[str] = []
    last_state: State | None = None

    wall_start = time.monotonic()
    ticks = 0
    # Simulated-time bound: the FSM self-terminates at the 570 s watchdog floor; give it a
    # little headroom in ticks so DONE is always reached before this cap.
    max_ticks = int((650.0) * tick_hz) + 10
    while ctrl.state is not State.DONE and ticks < max_ticks:
        if ctrl.state is not last_state:
            states_visited.append(ctrl.state.value)
            last_state = ctrl.state
        ctrl.tick(io)
        _clock_advance(clock, dt)
        ticks += 1
        if time.monotonic() - wall_start > max_wall_s:
            break
    ctrl.tick(io)  # finalize DONE (dumps the flight recording)
    if ctrl.state is not last_state:
        states_visited.append(ctrl.state.value)

    wall_s = time.monotonic() - wall_start

    answer, published = _published_answer(io)
    checkpoint_calls = ctrl.ledger.count("parse") + ctrl.ledger.count("verification") if ctrl.ledger else 0

    return RunResult(
        answer=answer,
        qtype=ctrl.qtype,
        elapsed_sim_s=ctrl.budget.elapsed() if ctrl.budget is not None else 0.0,
        states_visited=states_visited,
        checkpoint_calls=checkpoint_calls,
        flight_log=ctrl.flight_recording(),
        published=published,
        ticks=ticks,
        wall_s=wall_s,
    )
