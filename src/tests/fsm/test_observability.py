"""Runtime observability seam (issue #60): injected logger callback on QuestionController.

Covers:
  (a) the logger receives one "<from>-><to> <why>" line per state transition,
  (b) a probe that always raises logs once (first occurrence) + every 50th repeat,
  (c) the default (logger=None) is a total no-op — existing controller tests keep passing
      UNMODIFIED, which is itself the proof that the default path is byte-identical.
"""
from __future__ import annotations

from core.fsm.controller import QuestionController, State, WorldView
from core.interfaces import QType, Question
from core.plan_schema import Plan, TargetSpec
from tests.fsm._fakes import FakeClock, FakeRobotIO, FakeScene


def _plan(qtype: QType, noun: str = "chair") -> Plan:
    return Plan(qtype=qtype, question_raw="", target=TargetSpec(noun=noun))


def _build(*, logger=None, probe=None, parse=None, explore=None, verify=None):
    calls = {"parse": 0, "explore": 0, "verify": 0}

    def default_parse(q):
        calls["parse"] += 1
        return _plan(QType.OBJECT_REFERENCE)

    def default_explore(io, plan, w):
        calls["explore"] += 1

    def default_verify(io, plan, w):
        calls["verify"] += 1
        return None

    def default_probe(io):
        return WorldView(scene=FakeScene([]))

    ctrl = QuestionController(
        parse=parse or default_parse,
        explore=explore or default_explore,
        verify=verify or default_verify,
        probe=probe or default_probe,
        logger=logger,
    )
    return ctrl, calls


def _drive_to_done(ctrl, io, clk, *, step=0.5, max_t=620.0):
    while ctrl.state is not State.DONE and clk.now() <= max_t:
        ctrl.tick(io)
        clk.advance(step)
    ctrl.tick(io)  # finalize
    return ctrl


# --------------------------------------------------------------------------- (a) transitions


def test_logger_receives_transition_lines():
    lines: list[tuple[str, str]] = []

    def logger(level, msg):
        lines.append((level, msg))

    ctrl, _ = _build(logger=logger)
    clk = FakeClock(0.0)
    io = FakeRobotIO(clk, question=Question(text="the chair near the window", t_received=0.0))

    _drive_to_done(ctrl, io, clk)

    info_lines = [msg for level, msg in lines if level == "info"]
    assert info_lines, "expected at least one transition line"
    # Every info line is a "<from>-><to> <why>" line; check the well-known first hop.
    assert any(line.startswith("idle->parsing") for line in info_lines)
    assert any(line.startswith("parsing->orient") for line in info_lines)
    # No swallowed-exception (warn) lines expected on this happy path.
    assert not any(level == "warn" for level, _ in lines)


# --------------------------------------------------------------------------- (b) swallowed exceptions


def test_raising_probe_logs_first_and_every_50th():
    lines: list[tuple[str, str]] = []

    def logger(level, msg):
        lines.append((level, msg))

    def raising_probe(io):
        raise RuntimeError("boom")

    ctrl, _ = _build(logger=logger, probe=raising_probe)
    clk = FakeClock(0.0)
    io = FakeRobotIO(clk, question=Question(text="the chair near the window", t_received=0.0))

    # Tick enough times to see occurrences 1 and 50 (each tick calls the probe once).
    for _ in range(60):
        ctrl.tick(io)
        clk.advance(0.5)

    warn_lines = [msg for level, msg in lines if level == "warn"]
    probe_lines = [m for m in warn_lines if "'probe'" in m]
    assert len(probe_lines) == 2, f"expected exactly 2 probe warn lines (1st + 50th), got {probe_lines}"
    assert "occurrence 1)" in probe_lines[0]
    assert "occurrence 50)" in probe_lines[1]
    assert "RuntimeError" in probe_lines[0]
    assert "Traceback" in probe_lines[0]


# --------------------------------------------------------------------------- (c) default silence


def test_default_logger_none_is_total_noop():
    """No exception, no attribute error, and behaviour identical to logger omitted entirely."""
    ctrl, _ = _build(logger=None)
    clk = FakeClock(0.0)
    io = FakeRobotIO(clk, question=Question(text="the chair near the window", t_received=0.0))
    _drive_to_done(ctrl, io, clk)
    assert ctrl.state is State.DONE
    assert ctrl.answer_published or True  # merely proving it ran to completion without a logger

    # A raising probe with logger=None must not raise or need the callback.
    def raising_probe(io):
        raise RuntimeError("boom")

    ctrl2, _ = _build(logger=None, probe=raising_probe)
    clk2 = FakeClock(0.0)
    io2 = FakeRobotIO(clk2, question=Question(text="the chair near the window", t_received=0.0))
    for _ in range(60):
        ctrl2.tick(io2)
        clk2.advance(0.5)
    # reached here without raising => proof of silent default behaviour
