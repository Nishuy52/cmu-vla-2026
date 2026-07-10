"""EventLog ring-buffer flight recorder."""
from __future__ import annotations

import pytest

from core.fsm.events import EventLog, LogRecord


def test_record_and_dump_order():
    log = EventLog()
    log.record(0.0, "idle", "start")
    log.record(1.0, "parsing", "parsed", "tier=api")
    dump = log.dump()
    assert [r.event for r in dump] == ["start", "parsed"]
    assert dump[1].detail == "tier=api"
    assert isinstance(dump[0], LogRecord)


def test_ring_buffer_drops_oldest():
    log = EventLog(capacity=3)
    for i in range(5):
        log.record(float(i), "s", f"e{i}")
    dump = log.dump()
    assert len(dump) == 3
    assert [r.event for r in dump] == ["e2", "e3", "e4"]


def test_len_and_iter():
    log = EventLog()
    log.record(0.0, "s", "a")
    log.record(1.0, "s", "b")
    assert len(log) == 2
    assert [r.event for r in log] == ["a", "b"]


def test_capacity_must_be_positive():
    with pytest.raises(ValueError):
        EventLog(capacity=0)
