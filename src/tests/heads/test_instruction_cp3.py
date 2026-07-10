"""CP3 rich anchor-confirmation seam on InstructionHead: confirm, demote-and-replan, compat.

Stubs match the vision seam signature ``run(anchor_desc, crop) -> AnchorConfirmOutcome``
(duck-typed on ``.action`` in {"confirm","demote"}); the real CP3 builder is not called.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from core.heads.instruction import InstructionHead, _is_legacy_anchor_seam
from core.mocks.synthetic_scene import SyntheticScene
from core.perception.scene_index import BasicSceneIndex
from core.plan_schema import Anchor, LegKind, RouteLeg
from tests.heads._helpers import instruction_plan


@dataclass
class _AC:
    action: str
    confidence: float = 0.9
    actual_label: str | None = None
    match: bool | None = None


def _goto(noun):
    return RouteLeg(kind=LegKind.GOTO, anchors=[Anchor(noun=noun)])


class _DriveIO:
    def __init__(self, sc, start=(0.7, 3.0), step=0.15):
        self._sc = sc
        self._x, self._y = start
        self._t = 0.0
        self._step = step
        self.waypoints = []

    def latest_terrain(self, extended=False):
        return self._sc.terrain_patch(extended=extended, t=self._t)

    def latest_odom(self):
        from core.interfaces import OdomState

        return OdomState(t=self._t, x=self._x, y=self._y, z=0.0, yaw=0.0)

    def publish_waypoint(self, wp):
        self.waypoints.append(wp)

    def tick_motion(self):
        if not self.waypoints:
            self._t += 0.2
            return
        wp = self.waypoints[-1]
        dx, dy = wp.x - self._x, wp.y - self._y
        d = (dx * dx + dy * dy) ** 0.5
        s = min(self._step, d)
        if d > 1e-6:
            self._x += dx / d * s
            self._y += dy / d * s
        self._t += 0.2

    @property
    def pose(self):
        return (self._x, self._y)


def _if_scene():
    sc = SyntheticScene(0)
    sc.rooms = [type(sc.rooms[0])(0.0, 0.0, 8.0, 6.0)]
    sc.place_box("sofa", 6.0, 3.0, 0.5, 0.5, 0.5)
    return sc, BasicSceneIndex(sc.instances())


def _run(head, io, idx, max_ticks=4000):
    for _ in range(max_ticks):
        head.advance(io, idx)
        io.tick_motion()
        tw = head.terminal_waypoint()
        if tw is not None and ((io.pose[0] - tw.x) ** 2 + (io.pose[1] - tw.y) ** 2) ** 0.5 < 0.3:
            break


def test_cp3_confirm_fires_and_passes_anchor_desc():
    sc, idx = _if_scene()
    seen = []

    def confirm(anchor_desc, crop):
        seen.append(anchor_desc)
        return _AC("confirm")

    head = InstructionHead(plan=instruction_plan([_goto("sofa")]), anchor_confirm=confirm)
    _run(head, io := _DriveIO(sc), idx)
    assert "sofa" in seen  # rich seam called with the anchor's noun on arrival


def test_cp3_confirm_never_demotes():
    sc, idx = _if_scene()
    head = InstructionHead(
        plan=instruction_plan([_goto("sofa")]),
        anchor_confirm=lambda desc, crop: _AC("confirm"),
    )
    _run(head, _DriveIO(sc), idx)
    assert head._demoted == set()


def test_cp3_confident_mismatch_demotes_instance():
    # Two sofas: the ranked winner gets demoted on arrival -> demote set records it.
    sc = SyntheticScene(0)
    sc.rooms = [type(sc.rooms[0])(0.0, 0.0, 8.0, 6.0)]
    sc.place_box("sofa", 5.0, 3.0, 0.5, 0.5, 0.5)
    sc.place_box("sofa", 6.5, 3.0, 0.5, 0.5, 0.5)
    idx = BasicSceneIndex(sc.instances())

    def confirm(anchor_desc, crop):
        return _AC("demote", confidence=0.9)

    head = InstructionHead(plan=instruction_plan([_goto("sofa")]), anchor_confirm=confirm)
    _run(head, _DriveIO(sc), idx, max_ticks=200)
    # at least one arrival triggered a demote of the ranked winner instance.
    assert len(head._demoted) >= 1


def test_cp3_demote_never_blocks_drive_when_no_runner_up():
    sc, idx = _if_scene()  # single sofa -> no runner-up
    head = InstructionHead(
        plan=instruction_plan([_goto("sofa")]),
        anchor_confirm=lambda desc, crop: _AC("demote", confidence=0.95),
    )
    _run(head, io := _DriveIO(sc), idx)
    # the map's belief stands; a terminal waypoint is still produced (drive not blocked).
    assert head.terminal_waypoint() is not None


def test_cp3_exception_trusts_the_map():
    sc, idx = _if_scene()

    def boom(desc, crop):
        raise RuntimeError("dark checkpoint")

    head = InstructionHead(plan=instruction_plan([_goto("sofa")]), anchor_confirm=boom)
    _run(head, _DriveIO(sc), idx)
    assert head._demoted == set()  # exception == confirm (trust the map)


def test_legacy_bool_anchor_seam_detected_and_called():
    sc, idx = _if_scene()
    seen = []
    head = InstructionHead(
        plan=instruction_plan([_goto("sofa")]),
        anchor_confirm=lambda plan, i, summary: seen.append(i) or True,
    )
    assert head._legacy_confirm is True
    _run(head, _DriveIO(sc), idx)
    assert 0 in seen  # legacy (plan, leg_index, summary) seam still fires


def test_is_legacy_anchor_seam_classifier():
    assert _is_legacy_anchor_seam(lambda plan, i, summary: True) is True
    assert _is_legacy_anchor_seam(lambda anchor_desc, crop: None) is False
    assert _is_legacy_anchor_seam(lambda *a: None) is False
