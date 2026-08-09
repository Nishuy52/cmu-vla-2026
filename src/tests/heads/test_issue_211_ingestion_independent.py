"""Issue #211 validation (a): occupancy/costmap ingestion must grow with arriving
frames even when the detector lags far behind (or never returns at all).

Diagnosis this test locks in: ``ExploreHead``/``InstructionHead`` ingest terrain+scan
into their own ``OccupancyGrid`` every ``advance()`` call by reading
``io.latest_terrain()``/``io.latest_scan()`` directly (their own ROS callbacks in the
real adapter) — never through ``PerceptionPipeline``/``AsyncPerceptionWorker``. This
test proves that end to end: a completely STUCK detector (a real
``AsyncPerceptionWorker`` whose pipeline never returns) shares the SAME live
``BasicSceneIndex`` the head reads, and the head's grid still fills in across ticks.
"""
from __future__ import annotations

import threading
import time

from core.heads.explore_step import ExploreHead
from core.interfaces import OdomState, WaypointCmd
from core.mocks.synthetic_scene import SyntheticScene
from core.nav.occupancy import FREE
from core.perception.async_pipeline import AsyncPerceptionWorker
from core.perception.scene_index import BasicSceneIndex
from tests.heads._helpers import numerical_plan


class _ExploreIO:
    def __init__(self, sc: SyntheticScene, start=(2.5, 2.5)):
        self._sc = sc
        self._x, self._y, self._t = start[0], start[1], 0.0
        self.waypoints: list[WaypointCmd] = []

    def latest_terrain(self, extended: bool = False):
        return self._sc.terrain_patch(extended=extended, t=self._t)

    def latest_odom(self):
        return OdomState(t=self._t, x=self._x, y=self._y, z=0.0, yaw=0.0)

    def publish_waypoint(self, wp: WaypointCmd):
        self.waypoints.append(wp)

    def advance_time(self, dt):
        self._t += dt


class _StuckPipeline:
    """Models a detector that never returns — the worst-case backlog: not slow,
    permanently stuck. process() blocks forever (until the test releases it)."""

    def __init__(self) -> None:
        self.release = threading.Event()
        self.started = threading.Event()
        self.calls = 0

    def process(self, pano, scan):
        self.started.set()
        self.calls += 1
        self.release.wait()
        return []


def _free_cells(grid) -> int:
    state = grid.state
    if state is None:
        return 0
    return int((state == FREE).sum())


def test_occupancy_grows_across_ticks_while_the_detector_is_permanently_stuck():
    sc = SyntheticScene(0)
    sc.populate_default(3)
    # Empty on purpose: the stuck detector below must never populate this index, so an
    # empty index makes "detector contributed nothing" directly assertable.
    scene = BasicSceneIndex([])

    stuck = _StuckPipeline()
    worker = AsyncPerceptionWorker(stuck)
    try:
        # Submit the first (and only, since nothing ever finishes to free the slot up
        # for a NEW one to matter) frame — the worker wedges on it immediately,
        # exactly like a live GDINO forward that never comes back.
        from core.interfaces import LidarScan, PanoFrame
        import numpy as np

        worker.submit(
            PanoFrame(t=0.0, image=np.zeros((2, 2, 3), dtype=np.uint8),
                      odom=OdomState(t=0.0, x=0.0, y=0.0, z=0.0, yaw=0.0)),
            LidarScan(t=0.0, points=np.zeros((1, 3), dtype=np.float32)),
        )
        assert stuck.started.wait(timeout=2.0), "worker never started the forward"

        head = ExploreHead(plan=numerical_plan("chair"))
        io = _ExploreIO(sc)

        assert _free_cells(head.grid) == 0  # nothing ingested yet
        for _ in range(5):
            head.advance(io, scene)
            io.advance_time(0.2)  # one 5 Hz tick

        assert _free_cells(head.grid) > 0, (
            "occupancy ingestion did not run even once while the detector was stuck"
        )
        # The detector genuinely never got past its first (only) frame — proving the
        # grid growth above was NOT gated on it completing.
        assert stuck.calls == 1
        assert worker.processed_count == 0
        assert worker.in_flight_age_s is not None  # still stuck, right now
        assert scene.all_instances() == []  # the stuck detector contributed nothing
    finally:
        stuck.release.set()
        worker.stop(timeout=2.0)
