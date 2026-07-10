"""In-memory :class:`~core.interfaces.RobotIO` implementation backed by a
:class:`~core.mocks.synthetic_scene.SyntheticScene` and a manually-advanced clock.

Used by the offline test harness in place of the ROS adapter. Getters return the
latest synthesised message (or None before first receipt); publish_* record into
lists so tests can assert on emitted waypoints/markers/answers. No network, no ROS.
"""
from __future__ import annotations

import numpy as np

from core.interfaces import (
    IntAnswer,
    LidarScan,
    MarkerBox,
    OdomState,
    PanoFrame,
    Question,
    TerrainPatch,
    WaypointCmd,
)
from core.mocks.synthetic_scene import SyntheticScene

PANO_SHAPE = (640, 1920, 3)  # matches interfaces.PanoFrame contract


class FakeClock:
    """Manually-advanced monotonic clock (implements the Clock protocol)."""

    def __init__(self, start: float = 0.0) -> None:
        self._t = float(start)

    def now(self) -> float:
        return self._t

    def advance(self, dt: float) -> float:
        """Advance the clock by ``dt`` seconds and return the new time."""
        self._t += float(dt)
        return self._t

    def set(self, t: float) -> None:
        self._t = float(t)


class MockRobotIO:
    """RobotIO against a SyntheticScene. Odometry starts at the scene origin.

    Publish sinks are exposed as ``waypoints``, ``markers``, ``ints`` for asserts.
    """

    def __init__(
        self,
        scene: SyntheticScene,
        clock: FakeClock | None = None,
        *,
        start_x: float = 0.5,
        start_y: float = 0.5,
    ) -> None:
        self.scene = scene
        self._clock = clock if clock is not None else FakeClock()
        self._question: Question | None = None
        self._x = float(start_x)
        self._y = float(start_y)
        self._yaw = 0.0
        # publish sinks
        self.waypoints: list[WaypointCmd] = []
        self.markers: list[MarkerBox] = []
        self.ints: list[IntAnswer] = []

    # ---------------------------------------------------------------- injection

    def set_question(self, text: str) -> Question:
        """Inject the challenge question (as the ROS 1 Hz republish would)."""
        self._question = Question(text=text, t_received=self._clock.now())
        return self._question

    def set_pose(self, x: float, y: float, yaw: float = 0.0) -> None:
        """Teleport the mock vehicle (map frame)."""
        self._x, self._y, self._yaw = float(x), float(y), float(yaw)

    # ------------------------------------------------------------------ getters

    def question(self) -> Question | None:
        return self._question

    def latest_odom(self) -> OdomState | None:
        return OdomState(
            t=self._clock.now(), x=self._x, y=self._y, z=0.0, yaw=self._yaw
        )

    def latest_pano(self) -> PanoFrame | None:
        """Black equirectangular frame, shape-correct per the interface."""
        image = np.zeros(PANO_SHAPE, dtype=np.uint8)
        return PanoFrame(t=self._clock.now(), image=image, odom=self.latest_odom())

    def latest_scan(self) -> LidarScan | None:
        """Object-surface points sampled from every instance AABB (map frame)."""
        clouds = [
            rec.points
            for rec in self.scene.instances()
            if rec.points is not None and len(rec.points) > 0
        ]
        if clouds:
            pts = np.vstack(clouds).astype(np.float32)
        else:
            pts = np.empty((0, 3), dtype=np.float32)
        return LidarScan(t=self._clock.now(), points=pts)

    def latest_terrain(self, extended: bool = False) -> TerrainPatch | None:
        return self.scene.terrain_patch(extended=extended, t=self._clock.now())

    # ---------------------------------------------------------------- publishers

    def publish_waypoint(self, wp: WaypointCmd) -> None:
        self.waypoints.append(wp)

    def publish_marker(self, box: MarkerBox) -> None:
        self.markers.append(box)

    def publish_int(self, ans: IntAnswer) -> None:
        self.ints.append(ans)

    def clock(self) -> FakeClock:
        return self._clock
