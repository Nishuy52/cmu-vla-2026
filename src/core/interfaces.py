"""Frozen contracts for the VLA core. Everything in core/ codes against these types.

All spatial quantities are in the `map` frame, metres/radians/seconds, unless noted.
Mirrors the test-time ROS topic contract (docs/upstream_notes.md §3) without importing ROS.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, Sequence, runtime_checkable

import numpy as np

# --------------------------------------------------------------------------- sensors


@dataclass(frozen=True)
class OdomState:
    """Vehicle state from /state_estimation (map -> sensor)."""

    t: float
    x: float
    y: float
    z: float
    yaw: float  # radians


@dataclass(frozen=True)
class PanoFrame:
    """One /camera/image frame: 640x1920x3 uint8 RGB equirectangular strip.

    Column 0 is at vehicle yaw + pi (wrap); azimuth decreases left-to-right such that
    image centre column looks along the vehicle heading. Verify against sim in Phase 2.
    """

    t: float
    image: np.ndarray  # (640, 1920, 3) uint8
    odom: OdomState  # odometry nearest to t


@dataclass(frozen=True)
class LidarScan:
    """Registered scan from /registered_scan: (N, 3) float32 xyz, map frame."""

    t: float
    points: np.ndarray


@dataclass(frozen=True)
class TerrainPatch:
    """Terrain analysis cloud from /terrain_map (5 m) or /terrain_map_ext (20 m).

    points: (N, 4) float32 [x, y, z, intensity]; intensity = obstacle height above
    local ground in metres (docs/upstream_notes.md gotcha 9). intensity < FREE_MAX
    is traversable.
    """

    t: float
    points: np.ndarray
    extended: bool  # True for /terrain_map_ext

    FREE_MAX: float = 0.15  # m; traversability cutoff (tunable, see nav config)


@dataclass(frozen=True)
class Question:
    """From /challenge_question (republished at 1 Hz; latch the first receipt)."""

    text: str
    t_received: float


# --------------------------------------------------------------------------- outputs


@dataclass(frozen=True)
class WaypointCmd:
    """Published to /way_point_with_heading. theta is ALWAYS 0 (ignored this year).

    Keep targets near the vehicle (<= ~2.5 m ahead) per stack guidance.
    """

    x: float
    y: float


@dataclass(frozen=True)
class MarkerBox:
    """Axis-aligned box for /selected_object_marker (Marker CUBE, map frame).

    Scored by ground-truth overlap; center additionally serves as a nav goal.
    """

    cx: float
    cy: float
    cz: float
    sx: float  # full extents, metres
    sy: float
    sz: float
    label: str = ""


@dataclass(frozen=True)
class IntAnswer:
    """Published to /numerical_response (Int32)."""

    value: int


# --------------------------------------------------------------------------- world model


@dataclass(frozen=True)
class ColorBin:
    """One dominant-colour bin of an instance (VLA-3D 15-scheme quantisation).

    name:     scheme colour name (one of the closed 15, e.g. ``gray``, ``maroon``).
    rgb:      representative 0-255 RGB of the bin (the raw colour the scheme name
              quantises — carried because the *name* alone loses it).
    fraction: share of the object's surface points that fell in this bin, [0, 1].

    Carried so colour matching can apply cutoffs the scheme NAME cannot express
    (issues #11, #12): a near-black object whose points quantise to ``gray`` (RGB
    47,79,79) is separable from lighter grays only by luminance; a minor off-hue
    bin (18% maroon on a gray pillow) is separable from a dominant one only by
    fraction. Optional/defaulted on :class:`InstanceRecord`, so mocks and the live
    perception path that do not populate it are unaffected.
    """

    name: str
    rgb: tuple[int, int, int]
    fraction: float

    @property
    def luma(self) -> float:
        """Rec. 601 luminance (0-255 scale) of the bin's representative RGB."""
        r, g, b = self.rgb
        return 0.299 * r + 0.587 * g + 0.114 * b


@dataclass
class InstanceRecord:
    """One tracked object instance in the fused 3D map (lidar geometry, camera semantics)."""

    instance_id: int
    label: str  # canonical noun, lowercase singular
    score: float  # detector confidence, max over observations
    n_obs: int  # distinct keyframe observations (>=3 required for confident answers)
    centroid: np.ndarray  # (3,) float
    aabb_min: np.ndarray  # (3,) float — trimmed (2nd pct per axis)
    aabb_max: np.ndarray  # (3,) float — trimmed (98th pct per axis)
    points: np.ndarray | None = None  # (M, 3) retained fused points (may be decimated)
    caption: str = ""  # optional VLM caption (stretch feature)
    aliases: tuple[str, ...] = ()  # typo/synonym-tolerant match set
    #: Per-bin (scheme name, raw RGB, fraction) dominant colours; () when unknown
    #: (mocks / perception without colour quantisation). Enables luminance +
    #: dominance colour salience (issues #11/#12) that scheme names alone cannot.
    color_bins: tuple[ColorBin, ...] = ()

    @property
    def extents(self) -> np.ndarray:
        return self.aabb_max - self.aabb_min

    def to_marker(self) -> MarkerBox:
        c = (self.aabb_min + self.aabb_max) / 2.0
        e = self.extents
        return MarkerBox(*map(float, c), *map(float, e), label=self.label)


class SceneIndex(Protocol):
    """Read view over the instance map used by the toolbox and answer heads."""

    def all_instances(self) -> Sequence[InstanceRecord]: ...

    def by_label(self, noun: str) -> Sequence[InstanceRecord]:
        """Typo/plural/synonym-tolerant lookup ('refridgerator' -> fridge instances)."""
        ...


# --------------------------------------------------------------------------- robot I/O


@runtime_checkable
class Clock(Protocol):
    def now(self) -> float: ...


@runtime_checkable
class RobotIO(Protocol):
    """The single seam between core logic and the outside world (ROS adapter or mocks).

    Getters return the latest message or None before first receipt; they never block.
    """

    def question(self) -> Question | None: ...

    def latest_pano(self) -> PanoFrame | None: ...

    def latest_scan(self) -> LidarScan | None: ...

    def latest_terrain(self, extended: bool = False) -> TerrainPatch | None: ...

    def latest_odom(self) -> OdomState | None: ...

    def publish_waypoint(self, wp: WaypointCmd) -> None: ...

    def publish_marker(self, box: MarkerBox) -> None: ...

    def publish_int(self, ans: IntAnswer) -> None: ...

    def clock(self) -> Clock: ...


# --------------------------------------------------------------------------- budget

QUESTION_BUDGET_S: float = 600.0
FORCED_ASSEMBLY_S: float = 510.0  # T-90: begin best-effort answer assembly
WATCHDOG_FLOOR_S: float = 570.0  # T-30: publish floor answer unconditionally


class QType(str, Enum):
    NUMERICAL = "numerical"
    OBJECT_REFERENCE = "object_reference"
    INSTRUCTION_FOLLOWING = "instruction_following"


# Soft per-type exploration budgets (s) before answer-path pressure (architecture §5).
EXPLORE_BUDGET_S: dict[QType, float] = {
    QType.NUMERICAL: 210.0,
    QType.OBJECT_REFERENCE: 240.0,
    QType.INSTRUCTION_FOLLOWING: 270.0,
}
