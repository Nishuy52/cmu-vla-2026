"""Frozen contracts for the VLA core. Everything in core/ codes against these types.

All spatial quantities are in the `map` frame, metres/radians/seconds, unless noted.
Mirrors the test-time ROS topic contract (docs/upstream_notes.md §3) without importing ROS.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, IntEnum
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

    def __post_init__(self) -> None:
        rgb = self.rgb
        if len(rgb) != 3:
            raise ValueError(f"ColorBin.rgb must have 3 channels, got {len(rgb)}: {rgb!r}")
        for ch, v in zip(("r", "g", "b"), rgb):
            if not isinstance(v, int) or isinstance(v, bool):
                raise ValueError(f"ColorBin.rgb.{ch} must be an int, got {v!r}")
            if not 0 <= v <= 255:
                raise ValueError(f"ColorBin.rgb.{ch} must be in [0, 255], got {v!r}")
        if not 0.0 <= self.fraction <= 1.0:
            raise ValueError(f"ColorBin.fraction must be in [0.0, 1.0], got {self.fraction!r}")

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
    #: Original oriented-box centre/full-extents/yaw (radians, +Z), when the producer
    #: has them (currently only ``core.groundtruth.loader``, straight off the VLA-3D
    #: CSV). ``aabb_min``/``aabb_max`` stay the AABB-of-OBB over-approximation used by
    #: every scoring/toolbox geometry predicate (frozen semantics) — these three are
    #: PURELY ADDITIVE, read only by the mirror-costmap stamping path (issue #77,
    #: pre_grounding_movement_plan.md Pre-Stage 1a) to rasterize the true rotated
    #: footprint instead of its AABB hull. ``None``/``0.0`` (the default) means "no OBB
    #: info" — callers fall back to the AABB, unchanged prior behaviour.
    obb_center: np.ndarray | None = None  # (3,) float
    obb_extents: np.ndarray | None = None  # (3,) float, full xyz lengths
    obb_heading: float = 0.0  # radians, rotation about +Z

    @property
    def extents(self) -> np.ndarray:
        return self.aabb_max - self.aabb_min

    def to_marker(self) -> MarkerBox:
        c = (self.aabb_min + self.aabb_max) / 2.0
        e = self.extents
        return MarkerBox(*map(float, c), *map(float, e), label=self.label)


class MatchTier(IntEnum):
    """Label-match provenance, ordered best-first (lower value = stronger match).

    Lives here (rather than in :mod:`core.perception.scene_index`, the sole current
    producer) so the :class:`SceneIndex` Protocol can name it in
    :meth:`SceneIndex.by_label_tiered` without a core -> perception import cycle.
    """

    EXACT = 0
    SYNONYM = 1
    HEAD_NOUN = 2
    TYPO = 3


@runtime_checkable
class SceneIndex(Protocol):
    """Read view over the instance map used by the toolbox and answer heads."""

    def all_instances(self) -> Sequence[InstanceRecord]: ...

    def by_label(self, noun: str) -> Sequence[InstanceRecord]:
        """Typo/plural/synonym-tolerant lookup ('refridgerator' -> fridge instances)."""
        ...

    def by_label_tiered(self, noun: str) -> Sequence[tuple[InstanceRecord, MatchTier]]:
        """Like :meth:`by_label` but pairs each hit with its :class:`MatchTier`.

        Required (issue #24): anchor resolution (``geometry.toolbox._match_anchor_noun``)
        depends on tier provenance to keep a modified anchor's exact/synonym referent
        from being satisfied by a differently-modified head-noun cousin (#13). Declaring
        it only on :class:`~core.perception.scene_index.BasicSceneIndex` let a future
        conforming index (a caching wrapper, a ROS-adapter-native index) type-check
        while silently reintroducing that bug with no exception, test failure, or log
        line. Promoting it into the Protocol makes the absence a structural/type error
        instead — every conforming :class:`SceneIndex` must expose real tier provenance.
        """
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
#
# NUMERICAL (#150): raised from the original 210.0 to 450.0. NUMERICAL answers in place
# (no drive-out phase, unlike INSTRUCTION_FOLLOWING), so exploring longer only ever adds
# sensing coverage -- it never trades away answer-assembly time the way it would for a
# qtype with post-answer work. Live evidence (#150) shows the merged detector/NMS stack
# cut numerical over-counting by ~82%, leaving under-counting (never having SEEN an
# instance) as the entire remaining error source; the old 210 s cap threw away ~270 s of
# usable sensing before the controller's effective forced-assembly gate
# (`fsm.budget.DEFAULT_FORCED_ASSEMBLY_S` = 480.0) would have forced VERIFY anyway.
#
# 450.0 is chosen as "just under 480.0" with a documented 30 s settle margin, not the
# literal ceiling, so a NUMERICAL question that is still accumulating observations near
# the gate gets a full settle window to satisfy `heads.numerical.STABLE_TICKS` (3
# consecutive equal counts) before losing the distinction between "explore budget spent"
# (`fsm.controller._tick_explore`) and "forced assembly" (`fsm.controller.tick`'s
# watchdog overlay) -- both routes still land in VERIFY either way, so the margin is a
# clarity/logging hedge, not a correctness requirement. 30 s also mirrors the existing
# SYS-F6 hedge between the neutral (510/570) and effective (480/540) forced-assembly /
# watchdog-floor pairs (`fsm.budget.py`), reusing the codebase's established margin size
# rather than inventing a new one (generalization protocol: no fresh magic number).
#
# This costs nothing in the common case: NUMERICAL's early-answer gate
# (`fsm.controller.QuestionController._early_answer_ready`) already exits EXPLORE_EXECUTE
# the moment the count is stable, so a small room that stabilises in seconds still
# answers in seconds -- the raised ceiling only matters for scenes that genuinely need the
# extra ~240 s to keep discovering un-seen instances.
EXPLORE_BUDGET_S: dict[QType, float] = {
    QType.NUMERICAL: 450.0,
    QType.OBJECT_REFERENCE: 240.0,
    QType.INSTRUCTION_FOLLOWING: 270.0,
}
