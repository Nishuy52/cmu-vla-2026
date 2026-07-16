"""Seeded synthetic indoor scene generator for offline tests.

Builds a deterministic room (or two rooms joined by a doorway gap), a set of
ground-truth :class:`~core.interfaces.InstanceRecord` objects, and a floor-sampled
:class:`~core.interfaces.TerrainPatch` point cloud whose intensity channel follows
the interfaces contract: intensity = obstacle height above local ground (metres),
0 on free floor, > FREE_MAX under object footprints and walls.

All geometry is in the `map` frame; units metres. Determinism: every stochastic
choice is driven by a seeded numpy Generator, so `SyntheticScene(seed)` is a pure
function of `seed`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from core.interfaces import InstanceRecord, TerrainPatch

FLOOR_SPACING: float = 0.1  # m between terrain samples
WALL_HEIGHT: float = 2.4  # m; intensity stamped for wall/border cells
DEFAULT_OBJ_HEIGHT: float = 0.5  # m; footprint intensity when box extent z unknown

#: Base-height (m) at or above which an object's underside is treated as an OVERHANG
#: rather than a floor obstacle in the terrain mirror. Mirrors the real terrain stack:
#: terrainAnalysis.cpp filters /registered_scan to a thin slab (~0.2 m above the
#: vehicle) so tabletop/shelf/wall-mounted objects (a plant on a cabinet, a vase on a
#: shelf) are ABSENT from /terrain_map — the floor under/beside them reads FREE. Kept in
#: step with occupancy.OverheadConfig.overhead_min so mirror terrain and the overhead
#: layer agree on where "floor obstacle" ends and "drive-beside overhang" begins.
TERRAIN_SLAB_MAX_Z: float = 0.25


@dataclass(frozen=True)
class Room:
    """Axis-aligned rectangular room footprint in the map frame (metres)."""

    x0: float
    y0: float
    x1: float
    y1: float

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def depth(self) -> float:
        return self.y1 - self.y0

    def contains(self, x: float, y: float) -> bool:
        return self.x0 <= x <= self.x1 and self.y0 <= y <= self.y1


@dataclass
class GTObject:
    """Ground-truth placement record used to build both instances and terrain."""

    label: str
    cx: float
    cy: float
    sx: float
    sy: float
    sz: float
    #: base height (z of the object's underside above the floor). 0.0 = floor-mounted
    #: (the default, so pre-existing callers are unchanged). A tabletop object (a plant
    #: on a cabinet, a vase on a shelf) carries its real base, so the terrain mirror can
    #: tell floor obstacles from overhangs the vehicle drives beside/under (IF-F2/H13).
    cz: float = 0.0

    @property
    def aabb_min(self) -> np.ndarray:
        return np.array(
            [self.cx - self.sx / 2, self.cy - self.sy / 2, self.cz], dtype=float
        )

    @property
    def aabb_max(self) -> np.ndarray:
        return np.array(
            [self.cx + self.sx / 2, self.cy + self.sy / 2, self.cz + self.sz], dtype=float
        )

    def footprint_contains(self, x: float, y: float) -> bool:
        return (
            abs(x - self.cx) <= self.sx / 2 and abs(y - self.cy) <= self.sy / 2
        )


class SyntheticScene:
    """Deterministic synthetic scene, fully determined by its integer seed.

    Attributes
    ----------
    rooms : list[Room]
        One room, or two rooms joined by a doorway gap when ``two_rooms``.
    doorway : tuple[float, float] | None
        (x, y) centre of the doorway gap between rooms, if any.
    objects : list[GTObject]
        Ground-truth object placements.
    """

    def __init__(
        self,
        seed: int = 0,
        *,
        two_rooms: bool = False,
        wall_thickness: float = 0.15,
        doorway_width: float = 1.0,
    ) -> None:
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.two_rooms = bool(two_rooms)
        self.wall_thickness = float(wall_thickness)
        self.doorway_width = float(doorway_width)
        self.objects: list[GTObject] = []

        if two_rooms:
            # Two rooms side by side sharing the wall at x == split, with a gap.
            self.rooms = [Room(0.0, 0.0, 4.0, 5.0), Room(4.0, 0.0, 8.0, 5.0)]
            self._split_x = 4.0
            self.doorway = (4.0, 2.5)
        else:
            self.rooms = [Room(0.0, 0.0, 5.0, 5.0)]
            self._split_x = None
            self.doorway = None

    # ------------------------------------------------------------------ objects

    def place_box(
        self,
        label: str,
        x: float,
        y: float,
        sx: float = 0.5,
        sy: float = 0.5,
        sz: float = DEFAULT_OBJ_HEIGHT,
        cz: float = 0.0,
    ) -> GTObject:
        """Place a labeled box centred at (x, y) with full extents (sx, sy, sz).

        ``cz`` is the base height (z of the underside above the floor); 0.0 keeps the
        legacy floor-mounted behaviour. Returns the created :class:`GTObject`.
        """
        obj = GTObject(label=label.strip().lower(), cx=float(x), cy=float(y),
                       sx=float(sx), sy=float(sy), sz=float(sz), cz=float(cz))
        self.objects.append(obj)
        return obj

    def populate_default(self, n: int = 5) -> list[GTObject]:
        """Deterministically scatter ``n`` labelled boxes across the floor.

        Placement is seeded; labels cycle through a small furniture vocabulary.
        """
        vocab = ["sofa", "table", "chair", "pillow", "lamp", "picture", "vase"]
        placed: list[GTObject] = []
        for i in range(n):
            room = self.rooms[i % len(self.rooms)]
            # keep boxes clear of walls by a 0.6 m margin
            x = float(self.rng.uniform(room.x0 + 0.6, room.x1 - 0.6))
            y = float(self.rng.uniform(room.y0 + 0.6, room.y1 - 0.6))
            label = vocab[i % len(vocab)]
            sx = float(self.rng.uniform(0.3, 0.8))
            sy = float(self.rng.uniform(0.3, 0.8))
            sz = float(self.rng.uniform(0.4, 1.0))
            placed.append(self.place_box(label, x, y, sx, sy, sz))
        return placed

    def instances(self) -> list[InstanceRecord]:
        """Ground-truth :class:`InstanceRecord` list, one per placed object.

        Point clouds are the object-surface samples used by mock lidar; AABBs are
        the exact box bounds (a real perception stack would trim to percentiles).
        """
        recs: list[InstanceRecord] = []
        for i, obj in enumerate(self.objects):
            pts = self._object_surface_points(obj)
            amin = obj.aabb_min
            amax = obj.aabb_max
            centroid = (amin + amax) / 2.0
            recs.append(
                InstanceRecord(
                    instance_id=i,
                    label=obj.label,
                    score=1.0,
                    n_obs=3,
                    centroid=centroid,
                    aabb_min=amin,
                    aabb_max=amax,
                    points=pts,
                )
            )
        return recs

    def _object_surface_points(self, obj: GTObject) -> np.ndarray:
        """~(M, 3) points sampled on the object's AABB faces (deterministic)."""
        amin = obj.aabb_min
        amax = obj.aabb_max
        # Grid the top face plus a ring of side samples; cheap and deterministic.
        xs = np.linspace(amin[0], amax[0], 5)
        ys = np.linspace(amin[1], amax[1], 5)
        gx, gy = np.meshgrid(xs, ys)
        top = np.column_stack(
            [gx.ravel(), gy.ravel(), np.full(gx.size, amax[2])]
        )
        # side samples at mid-height around the footprint
        zc = (amin[2] + amax[2]) / 2.0
        ring = np.array(
            [
                [amin[0], obj.cy, zc],
                [amax[0], obj.cy, zc],
                [obj.cx, amin[1], zc],
                [obj.cx, amax[1], zc],
            ]
        )
        return np.vstack([top, ring]).astype(float)

    # ------------------------------------------------------------------ terrain

    def _bounds(self) -> tuple[float, float, float, float]:
        x0 = min(r.x0 for r in self.rooms)
        y0 = min(r.y0 for r in self.rooms)
        x1 = max(r.x1 for r in self.rooms)
        y1 = max(r.y1 for r in self.rooms)
        return x0, y0, x1, y1

    def _in_any_room(self, x: float, y: float) -> bool:
        return any(r.contains(x, y) for r in self.rooms)

    def _is_wall(self, x: float, y: float) -> bool:
        """A cell is a wall if within wall_thickness of a room border and not
        inside the doorway gap between two rooms."""
        t = self.wall_thickness
        x0, y0, x1, y1 = self._bounds()
        near_border = (
            abs(x - x0) <= t
            or abs(x - x1) <= t
            or abs(y - y0) <= t
            or abs(y - y1) <= t
        )
        wall = near_border
        # Interior shared wall for two-room scenes.
        if self._split_x is not None:
            near_split = abs(x - self._split_x) <= t
            if near_split:
                # doorway gap: no wall within +/- doorway_width/2 of doorway y
                if self.doorway is not None and abs(y - self.doorway[1]) <= (
                    self.doorway_width / 2
                ):
                    near_split = False
                wall = wall or near_split
        # A border cell that falls in the doorway gap is not a wall.
        if (
            self._split_x is not None
            and self.doorway is not None
            and abs(x - self._split_x) <= t
            and abs(y - self.doorway[1]) <= (self.doorway_width / 2)
        ):
            return False
        return wall

    def terrain_patch(self, *, extended: bool = False, t: float = 0.0) -> TerrainPatch:
        """Floor-sampled (N, 4) [x, y, z, intensity] cloud at ~0.1 m spacing.

        intensity semantics (interfaces.TerrainPatch): obstacle height above local
        ground in metres. 0 on free floor; object footprints carry the box height;
        walls carry WALL_HEIGHT. Points outside all rooms are omitted (unmapped).
        """
        x0, y0, x1, y1 = self._bounds()
        xs = np.arange(x0, x1 + FLOOR_SPACING / 2, FLOOR_SPACING)
        ys = np.arange(y0, y1 + FLOOR_SPACING / 2, FLOOR_SPACING)
        rows: list[tuple[float, float, float, float]] = []
        for x in xs:
            for y in ys:
                x = float(x)
                y = float(y)
                wall = self._is_wall(x, y)
                if not wall and not self._in_any_room(x, y):
                    continue  # unmapped void beyond the rooms
                if wall:
                    intensity = WALL_HEIGHT
                else:
                    intensity = 0.0
                    for obj in self.objects:
                        # An elevated object (base above the terrain slab) is an overhang,
                        # not a floor obstacle: the terrain stack filters it out, so the
                        # cell under/beside it reads FREE. Its surface points still feed the
                        # overhead layer (via mock lidar) so the planner soft-avoids it.
                        if obj.cz >= TERRAIN_SLAB_MAX_Z:
                            continue
                        if obj.footprint_contains(x, y):
                            intensity = max(intensity, obj.sz)
                rows.append((x, y, 0.0, intensity))
        points = np.array(rows, dtype=np.float32).reshape(-1, 4)
        return TerrainPatch(t=float(t), points=points, extended=bool(extended))
