"""ROS 2 bag reader: deserialise the five test-time topics into core dataclasses.

Pure Python via ``rosbags`` (no ROS). :class:`BagSource` iterates a bag as
``(topic, timestamp_s, converted)`` triples, applying a topic-name -> converter registry.
The five defaults mirror the challenge I/O contract (docs/upstream_notes.md §3):

    /camera/image        sensor_msgs/Image        -> PanoFrame  (odom attached separately)
    /registered_scan     sensor_msgs/PointCloud2  -> LidarScan  (N,3 float32 xyz, map frame)
    /terrain_map         sensor_msgs/PointCloud2  -> TerrainPatch (N,4 xyzi, extended=False)
    /terrain_map_ext     sensor_msgs/PointCloud2  -> TerrainPatch (extended=True)
    /state_estimation    nav_msgs/Odometry        -> OdomState  (yaw from quaternion)
    /challenge_question  std_msgs/String          -> Question

Point-cloud converters read the ``PointField`` offsets from the message rather than
assuming a fixed XYZI layout, so nonstandard byte layouts round-trip correctly.
Timestamps are the ROS *message* header stamp when present, else the bag-recorded time,
both normalised to float seconds.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Iterator

import numpy as np

from core.interfaces import (
    LidarScan,
    OdomState,
    PanoFrame,
    Question,
    TerrainPatch,
)

# --------------------------------------------------------------------------- constants

# ROS topic names (docs/upstream_notes.md §3). Kept as module constants so callers can
# build custom registries without magic strings.
TOPIC_CAMERA = "/camera/image"
TOPIC_SCAN = "/registered_scan"
TOPIC_TERRAIN = "/terrain_map"
TOPIC_TERRAIN_EXT = "/terrain_map_ext"
TOPIC_ODOM = "/state_estimation"
TOPIC_QUESTION = "/challenge_question"

# sensor_msgs/PointField datatype enum -> numpy dtype (values are the ROS constants).
_PF_DTYPE: dict[int, np.dtype] = {
    1: np.dtype(np.int8),
    2: np.dtype(np.uint8),
    3: np.dtype(np.int16),
    4: np.dtype(np.uint16),
    5: np.dtype(np.int32),
    6: np.dtype(np.uint32),
    7: np.dtype(np.float32),
    8: np.dtype(np.float64),
}


# --------------------------------------------------------------------------- time helpers


def _stamp_to_s(stamp: Any) -> float | None:
    """builtin_interfaces/Time -> float seconds, or None if the field is absent."""
    if stamp is None:
        return None
    sec = getattr(stamp, "sec", None)
    nsec = getattr(stamp, "nanosec", None)
    if sec is None or nsec is None:
        return None
    return float(sec) + float(nsec) * 1e-9


def _header_time(msg: Any, bag_ns: int) -> float:
    """Prefer the message header stamp; fall back to the bag-recorded nanosecond time."""
    hdr = getattr(msg, "header", None)
    if hdr is not None:
        t = _stamp_to_s(getattr(hdr, "stamp", None))
        if t is not None and t > 0.0:
            return t
    return float(bag_ns) * 1e-9


# --------------------------------------------------------------------------- converters


def image_to_pano(msg: Any, bag_ns: int, *, odom: OdomState | None = None) -> PanoFrame:
    """sensor_msgs/Image -> PanoFrame (H, W, 3) uint8 RGB.

    Handles ``rgb8`` and ``bgr8`` encodings (bgr8 channels are swapped to RGB).
    ``odom`` is the nearest odometry to attach; a zero placeholder is used if None
    (BagSource fills this from the odom stream — see :meth:`BagSource.frames`).
    """
    height = int(msg.height)
    width = int(msg.width)
    encoding = str(msg.encoding).lower()
    data = np.asarray(msg.data, dtype=np.uint8)

    if encoding in ("rgb8", "bgr8"):
        channels = 3
    elif encoding in ("rgba8", "bgra8"):
        channels = 4
    else:
        raise ValueError(f"unsupported image encoding {encoding!r} (expected rgb8/bgr8)")

    expected = height * width * channels
    if data.size < expected:
        raise ValueError(
            f"image data too short: {data.size} < {expected} for {height}x{width}x{channels}"
        )
    # step may pad rows; slice per row to be layout-safe.
    step = int(msg.step) if int(msg.step) else width * channels
    if step == width * channels:
        img = data[:expected].reshape(height, width, channels)
    else:
        rows = [data[r * step : r * step + width * channels] for r in range(height)]
        img = np.stack(rows).reshape(height, width, channels)

    if channels == 4:
        img = img[:, :, :3]
    if encoding.startswith("bgr"):
        img = img[:, :, ::-1]

    img = np.ascontiguousarray(img, dtype=np.uint8)
    if odom is None:
        t0 = _header_time(msg, bag_ns)
        odom = OdomState(t=t0, x=0.0, y=0.0, z=0.0, yaw=0.0)
    return PanoFrame(t=_header_time(msg, bag_ns), image=img, odom=odom)


def _read_pointcloud(msg: Any) -> dict[str, np.ndarray]:
    """Decode a PointCloud2 into a per-field column dict, honouring PointField offsets.

    Reads each field's byte ``offset``/``datatype`` from the message description rather
    than assuming a fixed stride/layout, so nonstandard packings decode correctly.
    """
    point_step = int(msg.point_step)
    n_points = int(msg.width) * int(msg.height)
    buf = np.asarray(msg.data, dtype=np.uint8)
    if buf.size < n_points * point_step:
        raise ValueError(
            f"pointcloud data too short: {buf.size} < {n_points * point_step}"
        )
    buf = buf[: n_points * point_step].reshape(n_points, point_step)

    out: dict[str, np.ndarray] = {}
    for field in msg.fields:
        dt = _PF_DTYPE.get(int(field.datatype))
        if dt is None:
            continue  # skip unknown datatypes rather than fail the whole cloud
        offset = int(field.offset)
        width = dt.itemsize
        if offset + width > point_step:
            raise ValueError(
                f"field {field.name!r} offset {offset}+{width} exceeds point_step {point_step}"
            )
        raw = buf[:, offset : offset + width].copy()
        col = raw.view(dt).reshape(n_points)
        if bool(getattr(msg, "is_bigendian", False)):
            col = col.byteswap()
        out[str(field.name)] = col.astype(np.float32, copy=False)
    return out


def pointcloud_to_lidar(msg: Any, bag_ns: int) -> LidarScan:
    """sensor_msgs/PointCloud2 -> LidarScan: (N, 3) float32 xyz (map frame)."""
    cols = _read_pointcloud(msg)
    n = len(next(iter(cols.values()))) if cols else 0
    x = cols.get("x", np.zeros(n, np.float32))
    y = cols.get("y", np.zeros(n, np.float32))
    z = cols.get("z", np.zeros(n, np.float32))
    pts = np.column_stack([x, y, z]).astype(np.float32, copy=False)
    return LidarScan(t=_header_time(msg, bag_ns), points=pts)


def pointcloud_to_terrain(
    msg: Any, bag_ns: int, *, extended: bool = False
) -> TerrainPatch:
    """sensor_msgs/PointCloud2 -> TerrainPatch: (N, 4) float32 [x, y, z, intensity]."""
    cols = _read_pointcloud(msg)
    n = len(next(iter(cols.values()))) if cols else 0
    x = cols.get("x", np.zeros(n, np.float32))
    y = cols.get("y", np.zeros(n, np.float32))
    z = cols.get("z", np.zeros(n, np.float32))
    i = cols.get("intensity", np.zeros(n, np.float32))
    pts = np.column_stack([x, y, z, i]).astype(np.float32, copy=False)
    return TerrainPatch(t=_header_time(msg, bag_ns), points=pts, extended=extended)


def _quat_to_yaw(x: float, y: float, z: float, w: float) -> float:
    """Yaw (radians, REP-103 CCW-from-+x) from a quaternion."""
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def odom_to_state(msg: Any, bag_ns: int) -> OdomState:
    """nav_msgs/Odometry -> OdomState; yaw extracted from the pose quaternion."""
    pose = msg.pose.pose
    p = pose.position
    q = pose.orientation
    yaw = _quat_to_yaw(float(q.x), float(q.y), float(q.z), float(q.w))
    return OdomState(
        t=_header_time(msg, bag_ns),
        x=float(p.x),
        y=float(p.y),
        z=float(p.z),
        yaw=yaw,
    )


def string_to_question(msg: Any, bag_ns: int) -> Question:
    """std_msgs/String -> Question (t_received = header/bag time)."""
    return Question(text=str(msg.data), t_received=float(bag_ns) * 1e-9)


# A converter takes (deserialized_msg, bag_timestamp_ns) -> a core dataclass.
Converter = Callable[[Any, int], Any]

# Default topic-name -> converter registry (overridable per BagSource).
DEFAULT_TOPIC_MAP: dict[str, Converter] = {
    TOPIC_CAMERA: image_to_pano,
    TOPIC_SCAN: pointcloud_to_lidar,
    TOPIC_TERRAIN: pointcloud_to_terrain,
    TOPIC_TERRAIN_EXT: lambda m, ns: pointcloud_to_terrain(m, ns, extended=True),
    TOPIC_ODOM: odom_to_state,
    TOPIC_QUESTION: string_to_question,
}


# --------------------------------------------------------------------------- source


@dataclass(frozen=True)
class BagRecord:
    """One converted message: ``(topic, t, msg)`` where ``t`` is float seconds."""

    topic: str
    t: float
    msg: Any


class BagSource:
    """Iterate a ROS 2 bag, converting the registered topics to core dataclasses.

    Example::

        for rec in BagSource(path):
            rec.topic, rec.t, rec.msg  # e.g. LidarScan

    ``topic_map`` overrides/extends :data:`DEFAULT_TOPIC_MAP`. Topics with no registered
    converter are skipped. Iteration yields records in bag time order.
    """

    def __init__(
        self,
        path,
        topic_map: dict[str, Converter] | None = None,
    ) -> None:
        # Lazy import so importing this module never requires rosbags to be installed
        # until a bag is actually read.
        from pathlib import Path

        self.path = Path(path)
        self.topic_map: dict[str, Converter] = dict(DEFAULT_TOPIC_MAP)
        if topic_map:
            self.topic_map.update(topic_map)

    def __iter__(self) -> Iterator[BagRecord]:
        from rosbags.highlevel import AnyReader

        with AnyReader([self.path]) as reader:
            wanted = {
                c.topic for c in reader.connections if c.topic in self.topic_map
            }
            gen = reader.messages(
                connections=[c for c in reader.connections if c.topic in wanted]
            )
            for conn, timestamp, raw in gen:
                conv = self.topic_map.get(conn.topic)
                if conv is None:
                    continue
                msg = reader.deserialize(raw, conn.msgtype)
                yield BagRecord(conn.topic, float(timestamp) * 1e-9, conv(msg, timestamp))

    def frames(self) -> Iterator[BagRecord]:
        """Like ``__iter__`` but attaches the nearest-preceding odom to each PanoFrame.

        Camera frames converted through :func:`image_to_pano` carry a placeholder odom;
        this pass replaces it with the newest :class:`OdomState` seen at or before the
        frame time so downstream consumers get a populated ``PanoFrame.odom``.
        """
        latest_odom: OdomState | None = None
        for rec in self:
            if isinstance(rec.msg, OdomState):
                latest_odom = rec.msg
                yield rec
            elif isinstance(rec.msg, PanoFrame) and latest_odom is not None:
                yield BagRecord(
                    rec.topic,
                    rec.t,
                    PanoFrame(t=rec.msg.t, image=rec.msg.image, odom=latest_odom),
                )
            else:
                yield rec
