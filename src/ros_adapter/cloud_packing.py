"""Pure-numpy PointCloud2 byte packing for the colored voxel map (no ROS imports).

The rclpy adapter cannot run on the Windows dev box, so the byte-level layout of the
colored-map ``sensor_msgs/PointCloud2`` is built here as a pure function and unit-tested
on Windows (``src/tests/ros_adapter/test_cloud_packing.py``); the node wrapper only
stuffs these bytes/fields into the ROS message struct.

Layout — the standard RViz-friendly colored point (``PointField`` offsets in bytes)::

    x    float32  offset 0
    y    float32  offset 4
    z    float32  offset 8
    rgb  float32  offset 12   # uint32 0x00RRGGBB reinterpreted as float32

``point_step`` is 16 bytes; the buffer is ``row_step = point_step * N`` bytes,
little-endian, ``height = 1``, ``is_dense = True``. The ``rgb`` field is the RViz
convention: pack ``(R<<16) | (G<<8) | B`` into a uint32 and reinterpret its bits as a
float32 so it rides in a float field (RViz's "RGB8" color transformer decodes it back).

:data:`FLOAT32` (7) is the ``sensor_msgs/PointField`` datatype code for float32.
"""
from __future__ import annotations

import numpy as np

#: sensor_msgs/PointField datatype code for float32 (see PointField.msg constants).
FLOAT32: int = 7

#: Bytes per point in the packed buffer: 4 float32 fields (x, y, z, rgb).
POINT_STEP: int = 16

#: (name, offset_bytes, datatype_code, count) for each field in the packed layout.
FIELDS: list[tuple[str, int, int, int]] = [
    ("x", 0, FLOAT32, 1),
    ("y", 4, FLOAT32, 1),
    ("z", 8, FLOAT32, 1),
    ("rgb", 12, FLOAT32, 1),
]

#: Structured dtype matching the packed layout (little-endian), used to build the buffer.
_CLOUD_DTYPE = np.dtype(
    {
        "names": ["x", "y", "z", "rgb"],
        "formats": ["<f4", "<f4", "<f4", "<f4"],
        "offsets": [0, 4, 8, 12],
        "itemsize": POINT_STEP,
    }
)


def pack_rgb_float(rgb: np.ndarray) -> np.ndarray:
    """Pack ``rgb`` (N, 3) uint8 into the RViz ``rgb`` float32 field (bit-reinterpreted).

    Each row becomes ``(R << 16) | (G << 8) | B`` as a uint32 whose bits are reinterpreted
    as a float32 — the layout RViz's RGB8 color transformer decodes.
    """
    rgb = np.ascontiguousarray(rgb).astype(np.uint32)
    packed = (rgb[:, 0] << 16) | (rgb[:, 1] << 8) | rgb[:, 2]
    return packed.view(np.float32)


def pack_colored_cloud(
    xyz: np.ndarray, rgb: np.ndarray
) -> tuple[bytes, list[tuple[str, int, int, int]], int]:
    """Pack ``(xyz, rgb)`` into a PointCloud2 byte buffer + field descriptors.

    ``xyz`` is (N, 3) float32 map-frame positions, ``rgb`` is (N, 3) uint8 RGB. Returns
    ``(data, fields, point_step)`` where ``data`` is the little-endian buffer
    (``point_step * N`` bytes), ``fields`` is :data:`FIELDS` (``(name, offset, datatype,
    count)`` per field), and ``point_step`` is :data:`POINT_STEP` (16). ``height`` is 1
    and ``is_dense`` true; the caller stuffs these into the ROS message.
    """
    xyz = np.ascontiguousarray(xyz, dtype=np.float32)
    rgb = np.ascontiguousarray(rgb).astype(np.uint8)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"xyz must be (N, 3), got {xyz.shape}")
    if rgb.shape != xyz.shape:
        raise ValueError(f"rgb must match xyz shape {xyz.shape}, got {rgb.shape}")

    n = len(xyz)
    buf = np.empty(n, dtype=_CLOUD_DTYPE)
    buf["x"] = xyz[:, 0]
    buf["y"] = xyz[:, 1]
    buf["z"] = xyz[:, 2]
    buf["rgb"] = pack_rgb_float(rgb)
    return buf.tobytes(), list(FIELDS), POINT_STEP
