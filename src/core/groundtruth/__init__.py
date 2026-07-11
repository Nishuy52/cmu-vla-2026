"""Ground-truth loading and REAL scoring for the VLA-3D Unity challenge scenes.

This package turns a VLA-3D per-scene folder (``_object_result.csv`` +
``_scene_graph.json`` + ``_region_result.csv``) into the core world-model type
(:class:`~core.interfaces.InstanceRecord`) so the battery can be driven against a
*fully observed* ground-truth scene and scored for ACCURACY (not just structural
health as :mod:`core.runner.battery` does).

Modules:
* :mod:`core.groundtruth.loader`  — VLA-3D folder -> ``GTScene`` (instances + region map)
* :mod:`core.groundtruth.scoring` — per-qtype scorers (count exact-match, 3D IoU,
  discrete Frechet + path-proximity for instruction following)
"""
from __future__ import annotations

from core.groundtruth.loader import (
    GTScene,
    GTRegion,
    load_scene,
    obb_to_aabb,
    parse_object_csv,
)

__all__ = [
    "GTScene",
    "GTRegion",
    "load_scene",
    "obb_to_aabb",
    "parse_object_csv",
]
