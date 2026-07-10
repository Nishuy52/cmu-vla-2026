"""SyntheticScene determinism + terrain intensity semantics."""
from __future__ import annotations

import numpy as np

from core.interfaces import TerrainPatch
from core.mocks.synthetic_scene import SyntheticScene


def test_same_seed_is_deterministic():
    a = SyntheticScene(42)
    b = SyntheticScene(42)
    a.populate_default(5)
    b.populate_default(5)
    ta = a.terrain_patch()
    tb = b.terrain_patch()
    assert np.array_equal(ta.points, tb.points)
    assert [o.label for o in a.objects] == [o.label for o in b.objects]
    assert np.allclose(
        [o.cx for o in a.objects], [o.cx for o in b.objects]
    )


def test_different_seed_differs():
    a = SyntheticScene(1)
    b = SyntheticScene(2)
    a.populate_default(5)
    b.populate_default(5)
    assert not np.allclose(
        [o.cx for o in a.objects], [o.cx for o in b.objects]
    )


def test_terrain_shape_and_dtype():
    s = SyntheticScene(0)
    s.populate_default(3)
    patch = s.terrain_patch()
    assert isinstance(patch, TerrainPatch)
    assert patch.points.ndim == 2
    assert patch.points.shape[1] == 4  # x, y, z, intensity
    assert patch.points.dtype == np.float32
    assert patch.extended is False


def test_terrain_spacing_about_point_one():
    s = SyntheticScene(0)
    patch = s.terrain_patch()
    xs = np.unique(patch.points[:, 0])
    diffs = np.diff(np.sort(xs))
    assert np.allclose(diffs, 0.1, atol=1e-3)


def test_free_floor_is_zero_intensity():
    s = SyntheticScene(0)  # no objects placed
    patch = s.terrain_patch()
    # a clearly interior point away from any wall
    interior = patch.points[
        (np.abs(patch.points[:, 0] - 2.5) < 0.05)
        & (np.abs(patch.points[:, 1] - 2.5) < 0.05)
    ]
    assert len(interior) >= 1
    assert np.all(interior[:, 3] == 0.0)


def test_object_footprint_raises_intensity():
    s = SyntheticScene(0)
    obj = s.place_box("sofa", 2.5, 2.5, sx=0.6, sy=0.6, sz=0.7)
    patch = s.terrain_patch()
    under = patch.points[
        (np.abs(patch.points[:, 0] - obj.cx) < 0.05)
        & (np.abs(patch.points[:, 1] - obj.cy) < 0.05)
    ]
    assert len(under) >= 1
    assert np.all(under[:, 3] > TerrainPatch.FREE_MAX)
    assert np.all(under[:, 3] >= 0.7 - 1e-6)


def test_walls_are_obstacles():
    s = SyntheticScene(0)
    patch = s.terrain_patch()
    # border row at y ~ 0 should carry wall intensity
    border = patch.points[np.abs(patch.points[:, 1] - 0.0) < 0.05]
    assert len(border) >= 1
    assert np.all(border[:, 3] > TerrainPatch.FREE_MAX)


def test_two_room_doorway_is_traversable():
    s = SyntheticScene(0, two_rooms=True)
    assert s.doorway is not None
    dx, dy = s.doorway
    patch = s.terrain_patch()
    gap = patch.points[
        (np.abs(patch.points[:, 0] - dx) < 0.05)
        & (np.abs(patch.points[:, 1] - dy) < 0.05)
    ]
    assert len(gap) >= 1
    # the doorway cell on the shared wall line must be free (traversable)
    assert np.all(gap[:, 3] <= TerrainPatch.FREE_MAX)


def test_two_room_shared_wall_blocks_off_doorway():
    s = SyntheticScene(0, two_rooms=True)
    dx, dy = s.doorway
    patch = s.terrain_patch()
    # a point on the shared wall well away from the doorway must be a wall
    off = patch.points[
        (np.abs(patch.points[:, 0] - dx) < 0.05)
        & (np.abs(patch.points[:, 1] - (dy + 2.0)) < 0.05)
    ]
    assert len(off) >= 1
    assert np.all(off[:, 3] > TerrainPatch.FREE_MAX)


def test_instances_match_placed_objects():
    s = SyntheticScene(7)
    s.populate_default(4)
    recs = s.instances()
    assert len(recs) == 4
    for rec, obj in zip(recs, s.objects):
        assert rec.label == obj.label
        assert rec.points is not None and len(rec.points) > 0
        assert np.allclose(rec.aabb_min, obj.aabb_min)
        assert np.allclose(rec.aabb_max, obj.aabb_max)
