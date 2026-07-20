"""SyntheticScene determinism + terrain intensity semantics."""
from __future__ import annotations

import math

import numpy as np

from core.interfaces import TerrainPatch
from core.mocks.synthetic_scene import GTObject, Room, SyntheticScene


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


def test_elevated_object_is_not_a_floor_obstacle():
    # A tabletop object (base above the terrain slab, e.g. a plant ON a cabinet) must
    # NOT stamp its footprint as a floor obstacle: the terrain stack filters overhangs
    # out, so the floor under/beside it reads FREE (T11 sig-1/2 root cause — every AABB
    # was stamped floor-to-top, sealing floor near most leg anchors).
    s = SyntheticScene(0)
    obj = s.place_box("potted plant", 2.5, 2.5, sx=0.3, sy=0.3, sz=0.2, cz=0.83)
    patch = s.terrain_patch()
    under = patch.points[
        (np.abs(patch.points[:, 0] - obj.cx) < 0.05)
        & (np.abs(patch.points[:, 1] - obj.cy) < 0.05)
    ]
    assert len(under) >= 1
    assert np.all(under[:, 3] == 0.0)  # free floor under the overhang
    # its AABB still carries the true base/top so resolvers + overhead layer see it.
    assert np.isclose(obj.aabb_min[2], 0.83)
    assert np.isclose(obj.aabb_max[2], 0.83 + 0.2)


def test_floor_mounted_object_still_blocks():
    # A floor-mounted object (base below the slab) stays a hard floor obstacle.
    s = SyntheticScene(0)
    obj = s.place_box("cabinet", 2.5, 2.5, sx=0.6, sy=0.6, sz=0.8, cz=0.0)
    patch = s.terrain_patch()
    under = patch.points[
        (np.abs(patch.points[:, 0] - obj.cx) < 0.05)
        & (np.abs(patch.points[:, 1] - obj.cy) < 0.05)
    ]
    assert len(under) >= 1
    assert np.all(under[:, 3] > TerrainPatch.FREE_MAX)


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


# --------------------------------------------------------------------------- #77 OBB rasterization


def _axis_aligned_contains(obj: GTObject, x: float, y: float) -> bool:
    """The pre-fix axis-aligned footprint test, independent of GTObject."""
    return abs(x - obj.cx) <= obj.sx / 2 and abs(y - obj.cy) <= obj.sy / 2


def test_footprint_contains_zero_heading_is_byte_identical_to_axis_aligned():
    """Regression guard (issue #77 Pre-Stage 1a): an unrotated box's stamp is
    unchanged by the OBB rasterization — heading=0.0 must reproduce the exact
    pre-fix axis-aligned test, not merely an equivalent-up-to-float-error one."""
    obj = GTObject(label="table", cx=1.3, cy=-0.7, sx=0.8, sy=1.4, sz=0.5, heading=0.0)
    xs = np.linspace(-2.0, 4.0, 61)
    ys = np.linspace(-3.0, 2.0, 51)
    for x in xs:
        for y in ys:
            assert obj.footprint_contains(float(x), float(y)) == _axis_aligned_contains(
                obj, float(x), float(y)
            )


def test_footprint_contains_90deg_heading_swaps_extents_exactly():
    """A box rotated exactly 90 deg is still axis-aligned (its own local x/y
    extents swap in world space) — no over- or under-coverage either way."""
    obj = GTObject(label="sofa", cx=0.0, cy=0.0, sx=2.0, sy=0.6, sz=0.7, heading=math.pi / 2)
    # World-x half-extent should now be the LOCAL y half-extent (0.3), and vice versa.
    assert obj.footprint_contains(0.29, 0.0)
    assert not obj.footprint_contains(0.31, 0.0)
    assert obj.footprint_contains(0.0, 0.99)
    assert not obj.footprint_contains(0.0, 1.01)


def test_footprint_contains_45deg_covers_fewer_cells_than_its_aabb_hull():
    """A rotated box's TRUE footprint (rasterized) covers strictly fewer lattice
    cells than its AABB hull (the pre-fix over-approximation) — the core claim
    of issue #77 Pre-Stage 1a."""
    heading = math.pi / 4
    sx, sy = 2.0, 0.6
    obj = GTObject(label="sofa", cx=0.0, cy=0.0, sx=sx, sy=sy, sz=0.7, heading=heading)
    half_x, half_y = sx / 2, sy / 2
    cos_h, sin_h = math.cos(heading), math.sin(heading)
    # AABB-of-OBB half-extents (mirrors core.groundtruth.loader.obb_to_aabb's formula).
    aabb_half_x = abs(half_x * cos_h) + abs(half_y * sin_h)
    aabb_half_y = abs(half_x * sin_h) + abs(half_y * cos_h)

    step = 0.05
    xs = np.arange(-aabb_half_x - 0.2, aabb_half_x + 0.2, step)
    ys = np.arange(-aabb_half_y - 0.2, aabb_half_y + 0.2, step)
    n_true = 0
    n_hull = 0
    for x in xs:
        for y in ys:
            x, y = float(x), float(y)
            in_hull = abs(x) <= aabb_half_x and abs(y) <= aabb_half_y
            in_true = obj.footprint_contains(x, y)
            if in_true:
                assert in_hull, "true OBB footprint must stay inside its own AABB hull"
            n_true += int(in_true)
            n_hull += int(in_hull)
    assert n_true < n_hull


def test_footprint_contains_arabic_room_sofa_case():
    """arabic_room object_id 56 (a sofa) — heading -3.1328664 rad, ~0.5 deg off
    a true 180 deg / axis-aligned orientation. Real fixture values from
    ``data/vla3d/Unity/arabic_room/arabic_room_object_result.csv``."""
    cx, cy = 2.0790001107131033, 0.6679999141611369
    sx, sy = 2.1899273413427003, 0.7787988067917222
    heading = -3.1328664111144175
    obj = GTObject(label="sofa", cx=cx, cy=cy, sx=sx, sy=sy, sz=0.68, heading=heading)

    # The near-axis-aligned true footprint stays inside its (only slightly larger)
    # AABB hull, and the hull covers a strictly larger area at the same lattice
    # resolution the mirror costmap stamps at.
    half_x, half_y = sx / 2, sy / 2
    cos_h, sin_h = math.cos(heading), math.sin(heading)
    aabb_half_x = abs(half_x * cos_h) + abs(half_y * sin_h)
    aabb_half_y = abs(half_x * sin_h) + abs(half_y * cos_h)
    assert aabb_half_x >= half_x  # over-approximation, as documented
    assert aabb_half_y >= half_y

    step = 0.02
    xs = np.arange(cx - aabb_half_x - 0.1, cx + aabb_half_x + 0.1, step)
    ys = np.arange(cy - aabb_half_y - 0.1, cy + aabb_half_y + 0.1, step)
    n_true = sum(
        1
        for x in xs
        for y in ys
        if obj.footprint_contains(float(x), float(y))
    )
    n_hull = sum(
        1
        for x in xs
        for y in ys
        if abs(float(x) - cx) <= aabb_half_x and abs(float(y) - cy) <= aabb_half_y
    )
    assert n_true < n_hull


def test_place_box_default_heading_is_zero():
    s = SyntheticScene(0)
    obj = s.place_box("chair", 1.0, 2.0, 0.5, 0.5, 0.5)
    assert obj.heading == 0.0


def test_place_box_carries_heading_through():
    s = SyntheticScene(0)
    obj = s.place_box("sofa", 1.0, 2.0, 2.0, 0.6, 0.7, heading=math.pi / 4)
    assert obj.heading == math.pi / 4
    assert s.objects[-1] is obj


def test_terrain_patch_rotated_object_stamps_fewer_cells_than_unrotated_hull():
    """End-to-end through ``terrain_patch``: a rotated 45 deg box stamps fewer
    floor cells than an unrotated box using the equivalent AABB-hull size."""
    aabb_sx, aabb_sy = 1.6, 1.6  # a square AABB hull can hold the rotated box exactly

    s_rot = SyntheticScene(0)
    s_rot.rooms = [Room(0, 0, 5, 5)]
    s_rot.place_box("sofa", 2.5, 2.5, 1.6, 0.5, 0.7, heading=math.pi / 4)
    patch_rot = s_rot.terrain_patch()
    n_rot = int(np.sum(patch_rot.points[:, 3] > TerrainPatch.FREE_MAX))

    s_hull = SyntheticScene(0)
    s_hull.rooms = [Room(0, 0, 5, 5)]
    s_hull.place_box("sofa", 2.5, 2.5, aabb_sx, aabb_sy, 0.7)  # heading=0 hull-sized stamp
    patch_hull = s_hull.terrain_patch()
    n_hull = int(np.sum(patch_hull.points[:, 3] > TerrainPatch.FREE_MAX))

    assert n_rot < n_hull
