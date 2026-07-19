"""Corridor gate + threading check + avoid capsule/disc + violation detection."""
from __future__ import annotations

import numpy as np

from core.geometry import toolbox as T
from core.geometry.toolbox import DEFAULT_THRESHOLDS as TH
from core.plan_schema import Anchor, AvoidSpec
from tests.geometry._helpers import FakeIndex, rec


# --------------------------------------------------------------------------- gate


def test_corridor_gate_between_faces():
    # two boxes separated along x; gate spans the facing edges at the shared y.
    b1 = rec(1, "sofa", (-2, 0, 0), (1.0, 1.0, 1.0))  # x face at -1.5
    b2 = rec(2, "shelf", (2, 0, 0), (1.0, 1.0, 1.0))  # x face at 1.5
    g = T.corridor_gate(b1, b2)
    assert np.allclose(g.midpoint, [0.0, 0.0])
    assert abs(g.width - 3.0) < 1e-9  # face-to-face gap on x


def test_corridor_gate_midpoint_offset_boxes():
    b1 = rec(1, "a", (0, 0, 0), (1.0, 1.0, 1.0))
    b2 = rec(2, "b", (0, 4, 0), (1.0, 1.0, 1.0))  # separated along y
    g = T.corridor_gate(b1, b2)
    assert np.allclose(g.midpoint, [0.0, 2.0])


# --------------------------------------------------------------------------- issue #69 (D1)


def test_corridor_gate_nondegenerate_when_anchor_footprints_touch():
    """issue #69/D1: a big and a small anchor whose footprints already touch (e.g. a
    sofa and a round table wedged into its corner) used to collapse the gate to a
    single zero-width point via the axis-aligned face projection -- unthreadable by
    construction. The centroid-axis construction must produce a genuine two-point
    segment instead."""
    big = rec(1, "sofa", (-1.5, -2.0, 0.0), (2.0, 3.0, 1.0))  # x:[-2.5,-0.5] y:[-3.5,-0.5]
    small = rec(2, "round table", (-0.3, -2.4, 0.0), (0.8, 0.8, 0.3))  # touches big's corner
    g = T.corridor_gate(big, small)
    assert g.width > 0.01
    assert not np.allclose(g.p0, g.p1)


# --------------------------------------------------------------------------- issue #63


def test_corridor_gate_without_index_ignores_third_object_at_midpoint():
    """Pre-#63 behaviour preserved when no ``index`` is passed: the midpoint is never
    nudged, even if a third object's footprint would otherwise block it."""
    b1 = rec(1, "sofa", (-2, 0, 0), (1.0, 1.0, 1.0))
    b2 = rec(2, "shelf", (2, 0, 0), (1.0, 1.0, 1.0))
    blocker = rec(3, "bed frame", (0, 0, 0), (1.0, 1.0, 1.0))  # sits ON the raw midpoint
    idx = FakeIndex([b1, b2, blocker])
    g_no_index = T.corridor_gate(b1, b2)
    assert np.allclose(g_no_index.midpoint, [0.0, 0.0])
    assert idx is not None  # (idx unused: this asserts the no-index path is unaffected)


def test_corridor_gate_nudges_off_genuine_third_object_at_midpoint():
    """issue #63 (hotel_room_2 shape): a genuine third GT instance's footprint sits at
    the raw midpoint. With ``index`` given, the returned midpoint must be nudged clear
    of it, along the gate's own axis, while ``p0``/``p1`` (the verified anchor line
    used by threading checks) stay untouched."""
    b1 = rec(1, "bed", (-2, 0, 0), (1.0, 1.0, 1.0))  # x face at -1.5
    b2 = rec(2, "bench", (2, 0, 0), (1.0, 1.0, 1.0))  # x face at 1.5
    # A duplicate "bed frame" instance for the same physical bed, straddling the raw
    # midpoint (0, 0) but not either anchor's own AABB.
    blocker = rec(3, "bed frame", (0, 0, 0), (0.8, 0.8, 1.0))
    idx = FakeIndex([b1, b2, blocker])
    g = T.corridor_gate(b1, b2, idx)
    assert np.allclose(g.p0, [-1.5, 0.0])
    assert np.allclose(g.p1, [1.5, 0.0])
    assert not np.allclose(g.midpoint, [0.0, 0.0]), "midpoint must move off the blocker"
    # Nudged point must stay ON the gate's own axis (y == 0) and inside the segment.
    assert abs(float(g.midpoint[1])) < 1e-9
    assert -1.5 < float(g.midpoint[0]) < 1.5
    # And it must actually be clear of the blocker's footprint.
    assert not T.P.point_in_footprint_2d(g.midpoint, blocker.aabb_min, blocker.aabb_max)


def test_corridor_gate_falls_back_to_midpoint_when_gate_fully_blocked():
    """When nothing within the bounded slide clears (a blocker spanning the whole
    gate), corridor_gate must fall back to the untouched midpoint rather than nudge
    past a verified anchor."""
    b1 = rec(1, "wallA", (-2, 0, 0), (1.0, 1.0, 1.0))
    b2 = rec(2, "wallB", (2, 0, 0), (1.0, 1.0, 1.0))
    # Blocker footprint spans the entire gate span (x in [-1.5, 1.5]).
    blocker = rec(3, "unknown", (0, 0, 0), (10.0, 10.0, 1.0))
    idx = FakeIndex([b1, b2, blocker])
    g = T.corridor_gate(b1, b2, idx)
    assert np.allclose(g.midpoint, [0.0, 0.0])


# --------------------------------------------------------------------------- threading


def _gate_xy(p0, p1):
    return T.Gate(np.array(p0, float), np.array(p1, float),
                  (np.array(p0, float) + np.array(p1, float)) / 2, 1.0)


def test_threading_crossing():
    gate = _gate_xy([-1, 0], [1, 0])  # gate along x-axis at y=0
    traj = np.array([[0, -2], [0, 2]])  # goes straight through
    ok, expl = T.threading_check(traj, gate)
    assert ok and "crossed" in expl


def test_threading_same_side_approach():
    gate = _gate_xy([-1, 0], [1, 0])
    traj = np.array([[0, -2], [0, -0.5], [0.5, -0.3]])  # approaches but never crosses y=0
    ok, _ = T.threading_check(traj, gate)
    assert not ok


def test_threading_touching_endpoint():
    gate = _gate_xy([-1, 0], [1, 0])
    # trajectory ends exactly on the gate endpoint (touch counts as crossing)
    traj = np.array([[1, -2], [1, 0]])
    ok, _ = T.threading_check(traj, gate)
    assert ok


def test_threading_parallel_no_cross():
    gate = _gate_xy([-1, 0], [1, 0])
    traj = np.array([[-2, 1], [2, 1]])  # parallel, 1 m above; never crosses
    ok, _ = T.threading_check(traj, gate)
    assert not ok


def test_threading_short_trajectory():
    gate = _gate_xy([-1, 0], [1, 0])
    ok, expl = T.threading_check(np.array([[0, 0]]), gate)
    assert not ok and ">=2" in expl


# --------------------------------------------------------------------------- avoid capsule (between)


def test_avoid_capsule_between():
    b1 = rec(1, "desk", (-3, 0, 0), (1.0, 1.0, 1.0))  # half-width 0.5
    b2 = rec(2, "cabinet", (3, 0, 0), (2.0, 1.0, 1.0))  # half-width 1.0
    idx = FakeIndex([b1, b2])
    spec = AvoidSpec(between=[Anchor(noun="desk"), Anchor(noun="cabinet")])
    cap = T.avoid_capsule(spec, idx)
    assert np.allclose(cap.a, [-3, 0]) and np.allclose(cap.b, [3, 0])
    # radius = max half-width (1.0) + 0.25 inflation
    assert abs(cap.radius - (1.0 + TH.avoid_inflate)) < 1e-9


def test_avoid_disc_near():
    anchor = rec(1, "fountain", (0, 0, 0), (0.2, 0.2, 0.5))  # small -> near floor 1.2
    idx = FakeIndex([anchor])
    spec = AvoidSpec(near=Anchor(noun="fountain"))
    cap = T.avoid_capsule(spec, idx)
    assert np.allclose(cap.a, cap.b)  # disc: degenerate segment
    assert abs(cap.radius - (1.2 + TH.avoid_inflate)) < 1e-9


def test_avoid_capsule_missing_anchor_raises():
    idx = FakeIndex([rec(1, "desk", (0, 0, 0))])
    spec = AvoidSpec(between=[Anchor(noun="desk"), Anchor(noun="ghost")])
    try:
        T.avoid_capsule(spec, idx)
        assert False, "expected ValueError"
    except ValueError:
        pass


# --------------------------------------------------------------------------- capsule violation


def test_capsule_violated_inside():
    cap = T.Capsule(np.array([-3.0, 0.0]), np.array([3.0, 0.0]), 1.0)
    traj = np.array([[-5, 0.5], [0, 0.5], [5, 0.5]])  # midpoint inside (dist 0.5 < 1.0)
    hit, pt = T.capsule_violated(traj, cap)
    assert hit and pt is not None


def test_capsule_not_violated_around():
    cap = T.Capsule(np.array([-3.0, 0.0]), np.array([3.0, 0.0]), 1.0)
    traj = np.array([[-5, 3], [0, 3], [5, 3]])  # 3 m above the capsule axis
    hit, pt = T.capsule_violated(traj, cap)
    assert not hit and pt is None


def test_capsule_violated_disc():
    cap = T.Capsule(np.array([0.0, 0.0]), np.array([0.0, 0.0]), 1.2)  # disc
    traj = np.array([[3, 3], [0.5, 0.5]])  # last point within 1.2 of origin
    hit, pt = T.capsule_violated(traj, cap)
    assert hit


def test_capsule_violated_mid_edge():
    # both vertices are outside, but the edge dips through the capsule.
    cap = T.Capsule(np.array([0.0, 0.0]), np.array([0.0, 0.0]), 0.5)  # small disc at origin
    traj = np.array([[-2, 0], [2, 0]])  # straight through origin; vertices far, midpoint hits
    hit, pt = T.capsule_violated(traj, cap)
    assert hit
