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
