"""Issue #132: standoff/via clearance-field accounting gap.

``_near_thresh`` (shared by the terminal-leg standoff, #126, and the
long-standing VIA_NEAR placement) computes its push distance from the anchor
CENTROID as ``half_diag + VIA_NEAR_CLEARANCE_M``. But the costmap's blocked
mask (``Costmap.base_blocked``, read by ``_clearance_field``) is the anchor's
raw footprint ALREADY inflated by ``Costmap.vehicle_radius_m`` -- so the true
distance from the centroid to the blocked boundary is
``half_diag + vehicle_radius_m``, not ``half_diag``. ``_near_thresh`` omits
the inflation radius entirely, so a push aimed at "``VIA_NEAR_CLEARANCE_M``
past the boundary" actually lands only
``VIA_NEAR_CLEARANCE_M - vehicle_radius_m`` past it -- the goal falls short
of the target clearance by (approximately) one inflation radius.

This test reproduces the gap on the four real terminal-leg anchor footprints
from the wedging scenes (issue #132), using the actual production
``InstructionHead`` GOTO-standoff path end to end.
"""
from __future__ import annotations

import math

from core.heads.instruction import ARRIVAL_TOL_M, VIA_NEAR_CLEARANCE_M
from core.interfaces import OdomState, WaypointCmd
from core.mocks.synthetic_scene import Room, SyntheticScene
from core.perception.scene_index import BasicSceneIndex
from core.plan_schema import Anchor, LegKind, RouteLeg
from tests.heads._helpers import instruction_plan

# (label, sx, sy) -- the four real terminal-leg anchor footprints from #132.
ANCHOR_FOOTPRINTS = [
    ("arabic_jar", 0.18, 0.13),
    ("trash_can", 0.46, 0.46),
    ("potted_plant", 0.22, 0.22),
    ("mirror", 0.26, 0.23),
]


def _goto(noun: str) -> RouteLeg:
    return RouteLeg(kind=LegKind.GOTO, anchors=[Anchor(noun=noun)])


class _IO:
    def __init__(self, sc: SyntheticScene, start=(0.7, 0.7)):
        self._sc = sc
        self._x, self._y = start
        self._t = 0.0
        self.waypoints: list[WaypointCmd] = []

    def latest_terrain(self, extended: bool = False):
        return self._sc.terrain_patch(extended=extended, t=self._t)

    def latest_odom(self):
        return OdomState(t=self._t, x=self._x, y=self._y, z=0.0, yaw=0.0)

    def publish_waypoint(self, wp: WaypointCmd):
        self.waypoints.append(wp)

    def advance_time(self, dt: float):
        self._t += dt


def _achieved_vs_target(label: str, sx: float, sy: float):
    """Run the real GOTO-standoff path for a single anchor; return (achieved, raw)."""
    from core.heads.instruction import InstructionHead

    sc = SyntheticScene(0)
    sc.rooms = [Room(0.0, 0.0, 10.0, 10.0)]
    sc.place_box(label, 5.0, 5.0, sx, sy, 0.6)
    idx = BasicSceneIndex(sc.instances())
    head = InstructionHead(plan=instruction_plan([_goto(label)]))
    io = _IO(sc, start=(1.0, 1.0))
    for _ in range(6):
        head.advance(io, idx)
        io.advance_time(1.0)
        if head._follower is not None and head._follower.path:
            break

    leg = head._legs[0]
    assert leg.geom is not None, f"{label}: leg never grounded a goal"
    goal = leg.geom
    return head._clearance_m(goal), goal


# Before the #132 fix (``_near_thresh`` omitting the costmap's inflation radius),
# this exact assertion FAILED with:
#   arabic_jar=0.300m, trash_can=0.300m, potted_plant=0.200m, mirror=0.200m
# (all short of the 0.45 m VIA_NEAR_CLEARANCE_M target -- matching the issue's
# reproduction numbers exactly). Kept green post-fix as the regression guard.
def test_standoff_clearance_meets_target_and_stays_in_arrival_tolerance():
    """Post-fix: every real footprint's standoff goal reaches (or the anchor is large
    enough that ARRIVAL_TOL_M legitimately caps it short -- none of these four are)
    VIA_NEAR_CLEARANCE_M, and never leaves ARRIVAL_TOL_M of the centroid."""
    for label, sx, sy in ANCHOR_FOOTPRINTS:
        achieved, goal = _achieved_vs_target(label, sx, sy)
        d = math.hypot(goal[0] - 5.0, goal[1] - 5.0)
        assert achieved >= VIA_NEAR_CLEARANCE_M - 1e-6, (
            f"{label}: achieved clearance {achieved:.3f} m short of "
            f"{VIA_NEAR_CLEARANCE_M} m target"
        )
        assert d <= ARRIVAL_TOL_M + 1e-6, (
            f"{label}: standoff goal {d:.3f} m from centroid exceeds ARRIVAL_TOL_M "
            f"({ARRIVAL_TOL_M} m)"
        )


def _via_leg(noun: str) -> RouteLeg:
    return RouteLeg(kind=LegKind.VIA_NEAR, anchors=[Anchor(noun=noun)])


def test_via_near_shares_the_defect_and_is_fixed_by_the_same_near_thresh():
    """VIA_NEAR (``_via_point`` / ``free_space_via_point``) is placed using the same
    ``_near_thresh`` as the terminal-leg standoff, so it shares the #132 accounting
    gap and is fixed by the same change: each real anchor footprint's VIA_NEAR goal
    must reach VIA_NEAR_CLEARANCE_M."""
    from core.heads.instruction import InstructionHead

    for label, sx, sy in ANCHOR_FOOTPRINTS:
        sc = SyntheticScene(0)
        sc.rooms = [Room(0.0, 0.0, 10.0, 10.0)]
        sc.place_box(label, 5.0, 5.0, sx, sy, 0.6)
        idx = BasicSceneIndex(sc.instances())
        head = InstructionHead(plan=instruction_plan([_via_leg(label)]))
        io = _IO(sc, start=(1.0, 1.0))
        for _ in range(6):
            head.advance(io, idx)
            io.advance_time(1.0)
            if head._follower is not None and head._follower.path:
                break

        leg = head._legs[0]
        assert leg.geom is not None, f"{label}: VIA_NEAR leg never grounded a goal"
        goal = leg.geom
        clr = head._clearance_m(goal)
        assert clr >= VIA_NEAR_CLEARANCE_M - 1e-6, (
            f"{label}: VIA_NEAR goal clearance {clr:.3f} m short of "
            f"{VIA_NEAR_CLEARANCE_M} m target"
        )
