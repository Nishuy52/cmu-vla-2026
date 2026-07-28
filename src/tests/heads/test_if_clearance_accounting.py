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


def test_via_near_on_real_footprints_already_clears_target_via_ring_slack():
    """VIA_NEAR (``_via_point`` / ``free_space_via_point``) is placed using the same
    ``_near_thresh`` as the terminal-leg standoff, so it shares the #132 EXPRESSION
    (the same missing ``vehicle_radius_m`` term). But this does NOT reproduce as an
    observable shortfall for these four real, compact footprints in the open:
    ``free_space_via_point`` searches a ring spanning +/-0.3 m around ``near_thresh``
    and keeps whichever passable cell in that band has the MOST clearance -- for a
    compact anchor in open space that band already reaches comfortably past
    VIA_NEAR_CLEARANCE_M regardless of the pre-#132 near_thresh error, so this test is
    a non-regression check (still true post-fix, with more margin), not a
    reproduction. The genuine VIA_NEAR reproduction (a confined geometry where the
    ring's slack canNOT absorb the gap) is
    ``test_via_near_confined_geometry_reproduces_and_is_fixed`` below."""
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


def _via_confined_goal(*, use_pre_132_near_thresh: bool):
    """A small anchor ("plant", 0.3x0.3) sitting just inside a dead-end alcove: three
    walls confine it, with the only opening 1.0 m out from the centroid along +x into
    the open room. The ring search's clearance-maximizing pick is still bounded BY THE
    ALCOVE for the pre-#132 near_thresh (too short to reach the opening), but the
    #132-corrected near_thresh reaches past the opening into the open room -- so
    (unlike ``ANCHOR_FOOTPRINTS`` in the open) the ring's +/-0.3 m slack canNOT paper
    over the missing ``vehicle_radius_m`` term here. Returns (clearance_m, goal)."""
    from core.heads.instruction import InstructionHead

    orig = InstructionHead._near_thresh

    def _pre_132_near_thresh(self, rec):
        ext = rec.aabb_max - rec.aabb_min
        half_diag = 0.5 * float((float(ext[0]) ** 2 + float(ext[1]) ** 2) ** 0.5)
        return half_diag + VIA_NEAR_CLEARANCE_M

    if use_pre_132_near_thresh:
        InstructionHead._near_thresh = _pre_132_near_thresh
    try:
        sc = SyntheticScene(0)
        sc.rooms = [Room(0.0, 0.0, 20.0, 20.0)]
        sc.place_box("plant", 5.0, 5.0, 0.3, 0.3, 0.6)
        wall_t = 0.2
        corridor_half_w, alcove_len = 0.6, 1.0
        x0, x1 = 4.6, 5.0 + alcove_len
        sc.place_box("wall_n", (x0 + x1) / 2.0, 5.0 + corridor_half_w + wall_t / 2.0,
                     x1 - x0, wall_t, 1.0)
        sc.place_box("wall_s", (x0 + x1) / 2.0, 5.0 - corridor_half_w - wall_t / 2.0,
                     x1 - x0, wall_t, 1.0)
        sc.place_box("wall_w", x0 - wall_t / 2.0, 5.0,
                     wall_t, 2 * (corridor_half_w + wall_t), 1.0)
        idx = BasicSceneIndex(sc.instances())
        head = InstructionHead(plan=instruction_plan([_via_leg("plant")]))
        io = _IO(sc, start=(1.0, 1.0))
        for _ in range(10):
            head.advance(io, idx)
            io.advance_time(1.0)
            if head._follower is not None and head._follower.path:
                break
        leg = head._legs[0]
        assert leg.geom is not None, "confined VIA_NEAR leg never grounded a goal"
        goal = leg.geom
        return head._clearance_m(goal), goal
    finally:
        InstructionHead._near_thresh = orig


def test_via_near_confined_geometry_reproduces_and_is_fixed():
    """The genuine VIA_NEAR reproduction of #132: in the confined alcove geometry, the
    pre-#132 ``_near_thresh`` (missing ``vehicle_radius_m``) leaves the ring search
    unable to reach past the alcove's opening, landing well short of
    VIA_NEAR_CLEARANCE_M; the #132-corrected ``_near_thresh`` (used by the current,
    unpatched code) reaches into the open room beyond the opening and meets it."""
    pre_clr, _ = _via_confined_goal(use_pre_132_near_thresh=True)
    assert pre_clr < VIA_NEAR_CLEARANCE_M - 1e-6, (
        f"test setup sanity: expected the pre-#132 near_thresh to fall short of the "
        f"{VIA_NEAR_CLEARANCE_M} m target in this confined geometry, got {pre_clr:.3f} m"
    )

    post_clr, _ = _via_confined_goal(use_pre_132_near_thresh=False)
    assert post_clr >= VIA_NEAR_CLEARANCE_M - 1e-6, (
        f"VIA_NEAR clearance {post_clr:.3f} m in the confined geometry still short of "
        f"the {VIA_NEAR_CLEARANCE_M} m target after the #132 fix"
    )
