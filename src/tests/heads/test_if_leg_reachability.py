"""T11 IF-leg-threading regressions: leg goals must land on floor the drive can
actually reach, so ordered intermediate legs are threaded (not skipped to the terminal).

Root cause (office_2 sig-1): ``_goto_point`` projected the anchor centroid to the
nearest *passable* cell, which can be an isolated pocket disconnected from the drivable
free-space component. ``plan_through``'s A* to that pocket then fails, the leg is
dropped, and the vehicle beelines to the terminal — the intermediate leg scores 0. The
fix snaps the goal to the nearest cell REACHABLE from the pose.
"""
from __future__ import annotations

import math

from core.heads.instruction import InstructionHead
from core.interfaces import OdomState, WaypointCmd
from core.mocks.synthetic_scene import Room, SyntheticScene
from core.perception.scene_index import BasicSceneIndex
from core.plan_schema import Anchor, LegKind, RouteLeg
from tests.heads._helpers import inst, instruction_plan, scene


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

    def set_pose(self, x: float, y: float):
        self._x, self._y = x, y

    def advance_time(self, dt: float):
        self._t += dt


def _reachable(cm, grid, pose, xy) -> bool:
    reach = cm.nearest_reachable_point(xy, pose)
    return grid.world_to_cell(*reach) == grid.world_to_cell(*xy)


def test_goto_goal_snaps_to_reachable_cell_not_a_sealed_pocket():
    """An anchor whose centroid-nearest passable cell sits in a sealed pocket must not
    yield an unreachable leg goal: the head snaps it to a pose-reachable cell, so the
    committed route commits and the follower fires (route not dropped)."""
    sc = SyntheticScene(0)
    sc.rooms = [Room(0.0, 0.0, 10.0, 10.0)]
    # Wall off a small pocket in the far corner around (8.5, 8.5): three thick bars
    # leaving only a hairline sliver so the centroid's nearest passable cell is there
    # but disconnected from the open floor where the robot spawns.
    sc.place_box("wall", 8.5, 6.7, 3.0, 0.3, 1.0)   # bottom bar
    sc.place_box("wall", 6.7, 8.5, 0.3, 3.0, 1.0)   # left bar
    # the anchor sits inside the near-sealed pocket
    sc.place_box("cabinet", 8.5, 8.5, 0.4, 0.4, 0.8)
    idx = BasicSceneIndex(sc.instances())

    inst = InstructionHead(plan=instruction_plan([_goto("cabinet")]))
    io = _IO(sc, start=(1.0, 1.0))
    for _ in range(6):
        inst.advance(io, idx)
        io.advance_time(1.0)
        if inst._follower is not None and inst._follower.path:
            break

    assert inst._costmap is not None
    # The grounded leg's goal must be reachable from the spawn pose (the whole point).
    leg = inst._legs[0]
    assert leg.geom is not None
    goal = leg.geom[1] if isinstance(leg.geom[0], tuple) else leg.geom
    assert _reachable(inst._costmap, inst.grid, inst._pose, goal), (
        "leg goal must be in the same passable component as the pose"
    )
    # And a route committed (follower firmed up) rather than being dropped as unreachable.
    assert inst._follower is not None and inst._follower.path


def test_reachable_goal_lets_open_anchor_be_reached_directly():
    """Sanity: for a plainly open anchor the goal is the (reachable) centroid projection,
    and the follower plans a path whose end is at that goal."""
    sc = SyntheticScene(0)
    sc.rooms = [Room(0.0, 0.0, 10.0, 6.0)]
    sc.place_box("sofa", 8.0, 3.0, 0.5, 0.5, 0.5)
    idx = BasicSceneIndex(sc.instances())
    inst = InstructionHead(plan=instruction_plan([_goto("sofa")]))
    io = _IO(sc, start=(1.0, 3.0))
    inst.advance(io, idx)
    assert inst._follower is not None and inst._follower.path
    leg = inst._legs[0]
    goal = leg.geom
    assert _reachable(inst._costmap, inst.grid, inst._pose, goal)


# --------------------------------------------------------------------------- #126
# GOTO goal standoff: the naive nearest-reachable-cell projection snaps a GOTO goal
# right up against the anchor's own inflation boundary (near-zero clearance to the
# nearest blocked cell), which the stock local planner refuses to drive the final
# stretch into -- 15/15 live IF runs wedge 0.16-2.43 m short, stationary for the
# tail. The fix pulls the goal back to a free-space standoff (reusing the VIA_NEAR
# placement machinery) whenever the raw goal's own clearance is below the planner's
# floor, and leaves an already-clear raw goal untouched.


def test_goto_goal_stands_off_a_close_compact_anchor():
    """A compact anchor's naive projected goal lands with near-zero clearance (right
    at the inflation boundary); the head must pull it back to MEANINGFULLY more
    clearance (the standoff push is a bounded, best-effort local nudge -- see
    ``InstructionHead._standoff_push`` -- not a guarantee of exactly clearing
    VIA_NEAR_CLEARANCE_M in every corner geometry) while staying within
    ARRIVAL_TOL_M of the anchor centroid (the rubric's terminal arrival
    tolerance) — the sole leg of a single-leg route is always the terminal one."""
    from core.heads.instruction import ARRIVAL_TOL_M, VIA_NEAR_CLEARANCE_M

    sc = SyntheticScene(0)
    sc.rooms = [Room(0.0, 0.0, 10.0, 10.0)]
    sc.place_box("table", 5.0, 5.0, 0.4, 0.4, 0.6)
    idx = BasicSceneIndex(sc.instances())
    head = InstructionHead(plan=instruction_plan([_goto("table")]))
    io = _IO(sc, start=(1.0, 1.0))
    for _ in range(6):
        head.advance(io, idx)
        io.advance_time(1.0)
        if head._follower is not None and head._follower.path:
            break

    leg = head._legs[0]
    assert leg.geom is not None
    goal = leg.geom
    rec = leg.record
    raw = head._goto_point_raw(rec)
    raw_clr = head._clearance_m(raw)

    # The raw (pre-fix) projection is close enough to reproduce the wedge: near-zero
    # clearance, well under the planner's floor.
    assert raw_clr < VIA_NEAR_CLEARANCE_M

    # The actual (post-fix) goal clears meaningfully more than the raw goal did...
    assert head._clearance_m(goal) > raw_clr + 0.05
    assert goal != raw
    # ...without overshooting the rubric's terminal arrival tolerance.
    d = math.hypot(goal[0] - 5.0, goal[1] - 5.0)
    assert d <= ARRIVAL_TOL_M


def test_goto_goal_unchanged_when_raw_goal_already_clear():
    """An anchor with no physical obstacle in the occupancy grid (e.g. grounded from
    perception alone, nothing yet integrated into the terrain) already has maximal
    clearance -- the standoff machinery must be a no-op, not move a goal that was
    already fine."""
    sc = SyntheticScene(0)
    sc.rooms = [Room(0.0, 0.0, 10.0, 10.0)]  # no place_box: nothing blocks the grid
    idx = scene(inst(1, "sofa", n_obs=3, centroid=(5.0, 5.0, 0.0), extent=(0.4, 0.4, 0.6)))
    head = InstructionHead(plan=instruction_plan([_goto("sofa")]))
    io = _IO(sc, start=(1.0, 1.0))
    for _ in range(6):
        head.advance(io, idx)
        io.advance_time(1.0)
        if head._follower is not None and head._follower.path:
            break

    leg = head._legs[0]
    assert leg.geom is not None
    goal = leg.geom
    rec = leg.record
    raw = head._goto_point_raw(rec)
    assert goal == raw, "goal moved even though the raw projection was already clear"


def test_goto_standoff_only_applies_to_the_terminal_leg():
    """The standoff is scoped to the TERMINAL leg only (issue #126): a non-terminal
    GOTO leg is a breadcrumb the follower threads through en route to the next leg,
    never a point the vehicle stops and holds at, so it was never exposed to the
    planner obstacle-clearance refusal -- standing it off too only adds scoring
    drift no live run exhibits. A close/compact anchor on a non-terminal leg must
    therefore keep its raw (un-stood-off) goal; the terminal leg's own close/compact
    anchor still gets the standoff."""
    sc = SyntheticScene(0)
    sc.rooms = [Room(0.0, 0.0, 12.0, 10.0)]
    sc.place_box("table", 5.0, 5.0, 0.4, 0.4, 0.6)  # non-terminal leg's close anchor
    sc.place_box("lamp", 9.0, 5.0, 0.4, 0.4, 0.6)  # terminal leg's close anchor
    idx = BasicSceneIndex(sc.instances())
    head = InstructionHead(plan=instruction_plan([_goto("table"), _goto("lamp")]))
    io = _IO(sc, start=(1.0, 1.0))
    for _ in range(8):
        head.advance(io, idx)
        io.advance_time(1.0)
        if head._follower is not None and head._follower.path:
            break

    leg0, leg1 = head._legs[0], head._legs[1]
    assert leg0.geom is not None and leg1.geom is not None
    raw0 = head._goto_point_raw(leg0.record)
    raw1 = head._goto_point_raw(leg1.record)

    # Non-terminal leg 0: unchanged, even though its raw goal is just as close to
    # its anchor as the terminal leg's.
    assert leg0.geom == raw0
    # Terminal leg 1: stood off (clears meaningfully more than its own raw goal).
    assert head._clearance_m(leg1.geom) > head._clearance_m(raw1) + 0.05


def test_goto_standoff_stays_within_arrival_tolerance_for_a_large_anchor():
    """issue #126 verification fix: a LARGE terminal anchor (a 2.5 m x 1.8 m bed,
    half-diagonal ~1.54 m) pushed a full VIA_NEAR-style near_thresh out (half_diag +
    VIA_NEAR_CLEARANCE_M ~= 1.99 m) lands OUTSIDE ARRIVAL_TOL_M (1.7463 m) -- trading
    a short-of-goal wedge for an unrecoverable tolerance miss, which is not a fix.
    The standoff must clamp the push so the final goal stays within ARRIVAL_TOL_M of
    the anchor centroid, with margin (not landing exactly on the boundary)."""
    from core.heads.instruction import ARRIVAL_TOL_M, STANDOFF_ARRIVAL_MARGIN_M

    sc = SyntheticScene(0)
    sc.rooms = [Room(0.0, 0.0, 14.0, 14.0)]
    sc.place_box("bed", 7.0, 7.0, 2.5, 1.8, 0.6)
    idx = BasicSceneIndex(sc.instances())
    head = InstructionHead(plan=instruction_plan([_goto("bed")]))
    io = _IO(sc, start=(1.0, 1.0))
    for _ in range(8):
        head.advance(io, idx)
        io.advance_time(1.0)
        if head._follower is not None and head._follower.path:
            break

    leg = head._legs[0]
    assert leg.geom is not None
    goal = leg.geom
    d = math.hypot(goal[0] - 7.0, goal[1] - 7.0)
    assert d <= ARRIVAL_TOL_M - STANDOFF_ARRIVAL_MARGIN_M + 1e-6, (
        f"standoff goal at {d:.3f} m from centroid is outside the (margined) "
        f"arrival tolerance"
    )
