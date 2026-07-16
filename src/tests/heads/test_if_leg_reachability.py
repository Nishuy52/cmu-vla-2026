"""T11 IF-leg-threading regressions: leg goals must land on floor the drive can
actually reach, so ordered intermediate legs are threaded (not skipped to the terminal).

Root cause (office_2 sig-1): ``_goto_point`` projected the anchor centroid to the
nearest *passable* cell, which can be an isolated pocket disconnected from the drivable
free-space component. ``plan_through``'s A* to that pocket then fails, the leg is
dropped, and the vehicle beelines to the terminal — the intermediate leg scores 0. The
fix snaps the goal to the nearest cell REACHABLE from the pose.
"""
from __future__ import annotations

from core.heads.instruction import InstructionHead
from core.interfaces import OdomState, WaypointCmd
from core.mocks.synthetic_scene import Room, SyntheticScene
from core.perception.scene_index import BasicSceneIndex
from core.plan_schema import Anchor, LegKind, RouteLeg
from tests.heads._helpers import instruction_plan


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
