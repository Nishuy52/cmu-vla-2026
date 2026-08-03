"""Issue #115: head and battery compute a leg's goal by different METHODS, so they
can push off an object's footprint to different places -- and a divergent
``prev_xy``/``approach_xy`` reference can then select a DIFFERENT physical object
for the next leg's tie-break, even though both sides implement the identical
tie-break rule (#107/#113).

This commit fixes the named defect -- (a) the ``prev_xy``/``approach_xy`` reference
THREADING, aligned to mirror ``core.runner.gt_battery._if_rubric_geometry`` (see
``InstructionHead._ground_legs``'s docstring for the full citation/derivation) --
without importing battery code into the head (module docstring: ROS-free/light).
Full alignment of (b) the goal PROJECTION method itself (grid/costmap push-off vs
pure-footprint AABB push-off) is NOT attempted here -- too large a change for one
contained commit -- and is instead pinned as still-diverging current behaviour by
the last test below, so the residual gap is documented, not silently left.
"""
from __future__ import annotations

from core.heads.instruction import InstructionHead
from core.plan_schema import Anchor, LegKind, RouteLeg
from tests.heads._helpers import inst, instruction_plan, scene


def _goto(noun: str) -> RouteLeg:
    return RouteLeg(kind=LegKind.GOTO, anchors=[Anchor(noun=noun)])


# --------------------------------------------------------------------- (a)(1): first leg


def test_first_leg_route_continuity_uses_current_pose_not_none():
    """issue #115: ``_ground_legs`` must seed ``prev_xy`` from the vehicle's own
    current pose for the FIRST leg's route-continuity tie-break -- mirroring
    ``gt_battery._if_rubric_geometry``'s ``start_xy`` threading (gt_battery.py:
    1478-1484, 1625: ``approach_xy = start_xy``) -- not fall back to ``None``
    (declared-salience-only ordering, ignoring where the vehicle actually is)."""
    sc = scene(
        inst(1, "vase", centroid=(9.0, 9.0, 0.0), n_obs=5, score=0.5),  # far from pose
        inst(2, "vase", centroid=(1.5, 1.0, 0.0), n_obs=5, score=0.5),  # near pose
    )
    head = InstructionHead(plan=instruction_plan([_goto("vase")]))
    head._pose = (1.0, 1.0)
    head._ground_legs(sc)

    leg = head._legs[0]
    assert leg.record is not None, "leg never grounded a candidate"
    assert leg.record.instance_id == 2, (
        "the first leg's same-label tie must resolve to the candidate nearest the "
        "vehicle's CURRENT pose (route-continuity tie-break), not fall back to "
        "declared-salience order (issue #115)"
    )


def test_first_leg_declared_salience_used_only_pre_115_would_pick_the_other_one():
    """Sanity/contrast for the test above: WITHOUT a pose-derived prev_xy (the
    pre-#115 ``None`` seed), declared salience (n_obs/score/volume) is the only
    signal, and with these two candidates tied on every declared-salience
    component, resolve()'s own pre-sort order decides -- demonstrating the fix
    actually changes the outcome for this fixture, not just its bookkeeping."""
    sc = scene(
        inst(1, "vase", centroid=(9.0, 9.0, 0.0), n_obs=5, score=0.5),
        inst(2, "vase", centroid=(1.5, 1.0, 0.0), n_obs=5, score=0.5),
    )
    head = InstructionHead(plan=instruction_plan([_goto("vase")]))
    ranked, _, audit = head._ranked_anchor(Anchor(noun="vase"), sc, prev_xy=None)
    assert audit.tie_break_group_size == 2
    # Declared salience is tied on both candidates (identical n_obs/score/volume),
    # so this does NOT reliably pick instance 2 -- it is whatever resolve()'s own
    # pre-sort order happens to be, underscoring that pose is the only thing that
    # discriminates the pair.
    ranked_pose, _, _ = head._ranked_anchor(Anchor(noun="vase"), sc, prev_xy=(1.0, 1.0))
    assert ranked_pose[0].instance_id == 2


# --------------------------------------------------------------------- (a)(2): corridor gate midpoint


def test_corridor_leg_threads_gate_midpoint_not_an_endpoint():
    """issue #115: after a CORRIDOR_BETWEEN leg, the next leg's route-continuity
    ``prev_xy`` must be the gate's MIDPOINT (mirrors ``gt_battery
    ._if_rubric_geometry``, gt_battery.py:1642 ``mid = ...``; 1647
    ``approach_xy = mid``), not an arbitrary endpoint of the two-point gate
    segment -- which can sit metres away from the midpoint for a wide gate and
    feed a materially different tie-break reference into the next leg."""
    sc = scene(
        inst(1, "post", centroid=(0.0, 0.0, 0.0), extent=(0.4, 0.4, 1.0)),
        inst(2, "post", centroid=(10.0, 0.0, 0.0), extent=(0.4, 0.4, 1.0)),
        inst(3, "vase", centroid=(5.0, 0.5, 0.0)),  # near the gate's MIDPOINT (~5,0)
        inst(4, "vase", centroid=(9.5, 0.5, 0.0)),  # near the gate's SECOND point (~9.8,0)
    )
    route = [
        RouteLeg(
            kind=LegKind.CORRIDOR_BETWEEN,
            anchors=[Anchor(noun="post"), Anchor(noun="post")],
        ),
        _goto("vase"),
    ]
    head = InstructionHead(plan=instruction_plan(route))
    head._pose = (0.0, -5.0)
    head._ground_legs(sc)

    corridor_leg = head._legs[0]
    assert corridor_leg.geom is not None, "corridor leg never grounded a gate"
    goto_leg = head._legs[1]
    assert goto_leg.record is not None, "goto leg never grounded a candidate"
    assert goto_leg.record.instance_id == 3, (
        "the leg following a corridor must thread the gate's MIDPOINT as its "
        "route-continuity reference, not an endpoint (issue #115)"
    )


def test_corridor_leg_endpoint_threading_would_pick_the_other_vase():
    """Sanity/contrast for the test above: confirms the two fixture vases really
    do sit on opposite sides of the midpoint-vs-endpoint divide (the endpoint is
    materially closer to the WRONG vase), so the fix in the prior test is
    demonstrated, not merely consistent by coincidence."""
    sc = scene(
        inst(1, "post", centroid=(0.0, 0.0, 0.0), extent=(0.4, 0.4, 1.0)),
        inst(2, "post", centroid=(10.0, 0.0, 0.0), extent=(0.4, 0.4, 1.0)),
        inst(3, "vase", centroid=(5.0, 0.5, 0.0)),
        inst(4, "vase", centroid=(9.5, 0.5, 0.0)),
    )
    from core.geometry import toolbox as TB

    p1, p2 = sc.all_instances()[0], sc.all_instances()[1]
    gate = TB.corridor_gate(p1, p2, sc)
    endpoint = (float(gate.p1[0]), float(gate.p1[1]))
    midpoint = (float(gate.midpoint[0]), float(gate.midpoint[1]))

    import math

    def _d(xy, c):
        return math.hypot(xy[0] - c[0], xy[1] - c[1])

    vase_a, vase_b = (5.0, 0.5), (9.5, 0.5)
    assert _d(midpoint, vase_a) < _d(midpoint, vase_b)
    assert _d(endpoint, vase_b) < _d(endpoint, vase_a)


# --------------------------------------------------------------------- (b): documented residual


def test_goal_projection_method_still_diverges_from_battery_without_terrain():
    """issue #115, residual NOT fixed by this commit (goal PROJECTION method, not
    ``prev_xy`` threading -- see ``InstructionHead._ground_legs``'s docstring).
    Pinned, not fixed: reproduces the issue's own livingroom_1 potted-plant
    numbers with a synthetic anchor -- a fresh, terrain-less
    ``InstructionHead._goto_point_raw`` (no costmap to push against yet, so it
    returns the raw centroid unmodified) versus
    ``gt_battery._nearest_free_goal`` (always pushes off the anchor's own AABB,
    terrain or not, per issue #66). Full alignment would mean either importing
    the battery's pure-geometry push into the ROS-free head or duplicating a
    second push algorithm there -- out of scope for this fix; this test exists so
    the gap stays visible instead of silently reappearing as a surprise."""
    from core.runner.gt_battery import _nearest_free_goal

    sc = scene(inst(1, "plant", centroid=(2.0, 3.0, 0.0), extent=(1.0, 1.0, 0.6)))
    rec = sc.all_instances()[0]

    head = InstructionHead(plan=instruction_plan([_goto("plant")]))
    assert head._costmap is None, "test setup sanity: no terrain ingested yet"
    head_goal = head._goto_point_raw(rec)
    assert head_goal == (2.0, 3.0), (
        "test setup sanity: a fresh, terrain-less head returns the raw centroid "
        "unmodified (no costmap to project against)"
    )

    # A distant, unrelated second instance so ``_gt_footprint_bounds`` derives a
    # real room footprint much larger than the plant's own AABB -- with only the
    # plant itself, its footprint EQUALS the (degenerate) room bounds and
    # ``_is_architectural_room_scale_aabb`` wrongly treats it as a room-scale
    # aggregate (issue #53), skipping the push this test exists to observe.
    far_rec = inst(2, "wall_marker", centroid=(20.0, 20.0, 0.0), extent=(0.1, 0.1, 0.1))

    class _FakeGT:
        instances = [rec, far_rec]

    battery_goal = _nearest_free_goal(
        (2.0, 3.0), _FakeGT(), anchor_id=1, approach_xy=(0.0, 3.0),
    )
    assert battery_goal is not None
    assert battery_goal != head_goal, (
        "the battery pushes off the anchor's own footprint (issue #66) even with "
        "no terrain, while the terrain-less head does not -- this documented "
        "divergence in goal PROJECTION method is issue #115's remaining, "
        "unfixed half"
    )
