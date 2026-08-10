"""Issue #215 (11 Aug #202 chained replay): reverting resolve()'s target-level
superlative anchor pick to the pre-#186 ``anchor_recs[0]`` is not confined to the
leg that carries the superlative. Leg 0's own pick sets its GOTO goal, and
``InstructionHead._ground_legs`` threads that goal forward as ``prev_xy`` — the
route-continuity reference the NEXT leg's same-label tie-break sorts by (issue
#115). So a changed leg-0 superlative pick can silently re-target leg 1's own
anchor resolution, including which physical gate a CORRIDOR_BETWEEN leg forms.

The #202 replay found exactly this cascade on live archived routes (the
`9_hotel` TV-bed gate and the `1_hotel2` curtain-TV gate): with route-continuity
chaining, the pre-#186 pick is what the pre-#186 (old, higher-scoring) tree
actually threaded, and the #186 evidence-based pick broke it downstream even
though #186's own single-leg tests never exercised a second leg. This module
pins the cascade with a synthetic two-leg route (no archived fixture data is
reconstructible in this worktree) so the threading effect stays covered by a
fast unit test, not only by a cluster replay.
"""
from __future__ import annotations

import math

from core.geometry import toolbox as TB
from core.geometry.toolbox import DEFAULT_THRESHOLDS, TargetSpec
from core.heads.instruction import InstructionHead
from core.plan_schema import Anchor, LegKind, RouteLeg
from tests.heads._helpers import closest_clause, inst, instruction_plan, scene


def _leg0_target_spec() -> TargetSpec:
    return TargetSpec(noun="trash can", clauses=[closest_clause("bench")])


def _fixture():
    """Two benches and two trash cans that make leg 0's superlative anchor pick
    diverge between the pre-#186 index-order rule and #186's evidence-based rule,
    plus two clusters of "post" instances (for a leg-1 CORRIDOR_BETWEEN) sitting
    one next to each candidate leg-0 winner.

    ``bench_far`` (id 1) is inserted FIRST -- the pre-#186 ``anchor_recs[0]`` pick
    -- but sits far from both trash cans; ``bench_near`` (id 2) is inserted
    second but is the evidence-grounded pick (nearest to the trash-can pool,
    ``_select_sub_anchor``'s own key). Ranking the trash cans by distance to
    whichever bench wins picks a DIFFERENT trash can under each rule, so leg 0's
    grounded goal -- and the ``prev_xy`` threaded to leg 1 -- lands in a
    different place depending on the anchor-pick rule alone.
    """
    bench_far = inst(1, "bench", centroid=(50.0, 50.0, 0.0))
    bench_near = inst(2, "bench", centroid=(0.5, 0.5, 0.0))
    trash_near_far_bench = inst(11, "trash can", centroid=(50.6, 50.0, 0.0))
    trash_near_near_bench = inst(10, "trash can", centroid=(0.6, 0.5, 0.0))
    # Cluster near the pre-#186 (reverted) leg-0 winner (trash can 11, ~(50.6, 50.0)).
    post_a = inst(20, "post", centroid=(50.0, 50.0, 0.0))
    post_b = inst(21, "post", centroid=(51.0, 50.0, 0.0))
    # Cluster near the #186 evidence-based leg-0 winner (trash can 10, ~(0.6, 0.5)).
    post_c = inst(22, "post", centroid=(0.0, 0.0, 0.0))
    post_d = inst(23, "post", centroid=(1.0, 0.0, 0.0))
    return scene(
        bench_far, bench_near,
        trash_near_far_bench, trash_near_near_bench,
        post_a, post_b, post_c, post_d,
    )


def test_fixture_leg0_pick_actually_diverges_between_the_two_rules():
    """Sanity/contrast: confirms the fixture really does make the two anchor-pick
    rules disagree, before trusting the cascade assertion below."""
    sc = _fixture()
    spec = _leg0_target_spec()
    reverted = TB.resolve(spec, sc, DEFAULT_THRESHOLDS, superlative_anchor_evidence=False)
    evidence = TB.resolve(spec, sc, DEFAULT_THRESHOLDS, superlative_anchor_evidence=True)
    assert reverted.candidates_ranked[0].instance_id == 11, (
        "pre-#186 anchor_recs[0] (bench_far, inserted first) must pick the trash "
        "can nearest to bench_far"
    )
    assert evidence.candidates_ranked[0].instance_id == 10, (
        "#186's evidence-based pick (bench_near, nearest to the trash-can pool) "
        "must pick the trash can nearest to bench_near"
    )


def test_leg0_superlative_pick_cascades_into_leg1_corridor_gate_reference():
    """The regression this issue guards: with the #215 revert live in
    ``InstructionHead`` (route resolution always calls ``resolve`` with
    ``superlative_anchor_evidence=False``), leg 0's pre-#186 trash-can pick
    threads its goal forward as leg 1's ``prev_xy``, and leg 1's own
    CORRIDOR_BETWEEN anchor ("post", tied 4-way, no disambiguator of its own)
    resolves to the post CLUSTER nearest that goal -- not the cluster the #186
    evidence-based pick would have threaded instead.
    """
    sc = _fixture()
    route = [
        RouteLeg(
            kind=LegKind.GOTO,
            anchors=[Anchor(noun="trash can", disambiguator=closest_clause("bench"))],
        ),
        RouteLeg(
            kind=LegKind.CORRIDOR_BETWEEN,
            anchors=[Anchor(noun="post"), Anchor(noun="post")],
        ),
    ]
    head = InstructionHead(plan=instruction_plan(route))
    head._pose = (1000.0, 1000.0)  # far from both clusters; must not itself decide
    head._ground_legs(sc)

    leg0 = head._legs[0]
    assert leg0.record is not None, "leg 0 never grounded a candidate"
    assert leg0.record.instance_id == 11, (
        "leg 0's own superlative anchor pick must follow the #215-reverted "
        "(pre-#186 anchor_recs[0]) rule"
    )

    leg1 = head._legs[1]
    assert leg1.geom is not None, "leg 1 never grounded a corridor gate"
    assert leg1.record is not None
    assert leg1.record.instance_id in (20, 21), (
        "leg 1's corridor-gate reference must follow leg 0's REVERTED goal "
        "(threaded as prev_xy), i.e. resolve to the post cluster next to trash "
        "can 11 -- not the cluster next to trash can 10, which is where the "
        "#186 evidence-based leg-0 pick would instead have threaded prev_xy "
        "(issue #215, #202 replay: the cascade the 8 Aug independent-leg replay "
        "missed)"
    )
    p0, p1 = leg1.geom
    gate_mid = ((p0[0] + p1[0]) / 2.0, (p0[1] + p1[1]) / 2.0)
    dist_to_reverted_cluster = math.hypot(gate_mid[0] - 50.5, gate_mid[1] - 50.0)
    dist_to_evidence_cluster = math.hypot(gate_mid[0] - 0.5, gate_mid[1] - 0.0)
    assert dist_to_reverted_cluster < dist_to_evidence_cluster
