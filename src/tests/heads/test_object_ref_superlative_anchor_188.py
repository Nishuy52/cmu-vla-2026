"""Issue #188: ``ObjectRefHead._ladder_severity`` (see ``core/heads/object_ref.py``)
deliberately excludes anchor-side relaxation steps from its severity score, by design
(#184's own docstring: those steps concern an anchor's own resolution, not whether the
target's clause was enforced). That is correct for the ladder itself, but it also means
an established-pool SUPERLATIVE whose own anchor never grounds --
``superlative_anchor_missing`` (the anchor noun has no instances in the established
pool) or ``superlative_tie_unresolved`` (every survivor tied once the anchor failed to
ground) -- reads severity 0 and sails straight through ``_resolve``'s severity==0
short-circuit, so the raw pool is never even consulted. Even when the raw pool grounds
that SAME anchor cleanly and ranks a genuinely different (correct) winner.

Verifier fixture (batch-3 re-verification, 5 Aug 2026): an established anchor pool
that's empty for the superlative (every instance of the anchor's own noun sits below
``ESTABLISH_N_OBS`` in the established view) while the raw pool grounds it fine.
Reproduced here as a small, hand-built scene (not archived live data -- the verifier's
repro was itself synthetic) that isolates the exact mechanism:

* target noun "vase", clause ``closest_to(shelf)``.
* two established (n_obs >= ESTABLISH_N_OBS) vases: one genuinely closest to the real
  'shelf' position, one far away but carrying the LOWER instance id.
* one 'shelf' instance, itself only single-observation (n_obs=1) -- a plausible ghost
  detection of a real shelf, but not established.

Established pool: the shelf anchor pool is empty (its only instance sits below the
established floor) -- ``resolve()`` records ``superlative_anchor_missing`` and falls to
``_ungrounded_superlative_order``, which (bare noun, no other relation clause to give it
n_obs/clause-score evidence to break the tie on) degenerates to plain ascending
instance-id order -- the WRONG (farther) vase, by construction, always has the lower id
here to make the pre-fix defect deterministic and visible.

Raw pool: the same single 'shelf' instance IS present (raw is unfiltered by
establishment), so the anchor grounds trivially and ``closest_to`` correctly ranks the
genuinely-nearer vase first -- no anchor-relaxation step on raw's audit at all.

Pre-fix, ``_resolve`` returns the established pool's wrong (id-order) winner outright
(severity 0, no raw comparison ever attempted). Post-fix (the #188 compare-both-audits
rule), the raw pool's clean grounding wins.
"""
from __future__ import annotations

from core.heads.object_ref import ObjectRefHead
from tests.heads._helpers import closest_clause, inst, object_plan, scene


def test_established_anchor_missing_but_raw_grounds_cleanly_flips_to_raw():
    # (#188) The verifier's scenario: established anchor pool empty for the
    # superlative, raw pool grounds it -- the winner flips from the established
    # pool's wrong id-order pick to the raw pool's genuinely-closest vase.
    vase_far_lower_id = inst(1, "vase", n_obs=5, centroid=(10.0, 0.0, 0.0))
    vase_near_higher_id = inst(2, "vase", n_obs=9, centroid=(0.5, 0.0, 0.0))
    shelf_ghost = inst(3, "shelf", n_obs=1, centroid=(0.0, 0.0, 0.0))  # below ESTABLISH_N_OBS
    sc = scene(vase_far_lower_id, vase_near_higher_id, shelf_ghost)

    plan = object_plan("vase", clauses=[closest_clause("shelf")])
    head = ObjectRefHead(plan=plan)
    head.advance(sc)

    assert head.best_candidate is not None
    assert head.best_candidate.instance_id == 2  # the genuinely-closest vase, not id 1
    assert any(r.step == "established_gate_pool_choice" for r in head._result.audit)
    assert "raw pool selected" in head._result.audit[-1].detail
    assert "grounded the superlative anchor" in head._result.audit[-1].detail

    # sanity: the established-only pool genuinely could not ground the anchor at all
    # (confirms the fixture reproduces the reported mechanism, not a vacuous setup).
    from core.geometry.toolbox import DEFAULT_THRESHOLDS, resolve
    from core.heads.scene_established import ESTABLISH_N_OBS, EstablishedView

    established_only = EstablishedView(sc, floor=ESTABLISH_N_OBS)
    established_result = resolve(plan.target, established_only, DEFAULT_THRESHOLDS)
    steps = {r.step for r in established_result.audit}
    assert "superlative_anchor_missing" in steps
    assert established_result.candidates_ranked[0].instance_id == 1  # the wrong, id-order pick


def test_both_pools_fail_to_ground_the_anchor_keeps_established():
    # Control (#188 + #184's own tie rule, preserved): when the anchor noun is
    # wholly ABSENT from the scene, neither pool can ground the superlative -- raw
    # carries no new evidence over established, so the established pool's pick
    # still wins (deterministic id-order tiebreak), exactly as #184 intends when
    # the two pools are evidentially indistinguishable.
    vase_a = inst(1, "vase", n_obs=5, centroid=(0.0, 0.0, 0.0))
    vase_b = inst(2, "vase", n_obs=9, centroid=(10.0, 0.0, 0.0))
    sc = scene(vase_a, vase_b)  # no 'shelf' instance at all, in either pool

    plan = object_plan("vase", clauses=[closest_clause("shelf")])
    head = ObjectRefHead(plan=plan)
    head.advance(sc)

    assert head.best_candidate is not None
    assert head.best_candidate.instance_id == 1  # established pool's own (id-order) pick
    steps = [r.step for r in head._result.audit]
    assert "superlative_anchor_missing" in steps
    assert "established pool kept" in head._result.audit[-1].detail


def test_superlative_tie_unresolved_established_but_raw_grounds_cleanly_flips_to_raw():
    # (#188) The other anchor-relaxation step the ladder excludes: the established
    # anchor pool is non-empty but every established candidate is genuinely tied
    # (no clause-score/tier evidence to discriminate), so ``resolve()`` also records
    # ``superlative_tie_unresolved`` on top of ``superlative_anchor_missing`` for the
    # bare-noun case above. This test isolates the tie-unresolved step by keeping the
    # SAME reproduction shape (bare noun, no discriminating hard clause) -- both
    # relaxation steps land in ``_SUPERLATIVE_ANCHOR_STEPS`` and both must trigger the
    # compare-both-audits rule identically.
    from core.heads.object_ref import _has_superlative_anchor_relaxation
    from core.geometry.toolbox import DEFAULT_THRESHOLDS, resolve
    from core.heads.scene_established import ESTABLISH_N_OBS, EstablishedView

    vase_far_lower_id = inst(1, "vase", n_obs=5, centroid=(10.0, 0.0, 0.0))
    vase_near_higher_id = inst(2, "vase", n_obs=9, centroid=(0.5, 0.0, 0.0))
    shelf_ghost = inst(3, "shelf", n_obs=1, centroid=(0.0, 0.0, 0.0))
    sc = scene(vase_far_lower_id, vase_near_higher_id, shelf_ghost)
    plan = object_plan("vase", clauses=[closest_clause("shelf")])

    established_only = EstablishedView(sc, floor=ESTABLISH_N_OBS)
    established_result = resolve(plan.target, established_only, DEFAULT_THRESHOLDS)
    assert _has_superlative_anchor_relaxation(established_result.audit)
    assert "superlative_tie_unresolved" in {r.step for r in established_result.audit}

    head = ObjectRefHead(plan=plan)
    head.advance(sc)
    assert head.best_candidate.instance_id == 2  # same flip as the primary test above
