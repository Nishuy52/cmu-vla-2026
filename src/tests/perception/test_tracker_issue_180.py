"""Issue #180: the #176 extent-growth branch has no cumulative cap.

``_match_plausible``'s two waivers (#153's near-zero-distance floor, #176's
per-step growth tolerance) each judge only ONE match. Neither caps how far a
CHAIN of individually-plausible matches can walk a track's box:

* The #153 floor (``dist <= extent_veto_min_sep``) waives every other check
  outright -- however large the resulting box grows -- whenever the new
  detection's own centroid lands close to the candidate's. A class with no
  dimension prior has no absolute ceiling to fall back on either, so nothing
  in the pre-#180 code stops a chain of such matches from walking the box
  outward without limit.
* The #176 growth branch (``growth <= extent_growth_tol``) only bounds a
  single match's own contribution; repeated sub-tolerance merges each look
  individually safe while their sum is unbounded.

The fix (``TrackerConfig.extent_cumulative_growth_cap``): :func:`associate`
now remembers each track's FIRST accepted box (kept on the index object, the
same way ``BasicSceneIndex`` already keeps its colour tally outside
``InstanceRecord``, issue #121) and ``_match_plausible`` vetoes any match that
would grow the box more than the cap past that first box, on any axis --
checked FIRST, ahead of and regardless of the #153/#176 waivers.

This module reproduces the adversarial chain directly against
``_match_plausible`` (no bag reads, no live replay -- synthetic corner-point
detections built by hand, mirroring test_tracker_issue_176.py's
``_corner_cloud`` technique so the real scene index's percentile AABB
trimming is a no-op and every intermediate box is exactly what the test
computes), then confirms the same wiring holds end-to-end through
``associate()``/``BasicSceneIndex``.
"""
from __future__ import annotations

import numpy as np

from core.interfaces import InstanceRecord
from core.perception.detector import Detection
from core.perception.dimension_priors import prior_for
from core.perception.fusion import Fused3D
from core.perception.scene_index import BasicSceneIndex
from core.perception.tracker import TrackerConfig, _match_plausible, associate

# A label deliberately absent from the dimension-prior table: no absolute
# ceiling exists to catch unbounded growth by itself, isolating the
# cumulative-cap mechanism this module tests (the #180 construction rides
# exactly this gap -- "a class with no dimension prior falls back to
# cfg.gate unchanged", never vetoed on size, pre-#180).
_LABEL = "gizmo_no_prior"

_STEP = 0.15  # m per accepted merge; well under extent_veto_min_sep (0.2)


def _det(label=_LABEL, score=0.4):
    return Detection(tile_id=0, bbox_xyxy=(0.0, 0.0, 10.0, 10.0), label=label, score=score)


def _fused_from_points(pts):
    pts = np.asarray(pts, dtype=np.float32)
    centroid = pts.mean(axis=0)
    return Fused3D(centroid=centroid, points=pts, n_points=len(pts), range_m=float(np.linalg.norm(centroid)))


def test_precondition_label_has_no_dimension_prior():
    """Pins the test's own precondition: no class-prior ceiling exists to save
    this construction by itself -- the #180 cumulative cap must do it alone."""
    assert prior_for(_LABEL) is None


# --------------------------------------------------------------- unit-level


def _run_chain(cfg, first_box, max_steps=200):
    """Chain ``_match_plausible`` calls directly: each step extends the box's
    +x edge by ``_STEP`` while keeping the new detection's OWN centroid within
    ``extent_veto_min_sep`` of the running candidate's -- exactly the #153
    floor's exploit condition, re-satisfied every single step. Returns the
    number of accepted merges and the final candidate extent.
    """
    aabb_min = np.array([0.0, -0.2, 0.4])
    aabb_max = np.array([0.3, 0.2, 0.7])
    candidate = InstanceRecord(
        instance_id=0,
        label=_LABEL,
        score=0.4,
        n_obs=1,
        centroid=(aabb_min + aabb_max) / 2.0,
        aabb_min=aabb_min.copy(),
        aabb_max=aabb_max.copy(),
    )
    accepted = 0
    for _ in range(max_steps):
        new_min = candidate.aabb_min.copy()
        new_max = candidate.aabb_max.copy()
        new_max[0] += _STEP
        fused = _fused_from_points(np.vstack([new_min, new_max]))
        dist = float(np.linalg.norm(fused.centroid - candidate.centroid))
        assert dist <= cfg.extent_veto_min_sep, (
            "test construction error: each step must keep riding the #153 floor"
        )
        if not _match_plausible(candidate, fused, cfg, dist, first_box=first_box):
            break
        # Accept: fold in exactly as BasicSceneIndex._fuse would union/trim a
        # 2-point cloud (a no-op trim, per test_tracker_issue_176.py's own
        # _corner_cloud rationale) -- no class prior means #104's
        # cap_fused_extent is also a no-op here, isolating this module's fix.
        candidate.aabb_min = np.minimum(candidate.aabb_min, new_min)
        candidate.aabb_max = np.maximum(candidate.aabb_max, new_max)
        candidate.centroid = (candidate.aabb_min + candidate.aabb_max) / 2.0
        accepted += 1
    return accepted, candidate.aabb_min, candidate.aabb_max


def test_PRE_FIX_no_first_box_walks_unbounded():
    """Pins the pre-#180 defect: with no ``first_box`` supplied (the
    signature's default, matching every pre-#180 caller), the #153 floor
    waives every one of 200 chained sub-tolerance merges -- nothing stops the
    box from walking indefinitely."""
    cfg = TrackerConfig()
    accepted, _, aabb_max = _run_chain(cfg, first_box=None, max_steps=200)
    assert accepted == 200
    assert aabb_max[0] >= 0.3 + 200 * _STEP - 1e-6


def test_cumulative_cap_stops_the_walk():
    """The #180 fix: supplying ``first_box`` (as :func:`associate` now does)
    ratchets the SAME chain to a stop once cumulative growth from that first
    box would exceed ``cfg.extent_cumulative_growth_cap``, however small each
    individual step's own contribution looked."""
    cfg = TrackerConfig()
    first_min = np.array([0.0, -0.2, 0.4])
    first_max = np.array([0.3, 0.2, 0.7])
    accepted, _, aabb_max = _run_chain(
        cfg, first_box=(first_min.copy(), first_max.copy()), max_steps=200
    )
    growth_x = float(aabb_max[0] - first_max[0])
    assert accepted < 200, "the walk never stopped -- #180 regression"
    # Ratcheted right up to (not stopped well short of) the cap: the last
    # accepted step must have landed within one step of it, and no accepted
    # step's resulting growth may ever have exceeded it.
    assert growth_x <= cfg.extent_cumulative_growth_cap + 1e-9
    assert growth_x > cfg.extent_cumulative_growth_cap - _STEP


# ---------------------------------------------------------- associate()-level


def _local_cloud(centre, seed, half=(0.03, 0.03, 0.03), n=20):
    """A small, tightly-bounded cluster around ``centre`` -- a plausible
    single-frame re-observation reaching just past the track's current edge
    (unlike the unit-level helper's bare 2-point corners, this stays LOW-IoU
    against the accumulated track once it has grown large, so a tracker veto
    is not silently overridden by ``BasicSceneIndex.add``'s own independent
    IoU-based dedup -- see ``merge_into``'s docstring for why that dedup
    exists and why it must not be the thing this test's rejection depends
    on)."""
    rng = np.random.default_rng(seed)
    lo = np.asarray(centre) - np.asarray(half)
    hi = np.asarray(centre) + np.asarray(half)
    return rng.uniform(lo, hi, size=(n, 3)).astype(np.float32)


def test_associate_wires_first_box_and_stops_the_walk_end_to_end():
    """Integration check: the SAME adversarial chain, driven through the real
    ``associate()``/``BasicSceneIndex`` path (no direct ``_match_plausible``
    calls, small local per-step clusters rather than the unit-level helper's
    bare corners), also stops -- proving :func:`associate` actually threads
    ``first_box`` through, not just the unit-level helper above."""
    idx = BasicSceneIndex()
    cfg = TrackerConfig()

    aabb_min = np.array([0.0, -0.2, 0.4])
    aabb_max = np.array([0.3, 0.2, 0.7])
    associate(
        [(_det(), _fused_from_points(np.vstack([aabb_min, aabb_max])))], idx, cfg=cfg
    )
    assert len(idx.all_instances()) == 1

    n_steps = 30  # comfortably past cap/_STEP (~10) if the walk were unbounded
    for i in range(n_steps):
        insts = [r for r in idx.all_instances() if r.label == _LABEL]
        if len(insts) != 1:
            break
        cand = insts[0]
        reach = cand.aabb_max.copy()
        reach[0] += _STEP
        associate([(_det(), _fused_from_points(_local_cloud(reach, seed=i)))], idx, cfg=cfg)

    final = [r for r in idx.all_instances() if r.label == _LABEL]
    # The walk must have minted a second (ghost) instance well before
    # exhausting n_steps -- i.e. the cumulative cap actually fired through the
    # real associate()/BasicSceneIndex path, not just the unit-level helper.
    assert len(final) >= 2, (
        f"expected the cumulative cap to split off a fresh instance within "
        f"{n_steps} steps, got {len(final)} '{_LABEL}' instance(s) -- #180 regression"
    )
