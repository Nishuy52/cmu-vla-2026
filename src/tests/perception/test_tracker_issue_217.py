"""Issue #217: the #94 extent veto rejects thin-class re-detections -- the
real #216 flicker mechanism.

The #216 event-level replay (test_tracker_issue_216.py) found the #94
absolute-ceiling check (``extent_veto_factor x typ_ext``, ONE flat ratio for
every class on every axis) is what actually starves the #216 flicker
instances, 9 of 10 rejections -- not the #180 growth-cap ordering #216 first
suspected. This module implements and proves the fix:
:func:`core.perception.tracker._extent_veto_bound` widens that ceiling per
axis (never tightens it) by the larger of

  1. an absolute allowance (``TrackerConfig.extent_veto_abs_slack_m``), and
  2. the class's own recorded per-axis ``ClassPrior.cap_factor`` (issue #201's
     P95/median triple, unused metadata until now) where a rank carries real
     evidence -- i.e. sits strictly above
     :data:`~core.perception.dimension_priors.CAP_FACTOR_FLOOR` (a rank AT the
     floor is #201's "not enough/not unusual variance" default, not a
     measurement, and does not widen anything).

``extent_veto_abs_slack_m``'s value (0.45 m) is not the 0.15 m first
proposed: that only admitted 2 of the #216 replay's 9 flagged rejections
(measured directly below, ``test_before_after_table_matches_the_acceptance_
criteria``). 0.45 m was found by binary search against the FULL perception
suite (every existing #94/#153/#161/#176 guard, run at each candidate value):
it admits 8 of the 9; the 9th (a "door" pair whose reconstructed box implies a
0.71 m door thickness -- itself suspect, see
``test_the_one_residual_rejection_needs_an_implausible_box`` below) would need
0.547 m, which flips ``test_distinct_archived_chairs_stay_separate_post_fix``
(7 real chairs collapse to 6) at any slack past ~0.48 m. 0.45 m is the chosen
value: comfortably under that boundary (issue #161's two-TVs decisive-band
guard, the next-tightest, keeps a 0.109 m margin at 0.45 m) while admitting
every thin-class pair the #94/#153/#161/#176 guards do not structurally forbid.
"""
from __future__ import annotations

import numpy as np
import pytest

from core.perception.dimension_priors import CAP_FACTOR_FLOOR, prior_for
from core.perception.scene_index import BasicSceneIndex
from core.perception.tracker import (
    DEFAULT_TRACKER_CONFIG,
    TrackerConfig,
    _extent_veto_bound,
    associate,
    canonical_for_match,
    decay_singletons,
)

from tests.perception.test_tracker_issue_216 import (
    _all_pairs,
    _FakeFused,
    _record_from_geometry,
    _replay_pair,
)

# --------------------------------------------------------------------------- widening 2: floor gating


def test_cap_factor_at_the_floor_does_not_widen():
    """A rank sitting exactly at CAP_FACTOR_FLOOR carries no real evidence
    (issue #201's own "not enough data / not unusually variable" default) --
    the bound on that rank must equal the un-widened #94 default, not 1.5x."""
    prior = prior_for("television")  # long/mid cap_factor == 1.500 == the floor
    assert prior.cap_factor[1] == CAP_FACTOR_FLOOR
    assert prior.cap_factor[2] == CAP_FACTOR_FLOOR
    bound = _extent_veto_bound(prior, DEFAULT_TRACKER_CONFIG)
    typ = prior.typ_ext
    # mid/long: no cap_factor signal -> bound is max(1.3x typ, typ + abs_slack)
    for axis in (1, 2):
        expected = max(
            DEFAULT_TRACKER_CONFIG.extent_veto_factor * typ[axis],
            typ[axis] + DEFAULT_TRACKER_CONFIG.extent_veto_abs_slack_m,
        )
        assert bound[axis] == pytest.approx(expected)


def test_cap_factor_above_the_floor_does_widen():
    """A rank with real recorded variance (window's thin axis, cap_factor 6.0,
    the #201 policy ceiling) widens past both the #94 default AND the floor
    value -- the whole point of consuming this metadata."""
    prior = prior_for("window")
    assert prior.cap_factor[0] > CAP_FACTOR_FLOOR
    bound = _extent_veto_bound(prior, DEFAULT_TRACKER_CONFIG)
    typ = prior.typ_ext
    ratio_bound = prior.cap_factor[0] * typ[0]
    abs_bound = typ[0] + DEFAULT_TRACKER_CONFIG.extent_veto_abs_slack_m
    assert bound[0] == pytest.approx(max(ratio_bound, abs_bound))
    assert bound[0] > DEFAULT_TRACKER_CONFIG.extent_veto_factor * typ[0]


def test_no_prior_data_falls_back_to_widening_1_alone():
    """A hand-curated class (no GT distribution, cap_factor is None) only ever
    gets the absolute-allowance widening, never a per-axis ratio boost."""
    prior = prior_for("sphere")
    assert prior.cap_factor is None
    bound = _extent_veto_bound(prior, DEFAULT_TRACKER_CONFIG)
    typ = prior.typ_ext
    expected = np.maximum(
        DEFAULT_TRACKER_CONFIG.extent_veto_factor * typ,
        typ + DEFAULT_TRACKER_CONFIG.extent_veto_abs_slack_m,
    )
    assert np.allclose(bound, expected)


def test_bound_never_tightens_the_94_default():
    """Structural guarantee: for EVERY data-backed and hand-curated class, the
    #217 bound is never smaller than the plain #94 ratio on any axis."""
    from core.perception.dimension_priors import _DATA_PRIORS, _PHRASE_DATA_PRIORS, _HAND_PRIORS

    for label in {**_DATA_PRIORS, **_PHRASE_DATA_PRIORS, **_HAND_PRIORS}:
        prior = prior_for(label)
        bound = _extent_veto_bound(prior, DEFAULT_TRACKER_CONFIG)
        assert np.all(bound >= DEFAULT_TRACKER_CONFIG.extent_veto_factor * prior.typ_ext - 1e-9), label


def test_extent_veto_use_cap_factor_off_reconstructs_the_pre_217_ratio_only():
    """The historical-pin escape hatch: with widening 2 disabled and slack
    zeroed, the bound is exactly the old flat extent_veto_factor x typ_ext."""
    cfg = TrackerConfig(extent_veto_abs_slack_m=0.0, extent_veto_use_cap_factor=False)
    prior = prior_for("window")
    bound = _extent_veto_bound(prior, cfg)
    assert np.allclose(bound, cfg.extent_veto_factor * prior.typ_ext)


# --------------------------------------------------------------------------- #94 guarantee: sofa-vs-cup


def test_sofa_sized_box_still_vetoed_against_a_cup_track_by_an_order_of_magnitude():
    """PRESERVE #94's guarantee, the coordinator's own acceptance bar: a
    genuinely wrong-size association (a sofa-sized box merged into a cup
    track) must still be rejected by the widened bound, with margin to spare.
    Compares a cup's OWN widened per-axis bound against a real sofa's typical
    extent (the combined box of "cup track + sofa-sized detection" converges
    to ~the sofa's own extent, since the cup contributes negligibly to the
    union) -- every axis overshoots 1.6x-3.8x, ~11.2x by volume (three
    independent axis margins compound multiplicatively) -- a decisive,
    order-of-magnitude reject, not a coincidence of one axis."""
    cup = prior_for("cup")
    sofa = prior_for("sofa")
    bound = _extent_veto_bound(cup, DEFAULT_TRACKER_CONFIG)
    ratio = sofa.typ_ext / bound
    assert np.all(ratio > 1.5), ratio  # every axis overshoots the widened cup ceiling
    assert ratio.prod() > 10.0, ratio.prod()  # ~an order of magnitude by volume

    # End-to-end through the real associate()/BasicSceneIndex path, not just the
    # bound arithmetic: a cup track offered a sofa-sized re-detection at the
    # SAME centroid (the most generous possible case for a false merge) must
    # still mint a second instance, never fuse.
    idx = BasicSceneIndex()
    from core.perception.detector import Detection

    cup_det = Detection(tile_id=0, bbox_xyxy=(0.0, 0.0, 10.0, 10.0), label="cup", score=0.5)
    cup_half = cup.typ_ext / 2.0
    cup_pts = np.array([-cup_half, cup_half])
    cup_fused = _FakeFused(np.zeros(3), cup_pts)
    associate([(cup_det, cup_fused)], idx, cfg=DEFAULT_TRACKER_CONFIG)
    assert len(idx.all_instances()) == 1

    sofa_det = Detection(tile_id=0, bbox_xyxy=(0.0, 0.0, 10.0, 10.0), label="cup", score=0.5)
    sofa_half = sofa.typ_ext / 2.0
    sofa_pts = np.array([-sofa_half, sofa_half])  # same centroid as the cup track
    sofa_fused = _FakeFused(np.zeros(3), sofa_pts)
    associate([(sofa_det, sofa_fused)], idx, cfg=DEFAULT_TRACKER_CONFIG)
    assert len(idx.all_instances()) == 2, "a sofa-sized box must never fuse into a cup track"


def test_the_one_residual_rejection_needs_an_implausible_box():
    """Documents why the door pair (id47->id72, livingroom_1_g719195) stays
    rejected: its reconstructed combined thin-axis extent is 0.714 m -- a real
    door panel does not run 0.71 m thick. This is consistent with the pair
    being a heuristic false positive in the #216 discovery pass (two
    coincidentally same-labelled, spatially-close but physically DIFFERENT
    detections -- an angled doorway's AABB does not align with the door's true
    thin axis) rather than a genuine single re-observation, so leaving it
    rejected trades nothing real away."""
    pairs = {(t, a, b): (t, a, b, lb, dist, g1, g2) for t, a, b, lb, dist, born1, born2, g1, g2 in _all_pairs()}
    tag, id1, id2, label, dist, g1, g2 = pairs[("livingroom_1_g719195", 47, 72)]
    assert label == "door"
    rec1 = _record_from_geometry(id1, g1, label)
    amin2 = np.array(g2["aabb_min"], dtype=float)
    amax2 = np.array(g2["aabb_max"], dtype=float)
    combined_min = np.minimum(rec1.aabb_min, amin2)
    combined_max = np.maximum(rec1.aabb_max, amax2)
    thin_extent = float(np.sort(combined_max - combined_min)[0])
    assert thin_extent > 0.6  # an implausible door thickness

    v, _, _, decider = _replay_pair(id1, id2, label, dist, g1, g2, DEFAULT_TRACKER_CONFIG)
    assert v is False
    assert decider == "issue_94_extent_veto_factor_reject"


# --------------------------------------------------------------------------- #216 before/after: the acceptance table


def _before_after_table():
    old_cfg = TrackerConfig(extent_veto_abs_slack_m=0.0, extent_veto_use_cap_factor=False)
    new_cfg = DEFAULT_TRACKER_CONFIG
    rows = []
    for tag, id1, id2, label, dist, born1, born2, g1, g2 in _all_pairs():
        v_old, a_old, b_old, d_old = _replay_pair(id1, id2, label, dist, g1, g2, old_cfg)
        v_new, a_new, b_new, d_new = _replay_pair(id1, id2, label, dist, g1, g2, new_cfg)
        rows.append((tag, id1, id2, label, dist, v_old, v_new, a_old, a_new))
    return rows


def test_before_after_table_matches_the_acceptance_criteria():
    """The coordinator's own acceptance bar, checked mechanically per pair:
    the 8 admittable thin-class rejections flip False->True; the door pair and
    the growth-cap (#180) chair pair are UNCHANGED (still False) -- #217 does
    not touch the #180 mechanism at all, by construction (the growth-cap check
    runs before ANY of this file's code, unconditionally)."""
    rows = _before_after_table()
    print("\ntag                     id1   id2  label            dist   old    new")
    for tag, id1, id2, label, dist, v_old, v_new, a_old, a_new in rows:
        print(f"{tag:24s}{id1:5d} {id2:5d}  {label:15s}  {dist:.3f}  {v_old!s:6s} {v_new!s:6s}")

    flips = [(id1, id2, label) for tag, id1, id2, label, dist, v_old, v_new, a_old, a_new in rows
             if (not v_old) and v_new]
    still_false = [(id1, id2, label) for tag, id1, id2, label, dist, v_old, v_new, a_old, a_new in rows
                   if not v_new]
    already_true = [(id1, id2, label) for tag, id1, id2, label, dist, v_old, v_new, a_old, a_new in rows
                     if v_old]

    assert len(rows) == 13
    assert len(flips) == 8, flips
    assert {(id1, id2) for id1, id2, label in still_false} == {(47, 72), (89, 165)}, still_false
    assert len(already_true) == 3  # potted plant + 2 cabinet-shelf pairs (never needed the fix)

    # The growth-cap pair's own site-A verdict is untouched by #217 (it fires
    # identically under both configs -- #217 only reaches the LAST-resort
    # branch after site A and the #153 floor have already run).
    growth_cap_row = next(r for r in rows if (r[1], r[2]) == (89, 165))
    assert growth_cap_row[7] is True and growth_cap_row[8] is True  # a_old, a_new both fired


# --------------------------------------------------------------------------- downstream: consolidation, not waves


def _consolidation_replay(id1, id2, g1, g2, label, cfg, *, extra_reobs=2):
    """Drive the REAL associate()/decay_singletons() pipeline for one #216
    flicker pair PLUS a few more synthetic re-observations at the same spot
    (small jitter around the second archived box, mirroring how a real object
    keeps being seen while in view) -- the actual downstream claim: does the
    track CONSOLIDATE (one id, n_obs climbing, survives H15(a) decay) instead
    of the wave pattern (prune, respawn, repeat)?

    Returns (final live instances for this label, the set of EVERY raw
    instance_id ever minted across the replay, the set of ids H15(a) actually
    pruned) -- the wave pattern shows up as multiple minted ids with at least
    one pruned; consolidation shows up as exactly one minted id, never pruned."""
    from core.perception.detector import Detection

    idx = BasicSceneIndex()
    first_seen: dict[int, int] = {}
    decay_k = cfg.decay_k
    ever_minted: set[int] = set()
    ever_pruned: set[int] = set()

    def _det():
        return Detection(tile_id=0, bbox_xyxy=(0.0, 0.0, 10.0, 10.0), label=label, score=0.4)

    amin1 = np.array(g1["aabb_min"], dtype=float)
    amax1 = np.array(g1["aabb_max"], dtype=float)
    fused1 = _FakeFused(np.array(g1["position"], dtype=float), np.stack([amin1, amax1]))
    touched = associate([(_det(), fused1)], idx, cfg=cfg)
    ever_minted.update(touched)
    for iid in touched:
        first_seen.setdefault(iid, 0)
    ever_pruned.update(decay_singletons(idx, first_seen, keyframe_idx=1, decay_k=decay_k))

    amin2 = np.array(g2["aabb_min"], dtype=float)
    amax2 = np.array(g2["aabb_max"], dtype=float)
    fused2 = _FakeFused(np.array(g2["position"], dtype=float), np.stack([amin2, amax2]))
    kf = decay_k + 1  # the exact decisive re-observation: right at the H15(a) boundary
    touched = associate([(_det(), fused2)], idx, cfg=cfg)
    ever_minted.update(touched)
    for iid in touched:
        first_seen.setdefault(iid, kf)
    ever_pruned.update(decay_singletons(idx, first_seen, keyframe_idx=kf, decay_k=decay_k))

    rng = np.random.default_rng(0)
    for i in range(extra_reobs):
        kf += decay_k + 1
        jitter = rng.uniform(-0.02, 0.02, size=3)
        pts = np.stack([amin2 + jitter, amax2 + jitter])
        fused_i = _FakeFused(np.array(g2["position"], dtype=float) + jitter, pts)
        touched = associate([(_det(), fused_i)], idx, cfg=cfg)
        ever_minted.update(touched)
        for iid in touched:
            first_seen.setdefault(iid, kf)
        ever_pruned.update(decay_singletons(idx, first_seen, keyframe_idx=kf, decay_k=decay_k))

    insts = [r for r in idx.all_instances() if r.label == canonical_for_match(label)]
    return insts, ever_minted, ever_pruned


@pytest.mark.parametrize(
    "tag, id1, id2, label",
    [
        ("office_1_g719515", 120, 191, "window"),
        ("office_1_g719515", 144, 202, "window"),
    ],
)
def test_flickering_window_tracks_consolidate_under_the_fix(tag, id1, id2, label):
    """The downstream claim: replaying the SAME flicker-pair geometry (plus a
    couple more re-observations at the same spot, since a real object stays
    in view) through the OLD ceiling produces the wave pattern (repeated
    prune+respawn, n_obs stuck at 1); through the NEW ceiling it consolidates
    to ONE id whose n_obs climbs past 1 and survives H15(a) decay."""
    pairs = {(t, a, b): (t, a, b, lb, dist, g1, g2) for t, a, b, lb, dist, born1, born2, g1, g2 in _all_pairs()}
    _, _, _, lb, dist, g1, g2 = pairs[(tag, id1, id2)]
    assert lb == label

    old_cfg = TrackerConfig(extent_veto_abs_slack_m=0.0, extent_veto_use_cap_factor=False)
    old_insts, old_minted, old_pruned = _consolidation_replay(id1, id2, g1, g2, label, old_cfg)
    # the wave pattern: the archived pair's own re-detection is vetoed, so a
    # SECOND raw id gets minted and the first is pruned by H15(a) -- exactly
    # the "born, starved, pruned, respawned" signature, even though the
    # respawned id then happens to self-consolidate against its own later
    # near-identical jitter (a different, #153-gated mechanism, unaffected by
    # #217) and so only one instance survives to the end.
    assert len(old_minted) >= 2, "expected the pre-#217 wave pattern (a second id minted)"
    assert len(old_pruned) >= 1, "expected H15(a) to have pruned the starved first id"

    new_cfg = DEFAULT_TRACKER_CONFIG
    new_insts, new_minted, new_pruned = _consolidation_replay(id1, id2, g1, g2, label, new_cfg)
    assert len(new_minted) == 1, f"expected ONE id minted (no wave), got {len(new_minted)}"
    assert len(new_pruned) == 0, "expected consolidation to survive decay entirely, no prune"
    assert len(new_insts) == 1
    assert new_insts[0].n_obs >= 3  # climbed past the original 2 archived hits via the extra re-observations
