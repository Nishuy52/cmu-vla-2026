"""Cross-frame association + the glued perception pipeline.

New 3D detections (from :mod:`core.perception.fusion`) are matched to existing
:class:`~core.interfaces.InstanceRecord`s by:

* **label compatibility** — canonicalised through the scene index's
  :func:`~core.perception.scene_index.normalize_label` *and* the vocab's
  :data:`~core.parsing.vocab.NOUN_ALIASES` table (so ``television`` associates with a
  ``tv`` instance, ``fridge`` with ``refrigerator``), and
* **centroid proximity** — 3D distance below ``gate`` (default 0.75 m).

Matching is greedy nearest: candidate (detection, instance) pairs are considered in
ascending distance and each side consumed at most once. Matched detections are fed
to :meth:`~core.perception.scene_index.BasicSceneIndex.add`, which reuses the existing
IoU-gated merge machinery (points concatenated, trimmed AABB recomputed, ``n_obs``
incremented, max ``score`` kept). Unmatched detections become new instances.

:class:`PerceptionPipeline` glues tiling -> detector -> fusion -> tracker behind a
keyframe gate so we only pay for perception every K-th frame or after enough vehicle
motion (translation / rotation), mirroring the geometry toolbox's ``Thresholds``
pattern.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

import numpy as np

from core.interfaces import InstanceRecord, LidarScan, OdomState, PanoFrame
from core.parsing.vocab import NOUN_ALIASES
from core.perception.colour import ColourTally, tally_from_colors
from core.perception.pano_projection import project_points_to_pano, sample_colors
from core.perception.detector import (
    CAPTION_JOIN_MARKER,
    Detection,
    DetectorFn,
    ENV_RAW_DETECTION_DUMP_PATH,
    GATE_ACCEPTED,
    GATE_NO_LIDAR_CLUSTER,
    dump_raw_detections,
)

_LOGGER = logging.getLogger(__name__)
from core.perception.fusion import (
    DEFAULT_FUSION_CONFIG,
    Fused3D,
    FusionConfig,
    fuse_detection,
)
from core.perception.scene_index import (
    BasicSceneIndex,
    DEFAULT_INSTANCE_DUMP_INTERVAL_S,
    ENV_INSTANCE_DUMP_INTERVAL_S,
    ENV_INSTANCE_DUMP_PATH,
    dump_instance_index,
    labels_foldable,
    normalize_label,
)
from core.perception.tiling import (
    DEFAULT_N_TILES,
    DEFAULT_TILE_HFOV,
    DEFAULT_TILE_VFOV,
    project_tiles,
)

# --------------------------------------------------------------------------- label compat

#: Reverse of NOUN_ALIASES: every alias surface form -> its canonical noun. Combined
#: with normalize_label this lets 'television' match a 'tv' instance and vice-versa.
_ALIAS_TO_CANON: dict[str, str] = {}
for _canon, _aliases in NOUN_ALIASES.items():
    for _a in _aliases:
        _ALIAS_TO_CANON[normalize_label(_a)] = normalize_label(_canon)
    _ALIAS_TO_CANON[normalize_label(_canon)] = normalize_label(_canon)


def canonical_for_match(label: str) -> str:
    """Fully canonicalise a label for association: scene-index normalise then fold
    through NOUN_ALIASES (so 'television' and 'tv' collapse to one key)."""
    base = normalize_label(label)
    return _ALIAS_TO_CANON.get(base, base)


def labels_compatible(a: str, b: str) -> bool:
    """True when two labels denote the same canonical noun (alias/synonym-aware), OR
    are a subphrase/duplicated-token fold of one another (issue #89: GDINO phrase
    decode fragments -- "potted"/"plant" vs "potted plant", "door door" vs "door" --
    see :func:`~core.perception.scene_index.labels_foldable`). Association is already
    centroid-gated (the caller only considers pairs within ``cfg.gate``), so folding
    here only ever widens which CO-LOCATED detection can join an existing track; it
    never on its own decides two spatially-unrelated detections are the same object."""
    return canonical_for_match(a) == canonical_for_match(b) or labels_foldable(a, b)


# --------------------------------------------------------------------------- colour


def colour_tally_for_cluster(points: np.ndarray, pano: PanoFrame) -> ColourTally | None:
    """Issue #121: quantise a fused detection's lidar points into a colour tally.

    Projects the cluster's map-frame ``points`` back onto ``pano`` with the same
    forward-projection machinery :class:`~core.perception.colored_map.ColoredVoxelMap`
    uses (:mod:`core.perception.pano_projection`) — reused, not reimplemented, so
    live colour and the debug-viz colored map agree pixel-for-pixel. Returns
    ``None`` when the cluster has no points or none of them land on the panorama
    this keyframe (out of VFOV / range-gated) — no observation, not a "gray"
    observation (see :func:`core.perception.colour.tally_from_colors`).
    """
    if points is None or len(points) == 0:
        return None
    rows, cols, valid = project_points_to_pano(points, pano.odom)
    if not valid.any():
        return None
    colors = sample_colors(pano.image, rows, cols, valid)
    return tally_from_colors(colors, valid)


# --------------------------------------------------------------------------- config


@dataclass(frozen=True)
class TrackerConfig:
    """Association gate. Non-spec values flagged in the task report.

    Issues #94/#89: a single fixed-radius centroid gate is the wrong criterion in
    kind, not just in magnitude -- one value cannot simultaneously be tight enough to
    keep desk-spaced monitors (~0.4-0.5 m apart) separate AND loose enough to
    re-identify a chair across viewpoint-driven centroid jitter. ``gate`` stays the
    ceiling (and the value used for classes with no dimension prior); actual
    per-match association below now scales down from it per class footprint
    (:func:`_assoc_gate`) and is additionally vetoed when it would produce an
    implausibly large merged box for the class (:func:`_match_plausible`).
    """

    gate: float = 0.75  # m; ceiling on centroid distance for a match
    gate_min: float = 0.25         # m; floor -- a class prior never gates tighter than this
    gate_extent_frac: float = 0.5  # fraction of a class's typical diagonal used as its gate
    # Issue #94: reject a match that would grow the fused box's sorted-axis extent past
    # this multiple of the class's typical (data-derived) extent -- a merged instance
    # whose extent grossly exceeds its class prior is self-evidently a bad merge,
    # however close the centroids landed. Generous enough that legitimate multi-view
    # growth of ONE real object (occlusion revealing more of it over time) is never
    # blocked; tight enough that two distinct same-class objects at typical spacing
    # (e.g. adjacent desk monitors) cannot fuse into one oversized box.
    extent_veto_factor: float = 1.3
    # Issue #153: floor below which the extent veto never fires, regardless of how
    # far the accumulated union has grown. A match landing THIS close to the
    # candidate's own centroid is, by construction, a re-observation of the same
    # physical object (the fused centroid is a noisy summary of the same points);
    # rejecting it just mints a duplicate instance at the same spot instead of
    # preventing anything. Chosen well below every real inter-object spacing this
    # veto exists to police -- #94's desk monitors (0.5 m) and #89's dense chair
    # cluster (0.8 m) -- and well above single-object depth-noise centroid drift
    # (millimetres to a few centimetres), so it only short-circuits the
    # ~zero-distance case the veto's own docstring calls out as wrong to reject.
    extent_veto_min_sep: float = 0.2
    # Issue #176: once a candidate's OWN accumulated box already sits at or past the
    # class-typical ceiling (a legitimately large object, or a noisy/oversized
    # detection that pushed it there), the #153 floor above stops helping the moment
    # later re-observations drift past 0.2 m -- which large/noisy clusters routinely
    # do (a bbox spanning half the panorama has far more centroid jitter than a
    # tightly-boxed monitor). Every such re-observation then fails BOTH #153's
    # distance floor AND the absolute typical-size ceiling (which the candidate
    # already exceeds on its own), so it is vetoed regardless of how little the box
    # would actually grow -- ratcheting one real, already-oversized object into a
    # fresh duplicate instance every keyframe (the live sweep's 46-chair-for-6-GT
    # overcount, #176). The fix: judge GROWTH relative to what the candidate already
    # spans, not just the absolute class ceiling. A merge that does not enlarge the
    # box beyond this tolerance on any axis is by construction absorbing more of the
    # SAME already-established (however oversized) object, not annexing a distinct
    # one -- a genuinely separate object at typical class spacing adds most of its
    # own footprint to the union (issue #161's two TVs at 0.65 m spacing grow the
    # union's width by 0.65 m, far past this), so this never masks that case.
    extent_growth_tol: float = 0.25
    # Issue #180: neither waiver above caps CUMULATIVE growth. The #153 floor
    # (dist <= extent_veto_min_sep) waives every check outright, no matter how
    # large a single accepted jump is; the #176 growth branch only bounds ONE
    # match's own contribution, not the running total across many accepted
    # matches. A chain of individually-plausible merges can therefore walk a
    # track's box outward without limit -- one adversarial construction does
    # exactly this by riding the #153 floor every step. This is the backstop:
    # regardless of which branch above would otherwise pass a match, the
    # resulting box may never grow more than this many metres, on any axis,
    # past the track's FIRST accepted box (tracked by :func:`associate`, kept
    # off :class:`~core.interfaces.InstanceRecord` the same way the scene
    # index keeps its colour tally external, issue #121). Generous enough that
    # the realistic multi-merge growth #176's own fix chain measured (~1.15 m
    # on a chair's long axis before the association gate itself stops it)
    # still completes; tight enough that no chain of "individually plausible"
    # merges can walk a box arbitrarily far.
    extent_cumulative_growth_cap: float = 1.5
    # Issue #217: #94's absolute ceiling (extent_veto_factor x typ_ext) is a SINGLE
    # ratio for every class on every axis. The #216 event-level replay (13 archived
    # flicker-wave pairs) found this is the actual, dominant rejector behind the
    # #216 track-flicker signature -- 9 of 10 rejections, not the #180 growth-cap
    # ordering #216 first suspected -- concentrated on THIN classes: a window's
    # typical depth is 0.06 m, so 1.3x typical (~0.08 m) is an impossible bound
    # against any real lidar-fused thickness. Two per-axis widenings, taken as the
    # MAX (never tighter than the un-widened #94 ceiling on any axis):
    #  1. an absolute allowance -- an axis may grow to typ_ext[axis] + this many
    #     metres even where the ratio bound would clip a class whose typical
    #     extent is itself tiny (a thin axis's own ratio bound is always small in
    #     absolute terms, however large the ratio). NOT the 0.15 m first proposed:
    #     that admitted only 2 of the #216 replay's 9 flagged rejections (the
    #     rest needed 0.36-0.55 m more headroom than their own cap_factor-widened
    #     ratio bound gave them). 0.45 m is the largest value measured, by binary
    #     search against the FULL perception test suite, that still admits 8 of
    #     those 9 WITHOUT flipping any existing #94/#153/#161/#176 guard (the
    #     tightest of which, issue #161's two-TVs-at-0.65m decisive-band split,
    #     starts failing between 0.46 and 0.48 m of slack) -- see
    #     tests/perception/test_tracker_issue_217.py's before/after table. The
    #     9th (the door pair) needs 0.547 m, which DOES flip a guard
    #     (test_distinct_archived_chairs_stay_separate_post_fix, 7 real chairs ->
    #     6) at any slack past ~0.48 m; it is left rejected rather than trade a
    #     live false-negative guard for one flicker pair whose own reconstructed
    #     box (0.71 m "thick" for a door) is itself suspect (see that test's own
    #     note in test_tracker_issue_217.py).
    #  2. the class's own recorded per-axis cap_factor (issue #201's P95/median
    #     triple, until now unused metadata -- see dimension_priors.py) where a
    #     data-backed prior exists, taken as max(cap_factor[axis],
    #     extent_veto_factor) so a class's DATA-MEASURED real size variance never
    #     tightens the ceiling, only ever loosens or matches it. Classes with no
    #     data-backed distribution (cap_factor is None) fall back to widening 1
    #     alone -- no measured variance to consult.
    extent_veto_abs_slack_m: float = 0.45
    # Issue #217: widening 2's own on/off switch, default on. Exists ONLY so a
    # test that needs to reconstruct the exact pre-#217 ceiling (a "PRE_FIX"
    # historical-numbers pin, matching the existing convention other issues'
    # PRE_FIX tests already use for their own superseded checks) can disable it
    # explicitly rather than depending on every class's recorded cap_factor
    # happening to sit at the floor. No non-test caller has a reason to set this
    # False.
    extent_veto_use_cap_factor: bool = True
    # H15(a) track decay: an instance still at n_obs==1 that has not been re-observed
    # within this many keyframes of first sighting is a one-frame ghost and is pruned.
    # Confirmed tracks (n_obs>=2) are NEVER decayed. 0 disables decay.
    decay_k: int = 5


DEFAULT_TRACKER_CONFIG = TrackerConfig()


@dataclass(frozen=True)
class KeyframeConfig:
    """Keyframe gate tunables (geometry-Thresholds style, one dataclass).

    Process a frame when it is at least ``every_k`` frames since the last keyframe,
    OR the vehicle has moved ``min_translation`` metres, OR turned ``min_rotation``
    radians, since the last processed frame.
    """

    every_k: int = 1
    min_translation: float = 0.5           # m
    min_rotation: float = np.deg2rad(30.0)  # rad


DEFAULT_KEYFRAME_CONFIG = KeyframeConfig()


# --------------------------------------------------------------------------- tracker


def _fused_to_record(det: Detection, fused: Fused3D, instance_id: int) -> InstanceRecord:
    """Build an InstanceRecord from a fused detection (AABB from its point set).

    The scene index recomputes a trimmed AABB on merge; here we seed a raw
    min/max box + centroid from the cluster so a brand-new instance is well-formed.
    """
    from core.perception.dimension_priors import floor_degenerate_aabb

    pts = fused.points
    aabb_min = pts.min(axis=0).astype(float)
    aabb_max = pts.max(axis=0).astype(float)
    canon = normalize_label(det.label)
    # Issue #125: a single-viewpoint cluster can be (near-)flat on an axis (e.g. a
    # front-shell-only sighting); raise only that broken axis to a plausible floor
    # so the record is never geometrically degenerate at birth.
    aabb_min, aabb_max = floor_degenerate_aabb(aabb_min, aabb_max, canon)
    aliases = NOUN_ALIASES.get(canon, ())
    return InstanceRecord(
        instance_id=instance_id,
        label=canon,
        score=float(det.score),
        n_obs=1,
        centroid=fused.centroid.astype(float),
        aabb_min=aabb_min,
        aabb_max=aabb_max,
        points=pts.astype(float),
        aliases=tuple(aliases),
    )


def _assoc_gate(label: str, cfg: TrackerConfig) -> float:
    """Per-class association gate (issues #94/#89): scaled from the class's
    data-derived typical diagonal instead of one fixed radius for every class.

    A monitor's typical diagonal (~0.72 m) yields a much tighter gate than a sofa's
    (~2.5 m, clipped back down to the ``cfg.gate`` ceiling) -- proportional to what
    "the same object, re-observed" plausibly looks like for THAT class, rather than
    one radius that is simultaneously too loose for desk-spaced monitors and
    (coincidentally) about right for a sofa. Classes with no dimension prior fall
    back to ``cfg.gate`` unchanged (unable to judge, so don't tighten blindly).
    """
    from core.perception.dimension_priors import prior_for

    prior = prior_for(canonical_for_match(label))
    if prior is None:
        return cfg.gate
    diag = float(np.linalg.norm(prior.typ_ext))
    return float(np.clip(cfg.gate_extent_frac * diag, cfg.gate_min, cfg.gate))


def _match_plausible(
    candidate: InstanceRecord,
    fused: Fused3D,
    cfg: TrackerConfig,
    dist: float,
    first_box: tuple[np.ndarray, np.ndarray] | None = None,
) -> bool:
    """Issue #94: veto a match that would blow the fused box past a plausible size
    for its class -- UNLESS (issue #153) the centroids landed within
    ``cfg.extent_veto_min_sep`` of each other, in which case this is by construction
    a re-observation of the same physical object and the veto never fires.

    Compares the SORTED (thin, mid, long) extent of the union of ``candidate``'s
    current trimmed AABB and the incoming detection's raw cluster AABB against the
    class's sorted typical extent (:mod:`core.perception.dimension_priors`, already
    used for marker clamping -- H12/red-team OR-F6). No prior -> can't judge -> never
    vetoes (unchanged behaviour for the ~1/3 of the vocabulary without a prior).

    This is what makes the sequential same-batch folding in :func:`associate` safe:
    genuine repeat/duplicate sightings of ONE real object keep the union near its own
    natural size and pass; two distinct same-class objects at roughly their class's
    typical spacing (e.g. adjacent desk monitors, adjacent dense-packed chairs) blow
    the union past ``extent_veto_factor`` x typical and are correctly kept apart.

    Issue #153: without the ``dist`` floor, a real object whose accumulated AABB has
    already grown past ``extent_veto_factor`` x typical (e.g. after one prior fuse
    revealed more of it) permanently vetoes EVERY later re-observation, however
    close the centroid -- each veto falls to ``associate``'s else-branch and mints a
    fresh duplicate instance at the same spot, forever. Extent is only meaningful
    evidence of "different object" when the two detections are actually apart; at
    near-zero separation it is not evidence of anything and must not veto.

    Issue #176: #153's floor only covers near-ZERO separation. A candidate that is
    already large/noisy enough to sit at or past the typical-size ceiling (a big
    real object, or a bbox-unstable detection) keeps drifting past that 0.2 m floor
    on ordinary re-observations, so it fails BOTH the floor AND the absolute ceiling
    (which it already exceeded before this match was even attempted) on every later
    keyframe -- ratcheting into a fresh duplicate each time. Below, a merge that does
    not enlarge the box beyond ``cfg.extent_growth_tol`` on ANY axis is accepted
    regardless of the absolute ceiling: it is, by construction, absorbing more of the
    object the candidate already spans, not annexing a separate one -- a genuinely
    distinct same-class object at typical spacing adds most of its own footprint to
    the union (issue #161's two TVs 0.65 m apart grow the union's width by 0.65 m),
    which this growth check still correctly rejects.

    Issue #180: everything above judges only THIS ONE match -- neither waiver
    caps how far a chain of individually-plausible matches walks the box over
    many keyframes. ``first_box`` (the track's aabb at the moment it was FIRST
    accepted, threaded in by :func:`associate`; ``None`` for any caller that
    predates #180) is checked FIRST, ahead of and regardless of the #153/#176
    waivers below: the union of ``first_box`` with this match's own combined
    box may grow by at most ``cfg.extent_cumulative_growth_cap`` on any axis,
    full stop. This is the one check in this function that a match cannot
    talk its way around by landing close (#153) or growing little (#176) --
    those only ever describe ONE step, never the running total.

    Issue #217: the LAST resort below (the absolute-ceiling ratio,
    ``cfg.extent_veto_factor x typ_ext``) used ONE flat ratio for every class,
    every axis. The #216 event-level replay found this is what actually starves
    real re-detections: on a thin class (window typ. depth 0.06 m), 1.3x typical
    is an impossible ~0.08 m bound against any real lidar-fused thickness, so a
    genuinely-close re-detection is vetoed here 9 times out of 10 rejections --
    not by the #180 growth cap #216 first suspected. The bound
    (:func:`_extent_veto_bound`) widens per axis (never tightens) by the larger
    of an absolute allowance and the class's own recorded per-axis size-variance
    (:attr:`~core.perception.dimension_priors.ClassPrior.cap_factor`, issue
    #201's P95/median triple -- until now unused metadata). A genuinely
    wrong-size match (issue #161's guard: two TVs at typical spacing, a sofa
    box matched to a cup track) is unaffected: those fail by ratios/absolute
    margins the widened bound does not reach either.
    """
    new_min = fused.points.min(axis=0)
    new_max = fused.points.max(axis=0)
    combined_min = np.minimum(candidate.aabb_min, new_min)
    combined_max = np.maximum(candidate.aabb_max, new_max)
    combined_ext = combined_max - combined_min
    candidate_ext = candidate.aabb_max - candidate.aabb_min

    if first_box is not None:
        first_min, first_max = first_box
        cumulative_min = np.minimum(first_min, combined_min)
        cumulative_max = np.maximum(first_max, combined_max)
        cumulative_growth = (cumulative_max - cumulative_min) - (first_max - first_min)
        if not bool(np.all(cumulative_growth <= cfg.extent_cumulative_growth_cap)):
            return False

    if dist <= cfg.extent_veto_min_sep:
        return True
    growth = combined_ext - candidate_ext
    if bool(np.all(growth <= cfg.extent_growth_tol)):
        return True
    from core.perception.dimension_priors import prior_for

    prior = prior_for(canonical_for_match(candidate.label))
    if prior is None:
        return True
    ext = np.sort(combined_ext)
    return bool(np.all(ext <= _extent_veto_bound(prior, cfg)))


def _extent_veto_bound(prior, cfg: TrackerConfig) -> np.ndarray:
    """Issue #217: the per-axis (thin, mid, long) ceiling :func:`_match_plausible`'s
    last resort compares a combined box against -- widened past the old flat
    ``cfg.extent_veto_factor x typ_ext`` on every axis (never tightened), by the
    larger of two per-axis widenings:

      1. an absolute allowance (``cfg.extent_veto_abs_slack_m``): a ratio bound
         computed off a tiny typ_ext (a thin axis) is tiny in absolute terms
         however generous the ratio, so every class gets at least this many
         metres of headroom past its typical extent on every axis.
      2. the class's own recorded per-axis cap_factor (issue #201's P95/median
         triple) where a data-backed prior exists AND that rank carries real
         evidence -- i.e. sits strictly above
         :data:`~core.perception.dimension_priors.CAP_FACTOR_FLOOR`. A rank AT
         the floor is #201's own "not enough data / not unusually variable"
         default, not a measurement, so it does not widen anything: the #94
         default (``cfg.extent_veto_factor``) applies unchanged on that rank.
         This is what keeps issue #161's decisive-band guard (two televisions,
         whose long/mid ranks sit exactly at the recorded floor) rejecting
         genuinely distinct objects exactly as before -- only a class's ranks
         with REAL measured variance (window's thin/mid, chair's long, book,
         bookcase, ...) get widened. ``prior.cap_factor`` is ``None`` for every
         hand-curated class (no GT distribution to measure one from) -- those
         fall back to widening 1 alone.
    """
    from core.perception.dimension_priors import CAP_FACTOR_FLOOR

    typ = prior.typ_ext  # already sorted (thin, mid, long)
    factor = np.full(3, cfg.extent_veto_factor)
    if cfg.extent_veto_use_cap_factor and prior.cap_factor is not None:
        has_signal = prior.cap_factor > CAP_FACTOR_FLOOR
        factor = np.where(
            has_signal, np.maximum(prior.cap_factor, cfg.extent_veto_factor), factor
        )
    ratio_bound = factor * typ
    abs_bound = typ + cfg.extent_veto_abs_slack_m
    return np.maximum(ratio_bound, abs_bound)


def associate(
    fused_dets: list[tuple[Detection, Fused3D]],
    index: BasicSceneIndex,
    cfg: TrackerConfig = DEFAULT_TRACKER_CONFIG,
    colour_obs: list[ColourTally | None] | None = None,
    odom: OdomState | None = None,
    keyframe_cfg: KeyframeConfig = DEFAULT_KEYFRAME_CONFIG,
) -> list[int]:
    """Sequential nearest-plausible associate fused detections to instances.

    Issues #94 (under-segmentation: distinct same-class objects fused into one) and
    #89 (over-segmentation: one real object spawning several instances) are opposite
    symptoms of the same wrong criterion: a single fixed-radius centroid gate,
    checked only against the PRE-BATCH instance snapshot. That snapshot-only check
    has a structural gap -- two detections of the very same physical object arriving
    in the SAME batch (e.g. duplicate proposals from overlapping detector tiles) are
    each compared only against instances that existed BEFORE this batch, never
    against each other, so the second one can find no candidate and mints a fresh
    ghost instance (#89) even though it is sitting right on top of the instance the
    first one just created or matched.

    Fixed here by folding sequentially: each detection is matched against a running
    pool that starts as the pre-batch instances and gains every instance
    matched-into or newly created earlier in THIS batch, so within-frame duplicates
    fold together instead of each independently spawning a ghost. What makes this
    safe against wrongly fusing two distinct nearby same-class objects (which the
    pre-#94-fix code prevented only by construction, via a stricter "hidden"
    contributor -- the one-shot IoU re-check inside ``index.add``/``merge_into``,
    since removed) is that every candidate match, pre-batch or same-batch, is now
    ALSO checked by :func:`_match_plausible`: fusing is vetoed once the resulting box
    would grossly exceed the class's typical size, regardless of which pool the
    candidate came from. The gate itself is also now per-class
    (:func:`_assoc_gate`) instead of one fixed radius for every object size.

    Returns the list of instance_ids touched (matched-into or newly created), ALIGNED
    to the input ``fused_dets`` order (i.e. ``touched[i]`` is the instance the i-th
    input detection landed in) -- NOT the order detections were internally processed.

    Issue #112: the sequential same-batch fold above makes the result depend on
    processing order UNLESS that order is itself a deterministic function of the
    detection set rather than of however the caller happened to list them (GDINO's
    raw per-tile output order, which correlates with nothing physical). Before
    folding, the batch is therefore canonicalised into a stable order keyed on
    world-grounded, data-derived properties only -- canonical label, then centroid
    (x, y, z), then cluster extent as a final tiebreak for the vanishingly rare
    exact-centroid tie. Deliberately excludes ``instance_id`` and input/arrival
    index as sort keys: #111/#113 were closed precisely because such numbering
    artifacts were allowed to decide physical outcomes, and using them here would
    reintroduce the same defect class this fix exists to close.

    ``colour_obs`` (issue #121), if given, is a list of per-detection colour tallies
    (:mod:`core.perception.colour`) parallel to ``fused_dets`` -- index ``i``'s tally
    is threaded into the scene index for detection ``i`` so it accumulates into
    whichever instance that detection lands in. It is indexed by the detection's
    ORIGINAL input position (``orig_idx``), never by processing position, so #112's
    canonicalisation above cannot pair a tally with a different detection's
    instance. ``None`` (the default, and every pre-#121 caller) leaves colour
    untouched, unchanged behaviour.

    ``odom`` (issue #191): the vehicle pose this batch of detections was taken
    from, used ONLY to maintain ``InstanceRecord.n_views`` -- a second,
    purely-additive counter alongside ``n_obs``. A live-replay measurement
    (5 slots, 94 GT-matched real instances) REFUTED an earlier version of
    this fix that gated ``n_obs`` itself on pose: every MIN_GROUND_OBS/
    ESTABLISH_N_OBS threshold in the codebase (#151/#184/#186) was calibrated
    against dwell-counted n_obs, so gating n_obs broke 51% of real instances
    out of those gates in one step (n_obs>=3 pass rate 56% -> 5%). ``n_obs``
    therefore keeps its exact pre-#191 dwell semantics here, completely
    unaffected by ``odom`` -- every existing gate/threshold sees identical
    behaviour to main. ``n_views`` is the new, separate signal: an
    observation that matches an EXISTING instance counts toward ``n_views``
    only if the pose has moved >= ``keyframe_cfg.min_translation`` or turned
    >= ``keyframe_cfg.min_rotation`` since that instance's last COUNTED
    viewpoint -- reusing the same two keyframe-gate constants, not a new
    tunable. A brand-new instance's first observation always counts (nothing
    to compare against yet). Nothing in this codebase reads ``n_views`` yet
    (ranking consumers are a separate, later change); it exists purely to be
    computed and dumped. ``None`` (the default, and every pre-#191 caller,
    including every test that calls ``associate()`` without a pose) leaves
    ``n_views`` ungated too -- it then just tracks ``n_obs`` exactly (one
    counted per accepted detection), since there is no pose to gate against.
    """
    pool: list[InstanceRecord] = list(index.all_instances())

    # Issue #180: the cumulative growth cap needs each track's FIRST accepted
    # box -- state that outlives any one associate() call, which
    # InstanceRecord itself does not carry (frozen semantics; owned surface
    # excludes core.interfaces). Kept on the index object instead, the same
    # convention BasicSceneIndex already uses for its own colour tally
    # (issue #121, kept OUTSIDE InstanceRecord) -- so it lives and dies with
    # the index it belongs to and never leaks between independent
    # SceneIndex instances (e.g. separate test cases, separate episodes).
    first_boxes: dict[int, tuple[np.ndarray, np.ndarray]] = getattr(
        index, "_tracker_first_boxes", None
    )
    if first_boxes is None:
        first_boxes = {}
        index._tracker_first_boxes = first_boxes
    for rec in pool:
        first_boxes.setdefault(rec.instance_id, (rec.aabb_min.copy(), rec.aabb_max.copy()))

    # Issue #191: per-instance "last pose an observation of THIS instance was
    # last counted toward n_views from" -- the same external-state convention
    # as `first_boxes` above (and the index's own colour tally, #121): it has
    # to outlive any one associate() call and must never leak between
    # independent SceneIndex instances, so it lives on the index object, not
    # on the (frozen-semantics) InstanceRecord. Feeds ONLY n_views -- n_obs is
    # never gated by this (see the `odom` docstring above for why).
    last_counted_pose: dict[int, OdomState] = getattr(
        index, "_tracker_last_counted_pose", None
    )
    if last_counted_pose is None:
        last_counted_pose = {}
        index._tracker_last_counted_pose = last_counted_pose

    def _pose_advanced(iid: int) -> bool:
        """True if ``odom`` has moved/turned enough since ``iid``'s last
        n_views-COUNTED observation to count this one too (issue #191)."""
        if odom is None:
            return True  # no pose given -- ungated, matches pre-#191 behaviour
        prev = last_counted_pose.get(iid)
        if prev is None:
            return True  # never counted before -- nothing to compare against
        moved = np.hypot(odom.x - prev.x, odom.y - prev.y)
        turned = abs(float(np.arctan2(
            np.sin(odom.yaw - prev.yaw), np.cos(odom.yaw - prev.yaw),
        )))
        return moved >= keyframe_cfg.min_translation or turned >= keyframe_cfg.min_rotation

    def _order_key(item: tuple[int, tuple[Detection, Fused3D]]):
        _, (det, fused) = item
        cx, cy, cz = (float(c) for c in fused.centroid)
        ext = fused.points.max(axis=0) - fused.points.min(axis=0)
        return (
            canonical_for_match(det.label),
            round(cx, 6),
            round(cy, 6),
            round(cz, 6),
            tuple(round(float(e), 6) for e in ext),
        )

    indexed = list(enumerate(fused_dets))
    indexed.sort(key=_order_key)

    touched_by_input_idx: dict[int, int] = {}
    for orig_idx, (det, fused) in indexed:
        obs = colour_obs[orig_idx] if colour_obs is not None else None
        gate = _assoc_gate(det.label, cfg)
        best: InstanceRecord | None = None
        best_dist = gate
        for cand in pool:
            if not labels_compatible(det.label, cand.label):
                continue
            dist = float(np.linalg.norm(fused.centroid - cand.centroid))
            if dist > best_dist:
                continue
            first_box = first_boxes.get(cand.instance_id)
            if not _match_plausible(cand, fused, cfg, dist, first_box=first_box):
                continue
            best = cand
            best_dist = dist
        if best is not None:
            rec = _fused_to_record(det, fused, instance_id=best.instance_id)
            # Issue #191: this detection matched an EXISTING instance -- gate
            # ONLY the n_views increment on whether the pose has actually moved
            # since the last n_views-COUNTED observation of that instance.
            # n_obs (rec.n_obs, still 1 from _fused_to_record) is left exactly
            # alone -- it fuses via _fuse's unconditional `target.n_obs +=
            # other.n_obs`, byte-identical to pre-#191/main. Geometry/score
            # fusion below (merge_into -> _fuse) always runs regardless of
            # either counter.
            counted = _pose_advanced(best.instance_id)
            if not counted:
                rec.n_views = 0
            elif odom is not None:
                last_counted_pose[best.instance_id] = odom
            # Issue #89/#84: trust THIS association's own match decision (alias-bridged
            # label compatibility + gate + extent-plausibility, all already checked
            # above) and fuse directly into `best` — do NOT hand off to index.add(),
            # whose independent label/IoU re-derivation can (and under live pose
            # jitter routinely does) disagree with this decision and silently mint a
            # duplicate instance instead of fusing (see BasicSceneIndex.merge_into's
            # docstring for the full story).
            survivor = index.merge_into(best.instance_id, rec, colour_obs=obs)
            for pool_idx, cand in enumerate(pool):
                if cand.instance_id == survivor.instance_id:
                    pool[pool_idx] = survivor
                    break
        else:
            new_id = index.next_id()
            rec = _fused_to_record(det, fused, instance_id=new_id)
            survivor = index.add(rec, colour_obs=obs)
            pool.append(survivor)
            # Issue #180: only seed a first-box the FIRST time this instance_id
            # is ever seen -- index.add() can still independently IoU-merge
            # into an already-established instance (its own re-derivation,
            # documented on merge_into above) rather than truly minting a new
            # one, and that established track's real first box must not be
            # overwritten by this later observation.
            first_boxes.setdefault(
                survivor.instance_id, (survivor.aabb_min.copy(), survivor.aabb_max.copy())
            )
            # Issue #191: same setdefault-only precedent as first_boxes just
            # above -- seed the pose this instance's first n_views-COUNTED
            # observation (n_views==1, from _fused_to_record) came from,
            # without clobbering an already-established track's real first
            # pose if index.add() actually folded this into one via its own
            # IoU re-derivation instead of truly minting `new_id`.
            if odom is not None:
                last_counted_pose.setdefault(survivor.instance_id, odom)
        touched_by_input_idx[orig_idx] = survivor.instance_id
    return [touched_by_input_idx[i] for i in range(len(fused_dets))]


def decay_singletons(
    index: BasicSceneIndex,
    first_seen: dict[int, int],
    keyframe_idx: int,
    decay_k: int,
) -> list[int]:
    """Prune one-frame ghosts (H15a): remove n_obs==1 instances not re-observed in time.

    An instance whose ``n_obs`` is still 1 ``decay_k`` keyframes after it was first
    seen never got a second observation — a spurious single-frame detection. It is
    removed from the index. Instances with ``n_obs >= 2`` are confirmed tracks and are
    NEVER decayed, regardless of age. ``first_seen`` maps instance_id -> the keyframe
    index at which it was minted; stale entries for removed/absent ids are cleaned up.

    Returns the list of pruned instance_ids. ``decay_k <= 0`` disables decay.
    """
    if decay_k <= 0:
        return []
    live = {r.instance_id: r for r in index.all_instances()}
    pruned: list[int] = []
    for iid in list(first_seen):
        rec = live.get(iid)
        if rec is None:
            del first_seen[iid]  # already gone (merged away / previously pruned)
            continue
        if rec.n_obs >= 2:
            del first_seen[iid]  # confirmed — stop tracking its age, never decays
            continue
        if keyframe_idx - first_seen[iid] >= decay_k:
            if index.remove(iid):
                pruned.append(iid)
            del first_seen[iid]
    return pruned


# --------------------------------------------------------------------------- pipeline


class PerceptionPipeline:
    """Glue: tiling -> detector -> fusion -> tracker, behind a keyframe gate.

    ``process(pano, scan)`` returns the list of instance_ids updated on that frame,
    or an empty list when the keyframe gate skips the frame. Instances accumulate in
    ``self.index`` (a :class:`BasicSceneIndex`).
    """

    def __init__(
        self,
        detector: DetectorFn,
        *,
        index: BasicSceneIndex | None = None,
        fusion_cfg: FusionConfig = DEFAULT_FUSION_CONFIG,
        tracker_cfg: TrackerConfig = DEFAULT_TRACKER_CONFIG,
        keyframe_cfg: KeyframeConfig = DEFAULT_KEYFRAME_CONFIG,
        n_tiles: int = DEFAULT_N_TILES,
        hfov: float = DEFAULT_TILE_HFOV,
        vfov: float = DEFAULT_TILE_VFOV,
    ) -> None:
        self.detector = detector
        self.index = index if index is not None else BasicSceneIndex()
        self.fusion_cfg = fusion_cfg
        self.tracker_cfg = tracker_cfg
        self.keyframe_cfg = keyframe_cfg
        self.n_tiles = n_tiles
        self.hfov = hfov
        self.vfov = vfov
        self._frame_count = 0
        self._last_keyframe: OdomState | None = None
        self._keyframe_idx = 0                    # keyframes processed
        self._det_kf_idx = 0                      # DETECTION-BEARING keyframes (H15a decay clock)
        self._first_seen: dict[int, int] = {}     # instance_id -> det-keyframe it was minted
        # Issues #84/#89 instrumentation: question-clock time (the PanoFrame's own `t`,
        # not wall-clock) of the last periodic instance-index dump (None -> unconditionally
        # dumps once VLA_INSTANCE_DUMP_PATH is set). Keeping this off frame time rather
        # than wall-clock stays deterministic/testable and matches
        # core.heads.explore_debug's throttle style.
        self._last_dump_t: float | None = None

    def _is_keyframe(self, odom: OdomState) -> bool:
        kf = self.keyframe_cfg
        if self._last_keyframe is None:
            return True
        dt_frames = self._frame_count - self._last_frame_idx
        moved = np.hypot(odom.x - self._last_keyframe.x, odom.y - self._last_keyframe.y)
        turned = abs(float(np.arctan2(
            np.sin(odom.yaw - self._last_keyframe.yaw),
            np.cos(odom.yaw - self._last_keyframe.yaw),
        )))
        return (
            dt_frames >= kf.every_k
            or moved >= kf.min_translation
            or turned >= kf.min_rotation
        )

    def process(self, pano: PanoFrame, scan: LidarScan) -> list[int]:
        """Run one frame through the pipeline; return updated instance_ids."""
        odom = pano.odom
        idx_before = self._frame_count
        self._frame_count += 1
        if not self._is_keyframe(odom):
            return []
        self._last_keyframe = odom
        self._last_frame_idx = idx_before

        tiles = project_tiles(pano.image, self.n_tiles, self.hfov, self.vfov)
        per_tile = self.detector(tiles)

        # Issue #84: opt-in raw pre-gate dump. The env lookup is the only cost paid
        # when unset (matches _maybe_dump_instances' cost contract); raw_records stays
        # empty and dump_raw_detections() below no-ops on its own env check anyway, but
        # skipping the list-building here too avoids paying for it at all when off.
        dump_raw = bool(os.environ.get(ENV_RAW_DETECTION_DUMP_PATH))
        raw_records: list[list] = []  # [Detection, gate, instance_id] triples (mutable placeholder)

        fused_dets: list[tuple[Detection, Fused3D]] = []
        # Issue #121: one colour tally per accepted fused detection, parallel to
        # fused_dets (index i's tally belongs to fused_dets[i]) — computed here where
        # the current pano image is in scope, then threaded through associate() into
        # whichever instance each detection lands in.
        colour_obs: list[ColourTally | None] = []
        for tile_dets in per_tile:
            for det in tile_dets:
                # Issue #172 defence-in-depth: the detector must never emit a raw,
                # still-joined caption as a label, but this is the one place every
                # detection (local or remote path) passes through before it can ever
                # become a tracked instance, so guard here too, cheaply, in case a
                # future/other detector path misses the fix.
                if CAPTION_JOIN_MARKER in det.label:
                    _LOGGER.warning(
                        "PerceptionPipeline.process: dropping a detection whose label "
                        "still contains the caption separator %r (label=%r, "
                        "score=%.4f) -- issue #172 guard.",
                        CAPTION_JOIN_MARKER, det.label, det.score,
                    )
                    continue
                fused = fuse_detection(
                    det, scan, odom, self.fusion_cfg,
                    n_tiles=self.n_tiles, hfov=self.hfov, vfov=self.vfov,
                )
                if fused is not None:
                    fused_dets.append((det, fused))
                    colour_obs.append(colour_tally_for_cluster(fused.points, pano))
                    if dump_raw:
                        raw_records.append([det, GATE_ACCEPTED, None])
                elif dump_raw:
                    raw_records.append([det, GATE_NO_LIDAR_CLUSTER, None])

        # Issue #191: thread this frame's pose through so associate() can gate
        # n_obs on distinct viewpoints instead of dwell (see associate()'s
        # `odom` docstring) -- the live/replay path always has a pose here.
        touched = associate(
            fused_dets, self.index, self.tracker_cfg, colour_obs=colour_obs,
            odom=odom, keyframe_cfg=self.keyframe_cfg,
        )

        if dump_raw:
            # Issue #211 counter note: this call writes ONE JSONL ROW PER RAW PER-TILE
            # DETECTION CANDIDATE this single process() call produced (raw_records, built
            # above -- one entry per Detection before/after the lidar-cluster gate), while
            # `self._keyframe_idx` below increments ONCE per process() call that reaches
            # this point (i.e. once per keyframe-gated FRAME). A frame with several boxes
            # across its tiles writes several raw_detections rows for that ONE keyframe
            # increment, so raw_detections having more records than keyframes_processed
            # has increments is the expected shape of two counters at different
            # granularities (frame vs per-detection), not a queue drop -- see
            # AsyncPerceptionWorker's module docstring for the full #211 accounting.
            # Back-fill the instance id each accepted detection landed in: associate()
            # returns `touched` in fused_dets order, and raw_records' GATE_ACCEPTED
            # entries were appended in that exact same order above.
            fused_i = 0
            for rec in raw_records:
                if rec[1] == GATE_ACCEPTED:
                    rec[2] = touched[fused_i]
                    fused_i += 1
            dump_raw_detections(
                [(d, g, iid) for d, g, iid in raw_records],
                keyframe_idx=self._keyframe_idx,
            )

        # H15a: record first sighting for any newly minted instance, then decay
        # one-frame ghosts that never got a second look. The decay clock counts
        # DETECTION-BEARING keyframes only: a keyframe on which the detector saw
        # nothing at all is no evidence against a singleton (cold start, occlusion,
        # scripted single-keyframe labels) — pruning requires decay_k keyframes on
        # which the detector demonstrably produced detections yet never re-observed
        # this instance.
        for iid in touched:
            self._first_seen.setdefault(iid, self._det_kf_idx)
        if fused_dets:
            self._det_kf_idx += 1
        decay_singletons(
            self.index, self._first_seen, self._det_kf_idx, self.tracker_cfg.decay_k
        )
        self._keyframe_idx += 1
        self._maybe_dump_instances(pano.t)
        return touched

    def _maybe_dump_instances(self, t: float) -> None:
        """Issues #84/#89: throttled periodic instance-index dump (opt-in via
        ``VLA_INSTANCE_DUMP_PATH``; no-op — not even the env lookup's cost matters,
        this is one dict-get per keyframe — when unset)."""
        if not os.environ.get(ENV_INSTANCE_DUMP_PATH):
            return
        try:
            interval = float(
                os.environ.get(ENV_INSTANCE_DUMP_INTERVAL_S, DEFAULT_INSTANCE_DUMP_INTERVAL_S)
            )
        except (TypeError, ValueError):
            interval = DEFAULT_INSTANCE_DUMP_INTERVAL_S
        if self._last_dump_t is not None and (t - self._last_dump_t) < interval:
            return
        self._last_dump_t = t
        dump_instance_index(self.index, tag="periodic", keyframes_processed=self._keyframe_idx)

    # convenience for tests / callers
    @property
    def frame_count(self) -> int:
        return self._frame_count

    # track last keyframe frame index for the every_k gate
    _last_frame_idx: int = -(10 ** 9)
