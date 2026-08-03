"""Deterministic spatial predicate + resolution toolbox over ``InstanceRecord``.

This is the answer-path geometry described in ``docs/architecture.md`` §4 and
``docs/proposals/proposal_A.md`` §3: pure, verifiable functions over tracked
instance AABBs/centroids. Every predicate returns a :class:`PredResult` carrying
a boolean, a graded score, a comparison margin, and a human-readable explanation
string that the pre-answer verification checkpoint logs and inspects.

Frames/units: ``map`` frame, metres. Footprint = XY projection of the AABB;
height axis = Z. No network, no wall-clock, no RNG — fully deterministic.

Calibration constants that go beyond values fixed by the spec are collected in
:class:`Thresholds` so they can be tuned offline (see report / calibration
config); the spec-mandated values (near, avoid inflation) are the defaults.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from core.interfaces import InstanceRecord, MatchTier, SceneIndex
from core.plan_schema import Anchor, AvoidSpec, Clause, Pred, TargetSpec
from core.geometry import primitives as P
from core.nav.costmap import VEHICLE_RADIUS_M as _VEHICLE_RADIUS_M
from core.perception.vocab import COLOUR_NEUTRAL, colour_cross_hue, colour_synonyms


# --------------------------------------------------------------------------- config


@dataclass(frozen=True)
class Thresholds:
    """Tunable calibration constants (metres unless noted).

    Spec-fixed defaults; anything not pinned by the spec is flagged in the task
    report as an invented threshold and lives here for offline calibration.
    """

    near_floor: float = 1.2  # near_thresh = max(near_floor, near_scale * diag)
    near_scale: float = 0.6
    next_to_gap: float = 0.75  # <= this AABB gap counts as adjacency
    # on() support semantics (H5/T8-C2/C3, D3): footprint IoM-over-min gate +
    # target bottom inside the supporter's UPPER z-span. The old top-face-only
    # on_vert_tol is gone — a pillow resting among sofa cushions sits 0.2-1.1 m
    # below the AABB top and must still count "on".
    on_min_overlap_frac: float = 0.50  # footprint intersection-over-min gate (was 0.30 over-target)
    on_upper_span_frac: float = 0.25  # upper z-span starts at zmin + this * height
    on_top_tol: float = 0.15  # a.bottom may sit this far above b's AABB top
    in_containment_frac: float = 0.60  # a-footprint fraction inside b  (invented)
    in_vert_slack: float = 0.10  # a within b's z-span, this much slack  (invented)
    # above() lateral-offset form (H5/T8-C4, D4): REPLACES the footprint-overlap
    # gate — wall-hung pictures over a bed have zero footprint overlap. XY centre
    # of the target must fall within the anchor footprint inflated by this margin.
    above_lateral_infl: float = 0.50  # anchor-footprint inflation for the above() lateral gate
    under_iom_min: float = 0.50  # footprint IoM-over-min gate for both under()/below() branches
    under_tuck_tol: float = 0.15  # tuck-under: target.min_z <= anchor.zmin + this
    with_feature_pad: float = 0.30  # "near" pad for possession relaxation rung  (invented)
    avoid_inflate: float = 0.25  # capsule/disc inflation for avoid geometry
    superlative_margin_frac: float = 0.25  # early-answer winner-margin gate
    size_sep_gap: float = 1.20  # size resolver: min largest-face-area ratio for a "small"/"big" extreme
    # Colour salience (issues #11/#12), tuned against the 15-scene GT battery:
    # a cross-hue bridged bin (red->maroon, blue->navy) or a black/white query
    # reaching a mis-binned `gray` bin only counts when the raw colour data clears
    # these cutoffs. Values chosen by principle (below), then validated on the
    # battery (loft black 2/2, home_building_2 red 2/2), so the sweep can retune.
    colour_dominance_floor: float = 0.50  # a cross-hue / luminance-bridged bin counts
    #   only if it is the object's MAJORITY component; rejects hb2 pillow 94's 18%
    #   3rd-bin maroon, admits the 78% maroon of pillows 85/213.
    dark_luma_max: float = 96.0  # Rec.601 luma (0-255): a neutral bin at/below this
    #   reads "black". Admits dark-slate-gray (47,79,79)=69.4, rejects slate-gray
    #   (112,128,144)=125.0 and gray (169,169,169) — the loft black-pillow separation.
    light_luma_min: float = 220.0  # symmetric brightness cutoff for "white" on a
    #   neutral bin. Conservative (no white-query battery evidence); a provision for
    #   the sweep, kept high so mid-grays never read white.
    # Issue #160: ceiling on a contiguous-fragment cluster's merged sorted-axis
    # extent, as a multiple of the class's typical (data-derived) extent -- SAME
    # constant and SAME semantics as core.perception.tracker.TrackerConfig
    # .extent_veto_factor (not a new tunable, just reused across the module
    # boundary: that veto polices same-batch tracker MERGES, this one polices
    # anchor-RESOLUTION-time fragment clustering). A merged footprint under this
    # bound is a plausible reconstruction of one real object (e.g. a pillar's
    # height-sliced fragments); one that exceeds it is a bridged blob spanning
    # more than one real object (e.g. two sofas plus the gap between them) and
    # must not be merged.
    cluster_extent_veto_factor: float = 1.3


DEFAULT_THRESHOLDS = Thresholds()


# --------------------------------------------------------------------------- results


@dataclass(frozen=True)
class PredResult:
    """Outcome of one predicate evaluation.

    passed: hard boolean verdict.
    score: graded [0,1]-ish quality (1 = clearly satisfied), for ranking/audit.
    margin: signed slack in metres (or metric units) to the decision boundary;
            positive = inside the predicate, negative = outside.
    explanation: one-line human-readable justification for the verification log.
    """

    passed: bool
    score: float
    margin: float
    explanation: str

    def __bool__(self) -> bool:  # predicates read naturally in boolean context
        return self.passed


@dataclass(frozen=True)
class Ranked:
    """A superlative ranking result for closest_to / farthest_from."""

    order: list[int]  # instance_ids best-first
    distances: dict[int, float]  # instance_id -> metric distance to anchor
    margin: float  # winner vs runner-up gap (metres); 0 if <2 candidates
    margin_frac: float  # margin / winner-metric (relative gap)
    explanation: str


@dataclass(frozen=True)
class Gate:
    """A corridor gate: the segment between two anchors' nearest faces."""

    p0: np.ndarray  # (2,) XY endpoint on anchor 1's face
    p1: np.ndarray  # (2,) XY endpoint on anchor 2's face
    midpoint: np.ndarray  # (2,) XY gate midpoint (mandatory via-point)
    width: float  # gate length, metres
    #: issue #155: True when the two anchors' AABB footprints already overlap in
    #: the XY plane (:func:`core.geometry.primitives.footprints_overlap`) AND the
    #: resulting ``width`` is narrower than the vehicle can physically fit through
    #: (see :data:`MIN_PASSABLE_GATE_WIDTH_M`) — the anchors are flush/overlapping
    #: furniture with no real navigable gap between them along the connecting
    #: axis, so ``p0``/``p1``/``width`` are the true (degenerate) construction,
    #: not a usable gate. Callers must treat a degenerate gate's threading
    #: requirement as unevaluable rather than record a violation the robot could
    #: never have avoided (see :func:`corridor_gate`).
    degenerate: bool = False


@dataclass(frozen=True)
class Capsule:
    """A stadium (segment + radius) or disc (a==b) avoid region, XY, metres."""

    a: np.ndarray  # (2,) segment start
    b: np.ndarray  # (2,) segment end (== a for a disc)
    radius: float


@dataclass(frozen=True)
class Relaxation:
    """One entry in the resolution fallback audit trail."""

    step: str  # "relax_attributes" | "drop_relation" | "category_only"
    detail: str


def has_unresolved_disambiguator(audit: Sequence["Relaxation"]) -> bool:
    """True if ``audit`` records a disambiguator drop that more perception could fix.

    (#93) A dropped nested disambiguator (e.g. "the table WITH a vase" degrading to
    "any table") widens a hard filter -- fine for :func:`resolve` (some box must be
    published) but dangerous for :func:`counting`, where it can turn "I couldn't
    verify this constraint" into a confident, silently-wrong integer (the vase was
    never detected -> "near the table" admits every table -> full-category count).

    Distinguishes two drop causes recorded under the ``"drop_disambiguator"`` step:

    * the referenced class was never detected, or no candidate satisfied the
      clause -- perception simply hasn't seen it yet; more exploration ticks may
      still resolve it. This is the case callers should treat as "not yet safe to
      commit to".
    * the nesting depth limit (``_MAX_ANCHOR_DEPTH``) was reached -- a structural
      parse-depth cap, not a perception gap. No amount of waiting changes it, so it
      is excluded here: gating on it would only burn exploration budget for no
      chance of resolution.
    """
    return any(
        r.step == "drop_disambiguator" and "nesting depth limit" not in r.detail
        for r in audit
    )


@dataclass(frozen=True)
class ResolveResult:
    """Output of :func:`resolve`.

    candidates_ranked: surviving InstanceRecords, best-first.
    pass_matrix: {instance_id: [per-clause PredResult]} over survivors — feeds
                 the per-clause verification checkpoint.
    margins: {instance_id: superlative margin} (0.0 if no superlative clause).
    audit: ordered relaxation trail (empty if resolved without fallback).
    """

    candidates_ranked: list[InstanceRecord]
    pass_matrix: dict[int, list[PredResult]]
    margins: dict[int, float]
    audit: list[Relaxation] = field(default_factory=list)


# --------------------------------------------------------------------------- helpers


def near_thresh(b: InstanceRecord, th: Thresholds = DEFAULT_THRESHOLDS) -> float:
    """Scale-adaptive proximity radius for anchor b: max(1.2, 0.6*footprint_diag) m."""
    diag = P.footprint_diagonal(b.aabb_min, b.aabb_max)
    return float(max(th.near_floor, th.near_scale * diag))


def _label_text(a: InstanceRecord) -> str:
    """Concatenated searchable text for attribute keyword matching."""
    return " ".join([a.label, a.caption, *a.aliases]).lower()


def _colour_present(a: InstanceRecord, colour: str, th: Thresholds) -> bool:
    """True if a colour-word attribute matches the record.

    When the record carries raw colour bins (:attr:`InstanceRecord.color_bins`,
    populated from GT / quantised perception), matching is bin-aware and applies the
    salience cutoffs the 15-scheme NAME alone cannot express:

    * *identity / same-hue bin* (the scheme name IS the query hue, incl. spelling
      bridges ``grey``->``gray``): always counts, no floor.
    * *cross-hue bridged bin* (``red``->``maroon``, ``blue``->``navy``): counts only
      when it is the object's majority component (fraction >= ``colour_dominance_floor``)
      — rejects a minor off-hue bin (issue #12).
    * *black / white via luminance*: a neutral (``gray``/``black``/``white``) bin
      counts as ``black`` when its RGB luma <= ``dark_luma_max`` (``white`` when >=
      ``light_luma_min``), still dominance-gated — separates a near-black neutral-binned
      object from lighter ones (issue #11) without a blanket black<->gray name merge.
      (The cross-name case is inert in practice: a same-named bin — e.g. a ``black``
      bin queried for ``black`` — hits the identity branch above first.)

    When the record has NO colour bins (mocks, perception without quantisation), it
    falls back to the legacy scheme-name substring test over the record text, so
    existing behaviour is preserved.
    """
    colour = colour.lower()
    schemes = colour_synonyms(colour)
    if not schemes:
        return False  # unknown colour word: match nothing rather than guess
    bins = a.color_bins
    if not bins:
        return any(name in _label_text(a) for name in schemes)
    cross = colour_cross_hue(colour)
    is_black, is_white = colour == "black", colour == "white"
    floor = th.colour_dominance_floor
    for b in bins:
        name = b.name
        # identity / same-hue scheme bin (incl. spelling bridges): always counts
        if name in schemes and name not in cross:
            return True
        # cross-hue bridged bin: only a dominant component counts
        if name in cross and b.fraction >= floor:
            return True
        # black/white via luminance on a neutral (gray) bin, dominance-gated
        if name in COLOUR_NEUTRAL and b.fraction >= floor:
            if is_black and b.luma <= th.dark_luma_max:
                return True
            if is_white and b.luma >= th.light_luma_min:
                return True
    return False


def _attr_present(a: InstanceRecord, attr: str, th: Thresholds = DEFAULT_THRESHOLDS) -> bool:
    """True if a single requested attribute keyword matches the record.

    Colour attributes go through :func:`_colour_present` (bin-aware salience, with a
    scheme-name text fallback for records without colour bins). Non-colour attributes
    keep the plain substring test over the record's searchable text.
    """
    attr = attr.lower()
    if colour_synonyms(attr):
        return _colour_present(a, attr, th)
    return attr in _label_text(a)


# Size qualifiers handled by the relative per-class resolver (DD-A12 = T8-C6).
# "smallest"/"largest" are the argmin/argmax forms; "small"/"big"/"large" the
# comparative forms — both rank on largest-face area within the same-class pool.
_SIZE_LARGE: frozenset[str] = frozenset({"big", "large", "largest", "biggest"})
_SIZE_SMALL: frozenset[str] = frozenset({"small", "smallest", "little", "tiny"})
_SIZE_ATTRS: frozenset[str] = _SIZE_LARGE | _SIZE_SMALL


def _size_attr_match(
    a: InstanceRecord,
    size_attr: str,
    pool: Sequence[InstanceRecord],
    th: Thresholds,
) -> bool:
    """Relative per-class size match (DD-A12): largest-face-area ranking + 1.2x gap.

    ``a`` matches a "big"/"large"/"largest" attribute iff it is the largest-face
    extreme of ``pool`` AND its largest-face area is at least ``size_sep_gap`` (1.2x)
    times the next candidate's; symmetrically for "small"/"smallest" against the
    smallest extreme. When no extreme is separated by the gap the size attribute
    matches NOTHING (honest none) — a size qualifier is only asserted when the
    generator would have (it assigns "small"/"big" only at a >=1.2x separation).

    With fewer than two candidates the ranking is undefined; a lone candidate cannot
    be "the small(est)/big(gest)" relative to nothing, so it does not match.
    """
    if size_attr not in _SIZE_ATTRS or len(pool) < 2:
        return False
    areas = sorted(
        (P.largest_face_area(c.aabb_min, c.aabb_max) for c in pool), reverse=True
    )
    a_area = P.largest_face_area(a.aabb_min, a.aabb_max)
    if size_attr in _SIZE_LARGE:
        top, runner = areas[0], areas[1]
        separated = top >= th.size_sep_gap * runner if runner > P.EPS else top > P.EPS
        return bool(separated and a_area >= top - P.EPS)
    # small end: smallest must be <= runner-up / gap (i.e. runner >= gap * smallest)
    smallest, runner = areas[-1], areas[-2]
    separated = runner >= th.size_sep_gap * smallest if smallest > P.EPS else runner > P.EPS
    return bool(separated and a_area <= smallest + P.EPS)


def _attrs_match(
    a: InstanceRecord,
    attributes: Sequence[str],
    pool: Sequence[InstanceRecord] | None = None,
    th: Thresholds = DEFAULT_THRESHOLDS,
) -> bool:
    """True if every requested attribute matches the record.

    Non-size attributes go through the text/colour-bridge test. A size qualifier
    ("small"/"big"/"largest"/...) is resolved RELATIVELY against ``pool`` (the
    same-class candidate set) via :func:`_size_attr_match` — largest-face ranking
    with the 1.2x separation gap (DD-A12). When ``pool`` is None (no same-class
    context, e.g. a single-candidate disambiguator check) a size attribute cannot be
    ranked and does not match, so callers with a pool must pass it.
    """
    if not attributes:
        return True
    for attr in attributes:
        low = attr.lower()
        if low in _SIZE_ATTRS:
            if pool is None or not _size_attr_match(a, low, pool, th):
                return False
        elif not _attr_present(a, low, th):
            return False
    return True


# Anchor classes that have an "under-space" a target can tuck into (VLA-3D
# `special_relation_classes.UNDER_RELATION`, verbatim from docs/prior_art/vla_3d.md).
# A stool tucked under a table sits at floor level with its top below the table's
# AABB top, so branch (ii) of under()/below() fires only when the ANCHOR is one of
# these. Stored space-normalised ("night stand" and "night_stand" both match).
UNDER_RELATION: frozenset[str] = frozenset(
    {
        "cabinet", "counter", "table", "desk", "stool", "shelf", "drawer",
        "dresser", "bed", "bookshelf", "tv stand", "bench", "chest",
        "piano bench", "bar", "night stand", "coffee table",
    }
)


def _under_relation_anchor(b: InstanceRecord) -> bool:
    """True if b's class is a VLA-3D UNDER_RELATION class (underscore/space tolerant)."""
    lbl = b.label.lower().replace("_", " ").strip()
    return lbl in UNDER_RELATION


# --------------------------------------------------------------------------- predicates


def on(a: InstanceRecord, b: InstanceRecord, th: Thresholds = DEFAULT_THRESHOLDS) -> PredResult:
    """a is supported by b (support semantics, H5/T8-C2/C3, D3).

    Three gates, matching the VLA-3D generation form as reconciled by the T8
    instance-level evidence:

    * footprint IoM-over-min >= ``on_min_overlap_frac`` (a small pillow fully on a
      big sofa scores 1.0);
    * anchor-larger gate: b's footprint area strictly exceeds a's (a sofa cannot be
      "on" a cushion);
    * vertical: a's bottom lies in the supporter's UPPER z-span,
      ``[b.zmin + on_upper_span_frac * b_height, b.ztop + on_top_tol]`` — so a
      pillow resting among sofa cushions (bottom 0.2-1.1 m below the AABB top, where
      the backrest is) still counts, while an object sitting near b's floor does not.
    """
    a_bottom = float(a.aabb_min[2])
    b_zmin = float(b.aabb_min[2])
    b_top = float(b.aabb_max[2])
    b_height = max(b_top - b_zmin, 0.0)
    band_lo = b_zmin + th.on_upper_span_frac * b_height
    band_hi = b_top + th.on_top_tol
    vert_ok = (a_bottom >= band_lo - P.EPS) and (a_bottom <= band_hi + P.EPS)

    frac = P.footprint_iom(a.aabb_min, a.aabb_max, b.aabb_min, b.aabb_max)
    over_ok = frac >= th.on_min_overlap_frac
    a_fp = P.footprint_area(a.aabb_min, a.aabb_max)
    b_fp = P.footprint_area(b.aabb_min, b.aabb_max)
    anchor_larger = b_fp > a_fp + P.EPS

    passed = bool(vert_ok and over_ok and anchor_larger)
    # signed slack to the nearest z-band edge (positive = inside the band)
    vmargin = min(a_bottom - band_lo, band_hi - a_bottom)
    # score (soft ranking signal, NOT the hard pass/fail above) is gated on vertical
    # alignment only, not `anchor_larger` (issue #59): a wide-canopy plant sitting
    # squarely atop a small side table (strong footprint overlap, correct height)
    # should still rank as a plausible "on" match over an unrelated candidate even
    # though the coarse anchor-size sanity check keeps `passed` False — the size
    # gate exists to block hard misclassifications ("a sofa on a cushion"), not to
    # zero out every near-miss's ranking signal (see resolve()'s category-only
    # fallback, which ranks by this score when no candidate can pass as a hard
    # filter).
    score = float(max(0.0, min(1.0, frac)) * (1.0 if vert_ok else 0.0))
    expl = (
        f"on: bottom {a_bottom:.2f} in upper z-band [{band_lo:.2f}, {band_hi:.2f}] "
        f"-> {'ok' if vert_ok else 'FAIL'}; footprint IoM {frac*100:.0f}% "
        f"(>= {th.on_min_overlap_frac*100:.0f}% -> {'ok' if over_ok else 'FAIL'}); "
        f"anchor larger ({b_fp:.2f} > {a_fp:.2f} m2 -> {'ok' if anchor_larger else 'FAIL'})"
    )
    return PredResult(passed, score, float(vmargin), expl)


def in_(a: InstanceRecord, b: InstanceRecord, th: Thresholds = DEFAULT_THRESHOLDS) -> PredResult:
    """a inside b (room/container): a's footprint mostly within b AND within b's z-span."""
    area = P.footprint_overlap_area(a.aabb_min, a.aabb_max, b.aabb_min, b.aabb_max)
    a_area = max(P.footprint_overlap_area(a.aabb_min, a.aabb_max, a.aabb_min, a.aabb_max), P.EPS)
    frac = area / a_area
    horiz_ok = frac >= th.in_containment_frac
    vert_ok = (
        float(a.aabb_min[2]) >= float(b.aabb_min[2]) - th.in_vert_slack - P.EPS
        and float(a.aabb_max[2]) <= float(b.aabb_max[2]) + th.in_vert_slack + P.EPS
    )
    passed = bool(horiz_ok and vert_ok)
    score = float(max(0.0, min(1.0, frac)) * (1.0 if vert_ok else 0.0))
    expl = (
        f"in: {frac*100:.0f}% of a's footprint inside b (>= "
        f"{th.in_containment_frac*100:.0f}% -> {'ok' if horiz_ok else 'FAIL'}); "
        f"vertical containment {'ok' if vert_ok else 'FAIL'}"
    )
    return PredResult(passed, score, float(frac - th.in_containment_frac), expl)


def near(a: InstanceRecord, b: InstanceRecord, th: Thresholds = DEFAULT_THRESHOLDS) -> PredResult:
    """a near b within scale-adaptive threshold max(1.2, 0.6*footprint_diag(b)) m."""
    thr = near_thresh(b, th)
    gap = P.aabb_gap(a.aabb_min, a.aabb_max, b.aabb_min, b.aabb_max)
    passed = bool(gap <= thr + P.EPS)
    margin = thr - gap
    score = float(max(0.0, min(1.0, 1.0 - gap / thr))) if thr > 0 else 0.0
    expl = f"near: AABB gap {gap:.2f}m <= adaptive thresh {thr:.2f}m -> {'ok' if passed else 'FAIL'}"
    return PredResult(passed, score, float(margin), expl)


def next_to(a: InstanceRecord, b: InstanceRecord, th: Thresholds = DEFAULT_THRESHOLDS) -> PredResult:
    """a tightly adjacent to b: AABB gap <= 0.75 m (tighter than near).

    DD-A5: "next to"/"beside"/"adjacent to"/"close to" are SYNONYMS of ``near`` in
    the VLA-3D generation templates, and questions phrased with a near-synonym were
    generated with the ``near`` threshold — so this tighter predicate is NOT the one
    the parser routes those phrasings to (see :data:`_BINARY_PREDS`, which maps
    ``Pred.NEXT_TO`` to :func:`near`). It is kept available under this name for any
    future parser-level distinction, but nothing routes to it today. The tight-gap
    behaviour lives here so a caller that genuinely wants strict adjacency can still
    call it explicitly.
    """
    gap = P.aabb_gap(a.aabb_min, a.aabb_max, b.aabb_min, b.aabb_max)
    passed = bool(gap <= th.next_to_gap + P.EPS)
    margin = th.next_to_gap - gap
    score = float(max(0.0, min(1.0, 1.0 - gap / th.next_to_gap))) if th.next_to_gap > 0 else 0.0
    expl = f"next_to: AABB gap {gap:.2f}m <= {th.next_to_gap}m -> {'ok' if passed else 'FAIL'}"
    return PredResult(passed, score, float(margin), expl)


def between(
    a: InstanceRecord,
    b1: InstanceRecord,
    b2: InstanceRecord,
    th: Thresholds = DEFAULT_THRESHOLDS,
) -> PredResult:
    """a between b1 and b2: a's centroid within capsule(seg b1-b2, r=max anchor half-width)."""
    radius = max(
        P.footprint_half_width(b1.aabb_min, b1.aabb_max),
        P.footprint_half_width(b2.aabb_min, b2.aabb_max),
    )
    dist, t = P.point_to_segment_2d(a.centroid, b1.centroid, b2.centroid)
    # Strict betweenness (H5/T8-C5): the projection must land in the OPEN interval
    # 0 < t < 1. A target sitting beside one anchor projects to a clamped t of 0 or
    # 1 (off the segment end) and must fail even when it is within the capsule
    # radius — "between" is exclusive of the anchor positions themselves.
    strict_t = P.EPS < t < 1.0 - P.EPS
    passed = bool(dist <= radius + P.EPS and strict_t)
    margin = radius - dist
    score = (
        float(max(0.0, min(1.0, 1.0 - dist / radius))) if (radius > 0 and strict_t) else 0.0
    )
    expl = (
        f"between: centroid {dist:.2f}m from b1-b2 segment (t={t:.2f}, "
        f"strict 0<t<1 -> {'ok' if strict_t else 'FAIL'}) <= capsule "
        f"radius {radius:.2f}m -> {'ok' if passed else 'FAIL'}"
    )
    return PredResult(passed, score, float(margin), expl)


def above(a: InstanceRecord, b: InstanceRecord, th: Thresholds = DEFAULT_THRESHOLDS) -> PredResult:
    """a above b: lateral-offset tolerance + positive vertical gap (H5/T8-C4, D4).

    The old footprint-overlap requirement is REPLACED (not gated) by a lateral
    tolerance: the XY centre of ``a`` must fall within ``b``'s footprint inflated by
    ``above_lateral_infl``. Wall-hung pictures "above the bed" have zero footprint
    overlap yet a small lateral offset, so the overlap gate rejected exactly the
    cases the questions ask about; adding an IoM gate would make it worse. The
    positive-gap requirement (a's bottom strictly above b's top) is kept.

    (The former ``above_gap_max`` cap was declared but never consumed in the body;
    it is dropped rather than wired — an upper bound on the vertical gap has no
    generation-spec counterpart and would spuriously reject a high picture over a
    low headboard. See docs/redteam/dossier_deltas.md A8 / hardening_backlog.md H5.)
    """
    c = P._as3(a.centroid)[:2]
    lo = P.footprint_min(b.aabb_min, b.aabb_max) - th.above_lateral_infl
    hi = P.footprint_max(b.aabb_min, b.aabb_max) + th.above_lateral_infl
    lat_ok = bool(np.all(c >= lo - P.EPS) and np.all(c <= hi + P.EPS))
    # lateral slack: signed distance from a's centre to the inflated footprint edge
    d = np.maximum.reduce([lo - c, c - hi, np.zeros(2)])
    lat_dist = float(np.linalg.norm(d))
    gap = float(a.aabb_min[2]) - float(b.aabb_max[2])  # >0 => a strictly above
    vert_ok = gap > P.EPS
    passed = bool(lat_ok and vert_ok)
    score = 1.0 if passed else 0.0
    expl = (
        f"above: lateral {'inside' if lat_ok else f'{lat_dist:.2f}m outside'} "
        f"inflated footprint (infl {th.above_lateral_infl}m -> "
        f"{'ok' if lat_ok else 'FAIL'}); vertical gap {gap:+.2f}m "
        f"(a above b -> {'ok' if vert_ok else 'FAIL'})"
    )
    return PredResult(passed, score, float(gap), expl)


def under(a: InstanceRecord, b: InstanceRecord, th: Thresholds = DEFAULT_THRESHOLDS) -> PredResult:
    """a under/below b: three-branch form (H5/DD-A7 + the below/above lateral mirror).

    Branches (i) and (ii) require footprint IoM-over-min >= ``under_iom_min``. Then:

    * (i) strict below: a's top <= b's bottom + ``under_tuck_tol`` — a rug under a
      table top, an object on a lower shelf below an upper one; OR
    * (ii) tuck-under: gated to ANCHOR classes with an under-space
      (:data:`UNDER_RELATION`): a rests at floor level relative to b
      (``a.min_z <= b.min_z + under_tuck_tol``) AND a's top is below b's AABB top
      (``a.max_z <= b.max_z``). This is the ONLY way "the stool under the table"
      resolves — the stool's top rises above the table's AABB min_z (~floor), so
      the strict branch can never pass it; OR
    * (iii) wall-relative below (lateral-offset, inverse of :func:`above`):
      ``below(a, b) == above(b, a)`` — b's XY centre lies within a's inflated
      footprint AND b is strictly above a (``b.min_z > a.max_z``). This is the
      vertical mirror of the accepted D4/H5 ``above()`` lateral form and carries NO
      footprint-IoM requirement: a sofa "below a window" (or under wall-hung
      pictures) has ~0% footprint overlap with the wall-mounted anchor, so the
      IoM-gated branches (i)/(ii) are blind to exactly the case the questions ask
      about — the same wall-hung blindness D4 removed from ``above()``. Because it
      fires only when the anchor is STRICTLY above the target, it never collides
      with tuck-under (a table extends to the floor, so it is not strictly above a
      stool it shelters).
    """
    a_top = float(a.aabb_max[2])
    a_bottom = float(a.aabb_min[2])
    b_zmin = float(b.aabb_min[2])
    b_top = float(b.aabb_max[2])

    frac = P.footprint_iom(a.aabb_min, a.aabb_max, b.aabb_min, b.aabb_max)
    over_ok = frac >= th.under_iom_min

    strict_ok = a_top <= b_zmin + th.under_tuck_tol + P.EPS
    tuck_gated = _under_relation_anchor(b)
    tuck_ok = tuck_gated and (
        a_bottom <= b_zmin + th.under_tuck_tol + P.EPS and a_top <= b_top + P.EPS
    )
    iom_branch = over_ok and (strict_ok or tuck_ok)

    # branch (iii): wall-relative below == above(anchor, target), no IoM gate.
    lateral_ok = above(b, a, th).passed

    passed = bool(iom_branch or lateral_ok)
    if iom_branch:
        which = "strict" if strict_ok else "tuck-under"
    elif lateral_ok:
        which = "lateral (inverse-above)"
    else:
        which = "neither"
    gap = b_zmin - a_top  # >0 => a strictly under b's bottom (branch i slack)
    score = 1.0 if lateral_ok and not iom_branch else (
        float(max(0.0, min(1.0, frac))) if passed else 0.0
    )
    expl = (
        f"under: footprint IoM {frac*100:.0f}% (>= {th.under_iom_min*100:.0f}% -> "
        f"{'ok' if over_ok else 'FAIL'}); branch={which} "
        f"(strict gap {gap:+.2f}m; anchor '{b.label}' "
        f"{'in' if tuck_gated else 'not in'} UNDER_RELATION; lateral "
        f"{'ok' if lateral_ok else 'FAIL'}) -> {'ok' if passed else 'FAIL'}"
    )
    return PredResult(passed, score, float(gap), expl)


def with_feature(
    a: InstanceRecord,
    b: InstanceRecord,
    th: Thresholds = DEFAULT_THRESHOLDS,
    *,
    allow_pad_rung: bool = True,
) -> PredResult:
    """a possesses feature b ("a with the b on it") == ``on(b, a)`` (H5/DD-A6).

    The VLA-3D generator has no "with" relation: "the table with the elephant
    figurine on it" is the INVERSE of ``on`` — the figurine is ON the table. So the
    primary test is ``on(b, a)`` with the new support semantics. When that fails, an
    explicitly-audited relaxation rung falls back to the old footprint-pad test
    (b's centroid inside/near a's XY footprint, z ignored) — this is a looser
    "contents-ish" match kept only so a mis-heighted detection still links contents
    to their container; the PredResult explanation names which rung fired so the
    verification checkpoint can see a relaxed match. Set ``allow_pad_rung=False`` to
    require the strict inverse-on form.
    """
    primary = on(b, a, th)
    if primary.passed or not allow_pad_rung:
        expl = f"with (== on(feature, a)): {primary.explanation}"
        return PredResult(primary.passed, primary.score, primary.margin, expl)

    # relaxation rung: footprint-pad possession (z ignored) — audited as relaxed.
    c = P._as3(b.centroid)[:2]
    lo = P.footprint_min(a.aabb_min, a.aabb_max)
    hi = P.footprint_max(a.aabb_min, a.aabb_max)
    inside = bool(np.all(c >= lo - P.EPS) and np.all(c <= hi + P.EPS))
    d = np.maximum.reduce([lo - c, c - hi, np.zeros(2)])
    dist = float(np.linalg.norm(d))
    passed = bool(inside or dist <= th.with_feature_pad + P.EPS)
    margin = th.with_feature_pad - dist
    score = (
        0.5
        if inside
        else float(max(0.0, min(0.5, 0.5 * (1.0 - dist / th.with_feature_pad))))
    )  # capped below any strict-on score so inverse-on always ranks first
    expl = (
        f"with (RELAXED footprint-pad rung; strict on(feature,a) failed): feature "
        f"centroid {'inside' if inside else f'{dist:.2f}m from'} a's footprint "
        f"(pad {th.with_feature_pad}m -> {'ok' if passed else 'FAIL'})"
    )
    return PredResult(passed, score, float(margin), expl)


def _centroid_dist(a: InstanceRecord, anchor: InstanceRecord) -> float:
    """Centroid-to-centroid 3D distance, metres."""
    return float(np.linalg.norm(P._as3(a.centroid) - P._as3(anchor.centroid)))


def closest_to(
    candidates: Sequence[InstanceRecord],
    anchor: InstanceRecord,
    th: Thresholds = DEFAULT_THRESHOLDS,
) -> Ranked:
    """Rank candidates by ascending centroid distance to anchor; return full ranking."""
    return _rank(candidates, anchor, farthest=False)


def farthest_from(
    candidates: Sequence[InstanceRecord],
    anchor: InstanceRecord,
    th: Thresholds = DEFAULT_THRESHOLDS,
) -> Ranked:
    """Rank candidates by descending centroid distance to anchor; return full ranking."""
    return _rank(candidates, anchor, farthest=True)


def _rank(candidates: Sequence[InstanceRecord], anchor: InstanceRecord, farthest: bool) -> Ranked:
    dists = {c.instance_id: _centroid_dist(c, anchor) for c in candidates}
    # deterministic tie-break on instance_id
    order = sorted(dists, key=lambda i: (dists[i], i), reverse=farthest)
    margin = 0.0
    margin_frac = 0.0
    if len(order) >= 2:
        d0, d1 = dists[order[0]], dists[order[1]]
        margin = abs(d1 - d0)
        winner = d0 if not farthest else max(d0, P.EPS)
        margin_frac = margin / winner if winner > P.EPS else 0.0
    kind = "farthest_from" if farthest else "closest_to"
    expl = (
        f"{kind}: {len(order)} candidates; winner id={order[0] if order else None} at "
        f"{dists[order[0]]:.2f}m; margin {margin:.2f}m ({margin_frac*100:.0f}%)"
        if order
        else f"{kind}: no candidates"
    )
    return Ranked(order, dists, float(margin), float(margin_frac), expl)


# clause-pred -> binary predicate function (BETWEEN handled separately).
# DD-A5: Pred.NEXT_TO routes to `near`, not the tight `next_to` — near-synonyms
# ("next to", "beside", "adjacent to", "close to") were generated with the `near`
# threshold, so a tighter 0.75 m gate would reject true targets. The tight
# `next_to` stays defined but nothing routes to it.
_BINARY_PREDS = {
    Pred.ON: on,
    Pred.IN: in_,
    Pred.NEAR: near,
    Pred.NEXT_TO: near,
    Pred.ABOVE: above,
    Pred.UNDER: under,
    Pred.WITH: with_feature,
}
_SUPERLATIVE_PREDS = {Pred.CLOSEST_TO, Pred.FARTHEST_FROM}


# --------------------------------------------------------------------------- resolution


def _match_noun(index: SceneIndex, noun: str) -> list[InstanceRecord]:
    """Typo/synonym-tolerant category lookup via the index."""
    return list(index.by_label(noun))


def _match_anchor_noun(index: SceneIndex, noun: str) -> list[InstanceRecord]:
    """Anchor-noun lookup: MODIFIED anchors keep only the STRONGEST non-empty match
    tier; BARE-noun anchors merge all tiers (see #21 note at the end).

    A disambiguator/relation anchor names a specific referent ("the table WITH the
    *horse figurine* on it", "the potted plant ON the *dressing table*"). The index's
    label matcher deliberately pools head-noun cousins so a bare-noun query still
    generalises ("figurine" -> every X figurine, "table" -> every X table). But when a
    *modified* anchor noun has an exact/synonym referent, the differently-modified
    cousins it also drags in ("elephant figurine", "side table") are the WRONG object:
    an existential relation clause then passes for a same-noun target sitting by the
    cousin, and the true target (by the exact referent) no longer uniquely survives —
    ranking collapses to instance-id order and an earlier-leg distractor wins (#13).

    So for anchor resolution we honour the index's own tier ranking: take the best
    (lowest :class:`MatchTier`) tier that produced any hit and drop the weaker tiers.
    This keeps the bare-noun generalisation intact (a bare query's cousins live in the
    same head-noun tier and are all kept) while stopping a specific anchor from being
    satisfied by a head-noun cousin when its exact referent is present. Indexes that
    don't expose :meth:`by_label_tiered` (minimal test doubles) fall back to the flat
    :meth:`by_label` result unchanged.
    """
    tiered = getattr(index, "by_label_tiered", None)
    if tiered is None:
        return list(index.by_label(noun))
    hits = list(tiered(noun))
    if not hits:
        return []
    # Bare-noun anchors are exempt from tier discipline (#21). Detected with the same
    # machinery the index tiering uses: a query whose normalised form equals its own
    # head noun carries no modifiers, so a bare "table" legitimately refers to ANY
    # table — its exact/synonym referent and its head-noun cousins ("coffee table")
    # are all the SAME class word and must all be admitted (pre-#13 flat behaviour).
    # Narrowing to the strongest tier would drop supporters on the cousins.
    from core.perception.scene_index import normalize_label
    from core.perception.vocab import head_noun

    query = normalize_label(noun)
    if query == head_noun(query):
        return [rec for rec, _ in hits]

    # Modified anchor ("horse figurine", "dressing table"): keep the #13 behaviour —
    # honour the index's tier ranking and take the best (lowest MatchTier) non-empty
    # tier only, so a differently-modified cousin cannot satisfy a specific anchor
    # when its exact referent is present.
    best_tier = min(tier for _, tier in hits)
    return [rec for rec, tier in hits if tier == best_tier]


# Recursion guard for nested-disambiguator resolution: an anchor's disambiguator
# may itself reference an anchored clause, so cap the nesting depth defensively.
_MAX_ANCHOR_DEPTH: int = 4


def _audit_add(audit: list[Relaxation] | None, step: str, detail: str) -> None:
    """Append a Relaxation, de-duplicating identical entries.

    Anchor resolution runs once per candidate, so a single disambiguator drop
    would otherwise be logged once per candidate; the drop is a property of the
    anchor set, not the candidate, so collapse repeats to one honest entry.
    """
    if audit is None:
        return
    entry = Relaxation(step, detail)
    if entry not in audit:
        audit.append(entry)


def _merge_fragment_footprint(members: Sequence[InstanceRecord]) -> InstanceRecord:
    """Build a synthetic candidate spanning ``members``' union AABB (RESOLUTION ONLY).

    Never inserted into any index -- the caller uses it as one anchor candidate
    among others and discards it once resolution picks a winner. ``instance_id``
    is the smallest member id (already a real, globally-unique tracker id that no
    other live candidate in this anchor's pool can collide with, since every
    member it was drawn from is removed from the pool in its place); ``score`` and
    ``n_obs`` accumulate (max confidence observed, total distinct sightings across
    the fragments); ``label``/``caption``/``aliases``/``color_bins`` are copied from
    the largest-footprint member as the most representative single fragment.
    """
    lo = np.minimum.reduce([m.aabb_min for m in members])
    hi = np.maximum.reduce([m.aabb_max for m in members])
    rep = max(members, key=lambda m: P.footprint_area(m.aabb_min, m.aabb_max))
    return InstanceRecord(
        instance_id=min(m.instance_id for m in members),
        label=rep.label,
        score=max(m.score for m in members),
        n_obs=sum(m.n_obs for m in members),
        centroid=(lo + hi) / 2.0,
        aabb_min=lo,
        aabb_max=hi,
        caption=rep.caption,
        aliases=rep.aliases,
        color_bins=rep.color_bins,
    )


def _cluster_plausible(merged: InstanceRecord, th: Thresholds) -> bool:
    """True if ``merged``'s sorted-axis extent is within a plausible size for its
    class (fail-open: no dimension prior for the class -> can't judge -> allow).

    Mirrors ``core.perception.tracker._match_plausible``'s per-axis sorted-extent
    comparison exactly (same veto value, see :attr:`Thresholds
    .cluster_extent_veto_factor`), applied here to a candidate CLUSTER footprint
    instead of a tracker MERGE.
    """
    from core.perception.dimension_priors import prior_for

    prior = prior_for(merged.label)
    if prior is None:
        return True
    ext = np.sort(merged.extents)
    typ = np.sort(prior.typ_ext)
    return bool(np.all(ext <= th.cluster_extent_veto_factor * typ))


def _cluster_same_label(
    indexed_members: Sequence[tuple[int, InstanceRecord]], th: Thresholds
) -> list[tuple[int, InstanceRecord]]:
    """Union-find one label group's fragments by XY footprint contiguity (#160),
    returning ``(sort_key, record)`` pairs: one merged candidate per plausible
    component, the untouched original for singletons or implausible (bridged-
    blob) components. ``sort_key`` is the lowest original ``cands`` index feeding
    that output record, so :func:`_cluster_anchor_candidates` can restore a
    stable, first-occurrence order across label groups.
    """
    n = len(indexed_members)
    if n < 2:
        return list(indexed_members)

    members = [r for _, r in indexed_members]
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for i in range(n):
        for j in range(i + 1, n):
            if P.footprints_overlap(
                members[i].aabb_min, members[i].aabb_max,
                members[j].aabb_min, members[j].aabb_max,
            ):
                union(i, j)

    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    out: list[tuple[int, InstanceRecord]] = []
    for idxs in groups.values():
        orig_idxs = [indexed_members[k][0] for k in idxs]
        first = min(orig_idxs)
        if len(idxs) == 1:
            out.append((first, members[idxs[0]]))
            continue
        merged = _merge_fragment_footprint([members[k] for k in idxs])
        if _cluster_plausible(merged, th):
            out.append((first, merged))
        else:
            # #160: adjacency chained past a plausible single-object footprint --
            # a bridged blob, not a reconstruction. Leave the fragments unmerged
            # rather than hand resolution an impossible anchor.
            out.extend((indexed_members[k][0], members[k]) for k in idxs)
    return out


def _cluster_anchor_candidates(
    cands: Sequence[InstanceRecord], th: Thresholds
) -> list[InstanceRecord]:
    """#160: cluster contiguous same-canonical-class fragments into one candidate
    footprint, for anchor RESOLUTION ONLY -- never mutates the scene index.

    A live-quality index tracks a real fragmented object (a pillar sliced by
    height, a sofa split across the frustum) as several small same-label
    instances whose union is the real object's footprint but no single instance
    covers it. Containment/support predicates (``on``/``above``) and multi-anchor
    predicates (``between``) evaluated against any ONE fragment therefore see the
    wrong-sized anchor. Grouping happens strictly within each distinct ``label``
    (never across classes) and only merges components whose union footprint
    clears :func:`_cluster_plausible`, so an adjacency chain that has bridged past
    two real, distinct objects (or the gap between them) is left unclustered
    rather than handed to resolution as one implausible blob.
    """
    if len(cands) < 2:
        return list(cands)
    by_label: dict[str, list[tuple[int, InstanceRecord]]] = {}
    for i, c in enumerate(cands):
        by_label.setdefault(c.label, []).append((i, c))

    out: list[tuple[int, InstanceRecord]] = []
    for group in by_label.values():
        out.extend(_cluster_same_label(group, th))
    out.sort(key=lambda t: t[0])
    return [r for _, r in out]


def _resolve_anchor(
    anchor: Anchor,
    index: SceneIndex,
    th: Thresholds,
    audit: list[Relaxation] | None = None,
    _depth: int = 0,
) -> list[InstanceRecord]:
    """Resolve an anchor to matching records (noun + attributes + nested disambiguator).

    Base pool = noun (typo-tolerant) filtered by the anchor's own attributes, then
    clustered (#160): contiguous same-label fragments merge into one candidate
    footprint for resolution only (see :func:`_cluster_anchor_candidates`; the
    scene index itself is never touched). When the anchor carries a nested
    disambiguator clause it is bound here (this is the dominant multi-constraint
    object-reference form, e.g. "the bowl on the table CLOSEST TO the screen"):

    * non-superlative disambiguator -> keep only anchor candidates for which the
      clause holds (anchor as subject);
    * superlative disambiguator -> rank the anchor candidates by it and keep the
      argmin/argmax only.

    A disambiguator that cannot be applied (its own anchor noun has no instances,
    or filtering would empty the pool) is dropped and recorded in ``audit`` as a
    ``Relaxation`` so the drop is never silent. ``_depth`` guards against a
    disambiguator that (transitively) references another anchored clause.
    """
    cands = _match_anchor_noun(index, anchor.noun)
    if anchor.attributes:
        _class_pool = cands  # same-class pool for relative size ranking (DD-A12)
        cands = [c for c in cands if _attrs_match(c, anchor.attributes, _class_pool, th)]
    cands = _cluster_anchor_candidates(cands, th)

    disamb = anchor.disambiguator
    if disamb is None or not cands or _depth >= _MAX_ANCHOR_DEPTH:
        if disamb is not None and _depth >= _MAX_ANCHOR_DEPTH:
            _audit_add(
                audit,
                "drop_disambiguator",
                f"anchor '{anchor.noun}' disambiguator dropped: nesting depth "
                f"limit ({_MAX_ANCHOR_DEPTH}) reached",
            )
        return cands

    narrowed = _apply_disambiguator(disamb, cands, index, th, audit, _depth)
    return narrowed


def _select_sub_anchor(
    sub_anchor_recs: Sequence[InstanceRecord],
    cands: Sequence[InstanceRecord],
) -> InstanceRecord:
    """Pick one instance from a resolved sub-anchor to serve as a disambiguator's
    reference point, when the sub-anchor noun resolves to more than one instance.

    A disambiguator's own anchor (e.g. "map wall decal" in "the table closest to
    the map wall decal") can legitimately resolve to several duplicate/near-
    duplicate detections when the fused map hasn't merged them into one track.
    Picking ``sub_anchor_recs[0]`` is an annotation/detection-order artifact
    (#151): plain list position carries no physical meaning, so the
    disambiguator's answer — and everything downstream of it — ends up
    depending on shuffle order of the banked instance list. #111 established
    that merely making such a choice *deterministic* (e.g. sort by
    ``instance_id`` alone) is not sufficient; the ranking must be grounded in
    physical evidence, with identity used only as a final, never-reached-in-
    practice tiebreak.

    Ranking key, each field a fallback for the previous only on an EXACT tie
    (never blended into one composite score):

    1. ascending distance from the sub-anchor candidate to its nearest instance
       in ``cands`` (the pool actually being disambiguated, e.g. the tables).
       This is physical evidence: of several "decal" detections, the one that
       is actually near a candidate table is the one "the table closest to the
       decal" can plausibly be about — a duplicate/ghost detection off in an
       unrelated part of the scene is not a real disambiguator for this query.
    2. descending detector ``score`` (max confidence over its own
       observations) — physical evidence of how likely the detection reflects
       a real object at that pose, versus a low-confidence fusion split.
    3. descending ``n_obs`` (distinct keyframe observations) — physical
       evidence of how much of the traversal actually observed this instance;
       more observations is a more robust centroid estimate.
    4. ``instance_id`` ascending — reached only on an exact tie across 1-3,
       which in practice never happens (#111); kept only so the function is
       total and reproducible if it ever does.
    """

    def key(rec: InstanceRecord) -> tuple[float, float, int, int]:
        nearest = min(_centroid_dist(rec, c) for c in cands)
        return (nearest, -rec.score, -rec.n_obs, rec.instance_id)

    return min(sub_anchor_recs, key=key)


def _apply_disambiguator(
    disamb: Clause,
    cands: list[InstanceRecord],
    index: SceneIndex,
    th: Thresholds,
    audit: list[Relaxation] | None,
    _depth: int,
) -> list[InstanceRecord]:
    """Narrow anchor candidates by their own disambiguator clause; drop it (audited)
    when it cannot be applied. Returns the narrowed (never empty unless cands was)."""
    if disamb.pred in _SUPERLATIVE_PREDS:
        sub_anchor_recs = _resolve_anchor(
            disamb.anchors[0], index, th, audit, _depth + 1
        )
        if not sub_anchor_recs:
            _audit_add(
                audit,
                "drop_disambiguator",
                f"disambiguator {disamb.pred.value} dropped: anchor "
                f"'{disamb.anchors[0].noun}' not found",
            )
            return cands
        sub_anchor = _select_sub_anchor(sub_anchor_recs, cands)
        ranked = (
            closest_to(cands, sub_anchor, th)
            if disamb.pred is Pred.CLOSEST_TO
            else farthest_from(cands, sub_anchor, th)
        )
        by_id = {c.instance_id: c for c in cands}
        return [by_id[ranked.order[0]]]

    # non-superlative disambiguator: keep candidates for which the clause holds.
    kept = [
        c
        for c in cands
        if _eval_clause(c, disamb, index, th, audit, _depth + 1).passed
    ]
    if not kept:
        _audit_add(
            audit,
            "drop_disambiguator",
            f"disambiguator {disamb.pred.value} dropped: no "
            f"'{cands[0].label}' candidate satisfied it",
        )
        return cands
    return kept


def _anchor_disambiguator_order(
    anchor: Anchor,
    index: SceneIndex,
    th: Thresholds,
    audit: list[Relaxation] | None,
    _depth: int,
) -> list[InstanceRecord] | None:
    """The anchor's own closest_to/farthest_from disambiguator ranking, best-first.

    Returns ``None`` when ``anchor`` carries no single-anchor superlative
    disambiguator to rank (nothing to fall back through) — the anchor's own
    noun/attribute pool is empty, its disambiguator's own anchor can't be
    resolved, or the nesting-depth guard is reached.

    #151: mirrors the candidate-pool construction ``_resolve_anchor`` /
    ``_apply_disambiguator`` use to pick the top-ranked anchor, so
    ``order[0]`` here is always the same instance those functions commit to.
    Deliberately does not alter or call into either of them — this is a
    read-only probe used by :func:`_disambiguator_anchor_fallback` to see
    what the ranking looks like *beyond* the top pick; the committed
    anchor-resolution path (`resolve()` / `_eval_clause`'s normal walk) is
    untouched.
    """
    disamb = anchor.disambiguator
    if disamb is None or disamb.pred not in _SUPERLATIVE_PREDS or _depth >= _MAX_ANCHOR_DEPTH:
        return None
    cands = _match_anchor_noun(index, anchor.noun)
    if anchor.attributes:
        _class_pool = cands
        cands = [c for c in cands if _attrs_match(c, anchor.attributes, _class_pool, th)]
    if not cands:
        return None
    sub_anchor_recs = _resolve_anchor(disamb.anchors[0], index, th, audit, _depth + 1)
    if not sub_anchor_recs:
        return None
    sub_anchor = _select_sub_anchor(sub_anchor_recs, cands)
    ranked = (
        closest_to(cands, sub_anchor, th)
        if disamb.pred is Pred.CLOSEST_TO
        else farthest_from(cands, sub_anchor, th)
    )
    by_id = {c.instance_id: c for c in cands}
    return [by_id[i] for i in ranked.order]


def _disambiguator_anchor_fallback(
    pool: Sequence[InstanceRecord],
    hard_clauses: Sequence[Clause],
    index: SceneIndex,
    th: Thresholds,
    audit: list[Relaxation],
) -> tuple[set[int], list[InstanceRecord]] | None:
    """#151: retry a strict count that a top-ranked disambiguator anchor emptied.

    Precedent (#122, merged ``2e85b1f``): ``_eval_clause`` was picking the
    highest-SCORING anchor result rather than any PASSING one, so a
    failing-but-high-scoring anchor could mask a genuinely passing one. This
    extends the same principle one level up, to disambiguator anchor
    SELECTION: ``closest_to``/``farthest_from`` picks the single
    nearest/farthest anchor purely by distance, with no regard to whether
    that anchor actually supports the relation the question presupposes. A
    phantom table can beat the real table on raw distance while carrying
    none of the target class ("the table closest to the decal" picks a
    monitor-free phantom over the real table 1.5 m further away).

    Fires ONLY when both hold:

    1. the top-ranked anchor's strict filter yields an empty count, AND
    2. the target class (``pool``) has at least one answer-eligible instance.

    Condition 2 is the load-bearing guard: it is what distinguishes "there
    are no monitors" (leave the 0 alone) from "there are 8 monitors and none
    sit on the anchor we chose" (the anchor choice is the more likely error).
    Callers must only invoke this when ``pool`` is non-empty AND the initial
    strict-filter count came out empty — this function does not re-check
    those, it assumes them.

    Known generalization risk: no question in the 15-scene training battery
    has a true answer of 0, so this branch is never exercised by that
    sample. Condition 2 (not sample evidence) is what makes the change
    defensible on held-out data — without it this would systematically bias
    away from ever answering zero.

    Only single-anchor clauses (BETWEEN excluded) whose anchor carries a
    closest_to/farthest_from disambiguator are retried; deeper nesting stays
    on the existing top-pick behaviour. Returns ``(ids, survivors)`` for the
    first alternate anchor (walked in ranking order) that yields a non-empty
    result, or ``None`` if no clause has such an anchor or none of its
    alternates help.
    """
    for ci, clause in enumerate(hard_clauses):
        if len(clause.anchors) != 1:
            continue
        fn = _BINARY_PREDS.get(clause.pred)
        if fn is None:
            continue
        order = _anchor_disambiguator_order(clause.anchors[0], index, th, audit, 0)
        if order is None or len(order) < 2:
            continue  # no superlative disambiguator here, or nothing to fall back to

        other_clauses = [cl for j, cl in enumerate(hard_clauses) if j != ci]
        for alt_anchor in order[1:]:
            survivors = [
                c
                for c in pool
                if _apply_negation(fn(c, alt_anchor, th), clause).passed
                and all(_eval_clause(c, cl, index, th).passed for cl in other_clauses)
            ]
            if survivors:
                ids = {r.instance_id for r in survivors}
                _audit_add(
                    audit,
                    "disambiguator_anchor_fallback",
                    f"anchor '{clause.anchors[0].noun}' top pick "
                    f"(id={order[0].instance_id}) yielded empty "
                    f"'{clause.pred.value}'; {len(pool)} eligible "
                    f"'{pool[0].label}' target(s) exist, so fell back to "
                    f"next-ranked anchor id={alt_anchor.instance_id}",
                )
                return ids, survivors
    return None


def _eval_clause(
    cand: InstanceRecord,
    clause: Clause,
    index: SceneIndex,
    th: Thresholds,
    audit: list[Relaxation] | None = None,
    _depth: int = 0,
) -> PredResult:
    """Evaluate one non-superlative clause for a candidate; honours negation.

    Existential over resolved anchors: passes if the relation holds for ANY
    matching anchor instance (the definite/indefinite distinction is a counting
    concern, not a filter concern here). Anchors are resolved through
    :func:`_resolve_anchor`, so a nested disambiguator on the clause's anchor is
    bound (and any drop recorded in ``audit``).
    """
    anchor_recs = [_resolve_anchor(a, index, th, audit, _depth) for a in clause.anchors]
    if any(len(r) == 0 for r in anchor_recs):
        base = PredResult(False, 0.0, float("-inf"), f"{clause.pred.value}: anchor not found")
        return _apply_negation(base, clause)

    if clause.pred is Pred.BETWEEN:
        best = None
        best_passing = None
        for b1 in anchor_recs[0]:
            for b2 in anchor_recs[1]:
                r = between(cand, b1, b2, th)
                if best is None or r.score > best.score:
                    best = r
                if r.passed and (best_passing is None or r.score > best_passing.score):
                    best_passing = r
        return _apply_negation(best_passing if best_passing is not None else best, clause)

    fn = _BINARY_PREDS[clause.pred]
    best = None
    best_passing = None
    for b in anchor_recs[0]:
        r = fn(cand, b, th)
        if best is None or r.score > best.score:
            best = r
        if r.passed and (best_passing is None or r.score > best_passing.score):
            best_passing = r
    return _apply_negation(best_passing if best_passing is not None else best, clause)


def _apply_negation(r: PredResult, clause: Clause) -> PredResult:
    """Flip a predicate result for a negated clause ('not near the door')."""
    if not clause.negated:
        return r
    return PredResult(
        passed=not r.passed,
        score=1.0 - r.score,
        margin=-r.margin,
        explanation=f"NOT[{r.explanation}]",
    )


def _clause_selectivity(
    clause: Clause, cands: Sequence[InstanceRecord], index: SceneIndex, th: Thresholds
) -> float:
    """Fraction of candidates that pass a clause; lower = more selective.

    Used to drop the *weakest* (least selective) relation in the fallback ladder.
    Audit is intentionally not threaded here: this is a scoring pass over the pool,
    not the committed evaluation, so disambiguator drops must not be double-logged.
    """
    if not cands:
        return 1.0
    passes = sum(1 for c in cands if _eval_clause(c, clause, index, th).passed)
    return passes / len(cands)


def _superlative_clause(target: TargetSpec) -> Clause | None:
    for cl in target.clauses:
        if cl.pred in _SUPERLATIVE_PREDS:
            return cl
    return None


def resolve(
    target: TargetSpec,
    index: SceneIndex,
    th: Thresholds = DEFAULT_THRESHOLDS,
) -> ResolveResult:
    """Resolve a TargetSpec to ranked candidates with a per-clause pass matrix.

    Pipeline: hard-filter by noun (typo-tolerant) + target attributes; AND over
    non-superlative clauses; superlative clause ranks survivors rather than
    filtering. Empty result triggers the exact architecture fallback ladder:
    relax attributes -> drop weakest relation -> category-only. Every relaxation
    is recorded in the audit trail.
    """
    audit: list[Relaxation] = []
    base = _match_noun(index, target.noun)

    # attribute-filtered pool (base is the same-class pool for relative size ranking)
    pool = [c for c in base if _attrs_match(c, target.attributes, base, th)]
    hard_clauses = [c for c in target.clauses if c.pred not in _SUPERLATIVE_PREDS]
    #: The relation clauses as originally stated, kept even once the fallback ladder
    #: relaxes/drops them from ``hard_clauses`` (issue #59 probe): lets the
    #: category-only rung rank by how well a candidate still MATCHES the dropped
    #: relation instead of discarding it outright (see the ranking step below).
    original_hard_clauses = list(hard_clauses)
    sup = _superlative_clause(target)

    survivors = _filter_and(pool, hard_clauses, index, th)

    # --- fallback ladder -----------------------------------------------------
    if not survivors:
        # step 1: relax attributes
        if target.attributes:
            audit.append(Relaxation("relax_attributes", f"dropped {target.attributes}"))
            pool = list(base)
            survivors = _filter_and(pool, hard_clauses, index, th)

    if not survivors and len(hard_clauses) >= 2:
        # step 2: drop the weakest (least selective) relation, one at a time,
        # while at least one relation still constrains the result. Dropping the
        # final relation is the category-only rung (step 3), not this one.
        remaining = list(hard_clauses)
        while len(remaining) >= 2 and not survivors:
            # least selective = highest pass fraction
            weakest = max(
                remaining,
                key=lambda cl: _clause_selectivity(cl, pool, index, th),
            )
            remaining = [c for c in remaining if c is not weakest]
            audit.append(
                Relaxation("drop_relation", f"dropped weakest clause {weakest.pred.value}")
            )
            survivors = _filter_and(pool, remaining, index, th)
        hard_clauses = remaining  # matrix reflects the clauses actually applied

    if not survivors:
        # step 3: category-only
        audit.append(Relaxation("category_only", f"fell back to all '{target.noun}' instances"))
        survivors = list(pool) if pool else list(base)
        hard_clauses = []

    # --- ranking -------------------------------------------------------------
    margins: dict[int, float] = {c.instance_id: 0.0 for c in survivors}
    if sup is not None and survivors:
        anchor_recs = _resolve_anchor(sup.anchors[0], index, th, audit)
        if anchor_recs:
            anchor = anchor_recs[0]  # salience: first (index order); deterministic
            ranked = (
                closest_to(survivors, anchor, th)
                if sup.pred is Pred.CLOSEST_TO
                else farthest_from(survivors, anchor, th)
            )
            by_id = {c.instance_id: c for c in survivors}
            survivors = [by_id[i] for i in ranked.order]
            for i in survivors:
                margins[i.instance_id] = ranked.margin
        else:
            audit.append(
                Relaxation("superlative_anchor_missing", f"{sup.anchors[0].noun} not found")
            )
            survivors = _tier_priority_order(survivors, index, target.noun, hard_clauses, th)
    elif not hard_clauses and original_hard_clauses and len(survivors) > 1:
        # issue #59: the fallback ladder dropped every relation clause (category-only
        # rung) because no candidate passed it as a HARD filter — e.g. "the potted
        # plant on the table" where the plant's own footprint AABB (a wide canopy)
        # narrowly fails the `on()` anchor-larger gate against a small side table
        # despite strong overlap and correct height. Ranking then fell through to
        # `_tier_priority_order` (effectively instance-id order), an arbitrary
        # tie-break with no relation to the dropped clause — so an unrelated
        # candidate (e.g. a floor-standing plant nowhere near any table) could
        # outrank the one that actually sits on a table in every way but the exact
        # gate. Rank by best-effort match to the ORIGINAL clauses instead (soft
        # PredResult.score, never a hard requirement) so the candidate closest to
        # satisfying the dropped relation wins, before falling back to the same
        # deterministic tie-break for genuine ties.
        survivors = _relaxed_relation_order(survivors, original_hard_clauses, index, th)
    elif len(survivors) > 1:
        # No top-level superlative: if a surviving hard clause's anchor carried a
        # nested superlative disambiguator, break ties by that nested metric rather
        # than by instance id (OR-F1: two-tables-two-bowls). Otherwise tier-priority
        # (issue #51): an exact/synonym label match must not lose a tie to an
        # unrelated head-noun cousin merely by carrying a lower instance id.
        survivors = _nested_superlative_order(survivors, hard_clauses, index, th, target.noun)
    else:
        survivors = _tier_priority_order(survivors, index, target.noun)

    # --- pass matrix (over the clauses actually applied) ---------------------
    pass_matrix: dict[int, list[PredResult]] = {}
    for c in survivors:
        row = [_eval_clause(c, cl, index, th, audit) for cl in hard_clauses]
        pass_matrix[c.instance_id] = row

    return ResolveResult(survivors, pass_matrix, margins, audit)


def _relaxed_relation_order(
    survivors: list[InstanceRecord],
    dropped_clauses: Sequence[Clause],
    index: SceneIndex,
    th: Thresholds,
) -> list[InstanceRecord]:
    """Order survivors by best-effort match to relation clauses the fallback ladder
    dropped as a hard filter (issue #59), instead of an arbitrary id-order tie-break.

    Score = sum of each dropped clause's soft ``PredResult.score`` (in [0, 1] per
    clause; never a HARD requirement, so this never re-excludes a survivor — it only
    orders the category-only pool by relevance). Ties broken by instance_id for
    determinism, matching every other tie-break in this module.
    """
    def _score(c: InstanceRecord) -> float:
        return sum(_eval_clause(c, cl, index, th).score for cl in dropped_clauses)

    return sorted(survivors, key=lambda c: (-_score(c), c.instance_id))


def _nested_superlative_order(
    survivors: list[InstanceRecord],
    hard_clauses: Sequence[Clause],
    index: SceneIndex,
    th: Thresholds,
    noun: str | None = None,
) -> list[InstanceRecord]:
    """Order survivors by the nested superlative metric of the first hard clause
    whose anchor carries one; fall back to tier-priority (then stable-by-id) when
    none applies.

    Each survivor is scored by the distance from the survivor to its own
    best-matching disambiguated anchor (the anchor kept by the nested superlative),
    so the survivor sitting by the argmin/argmax anchor ranks first — never an
    instance-id accident.
    """
    for clause in hard_clauses:
        if clause.pred in _SUPERLATIVE_PREDS:
            continue
        anchor = clause.anchors[0]
        disamb = anchor.disambiguator
        if disamb is None or disamb.pred not in _SUPERLATIVE_PREDS:
            continue
        anchor_recs = _resolve_anchor(anchor, index, th)  # already narrowed to argmin/argmax
        if not anchor_recs:
            continue
        target_anchor = anchor_recs[0]
        # Rank survivors by proximity to the disambiguated (argmin/argmax) anchor:
        # the survivor most bound to the selected anchor wins, never instance id.
        dists = {c.instance_id: _centroid_dist(c, target_anchor) for c in survivors}
        return sorted(survivors, key=lambda c: (dists[c.instance_id], c.instance_id))
    if noun is not None:
        return _tier_priority_order(survivors, index, noun, hard_clauses, th)
    return _stable_by_id(survivors)


def _tier_priority_order(
    recs: Sequence[InstanceRecord],
    index: SceneIndex,
    noun: str,
    hard_clauses: Sequence[Clause] = (),
    th: Thresholds | None = None,
) -> list[InstanceRecord]:
    """Stable-by-id, but an EXACT/SYNONYM label match is never outranked by a
    HEAD_NOUN cousin dragged in by the same-class pool (issue #51 root cause: with
    no disambiguating clause, ``resolve()`` fell straight to instance-id order, so a
    query like "coffee table" could rank an unrelated "dressing table" (lower
    instance id, HEAD_NOUN tier) ahead of the exact "coffee table" — both a scoring
    artifact AND a real navigation defect, since corridor-leg anchors ("the sofa and
    the coffee table") carry no disambiguator by construction.

    BARE-noun queries are exempt from the TIER grouping only (mirrors
    ``_match_anchor_noun``'s #21 exemption): "table" legitimately means every
    table, cousins and exact alike, so imposing a tier preference there would
    wrongly bias a same-class superlative/count pool. Falls back to plain
    stable-by-id when the index exposes no tiered lookup (test doubles) --
    still only for the tier component; see #116 below for why a bare noun must
    NOT also skip the clause-score ordering.

    Issue #73: an exact label is strictly more specific than an alias/cousin
    label — that specificity ordering does not stop at the label tier. When
    ``hard_clauses`` (the relation clauses this candidate pool actually survived,
    e.g. ``between()``/``on()``) are supplied, ties WITHIN one label tier are broken
    by each survivor's own summed clause ``PredResult.score`` before falling to
    instance id: a same-label survivor that satisfies the SAME clause more
    specifically (a stronger continuous margin) is a more specific match than one
    that merely cleared the hard-filter threshold, so it must not lose a tie to a
    detection-order accident (home_building_2/office_2, issue #71 audit residual).
    This never re-ranks ACROSS tiers (exact still always beats alias/cousin) and
    never fires when no clause context is given (``hard_clauses`` empty), so every
    existing tier-only call site is unaffected.

    Issue #116 (livingroom_1 "vase between the TV and the door", surfaced once
    #110 gave the scene a resolvable "tv"): the ORIGINAL bare-noun exemption
    above returned ``_stable_by_id`` unconditionally, which threw away the
    ``hard_clauses`` score-based ordering too -- not just the tier grouping --
    for every bare-noun query (e.g. plain "vase", not "coffee table"). That let
    two same-label survivors with genuinely DIFFERENT ``between()`` margins
    (real, world-grounded discriminating evidence -- the caller's own
    ``_same_label_group_is_tied`` correctly judges this pair NOT tied, so it
    never engages its own quantised-position reorder) fall through to bare
    instance_id order regardless, which flips under renumbering -- reproducing
    exactly the #113 defect this whole tier-priority scheme exists to prevent.
    A bare noun has no tier distinction to make (every candidate is
    trivially the same "tier"), but that is orthogonal to whether clause
    evidence exists to order same-tier survivors by; the fix is to skip ONLY
    the tier grouping for bare nouns, never the score-based ordering.
    """
    from core.perception.scene_index import normalize_label
    from core.perception.vocab import head_noun

    query = normalize_label(noun)
    bare = query == head_noun(query)
    tiered = None if bare else getattr(index, "by_label_tiered", None)
    tier_by_id: dict[int, MatchTier] = (
        {} if bare or tiered is None else {rec.instance_id: tier for rec, tier in tiered(noun)}
    )
    worst = MatchTier.TYPO
    score_by_id: dict[int, float] = {}
    if hard_clauses and th is not None:
        score_by_id = {
            r.instance_id: sum(
                _eval_clause(r, cl, index, th).score for cl in hard_clauses
            )
            for r in recs
        }
    if not tier_by_id and not score_by_id:
        return _stable_by_id(recs)
    return sorted(
        recs,
        key=lambda r: (
            tier_by_id.get(r.instance_id, worst),
            -score_by_id.get(r.instance_id, 0.0),
            r.instance_id,
        ),
    )


def _filter_and(
    pool: Sequence[InstanceRecord],
    clauses: Sequence[Clause],
    index: SceneIndex,
    th: Thresholds,
    audit: list[Relaxation] | None = None,
) -> list[InstanceRecord]:
    """Keep candidates passing ALL clauses (AND)."""
    out = []
    for c in pool:
        if all(_eval_clause(c, cl, index, th, audit).passed for cl in clauses):
            out.append(c)
    return out


def _stable_by_id(recs: Sequence[InstanceRecord]) -> list[InstanceRecord]:
    return sorted(recs, key=lambda r: r.instance_id)


# --------------------------------------------------------------------------- counting

# Generic scene-scope nouns: the room is the universe of discourse for a count
# ("how many stools are in the room?"), so a clause anchored ONLY on one of
# these is vacuous scoping, not a filter — no instance is ever labeled "room".
# Named room types (kitchen, bedroom, office, ...) are NOT in this set and stay
# strict: "how many stools are in the kitchen?" must still filter on "kitchen".
_SCOPE_NOUNS = frozenset({"room", "scene", "area", "house", "home", "building", "apartment"})


def _normalize_noun(noun: str) -> str:
    """Lowercase + strip a trailing 's' so plural scope nouns ('rooms') match."""
    n = noun.strip().lower()
    return n[:-1] if n.endswith("s") and len(n) > 1 else n


def _is_scope_clause(clause: Clause) -> bool:
    """True if every anchor of ``clause`` is a generic scene-scope noun."""
    return bool(clause.anchors) and all(
        _normalize_noun(a.noun) in _SCOPE_NOUNS for a in clause.anchors
    )


@dataclass(frozen=True)
class CountResult:
    """Output of :func:`counting`.

    count: cardinality of the strictly-filtered set (0 is a legal answer).
    ids: contributing instance_ids.
    explanations: when ``count == 0`` and a clause emptied the set, the failing
                  clause explanation(s) — so the head can distinguish "relation
                  unmeasurable" from "genuinely zero" and decide for itself; empty
                  otherwise.
    audit: any disambiguator drops surfaced while evaluating the clauses (the
           counting path never relaxes noun/attribute/relation filters, but a
           nested disambiguator on an anchor may still be dropped and must remain
           visible).

    Iterable as ``(count, ids)`` so existing ``n, ids = counting(...)`` callers are
    unchanged; ``.explanations`` / ``.audit`` are available to callers that read them.
    """

    count: int
    ids: set[int]
    explanations: list[str] = field(default_factory=list)
    audit: list[Relaxation] = field(default_factory=list)

    def __iter__(self):
        yield self.count
        yield self.ids


def counting(
    target: TargetSpec,
    index: SceneIndex,
    min_obs: int = 1,
    th: Thresholds = DEFAULT_THRESHOLDS,
) -> CountResult:
    """Strict set-cardinality: noun + attributes + hard clauses, NO relaxation.

    Unlike :func:`resolve` (which must publish *some* box and so runs the fallback
    ladder), counting is the cardinality of a filtered set — a relaxed or dropped
    filter must yield 0, never the whole-category total (red-team NUM-F1). The
    pipeline here is therefore the AND-filter only: category match, target
    attributes, then every non-superlative clause. A superlative clause never
    filters (it ranks), so it is ignored for counting. When the filtered set is
    empty, the failing clause explanation(s) are attached to the result so the
    head can tell "relation unmeasurable" from "genuinely zero"; the toolbox does
    not guess a fallback integer.

    Returns a :class:`CountResult`; deduplicates by instance_id (the tracker/NMS
    layer upstream guarantees one id per physical object).
    """
    audit: list[Relaxation] = []
    base = _match_noun(index, target.noun)
    pool = [c for c in base if _attrs_match(c, target.attributes, base, th)]
    hard_clauses = []
    for c in target.clauses:
        if c.pred in _SUPERLATIVE_PREDS:
            continue
        if _is_scope_clause(c):
            anchor_desc = ", ".join(a.noun for a in c.anchors)
            audit.append(
                Relaxation(
                    "scope_clause",
                    f"'{c.pred.value}({anchor_desc})' is scene-scope, not a filter — skipped",
                )
            )
            continue
        hard_clauses.append(c)

    survivors = _filter_and(pool, hard_clauses, index, th, audit)
    ids = {r.instance_id for r in survivors if r.n_obs >= min_obs}

    if not ids and pool:
        fallback = _disambiguator_anchor_fallback(pool, hard_clauses, index, th, audit)
        if fallback is not None:
            fb_survivors = [r for r in fallback[1] if r.n_obs >= min_obs]
            if fb_survivors:
                survivors = fb_survivors
                ids = {r.instance_id for r in fb_survivors}

    explanations: list[str] = []
    if not ids:
        explanations = _empty_count_explanations(target, base, pool, hard_clauses, index, th)

    return CountResult(len(ids), ids, explanations, audit)


def _empty_count_explanations(
    target: TargetSpec,
    base: Sequence[InstanceRecord],
    pool: Sequence[InstanceRecord],
    hard_clauses: Sequence[Clause],
    index: SceneIndex,
    th: Thresholds,
) -> list[str]:
    """Name why a count came out 0: which stage of the strict filter emptied it.

    Ordered most-specific-first so the head sees the operative reason first.
    """
    out: list[str] = []
    if not base:
        out.append(f"no '{target.noun}' instances in scene")
        return out
    if target.attributes and not pool:
        out.append(f"no '{target.noun}' matched attributes {list(target.attributes)}")
        return out
    # noun (+ attributes) matched; a relation clause is what emptied the set.
    for cl in hard_clauses:
        if not _filter_and(pool, [cl], index, th):
            out.append(f"no '{target.noun}' satisfied {cl.pred.value} clause")
    if not out:
        # every clause individually retained something, but their conjunction did
        # not — report the joint failure rather than a single clause.
        preds = ", ".join(cl.pred.value for cl in hard_clauses)
        out.append(f"no '{target.noun}' satisfied all clauses jointly ({preds})")
    return out


# --------------------------------------------------------------------------- corridor / avoid

#: issue #155: minimum gate width (metres) the vehicle can physically fit through,
#: derived from the SAME constant the planner already uses to decide a gap is
#: sealed by its own body — ``core.nav.planner._pinch_costmap``'s docstring notes
#: "a physical gap narrower than ``2 * vehicle_radius_m`` gets sealed end-to-end by
#: the vehicle's own inflation margin from BOTH anchors" (`core/nav/planner.py`,
#: `_pinch_costmap` docstring). Not a new tuned number — the vehicle's own footprint
#: radius, doubled, exactly as the planner already treats it.
MIN_PASSABLE_GATE_WIDTH_M: float = 2.0 * _VEHICLE_RADIUS_M


def corridor_gate(
    b1: InstanceRecord, b2: InstanceRecord, index: SceneIndex | None = None
) -> Gate:
    """Gate segment between the two anchors' closest AABB faces, plus its midpoint.

    issue #69 (D1): the gate's face points walk the anchors' CENTROID-to-centroid
    line out to each footprint's own boundary
    (:func:`P.centroid_axis_face_points_2d`) — the same axis :func:`between` already
    uses to define "a between b1 and b2" — rather than the independent per-axis
    footprint-overlap projection this used before (:func:`P.aabb_face_points_2d`,
    still available for callers that want the old axis-aligned construction). The
    old projection degenerates to a near-zero-width, unusable "gate" whenever the
    two footprints already touch or overlap (adjacent furniture — no gap exists to
    project a shared-overlap midpoint from) and can misplace the gate entirely for a
    diagonally-offset anchor pair; the centroid axis stays well-defined (and
    consistent with the module's own "between" semantics) in both cases.

    issue #155: "well-defined" is not the same as "usable". When the two anchors'
    AABB footprints already overlap in the XY plane (adjacent/flush furniture with
    no real floor gap between them — e.g. livingroom_1's sofa/round-table pair,
    which overlap by 1.3 cm along their connecting axis), the centroid-axis
    construction above still returns SOME p0/p1 (it is well-defined, per #69), but
    those points land inside the shared overlap sliver: a sub-2-cm "gate" a metre
    away from the room's actual navigable gap, not a physically passable opening.
    No fallback construction recovers a real gate here — the objects are genuinely
    touching, so there IS no navigable gap between these two anchors along this
    axis to construct. Flagging ``Gate.degenerate`` (footprints overlap AND the
    resulting width is under :data:`MIN_PASSABLE_GATE_WIDTH_M`, the vehicle's own
    fit-through width) lets callers treat the leg's threading requirement as
    unevaluable instead of recording a violation the robot could never have
    avoided (`core.groundtruth.scoring.score_instruction_rubric`). A gate that is
    merely narrow but has NO footprint overlap (a real, if tight, doorway) is left
    untouched — narrow-but-real gates still score exactly as before.

    issue #63: when ``index`` is given, a genuine THIRD instance's footprint sitting
    at the raw midpoint (e.g. hotel_room_2's duplicate GT "bed frame" instance for
    the same physical bed as the "bed" anchor — a real obstruction, not either
    anchor's own footprint) is not a usable via-point, so the midpoint is nudged
    along the gate's own axis to the nearest clear point via
    ``P.usable_gate_point`` — the SAME helper `core.nav.planner.plan_through` calls
    for its own corridor target, so scoring and planning can never disagree about
    where a blocked gate's usable crossing is (the #51 mismatch class). ``gate.p0``/
    ``gate.p1`` (the verified anchor-to-anchor line used by threading checks) are
    never nudged — only the mandatory via-point. Omit ``index`` (or pass one with no
    blocking third instance, or a gate whose block has no clear point within the
    bounded search) to get the untouched exact midpoint. A degenerate gate skips
    this nudge entirely: sliding a via-point along an axis that never had a real
    gap to begin with is meaningless.
    """
    pa, pb = P.centroid_axis_face_points_2d(
        b1.centroid, b1.aabb_min, b1.aabb_max, b2.centroid, b2.aabb_min, b2.aabb_max
    )
    mid = (pa + pb) / 2.0
    width = float(np.linalg.norm(pb - pa))
    degenerate = (
        width < MIN_PASSABLE_GATE_WIDTH_M
        and P.footprints_overlap(b1.aabb_min, b1.aabb_max, b2.aabb_min, b2.aabb_max)
    )
    if index is not None and not degenerate:
        exclude_ids = {b1.instance_id, b2.instance_id}
        others = [r for r in index.all_instances() if r.instance_id not in exclude_ids]

        def _blocked(pt: np.ndarray) -> bool:
            return any(P.point_in_footprint_2d(pt, r.aabb_min, r.aabb_max) for r in others)

        mid = P.usable_gate_point(pa, pb, mid, _blocked)
    return Gate(pa, pb, mid, width, degenerate=degenerate)


def resolve_corridor_pair(
    anchor: Anchor,
    index: SceneIndex,
    th: Thresholds,
    from_xy: tuple[float, float] | None,
    to_xy: tuple[float, float] | None,
) -> tuple[InstanceRecord, InstanceRecord] | None:
    """#167: pick the two DISTINCT anchor instances for a bare "the two X" /
    "between the two X" corridor leg, with a route-context tie-break when
    clustering (#160) still leaves more than two same-label candidates.

    ``_resolve_anchor`` (with #160 clustering applied) supplies the candidate
    pool. With 0 or 1 candidates no pair exists (``None``). With exactly 2 the
    choice is unambiguous -- return them, lower ``instance_id`` first for a
    deterministic, reproducible order. With more than 2 (the corridor scorer
    intentionally leaves untouched -- see the #167 commit body: it resolves on
    GT, which has exactly two "column" instances for arabic_room and so never
    hits this branch), every unordered pair is scored by DETOUR: the extra
    distance the route would travel by threading through that pair's gate
    midpoint versus going straight from ``from_xy`` to ``to_xy``,

        detour(pair) = |from_xy -> gate.midpoint| + |gate.midpoint -> to_xy|
                        - |from_xy -> to_xy|

    and the minimum-detour pair wins -- the physically closest reading of
    "between the two columns" when more than two same-label objects exist: the
    pair that keeps the route most nearly on its already-intended line. Ties
    (equal detour, float-tolerant) break on ascending
    ``(instance_id, instance_id)`` for determinism.

    ``from_xy``/``to_xy`` are the previous leg's resolved goal and the next
    leg's resolved goal (or the robot's current pose when this corridor leg is
    the route's last leg) -- both purely geometric inputs, no caller state read
    here. Either or both may be ``None`` (first leg / terminal leg with no pose
    tracked here); the corresponding term of the detour sum is simply omitted,
    degrading gracefully to "closest gate to the one known point" or, with
    both missing, straight to the ascending-``instance_id`` pair (still
    deterministic, just uninformed by route context).
    """
    cands = _resolve_anchor(anchor, index, th)
    if len(cands) < 2:
        return None
    cands = sorted(cands, key=lambda c: c.instance_id)
    if len(cands) == 2:
        return cands[0], cands[1]

    def _detour(b1: InstanceRecord, b2: InstanceRecord) -> float:
        gate = corridor_gate(b1, b2)
        mid = gate.midpoint
        total = 0.0
        straight = 0.0
        if from_xy is not None:
            total += float(np.linalg.norm(mid - np.asarray(from_xy, dtype=float)))
        if to_xy is not None:
            total += float(np.linalg.norm(mid - np.asarray(to_xy, dtype=float)))
        if from_xy is not None and to_xy is not None:
            straight = float(
                np.linalg.norm(
                    np.asarray(to_xy, dtype=float) - np.asarray(from_xy, dtype=float)
                )
            )
        return total - straight

    best_pair = None
    best_detour = None
    for i in range(len(cands)):
        for j in range(i + 1, len(cands)):
            b1, b2 = cands[i], cands[j]
            d = _detour(b1, b2)
            key = (round(d, 9), b1.instance_id, b2.instance_id)
            if best_detour is None or key < best_detour:
                best_detour = key
                best_pair = (b1, b2)
    return best_pair


def threading_check(trajectory: np.ndarray, gate: Gate) -> tuple[bool, str]:
    """True if the trajectory polyline properly crosses the gate segment.

    Proper segment intersection (not proximity): some trajectory edge must cross
    the gate p0-p1 line. Approaching from one side without crossing returns False.
    """
    traj = np.asarray(trajectory, dtype=float)
    if traj.ndim != 2 or traj.shape[0] < 2 or traj.shape[1] < 2:
        return False, "threading: trajectory needs >=2 (x,y) points"
    for i in range(traj.shape[0] - 1):
        if P.segments_intersect_2d(traj[i], traj[i + 1], gate.p0, gate.p1):
            return True, f"threading: crossed gate on edge {i}->{i+1}"
    return False, "threading: trajectory never crossed the gate segment"


def avoid_capsule(
    spec: AvoidSpec, index: SceneIndex, th: Thresholds = DEFAULT_THRESHOLDS
) -> Capsule:
    """Build the avoid region for an AvoidSpec: capsule for .between, disc for .near.

    .between -> stadium along the segment between the two anchors' centroids,
    radius = max anchor half-width + 0.25 m inflation.
    .near -> disc (segment a==b) of radius = near_thresh(anchor) + 0.25 m.
    Raises ValueError if the spec's anchors are unresolvable or the spec is malformed.
    """
    if (spec.between is None) == (spec.near is None):
        raise ValueError("AvoidSpec must set exactly one of between/near")

    if spec.between is not None:
        recs0 = _resolve_anchor(spec.between[0], index, th)
        recs1 = _resolve_anchor(spec.between[1], index, th)
        if not recs0 or not recs1:
            raise ValueError("avoid_capsule: between anchor(s) not found")
        b1, b2 = recs0[0], recs1[0]
        radius = (
            max(
                P.footprint_half_width(b1.aabb_min, b1.aabb_max),
                P.footprint_half_width(b2.aabb_min, b2.aabb_max),
            )
            + th.avoid_inflate
        )
        return Capsule(P._as3(b1.centroid)[:2], P._as3(b2.centroid)[:2], float(radius))

    recs = _resolve_anchor(spec.near, index, th)
    if not recs:
        raise ValueError("avoid_capsule: near anchor not found")
    anchor = recs[0]
    radius = near_thresh(anchor, th) + th.avoid_inflate
    c = P._as3(anchor.centroid)[:2]
    return Capsule(c, c, float(radius))


def capsule_violated(
    trajectory: np.ndarray, capsule: Capsule
) -> tuple[bool, np.ndarray | None]:
    """True + first violation point if the trajectory enters the avoid capsule.

    Samples each trajectory vertex and the closest approach of each edge to the
    capsule segment; a violation is any point within the capsule radius.
    """
    traj = np.asarray(trajectory, dtype=float)
    if traj.ndim != 2 or traj.shape[0] < 1 or traj.shape[1] < 2:
        return False, None
    for i in range(traj.shape[0]):
        p = traj[i, :2]
        dist, _ = P.point_to_segment_2d(p, capsule.a, capsule.b)
        if dist <= capsule.radius + P.EPS:
            return True, p.copy()
        if i + 1 < traj.shape[0]:
            # sample the edge for a mid-edge dip inside the capsule
            q = traj[i + 1, :2]
            for s in (0.25, 0.5, 0.75):
                m = p + s * (q - p)
                dm, _ = P.point_to_segment_2d(m, capsule.a, capsule.b)
                if dm <= capsule.radius + P.EPS:
                    return True, m.copy()
    return False, None
