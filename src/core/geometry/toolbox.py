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
config); the spec-mandated values (near, next_to, avoid inflation) are the
defaults.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from core.interfaces import InstanceRecord, SceneIndex
from core.plan_schema import Anchor, AvoidSpec, Clause, Pred, TargetSpec
from core.geometry import primitives as P


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
    on_vert_tol: float = 0.15  # a.bottom within +/- this of b.top  (invented)
    on_min_overlap_frac: float = 0.30  # footprint overlap / a-footprint  (invented)
    in_containment_frac: float = 0.60  # a-footprint fraction inside b  (invented)
    in_vert_slack: float = 0.10  # a within b's z-span, this much slack  (invented)
    above_gap_max: float = 3.0  # cap on above/under vertical gap  (invented)
    with_feature_pad: float = 0.30  # "near" pad for possession test  (invented)
    avoid_inflate: float = 0.25  # capsule/disc inflation for avoid geometry
    superlative_margin_frac: float = 0.25  # early-answer winner-margin gate


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


def _attrs_match(a: InstanceRecord, attributes: Sequence[str]) -> bool:
    """True if every requested attribute keyword appears in the record's text."""
    if not attributes:
        return True
    text = _label_text(a)
    return all(attr.lower() in text for attr in attributes)


# --------------------------------------------------------------------------- predicates


def on(a: InstanceRecord, b: InstanceRecord, th: Thresholds = DEFAULT_THRESHOLDS) -> PredResult:
    """a rests on b: a's bottom within vert tol of b's top AND footprint overlap."""
    a_bottom = float(a.aabb_min[2])
    b_top = float(b.aabb_max[2])
    vgap = a_bottom - b_top  # >0 hovering above, <0 sunk into b
    vert_ok = abs(vgap) <= th.on_vert_tol + P.EPS
    area = P.footprint_overlap_area(a.aabb_min, a.aabb_max, b.aabb_min, b.aabb_max)
    a_area = max(P.footprint_overlap_area(a.aabb_min, a.aabb_max, a.aabb_min, a.aabb_max), P.EPS)
    frac = area / a_area
    over_ok = frac >= th.on_min_overlap_frac
    passed = bool(vert_ok and over_ok)
    vmargin = th.on_vert_tol - abs(vgap)
    score = float(max(0.0, min(1.0, frac)) * (1.0 if vert_ok else 0.0))
    expl = (
        f"on: bottom {a_bottom:.2f} vs top {b_top:.2f} (vgap {vgap:+.2f}m, tol "
        f"{th.on_vert_tol}m -> {'ok' if vert_ok else 'FAIL'}); footprint overlap "
        f"{frac*100:.0f}% (>= {th.on_min_overlap_frac*100:.0f}% -> "
        f"{'ok' if over_ok else 'FAIL'})"
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
    """a adjacent to b: AABB gap <= 0.75 m (tighter than near)."""
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
    passed = bool(dist <= radius + P.EPS)
    margin = radius - dist
    score = float(max(0.0, min(1.0, 1.0 - dist / radius))) if radius > 0 else 0.0
    expl = (
        f"between: centroid {dist:.2f}m from b1-b2 segment (t={t:.2f}) <= capsule "
        f"radius {radius:.2f}m -> {'ok' if passed else 'FAIL'}"
    )
    return PredResult(passed, score, float(margin), expl)


def above(a: InstanceRecord, b: InstanceRecord, th: Thresholds = DEFAULT_THRESHOLDS) -> PredResult:
    """a above b: footprint overlap AND a's bottom above b's top (no support/contact)."""
    over = P.footprints_overlap(a.aabb_min, a.aabb_max, b.aabb_min, b.aabb_max)
    gap = float(a.aabb_min[2]) - float(b.aabb_max[2])  # >0 => a strictly above
    vert_ok = gap > P.EPS
    passed = bool(over and vert_ok)
    score = 1.0 if passed else 0.0
    expl = (
        f"above: footprint overlap {'ok' if over else 'FAIL'}; vertical gap "
        f"{gap:+.2f}m (a above b -> {'ok' if vert_ok else 'FAIL'})"
    )
    return PredResult(passed, score, float(gap), expl)


def under(a: InstanceRecord, b: InstanceRecord, th: Thresholds = DEFAULT_THRESHOLDS) -> PredResult:
    """a under b: footprint overlap AND a's top below b's bottom (no contact required)."""
    over = P.footprints_overlap(a.aabb_min, a.aabb_max, b.aabb_min, b.aabb_max)
    gap = float(b.aabb_min[2]) - float(a.aabb_max[2])  # >0 => a strictly under
    vert_ok = gap > P.EPS
    passed = bool(over and vert_ok)
    score = 1.0 if passed else 0.0
    expl = (
        f"under: footprint overlap {'ok' if over else 'FAIL'}; vertical gap "
        f"{gap:+.2f}m (a below b -> {'ok' if vert_ok else 'FAIL'})"
    )
    return PredResult(passed, score, float(gap), expl)


def with_feature(
    a: InstanceRecord, b: InstanceRecord, th: Thresholds = DEFAULT_THRESHOLDS
) -> PredResult:
    """a possesses feature b: b's centroid inside or near a's AABB footprint."""
    c = P._as3(b.centroid)[:2]
    lo = P.footprint_min(a.aabb_min, a.aabb_max)
    hi = P.footprint_max(a.aabb_min, a.aabb_max)
    inside = bool(np.all(c >= lo - P.EPS) and np.all(c <= hi + P.EPS))
    # distance from b's centroid to a's footprint (0 if inside)
    d = np.maximum.reduce([lo - c, c - hi, np.zeros(2)])
    dist = float(np.linalg.norm(d))
    passed = bool(inside or dist <= th.with_feature_pad + P.EPS)
    margin = th.with_feature_pad - dist
    score = 1.0 if inside else float(max(0.0, min(1.0, 1.0 - dist / th.with_feature_pad)))
    expl = (
        f"with: feature centroid {'inside' if inside else f'{dist:.2f}m from'} "
        f"a's footprint (pad {th.with_feature_pad}m -> {'ok' if passed else 'FAIL'})"
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


# clause-pred -> binary predicate function (BETWEEN handled separately)
_BINARY_PREDS = {
    Pred.ON: on,
    Pred.IN: in_,
    Pred.NEAR: near,
    Pred.NEXT_TO: next_to,
    Pred.ABOVE: above,
    Pred.UNDER: under,
    Pred.WITH: with_feature,
}
_SUPERLATIVE_PREDS = {Pred.CLOSEST_TO, Pred.FARTHEST_FROM}


# --------------------------------------------------------------------------- resolution


def _match_noun(index: SceneIndex, noun: str) -> list[InstanceRecord]:
    """Typo/synonym-tolerant category lookup via the index."""
    return list(index.by_label(noun))


def _resolve_anchor(
    anchor: Anchor, index: SceneIndex, th: Thresholds
) -> list[InstanceRecord]:
    """Resolve an anchor to matching records (noun + attributes; ignores disambiguator
    nesting for the toolbox's binary-predicate needs — nested disambiguation is a
    resolver concern handled by resolve() on the top-level target)."""
    cands = _match_noun(index, anchor.noun)
    if anchor.attributes:
        cands = [c for c in cands if _attrs_match(c, anchor.attributes)]
    return cands


def _eval_clause(
    cand: InstanceRecord,
    clause: Clause,
    index: SceneIndex,
    th: Thresholds,
) -> PredResult:
    """Evaluate one non-superlative clause for a candidate; honours negation.

    Existential over resolved anchors: passes if the relation holds for ANY
    matching anchor instance (the definite/indefinite distinction is a counting
    concern, not a filter concern here).
    """
    anchor_recs = [_resolve_anchor(a, index, th) for a in clause.anchors]
    if any(len(r) == 0 for r in anchor_recs):
        base = PredResult(False, 0.0, float("-inf"), f"{clause.pred.value}: anchor not found")
        return _apply_negation(base, clause)

    if clause.pred is Pred.BETWEEN:
        best = None
        for b1 in anchor_recs[0]:
            for b2 in anchor_recs[1]:
                r = between(cand, b1, b2, th)
                if best is None or r.score > best.score:
                    best = r
        return _apply_negation(best, clause)

    fn = _BINARY_PREDS[clause.pred]
    best = None
    for b in anchor_recs[0]:
        r = fn(cand, b, th)
        if best is None or r.score > best.score:
            best = r
    return _apply_negation(best, clause)


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

    # attribute-filtered pool
    pool = [c for c in base if _attrs_match(c, target.attributes)]
    hard_clauses = [c for c in target.clauses if c.pred not in _SUPERLATIVE_PREDS]
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
        anchor_recs = _resolve_anchor(sup.anchors[0], index, th)
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
            survivors = _stable_by_id(survivors)
    else:
        survivors = _stable_by_id(survivors)

    # --- pass matrix (over the clauses actually applied) ---------------------
    pass_matrix: dict[int, list[PredResult]] = {}
    for c in survivors:
        row = [_eval_clause(c, cl, index, th) for cl in hard_clauses]
        pass_matrix[c.instance_id] = row

    return ResolveResult(survivors, pass_matrix, margins, audit)


def _filter_and(
    pool: Sequence[InstanceRecord],
    clauses: Sequence[Clause],
    index: SceneIndex,
    th: Thresholds,
) -> list[InstanceRecord]:
    """Keep candidates passing ALL clauses (AND)."""
    out = []
    for c in pool:
        if all(_eval_clause(c, cl, index, th).passed for cl in clauses):
            out.append(c)
    return out


def _stable_by_id(recs: Sequence[InstanceRecord]) -> list[InstanceRecord]:
    return sorted(recs, key=lambda r: r.instance_id)


# --------------------------------------------------------------------------- counting


def counting(
    target: TargetSpec,
    index: SceneIndex,
    min_obs: int = 1,
    th: Thresholds = DEFAULT_THRESHOLDS,
) -> tuple[int, set[int]]:
    """Set-cardinality over resolve() survivors with n_obs >= min_obs.

    Returns (count, contributing_instance_ids). Deduplicates by instance_id (the
    tracker/NMS layer upstream guarantees one id per physical object; this guards
    against a survivor list that repeated an id).
    """
    res = resolve(target, index, th)
    ids = {r.instance_id for r in res.candidates_ranked if r.n_obs >= min_obs}
    return len(ids), ids


# --------------------------------------------------------------------------- corridor / avoid


def corridor_gate(b1: InstanceRecord, b2: InstanceRecord) -> Gate:
    """Gate segment between the two anchors' closest AABB faces, plus its midpoint."""
    pa, pb = P.aabb_face_points_2d(b1.aabb_min, b1.aabb_max, b2.aabb_min, b2.aabb_max)
    mid = (pa + pb) / 2.0
    width = float(np.linalg.norm(pb - pa))
    return Gate(pa, pb, mid, width)


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
