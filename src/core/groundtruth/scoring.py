"""Per-qtype REAL scorers for the ground-truth battery.

Unlike :mod:`core.runner.battery` (structural health, no ground truth), these score
answers our pipeline produces against the VLA-3D ground-truth scene. Three scorers,
one per :class:`~core.interfaces.QType`:

* NUMERICAL  -> :func:`score_numerical`. Exact-match of our count against a
  ground-truth count. The **primary** GT count is computed by running OUR
  :func:`core.geometry.toolbox.resolve`/``counting`` against the GT index.
  **Circularity, stated honestly:** the "ground truth" here is our own resolver run
  over ground-truth geometry, so an exact match validates *pipeline self-consistency*
  (parse -> resolve -> count is stable and geometry-driven), NOT that the count equals
  the absolute human-annotated answer. Where an independent count is derivable —
  from ``_referential_statements.json`` (distinct annotated target instances of the
  same class+relation) or the scene-graph relations — we compute it as a **second
  opinion** and report both numbers side by side rather than collapsing them.

* OBJECT_REFERENCE -> :func:`score_object_reference`. 3D IoU between our answer box
  (the :class:`~core.interfaces.MarkerBox` AABB our resolver selects) and the GT
  target's AABB. The GT target is taken, when available, from the referential-
  statement annotation matching the question (``target_index`` -> that object's GT
  AABB); where the question can't be matched to an annotation unambiguously, the
  target is flagged (``target_source == "ambiguous"``) rather than guessed, and IoU
  is reported against our own top pick as a lower-confidence signal.

* INSTRUCTION_FOLLOWING -> :func:`score_instruction_following`. Two numbers, never a
  single fake composite: (1) **discrete Frechet distance** between our waypoint path
  and the GT ``trajectory_qN.ply`` path, and (2) **fraction of GT path within 1.0 m**
  of our path (coverage). Both in metres / [0,1]; low Frechet + high coverage == good.

Pure/deterministic: numpy only, no network, no RNG.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from core.geometry import toolbox as T
from core.interfaces import InstanceRecord, MarkerBox, SceneIndex
from core.parsing.regex_tier import parse_regex
from core.perception.scene_index import BasicSceneIndex, normalize_label

PROXIMITY_M = 1.0  # coverage radius for path scoring


def _anchor_agrees(question_anchor: str, ann_anchor_class: str) -> bool:
    """Loose anchor-class agreement tolerant of surface drift.

    Challenge nouns and referential ``target_class``/anchor ``class`` values differ in
    surface form ("coffee cup" vs "cup", "potted plant" vs "plant"). We accept a match
    when the normalised strings are equal, one contains the other, or they share a
    content token — enough to link a question anchor to an annotation without the
    exact-string brittleness that produced spurious "none" matches.
    """
    a = normalize_label(question_anchor)
    b = normalize_label(ann_anchor_class)
    if not a or not b:
        return False
    if a == b or a in b or b in a:
        return True
    ta, tb = set(a.split()), set(b.split())
    return bool(ta & tb)


# --------------------------------------------------------------------------- 3D IoU


def aabb_iou_3d(box_a: MarkerBox, box_b_min: np.ndarray, box_b_max: np.ndarray) -> float:
    """3D IoU between a MarkerBox (center+extents) and a GT AABB (min/max corners)."""
    a_min = np.array([box_a.cx - box_a.sx / 2, box_a.cy - box_a.sy / 2, box_a.cz - box_a.sz / 2])
    a_max = np.array([box_a.cx + box_a.sx / 2, box_a.cy + box_a.sy / 2, box_a.cz + box_a.sz / 2])
    return iou_from_corners(a_min, a_max, box_b_min, box_b_max)


def iou_from_corners(
    a_min: np.ndarray, a_max: np.ndarray, b_min: np.ndarray, b_max: np.ndarray
) -> float:
    """3D IoU of two axis-aligned boxes given as (3,) min/max corners."""
    lo = np.maximum(a_min, b_min)
    hi = np.minimum(a_max, b_max)
    inter_dims = np.clip(hi - lo, 0.0, None)
    inter = float(np.prod(inter_dims))
    if inter <= 0.0:
        return 0.0
    vol_a = float(np.prod(np.clip(a_max - a_min, 0.0, None)))
    vol_b = float(np.prod(np.clip(b_max - b_min, 0.0, None)))
    union = vol_a + vol_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


# --------------------------------------------------------------------------- numerical


@dataclass
class NumericalScore:
    """Result of scoring one numerical question."""

    our_count: int
    gt_count_pipeline: int  # primary: OUR resolve() over the GT index (circular)
    exact_match: bool  # our_count == gt_count_pipeline
    gt_count_independent: int | None  # second opinion from referential/scene-graph
    independent_source: str  # "referential" | "none"
    note: str = ""


def _independent_count(
    text: str, referential: dict | None
) -> tuple[int | None, str]:
    """Second-opinion count from referential statements: distinct annotated targets.

    Matches the question's target class and (loosely) its relation phrasing against
    the annotated statement set, counting distinct ``target_index`` values. Returns
    (None, "none") when no independent evidence is derivable — we do NOT fabricate a
    number in that case.
    """
    if not referential:
        return None, "none"
    plan = parse_regex(text)
    if plan.target is None:
        return None, "none"
    tgt_noun = normalize_label(plan.target.noun)
    # anchors mentioned in the question (for a loose relation match)
    anchor_nouns = {
        normalize_label(a.noun) for cl in plan.target.clauses for a in cl.anchors
    }
    # Relation-aware set: target class matches AND (if anchors named) at least one
    # named anchor class appears in the annotation's anchors.
    rel_ids: set[str] = set()
    # Relation-agnostic set: target class matches (coarser second opinion — counts
    # every distinct annotated instance of the class, ignoring the spatial relation).
    class_ids: set[str] = set()
    for _rid, stmts in (referential.get("regions") or {}).items():
        for _stmt, anns in stmts.items():
            if not isinstance(anns, list):  # e.g. a "region": "<label>" metadata key
                continue
            for ann in anns:
                if not isinstance(ann, dict):
                    continue
                tclass = str(ann.get("target_class", ""))
                if not _anchor_agrees(tgt_noun, tclass):
                    continue
                tid = str(ann.get("target_index", ""))
                if not tid:
                    continue
                class_ids.add(tid)
                if anchor_nouns:
                    ann_anchor_classes = [
                        str(v.get("class", ""))
                        for v in (ann.get("anchors") or {}).values()
                    ]
                    if any(
                        _anchor_agrees(qa, ac)
                        for qa in anchor_nouns
                        for ac in ann_anchor_classes
                    ):
                        rel_ids.add(tid)
                else:
                    rel_ids.add(tid)
    if rel_ids:
        return len(rel_ids), "referential"
    if class_ids:
        # No statement matched the named relation/anchor, but the class is annotated:
        # report the class-level count, flagged as relation-agnostic (coarser).
        return len(class_ids), "referential_class_only"
    return None, "none"


def score_numerical(
    text: str,
    index: SceneIndex,
    *,
    referential: dict | None = None,
) -> NumericalScore:
    """Score a numerical question against the GT index.

    Primary GT count = OUR resolver's ``counting`` over the GT index (circular — see
    module docstring). Second opinion (when derivable) from referential statements.
    Our answer count is also our resolver's count, so ``exact_match`` here really
    asks "is the count deterministic / stable" — the independent number is what tells
    you whether the pipeline count is *right*.
    """
    plan = parse_regex(text)
    if plan.target is None:
        return NumericalScore(
            our_count=0,
            gt_count_pipeline=0,
            exact_match=True,
            gt_count_independent=None,
            independent_source="none",
            note="no target parsed",
        )
    # min_obs=1: GT is fully observed (n_obs=3), so this is not the gate here.
    gt_count, _ids = T.counting(plan.target, index, min_obs=1)
    our_count = gt_count  # same path; the exact-match records determinism
    indep, src = _independent_count(text, referential)
    note = ""
    if indep is not None and indep != gt_count:
        note = f"pipeline={gt_count} vs independent={indep} (disagreement)"
    return NumericalScore(
        our_count=our_count,
        gt_count_pipeline=gt_count,
        exact_match=(our_count == gt_count),
        gt_count_independent=indep,
        independent_source=src,
        note=note,
    )


# --------------------------------------------------------------------------- object ref


@dataclass
class ObjectRefScore:
    """Result of scoring one object-reference question."""

    iou: float
    our_marker: MarkerBox | None
    gt_target_id: int | None
    target_source: str  # "referential" | "ambiguous" | "none"
    note: str = ""


def _gt_target_from_referential(
    text: str, referential: dict | None, instances: list[InstanceRecord]
) -> tuple[int | None, str]:
    """Find the GT target object id for an object-reference question.

    Strategy: exact statement-string match first (the question phrasing often matches
    a generated statement verbatim up to punctuation/case); else fall back to a
    class+anchor match returning the id only when it is unique (else "ambiguous").
    """
    if not referential:
        return None, "none"

    def norm(s: str) -> str:
        return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()

    q = norm(text)
    # 1) exact / substring statement match
    for _rid, stmts in (referential.get("regions") or {}).items():
        for stmt, anns in stmts.items():
            if not isinstance(anns, list) or not anns:
                continue
            ns = norm(stmt)
            if ns and (ns == q or ns in q or q in ns):
                if isinstance(anns[0], dict):
                    tid = anns[0].get("target_index")
                    if tid is not None:
                        return int(tid), "referential"

    # 2) class + anchor match, unique-id only
    plan = parse_regex(text)
    if plan.target is None:
        return None, "ambiguous"
    tgt_noun = normalize_label(plan.target.noun)
    anchor_nouns = {
        normalize_label(a.noun) for cl in plan.target.clauses for a in cl.anchors
    }
    matched: set[int] = set()
    for _rid, stmts in (referential.get("regions") or {}).items():
        for _stmt, anns in stmts.items():
            if not isinstance(anns, list):
                continue
            for ann in anns:
                if not isinstance(ann, dict):
                    continue
                if not _anchor_agrees(tgt_noun, str(ann.get("target_class", ""))):
                    continue
                if anchor_nouns:
                    ann_anchor_classes = [
                        str(v.get("class", ""))
                        for v in (ann.get("anchors") or {}).values()
                    ]
                    if not any(
                        _anchor_agrees(qa, ac)
                        for qa in anchor_nouns
                        for ac in ann_anchor_classes
                    ):
                        continue
                tid = ann.get("target_index")
                if tid is not None:
                    matched.add(int(tid))
    if len(matched) == 1:
        return next(iter(matched)), "referential"
    if matched:
        return None, "ambiguous"
    return None, "none"


def score_object_reference(
    text: str,
    index: SceneIndex,
    instances: list[InstanceRecord],
    *,
    referential: dict | None = None,
) -> ObjectRefScore:
    """Score an object-reference question: 3D IoU of our box vs the GT target box.

    Our answer: the top candidate our resolver selects on the GT index, as a
    MarkerBox. GT target, in priority order: (1) the referential-statement annotation
    matching the question (``target_source == "referential"``); (2) category
    uniqueness — if the scene holds exactly one instance of the target noun it needs
    no disambiguation and IS the target (``"unique_in_scene"``). When neither pins a
    trustworthy target the IoU is left undefined (NaN) and flagged
    (``"ambiguous"``/``"none"``) rather than guessed — we never fabricate a target.
    """
    plan = parse_regex(text)
    our_marker: MarkerBox | None = None
    unique_target_id: int | None = None
    if plan.target is not None:
        res = T.resolve(plan.target, index)
        if res.candidates_ranked:
            our_marker = res.candidates_ranked[0].to_marker()
        # Category-level uniqueness: if the scene has exactly ONE instance of the
        # target noun, no disambiguation is possible or needed — that instance is the
        # ground-truth target unambiguously (a defensible GT independent of the
        # referential annotations).
        cat = list(index.by_label(plan.target.noun))
        if len(cat) == 1:
            unique_target_id = cat[0].instance_id

    gt_id, source = _gt_target_from_referential(text, referential, instances)
    # Prefer a referential-annotated target; else fall back to category-uniqueness.
    if gt_id is None and unique_target_id is not None:
        gt_id, source = unique_target_id, "unique_in_scene"
    by_id = {r.instance_id: r for r in instances}

    if our_marker is None:
        return ObjectRefScore(
            iou=0.0, our_marker=None, gt_target_id=gt_id, target_source=source,
            note="our resolver returned no candidate",
        )

    if gt_id is not None and gt_id in by_id:
        gt = by_id[gt_id]
        iou = aabb_iou_3d(our_marker, gt.aabb_min, gt.aabb_max)
        note = "" if source in ("referential", "unique_in_scene") else "target inferred (non-exact)"
        return ObjectRefScore(iou, our_marker, gt_id, source, note)

    # No trustworthy GT target: report self-IoU (1.0) but flag it clearly.
    return ObjectRefScore(
        iou=float("nan"),
        our_marker=our_marker,
        gt_target_id=None,
        target_source="ambiguous" if source != "none" else "none",
        note="no GT target matched; IoU undefined (flagged, not guessed)",
    )


# --------------------------------------------------------------------------- IF paths


def load_trajectory_ply(path: os.PathLike | str) -> np.ndarray:
    """Parse an ASCII vertex-only PLY trajectory into an (N, 3) float array.

    The challenge trajectory PLYs are ``format ascii 1.0``, vertex-only, three float
    properties (x y z), z fixed at 0.75 (see docs/vla3d_notes.md §5). We parse the
    header for the vertex count then read that many ``x y z`` rows — by hand, no PLY
    library (offline / no new deps).
    """
    pts: list[list[float]] = []
    n_expected: int | None = None
    with open(path, encoding="utf-8") as fh:
        in_header = True
        for line in fh:
            s = line.strip()
            if in_header:
                if s.startswith("element vertex"):
                    n_expected = int(s.split()[-1])
                elif s == "end_header":
                    in_header = False
                continue
            if not s:
                continue
            parts = s.split()
            if len(parts) < 3:
                continue
            pts.append([float(parts[0]), float(parts[1]), float(parts[2])])
            if n_expected is not None and len(pts) >= n_expected:
                break
    return np.asarray(pts, dtype=float) if pts else np.empty((0, 3), dtype=float)


def _xy(path: np.ndarray) -> np.ndarray:
    """XY projection of an (N, >=2) path (trajectories are fixed-z)."""
    p = np.asarray(path, dtype=float)
    if p.ndim != 2 or p.shape[0] == 0:
        return np.empty((0, 2), dtype=float)
    return p[:, :2]


def discrete_frechet(path_a: np.ndarray, path_b: np.ndarray) -> float:
    """Discrete Frechet distance between two polylines (XY), metres.

    Standard Eiter-Mannila dynamic-programming recurrence over the coupling matrix,
    iterative (no recursion-depth risk on long paths). Returns +inf if either path is
    empty.
    """
    a = _xy(path_a)
    b = _xy(path_b)
    n, m = a.shape[0], b.shape[0]
    if n == 0 or m == 0:
        return float("inf")
    # pairwise distances
    diff = a[:, None, :] - b[None, :, :]  # (n, m, 2)
    dist = np.sqrt((diff * diff).sum(axis=2))  # (n, m)
    ca = np.full((n, m), -1.0)
    ca[0, 0] = dist[0, 0]
    for i in range(1, n):
        ca[i, 0] = max(ca[i - 1, 0], dist[i, 0])
    for j in range(1, m):
        ca[0, j] = max(ca[0, j - 1], dist[0, j])
    for i in range(1, n):
        for j in range(1, m):
            ca[i, j] = max(
                min(ca[i - 1, j], ca[i - 1, j - 1], ca[i, j - 1]),
                dist[i, j],
            )
    return float(ca[n - 1, m - 1])


def path_coverage(
    gt_path: np.ndarray, our_path: np.ndarray, radius: float = PROXIMITY_M
) -> float:
    """Fraction of GT path vertices within ``radius`` metres of our path (XY).

    "Within our path" = distance to the nearest of our vertices (point-to-vertex, a
    conservative lower bound vs point-to-segment). Returns 0.0 if our path is empty.
    """
    g = _xy(gt_path)
    o = _xy(our_path)
    if g.shape[0] == 0:
        return 0.0
    if o.shape[0] == 0:
        return 0.0
    diff = g[:, None, :] - o[None, :, :]  # (G, O, 2)
    dmin = np.sqrt((diff * diff).sum(axis=2)).min(axis=1)  # (G,)
    return float((dmin <= radius).mean())


@dataclass
class InstructionScore:
    """Result of scoring one instruction-following question (two numbers)."""

    frechet_m: float
    coverage_1m: float  # fraction of GT path within 1.0 m of ours
    our_n_waypoints: int
    gt_n_waypoints: int
    frame_aligned: bool = True  # False when a frame offset was detected/uncorrected
    note: str = ""


#: If the two paths' centroids are farther apart than this, we treat the frames as
#: unaligned and flag the score — the VLA-3D object CSV frame and the challenge
#: trajectory (robot/map) frame are NOT guaranteed to share an origin (observed on
#: loft: object x-extent runs to +11 m while the trajectory tops out at ~7.7 m, a
#: translation the dataset does not ship a transform for). Absolute Frechet/coverage
#: across unaligned frames is not meaningful, so we report the raw numbers but mark
#: them un-trustworthy rather than fabricate an aligned score.
_FRAME_OFFSET_FLAG_M = 2.0


def score_instruction_following(
    our_path: np.ndarray,
    trajectory_ply: os.PathLike | str,
) -> InstructionScore:
    """Score our waypoint path against a GT trajectory PLY (Frechet + coverage).

    Reports two numbers, never a composite. Also flags a probable frame offset (see
    ``_FRAME_OFFSET_FLAG_M``): when our path and the GT path centroids are far apart,
    the frames are likely unaligned and the two numbers should be read as a lower
    bound / diagnostic, not a clean accuracy.
    """
    gt = load_trajectory_ply(trajectory_ply)
    frech = discrete_frechet(gt, our_path)
    cov = path_coverage(gt, our_path)
    note = ""
    op = np.asarray(our_path, dtype=float)
    empty = op.ndim != 2 or op.shape[0] == 0
    frame_aligned = True
    if empty:
        note = "our path empty — pipeline produced no waypoints"
    elif gt.shape[0]:
        gc = _xy(gt).mean(axis=0)
        oc = _xy(op).mean(axis=0)
        if float(np.linalg.norm(gc - oc)) > _FRAME_OFFSET_FLAG_M:
            frame_aligned = False
            note = (
                "probable frame offset: our-path vs GT-path centroids far apart; "
                "VLA-3D object frame and challenge trajectory frame are not aligned "
                "(no transform shipped) — read Frechet/coverage as diagnostic only"
            )
    return InstructionScore(
        frechet_m=frech,
        coverage_1m=cov,
        our_n_waypoints=int(op.shape[0]) if op.ndim == 2 else 0,
        gt_n_waypoints=int(gt.shape[0]),
        frame_aligned=frame_aligned,
        note=note,
    )
