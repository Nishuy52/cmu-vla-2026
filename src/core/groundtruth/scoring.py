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

* INSTRUCTION_FOLLOWING -> :func:`score_instruction_rubric` (HEADLINE, IF-F2) and
  :func:`score_instruction_following` (SECONDARY diagnostics). The challenge scores the
  *driven* trajectory on ordered path-constraint adherence with forbidden-region
  penalties, so the headline is a rubric proxy over a simulated driven trajectory:
  ordered per-leg arrival credit (partial credit per leg reached in order) minus
  ``threading_check`` / ``capsule_violated`` penalties. The old discrete-Frechet +
  coverage@1m of the PLANNED path are kept ONLY as secondary diagnostics (shape
  similarity to the reference PLY, which the rubric does not pay for).

Pure/deterministic: numpy only, no network, no RNG.
"""
from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from core.geometry import primitives as P
from core.geometry import toolbox as T
from core.geometry.toolbox import (
    DEFAULT_THRESHOLDS,
    Capsule,
    Gate,
    Thresholds,
    capsule_violated,
    threading_check,
)
from core.groundtruth.arrival import NOMINAL_ARRIVAL_TOL_M, derived_arrival_tol_m
from core.groundtruth.vocab_bridge import bridge_synonyms, bridged_agree
from core.interfaces import InstanceRecord, MarkerBox, SceneIndex
from core.parsing.regex_tier import _SUPERLATIVE_PREDS as _PARSER_SUPERLATIVE_PREDS
from core.parsing.regex_tier import parse_regex
from core.perception.scene_index import BasicSceneIndex, normalize_label
from core.plan_walk import iter_clause_anchors, iter_target_anchors

PROXIMITY_M = 1.0  # coverage radius for path scoring

#: Fuzzy statement-match acceptance threshold (Jaccard over content tokens). At/above
#: this the question phrasing is treated as the same statement despite surface drift.
_FUZZY_JACCARD_MIN = 0.8

#: Map a parsed :class:`~core.plan_schema.Pred` value to the set of VLA-3D statement
#: ``relation`` strings (and scene-graph relationship keys) that express the same
#: spatial predicate. Kept loose on purpose — a question's "closest to" and a
#: statement's "closest" are the same relation; "on"/"near"/"above" all describe
#: physical support/adjacency the scene graph splits into separate keys.
_PRED_TO_RELATIONS: dict[str, tuple[str, ...]] = {
    "closest_to": ("closest",),
    "farthest_from": ("farthest",),
    "between": ("between",),
    "near": ("near", "beside"),
    "next_to": ("near", "beside"),
    "on": ("on", "near", "above", "hanging_on"),
    "in": ("in",),
    "above": ("above",),
    "under": ("below",),
    "with": ("on", "near", "hanging_on"),
}


def _pred_relations(pred) -> tuple[str, ...]:
    """Statement/scene-graph relation strings a parsed predicate may correspond to."""
    if pred is None:
        return ()
    val = getattr(pred, "value", pred)
    return _PRED_TO_RELATIONS.get(str(val), ())


#: Parsed predicate values that express a ranked (superlative) relation. Derived from
#: the parser's canonical Pred set (:data:`core.parsing.regex_tier._SUPERLATIVE_PREDS`)
#: so there is a single source of truth — adding a superlative predicate there flows
#: through to this scorer automatically instead of drifting out of sync.
_SUPERLATIVE_PRED_VALUES = frozenset(p.value for p in _PARSER_SUPERLATIVE_PREDS)


def _is_superlative_clause(clause) -> bool:
    """True if a parsed clause's predicate is a superlative ("closest"/"farthest")."""
    pred = getattr(clause, "pred", None)
    return str(getattr(pred, "value", pred)) in _SUPERLATIVE_PRED_VALUES


def _norm_stmt(s: str) -> str:
    """Whitespace/case/punctuation-normalised statement or question string."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", s.lower())).strip()


def _jaccard(a: str, b: str) -> float:
    """Token-set Jaccard overlap of two normalised strings ([0,1])."""
    ta, tb = set(a.split()), set(b.split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _bridge_normalise_stmt(stmt_norm: str, question_noun: str) -> str:
    """Rewrite a bridged annotation-vocabulary noun in a statement to the question noun.

    The phrasing tie-break (step 3 below) scores a statement's token overlap with the
    question to pick, among ordinal variants ("farthest" vs "second farthest"), the one
    that best matches the question's phrasing. When the question and statement use a
    *bridged synonym* for the same object ("bedside table" vs "night stand"), the noun
    tokens never overlap, deflating that overlap uniformly across all variants and below
    the tie-break's absolute floor — even though the ordinal words that actually
    discriminate ARE present. To restore a meaningful comparison we substitute, in the
    normalised statement, any whitelisted synonym surface of ``question_noun`` with the
    question noun itself, so the shared object tokens count and the ordinal qualifier
    becomes the deciding difference. Only bridged (explicitly whitelisted) synonyms are
    rewritten — this never invents overlap between genuinely different objects.
    """
    qn = normalize_label(question_noun)
    if not qn:
        return stmt_norm
    syns = bridge_synonyms(qn)
    if not syns:
        return stmt_norm
    out = stmt_norm
    for syn in sorted(syns, key=len, reverse=True):  # longest first (multi-word synonyms)
        if syn and syn in out:
            out = out.replace(syn, qn)
    return out


def _question_relations(plan) -> tuple[str, ...]:
    """Union of statement-relation strings implied by a parsed target's clauses."""
    rels: set[str] = set()
    if plan.target is not None:
        for cl in plan.target.clauses:
            rels.update(_pred_relations(getattr(cl, "pred", None)))
    return tuple(sorted(rels))


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
    if ta & tb:
        return True
    # Final leniency: a whitelisted surface-form synonym linking the SAME object class
    # across the challenge/annotation vocabulary drift ("bedside table" <-> "night
    # stand", "potted plant" <-> "plant"). Adds only explicitly bridged pairs — it does
    # not widen the substring/token guard, so honest-none is preserved for true misses.
    return bridged_agree(question_anchor, ann_anchor_class)


def _class_equal(question_noun: str, ann_class: str) -> bool:
    """Strict target-class equality: normalised equality OR a whitelisted bridge only.

    NUM-F6 fix. Where :func:`_anchor_agrees` deliberately accepts substring / shared-
    token overlap (right for *anchor* linkage, which tolerates surface drift like
    "coffee cup" vs "cup"), that leniency *inflates independent counts* when used to
    pick which annotated targets belong to the queried class: "photo frame" targets
    counted as "photo" (livingroom_3: 10 vs 9), "tv cabinet" matched by the shared "tv"
    token to any TV relation. For counting a *class*, the match must be strict — the
    normalised nouns are equal, or they are explicitly bridged as the same object class
    (``vocab_bridge`` whitelist). No substring, no shared-token fallback.
    """
    a = normalize_label(question_noun)
    b = normalize_label(ann_class)
    if not a or not b:
        return False
    if a == b:
        return True
    return bridged_agree(question_noun, ann_class)


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
    gt_count_independent: int | None  # 2nd opinion from referential statements
    independent_source: str  # "referential" | "referential_class_only" | "none"
    gt_count_scenegraph: int | None = None  # 3rd opinion from scene-graph relations
    scenegraph_source: str = "none"  # "scene_graph" | "scene_graph_class_only" | "none"
    #: NUM-F6 annotation-coverage: (annotated target instances of the class, CSV
    #: instances of the class). When the first is far below the second the class is
    #: under-annotated and any independent count over it is deflated / untrustworthy.
    annotated_targets_of_class: int | None = None
    csv_instances_of_class: int | None = None
    note: str = ""
    #: Parse-time notes off the resolved Plan (e.g. "unparsed clause text dropped: ...")
    #: — regex-tier ctx.notes joined by parse_regex. Empty when the parse was clean.
    parse_notes: str = ""


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
    # anchors mentioned in the question (for a loose relation match) — recurses through
    # each anchor's nested ``disambiguator`` chain (issue #95) so an anchor named only
    # as a disambiguator still participates in the relation-aware match.
    anchor_nouns = {normalize_label(a.noun) for a in iter_target_anchors(plan.target)}
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
                # NUM-F6: strict class equality for the TARGET class (no substring/token
                # fallback) — that leniency inflated "photo" with "photo frame" targets.
                if not _class_equal(tgt_noun, tclass):
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


def _annotation_coverage(
    text: str, referential: dict | None, index: SceneIndex
) -> tuple[int | None, int | None]:
    """(annotated target instances of the class, CSV instances of the class).

    NUM-F6 coverage column. The first counts distinct ``target_index`` values whose
    ``target_class`` strictly equals the question's target noun across every referential
    statement (relation-agnostic — how many instances of the class the generator ever
    annotated as a target). The second is the scene's instance count of the same class
    (via the index's strict label lookup). A first value well below the second flags an
    under-annotated class whose independent count is deflated and untrustworthy. Returns
    (None, csv) when no referential set is present.
    """
    plan = parse_regex(text)
    if plan.target is None:
        return None, None
    tgt_noun = normalize_label(plan.target.noun)
    csv_n: int | None = None
    try:
        csv_n = sum(
            1
            for r in index.by_label(plan.target.noun)
            if _class_equal(tgt_noun, r.label)
        )
    except Exception:  # noqa: BLE001 — index lookup is best-effort for the coverage column
        csv_n = None
    if not referential:
        return None, csv_n
    ann_ids: set[str] = set()
    for _rid, stmts in (referential.get("regions") or {}).items():
        if not isinstance(stmts, dict):
            continue
        for _stmt, anns in stmts.items():
            if not isinstance(anns, list):
                continue
            for ann in anns:
                if not isinstance(ann, dict):
                    continue
                if not _class_equal(tgt_noun, str(ann.get("target_class", ""))):
                    continue
                tid = str(ann.get("target_index", ""))
                if tid:
                    ann_ids.add(tid)
    return len(ann_ids), csv_n


def _scene_graph_count(
    text: str, scene_graph: dict | None
) -> tuple[int | None, str]:
    """Third-opinion count directly from ``_scene_graph.json`` relation edges.

    The scene graph stores, per region, ``relationships[relation][target_id] =
    [anchor_ids]`` plus a per-object ``raw_label``. We count distinct objects whose
    label matches the question's target noun AND that appear as the *target* of a
    relation the question's predicate maps to (:data:`_PRED_TO_RELATIONS`) with an
    anchor whose label matches a question anchor. When the question names no
    resolvable relation/anchor we fall back to a class-only count (flagged coarser).
    Returns (None, "none") when nothing is derivable — never a fabricated number.
    """
    if not scene_graph:
        return None, "none"
    plan = parse_regex(text)
    if plan.target is None:
        return None, "none"
    tgt_noun = normalize_label(plan.target.noun)
    # Recurses through each anchor's nested ``disambiguator`` chain (issue #95).
    anchor_nouns = {normalize_label(a.noun) for a in iter_target_anchors(plan.target)}
    q_rels = set(_question_relations(plan))

    # id -> label, and merged relation edges across all regions.
    id2label: dict[str, str] = {}
    # relation -> {target_id: set(anchor_ids)}
    edges: dict[str, dict[str, set[str]]] = {}
    for _rid, reg in (scene_graph.get("regions") or {}).items():
        if not isinstance(reg, dict):
            continue
        for o in reg.get("objects", []):
            oid = str(o.get("object_id", ""))
            if oid:
                id2label[oid] = str(o.get("raw_label", "")).lower()
        for relname, tgts in (reg.get("relationships") or {}).items():
            if not isinstance(tgts, dict):
                continue
            bucket = edges.setdefault(relname, {})
            for tgt, anchors in tgts.items():
                if not isinstance(anchors, list):
                    continue
                # T7-S2: `between` stores each entry as an [id, id] PAIR, not a flat
                # anchor id. Flattening the pair with str(a) garbles it into
                # "[3, 7]"; expand nested pairs into their member ids so between
                # anchors are counted correctly (all other relations are flat lists).
                flat: list[str] = []
                for a in anchors:
                    if isinstance(a, (list, tuple)):
                        flat.extend(str(x) for x in a)
                    else:
                        flat.append(str(a))
                bucket.setdefault(str(tgt), set()).update(flat)

    # NUM-F6: strict class equality for the target class (anchor linkage below still
    # uses the looser _anchor_agrees, which is correct for surface-drifting anchors).
    class_ids = {oid for oid, lbl in id2label.items() if _class_equal(tgt_noun, lbl)}
    if not class_ids:
        return None, "none"

    # Relation-aware: target-class objects that are the target of a mapped relation
    # with an anchor whose label matches a question anchor.
    rel_names = q_rels or {"on", "near", "beside", "above"}
    rel_ids: set[str] = set()
    for oid in class_ids:
        for rn in rel_names:
            anchors = edges.get(rn, {}).get(oid, set())
            if not anchors:
                continue
            if not anchor_nouns:
                rel_ids.add(oid)
                break
            anchor_labels = [id2label.get(a, "") for a in anchors]
            if any(_anchor_agrees(qa, al) for qa in anchor_nouns for al in anchor_labels):
                rel_ids.add(oid)
                break
    if rel_ids:
        return len(rel_ids), "scene_graph"
    return len(class_ids), "scene_graph_class_only"


def score_numerical(
    text: str,
    index: SceneIndex,
    *,
    referential: dict | None = None,
    scene_graph: dict | None = None,
    thresholds: Thresholds = DEFAULT_THRESHOLDS,
    min_obs: int = 1,
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
            parse_notes=plan.notes,
        )
    # min_obs default 1: GT is fully observed (every instance has n_obs=3), so raising
    # this gate does not change the count on GT scenes — the sweep may vary it but it is
    # inert here by construction (kept wired so the harness is faithful, not so it moves).
    gt_count, _ids = T.counting(plan.target, index, min_obs=min_obs, th=thresholds)
    our_count = gt_count  # same path; the exact-match records determinism
    indep, src = _independent_count(text, referential)
    sg_count, sg_src = _scene_graph_count(text, scene_graph)
    ann_cov, csv_cov = _annotation_coverage(text, referential, index)
    # NUM-F6(b): a ``*_class_only`` count is a relation-agnostic total, NOT independent
    # evidence for the relation-filtered question — comparing it as a disagreement is a
    # scorer artifact (arabic_room "1 vs 3"). Such rows are excluded from the
    # disagreement signal and reported as "no independent evidence" instead.
    indep_is_evidence = indep is not None and src == "referential"
    sg_is_evidence = sg_count is not None and sg_src == "scene_graph"
    disagree = [f"pipeline={gt_count}"]
    if indep_is_evidence:
        disagree.append(f"independent={indep}")
    if sg_is_evidence:
        disagree.append(f"scene_graph={sg_count}")
    distinct = {gt_count}
    if indep_is_evidence:
        distinct.add(indep)
    if sg_is_evidence:
        distinct.add(sg_count)
    note = ""
    if len(distinct) > 1:
        note = "; ".join(disagree) + " (disagreement)"
    elif not indep_is_evidence and not sg_is_evidence:
        note = "no independent evidence (relation-agnostic class-only counts only)"
    # NUM-F6(c): flag a deflated (under-annotated) class so the row reads as untrustworthy.
    if ann_cov is not None and csv_cov is not None and ann_cov < csv_cov:
        cov_note = (
            f"annotation coverage {ann_cov}/{csv_cov} — class under-annotated, "
            "independent count deflated"
        )
        note = f"{note}; {cov_note}" if note else cov_note
    return NumericalScore(
        our_count=our_count,
        gt_count_pipeline=gt_count,
        exact_match=(our_count == gt_count),
        gt_count_independent=indep,
        independent_source=src,
        gt_count_scenegraph=sg_count,
        scenegraph_source=sg_src,
        annotated_targets_of_class=ann_cov,
        csv_instances_of_class=csv_cov,
        note=note,
        parse_notes=plan.notes,
    )


# --------------------------------------------------------------------------- object ref


@dataclass
class ObjectRefScore:
    """Result of scoring one object-reference question."""

    iou: float
    our_marker: MarkerBox | None
    gt_target_id: int | None
    # "referential" | "unique_in_scene" | "geometry" | "ambiguous" | "none"
    target_source: str
    # "exact" | "fuzzy" | "relation" | "unique" | "geometric" | "none"
    match_method: str = "none"
    our_target_id: int | None = None  # instance id of our top resolver pick, if any
    note: str = ""
    #: Parse-time notes off the resolved Plan (e.g. "unparsed clause text dropped: ...")
    #: — regex-tier ctx.notes joined by parse_regex. Empty when the parse was clean.
    parse_notes: str = ""


def _iter_statements(referential: dict | None):
    """Yield (statement_string, annotation_dict) over every referential statement.

    Skips metadata keys (e.g. a region's ``"region": "<label>"``) and any non-dict
    annotation entries, so callers can assume ``ann`` is a grounding dict.
    """
    for _rid, stmts in (referential.get("regions") or {}).items() if referential else []:
        if not isinstance(stmts, dict):
            continue
        for stmt, anns in stmts.items():
            if not isinstance(anns, list):
                continue
            for ann in anns:
                if isinstance(ann, dict):
                    yield stmt, ann


def _gt_target_from_referential(
    text: str, referential: dict | None, instances: list[InstanceRecord]
) -> tuple[int | None, str, str]:
    """Find the GT target object id for an object-reference question.

    Returns ``(target_index, source, match_method)`` where ``match_method`` is one of
    ``"exact"`` / ``"fuzzy"`` / ``"relation"`` / ``"none"``. Matching ladder:

    1. **exact** — the question, normalised (whitespace/case/punctuation), equals a
       generated statement string. Highest confidence, no guessing.
    2. **fuzzy** — token-set Jaccard vs a statement ``>= _FUZZY_JACCARD_MIN`` AND the
       statement's ``target_class`` agrees with the parsed target noun (the class guard
       stops a high-overlap-but-wrong-referent match, e.g. a "closest to the chair"
       statement whose target is a *light*, not the queried chair).
    3. **relation** — class + relation + anchor agreement: the statement's target class
       matches the question's target noun, its ``relation`` is one the question's parsed
       predicate maps to, and at least one of its anchor classes matches a question
       anchor. Returned when it pins a **unique** target id, or — when several ids share
       the mapped relation (ordinal variants like "closest"/"second closest") — when one
       statement's phrasing clearly and strictly best-matches the question; otherwise
       "ambiguous".

    ``"none"`` is returned when nothing matches — never a guess.
    """
    if not referential:
        return None, "none", "none"

    q = _norm_stmt(text)
    plan = parse_regex(text)
    tgt_noun = normalize_label(plan.target.noun) if plan.target is not None else ""

    # 1) exact normalised statement match.
    for stmt, ann in _iter_statements(referential):
        if _norm_stmt(stmt) == q:
            tid = ann.get("target_index")
            if tid is not None:
                return int(tid), "referential", "exact"

    # 2) fuzzy (Jaccard >= threshold) with a target-class guard.
    best_tid: int | None = None
    best_j = _FUZZY_JACCARD_MIN
    for stmt, ann in _iter_statements(referential):
        if tgt_noun and not _anchor_agrees(tgt_noun, str(ann.get("target_class", ""))):
            continue
        j = _jaccard(q, _norm_stmt(stmt))
        if j >= best_j:
            tid = ann.get("target_index")
            if tid is not None:
                best_j, best_tid = j, int(tid)
    if best_tid is not None:
        return best_tid, "referential", "fuzzy"

    # 3) class + relation + anchor match. Ordered relations ("closest" vs "second
    # closest"/"third closest") all share one ``relation`` string, so a plain class+
    # relation+anchor filter yields several candidate targets. We keep, per candidate
    # id, its best statement-text Jaccard vs the question, then: (a) unique id -> take
    # it; (b) multiple ids but one statement clearly closest in phrasing -> take that
    # (this is how "closest to the guitar" selects the *closest* vase, not the second/
    # third); (c) genuine tie -> "ambiguous", never guessed.
    if plan.target is None:
        return None, "ambiguous", "none"
    q_rels = set(_question_relations(plan))
    # Recurses through each anchor's nested ``disambiguator`` chain (issue #95).
    anchor_nouns = {normalize_label(a.noun) for a in iter_target_anchors(plan.target)}
    # Superlative honesty (#20): when the question carries a superlative ("closest to
    # Z" / "farthest from Z"), the discriminating constraint IS that superlative — a
    # genuine referential match must be about the anchor the superlative names, not an
    # anchor pulled in from some other, non-superlative clause. So we restrict the
    # relation and anchor filters to the superlative clause(s) ALONE: a non-superlative
    # clause's anchor no longer participates in the filter at all, so a "near the small
    # cabinet" clause can no longer stand in for "closest to the potted plant" (the
    # question's real ask). Note the per-anchor comparison below is still ``_anchor_agrees``,
    # which is unchanged and deliberately loose (head-noun-level) — this fix narrows WHICH
    # anchors are filtered on, not how strictly each is compared. That is related to but
    # distinct from the MatchTier ranking in geometry.toolbox._match_anchor_noun, which
    # grades modified-vs-bare anchor matches; we do no such grading here. With no genuine
    # speaker-to-potted-plant statement in the scene, the honest result is ambiguous.
    superl_clauses = [c for c in plan.target.clauses if _is_superlative_clause(c)]
    if superl_clauses:
        q_rels = set()
        for c in superl_clauses:
            q_rels.update(_pred_relations(getattr(c, "pred", None)))
        # Recurses through each anchor's nested ``disambiguator`` chain (issue #95).
        anchor_nouns = {
            normalize_label(a.noun) for a in iter_clause_anchors(superl_clauses)
        }
    id_best_j: dict[int, float] = {}
    id_best_union: dict[int, int] = {}
    for stmt, ann in _iter_statements(referential):
        if not _anchor_agrees(tgt_noun, str(ann.get("target_class", ""))):
            continue
        # Relation guard: when the question implies a relation, require the statement's
        # relation to be one of the mapped strings (skip when we can't tell).
        if q_rels:
            if str(ann.get("relation", "")).lower() not in q_rels:
                continue
        if anchor_nouns:
            ann_anchor_classes = [
                str(v.get("class", "")) for v in (ann.get("anchors") or {}).values()
            ]
            if not any(
                _anchor_agrees(qa, ac)
                for qa in anchor_nouns
                for ac in ann_anchor_classes
            ):
                continue
        tid = ann.get("target_index")
        if tid is None:
            continue
        # Bridge-normalise the statement noun to the question's so a whitelisted synonym
        # swap ("night stand" -> "bedside table") does not deflate the ordinal-phrasing
        # overlap the tie-break depends on.
        stmt_norm = _bridge_normalise_stmt(_norm_stmt(stmt), tgt_noun)
        j = _jaccard(q, stmt_norm)
        tid = int(tid)
        if j > id_best_j.get(tid, -1.0):
            id_best_j[tid] = j
            id_best_union[tid] = len(set(q.split()) | set(stmt_norm.split()))
    if len(id_best_j) == 1:
        return next(iter(id_best_j)), "referential", "relation"
    if id_best_j:
        # Phrasing tie-break: pick the id whose best statement most overlaps the
        # question, but only when it strictly and clearly beats the runner-up (guards
        # against picking arbitrarily among equally-worded distractors).
        ranked = sorted(id_best_j.items(), key=lambda kv: kv[1], reverse=True)
        (top_id, top_j), (_, second_j) = ranked[0], ranked[1]
        # Length-scaled margin (issue #92 fix 2). A FIXED absolute floor (the old
        # ``0.05``) is wrong: Jaccard is a token-SET ratio, so the delta a single
        # discriminating token contributes shrinks as the statement gets longer. Two
        # ordinal variants ("closest" vs "second closest") differ from each other by
        # exactly one token that is absent from the question (the word "second"):
        # adding one token neither side shares with the question changes
        #   j = I/U  ->  j' = I/(U+1)   (I = intersection size, U = union size)
        # so a genuine one-token distinction produces a margin of
        #   j - j' = I/(U*(U+1)) ~= top_j/(U+1)   (since I ~= top_j * U)
        # i.e. the SAME discriminating token is worth less absolute Jaccard on a
        # longer statement, so the floor must shrink with length too, not stay fixed.
        # We require the observed margin to clear HALF that theoretical one-token
        # delta: comfortable headroom above 0 (where every genuine phrasing tie in
        # this corpus sits exactly, since no discriminating token exists to move the
        # score) while still safely below what a real one-token ordinal distinction
        # produces (verified empirically ~1.7-2x this floor across every "closest" /
        # "second closest" pair in the training corpus). ``top_id``'s own matched
        # statement's union size against the question is the length reference, since
        # that pairing is what determines the size of a one-token nudge here.
        union = id_best_union[top_id]
        margin_required = top_j / (2.0 * (union + 1))
        if top_j >= 0.5 and top_j - second_j >= margin_required:
            return top_id, "referential", "relation"
        return None, "ambiguous", "none"
    return None, "none", "none"


def _gt_target_from_geometry(
    plan, index: SceneIndex, instances: list[InstanceRecord], thresholds: Thresholds
) -> tuple[int | None, str, str]:
    """Geometry-grounded fallback GT target (issue #92 fix 3 — corpus coverage).

    Scope, established before writing this (issue #92): 10 of the 30 training
    object-reference questions get ZERO candidates out of the referential-statement
    ladder (:func:`_gt_target_from_referential` step 3 returns ``id_best_j`` empty,
    ``target_source == "none"``) — the generated statement corpus simply never pairs
    that target class with the relation+anchor the question needs, even when the
    real physical relation holds in the scene. office_1's "potted plant on the file
    cabinet" is the flagship case: the corpus has 25 'near'-relation statements for
    class 'plant' in that scene, but every one is anchored to 'book', never
    'cabinet' — and zero 'on'-relation statements for 'plant' at all — although
    instance 55 ('potted plant') is verifiably ``on`` instance 69 ('file cabinet')
    by the exact geometry predicate the resolver itself trusts.

    Of those 10, this fallback safely resolves the subset with exactly ONE clause,
    ONE anchor (no nested disambiguator), and a predicate the resolver's own
    ``core.geometry.toolbox._BINARY_PREDS`` maps to a single physical relation
    (``on``/``near``/``above``/``under``/``with``/``in`` — NOT the broadened,
    synonym-merged relation-string family :data:`_PRED_TO_RELATIONS` uses for text
    matching, which is deliberately loose because the text corpus splits one real
    relation across several annotation strings; geometry has no such limitation, so
    using the loose family here would accept "near" as evidence of "on" and let two
    genuinely different objects tie as "winners"). This resolves 1 of the 10 in the
    training set (the office_1 case above); the rest need either ordinal ranking (5,
    which would just re-derive our own resolver's answer as "ground truth" — not an
    independent check, so deliberately out of scope), a ``between`` triple (3), or
    multiple/nested anchors (1) that would widen this fallback's honesty guarantee
    beyond what a single verified geometric fact gives us. Those remain "none",
    flagged, same as before.

    Honesty is preserved exactly like the text ladder: the target and anchor noun
    still go through :meth:`SceneIndex.by_label` (the SAME vocab bridge the rest of
    the pipeline trusts — this does not loosen or bypass it), and we only return an
    id when it is the sole target-class instance that geometrically satisfies the
    EXACT predicate against the sole matching anchor instance. Any ambiguity (more
    than one candidate anchor, more than one candidate target satisfying the
    predicate, an unmapped/ordinal/``between``/multi-anchor clause) falls through to
    ``"none"`` rather than guessing.
    """
    if plan.target is None:
        return None, "none", "none"
    clauses = plan.target.clauses
    if len(clauses) != 1 or _is_superlative_clause(clauses[0]):
        return None, "none", "none"
    clause = clauses[0]
    if clause.negated or len(clause.anchors) != 1:
        return None, "none", "none"
    anchor = clause.anchors[0]
    if anchor.disambiguator is not None:
        return None, "none", "none"
    fn = T._BINARY_PREDS.get(clause.pred)
    if fn is None or fn is T.in_ or fn is T.with_feature:
        # `in`/`with` describe containment/possession, not a support/adjacency
        # relation the referential-statement generator would encode as `on`/`near`/
        # `above` in the first place -- out of this fallback's stated scope.
        return None, "none", "none"
    anchor_matches = index.by_label(anchor.noun)
    if len(anchor_matches) != 1:
        return None, "none", "none"
    anchor_inst = anchor_matches[0]
    candidates = index.by_label(plan.target.noun)
    winners = [
        r.instance_id for r in candidates if fn(r, anchor_inst, thresholds).passed
    ]
    if len(winners) == 1:
        return winners[0], "geometry", "geometric"
    return None, "none", "none"


def score_object_reference(
    text: str,
    index: SceneIndex,
    instances: list[InstanceRecord],
    *,
    referential: dict | None = None,
    thresholds: Thresholds = DEFAULT_THRESHOLDS,
) -> ObjectRefScore:
    """Score an object-reference question: 3D IoU of our box vs the GT target box.

    Our answer: the top candidate our resolver selects on the GT index, as a
    MarkerBox. GT target, in priority order: (1) the referential-statement annotation
    matching the question (``target_source == "referential"``); (2) category
    uniqueness — if the scene holds exactly one instance of the target noun it needs
    no disambiguation and IS the target (``"unique_in_scene"``); (3) — only when (1)
    found genuinely ZERO candidates, never when it found some but couldn't disambiguate
    — a direct geometric relation check against the GT AABBs for single-anchor
    physical relations the statement corpus doesn't cover (``"geometry"``, issue #92
    fix 3; see :func:`_gt_target_from_geometry`). When none of the three pins a
    trustworthy target the IoU is left undefined (NaN) and flagged
    (``"ambiguous"``/``"none"``) rather than guessed — we never fabricate a target.
    """
    plan = parse_regex(text)
    our_marker: MarkerBox | None = None
    our_target_id: int | None = None
    unique_target_id: int | None = None
    if plan.target is not None:
        res = T.resolve(plan.target, index, thresholds)
        if res.candidates_ranked:
            our_marker = res.candidates_ranked[0].to_marker()
            our_target_id = res.candidates_ranked[0].instance_id
        # Category-level uniqueness: if the scene has exactly ONE instance of the
        # target noun, no disambiguation is possible or needed — that instance is the
        # ground-truth target unambiguously (a defensible GT independent of the
        # referential annotations).
        cat = list(index.by_label(plan.target.noun))
        if len(cat) == 1:
            unique_target_id = cat[0].instance_id

    gt_id, source, method = _gt_target_from_referential(text, referential, instances)
    # Prefer a referential-annotated target; else fall back to category-uniqueness;
    # else (issue #92 fix 3) a direct geometric relation check, but ONLY when the
    # statement ladder found genuinely ZERO candidates ("none") — an "ambiguous"
    # result means candidates exist but a text tie-break failed to pick one, which
    # is a different failure mode this geometric check must not paper over.
    if gt_id is None and unique_target_id is not None:
        gt_id, source, method = unique_target_id, "unique_in_scene", "unique"
    elif gt_id is None and source == "none" and plan.target is not None:
        gt_id, source, method = _gt_target_from_geometry(plan, index, instances, thresholds)
    by_id = {r.instance_id: r for r in instances}

    if our_marker is None:
        return ObjectRefScore(
            iou=0.0, our_marker=None, gt_target_id=gt_id, target_source=source,
            match_method=method, our_target_id=our_target_id,
            note="our resolver returned no candidate",
            parse_notes=plan.notes,
        )

    if gt_id is not None and gt_id in by_id:
        gt = by_id[gt_id]
        iou = aabb_iou_3d(our_marker, gt.aabb_min, gt.aabb_max)
        note = "" if method in ("exact", "unique") else f"target matched via {method}"
        return ObjectRefScore(
            iou, our_marker, gt_id, source, method,
            our_target_id=our_target_id, note=note,
            parse_notes=plan.notes,
        )

    # No trustworthy GT target: report self-IoU (1.0) but flag it clearly.
    return ObjectRefScore(
        iou=float("nan"),
        our_marker=our_marker,
        gt_target_id=None,
        target_source="ambiguous" if source != "none" else "none",
        match_method="none",
        our_target_id=our_target_id,
        note="no GT target matched; IoU undefined (flagged, not guessed)",
        parse_notes=plan.notes,
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


# --------------------------------------------------------------------------- frame fit


@dataclass
class Frame2D:
    """A 2D rigid transform (yaw ``theta`` about origin, then translation ``t``).

    Maps a point ``p`` in the *source* (sim/trajectory) frame into the *destination*
    (VLA-3D object) frame: ``R(theta) @ p + t``.
    """

    theta: float  # radians
    t: np.ndarray  # (2,)

    def apply(self, pts: np.ndarray) -> np.ndarray:
        p = np.asarray(pts, dtype=float)
        if p.ndim != 2 or p.shape[0] == 0:
            return np.empty((0, 2), dtype=float)
        p = p[:, :2]
        c, s = math.cos(self.theta), math.sin(self.theta)
        rot = np.array([[c, -s], [s, c]])
        return p @ rot.T + self.t


def _fit_translation(src: np.ndarray, dst: np.ndarray) -> Frame2D:
    """Best translation-only fit (theta=0): t = mean(dst - src)."""
    t = (dst - src).mean(axis=0)
    return Frame2D(theta=0.0, t=t)


def _fit_similarity(src: np.ndarray, dst: np.ndarray) -> Frame2D:
    """Best rigid (rotation+translation) fit of ``src`` onto ``dst`` (Umeyama, no scale).

    Needs >= 2 non-degenerate correspondence points. For 2 points this recovers the
    exact yaw + translation aligning the segment; for more it is the least-squares
    rigid fit. Falls back to translation-only when the point spread is degenerate
    (identical source points), where yaw is unidentifiable.
    """
    src = np.asarray(src, dtype=float)[:, :2]
    dst = np.asarray(dst, dtype=float)[:, :2]
    if src.shape[0] < 2:
        return _fit_translation(src, dst)
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    sc = src - mu_s
    dc = dst - mu_d
    if float((sc * sc).sum()) < 1e-9:  # source points coincide -> yaw undefined
        return _fit_translation(src, dst)
    h = sc.T @ dc  # (2, 2) covariance
    u, _s, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1.0, d]) @ u.T  # (2, 2) proper rotation
    theta = float(math.atan2(r[1, 0], r[0, 0]))
    t = mu_d - r @ mu_s
    return Frame2D(theta=theta, t=np.asarray(t, dtype=float))


def _fit_residual(frame: Frame2D, src: np.ndarray, dst: np.ndarray) -> float:
    """Mean correspondence residual (metres) after applying ``frame`` to ``src``."""
    if src.shape[0] == 0:
        return float("inf")
    mapped = frame.apply(src)
    return float(np.linalg.norm(mapped - np.asarray(dst)[:, :2], axis=1).mean())


def fit_frame(
    src: np.ndarray,
    dst: np.ndarray,
    *,
    yaw_residual_gate: float = 0.5,
) -> tuple[Frame2D, float]:
    """Fit a 2D transform from ``src`` correspondences to ``dst`` (translation, +yaw).

    Tries translation-only first; if its mean residual exceeds ``yaw_residual_gate``
    and >= 2 correspondences are available, tries the full rigid (yaw+translation) fit
    and keeps it when it lowers the residual. Returns ``(frame, mean_residual_m)``.
    """
    src = np.asarray(src, dtype=float)[:, :2]
    dst = np.asarray(dst, dtype=float)[:, :2]
    tr = _fit_translation(src, dst)
    res_tr = _fit_residual(tr, src, dst)
    if res_tr <= yaw_residual_gate or src.shape[0] < 2:
        return tr, res_tr
    sim = _fit_similarity(src, dst)
    res_sim = _fit_residual(sim, src, dst)
    if res_sim < res_tr:
        return sim, res_sim
    return tr, res_tr


def align_scene_trajectories(
    pairs: list[tuple[np.ndarray, np.ndarray | None]],
    *,
    yaw_residual_gate: float = 0.5,
) -> tuple[Frame2D | None, float | None]:
    """Fit ONE per-scene sim->object transform from trajectory endpoints (+shared start).

    ``pairs`` is one ``(gt_trajectory_xyz, goal_centroid_xy)`` per instruction-following
    question in the scene. Correspondences fed to the fit:

    * each trajectory's **endpoint** (sim frame) -> that question's **goal centroid**
      (object frame, the resolved terminal-GOTO anchor). Skipped when the goal is None.
    * a **start-start** row when >= 2 trajectories share a near-identical start point:
      their common sim start maps to the object-frame spawn. The spawn's object-frame
      position is unknown independently, so this row does not over-constrain — it is
      added only to stabilise the fit, using the mean of the endpoint-implied spawn as
      the target (a soft anchor). In practice the two endpoints already determine the
      rigid transform; start-start is a consistency check, not new information.

    Returns ``(frame, mean_residual_m)`` over the endpoint correspondences, or
    ``(None, None)`` when fewer than one usable endpoint correspondence exists.
    """
    src_ends: list[np.ndarray] = []
    dst_ends: list[np.ndarray] = []
    starts: list[np.ndarray] = []
    for traj, goal in pairs:
        t = _xy(traj)
        if t.shape[0] == 0:
            continue
        starts.append(t[0])
        if goal is not None:
            g = np.asarray(goal, dtype=float).reshape(-1)[:2]
            src_ends.append(t[-1])
            dst_ends.append(g)
    if not src_ends:
        return None, None

    src = np.asarray(src_ends, dtype=float)
    dst = np.asarray(dst_ends, dtype=float)

    if src.shape[0] == 1:
        # One endpoint => translation only (yaw unidentifiable from a single point).
        frame = _fit_translation(src, dst)
        return frame, _fit_residual(frame, src, dst)

    frame, residual = fit_frame(src, dst, yaw_residual_gate=yaw_residual_gate)
    return frame, residual


@dataclass
class InstructionScore:
    """Result of scoring one instruction-following question (two numbers)."""

    frechet_m: float
    coverage_1m: float  # fraction of GT path within 1.0 m of ours
    our_n_waypoints: int
    gt_n_waypoints: int
    frame_aligned: bool = True  # False when a frame offset was detected/uncorrected
    fit_residual_m: float | None = None  # scene-level correspondence residual, if fit
    note: str = ""


#: If the two paths' centroids are farther apart than this, we treat the frames as
#: unaligned and flag the score — the VLA-3D object CSV frame and the challenge
#: trajectory (robot/map) frame are NOT guaranteed to share an origin (observed on
#: loft: object x-extent runs to +11 m while the trajectory tops out at ~7.7 m, a
#: translation the dataset does not ship a transform for). Absolute Frechet/coverage
#: across unaligned frames is not meaningful, so we report the raw numbers but mark
#: them un-trustworthy rather than fabricate an aligned score.
_FRAME_OFFSET_FLAG_M = 2.0


#: Scene-level fit residual (m) above which we declare the frames un-alignable and
#: report Frechet/coverage as diagnostic-only rather than trustworthy accuracy.
_ALIGN_RESIDUAL_GATE_M = 1.0


def score_instruction_following(
    our_path: np.ndarray,
    trajectory_ply: os.PathLike | str,
    *,
    frame: Frame2D | None = None,
    fit_residual_m: float | None = None,
) -> InstructionScore:
    """Score our waypoint path against a GT trajectory PLY (Frechet + coverage).

    Reports two numbers, never a composite. When a scene-level ``frame`` is supplied
    (fitted by :func:`align_scene_trajectories` — maps the sim/trajectory frame into
    the VLA-3D object frame), the GT trajectory is transformed into our frame before
    scoring, so Frechet/coverage are directly comparable. ``fit_residual_m`` (the
    scene's correspondence residual) is carried through; residuals above
    :data:`_ALIGN_RESIDUAL_GATE_M` mark the scene unaligned (diagnostic-only). With no
    ``frame`` we fall back to the raw centroid-offset flag (legacy diagnostic path).
    """
    gt_raw = load_trajectory_ply(trajectory_ply)
    op = np.asarray(our_path, dtype=float)
    empty = op.ndim != 2 or op.shape[0] == 0

    if frame is not None:
        gt = frame.apply(gt_raw)  # into our (object) frame
    else:
        gt = _xy(gt_raw)

    frech = discrete_frechet(gt, our_path)
    cov = path_coverage(gt, our_path)
    note = ""
    frame_aligned = True

    if empty:
        note = "our path empty — pipeline produced no waypoints"
        frame_aligned = frame is not None and (
            fit_residual_m is None or fit_residual_m <= _ALIGN_RESIDUAL_GATE_M
        )
    elif frame is not None:
        if fit_residual_m is not None and fit_residual_m > _ALIGN_RESIDUAL_GATE_M:
            frame_aligned = False
            note = (
                f"scene frame fit residual {fit_residual_m:.2f} m > "
                f"{_ALIGN_RESIDUAL_GATE_M:.1f} m — endpoints/spawn did not co-locate "
                "under a single rigid transform; Frechet/coverage diagnostic only"
            )
        else:
            note = (
                f"aligned via fitted scene transform (residual "
                f"{fit_residual_m:.2f} m)" if fit_residual_m is not None else "aligned"
            )
    elif gt.shape[0]:
        gc = gt.mean(axis=0)
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
        gt_n_waypoints=int(gt_raw.shape[0]),
        frame_aligned=frame_aligned,
        fit_residual_m=fit_residual_m,
        note=note,
    )


# ----------------------------------------------------------------- IF rubric proxy (IF-F2)

#: Distance (m) within which a driven pose counts as "arrived" at a leg goal.
#:
#: DERIVED, not asserted (issue #70): this is
#: :func:`core.groundtruth.arrival.derived_arrival_tol_m` evaluated at the
#: NOMINAL documented residual (:data:`core.groundtruth.arrival.
#: NOMINAL_FIT_RESIDUAL_P95_M`) -- the same nominal residual
#: :data:`core.heads.instruction.ARRIVAL_TOL_M` derives from, so the two agree
#: by construction at that residual (coupling verified by
#: ``tests/groundtruth/test_arrival.py::test_scoring_and_head_agree_at_nominal``).
#:
#: This module-level constant is a NO-LIVE-RESIDUAL FALLBACK only -- it is the
#: default for :func:`score_instruction_rubric`'s ``tol`` argument and for any
#: caller not yet wired to a live per-run residual. A real battery run should
#: NOT use it blind: it should compute its OWN p95 fit residual across its
#: scenes' :func:`fit_frame`/:func:`align_scene_trajectories` results and call
#: :func:`core.groundtruth.arrival.derived_arrival_tol_m` on that live p95
#: directly, passing the result as ``tol=`` -- making the tolerance
#: per-battery-run derived rather than this fixed nominal (issue #70 Phase B
#: wiring: ``core.runner.gt_battery``).
LEG_ARRIVAL_TOL_M: float = NOMINAL_ARRIVAL_TOL_M


def leg_arrival_tol_m(fit_residual_p95_m: float, **kwargs) -> float:
    """Per-battery-run leg-arrival tolerance.

    Thin re-export of :func:`core.groundtruth.arrival.derived_arrival_tol_m` so
    a battery-run caller needs only this module to both derive ITS run's
    tolerance (from that run's own p95 fit residual across scenes) and pass
    the result as :func:`score_instruction_rubric`'s ``tol=`` -- see
    :data:`LEG_ARRIVAL_TOL_M` for why the module-level constant alone is not
    enough for a real run.
    """
    return derived_arrival_tol_m(fit_residual_p95_m, **kwargs)

#: Step (m) the driven polyline is densified to before leg-arrival checks. The v1
#: kinematic follower emits one pose per planned waypoint (no densification), so a
#: route that passes centimeters from a goal mid-segment could be scored NOT reached
#: when the bracketing waypoints are both farther than LEG_ARRIVAL_TOL_M away — issue #58.
ARRIVAL_RESAMPLE_STEP_M: float = 0.25

#: Per-violation penalty (fraction of one leg's worth of credit) subtracted from the
#: ordered-leg credit for each threading miss / avoid-capsule breach. Kept at one full
#: leg-equivalent so a forbidden-region breach or a missed corridor gate costs as much
#: as failing to reach a leg — the rubric penalises them explicitly (question_analysis
#: §2/§5: forbidden regions are scored, ordered adherence is scored).
IF_PENALTY_PER_VIOLATION: float = 1.0


#: A leg is scored PASS-BY (closest-approach within a class-conditional radius)
#: rather than STOP (fixed-tolerance arrival) when it is a VIA_NEAR leg at any
#: position, or a GOTO leg that is not the route's LAST leg. This mirrors the
#: ceiling evidence (``reports/ceiling_diagnosis_2026-07-19/classification.md``,
#: ``reports/issue59_probe.md`` terminal-vs-intermediate split: terminal legs
#: reach the GT path 83% of the time, non-terminal legs only 31%) that GT
#: demonstrators satisfy intermediate landmark mentions by driving PAST them,
#: not by stopping at a projected point — only the FINAL "stop at X" leg is a
#: genuine stop point. CORRIDOR_BETWEEN legs are left as STOP (unchanged): a
#: gate midpoint has no single "referenced instance" to measure closest-
#: approach against (two anchors), and the corridor-leg-specific failure mode
#: (issue #70's classification bucket (d), the gate-midpoint-vs-crossing-point
#: definition) is a distinct, out-of-scope-for-this-issue fix.
def _is_pass_by_leg(kind: str, index: int, n_legs: int) -> bool:
    is_terminal = index == n_legs - 1
    if kind == "via_near":
        return True
    if kind == "goto" and not is_terminal:
        return True
    return False


@dataclass
class IFLegOutcome:
    """Per-leg ordered-arrival record for the rubric-proxy score."""

    index: int
    kind: str  # "goto" | "via_near" | "corridor_between"
    goal_xy: tuple[float, float]
    reached: bool  # some driven pose came within tolerance...
    reached_in_order: bool  # ...AND after the previous ordered leg's arrival
    #: Corridor legs only: whether the driven trajectory threaded the gate
    #: (True/False). ``None`` for non-corridor legs (threading does not apply).
    threaded: bool | None = None
    #: True when this leg was scored PASS-BY (closest-approach within a
    #: class-conditional radius) rather than STOP (fixed-tolerance arrival) —
    #: see :func:`_is_pass_by_leg`.
    pass_by: bool = False
    #: The actual tolerance (m) applied to THIS leg. Equals the caller's
    #: ``tol`` for a STOP leg; equals ``instance_aabb_half_diagonal + tol`` for
    #: a PASS-BY leg with a resolvable instance AABB (parameter-free —
    #: derived from the instance's own footprint, not a new tuned constant);
    #: falls back to the caller's ``tol`` for a PASS-BY leg with no AABB
    #: supplied (old callers, :data:`None` in ``leg_instance_aabbs``).
    tol_used: float = 0.0


@dataclass
class InstructionRubricScore:
    """Rubric-proxy score of a DRIVEN instruction-following trajectory (IF-F2).

    Headline is :attr:`rubric_score` in [0, 1]: ordered per-leg arrival credit minus
    threading/avoid penalties. Fréchet/coverage are carried as SECONDARY diagnostics
    only — never in the headline (they measure shape similarity to the reference PLY,
    which the rubric does not pay for).
    """

    rubric_score: float  # HEADLINE: ordered-leg credit minus penalties, clamped [0,1]
    ordered_leg_credit: float  # fraction of legs reached IN ORDER, before penalties
    n_legs: int
    n_legs_reached_in_order: int
    leg_outcomes: list[IFLegOutcome] = field(default_factory=list)
    n_threading_legs: int = 0
    n_threading_violations: int = 0  # corridor legs the driven traj never threaded
    threading_details: list[str] = field(default_factory=list)
    n_avoid_specs: int = 0
    n_avoid_violations: int = 0  # avoid capsules the driven traj entered
    avoid_details: list[str] = field(default_factory=list)
    penalty: float = 0.0
    # secondary diagnostics (never headline)
    frechet_m: float | None = None
    coverage_1m: float | None = None
    driven_n_poses: int = 0
    note: str = ""


def _first_arrival_index(
    traj: np.ndarray, goal: tuple[float, float], tol: float, start: int
) -> int | None:
    """First index >= ``start`` where a driven pose is within ``tol`` of ``goal``."""
    if traj.ndim != 2 or traj.shape[0] == 0:
        return None
    g = np.asarray(goal, dtype=float)[:2]
    for i in range(max(start, 0), traj.shape[0]):
        if float(np.linalg.norm(traj[i, :2] - g)) <= tol:
            return i
    return None


def _densify_polyline(traj: np.ndarray, step: float) -> np.ndarray:
    """Linearly interpolate ``traj`` so no consecutive-point gap exceeds ``step``.

    Keeps every original vertex; inserted points are evenly spaced along each
    consecutive pair. Segment geometry is unchanged (each inserted point lies
    exactly on the original segment), so threading/capsule checks — which are
    segment-intersection based — see identical verdicts before/after. Single-point
    and empty trajectories pass through unchanged.
    """
    if traj.ndim != 2 or traj.shape[0] < 2 or step <= 0:
        return traj
    out_rows = [traj[0:1]]
    for i in range(traj.shape[0] - 1):
        p0 = traj[i]
        p1 = traj[i + 1]
        dist = float(np.linalg.norm(p1[:2] - p0[:2]))
        n_extra = int(math.ceil(dist / step)) - 1 if dist > step else 0
        if n_extra > 0:
            for k in range(1, n_extra + 1):
                frac = k / (n_extra + 1)
                out_rows.append((p0 + frac * (p1 - p0))[None, :])
        out_rows.append(p1[None, :])
    return np.concatenate(out_rows, axis=0)


def score_instruction_rubric(
    driven_traj: np.ndarray,
    leg_goals: list[tuple[str, tuple[float, float]]],
    *,
    corridor_gates: list[tuple[int, Gate]] | None = None,
    avoid_capsules: list[Capsule] | None = None,
    trajectory_ply: os.PathLike | str | None = None,
    frame: Frame2D | None = None,
    tol: float = LEG_ARRIVAL_TOL_M,
    leg_instance_aabbs: list[tuple[np.ndarray, np.ndarray] | None] | None = None,
) -> InstructionRubricScore:
    """Score a DRIVEN trajectory against the instruction rubric proxy (IF-F2).

    Args:
      driven_traj: (N, 2) pose stream the vehicle actually followed (object frame).
      leg_goals: ordered ``(kind, (x, y))`` per route leg — the point each ordered leg
        must be reached at, in order. Corridor legs use the gate midpoint as the goal.
      corridor_gates: ``(leg_index, Gate)`` per corridor leg, for threading checks.
      avoid_capsules: forbidden regions active for the whole traversal.
      trajectory_ply / frame: optional reference PLY (+ scene frame) for the SECONDARY
        Fréchet/coverage diagnostics only.
      leg_instance_aabbs: OPTIONAL, parallel to ``leg_goals`` (issue #70) — the
        resolved anchor's own ``(aabb_min, aabb_max)`` per leg, ``None`` where
        unresolved/unavailable. Mirrors the existing ``leg_instance_ids``
        parallel-list convention (``core.runner.gt_battery._if_rubric_geometry``).
        Compatible extension: omitted entirely (the default), every leg falls
        back to the plain STOP semantics below — old callers keep working
        unmodified. Only consumed for PASS-BY legs (see below); STOP legs
        never read it.

    Scoring:
      * (a) per-leg-kind arrival semantics (issue #70): each leg is classified
        STOP or PASS-BY by :func:`_is_pass_by_leg` (VIA_NEAR legs and
        non-terminal GOTO legs are PASS-BY; the terminal GOTO/"stop-at" leg
        and CORRIDOR_BETWEEN legs stay STOP). A STOP leg's tolerance is
        ``tol`` unchanged. A PASS-BY leg's tolerance is widened to
        ``instance_aabb_half_diagonal + tol`` when ``leg_instance_aabbs``
        supplies an AABB for it (parameter-free: derived from the instance's
        own footprint, not a new tuned constant) — modelling "the demonstrator
        drove past the landmark" as closest-approach-within-the-object's-own-
        reach rather than exact-point arrival, per the ceiling evidence (see
        :func:`_is_pass_by_leg`'s docstring). Falls back to plain ``tol`` when
        no AABB is available for a PASS-BY leg.
      * (b) ordered per-leg arrival: walk the trajectory once; a leg counts only if a
        pose reaches it within its (per-(a)) tolerance AND at/after the previous
        ordered leg's arrival index (partial credit = ordered legs reached / total
        legs). The ordered-arrival CURSOR logic itself is unchanged by (a) — only
        the per-leg tolerance radius differs.
      * (c) threading_check per corridor leg and capsule_violated per avoid spec: each
        miss/breach is a penalty of :data:`IF_PENALTY_PER_VIOLATION` leg-equivalents.
      * headline ``rubric_score`` = clamp(ordered_leg_credit - penalty/n_legs, 0, 1).
    """
    corridor_gates = corridor_gates or []
    avoid_capsules = avoid_capsules or []
    traj = np.asarray(driven_traj, dtype=float)
    if traj.ndim != 2 or traj.shape[1] < 2:
        traj = np.empty((0, 2), dtype=float)

    # Densify before arrival/threading/capsule checks so goals can't be stepped over
    # between sparse driven waypoints (issue #58).
    traj = _densify_polyline(traj, ARRIVAL_RESAMPLE_STEP_M)

    n_legs = len(leg_goals)
    outcomes: list[IFLegOutcome] = []
    cursor = 0  # ordered arrival index frontier
    n_in_order = 0
    for i, (kind, goal) in enumerate(leg_goals):
        pass_by = _is_pass_by_leg(kind, i, n_legs)
        leg_tol = tol
        if pass_by and leg_instance_aabbs is not None and i < len(leg_instance_aabbs):
            aabb = leg_instance_aabbs[i]
            if aabb is not None:
                half_diag = P.footprint_diagonal(aabb[0], aabb[1]) / 2.0
                leg_tol = half_diag + tol
        # "reached" ignores order (did we ever get there); "reached_in_order" requires
        # arrival at/after the previous ordered leg's arrival.
        any_idx = _first_arrival_index(traj, goal, leg_tol, 0)
        ordered_idx = _first_arrival_index(traj, goal, leg_tol, cursor)
        reached = any_idx is not None
        in_order = ordered_idx is not None
        if in_order:
            cursor = ordered_idx + 1
            n_in_order += 1
        outcomes.append(
            IFLegOutcome(
                index=i, kind=kind,
                goal_xy=(float(goal[0]), float(goal[1])),
                reached=reached, reached_in_order=in_order,
                pass_by=pass_by, tol_used=leg_tol,
            )
        )

    ordered_leg_credit = (n_in_order / n_legs) if n_legs else 0.0

    # (c) threading + avoid penalties over the DRIVEN trajectory.
    threading_details: list[str] = []
    n_thread_viol = 0
    threaded_by_leg: dict[int, bool] = {}
    for leg_i, gate in corridor_gates:
        ok, msg = threading_check(traj, gate)
        threaded_by_leg[leg_i] = ok
        if not ok:
            n_thread_viol += 1
            threading_details.append(f"leg {leg_i}: {msg}")
    # Fold the per-corridor-leg threading result back onto its leg outcome (leaves
    # non-corridor legs' ``threaded`` at None).
    for o in outcomes:
        if o.index in threaded_by_leg:
            o.threaded = threaded_by_leg[o.index]

    avoid_details: list[str] = []
    n_avoid_viol = 0
    for k, cap in enumerate(avoid_capsules):
        hit, pt = capsule_violated(traj, cap)
        if hit:
            n_avoid_viol += 1
            where = f" at ({pt[0]:.2f}, {pt[1]:.2f})" if pt is not None else ""
            avoid_details.append(f"avoid[{k}]: trajectory entered capsule{where}")

    penalty = IF_PENALTY_PER_VIOLATION * (n_thread_viol + n_avoid_viol)
    penalty_fraction = (penalty / n_legs) if n_legs else penalty
    rubric_score = max(0.0, min(1.0, ordered_leg_credit - penalty_fraction))

    # secondary diagnostics only.
    frech: float | None = None
    cov: float | None = None
    if trajectory_ply is not None:
        diag = score_instruction_following(traj, trajectory_ply, frame=frame)
        frech = diag.frechet_m if np.isfinite(diag.frechet_m) else None
        cov = diag.coverage_1m

    note_parts: list[str] = []
    if traj.shape[0] == 0:
        note_parts.append("driven trajectory empty — pipeline produced no motion")
    if n_thread_viol:
        note_parts.append(f"{n_thread_viol} corridor leg(s) never threaded")
    if n_avoid_viol:
        note_parts.append(f"{n_avoid_viol} avoid capsule(s) breached")
    return InstructionRubricScore(
        rubric_score=rubric_score,
        ordered_leg_credit=ordered_leg_credit,
        n_legs=n_legs,
        n_legs_reached_in_order=n_in_order,
        leg_outcomes=outcomes,
        n_threading_legs=len(corridor_gates),
        n_threading_violations=n_thread_viol,
        threading_details=threading_details,
        n_avoid_specs=len(avoid_capsules),
        n_avoid_violations=n_avoid_viol,
        avoid_details=avoid_details,
        penalty=penalty,
        frechet_m=frech,
        coverage_1m=cov,
        driven_n_poses=int(traj.shape[0]),
        note="; ".join(note_parts),
    )
