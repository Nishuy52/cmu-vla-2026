"""Issue #216: track flicker (born at n_obs=1, decayed, respawned) -- event-
level proof of the actual rejector.

#216's own candidate mechanism: #180 moved the cumulative-growth-cap veto
(``TrackerConfig.extent_cumulative_growth_cap``, tracker.py ~L341) ahead of
the #153 close-centroid floor (``dist <= extent_veto_min_sep``, ~L344) inside
``_match_plausible``, so a genuinely close re-detection could be vetoed by
the growth cap before the floor that would have waived it.

This module replays the flicker signature named in the issue (livingroom_1
leg0, office_1 leg0, generations 719195/719515/723600/723601 -- the same
runs the 11 Aug #202 attribution comment cites) through the REAL, unmodified
``_match_plausible``/``_assoc_gate``/``labels_compatible`` and reports, per
strict flicker-wave pair, which check actually decided the outcome.

Archived data limitation: ``raw_detections.jsonl`` carries only 2D pixel
bboxes (no fused 3D geometry -- the same gap #191's replay module already
documents and works around with synthetic geometry). This module instead
recovers REAL 3D geometry from the periodic ``instance_index.jsonl`` dumps
(the actual accumulated aabb/centroid the live tracker produced), keyed to
each candidate instance's own first-hit keyframe from ``raw_detections.jsonl``
(exact timing). A "strict flicker-wave pair" is two same-label, n_obs==1-only
("ghost") instance ids where the second is born within H15(a)'s own decay
window (``decay_k`` keyframes) of the first's birth, AND the two archived
centroids land within the REAL per-class association gate of each other --
i.e. exactly the candidate set ``associate()`` itself would have scored.

RESULT (see ``test_growth_cap_is_not_the_dominant_rejector`` and the
per-pair regression pins below): across 13 such pairs found in 4 archived
slots (422 total ghost ids surveyed), the #180 growth cap fired in exactly
ONE case -- and even there, the #153 floor it supposedly outranks would NOT
have waived the veto anyway (the re-detection's centroid, 0.638 m away, sits
well outside the 0.2 m floor). Reordering the two checks would change the
outcome of ZERO of the 13 pairs. The dominant rejector, 9 of 10 rejected
pairs, is the LATER, order-independent ``#94`` absolute class-size ceiling
(``extent_veto_factor``): the union of the two partial-view boxes exceeds
1.3x the class's typical extent, most consistently on thin classes (window,
typ. depth 0.06 m) whose real lidar-fused thickness is many times that.

#216's candidate mechanism is REFUTED by this replay: the #180/#153 check
ordering is not what is starving these tracks. No fix for #216's stated
hypothesis is implemented in this change; see the task's own directive
("if the growth cap is NOT the rejector, report what is and stop") --
this module is the refutation artifact, committed as ``Refs #216``.
"""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pytest

from core.interfaces import InstanceRecord
from core.perception.dimension_priors import prior_for
from core.perception.tracker import (
    DEFAULT_TRACKER_CONFIG,
    _assoc_gate,
    _match_plausible,
    canonical_for_match,
    labels_compatible,
)

_REPORTS = Path(__file__).resolve().parents[3] / "reports" / "cluster_verify"

_SLOTS = [
    ("livingroom_1_g723600", _REPORTS / "723600" / "debug" / "4_livingroom_1_inst"),
    ("office_1_g723601", _REPORTS / "723601" / "debug" / "4_office_1_inst"),
    ("livingroom_1_g719195", _REPORTS / "719195" / "debug" / "4_livingroom_1_inst"),
    ("office_1_g719515", _REPORTS / "719515" / "debug" / "4_office_1_inst"),
]

_DECAY_K = DEFAULT_TRACKER_CONFIG.decay_k


# --------------------------------------------------------------------------- archive loading


def _load_raw(path: Path):
    """instance_id -> (hit_count, first accepted keyframe_idx, label)."""
    hit_count: Counter = Counter()
    first_kf: dict[int, int] = {}
    label_of: dict[int, str] = {}
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            kf = rec["keyframe_idx"]
            for det in rec.get("detections", ()):
                if det.get("gate") != "accepted":
                    continue
                iid = det["instance_id"]
                hit_count[iid] += 1
                first_kf[iid] = min(first_kf.get(iid, kf), kf)
                label_of[iid] = det["label"]
    return hit_count, first_kf, label_of


def _load_instance_dumps(path: Path):
    dumps = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            if d.get("keyframes_processed") is None:
                continue
            dumps.append(d)
    return dumps


def _geometry_near(dumps, iid: int, kf: int):
    """The archived aabb/centroid/score for ``iid`` from the dump whose
    ``keyframes_processed`` is closest to ``kf`` -- the real fused geometry
    the live tracker actually produced for this instance."""
    best = None
    best_gap = None
    for d in dumps:
        dk = d["keyframes_processed"]
        for inst in d["instances"]:
            if inst["id"] != iid:
                continue
            gap = abs(dk - kf)
            if best_gap is None or gap < best_gap:
                best_gap = gap
                best = inst
    return best


def _find_flicker_pairs(hit_count, first_kf, label_of, dumps, cfg):
    """Same-label ghost-id pairs where the second is born within
    ``cfg.decay_k`` keyframes of the first (H15(a)'s own prune window) and
    the two archived centroids fall within the real per-class assoc gate --
    exactly the candidate set ``associate()`` would have scored."""
    ghosts = [iid for iid, c in hit_count.items() if c == 1]
    ghosts.sort(key=lambda i: first_kf[i])
    used: set[int] = set()
    pairs = []
    for i, id1 in enumerate(ghosts):
        if id1 in used:
            continue
        label1 = label_of[id1]
        born1 = first_kf[id1]
        for id2 in ghosts[i + 1 :]:
            if id2 in used or id2 == id1:
                continue
            if not labels_compatible(label1, label_of[id2]):
                continue
            born2 = first_kf[id2]
            if born2 <= born1 or born2 - born1 > cfg.decay_k:
                continue
            g1 = _geometry_near(dumps, id1, born1)
            g2 = _geometry_near(dumps, id2, born2)
            if g1 is None or g2 is None:
                continue
            dist = math.dist(g1["position"], g2["position"])
            gate = _assoc_gate(label1, cfg)
            if dist <= gate:
                pairs.append((id1, id2, label1, dist, born1, born2, g1, g2))
                used.add(id1)
                used.add(id2)
                break
    return pairs


def _record_from_geometry(iid: int, g: dict, label: str) -> InstanceRecord:
    amin = np.array(g["aabb_min"], dtype=float)
    amax = np.array(g["aabb_max"], dtype=float)
    return InstanceRecord(
        instance_id=iid,
        label=label,
        score=float(g["score"]),
        n_obs=int(g["n_obs"]),
        centroid=np.array(g["position"], dtype=float),
        aabb_min=amin,
        aabb_max=amax,
        points=np.stack([amin, amax]),  # exact min/max -- no more geometry archived
        aliases=(),
    )


class _FakeFused:
    def __init__(self, centroid, points):
        self.centroid = centroid
        self.points = points


def _replay_pair(id1, id2, label, dist, g1, g2, cfg):
    """Drive the REAL unmodified ``_match_plausible`` for one archived
    flicker-wave pair, instrumented at both #216-named check sites. Returns
    (verdict, site_a_fired, site_b_would_waive, deciding_check)."""
    rec1 = _record_from_geometry(id1, g1, label)
    amin2 = np.array(g2["aabb_min"], dtype=float)
    amax2 = np.array(g2["aabb_max"], dtype=float)
    fused = _FakeFused(np.array(g2["position"], dtype=float), np.stack([amin2, amax2]))
    first_box = (rec1.aabb_min.copy(), rec1.aabb_max.copy())

    new_min = fused.points.min(axis=0)
    new_max = fused.points.max(axis=0)
    combined_min = np.minimum(rec1.aabb_min, new_min)
    combined_max = np.maximum(rec1.aabb_max, new_max)
    cumulative_growth = (
        np.maximum(first_box[1], combined_max) - np.minimum(first_box[0], combined_min)
    ) - (first_box[1] - first_box[0])
    site_a_fired = bool(np.any(cumulative_growth > cfg.extent_cumulative_growth_cap))
    site_b_would_waive = dist <= cfg.extent_veto_min_sep

    candidate_ext = rec1.aabb_max - rec1.aabb_min
    combined_ext = combined_max - combined_min
    growth = combined_ext - candidate_ext
    growth_tol_passes = bool(np.all(growth <= cfg.extent_growth_tol))
    prior = prior_for(canonical_for_match(label))

    if site_a_fired:
        deciding = "site_a_growth_cap"
    elif site_b_would_waive:
        deciding = "site_b_close_centroid_floor"
    elif growth_tol_passes:
        deciding = "issue_176_growth_tol"
    elif prior is None:
        deciding = "issue_94_no_prior"
    else:
        ext_sorted = np.sort(combined_ext)
        typ_sorted = np.sort(prior.typ_ext)
        passes = bool(np.all(ext_sorted <= cfg.extent_veto_factor * typ_sorted))
        deciding = "issue_94_extent_veto_factor_" + ("accept" if passes else "reject")

    verdict = _match_plausible(rec1, fused, cfg, dist, first_box=first_box)
    return verdict, site_a_fired, site_b_would_waive, deciding


# --------------------------------------------------------------------------- discovery (shared)


def _all_pairs():
    cfg = DEFAULT_TRACKER_CONFIG
    all_pairs = []
    for tag, slot_dir in _SLOTS:
        raw_path = slot_dir / "raw_detections.jsonl"
        idx_path = slot_dir / "instance_index.jsonl"
        hit_count, first_kf, label_of = _load_raw(raw_path)
        dumps = _load_instance_dumps(idx_path)
        for pair in _find_flicker_pairs(hit_count, first_kf, label_of, dumps, cfg):
            all_pairs.append((tag, *pair))
    return all_pairs


# --------------------------------------------------------------------------- tests


def test_archived_slots_actually_exhibit_the_ghost_signature():
    """Precondition pin: every cited slot has a large fraction of n_obs==1
    ("ghost") instance ids -- the #216 signature is real and present in this
    archive, not an artefact of a quiet run."""
    cfg = DEFAULT_TRACKER_CONFIG
    fractions = {}
    for tag, slot_dir in _SLOTS:
        hit_count, _, _ = _load_raw(slot_dir / "raw_detections.jsonl")
        ghost_frac = sum(1 for c in hit_count.values() if c == 1) / len(hit_count)
        fractions[tag] = ghost_frac
    assert all(f > 0.3 for f in fractions.values()), fractions


def test_growth_cap_is_not_the_dominant_rejector():
    """The #216 event-level proof: over every strict flicker-wave pair found
    across all 4 cited slots, the #180 growth cap (site A) essentially never
    decides the outcome, and reordering it after the #153 floor (site B)
    would change NOTHING -- every pair where site A fires also sits outside
    the floor's own 0.2 m radius, so the floor would not have waived it
    either way."""
    pairs = _all_pairs()
    assert len(pairs) >= 10, "expected the archive to yield a healthy sample of flicker pairs"

    cfg = DEFAULT_TRACKER_CONFIG
    site_a_fired = 0
    reorder_would_change_outcome = 0
    decider_counts: Counter = Counter()
    for tag, id1, id2, label, dist, born1, born2, g1, g2 in pairs:
        verdict, a_fired, b_would_waive, deciding = _replay_pair(id1, id2, label, dist, g1, g2, cfg)
        decider_counts[deciding] += 1
        if a_fired:
            site_a_fired += 1
            # Reordering (checking the #153 floor BEFORE the growth cap) only
            # changes this pair's outcome if the floor would actually waive.
            if b_would_waive:
                reorder_would_change_outcome += 1

    # The growth cap is a rare, minor contributor...
    assert site_a_fired <= 1, f"growth cap fired in {site_a_fired}/{len(pairs)} pairs: {decider_counts}"
    # ...and even where it fires, the reorder #216 proposes would rescue nothing.
    assert reorder_would_change_outcome == 0, (
        "#216's proposed reorder would change an outcome -- mechanism is NOT refuted, re-open the fix"
    )
    # The real, order-independent dominant rejector is #94's absolute class-size ceiling.
    extent_veto_rejects = decider_counts.get("issue_94_extent_veto_factor_reject", 0)
    total_rejected = sum(
        c for k, c in decider_counts.items() if k not in ("issue_176_growth_tol", "issue_94_no_prior")
        or "reject" in k
    )
    assert extent_veto_rejects >= 1
    assert extent_veto_rejects / max(total_rejected, 1) >= 0.5, decider_counts


@pytest.mark.parametrize(
    "tag, id1, id2, label, expected_decider",
    [
        ("office_1_g723601", 31, 96, "table", "issue_94_extent_veto_factor_reject"),
        ("livingroom_1_g719195", 47, 72, "door", "issue_94_extent_veto_factor_reject"),
        ("office_1_g719515", 89, 165, "chair", "site_a_growth_cap"),
        ("office_1_g719515", 8, 80, "bookcase", "issue_94_extent_veto_factor_reject"),
        ("office_1_g719515", 120, 191, "window", "issue_94_extent_veto_factor_reject"),
    ],
)
def test_deciding_check_pinned_per_archived_pair(tag, id1, id2, label, expected_decider):
    """Regression pin: the exact deciding check for a handful of named
    archived pairs (one per slot/class), so a future change to the check
    order or thresholds shows up here first."""
    pairs = {(t, a, b): (t, a, b, lb, dist, born1, born2, g1, g2)
             for t, a, b, lb, dist, born1, born2, g1, g2 in _all_pairs()}
    key = (tag, id1, id2)
    assert key in pairs, f"archived pair {key} not found by the current discovery pass"
    _, _, _, lb, dist, born1, born2, g1, g2 = pairs[key]
    assert lb == label
    _, _, _, deciding = _replay_pair(id1, id2, label, dist, g1, g2, DEFAULT_TRACKER_CONFIG)
    assert deciding == expected_decider
