"""Issue #217: the #94 extent veto rejects thin-class re-detections -- two
widening designs tried, both refuted. #217 is DROPPED; this module is the
refutation-and-drop evidence record (mirrors test_tracker_issue_216.py's
"proof, not blind fix" convention), not a fix. ``core/perception/tracker.py``
and ``core/perception/dimension_priors.py`` are UNCHANGED from main.

THE MECHANISM (confirmed, #216): the #94 absolute-ceiling check
(``extent_veto_factor x typ_ext``, one flat ratio for every class, every
sorted axis) is what starves the #216 flicker instances, 9 of 10 rejections
in the #216 replay -- not the #180 check-order hypothesis #216 first tested.

DESIGN 1 (tried, reverted): widen every sorted rank by the larger of an
absolute allowance and the class's recorded ``ClassPrior.cap_factor`` (issue
#201's P95/median metadata). REFUTED by a real-GT probe (every same-class
pair from ``data/vla3d/Unity/*/*_object_result.csv``, both configs, through
the real ``_match_plausible``): 19 of 139 genuinely-distinct in-gate real
pairs flipped from correctly-rejected to wrongly admitted, 10 of them pillow
pairs (a top-frequency question target). Root cause:
``test_design_1_would_widen_mid_and_long_ranks_not_just_thin`` below pins
why -- cap_factor widens MID/LONG ranks too (e.g. sofa's long rank +36.8%),
and two DISTINCT same-class objects at typical spacing overwhelmingly union
on a mid/long rank (exactly what #94/#161 exist to catch).

DESIGN 2 (tried, reverted): widen ONLY the thin (smallest sorted) rank, to a
physical floor (``max(extent_veto_factor * typ_ext[0], THIN_RANK_FLOOR_M)``),
mid/long UNCHANGED, no cap_factor. Narrower and physically motivated (two
partial views of one flat surface union on the thin/depth axis; two distinct
objects union on mid/long). Still REFUTED by the same real-GT probe
(``test_design_2_at_the_safe_floor_admits_none_of_the_9`` below): a floor low
enough to keep the probe's flip count at 0 (<= ~0.10 m -- the binding
constraint is a genuine pair of vertically-stacked office_2 windows whose
real thin-axis union is 0.104 m) admits NONE of the #216 replay's 9 flagged
rejections: 7 of the 9 fail on a MID or LONG rank the design deliberately
never touches (unrescuable regardless of floor), and the 2 that fail on thin
alone need >= 0.475 m -- well past the real-GT probe's safe zone (which
starts flipping distinct pillow and door pairs at 0.475-0.50 m, and a mirror
pair by 0.858 m, the floor a bookcase pair would additionally need).

CONCLUSION: no single per-class-agnostic threshold on AABB extent (ratio,
absolute, or physical floor) separates "two views of one thin object" from
"two distinct same-class objects at typical spacing" -- the archived #216
flicker geometry's own thin-axis unions (0.45-0.86 m) are NOT smaller than
genuine distinct-object unions found in real GT data (0.10 m and up); the two
distributions overlap. Recommended next step (out of this module's scope): a
geometry-aware redesign that does not rely on AABB extent alone -- e.g. an
orientation-aware thin-axis measurement, or a same-pose/near-identical-view
fast path -- tracked as its own follow-up issue, not a blind retry of #217's
same family of fixes.
"""
from __future__ import annotations

import csv
import glob
import os
from pathlib import Path

import numpy as np
import pytest

from core.interfaces import InstanceRecord
from core.perception.dimension_priors import prior_for
from core.perception.tracker import (
    DEFAULT_TRACKER_CONFIG,
    TrackerConfig,
    _assoc_gate,
    _match_plausible,
    canonical_for_match,
    labels_compatible,
)

from tests.perception.test_tracker_issue_216 import _all_pairs

_DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "vla3d" / "Unity"


# --------------------------------------------------------------------------- design 1: why it was unsafe


def test_design_1_would_widen_mid_and_long_ranks_not_just_thin():
    """Pins WHY design 1 (per-axis cap_factor widening, tried and reverted)
    was unsafe: its own recorded metadata widens ranks OTHER than thin for
    ordinary common classes -- sofa's LONG rank (cap_factor 1.778, +36.8% over
    the plain #94 ratio: 2.76 m -> 3.77 m) and window's MID rank (cap_factor
    5.083) -- exactly the ranks a genuinely distinct same-class pair unions
    on. Design 2 (below) never reads cap_factor at all."""
    sofa = prior_for("sofa")
    assert sofa.cap_factor[2] > 1.3  # long rank: real recorded variance signal
    old_long_bound = 1.3 * sofa.typ_ext[2]
    widened_long_bound = sofa.cap_factor[2] * sofa.typ_ext[2]
    growth_pct = (widened_long_bound / old_long_bound - 1.0) * 100.0
    assert growth_pct == pytest.approx(36.8, abs=0.5)
    assert old_long_bound == pytest.approx(2.76, abs=0.01)
    assert widened_long_bound == pytest.approx(3.77, abs=0.01)

    window = prior_for("window")
    assert window.cap_factor[1] > 1.3  # mid rank: also real recorded variance


# --------------------------------------------------------------------------- design 2: the standalone bound (refutation evidence only)


def _design_2_bound(prior, floor: float) -> np.ndarray:
    """Design 2's per-axis ceiling, reconstructed HERE only (not in
    tracker.py -- #217 is dropped). Thin rank widened to a physical floor;
    mid/long keep the exact, unmodified #94 ratio."""
    typ = prior.typ_ext
    bound = 1.3 * typ  # TrackerConfig.extent_veto_factor's default, unchanged
    bound = bound.copy()
    bound[0] = max(bound[0], floor)
    return bound


def _design_2_verdict(cand_min, cand_max, other_min, other_max, dist: float, prior, floor: float) -> bool:
    """A standalone re-implementation of ``_match_plausible``'s control flow
    with ONLY the last-resort ceiling swapped for :func:`_design_2_bound` --
    everything else (the #180 growth cap is out of scope here since these are
    single-observation candidates with no chain history, the #153 floor, the
    #176 growth-relative rescue) matches tracker.py's real, unmodified
    ``_match_plausible`` exactly."""
    combined_min = np.minimum(cand_min, other_min)
    combined_max = np.maximum(cand_max, other_max)
    if dist <= 0.2:  # TrackerConfig.extent_veto_min_sep default
        return True
    growth = (combined_max - combined_min) - (cand_max - cand_min)
    if bool(np.all(growth <= 0.25)):  # TrackerConfig.extent_growth_tol default
        return True
    if prior is None:
        return True
    combined_ext = np.sort(combined_max - combined_min)
    return bool(np.all(combined_ext <= _design_2_bound(prior, floor)))


# --------------------------------------------------------------------------- real-GT probe (the acceptance oracle for both designs)


def _load_scene_objects(csv_path: Path):
    objs = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                cx = float(row["object_bbox_cx"])
                cy = float(row["object_bbox_cy"])
                cz = float(row["object_bbox_cz"])
                dx = float(row["object_bbox_xlength"])
                dy = float(row["object_bbox_ylength"])
                dz = float(row["object_bbox_zlength"])
            except (ValueError, KeyError):
                continue
            if min(dx, dy, dz) <= 0:
                continue
            centroid = np.array([cx, cy, cz])
            half = np.array([dx, dy, dz]) / 2.0
            objs.append((row["object_id"], row["raw_label"], centroid, centroid - half, centroid + half))
    return objs


def _real_gt_in_gate_pairs():
    """Every same-class pair, every VLA-3D Unity scene, whose real GT
    centroids fall within the real per-class association gate -- the exact
    candidate set ``associate()`` itself would score."""
    pairs = []
    for csv_path in sorted(glob.glob(os.path.join(str(_DATA_DIR), "*", "*_object_result.csv"))):
        scene = os.path.basename(os.path.dirname(csv_path))
        objs = _load_scene_objects(Path(csv_path))
        for i in range(len(objs)):
            for j in range(i + 1, len(objs)):
                oid_a, label_a, ca, amin_a, amax_a = objs[i]
                oid_b, label_b, cb, amin_b, amax_b = objs[j]
                if not labels_compatible(label_a, label_b):
                    continue
                dist = float(np.linalg.norm(ca - cb))
                gate = _assoc_gate(label_a, DEFAULT_TRACKER_CONFIG)
                if dist > gate:
                    continue
                pairs.append((scene, oid_a, label_a, ca, amin_a, amax_a, oid_b, label_b, cb, amin_b, amax_b, dist))
    return pairs


@pytest.fixture(scope="module")
def real_gt_pairs():
    pairs = _real_gt_in_gate_pairs()
    if len(pairs) < 50:
        pytest.skip("VLA-3D Unity data checkout not present or too small for this probe")
    return pairs


def _flips_at_floor(pairs, floor: float):
    """Every real-GT pair (either candidate direction) that the plain #94
    ratio correctly rejects but design 2's floor wrongly admits."""
    flips = []
    for scene, oid_a, label_a, ca, amin_a, amax_a, oid_b, label_b, cb, amin_b, amax_b, dist in pairs:
        prior = prior_for(canonical_for_match(label_a))
        if prior is None:
            continue
        for cand_min, cand_max, other_min, other_max in (
            (amin_a, amax_a, amin_b, amax_b),
            (amin_b, amax_b, amin_a, amax_a),
        ):
            v_old = _design_2_verdict(cand_min, cand_max, other_min, other_max, dist, prior, 0.0)
            v_new = _design_2_verdict(cand_min, cand_max, other_min, other_max, dist, prior, floor)
            if (not v_old) and v_new:
                flips.append((scene, oid_a, oid_b, label_a, dist))
                break
    return flips


def test_real_gt_probe_finds_a_substantial_candidate_set(real_gt_pairs):
    """Precondition pin: the probe surfaces a real, substantial in-gate
    same-class candidate set across the 15 VLA-3D scenes, not a near-empty
    accidental pass."""
    assert len(real_gt_pairs) > 100


def test_design_2_safe_floor_has_zero_real_gt_flips(real_gt_pairs):
    """0.10 m is the largest thin-rank floor with zero real-GT flips. Above
    it (0.104 m+), a genuine pair of vertically-stacked office_2 windows
    (real centroid distance 0.691 m, thin-axis union 0.104 m) flips first."""
    assert _flips_at_floor(real_gt_pairs, 0.10) == []
    flips_just_above = _flips_at_floor(real_gt_pairs, 0.11)
    assert len(flips_just_above) >= 1
    assert any(f[3] == "window" for f in flips_just_above)


def test_design_2_at_the_safe_floor_admits_none_of_the_9():
    """The decisive #217 acceptance check: at the largest thin-rank floor
    that keeps the real-GT probe clean (0.10 m), design 2 admits ZERO of the
    #216 replay's 9 flagged rejections -- the "useful fraction" this design
    would need to justify itself never materializes at a safe floor."""
    safe_floor = 0.10
    admitted = []
    for tag, id1, id2, label, dist, born1, born2, g1, g2 in _all_pairs():
        prior = prior_for(canonical_for_match(label))
        if prior is None:
            continue
        cand_min = np.array(g1["aabb_min"], dtype=float)
        cand_max = np.array(g1["aabb_max"], dtype=float)
        other_min = np.array(g2["aabb_min"], dtype=float)
        other_max = np.array(g2["aabb_max"], dtype=float)
        v_old = _design_2_verdict(cand_min, cand_max, other_min, other_max, dist, prior, 0.0)
        v_new = _design_2_verdict(cand_min, cand_max, other_min, other_max, dist, prior, safe_floor)
        if (not v_old) and v_new:
            admitted.append((id1, id2, label))
    assert admitted == [], (
        f"expected 0 admitted at the safe floor, got {admitted} -- update the module "
        "docstring's conclusion if this ever changes"
    )


def test_design_2_needs_an_unsafe_floor_to_admit_even_two_of_the_9():
    """Names the floor where design 2 first admits ANYTHING from the #216
    replay (0.50 m, two window pairs, thin-only failures) and confirms that
    floor already reopens real-GT flips well past "near zero" -- pillow and
    door pairs among them, the exact failure class the design-1 verifier
    first flagged."""
    unsafe_floor = 0.50
    admitted = []
    for tag, id1, id2, label, dist, born1, born2, g1, g2 in _all_pairs():
        prior = prior_for(canonical_for_match(label))
        if prior is None:
            continue
        cand_min = np.array(g1["aabb_min"], dtype=float)
        cand_max = np.array(g1["aabb_max"], dtype=float)
        other_min = np.array(g2["aabb_min"], dtype=float)
        other_max = np.array(g2["aabb_max"], dtype=float)
        v_old = _design_2_verdict(cand_min, cand_max, other_min, other_max, dist, prior, 0.0)
        v_new = _design_2_verdict(cand_min, cand_max, other_min, other_max, dist, prior, unsafe_floor)
        if (not v_old) and v_new:
            admitted.append((id1, id2, label))
    assert len(admitted) == 2
    assert {label for _, _, label in admitted} == {"window"}


def test_design_2_unsafe_floor_flips_pillow_and_door_pairs(real_gt_pairs):
    """Confirms the trade-off named above directly: at 0.50 m (needed for ANY
    #216 benefit), the real-GT probe already flips pillow and door pairs --
    not "near zero", the same failure class design 1's verifier flagged."""
    flips = _flips_at_floor(real_gt_pairs, 0.50)
    labels = {f[3] for f in flips}
    assert len(flips) >= 5
    assert "pillow" in labels
    assert "door" in labels


# --------------------------------------------------------------------------- current code is untouched


def test_current_tracker_config_has_no_217_knob():
    """#217 is dropped: TrackerConfig carries none of either design's fields."""
    import dataclasses

    field_names = {f.name for f in dataclasses.fields(TrackerConfig)}
    assert "extent_veto_abs_slack_m" not in field_names
    assert "extent_veto_use_cap_factor" not in field_names
    assert "extent_veto_thin_rank_floor_m" not in field_names


def test_current_match_plausible_is_the_plain_94_ratio():
    """The live ``_match_plausible`` ceiling is exactly ``extent_veto_factor x
    typ_ext`` -- no widening, matching main pre-#217."""
    from core.perception.tracker import _match_plausible as real_match_plausible

    prior = prior_for("window")
    cand_min = np.array([0.0, 0.0, 0.0])
    cand_max = np.array([0.05, 0.8, 1.6])
    cand = InstanceRecord(
        instance_id=0, label="window", score=0.5, n_obs=1,
        centroid=(cand_min + cand_max) / 2.0, aabb_min=cand_min, aabb_max=cand_max,
    )
    # An incoming detection whose union pushes the thin axis to 0.55 m (past
    # both the #176 growth-relative rescue, tol 0.25, and the plain #94
    # ceiling) -- under either dropped design's floor (0.30/0.45 m) this
    # would have been admitted; under the current (dropped) code it must be
    # rejected, matching the plain 1.3x typ_ext ceiling (0.078 m).
    other_min = np.array([-0.50, 0.0, 0.0])
    other_max = np.array([0.05, 0.8, 1.6])

    class _F:
        points = np.stack([other_min, other_max])

    dist = 0.3  # past the #153 floor
    verdict = real_match_plausible(cand, _F(), TrackerConfig(), dist, first_box=None)
    assert verdict is False
