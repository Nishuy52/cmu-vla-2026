"""Per-class dimension priors (H12 / OR-F6): min-clamp, under-obs inflate, GT passthrough.

Issue #201: a marker-time upper cap (grow-AND-shrink) was designed, implemented in
three successively refined versions, measured against real GT instances each time,
and REJECTED before shipping -- clamp_extents stays GROW-ONLY. See the module
docstring in core.perception.dimension_priors for the full history. The recorded
per-class per-axis cap-factor metadata (_CAP_FACTOR / ClassPrior.cap_factor) is
kept as diagnosis data; this file both verifies that metadata loads correctly AND
pins that nothing in the live clamp path reads it.
"""
from __future__ import annotations

import ast
import inspect

import numpy as np

from core.interfaces import InstanceRecord
from core.perception.dimension_priors import (
    _CAP_FACTOR,
    _DATA_PRIORS,
    _HAND_PRIORS,
    _PHRASE_DATA_PRIORS,
    ABS_MIN_EXTENT_FALLBACK_M,
    DEGENERATE_EXTENT_M,
    FUSE_EXTENT_CAP_FACTOR,
    UNDEROBS_MAX_N_OBS,
    clamp_extents,
    clamp_record_marker,
    floor_degenerate_aabb,
    prior_for,
)
from core.perception.scene_index import BasicSceneIndex


def _rec(label, extent, *, n_obs=3, centroid=(0.0, 0.0, 0.0)):
    c = np.array(centroid, dtype=float)
    half = np.array(extent, dtype=float) / 2.0
    return InstanceRecord(
        instance_id=1,
        label=label,
        score=0.9,
        n_obs=n_obs,
        centroid=c,
        aabb_min=c - half,
        aabb_max=c + half,
    )


# --------------------------------------------------------------- table sanity

def test_priors_cover_data_and_hand_classes():
    # data-derived seed (VLA-3D) plus the seven hand-curated absent classes.
    assert len(_DATA_PRIORS) >= 60
    assert set(_HAND_PRIORS) == {
        "elephant", "fan", "fossil", "horse", "projector", "pyramid", "sphere",
    }
    # data classes and hand classes are disjoint (data wins if ever overlapping).
    assert not (set(_DATA_PRIORS) & set(_HAND_PRIORS))


def test_prior_for_normalises_label():
    # plural / synonym fold through normalize_label into the same prior.
    assert prior_for("chairs") is prior_for("chair")
    assert prior_for("nonexistent-class") is None


def test_prior_min_le_typical_every_class():
    for label, (mn, tp) in {**_DATA_PRIORS, **_PHRASE_DATA_PRIORS, **_HAND_PRIORS}.items():
        assert all(m <= t for m, t in zip(mn, tp)), label


# --------------------------------------------------------------- issue #201: five missing priors

def test_five_previously_missing_priors_now_load():
    # #118 measured these five challenge-vocabulary classes with NO prior at all.
    for label in (
        "bedside table", "fossil decoration", "potted plant", "paper cup", "beer bottle",
    ):
        assert prior_for(label) is not None, label


def test_five_new_priors_have_plausible_magnitudes():
    # Sanity band: no axis under 2 cm (degenerate) or over 3 m (would swallow a room).
    for label in (
        "bedside table", "fossil decoration", "potted plant", "paper cup", "beer bottle",
    ):
        prior = prior_for(label)
        assert np.all(prior.min_ext > 0.02), label
        assert np.all(prior.typ_ext < 3.0), label
        assert np.all(prior.min_ext <= prior.typ_ext), label
    # potted plant (n=57) typical extents should land in real-plant-scale territory.
    p = prior_for("potted plant")
    assert 0.1 < p.typ_ext[0] < 1.0
    assert 0.3 < p.typ_ext[-1] < 2.0
    # beer bottle / paper cup are small handheld items.
    for label in ("beer bottle", "paper cup"):
        prior = prior_for(label)
        assert prior.typ_ext[-1] < 0.5, label


# --------------------------------------------------------------- min-clamp

def test_thin_underbox_clamped_to_class_min():
    # A pillow seen edge-on: near-zero depth. Class-min thin axis is ~0.187 m.
    thin = _DATA_PRIORS["pillow"][0][0]
    out = clamp_extents(np.array([0.01, 0.30, 0.40]), "pillow", n_obs=3)
    # thinnest axis lifted to at least the class-min.
    assert min(out) >= thin - 1e-9


def test_clamp_preserves_axis_assignment():
    # thin axis is y here; after clamp the y axis (not x) should be the one grown.
    # x=0.90 is above every pillow class-min rank, so it must stay put.
    out = clamp_extents(np.array([0.90, 0.01, 0.50]), "pillow", n_obs=3)
    assert out[1] >= _DATA_PRIORS["pillow"][0][0] - 1e-9  # thin y axis lifted
    assert out[0] == 0.90  # largest axis unchanged (already above its rank min)


def test_grossly_oversized_axis_is_not_capped_grow_only_by_design():
    # Issue #201: a marker-time upper cap (grow-AND-shrink) was built, measured
    # against real GT instances, and REJECTED -- it clipped 10-21% of real
    # instances on some axis for every high-variance class measured (table,
    # window, sofa, shelf, chair) against a measured score upside of only +0.04
    # points for the whole cap concept (see core.perception.dimension_priors's
    # module docstring, "Issue #201" section, for the full clip-rate table and
    # the decision). x here (1.50 m) is far past what any of the three rejected
    # cap designs would have allowed for pillow's long rank -- it must still pass
    # through UNTOUCHED: clamp_extents is grow-only, y (thin, under min) is the
    # only axis that moves.
    out = clamp_extents(np.array([1.50, 0.01, 0.50]), "pillow", n_obs=3)
    prior = prior_for("pillow")
    assert out[1] >= prior.min_ext[0] - 1e-9  # thin y axis lifted toward min
    assert out[0] == 1.50  # oversized x axis passes through unchanged, NOT capped


def test_no_prior_returns_input_unchanged():
    ext = np.array([0.02, 0.02, 0.02])
    out = clamp_extents(ext, "nonexistent-class", n_obs=1)
    assert np.array_equal(out, ext)


# --------------------------------------------------------------- GT passthrough (named invariant)

def test_gt_perfect_box_passes_through_unchanged():
    """A well-observed box already at typical extents is returned identical.

    GT battery instances are n_obs=3 and built from GT geometry; the clamp must be a
    no-op on them so both H12 and the GT battery are unaffected by default.
    """
    typ = _DATA_PRIORS["chair"][1]  # typical (thin, mid, long)
    ext = np.array(typ)
    out = clamp_extents(ext, "chair", n_obs=3)
    assert np.allclose(out, ext)


def test_well_observed_thin_box_not_inflated():
    # thin axis below class-min but n_obs high => min-clamp only, NO inflate to typical.
    prior = prior_for("pillow")
    out = clamp_extents(np.array([0.01, 0.30, 0.40]), "pillow", n_obs=5)
    # clamped up to min, but not all the way to typical.
    assert min(out) < prior.typ_ext[0]
    assert min(out) >= prior.min_ext[0] - 1e-9


def test_underobserved_thin_axis_inflated_toward_typical():
    # single-view, thin front-shell => inflate least-observed axis to class-typical.
    prior = prior_for("pillow")
    out = clamp_extents(np.array([0.01, 0.30, 0.40]), "pillow", n_obs=1)
    assert min(out) >= prior.typ_ext[0] - 1e-9


def test_inflate_only_up_to_max_n_obs():
    prior = prior_for("pillow")
    at = clamp_extents(np.array([0.01, 0.30, 0.40]), "pillow", n_obs=UNDEROBS_MAX_N_OBS)
    above = clamp_extents(np.array([0.01, 0.30, 0.40]), "pillow", n_obs=UNDEROBS_MAX_N_OBS + 1)
    assert min(at) >= prior.typ_ext[0] - 1e-9      # still inflated at the boundary
    assert min(above) < prior.typ_ext[0]           # not inflated just above it


# --------------------------------------------------------------- degenerate floor (#125)

def test_degenerate_axis_raised_to_prior_floor():
    # zero-thickness z axis (e.g. a single lidar sweep plane) on a chair.
    lo = np.array([0.0, 0.0, 0.0])
    hi = np.array([0.6, 0.7, 0.0])
    new_lo, new_hi = floor_degenerate_aabb(lo, hi, "chair")
    ext = new_hi - new_lo
    assert ext[2] >= DEGENERATE_EXTENT_M
    # thin-rank floor for chair is the prior's smallest sorted-axis minimum.
    prior = prior_for("chair")
    assert ext[2] >= prior.min_ext.min() - 1e-9
    # healthy axes untouched.
    assert new_lo[0] == 0.0 and new_hi[0] == 0.6
    assert new_lo[1] == 0.0 and new_hi[1] == 0.7


def test_degenerate_floor_preserves_centroid():
    lo = np.array([1.0, -2.0, 0.5])
    hi = np.array([1.6, -1.3, 0.5])  # z is degenerate, centred at 0.5
    new_lo, new_hi = floor_degenerate_aabb(lo, hi, "chair")
    centre = (new_lo + new_hi) / 2.0
    assert np.allclose(centre, [(1.0 + 1.6) / 2.0, (-2.0 + -1.3) / 2.0, 0.5])


def test_healthy_box_is_untouched_by_floor():
    lo = np.array([0.0, 0.0, 0.0])
    hi = np.array([0.6, 0.7, 0.8])  # every axis well above the threshold
    new_lo, new_hi = floor_degenerate_aabb(lo, hi, "chair")
    assert np.array_equal(new_lo, lo)
    assert np.array_equal(new_hi, hi)


def test_no_prior_class_uses_absolute_fallback():
    lo = np.array([0.0, 0.0, 0.0])
    hi = np.array([0.3, 0.3, 0.0])
    new_lo, new_hi = floor_degenerate_aabb(lo, hi, "nonexistent-class")
    ext = new_hi - new_lo
    assert ext[2] == ABS_MIN_EXTENT_FALLBACK_M
    assert ext[2] >= DEGENERATE_EXTENT_M


def test_thin_but_not_degenerate_axis_left_alone():
    # 3 cm axis is above the 2 cm degenerate threshold: not touched, even though it
    # is below several classes' prior min -- this is NOT the general clamp.
    lo = np.array([0.0, 0.0, 0.0])
    hi = np.array([0.03, 0.7, 0.8])
    new_lo, new_hi = floor_degenerate_aabb(lo, hi, "chair")
    assert new_hi[0] - new_lo[0] == 0.03


def test_multiple_degenerate_axes_each_raised():
    lo = np.array([0.0, 0.0, 0.0])
    hi = np.array([0.0, 0.0, 0.8])
    new_lo, new_hi = floor_degenerate_aabb(lo, hi, "chair")
    ext = new_hi - new_lo
    assert ext[0] >= DEGENERATE_EXTENT_M
    assert ext[1] >= DEGENERATE_EXTENT_M
    assert ext[2] == 0.8


# --------------------------------------------------------------- marker seam

def test_clamp_record_marker_preserves_centre():
    rec = _rec("pillow", (0.01, 0.30, 0.40), n_obs=1, centroid=(2.0, -1.0, 0.5))
    m = clamp_record_marker(rec)
    assert (m.cx, m.cy, m.cz) == (2.0, -1.0, 0.5)
    assert m.label == "pillow"
    # extents inflated relative to the raw box.
    assert m.sx >= 0.01  # thin x axis lifted


def test_scene_index_marker_for_matches_clamp():
    rec = _rec("pillow", (0.01, 0.30, 0.40), n_obs=1)
    idx = BasicSceneIndex([rec])
    m = idx.marker_for(rec)
    ext = clamp_extents(rec.extents, rec.label, rec.n_obs)
    assert np.allclose([m.sx, m.sy, m.sz], ext)


def test_gt_record_marker_equals_raw_marker():
    # GT-perfect instance: marker_for == to_marker (clamp inert).
    typ = _DATA_PRIORS["chair"][1]
    rec = _rec("chair", typ, n_obs=3, centroid=(1.0, 2.0, 0.0))
    m = clamp_record_marker(rec)
    raw = rec.to_marker()
    assert np.allclose([m.sx, m.sy, m.sz], [raw.sx, raw.sy, raw.sz])
    assert (m.cx, m.cy, m.cz) == (raw.cx, raw.cy, raw.cz)


# --------------------------------------------------------------- issue #201: upper cap
# MEASURED AND REJECTED. clamp_extents stays GROW-ONLY.
#
# Three cap designs were built and measured against real GT instances in turn, and
# every one clipped a material fraction of otherwise-correct real answer boxes at
# the scored marker path, against a measured score upside (the #118 diagnosis that
# motivated this whole issue, applying a two-sided clamp directly to the published
# marker) of only +0.04 points:
#
#   design                         | worst measured any-axis real-instance clip rate
#   ------------------------------ | -------------------------------------------------
#   1. flat 1.5x, every class      | 48.8% (window), 36.8% (table), on the long axis
#   2. one scalar per class        | 27.9% (window), any axis
#   3. one triple per class+axis   | 10-21% (table/window/sofa/shelf/chair), any axis
#
# Negative expected value at every tier; the underlying oversize error is a FUSION
# defect and is treated at its cause (#199's component selection and outlier-core
# filtering), not papered over at the marker seam. See the module docstring in
# core.perception.dimension_priors, "Issue #201" section, for the full history.
#
# The per-class, per-axis cap-factor RESEARCH (:data:`_CAP_FACTOR`,
# :attr:`ClassPrior.cap_factor`) is kept as recorded diagnosis metadata -- it
# still loads correctly (verified below) -- but :func:`clamp_extents` (the
# marker/answer-time clamp) must never read it, which the source-inspection
# test below pins directly against the function body, not just its observed
# behaviour. Issue #217 gave this metadata its first sanctioned consumer
# outside this module: core.perception.tracker._extent_veto_bound, which
# widens the tracker's own association-time #94 extent veto from it (see
# tests/perception/test_tracker_issue_217.py) -- a DIFFERENT seam from the one
# this file's tests guard, so the guard below is unchanged, not weakened.

def test_cap_factor_metadata_still_loads_as_a_per_axis_triple():
    # Recorded diagnosis data, NOT a live parameter (see section note above): every
    # _DATA_PRIORS / _PHRASE_DATA_PRIORS class still carries a (thin, mid, long)
    # cap factor triple, each rank in [1.5, 6.0], for any future fusion-side use.
    for label in {**_DATA_PRIORS, **_PHRASE_DATA_PRIORS}:
        prior = prior_for(label)
        assert prior.cap_factor is not None, label
        assert prior.cap_factor.shape == (3,), label
        assert np.all((1.5 <= prior.cap_factor) & (prior.cap_factor <= 6.0)), label

    # _HAND_PRIORS classes (no GT distribution) get NO cap_factor at all -- there
    # was never a distribution to measure one from.
    for label in _HAND_PRIORS:
        prior = prior_for(label)
        assert prior.cap_factor is None, label

    # A class with no prior at all: cap_factor is moot (prior_for returns None).
    assert prior_for("nonexistent-class") is None


def test_cap_factor_table_matches_prior_for():
    # _CAP_FACTOR is the single source of truth prior_for() reads the metadata from.
    for label, expected in _CAP_FACTOR.items():
        assert np.allclose(prior_for(label).cap_factor, expected), label


def test_clamp_extents_never_reads_cap_factor():
    """Source-inspection pin (issue #201): clamp_extents must NEVER apply the
    recorded cap_factor metadata to a box, however tempting re-adding a shrink
    step might look later. Walks clamp_extents' own AST for any ``.cap_factor``
    attribute access -- independent of comments/docstrings, so a future edit
    that quietly re-wires the function trips this even if nobody updates prose.
    If this test ever needs to change, re-read the rejection history in the
    module docstring first: a marker-time upper cap was measured at 10-21%
    any-axis clipping of real GT instances for every high-variance class, against
    a measured +0.04 point score upside for the whole cap concept, and rejected.
    """
    src = inspect.getsource(clamp_extents)
    tree = ast.parse(src)
    hits = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and node.attr == "cap_factor"
    ]
    assert not hits, "clamp_extents reads .cap_factor -- issue #201 rejected this design"


def test_grossly_oversized_low_variance_class_box_passes_through_uncapped():
    """A marker-time upper cap for nightstand (a genuinely LOW-variance class,
    every rank of the recorded cap_factor sits at the 1.5 floor) was measured
    against real GT instances and rejected along with every other class's cap
    (see the section note above: 10-21% any-axis clip rate on high-variance
    classes, +0.04 points total upside). #118 measured 11 of 14 live boxes too
    LARGE (median ~4x GT volume); a grossly over-fused nightstand (each axis 4x
    typical) is exactly that failure mode, and it now passes through
    clamp_extents completely UNTOUCHED -- this is the deliberate, documented
    absence of a fix, not an oversight.
    """
    prior = prior_for("nightstand")
    assert np.allclose(prior.cap_factor, 1.5)  # precondition: recorded floor on every rank
    typ = prior.typ_ext
    bloated = typ * 4.0
    out = clamp_extents(bloated, "nightstand", n_obs=3)
    assert np.allclose(np.sort(out), np.sort(bloated))  # unchanged -- NOT capped


def test_grossly_oversized_high_variance_class_box_also_passes_through_uncapped():
    """Same as above for chair -- a genuine 2.93 m-long "chair"-labelled GT asset
    repeats 6x in home_building_1, pushing chair's recorded long-axis cap_factor
    to 2.794 (higher than the rejected flat-1.5x design's own value). Even a box
    8x chair's typical extent on every axis passes through UNTOUCHED: there is no
    ceiling of any kind left in the live path, high-variance or not.
    """
    prior = prior_for("chair")
    assert prior.cap_factor[-1] > 2.5  # precondition: chair's recorded long rank is not floor-capped
    typ = prior.typ_ext
    bloated = typ * 8.0
    out = clamp_extents(bloated, "chair", n_obs=3)
    assert np.allclose(np.sort(out), np.sort(bloated))  # unchanged -- NOT capped


def test_real_large_table_instance_passes_unclipped_regression_guard():
    """Trivial by construction now (clamp_extents never clips anything) -- kept as
    a regression guard, not a design proof: this is a REAL GT table instance
    (data/vla3d/Unity/*/*_object_result.csv, sorted (thin, mid, long) extents,
    long axis ~1.80x the class median) that every one of the three rejected cap
    designs would have clipped to some degree. If a future change re-adds any
    upper cap to clamp_extents and this test starts failing, that is the signal
    to re-read the module docstring's "Issue #201" section (three designs
    measured, 10-49% real-instance clip rates depending on tier, +0.04 points
    total measured upside) before shipping it again.
    """
    real_table = np.array([0.80515512, 1.10492237, 2.27493666])
    out = clamp_extents(real_table, "table", n_obs=3)
    assert np.allclose(np.sort(out), real_table, atol=1e-6)


def test_real_large_window_instance_passes_unclipped_regression_guard():
    """Same as above for window, the class the cap refutation centred on at every
    tier (48.8% flat-cap, 27.9% per-class-scalar, 20.9% per-axis any-axis clip
    rate on real instances). Trivial now by construction; kept as the regression
    guard for the highest-variance class in the table. See the module docstring's
    "Issue #201" section before re-adding any upper cap.
    """
    real_window = np.array([0.11500040, 4.17200059, 7.22500070])
    out = clamp_extents(real_window, "window", n_obs=3)
    assert np.allclose(np.sort(out), real_window, atol=1e-6)


def test_undersized_box_inflated_with_no_upper_bound_anywhere():
    # Grow-only means exactly that: an under-observed thin pillow is still
    # inflated toward typical (unchanged pre-#201 behaviour), and there is no
    # upper bound anywhere in clamp_extents to interact with it.
    prior = prior_for("pillow")
    out = clamp_extents(np.array([0.01, 0.30, 0.40]), "pillow", n_obs=1)
    assert min(out) >= prior.typ_ext[0] - 1e-9


def test_in_range_box_untouched():
    # A box sitting between class-min and typical, well-observed: true no-op, the
    # named GT-passthrough invariant.
    prior = prior_for("chair")
    mid = (prior.min_ext + prior.typ_ext) / 2.0
    out = clamp_extents(mid, "chair", n_obs=5)
    assert np.allclose(np.sort(out), mid)


def test_cap_fused_extent_unaffected_by_the_rejected_marker_time_cap():
    # cap_fused_extent (#104, fuse time) is a DIFFERENT function on a different
    # part of the pipeline (live fused boxes mid-pipeline, not the scored answer
    # marker) and was never part of the #201 refutation -- it keeps its own flat
    # FUSE_EXTENT_CAP_FACTOR unconditionally. clamp_extents (marker time) applies
    # NO cap at all now, so re-running it over an already fuse-capped box is
    # always a pure no-op, for every class, regardless of that class's recorded
    # (unused) cap_factor.
    from core.perception.dimension_priors import cap_fused_extent

    typ = np.array(_DATA_PRIORS["window"][1])
    lo = np.array([0.0, 0.0, 0.0])
    hi = lo + typ * 5.0  # 5x typical -- well past cap_fused_extent's flat 1.5x
    capped_lo, capped_hi = cap_fused_extent(lo, hi, "window")
    capped_ext = np.sort(capped_hi - capped_lo)
    assert np.allclose(capped_ext, FUSE_EXTENT_CAP_FACTOR * np.sort(typ), atol=1e-6)
    reclamped = clamp_extents(capped_hi - capped_lo, "window", n_obs=3)
    assert np.allclose(np.sort(reclamped), capped_ext, atol=1e-6)


# --------------------------------------------------------------- issue #201 / #199 interaction

def test_new_priors_flip_depth_gate_from_fail_open_to_active():
    """Issue #199's depth-plausibility gate (core.perception.fusion) reads
    prior_for() and fails OPEN (never second-guesses depth) for a class with no
    prior. These five classes used to have none; now they do, so the gate is
    ACTIVE for them going forward. This is a deliberate, flagged behaviour change,
    not a regression -- #199's own fail-open tests use a synthetic
    never-in-vocabulary label ('not-a-real-class' / 'not-a-real-vocab-class' /
    'gizmo_no_prior'), not any of these five, so #199's fail-open coverage itself
    is untouched by this change; this test just pins the new fact for the record.
    """
    for label in (
        "bedside table", "fossil decoration", "potted plant", "paper cup", "beer bottle",
    ):
        assert prior_for(label) is not None
