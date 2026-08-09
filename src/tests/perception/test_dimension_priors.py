"""Per-class dimension priors (H12 / OR-F6): min-clamp, under-obs inflate, GT passthrough."""
from __future__ import annotations

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
    # x=0.60 sits between every pillow class-min rank and pillow's #201 per-AXIS
    # long-rank cap (cap_factor[-1] 1.777 x typical-long 0.528 = 0.938), so it must
    # stay put -- only y (thin, under min) moves.
    out = clamp_extents(np.array([0.60, 0.01, 0.50]), "pillow", n_obs=3)
    assert out[1] >= _DATA_PRIORS["pillow"][0][0] - 1e-9  # thin y axis lifted
    assert out[0] == 0.60  # largest axis unchanged (already in [min, cap] for its rank)


def test_oversized_axis_capped_preserves_other_axis_assignment():
    # Issue #201: the mirror case -- x is now the OVERSIZED axis (well past
    # pillow's own per-axis long-rank cap, 0.938 m), y is thin (under min). Both
    # directions must apply to their own axis only.
    out = clamp_extents(np.array([1.50, 0.01, 0.50]), "pillow", n_obs=3)
    prior = prior_for("pillow")
    assert out[1] >= prior.min_ext[0] - 1e-9  # thin y axis lifted toward min
    assert out[0] < 1.50  # oversized x axis capped down
    assert out[0] <= prior.cap_factor[-1] * prior.typ_ext[-1] + 1e-9


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


# --------------------------------------------------------------- issue #201: two-sided cap
#
# Rework note (round 2): a fresh verifier REFUTED TWO earlier versions of this fix
# against real GT data. Round 1: one flat 1.5x-typical cap for every class clipped
# 28.9% of real table instances and 37.2% of real window instances on their long
# axis alone, because real same-class size variance differs by an order of
# magnitude between classes. Round 2: one scalar PER CLASS (long-axis P95/median)
# fixed that but still clipped 27.9% of real window instances on SOME axis, because
# one axis's variance cannot bound another, independently-varying axis. The cap is
# now PER-CLASS AND PER-AXIS (:data:`_CAP_FACTOR`, a (thin, mid, long) triple, each
# rank's own P95/median ratio, floored 1.5, capped 6.0); see the module docstring
# and :data:`_CAP_FACTOR`'s comment.

def test_cap_factor_loads_as_a_per_axis_triple_for_data_backed_classes():
    # Every _DATA_PRIORS / _PHRASE_DATA_PRIORS class has a (thin, mid, long) cap
    # factor triple, each rank in [1.5, 6.0].
    for label in {**_DATA_PRIORS, **_PHRASE_DATA_PRIORS}:
        prior = prior_for(label)
        assert prior.cap_factor is not None, label
        assert prior.cap_factor.shape == (3,), label
        assert np.all((1.5 <= prior.cap_factor) & (prior.cap_factor <= 6.0)), label

    # _HAND_PRIORS classes (no GT distribution) get NO cap -- fail-open, exactly
    # as the pre-#201 grow-only clamp behaved for every class.
    for label in _HAND_PRIORS:
        prior = prior_for(label)
        assert prior.cap_factor is None, label

    # A class with no prior at all: cap_factor is moot (prior_for returns None).
    assert prior_for("nonexistent-class") is None


def test_cap_factor_table_matches_prior_for():
    # _CAP_FACTOR is the single source of truth prior_for() reads from.
    for label, expected in _CAP_FACTOR.items():
        assert np.allclose(prior_for(label).cap_factor, expected), label


def test_window_cap_factor_axes_differ_from_each_other():
    # The whole point of the round-2 rework: window's three ranks do NOT share one
    # ratio (thin varies far more than long) -- a single scalar could not express
    # this, which is exactly what made round 1's per-class-scalar cap leak on
    # window's thin/mid axes.
    prior = prior_for("window")
    assert prior.cap_factor[0] != prior.cap_factor[-1]


def test_low_variance_class_oversized_box_still_capped_toward_typical():
    # #118: 11 of 14 live boxes were TOO LARGE, median ~4x GT volume. nightstand is
    # a genuinely LOW-variance class (every rank of cap_factor sits at the 1.5
    # floor, n=12): an over-fused nightstand (each axis ~4x typical) must still be
    # capped.
    prior = prior_for("nightstand")
    assert np.allclose(prior.cap_factor, 1.5)  # precondition: floor-capped on every rank
    typ = prior.typ_ext
    bloated = typ * 4.0
    out = clamp_extents(bloated, "nightstand", n_obs=3)
    cap = prior.cap_factor * np.sort(typ)
    assert np.all(np.sort(out) <= cap + 1e-9)
    assert np.all(out < bloated)  # actually shrunk, not just left alone


def test_high_variance_class_still_corrects_a_grossly_oversized_box():
    # chair turned out NOT to be a low-variance class in the real data on its LONG
    # axis (a genuine 2.93 m-long "chair"-labelled GT asset repeats 6x in
    # home_building_1, pushing chair's long-axis P95/median to 2.794 -- higher than
    # the flat cap this issue started from), while its thin/mid axes stay at the
    # 1.5 floor. The per-axis cap still corrects an EXTREME oversize on every axis
    # (well past even chair's own higher long-axis cap), just tolerates more real
    # long-axis variance first.
    prior = prior_for("chair")
    assert prior.cap_factor[-1] > 2.5  # precondition: chair's long rank is NOT floor-capped
    typ = prior.typ_ext
    bloated = typ * 8.0  # well past chair's own (higher) cap on every rank
    out = clamp_extents(bloated, "chair", n_obs=3)
    cap = prior.cap_factor * np.sort(typ)
    assert np.all(np.sort(out) <= cap + 1e-9)
    assert np.all(out < bloated)


def test_real_large_table_instance_passes_unclipped_on_all_three_axes():
    # A REAL GT table instance (data/vla3d/Unity/*/*_object_result.csv, sorted
    # (thin, mid, long) extents) that a flat 1.5x cap WOULD have clipped (its long
    # axis is ~1.80x the class median) must now pass through clamp_extents
    # untouched on EVERY axis -- this is exactly the case the flat-cap verifier
    # refuted.
    real_table = np.array([0.80515512, 1.10492237, 2.27493666])
    prior = prior_for("table")
    cap_vec = prior.cap_factor * prior.typ_ext
    assert np.all(real_table <= cap_vec + 1e-6)  # precondition: in bound on all 3 ranks
    out = clamp_extents(real_table, "table", n_obs=3)
    assert np.allclose(np.sort(out), real_table, atol=1e-6)


def test_real_large_window_instance_passes_unclipped_on_all_three_axes():
    # Likewise for window, the class the round-1 (long-axis-only) cap refutation
    # centred on: a real GT window with a long axis ~4.37x the class median (badly
    # clipped by a flat 1.5x cap, and previously at risk from a per-class-scalar
    # cap too) is untouched now on ALL THREE axes -- thin, mid, AND long.
    real_window = np.array([0.11500040, 4.17200059, 7.22500070])
    prior = prior_for("window")
    cap_vec = prior.cap_factor * prior.typ_ext
    assert np.all(real_window <= cap_vec + 1e-6)  # precondition: in bound on all 3 ranks
    out = clamp_extents(real_window, "window", n_obs=3)
    assert np.allclose(np.sort(out), real_window, atol=1e-6)


def test_undersized_box_still_inflated_two_sided_cap_does_not_fight_it():
    # The pre-existing under-box behaviour (steps 2-3) must be unaffected by the new
    # upper cap: an under-observed thin pillow is still inflated toward typical.
    prior = prior_for("pillow")
    out = clamp_extents(np.array([0.01, 0.30, 0.40]), "pillow", n_obs=1)
    assert min(out) >= prior.typ_ext[0] - 1e-9
    assert np.all(np.sort(out) <= prior.cap_factor * prior.typ_ext + 1e-9)


def test_in_range_box_untouched_by_two_sided_clamp():
    # A box sitting between class-min and the class's own cap, well-observed: true
    # no-op, same GT-passthrough invariant the min-only clamp already guaranteed.
    prior = prior_for("chair")
    mid = (prior.min_ext + prior.typ_ext) / 2.0  # between min and typical < cap
    out = clamp_extents(mid, "chair", n_obs=5)
    assert np.allclose(np.sort(out), mid)


def test_cap_applied_after_min_clamp_and_inflate_no_conflict():
    # Every DATA-backed class's cap is >= its min (a P95/median ratio floored at
    # 1.5 is always >= min-ext/typ-ext, since min <= typ by construction): the two
    # directions of the clamp can never fight for any class in the table.
    for label in {**_DATA_PRIORS, **_PHRASE_DATA_PRIORS}:
        prior = prior_for(label)
        assert np.all(prior.min_ext <= prior.cap_factor * prior.typ_ext), label


def test_cap_fused_extent_stays_flat_1_5x_unaffected_by_the_per_class_rework():
    # cap_fused_extent (#104, fuse time) was NOT part of the flat-cap refutation
    # (it acts on live fused boxes mid-pipeline, not the scored answer marker) and
    # keeps its own flat FUSE_EXTENT_CAP_FACTOR unconditionally -- verify it still
    # does, so a reader does not assume the #201 per-class cap silently propagated
    # there too.
    from core.perception.dimension_priors import cap_fused_extent

    typ = np.array(_DATA_PRIORS["window"][1])  # window: highest per-class cap (4.82)
    lo = np.array([0.0, 0.0, 0.0])
    hi = lo + typ * 5.0  # 5x typical -- well past cap_fused_extent's flat 1.5x
    capped_lo, capped_hi = cap_fused_extent(lo, hi, "window")
    capped_ext = np.sort(capped_hi - capped_lo)
    assert np.allclose(capped_ext, FUSE_EXTENT_CAP_FACTOR * np.sort(typ), atol=1e-6)
    # ...but clamp_extents applied to that SAME result uses window's own (looser)
    # per-class cap, so it is a true no-op here -- the fuse-time cap already left
    # the box well inside the marker-time cap.
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
