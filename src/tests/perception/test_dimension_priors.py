"""Per-class dimension priors (H12 / OR-F6): min-clamp, under-obs inflate, GT passthrough."""
from __future__ import annotations

import numpy as np

from core.interfaces import InstanceRecord
from core.perception.dimension_priors import (
    _DATA_PRIORS,
    _HAND_PRIORS,
    ABS_MIN_EXTENT_FALLBACK_M,
    DEGENERATE_EXTENT_M,
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
    for label, (mn, tp) in {**_DATA_PRIORS, **_HAND_PRIORS}.items():
        assert all(m <= t for m, t in zip(mn, tp)), label


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
