"""Tests for core.perception.colour (issue #121: populate live colour bins).

Covers: a clear primary colour classifying confidently, a greyscale/low-saturation
observation abstaining (returning no colour bins rather than a wrong label), and
that abstain propagating end-to-end through BasicSceneIndex (add()/merge_into()
never populate color_bins/caption from a weak observation).
"""
from __future__ import annotations

import numpy as np

from core.interfaces import InstanceRecord
from core.perception.colour import (
    HIGH_AGREEMENT_FRACTION,
    MAX_UNCLASSIFIABLE_FRACTION,
    MIN_DOMINANT_FRACTION,
    MIN_MEAN_SATURATION,
    abstain_reason,
    build_caption,
    dominant_fraction,
    mean_saturation,
    merge_tallies,
    quantise_pixel_names,
    tally_from_colors,
    top_bins,
    unclassifiable_fraction,
)
from core.perception.scene_index import BasicSceneIndex


def _rgb_block(rgb, n, rng, noise=0.0):
    """``n`` copies of ``rgb`` (0-255) with optional small per-channel noise."""
    base = np.tile(np.asarray(rgb, dtype=float), (n, 1))
    if noise:
        base = base + rng.uniform(-noise, noise, size=base.shape)
    return np.clip(base, 0, 255)


# --------------------------------------------------------------------- quantisation


def test_quantise_pixel_names_pure_red():
    names = quantise_pixel_names(np.array([[220.0, 110.0, 100.0]]))
    assert names[0] == "red"


# --------------------------------------------------------------------- clear colour


def test_clear_primary_colour_does_not_abstain():
    """A saturated, unanimous red cluster classifies confidently as red."""
    rng = np.random.default_rng(0)
    colors = _rgb_block((220.0, 110.0, 100.0), 200, rng, noise=8.0)
    valid = np.ones(len(colors), dtype=bool)

    tally = tally_from_colors(colors, valid)
    assert tally is not None
    assert abstain_reason(tally) is None

    bins = top_bins(tally)
    assert len(bins) >= 1
    assert bins[0].name == "red"
    assert bins[0].fraction > 0.9


# --------------------------------------------------------------------- abstain: too few pixels


def test_too_few_pixels_abstains():
    rng = np.random.default_rng(1)
    colors = _rgb_block((220.0, 110.0, 100.0), 3, rng)  # well under MIN_TALLY_PIXELS
    valid = np.ones(len(colors), dtype=bool)

    tally = tally_from_colors(colors, valid)
    assert tally is not None
    assert abstain_reason(tally) == "too_few_pixels"
    assert top_bins(tally) == ()


# --------------------------------------------------------------------- abstain: mixed distribution


def test_mixed_distribution_abstains():
    """No clear majority colour: an even 3-way red/blue/green split -> abstain,
    not a coin flip that happens to pick one of the three."""
    rng = np.random.default_rng(2)
    # Centroid-exact RGBs (see CENTROIDS_RGB) so noise alone stays well inside
    # MAX_CENTROID_LAB_DISTANCE -- this test targets the dominance gate
    # specifically, not the separate out-of-gamut gate.
    red = _rgb_block((229.8, 117.5, 106.5), 40, rng, noise=5.0)
    blue = _rgb_block((69.6, 128.2, 183.2), 40, rng, noise=5.0)
    green = _rgb_block((44.5, 141.7, 78.1), 40, rng, noise=5.0)
    colors = np.concatenate([red, blue, green], axis=0)
    valid = np.ones(len(colors), dtype=bool)

    tally = tally_from_colors(colors, valid)
    assert tally is not None
    assert unclassifiable_fraction(tally) < MAX_UNCLASSIFIABLE_FRACTION
    assert dominant_fraction(tally) < MIN_DOMINANT_FRACTION
    assert abstain_reason(tally) == "mixed_distribution"
    assert top_bins(tally) == ()


# --------------------------------------------------------------------- abstain: greyscale / low saturation


def test_greyscale_low_saturation_abstains():
    """A washed-out near-achromatic cluster straddling the gray/black luma
    boundary: every pixel is exactly R=G=B (zero saturation -- a genuinely
    ambiguous dim/desaturated capture, e.g. low light or sensor clipping), and
    the split leaves no unanimous majority -- must abstain rather than assert
    either neutral name with false confidence."""
    lo = np.tile([55.0, 55.0, 55.0], (50, 1))
    hi = np.tile([63.0, 63.0, 63.0], (50, 1))
    colors = np.concatenate([lo, hi], axis=0)
    valid = np.ones(len(colors), dtype=bool)

    tally = tally_from_colors(colors, valid)
    assert tally is not None
    assert mean_saturation(tally) < MIN_MEAN_SATURATION
    assert dominant_fraction(tally) < HIGH_AGREEMENT_FRACTION
    assert abstain_reason(tally) == "low_saturation"
    assert top_bins(tally) == ()


def test_uniform_gray_does_not_abstain_despite_low_saturation():
    """A genuinely uniform low-saturation surface (every pixel agrees) is still
    trusted as 'gray' -- low_saturation only fires jointly with non-unanimous
    dominance, so this legitimate GT colour is not thrown away."""
    rng = np.random.default_rng(4)
    colors = _rgb_block((110.0, 121.0, 121.0), 100, rng, noise=1.0)
    valid = np.ones(len(colors), dtype=bool)

    tally = tally_from_colors(colors, valid)
    assert tally is not None
    assert dominant_fraction(tally) >= 0.90
    assert abstain_reason(tally) is None
    bins = top_bins(tally)
    assert bins[0].name == "gray"


# --------------------------------------------------------------------- no observation


def test_no_valid_pixels_returns_none_tally():
    colors = np.zeros((5, 3))
    valid = np.zeros(5, dtype=bool)
    assert tally_from_colors(colors, valid) is None
    assert abstain_reason(None) == "no_observation"
    assert top_bins(None) == ()


# --------------------------------------------------------------------- merge_tallies


def test_merge_tallies_accumulates_counts_and_saturation():
    rng = np.random.default_rng(5)
    a = tally_from_colors(_rgb_block((220.0, 110.0, 100.0), 20, rng), np.ones(20, dtype=bool))
    b = tally_from_colors(_rgb_block((220.0, 110.0, 100.0), 30, rng), np.ones(30, dtype=bool))
    merged = merge_tallies(a, b)
    assert merged.total == a.total + b.total
    assert merged.counts["red"] == a.counts["red"] + b.counts["red"]
    assert merged.sat_sum == a.sat_sum + b.sat_sum


# --------------------------------------------------------------------- caption


def test_build_caption_includes_colour_and_size():
    bins = top_bins(
        tally_from_colors(
            _rgb_block((220.0, 110.0, 100.0), 50, np.random.default_rng(6), noise=5.0),
            np.ones(50, dtype=bool),
        )
    )
    caption = build_caption(bins, np.array([0.3, 0.3, 0.3]))
    assert "red" in caption
    assert "medium" in caption


# --------------------------------------------------------------------- abstain propagation through BasicSceneIndex


def _rec(instance_id, cmin, cmax):
    amin = np.asarray(cmin, dtype=float)
    amax = np.asarray(cmax, dtype=float)
    return InstanceRecord(
        instance_id=instance_id,
        label="chair",
        score=0.9,
        n_obs=1,
        centroid=(amin + amax) / 2.0,
        aabb_min=amin,
        aabb_max=amax,
        points=None,
    )


def test_scene_index_add_with_confident_colour_populates_bins():
    rng = np.random.default_rng(7)
    tally = tally_from_colors(
        _rgb_block((220.0, 110.0, 100.0), 50, rng, noise=5.0), np.ones(50, dtype=bool)
    )
    index = BasicSceneIndex([])
    survivor = index.add(_rec(1, [0, 0, 0], [0.3, 0.3, 0.3]), colour_obs=tally)
    assert survivor.color_bins != ()
    assert survivor.color_bins[0].name == "red"
    assert "red" in survivor.caption


def test_scene_index_add_with_weak_colour_leaves_bins_empty():
    """A too-few-pixels observation must not populate color_bins/caption -- the
    instance stays 'colour unknown', not confidently mislabeled."""
    rng = np.random.default_rng(8)
    weak_tally = tally_from_colors(_rgb_block((220.0, 110.0, 100.0), 2, rng), np.ones(2, dtype=bool))
    assert abstain_reason(weak_tally) == "too_few_pixels"

    index = BasicSceneIndex([])
    survivor = index.add(_rec(1, [0, 0, 0], [0.3, 0.3, 0.3]), colour_obs=weak_tally)
    assert survivor.color_bins == ()
    assert survivor.caption == "medium"  # size token only, no colour word


def test_scene_index_merge_keeps_prior_confident_colour_when_new_obs_is_weak():
    """Fusing a second, weak observation into an instance that already has a
    confident colour must not erase it -- colour_obs=None (or an abstained tally)
    leaves the existing bins untouched."""
    rng = np.random.default_rng(9)
    strong = tally_from_colors(
        _rgb_block((220.0, 110.0, 100.0), 50, rng, noise=5.0), np.ones(50, dtype=bool)
    )
    index = BasicSceneIndex([])
    index.add(_rec(1, [0, 0, 0], [1, 1, 1]), colour_obs=strong)

    # A second observation of the same instance, projected onto too few pixels
    # this keyframe -- top_bins would abstain, so the caller passes colour_obs=None.
    survivor = index.merge_into(1, _rec(2, [0.1, 0.1, 0.1], [1.1, 1.1, 1.1]), colour_obs=None)
    assert survivor.color_bins != ()
    assert survivor.color_bins[0].name == "red"


# --------------------------------------------------------------------- out-of-gamut guard (white/off-white/beige/light-gray)


def test_pure_white_is_unclassifiable_and_abstains():
    colors = _rgb_block((250.0, 250.0, 250.0), 100, np.random.default_rng(10), noise=3.0)
    valid = np.ones(len(colors), dtype=bool)
    tally = tally_from_colors(colors, valid)
    assert tally is not None
    assert unclassifiable_fraction(tally) >= MAX_UNCLASSIFIABLE_FRACTION
    assert abstain_reason(tally) == "out_of_gamut"
    assert top_bins(tally) == ()


def test_off_white_is_unclassifiable_and_abstains():
    colors = _rgb_block((230.0, 232.0, 228.0), 100, np.random.default_rng(11), noise=3.0)
    valid = np.ones(len(colors), dtype=bool)
    tally = tally_from_colors(colors, valid)
    assert tally is not None
    assert unclassifiable_fraction(tally) >= MAX_UNCLASSIFIABLE_FRACTION
    assert abstain_reason(tally) == "out_of_gamut"
    assert top_bins(tally) == ()


def test_beige_is_unclassifiable_and_abstains():
    colors = _rgb_block((240.0, 235.0, 220.0), 100, np.random.default_rng(12), noise=3.0)
    valid = np.ones(len(colors), dtype=bool)
    tally = tally_from_colors(colors, valid)
    assert tally is not None
    assert unclassifiable_fraction(tally) >= MAX_UNCLASSIFIABLE_FRACTION
    assert abstain_reason(tally) == "out_of_gamut"
    assert top_bins(tally) == ()


def test_light_gray_is_unclassifiable_and_abstains():
    """(200, 200, 205) is the closest of the four reported failure RGBs to the
    nearest real centroid (~28.85 Lab units) -- still comfortably beyond
    MAX_CENTROID_LAB_DISTANCE (25.0)."""
    colors = _rgb_block((200.0, 200.0, 205.0), 100, np.random.default_rng(13), noise=3.0)
    valid = np.ones(len(colors), dtype=bool)
    tally = tally_from_colors(colors, valid)
    assert tally is not None
    assert unclassifiable_fraction(tally) >= MAX_UNCLASSIFIABLE_FRACTION
    assert abstain_reason(tally) == "out_of_gamut"
    assert top_bins(tally) == ()


def test_noisy_cream_beige_couch_abstains_not_confidently_pink():
    """Reproduces the reported failure: a cream/beige surface under realistic
    sensor noise (sigma ~8/channel) used to land >50% in a single named bin
    (pink/aqua) with mean_saturation/dominant_fraction both clearing the
    ordinary confidence gates -- the out-of-gamut guard must catch it before
    either of those gates gets a chance to (mis)approve it."""
    rng = np.random.default_rng(14)
    base = np.tile([222.0, 216.0, 190.0], (200, 1))
    colors = np.clip(base + rng.normal(0.0, 8.0, size=base.shape), 0, 255)
    valid = np.ones(len(colors), dtype=bool)

    tally = tally_from_colors(colors, valid)
    assert tally is not None
    assert unclassifiable_fraction(tally) >= MAX_UNCLASSIFIABLE_FRACTION
    assert abstain_reason(tally) == "out_of_gamut"
    assert top_bins(tally) == ()


def test_out_of_gamut_guard_does_not_disturb_clear_primary_colour():
    """Regression guard: the new out-of-gamut check must not affect a clearly
    in-gamut saturated colour (already covered by test_clear_primary_colour_
    does_not_abstain, re-asserted here against the specific new gate)."""
    rng = np.random.default_rng(15)
    colors = _rgb_block((220.0, 110.0, 100.0), 200, rng, noise=8.0)
    valid = np.ones(len(colors), dtype=bool)
    tally = tally_from_colors(colors, valid)
    assert tally is not None
    assert unclassifiable_fraction(tally) < MAX_UNCLASSIFIABLE_FRACTION
    assert abstain_reason(tally) is None
    assert top_bins(tally)[0].name == "red"


def test_out_of_gamut_guard_does_not_disturb_uniform_gray():
    """Regression guard: a genuinely uniform low-saturation gray surface (well
    within the scheme -- its own centroid is nearby) must keep classifying,
    unaffected by the new out-of-gamut gate."""
    rng = np.random.default_rng(16)
    colors = _rgb_block((110.0, 121.0, 121.0), 100, rng, noise=1.0)
    valid = np.ones(len(colors), dtype=bool)
    tally = tally_from_colors(colors, valid)
    assert tally is not None
    assert unclassifiable_fraction(tally) < MAX_UNCLASSIFIABLE_FRACTION
    assert abstain_reason(tally) is None
    assert top_bins(tally)[0].name == "gray"
