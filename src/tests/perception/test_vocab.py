"""Shared vocabulary bridges (H10): object-class bridge move + colour bridge.

Covers the neutral-location move (perception.vocab is now the source; the old
groundtruth.vocab_bridge re-exports), the head-noun helper, and the colour bridge
both at the map level and wired through toolbox._attrs_match.
"""
from __future__ import annotations

import numpy as np
import pytest

from core.geometry.toolbox import _attrs_match
from core.interfaces import ColorBin, InstanceRecord
from core.perception.vocab import (
    COLOUR_CROSS_HUE,
    COLOUR_NEUTRAL,
    COLOUR_SCHEME,
    DISAMBIGUATOR_PRIORITY_NOUNS,
    STRUCTURAL_EXACT,
    STRUCTURAL_HEADS,
    bridge_synonyms,
    bridged_agree,
    colour_cross_hue,
    colour_synonyms,
    head_noun,
    is_structural_class,
    prioritize_vocab_nouns,
)


def _rec(label: str, caption: str = "", aliases: tuple[str, ...] = ()) -> InstanceRecord:
    z = np.zeros(3)
    return InstanceRecord(
        instance_id=1,
        label=label,
        score=0.9,
        n_obs=3,
        centroid=z,
        aabb_min=z,
        aabb_max=np.ones(3),
        caption=caption,
        aliases=aliases,
    )


def _rec_bins(label: str, bins: list[ColorBin]) -> InstanceRecord:
    """A record carrying raw colour bins (as the GT loader / quantised perception do)."""
    z = np.zeros(3)
    return InstanceRecord(
        instance_id=1,
        label=label,
        score=0.9,
        n_obs=3,
        centroid=z,
        aabb_min=z,
        aabb_max=np.ones(3),
        caption=" ".join(b.name for b in bins),
        aliases=tuple(b.name for b in bins),
        color_bins=tuple(bins),
    )


# ------------------------------------------------------------- object bridge move


def test_object_bridge_reexport_matches_source():
    # the old scorer import path must still resolve to the moved implementation
    from core.groundtruth.vocab_bridge import bridged_agree as reexported

    assert reexported is bridged_agree
    assert bridged_agree("bedside table", "night stand")
    assert "night stand" in bridge_synonyms("bedside table")


def test_head_noun():
    assert head_noun("beer bottle") == "bottle"
    assert head_noun("computer monitor") == "monitor"
    assert head_noun("table") == "table"


# ------------------------------------------------------------------ colour bridge


def test_colour_synonyms_red_reaches_maroon():
    syns = colour_synonyms("red")
    assert "red" in syns and "maroon" in syns


def test_colour_synonyms_grey_spelling():
    # British 'grey' must reach the scheme name 'gray'
    assert "gray" in colour_synonyms("grey")
    assert "gray" in colour_synonyms("gray")


def test_colour_synonyms_black_does_not_reach_gray():
    # black and gray are distinct 15-scheme bins; must not be bridged
    assert "gray" not in colour_synonyms("black")
    assert "black" not in colour_synonyms("gray")


def test_colour_synonyms_unknown_colour_matches_nothing():
    assert colour_synonyms("teal") == frozenset()


def test_colour_scheme_is_the_15_names():
    assert len(COLOUR_SCHEME) == 15
    assert "maroon" in COLOUR_SCHEME and "gray" in COLOUR_SCHEME


# --------------------------------------------------- colour bridge via _attrs_match


def test_attrs_match_red_matches_maroon_caption():
    # NUM-F4: a "red" filter must match a maroon-captioned instance
    rec = _rec("pillow", caption="maroon medium")
    assert _attrs_match(rec, ["red"])


def test_attrs_match_black_does_not_match_gray_caption():
    rec = _rec("pillow", caption="gray medium")
    assert not _attrs_match(rec, ["black"])


def test_attrs_match_exact_colour_still_matches():
    rec = _rec("pillow", caption="black medium")
    assert _attrs_match(rec, ["black"])


def test_attrs_match_non_colour_attr_unchanged():
    # size/material attributes keep the plain substring test
    rec = _rec("table", caption="big wooden")
    assert _attrs_match(rec, ["wooden"])
    assert not _attrs_match(rec, ["metal"])


def test_attrs_match_colour_absent_fails():
    rec = _rec("pillow", caption="green medium")
    assert not _attrs_match(rec, ["red"])


# ------------------------------- bin-aware colour salience (issues #11 / #12)


def test_cross_hue_map_excludes_spelling_bridges():
    # cross-hue bridges are DIFFERENT hues (red->maroon); spelling bridges are not
    assert colour_cross_hue("red") == frozenset({"maroon"})
    assert colour_cross_hue("blue") == frozenset({"navy"})
    assert colour_cross_hue("grey") == frozenset()  # grey->gray is same hue
    assert "gray" in COLOUR_NEUTRAL and "black" in COLOUR_NEUTRAL


# --- issue #11: black reaches a mis-binned dark `gray` via luminance ---


def test_black_matches_dark_slate_gray_bin():
    # loft pillows 89/91: scheme 'gray' but RGB (47,79,79) reads black
    rec = _rec_bins("pillow", [ColorBin("gray", (47, 79, 79), 1.0)])
    assert _attrs_match(rec, ["black"])


def test_black_rejects_lighter_gray_bins():
    # the lighter (112,128,144) and (169,169,169) sofa pillows must NOT be black
    assert not _attrs_match(_rec_bins("pillow", [ColorBin("gray", (112, 128, 144), 1.0)]), ["black"])
    assert not _attrs_match(_rec_bins("pillow", [ColorBin("gray", (169, 169, 169), 1.0)]), ["black"])


def test_black_luminance_gate_is_neutral_only():
    # a DARK saturated bin (maroon) is red, not black — the luma gate is gray-only
    rec = _rec_bins("pillow", [ColorBin("maroon", (120, 20, 20), 1.0)])
    assert not _attrs_match(rec, ["black"])
    assert _attrs_match(rec, ["red"])


def test_white_matches_bright_gray_bin_only():
    assert _attrs_match(_rec_bins("wall", [ColorBin("gray", (240, 240, 240), 1.0)]), ["white"])
    assert not _attrs_match(_rec_bins("wall", [ColorBin("gray", (169, 169, 169), 1.0)]), ["white"])


# --- issue #12: a cross-hue bridged bin needs dominance ---


def test_red_rejects_minor_maroon_bin():
    # hb2 pillow 94: gray 0.36 / brown 0.21 / maroon 0.18 (3rd bin) -> NOT red
    rec = _rec_bins(
        "pillow",
        [
            ColorBin("gray", (105, 105, 105), 0.356),
            ColorBin("brown", (160, 82, 45), 0.215),
            ColorBin("maroon", (165, 42, 42), 0.180),
        ],
    )
    assert not _attrs_match(rec, ["red"])


def test_red_accepts_dominant_maroon_bin():
    # hb2 pillows 85/213: maroon 0.78 dominant -> red
    rec = _rec_bins(
        "pillow",
        [ColorBin("maroon", (205, 92, 92), 0.78), ColorBin("gray", (105, 105, 105), 0.14)],
    )
    assert _attrs_match(rec, ["red"])


def test_exact_scheme_bin_not_dominance_gated():
    # an explicit 'black' bin counts even below the dominance floor (identity, no floor):
    # livingroom_1 chair 22 is black 0.64 / brown 0.36 -> black
    rec = _rec_bins("chair", [ColorBin("black", (0, 0, 0), 0.30), ColorBin("brown", (160, 82, 45), 0.70)])
    assert _attrs_match(rec, ["black"])


def test_bins_absent_falls_back_to_text():
    # no colour bins -> legacy scheme-name substring test (mocks / perception path)
    assert _attrs_match(_rec("pillow", caption="maroon medium"), ["red"])
    assert not _attrs_match(_rec("pillow", caption="gray medium"), ["black"])


# ------------------------------------------- ColorBin construction-time validation


def test_colorbin_valid_construction_ok():
    # a well-formed bin constructs without error
    assert ColorBin("gray", (47, 79, 79), 0.5).rgb == (47, 79, 79)


def test_colorbin_rejects_bad_rgb_shape():
    with pytest.raises(ValueError):
        ColorBin("gray", (10, 20), 0.5)  # only 2 channels
    with pytest.raises(ValueError):
        ColorBin("gray", (10, 20, 30, 40), 0.5)  # 4 channels


def test_colorbin_rejects_rgb_out_of_range():
    with pytest.raises(ValueError):
        ColorBin("gray", (256, 0, 0), 0.5)  # channel above 255
    with pytest.raises(ValueError):
        ColorBin("gray", (0, -1, 0), 0.5)  # channel below 0


def test_colorbin_rejects_bad_fraction():
    with pytest.raises(ValueError):
        ColorBin("gray", (10, 20, 30), 1.5)  # above 1.0
    with pytest.raises(ValueError):
        ColorBin("gray", (10, 20, 30), -0.1)  # below 0.0


# ---------------------------- boundary semantics of the three colour-salience knobs
# Pins the inclusive comparisons in _colour_present: fraction >= colour_dominance_floor
# (0.50), luma <= dark_luma_max (96.0), luma >= light_luma_min (220.0).


def test_dominance_floor_boundary_inclusive():
    # cross-hue maroon-as-red is dominance-gated at 0.50: 0.500 accepts, 0.499 rejects
    assert _attrs_match(_rec_bins("pillow", [ColorBin("maroon", (165, 42, 42), 0.500)]), ["red"])
    assert not _attrs_match(_rec_bins("pillow", [ColorBin("maroon", (165, 42, 42), 0.499)]), ["red"])


def test_dark_luma_boundary_inclusive():
    # gray bin reads black at luma <= 96.0: (96,96,96)=96.0 accepts;
    # (96,96,97)=96.114 (just above) rejects
    assert _attrs_match(_rec_bins("pillow", [ColorBin("gray", (96, 96, 96), 1.0)]), ["black"])
    assert not _attrs_match(_rec_bins("pillow", [ColorBin("gray", (96, 96, 97), 1.0)]), ["black"])


def test_light_luma_boundary_inclusive():
    # gray bin reads white at luma >= 220.0: (220,220,220)=220.0 accepts;
    # (220,220,219)=219.886 (just below) rejects
    assert _attrs_match(_rec_bins("wall", [ColorBin("gray", (220, 220, 220), 1.0)]), ["white"])
    assert not _attrs_match(_rec_bins("wall", [ColorBin("gray", (220, 220, 219), 1.0)]), ["white"])


# ------------------------------------------------------- structural classification (#129)


@pytest.mark.parametrize(
    "label",
    ["floor", "ceiling", "wall", "window", "door", "column", "door frame"],
)
def test_is_structural_class_true_for_room_shell_and_openings(label):
    assert is_structural_class(label)


@pytest.mark.parametrize(
    "label",
    [
        "sofa", "chair", "table", "couch", "bed", "lamp", "curtain", "blinds",
        "picture frame", "carpet", "rug",
    ],
)
def test_is_structural_class_false_for_furniture_and_portables(label):
    assert not is_structural_class(label)


def test_is_structural_class_generalises_via_head_noun_not_substring():
    # A compound that CONTAINS a structural word but denotes a portable object is
    # correctly excluded -- substring matching would wrongly flag this.
    assert not is_structural_class("floor lamp")
    assert not is_structural_class("wall clock")
    assert not is_structural_class("window seat")
    # A structural word carrying its own modifier still generalises through the
    # head-noun match, without being individually enumerated.
    assert is_structural_class("bay window")
    assert is_structural_class("sliding door")
    assert is_structural_class("load-bearing wall")
    assert is_structural_class("stone column")


def test_is_structural_class_door_frame_is_a_named_exception_not_every_frame():
    # "door frame"'s head noun ("frame") is otherwise a portable-object word --
    # folding on the head alone would wrongly pull every "* frame" into structural.
    assert is_structural_class("door frame")
    assert not is_structural_class("picture frame")
    assert not is_structural_class("photo frame")


def test_is_structural_class_never_seen_class_predictions():
    # Classes never seen in this project's scenes, per the classification rule's own
    # stated justification (core.perception.vocab module docstring, issue #129).
    assert is_structural_class("column")  # room-shell support member
    assert is_structural_class("stone column")  # modified variant, same head noun
    # "pillar" is a DIFFERENT head noun from "column" and is NOT in the strict head
    # set -- the rule is deliberately conservative and does not guess that these are
    # the same class without data verification, even though a human reader would
    # treat them as synonyms.
    assert not is_structural_class("pillar")
    # Portable window dressing: hangs on a rod, independently countable and
    # removable without altering the room's shell -- exactly the class of mistake
    # the issue warns against (it originally lumped `window` in with `floor`).
    assert not is_structural_class("curtain")
    assert not is_structural_class("blinds")


def test_is_structural_class_handles_plurals_and_case():
    assert is_structural_class("Windows")
    assert is_structural_class("Doors")
    assert is_structural_class("FLOOR")


def test_is_structural_class_empty_and_blank_are_false():
    assert not is_structural_class("")
    assert not is_structural_class("   ")


def test_structural_heads_and_exact_sets_are_normalised_singular_lowercase():
    for h in STRUCTURAL_HEADS:
        assert h == h.lower()
        assert not h.endswith("s")
    for e in STRUCTURAL_EXACT:
        assert e == e.lower()

# ------------------------------------------- disambiguator vocab priority (issue #91)


def test_prioritize_vocab_nouns_is_pure_reorder_never_a_filter():
    # Every noun handed in must still be present afterward, exactly once -- this is a
    # reorder, never a filter (see the function's docstring guarantee).
    nouns = ["ball", "candle holder", "chair", "jar", "table", "pyramid candle holder"]
    result = prioritize_vocab_nouns(nouns)
    assert sorted(result) == sorted(nouns)
    assert len(result) == len(nouns)


def test_prioritize_vocab_nouns_pulls_priority_set_to_front():
    nouns = ["ball", "candle holder", "chair", "jar", "table", "pyramid candle holder"]
    result = prioritize_vocab_nouns(nouns)
    priority_prefix_len = len(DISAMBIGUATOR_PRIORITY_NOUNS)
    front = set(result[: sum(1 for n in nouns if n in DISAMBIGUATOR_PRIORITY_NOUNS)])
    assert front == {"candle holder", "jar", "pyramid candle holder"}


def test_prioritize_vocab_nouns_preserves_relative_order_within_each_tier():
    # Both the priority tier and the "rest" tier keep the caller's given order among
    # their own members (stable partition, not a re-sort).
    nouns = ["pyramid candle holder", "ball", "jar", "candle holder", "chair", "wall decal"]
    result = prioritize_vocab_nouns(nouns)
    priority_seen = [n for n in result if n in DISAMBIGUATOR_PRIORITY_NOUNS]
    assert priority_seen == ["pyramid candle holder", "jar", "candle holder", "wall decal"]
    rest_seen = [n for n in result if n not in DISAMBIGUATOR_PRIORITY_NOUNS]
    assert rest_seen == ["ball", "chair"]


def test_prioritize_vocab_nouns_never_invents_a_noun_not_given():
    # A priority noun absent from the input must not appear in the output.
    nouns = ["ball", "chair"]
    result = prioritize_vocab_nouns(nouns)
    assert "jar" not in result
    assert set(result) == {"ball", "chair"}


def test_prioritize_vocab_nouns_empty_input():
    assert prioritize_vocab_nouns([]) == []


def test_prioritize_vocab_nouns_all_priority_or_none():
    only_priority = list(DISAMBIGUATOR_PRIORITY_NOUNS)
    assert prioritize_vocab_nouns(only_priority) == only_priority
    no_priority = ["ball", "chair", "table"]
    assert prioritize_vocab_nouns(no_priority) == no_priority
