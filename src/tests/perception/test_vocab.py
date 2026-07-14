"""Shared vocabulary bridges (H10): object-class bridge move + colour bridge.

Covers the neutral-location move (perception.vocab is now the source; the old
groundtruth.vocab_bridge re-exports), the head-noun helper, and the colour bridge
both at the map level and wired through toolbox._attrs_match.
"""
from __future__ import annotations

import numpy as np

from core.geometry.toolbox import _attrs_match
from core.interfaces import InstanceRecord
from core.perception.vocab import (
    COLOUR_SCHEME,
    bridge_synonyms,
    bridged_agree,
    colour_synonyms,
    head_noun,
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
