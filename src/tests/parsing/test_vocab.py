"""Vocabulary unit tests: singularisation, typo map, phrase and attribute matching."""
from __future__ import annotations

from core.parsing import vocab


def test_singularize():
    assert vocab.singularize("pillows") == "pillow"
    assert vocab.singularize("boxes") == "box"
    assert vocab.singularize("benches") == "bench"
    assert vocab.singularize("stairs") == "stairs"  # plural-invariant
    assert vocab.singularize("glass") == "glass"  # no -ss stripping


def test_refridgerator_typo_canonicalises():
    assert vocab.canonical_word("refridgerator") == "refrigerator"
    assert vocab.canonical_word("fridge") == "refrigerator"


def test_phrase_matching():
    noun, attrs, raw, indef = vocab.match_noun("the potted plants".split())
    assert noun == "potted plant" and attrs == [] and raw == "potted plants"
    noun, _, raw, _ = vocab.match_noun("the cup of coffee".split())
    assert noun == "coffee cup" and raw == "cup of coffee"
    noun, _, _, _ = vocab.match_noun("the crystal ball decoration".split())
    assert noun == "crystal ball decoration"  # 'crystal' not stripped as attribute


def test_attribute_peeling_and_indefinite():
    noun, attrs, _, indef = vocab.match_noun("a small table".split())
    assert noun == "table" and attrs == ["small"] and indef is True
    noun, attrs, _, _ = vocab.match_noun("the round tables".split())
    assert noun == "table" and attrs == ["round"]


def test_unknown_noun_best_effort():
    noun, attrs, _, _ = vocab.match_noun("the wooden storage crate".split())
    assert noun == "crate" and "wooden" in attrs


def test_find_first_noun():
    assert vocab.find_first_noun("near the map wall decal please") == "map wall decal"
    assert vocab.find_first_noun("no such things here") is None
