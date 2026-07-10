"""build_scene_for / noun-extraction tests."""
from __future__ import annotations

from core.runner.scenegen import build_scene_for, nouns_in_text, numerical_target_nouns


def test_nouns_in_text_collects_all_and_phrases():
    nouns = nouns_in_text("Find the wall lamp between a door frame and a window")
    assert "wall lamp" in nouns  # multi-word phrase preferred over "lamp"
    assert "door frame" in nouns
    assert "window" in nouns
    assert "lamp" not in nouns  # consumed by the "wall lamp" phrase


def test_numerical_target_gets_two_instances():
    questions = {"numerical": ["How many chairs are near the table?"]}
    assert "chair" in numerical_target_nouns(questions)
    sc, spec = build_scene_for("s", questions)
    labels = [o.label for o in sc.objects]
    assert labels.count("chair") == 2  # counting target duplicated
    assert labels.count("table") == 1  # non-target noun single


def test_build_scene_for_is_deterministic():
    questions = {"object_reference": ["Find the vase on the table"]}
    a, _ = build_scene_for("s", questions, seed=3)
    b, _ = build_scene_for("s", questions, seed=3)
    pa = [(o.label, round(o.cx, 4), round(o.cy, 4)) for o in a.objects]
    pb = [(o.label, round(o.cx, 4), round(o.cy, 4)) for o in b.objects]
    assert pa == pb


def test_scene_contains_every_mentioned_noun():
    questions = {
        "object_reference": ["Find the pillow closest to the book on the stool."],
    }
    sc, spec = build_scene_for("s", questions)
    labels = {o.label for o in sc.objects}
    for noun in ("pillow", "book", "stool"):
        assert noun in labels
