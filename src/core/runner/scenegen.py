"""Synthesize a per-scene :class:`SyntheticScene` from the nouns its questions mention.

No ground-truth answers exist for the training questions, so the battery cannot score
correctness — it scores STRUCTURAL health. For that the heads need *a* scene that at
least contains an instance of every noun each question refers to, placed deterministically.

``nouns_in_text`` extracts every vocabulary noun in a question (a full n-gram sweep, not
just the first). ``build_scene_for`` lays one-or-more boxes per noun onto a seeded grid;
nouns that are the *target* of a numerical question get 2+ instances so counting is
nontrivial (a count of 1 would be a degenerate structural pass).
"""
from __future__ import annotations

from dataclasses import dataclass

from core.mocks.synthetic_scene import Room, SyntheticScene
from core.parsing import vocab
from core.parsing.regex_tier import classify_qtype
from core.interfaces import QType


def nouns_in_text(text: str) -> list[str]:
    """Every vocabulary noun in ``text``, left-to-right, longest-phrase-first, de-duped.

    Mirrors :func:`core.parsing.vocab.find_first_noun` but collects all matches and skips
    over the tokens a multi-word phrase already consumed.
    """
    words = [w.strip(",.;:'\"?!") for w in text.lower().split()]
    words = [w for w in words if w]
    out: list[str] = []
    i = 0
    while i < len(words):
        matched = False
        for n in (3, 2, 1):
            if i + n <= len(words):
                noun = vocab._lookup(tuple(words[i : i + n]))
                if noun is not None:
                    if noun not in out:
                        out.append(noun)
                    i += n
                    matched = True
                    break
        if not matched:
            i += 1
    return out


def numerical_target_nouns(questions: dict[str, list[str]]) -> set[str]:
    """The head noun each numerical question counts (first noun after the count prefix).

    These need >= 2 instances so the count is nontrivial.
    """
    targets: set[str] = set()
    for text in questions.get("numerical", []):
        for noun in nouns_in_text(text):
            targets.add(noun)
            break  # the first noun is the counting target
    return targets


@dataclass
class SceneSpec:
    """What was synthesized for one scene (for report provenance)."""

    scene_name: str
    nouns: list[str]
    numerical_targets: list[str]
    instance_count: int


def _all_nouns(questions: dict[str, list[str]]) -> list[str]:
    seen: list[str] = []
    for bucket in ("numerical", "object_reference", "instruction_following"):
        for text in questions.get(bucket, []):
            for noun in nouns_in_text(text):
                if noun not in seen:
                    seen.append(noun)
    return seen


def build_scene_for(
    scene_name: str, questions: dict[str, list[str]], *, seed: int = 0
) -> tuple[SyntheticScene, SceneSpec]:
    """Build a deterministic SyntheticScene containing an instance of every noun mentioned.

    Placement: a seeded grid across one wide room; numerical-target nouns get 2 instances.
    Determinism: fully a function of (scene_name, questions, seed).
    """
    nouns = _all_nouns(questions)
    num_targets = numerical_target_nouns(questions)

    # Deterministic grid placement; 2 boxes for counting targets, 1 otherwise.
    placements: list[str] = []
    for noun in nouns:
        placements.append(noun)
        if noun in num_targets:
            placements.append(noun)  # second instance so counting is nontrivial

    # Size the room tightly to the grid: terrain sampling is O(room area) and is recomputed
    # every explore tick, so an oversized room dominates wall time. A 0.7 m grid pitch with a
    # 0.4 m wall margin keeps the whole scene compact yet un-crowded.
    cols = 8
    step = 0.7
    margin = 0.6
    x0 = margin
    y0 = margin
    rows = (len(placements) + cols - 1) // cols if placements else 1
    room_w = x0 + (cols - 1) * step + margin
    room_d = y0 + max(rows - 1, 0) * step + margin

    sc = SyntheticScene(seed)
    sc.rooms = [Room(0.0, 0.0, room_w, room_d)]
    sc._split_x = None
    sc.doorway = None
    sc.objects = []

    for i, noun in enumerate(placements):
        row, col = divmod(i, cols)
        x = x0 + col * step
        y = y0 + row * step
        # small deterministic per-noun jitter keyed off the char sum, seed-mixed
        jitter = ((sum(ord(c) for c in noun) + sc.seed) % 5) * 0.02
        sc.place_box(noun, x + jitter, y + jitter, 0.3, 0.3, 0.4)

    spec = SceneSpec(
        scene_name=scene_name,
        nouns=nouns,
        numerical_targets=sorted(num_targets),
        instance_count=len(sc.objects),
    )
    return sc, spec
