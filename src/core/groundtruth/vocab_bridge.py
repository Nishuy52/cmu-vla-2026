"""Vocabulary bridge between challenge-question nouns and VLA-3D annotation labels.

The ground-truth referential statements (``*_referential_statements.json``) and the
object CSV ``raw_label`` use a *different surface vocabulary* than the challenge
questions for the **same physical object class**. The object-reference scorer matches
a question's target/anchor noun against an annotation's ``target_class`` / anchor
``class`` via :func:`core.groundtruth.scoring._anchor_agrees` (equality / substring /
shared-token over normalised strings). That test fails on pure *surface drift* where
neither string contains the other and they share no token, even though a human reads
them as the same object:

* ``bedside table`` (question) vs ``night stand`` (annotation)
* ``potted plant``  vs ``plant``
* ``beer bottle``   vs ``bottle``
* ``paper cup`` / ``cup of coffee`` / ``coffee cup`` vs ``cup``

These are the *legitimate* "none" cases in the baseline OR report: the annotation set
DOES contain the queried target under a synonymous class name, with the SAME relation
and anchor. Bridging them recovers a scoreable IoU without any guessing.

**Honest-none discipline.** This bridge is a *surface-form synonym* table only. It
equates two spellings of the same object class; it never relaxes the relation guard,
the anchor guard, or the phrasing tie-break in the matcher. A question whose relation
is genuinely absent from the annotations (e.g. "the picture closest to the window",
where only "the window closest to the picture" is annotated — an inverted referent)
stays "none". A noun that maps to a *different* physical class is never entered here.
Every entry below was verified against a real scene's annotations to link the same
object, and adding one must not turn a correct match into a wrong one.

Construction (two layers, unioned):

1. **Derived** from :data:`core.parsing.vocab.NOUN_ALIASES` — the parse ladder already
   records alias surface forms per canonical noun (``coffee cup`` <-> ``cup of
   coffee``, ``tv`` <-> ``television``, ...). We fold those in so the bridge tracks the
   parser's own synonym knowledge for free.

2. **Hand-extendable** :data:`VOCAB_BRIDGE` — the small, documented map of
   annotation-vocabulary drift the derived layer does not cover (``bedside table`` ->
   ``night stand``, ``potted plant`` -> ``plant``, ...). Extend THIS constant when a
   new legitimate drift is found; keep it minimal and object-identity-preserving.

The public entry point is :func:`bridged_agree`, a drop-in stricter-than-nothing
replacement for the surface comparison used inside the scorer: it returns True when
the base surface test agrees OR when the two normalised nouns are linked by the bridge.
Pure, deterministic, no new dependencies.
"""
from __future__ import annotations

from functools import lru_cache

from core.parsing.vocab import NOUN_ALIASES
from core.perception.scene_index import normalize_label


# --------------------------------------------------------------------------- hand map

#: Hand-extendable synonym pairs: challenge-question noun -> VLA-3D annotation labels
#: (``target_class`` / anchor ``class`` / ``raw_label``) that denote the SAME object
#: class. Keys and values are given in natural surface form; both sides are run through
#: :func:`core.perception.scene_index.normalize_label` (lowercase / singular / synonym
#: table) before comparison, so "flowers" and "flower" collapse automatically and need
#: no entry. Only genuine *surface drift for the same object* belongs here — never a
#: hyponym that changes the referent (e.g. do NOT map "chair" -> "stool").
#:
#: Each entry is annotated with the scene where the drift was observed in the baseline
#: OR "none" cases, so the link can be re-checked against real data.
VOCAB_BRIDGE: dict[str, tuple[str, ...]] = {
    # bedside table  vs  night stand   (hotel_room_1: "bedside table farthest from window")
    "bedside table": ("night stand", "nightstand"),
    "nightstand": ("night stand",),
    # potted plant   vs  plant         (office_1 / livingroom_3: "potted plant near/on ...")
    "potted plant": ("plant",),
    # beer bottle    vs  bottle        (studio: "beer bottle furthest from the couch")
    "beer bottle": ("bottle",),
    # paper cup / cup of coffee / coffee cup  vs  cup  (office_1: "paper cup on the table";
    # loft anchor "cup of coffee").  The parser canonicalises "cup of coffee" -> "coffee
    # cup"; the annotation calls it "cup".
    "paper cup": ("cup",),
    "coffee cup": ("cup",),
    # computer monitor  vs  monitor    (office_2 anchor/target surface drift)
    "computer monitor": ("monitor",),
    # wall lamp      vs  lamp          (arabic_room: "wall lamp between door frame and window")
    "wall lamp": ("lamp",),
}


# --------------------------------------------------------------------------- derive

def _derive_from_noun_aliases() -> dict[str, frozenset[str]]:
    """Normalised bidirectional synonym groups derived from ``vocab.NOUN_ALIASES``.

    ``NOUN_ALIASES`` maps a canonical noun to its alias surface forms. We normalise
    every member and build, per normalised member, the set of every OTHER normalised
    member in its group — so a lookup from either side finds the rest.
    """
    groups: list[set[str]] = []
    for canon, aliases in NOUN_ALIASES.items():
        members = {normalize_label(canon)} | {normalize_label(a) for a in aliases}
        members = {m for m in members if m}
        if len(members) >= 2:
            groups.append(members)
    out: dict[str, set[str]] = {}
    for members in groups:
        for m in members:
            out.setdefault(m, set()).update(members - {m})
    return {k: frozenset(v) for k, v in out.items()}


@lru_cache(maxsize=1)
def _bridge_map() -> dict[str, frozenset[str]]:
    """The full normalised bridge: ``normalized noun -> {normalized synonyms}``.

    Unions the hand map (both directions) with the NOUN_ALIASES-derived groups. Cached
    (the inputs are module constants), so building it is a one-off.
    """
    out: dict[str, set[str]] = {k: set(v) for k, v in _derive_from_noun_aliases().items()}

    def _add(a: str, b: str) -> None:
        if a and b and a != b:
            out.setdefault(a, set()).add(b)

    for key, vals in VOCAB_BRIDGE.items():
        nk = normalize_label(key)
        for v in vals:
            nv = normalize_label(v)
            _add(nk, nv)
            _add(nv, nk)  # bidirectional: match from either surface form
    return {k: frozenset(v) for k, v in out.items()}


def bridge_synonyms(noun: str) -> frozenset[str]:
    """Normalised synonym surface forms linked to ``noun`` by the bridge (excl. itself)."""
    return _bridge_map().get(normalize_label(noun), frozenset())


def bridged_agree(question_noun: str, annotation_class: str) -> bool:
    """True if a question noun and an annotation class denote the same object class.

    Consults the vocabulary bridge only — the caller is expected to have already tried
    the base surface comparison (equality / substring / shared token). This function is
    the *additional* leniency: it returns True when the two normalised nouns are linked
    in :func:`_bridge_map` (hand map or NOUN_ALIASES-derived), and False otherwise. It
    deliberately does NOT re-implement substring/shared-token logic, so it can only ever
    ADD matches that are explicitly whitelisted as the same object — never widen the
    guard into guessing.
    """
    a = normalize_label(question_noun)
    b = normalize_label(annotation_class)
    if not a or not b:
        return False
    if a == b:
        return True
    return b in _bridge_map().get(a, frozenset())


__all__ = [
    "VOCAB_BRIDGE",
    "bridge_synonyms",
    "bridged_agree",
]
