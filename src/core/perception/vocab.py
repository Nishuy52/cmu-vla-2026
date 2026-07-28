"""Shared vocabulary bridges: object-class surface drift + colour-name drift.

This module is the *neutral* home for the scene-verified vocabulary maps that
both the live matcher (:mod:`core.perception.scene_index`) and the offline scorer
(:mod:`core.groundtruth.scoring`) need. It lives under ``perception/`` so the live
path can consult it without importing from ``groundtruth/`` (a layering the scorer
respects one-directionally: ``groundtruth`` may import this, never the reverse).

Two independent bridges live here:

* :data:`VOCAB_BRIDGE` / :func:`bridge_synonyms` / :func:`bridged_agree` — object
  *class* surface drift between challenge-question nouns and VLA-3D annotation /
  detector labels for the SAME physical object (``bedside table`` <-> ``night
  stand``, ``potted plant`` <-> ``plant``, ...). Moved here verbatim from the old
  ``core.groundtruth.vocab_bridge`` (which now re-exports these names).
* :data:`COLOUR_BRIDGE` / :func:`colour_synonyms` — question colour words onto the
  closed 15-name VLA-3D colour scheme's neighbourhoods (``red`` -> {red, maroon},
  ``grey`` -> {gray}, ...), so a colour attribute filter can match a scheme-named
  caption without relaxing.

**Honest-none discipline (both bridges).** These are *surface-form synonym* tables
only. They equate two spellings of the same object class / colour bin; they never
relax a relation guard, an anchor guard, or a phrasing tie-break, and never map a
noun/colour to a *different* referent (never a hyponym: no ``chair`` -> ``stool``,
no ``red`` -> ``pink``). Every entry was verified against real VLA-3D data, and
adding one must not turn a correct match into a wrong one — a wrong bridge is worse
than none.
"""
from __future__ import annotations

from functools import lru_cache

from core.parsing.vocab import NOUN_ALIASES


# ------------------------------------------------------------------ object bridge

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


def _norm() :
    """Deferred handle to ``normalize_label`` (avoids an import cycle with scene_index).

    ``scene_index`` imports this module lazily inside ``by_label``; this module in turn
    needs ``normalize_label``. Importing it inside the builder keeps both module tops
    import-free of each other.
    """
    from core.perception.scene_index import normalize_label

    return normalize_label


def _derive_from_noun_aliases() -> dict[str, frozenset[str]]:
    """Normalised bidirectional synonym groups derived from ``vocab.NOUN_ALIASES``.

    ``NOUN_ALIASES`` maps a canonical noun to its alias surface forms. We normalise
    every member and build, per normalised member, the set of every OTHER normalised
    member in its group — so a lookup from either side finds the rest.
    """
    normalize_label = _norm()
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
    """The full normalised object bridge: ``normalized noun -> {normalized synonyms}``.

    Unions the hand map (both directions) with the NOUN_ALIASES-derived groups. Cached
    (the inputs are module constants), so building it is a one-off.
    """
    normalize_label = _norm()
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
    return _bridge_map().get(_norm()(noun), frozenset())


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
    normalize_label = _norm()
    a = normalize_label(question_noun)
    b = normalize_label(annotation_class)
    if not a or not b:
        return False
    if a == b:
        return True
    return b in _bridge_map().get(a, frozenset())


# -------------------------------------------------------------------- head noun

def head_noun(noun: str) -> str:
    """Last whitespace-delimited token of a (normalised) multi-word noun.

    ``"beer bottle" -> "bottle"``, ``"computer monitor" -> "monitor"``,
    ``"table" -> "table"``. The head noun is the grammatical class word of an
    English noun phrase, so a modified query ("X table") can fall back to matching
    bare "table"-headed labels as a *lower-priority* tier below exact/synonym.
    Single-word nouns are returned unchanged.
    """
    n = _norm()(noun)
    parts = n.split()
    return parts[-1] if parts else n


# ----------------------------------------------------------------- colour bridge

#: The closed 15-name VLA-3D colour scheme (``metadata.json`` ``colors_used``):
#: maroon, brown, green, blue, pink, white, orange, black, olive, navy, purple, red,
#: aqua, yellow, gray. Question colour words must land on one of these to match a
#: scheme-named caption; anything outside it is dropped rather than guessed.
COLOUR_SCHEME: frozenset[str] = frozenset(
    {
        "maroon", "brown", "green", "blue", "pink", "white", "orange", "black",
        "olive", "navy", "purple", "red", "aqua", "yellow", "gray",
    }
)

#: Question colour word -> scheme-name neighbourhood it may match. Conservative on
#: purpose: only same-hue-family drift the 15-bin quantiser genuinely produces
#: (``red`` reds get binned ``maroon``; ``blue`` navies get binned ``navy``; British
#: ``grey`` is the American ``gray`` spelling; ``cyan`` is ``aqua``). NOT included:
#: black<->gray (distinct bins; a dark-grey object is annotated ``gray``, not
#: ``black`` — merging them mis-answers "black X" on grey objects, and the notes give
#: no evidence they collide), red<->pink, red<->orange, brown<->maroon (a wrong bridge
#: is worse than none). Every value is itself a scheme name.
COLOUR_BRIDGE: dict[str, tuple[str, ...]] = {
    "red": ("red", "maroon"),
    "blue": ("blue", "navy"),
    "purple": ("purple", "navy"),
    "green": ("green", "olive"),
    "grey": ("gray",),
    "gray": ("gray", "grey"),
    "cyan": ("aqua",),
    "aqua": ("aqua", "cyan"),
    # exact-scheme colours map to themselves so a single lookup covers every case
    "maroon": ("maroon",),
    "brown": ("brown",),
    "pink": ("pink",),
    "white": ("white",),
    "orange": ("orange",),
    "black": ("black",),
    "olive": ("olive",),
    "navy": ("navy",),
    "yellow": ("yellow",),
}


#: Cross-hue colour bridges: a query hue the 15-bin quantiser also emits under a
#: *different* scheme name (dark reds land in ``maroon``, blues in ``navy``, ...).
#: These are the subset of :data:`COLOUR_BRIDGE` values that are a DIFFERENT hue
#: from the query word — so a cross-hue bin should count for the query only when it
#: is a DOMINANT component of the object (issue #12: an 18% 3rd-bin maroon on a
#: gray-dominant pillow must NOT make it "red"). Spelling/synonym bridges
#: (``grey``->``gray``, ``cyan``->``aqua``) are the SAME hue and are deliberately
#: NOT here — they always count, with no dominance floor.
COLOUR_CROSS_HUE: dict[str, frozenset[str]] = {
    "red": frozenset({"maroon"}),
    "blue": frozenset({"navy"}),
    "purple": frozenset({"navy"}),
    "green": frozenset({"olive"}),
}

#: Neutral (achromatic) scheme bins. A ``black``/``white`` query reaches a dark /
#: light ``gray`` bin through a LUMINANCE cutoff (issue #11): VLA-3D bins a
#: near-black dark-slate-gray object (RGB 47,79,79) as ``gray``, not ``black``, so
#: the scheme name cannot separate it from lighter grays — only its RGB luminance
#: can. The luminance thresholds themselves are calibration knobs (geometry
#: ``Thresholds``), applied at the match site; this set names the bins eligible.
COLOUR_NEUTRAL: frozenset[str] = frozenset({"gray", "black", "white"})


def colour_cross_hue(colour: str) -> frozenset[str]:
    """Scheme names that are a cross-hue (dominance-gated) bridge of ``colour``."""
    return COLOUR_CROSS_HUE.get(colour.strip().lower(), frozenset())


@lru_cache(maxsize=256)
def colour_synonyms(colour: str) -> frozenset[str]:
    """Scheme-name colour bins a question colour word may legitimately match.

    Returns the neighbourhood from :data:`COLOUR_BRIDGE` (always including the word
    itself), or — for a colour word not in the bridge — the singleton of the word if
    it is already a scheme name, else an empty set (unknown colour: match nothing
    rather than guess).
    """
    c = colour.strip().lower()
    if c in COLOUR_BRIDGE:
        return frozenset(COLOUR_BRIDGE[c]) | {c}
    if c in COLOUR_SCHEME:
        return frozenset({c})
    return frozenset()


# ------------------------------------------------------------- structural classes
#
# Issue #129: the room's architectural "shell" (floor/ceiling/walls) and its built-in
# openings (windows, doors, door frames, columns) are proposed by the open-vocab
# detector like any other class, but GT annotates exactly ONE of each per scene while
# a floor's box sweeps most of the camera frame and re-detects as dozens of fragments
# (up to 20 "floor" instances vs GT's 1 -- #129 evidence table). Tracked as ordinary
# countable instances, those fragments inflate any class census. Per the training
# corpus (docs/question_analysis.md), NO training question ever counts a structural
# class, so excluding them from a census costs nothing on the scored question set.
#
# But `window` is simultaneously the 4th most frequent object HEAD across the
# training questions and is used almost entirely as a spatial ANCHOR ("how many
# sofas are below a window?") -- so structural classes must stay fully visible to
# every existing anchor/target/relation lookup. This classifier is therefore
# consulted ONLY by this module's own callers and BasicSceneIndex's own census-only
# surfaces (:meth:`BasicSceneIndex.countable_instances`, ``dump_instance_index``'s
# ``by_class``); :meth:`BasicSceneIndex.by_label`/``by_label_tiered``/``all_instances``
# stay completely classification-blind, so every existing anchor/target/relation
# lookup (including this module's own ``resolve``/``counting`` callers) is
# byte-for-byte unchanged by this classifier's existence.
#
# Classification rule: a label is structural iff it names the room's architectural
# shell or a built-in opening set into that shell -- floor, ceiling, wall, window,
# door, door frame, column -- rather than a freestanding, portable, or functional
# object placed WITHIN the room. Matched on the label's HEAD NOUN (the last
# whitespace token of the normalised label, see :func:`head_noun`), never by
# substring, so a compound that merely CONTAINS a structural word but denotes a
# portable object is correctly excluded: "floor lamp" (head "lamp") is furniture,
# not structural, despite literally containing the word "floor". Conversely a
# structural word carrying its own modifier still generalises correctly through the
# head-noun match: "bay window" / "sliding door" / "load-bearing wall" (heads
# "window" / "door" / "wall") all classify as structural without being individually
# enumerated. "door frame" is listed as its own exact canonical form because its
# head noun ("frame") is otherwise a portable-object word ("picture frame") --
# folding on the head alone would wrongly pull every "* frame" into structural.
#
# What this rule predicts for classes never seen in this project's scenes:
# - "column" / "pillar" -> structural (a fixed shell/support member), matching the
#   issue's own "reasonably column" guidance.
# - "skylight" (head "skylight"), or a scene that instead labels the same thing
#   "roof window" (head "window") -> structural either way: an opening set into the
#   shell.
# - "curtain" / "blinds" -> NOT structural: portable window dressing hung on a rod,
#   independently countable and removable without altering the room's shell —
#   exactly the kind of class the issue explicitly warns against lumping in with
#   `floor` (it made the same mistake with `window` originally).
# - "baseboard" / "crown molding" -> NOT structural under the rule AS WRITTEN (their
#   head nouns are not one of the six listed heads). This is a deliberately
#   conservative miss: extending the head set to a genuinely new structural class
#   needs its own data-verified justification, not a guess bundled into this fix.
STRUCTURAL_HEADS: frozenset[str] = frozenset(
    {"floor", "ceiling", "wall", "window", "door", "column"}
)
#: Multi-word canonical forms that are structural despite a head noun ("frame")
#: that is NOT itself structural in isolation (see rule note above).
STRUCTURAL_EXACT: frozenset[str] = frozenset({"door frame"})


def is_structural_class(label: str) -> bool:
    """True iff ``label`` names room-shell structure or a built-in opening in it.

    See the module-level note above for the full rule, its justification, and what
    it predicts for classes not seen in this project's scenes. Consulted only by
    census/reporting surfaces (:meth:`~core.perception.scene_index.BasicSceneIndex.
    countable_instances`, ``dump_instance_index``'s ``by_class``) -- never by any
    label-matching method used for anchor or target resolution.
    """
    normalize_label = _norm()
    canon = normalize_label(label)
    if not canon:
        return False
    if canon in STRUCTURAL_EXACT:
        return True
    return head_noun(canon) in STRUCTURAL_HEADS


__all__ = [
    "VOCAB_BRIDGE",
    "bridge_synonyms",
    "bridged_agree",
    "head_noun",
    "COLOUR_SCHEME",
    "COLOUR_BRIDGE",
    "COLOUR_CROSS_HUE",
    "COLOUR_NEUTRAL",
    "colour_synonyms",
    "colour_cross_hue",
    "STRUCTURAL_HEADS",
    "STRUCTURAL_EXACT",
    "is_structural_class",
]
