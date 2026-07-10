"""Noun vocabulary for the parse ladder: canonical nouns, phrases, attributes, typos.

Built from the 75 training questions (docs/question_analysis.md section 4, 114 noun tokens).
Canonical form is lowercase singular; multi-word nouns keep internal spaces ('potted plant').
"""
from __future__ import annotations

# --------------------------------------------------------------------- attributes

#: The 10 attribute mentions observed in training plus common colors/sizes/materials.
ATTRIBUTES: frozenset[str] = frozenset(
    {
        # colors (training: red, black, blue)
        "red", "black", "blue", "green", "white", "yellow", "brown", "orange",
        "purple", "pink", "grey", "gray",
        # sizes (training: small, big)
        "small", "big", "large", "little", "tiny", "tall", "short", "wide", "narrow",
        # shapes / materials (training: round tables, stone, crystal)
        "round", "square", "wooden", "metal", "plastic", "glass", "leather",
        "stone", "crystal",
    }
)

# --------------------------------------------------------------------- typo / synonym map

#: Word-level canonicalisation applied after singularisation ('refridgerator' -> 'refrigerator').
SYNONYMS: dict[str, str] = {
    "refridgerator": "refrigerator",
    "fridge": "refrigerator",
    "television": "tv",
    "telly": "tv",
    "settee": "sofa",
    "picture-frame": "picture",
}

# --------------------------------------------------------------------- nouns

#: Single-word canonical nouns seen (or plausibly standing alone) in the training set.
SINGLE_NOUNS: frozenset[str] = frozenset(
    {
        "ball", "bed", "bench", "book", "bookcase", "bottle", "bowl", "box",
        "cabinet", "can", "candle", "chair", "clock", "column", "couch",
        "counter", "cup", "curtain", "decal", "decoration", "door", "easel",
        "elephant", "fan", "figurine", "fireplace", "floor", "flower",
        "folder", "fossil", "frame", "guitar", "holder", "hookah", "horse",
        "jar", "kettle", "knife", "lamp", "lantern", "ledge", "magazine",
        "map", "microwave", "mirror", "monitor", "nightstand", "ottoman",
        "painting", "phone", "photo", "picture", "pillow", "plant",
        "projector", "pyramid", "rack", "record", "refrigerator", "remote",
        "screen", "shelf", "sign", "sofa", "speaker", "sphere", "stairs",
        "stool", "suitcase", "sushi", "table", "tray", "tv", "vase", "wall",
        "wardrobe", "whiteboard", "window",
    }
)

#: Multi-word noun phrases -> canonical noun. Keys use the singular head word.
PHRASES: dict[tuple[str, ...], str] = {
    ("potted", "plant"): "potted plant",
    ("coffee", "table"): "coffee table",
    ("tea", "table"): "tea table",
    ("dining", "table"): "dining table",
    ("bedside", "table"): "bedside table",
    ("dressing", "table"): "dressing table",
    ("tv", "cabinet"): "tv cabinet",
    ("file", "cabinet"): "file cabinet",
    ("kitchen", "counter"): "kitchen counter",
    ("trash", "can"): "trash can",
    ("folding", "screen"): "folding screen",
    ("projector", "screen"): "projector screen",
    ("knife", "rack"): "knife rack",
    ("wall", "lamp"): "wall lamp",
    ("door", "frame"): "door frame",
    ("map", "wall", "decal"): "map wall decal",
    ("wall", "decal"): "wall decal",
    ("water", "cooler"): "water cooler",
    ("exit", "sign"): "exit sign",
    ("soccer", "ball"): "soccer ball",
    ("crystal", "ball", "decoration"): "crystal ball decoration",
    ("crystal", "ball"): "crystal ball decoration",
    ("sphere", "decoration"): "sphere decoration",
    ("fan", "decoration"): "fan decoration",
    ("stone", "decoration"): "stone decoration",
    ("fossil", "decoration"): "fossil decoration",
    ("elephant", "figurine"): "elephant figurine",
    ("horse", "figurine"): "horse figurine",
    ("pyramid", "candle", "holder"): "pyramid candle holder",
    ("candle", "holder"): "candle holder",
    ("computer", "monitor"): "computer monitor",
    ("display", "ledge"): "display ledge",
    ("calligraphy", "painting"): "calligraphy painting",
    ("framed", "record"): "framed record",
    ("beer", "bottle"): "beer bottle",
    ("paper", "cup"): "paper cup",
    ("tv", "remote"): "tv remote",
    ("wardrobe", "door"): "wardrobe door",
    ("cup", "of", "coffee"): "coffee cup",
    ("coffee", "cup"): "coffee cup",
    ("tv", "stand"): "tv cabinet",
}

#: Nouns whose surface form is inherently plural / uncountable (never singularised).
_PLURAL_INVARIANT: frozenset[str] = frozenset({"stairs", "sushi", "glasses", "scissors"})

#: Canonical noun -> alias surface forms, for downstream typo/synonym-tolerant lookup.
NOUN_ALIASES: dict[str, tuple[str, ...]] = {
    "refrigerator": ("refridgerator", "fridge"),
    "tv": ("television",),
    "coffee cup": ("cup of coffee",),
    "tv cabinet": ("tv stand",),
}

_ARTICLES = {"the", "that", "this", "these", "those"}
_INDEFINITE = {"a", "an"}
_QUANTIFIERS = {"two", "both", "three", "four", "pair"}


def singularize(word: str) -> str:
    """Return the naive English singular of a lowercase word (plural-invariants kept)."""
    if word in _PLURAL_INVARIANT or len(word) < 3:
        return word
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("es") and word[:-2].endswith(("s", "x", "z", "ch", "sh")):
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def canonical_word(word: str) -> str:
    """Singularise then apply the typo/synonym map to one lowercase word."""
    w = singularize(word)
    return SYNONYMS.get(w, SYNONYMS.get(word, w))


def _lookup(words: tuple[str, ...]) -> str | None:
    """Look up a word tuple (last word singularised/canonicalised) in the vocab."""
    if not words:
        return None
    key = tuple(words[:-1]) + (canonical_word(words[-1]),)
    if key in PHRASES:
        return PHRASES[key]
    if len(key) == 1 and key[0] in SINGLE_NOUNS:
        return key[0]
    # try canonicalising every word (e.g. 'wardrobe doors' with odd inflection)
    key2 = tuple(canonical_word(w) for w in words)
    if key2 in PHRASES:
        return PHRASES[key2]
    return None


def match_noun(words: list[str]) -> tuple[str, list[str], str, bool]:
    """Match a noun phrase word list -> (canonical noun, attributes, raw surface, indefinite).

    Strips articles/quantifiers, peels leading attribute words, prefers the longest
    vocabulary phrase; unknown nouns fall back to the singularised last word.
    """
    core: list[str] = []
    indefinite = False
    for w in words:
        w = w.strip(",.;:'\"?!").lower()
        if not w:
            continue
        if w in _ARTICLES or w in _QUANTIFIERS:
            continue
        if w in _INDEFINITE:
            indefinite = True
            continue
        core.append(w)
    if not core:
        return "object", [], " ".join(words), indefinite
    # prefer a phrase spanning core[i:], with core[:i] all attributes (smallest i first)
    for i in range(len(core)):
        if i > 0 and core[i - 1] not in ATTRIBUTES:
            break
        noun = _lookup(tuple(core[i:]))
        if noun is not None:
            return noun, core[:i], " ".join(core[i:]), indefinite
    # unknown noun: last word is the head, known attributes peel off, extras join them
    attrs = [w for w in core[:-1] if w in ATTRIBUTES]
    extras = [w for w in core[:-1] if w not in ATTRIBUTES]
    noun = canonical_word(core[-1])
    return noun, attrs + extras, core[-1], indefinite


def find_first_noun(text: str) -> str | None:
    """Return the first vocabulary noun found in free text (3/2/1-gram scan), or None."""
    words = [w.strip(",.;:'\"?!") for w in text.lower().split()]
    words = [w for w in words if w]
    for i in range(len(words)):
        for n in (3, 2, 1):
            if i + n <= len(words):
                noun = _lookup(tuple(words[i : i + n]))
                if noun is not None:
                    return noun
    return None
