"""BasicSceneIndex — a :class:`~core.interfaces.SceneIndex` over InstanceRecords.

Provides typo/plural/synonym-tolerant label lookup (used by the toolbox and answer
heads) plus the add/merge API that real perception uses to fuse cross-frame
detections: same-label instances overlapping in 3D (IoU > MERGE_IOU) are fused —
points concatenated, AABB recomputed as a robust-core-then-per-axis 2nd/98th-
percentile trimmed box (issue #199), ``n_obs`` incremented (and ``n_views``,
issue #191's separate distinct-viewpoint counter), ``score`` kept as the max.

Label matching ladder (higher tier wins; :meth:`by_label` returns higher tiers
first, and :meth:`by_label_tiered` surfaces which tier each hit came from):
  1. exact     — canonical (lowercase-singular) match
  2. synonym   — synonym-table / vocab-bridge equivalence (fridge<->refrigerator,
                 sofa<->couch, bedside table<->night stand, ...); matches on the
                 record canonical or its declared aliases (exact string only)
  3. head-noun — the query's head noun matches a candidate's head-noun-headed label
                 ("X table" matches "table"-headed labels, "beer bottle"->"bottle")
  4. typo      — edit-distance fuzzy match, length-scaled tolerance, returned ONLY
                 when tiers 1-3 are all empty; aliases NEVER participate here
                 ('refridgerator' -> 'refrigerator')
"""
from __future__ import annotations

import json
import os
import threading
import time
from functools import lru_cache

import numpy as np

from core.interfaces import InstanceRecord, MarkerBox, MatchTier
from core.perception.colour import ColourTally, build_caption, merge_tallies, top_bins

# Re-exported for backward compatibility: this module used to define MatchTier
# itself; it now lives on core.interfaces (see SceneIndex.by_label_tiered, #24) so
# the Protocol can name it without a core -> perception import cycle.
__all__ = [
    "BasicSceneIndex",
    "MatchTier",
    "normalize_label",
    "singularize",
    "labels_foldable",
    "dump_instance_index",
]

# --------------------------------------------------------------------------- instrumentation
#
# Issues #84/#89: quantifying live detection recall (#84) and instance-count blowups (#89)
# both need a per-run dump of the LIVE instance index (classes, counts, positions, scores,
# n_obs) — the offline battery has no equivalent need (its GT mocks are the ground truth,
# not something to audit). Opt-in only, following the existing explore-debug dump's
# contract (core/heads/explore_debug.py, issue #83/#84): unset env var == this module is
# byte-identical to before instrumentation existed, no behaviour or perf change.

#: Path to append JSONL instance-index records to. Unset (default) -> dump_instance_index
#: is a no-op (single os.environ.get, no I/O).
ENV_INSTANCE_DUMP_PATH: str = "VLA_INSTANCE_DUMP_PATH"

#: Minimum seconds between periodic dumps from the SAME caller (callers pass their own
#: clock reading; see :class:`~core.perception.tracker.PerceptionPipeline`). An
#: "answer_time" tag is never throttled — always dump exactly once (see
#: :func:`dump_instance_index`'s ``force`` argument).
ENV_INSTANCE_DUMP_INTERVAL_S: str = "VLA_INSTANCE_DUMP_INTERVAL_S"
DEFAULT_INSTANCE_DUMP_INTERVAL_S: float = 10.0


def dump_instance_index(
    index: "BasicSceneIndex",
    tag: str,
    *,
    keyframes_processed: int | None = None,
    extra: dict | None = None,
) -> None:
    """Append one JSONL record describing ``index``'s current instances, if
    :data:`ENV_INSTANCE_DUMP_PATH` is set. No-op (no I/O at all) when unset.

    Record shape: ``wall_time``, ``tag`` (caller-chosen, e.g. ``"periodic"`` for a
    throttled perception-tick dump or ``"answer_time"`` for the one fired when a head
    is about to publish), ``keyframes_processed`` (caller-supplied, or ``None``),
    ``total_instances``, ``by_class`` (label -> count), and ``instances`` — one entry
    per instance with ``id``/``label``/``position`` (rounded centroid)/``score``/
    ``n_obs``, exactly the fields #84/#89 need to compare live recall and instance
    counts against ground truth, plus ``n_views`` (issue #191's separate distinct-
    viewpoint counter). ``extra`` is merged into the top-level record as-is
    (e.g. a caller-specific qtype/answer value) when given.

    Issue #101: each instance entry also carries ``aabb_min``/``aabb_max`` (rounded
    trimmed AABB corners, same rounding as ``position``) so an on(table)-style
    geometry predicate can be replayed offline from the dump alone, without needing
    the live run's full point cloud.

    Any failure (bad path, unwritable dir, etc.) is swallowed — diagnostics must never
    break the run they are observing, matching ``core.heads.explore_debug.maybe_dump``.
    """
    path = os.environ.get(ENV_INSTANCE_DUMP_PATH)
    if not path:
        return
    # Deferred import: detector.py has no reverse dependency on this module today, but
    # importing it lazily here (matching this module's existing lazy-import style for
    # perception.vocab/dimension_priors) keeps this module's own import-time surface
    # unchanged for every caller that never sets ENV_INSTANCE_DUMP_PATH.
    from core.perception.detector import answer_eligibility_reason, is_answer_eligible

    try:
        instances = index.all_instances()
        # Issue #129: `by_class` is a literal class CENSUS (a count per label), so it
        # excludes structural classes (floor/ceiling/wall/window/door/door frame/column
        # -- see core.perception.vocab.is_structural_class) the same way
        # BasicSceneIndex.countable_instances() does. The full `instances` list below
        # is left untouched -- every structural instance (including window, needed as
        # a relation anchor) still appears there with its position/AABB, so an
        # offline consumer that wants structural geometry (not a count) still has it.
        by_class: dict[str, int] = {}
        for rec in index.countable_instances():
            by_class[rec.label] = by_class.get(rec.label, 0) + 1
        ordered = sorted(instances, key=lambda r: r.instance_id)
        record: dict = {
            "wall_time": time.time(),
            "tag": tag,
            "keyframes_processed": keyframes_processed,
            "total_instances": len(instances),
            "by_class": by_class,
            "instances": [
                {
                    "id": int(rec.instance_id),
                    "label": rec.label,
                    "position": [round(float(c), 3) for c in rec.centroid],
                    # Issue #101: trimmed AABB corners, rounded like position, so an
                    # on(table)-style geometry predicate can be replayed offline
                    # against this dump alone (centroid-only was insufficient).
                    "aabb_min": [round(float(c), 3) for c in rec.aabb_min],
                    "aabb_max": [round(float(c), 3) for c in rec.aabb_max],
                    "score": round(float(rec.score), 4),
                    "n_obs": int(rec.n_obs),
                    # Issue #191: distinct-viewpoint count, separate from n_obs's
                    # dwell count -- see InstanceRecord.n_views. Dumped alongside
                    # n_obs so future diagnoses can see both without needing the
                    # live pipeline's in-memory index.
                    "n_views": int(rec.n_views),
                    # Issue #121: colour bins + caption, so a replay dump alone lets
                    # tools/perception_eval.py-style consumers audit colour coverage
                    # and accuracy without needing the live pipeline's in-memory index.
                    "caption": rec.caption,
                    "color_bins": [
                        {"name": b.name, "rgb": list(b.rgb), "fraction": round(b.fraction, 4)}
                        for b in rec.color_bins
                    ],
                    # Issue #84 gate observability: whether THIS instance would win the
                    # answer-eligibility gate right now, and why not when it doesn't --
                    # makes the gate's rejections visible in the same stream that already
                    # shows the recall/count numbers, no separate log-scraping needed.
                    "answer_eligible": is_answer_eligible(rec),
                    "eligibility_reason": answer_eligibility_reason(rec),
                }
                for rec in ordered
            ],
        }
        if extra:
            record.update(extra)
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "a") as fh:
            fh.write(json.dumps(record) + "\n")
    except Exception:  # noqa: BLE001 - diagnostics must never break the run
        pass

MERGE_IOU: float = 0.3  # 3D IoU threshold for fusing same-label instances
TYPO_MAX_DIST: int = 2  # max Levenshtein distance for the longest length band
TRIM_LO_PCT: float = 2.0
TRIM_HI_PCT: float = 98.0
# Issue #199: robust-core pre-filter ahead of the percentile trim above -- see
# _trimmed_aabb. Mirrors core.perception.fusion.FusionConfig.outlier_k/
# outlier_min_mad/outlier_min_n_for_trim (not wired to the same FusionConfig
# instance since _fuse's merge path is not itself a FusionConfig consumer; kept
# numerically identical on purpose).
OUTLIER_K: float = 4.5
OUTLIER_MIN_MAD: float = 0.02
# Issue #199 (post-review fix): below this point count, robust_core_mask is a
# noisy per-axis median/MAD estimator that spuriously trims clean, zero-outlier
# clusters (measured 0.6-40% at n=6-50 through the fusion.py pipeline this
# constant mirrors -- see FusionConfig.outlier_min_n_for_trim's comment for the
# full curve and why 40, not a lower or a still-higher count, was picked).
OUTLIER_MIN_N: int = 40


def _typo_budget(query: str, candidate: str) -> int:
    """Length-scaled Levenshtein budget for a fuzzy match; 0 disables fuzzy.

    No fuzzy match when either side is shorter than 5 chars (short words collide
    too readily: 'door'/'floor', 'tap'/'cup', 'bag'/'bed'). Edit distance <= 1 for
    length 5-7, <= 2 only for length 8+. The band is set by the SHORTER of the two
    strings, so a long candidate cannot buy tolerance a short query never earns.
    """
    n = min(len(query), len(candidate))
    if n < 5:
        return 0
    if n <= 7:
        return 1
    return 2

# Canonical synonym groups. Every member maps to the group's canonical head
# (the first element). Lookup normalises a query to its canonical head, then
# matches instances whose label shares the same head.
_SYNONYM_GROUPS: tuple[tuple[str, ...], ...] = (
    ("refrigerator", "fridge"),
    ("sofa", "couch"),
    ("television", "tv"),
    ("picture", "photo"),
)


def _build_synonym_map() -> dict[str, str]:
    m: dict[str, str] = {}
    for group in _SYNONYM_GROUPS:
        head = group[0]
        for member in group:
            m[member] = head
    return m


_SYNONYM_MAP = _build_synonym_map()


def singularize(noun: str) -> str:
    """Cheap English plural stripping to a singular canonical form.

    Handles the common regular patterns present in the challenge vocabulary
    (…ies -> …y, …ses/…xes/…zes/…ches/…shes -> drop 'es', trailing 's').
    """
    w = noun.strip().lower()
    if len(w) <= 3 or not w.endswith("s"):
        return w
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith(("ches", "shes", "sses", "xes", "zes")):
        return w[:-2]
    if w.endswith("ss"):
        return w
    return w[:-1]


def normalize_label(noun: str) -> str:
    """Canonicalise a noun: lowercase, singular, mapped through the synonym table."""
    base = singularize(noun)
    return _SYNONYM_MAP.get(base, base)


# --------------------------------------------------------------------------- label folding
#
# Issue #89: open-vocab GDINO phrase decoding routinely emits FRAGMENTS of one class
# label instead of the label itself -- a duplicated leading token ("door" -> "door
# door", "projector screen" -> "screen projector screen") or a bare head/modifier split
# off a multi-word class ("potted plant" -> "potted" / "plant"). Each fragment fails
# the exact-label-equality tracker association/scene-index merge checks against the
# label the SAME physical object was already tracked under, so it mints its own
# instance instead of fusing -- observed office_1 instance_index.jsonl fragmentation
# ("potted plant"/"potted"/"plant", "door door"/"door door frame"/"door",
# "projector screen"/"screen projector screen") inflating live instance counts 2-3x on
# a single object.

def _fold_tokens(label: str) -> frozenset[str]:
    """Whitespace-tokenised, singularised, duplicate/order-insensitive token set.

    Converting to a ``set`` is what makes both fold shapes fall out of ONE test in
    :func:`labels_foldable`: a duplicated token ("door door" -> tokens ``[door,
    door]``) collapses to the same singleton set as the plain word ("door" ->
    ``{door}``), and a bare head/modifier fragment ("potted") is trivially a *subset*
    of the multi-word label's token set ("potted plant" -> ``{potted, plant}``).
    """
    return frozenset(singularize(t) for t in normalize_label(label).split() if t)


def labels_foldable(a: str, b: str) -> bool:
    """True iff labels ``a`` and ``b`` are fragments of ONE class via a subphrase or
    duplicated-token relation (issue #89) -- i.e. one's (deduped) token set is wholly
    contained in the other's. Deliberately conservative: this is a TOKEN-SET test, not
    a substring/edit-distance test, so two genuinely distinct single-word classes never
    fold merely for looking similar -- ``"door"`` (``{door}``) is not a subset of
    ``"floor"`` (``{floor}``, no shared tokens at all), nor is ``"chair"`` a subset of
    ``"table"``. Folding only fires when a token set is an actual subset of the other:
    ``"potted"`` (``{potted}``) subset of ``"potted plant"`` (``{potted, plant}``);
    ``"door door"`` (``{door}``) equal to ``"door"`` (``{door}``); ``"door door
    frame"`` (``{door, frame}``) superset of ``"door"`` (``{door}``); ``"screen
    projector screen"`` (``{screen, projector}``) equal to ``"projector screen"``.

    Two identical empty-token labels (blank strings) never fold (guards against a
    vacuous ``frozenset() <= frozenset()``).

    NOT transitive across differing fragments of the same object that never
    individually appeared together: e.g. if ``"potted"`` and ``"plant"`` (but never the
    compound ``"potted plant"``) are both observed for the same object, they do NOT
    fold against each other directly (``{potted}`` is not a subset of ``{plant}`` or
    vice versa) -- callers rely on the compound form (or SOME shared-superset label)
    appearing at least once to anchor the fold, which matches the observed live data
    (the detector's own caption prompt is usually the full compound noun).
    """
    ta, tb = _fold_tokens(a), _fold_tokens(b)
    if not ta or not tb:
        return False
    return ta <= tb or tb <= ta


# --------------------------------------------------------------- merged-label resolution
#
# Issue #154: GDINO phrase decoding also emits labels that are TWO DIFFERENT class
# names concatenated into one token span ("couch sofa", "cabinet shelf", "elephant
# figurine horse figurine") -- a different failure from #89's SAME-class fragments
# above. #89's labels_foldable (a token-SUBSET test) is deliberately blind to which
# side is the real class: fed "cabinet shelf", it folds against BOTH a "cabinet"
# instance and a "shelf" instance, because {cabinet} and {shelf} are each a subset
# of {cabinet, shelf} -- it was built to recognise a fragment of ONE class, not to
# adjudicate which of two real classes a merged label actually names. Telling the
# two apart needs an outside arbiter of what a REAL class even is: the training
# vocabulary (core.parsing.vocab.SINGLE_NOUNS / PHRASES), not a string heuristic
# over the labels observed so far.
#
# Design decision -- fold to canonical class(es) via alias, never invent a second
# box: splitting one detection into two synthetic instances would fabricate
# geometry nobody observed (the box only ever bounded ONE detected region, and nothing
# here knows how to divide it between two classes). So a merged label always
# resolves to exactly ONE instance: its label becomes the first recognised
# canonical class (reading order), and every OTHER canonical class the label also
# names is recorded as an alias. An alias participates in the SYNONYM match tier
# (by_label_tiered), so an anchor for any of the merged classes still finds this
# instance instead of reading zero hits -- e.g. the elephant/horse figurine case
# (#154 comment): both `elephant figurine` and `horse figurine` resolve to the one
# instance actually detected, rather than neither ever matching a phantom
# `elephant figurine horse figurine` class. That is deliberately the most a
# label-only fix can promise: a detection that only ever drew ONE box around what
# turned out to be two distinct real objects needs a decode fix in the detector
# itself to truly recover both (out of scope here -- see OWNERSHIP; neither
# vocab.py nor detector.py is touched by this fix).
#
# Three steps, tried in order, all driven by the vocabulary rather than a
# similarity heuristic -- each earlier step is a MORE conservative reading than
# the next, so it always gets first refusal:
#  1. self-match -- the label's own (per-token normalised) token set equals ONE
#     real class's token set exactly, order and duplicates ignored: "door door" ->
#     {door} == "door"'s own set -> the label IS "door", just doubled. No alias --
#     this is #89's case, just resolved eagerly at creation time instead of
#     waiting for a same-label instance to already exist to fold against (which is
#     how a doubled/fragment label could end up founding and permanently naming
#     its own instance before any "clean" observation of the same object arrived).
#     Legitimate compounds are ALWAYS their own canonical vocabulary entry (that's
#     what makes them legitimate), so this step protects every one of them before
#     steps 2/3 ever run: "pyramid candle holder" is itself a vocabulary phrase,
#     so it self-matches whole and is never handed to the later steps that would
#     otherwise be tempted to read it as "pyramid" + "candle holder" (both
#     independently real classes too).
#  2. superset -- only reached when step 1 fails. The label's token set is a
#     STRICT subset of exactly one real class's token set: "map wall" ({map,
#     wall}) is a subset of "map wall decal"'s own {map, wall, decal} -- read as
#     that ONE bigger class with an internal word ("decal") dropped, not as
#     "map" + "wall" concatenated. This has to run BEFORE decomposition: by
#     token shape alone a dropped-middle-word fragment of a real 3-word class is
#     indistinguishable from two real shorter classes standing side by side, and
#     "one real object, its longer name partly dropped" is the safer read of
#     that ambiguity (it costs nothing the object didn't already have; the
#     alternative invents a second class outright). This mirrors #89's existing
#     subset-fold philosophy (:func:`labels_foldable`), just grounded against the
#     real vocabulary instead of whichever instance happens to already be in the
#     index.
#  3. decompose -- only reached when steps 1 and 2 both fail (the label is not,
#     as a whole, any real class, nor a fragment of exactly one). Try to
#     partition the label's word sequence, left to right, into two or more
#     consecutive spans that EACH equal a real class exactly (a small
#     backtracking search over canonical class boundaries -- observed merged
#     labels top out around 4 tokens, so this is cheap regardless of vocabulary
#     size). Succeeds only when the ENTIRE label is accounted for by real classes
#     with nothing left over: "elephant figurine horse figurine" -> "elephant
#     figurine" + "horse figurine" (both real); "cabinet shelf" -> "cabinet" +
#     "shelf" (both real, single-word). If no such full partition exists (e.g. one
#     span is real but a leftover token names nothing in the vocabulary, as in the
#     rare "cabinet bedside file cabinet"), this deliberately gives up and leaves
#     the label untouched -- guessing which part is genuine and silently dropping
#     the rest would risk erasing a legitimate detection outright, which is worse
#     than leaving a rare unresolved label exactly as today.
#
# Scope note: only multi-word raw labels are ever considered (see the token-count
# guard in :func:`_resolve_merged_label`). A single-word label is never the merged-
# label bug by construction (there is nothing to have merged), so this never
# touches e.g. a bare "couch" detection -- normalize_label's existing couch<->sofa
# synonym handling is untouched and out of scope for this fix.


@lru_cache(maxsize=1)
def _canonical_class_tokens() -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Every canonical class in the training vocabulary
    (:data:`core.parsing.vocab.SINGLE_NOUNS` / :data:`core.parsing.vocab.PHRASES`
    values), each pre-tokenised through this module's own per-token
    :func:`normalize_label` so it compares to an incoming label on equal footing
    (the vocabulary's own "couch" and an incoming "sofa" both normalise to the
    token "sofa"). Cached: the vocabulary is a fixed module constant, built once.
    Deferred import (matching this module's existing lazy-import style for
    cross-package references) -- ``core.parsing.vocab`` has no reverse dependency
    on this module today, but importing it lazily keeps this module's own
    import-time surface unchanged either way.
    """
    from core.parsing.vocab import PHRASES, SINGLE_NOUNS

    names = set(SINGLE_NOUNS) | set(PHRASES.values())
    return tuple(
        (name, tuple(normalize_label(t) for t in name.split()))
        for name in sorted(names)
    )


def _normalized_tokens(label: str) -> tuple[str, ...]:
    """Whitespace-tokenised, per-token :func:`normalize_label`-normalised, ORDER-
    preserving tokens of ``label`` -- the order-sensitive counterpart to
    :func:`_fold_tokens` (which discards order/duplicates for the subset test).
    """
    return tuple(normalize_label(t) for t in label.strip().lower().split() if t)


def _canonical_token_set_match(tokens: tuple[str, ...]) -> str | None:
    """The canonical class name whose (deduped) token set exactly equals
    ``tokens``'s (deduped) set, or ``None`` if no such class exists. Order- and
    duplicate-insensitive on purpose: "door door" ({door}) and "coffee table
    table" ({coffee, table}) both need to reach their whole real class this way.
    """
    target = frozenset(tokens)
    if not target:
        return None
    for name, name_tokens in _canonical_class_tokens():
        if frozenset(name_tokens) == target:
            return name
    return None


def _canonical_superset_match(tokens: tuple[str, ...]) -> str | None:
    """A canonical class whose token set STRICTLY contains ``tokens``'s (deduped)
    set -- i.e. ``label`` reads as a FRAGMENT of one bigger real class with some
    internal word dropped ("map wall" missing "decal" from "map wall decal";
    "pyramid holder" missing "candle" from "pyramid candle holder"), rather than
    smaller real classes concatenated. Checked BEFORE decomposition on purpose:
    "one real object, its longer name partly dropped" is the more conservative
    reading of an ambiguous shorter label -- it matches #89's existing subset-
    fold philosophy (:func:`labels_foldable`), just grounded against the real
    vocabulary instead of whichever instance happens to already be in the index.
    A subset-of-decompose candidate (module note: "map wall" -> "map" + "wall",
    both independently real single-word classes) would otherwise be indistinguishable
    from this fragment case by shape alone; without this check running first,
    the decomposer would confidently but wrongly split what is really a single
    truncated detection into two phantom classes neither of which the object
    actually is. Returns the SMALLEST such superset class when more than one
    exists (invents the least beyond what was actually observed), or ``None``.
    """
    target = frozenset(tokens)
    if not target:
        return None
    best: tuple[int, str] | None = None
    for name, name_tokens in _canonical_class_tokens():
        superset = frozenset(name_tokens)
        if superset > target:  # strict superset only -- equality is step 1's job
            if best is None or len(superset) < best[0]:
                best = (len(superset), name)
    return best[1] if best else None


def _decompose_into_canonical_classes(tokens: tuple[str, ...]) -> list[str] | None:
    """Partition ``tokens`` left to right into consecutive spans that each equal a
    canonical class's token sequence exactly (order-sensitive), covering EVERY
    token with nothing left over. Returns the ordered list of matched class names,
    or ``None`` if no such full partition exists.

    Backtracking search, longest-span-first at each position (so a more specific
    class like "door frame" is tried before a shorter prefix like "door" when both
    fit -- this only affects WHICH accepting partition is found first when more
    than one exists, never whether one exists). Memoised on position: each
    position is resolved at most once. Cheap regardless of vocabulary size --
    observed merged labels top out around 4 tokens.
    """
    classes = sorted(_canonical_class_tokens(), key=lambda nc: -len(nc[1]))
    n = len(tokens)
    memo: dict[int, list[str] | None] = {}

    def helper(i: int) -> list[str] | None:
        if i == n:
            return []
        if i in memo:
            return memo[i]
        memo[i] = None  # guard against pathological re-entry; classes are static
        result: list[str] | None = None
        for name, name_tokens in classes:
            span = len(name_tokens)
            if span == 0 or i + span > n:
                continue
            if tokens[i : i + span] == name_tokens:
                rest = helper(i + span)
                if rest is not None:
                    result = [name] + rest
                    break
        memo[i] = result
        return result

    return helper(0)


def _resolve_merged_label(label: str) -> tuple[str, tuple[str, ...]] | None:
    """Resolve a raw incoming detection ``label`` against the training vocabulary
    (issue #154). Returns ``None`` when no vocabulary-grounded correction applies
    (leave the label exactly as given -- see the module note above), else
    ``(canonical_label, extra_aliases)``: ``extra_aliases`` is empty for a same-
    class fragment/duplicate ("door door" -> ("door", ())), and holds every OTHER
    real class the label also names for a genuine multi-class merge ("cabinet
    shelf" -> ("cabinet", ("shelf",))).
    """
    tokens = _normalized_tokens(label)
    if len(tokens) < 2:
        # a single-word label is never the merged-label bug -- nothing to have
        # merged (see the module-level scope note).
        return None

    whole = _canonical_token_set_match(tokens)
    if whole is not None:
        return normalize_label(whole), ()

    fragment_of = _canonical_superset_match(tokens)
    if fragment_of is not None:
        return normalize_label(fragment_of), ()

    parts = _decompose_into_canonical_classes(tokens)
    if not parts:
        return None
    ordered: list[str] = []
    for p in parts:
        canon = normalize_label(p)
        if canon not in ordered:
            ordered.append(canon)
    if len(ordered) < 2:
        # every decomposed chunk collapsed to the SAME class after synonym
        # folding (e.g. "couch" + "couch") -- a fragment/duplicate, not a merge.
        return ordered[0], ()
    return ordered[0], tuple(ordered[1:])


def _resolve_incoming_label(rec: InstanceRecord) -> InstanceRecord:
    """Apply issue #154's vocabulary-grounded label resolution to a fresh
    detection before it reaches merge-target lookup / instance creation in
    :meth:`BasicSceneIndex.add`. A no-op when :func:`_resolve_merged_label` finds
    no correction. Mutates ``rec`` in place (label + aliases) rather than
    copying: by the time a detection reaches ``add()`` it is a fresh, single-
    owner record built by the caller (tracker/fusion) purely to be handed off
    here -- matching how ``add()`` already treats it (appended directly into
    ``self._instances`` on the new-instance path with no defensive copy).
    """
    resolved = _resolve_merged_label(rec.label)
    if resolved is None:
        return rec
    canonical, extra_aliases = resolved
    if canonical == rec.label and not extra_aliases:
        return rec
    rec.label = canonical
    if extra_aliases:
        merged_aliases = list(rec.aliases)
        for alias in extra_aliases:
            if alias not in merged_aliases:
                merged_aliases.append(alias)
        rec.aliases = tuple(merged_aliases)
    return rec


def _levenshtein(a: str, b: str) -> int:
    """Standard Levenshtein edit distance (iterative two-row DP)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[-1]


def _aabb_iou_3d(min_a, max_a, min_b, max_b) -> float:
    """Axis-aligned 3D IoU of two boxes given as (3,) min/max corners."""
    lo = np.maximum(min_a, min_b)
    hi = np.minimum(max_a, max_b)
    inter_dims = np.clip(hi - lo, 0.0, None)
    inter = float(np.prod(inter_dims))
    if inter <= 0.0:
        return 0.0
    vol_a = float(np.prod(np.clip(max_a - min_a, 0.0, None)))
    vol_b = float(np.prod(np.clip(max_b - min_b, 0.0, None)))
    union = vol_a + vol_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def _trimmed_aabb(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Robust-core + per-axis 2nd/98th-percentile trimmed AABB of an (M, 3) cloud.

    Issue #199: a merge accumulates points from every re-observation, so a genuine
    outlier return (a stray background point one fuse chained in, or a far corner
    surviving from a bad component pick before this issue's fusion-side fixes)
    can still reach this function. :func:`~core.perception.fusion.robust_core_mask`
    (median +/- ``OUTLIER_K`` MADs per axis) drops it FIRST, without touching a
    clean cluster (see that function's docstring for why a MAD test, not a
    percentile-rank one, leaves a well-bounded cloud untouched); the percentile
    trim then runs on the surviving points exactly as before, so a genuinely large
    but clean cluster (e.g. the deliberate #104 accumulated-walk fixture) is
    unaffected. Falls back to the full point set if the core would drop it below
    2 points (a percentile trim needs at least that many to be meaningful).

    Issue #199 (post-review fix): the robust-core step above only runs once the
    cloud clears :data:`OUTLIER_MIN_N` points -- see that constant's comment.
    Below it, the median/MAD estimator is too noisy to trust and every point
    goes straight to the percentile trim, matching this function's behaviour
    before issue #199 (unaffected by this fix either way, since the percentile
    trim itself is unchanged).
    """
    from core.perception.fusion import robust_core_mask

    core = points
    if len(points) >= OUTLIER_MIN_N:
        mask = robust_core_mask(points, OUTLIER_K, OUTLIER_MIN_MAD)
        if mask.sum() >= 2:
            core = points[mask]
    lo = np.percentile(core, TRIM_LO_PCT, axis=0)
    hi = np.percentile(core, TRIM_HI_PCT, axis=0)
    return lo.astype(float), hi.astype(float)


class BasicSceneIndex:
    """Mutable in-memory scene index. Implements the SceneIndex protocol.

    Thread safety (issue #88): a live run now feeds this index from a dedicated
    perception worker thread (core.perception.async_pipeline.AsyncPerceptionWorker)
    while the tick thread's heads concurrently read it (by_label/all_instances/...)
    to resolve answers. An internal RLock (``self._lock``) guards every read and
    write method so index mutation (add/remove/_fuse) and iteration (by_label_tiered,
    all_instances) are never interleaved on the same underlying list — without it, a
    reader iterating ``self._instances`` while the worker thread deletes from it can
    raise ``RuntimeError: list changed size during iteration``. Callers (nav/heads)
    need no changes: the lock lives entirely inside this class.
    """

    def __init__(self, instances: list[InstanceRecord] | None = None) -> None:
        self._lock = threading.RLock()
        self._instances: list[InstanceRecord] = list(instances or [])
        self._next_id = 1 + max(
            (r.instance_id for r in self._instances), default=-1
        )
        # Issue #121: raw per-instance colour tally (point counts + RGB sums per
        # scheme name), kept OUTSIDE InstanceRecord so it can accumulate losslessly
        # across every re-observation — InstanceRecord.color_bins only ever holds the
        # up-to-3 dominant bins derived FROM this, recomputed on every fuse. Seeded
        # only for instances built through add()/merge_into() with a colour_obs
        # (the live perception path); GT-constructed indices (passed instances=...)
        # never populate this, unaffected.
        self._colour_tally: dict[int, ColourTally] = {}

    # ------------------------------------------------------------- read protocol

    def all_instances(self):
        with self._lock:
            return list(self._instances)

    def countable_instances(self):
        """``all_instances()`` minus structural classes (issue #129).

        The separate structural channel the census needs: floor/ceiling/wall/window/
        door/door frame/column (:func:`core.perception.vocab.is_structural_class`)
        are proposed and tracked like any other class, but GT annotates exactly one
        of each per scene while the detector re-proposes dozens of fragments of the
        same physical shell surface -- inflating any "how many X" enumeration or
        class census with instances nobody asks to count (docs/question_analysis.md:
        no training question counts a structural class).

        Deliberately NOT wired into :meth:`by_label`/:meth:`by_label_tiered`/
        :meth:`all_instances` -- those remain the anchor/target lookup surface every
        relation clause depends on (``"the sofa below a WINDOW"`` needs `window`,
        itself structural, to keep resolving exactly as before). This method is
        additive: only a caller that explicitly wants the countable/census view (this
        module's own :func:`dump_instance_index`) uses it; every existing consumer of
        the three methods above is completely unaffected by this method's existence.
        """
        # Deferred import: perception.vocab imports normalize_label from this module,
        # so we consult it at call time to avoid a module-load cycle (matching the
        # existing by_label_tiered import style).
        from core.perception.vocab import is_structural_class

        with self._lock:
            snapshot = list(self._instances)
        return [rec for rec in snapshot if not is_structural_class(rec.label)]

    def remove(self, instance_id: int) -> bool:
        """Drop an instance by id; return True if one was removed.

        Used by track decay (H15a) to prune one-frame ghosts. No-op (returns False)
        if the id is absent. Never mutates ``_next_id`` — freed ids are not recycled,
        so a pruned ghost's id cannot be silently reused by a later real object.
        """
        with self._lock:
            for i, rec in enumerate(self._instances):
                if rec.instance_id == instance_id:
                    del self._instances[i]
                    self._colour_tally.pop(instance_id, None)
                    return True
            return False

    def marker_for(self, record: InstanceRecord) -> MarkerBox:
        """Prior-clamped marker for a record — the H12 marker seam.

        The trimmed AABB an instance carries can be a single-viewpoint *under*-box
        (red-team OR-F6) or, per issue #201's live measurement, an over-fused
        *over*-box; the raw ``record.to_marker()`` would publish either verbatim and
        shed IoU against the GT hull either way. This routes the record through the
        per-class dimension prior (:mod:`core.perception.dimension_priors`): clamp
        every axis up to the class-min, inflate the least-observed axis toward
        class-typical only when the instance signals under-observation, AND (#201)
        cap every axis down at the class-typical cap, centre preserved. A GT-perfect /
        well-observed / in-bound box is returned identical to ``record.to_marker()``.

        Marker-path owners (``heads/object_ref.py``, ``fsm/floors.py``) should publish
        ``index.marker_for(rec)`` in place of ``rec.to_marker()`` — the seam lives here
        so the clamp is applied wherever the scene index is in scope. Deferred import
        avoids a load cycle (dimension_priors imports normalize_label from this module).
        """
        from core.perception.dimension_priors import clamp_record_marker

        return clamp_record_marker(record)

    def next_id(self) -> int:
        """The instance_id the next fresh (non-merged) instance would receive.

        Read-only peek used by the tracker to mint ids for unmatched detections;
        ``add`` still owns id assignment and will reassign on collision.
        """
        with self._lock:
            return self._next_id

    def by_label(self, noun: str):
        """Typo/plural/synonym-tolerant lookup; returns matching instances, best-first.

        Tiers are tried exact -> synonym -> head-noun -> typo (see the module
        docstring). The typo tier is a short-circuit: it contributes ONLY when the
        exact, synonym and head-noun tiers are all empty, so an exact match is never
        polluted by fuzzy cousins.
        """
        return [rec for rec, _ in self.by_label_tiered(noun)]

    def by_label_tiered(self, noun: str) -> list[tuple[InstanceRecord, MatchTier]]:
        """Like :meth:`by_label` but pairs each hit with its :class:`MatchTier`.

        Lets consumers (e.g. ``resolve``/audit) prefer stronger-tier candidates before
        superlative ranking without breaking the flat-list return type of
        :meth:`by_label`.
        """
        # Deferred import: perception.vocab imports normalize_label from this module,
        # so we consult it at call time to avoid a module-load cycle.
        from core.perception.vocab import bridge_synonyms, head_noun

        query = normalize_label(noun)
        query_head = head_noun(query)
        syns = bridge_synonyms(query)  # normalised object-bridge equivalents

        exact: list[InstanceRecord] = []
        syn: list[InstanceRecord] = []
        head: list[InstanceRecord] = []
        typo: list[InstanceRecord] = []
        with self._lock:
            instances_snapshot = list(self._instances)
        for rec in instances_snapshot:
            canon = normalize_label(rec.label)
            if canon == query:
                exact.append(rec)
                continue
            # synonym tier: the vocab bridge, plus the record's declared aliases —
            # but aliases are matched by EXACT string only, never fuzzily (colour
            # aliases like 'yellow' are edit-distance 2 from 'pillow').
            alias_canons = [normalize_label(a) for a in rec.aliases]
            if canon in syns or query in bridge_synonyms(canon) or query in alias_canons:
                syn.append(rec)
                continue
            # head-noun tier: modifier-stripped class-word match, either direction
            # ("beer bottle" query vs "bottle" label, or "table" query vs "X table").
            if head_noun(canon) == query_head:
                head.append(rec)
                continue
            # typo tier candidate (length-gated); aliases excluded on purpose.
            budget = _typo_budget(query, canon)
            if budget and _levenshtein(canon, query) <= budget:
                typo.append(rec)

        tiered: list[tuple[InstanceRecord, MatchTier]] = []
        tiered += [(r, MatchTier.EXACT) for r in exact]
        tiered += [(r, MatchTier.SYNONYM) for r in syn]
        tiered += [(r, MatchTier.HEAD_NOUN) for r in head]
        # typo short-circuit: only when every stronger tier is empty.
        if not tiered:
            tiered += [(r, MatchTier.TYPO) for r in typo]
        return tiered

    # -------------------------------------------------------------- write / merge

    def add(
        self, rec: InstanceRecord, colour_obs: ColourTally | None = None
    ) -> InstanceRecord:
        """Add an observation, fusing into an existing same-label instance when
        their 3D AABB IoU exceeds MERGE_IOU. Returns the surviving record.

        ``colour_obs`` (issue #121): this observation's raw pixel-colour tally
        (see :mod:`core.perception.colour`), if the caller computed one. On a
        brand-new instance it seeds that instance's colour bins/caption directly;
        on a fuse it is folded into the target's running tally by :meth:`_fuse`.
        ``None`` (the default, and every non-live caller) leaves colour untouched.
        """
        with self._lock:
            # Issue #154: resolve a GDINO-merged/fragmented label against the
            # training vocabulary BEFORE it can ever found a new instance or
            # participate in merge-target lookup -- see the module note above
            # _resolve_merged_label.
            rec = _resolve_incoming_label(rec)
            target = self._find_merge_target(rec)
            if target is None:
                if rec.instance_id in (r.instance_id for r in self._instances):
                    rec = InstanceRecord(
                        instance_id=self._next_id,
                        label=rec.label,
                        score=rec.score,
                        n_obs=rec.n_obs,
                        centroid=rec.centroid,
                        aabb_min=rec.aabb_min,
                        aabb_max=rec.aabb_max,
                        points=rec.points,
                        caption=rec.caption,
                        aliases=rec.aliases,
                        n_views=rec.n_views,  # issue #191
                    )
                self._next_id = max(self._next_id, rec.instance_id + 1)
                self._instances.append(rec)
                if colour_obs is not None:
                    self._seed_colour(rec, colour_obs)
                return rec
            self._fuse(target, rec, colour_obs=colour_obs)
            return target

    def _seed_colour(self, rec: InstanceRecord, colour_obs: ColourTally) -> None:
        """First colour observation for a brand-new instance (issue #121)."""
        self._colour_tally[rec.instance_id] = colour_obs
        rec.color_bins = top_bins(colour_obs)
        rec.caption = build_caption(rec.color_bins, rec.extents)

    def merge_into(
        self, instance_id: int, rec: InstanceRecord, colour_obs: ColourTally | None = None
    ) -> InstanceRecord:
        """Fuse ``rec`` (one fresh single-frame observation) directly into the existing
        instance identified by ``instance_id`` — trusting that decision unconditionally,
        with NO re-derivation of label/IoU compatibility.

        Issue #89 (live instance explosion) / #84 item 3 (label-variant ghosts): the
        tracker's own association (:func:`core.perception.tracker.associate`) already
        decided ``rec`` belongs to ``instance_id`` — centroid distance under the tracker
        gate, label compatibility through the FULL alias bridge
        (:func:`core.perception.tracker.canonical_for_match`, which folds in
        ``core.parsing.vocab.NOUN_ALIASES`` on top of this module's own narrower
        :data:`_SYNONYM_MAP`). :meth:`add`'s :meth:`_find_merge_target` used to be the
        ONLY way a matched detection reached the index, and it independently re-derives
        the merge decision from this module's own :func:`normalize_label` (missing any
        alias only known to ``NOUN_ALIASES``, e.g. 'refridgerator' vs 'refrigerator') AND
        an AABB IoU > :data:`MERGE_IOU` (0.3) test — a MUCH stricter, different
        criterion than the tracker's centroid gate (0.75 m). Under live pose jitter, or
        for a label that only the broader alias bridge recognises, these two independent
        decisions routinely disagree; when they do, ``add`` falls through to its
        "new instance" branch, finds ``rec.instance_id`` already taken by the very
        instance association just matched it to, and mints a completely FRESH id —
        silently defeating the association and spawning a duplicate instance for
        something already being tracked. That is the dominant contributor to #89's
        78-141 instance overcounts on chair-dense live scenes.

        Falls back to a normal :meth:`add` (which still runs its own IoU-based dedup)
        only if ``instance_id`` is not actually present — defensive; the tracker never
        calls this with an id it did not itself just read off ``all_instances()``.
        """
        with self._lock:
            # Issue #154: canonicalise/alias BEFORE fusing so a merged-label
            # observation's extra alias (e.g. "horse figurine" off an
            # "elephant figurine horse figurine" detection) survives into the
            # target even when the target's OWN label was already the plain,
            # unmerged form -- see _fuse's alias-union for the other half of
            # this: label wins from whichever side is the established target,
            # aliases accumulate from both sides.
            rec = _resolve_incoming_label(rec)
            for existing in self._instances:
                if existing.instance_id == instance_id:
                    self._fuse(existing, rec, colour_obs=colour_obs)
                    return existing
            return self.add(rec, colour_obs=colour_obs)

    def _find_merge_target(self, rec: InstanceRecord) -> InstanceRecord | None:
        # Called only from within add()'s locked section (RLock: reentrant).
        q = normalize_label(rec.label)
        best: InstanceRecord | None = None
        best_iou = MERGE_IOU
        for existing in self._instances:
            # Issue #89: exact-canonical equality OR a subphrase/duplicated-token fold
            # relation (labels_foldable) -- a "potted"/"door door" fragment still needs
            # to find its established "potted plant"/"door" instance here, not just via
            # the tracker's own association (BasicSceneIndex.add/_find_merge_target is
            # also reached directly by callers that bypass the tracker, e.g. scripted
            # replay/tests).
            if normalize_label(existing.label) != q and not labels_foldable(
                existing.label, rec.label
            ):
                continue
            iou = _aabb_iou_3d(
                existing.aabb_min, existing.aabb_max, rec.aabb_min, rec.aabb_max
            )
            if iou > best_iou:
                best_iou = iou
                best = existing
        return best

    def _fuse(
        self,
        target: InstanceRecord,
        other: InstanceRecord,
        colour_obs: ColourTally | None = None,
    ) -> None:
        """Fuse ``other`` into ``target`` in place: concat points, recompute the
        trimmed AABB and centroid, bump n_obs (and n_views, issue #191), keep max
        score.

        Issue #121: ``colour_obs`` (this observation's raw pixel-colour tally, if
        any) is folded into ``target``'s running :class:`ColourTally` — ACCUMULATED
        across every re-observation, never overwritten by the latest keyframe — and
        ``target.color_bins``/``target.caption`` are recomputed from the merged
        totals. ``None`` (no colour observation this frame, e.g. the whole cluster
        fell outside the panorama) leaves any existing colour untouched.

        Issue #154: ``other.aliases`` are unioned into ``target.aliases``
        (target's own aliases kept first, ``other``'s new ones appended, no
        duplicates) rather than discarded — a merged-label detection's extra
        alias (e.g. "horse figurine" off a canonicalised "elephant figurine
        horse figurine" observation, see :func:`_resolve_incoming_label`) must
        stay reachable even when it fuses by IoU into an already-established
        plain-labelled target that predates it and never carried that alias.
        """
        # Deferred import: dimension_priors imports normalize_label from this
        # module, so a module-level import here would be a load cycle.
        from core.perception.dimension_priors import cap_fused_extent, floor_degenerate_aabb

        clouds = [p for p in (target.points, other.points) if p is not None and len(p)]
        if clouds:
            fused = np.vstack(clouds).astype(float)
            target.points = fused
        else:
            fused = None
        if fused is not None:
            lo, hi = _trimmed_aabb(fused)
        else:
            # no points to trim with — union the boxes as a fallback
            lo = np.minimum(target.aabb_min, other.aabb_min)
            hi = np.maximum(target.aabb_max, other.aabb_max)
        # Issue #125: the trimmed/unioned box can still land (near-)flat on an axis
        # (duplicate/collinear points, or two already-degenerate boxes unioned) —
        # raise only that broken axis to a plausible floor, centre preserved.
        lo, hi = floor_degenerate_aabb(lo, hi, target.label)
        # Issue #104: the missing other half of the dimension-sanity table — a
        # match-time veto (tracker._match_plausible, #94/#153) cannot stop an
        # accumulated box drifting past a plausible size through many individually-
        # safe co-located merges, so this caps the RESULT of every fuse instead.
        # Fails open (no-op) for classes with no prior and boxes already in bound.
        lo, hi = cap_fused_extent(lo, hi, target.label, points=fused)
        target.aabb_min = lo
        target.aabb_max = hi
        target.centroid = (lo + hi) / 2.0
        target.n_obs += other.n_obs
        # Issue #191: n_views is a SEPARATE counter from n_obs -- the tracker's
        # associate() already decided whether this observation counts toward it
        # (0 on `other.n_views` when the pose gate vetoed it, 1 otherwise) before
        # calling merge_into/add; this just accumulates that decision the same
        # way n_obs accumulates its own count, unconditionally.
        target.n_views += other.n_views
        target.score = max(target.score, other.score)
        if other.aliases:
            merged_aliases = list(target.aliases)
            for alias in other.aliases:
                if alias not in merged_aliases:
                    merged_aliases.append(alias)
            target.aliases = tuple(merged_aliases)
        if colour_obs is not None:
            prior = self._colour_tally.get(target.instance_id)
            merged = merge_tallies(prior, colour_obs) if prior is not None else colour_obs
            self._colour_tally[target.instance_id] = merged
            target.color_bins = top_bins(merged)
            target.caption = build_caption(target.color_bins, target.extents)
