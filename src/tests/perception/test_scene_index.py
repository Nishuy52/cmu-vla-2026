"""BasicSceneIndex behaviour: label normalisation, synonym + typo tolerance,
and the IoU-gated merge/fuse API."""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from core.interfaces import InstanceRecord, SceneIndex
from core.perception.scene_index import (
    ENV_INSTANCE_DUMP_PATH,
    BasicSceneIndex,
    MatchTier,
    dump_instance_index,
    labels_foldable,
    normalize_label,
    singularize,
)


def _rec(instance_id, label, cmin, cmax, points=None, n_obs=1, score=0.5, aliases=()):
    amin = np.asarray(cmin, dtype=float)
    amax = np.asarray(cmax, dtype=float)
    return InstanceRecord(
        instance_id=instance_id,
        label=label,
        score=score,
        n_obs=n_obs,
        centroid=(amin + amax) / 2.0,
        aabb_min=amin,
        aabb_max=amax,
        points=None if points is None else np.asarray(points, dtype=float),
        aliases=aliases,
    )


def _box_points(cmin, cmax, n=6):
    """Deterministic grid of points filling a box, for percentile trimming."""
    xs = np.linspace(cmin[0], cmax[0], n)
    ys = np.linspace(cmin[1], cmax[1], n)
    zs = np.linspace(cmin[2], cmax[2], n)
    gx, gy, gz = np.meshgrid(xs, ys, zs)
    return np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])


# ---------------------------------------------------------------- normalisation


def test_singularize_common_plurals():
    assert singularize("sofas") == "sofa"
    assert singularize("pillows") == "pillow"
    assert singularize("boxes") == "box"
    assert singularize("vases") == "vase"
    assert singularize("photos") == "photo"


def test_normalize_folds_synonyms():
    assert normalize_label("fridge") == "refrigerator"
    assert normalize_label("refrigerator") == "refrigerator"
    assert normalize_label("couch") == "sofa"
    assert normalize_label("tv") == "television"
    assert normalize_label("photos") == "picture"


# ------------------------------------------------------------------ by_label


def _index_with_labels():
    return BasicSceneIndex(
        [
            _rec(1, "refrigerator", [0, 0, 0], [1, 1, 2]),
            _rec(2, "sofa", [3, 0, 0], [5, 1, 1]),
            _rec(3, "television", [0, 3, 0], [1, 4, 1]),
            _rec(4, "picture", [5, 5, 1], [6, 6, 2]),
        ]
    )


def test_by_label_exact_and_plural():
    idx = _index_with_labels()
    assert [r.instance_id for r in idx.by_label("refrigerator")] == [1]
    assert [r.instance_id for r in idx.by_label("sofas")] == [2]


def test_by_label_synonym():
    idx = _index_with_labels()
    assert [r.instance_id for r in idx.by_label("fridge")] == [1]
    assert [r.instance_id for r in idx.by_label("couch")] == [2]
    assert [r.instance_id for r in idx.by_label("tv")] == [3]
    assert [r.instance_id for r in idx.by_label("photo")] == [4]


def test_by_label_typo_refridgerator():
    # the canonical challenge-data misspelling must resolve to refrigerator
    idx = _index_with_labels()
    hits = idx.by_label("refridgerator")
    assert [r.instance_id for r in hits] == [1]


def test_by_label_typo_within_edit_distance_2():
    idx = _index_with_labels()
    assert [r.instance_id for r in idx.by_label("picure")] == [4]  # dist 1
    # far-off word matches nothing
    assert idx.by_label("elephant") == []


def test_by_label_orders_exact_before_typo():
    idx = BasicSceneIndex(
        [
            _rec(1, "table", [0, 0, 0], [1, 1, 1]),
            _rec(2, "cable", [3, 0, 0], [4, 1, 1]),  # edit distance 1 from 'table'
        ]
    )
    hits = idx.by_label("table")
    assert hits[0].instance_id == 1  # exact first


# ----------------------------------------------------- H10: bridge / tier discipline


def test_by_label_bridge_pair_matches_live():
    # bedside table <-> night stand: the vocab bridge must fire on the live matcher
    idx = BasicSceneIndex(
        [
            _rec(1, "night stand", [0, 0, 0], [1, 1, 1]),
            _rec(2, "nightstand", [3, 0, 0], [4, 1, 1]),
        ]
    )
    assert sorted(r.instance_id for r in idx.by_label("bedside table")) == [1, 2]
    # and the reverse direction
    idx2 = BasicSceneIndex([_rec(1, "plant", [0, 0, 0], [1, 1, 1])])
    assert [r.instance_id for r in idx2.by_label("potted plant")] == [1]


def test_by_label_bridge_is_synonym_tier():
    idx = BasicSceneIndex([_rec(1, "night stand", [0, 0, 0], [1, 1, 1])])
    tiered = idx.by_label_tiered("bedside table")
    assert [(r.instance_id, t) for r, t in tiered] == [(1, MatchTier.SYNONYM)]


def test_by_label_head_noun_tier():
    # "beer bottle" -> "bottle"; "X table" matches "table"-headed labels
    idx = BasicSceneIndex(
        [
            _rec(1, "bottle", [0, 0, 0], [1, 1, 1]),
            _rec(2, "coffee table", [3, 0, 0], [4, 1, 1]),
        ]
    )
    assert [r.instance_id for r in idx.by_label("beer bottle")] == [1]
    assert [r.instance_id for r in idx.by_label("table")] == [2]
    # a modified query NOT in the bridge lands in the head-noun tier
    # ("wooden table" -> "coffee table", both head "table")
    assert idx.by_label_tiered("wooden table")[0][1] == MatchTier.HEAD_NOUN


def test_typo_short_circuits_when_exact_present():
    # THE door pollution case: exact 'door' present => no 'floor'/'book' typo cousins
    idx = BasicSceneIndex(
        [
            _rec(1, "door", [0, 0, 0], [1, 1, 1]),
            _rec(2, "floor", [3, 0, 0], [4, 1, 1]),  # lev('door','floor') == 2
            _rec(3, "book", [6, 0, 0], [7, 1, 1]),
        ]
    )
    assert [r.instance_id for r in idx.by_label("door")] == [1]


def test_typo_length_gating_short_nouns_never_fuzzy():
    # 3-4 letter nouns must never fuzzy-match (door/floor, tap/cup, bag/bed)
    idx = BasicSceneIndex([_rec(1, "floor", [0, 0, 0], [1, 1, 1])])
    assert idx.by_label("door") == []  # query len 4 -> no fuzzy
    idx2 = BasicSceneIndex([_rec(1, "cup", [0, 0, 0], [1, 1, 1])])
    assert idx2.by_label("tap") == []  # len 3
    idx3 = BasicSceneIndex([_rec(1, "bed", [0, 0, 0], [1, 1, 1])])
    assert idx3.by_label("bag") == []  # len 3


def test_typo_length_gating_bands():
    # 5-7 chars: distance 1 ok, distance 2 rejected
    idx = BasicSceneIndex([_rec(1, "pillow", [0, 0, 0], [1, 1, 1])])  # len 6
    assert [r.instance_id for r in idx.by_label("pillo")] == [1]  # dist 1 ok
    assert idx.by_label("yellow") == []  # dist 2, band 5-7 -> rejected
    # 8+ chars: distance 2 ok
    idx2 = BasicSceneIndex([_rec(1, "refrigerator", [0, 0, 0], [1, 1, 1])])
    assert [r.instance_id for r in idx2.by_label("refridgerator")] == [1]


def test_alias_never_participates_in_fuzzy_match():
    # colour-alias 'yellow' is lev-2 from 'pillow' but must NOT phantom-match 'pillow'
    idx = BasicSceneIndex(
        [_rec(1, "lamp", [0, 0, 0], [1, 1, 1], aliases=("black", "brown", "yellow"))]
    )
    assert idx.by_label("pillow") == []


def test_alias_still_matches_exactly():
    # exact alias match stays in the synonym tier (aliases lose only fuzzy privileges)
    idx = BasicSceneIndex(
        [_rec(1, "sofa", [0, 0, 0], [1, 1, 1], aliases=("settee",))]
    )
    hits = idx.by_label_tiered("settee")
    assert [(r.instance_id, t) for r, t in hits] == [(1, MatchTier.SYNONYM)]


# --------------------------------------------------------------------- merge


def test_merge_fuses_overlapping_same_label():
    a_min, a_max = [0, 0, 0], [1, 1, 1]
    b_min, b_max = [0.2, 0.2, 0.2], [1.2, 1.2, 1.2]  # heavy overlap -> IoU > 0.3
    idx = BasicSceneIndex(
        [_rec(1, "chair", a_min, a_max, points=_box_points(a_min, a_max), n_obs=2)]
    )
    incoming = _rec(2, "chair", b_min, b_max, points=_box_points(b_min, b_max), n_obs=3, score=0.9)
    survivor = idx.add(incoming)

    assert len(idx.all_instances()) == 1  # fused, not appended
    assert survivor.instance_id == 1
    assert survivor.n_obs == 5  # 2 + 3
    assert survivor.score == 0.9  # max kept
    # fused AABB spans both boxes (trimmed), centroid recomputed inside the union
    assert survivor.aabb_min[0] >= 0.0
    assert survivor.aabb_max[0] <= 1.2 + 1e-6
    assert survivor.aabb_max[0] > 1.0  # grew toward b


def test_no_merge_when_disjoint():
    idx = BasicSceneIndex([_rec(1, "chair", [0, 0, 0], [1, 1, 1])])
    idx.add(_rec(2, "chair", [10, 10, 0], [11, 11, 1]))
    assert len(idx.all_instances()) == 2


def test_no_merge_across_labels_even_if_overlapping():
    idx = BasicSceneIndex([_rec(1, "chair", [0, 0, 0], [1, 1, 1])])
    idx.add(_rec(2, "table", [0, 0, 0], [1, 1, 1]))  # same box, different label
    assert len(idx.all_instances()) == 2


def test_merge_recomputes_trimmed_aabb_from_percentiles():
    base_min, base_max = [0, 0, 0], [2, 2, 2]
    pts = _box_points(base_min, base_max, n=10)
    idx = BasicSceneIndex([_rec(1, "vase", base_min, base_max, points=pts)])
    # incoming identical footprint plus one far outlier point that trimming rejects
    incoming_pts = np.vstack([pts, np.array([[100.0, 100.0, 100.0]])])
    idx.add(_rec(2, "vase", base_min, base_max, points=incoming_pts))
    survivor = idx.all_instances()[0]
    # 98th percentile must reject the lone outlier -> stays near 2, not 100
    assert survivor.aabb_max[0] < 10.0


def test_fuse_time_extent_cap_bounds_accumulated_walk_issue_104():
    """Issue #104: live mechanism -- hotel_room_1's bedside-table instance fused to
    ~3x its class's typical extent (2.47x1.89x1.53 m vs a ~0.63-0.81 m typical
    nightstand) even though every contributing match individually cleared
    tracker._match_plausible (#94/#153: co-located re-observations, dist under
    extent_veto_min_sep, never veto regardless of how large the RUNNING union has
    already grown). The bug is therefore in the RESULT of many individually-safe
    fuses, not in any one match decision -- reproduced here directly against
    BasicSceneIndex.merge_into (the same unconditional-fuse entrypoint the tracker
    calls, see its own docstring), independent of the tracker's match logic.

    A same-class instance is walked in small (0.15 m) steps -- comfortably under
    tracker.TrackerConfig.extent_veto_min_sep (0.2 m), so #153 says a real tracker
    would never veto any single one of these -- for enough steps that the raw
    accumulated span (2.4 m) is ~3x the class's typical long axis (0.811 m), #104's
    exact failure mode. cap_fused_extent (issue #104) must keep the final fused
    box's sorted extent within FUSE_EXTENT_CAP_FACTOR x the class's typical extent,
    on every axis.
    """
    from core.perception.dimension_priors import FUSE_EXTENT_CAP_FACTOR, prior_for

    label = "nightstand"
    half = 0.075  # 0.15 m cube cluster per observation
    seed_pts = _box_points([-half, -half, -half], [half, half, half], n=3)
    idx = BasicSceneIndex(
        [_rec(1, label, [-half, -half, -half], [half, half, half], points=seed_pts)]
    )
    n_steps = 16
    step = 0.15  # < extent_veto_min_sep -> every step is a "same object" re-observation
    for i in range(1, n_steps + 1):
        cx = i * step
        cmin = [cx - half, -half, -half]
        cmax = [cx + half, half, half]
        rec = _rec(100 + i, label, cmin, cmax, points=_box_points(cmin, cmax, n=3))
        idx.merge_into(1, rec)

    survivor = idx.all_instances()[0]
    assert survivor.n_obs == n_steps + 1  # every step really did fuse, none split off
    ext = np.sort(survivor.aabb_max - survivor.aabb_min)
    typ = np.sort(prior_for(label).typ_ext)

    raw_span = n_steps * step  # 2.4 m -- what the uncapped walk would have spanned
    assert raw_span > FUSE_EXTENT_CAP_FACTOR * typ[-1]  # the walk really would have blown up
    assert np.all(ext <= FUSE_EXTENT_CAP_FACTOR * typ + 1e-6)  # the fix bounds it


def test_fuse_time_extent_cap_fails_open_with_no_class_prior():
    """Issue #104, fail-open half (same convention as tracker._match_plausible/#94):
    a genuinely large real object of a class with NO dimension prior must not be
    second-guessed by cap_fused_extent -- with nothing to judge plausibility
    against, the accumulated box is left exactly as measured, however large."""
    from core.perception.dimension_priors import prior_for

    label = "not-a-real-vocab-class"
    assert prior_for(label) is None  # precondition: genuinely unpriored

    a_min, a_max = [0, 0, 0], [4, 4, 4]  # deliberately huge -- no prior to cap it against
    b_min, b_max = [0.2, 0.2, 0.2], [5, 5, 5]  # heavy overlap -> IoU > 0.3, merge fires
    idx = BasicSceneIndex(
        [_rec(1, label, a_min, a_max, points=_box_points(a_min, a_max))]
    )
    incoming = _rec(2, label, b_min, b_max, points=_box_points(b_min, b_max))
    survivor = idx.add(incoming)

    assert len(idx.all_instances()) == 1  # fused
    # untouched by any class-based cap -- the fused box legitimately spans a->b
    assert survivor.aabb_max[0] > 4.5
    assert survivor.aabb_min[0] < 0.5


def test_merge_into_fuses_by_id_even_when_disjoint():
    """Issue #89: merge_into trusts the caller's association decision unconditionally
    -- unlike add()'s IoU re-derivation, disjoint boxes still fuse when the caller
    (the tracker) already matched them by id."""
    idx = BasicSceneIndex([_rec(1, "chair", [0, 0, 0], [1, 1, 1], n_obs=1)])
    incoming = _rec(1, "chair", [10, 10, 0], [11, 11, 1], n_obs=1, score=0.7)
    survivor = idx.merge_into(1, incoming)
    assert len(idx.all_instances()) == 1  # fused, no duplicate spawned
    assert survivor.instance_id == 1
    assert survivor.n_obs == 2
    assert survivor.score == 0.7


def test_merge_into_fuses_by_id_even_across_label_variants():
    """Issue #84 item 3: merge_into ignores label text entirely -- a detector label
    variant the tracker's association already alias-bridged (e.g. 'refridgerator' vs
    the tracked 'refrigerator') still fuses into the SAME instance, since merge_into
    never re-derives label compatibility the way add()/_find_merge_target does."""
    idx = BasicSceneIndex([_rec(1, "refrigerator", [0, 0, 0], [1, 1, 1], n_obs=1)])
    incoming = _rec(1, "refridgerator", [0.1, 0.1, 0.1], [1.1, 1.1, 1.1], n_obs=1)
    survivor = idx.merge_into(1, incoming)
    assert len(idx.all_instances()) == 1
    assert survivor.instance_id == 1
    assert survivor.n_obs == 2


def test_merge_into_falls_back_to_add_when_id_absent():
    idx = BasicSceneIndex()
    incoming = _rec(1, "chair", [0, 0, 0], [1, 1, 1])
    survivor = idx.merge_into(1, incoming)
    assert len(idx.all_instances()) == 1
    assert survivor.instance_id == 1


def test_add_merges_across_subphrase_fragment_label():
    """Issue #89: a bare 'potted' fragment overlapping a 'potted plant' instance's box
    must fuse via add()'s own IoU-gated merge path, not just via the tracker's
    merge_into (add()/_find_merge_target is also reached directly, e.g. scripted
    replay/tests bypassing the tracker)."""
    a_min, a_max = [0, 0, 0], [1, 1, 1]
    b_min, b_max = [0.2, 0.2, 0.2], [1.2, 1.2, 1.2]  # heavy overlap -> IoU > 0.3
    idx = BasicSceneIndex(
        [_rec(1, "potted plant", a_min, a_max, points=_box_points(a_min, a_max), n_obs=1)]
    )
    incoming = _rec(2, "potted", b_min, b_max, points=_box_points(b_min, b_max), n_obs=1)
    survivor = idx.add(incoming)
    assert len(idx.all_instances()) == 1
    assert survivor.instance_id == 1
    assert survivor.label == "potted plant"  # established label kept
    assert survivor.n_obs == 2


def test_add_merges_across_duplicated_token_label():
    a_min, a_max = [0, 0, 0], [1, 1, 1]
    b_min, b_max = [0.2, 0.2, 0.2], [1.2, 1.2, 1.2]
    idx = BasicSceneIndex(
        [_rec(1, "door", a_min, a_max, points=_box_points(a_min, a_max), n_obs=1)]
    )
    incoming = _rec(2, "door door", b_min, b_max, points=_box_points(b_min, b_max), n_obs=1)
    survivor = idx.add(incoming)
    assert len(idx.all_instances()) == 1
    assert survivor.n_obs == 2


def test_add_does_not_merge_door_and_floor_even_when_overlapping():
    idx = BasicSceneIndex([_rec(1, "door", [0, 0, 0], [1, 1, 1])])
    idx.add(_rec(2, "floor", [0, 0, 0], [1, 1, 1]))  # same box, unrelated label
    assert len(idx.all_instances()) == 2


# --------------------------------------------------------------------- labels_foldable


def test_labels_foldable_subphrase_fragments():
    assert labels_foldable("potted", "potted plant")
    assert labels_foldable("plant", "potted plant")
    assert labels_foldable("potted plant", "potted")  # symmetric


def test_labels_foldable_duplicated_token_forms():
    assert labels_foldable("door door", "door")
    assert labels_foldable("door door frame", "door")
    assert labels_foldable("screen projector screen", "projector screen")


def test_labels_foldable_guards_against_over_merge():
    assert not labels_foldable("door", "floor")
    assert not labels_foldable("chair", "table")
    assert not labels_foldable("", "door")
    assert not labels_foldable("", "")


def test_labels_foldable_unrelated_fragments_of_same_object_not_transitive():
    """Documented limitation: 'potted' and 'plant' alone (neither the compound) do not
    fold against each other directly."""
    assert not labels_foldable("potted", "plant")


# --------------------------------------------------------------- issue #154: merged labels


def test_resolve_merged_label_door_door_folds_to_door():
    from core.perception.scene_index import _resolve_merged_label

    assert _resolve_merged_label("door door") == ("door", ())
    assert _resolve_merged_label("door door frame") == ("door frame", ())
    assert _resolve_merged_label("counter counter") == ("counter", ())


def test_resolve_merged_label_two_class_merge_keeps_both_as_alias():
    from core.perception.scene_index import _resolve_merged_label

    assert _resolve_merged_label("cabinet shelf") == ("cabinet", ("shelf",))
    assert _resolve_merged_label("elephant figurine horse figurine") == (
        "elephant figurine",
        ("horse figurine",),
    )
    assert _resolve_merged_label("computer monitor projector screen") == (
        "computer monitor",
        ("projector screen",),
    )


def test_resolve_merged_label_dropped_middle_word_folds_to_the_bigger_class():
    """'map wall' (missing 'decal') and 'pyramid holder' (missing 'candle') are
    fragments of one bigger real class with an internal word dropped, not a
    merge of two smaller real classes ('map' + 'wall', 'pyramid' + 'holder') --
    real live data (cluster_verify job 702059) contains exactly this case."""
    from core.perception.scene_index import _resolve_merged_label

    assert _resolve_merged_label("map wall") == ("map wall decal", ())
    assert _resolve_merged_label("pyramid holder") == ("pyramid candle holder", ())


def test_resolve_merged_label_unresolvable_merge_is_left_untouched():
    """'cabinet bedside file cabinet' has no full decomposition into real classes
    (a bare 'bedside' names nothing) -- guessing which part to keep would risk
    discarding a legitimate detection outright, so this deliberately no-ops."""
    from core.perception.scene_index import _resolve_merged_label

    assert _resolve_merged_label("cabinet bedside file cabinet") is None


def test_resolve_merged_label_single_word_never_touched():
    """A single-word label can never BE the merged-label bug -- nothing merged
    into it -- so the couch<->sofa synonym collapse (out of scope here) is left
    entirely to normalize_label, unaffected by this fix."""
    from core.perception.scene_index import _resolve_merged_label

    assert _resolve_merged_label("couch") is None
    assert _resolve_merged_label("door") is None


# Every legitimate multi-word class named in issue #154 as a getting-it-wrong
# trap -- each MUST resolve to exactly itself, no split, no alias.
_LEGITIMATE_COMPOUNDS = (
    "coffee table", "dining table", "bedside table", "potted plant",
    "pyramid candle holder", "map wall decal", "file cabinet",
    "projector screen", "computer monitor", "door frame", "kitchen counter",
    "calligraphy painting",
)


@pytest.mark.parametrize("label", _LEGITIMATE_COMPOUNDS)
def test_resolve_merged_label_leaves_legitimate_compounds_untouched(label):
    from core.perception.scene_index import _resolve_merged_label

    assert _resolve_merged_label(label) == (label, ())


@pytest.mark.parametrize("label", _LEGITIMATE_COMPOUNDS)
def test_add_leaves_legitimate_compounds_as_their_own_single_instance(label):
    """End-to-end through the real entry point (add()): a legitimate compound
    detection must still found exactly one instance under its own name, with no
    alias invented -- the getting-it-wrong-is-worse-than-the-bug guard from
    issue #154's design constraints."""
    idx = BasicSceneIndex()
    survivor = idx.add(_rec(1, label, [0, 0, 0], [1, 1, 1]))
    assert len(idx.all_instances()) == 1
    assert survivor.label == label
    assert survivor.aliases == ()
    assert idx.by_label(label) == [survivor]


def test_add_founds_door_door_as_a_plain_door_instance():
    """Issue #154: unlike the pre-fix behaviour (a 'door door' detection with no
    prior 'door' instance to fold against permanently founds its own 'door door'
    class), a brand-new instance is canonicalised at creation time."""
    idx = BasicSceneIndex()
    survivor = idx.add(_rec(1, "door door", [0, 0, 0], [1, 1, 1]))
    assert survivor.label == "door"
    assert [r.label for r in idx.all_instances()] == ["door"]


def test_add_elephant_figurine_horse_figurine_both_findable_by_anchor():
    """Issue #154 worst case: a single merged detection must not become a
    phantom 'elephant figurine horse figurine' class that no anchor can ever
    reach. Splitting into two synthetic instances would invent geometry no
    detection ever supported (one box, one detected region) -- so this resolves
    to ONE instance findable by EITHER of the real anchors the question turns
    on, via the alias/synonym match tier."""
    idx = BasicSceneIndex()
    survivor = idx.add(_rec(1, "elephant figurine horse figurine", [0, 0, 0], [1, 1, 1]))
    assert len(idx.all_instances()) == 1
    assert survivor.label == "elephant figurine"
    assert idx.by_label("elephant figurine") == [survivor]
    assert idx.by_label("horse figurine") == [survivor]


def test_add_two_real_single_word_classes_merge_both_findable():
    """'cabinet shelf' merges two genuinely distinct real classes (unlike 'door
    door', neither is a fragment of the other) -- must not become its own
    phantom 'cabinet shelf' class, and must not silently vanish either."""
    idx = BasicSceneIndex()
    survivor = idx.add(_rec(1, "cabinet shelf", [0, 0, 0], [1, 1, 1]))
    assert survivor.label == "cabinet"
    assert idx.by_label("cabinet") == [survivor]
    assert idx.by_label("shelf") == [survivor]


def test_add_merged_label_still_merges_by_iou_into_existing_canonical_instance():
    """A merged-label detection that overlaps an already-established canonical
    instance must fuse into it, not mint a second phantom instance. "door door
    frame" canonicalises whole to "door frame" (issue #154), which is still a
    #89-style fold of the established "door" instance's own label (a "door
    frame" is a superset-fold of bare "door"), so the two observations of what
    is geometrically the same object correctly fuse into one -- established
    label kept, n_obs incremented."""
    a_min, a_max = [0, 0, 0], [1, 1, 1]
    b_min, b_max = [0.2, 0.2, 0.2], [1.2, 1.2, 1.2]
    idx = BasicSceneIndex(
        [_rec(1, "door", a_min, a_max, points=_box_points(a_min, a_max), n_obs=1)]
    )
    incoming = _rec(
        2, "door door frame", b_min, b_max, points=_box_points(b_min, b_max), n_obs=1
    )
    survivor = idx.add(incoming)
    assert len(idx.all_instances()) == 1
    assert survivor.label == "door"
    assert survivor.n_obs == 2


def test_add_merged_label_alias_survives_fuse_into_established_plain_target():
    """Issue #154: if a plain "elephant figurine" instance was already
    established (no alias) BEFORE a merged "elephant figurine horse figurine"
    detection of the same object arrives, the merged detection's extra alias
    ("horse figurine") must still end up reachable on the fused survivor --
    not silently dropped just because the target predates it."""
    a_min, a_max = [0, 0, 0], [1, 1, 1]
    b_min, b_max = [0.2, 0.2, 0.2], [1.2, 1.2, 1.2]
    idx = BasicSceneIndex(
        [_rec(1, "elephant figurine", a_min, a_max, points=_box_points(a_min, a_max), n_obs=1)]
    )
    incoming = _rec(
        2,
        "elephant figurine horse figurine",
        b_min,
        b_max,
        points=_box_points(b_min, b_max),
        n_obs=1,
    )
    survivor = idx.add(incoming)
    assert len(idx.all_instances()) == 1
    assert survivor.label == "elephant figurine"
    assert idx.by_label("horse figurine") == [survivor]


def test_add_disjoint_reassigns_colliding_id():
    idx = BasicSceneIndex([_rec(1, "chair", [0, 0, 0], [1, 1, 1])])
    # incoming reuses id 1 but is disjoint -> must be appended with a fresh id
    added = idx.add(_rec(1, "chair", [10, 10, 0], [11, 11, 1]))
    ids = sorted(r.instance_id for r in idx.all_instances())
    assert len(ids) == 2
    assert len(set(ids)) == 2  # no duplicate ids
    assert added.instance_id != 1


# --------------------------------------------------------------------------- #24:
# SceneIndex Protocol structurally requires by_label_tiered


class _ConformingFakeIndex:
    """Minimal SceneIndex: implements all_instances, by_label AND by_label_tiered."""

    def all_instances(self):
        return []

    def by_label(self, noun):
        return []

    def by_label_tiered(self, noun):
        return []


class _NonConformingFakeIndex:
    """Missing by_label_tiered — pre-#24 this satisfied SceneIndex; it must not now."""

    def all_instances(self):
        return []

    def by_label(self, noun):
        return []


def test_basic_scene_index_conforms_to_scene_index_protocol():
    idx = BasicSceneIndex([_rec(1, "chair", [0, 0, 0], [1, 1, 1])])
    assert isinstance(idx, SceneIndex)
    assert hasattr(idx, "by_label_tiered")


def test_scene_index_protocol_structurally_requires_by_label_tiered():
    # A minimal fake that implements by_label_tiered duck-type-conforms...
    assert isinstance(_ConformingFakeIndex(), SceneIndex)
    # ...but one that only implements the pre-#24 by_label surface does not, because
    # by_label_tiered is now a required Protocol member (not an optional extra that
    # geometry.toolbox._match_anchor_noun quietly shrugs off via getattr).
    assert not isinstance(_NonConformingFakeIndex(), SceneIndex)


# --------------------------------------------------------------------------- #84/#89:
# dump_instance_index instrumentation


def test_dump_instance_index_is_noop_without_env_var(monkeypatch, tmp_path):
    monkeypatch.delenv(ENV_INSTANCE_DUMP_PATH, raising=False)
    idx = BasicSceneIndex([_rec(1, "chair", [0, 0, 0], [1, 1, 1])])
    dump_instance_index(idx, tag="periodic")
    assert list(tmp_path.iterdir()) == []  # nothing written anywhere


def test_dump_instance_index_writes_jsonl_record(monkeypatch, tmp_path):
    out = tmp_path / "instances.jsonl"
    monkeypatch.setenv(ENV_INSTANCE_DUMP_PATH, str(out))
    idx = BasicSceneIndex(
        [_rec(1, "chair", [0, 0, 0], [1, 1, 1], n_obs=3, score=0.42)]
    )
    dump_instance_index(idx, tag="answer_time", keyframes_processed=7)
    lines = out.read_text().strip().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec["tag"] == "answer_time"
    assert rec["keyframes_processed"] == 7
    assert rec["total_instances"] == 1
    assert rec["by_class"] == {"chair": 1}
    assert rec["instances"][0]["id"] == 1
    assert rec["instances"][0]["label"] == "chair"
    assert rec["instances"][0]["n_obs"] == 3
    assert rec["instances"][0]["score"] == 0.42
    assert len(rec["instances"][0]["position"]) == 3
    # n_obs=3 & score=0.42 clear both default answer-eligibility floors (issue #84)
    assert rec["instances"][0]["answer_eligible"] is True
    assert rec["instances"][0]["eligibility_reason"] == "eligible"


def test_dump_instance_index_includes_aabb(monkeypatch, tmp_path):
    # Issue #101: per-instance AABB corners so an on(table)-style geometry predicate
    # can be replayed offline from the dump alone (centroid alone can't tell you).
    out = tmp_path / "instances.jsonl"
    monkeypatch.setenv(ENV_INSTANCE_DUMP_PATH, str(out))
    idx = BasicSceneIndex([_rec(1, "table", [0.0, 0.5, 1.0], [2.0, 1.5, 1.25])])
    dump_instance_index(idx, tag="periodic")
    rec = json.loads(out.read_text().strip())
    inst = rec["instances"][0]
    assert inst["aabb_min"] == [0.0, 0.5, 1.0]
    assert inst["aabb_max"] == [2.0, 1.5, 1.25]
    # plain JSON lists, not numpy arrays/scalars leaking through
    assert isinstance(inst["aabb_min"], list)
    assert all(isinstance(c, float) for c in inst["aabb_min"])


def test_dump_instance_index_surfaces_gate_rejection_reason(monkeypatch, tmp_path):
    """Issue #84 gate observability: an instance that fails the answer-eligibility gate
    (n_obs=1, below the default min-obs floor of 2) shows up in the dump as ineligible
    with the reason -- this is the visibility the offline battery's n_obs=3 GT mocks
    never exercise."""
    out = tmp_path / "instances.jsonl"
    monkeypatch.setenv(ENV_INSTANCE_DUMP_PATH, str(out))
    idx = BasicSceneIndex([_rec(1, "teapot", [0, 0, 0], [1, 1, 1], n_obs=1, score=0.9)])
    dump_instance_index(idx, tag="periodic")
    rec = json.loads(out.read_text().strip())
    assert rec["instances"][0]["answer_eligible"] is False
    assert rec["instances"][0]["eligibility_reason"] == "n_obs_below_floor"


def test_dump_instance_index_appends_across_calls(monkeypatch, tmp_path):
    out = tmp_path / "instances.jsonl"
    monkeypatch.setenv(ENV_INSTANCE_DUMP_PATH, str(out))
    idx = BasicSceneIndex([_rec(1, "chair", [0, 0, 0], [1, 1, 1])])
    dump_instance_index(idx, tag="periodic")
    dump_instance_index(idx, tag="periodic")
    assert len(out.read_text().strip().splitlines()) == 2


def test_dump_instance_index_never_raises_on_bad_path(monkeypatch):
    # A path under a file (not a directory) cannot be created -> swallowed, not raised.
    monkeypatch.setenv(ENV_INSTANCE_DUMP_PATH, "/dev/null/nonexistent/instances.jsonl")
    idx = BasicSceneIndex([_rec(1, "chair", [0, 0, 0], [1, 1, 1])])
    dump_instance_index(idx, tag="periodic")  # must not raise


def test_non_conforming_index_lacks_the_attribute_toolbox_depends_on():
    # This is exactly the failure mode #24 closes off: code consuming a SceneIndex
    # (geometry.toolbox._match_anchor_noun) can no longer rely on by_label_tiered
    # being there just because a value type-checks as SceneIndex -- a value that is
    # actually missing it now also fails the structural isinstance check above, so
    # the gap is caught at the type boundary instead of surfacing only as a silent
    # anchor-resolution regression (#13) deep in toolbox's getattr fallback.
    idx = _NonConformingFakeIndex()
    assert not isinstance(idx, SceneIndex)
    assert not hasattr(idx, "by_label_tiered")


# --------------------------------------------------------------------------- #129:
# structural classes excluded from the census, kept for anchor/target resolution


def test_countable_instances_excludes_structural_classes():
    idx = BasicSceneIndex(
        [
            _rec(1, "floor", [0, 0, 0], [1, 1, 0.05]),
            _rec(2, "floor", [3, 3, 0], [4, 4, 0.05]),
            _rec(3, "window", [0, 0, 1], [1, 0.1, 2]),
            _rec(4, "sofa", [2, 2, 0], [3, 3, 1]),
        ]
    )
    countable = idx.countable_instances()
    assert {r.label for r in countable} == {"sofa"}
    assert len(countable) == 1


def test_countable_instances_leaves_non_structural_counts_unchanged():
    idx = BasicSceneIndex(
        [
            _rec(1, "floor", [0, 0, 0], [1, 1, 0.05]),
            _rec(2, "chair", [0, 0, 0], [1, 1, 1]),
            _rec(3, "chair", [2, 2, 0], [3, 3, 1]),
            _rec(4, "table", [4, 4, 0], [5, 5, 1]),
        ]
    )
    countable = idx.countable_instances()
    by_label: dict[str, int] = {}
    for r in countable:
        by_label[r.label] = by_label.get(r.label, 0) + 1
    assert by_label == {"chair": 2, "table": 1}


def test_all_instances_is_unaffected_by_structural_classification():
    # all_instances() -- the surface every existing anchor/target/census caller
    # already uses -- must be byte-for-byte unchanged: still returns EVERY instance,
    # structural or not.
    idx = BasicSceneIndex(
        [
            _rec(1, "floor", [0, 0, 0], [1, 1, 0.05]),
            _rec(2, "window", [0, 0, 1], [1, 0.1, 2]),
            _rec(3, "sofa", [2, 2, 0], [3, 3, 1]),
        ]
    )
    assert {r.label for r in idx.all_instances()} == {"floor", "window", "sofa"}
    assert len(idx.all_instances()) == 3


def test_by_label_still_resolves_structural_anchors_unchanged():
    # The gate that matters most (#129 acceptance): a `window` anchor must keep
    # resolving through the exact same by_label/by_label_tiered surface as before --
    # this method is deliberately NOT touched by the structural classifier.
    idx = BasicSceneIndex(
        [
            _rec(1, "window", [0, 0, 1], [1, 0.1, 2]),
            _rec(2, "window", [5, 0, 1], [6, 0.1, 2]),
            _rec(3, "floor", [0, 0, 0], [6, 6, 0.05]),
        ]
    )
    windows = idx.by_label("window")
    assert len(windows) == 2
    assert {r.instance_id for r in windows} == {1, 2}
    tiered = idx.by_label_tiered("window")
    assert len(tiered) == 2
    assert all(tier == MatchTier.EXACT for _, tier in tiered)
    floors = idx.by_label("floor")
    assert len(floors) == 1


def test_dump_instance_index_by_class_excludes_structural(monkeypatch, tmp_path):
    # Issue #129: the class CENSUS (`by_class`) drops structural classes; the full
    # `instances` list stays untouched (a structural instance's geometry is still
    # there for an offline consumer that wants it, e.g. as a relation anchor).
    out = tmp_path / "instances.jsonl"
    monkeypatch.setenv(ENV_INSTANCE_DUMP_PATH, str(out))
    idx = BasicSceneIndex(
        [
            _rec(1, "floor", [0, 0, 0], [1, 1, 0.05], n_obs=3, score=0.9),
            _rec(2, "floor", [3, 3, 0], [4, 4, 0.05], n_obs=3, score=0.9),
            _rec(3, "window", [0, 0, 1], [1, 0.1, 2], n_obs=3, score=0.9),
            _rec(4, "sofa", [2, 2, 0], [3, 3, 1], n_obs=3, score=0.9),
        ]
    )
    dump_instance_index(idx, tag="answer_time")
    rec = json.loads(out.read_text().strip())
    assert rec["by_class"] == {"sofa": 1}
    assert rec["total_instances"] == 4  # total_instances is unaffected
    assert {i["label"] for i in rec["instances"]} == {"floor", "window", "sofa"}
    assert len(rec["instances"]) == 4  # the full instances list is unaffected
