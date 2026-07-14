"""BasicSceneIndex behaviour: label normalisation, synonym + typo tolerance,
and the IoU-gated merge/fuse API."""
from __future__ import annotations

import numpy as np

from core.interfaces import InstanceRecord
from core.perception.scene_index import (
    BasicSceneIndex,
    MatchTier,
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


def test_add_disjoint_reassigns_colliding_id():
    idx = BasicSceneIndex([_rec(1, "chair", [0, 0, 0], [1, 1, 1])])
    # incoming reuses id 1 but is disjoint -> must be appended with a fresh id
    added = idx.add(_rec(1, "chair", [10, 10, 0], [11, 11, 1]))
    ids = sorted(r.instance_id for r in idx.all_instances())
    assert len(ids) == 2
    assert len(set(ids)) == 2  # no duplicate ids
    assert added.instance_id != 1
