"""BasicSceneIndex behaviour: label normalisation, synonym + typo tolerance,
and the IoU-gated merge/fuse API."""
from __future__ import annotations

import json
import os

import numpy as np

from core.interfaces import InstanceRecord, SceneIndex
from core.perception.scene_index import (
    ENV_INSTANCE_DUMP_PATH,
    BasicSceneIndex,
    MatchTier,
    dump_instance_index,
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
