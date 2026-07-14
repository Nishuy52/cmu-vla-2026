"""Track decay (H15a / NUM-F8): prune one-frame ghosts, never confirmed tracks."""
from __future__ import annotations

import numpy as np

from core.interfaces import InstanceRecord
from core.perception.scene_index import BasicSceneIndex
from core.perception.tracker import decay_singletons


def _rec(iid, label="vase", *, n_obs=1):
    c = np.zeros(3)
    return InstanceRecord(
        instance_id=iid,
        label=label,
        score=0.9,
        n_obs=n_obs,
        centroid=c,
        aabb_min=c - 0.1,
        aabb_max=c + 0.1,
    )


def test_remove_returns_true_only_when_present():
    idx = BasicSceneIndex([_rec(1)])
    assert idx.remove(1) is True
    assert idx.remove(1) is False
    assert idx.all_instances() == []


def test_ghost_pruned_after_k_keyframes():
    idx = BasicSceneIndex([_rec(1, n_obs=1)])
    first_seen = {1: 0}
    # not yet due at keyframe 4 (k=5): 4 - 0 = 4 < 5
    assert decay_singletons(idx, first_seen, keyframe_idx=4, decay_k=5) == []
    assert len(idx.all_instances()) == 1
    # due at keyframe 5: 5 - 0 = 5 >= 5
    assert decay_singletons(idx, first_seen, keyframe_idx=5, decay_k=5) == [1]
    assert idx.all_instances() == []
    assert 1 not in first_seen  # bookkeeping cleaned up


def test_confirmed_track_never_pruned():
    # n_obs>=2 is a confirmed track and must survive arbitrarily long.
    idx = BasicSceneIndex([_rec(1, n_obs=2)])
    first_seen = {1: 0}
    assert decay_singletons(idx, first_seen, keyframe_idx=1000, decay_k=5) == []
    assert len(idx.all_instances()) == 1
    # confirmed tracks stop being aged (dropped from first_seen).
    assert 1 not in first_seen


def test_ghost_that_gets_reobserved_is_kept():
    # a ghost re-observed (n_obs bumped to 2) before the deadline is a real track now.
    idx = BasicSceneIndex([_rec(1, n_obs=1)])
    first_seen = {1: 0}
    idx.all_instances()[0].n_obs = 2  # re-observed on a later keyframe
    assert decay_singletons(idx, first_seen, keyframe_idx=10, decay_k=5) == []
    assert len(idx.all_instances()) == 1


def test_decay_disabled_when_k_zero():
    idx = BasicSceneIndex([_rec(1, n_obs=1)])
    first_seen = {1: 0}
    assert decay_singletons(idx, first_seen, keyframe_idx=100, decay_k=0) == []
    assert len(idx.all_instances()) == 1


def test_stale_firstseen_entry_cleaned_for_absent_id():
    idx = BasicSceneIndex([])  # id 1 already gone (merged away earlier)
    first_seen = {1: 0}
    assert decay_singletons(idx, first_seen, keyframe_idx=10, decay_k=5) == []
    assert first_seen == {}


def test_gt_style_index_unaffected():
    # GT battery instances are n_obs=3 by construction — decay is inert on them.
    idx = BasicSceneIndex([_rec(i, n_obs=3) for i in range(1, 6)])
    first_seen = {i: 0 for i in range(1, 6)}
    pruned = decay_singletons(idx, first_seen, keyframe_idx=999, decay_k=5)
    assert pruned == []
    assert len(idx.all_instances()) == 5
