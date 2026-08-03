"""Issue #160: cluster contiguous same-class fragments into one anchor footprint
at RESOLVE time.

Ground-truth reference (arabic_room, ``data/vla3d/Unity/arabic_room
/arabic_room_object_result.csv``): exactly TWO ``column`` instances,
    id=23  center (-0.993,  1.081, 1.392)  extents (0.503, 0.503, 2.782)
    id=30  center (-0.993, -1.133, 1.392)  extents (0.503, 0.503, 2.782)
matching #167's later evidence ("GT arabic_room has exactly TWO column
instances"). The live index (#160's own evidence, 18-23 ``column`` instances)
fragments each real column into several small same-footprint, different-height
detections -- this file's 18-fragment fixture reconstructs that shape without
the (unavailable locally) banked ``instance_index.jsonl``.
"""
from __future__ import annotations

import math

from core.geometry import toolbox as T
from core.plan_schema import Anchor
from tests.geometry._helpers import FakeIndex, rec


# --------------------------------------------------------------------------- #160


def _column_fragment(iid: int, cx: float, cy: float, z_lo: float, z_hi: float):
    """One height-sliced fragment: same small XY footprint, a Z band of the
    real column's height range -- the exact shape #160 attributes the live
    18-23x ``column`` count to ("fragments of pillars/arches at multiple
    heights/tiles")."""
    return rec(iid, "column", (cx, cy, (z_lo + z_hi) / 2.0), (0.4, 0.4, z_hi - z_lo))


def _arabic_room_column_fragments() -> list:
    """18 synthetic ``column`` fragments: 9 height-slices at each of the two GT
    column centres (-0.993, 1.081) and (-0.993, -1.133)."""
    col_centers = [(-0.993, 1.081), (-0.993, -1.133)]
    n_bands = 9
    height = 2.78
    band = height / n_bands
    frags = []
    iid = 1
    for cx, cy in col_centers:
        for k in range(n_bands):
            z_lo = k * band
            z_hi = z_lo + band
            frags.append(_column_fragment(iid, cx, cy, z_lo, z_hi))
            iid += 1
    return frags


def test_cluster_18_column_fragments_resolves_to_gt_pair():
    idx = FakeIndex(_arabic_room_column_fragments())
    resolved = T._resolve_anchor(Anchor(noun="column"), idx, T.DEFAULT_THRESHOLDS)
    assert len(resolved) == 2, f"expected 2 clustered columns, got {len(resolved)}"

    centers = sorted((float(c.centroid[0]), float(c.centroid[1])) for c in resolved)
    gt = sorted([(-0.993, -1.133), (-0.993, 1.081)])
    for (gx, gy), (rx, ry) in zip(gt, centers):
        assert math.hypot(gx - rx, gy - ry) < 0.05

    # each cluster accumulates every fragment's observations, not just one
    assert all(c.n_obs == 9 * 3 for c in resolved)  # rec()'s default n_obs=3


def test_cluster_never_mutates_the_index():
    frags = _arabic_room_column_fragments()
    idx = FakeIndex(frags)
    before = len(idx.all_instances())
    T._resolve_anchor(Anchor(noun="column"), idx, T.DEFAULT_THRESHOLDS)
    after = len(idx.all_instances())
    assert before == after == 18
    # the index's own records are untouched objects, not replaced/edited copies
    assert idx.all_instances() == frags


def test_cluster_single_fragment_column_unaffected():
    """A class with only one detected fragment (no clustering possible) resolves
    exactly as before -- clustering is a no-op when there is nothing to merge."""
    idx = FakeIndex([rec(1, "column", (0.0, 0.0, 1.4), (0.5, 0.5, 2.8))])
    resolved = T._resolve_anchor(Anchor(noun="column"), idx, T.DEFAULT_THRESHOLDS)
    assert [c.instance_id for c in resolved] == [1]


# ------------------------------------------------------- must NOT merge (negative)


def test_two_distinct_columns_at_normal_spacing_do_not_merge():
    """The GT pair itself (single-fragment, non-overlapping, ~2.2 m apart): a
    genuine pair of distinct same-class objects at ordinary spacing must survive
    resolution as two separate candidates, not collapse into one."""
    col_a = rec(23, "column", (-0.993, 1.081, 1.392), (0.503, 0.503, 2.782))
    col_b = rec(30, "column", (-0.993, -1.133, 1.392), (0.503, 0.503, 2.782))
    idx = FakeIndex([col_a, col_b])
    resolved = T._resolve_anchor(Anchor(noun="column"), idx, T.DEFAULT_THRESHOLDS)
    assert sorted(c.instance_id for c in resolved) == [23, 30]


def test_bridged_fragments_spanning_two_real_objects_do_not_merge():
    """#160's own named failure mode: adjacency chains ACROSS the gap between two
    neighbouring real objects via a bridging fragment, producing a blob far
    larger than any real instance of the class (the issue's 11.06 m2 example).
    The plausibility ceiling must keep the chain unmerged rather than hand
    resolution one implausible anchor spanning both real objects."""
    # Two column-sized objects (real single-column footprint) ~2 m apart in X,
    # bridged by a wide, thin "fragment" (e.g. a shadow/floor mis-detection)
    # whose footprint overlaps both -- the union would be a >2 m wide "column".
    obj_a = rec(1, "column", (0.0, 0.0, 1.4), (0.5, 0.5, 2.8))
    bridge = rec(2, "column", (0.9, 0.0, 1.4), (1.8, 0.2, 2.8))  # spans x:[0,1.8]
    obj_b = rec(3, "column", (1.8, 0.0, 1.4), (0.5, 0.5, 2.8))
    idx = FakeIndex([obj_a, bridge, obj_b])
    resolved = T._resolve_anchor(Anchor(noun="column"), idx, T.DEFAULT_THRESHOLDS)
    assert sorted(c.instance_id for c in resolved) == [1, 2, 3]


def test_cluster_respects_label_boundary_never_merges_across_classes():
    """A same-footprint different-label pair (e.g. mislabelled cousin sitting
    exactly where a column fragment is) must never merge -- clustering is
    strictly within one canonical label."""
    a = rec(1, "column", (0.0, 0.0, 1.4), (0.5, 0.5, 2.8))
    b = rec(2, "pillar", (0.0, 0.0, 1.4), (0.5, 0.5, 2.8))  # overlapping, diff label
    idx = FakeIndex([a, b])
    resolved = T._resolve_anchor(Anchor(noun="column"), idx, T.DEFAULT_THRESHOLDS)
    assert [c.instance_id for c in resolved] == [1]


def test_plausible_merge_of_two_touching_fragments_of_one_real_object():
    """The positive companion to the bridged-blob test: two fragments of ONE real
    object whose union stays within a plausible size for the class DO merge into
    a single wider-footprint candidate."""
    frag_a = rec(1, "column", (-0.1, 0.0, 0.5), (0.3, 0.5, 1.0))
    frag_b = rec(2, "column", (0.1, 0.0, 1.9), (0.3, 0.5, 1.8))  # overlaps frag_a in XY
    idx = FakeIndex([frag_a, frag_b])
    resolved = T._resolve_anchor(Anchor(noun="column"), idx, T.DEFAULT_THRESHOLDS)
    assert len(resolved) == 1
    assert resolved[0].n_obs == 6  # 3 + 3
