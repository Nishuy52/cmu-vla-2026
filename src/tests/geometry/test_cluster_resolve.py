"""Issue #160: cluster contiguous same-class fragments into one anchor footprint
at RESOLVE time, and issue #167: route-context tie-break for a corridor leg's
anchor pair once clustering still leaves more than two candidates.

Ground-truth reference (arabic_room, ``data/vla3d/Unity/arabic_room
/arabic_room_object_result.csv``): exactly TWO ``column`` instances,
    id=23  center (-0.993,  1.081, 1.392)  extents (0.503, 0.503, 2.782)
    id=30  center (-0.993, -1.133, 1.392)  extents (0.503, 0.503, 2.782)
matching #167's "GT arabic_room has exactly TWO column instances" evidence. The
live index (#160's own evidence, 18-23 ``column`` instances) fragments each real
column into several small same-footprint, different-height detections -- this
file's 18-fragment fixture reconstructs that shape without the (unavailable
locally) banked ``instance_index.jsonl``.
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


# ------------------------------------------------ #151 replay finding: head-noun prior
#
# tools/replay_live_numerical.py against reports/cluster_verify/699819 caught a real
# defect (not a synthetic worry): "tv cabinet" has no OWN dimension-prior row (only
# the generic "cabinet" head noun does), so the original fail-open ("no prior -> can't
# judge -> allow") let livingroom_3's 27 scattered `tv cabinet` detections -- real,
# spatially UNRELATED objects that merely happen to chain via footprint adjacency --
# cluster into one 74 sq m "anchor" with no size check at all, changing "photos on the
# TV cabinet" from count=6 to count=13 against a truth of 2 (a real regression this
# file's own replay test below pins). Falling back to the HEAD NOUN's prior
# ("cabinet") before failing open fixes it.


def test_cluster_falls_back_to_head_noun_prior_when_no_exact_prior():
    """A compound label with no OWN prior row ("tv cabinet") must still be held to
    its head noun's plausible size ("cabinet"), not fail open. Mirrors the
    bridged-blob shape: three real, distinct "tv cabinet"-labelled objects
    chained by a bridging fragment must NOT merge into one anchor."""
    obj_a = rec(1, "tv cabinet", (0.0, 0.0, 1.0), (0.5, 0.5, 1.0))
    bridge = rec(2, "tv cabinet", (1.5, 0.0, 1.0), (3.5, 0.2, 1.0))  # spans x:[-0.25,3.25]
    obj_b = rec(3, "tv cabinet", (3.0, 0.0, 1.0), (0.5, 0.5, 1.0))
    idx = FakeIndex([obj_a, bridge, obj_b])
    resolved = T._resolve_anchor(Anchor(noun="tv cabinet"), idx, T.DEFAULT_THRESHOLDS)
    assert sorted(c.instance_id for c in resolved) == [1, 2, 3]


def test_cluster_head_noun_prior_still_admits_a_plausible_merge():
    """The positive companion: two small "tv cabinet" fragments of ONE real
    object, well within the head noun's ("cabinet") plausible size, still
    merge -- the head-noun fallback is a real size check, not a blanket veto."""
    frag_a = rec(1, "tv cabinet", (-0.1, 0.0, 0.4), (0.3, 0.5, 0.8))
    frag_b = rec(2, "tv cabinet", (0.1, 0.0, 1.2), (0.3, 0.5, 0.8))
    idx = FakeIndex([frag_a, frag_b])
    resolved = T._resolve_anchor(Anchor(noun="tv cabinet"), idx, T.DEFAULT_THRESHOLDS)
    assert len(resolved) == 1


# --------------------------------------------------------------------------- #167


def _three_column_clusters_fixture() -> list:
    """Three well-separated single-instance ``column`` candidates -- clustering
    (#160) has nothing to merge here (no footprint overlap), so this exercises
    ``resolve_corridor_pair``'s >2-candidate route-context tie-break directly.

    col_a=(0,0), col_b=(4,0), col_c=(0,4): a route travelling along the X axis
    from (-2,0) to (6,0) passes almost exactly between col_a and col_b (near-zero
    detour); threading it between col_a and col_c (perpendicular) or col_b and
    col_c requires a large detour off that line.
    """
    return [
        rec(1, "column", (0.0, 0.0, 1.4), (0.4, 0.4, 2.8)),
        rec(2, "column", (4.0, 0.0, 1.4), (0.4, 0.4, 2.8)),
        rec(3, "column", (0.0, 4.0, 1.4), (0.4, 0.4, 2.8)),
    ]


def test_corridor_pair_picks_minimum_detour_of_three_clusters():
    idx = FakeIndex(_three_column_clusters_fixture())
    anchor = Anchor(noun="column")
    pair = T.resolve_corridor_pair(
        anchor, idx, T.DEFAULT_THRESHOLDS, from_xy=(-2.0, 0.0), to_xy=(6.0, 0.0)
    )
    assert pair is not None
    ids = sorted(c.instance_id for c in pair)
    assert ids == [1, 2]  # col_a/col_b straddle the route; col_c is a detour


def test_corridor_pair_route_context_flips_the_choice():
    """Same three clusters, a route travelling along the Y axis instead: the
    minimum-detour pair must flip to the pair straddling THAT line, proving the
    tie-break reads route context rather than a fixed geometric bias."""
    idx = FakeIndex(_three_column_clusters_fixture())
    anchor = Anchor(noun="column")
    pair = T.resolve_corridor_pair(
        anchor, idx, T.DEFAULT_THRESHOLDS, from_xy=(0.0, -2.0), to_xy=(0.0, 6.0)
    )
    assert pair is not None
    ids = sorted(c.instance_id for c in pair)
    assert ids == [1, 3]  # col_a/col_c now straddle the (vertical) route


def test_corridor_pair_terminal_leg_only_from_xy_known():
    """A terminal corridor leg has no "next leg" goal; the #167 commit body's
    contract is that the caller passes the robot's current pose as ``to_xy``
    instead, so this only exercises the degenerate ``to_xy=None`` path directly:
    with no destination term, the detour collapses to "closest gate to
    ``from_xy``" -- col_a/col_c's gate (near (0, 2)) is closest to (-2, 0), not
    col_a/col_b's (near (2, 0))."""
    idx = FakeIndex(_three_column_clusters_fixture())
    anchor = Anchor(noun="column")
    pair = T.resolve_corridor_pair(
        anchor, idx, T.DEFAULT_THRESHOLDS, from_xy=(-2.0, 0.0), to_xy=None
    )
    assert pair is not None
    assert sorted(c.instance_id for c in pair) == [1, 3]


def test_corridor_pair_deterministic_with_no_route_context():
    """Both endpoints unknown: falls back to the lowest two instance ids,
    deterministic and reproducible (never an accident of list order)."""
    idx = FakeIndex(_three_column_clusters_fixture())
    anchor = Anchor(noun="column")
    pair = T.resolve_corridor_pair(
        anchor, idx, T.DEFAULT_THRESHOLDS, from_xy=None, to_xy=None
    )
    assert pair is not None
    assert sorted(c.instance_id for c in pair) == [1, 2]


def test_corridor_pair_exactly_two_candidates_returns_them_unconditionally():
    col_a = rec(23, "column", (-0.993, 1.081, 1.392), (0.503, 0.503, 2.782))
    col_b = rec(30, "column", (-0.993, -1.133, 1.392), (0.503, 0.503, 2.782))
    idx = FakeIndex([col_a, col_b])
    pair = T.resolve_corridor_pair(
        Anchor(noun="column"), idx, T.DEFAULT_THRESHOLDS,
        from_xy=(100.0, 100.0), to_xy=(-100.0, -100.0),  # irrelevant, only 2 exist
    )
    assert pair is not None
    assert sorted(c.instance_id for c in pair) == [23, 30]


def test_corridor_pair_none_with_fewer_than_two_candidates():
    idx = FakeIndex([rec(1, "column", (0.0, 0.0, 1.4), (0.4, 0.4, 2.8))])
    pair = T.resolve_corridor_pair(
        Anchor(noun="column"), idx, T.DEFAULT_THRESHOLDS, from_xy=None, to_xy=None
    )
    assert pair is None
