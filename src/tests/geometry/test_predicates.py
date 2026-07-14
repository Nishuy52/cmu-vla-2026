"""Per-predicate positive / negative / boundary tests over synthetic AABBs."""
from __future__ import annotations

import numpy as np

from core.geometry import toolbox as T
from core.geometry.toolbox import DEFAULT_THRESHOLDS as TH
from tests.geometry._helpers import rec


# --------------------------------------------------------------------------- on


def test_on_positive():
    table = rec(1, "table", (0, 0, 0.5), (1.0, 1.0, 1.0))  # top at z=1.0
    book = rec(2, "book", (0, 0, 1.05), (0.3, 0.3, 0.1))  # bottom at z=1.0
    r = T.on(book, table)
    assert r.passed and r.score > 0
    assert "on:" in r.explanation


def test_on_negative_floating():
    table = rec(1, "table", (0, 0, 0.5), (1.0, 1.0, 1.0))  # top z=1.0
    lamp = rec(2, "lamp", (0, 0, 2.0), (0.2, 0.2, 0.4))  # bottom z=1.8, far above
    assert not T.on(lamp, table).passed


def test_on_negative_no_overlap():
    table = rec(1, "table", (0, 0, 0.5), (1.0, 1.0, 1.0))
    book = rec(2, "book", (5, 5, 1.05), (0.3, 0.3, 0.1))  # correct height, no footprint overlap
    assert not T.on(book, table).passed


def test_on_boundary_top_tol():
    # H5 support-band form: a.bottom may sit up to on_top_tol above b's AABB top.
    table = rec(1, "table", (0, 0, 0.5), (1.0, 1.0, 1.0))  # top z=1.0
    # bottom exactly on_top_tol above the top -> still on (upper band's top edge)
    book = rec(2, "book", (0, 0, 1.0 + TH.on_top_tol + 0.05), (0.3, 0.3, 0.1))
    assert book.aabb_min[2] == 1.0 + TH.on_top_tol
    assert T.on(book, table).passed
    # a hair further above the band -> off
    book2 = rec(3, "book", (0, 0, 1.0 + TH.on_top_tol + 0.05 + 0.02), (0.3, 0.3, 0.1))
    assert not T.on(book2, table).passed


def test_on_lower_band_edge_rejects_floor_object():
    # Support semantics: an object resting near b's FLOOR (below the upper z-band)
    # is not "on" b — only the upper span counts (H5/T8-C3/D3).
    table = rec(1, "table", (0, 0, 0.5), (1.0, 1.0, 1.0))  # zmin 0, top 1.0, height 1.0
    # object bottom just below zmin + on_upper_span_frac*height (band lower edge)
    low = rec(2, "book", (0, 0, TH.on_upper_span_frac - 0.06 + 0.05), (0.3, 0.3, 0.1))
    assert low.aabb_min[2] < TH.on_upper_span_frac * 1.0
    assert not T.on(low, table).passed


def test_on_anchor_larger_gate():
    # A big supporter cannot be "on" a small object even with full IoM + z overlap.
    cushion = rec(1, "cushion", (0, 0, 0.55), (0.4, 0.4, 0.1))  # small, top 0.6
    sofa = rec(2, "sofa", (0, 0, 0.5), (2.0, 1.0, 1.0))  # large, bottom sits in cushion band
    assert not T.on(sofa, cushion).passed  # anchor-larger gate fails (sofa > cushion)


# --------------------------------------------------------------------------- in


def test_in_positive():
    room = rec(1, "room", (0, 0, 1.5), (6.0, 6.0, 3.0))
    chair = rec(2, "chair", (1, 1, 0.5), (0.6, 0.6, 1.0))
    assert T.in_(chair, room).passed


def test_in_negative_outside():
    room = rec(1, "room", (0, 0, 1.5), (6.0, 6.0, 3.0))
    chair = rec(2, "chair", (10, 10, 0.5), (0.6, 0.6, 1.0))
    assert not T.in_(chair, room).passed


def test_in_boundary_partial():
    room = rec(1, "room", (0, 0, 1.5), (6.0, 6.0, 3.0))  # x in [-3,3]
    # chair straddling the wall: half its footprint inside -> below 0.6 frac
    chair = rec(2, "chair", (3.0, 0, 0.5), (1.0, 1.0, 1.0))  # x in [2.5,3.5]
    assert not T.in_(chair, room).passed


# --------------------------------------------------------------------------- near


def test_near_positive():
    a = rec(1, "cup", (0, 0, 0), (0.2, 0.2, 0.2))
    b = rec(2, "plate", (0.5, 0, 0), (0.2, 0.2, 0.2))
    assert T.near(a, b).passed


def test_near_negative():
    a = rec(1, "cup", (0, 0, 0), (0.2, 0.2, 0.2))
    b = rec(2, "plate", (5, 0, 0), (0.2, 0.2, 0.2))
    assert not T.near(a, b).passed


def test_near_uses_floor_for_small_anchor():
    # tiny anchor -> threshold is the 1.2 m floor, not 0.6*diag
    b = rec(2, "cup", (0, 0, 0), (0.1, 0.1, 0.1))
    assert T.near_thresh(b) == 1.2
    a = rec(1, "spoon", (1.1, 0, 0), (0.1, 0.1, 0.1))  # gap ~0.9 m < 1.2
    assert T.near(a, b).passed


def test_near_scale_adaptive_same_gap():
    # SAME edge gap of 1.5 m: too far for a small anchor, near for a huge one.
    small = rec(1, "cup", (0, 0, 0), (0.2, 0.2, 0.2))  # thresh = 1.2 floor
    huge = rec(2, "sofa", (0, 0, 0), (4.0, 2.0, 1.0))  # diag ~4.47, thresh ~2.68
    # small anchor half-x=0.1, probe half-x=0.05 -> center at 0.1+0.05+1.5=1.65 -> gap 1.5
    probe_small = rec(3, "spoon", (1.65, 0, 0), (0.1, 0.1, 0.1))
    assert not T.near(probe_small, small).passed  # 1.5 > 1.2 floor
    # huge anchor half-x=2.0, probe half-x=0.05 -> center at 2.0+0.05+1.5=3.55 -> gap 1.5
    probe_huge = rec(4, "spoon", (3.55, 0, 0), (0.1, 0.1, 0.1))
    assert T.near(probe_huge, huge).passed  # 1.5 <= ~2.68


# --------------------------------------------------------------------------- next_to


def test_next_to_positive():
    a = rec(1, "nightstand", (0, 0, 0), (0.5, 0.5, 0.5))
    b = rec(2, "bed", (0.9, 0, 0), (0.6, 2.0, 0.5))  # edge gap ~0.35 m
    assert T.next_to(a, b).passed


def test_next_to_negative_beyond_gap():
    a = rec(1, "nightstand", (0, 0, 0), (0.5, 0.5, 0.5))  # half-x 0.25
    b = rec(2, "bed", (2.0, 0, 0), (0.6, 2.0, 0.5))  # half-x 0.3, gap = 2-0.25-0.3=1.45
    assert not T.next_to(a, b).passed  # 1.45 > 0.75


def test_next_to_boundary():
    a = rec(1, "x", (0, 0, 0), (1.0, 1.0, 1.0))  # x in [-0.5,0.5]
    # gap exactly 0.75
    b = rec(2, "y", (0.5 + 0.75 + 0.5, 0, 0), (1.0, 1.0, 1.0))
    from core.geometry import primitives as PR

    assert abs(PR.aabb_gap(a.aabb_min, a.aabb_max, b.aabb_min, b.aabb_max) - 0.75) < 1e-9
    assert T.next_to(a, b).passed  # inclusive at the boundary


# --------------------------------------------------------------------------- between


def test_between_positive_midpoint():
    b1 = rec(1, "chair", (-3, 0, 0), (0.6, 0.6, 1.0))
    b2 = rec(2, "chair", (3, 0, 0), (0.6, 0.6, 1.0))
    a = rec(3, "rug", (0, 0.1, 0), (1.0, 1.0, 0.1))
    assert T.between(a, b1, b2).passed


def test_between_negative_off_axis():
    b1 = rec(1, "chair", (-3, 0, 0), (0.6, 0.6, 1.0))  # half-width 0.3 -> radius 0.3
    b2 = rec(2, "chair", (3, 0, 0), (0.6, 0.6, 1.0))
    a = rec(3, "rug", (0, 2.0, 0), (0.2, 0.2, 0.1))  # 2 m off the segment
    assert not T.between(a, b1, b2).passed


def test_between_at_endpoint_fails_strict():
    # H5/T8-C5 strict betweenness: a sitting exactly at b1's centroid projects to
    # t=0 (the segment end) and must FAIL — "between" excludes the anchor positions.
    b1 = rec(1, "post", (0, 0, 0), (1.0, 1.0, 1.0))
    b2 = rec(2, "post", (5, 0, 0), (1.0, 1.0, 1.0))
    a = rec(3, "dot", (0, 0, 0), (0.1, 0.1, 0.1))
    assert not T.between(a, b1, b2).passed


def test_between_radius_is_max_halfwidth():
    # wide b2 sets the capsule radius; a offset within that radius passes.
    b1 = rec(1, "post", (-4, 0, 0), (0.4, 0.4, 1.0))  # half-w 0.2
    b2 = rec(2, "wall", (4, 0, 0), (0.4, 3.0, 1.0))  # footprint half-width = max(0.2,1.5)=1.5
    a = rec(3, "dot", (0, 1.3, 0), (0.1, 0.1, 0.1))  # 1.3 < 1.5 radius
    assert T.between(a, b1, b2).passed
    a2 = rec(4, "dot", (0, 1.7, 0), (0.1, 0.1, 0.1))  # 1.7 > 1.5
    assert not T.between(a2, b1, b2).passed


# --------------------------------------------------------------------------- above / under


def test_above_positive():
    bed = rec(1, "bed", (0, 0, 0.3), (2.0, 1.5, 0.6))  # top z=0.6
    picture = rec(2, "picture", (0, 0, 1.5), (1.0, 0.1, 0.5))  # bottom z=1.25
    assert T.above(picture, bed).passed


def test_above_negative_no_overlap():
    bed = rec(1, "bed", (0, 0, 0.3), (2.0, 1.5, 0.6))
    picture = rec(2, "picture", (5, 0, 1.5), (1.0, 0.1, 0.5))  # no footprint overlap
    assert not T.above(picture, bed).passed


def test_above_negative_below():
    bed = rec(1, "bed", (0, 0, 1.0), (2.0, 1.5, 0.6))  # bottom z=0.7
    rug = rec(2, "rug", (0, 0, 0.05), (2.0, 1.5, 0.1))  # below the bed
    assert not T.above(rug, bed).passed


def test_under_positive():
    table = rec(1, "table", (0, 0, 1.0), (1.5, 1.5, 0.1))  # bottom z=0.95
    box = rec(2, "box", (0, 0, 0.3), (0.5, 0.5, 0.5))  # top z=0.55
    assert T.under(box, table).passed


def test_under_negative():
    table = rec(1, "table", (0, 0, 1.0), (1.5, 1.5, 0.1))
    box = rec(2, "box", (0, 0, 2.0), (0.5, 0.5, 0.5))  # above the table
    assert not T.under(box, table).passed


# --------------------------------------------------------------------------- with_feature


def test_with_feature_inside():
    table = rec(1, "table", (0, 0, 0.5), (2.0, 1.0, 1.0))
    lamp = rec(2, "lamp", (0.3, 0.2, 1.2), (0.2, 0.2, 0.4))  # centroid over the table
    assert T.with_feature(table, lamp).passed


def test_with_feature_near_pad():
    table = rec(1, "table", (0, 0, 0.5), (2.0, 1.0, 1.0))  # x in [-1,1]
    lamp = rec(2, "lamp", (1.2, 0, 1.2), (0.2, 0.2, 0.4))  # 0.2 m outside footprint edge
    assert T.with_feature(table, lamp).passed  # within with_feature_pad=0.30


def test_with_feature_negative():
    table = rec(1, "table", (0, 0, 0.5), (2.0, 1.0, 1.0))
    lamp = rec(2, "lamp", (3.0, 0, 1.2), (0.2, 0.2, 0.4))  # far away
    assert not T.with_feature(table, lamp).passed


# --------------------------------------------------------------------------- superlatives


def test_closest_to_ranking_and_margin():
    anchor = rec(1, "door", (0, 0, 0), (0.1, 2.0, 2.0))
    c1 = rec(2, "chair", (1, 0, 0), (0.5, 0.5, 1.0))  # dist 1
    c2 = rec(3, "chair", (3, 0, 0), (0.5, 0.5, 1.0))  # dist 3
    c3 = rec(4, "chair", (2, 0, 0), (0.5, 0.5, 1.0))  # dist 2
    r = T.closest_to([c1, c2, c3], anchor)
    assert r.order == [2, 4, 3]  # ids by ascending distance
    assert abs(r.margin - 1.0) < 1e-9  # runner-up (2) minus winner (1)
    assert r.distances[2] == 1.0


def test_farthest_from_ranking():
    anchor = rec(1, "door", (0, 0, 0), (0.1, 2.0, 2.0))
    c1 = rec(2, "chair", (1, 0, 0), (0.5, 0.5, 1.0))
    c2 = rec(3, "chair", (3, 0, 0), (0.5, 0.5, 1.0))
    r = T.farthest_from([c1, c2], anchor)
    assert r.order == [3, 2]
    assert abs(r.margin - 2.0) < 1e-9


def test_superlative_tie_break_by_id():
    anchor = rec(1, "door", (0, 0, 0), (0.1, 0.1, 0.1))
    c1 = rec(5, "chair", (2, 0, 0), (0.5, 0.5, 1.0))
    c2 = rec(3, "chair", (0, 2, 0), (0.5, 0.5, 1.0))  # same distance, lower id
    r = T.closest_to([c1, c2], anchor)
    assert r.order == [3, 5]  # tie broken by ascending id
    assert r.margin == 0.0


# ----------------------------------------------------- H5 NUM-F2 regressions


def test_on_pillow_on_sofa_num_f2():
    # NUM-F2: a pillow resting among sofa cushions sits 0.2-1.1 m BELOW the sofa's
    # AABB top (the backrest raises the top above the seat) with 100% footprint
    # overlap. The old top-face-only on() rejected it; the H5 support-band form must
    # accept it (on() TRUE).
    sofa = rec(1, "sofa", (0, 0, 0.45), (2.0, 1.0, 0.9))  # zmin 0, top 0.9 (backrest)
    # pillow on the seat ~0.45 m: bottom 0.45 is 0.45 m below the AABB top (in-spec 0.2-1.1)
    pillow = rec(2, "pillow", (0, 0, 0.55), (0.4, 0.4, 0.2))  # bottom z=0.45
    vgap_below_top = sofa.aabb_max[2] - pillow.aabb_min[2]
    assert 0.2 <= vgap_below_top <= 1.1  # NUM-F2 empirical range
    r = T.on(pillow, sofa)
    assert r.passed and r.score > 0


def test_on_pillow_on_bed_num_f2():
    # Companion NUM-F2 case: pillow on a bed with a headboard raising the AABB top.
    bed = rec(1, "bed", (0, 0, 0.5), (2.0, 1.6, 1.0))  # zmin 0, top 1.0 (headboard)
    pillow = rec(2, "pillow", (0, 0, 0.5), (0.5, 0.5, 0.2))  # bottom z=0.4
    vgap_below_top = bed.aabb_max[2] - pillow.aabb_min[2]
    assert 0.2 <= vgap_below_top <= 1.1  # pillow bottom within NUM-F2 range below AABB top
    assert T.on(pillow, bed).passed


def test_above_wall_picture_num_f2():
    # NUM-F2: a wall-hung picture "above the bed" has ZERO footprint overlap with
    # the bed (it is on the wall behind the headboard) but a small lateral offset.
    # The H5 lateral-offset form must accept it (above() TRUE); the old overlap
    # gate rejected all 5 such pictures.
    bed = rec(1, "bed", (0, 0, 0.3), (2.0, 1.6, 0.6))  # top z=0.6, y in [-0.8, 0.8]
    # picture on the wall just past the bed's y-edge, well above it
    picture = rec(2, "picture", (0, 1.0, 1.6), (1.0, 0.1, 0.5))  # centre y=1.0, no overlap
    from core.geometry import primitives as PR

    assert not PR.footprints_overlap(
        picture.aabb_min, picture.aabb_max, bed.aabb_min, bed.aabb_max
    )
    assert T.above(picture, bed).passed  # within above_lateral_infl (0.5 m) + above bed


def test_above_far_picture_still_rejected():
    # Guard the lateral tolerance is not unbounded: a picture 3 m to the side of the
    # bed is NOT above it.
    bed = rec(1, "bed", (0, 0, 0.3), (2.0, 1.6, 0.6))
    far = rec(2, "picture", (0, 3.0, 1.6), (1.0, 0.1, 0.5))
    assert not T.above(far, bed).passed


# ----------------------------------------------------- H5 tuck-under (DD-A7)


def test_under_stool_tucked_under_table():
    # A stool tucked under a table: its top rises above the table's AABB min_z
    # (~floor), so the STRICT branch can never pass. Branch (ii) tuck-under fires
    # because 'table' is an UNDER_RELATION anchor and the stool sits at floor level
    # with its top below the table's AABB top.
    table = rec(1, "table", (0, 0, 0.4), (1.2, 1.2, 0.8))  # zmin 0, top 0.8
    stool = rec(2, "stool", (0, 0, 0.25), (0.4, 0.4, 0.5))  # zmin 0, top 0.5 < table top
    assert stool.aabb_max[2] > table.aabb_min[2]  # top above table floor -> strict fails
    r = T.under(stool, table)
    assert r.passed and "tuck-under" in r.explanation


def test_under_tuck_gated_to_under_relation_class():
    # The tuck-under branch is class-gated: the SAME geometry under a non-UNDER
    # anchor (e.g. a 'picture') must NOT pass tuck-under (only strict below counts).
    picture = rec(1, "picture", (0, 0, 0.4), (1.2, 1.2, 0.8))  # not an UNDER_RELATION class
    stool = rec(2, "stool", (0, 0, 0.25), (0.4, 0.4, 0.5))
    assert not T.under(stool, picture).passed


def test_under_strict_branch_still_works():
    # Branch (i) strict below is unchanged in spirit: a rug directly under a table
    # top (top <= anchor bottom + tol) with footprint IoM passes.
    table = rec(1, "table", (0, 0, 1.0), (1.5, 1.5, 0.1))  # bottom z=0.95
    rug = rec(2, "rug", (0, 0, 0.05), (1.4, 1.4, 0.1))  # top z=0.1 << 0.95
    assert T.under(rug, table).passed


# ----------------------------------------------------- H5 strict betweenness


def test_between_off_segment_end_rejected():
    # T8-C5 strict betweenness: a target BESIDE anchor b1, off the segment end
    # (projects to t clamped at 0) must FAIL even inside the capsule radius.
    b1 = rec(1, "chair", (0, 0, 0), (0.6, 0.6, 1.0))
    b2 = rec(2, "chair", (4, 0, 0), (0.6, 0.6, 1.0))  # segment along +x
    # target at x=-1 (before b1), y within radius: projects off the near end
    beside = rec(3, "rug", (-1.0, 0.1, 0), (0.3, 0.3, 0.1))
    r = T.between(beside, b1, b2)
    assert not r.passed and "strict" in r.explanation


# ----------------------------------------------------- H5 size resolver (DD-A12)


def test_size_attr_match_largest_with_gap():
    # Relative per-class largest-face ranking: the big table (>=1.2x the next) is the
    # 'big' one; a table only 1.1x larger is NOT separated -> matches nothing.
    from core.geometry.toolbox import _size_attr_match

    big = rec(1, "table", (0, 0, 0), (2.0, 2.0, 0.1))  # face 4.0
    mid = rec(2, "table", (5, 0, 0), (1.0, 1.0, 0.1))  # face 1.0
    pool = [big, mid]  # ratio 4.0 -> separated
    assert _size_attr_match(big, "big", pool, TH)
    assert not _size_attr_match(mid, "big", pool, TH)

    a = rec(1, "table", (0, 0, 0), (1.1, 1.0, 0.1))  # face 1.1
    b = rec(2, "table", (5, 0, 0), (1.0, 1.0, 0.1))  # face 1.0, ratio 1.1 < 1.2
    close_pool = [a, b]
    assert not _size_attr_match(a, "big", close_pool, TH)  # honest none: not separated


def test_size_attr_match_smallest_with_gap():
    from core.geometry.toolbox import _size_attr_match

    tiny = rec(1, "chair", (0, 0, 0), (0.5, 0.5, 0.1))  # face 0.25
    big = rec(2, "chair", (5, 0, 0), (1.0, 1.0, 0.1))  # face 1.0, ratio 4x
    pool = [tiny, big]
    assert _size_attr_match(tiny, "small", pool, TH)
    assert not _size_attr_match(big, "small", pool, TH)
