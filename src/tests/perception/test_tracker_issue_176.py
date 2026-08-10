"""Issue #176: the tracker keeps duplicate instances of one physical object,
inflating per-class counts up to 8x.

Live evidence (tree 1fe2167, sweep jobs 712650-712658, see the issue body):
``reports/cluster_verify/712652/debug/1_chinese_room_nume/instance_index.jsonl``
holds 46 'chair' instances for 6 physical chairs in the scene. Inspecting that
archive shows two populations:

* 7 well-separated, well-formed chairs (ids 90, 111, 112, 113, 115, 116, 120 --
  their positions are metres apart and their AABBs match the class's typical
  extent) -- these are genuinely distinct real chairs and MUST stay separate.
* 25 near-duplicate instances (ids 114, 122, 130, 131, 133-153) whose AABBs are
  byte-identical to three decimal places, taken verbatim from that archive:
  ``aabb_min ~= [4.10, -0.72, 0.44]``, ``aabb_max ~= [5.48, -0.18, 0.77]`` -- a
  1.38 x 0.53 x 0.34 m box, already PAST the chair prior's veto ceiling (sorted
  typ_ext (0.583, 0.652, 1.047) * extent_veto_factor(1.3) == (0.758, 0.848,
  1.361) -- the box's own 1.38 m long axis exceeds the 1.361 m ceiling on its
  own, before any new detection is even considered). Every one of the 25
  persists for exactly one keyframe before a fresh id takes over at (near-)the
  same box.

Root cause (:mod:`core.perception.tracker`'s ``_match_plausible``): the #153 fix
only waives the extent veto at near-zero (<=0.2 m) centroid separation. Once a
candidate's OWN accumulated box already sits at/past the class-typical ceiling
(exactly the archived case above), EVERY later re-observation that drifts past
0.2 m -- plausible for a detection this large and noisy (a bbox spanning half
the panorama has far more registration jitter than a tightly-boxed detection's
few cm) -- fails BOTH the #153 floor and the absolute ceiling (which the
candidate already exceeded on its own, independent of anything the new
detection contributes). ``associate()`` then mints a fresh duplicate instead of
fusing: one real chair ratchets into 25 ghost instances.

The fix (``TrackerConfig.extent_growth_tol``): judge a match by how much it
GROWS the box relative to what the candidate already spans, not just by the
absolute ceiling. A re-observation that does not enlarge the box is, by
construction, more of the SAME (already established, however oversized)
object.

This module replays the archived AABBs (no bag reads, no live_replay -- the
numbers below are the archive's own instance_index.jsonl fields, hardcoded with
their provenance) through the real ``associate()`` path.
"""
from __future__ import annotations

import numpy as np

from core.perception.detector import Detection
from core.perception.fusion import Fused3D
from core.perception.scene_index import BasicSceneIndex
from core.perception.tracker import TrackerConfig, associate

# --------------------------------------------------------------------------- fixture data
#
# Archived aabb_min/aabb_max, one entry per archived instance id, taken verbatim
# (3dp) from
# reports/cluster_verify/712652/debug/1_chinese_room_nume/instance_index.jsonl's
# final periodic record (chair ids 114, 122, 130, 131, 133-153 -- 25 records).
_DUPLICATE_CHAIN_AABBS: list[tuple[tuple[float, float, float], tuple[float, float, float]]] = [
    ((4.101, -0.716, 0.435), (5.480, -0.183, 0.772)),  # id 114
    ((4.104, -0.716, 0.435), (5.480, -0.183, 0.772)),  # id 122
    ((4.101, -0.716, 0.435), (5.480, -0.183, 0.772)),  # id 130
    ((4.112, -0.716, 0.436), (5.480, -0.183, 0.772)),  # id 131
    ((4.101, -0.716, 0.435), (5.480, -0.183, 0.772)),  # id 133
    ((4.104, -0.716, 0.435), (5.480, -0.183, 0.772)),  # id 134
    ((4.104, -0.716, 0.435), (5.480, -0.183, 0.772)),  # id 135
    ((4.104, -0.716, 0.435), (5.480, -0.183, 0.772)),  # id 136
    ((4.104, -0.716, 0.435), (5.480, -0.183, 0.772)),  # id 137
    ((4.108, -0.716, 0.435), (5.480, -0.183, 0.772)),  # id 138
    ((4.101, -0.716, 0.435), (5.480, -0.183, 0.772)),  # id 139
    ((4.104, -0.716, 0.435), (5.480, -0.183, 0.772)),  # id 140
    ((4.101, -0.716, 0.435), (5.480, -0.183, 0.772)),  # id 141
    ((4.104, -0.716, 0.435), (5.480, -0.183, 0.772)),  # id 142
    ((4.101, -0.716, 0.435), (5.480, -0.183, 0.772)),  # id 143
    ((4.108, -0.716, 0.435), (5.480, -0.183, 0.772)),  # id 144
    ((4.101, -0.716, 0.435), (5.480, -0.183, 0.772)),  # id 145
    ((4.108, -0.716, 0.435), (5.480, -0.183, 0.772)),  # id 146
    ((4.101, -0.716, 0.435), (5.480, -0.183, 0.772)),  # id 147
    ((4.108, -0.716, 0.435), (5.480, -0.183, 0.772)),  # id 148
    ((4.101, -0.716, 0.435), (5.480, -0.183, 0.772)),  # id 149
    ((4.104, -0.716, 0.435), (5.480, -0.183, 0.772)),  # id 150
    ((4.108, -0.716, 0.435), (5.480, -0.183, 0.772)),  # id 151
    ((4.101, -0.716, 0.435), (5.480, -0.183, 0.772)),  # id 152
    ((3.100, -0.964, 0.334), (5.484, -0.169, 0.833)),  # id 153
]

# The 7 genuinely distinct, well-formed chairs from the SAME archived record
# (ids 90, 111, 112, 113, 115, 116, 120) -- metres apart from the duplicate chain
# and from each other; a fix must not pull these together.
_DISTINCT_CHAIR_AABBS: list[tuple[tuple[float, float, float], tuple[float, float, float]]] = [
    ((-0.241, -0.241, -0.241), (0.241, 0.241, 0.241)),   # id 90
    ((3.798, -4.886, 0.029), (4.356, -4.713, 1.600)),    # id 111
    ((4.492, -0.026, 0.634), (4.797, 0.623, 0.825)),     # id 112
    ((4.394, -3.267, -0.251), (5.372, -1.696, 0.231)),   # id 113
    ((5.927, -4.920, -0.004), (6.436, -4.438, 0.783)),   # id 115
    ((5.940, -3.013, -0.075), (6.626, -1.442, 0.708)),   # id 116
    ((6.124, -3.165, 0.112), (6.614, -1.595, 1.090)),    # id 120
]

# Issue #176 diagnosis: the archived AABBs alone don't preserve the underlying
# raw per-frame centroid (only the final, already-merged instance's trimmed
# box) -- fine detail the JSONL dump never recorded. 0.35 m is the reproduction
# parameter this module supplies: within the chair centroid gate (0.682 m, see
# _assoc_gate('chair', ...)) but past the #153 floor (0.2 m) -- exactly the
# "decisive band" issue #161 names, plausible registration jitter for a
# detection whose bbox spans half the panorama (see the module docstring).
_REOBSERVATION_OFFSET_M = 0.35


def _det(score=0.3):
    # bbox is unused by associate()/labels_compatible -- only label/score matter
    # once a Fused3D is already built.
    return Detection(tile_id=0, bbox_xyxy=(0.0, 0.0, 10.0, 10.0), label="chair", score=score)


def _fused_from_points(pts):
    centroid = pts.mean(axis=0)
    return Fused3D(centroid=centroid, points=pts, n_points=len(pts), range_m=float(np.linalg.norm(centroid)))


def _corner_cloud(aabb_min, aabb_max):
    """The two extreme corners of an archived AABB -- with only 2 points, the
    real scene index's percentile trim (2nd/98th) is a no-op, so a candidate
    built by folding two of these together lands EXACTLY on the archived box,
    reproducing its real (already past the class ceiling) extent."""
    return np.vstack([aabb_min, aabb_max]).astype(np.float32)


def _local_cloud(centre, seed, half=(0.2, 0.15, 0.15), n=30):
    """A small, local cluster (well within the chair class's typical footprint
    on its own) around ``centre`` -- a plausible SINGLE-frame re-observation."""
    rng = np.random.default_rng(seed)
    lo = np.asarray(centre) - np.asarray(half)
    hi = np.asarray(centre) + np.asarray(half)
    return rng.uniform(lo, hi, size=(n, 3)).astype(np.float32)


def _make_oversized_candidate(aabb_min, aabb_max, cfg):
    """Fold two corner-only detections of an archived duplicate-chain AABB into
    ONE candidate, mirroring how the real ghost was born (a same-keyframe fold,
    #153's near-zero floor always waives an exact-corner match regardless of
    the fix under test)."""
    idx = BasicSceneIndex()
    a = _corner_cloud(aabb_min, aabb_max)
    b = _corner_cloud(aabb_min, aabb_max)
    associate([(_det(), _fused_from_points(a)), (_det(), _fused_from_points(b))], idx, cfg=cfg)
    return idx


def _replay_reobservation(aabb, cfg, offset_sign, seed):
    """Build the oversized candidate from ``aabb``, then feed ONE more, local
    re-observation offset by :data:`_REOBSERVATION_OFFSET_M` from its centroid
    -- the archived ghost-chain mechanism's decisive step. Returns the resulting
    instance count (1 == fused, 2 == a fresh duplicate ghost was minted)."""
    idx = _make_oversized_candidate(aabb[0], aabb[1], cfg)
    cand = idx.all_instances()[0]
    offset = np.array([_REOBSERVATION_OFFSET_M * offset_sign, 0.0, 0.0])
    pts = _local_cloud(cand.centroid + offset, seed=seed)
    associate([(_det(), _fused_from_points(pts))], idx, cfg=cfg)
    return len(idx.all_instances())


# --------------------------------------------------------------------------- reproduction


def test_oversized_candidate_matches_the_archived_extent():
    """Sanity check pinning the archived numbers this module replays: folding
    two corner-only detections of the archived duplicate-chain AABB produces a
    candidate whose own extent already exceeds the chair prior's veto ceiling
    -- the fact the #176 root cause depends on."""
    from core.perception.dimension_priors import prior_for

    idx = _make_oversized_candidate(*_DUPLICATE_CHAIN_AABBS[0], cfg=TrackerConfig())
    cand = idx.all_instances()[0]
    ext = np.sort(cand.aabb_max - cand.aabb_min)
    ceiling = np.sort(prior_for("chair").typ_ext) * TrackerConfig().extent_veto_factor
    assert ext[-1] > ceiling[-1], (
        f"expected the archived box's long axis ({ext[-1]:.3f}) to already exceed "
        f"the veto ceiling ({ceiling[-1]:.3f}) -- the #176 precondition"
    )


def test_PRE_FIX_every_archived_ghost_fails_to_rejoin_its_own_track():
    """Pins the PRE-#176-fix number: with the growth-relative rescue disabled
    (``extent_growth_tol`` set impossibly negative so the check can never pass
    -- only the #153 near-zero floor and the absolute ceiling remain, matching
    the tracker as it stood on tree 1fe2167), replaying the decisive
    re-observation step for EVERY one of the 25 archived duplicate-chain AABBs
    mints a fresh ghost instance every time -- reproducing the live sweep's
    46-chair-for-6-GT overcount mechanism. Issue #217 later widened the
    absolute ceiling itself (chair's own recorded long-axis cap_factor is
    2.794, real measured variance above the floor -- see
    dimension_priors._CAP_FACTOR), so both #217 widenings are ALSO disabled
    here (``extent_veto_abs_slack_m=0.0``, ``extent_veto_use_cap_factor=False``)
    to keep reconstructing the exact pre-#176/#217 ceiling this test names."""
    pre_fix_cfg = TrackerConfig(
        extent_growth_tol=-1.0,
        extent_veto_abs_slack_m=0.0,
        extent_veto_use_cap_factor=False,
    )
    results = [
        _replay_reobservation(aabb, pre_fix_cfg, offset_sign=1 if i % 2 == 0 else -1, seed=i)
        for i, aabb in enumerate(_DUPLICATE_CHAIN_AABBS)
    ]
    assert results == [2] * len(_DUPLICATE_CHAIN_AABBS), (
        f"expected every archived ghost to mint a duplicate pre-fix, got {results}"
    )


def test_duplicate_chain_collapses_post_fix():
    """Issue #176 fix: replaying the SAME decisive re-observation step for every
    archived duplicate-chain AABB, through the DEFAULT tracker config, now fuses
    into the established track every time -- one physical object, one tracked
    instance, instead of one ghost per keyframe."""
    results = [
        _replay_reobservation(aabb, TrackerConfig(), offset_sign=1 if i % 2 == 0 else -1, seed=i)
        for i, aabb in enumerate(_DUPLICATE_CHAIN_AABBS)
    ]
    assert results == [1] * len(_DUPLICATE_CHAIN_AABBS), (
        f"expected every archived ghost to fuse into its track post-fix, got {results}"
    )


def test_distinct_archived_chairs_stay_separate_post_fix():
    """Do-not-over-correct guard: the 7 genuinely distinct, well-formed chairs
    from the SAME archived record (metres apart, well-formed AABBs) must stay 7
    separate instances under the fix -- the growth-relative rescue must not
    become a licence to merge real, distinct same-class objects (the
    #94/#154/#161 regression class this task must not reintroduce)."""
    idx = BasicSceneIndex()
    cfg = TrackerConfig()
    for i, (aabb_min, aabb_max) in enumerate(_DISTINCT_CHAIR_AABBS):
        pts = _local_cloud(
            (np.asarray(aabb_min) + np.asarray(aabb_max)) / 2.0,
            seed=100 + i,
            half=(np.asarray(aabb_max) - np.asarray(aabb_min)) / 2.0,
        )
        associate([(_det(), _fused_from_points(pts))], idx, cfg=cfg)

    insts = [r for r in idx.all_instances() if r.label == "chair"]
    assert len(insts) == len(_DISTINCT_CHAIR_AABBS)
    assert all(r.n_obs == 1 for r in insts)


def test_two_distinct_tvs_in_the_161_decisive_band_still_split_post_fix():
    """Must-not-regress (#161): two genuinely distinct televisions 0.65 m apart
    (centroid distance ~0.61 m -- inside the television centroid gate of
    0.745 m, so the extent veto is the ONLY thing keeping them apart) must stay
    2 instances under the #176 fix. A genuinely separate object at typical class
    spacing adds most of its own footprint to the union (the union's width here
    grows by ~0.65 m), which the growth-relative check still rejects -- unlike
    the archived chair case above, where a re-observation adds ~nothing new."""
    from core.perception import tiling as T

    def _det_for_point(x, y, z, od, label, half_px=45):
        apex = np.array([od.x, od.y, od.z])
        dx, dy, dz = x - apex[0], y - apex[1], z - apex[2]
        horiz = np.hypot(dx, dy)
        bearing = np.arctan2(dy, dx)
        elevation = np.arctan2(dz, horiz)
        az, el = T.map_ray_to_camera(bearing, elevation, od.yaw)
        spec = T.tile_specs()[0]
        local_az = az - spec.yaw_center
        xcam = -np.tan(local_az)
        ycam = -np.tan(el) * np.hypot(xcam, 1.0)
        u = xcam * spec.focal_x + spec.cx
        v = ycam * spec.focal_y + spec.cy
        bb = (u - half_px, v - half_px, u + half_px, v + half_px)
        return Detection(tile_id=0, bbox_xyxy=bb, label=label, score=0.9)

    def _tv_cloud(y_centre, seed):
        rng = np.random.default_rng(seed)
        return np.column_stack([
            2.60 + rng.uniform(-0.02, 0.02, 60),
            y_centre + rng.uniform(-0.6, 0.6, 60),
            0.735 + rng.uniform(-0.4, 0.4, 60),
        ]).astype(np.float32)

    from core.interfaces import OdomState

    od = OdomState(t=0.0, x=0.0, y=0.0, z=0.0, yaw=0.0)
    det_a = _det_for_point(2.60, 0.0, 0.735, od, "television")
    det_b = _det_for_point(2.60, 0.65, 0.735, od, "television")
    f_a = _fused_from_points(_tv_cloud(0.0, seed=1))
    f_b = _fused_from_points(_tv_cloud(0.65, seed=2))

    idx = BasicSceneIndex()
    associate([(det_a, f_a), (det_b, f_b)], idx, cfg=TrackerConfig())
    assert len(idx.all_instances()) == 2
