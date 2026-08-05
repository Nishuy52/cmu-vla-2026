"""Issue #184: the object_reference head ranks candidates and resolves anchors over
ALL tracks, so phantom same-class tracks (single/few-observation detector noise) win a
relation clause, a candidate rank, or a superlative anchor pick they have no business
winning -- deterministically, not noise (the #118 mechanism table, 5 Aug comment: windows
31 vs GT 1, lamps 92 vs GT 3, bedside tables 22 vs GT 2, folding screens 12 vs GT 1).

The fix (``ObjectRefHead._resolve``, ``core.heads.scene_established.EstablishedView``)
extends the #151 pattern (``core.heads.numerical._count`` / ``ANCHOR_ESTABLISHED_N_OBS``)
from NUMERICAL's relation-clause anchor filtering to OBJECT_REFERENCE's own candidate pool
AND anchor pool: rank/resolve first against instances seen ``ESTABLISH_N_OBS`` times, with
a two-tier fall-open to the raw pool so the filter can only ever move the pick toward more
plausible instances and can never fabricate an empty result.

Live evidence (job 713818, tree matching the #118 comment) reproduced here verbatim from
``reports/cluster_verify/713818/debug/*/instance_index.jsonl``'s final ``answer_time``
record (positions/AABBs/n_obs/score, as archived) and ``resolved_plan.jsonl``'s parsed
clause for the same question:

* ``10_hotel_room_2_obje`` -- "Find the flowers near the window." 20 'flower' tracks
  exist for ONE physical flower vase (GT unique_in_scene); 31 'window' tracks exist for
  ONE physical window. The live answer published instance 0 (n_obs=2, a single/few-frame
  ghost) -- confirmed here by matching the archived ``captures/scores.json`` live marker
  position exactly. Post-fix, the established view excludes id 0 (and every other n_obs<3
  ghost) from the ranked pool entirely, so the winner is now a genuinely re-observed
  (established) track, not a ghost -- see ``test_flowers_near_window_excludes_the_ghost``.
  KNOWN RESIDUAL GAP (documented, not silently papered over): the established winner
  (id 2, ~3.1 m from GT) is still not the GT-nearest established candidate (id 82,
  ~0.76 m from GT, also established but loses on `resolve()`'s own final instance-id
  tie-break inside `_tier_priority_order` once `near()`'s score saturates to 1.0 for both
  -- an AABB-gap-based score with no distance headroom above the near-threshold). Fixing
  that tie-break is a `core/geometry/toolbox.py` ranking change, outside #184's stated
  SceneIndex-view-only scope (predicates/ranking untouched) -- filed as a follow-up
  (see issue tracker).
* ``8_hotel_room_1_obje`` -- "Find the bedside table farthest from the window." 22
  'bedside table' tracks exist for GT's 2 real ones; 40 'window' tracks exist for GT's 1
  real one. The live answer published instance 116 (n_obs=35 -- itself well "established",
  so no observation-count gate can ever exclude it) -- confirmed here the same way. The
  root cause is `resolve()`'s own superlative anchor pick, `anchor_recs[0]`, documented in
  `core/geometry/toolbox.py` as "salience: first (index order); deterministic": it lands on
  window id 26, which happens to have exactly `ESTABLISH_N_OBS` (3) observations -- already
  established, so the #184 gate is a no-op for this ONE anchor pick specifically (a floor
  set any higher would be tuning to this issue's own sample, against the generalization
  protocol). See ``test_bedside_table_farthest_established_gate_is_a_documented_no_op``
  for the verified (not fabricated) before/after and the follow-up recommendation.

Issue #186 (this module's characterization tests, updated) closes both `toolbox.py`
ranking gaps the paragraphs above flagged as follow-ups:

* ``_tier_priority_order``'s final tie-break now ranks same-tier/same-clause-score
  survivors by DESCENDING ``n_obs`` before falling to ascending ``instance_id`` (the
  never-reached-in-practice last resort, unchanged) -- the same "identity only as a
  final tiebreak" discipline ``_select_sub_anchor`` (#151) already applies.
* ``resolve()``'s superlative anchor pick now calls ``_select_sub_anchor`` itself
  (nearest to the ranked pool > score > n_obs > id) instead of ``anchor_recs[0]``.

VERIFIED (not assumed) effect on both archived rows, recomputed against this exact
fixture data after the fix:

* Flowers-near-window: the established n_obs-descending winner is instance 5 (n_obs=59),
  not instance 2 (the old id-ascending winner) NOR instance 82 (the GT-nearest
  candidate an earlier analysis of this issue expected). Instance 5 is itself near
  (AABB gap 0.00 m, `near()` saturates to 1.0) an ESTABLISHED window (id 48, n_obs=21)
  that is NOT the real GT window (~7.5 m away) -- a second, independently-established
  ghost pair that also saturates `near()`, and that carries both higher `n_obs` (59 vs
  38) AND higher detector `score` (0.57 vs 0.28) than the GT-nearest candidate (id 82).
  No per-candidate evidence signal available on `InstanceRecord` (score, n_obs, or the
  unclamped `near()` margin, which is numerically IDENTICAL -- gap 0.00 m, margin
  1.2 -- for both ties) can tell these two established, `near()`-saturated pairs
  apart without GT. This is a real, narrower residual gap than #184 left (a
  ghost-vs-ghost tie among ESTABLISHED, non-index-order-selected candidates, caused by
  `near()`'s own AABB-gap saturation admitting more than one truly-independent
  candidate/anchor pair at score 1.0) -- documented here, not silently papered over,
  and filed as a follow-up rather than fixed by changing `near()`'s scoring semantics
  (out of this issue's stated scope).
* Bedside-table-farthest-from-window: the new evidence-based anchor pick selects
  window id 148 (n_obs=7, nearest among established windows to the established
  bedside-table pool) instead of id 26 (index-order pick); ranking established bedside
  tables by distance from that anchor now picks instance 45 (n_obs=3) -- phantom
  instance 116 (the old winner) no longer wins. id 148 is not itself the single
  closest established window to the real GT window position (see
  ``test_bedside_table_farthest_established_gate_is_a_documented_no_op`` for the full
  verified before/after), so this remains an evidence-based, deterministic, but not
  provably GT-optimal pick -- the anchor pool is genuinely noisy detector output and no
  purely local (score/n_obs/distance-to-pool) signal recovers the literal GT window
  from it in every case. What #186 guarantees is that the pick is no longer an
  index-order artifact and that a genuinely well-observed phantom no longer wins by
  construction.
"""
from __future__ import annotations

import numpy as np

from core.heads.object_ref import ObjectRefHead
from core.heads.scene_established import ESTABLISH_N_OBS
from core.geometry.toolbox import DEFAULT_THRESHOLDS, resolve
from core.interfaces import InstanceRecord
from core.plan_schema import Anchor, Clause, Plan, Pred, QType, TargetSpec
from tests.heads._helpers import inst, near_clause, object_plan, scene


def _rec(iid: int, label: str, aabb_min, aabb_max, n_obs: int, score: float = 0.3) -> InstanceRecord:
    lo = np.array(aabb_min, dtype=float)
    hi = np.array(aabb_max, dtype=float)
    return InstanceRecord(
        instance_id=iid,
        label=label,
        score=score,
        n_obs=n_obs,
        centroid=(lo + hi) / 2.0,
        aabb_min=lo,
        aabb_max=hi,
    )


def _head_winner(target: TargetSpec, question: str, *records: InstanceRecord) -> InstanceRecord:
    sc = scene(*records)
    plan = Plan(qtype=QType.OBJECT_REFERENCE, question_raw=question, target=target)
    head = ObjectRefHead(plan=plan)
    head.advance(sc)
    head.verify()
    return head.best_candidate


# --------------------------------------------------------------------------- #184
# archived-data fixtures (verbatim aabb_min/aabb_max/n_obs/score, as archived)

_HOTELROOM2_FLOWERS = [
    _rec(0, "flower", (-2.7, 2.46, 1.124), (-2.08, 2.482, 1.455), 2, 0.3331),
    _rec(1, "flower", (-2.003, 2.433, 0.88), (-1.672, 2.513, 1.562), 2, 0.3478),
    _rec(2, "flower", (-1.301, 2.434, 0.907), (-0.97, 2.514, 1.584), 3, 0.3311),
    _rec(3, "flower", (0.494, 0.875, 0.735), (0.691, 1.586, 1.067), 3, 0.2332),
    _rec(4, "flower", (0.638, 0.363, 0.558), (1.349, 0.585, 0.89), 2, 0.2058),
    _rec(5, "flower", (2.248, -1.356, 1.167), (2.959, -1.276, 1.498), 59, 0.5697),
    _rec(27, "flower", (-2.357, 2.13, 0.604), (-1.646, 2.461, 0.839), 4, 0.3522),
    _rec(29, "flower", (-0.665, 2.441, 0.861), (-0.334, 2.521, 1.414), 2, 0.2974),
    _rec(30, "flower", (-0.459, 2.444, 0.994), (0.228, 2.524, 1.325), 7, 0.386),
    _rec(31, "flower", (0.152, 2.444, 0.893), (0.484, 2.524, 1.498), 4, 0.4273),
    _rec(32, "flower", (0.99, -1.389, 0.909), (1.197, -1.309, 1.373), 2, 0.4695),
    _rec(33, "flower", (0.925, -1.856, 0.996), (1.257, -1.776, 1.513), 4, 0.4949),
    _rec(49, "flower", (-1.886, 2.115, 0.579), (-1.175, 2.446, 0.815), 2, 0.3155),
    _rec(51, "flower", (0.788, -1.856, 0.858), (1.071, -1.776, 1.526), 2, 0.4578),
    _rec(52, "flower", (1.494, 0.177, 1.181), (2.205, 0.412, 1.512), 2, 0.3342),
    _rec(56, "flower", (-4.198, 1.752, 0.448), (-3.963, 2.046, 0.745), 3, 0.3608),
    _rec(58, "flower", (1.837, -1.357, 1.296), (2.548, -1.277, 1.627), 2, 0.3977),
    _rec(82, "flower", (-4.667, 1.103, 0.872), (-4.587, 1.814, 1.133), 38, 0.2786),
    _rec(83, "flower", (-1.416, -0.943, 0.615), (-1.086, -0.786, 0.743), 19, 0.2083),
    _rec(84, "flower", (0.754, 0.36, 1.0), (0.99, 0.691, 1.711), 19, 0.4378),
]

_HOTELROOM2_WINDOWS = [
    _rec(6, "window", (-4.629, 0.373, 1.642), (-4.586, 1.959, 2.449), 4, 0.2292),
    _rec(8, "window", (-3.163, 2.414, 0.772), (-1.94, 2.504, 1.981), 29, 0.4195),
    _rec(9, "window", (-2.335, 2.461, 0.849), (-1.41, 2.484, 1.747), 27, 0.4049),
    _rec(11, "window", (0.56, 0.806, 0.488), (0.622, 2.395, 1.503), 4, 0.4583),
    _rec(12, "window", (0.544, -1.578, 0.115), (1.445, -1.488, 1.764), 3, 0.5518),
    _rec(14, "window", (1.728, 0.394, 0.191), (3.432, 0.484, 1.422), 2, 0.3334),
    _rec(42, "window", (-3.96, 2.194, 0.707), (-1.48, 2.284, 1.938), 2, 0.28),
    _rec(43, "window", (-0.896, 2.421, 0.796), (-0.14, 2.511, 1.492), 3, 0.3723),
    _rec(44, "window", (-0.384, 2.441, 0.773), (0.622, 2.484, 1.522), 5, 0.3479),
    _rec(45, "window", (-0.57, -1.378, 0.805), (0.662, -1.288, 2.079), 11, 0.4561),
    _rec(46, "window", (0.367, 0.437, 0.561), (1.159, 0.527, 1.049), 4, 0.4032),
    _rec(47, "window", (0.555, -1.726, 0.787), (1.135, -1.636, 1.742), 2, 0.3017),
    _rec(48, "window", (2.103, -1.344, 0.49), (3.235, -1.254, 2.292), 21, 0.3145),
    _rec(53, "window", (-3.551, 2.275, 0.545), (-1.072, 2.365, 1.776), 5, 0.3231),
    _rec(54, "window", (-2.715, 2.345, 0.45), (-1.484, 2.435, 1.701), 4, 0.2928),
    _rec(55, "window", (-2.264, 2.391, 0.694), (-0.632, 2.481, 1.926), 24, 0.3698),
    _rec(59, "window", (-4.629, 1.02, 0.867), (-4.586, 2.05, 2.535), 21, 0.3303),
    _rec(61, "window", (-4.629, 1.778, 0.891), (-4.586, 2.39, 1.985), 2, 0.2544),
    _rec(63, "window", (-4.297, 1.765, 0.561), (-3.77, 2.18, 0.651), 2, 0.2064),
    _rec(64, "window", (-4.034, 1.234, 0.442), (-3.847, 1.324, 0.9), 4, 0.3492),
    _rec(65, "window", (0.279, -1.505, 0.916), (1.159, -1.415, 2.366), 2, 0.2982),
    _rec(76, "window", (-4.567, 2.292, 0.499), (-3.336, 2.382, 2.492), 2, 0.4554),
    _rec(79, "window", (0.589, 0.427, 0.632), (0.679, 0.841, 1.146), 21, 0.3425),
    _rec(85, "window", (-4.473, 1.514, 1.141), (-4.383, 2.57, 2.743), 19, 0.2513),
    _rec(86, "window", (-4.213, 0.272, 0.647), (-3.896, 0.562, 0.737), 57, 0.3785),
    _rec(87, "window", (-0.026, 0.703, 2.554), (0.252, 0.867, 2.594), 19, 0.2215),
    _rec(88, "window", (0.555, -1.715, 0.17), (1.157, -1.625, 1.171), 38, 0.2116),
    _rec(89, "window", (0.814, -0.445, 0.771), (0.904, 1.583, 2.002), 19, 0.2254),
    _rec(90, "window", (0.427, -1.53, 0.081), (1.927, -1.44, 1.219), 19, 0.4924),
    _rec(91, "window", (2.092, 0.509, 0.585), (3.323, 0.599, 2.696), 19, 0.3924),
    _rec(92, "window", (3.22, -1.102, 0.101), (3.31, 0.129, 2.402), 19, 0.2445),
]

_HOTELROOM1_BEDSIDE_TABLES = [
    _rec(2, "bedside table", (-1.919, 0.922, 0.4), (-1.487, 1.891, 2.08), 3, 0.4426),
    _rec(3, "bedside table", (-1.763, -2.456, 0.73), (-1.309, -1.223, 0.949), 2, 0.6129),
    _rec(4, "bedside table", (-1.919, -3.499, -0.002), (-0.726, -0.796, 0.917), 7, 0.4603),
    _rec(5, "bedside table", (1.02, 1.099, -0.002), (2.975, 1.891, 0.689), 2, 0.4976),
    _rec(6, "bedside table", (2.384, -0.33, -0.002), (3.853, 1.533, 0.789), 6, 0.6822),
    _rec(32, "bedside table", (-2.071, 1.292, -0.002), (-0.759, 2.064, 1.214), 6, 0.5846),
    _rec(35, "bedside table", (-1.512, -3.874, -0.002), (-0.348, -1.318, 0.778), 3, 0.4959),
    _rec(36, "bedside table", (-1.64, 4.031, 0.038), (0.434, 4.646, 1.244), 33, 0.4565),
    _rec(37, "bedside table", (0.949, -3.119, 0.158), (1.477, -2.092, 0.705), 4, 0.3968),
    _rec(44, "bedside table", (-1.487, -3.692, -0.002), (-0.669, -3.12, 0.306), 2, 0.3358),
    _rec(45, "bedside table", (-0.967, -5.348, -0.017), (0.669, -2.801, 0.013), 3, 0.3759),
    _rec(46, "bedside table", (1.641, -4.781, 0.411), (2.065, -4.591, 0.808), 2, 0.2433),
    _rec(56, "bedside table", (-1.933, -4.66, 0.426), (-1.752, -4.219, 0.766), 3, 0.2846),
    _rec(59, "bedside table", (3.016, -2.967, 0.831), (3.653, -2.082, 1.299), 2, 0.3402),
    _rec(91, "bedside table", (-4.818, -5.897, -0.002), (-4.415, -4.473, 1.698), 2, 0.1992),
    _rec(114, "bedside table", (-2.071, 3.641, 0.001), (-0.705, 4.646, 1.338), 2, 0.3121),
    _rec(116, "bedside table", (1.058, 3.997, 0.001), (2.694, 4.646, 1.2), 35, 0.6203),
    _rec(123, "bedside table", (-0.203, 4.034, 0.377), (1.281, 4.646, 1.049), 41, 0.5792),
    _rec(124, "bedside table", (1.102, 2.053, 0.411), (1.401, 2.831, 0.639), 30, 0.3265),
    _rec(132, "bedside table", (-0.015, -0.015, -0.015), (0.015, 0.015, 0.015), 2, 0.7241),
    _rec(156, "bedside table", (0.355, 1.979, 0.48), (0.607, 2.063, 0.825), 3, 0.5415),
    _rec(175, "bedside table", (-2.071, 4.061, 0.288), (-0.52, 4.646, 2.84), 56, 0.3543),
]

_HOTELROOM1_WINDOWS = [
    _rec(26, "window", (-1.919, -2.034, 1.128), (-1.829, -0.928, 2.024), 3, 0.2948),
    _rec(28, "window", (-0.882, 4.246, 0.887), (0.349, 4.336, 3.366), 2, 0.3954),
    _rec(29, "window", (1.07, -4.936, 0.182), (2.147, -4.846, 1.032), 3, 0.3264),
    _rec(31, "window", (3.672, -4.46, 0.888), (3.762, -3.16, 1.451), 3, 0.2241),
    _rec(38, "window", (-1.914, -4.732, 1.212), (-1.824, -2.913, 2.444), 6, 0.3534),
    _rec(39, "window", (-1.916, -3.116, 1.119), (-1.826, -1.368, 2.192), 3, 0.3962),
    _rec(40, "window", (-1.391, -0.69, 0.018), (-1.301, 0.542, 1.341), 2, 0.364),
    _rec(41, "window", (-0.543, 4.202, 1.041), (0.688, 4.292, 3.52), 2, 0.2878),
    _rec(43, "window", (-0.26, -5.835, 1.474), (2.219, -5.745, 2.705), 2, 0.1939),
    _rec(47, "window", (-1.939, 0.938, 0.384), (-1.899, 1.578, 1.971), 2, 0.2572),
    _rec(48, "window", (-1.884, -5.843, 0.755), (-0.928, -5.803, 2.801), 3, 0.2651),
    _rec(50, "window", (-1.032, 3.453, 2.702), (1.447, 4.685, 2.792), 2, 0.2135),
    _rec(54, "window", (3.069, -3.012, 1.009), (3.706, -1.932, 1.099), 2, 0.2178),
    _rec(75, "window", (-0.961, 4.62, 1.253), (0.151, 4.66, 2.534), 2, 0.5264),
    _rec(76, "window", (-0.691, 4.314, 1.685), (0.541, 4.404, 3.48), 2, 0.2919),
    _rec(78, "window", (-0.501, -5.981, 1.211), (1.978, -5.891, 2.443), 2, 0.3354),
    _rec(85, "window", (-2.431, -5.922, 0.906), (0.049, -5.832, 2.137), 3, 0.5507),
    _rec(88, "window", (0.917, 1.871, 0.754), (2.148, 1.911, 2.736), 3, 0.3028),
    _rec(89, "window", (1.525, 1.871, 0.785), (2.638, 1.911, 2.875), 4, 0.3511),
    _rec(92, "window", (-4.797, -5.75, 0.046), (-4.707, -4.518, 1.744), 2, 0.4912),
    _rec(93, "window", (-1.934, -4.68, 0.708), (-1.759, -4.64, 0.984), 2, 0.5121),
    _rec(95, "window", (-0.023, 4.62, 1.291), (0.855, 4.66, 2.047), 31, 0.5168),
    _rec(111, "window", (-3.069, -6.007, 0.295), (-2.213, -5.967, 1.103), 2, 0.272),
    _rec(117, "window", (-2.037, 3.069, 0.756), (-1.947, 3.851, 1.298), 2, 0.3322),
    _rec(118, "window", (-2.012, 2.016, 0.491), (-0.585, 2.106, 1.723), 4, 0.2404),
    _rec(122, "window", (2.798, 2.853, 0.357), (2.888, 4.085, 2.836), 2, 0.7168),
    _rec(126, "window", (-2.405, 2.354, 1.694), (-1.173, 2.444, 3.475), 2, 0.3783),
    _rec(129, "window", (0.692, 4.594, 0.692), (1.589, 4.684, 2.006), 4, 0.6223),
    _rec(130, "window", (2.634, 3.664, 0.134), (2.694, 3.904, 0.977), 2, 0.5321),
    _rec(131, "window", (2.908, 1.941, 0.116), (2.998, 3.628, 1.348), 2, 0.435),
    _rec(145, "window", (-2.049, 2.296, 0.483), (-1.959, 3.528, 2.963), 2, 0.3674),
    _rec(148, "window", (0.434, 2.038, 0.216), (0.741, 2.078, 1.299), 7, 0.3371),
    _rec(149, "window", (-2.08, 2.684, 0.624), (-1.99, 3.915, 2.379), 2, 0.4467),
    _rec(150, "window", (-1.629, -0.002, 1.532), (-1.539, 1.23, 3.391), 2, 0.3697),
    _rec(152, "window", (-1.363, 4.582, 1.148), (0.324, 4.672, 2.045), 31, 0.5471),
    _rec(154, "window", (2.714, 2.831, 0.229), (2.804, 4.062, 1.666), 2, 0.4252),
    _rec(157, "window", (-2.12, -5.944, 1.396), (-1.882, -5.854, 2.915), 3, 0.2856),
    _rec(176, "window", (-1.823, -4.266, 2.26), (-1.733, -3.915, 3.253), 28, 0.372),
    _rec(177, "window", (2.544, 3.36, 0.002), (2.634, 4.592, 1.604), 28, 0.5508),
    _rec(178, "window", (2.447, 2.899, -0.082), (3.678, 2.989, 2.397), 56, 0.5242),
]


def test_flowers_near_window_raw_resolve_reproduces_the_live_bug():
    """(#186) The toolbox's own ranking, applied directly with no establishment gate,
    no longer reproduces the archived live pick (instance 0, n_obs=2, a single/few-frame
    ghost) once `_tier_priority_order`'s final tie-break prefers higher `n_obs` over
    ascending `instance_id`: among the many candidates tied at `near()` == 1.0, instance
    5 (n_obs=59) is the most-observed, so it now wins even with NO establishment floor
    applied. This still moves the raw pick off the id-ascending artifact (id 0) -- the
    module docstring's #186 section documents why it does not reach the GT-nearest
    candidate (id 82) either; #186 fixed the id-order tie-break, not `near()`'s own
    AABB-gap saturation."""
    target = TargetSpec(noun="flowers", raw="flowers", attributes=[], clauses=[near_clause("window")])
    sc = scene(*_HOTELROOM2_FLOWERS, *_HOTELROOM2_WINDOWS)
    raw = resolve(target, sc, DEFAULT_THRESHOLDS)
    assert raw.candidates_ranked[0].instance_id == 5
    assert raw.candidates_ranked[0].n_obs == 59  # most-observed among the near()==1.0 tie


def test_flowers_near_window_excludes_the_ghost():
    """(#184 gate + #186 tie-break) Post-fix, ``ObjectRefHead`` no longer publishes the
    n_obs=2 ghost (instance 0): the established view excludes every candidate under
    ``ESTABLISH_N_OBS`` from the ranked pool, and `_tier_priority_order`'s n_obs-descending
    tie-break (#186) picks the most-observed established survivor among the `near()`==1.0
    tie -- instance 5 (n_obs=59), never an id-ascending accident. See the module docstring
    for why this is not the GT-nearest established candidate (id 82): both are genuinely
    established, `near()`-saturated picks, and no per-candidate evidence signal
    distinguishes them without ground truth -- documented as a residual gap, not silently
    papered over."""
    target = TargetSpec(noun="flowers", raw="flowers", attributes=[], clauses=[near_clause("window")])
    winner = _head_winner(
        target, "Find the flowers near the window.", *_HOTELROOM2_FLOWERS, *_HOTELROOM2_WINDOWS
    )
    assert winner.instance_id != 0  # the ghost no longer wins
    assert winner.n_obs >= ESTABLISH_N_OBS  # the winner is always established now
    assert winner.instance_id == 5  # most-observed (n_obs=59) survivor of the near()==1.0 tie


def _farthest_from_window_target(noun: str) -> TargetSpec:
    return TargetSpec(
        noun=noun, raw=noun, attributes=[],
        clauses=[Clause(pred=Pred.FARTHEST_FROM, anchors=[Anchor(noun="window", raw="window")])],
    )


def test_bedside_table_farthest_raw_resolve_reproduces_the_live_bug():
    """(#186) The toolbox's own ranking no longer reproduces the archived live pick
    (instance 116) once `resolve()`'s superlative anchor pick uses `_select_sub_anchor`
    (nearest to the ranked pool > score > n_obs > id) instead of `anchor_recs[0]`
    (index order). Over the FULL raw (unfiltered) window pool the nearest-to-the-
    bedside-table-pool window is id 54 (n_obs=2, itself a low-observation detection --
    the raw/no-establishment path has no floor to exclude it, same as before #184);
    ranking the raw bedside-table pool by distance from that anchor now picks instance
    91."""
    target = _farthest_from_window_target("bedside table")
    sc = scene(*_HOTELROOM1_BEDSIDE_TABLES, *_HOTELROOM1_WINDOWS)
    raw = resolve(target, sc, DEFAULT_THRESHOLDS)
    assert raw.candidates_ranked[0].instance_id == 91
    assert raw.candidates_ranked[0].instance_id != 116  # the old anchor_recs[0] winner


def test_bedside_table_farthest_established_gate_is_a_documented_no_op():
    """(#184 gate + #186 anchor pick) Verified, not assumed: `ObjectRefHead` narrows both
    the candidate and anchor pools to established (n_obs >= ESTABLISH_N_OBS) instances,
    then `resolve()`'s new evidence-based anchor pick (#186, `_select_sub_anchor` instead
    of `anchor_recs[0]`) selects window id 148 (n_obs=7, nearest among established windows
    to the established bedside-table pool) rather than the old index-order pick (id 26).
    Ranking established bedside tables by distance from that anchor now picks instance 45
    (n_obs=3) -- phantom instance 116 no longer wins. id 148 is not itself provably the
    single closest established window to the real GT window position (see the module
    docstring's #186 section): the anchor pool is genuinely noisy detector output, and
    #186 guarantees the pick is evidence-based and deterministic, not that it is
    GT-optimal in every archived row."""
    target = _farthest_from_window_target("bedside table")
    winner = _head_winner(
        target,
        "Find the bedside table farthest from the window.",
        *_HOTELROOM1_BEDSIDE_TABLES,
        *_HOTELROOM1_WINDOWS,
    )
    assert winner.instance_id != 116  # the phantom no longer wins
    assert winner.instance_id == 45
    assert winner.n_obs >= ESTABLISH_N_OBS  # not itself a low-observation ghost


# --------------------------------------------------------------------------- controls


def test_no_clause_is_unaffected_by_the_fix():
    # A question with no relation/superlative clause counts exactly as before -- the
    # established view still applies (per #184's spec: candidate pools too), but with no
    # clause to gate, `resolve()`'s own noun-match ladder is the only thing engaged.
    sc = scene(inst(1, "chair", n_obs=5), inst(2, "chair", n_obs=5), inst(3, "table", n_obs=5))
    plan = object_plan("chair")
    head = ObjectRefHead(plan=plan)
    head.advance(sc)
    assert head.best_candidate is not None
    assert head.best_candidate.label == "chair"


def test_established_pool_empty_falls_open_to_full_pool():
    """(#184 constraint, mirrors #151's own fall-open control) When NO instance of the
    target noun has reached ``ESTABLISH_N_OBS`` yet (cold start), the established view
    would rank nothing -- the head must fall open to the raw (unfiltered-by-establishment)
    ``resolve()`` result rather than publish no marker at all."""
    sc = scene(
        inst(1, "chair", n_obs=1, centroid=(0.0, 0.0, 0.0)),
        inst(2, "chair", n_obs=2, centroid=(5.0, 0.0, 0.0)),
    )
    plan = object_plan("chair")
    head = ObjectRefHead(plan=plan)
    head.advance(sc)
    m = head.verify()
    assert m is not None
    assert head.best_candidate is not None  # never silence on a cold-start pool


def test_already_correct_selection_is_unchanged_by_the_fix():
    """Control: a question the pipeline already answers correctly (no phantom-track
    ambiguity -- every candidate/anchor is already well-established) must not regress
    under the new established-instance gate."""
    sc = scene(
        inst(1, "table", n_obs=6, centroid=(0.0, 0.0, 0.0)),
        inst(2, "table", n_obs=6, centroid=(10.0, 0.0, 0.0)),
        inst(3, "lamp", n_obs=6, centroid=(0.6, 0.0, 0.0), extent=(0.2, 0.2, 0.3)),
    )
    plan = object_plan("table", clauses=[near_clause("lamp")])
    head = ObjectRefHead(plan=plan)
    head.advance(sc)
    assert head.best_candidate.instance_id == 1  # unchanged: the near-lamp table wins


# --------------------------------------------------------------------------- post-#184
# correction: relaxation-aware fall-open (merge-verifier repro)


def test_established_category_only_loses_to_clause_satisfying_raw_candidate():
    """Regression (merge-verifier repro): ``resolve()``'s OWN internal relaxation ladder
    can rescue an EMPTY established-only pool by dropping the clause entirely
    (category_only), silently returning every established candidate UNFILTERED -- none of
    which actually satisfy the clause. Meanwhile an unestablished candidate the
    established view excluded genuinely satisfies it. Pre-#184 (and pre-this-fix), the
    established, clause-FAILING chairs would win outright (their non-empty
    ``candidates_ranked`` short-circuits the #184 fall-open, which only ever checks
    "is it empty"). A clause-failing established candidate must never beat a
    clause-satisfying candidate -- the head must notice the established pool needed a
    severe relaxation and prefer the raw pool's genuinely clause-satisfying survivor."""
    table = inst(10, "table", n_obs=5, centroid=(0.0, 0.0, 0.0))
    far_established_1 = inst(1, "chair", n_obs=5, centroid=(20.0, 0.0, 0.0))
    far_established_2 = inst(2, "chair", n_obs=5, centroid=(25.0, 0.0, 0.0))
    near_unestablished = inst(3, "chair", n_obs=1, centroid=(0.3, 0.0, 0.0))
    sc = scene(table, far_established_1, far_established_2, near_unestablished)

    plan = object_plan("chair", clauses=[near_clause("table")])
    head = ObjectRefHead(plan=plan)
    head.advance(sc)

    assert head.best_candidate is not None
    assert head.best_candidate.instance_id == 3  # the clause-satisfying chair wins
    # the winning pool (raw) is surfaced in the audit trail; the returned audit reflects
    # the WINNING pool's own (empty, unrelaxed) history plus the pool-choice note, not the
    # established pool's internal category_only step -- confirmed separately below.
    assert any(r.step == "established_gate_pool_choice" for r in head._result.audit)
    assert "raw pool selected" in head._result.audit[-1].detail

    # sanity: the established-only pool genuinely DID need category_only (confirms the
    # test fixture actually reproduces the reported mechanism, not a vacuous setup).
    from core.heads.scene_established import EstablishedView

    established_only = EstablishedView(sc, floor=ESTABLISH_N_OBS)
    established_result = resolve(
        object_plan("chair", clauses=[near_clause("table")]).target,
        established_only,
        DEFAULT_THRESHOLDS,
    )
    assert any(r.step == "category_only" for r in established_result.audit)
    assert {c.instance_id for c in established_result.candidates_ranked} == {1, 2}


def test_equal_severity_relaxation_keeps_the_established_pool():
    """Control: when BOTH the established and raw pools need the SAME severity of
    relaxation (here: the anchor class is wholly absent from the scene, so ``near(table)``
    is unevaluable for every candidate in either pool alike, and both bottom out at
    category_only), the established pool still wins -- preserving #184's own intent of
    preferring established tracks when the clause evidence does not distinguish the two
    pools."""
    established_1 = inst(1, "chair", n_obs=5, centroid=(0.0, 0.0, 0.0))
    established_2 = inst(2, "chair", n_obs=5, centroid=(1.0, 0.0, 0.0))
    unestablished = inst(3, "chair", n_obs=1, centroid=(2.0, 0.0, 0.0))
    sc = scene(established_1, established_2, unestablished)  # no 'table' in the scene at all

    plan = object_plan("chair", clauses=[near_clause("table")])
    head = ObjectRefHead(plan=plan)
    head.advance(sc)

    assert head.best_candidate is not None
    assert head.best_candidate.n_obs >= ESTABLISH_N_OBS  # established candidate, not id 3
    assert any(r.step == "established_gate_pool_choice" for r in head._result.audit)
