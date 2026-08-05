"""Issue #151: the numerical head's relation-clause filter passes almost everything
because it is EXISTENTIAL over every resolved anchor instance, and the live anchor
pool is dominated by single-frame detector noise.

Live evidence (job 713413, tree b2b4852, matched against GT) reproduced here
verbatim from ``reports/cluster_verify/713413/debug/*/instance_index.jsonl``'s
final ``answer_time`` record (positions/AABBs/n_obs/score, 3dp, as archived) and
``resolved_plan.jsonl``'s parsed clause for the same question:

* ``9_livingroom_3_nume`` -- "How many photos are on the TV cabinet?" answered 9
  (GT 2). 19 'tv cabinet' instances exist for ONE physical cabinet, scattered
  across the whole room (x in [-1.8, 8.9], y in [-5.9, 5.5]); all but a couple
  have n_obs<=6, most n_obs==2 -- single-frame ghosts, not fragments of the real
  object. ``on(photo, ANY of 19)`` is satisfied by nearly the whole photo class.
* ``8_livingroom_2_nume`` -- "How many cups are on the coffee table?" answered 7
  (GT 2). Same shape: 18 'coffee table' instances, mostly n_obs<=2.
* ``1_chinese_room_nume`` -- "Count the number of chairs with pillows on them."
  answered 10 (GT 6). Same shape on the inverse-support (WITH) predicate: 17
  'pillow' instances, most n_obs<=4.
* ``13_office_2_nume`` -- "How many potted plants are on a table?" answered 1
  (GT 1, exact) -- included as the control the fix must NOT regress: the target
  pool itself is already sparse enough (H15(b) establishment gating) that the
  existing filter already lands on the right answer.

The fix (``core.heads.numerical._AnchorEstablishedView`` /
``NumericalHead._count``) gates the ANCHOR side of a relation clause to
instances seen ``ANCHOR_ESTABLISHED_N_OBS`` times before it may filter, with a
three-tier fall-open documented on ``_count`` so a too-strict floor can only
ever shrink a count toward the truth, never fabricate a zero.
"""
from __future__ import annotations

import numpy as np

from core.heads.numerical import ANCHOR_ESTABLISHED_N_OBS, NumericalHead
from core.geometry.toolbox import DEFAULT_THRESHOLDS, counting
from core.interfaces import InstanceRecord
from core.plan_schema import Anchor, Clause, Plan, Pred, QType, TargetSpec
from tests.heads._helpers import inst, near_clause, numerical_plan, scene


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


def _clause_plan(qtype: QType, noun: str, pred: Pred, anchor_noun: str) -> Plan:
    clause = Clause(pred=pred, anchors=[Anchor(noun=anchor_noun)])
    return Plan(
        qtype=qtype,
        question_raw=f"how many {noun} {pred.value} {anchor_noun}",
        target=TargetSpec(noun=noun, clauses=[clause]),
    )


# --------------------------------------------------------------------------- #151
# archived-data fixtures (verbatim aabb_min/aabb_max/n_obs/score, 3dp)

_LIVINGROOM3_PHOTOS = [
    _rec(3, "picture", (-0.785, 1.902, 1.371), (-0.754, 2.75, 2.133), 2, 0.6428),
    _rec(5, "picture", (0.931, 1.635, 1.242), (0.963, 2.482, 2.004), 3, 0.2944),
    _rec(9, "picture", (9.026, -1.681, 1.405), (9.057, -0.833, 2.167), 2, 0.1897),
    _rec(22, "picture", (-1.694, -2.284, 1.65), (-1.662, -1.437, 2.412), 2, 0.592),
    _rec(24, "picture", (0.849, 3.606, 2.21), (0.88, 4.454, 2.972), 2, 0.285),
    _rec(46, "picture", (-0.01, -0.01, -0.01), (0.01, 0.01, 0.01), 2, 0.5714),
    _rec(48, "picture", (1.58, 0.689, 0.648), (2.428, 0.72, 1.41), 35, 0.6261),
    _rec(50, "picture", (2.209, -1.619, 0.743), (2.589, -1.588, 0.991), 2, 0.2252),
    _rec(59, "picture", (4.889, -0.441, 1.265), (5.737, 0.321, 1.297), 2, 0.5858),
    _rec(62, "picture", (9.013, -1.865, 1.786), (9.045, -1.017, 2.548), 70, 0.3889),
    _rec(63, "picture", (9.009, -1.475, 1.162), (9.04, -0.627, 1.924), 2, 0.3796),
    _rec(64, "picture", (9.017, -1.242, 1.801), (9.049, -0.394, 2.563), 35, 0.3825),
    _rec(65, "picture", (9.019, -2.174, 1.854), (9.05, -1.327, 2.616), 2, 0.2246),
    _rec(66, "picture", (9.019, -2.326, 1.11), (9.051, -1.564, 1.958), 2, 0.3907),
    _rec(73, "picture", (4.127, -6.128, 0.762), (4.975, -5.497, 0.794), 2, 0.3172),
    _rec(74, "picture", (4.524, -6.071, 0.678), (5.372, -6.039, 1.44), 2, 0.4914),
    _rec(77, "picture", (9.017, -2.156, 1.395), (9.048, -1.394, 2.242), 35, 0.222),
    _rec(96, "picture", (-0.246, -3.366, 0.43), (-0.214, -2.761, 1.25), 36, 0.5522),
    _rec(116, "picture", (2.381, -1.55, 0.234), (3.228, -1.518, 0.996), 34, 0.5117),
    _rec(118, "picture", (7.292, 0.753, 1.947), (7.487, 0.785, 2.512), 34, 0.5231),
    _rec(122, "picture", (-0.548, -6.158, 1.575), (0.214, -6.126, 2.423), 33, 0.2734),
    _rec(123, "picture", (1.127, 0.553, 0.329), (1.948, 0.584, 0.9), 25, 0.1816),
    _rec(124, "picture", (4.58, 0.722, 1.462), (5.428, 0.753, 2.224), 33, 0.1891),
    _rec(125, "picture", (9.012, -1.458, 1.623), (9.044, -0.61, 2.029), 33, 0.2058),
]
_LIVINGROOM3_TV_CABINETS = [
    _rec(12, "tv cabinet", (-1.778, -3.982, 0.938), (-1.474, 1.782, 2.959), 6, 0.3618),
    _rec(13, "tv cabinet", (-1.778, -0.256, 0.964), (-1.739, 2.069, 1.865), 2, 0.2346),
    _rec(16, "tv cabinet", (-1.265, 5.491, 0.057), (0.419, 5.536, 2.697), 2, 0.2821),
    _rec(17, "tv cabinet", (0.888, -6.161, 0.0), (3.65, -4.411, 0.85), 2, 0.2791),
    _rec(18, "tv cabinet", (-1.02, -6.161, 0.0), (4.607, -3.577, 2.959), 2, 0.2106),
    _rec(20, "tv cabinet", (0.538, -6.161, 0.0), (4.579, -3.683, 2.959), 5, 0.2504),
    _rec(21, "tv cabinet", (1.909, -6.161, 1.154), (5.59, -5.598, 2.959), 40, 0.5995),
    _rec(33, "tv cabinet", (3.836, -6.161, 0.302), (5.398, -5.53, 0.801), 3, 0.5673),
    _rec(34, "tv cabinet", (3.016, -6.202, 1.316), (7.691, -4.526, 2.959), 2, 0.4829),
    _rec(55, "tv cabinet", (-1.778, -1.174, 0.0), (1.254, 0.791, 0.316), 2, 0.4673),
    _rec(56, "tv cabinet", (1.552, -2.61, 0.533), (2.03, -1.849, 0.846), 2, 0.2724),
    _rec(57, "tv cabinet", (2.546, -0.342, 0.0), (5.612, 0.772, 1.111), 35, 0.8053),
    _rec(68, "tv cabinet", (1.553, -6.161, 0.0), (4.877, -3.544, 0.983), 3, 0.2273),
    _rec(72, "tv cabinet", (8.713, -2.416, 0.984), (9.056, -0.333, 2.959), 2, 0.2996),
    _rec(79, "tv cabinet", (5.818, -4.564, 0.586), (5.983, -3.677, 0.854), 2, 0.5295),
    _rec(114, "tv cabinet", (4.733, -0.393, 0.0), (6.654, 0.778, 1.241), 2, 0.4764),
    _rec(126, "tv cabinet", (-1.778, -5.503, 0.0), (0.253, 1.255, 2.706), 33, 0.371),
    _rec(127, "tv cabinet", (0.486, -3.41, 0.0), (2.071, -1.098, 0.631), 33, 0.2075),
    _rec(128, "tv cabinet", (2.039, -1.602, 0.108), (4.208, -1.494, 0.965), 33, 0.1962),
]


def test_livingroom3_photos_on_tv_cabinet_moves_toward_gt():
    sc = scene(*_LIVINGROOM3_PHOTOS, *_LIVINGROOM3_TV_CABINETS)
    plan = _clause_plan(QType.NUMERICAL, "photo", Pred.ON, "tv cabinet")

    # Reproduce the live overcount: the toolbox's own clause filter, applied
    # directly with no establishment gate, already lands on the archived 9.
    raw = counting(plan.target, sc, min_obs=1, th=DEFAULT_THRESHOLDS)
    assert raw.count == 9  # matches the live answer (GT 2)

    head = NumericalHead(plan=plan)
    head.advance(sc)
    assert head.count < raw.count  # #151 fix: materially fewer than the overcount
    assert head.count > 0  # never a fabricated 0
    assert head.count <= 4  # within reach of GT=2, not still ~the whole class


def test_livingroom2_cups_on_coffee_table_moves_toward_gt():
    cups = [
        _rec(8, "cup", (0.079, -2.39, 0.47), (0.245, -2.242, 0.524), 3, 0.4457),
        _rec(10, "cup", (0.607, 4.163, 0.79), (0.774, 4.294, 0.939), 3, 0.4107),
        _rec(11, "cup", (0.782, 0.333, 0.76), (0.833, 0.499, 0.854), 2, 0.1888),
        _rec(12, "cup", (2.468, -5.198, 0.236), (2.617, -5.067, 0.403), 2, 0.5635),
        _rec(37, "cup", (-0.63, 4.75, 1.193), (-0.464, 4.835, 1.342), 2, 0.2899),
        _rec(39, "cup", (0.619, 0.975, 0.877), (0.785, 1.123, 1.003), 2, 0.2311),
        _rec(60, "cup", (2.768, 4.558, 1.305), (2.899, 4.707, 1.471), 3, 0.264),
        _rec(61, "cup", (2.787, -2.075, 2.106), (2.917, -1.909, 2.255), 2, 0.2857),
        _rec(62, "cup", (2.869, -2.312, 0.989), (2.954, -2.145, 1.138), 2, 0.2786),
        _rec(66, "cup", (0.454, 1.323, 0.695), (0.487, 1.472, 0.861), 4, 0.4694),
        _rec(82, "cup", (0.047, -3.206, 0.448), (0.177, -3.04, 0.597), 2, 0.3589),
        _rec(105, "cup", (0.24, -7.402, 0.267), (0.389, -7.235, 0.397), 35, 0.5833),
        _rec(108, "cup", (1.527, -6.936, 0.437), (1.694, -6.903, 0.586), 3, 0.65),
    ]
    tables = [
        _rec(1, "coffee table", (-1.935, 1.901, -0.004), (-0.866, 4.806, 1.09), 2, 0.3752),
        _rec(3, "coffee table", (-0.001, -2.419, 0.384), (0.726, -2.289, 0.524), 2, 0.1988),
        _rec(5, "coffee table", (0.772, -0.658, -0.004), (2.912, 1.151, 0.823), 4, 0.6574),
        _rec(6, "coffee table", (1.527, -3.648, -0.004), (2.912, -1.042, 0.995), 2, 0.307),
        _rec(17, "coffee table", (-1.964, 1.043, -0.004), (-1.341, 2.514, 0.949), 2, 0.365),
        _rec(18, "coffee table", (0.391, 0.493, -0.004), (2.912, 1.471, 1.219), 5, 0.5773),
        _rec(29, "coffee table", (-1.935, 3.295, -0.004), (-0.764, 4.806, 1.11), 3, 0.5091),
        _rec(30, "coffee table", (-1.362, -5.194, -0.004), (0.172, -1.107, 1.319), 2, 0.5335),
        _rec(31, "coffee table", (2.295, -4.583, -0.003), (2.571, -4.285, 0.331), 2, 0.7607),
        _rec(56, "coffee table", (2.897, -0.154, 0.02), (2.927, 0.88, 0.884), 2, 0.2655),
        _rec(64, "coffee table", (0.457, 0.999, 0.305), (0.487, 1.495, 0.668), 2, 0.3082),
        _rec(78, "coffee table", (-0.472, -4.381, -0.002), (1.099, -4.015, 0.036), 2, 0.7445),
        _rec(79, "coffee table", (0.773, 0.466, 0.635), (0.801, 0.552, 0.847), 2, 0.499),
        _rec(99, "coffee table", (-0.015, -0.015, -0.015), (0.015, 0.015, 0.015), 14, 0.2424),
        _rec(106, "coffee table", (1.254, -6.943, 0.334), (2.013, -6.791, 0.77), 12, 0.7246),
        _rec(112, "coffee table", (-2.337, -9.927, -0.022), (1.187, -8.896, 1.567), 12, 0.471),
        _rec(121, "coffee table", (-1.276, -8.132, -0.022), (0.448, -6.489, 0.478), 14, 0.6678),
        _rec(122, "coffee table", (1.892, -7.97, 0.686), (2.032, -7.836, 0.815), 11, 0.2689),
    ]
    sc = scene(*cups, *tables)
    plan = _clause_plan(QType.NUMERICAL, "cup", Pred.ON, "coffee table")

    raw = counting(plan.target, sc, min_obs=1, th=DEFAULT_THRESHOLDS)
    assert raw.count == 7  # matches the live answer (GT 2)

    head = NumericalHead(plan=plan)
    head.advance(sc)
    assert head.count < raw.count
    assert head.count > 0
    assert head.count <= 4


def test_chinese_room_chairs_with_pillows_moves_toward_gt():
    chairs = [
        _rec(3, "chair", (-0.424, -3.976, -0.008), (0.733, -3.166, 0.864), 2, 0.6183),
        _rec(5, "chair", (6.073, 1.686, 0.359), (6.497, 2.136, 1.391), 90, 0.5801),
        _rec(27, "chair", (-2.297, 0.997, -0.01), (-1.78, 2.046, 0.831), 4, 0.4469),
        _rec(38, "chair", (0.508, -4.679, 0.633), (2.022, -4.646, 1.068), 2, 0.5541),
        _rec(44, "chair", (1.454, -2.197, 0.042), (2.328, -0.626, 1.02), 2, 0.2251),
        _rec(67, "chair", (1.835, -0.551, 0.614), (2.088, -0.199, 0.798), 4, 0.488),
        _rec(68, "chair", (2.9, -0.952, 0.009), (4.471, 0.026, 0.826), 2, 0.5764),
        _rec(69, "chair", (3.439, 0.139, -0.098), (5.009, 0.886, 0.713), 3, 0.7626),
        _rec(70, "chair", (6.068, -0.585, -0.01), (6.497, 0.095, 1.003), 3, 0.2239),
        _rec(80, "chair", (5.765, -3.811, -0.01), (6.497, -2.965, 0.72), 2, 0.2528),
        _rec(100, "chair", (4.479, 0.016, 0.739), (4.797, 0.631, 0.822), 99, 0.6591),
        _rec(101, "chair", (4.642, -3.125, -0.251), (5.519, -1.554, 0.231), 100, 0.359),
        _rec(102, "chair", (4.445, -0.703, 0.452), (6.016, -0.084, 0.817), 50, 0.4321),
        _rec(103, "chair", (5.92, -3.218, -0.086), (6.618, -1.648, 0.686), 22, 0.182),
        _rec(104, "chair", (3.852, -4.809, -0.01), (4.762, -4.55, 1.537), 50, 0.2702),
        _rec(126, "chair", (5.984, -3.907, -0.01), (6.497, -1.822, 1.224), 1, 0.1842),
    ]
    pillows = [
        _rec(22, "pillow", (1.257, -1.63, 0.471), (2.049, -1.048, 0.835), 3, 0.4468),
        _rec(24, "pillow", (1.86, -3.108, 0.47), (2.224, -2.316, 1.052), 3, 0.4549),
        _rec(30, "pillow", (-2.297, 1.124, 0.633), (-2.267, 1.594, 1.014), 2, 0.3027),
        _rec(31, "pillow", (1.839, -0.982, 0.669), (2.254, -0.461, 0.955), 4, 0.3658),
        _rec(32, "pillow", (2.16, -1.3, 0.877), (2.347, -0.856, 0.998), 2, 0.3715),
        _rec(34, "pillow", (5.945, 1.87, 0.134), (6.527, 2.2, 0.926), 2, 0.3925),
        _rec(40, "pillow", (1.098, -4.677, 0.341), (1.89, -4.643, 0.923), 2, 0.4426),
        _rec(55, "pillow", (1.57, -1.584, 0.537), (2.362, -1.002, 0.901), 2, 0.4551),
        _rec(56, "pillow", (1.88, -2.615, 0.484), (2.244, -1.823, 1.066), 2, 0.4674),
        _rec(57, "pillow", (1.944, -1.479, 0.45), (2.309, -0.814, 1.002), 3, 0.4191),
        _rec(64, "pillow", (6.108, 1.721, 0.773), (6.472, 2.125, 1.33), 51, 0.2992),
        _rec(65, "pillow", (6.275, 1.446, 0.126), (6.639, 2.028, 0.918), 2, 0.4431),
        _rec(73, "pillow", (3.961, 0.482, 0.313), (4.468, 0.725, 0.745), 3, 0.4638),
        _rec(105, "pillow", (4.515, 0.355, 0.748), (4.537, 0.466, 0.777), 50, 0.2963),
        _rec(106, "pillow", (4.763, -0.676, 0.441), (5.457, -0.228, 0.724), 51, 0.5105),
        _rec(107, "pillow", (6.193, -2.724, 0.551), (6.557, -1.932, 1.133), 50, 0.2252),
        _rec(108, "pillow", (5.393, -0.71, 0.67), (5.489, -0.424, 0.817), 49, 0.202),
    ]
    sc = scene(*chairs, *pillows)
    plan = _clause_plan(QType.NUMERICAL, "chair", Pred.WITH, "pillow")

    raw = counting(plan.target, sc, min_obs=1, th=DEFAULT_THRESHOLDS)
    assert raw.count in (9, 10)  # matches the live answer band (GT 6)

    head = NumericalHead(plan=plan)
    head.advance(sc)
    assert head.count < raw.count
    assert head.count > 0
    assert head.count <= 8  # closer to GT=6 than the raw overcount


def test_office2_potted_plants_on_table_stays_exact():
    # Control (#151): the ONE archived case that already answers correctly must
    # not regress under the new anchor-establishment gate.
    plants = [
        _rec(36, "potted plant", (-0.163, 1.998, 0.12), (0.149, 2.437, 0.974), 1, 0.4927),
        _rec(37, "potted plant", (2.992, -2.835, 0.067), (4.187, -2.81, 0.384), 1, 0.1913),
        _rec(38, "potted plant", (3.165, -2.835, 0.413), (4.971, -2.791, 1.471), 1, 0.2237),
        _rec(56, "potted plant", (2.364, -0.741, -0.003), (3.153, -0.583, 0.546), 1, 0.3003),
        _rec(57, "potted plant", (3.109, -0.617, 0.689), (3.646, -0.337, 1.08), 1, 0.1871),
        _rec(63, "potted plant", (4.132, -0.786, 0.68), (5.17, -0.756, 0.989), 4, 0.5686),
    ]
    tables = [
        _rec(17, "table", (-1.17, -0.936, 0.657), (-0.212, -0.614, 1.588), 8, 0.261),
        _rec(18, "table", (0.165, 1.205, -0.01), (1.354, 3.099, 0.749), 2, 0.7398),
        _rec(19, "table", (2.501, -0.741, -0.003), (3.666, -0.198, 0.941), 6, 0.3812),
        _rec(20, "table", (2.345, 1.029, -0.051), (4.116, 2.218, 0.708), 5, 0.2119),
        _rec(22, "table", (5.301, 2.471, -0.003), (5.586, 3.502, 0.753), 10, 0.4824),
        _rec(24, "table", (-1.332, -0.207, 0.405), (-1.292, 0.907, 1.713), 4, 0.2139),
        _rec(25, "table", (-1.318, 0.858, 0.175), (-1.292, 1.175, 0.313), 4, 0.2919),
        _rec(26, "table", (0.391, 1.09, 0.026), (1.581, 2.984, 0.785), 9, 0.396),
        _rec(27, "table", (-0.109, 1.74, -0.025), (1.08, 3.635, 0.734), 3, 0.4843),
        _rec(39, "table", (-1.337, 0.666, 0.321), (-1.292, 1.456, 1.607), 1, 0.1988),
        _rec(40, "table", (-1.318, -0.775, 0.82), (-1.084, -0.076, 1.682), 1, 0.2689),
        _rec(41, "table", (-0.215, 3.158, -0.003), (0.311, 4.092, 0.905), 1, 0.2619),
        _rec(42, "table", (1.262, 0.546, -0.221), (3.156, 1.736, 0.538), 2, 0.2271),
        _rec(43, "table", (2.462, -0.546, -0.003), (5.105, 0.177, 0.953), 1, 0.6526),
        _rec(44, "table", (2.797, -2.835, -0.003), (5.909, -2.472, 1.121), 1, 0.5715),
        _rec(59, "table", (-1.024, -2.197, 0.265), (-0.673, -1.224, 1.725), 2, 0.3748),
        _rec(60, "table", (2.221, -1.168, 0.05), (3.268, -0.36, 0.809), 3, 0.4211),
        _rec(61, "table", (2.94, 1.389, 0.207), (4.004, 2.402, 0.889), 2, 0.5154),
        _rec(62, "table", (5.035, 0.953, 0.073), (6.225, 2.847, 0.832), 2, 0.3093),
        _rec(64, "table", (3.381, -0.932, 0.301), (5.275, -0.61, 0.845), 3, 0.3713),
        _rec(65, "table", (4.962, -0.932, 0.436), (5.248, -0.61, 0.809), 1, 0.5373),
        _rec(66, "table", (5.628, -2.996, 0.648), (5.759, -2.674, 0.68), 1, 0.3596),
        _rec(67, "table", (5.891, -1.972, 0.234), (6.213, -1.326, 1.092), 2, 0.2644),
    ]
    sc = scene(*plants, *tables)
    plan = _clause_plan(QType.NUMERICAL, "potted plant", Pred.ON, "table")

    head = NumericalHead(plan=plan)
    head.advance(sc)
    assert head.count == 1


# --------------------------------------------------------------------------- fallback safety


def test_no_clause_is_unaffected_by_the_fix():
    # (#151 constraint) A question with NO relation clause must count exactly as
    # today -- the established-anchor view is only constructed when a clause is
    # present, so this must be byte-identical to a plain toolbox.counting() call.
    sc = scene(inst(1, "chair"), inst(2, "chair"), inst(3, "table"))
    plan = numerical_plan("chair")
    baseline = counting(plan.target, sc, min_obs=1, th=DEFAULT_THRESHOLDS)

    head = NumericalHead(plan=plan)
    head.advance(sc)
    assert head.count == baseline.count == 2


def test_anchor_wholly_absent_falls_back_to_unfiltered_not_zero():
    # (#151 constraint) An anchor class not present in the scene at all makes the
    # clause unevaluable -- the head must fall back to the unfiltered target
    # count, never fabricate a 0 (the pre-#151 undercount-by-overfiltering
    # failure mode this issue exists to close).
    sc = scene(inst(1, "chair"), inst(2, "chair"), inst(3, "chair"))
    plan = numerical_plan("chair", clauses=[near_clause("table")])  # no table at all

    head = NumericalHead(plan=plan)
    head.advance(sc)
    assert head.count == 3  # falls back to the full chair census, not 0


def test_cold_start_anchor_falls_back_to_raw_clause_result():
    # The anchor exists but hasn't been established yet (n_obs below
    # ANCHOR_ESTABLISHED_N_OBS) -- the established view drops it, so the head
    # must fall back to the toolbox's plain (unfiltered-by-establishment) clause
    # result rather than treat "not yet established" as "absent".
    assert ANCHOR_ESTABLISHED_N_OBS >= 2  # sanity: the test needs headroom below it
    sc = scene(
        inst(1, "chair", centroid=(0.0, 0.0, 0.0)),
        inst(2, "chair", centroid=(10.0, 10.0, 0.0)),
        inst(3, "table", centroid=(0.6, 0.0, 0.0), n_obs=1),  # under-observed
    )
    plan = numerical_plan("chair", clauses=[near_clause("table")])

    head = NumericalHead(plan=plan)
    head.advance(sc)
    assert head.count == 1  # the near-table chair, via the raw-clause fallback tier


def test_established_anchor_filters_normally_when_evidence_is_solid():
    # Control: when the anchor IS well-established, the fix's filtered tier
    # applies directly (no fallback needed) and narrows the count as before.
    sc = scene(
        inst(1, "chair", centroid=(0.0, 0.0, 0.0)),
        inst(2, "chair", centroid=(10.0, 10.0, 0.0)),
        inst(3, "table", centroid=(0.6, 0.0, 0.0), n_obs=ANCHOR_ESTABLISHED_N_OBS),
    )
    plan = numerical_plan("chair", clauses=[near_clause("table")])

    head = NumericalHead(plan=plan)
    head.advance(sc)
    assert head.count == 1
