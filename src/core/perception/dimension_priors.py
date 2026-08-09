"""Per-class dimension priors — the "dimension sanity table" (red-team OR-F6 / H12).

Real single-viewpoint perception produces AABBs that are *under*-approximations of
the ground-truth hull: only the observed faces carry lidar points, and the trimmed
2nd/98th-percentile box shaves a further sliver per axis. Against the GT hull (an
axis-aligned *over*-approximation of an oriented box) this compounds into IoU
cliffs at 0.25 / 0.5, costing 2-pt object-reference wins even when the *instance*
selected is correct.

This module supplies the missing clamp: per-class minimum and typical extents, and
:func:`clamp_extents` / :func:`clamp_record_marker`, which
  * clamp every axis up to at least the class minimum (so a paper-thin front-shell
    box can never fall below a plausible object size), and
  * inflate the single least-observed axis toward the class-typical value, but ONLY
    when the instance signals under-observation (few observations AND a thin
    least-axis relative to the prior). A well-observed, near-typical box is left
    exactly as measured, so a GT-perfect box passes through UNCHANGED.

Extents are stored orientation-invariantly as the sorted (thin, mid, long) axis
lengths, so a class prior applies regardless of how the instance's AABB happens to
be axis-aligned in the map frame. Clamping sorts the instance extents, adjusts the
sorted values, then maps the deltas back onto the original axes.

Provenance
----------
``_DATA_PRIORS`` was generated offline from the VLA-3D Unity release (all 15 scenes'
``*_object_result.csv`` per-instance bbox extents) by :func:`_generate_data_priors`
(run ``python -m core.perception.dimension_priors`` to regenerate). Each entry is the
per-sorted-axis 10th-percentile extent (min) and median extent (typical), aggregated
over every instance whose ``raw_label`` folds onto that challenge-vocabulary class
(exact canonical, object bridge, or head-noun match). The trailing ``# n=..,
scenes=..`` comment on each row is the sample support.

``_HAND_PRIORS`` covers the seven challenge-vocabulary classes with no VLA-3D
instances (all small decorative/figurine-scale items); values are hand-curated order-
of-magnitude estimates, deliberately conservative (small mins so the clamp never
inflates a real detection past plausibility).
"""
from __future__ import annotations

import numpy as np

from core.interfaces import InstanceRecord, MarkerBox
from core.perception.scene_index import normalize_label

# --------------------------------------------------------------------------- priors

# Sorted-axis (thin, mid, long) extents in metres: (class-min[10th pct], class-typical[median]).
# DATA-DERIVED — VLA-3D Unity, 15 scenes, per-instance bbox extents. See module docstring
# and _generate_data_priors (the __main__ regenerator). Do not hand-edit rows; regenerate.
_DATA_PRIORS: dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]] = {
    "ball": ((0.104, 0.105, 0.106), (0.125, 0.127, 0.128)),  # n=4, scenes=2
    "bed": ((0.356, 1.974, 2.109), (1.165, 2.589, 3.101)),  # n=5, scenes=4
    "bench": ((0.377, 0.493, 1.378), (0.432, 0.618, 1.684)),  # n=6, scenes=4
    "book": ((0.027, 0.153, 0.235), (0.041, 0.179, 0.256)),  # n=68, scenes=12
    "bookcase": ((0.382, 1.134, 2.080), (0.433, 1.514, 2.367)),  # n=2, scenes=2
    "bottle": ((0.075, 0.075, 0.208), (0.123, 0.128, 0.324)),  # n=54, scenes=9
    "bowl": ((0.092, 0.158, 0.159), (0.115, 0.252, 0.259)),  # n=13, scenes=5
    "box": ((0.141, 0.202, 0.283), (0.224, 0.314, 0.405)),  # n=20, scenes=5
    "cabinet": ((0.427, 0.588, 1.004), (0.538, 0.829, 1.982)),  # n=25, scenes=13
    "can": ((0.276, 0.280, 0.391), (0.322, 0.326, 0.397)),  # n=6, scenes=4
    "candle": ((0.063, 0.066, 0.087), (0.074, 0.076, 0.115)),  # n=12, scenes=3
    "chair": ((0.482, 0.539, 0.801), (0.583, 0.652, 1.047)),  # n=84, scenes=13
    "clock": ((0.038, 0.163, 0.193), (0.060, 0.274, 0.307)),  # n=10, scenes=7
    "column": ((0.250, 0.250, 2.463), (0.491, 0.499, 2.734)),  # n=18, scenes=5
    "counter": ((1.056, 1.608, 2.670), (1.056, 1.608, 2.670)),  # n=1, scenes=1
    "cup": ((0.085, 0.092, 0.101), (0.087, 0.099, 0.111)),  # n=26, scenes=5
    "curtain": ((0.072, 0.806, 2.367), (0.253, 1.293, 3.481)),  # n=21, scenes=6
    "decal": ((0.005, 1.290, 1.903), (0.023, 1.829, 2.343)),  # n=2, scenes=2
    "decoration": ((0.057, 0.179, 0.210), (0.118, 0.220, 0.319)),  # n=15, scenes=8
    "door": ((0.041, 0.527, 1.572), (0.167, 0.976, 2.194)),  # n=68, scenes=15
    "easel": ((0.658, 0.683, 2.139), (0.658, 0.683, 2.139)),  # n=1, scenes=1
    "figurine": ((0.141, 0.247, 0.261), (0.142, 0.320, 0.324)),  # n=3, scenes=2
    "fireplace": ((0.070, 0.482, 1.127), (0.080, 0.550, 1.287)),  # n=3, scenes=3
    "floor": ((0.017, 2.750, 6.368), (0.067, 6.682, 8.097)),  # n=19, scenes=15
    "flower": ((0.080, 0.125, 0.389), (0.157, 0.221, 0.474)),  # n=8, scenes=5
    "folder": ((0.063, 0.255, 0.342), (0.160, 0.297, 0.679)),  # n=10, scenes=2
    "frame": ((0.120, 0.955, 1.721), (0.153, 1.159, 2.460)),  # n=49, scenes=14
    "guitar": ((0.509, 0.673, 1.261), (0.509, 0.673, 1.261)),  # n=1, scenes=1
    "holder": ((0.147, 0.163, 0.220), (0.181, 0.200, 0.237)),  # n=4, scenes=3
    "hookah": ((0.096, 0.098, 0.407), (0.096, 0.098, 0.407)),  # n=1, scenes=1
    "jar": ((0.073, 0.077, 0.185), (0.073, 0.077, 0.185)),  # n=11, scenes=3
    "kettle": ((0.196, 0.233, 0.306), (0.204, 0.244, 0.346)),  # n=4, scenes=2
    "knife": ((0.021, 0.084, 0.235), (0.022, 0.120, 0.243)),  # n=5, scenes=1
    "lamp": ((0.101, 0.200, 0.348), (0.325, 0.381, 0.863)),  # n=73, scenes=15
    "lantern": ((0.128, 0.128, 0.374), (0.162, 0.162, 0.421)),  # n=5, scenes=2
    "ledge": ((0.180, 0.894, 2.256), (0.180, 0.894, 2.256)),  # n=1, scenes=1
    "magazine": ((0.020, 0.223, 0.333), (0.041, 0.279, 0.337)),  # n=4, scenes=3
    "map": ((0.013, 0.161, 0.245), (0.027, 0.241, 0.339)),  # n=2, scenes=1
    "microwave": ((0.098, 0.444, 0.580), (0.337, 0.549, 0.772)),  # n=3, scenes=2
    "mirror": ((0.023, 0.572, 0.653), (0.040, 0.799, 0.988)),  # n=9, scenes=7
    "monitor": ((0.150, 0.469, 0.493), (0.193, 0.491, 0.493)),  # n=9, scenes=2
    "nightstand": ((0.350, 0.398, 0.661), (0.628, 0.664, 0.811)),  # n=12, scenes=4
    "ottoman": ((0.453, 0.506, 0.592), (0.494, 0.749, 1.034)),  # n=2, scenes=2
    "painting": ((0.025, 0.509, 0.699), (0.038, 0.765, 1.132)),  # n=17, scenes=7
    "phone": ((0.007, 0.072, 0.151), (0.060, 0.142, 0.219)),  # n=10, scenes=7
    "picture": ((0.012, 0.199, 0.280), (0.021, 0.508, 0.565)),  # n=48, scenes=10
    "pillow": ((0.187, 0.324, 0.409), (0.243, 0.388, 0.528)),  # n=94, scenes=13
    "plant": ((0.171, 0.199, 0.210), (0.642, 0.730, 1.008)),  # n=57, scenes=13
    "rack": ((0.151, 0.273, 0.468), (0.274, 0.453, 0.994)),  # n=5, scenes=4
    "record": ((0.040, 0.673, 0.806), (0.040, 0.673, 0.806)),  # n=3, scenes=1
    "refrigerator": ((0.759, 0.946, 2.212), (0.788, 1.195, 2.361)),  # n=2, scenes=2
    "remote": ((0.019, 0.051, 0.201), (0.019, 0.051, 0.204)),  # n=6, scenes=5
    "screen": ((0.169, 1.554, 1.994), (0.215, 1.620, 2.288)),  # n=2, scenes=2
    "shelf": ((0.152, 0.477, 1.210), (0.400, 0.993, 2.389)),  # n=7, scenes=6
    "sign": ((0.006, 0.111, 0.282), (0.026, 0.138, 0.307)),  # n=2, scenes=2
    "sofa": ((0.678, 0.789, 1.022), (0.860, 1.017, 2.122)),  # n=24, scenes=10
    "speaker": ((0.106, 0.153, 0.295), (0.173, 0.238, 0.579)),  # n=12, scenes=3
    "stair": ((0.273, 0.757, 2.687), (0.943, 1.547, 3.235)),  # n=2, scenes=2
    "stool": ((0.363, 0.364, 0.682), (0.388, 0.388, 0.845)),  # n=8, scenes=4
    "suitcase": ((0.265, 0.507, 0.826), (0.265, 0.507, 0.826)),  # n=1, scenes=1
    "sushi": ((0.024, 0.030, 0.052), (0.027, 0.051, 0.052)),  # n=3, scenes=1
    "table": ((0.322, 0.479, 0.549), (0.506, 0.793, 1.263)),  # n=38, scenes=14
    "television": ((0.039, 0.626, 0.978), (0.060, 0.791, 1.262)),  # n=13, scenes=9
    "tray": ((0.010, 0.201, 0.205), (0.014, 0.370, 0.371)),  # n=5, scenes=5
    "vase": ((0.113, 0.130, 0.180), (0.176, 0.217, 0.347)),  # n=32, scenes=10
    "wall": ((0.070, 0.974, 3.287), (0.132, 2.786, 6.599)),  # n=84, scenes=15
    "wardrobe": ((0.503, 1.412, 2.034), (0.642, 2.523, 3.001)),  # n=5, scenes=4
    "whiteboard": ((0.164, 1.027, 1.779), (0.164, 1.027, 1.779)),  # n=1, scenes=1
    "window": ((0.040, 0.642, 0.882), (0.060, 0.821, 1.653)),  # n=43, scenes=14
}

# HAND-CURATED fallback for challenge-vocabulary classes absent from the VLA-3D data
# (all small decorative / figurine-scale items). Conservative order-of-magnitude
# estimates; sorted (thin, mid, long) as (min, typical). No provenance beyond common
# object sizes — kept intentionally small so the clamp never over-inflates.
_HAND_PRIORS: dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]] = {
    "elephant": ((0.08, 0.15, 0.20), (0.15, 0.25, 0.35)),   # figurine / decorative
    "fan": ((0.10, 0.30, 0.35), (0.20, 0.45, 0.55)),        # table/standing fan
    "fossil": ((0.05, 0.15, 0.20), (0.10, 0.28, 0.32)),     # decorative fossil
    "horse": ((0.08, 0.15, 0.25), (0.15, 0.28, 0.40)),      # figurine
    "projector": ((0.10, 0.25, 0.30), (0.15, 0.30, 0.38)),  # ceiling/desk projector
    "pyramid": ((0.08, 0.10, 0.12), (0.15, 0.18, 0.20)),    # decorative pyramid
    "sphere": ((0.10, 0.10, 0.10), (0.18, 0.18, 0.18)),     # decorative sphere/orb
}


class ClassPrior:
    """Sorted-axis (thin, mid, long) minimum and typical extents for a class."""

    __slots__ = ("min_ext", "typ_ext")

    def __init__(self, min_ext: tuple[float, float, float], typ_ext: tuple[float, float, float]) -> None:
        self.min_ext = np.asarray(min_ext, dtype=float)
        self.typ_ext = np.asarray(typ_ext, dtype=float)


_PRIOR_CACHE: dict[str, ClassPrior] = {}


def prior_for(label: str) -> ClassPrior | None:
    """Return the :class:`ClassPrior` for a label, or None if no prior exists.

    The label is canonicalised through :func:`normalize_label`; data priors win over
    hand priors on the (currently empty) intersection.
    """
    key = normalize_label(label)
    if key in _PRIOR_CACHE:
        return _PRIOR_CACHE[key]
    row = _DATA_PRIORS.get(key) or _HAND_PRIORS.get(key)
    if row is None:
        return None
    cp = ClassPrior(row[0], row[1])
    _PRIOR_CACHE[key] = cp
    return cp


# --------------------------------------------------------------------------- clamp

# Under-observation trigger. A GT-perfect / well-observed box must pass UNCHANGED, so
# the inflate branch fires ONLY when BOTH hold: the instance has few observations AND
# its thinnest axis is markedly below the class-min (a front-shell sliver). Values are
# deliberately conservative — a box that is merely small but plausibly-shaped is left
# alone.
UNDEROBS_MAX_N_OBS: int = 2       # inflate only for n_obs <= this (single/near-single view)
UNDEROBS_THIN_FRAC: float = 0.60  # thin axis below this * class-min => under-observed


def _underobserved(sorted_ext: np.ndarray, prior: ClassPrior, n_obs: int) -> bool:
    """True when the instance looks like a thin single-view shell of this class.

    Requires few observations AND a thinnest-axis extent well under the class-min.
    The min-clamp always runs; only the *inflate-toward-typical* step is gated on this.
    """
    if n_obs > UNDEROBS_MAX_N_OBS:
        return False
    return bool(sorted_ext[0] < UNDEROBS_THIN_FRAC * prior.min_ext[0])


# --------------------------------------------------------------- degenerate floor

# Below this per-axis extent (metres) an AABB is geometrically broken, not merely
# thin: a single-viewpoint trimmed box can land with (near-)zero width on an axis
# when duplicate/collinear points survive the 2nd/98th-pct trim, and a zero-extent
# axis can NEVER satisfy the footprint-IoM gate that on()/near()/between() share
# (core.geometry.toolbox), regardless of how correct the instance semantics are
# (issue #125). Matches the measurement threshold used to size the problem.
DEGENERATE_EXTENT_M: float = 0.02  # 2 cm

# Fallback floor for the minority of challenge-vocabulary classes with no dimension
# prior at all. Small and conservative — enough to clear DEGENERATE_EXTENT_M with a
# little headroom, never large enough to look like a real clamp/inflate.
ABS_MIN_EXTENT_FALLBACK_M: float = 0.03  # 3 cm


def floor_degenerate_aabb(
    aabb_min: np.ndarray, aabb_max: np.ndarray, label: str
) -> tuple[np.ndarray, np.ndarray]:
    """Raise any (near-)zero-extent axis up to a plausible floor, centre preserved.

    Issue #125: a trimmed single-viewpoint AABB can land with an axis extent below
    :data:`DEGENERATE_EXTENT_M` — most often an exact-zero XY footprint — which
    permanently fails every footprint-gated relation predicate no matter how correct
    the instance's semantics are. This grows ONLY the broken axis (or axes), about
    the box's existing centre, up to the class prior's thinnest-rank minimum
    (:attr:`ClassPrior.min_ext`), or :data:`ABS_MIN_EXTENT_FALLBACK_M` when the class
    has no prior. An axis already at/above the threshold is returned unchanged.

    Deliberately NOT :func:`clamp_extents`: that function clamps every axis to the
    class-min unconditionally (the marker-rendering path) — a general resize of
    already-valid boxes, which #123 measured as a net-negative change when applied
    more broadly. This is narrower on purpose: a healthy box is a true no-op here.
    """
    lo = np.asarray(aabb_min, dtype=float).copy()
    hi = np.asarray(aabb_max, dtype=float).copy()
    ext = hi - lo
    degenerate = ext < DEGENERATE_EXTENT_M
    if not np.any(degenerate):
        return lo, hi

    prior = prior_for(label)
    floor = float(prior.min_ext.min()) if prior is not None else ABS_MIN_EXTENT_FALLBACK_M
    floor = max(floor, DEGENERATE_EXTENT_M)

    centre = (lo + hi) / 2.0
    for axis in np.nonzero(degenerate)[0]:
        half = max(float(ext[axis]), floor) / 2.0
        lo[axis] = centre[axis] - half
        hi[axis] = centre[axis] + half
    return lo, hi


# --------------------------------------------------------------- fuse-time extent cap

# Issue #104: a hard backstop on the box a FUSE actually produces, looser than
# tracker.TrackerConfig.extent_veto_factor (1.3). That match-time veto
# (core.perception.tracker._match_plausible) polices any ONE match -- but issue
# #153 floors the veto at near-zero centroid separation (extent_veto_min_sep) so a
# genuine re-observation is never blocked, and a long run of such individually-tiny,
# co-located merges can each legitimately pass while the ACCUMULATED union still
# creeps well past a plausible size one small step at a time (no single step looks
# unsafe; the cumulative drift is). #104's reproduction is exactly this: a
# hotel-room bedside-table instance fused to ~3x its class's typical extent despite
# every contributing match individually clearing #94/#153's checks. Deliberately
# looser than the match-time veto (which would fight legitimate single-merge growth,
# e.g. occlusion revealing more of one real object in one step) -- this only trims
# the RESULT once it has drifted implausibly large, regardless of how it got there.
FUSE_EXTENT_CAP_FACTOR: float = 1.5


def cap_fused_extent(
    aabb_min: np.ndarray, aabb_max: np.ndarray, label: str, points: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Cap a freshly fused AABB's sorted extent to :data:`FUSE_EXTENT_CAP_FACTOR` x
    the class's typical extent (issue #104), re-centred on the accumulated point
    mass (``points.mean(axis=0)``) when available, else the box's own centre.

    Fails open -- returns the box completely unchanged -- when the class has no
    dimension prior, the same convention :func:`core.perception.tracker._match_plausible`
    (#94) uses: with nothing to judge plausibility against, a genuinely huge (or
    off-vocabulary-shaped) object is never second-guessed against a made-up bound.
    Also a no-op whenever every sorted axis is already within the cap -- a
    well-behaved fuse is returned byte-identical, only a box that has actually blown
    past the bound is touched.

    Unlike :func:`floor_degenerate_aabb` (which only ever grows a too-thin axis),
    this only ever SHRINKS: it is the missing other half of the dimension-sanity
    table -- a size backstop on the box a fuse produces, not on a match decision.
    """
    lo = np.asarray(aabb_min, dtype=float)
    hi = np.asarray(aabb_max, dtype=float)
    prior = prior_for(label)
    if prior is None:
        return lo, hi
    ext = hi - lo
    limit = FUSE_EXTENT_CAP_FACTOR * prior.typ_ext  # already sorted (thin, mid, long)
    order = np.argsort(ext, kind="stable")
    sorted_ext = ext[order]
    if np.all(sorted_ext <= limit):
        return lo, hi
    capped_sorted = np.minimum(sorted_ext, limit)
    out_ext = ext.copy()
    for rank, axis in enumerate(order):
        out_ext[axis] = capped_sorted[rank]
    if points is not None and len(points):
        centre = np.asarray(points, dtype=float).mean(axis=0)
    else:
        centre = (lo + hi) / 2.0
    new_lo = centre - out_ext / 2.0
    new_hi = centre + out_ext / 2.0
    return new_lo, new_hi


def clamp_extents(
    extents: np.ndarray,
    label: str,
    n_obs: int,
) -> np.ndarray:
    """Clamp a per-axis extent triple against the class prior; return adjusted extents.

    ``extents`` is the raw ``(sx, sy, sz)`` in map-frame axis order. Steps:
      1. sort to (thin, mid, long) so the prior (also sorted) aligns rank-for-rank;
      2. clamp each ranked extent up to the class-min;
      3. if the instance is under-observed (:func:`_underobserved`), inflate the
         thinnest (least-observed) ranked axis up to the class-typical value;
      4. distribute the per-rank deltas back onto the original axes.

    Returns ``extents`` unchanged (a copy) when no prior exists for the class. A box
    already at/above class-min on every axis with enough observations is returned
    identical to the input — the named GT-passthrough invariant.
    """
    ext = np.asarray(extents, dtype=float).copy()
    prior = prior_for(label)
    if prior is None:
        return ext

    order = np.argsort(ext, kind="stable")  # ascending: order[0] = thinnest axis
    sorted_ext = ext[order]

    adjusted = np.maximum(sorted_ext, prior.min_ext)  # step 2: min-clamp per rank
    if _underobserved(sorted_ext, prior, n_obs):
        # step 3: pull the thinnest (least-observed) axis toward typical.
        adjusted[0] = max(adjusted[0], prior.typ_ext[0])

    # step 4: map ranked deltas back onto original axes.
    out = ext.copy()
    for rank, axis in enumerate(order):
        out[axis] = adjusted[rank]
    return out


def clamp_record_marker(record: InstanceRecord) -> MarkerBox:
    """Prior-clamped :class:`MarkerBox` for an instance (centre preserved).

    The seam that other agents' marker path should call instead of
    ``record.to_marker()`` when a per-class dimension prior should apply. The box
    centre (== nav goal) is preserved; only extents are clamped/inflated, grown
    symmetrically about the existing centre so the correct instance is never moved.
    """
    centre = (record.aabb_min + record.aabb_max) / 2.0
    ext = clamp_extents(record.extents, record.label, record.n_obs)
    return MarkerBox(
        float(centre[0]), float(centre[1]), float(centre[2]),
        float(ext[0]), float(ext[1]), float(ext[2]),
        label=record.label,
    )


# --------------------------------------------------------------------------- regenerator

def _generate_data_priors(data_dir: str) -> str:
    """Recompute ``_DATA_PRIORS`` from the VLA-3D Unity object CSVs; return the literal.

    Offline seed generator (kept as a ``__main__`` block, per H12). Reads every
    ``*_object_result.csv`` under ``data_dir``, folds each instance's ``raw_label``
    onto a challenge-vocabulary class (exact canonical / object bridge / head-noun),
    and emits per-sorted-axis 10th-percentile (min) and median (typical) extents.

    Requires the VLA-3D data checkout (not shipped with the repo); intended to be run
    manually when the vocabulary or data changes, and the printed table pasted above.
    """
    import csv
    import glob
    import os

    from core.parsing.vocab import SINGLE_NOUNS
    from core.perception.vocab import bridge_synonyms, head_noun

    try:
        from core.parsing.vocab import PHRASE_NOUNS  # type: ignore
    except Exception:  # pragma: no cover - vocab may not expose phrases
        PHRASE_NOUNS = frozenset()

    vocab = {normalize_label(n) for n in SINGLE_NOUNS} | {normalize_label(n) for n in PHRASE_NOUNS}

    def to_vocab_class(raw: str) -> str | None:
        lab = normalize_label(raw)
        if lab in vocab:
            return lab
        for s in bridge_synonyms(lab):
            if s in vocab:
                return s
        h = head_noun(lab)
        return h if h in vocab else None

    per: dict[str, list[tuple[float, float, float]]] = {}
    scenes: dict[str, set[str]] = {}
    for path in glob.glob(os.path.join(data_dir, "*", "*_object_result.csv")):
        scene = os.path.basename(os.path.dirname(path))
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                cls = to_vocab_class(row["raw_label"])
                if cls is None:
                    continue
                try:
                    dx = float(row["object_bbox_xlength"])
                    dy = float(row["object_bbox_ylength"])
                    dz = float(row["object_bbox_zlength"])
                except (ValueError, KeyError):
                    continue
                if min(dx, dy, dz) <= 0:
                    continue
                per.setdefault(cls, []).append(tuple(sorted((dx, dy, dz))))
                scenes.setdefault(cls, set()).add(scene)

    lines = ["_DATA_PRIORS = {"]
    for cls in sorted(per):
        arr = np.array(per[cls])
        p10 = np.percentile(arr, 10, axis=0)
        med = np.percentile(arr, 50, axis=0)
        mn = ", ".join(f"{v:.3f}" for v in p10)
        tp = ", ".join(f"{v:.3f}" for v in med)
        lines.append(f'    "{cls}": (({mn}), ({tp})),  # n={len(arr)}, scenes={len(scenes[cls])}')
    lines.append("}")
    missing = sorted(vocab - set(per))
    lines.append(f"# vocab classes without data (hand-curated): {missing}")
    return "\n".join(lines)


if __name__ == "__main__":  # pragma: no cover - offline regeneration utility
    import sys

    default = r"C:/Users/jyuc1/Documents/Uni/2026 CMU VLA/data/vla3d/Unity"
    data_dir = sys.argv[1] if len(sys.argv) > 1 else default
    print(_generate_data_priors(data_dir))
