"""Live colour quantisation: pixel RGB -> VLA-3D 15-scheme colour name.

Issue #121. Both live instance-construction paths (``tracker._fused_to_record``,
``scene_index`` merge) build :class:`~core.interfaces.InstanceRecord`\\ s with
``color_bins=()`` and ``caption=""`` — nothing ever fills them, so a colour-word
question attribute can never match a live instance (``_colour_present`` in
``core.geometry.toolbox`` falls back to a substring test that a plain label never
satisfies). This module is the missing piece: it turns the raw pixel colours a
fused lidar cluster projects onto in the panorama into up-to-3 dominant
:class:`~core.interfaces.ColorBin`\\ s, the same shape GT's own loader produces
(``core.groundtruth.loader._color_slots``/``_color_bins``).

**Conservative by construction: a wrong colour label is worse than an absent
one.** A colour-qualified question is answered with confidence downstream, so a
bad label is a confident wrong answer, not a graceful "don't know". Every path
into :class:`ColorBin` production funnels through :func:`top_bins`, which
abstains (returns ``()``, the same shape as "never observed") whenever
:func:`abstain_reason` finds the evidence weak — too few quantised pixels, no
clear majority bin ("mixed distribution"), or a low-saturation capture without
near-unanimous agreement. An abstained instance keeps ``color_bins=()``, which
the rest of the codebase already treats as "colour unknown" (see
``InstanceRecord.color_bins``'s docstring and ``_colour_present``'s no-bins
fallback) — never as a confident colour mismatch.

**Where the 14-name centroid table comes from (read this before touching it).**
The mapping from scheme NAME to RGB is not invented or hand-tuned here — it is
read straight off every VLA-3D ``*_object_result.csv`` in ``data/vla3d/Unity/*``:
each row's ``object_color_scheme{i}`` + ``object_color_r/g/b{i}`` columns pair a
scheme name with the raw RGB GT itself assigns to it. :data:`CENTROIDS_RGB` is the
per-name mean RGB over every (name, rgb) slot across all 15 scenes (reproduce with
the one-pass script in this docstring's sibling task notes; the exact figures were
last verified 28 Jul 2026). This is reading the dataset's OWN fixed encoding, not
fitting a model hyperparameter to training-sample evidence — the generalisation
protocol's holdout rule does not apply to it, precisely because it never looks at
which scene a question came from or what answer it implies; it only decodes what
GT's own colour-scheme column already means in RGB terms. Do NOT add, remove, or
re-weight a centroid to move a specific question's answer.

Only 14 of the 15 scheme names have GT slots — **``white`` never appears in the
VLA-3D colour data at all** (0 of ~2990 slots across all scenes), so it is
deliberately absent from :data:`CENTROIDS_RGB`; the quantiser can never emit
``white`` from a live pixel, matching what GT itself would (not) say.

**Nearest-centroid distance is computed in CIELab, not raw RGB.** ``gray`` alone
is 46% of all GT colour slots with a very large per-axis spread; Euclidean RGB
distance lets it swallow blues/purples/greens that are perceptually distinct but
numerically close in raw RGB. Lab is closer to perceptually uniform, so the same
nearest-centroid rule separates those hues far more often (see
``rgb_to_lab``/``quantise_pixel_names``).

**The black/gray split is NOT special-cased.** ``black``'s GT centroid sits at
the exact synthetic RGB (0, 0, 0) with zero spread (every ``black``-annotated
slot in the CSVs is exactly (0,0,0) — an authoring convention, not something a
camera pixel will ever exactly equal). Nearest-centroid in Lab naturally sends
very-dark pixels to ``black`` and moderately-dark pixels to ``gray`` (verified:
(10,10,12) -> black, (47,79,79) "dark-slate-gray" -> gray, matching the exact
example already documented on ``Thresholds.dark_luma_max`` in
``core.geometry.toolbox``). Nothing here bridges ``black``<->``gray`` — that
remains `vocab.COLOUR_BRIDGE`'s deliberate call; a genuinely dark-but-not-black
pixel that lands in ``gray`` is still recoverable for a "black" query through
``_colour_present``'s luminance cutoff on a neutral bin, unchanged and untouched
by this module.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from core.interfaces import ColorBin

# --------------------------------------------------------------------- centroids

#: Per-name mean raw RGB over every (name, rgb) colour slot in every VLA-3D
#: ``*_object_result.csv`` (all 15 Unity scenes) — GT's OWN name<->RGB encoding,
#: read off the data, not invented. ``white`` is absent: it never appears as a
#: ``object_color_scheme{i}`` value anywhere in the corpus.
CENTROIDS_RGB: dict[str, tuple[float, float, float]] = {
    "gray": (110.0, 121.3, 121.5),
    "black": (0.0, 0.0, 0.0),
    "brown": (168.6, 98.9, 58.6),
    "maroon": (165.5, 37.4, 37.4),
    "olive": (87.7, 110.4, 45.0),
    "blue": (69.6, 128.2, 183.2),
    "pink": (197.2, 133.8, 144.2),
    "red": (229.8, 117.5, 106.5),
    "yellow": (195.0, 163.4, 58.2),
    "purple": (77.2, 62.1, 142.3),
    "green": (44.5, 141.7, 78.1),
    "aqua": (95.0, 158.0, 160.0),
    "navy": (25.0, 25.0, 112.0),
    "orange": (225.0, 93.0, 20.0),
}

_CENTROID_NAMES: tuple[str, ...] = tuple(CENTROIDS_RGB.keys())


def srgb_to_linear(c: np.ndarray) -> np.ndarray:
    """Elementwise sRGB (0-255) -> linear-light (0-1), IEC 61966-2-1."""
    c = c / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Vectorized sRGB (..., 3) 0-255 -> CIELab (..., 3), D65 white point.

    Standard sRGB -> linear -> XYZ (D65) -> Lab pipeline. Used for both the
    quantiser's centroid table and every pixel it classifies, so the distance
    comparison is always apples-to-apples.
    """
    rgb = np.asarray(rgb, dtype=np.float64)
    lin = srgb_to_linear(rgb)
    r, g, b = lin[..., 0], lin[..., 1], lin[..., 2]
    x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041
    xn, yn, zn = 0.95047, 1.0, 1.08883
    xr, yr, zr = x / xn, y / yn, z / zn

    delta = 6.0 / 29.0

    def f(t: np.ndarray) -> np.ndarray:
        return np.where(t > delta ** 3, np.cbrt(t), t / (3 * delta * delta) + 4.0 / 29.0)

    fx, fy, fz = f(xr), f(yr), f(zr)
    ell = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    bb = 200.0 * (fy - fz)
    return np.stack([ell, a, bb], axis=-1)


_CENTROID_LAB = rgb_to_lab(np.array([CENTROIDS_RGB[n] for n in _CENTROID_NAMES]))  # (14, 3)


def quantise_pixel_names(colors: np.ndarray) -> np.ndarray:
    """Nearest-centroid scheme name (in Lab space) for each RGB row in ``colors``.

    ``colors``: (N, 3) uint8/float RGB. Returns an (N,) array of scheme-name
    strings, one of :data:`CENTROIDS_RGB`'s 14 keys per row (never ``white`` —
    it has no centroid; see module docstring).
    """
    lab = rgb_to_lab(np.asarray(colors, dtype=np.float64))
    diffs = lab[:, None, :] - _CENTROID_LAB[None, :, :]
    d2 = np.einsum("nkc,nkc->nk", diffs, diffs)
    idx = np.argmin(d2, axis=1)
    return np.asarray(_CENTROID_NAMES, dtype=object)[idx]


# ----------------------------------------------------------------------- tally


@dataclass
class ColourTally:
    """Running per-name observation tally: point count + summed raw RGB.

    Accumulated across every keyframe an instance is re-observed in (issue #121:
    colour must accumulate, not be overwritten by the latest keyframe) so the
    eventual dominant-bin fractions reflect the WHOLE observation history, the
    same way GT's own percentages describe the whole (simulated) object surface.

    ``sat_sum`` accumulates raw-pixel HSV saturation (see :func:`_pixel_saturation`)
    over every quantised pixel across every observation, alongside ``counts`` —
    together they give :func:`mean_saturation` a running mean without needing to
    retain the raw pixels themselves (see the confidence-gating section below).
    """

    counts: dict[str, int] = field(default_factory=dict)
    rgb_sums: dict[str, np.ndarray] = field(default_factory=dict)
    sat_sum: float = 0.0

    @property
    def total(self) -> int:
        return sum(self.counts.values())


def _pixel_saturation(colors: np.ndarray) -> np.ndarray:
    """Classic HSV saturation, (N,) in [0, 1], for (N, 3) 0-255 RGB rows.

    ``S = 0`` when ``V == 0`` (pure black — achromatic, not undefined-as-1)."""
    c = np.asarray(colors, dtype=np.float64)
    v = c.max(axis=-1)
    span = v - c.min(axis=-1)
    return np.divide(span, v, out=np.zeros_like(v), where=v > 0)


def tally_from_colors(colors: np.ndarray | None, valid: np.ndarray) -> ColourTally | None:
    """Quantise the VALID rows of ``colors`` into a fresh :class:`ColourTally`.

    ``valid`` marks which rows are real panorama samples versus the
    :data:`~core.perception.pano_projection.GRAY` placeholder filled in for
    out-of-VFOV / range-gated points (``pano_projection.sample_colors``) — those
    placeholder rows are NOT a colour observation of the object and must never be
    tallied as ``gray``, so only ``colors[valid]`` is ever quantised. Returns
    ``None`` when nothing valid landed (e.g. the whole cluster projected outside
    the panorama this keyframe) rather than an empty-but-truthy tally.
    """
    if colors is None or len(colors) == 0:
        return None
    obs = np.asarray(colors)[np.asarray(valid, dtype=bool)]
    if len(obs) == 0:
        return None
    names = quantise_pixel_names(obs)
    tally = ColourTally(sat_sum=float(_pixel_saturation(obs).sum()))
    for name in _CENTROID_NAMES:
        mask = names == name
        n = int(mask.sum())
        if n:
            tally.counts[name] = n
            tally.rgb_sums[name] = obs[mask].astype(np.float64).sum(axis=0)
    return tally if tally.counts else None


def merge_tallies(a: ColourTally, b: ColourTally) -> ColourTally:
    """Elementwise-sum two tallies (accumulate-across-observations, issue #121)."""
    counts = dict(a.counts)
    rgb_sums = {k: v.copy() for k, v in a.rgb_sums.items()}
    for name, n in b.counts.items():
        counts[name] = counts.get(name, 0) + n
        rgb_sums[name] = rgb_sums.get(name, np.zeros(3, dtype=np.float64)) + b.rgb_sums[name]
    return ColourTally(counts=counts, rgb_sums=rgb_sums, sat_sum=a.sat_sum + b.sat_sum)


# ------------------------------------------------------------- confidence / abstain

#: Issue #121: a WRONG colour label is worse than an absent one (the head answers
#: colour-qualified questions with confidence, so a bad label is a confident wrong
#: answer, not a graceful "don't know"). These three gates decide whether a tally
#: is trustworthy enough to commit to ANY colour bin; failing any one means
#: :func:`top_bins` returns ``()`` — the same "no colour evidence" shape the rest
#: of the codebase already treats as unknown (``InstanceRecord.color_bins: ()
#: when unknown``, ``core.geometry.toolbox._colour_present``'s no-bins branch),
#: so an abstained instance is answered as "cannot conclude", never as a
#: confident colour mismatch.

#: Fewer quantised pixels than this: not enough samples to trust any bin.
MIN_TALLY_PIXELS: int = 12

#: Winning bin's share of the tally below this: no clear majority colour (the
#: cluster's surface is genuinely mixed, or the observation is too noisy to agree
#: with itself) — "mixed distribution" in the issue's language.
MIN_DOMINANT_FRACTION: float = 0.35

#: Mean per-pixel HSV saturation below this is a washed-out / near-achromatic
#: capture where Lab nearest-centroid assignment is least reliable (small sensor
#: noise easily flips a pixel between neighbouring centroids). Gated jointly with
#: a HIGH agreement requirement below so a genuinely uniform gray/black surface
#: (every pixel agrees, low saturation is simply correct for that object) is
#: still trusted — only a low-saturation AND non-unanimous tally abstains.
MIN_MEAN_SATURATION: float = 0.10

#: Dominant fraction at/above which a low-saturation tally is trusted anyway
#: (near-unanimous agreement overrides the low-saturation gate).
HIGH_AGREEMENT_FRACTION: float = 0.90


def dominant_fraction(tally: ColourTally) -> float:
    """Winning bin's share of the tally's total quantised pixels, in [0, 1]."""
    total = tally.total
    if total == 0:
        return 0.0
    return max(tally.counts.values()) / total


def mean_saturation(tally: ColourTally) -> float:
    """Mean HSV saturation across every quantised pixel in the tally, in [0, 1]."""
    total = tally.total
    if total == 0:
        return 0.0
    return tally.sat_sum / total


def abstain_reason(tally: ColourTally | None) -> str | None:
    """``None`` if ``tally`` is confident enough to commit to a colour bin, else
    a short machine-readable reason (``"no_observation"``, ``"too_few_pixels"``,
    ``"mixed_distribution"``, ``"low_saturation"``)."""
    if tally is None or tally.total == 0:
        return "no_observation"
    if tally.total < MIN_TALLY_PIXELS:
        return "too_few_pixels"
    frac = dominant_fraction(tally)
    if frac < MIN_DOMINANT_FRACTION:
        return "mixed_distribution"
    if mean_saturation(tally) < MIN_MEAN_SATURATION and frac < HIGH_AGREEMENT_FRACTION:
        return "low_saturation"
    return None


#: Cap on how many dominant colour bins an instance carries — mirrors GT's own
#: up-to-3 ``object_color_scheme{1,2,3}`` slots (core.groundtruth.loader._color_slots).
MAX_COLOUR_BINS: int = 3


def top_bins(tally: ColourTally | None, max_bins: int = MAX_COLOUR_BINS) -> tuple[ColorBin, ...]:
    """The up-to-``max_bins`` dominant :class:`ColorBin`\\ s of a tally.

    Fraction is each bin's point share of the tally's TOTAL observed points
    (every quantised point, not just the kept top bins) — matching GT's own
    percentage semantics, where the top-3 slots need not sum to 100% when a
    4th+ minor colour was also present. ``rgb`` is the bin's own mean observed
    RGB (not the scheme centroid) so luminance-based salience
    (``_colour_present``'s ``dark_luma_max``/``light_luma_min``) sees the true
    brightness of what was actually seen, not the scheme's canonical shade.

    Returns ``()`` — the same shape as "no observation" — whenever
    :func:`abstain_reason` finds the evidence too weak to commit to (too few
    pixels, no clear majority, or a low-saturation/non-unanimous capture; issue
    #121). This is the single choke point every caller (scene_index's seed/fuse
    paths) goes through, so a weak observation degrades to "unknown", never to a
    wrong colour name.
    """
    if abstain_reason(tally) is not None:
        return ()
    total = tally.total
    ranked = sorted(tally.counts.items(), key=lambda kv: -kv[1])[:max_bins]
    bins = []
    for name, n in ranked:
        mean_rgb = tally.rgb_sums[name] / n
        rgb = tuple(int(min(255, max(0, round(c)))) for c in mean_rgb)
        bins.append(ColorBin(name=name, rgb=rgb, fraction=n / total))
    return tuple(bins)


# --------------------------------------------------------------------- caption

#: Mirrors core.groundtruth.loader's ``_SIZE_LARGE_VOL``/``_SIZE_SMALL_VOL`` exactly
#: (same AABB-volume size token) so live and GT captions read the same way to the
#: toolbox's attribute matcher. Duplicated rather than imported: `groundtruth/`
#: is a one-directional consumer of `perception/` (see vocab.py's module
#: docstring) and must never be imported FROM here.
_SIZE_LARGE_VOL = 0.5  # m^3 AABB volume at/above which an object reads "big"/"large"
_SIZE_SMALL_VOL = 0.02  # m^3 AABB volume at/below which an object reads "small"


def _size_token(extents: np.ndarray) -> str:
    """Coarse size word from AABB volume — verbatim twin of
    ``core.groundtruth.loader._size_token``."""
    vol = float(np.prod(np.clip(extents, 0.0, None)))
    if vol >= _SIZE_LARGE_VOL:
        return "big large"
    if vol <= _SIZE_SMALL_VOL:
        return "small"
    return "medium"


def build_caption(color_bins: tuple[ColorBin, ...], extents: np.ndarray) -> str:
    """Caption text scanned by the toolbox attribute matcher (colours + size).

    Verbatim twin of ``core.groundtruth.loader._build_caption`` — same shape
    (colour names then a size token) so live and GT-derived instances present
    the same surface to ``_attr_present``/``_colour_present``.
    """
    parts = [b.name for b in color_bins]
    parts.append(_size_token(extents))
    return " ".join(parts)


__all__ = [
    "CENTROIDS_RGB",
    "HIGH_AGREEMENT_FRACTION",
    "MAX_COLOUR_BINS",
    "MIN_DOMINANT_FRACTION",
    "MIN_MEAN_SATURATION",
    "MIN_TALLY_PIXELS",
    "ColourTally",
    "abstain_reason",
    "build_caption",
    "dominant_fraction",
    "mean_saturation",
    "merge_tallies",
    "quantise_pixel_names",
    "rgb_to_lab",
    "srgb_to_linear",
    "tally_from_colors",
    "top_bins",
]
