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
clear majority bin ("mixed distribution"), a low-saturation capture without
near-unanimous agreement, or the top two buckets (named or
:data:`UNCLASSIFIABLE`) sitting within a corpus-derived margin of each other
("close margin", issue #147 — see :data:`MIN_TOP_TWO_MARGIN`). An abstained
instance keeps ``color_bins=()``, which
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
    it has no centroid; see module docstring). This is the raw nearest-centroid
    assignment with NO distance gate — always picks a name, however far it is.
    Callers that need to distinguish "genuinely this colour" from "nothing in
    the 14-name scheme is actually close" must use :func:`nearest_centroid` (see
    the out-of-gamut guard below); :func:`tally_from_colors` does exactly that.
    """
    lab = rgb_to_lab(np.asarray(colors, dtype=np.float64))
    diffs = lab[:, None, :] - _CENTROID_LAB[None, :, :]
    d2 = np.einsum("nkc,nkc->nk", diffs, diffs)
    idx = np.argmin(d2, axis=1)
    return np.asarray(_CENTROID_NAMES, dtype=object)[idx]


def nearest_centroid(colors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Nearest-centroid scheme name AND its Lab distance for each row in ``colors``.

    Returns ``(names, distances)``, both (N,) — ``names`` identical to
    :func:`quantise_pixel_names`, ``distances`` the Lab distance from the pixel
    to that assigned centroid (always >= 0, never gated). The distance is what
    the out-of-gamut guard (:data:`MAX_CENTROID_LAB_DISTANCE`) thresholds.
    """
    lab = rgb_to_lab(np.asarray(colors, dtype=np.float64))
    diffs = lab[:, None, :] - _CENTROID_LAB[None, :, :]
    d2 = np.einsum("nkc,nkc->nk", diffs, diffs)
    idx = np.argmin(d2, axis=1)
    names = np.asarray(_CENTROID_NAMES, dtype=object)[idx]
    dist = np.sqrt(d2[np.arange(len(idx)), idx])
    return names, dist


# ------------------------------------------------------------- out-of-gamut guard

#: Issue #121 follow-up: ``white`` (and off-white/cream/beige/light-gray more
#: generally) has NO centroid in the 14-name scheme — it never occurs in the GT
#: corpus (see module docstring) — so nearest-centroid alone has no notion of
#: "none of these 14 names actually fit"; it always snaps to whichever is
#: numerically closest, however far that is (pure white (250,250,250) landed on
#: ``aqua`` at Lab distance ~43). A pixel farther than this from every centroid
#: is UNCLASSIFIABLE: the truthful answer is "this scheme has no name for it",
#: not a confident guess.
#:
#: Threshold evidence (reproduced against every ``*_object_result.csv`` across
#: all 15 VLA-3D Unity scenes, 2990 colour slots): restricting to the 2776
#: slots where this module's OWN nearest-centroid assignment agrees with GT's
#: own declared scheme name (i.e. actually-correct classifications, not the
#: separate pre-existing gray/tan-vs-aqua/pink edge cases), the Lab distance to
#: the assigned centroid is <= 21.71 for 2775 of those 2776 slots (99.96%); the
#: single exception is one saturated "purple" GT slot (medium-orchid RGB
#: (186,85,211), a rare high-variance outlier for a name with only 22 samples)
#: at 38.58. 25.0 sits in the resulting gap: comfortably above the 21.71
#: ceiling of essentially every legitimately-classified corpus colour, and
#: comfortably below every one of pure white / off-white / beige / light-gray's
#: nearest-centroid distances (28.85 - 42.66, measured directly on those RGBs).
MAX_CENTROID_LAB_DISTANCE: float = 25.0

#: Sentinel tally key for pixels beyond :data:`MAX_CENTROID_LAB_DISTANCE` from
#: every centroid. Never a real scheme name (deliberately not in
#: :data:`CENTROIDS_RGB`), never emitted by :func:`top_bins`.
UNCLASSIFIABLE: str = "_unclassifiable"

#: If at least this fraction of a tally's pixels are unclassifiable, the whole
#: observation abstains (:func:`abstain_reason` -> ``"out_of_gamut"``) rather
#: than reporting whatever minority of pixels happened to land near a real
#: centroid. 0.5: unclassifiable pixels outright outnumbering every real-name
#: bin combined is the clearest "this object's colour isn't in the scheme"
#: signal available; a small minority of stray out-of-gamut pixels (specular
#: highlight, shadow fringe) on an otherwise clearly-coloured object should not
#: by itself veto an otherwise confident classification.
#:
#: Issue #147: this floor alone is NOT a sufficient abstention rule -- a tally
#: at 49% unclassifiable / 51% dominant-named clears it (0.49 < 0.5) and answers
#: confidently, even though the two shares are a coin flip apart. See
#: :data:`MIN_TOP_TWO_MARGIN` below, which composes with this gate rather than
#: replacing it: this floor still catches the "unclassifiable outright wins"
#: case, the margin gate catches the "unclassifiable/named (or named/named)
#: near-tie" case this floor structurally cannot.
MAX_UNCLASSIFIABLE_FRACTION: float = 0.5

#: Issue #147 follow-up to #121's out-of-gamut guard. ``MAX_UNCLASSIFIABLE_
#: FRACTION`` alone lets a 49%-unclassifiable / 51%-named split answer
#: confidently (0.49 < 0.5), reproducing the original "confident wrong colour"
#: failure mode at a smaller split. This gate instead compares the tally's
#: TOP TWO buckets directly -- named vs named, or named vs
#: :data:`UNCLASSIFIABLE` -- and abstains (:func:`abstain_reason` ->
#: ``"close_margin"``) whenever they are within :data:`MIN_TOP_TWO_MARGIN` of
#: each other, regardless of which floor either individually clears. See
#: :func:`top_two_margin`.
#:
#: Threshold evidence (measured 3 Aug 2026 against every ``*_object_result.csv``
#: across all 15 VLA-3D Unity scenes): of 1967 GT colour slots, 869 declare 2+
#: colour schemes (``object_color_scheme_percentage{1,2}``); the top1-top2
#: percentage-point gap across those 869 ranges from 0.005 to 1.0 with no clean
#: bimodal split -- density is roughly uniform (~10-20 slots per 1-point bucket)
#: from gap 0.00 up through ~0.30, then rises toward a large mass at gap ~1.0
#: (dominated by the 1098 single-colour-only slots, always gap=1.0, plus
#: genuinely near-unanimous multi-colour ones). Coarse buckets: gap<0.02: 24
#: slots, gap<0.05: 61, gap<0.10: 140, gap<0.15: 205, gap<0.20: 251 (out of 869).
#: Two anchors fix the usable window: a constructed 49%/51% split (gap 0.02)
#: must abstain (this issue); the livingroom_1 sofa (gray 0.68 / brown 0.32,
#: gap 0.36) must still answer "gray" confidently (#121's own worked example).
#: 0.15 sits inside that 0.02-0.36 window -- >7x the abstain anchor, well under
#: half the confident anchor -- and lands in a corpus region with no sharp
#: density change either side of it, so no single scene's outcome hinges on
#: exactly where within the gap it falls. It also matches the corpus' own
#: authoring: livingroom_1 has several near-50/50 wall/door slots (e.g. gray
#: 0.503/black 0.497, gap 0.006) that are themselves a coin flip between two
#: names in the source annotation -- abstaining on those, rather than picking
#: one, is the correct behaviour this gate is meant to produce.
MIN_TOP_TWO_MARGIN: float = 0.15


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
    names, dist = nearest_centroid(obs)
    # Out-of-gamut guard (issue #121 follow-up): a pixel farther than
    # MAX_CENTROID_LAB_DISTANCE from every centroid gets no name at all -- it is
    # tallied under the UNCLASSIFIABLE sentinel, never one of the 14 scheme
    # names, so a genuinely white/off-white/beige/light-gray surface can never
    # be reported as (say) "aqua" or "pink" just because that was numerically
    # closest.
    far = dist > MAX_CENTROID_LAB_DISTANCE
    tally = ColourTally(sat_sum=float(_pixel_saturation(obs).sum()))
    n_far = int(far.sum())
    if n_far:
        tally.counts[UNCLASSIFIABLE] = n_far
    for name in _CENTROID_NAMES:
        mask = (names == name) & ~far
        n = int(mask.sum())
        if n:
            tally.counts[name] = n
            tally.rgb_sums[name] = obs[mask].astype(np.float64).sum(axis=0)
    return tally if tally.counts else None


def merge_tallies(a: ColourTally, b: ColourTally) -> ColourTally:
    """Elementwise-sum two tallies (accumulate-across-observations, issue #121).

    ``UNCLASSIFIABLE`` (issue #121 follow-up) has a ``counts`` entry but
    deliberately no ``rgb_sums`` entry (:func:`tally_from_colors` never stores
    one — an out-of-gamut pixel is never turned into a :class:`ColorBin`, so
    there is nothing to average), so the ``rgb_sums`` side of this merge must
    default-and-skip rather than index ``b.rgb_sums[name]`` directly.
    """
    counts = dict(a.counts)
    rgb_sums = {k: v.copy() for k, v in a.rgb_sums.items()}
    for name, n in b.counts.items():
        counts[name] = counts.get(name, 0) + n
        if name in b.rgb_sums:
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
    """Winning NAMED bin's share of the tally's total quantised pixels, in [0, 1].

    The :data:`UNCLASSIFIABLE` sentinel is never a candidate winner here — an
    out-of-gamut pixel is not a colour bin, so it cannot "win" a dominance
    check. A tally that is mostly unclassifiable is instead caught by the
    dedicated ``out_of_gamut`` gate in :func:`abstain_reason`.
    """
    total = tally.total
    if total == 0:
        return 0.0
    named = {k: v for k, v in tally.counts.items() if k != UNCLASSIFIABLE}
    if not named:
        return 0.0
    return max(named.values()) / total


def unclassifiable_fraction(tally: ColourTally) -> float:
    """Share of the tally's pixels that landed farther than
    :data:`MAX_CENTROID_LAB_DISTANCE` from every centroid, in [0, 1]."""
    total = tally.total
    if total == 0:
        return 0.0
    return tally.counts.get(UNCLASSIFIABLE, 0) / total


def top_two_margin(tally: ColourTally) -> float:
    """Gap between the tally's top two buckets, as a fraction of its total.

    Unlike :func:`dominant_fraction` (which only ever considers NAMED bins),
    this ranks EVERY bucket including :data:`UNCLASSIFIABLE` -- issue #147: the
    failure mode is a near-tie between the winning named colour and whatever is
    in second place, and that runner-up is just as often the unclassifiable
    sentinel (the reported 49/51 case) as a second named colour. Returns
    ``1.0`` when there is no second bucket at all (a single-bucket tally is
    maximally unambiguous), ``0.0`` for an exact tie. See :data:`MIN_TOP_TWO_MARGIN`."""
    total = tally.total
    if total == 0:
        return 1.0
    counts = sorted(tally.counts.values(), reverse=True)
    top = counts[0]
    second = counts[1] if len(counts) > 1 else 0
    return (top - second) / total


def mean_saturation(tally: ColourTally) -> float:
    """Mean HSV saturation across every quantised pixel in the tally, in [0, 1]."""
    total = tally.total
    if total == 0:
        return 0.0
    return tally.sat_sum / total


def abstain_reason(tally: ColourTally | None) -> str | None:
    """``None`` if ``tally`` is confident enough to commit to a colour bin, else
    a short machine-readable reason (``"no_observation"``, ``"too_few_pixels"``,
    ``"out_of_gamut"``, ``"mixed_distribution"``, ``"low_saturation"``,
    ``"close_margin"``).

    ``close_margin`` (issue #147) is checked LAST, after every existing gate,
    so it only ever adds new abstentions on top of the pre-#147 behaviour --
    a tally that already abstains for some other reason keeps that reason
    (e.g. the 50/50 gray/black zero-saturation tally in the tests stays
    ``"low_saturation"``, not ``"close_margin"``); it only reclassifies a
    would-have-been-confident tally whose top two buckets are within
    :data:`MIN_TOP_TWO_MARGIN` of each other."""
    if tally is None or tally.total == 0:
        return "no_observation"
    if tally.total < MIN_TALLY_PIXELS:
        return "too_few_pixels"
    if unclassifiable_fraction(tally) >= MAX_UNCLASSIFIABLE_FRACTION:
        return "out_of_gamut"
    frac = dominant_fraction(tally)
    if frac < MIN_DOMINANT_FRACTION:
        return "mixed_distribution"
    if mean_saturation(tally) < MIN_MEAN_SATURATION and frac < HIGH_AGREEMENT_FRACTION:
        return "low_saturation"
    if top_two_margin(tally) < MIN_TOP_TWO_MARGIN:
        return "close_margin"
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
    named_counts = {k: v for k, v in tally.counts.items() if k != UNCLASSIFIABLE}
    ranked = sorted(named_counts.items(), key=lambda kv: -kv[1])[:max_bins]
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
    "MAX_CENTROID_LAB_DISTANCE",
    "MAX_COLOUR_BINS",
    "MAX_UNCLASSIFIABLE_FRACTION",
    "MIN_DOMINANT_FRACTION",
    "MIN_MEAN_SATURATION",
    "MIN_TALLY_PIXELS",
    "MIN_TOP_TWO_MARGIN",
    "UNCLASSIFIABLE",
    "ColourTally",
    "abstain_reason",
    "build_caption",
    "dominant_fraction",
    "mean_saturation",
    "merge_tallies",
    "nearest_centroid",
    "quantise_pixel_names",
    "rgb_to_lab",
    "srgb_to_linear",
    "tally_from_colors",
    "top_bins",
    "top_two_margin",
    "unclassifiable_fraction",
]
