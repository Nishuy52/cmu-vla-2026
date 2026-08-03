"""Flag GT instances whose ``raw_label`` is geometrically implausible for its class.

Issue #144: a GT label that contradicts its own geometry (#110's livingroom_1
object 91 — labelled ``tv remote``, actually a 1.70 x 0.98 m wall-mounted TV
panel) is invisible to everything downstream: nothing cross-checks ``raw_label``
against extents, so the object is unfindable under its true name and every
relational predicate reasons about it as the wrong kind of thing.

This tool reuses the existing per-class :mod:`core.perception.dimension_priors`
(min/typical sorted-axis extents, already derived from the same VLA-3D data) as
the plausibility oracle, instead of inventing a second one: an instance is
flagged when its true oriented-box *volume* (product of the three ``obb_extents``
— never the AABB-of-OBB, a deliberate over-approximation for rotated boxes) is
>= ``factor`` times the class's typical volume (``prior.typ_ext`` product) or
<= 1/``factor`` of it — the same "volume vs. same-label median, flagged at
>=15x / <=1/15x" shape #144's own heuristic scan used, just anchored on the
already-vetted per-class prior instead of a fresh same-run median (so a label
with only 1-2 GT instances still gets a stable yardstick, as long as *some*
scene contributed to the prior). This is deliberately a wider net than the
min-clamp in dimension_priors (which is tolerant by design, to avoid
over-correcting real detections); the validator is an offline human-facing
report, not a runtime clamp.

Instances are pooled by (normalised) ``raw_label`` across ALL 15 scenes before
the ``min_n`` (family-size) test is applied, matching #144's own heuristic
scan ("124 labels with >=3 instances" out of 1967 objects total, scanned
across the whole dataset, not per scene) — a label with 1-2 instances in every
individual scene but >=3 pooled across scenes still gets checked, and the
bounding note (a label that never reaches 3 anywhere is invisible to this
heuristic) is inherited unchanged.

Usage (from the repo root, needs the ``src`` package on the path)::

    python -m tools.gt_label_geometry_validator [--data-dir data/vla3d/Unity] \
        [--factor 12.0] [--min-n 3]

``--min-n`` mirrors #144's own heuristic scan (only labels with >= N same-scene
instances have a "family" to be out of — a label appearing once or twice is
invisible to this heuristic entirely; documented, not silently hidden).

Pure stdlib + numpy; imports the real loader/priors code so this never drifts
from what the pipeline actually does. Offline dev tool -- not part of the
scored pipeline.
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from core.groundtruth.loader import load_scene  # noqa: E402
from core.perception.dimension_priors import prior_for  # noqa: E402

#: #144's own heuristic scan used >=15x / <=1/15x against a same-run same-label
#: median. This validator anchors on the dimension_priors typical volume instead
#: (stable across runs, already vetted) — but that median is computed FROM the
#: full VLA-3D corpus, including any mislabelled outlier itself, which drags the
#: median toward the outlier and understates its ratio. Concretely: #144's flagged
#: livingroom_2 clock (object 53) scores ~13.5x against the dimension_priors
#: typical, just under a literal 15x cut. 12x is the smallest widening that still
#: catches every #144-listed candidate the priors can see (all but "hanger",
#: which has no dimension prior at all — see the per-candidate table in issue
#: #144's resolution).
DEFAULT_FACTOR = 12.0
DEFAULT_MIN_N = 3  # matches #144's own heuristic scan (needs a same-label family)


class Flag:
    __slots__ = ("scene", "object_id", "label", "extents", "reason")

    def __init__(self, scene: str, object_id: int, label: str, extents: np.ndarray, reason: str) -> None:
        self.scene = scene
        self.object_id = object_id
        self.label = label
        self.extents = extents
        self.reason = reason

    def row(self) -> tuple[str, ...]:
        ex = ", ".join(f"{v:.3f}" for v in self.extents)
        return (self.scene, str(self.object_id), self.label, ex, self.reason)


def _load_all_instances(data_dir: Path) -> list[tuple[str, object]]:
    """(scene_name, InstanceRecord) for every object across every scene folder."""
    out: list[tuple[str, object]] = []
    for scene_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        try:
            scene = load_scene(scene_dir)
        except FileNotFoundError:
            continue  # not a scene folder (no *_object_result.csv)
        for rec in scene.instances:
            out.append((scene.scene_name, rec))
    return out


def scan_all(data_dir: Path, *, factor: float, min_n: int) -> list[Flag]:
    """Flag instances whose raw-label AABB volume is >= ``factor``x or <= 1/``factor``x
    the class prior's typical volume, restricted to labels with >= ``min_n`` instances
    POOLED ACROSS ALL SCENES (matching #144's own heuristic scan, which scanned the
    whole dataset rather than per-scene families).
    """
    all_recs = _load_all_instances(data_dir)
    by_label: dict[str, list[tuple[str, object]]] = defaultdict(list)
    for scene_name, rec in all_recs:
        by_label[rec.label].append((scene_name, rec))

    flags: list[Flag] = []
    for label, entries in by_label.items():
        if len(entries) < min_n:
            continue  # no family to be out of (documented bounding note, #144)
        prior = prior_for(label)
        if prior is None:
            continue  # no dimension prior for this class; nothing to check against
        typ_vol = float(np.prod(prior.typ_ext))
        if typ_vol <= 0:
            continue
        for scene_name, rec in entries:
            # Use the TRUE oriented-box extents (obb_extents), never the AABB-of-OBB
            # (rec.extents): dimension_priors was generated straight from the CSV's
            # object_bbox_{x,y,z}length columns (the OBB dims), and for any object with
            # a non-axis-aligned heading the AABB is a deliberate over-approximation
            # (core.groundtruth.loader.obb_to_aabb, up to ~sqrt(2) per axis) — comparing
            # that inflated box against an OBB-derived prior would spuriously flag
            # correctly-labelled rotated objects (e.g. a 22.5-degree-heading lamp whose
            # AABB footprint reads ~1.7x its true footprint area).
            raw_ext = rec.obb_extents if rec.obb_extents is not None else rec.extents
            sorted_ext = np.sort(np.asarray(raw_ext, dtype=float))
            inst_vol = float(np.prod(sorted_ext))
            ratio = inst_vol / typ_vol
            if factor <= ratio or ratio <= 1.0 / factor:
                direction = "large" if ratio > 1 else "small"
                reason = (
                    f"volume {inst_vol:.4f}m^3 is {ratio:.1f}x class-typical "
                    f"{typ_vol:.4f}m^3 ({direction})"
                )
                flags.append(Flag(scene_name, rec.instance_id, label, sorted_ext, reason))
    return flags


def _print_table(flags: list[Flag]) -> None:
    if not flags:
        print("no flags")
        return
    header = ("scene", "object_id", "label", "extents (thin,mid,long)", "reason")
    rows = [f.row() for f in flags]
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(header)]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*header))
    print(fmt.format(*("-" * w for w in widths)))
    for r in rows:
        print(fmt.format(*r))
    print(f"\n{len(flags)} flag(s) across {len(set((f.scene, f.object_id) for f in flags))} instance(s)")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", default="data/vla3d/Unity", type=Path)
    ap.add_argument("--factor", default=DEFAULT_FACTOR, type=float)
    ap.add_argument("--min-n", default=DEFAULT_MIN_N, type=int)
    args = ap.parse_args(argv)

    if not args.data_dir.is_dir():
        print(f"not a directory: {args.data_dir}", file=sys.stderr)
        return 2

    flags = scan_all(args.data_dir, factor=args.factor, min_n=args.min_n)
    _print_table(flags)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
