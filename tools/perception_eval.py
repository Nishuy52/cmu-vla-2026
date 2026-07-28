"""Score a perception instance index (replay OR live) against scene ground truth.

Works on ANY index written in the live schema (``core.perception.scene_index.
dump_instance_index``): a full live run's JSONL (many periodic records, read the
last/``answer_time`` one) or a :mod:`tools.perception_replay` output (a single
``replay_final`` record) score identically -- the whole point is that a replay and a
live run on the same scene are directly comparable.

Frame note (issue #124): the VLA-3D CSV ids match the sim frame id-for-id, so no
transform is fit between live/replay instance positions and GT positions -- both are
consumed in the identity frame.

Metrics (per scene, plus a pooled row across every scene passed in one invocation):

1. **recall** -- a GT object counts as found if some same-label instance's centroid
   lies within ``max(0.5 m, half the GT object's diagonal)`` (inflation-robust: an
   IoU-based match would under-count matches against the heavily over-sized boxes
   perception currently produces).
2. **duplication factor** -- total instances / GT objects found.
3. **inflation** -- median(matched instance AABB diagonal / matched GT AABB diagonal).
4. **shell adjacency** -- fraction of instances with >=1 AABB face within 5 cm of the
   scene-wide extremum on that axis, next to the same statistic computed over the GT
   boxes (the "how many objects are genuinely against a wall" baseline).
5. **per-class table** -- live count vs GT count, sorted by over-production ratio.

CLI::

    python -m tools.perception_eval --index a.jsonl --scene office_1
    python -m tools.perception_eval --index a.jsonl --scene office_1 \\
        --index b.jsonl --scene livingroom_1   # per-scene + pooled table
    python -m tools.perception_eval --compare before.jsonl after.jsonl --scene office_1

Pure offline dev tool (tools/ -- never part of the scored pipeline). No GPU needed.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

import tools  # noqa: F401 -- inserts <repo>/src onto sys.path

from core.groundtruth.loader import load_scene
from core.perception.vocab import is_structural_class

DEFAULT_UNITY_DIR = Path("data/vla3d/Unity")
SHELL_TOL_M = 0.05  # face-to-scene-extremum tolerance for "on the shell"
RECALL_MIN_TOL_M = 0.5  # floor on the recall centroid tolerance


# --------------------------------------------------------------------------- loading


def load_snapshot(path: str | Path) -> dict[str, Any]:
    """Read the scoring snapshot out of an instance-index JSONL file.

    Prefers the record tagged ``"answer_time"`` (the one a live run fires right
    before publishing an answer -- the fairest "what would the robot have answered
    with" snapshot); falls back to the LAST record in the file otherwise. A replay
    index (a single ``"replay_final"`` record) trivially satisfies "last record" and
    is scored identically to a live index this way.
    """
    path = Path(path)
    best: dict[str, Any] | None = None
    last: dict[str, Any] | None = None
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            last = rec
            if rec.get("tag") == "answer_time":
                best = rec
    if best is not None:
        return best
    if last is not None:
        return last
    raise ValueError(f"{path}: no records found")


def load_gt(scene_name: str, unity_dir: str | Path = DEFAULT_UNITY_DIR) -> list:
    """Ground-truth InstanceRecords for ``scene_name`` (folder under ``unity_dir``)."""
    return load_scene(Path(unity_dir) / scene_name).instances


# --------------------------------------------------------------------------- geometry helpers


def _diag(lo: np.ndarray, hi: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(hi, dtype=float) - np.asarray(lo, dtype=float)))


def shell_fraction(lo: np.ndarray, hi: np.ndarray) -> float:
    """Fraction of boxes with >=1 face within :data:`SHELL_TOL_M` of the scene extremum.

    ``lo``/``hi`` are (N, 3) arrays of AABB corners over the SAME box population the
    fraction is computed for (the scene-wide extremum is derived from them, not from
    an external reference) -- matching the intent of "does an instance terminate on
    the room shell" for that population. Returns ``nan`` for an empty population.
    """
    if len(lo) == 0:
        return float("nan")
    gmin = lo.min(axis=0)
    gmax = hi.max(axis=0)
    on_min = np.abs(lo - gmin) <= SHELL_TOL_M
    on_max = np.abs(hi - gmax) <= SHELL_TOL_M
    return float((on_min | on_max).any(axis=1).mean())


# --------------------------------------------------------------------------- matching


@dataclass
class Match:
    gt_index: int
    live_index: int | None  # None == not found
    distance: float = float("nan")


def match_gt_to_live(gt: list, live: list[dict]) -> list[Match]:
    """For each GT object, find the NEAREST same-label live instance within tolerance.

    Tolerance is ``max(RECALL_MIN_TOL_M, gt_diagonal / 2)`` -- a small object still
    gets at least a 0.5 m allowance (odometry/lidar noise), a large object (a sofa, a
    counter run) gets proportionally more slack. Nearest-within-tolerance rather than
    first-within-tolerance, so results don't depend on live instance ordering; ties
    broken by higher detector score. A single live instance CAN satisfy more than one
    GT object (no live-side dedup) -- deliberately permissive, matching the informal
    "was this object found at all" reading of recall rather than a 1:1 assignment.
    """
    live_pos = np.array([inst["position"] for inst in live], dtype=float) if live else np.zeros((0, 3))
    live_labels = [inst["label"] for inst in live]
    matches: list[Match] = []
    for gi, g in enumerate(gt):
        tol = max(RECALL_MIN_TOL_M, _diag(g.aabb_min, g.aabb_max) / 2.0)
        best_li: int | None = None
        best_dist = float("inf")
        best_score = float("-inf")
        for li, label in enumerate(live_labels):
            if label != g.label:
                continue
            dist = float(np.linalg.norm(live_pos[li] - np.asarray(g.centroid, dtype=float)))
            if dist > tol:
                continue
            score = float(live[li].get("score", 0.0))
            if dist < best_dist or (dist == best_dist and score > best_score):
                best_li, best_dist, best_score = li, dist, score
        matches.append(Match(gt_index=gi, live_index=best_li, distance=best_dist if best_li is not None else float("nan")))
    return matches


# --------------------------------------------------------------------------- metrics


@dataclass
class ClassRow:
    label: str
    live: int
    gt: int

    @property
    def ratio(self) -> float:
        return self.live / self.gt if self.gt else float("inf")


@dataclass
class SceneMetrics:
    scene: str
    n_gt: int
    n_live: int
    n_found: int
    recall: float
    duplication_factor: float
    inflation: float
    shell_live: float
    shell_gt: float
    per_class: list[ClassRow] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scene": self.scene,
            "n_gt": self.n_gt,
            "n_live": self.n_live,
            "n_found": self.n_found,
            "recall": self.recall,
            "duplication_factor": self.duplication_factor,
            "inflation": self.inflation,
            "shell_live": self.shell_live,
            "shell_gt": self.shell_gt,
            "per_class": [
                {"label": r.label, "live": r.live, "gt": r.gt, "ratio": r.ratio}
                for r in sorted(self.per_class, key=lambda r: r.ratio, reverse=True)
            ],
        }


def compute_metrics(
    scene_name: str, live: list[dict], gt: list, *, exclude_structural: bool = False
) -> SceneMetrics:
    """Compute every metric in the module docstring for one scene's (live, gt) pair.

    ``exclude_structural`` (issue #129, default off -- existing callers/behaviour
    unchanged): when set, drops structural classes (floor/ceiling/wall/window/door/
    door frame/column; see ``core.perception.vocab.is_structural_class``) from the
    ``per_class`` table only. Every other metric (recall, duplication_factor,
    inflation, shell fractions, n_gt/n_live/n_found) is computed over the full
    population exactly as before -- this flag narrows only the class CENSUS table,
    matching the fact that no training question ever counts a structural class.
    """
    n_gt = len(gt)
    n_live = len(live)

    matches = match_gt_to_live(gt, live) if n_gt else []
    found = [m for m in matches if m.live_index is not None]
    n_found = len(found)
    recall = (n_found / n_gt) if n_gt else float("nan")
    duplication_factor = (n_live / n_found) if n_found else float("nan")

    ratios = []
    for m in found:
        g = gt[m.gt_index]
        inst = live[m.live_index]
        gt_diag = _diag(g.aabb_min, g.aabb_max)
        if gt_diag <= 0:
            continue
        inst_diag = _diag(np.asarray(inst["aabb_min"]), np.asarray(inst["aabb_max"]))
        ratios.append(inst_diag / gt_diag)
    inflation = float(np.median(ratios)) if ratios else float("nan")

    live_lo = np.array([i["aabb_min"] for i in live], dtype=float) if live else np.zeros((0, 3))
    live_hi = np.array([i["aabb_max"] for i in live], dtype=float) if live else np.zeros((0, 3))
    gt_lo = np.array([g.aabb_min for g in gt], dtype=float) if gt else np.zeros((0, 3))
    gt_hi = np.array([g.aabb_max for g in gt], dtype=float) if gt else np.zeros((0, 3))
    shell_live = shell_fraction(live_lo, live_hi)
    shell_gt = shell_fraction(gt_lo, gt_hi)

    live_by_label: dict[str, int] = {}
    for inst in live:
        live_by_label[inst["label"]] = live_by_label.get(inst["label"], 0) + 1
    gt_by_label: dict[str, int] = {}
    for g in gt:
        gt_by_label[g.label] = gt_by_label.get(g.label, 0) + 1
    labels = set(live_by_label) | set(gt_by_label)
    if exclude_structural:
        labels = {lab for lab in labels if not is_structural_class(lab)}
    per_class = [ClassRow(label=lab, live=live_by_label.get(lab, 0), gt=gt_by_label.get(lab, 0)) for lab in labels]

    return SceneMetrics(
        scene=scene_name,
        n_gt=n_gt,
        n_live=n_live,
        n_found=n_found,
        recall=recall,
        duplication_factor=duplication_factor,
        inflation=inflation,
        shell_live=shell_live,
        shell_gt=shell_gt,
        per_class=per_class,
    )


def pool_metrics(scenes: list[SceneMetrics]) -> SceneMetrics:
    """Sum/re-derive every metric across ``scenes`` into one pooled row."""
    n_gt = sum(s.n_gt for s in scenes)
    n_live = sum(s.n_live for s in scenes)
    n_found = sum(s.n_found for s in scenes)
    recall = (n_found / n_gt) if n_gt else float("nan")
    duplication_factor = (n_live / n_found) if n_found else float("nan")

    # inflation: pool the per-scene medians weighted by how many matches fed them
    # (re-deriving the raw ratio list would need scenes to hand it back; the pooled
    # median-of-medians is a reasonable, cheap approximation for a summary row).
    valid = [s.inflation for s in scenes if s.n_found and not np.isnan(s.inflation)]
    inflation = float(np.median(valid)) if valid else float("nan")

    shell_live_num = sum(s.shell_live * s.n_live for s in scenes if s.n_live and not np.isnan(s.shell_live))
    shell_gt_num = sum(s.shell_gt * s.n_gt for s in scenes if s.n_gt and not np.isnan(s.shell_gt))
    shell_live = (shell_live_num / n_live) if n_live else float("nan")
    shell_gt = (shell_gt_num / n_gt) if n_gt else float("nan")

    by_label: dict[str, list[int]] = {}
    for s in scenes:
        for row in s.per_class:
            acc = by_label.setdefault(row.label, [0, 0])
            acc[0] += row.live
            acc[1] += row.gt
    per_class = [ClassRow(label=lab, live=v[0], gt=v[1]) for lab, v in by_label.items()]

    return SceneMetrics(
        scene="POOLED",
        n_gt=n_gt,
        n_live=n_live,
        n_found=n_found,
        recall=recall,
        duplication_factor=duplication_factor,
        inflation=inflation,
        shell_live=shell_live,
        shell_gt=shell_gt,
        per_class=per_class,
    )


# --------------------------------------------------------------------------- printing


def _fmt_pct(x: float) -> str:
    return "n/a" if np.isnan(x) else f"{x:.1%}"


def _fmt_f(x: float) -> str:
    return "n/a" if np.isnan(x) else f"{x:.2f}"


def print_headline_table(rows: list[SceneMetrics], out=sys.stdout) -> None:
    hdr = (
        f"{'scene':<20}{'GT':>5}{'live':>6}{'found':>7}{'recall':>9}"
        f"{'dup x':>8}{'inflate':>9}{'shell live':>12}{'shell GT':>10}"
    )
    print(hdr, file=out)
    print("-" * len(hdr), file=out)
    for r in rows:
        print(
            f"{r.scene:<20}{r.n_gt:>5}{r.n_live:>6}{r.n_found:>7}"
            f"{_fmt_pct(r.recall):>9}{_fmt_f(r.duplication_factor):>8}"
            f"{_fmt_f(r.inflation):>9}{_fmt_pct(r.shell_live):>12}{_fmt_pct(r.shell_gt):>10}",
            file=out,
        )


def print_per_class_table(metrics: SceneMetrics, out=sys.stdout, limit: int = 20) -> None:
    print(f"\nper-class live-vs-GT for {metrics.scene} (worst over-production first):", file=out)
    hdr = f"{'ratio':>7}  {'class':<24}{'live':>6}{'GT':>5}"
    print(hdr, file=out)
    rows = sorted(metrics.per_class, key=lambda r: r.ratio, reverse=True)
    for row in rows[:limit]:
        ratio = "inf" if np.isinf(row.ratio) else f"{row.ratio:.1f}"
        print(f"{ratio:>7}  {row.label:<24}{row.live:>6}{row.gt:>5}", file=out)


# --------------------------------------------------------------------------- CLI


def _eval_one(
    index_path: str, scene_name: str, unity_dir: str, *, exclude_structural: bool = False
) -> SceneMetrics:
    snap = load_snapshot(index_path)
    live = snap.get("instances", [])
    gt = load_gt(scene_name, unity_dir)
    return compute_metrics(scene_name, live, gt, exclude_structural=exclude_structural)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--index", action="append", default=[],
        help="instance_index.jsonl path (repeatable; pair with --scene in the same order)",
    )
    ap.add_argument(
        "--scene", action="append", default=[],
        help="GT scene name under --unity-dir (repeatable; pair with --index in the same order)",
    )
    ap.add_argument(
        "--compare", nargs=2, metavar=("A", "B"), default=None,
        help="before/after A/B: two instance_index.jsonl paths scored against the SAME --scene",
    )
    ap.add_argument("--unity-dir", default=str(DEFAULT_UNITY_DIR), help="VLA-3D Unity scenes root")
    ap.add_argument("--out", default=None, help="JSON output path (default: <first index>.eval.json)")
    ap.add_argument(
        "--exclude-structural", action="store_true",
        help="drop structural classes (floor/ceiling/wall/window/door/door frame/"
        "column, issue #129) from the per-class census table only; every other "
        "metric is unaffected",
    )
    args = ap.parse_args(argv)

    if args.compare:
        if len(args.scene) != 1:
            ap.error("--compare requires exactly one --scene")
        a_path, b_path = args.compare
        a = _eval_one(a_path, args.scene[0], args.unity_dir, exclude_structural=args.exclude_structural)
        b = _eval_one(b_path, args.scene[0], args.unity_dir, exclude_structural=args.exclude_structural)
        a.scene, b.scene = f"A ({Path(a_path).name})", f"B ({Path(b_path).name})"
        print_headline_table([a, b])
        print_per_class_table(a)
        print_per_class_table(b)
        out_path = args.out or f"{a_path}.compare.json"
        payload = {"a": a.to_dict(), "b": b.to_dict()}
        Path(out_path).write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {out_path}")
        return 0

    if not args.index or len(args.index) != len(args.scene):
        ap.error("--index and --scene must be given the same number of times (or use --compare)")

    per_scene = [
        _eval_one(idx, sc, args.unity_dir, exclude_structural=args.exclude_structural)
        for idx, sc in zip(args.index, args.scene)
    ]
    rows = list(per_scene)
    if len(per_scene) > 1:
        rows.append(pool_metrics(per_scene))

    print_headline_table(rows)
    for m in per_scene:
        print_per_class_table(m)

    out_path = args.out or f"{args.index[0]}.eval.json"
    payload = {"scenes": [m.to_dict() for m in per_scene]}
    if len(per_scene) > 1:
        payload["pooled"] = pool_metrics(per_scene).to_dict()
    Path(out_path).write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
