"""issue #84: live instance-index recall vs GT scene classes.

Companion to ``core.heads.explore_debug``'s ``live_instances`` field: given one
``explore_debug*.jsonl`` capture and the scene it was captured against, this reads
the FINAL record's live-instance-index summary (the fullest picture the run ever
had) and compares its per-class detected counts against the ground-truth scene's
per-class object counts (``core.groundtruth.load_scene``, the same loader the
offline battery scores against), printing a per-class GT-vs-detected recall table.

Both sides are joined on the SAME canonical label ``core.perception.scene_index
.normalize_label`` applies live (the tracker canonicalises every detection label
the same way before it ever reaches the index — see ``tracker._fused_to_record``),
so a live 'tv' instance and a GT 'television' row land in one row here exactly as
they would resolve to the same instance in a live query.

Captures predate this field's #84 addition without it — every record before the
issue #84 explore_debug change simply lacks ``live_instances``. That is handled
gracefully: this tool says so and exits without a table rather than crashing on a
missing key.

Usage (repo root, project venv)::

    python -m tools.live_harness.instance_recall <path/to/explore_debug.jsonl> <scene_name>
    python -m tools.live_harness.instance_recall \\
        reports/issue83_live_captures/office_1_inst/explore_debug.jsonl office_1

Pure offline dev tool (tools/ — never part of the scored pipeline). CPU only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import tools  # noqa: F401 -- inserts <repo>/src onto sys.path

from core.groundtruth import load_scene
from core.perception.scene_index import normalize_label

_REPO = Path(__file__).resolve().parents[2]
DEFAULT_GROUNDTRUTH = _REPO / "data" / "vla3d" / "Unity"


def _load_last_record(jsonl_path: Path) -> dict | None:
    """The last well-formed JSON record in the capture, or None if there is none."""
    last: dict | None = None
    with open(jsonl_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                last = json.loads(line)
            except json.JSONDecodeError:
                continue
    return last


def _gt_class_counts(scene_dir: Path) -> dict[str, int]:
    gt = load_scene(scene_dir)
    counts: dict[str, int] = {}
    for rec in gt.instances:
        canon = normalize_label(rec.label)
        counts[canon] = counts.get(canon, 0) + 1
    return counts


def _detected_class_counts(record: dict) -> dict[str, int]:
    """Detected per-class counts off a record's ``live_instances.by_class``.

    ``by_class`` keys are already canonical (the tracker normalises every label
    before it reaches the index — see module docstring) so no re-normalisation is
    needed here; done defensively anyway in case a hand-built record is fed in.
    """
    by_class = record.get("live_instances", {}).get("by_class", {})
    counts: dict[str, int] = {}
    for label, n in by_class.items():
        canon = normalize_label(label)
        counts[canon] = counts.get(canon, 0) + int(n)
    return counts


def render_table(gt_counts: dict[str, int], det_counts: dict[str, int]) -> str:
    classes = sorted(set(gt_counts) | set(det_counts))
    rows = []
    for cls in classes:
        gt_n = gt_counts.get(cls, 0)
        det_n = det_counts.get(cls, 0)
        recall = f"{det_n / gt_n:.2f}" if gt_n > 0 else "n/a"
        rows.append((cls, gt_n, det_n, recall))

    header = ("class", "gt_count", "detected_count", "recall")
    widths = [
        max(len(header[i]), *(len(str(r[i])) for r in rows)) if rows else len(header[i])
        for i in range(4)
    ]
    lines = [
        " | ".join(h.ljust(widths[i]) for i, h in enumerate(header)),
        "-+-".join("-" * w for w in widths),
    ]
    for row in rows:
        lines.append(" | ".join(str(v).ljust(widths[i]) for i, v in enumerate(row)))

    gt_total = sum(gt_counts.values())
    det_total = sum(det_counts.get(c, 0) for c in gt_counts)
    overall = f"{det_total / gt_total:.2f}" if gt_total > 0 else "n/a"
    lines.append("")
    lines.append(
        f"overall: {det_total}/{gt_total} GT instances have >=1 detection-derived"
        f" count in-class (recall={overall}); {len(classes)} classes compared"
        f" ({sum(1 for c in classes if c not in gt_counts)} detected-only,"
        f" {sum(1 for c in classes if c not in det_counts)} GT-only / zero-recall)"
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("jsonl", type=Path, help="explore_debug*.jsonl capture")
    ap.add_argument("scene", help="GT scene name (e.g. office_1)")
    ap.add_argument(
        "--gt-dir",
        type=Path,
        default=DEFAULT_GROUNDTRUTH,
        help=f"VLA-3D Unity scenes root (default: {DEFAULT_GROUNDTRUTH})",
    )
    args = ap.parse_args(argv)

    record = _load_last_record(args.jsonl)
    if record is None:
        print(f"no well-formed JSON records in {args.jsonl}", file=sys.stderr)
        return 1

    if "live_instances" not in record:
        print(
            f"{args.jsonl}: last record has no 'live_instances' field — this capture "
            "predates the issue #84 explore_debug addition (dumped by an older build "
            "of core.heads.explore_debug). Nothing to compare; re-capture with a "
            "current build to get live-instance-index recall.",
        )
        return 0

    scene_dir = args.gt_dir / args.scene
    if not scene_dir.is_dir():
        print(f"GT scene folder not found: {scene_dir}", file=sys.stderr)
        return 1

    gt_counts = _gt_class_counts(scene_dir)
    det_counts = _detected_class_counts(record)
    print(
        f"# {args.jsonl.name} vs GT scene '{args.scene}' "
        f"(record t={record.get('question_clock_t')}, "
        f"keyframes_processed={record.get('keyframes_processed')})\n"
    )
    print(render_table(gt_counts, det_counts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
