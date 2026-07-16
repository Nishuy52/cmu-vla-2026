"""Diff two gt_battery results JSON files into a markdown before/after report.

Adjudicated fix for meth-F8: agent-authored prose once introduced fabricated
numbers into a report. This tool's output is the ONLY legal source of
before/after battery tables in reports -- no hand-typed numbers.

Tolerates schema drift: older results predate both the "provenance" stamp and
newer aggregate/row keys (meth-F7). All row and aggregate fields are treated
generically -- nothing is hardcoded to a specific key list.

Usage (from the repo root)::

    python -m tools.battery_diff <before_results.json> <after_results.json> [--out <path.md>]

Pure stdlib. Offline dev tool -- not part of the scored pipeline.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROW_KEY_FIELDS = ("scene", "qtype", "question")


def _load(path: Path) -> dict:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as e:
        raise ValueError(f"cannot read {path}: {e}") from e
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"invalid JSON in {path}: {e}") from e
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object at top level")
    return data


def _fmt_val(x) -> str:
    """Render a scalar value for display (numbers trimmed to <=4 decimals)."""
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, float):
        return f"{x:.4f}"
    return str(x)


def _fmt_delta(before, after) -> str:
    d = after - before
    return f"{'+' if d >= 0 else ''}{d:.4f}"


def _is_number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _flatten_leaves(d: dict, prefix: str = "") -> dict:
    """Flatten nested dicts to dotted keys -> leaf value (any type)."""
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten_leaves(v, key + "."))
        else:
            out[key] = v
    return out


def _provenance_block(data: dict, label: str) -> list[str]:
    prov = data.get("provenance")
    lines = [f"### {label}"]
    if not isinstance(prov, dict):
        lines.append("")
        lines.append("UNPROVENANCED (pre-meth-F7 run)")
        lines.append("")
        return lines
    tool = prov.get("tool", "—")
    generated = prov.get("generated_utc", "—")
    commit = prov.get("git_commit", "—")
    commit_short = commit[:12] if isinstance(commit, str) else commit
    dirty = prov.get("git_dirty", "—")
    calib = prov.get("calibration_sha1", "—")
    lines.append("")
    lines.append(f"- tool: {tool}")
    lines.append(f"- generated_utc: {generated}")
    lines.append(f"- git_commit: {commit_short}")
    lines.append(f"- git_dirty: {dirty}")
    lines.append(f"- calibration_sha1: {calib}")
    lines.append("")
    return lines


def _topline_section(before: dict, after: dict) -> tuple[list[str], bool]:
    before_leaves = _flatten_leaves(before.get("aggregate") or {})
    after_leaves = _flatten_leaves(after.get("aggregate") or {})
    keys = sorted(set(before_leaves) | set(after_leaves))

    rows = []
    any_change = False
    for k in keys:
        has_b = k in before_leaves
        has_a = k in after_leaves
        b = before_leaves.get(k)
        a = after_leaves.get(k)

        if has_b and has_a and _is_number(b) and _is_number(a):
            delta = _fmt_delta(float(b), float(a))
            if float(b) != float(a):
                any_change = True
            rows.append((k, _fmt_val(b), _fmt_val(a), delta))
        elif _is_number(b) or _is_number(a):
            # one-sided numeric leaf
            b_s = _fmt_val(b) if has_b and _is_number(b) else "—"
            a_s = _fmt_val(a) if has_a and _is_number(a) else "—"
            rows.append((k, b_s, a_s, "—"))
            any_change = True
        else:
            # non-numeric leaf (list/dict/string/None): skip unless it differs
            if has_b and has_a and b == a:
                continue
            if not has_b and not has_a:
                continue
            any_change = True
            b_s = json.dumps(b) if has_b else "—"
            a_s = json.dumps(a) if has_a else "—"
            rows.append((k, b_s, a_s, "n/a"))

    lines = ["## Topline", ""]
    if not rows:
        lines.append("(no aggregate leaves found)")
        lines.append("")
        return lines, any_change

    lines.append("| Key | Before | After | Δ |")
    lines.append("|---|---|---|---|")
    for k, b_s, a_s, d_s in rows:
        lines.append(f"| {k} | {b_s} | {a_s} | {d_s} |")
    lines.append("")
    return lines, any_change


def _row_key(row: dict) -> tuple:
    return tuple(row.get(f) for f in _ROW_KEY_FIELDS)


def _index_rows(data: dict) -> dict:
    scores = data.get("scores") or []
    idx = {}
    for row in scores:
        if isinstance(row, dict):
            idx[_row_key(row)] = row
    return idx


def _rows_section(before: dict, after: dict) -> tuple[list[str], bool]:
    before_idx = _index_rows(before)
    after_idx = _index_rows(after)

    common_keys = sorted(set(before_idx) & set(after_idx), key=lambda k: [str(x) for x in k])
    added_keys = sorted(set(after_idx) - set(before_idx), key=lambda k: [str(x) for x in k])
    removed_keys = sorted(set(before_idx) - set(after_idx), key=lambda k: [str(x) for x in k])

    changed_lines = ["## Changed rows", ""]
    any_change = bool(added_keys or removed_keys)
    changed_entries = []
    for key in common_keys:
        b_row = before_idx[key]
        a_row = after_idx[key]
        fields = sorted((set(b_row) | set(a_row)) - {"note"})
        diffs = []
        for f in fields:
            bv = b_row.get(f)
            av = a_row.get(f)
            if bv != av:
                diffs.append(f"{f}: {json.dumps(bv)} -> {json.dumps(av)}")
        if diffs:
            any_change = True
            scene, qtype, question = key
            changed_entries.append((scene, qtype, question, "; ".join(diffs)))

    if changed_entries:
        changed_lines.append("| Scene | Qtype | Question | Changed fields |")
        changed_lines.append("|---|---|---|---|")
        for scene, qtype, question, diffs in changed_entries:
            changed_lines.append(f"| {scene} | {qtype} | {question} | {diffs} |")
        changed_lines.append("")
    else:
        changed_lines.append("(no changed rows)")
        changed_lines.append("")

    added_removed_lines = ["## Added / removed rows", ""]
    if added_keys:
        added_removed_lines.append("### Added")
        added_removed_lines.append("")
        for scene, qtype, question in added_keys:
            added_removed_lines.append(f"- ({scene}, {qtype}, {question})")
        added_removed_lines.append("")
    if removed_keys:
        added_removed_lines.append("### Removed")
        added_removed_lines.append("")
        for scene, qtype, question in removed_keys:
            added_removed_lines.append(f"- ({scene}, {qtype}, {question})")
        added_removed_lines.append("")
    if not added_keys and not removed_keys:
        added_removed_lines.append("(none)")
        added_removed_lines.append("")

    return changed_lines + added_removed_lines, any_change


def build_report(before: dict, after: dict) -> str:
    lines = ["# Battery diff", ""]
    lines.append("## Provenance")
    lines.append("")
    lines.extend(_provenance_block(before, "Before"))
    lines.extend(_provenance_block(after, "After"))

    topline_lines, topline_changed = _topline_section(before, after)
    rows_lines, rows_changed = _rows_section(before, after)

    if not topline_changed and not rows_changed:
        lines.append("no changes")
        lines.append("")

    lines.extend(topline_lines)
    lines.extend(rows_lines)
    return "\n".join(lines).rstrip() + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("before", type=Path)
    ap.add_argument("after", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    try:
        before = _load(args.before)
        after = _load(args.after)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    report = build_report(before, after)

    if args.out is not None:
        args.out.write_text(report, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass
        print(report)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
