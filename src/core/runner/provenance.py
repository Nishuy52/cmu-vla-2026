"""Provenance stamps for battery / sweep results JSON (meth-F7/F8).

Battery and sweep numbers used to ship with no record of where they came from,
which is exactly the gap a fabricated-attribution incident exploited. This module
makes every results payload self-describing: which tool produced it, when, from
which commit and working-tree state, and against which calibration.

:func:`collect_provenance` is a hard non-raising boundary — any git/subprocess
failure degrades the affected fields to ``None`` and appends a human-readable
reason to ``"note"`` — so stamping can never take down a battery run. Every schema
key is always present.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from core import calibration as _cal

if TYPE_CHECKING:
    from core.calibration import Calibration

#: Repo root: src/core/runner/provenance.py -> parents[3] == the git working tree.
_REPO_ROOT = Path(__file__).resolve().parents[3]


def _git(args: list[str]) -> str:
    """Run a git command at the repo root and return stripped stdout.

    Raises on a non-zero exit / missing git — the caller degrades gracefully.
    """
    out = subprocess.run(
        ["git", *args],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.strip()


def _untracked_content_digest(status_porcelain: str) -> str:
    """Deterministic digest input for the CONTENTS of untracked paths.

    ``status_porcelain`` is the stripped ``git status --porcelain`` output.
    Untracked entries are the lines starting with ``"?? "``. For each such path,
    in sorted order, we fold in the file bytes so two trees that differ only
    inside an untracked file get distinct digests (issue #18).

    Untracked *directories* appear in porcelain collapsed to a single ``dir/``
    entry (git does not expand their contents), so we deliberately hash the
    directory name only — walking them could be arbitrarily large (e.g. an
    untracked ``reports/`` tree) and is a performance hazard for a stamp that
    runs on every battery. This is a documented, bounded trade-off: content
    changes strictly *inside* an untracked directory are not distinguished.

    Never raises: an unreadable path (deleted mid-flight, permission error, a
    path that resolves to a directory) degrades to hashing its name only.
    """
    parts: list[str] = []
    for line in sorted(status_porcelain.splitlines()):
        if not line.startswith("?? "):
            continue
        rel = line[3:]
        # Porcelain quotes paths with special characters (core.quotePath);
        # such a path won't resolve on disk, so its hash falls back to the
        # raw (quoted) name below — still deterministic.
        if rel.endswith("/"):
            # Untracked directory: hash the name only (see docstring).
            parts.append(f"{rel}\0dir")
            continue
        try:
            data = (_REPO_ROOT / rel).read_bytes()
            parts.append(f"{rel}\0" + hashlib.sha1(data).hexdigest())
        except Exception:  # noqa: BLE001 — provenance must never raise
            # Unreadable / vanished / directory: degrade to name only.
            parts.append(f"{rel}\0?")
    return "\n".join(parts)


def collect_provenance(
    tool: str,
    argv: list[str] | None = None,
    cal: "Calibration | None" = None,
) -> dict:
    """Provenance stamp for results JSON (meth-F7/F8).

    Never raises: any git/subprocess failure degrades the affected
    fields to None and appends a human-readable reason to "note".
    """
    notes: list[str] = []

    # Calibration identity is pure (no subprocess), so it is computed outside the
    # git try-blocks; still guarded so the whole function is genuinely non-raising.
    calibration_sha1: str | None = None
    calibration: dict | None = None
    try:
        if cal is None:
            cal = _cal.default_calibration()
        cal_json = _cal.to_json(cal)
        calibration_sha1 = hashlib.sha1(cal_json.encode("utf-8")).hexdigest()[:12]
        calibration = json.loads(cal_json)
    except Exception as exc:  # noqa: BLE001 — provenance must never raise
        notes.append(f"calibration snapshot unavailable ({exc.__class__.__name__})")

    git_commit: str | None = None
    git_branch: str | None = None
    git_dirty: bool | None = None
    dirty_digest: str | None = None
    try:
        git_commit = _git(["rev-parse", "HEAD"])
    except Exception as exc:  # noqa: BLE001
        notes.append(f"git commit unavailable ({exc.__class__.__name__})")
    try:
        git_branch = _git(["rev-parse", "--abbrev-ref", "HEAD"])
    except Exception as exc:  # noqa: BLE001
        notes.append(f"git branch unavailable ({exc.__class__.__name__})")
    try:
        status = _git(["status", "--porcelain"])
        git_dirty = bool(status)
        if git_dirty:
            # Digest identifying the dirty working tree without dumping it into
            # the payload. Covers: the porcelain status line set (tracked path
            # changes + untracked path names), the staged+unstaged diff vs HEAD
            # for tracked files, and the CONTENTS of untracked files (issue #18)
            # — since `git diff HEAD` omits untracked files, their bytes are
            # folded in separately (untracked dirs by name only; see
            # _untracked_content_digest).
            diff_text = _git(["diff", "HEAD"])
            untracked_text = _untracked_content_digest(status)
            dirty_digest = hashlib.sha1(
                (status + "\0" + diff_text + "\0" + untracked_text).encode("utf-8")
            ).hexdigest()[:12]
    except Exception as exc:  # noqa: BLE001
        notes.append(f"git status unavailable ({exc.__class__.__name__})")

    return {
        "tool": tool,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "argv": list(argv) if argv is not None else None,
        "git_commit": git_commit,
        "git_branch": git_branch,
        "git_dirty": git_dirty,
        "dirty_digest": dirty_digest,
        "calibration_sha1": calibration_sha1,
        "calibration": calibration,
        "note": "; ".join(notes),
    }


__all__ = ["collect_provenance"]
