"""Provenance stamp tests (meth-F7/F8)."""
from __future__ import annotations

import re
import subprocess

from core.runner import provenance as prov

#: Every key the stamp schema guarantees is always present.
SCHEMA_KEYS = {
    "tool",
    "generated_utc",
    "argv",
    "git_commit",
    "git_branch",
    "git_dirty",
    "dirty_digest",
    "calibration_sha1",
    "calibration",
    "note",
}


def test_stamp_has_every_schema_key():
    st = prov.collect_provenance("gt_battery", argv=["--groundtruth", "x"])
    assert set(st) == SCHEMA_KEYS
    assert st["tool"] == "gt_battery"
    assert st["argv"] == ["--groundtruth", "x"]
    assert isinstance(st["calibration"], dict)
    # ISO-8601 UTC timestamp (offset-aware).
    assert st["generated_utc"].endswith("+00:00")


def test_argv_none_serialises_null():
    st = prov.collect_provenance("cvsweep")
    assert st["argv"] is None


def test_in_repo_run_populates_git_and_calibration():
    st = prov.collect_provenance("gt_battery")
    # Running inside the repo: a real commit hash and a clean 12-hex calibration digest.
    assert st["git_commit"] is not None
    assert re.fullmatch(r"[0-9a-f]{40}", st["git_commit"])
    assert isinstance(st["git_dirty"], bool)
    assert st["calibration_sha1"] is not None
    assert re.fullmatch(r"[0-9a-f]{12}", st["calibration_sha1"])


def test_calibration_digest_matches_serialised_snapshot():
    import hashlib

    from core import calibration as C

    st = prov.collect_provenance("gt_battery")
    expected = hashlib.sha1(C.to_json(C.default_calibration()).encode("utf-8")).hexdigest()[:12]
    assert st["calibration_sha1"] == expected


def _init_temp_repo(root):
    """git-init a throwaway repo at ``root`` with one committed file."""
    def g(*args):
        subprocess.run(
            ["git", *args], cwd=str(root), check=True,
            capture_output=True, text=True,
        )

    g("init")
    g("config", "user.email", "t@example.com")
    g("config", "user.name", "t")
    (root / "committed.txt").write_text("base\n", encoding="utf-8")
    g("add", "committed.txt")
    g("commit", "-m", "base")


def test_dirty_digest_distinguishes_untracked_content(tmp_path, monkeypatch):
    """Two trees differing only inside an untracked file's CONTENT get distinct
    dirty_digests (issue #18) — untracked bytes are folded into the digest, not
    just the filename via `git status`."""
    _init_temp_repo(tmp_path)
    # Point provenance at the throwaway repo (module global drives cwd + reads).
    monkeypatch.setattr(prov, "_REPO_ROOT", tmp_path)

    untracked = tmp_path / "note.txt"

    untracked.write_text("alpha\n", encoding="utf-8")
    st_a = prov.collect_provenance("gt_battery")

    untracked.write_text("beta\n", encoding="utf-8")
    st_b = prov.collect_provenance("gt_battery")

    assert st_a["git_dirty"] is True
    assert st_b["git_dirty"] is True
    # Filename is identical in both `git status` outputs; only content differs.
    assert st_a["dirty_digest"] is not None
    assert st_a["dirty_digest"] != st_b["dirty_digest"]
    assert re.fullmatch(r"[0-9a-f]{12}", st_a["dirty_digest"])


def test_dirty_digest_untracked_unreadable_does_not_raise(tmp_path, monkeypatch):
    """An untracked path that vanishes / is a directory degrades to name-only
    without raising, preserving the non-raising invariant."""
    _init_temp_repo(tmp_path)
    monkeypatch.setattr(prov, "_REPO_ROOT", tmp_path)

    # An untracked directory (porcelain collapses it to `subdir/`, name-only hash)
    # plus an untracked file — exercises both branches.
    (tmp_path / "subdir").mkdir()
    (tmp_path / "subdir" / "inner.txt").write_text("x\n", encoding="utf-8")
    (tmp_path / "top.txt").write_text("y\n", encoding="utf-8")

    st = prov.collect_provenance("gt_battery")
    assert set(st) == SCHEMA_KEYS
    assert st["git_dirty"] is True
    assert re.fullmatch(r"[0-9a-f]{12}", st["dirty_digest"])


def test_graceful_degradation_when_git_unavailable(monkeypatch):
    """Any subprocess failure degrades git fields to None + a note, never raising."""

    def boom(*args, **kwargs):
        raise OSError("git not found")

    monkeypatch.setattr(prov.subprocess, "run", boom)
    st = prov.collect_provenance("gt_battery")
    assert set(st) == SCHEMA_KEYS
    assert st["git_commit"] is None
    assert st["git_branch"] is None
    assert st["git_dirty"] is None
    assert st["dirty_digest"] is None
    assert st["note"]  # degradation reasons recorded
    # Calibration does not touch subprocess, so it still resolves.
    assert re.fullmatch(r"[0-9a-f]{12}", st["calibration_sha1"])
