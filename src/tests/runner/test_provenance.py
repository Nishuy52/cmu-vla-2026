"""Provenance stamp tests (meth-F7/F8)."""
from __future__ import annotations

import re

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


def test_dirty_digest_changes_with_untracked_file_contents(tmp_path, monkeypatch):
    """dirty_digest must reflect untracked-file *contents*, not just their names."""
    import subprocess

    real_run = subprocess.run

    def fake_git(args, cwd, capture_output, text, check):
        if args[:1] == ["status"]:
            return subprocess.CompletedProcess(args, 0, stdout="?? scratch_untracked.txt\n", stderr="")
        if args[:1] == ["diff"]:
            return subprocess.CompletedProcess(args, 0, stdout="", stderr="")
        if args[:2] == ["ls-files", "--others"] or args[:1] == ["ls-files"]:
            return subprocess.CompletedProcess(args, 0, stdout="scratch_untracked.txt\n", stderr="")
        return real_run(["git", *args], cwd=cwd, capture_output=capture_output, text=text, check=check)

    def fake_subprocess_run(cmd, cwd, capture_output, text, check):
        assert cmd[0] == "git"
        return fake_git(cmd[1:], cwd, capture_output, text, check)

    monkeypatch.setattr(prov, "_REPO_ROOT", tmp_path)
    monkeypatch.setattr(prov.subprocess, "run", fake_subprocess_run)

    scratch = tmp_path / "scratch_untracked.txt"
    scratch.write_text("version one", encoding="utf-8")
    st1 = prov.collect_provenance("gt_battery")

    scratch.write_text("version two", encoding="utf-8")
    st2 = prov.collect_provenance("gt_battery")

    assert st1["dirty_digest"] is not None
    assert st1["dirty_digest"] != st2["dirty_digest"]


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
