"""Shared fixtures for ground-truth loader/scoring tests (real loft sample)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

# src/ -> repo root (…/2026 CMU VLA) -> upstream/VLA-3D/sample_data/Unity/loft
_SRC = Path(__file__).resolve().parents[2]
_REPO = _SRC.parent
LOFT_DIR = _REPO / "upstream" / "VLA-3D" / "sample_data" / "Unity" / "loft"
QUESTIONS_JSON = _REPO / "upstream" / "CMU-VLN-Challenge-2026" / "questions" / "questions.json"
QUESTIONS_DIR = QUESTIONS_JSON.parent
UNITY_SAMPLE_ROOT = _REPO / "upstream" / "VLA-3D" / "sample_data" / "Unity"

_have_loft = LOFT_DIR.is_dir() and (LOFT_DIR / "loft_object_result.csv").exists()
requires_loft = pytest.mark.skipif(not _have_loft, reason="loft sample data not present")


@pytest.fixture(scope="session")
def loft_dir() -> Path:
    if not _have_loft:
        pytest.skip("loft sample data not present")
    return LOFT_DIR


@pytest.fixture(scope="session")
def loft_referential() -> dict:
    p = LOFT_DIR / "loft_referential_statements.json"
    if not p.exists():
        pytest.skip("loft referential statements not present")
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)
