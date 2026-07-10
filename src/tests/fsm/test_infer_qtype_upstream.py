"""_infer_qtype heuristic vs. the upstream questions.json ground truth.

Skips gracefully if the upstream repo/file isn't present in this checkout.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.fsm.controller import _infer_qtype
from core.interfaces import Question, QType

_UPSTREAM_PATH = (
    Path(__file__).resolve().parents[2]
    / "upstream"
    / "CMU-VLN-Challenge-2026"
    / "questions"
    / "questions.json"
)

_QTYPE_BY_KEY = {
    "numerical": QType.NUMERICAL,
    "object_reference": QType.OBJECT_REFERENCE,
    "instruction_following": QType.INSTRUCTION_FOLLOWING,
}


def _load_upstream_cases():
    if not _UPSTREAM_PATH.is_file():
        return None
    data = json.loads(_UPSTREAM_PATH.read_text(encoding="utf-8"))
    cases = []
    for scene in data:
        for key, texts in scene["questions"].items():
            want = _QTYPE_BY_KEY[key]
            for text in texts:
                cases.append((scene.get("scene", "?"), text, want))
    return cases


_CASES = _load_upstream_cases()

if _CASES is None:
    pytestmark = pytest.mark.skip(reason="upstream questions.json not found in this checkout")
    _CASES = []


@pytest.mark.parametrize("scene,text,want", _CASES, ids=lambda v: v if isinstance(v, str) else None)
def test_infer_qtype_matches_upstream_ground_truth(scene, text, want):
    q = Question(text=text, t_received=0.0)
    got = _infer_qtype(q)
    assert got is want, f"[{scene}] {text!r}: expected {want}, got {got}"


def test_upstream_dataset_has_75_questions():
    if not _UPSTREAM_PATH.is_file():
        pytest.skip("upstream questions.json not found in this checkout")
    assert len(_CASES) == 75
