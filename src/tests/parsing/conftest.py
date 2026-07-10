"""Shared fixtures/helpers for parsing tests: golden loading + noun extraction."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).resolve().parents[2] / "core" / "parsing" / "fixtures"
QUESTIONS_JSON = (
    Path(__file__).resolve().parents[3]
    / "upstream" / "CMU-VLN-Challenge-2026" / "questions" / "questions.json"
)


def load_goldens() -> list[dict]:
    """Load every golden fixture: [{'scene', 'question', 'plan'}, ...]."""
    files = sorted(FIXTURES_DIR.glob("*.json"))
    assert files, f"no golden fixtures under {FIXTURES_DIR}"
    return [json.loads(p.read_text(encoding="utf-8")) | {"_file": p.name} for p in files]


def anchor_nouns(anchor: dict) -> list[str]:
    """Flatten one anchor dict into its noun plus all nested disambiguator nouns."""
    nouns = [anchor["noun"]]
    dis = anchor.get("disambiguator")
    if dis:
        for a in dis.get("anchors", []):
            nouns.extend(anchor_nouns(a))
    return nouns


def plan_nouns(plan: dict) -> Counter:
    """Multiset of every target/leg/clause/avoid noun in a Plan dict."""
    nouns: list[str] = []
    tgt = plan.get("target")
    if tgt:
        nouns.append(tgt["noun"])
        for cl in tgt.get("clauses", []):
            for a in cl.get("anchors", []):
                nouns.extend(anchor_nouns(a))
    for leg in plan.get("route", []):
        for a in leg.get("anchors", []):
            nouns.extend(anchor_nouns(a))
    for av in plan.get("avoid", []):
        for a in av.get("between") or []:
            nouns.extend(anchor_nouns(a))
        if av.get("near"):
            nouns.extend(anchor_nouns(av["near"]))
    return Counter(nouns)


@pytest.fixture(scope="session")
def goldens() -> list[dict]:
    """All golden fixtures."""
    return load_goldens()


@pytest.fixture(scope="session")
def all_questions() -> list[tuple[str, str, str]]:
    """All 75 training questions as (scene, qtype_label, question_text)."""
    assert QUESTIONS_JSON.exists(), f"missing training questions at {QUESTIONS_JSON}"
    data = json.loads(QUESTIONS_JSON.read_text(encoding="utf-8"))
    out = []
    for entry in data:
        for qtype_label, qs in entry["questions"].items():
            for q in qs:
                out.append((entry["scene"], qtype_label, q))
    assert len(out) == 75, f"expected 75 training questions, found {len(out)}"
    return out
