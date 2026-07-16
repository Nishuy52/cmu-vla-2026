"""IF leg-count census: hand tally vs. the pipeline's own parse (meth-F5).

The evaluation instrument (``_if_rubric_geometry`` in ``core.runner.gt_battery``)
builds each instruction-following plan by calling ``parse_regex(text)`` and then
reads ``plan.route`` / ``plan.avoid``. Because scoring and pipeline share that one
parser, a silently dropped route leg is invisible: both sides agree on the same
wrong plan. This test breaks the circularity by comparing ``parse_regex`` output
against an INDEPENDENT, hand-tallied count committed in ``if_leg_census.json``
(built by reading the question text, never by running the parser).

Rows where the parser disagrees with the hand tally today are marked
``xfail(strict=True)`` so the census documents the current dropped-leg gaps
without reddening the gate; any change in parser behaviour flips them loudly.
As of 2026-07-17 the parser agrees with the hand tally on all 30 questions, so
there are no xfails — the fixture then guards against future regressions.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

# src/ -> repo root (…/2026 CMU VLA) -> upstream/CMU-VLN-Challenge-2026/questions
_SRC = Path(__file__).resolve().parents[2]
_REPO = _SRC.parent
QUESTIONS_JSON = _REPO / "upstream" / "CMU-VLN-Challenge-2026" / "questions" / "questions.json"
_CENSUS_PATH = Path(__file__).with_name("if_leg_census.json")

_have_upstream = QUESTIONS_JSON.exists()
requires_upstream = pytest.mark.skipif(
    not _have_upstream, reason="upstream questions.json not present"
)

# Scenes/indices whose parse disagrees with the hand tally today. Determined
# empirically by running this test; kept honest (the tally wins on disagreement).
_XFAIL: dict[tuple[str, int], str] = {}


def _load_census() -> list[dict]:
    with open(_CENSUS_PATH, encoding="utf-8") as fh:
        return json.load(fh)["census"]


def _census_id(row: dict) -> str:
    return f"{row['scene']}-q{row['q_index']}"


def _census_params() -> list:
    params = []
    for row in _load_census():
        key = (row["scene"], row["q_index"])
        marks = ()
        if key in _XFAIL:
            marks = (pytest.mark.xfail(strict=True, reason=_XFAIL[key]),)
        params.append(pytest.param(row, id=_census_id(row), marks=marks))
    return params


def test_census_covers_all_if_questions():
    """The fixture must tally exactly the 30 upstream IF questions (15 scenes x 2)."""
    if not _have_upstream:
        pytest.skip("upstream questions.json not present")
    data = json.load(open(QUESTIONS_JSON, encoding="utf-8"))
    upstream = {
        (entry["scene"], qi)
        for entry in data
        for qi in range(len(entry["questions"]["instruction_following"]))
    }
    census = {(r["scene"], r["q_index"]) for r in _load_census()}
    assert census == upstream, (
        "census must cover exactly the upstream IF questions; "
        f"missing={upstream - census} extra={census - upstream}"
    )
    # The committed question text must still match upstream verbatim.
    text_by_key = {
        (entry["scene"], qi): q
        for entry in data
        for qi, q in enumerate(entry["questions"]["instruction_following"])
    }
    for row in _load_census():
        assert row["question"] == text_by_key[(row["scene"], row["q_index"])], (
            f"{_census_id(row)} question text drifted from upstream"
        )


@requires_upstream
@pytest.mark.parametrize("row", _census_params())
def test_leg_counts_match_hand_tally(row: dict):
    """``parse_regex`` (the plan builder the rubric uses) must produce exactly the
    hand-tallied number of route legs and avoid clauses for each IF question."""
    from core.parsing.regex_tier import parse_regex

    plan = parse_regex(row["question"])
    assert len(plan.route) == row["n_route_legs"], (
        f"{_census_id(row)}: route legs parser={len(plan.route)} "
        f"hand_tally={row['n_route_legs']} :: {row['question']}"
    )
    assert len(plan.avoid) == row["n_avoid"], (
        f"{_census_id(row)}: avoid clauses parser={len(plan.avoid)} "
        f"hand_tally={row['n_avoid']} :: {row['question']}"
    )
