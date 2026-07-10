"""End-to-end: build_chat_fns -> ladder.parse with a LocalStub returning valid Plan JSON.

Ties the provider layer to the parse ladder without any network: a stub slot returns a
schema-valid Plan and the ladder stamps the tier and returns a validating Plan.
"""
from __future__ import annotations

import json

from core.llm.config import LlmConfig, ProviderSpec, build_chat_fns
from core.parsing.ladder import parse
from core.plan_schema import Plan

QUESTION = "How many pillows are on the bed?"

VALID_PLAN_JSON = json.dumps(
    {
        "qtype": "numerical",
        "question_raw": QUESTION,
        "target": {
            "noun": "pillow",
            "raw": "pillows",
            "attributes": [],
            "clauses": [
                {
                    "pred": "on",
                    "anchors": [
                        {"noun": "bed", "raw": "bed", "attributes": [], "disambiguator": None}
                    ],
                    "negated": False,
                }
            ],
        },
        "route": [],
        "avoid": [],
        "notes": "",
        "parse_tier": "api",
    }
)


class FakeClock:
    def __init__(self, t: float = 0.0) -> None:
        self.t = t

    def now(self) -> float:
        return self.t


def test_primary_stub_drives_ladder_to_valid_plan():
    cfg = LlmConfig(primary=ProviderSpec(kind="stub", stub_reply=VALID_PLAN_JSON))
    fns = build_chat_fns(cfg)
    plan = parse(QUESTION, fns, FakeClock())
    assert isinstance(plan, Plan)
    assert plan.validate() == []
    assert plan.parse_tier == "api"  # first configured slot stamps the "api" tier
    assert plan.target is not None and plan.target.noun == "pillow"
    assert plan.question_raw == QUESTION


def test_secondary_stub_used_when_primary_returns_garbage():
    cfg = LlmConfig(
        primary=ProviderSpec(kind="stub", stub_reply="not json at all"),
        secondary=ProviderSpec(kind="stub", stub_reply=VALID_PLAN_JSON),
    )
    fns = build_chat_fns(cfg)
    plan = parse(QUESTION, fns, FakeClock())
    assert plan.validate() == []
    assert plan.parse_tier == "api2"  # second slot -> api2 tier stamp


def test_empty_config_falls_to_regex_floor():
    fns = build_chat_fns(LlmConfig())
    plan = parse(QUESTION, fns, FakeClock())
    assert plan.parse_tier == "regex"
    assert plan.validate() == []
