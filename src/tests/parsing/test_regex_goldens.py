"""Regex tier vs the hand-authored goldens: qtype, nouns, corridor/avoid detection."""
from __future__ import annotations

import json

from core.parsing.regex_tier import parse_regex
from tests.parsing.conftest import anchor_nouns, plan_nouns


def _parsed(golden: dict) -> dict:
    plan = parse_regex(golden["question"])
    assert plan.validate() == [], f"{golden['_file']}: invalid parse {plan.validate()}"
    return json.loads(plan.to_json())


def test_goldens_present_and_typed(goldens):
    """Golden coverage: all 15 numerical, >=8 object_reference, >=8 instruction_following."""
    by_type = {"numerical": 0, "object_reference": 0, "instruction_following": 0}
    for g in goldens:
        by_type[g["plan"]["qtype"]] += 1
    assert by_type["numerical"] == 15
    assert by_type["object_reference"] >= 8
    assert by_type["instruction_following"] >= 8
    assert len(goldens) >= 20


def test_qtype_100_percent(goldens):
    """The regex tier classifies every golden's qtype correctly."""
    wrong = []
    for g in goldens:
        got = _parsed(g)["qtype"]
        if got != g["plan"]["qtype"]:
            wrong.append((g["_file"], got))
    assert not wrong, f"qtype misclassifications: {wrong}"


def test_noun_match_at_least_90_percent(goldens):
    """Target/leg-anchor nouns from the regex tier match >=90% of golden nouns overall."""
    matched = total = 0
    misses = []
    for g in goldens:
        want = plan_nouns(g["plan"])
        got = plan_nouns(_parsed(g))
        inter = want & got
        matched += sum(inter.values())
        total += sum(want.values())
        if inter != want:
            misses.append((g["_file"], sorted((want - inter).elements())))
    ratio = matched / total
    assert ratio >= 0.90, f"noun match {ratio:.1%} ({matched}/{total}); misses: {misses}"


def test_corridor_detection_100_percent(goldens):
    """Every golden corridor leg is detected, with the same anchor-noun pair."""
    for g in goldens:
        want_pairs = [
            frozenset(a["noun"] for a in leg["anchors"])
            for leg in g["plan"].get("route", [])
            if leg["kind"] == "corridor_between"
        ]
        if not want_pairs:
            continue
        got = _parsed(g)
        got_pairs = [
            frozenset(a["noun"] for a in leg["anchors"])
            for leg in got.get("route", [])
            if leg["kind"] == "corridor_between"
        ]
        assert sorted(map(sorted, got_pairs)) == sorted(map(sorted, want_pairs)), (
            f"{g['_file']}: corridor mismatch want {want_pairs} got {got_pairs}"
        )


def test_avoid_detection_100_percent(goldens):
    """Every golden avoid spec is detected with the right kind and anchor nouns."""
    for g in goldens:
        want = g["plan"].get("avoid", [])
        if not want:
            continue
        got = _parsed(g).get("avoid", [])
        assert len(got) == len(want), f"{g['_file']}: avoid count {len(got)} != {len(want)}"
        for w, p in zip(want, got):
            if w.get("between"):
                assert p.get("between"), f"{g['_file']}: expected between-avoid"
                assert {a["noun"] for a in p["between"]} == {
                    a["noun"] for a in w["between"]
                }, f"{g['_file']}: avoid anchors differ"
            else:
                assert p.get("near"), f"{g['_file']}: expected near-avoid"
                assert p["near"]["noun"] == w["near"]["noun"]


def test_leg_order_and_kinds(goldens):
    """Regex-tier leg kind sequences match the goldens (ordering is penalty-scored)."""
    wrong = []
    for g in goldens:
        want = [leg["kind"] for leg in g["plan"].get("route", [])]
        if not want:
            continue
        got = [leg["kind"] for leg in _parsed(g).get("route", [])]
        if got != want:
            wrong.append((g["_file"], want, got))
    assert not wrong, f"leg-kind sequence mismatches: {wrong}"


def test_goldens_themselves_validate(goldens):
    """Every stored golden round-trips through Plan.from_json and validates."""
    from core.plan_schema import Plan

    for g in goldens:
        plan = Plan.from_json(json.dumps(g["plan"]))
        assert plan.validate() == [], f"{g['_file']}: golden invalid {plan.validate()}"
        for leaf in anchor_nouns(
            {"noun": "x", "disambiguator": None}
        ):  # sanity of helper
            assert isinstance(leaf, str)
