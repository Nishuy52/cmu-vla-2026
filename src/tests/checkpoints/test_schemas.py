"""Shared JSON schema layer: extraction, validators, one-repair loop."""
from __future__ import annotations

import pytest

from core.checkpoints import schemas


def test_extract_json_tolerates_fences_and_preamble():
    raw = "sure, here:\n```json\n{\"verdict\": \"confirm\"}\n```"
    assert schemas.extract_json(raw) == '{"verdict": "confirm"}'


def test_extract_json_nested_braces():
    raw = 'x {"a": {"b": 1}, "c": 2} trailing'
    assert schemas.loads(raw) == {"a": {"b": 1}, "c": 2}


def test_extract_json_raises_on_no_object():
    with pytest.raises(ValueError):
        schemas.extract_json("no braces here")


def test_validate_verification_ok_and_bad():
    assert schemas.validate_verification(
        {"verdict": "confirm", "missed_constraint": None, "reason": "r"}
    ) == []
    errs = schemas.validate_verification({"verdict": "maybe"})
    assert any("verdict" in e for e in errs)


def test_validate_miss_recovery_bounds():
    assert schemas.validate_miss_recovery(
        {"present": True, "tile": 3, "bbox_hint": [1, 2, 3, 4], "confidence": 0.5}
    ) == []
    errs = schemas.validate_miss_recovery(
        {"present": True, "tile": 7, "bbox_hint": [1, 2], "confidence": 2.0}
    )
    assert len(errs) == 3


def test_validate_anchor_confirm():
    assert schemas.validate_anchor_confirm(
        {"match": True, "actual_label": None, "confidence": 1.0}
    ) == []
    assert schemas.validate_anchor_confirm({"match": "yes", "confidence": 0.5})


def test_validate_frontier_select_range():
    assert schemas.validate_frontier_select({"choice": 3, "reason": "r"}, n_choices=5) == []
    assert schemas.validate_frontier_select({"choice": 6, "reason": "r"}, n_choices=5)


def test_bool_is_not_number_for_confidence():
    # True must not pass as a confidence number
    errs = schemas.validate_anchor_confirm({"match": True, "confidence": True})
    assert any("confidence" in e for e in errs)


def test_parse_with_repair_success_first_pass():
    obj, errs = schemas.parse_with_repair(
        '{"verdict": "confirm", "reason": "r"}', schemas.validate_verification
    )
    assert obj is not None and errs == []


def test_parse_with_repair_one_repair_then_valid():
    calls = {"n": 0}

    def repair(raw, errors):
        calls["n"] += 1
        return '{"verdict": "confirm", "reason": "fixed"}'

    obj, errs = schemas.parse_with_repair("garbage", schemas.validate_verification, repair)
    assert obj is not None and calls["n"] == 1


def test_parse_with_repair_gives_up_after_one():
    calls = {"n": 0}

    def repair(raw, errors):
        calls["n"] += 1
        return "still garbage"

    obj, errs = schemas.parse_with_repair("garbage", schemas.validate_verification, repair)
    assert obj is None
    assert calls["n"] == 1  # exactly one repair round, not a loop


def test_parse_with_repair_no_repair_callable():
    obj, errs = schemas.parse_with_repair("garbage", schemas.validate_verification, None)
    assert obj is None and errs


def test_parse_with_repair_swallows_repair_exception():
    def repair(raw, errors):
        raise RuntimeError("dead provider")

    obj, errs = schemas.parse_with_repair("garbage", schemas.validate_verification, repair)
    assert obj is None
    assert any("repair call failed" in e for e in errs)
