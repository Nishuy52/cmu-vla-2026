"""JSON schema validators + parse-with-one-repair for the five checkpoint contracts.

Every checkpoint returns strict JSON (design doc "Shared rules"): one repair round on a
validation failure, then the deterministic fallback. This module owns the wire-format
layer shared by all four checkpoint modules:

* :func:`extract_json` — the ladder's balanced-brace object extractor. We import it from
  :mod:`core.parsing.ladder` when available and reimplement it minimally otherwise, so
  the checkpoints tolerate code fences / preamble around the model's JSON exactly like
  the parse ladder does.
* one ``validate_*`` per checkpoint — a total, dependency-free structural check returning
  a list of human-readable errors (empty == valid). No ``jsonschema`` dependency: these
  contracts are tiny and fixed, so a hand-written validator keeps the "no new hard deps"
  constraint.
* :func:`parse_with_repair` — the shared decode loop: extract+parse+validate; on failure,
  invoke the injected ``repair`` callable exactly once with the raw reply and the errors,
  then validate again; still-invalid returns ``(None, errors)`` and the caller applies its
  deterministic fallback.

No network, no I/O, deterministic.
"""
from __future__ import annotations

import json
from typing import Any, Callable

# --------------------------------------------------------------------------- extraction

try:  # prefer the ladder's proven extractor so behaviour matches checkpoint 1 exactly
    from core.parsing.ladder import _extract_json as _ladder_extract_json
except Exception:  # pragma: no cover - defensive; the ladder is always importable in-repo
    _ladder_extract_json = None


def extract_json(raw: str) -> str:
    """Return the first balanced JSON object substring in ``raw`` (tolerates fences/preamble).

    Delegates to the parse ladder's extractor when importable (single source of truth),
    else applies an equivalent minimal balanced-brace scan that ignores braces inside
    string literals. Raises ``ValueError`` when no balanced object is present.
    """
    if _ladder_extract_json is not None:
        return _ladder_extract_json(raw)
    return _extract_json_fallback(raw)


def _extract_json_fallback(raw: str) -> str:
    """Minimal reimplementation of the ladder's balanced-brace object extractor."""
    start = raw.find("{")
    if start < 0:
        raise ValueError("no JSON object in reply")
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(raw)):
        ch = raw[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return raw[start : i + 1]
    raise ValueError("unbalanced JSON object in reply")


def loads(raw: str) -> dict[str, Any]:
    """Extract + JSON-decode the reply into a dict. Raises on malformed / non-object."""
    obj = json.loads(extract_json(raw))
    if not isinstance(obj, dict):
        raise ValueError("checkpoint reply is not a JSON object")
    return obj


# --------------------------------------------------------------------------- validators
#
# Each validator takes the decoded dict and returns a list of error strings (empty ==
# valid). They are total: any structural surprise becomes an error, never an exception.


def _is_bool(v: Any) -> bool:
    return isinstance(v, bool)


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _is_conf(v: Any) -> bool:
    return _is_num(v) and 0.0 <= float(v) <= 1.0


def _opt_str(v: Any) -> bool:
    return v is None or isinstance(v, str)


def validate_verification(d: dict[str, Any]) -> list[str]:
    """CP4: {verdict: confirm|runner_up|neither, missed_constraint: str|null, reason: str}."""
    errs: list[str] = []
    verdict = d.get("verdict")
    if verdict not in ("confirm", "runner_up", "neither"):
        errs.append(f"verdict must be confirm|runner_up|neither, got {verdict!r}")
    if not _opt_str(d.get("missed_constraint")):
        errs.append("missed_constraint must be str|null")
    if not isinstance(d.get("reason", ""), str):
        errs.append("reason must be str")
    return errs


def validate_miss_recovery(d: dict[str, Any]) -> list[str]:
    """CP2: {present: bool, tile: 0-3|null, bbox_hint: [x1,y1,x2,y2]|null, confidence: 0-1}."""
    errs: list[str] = []
    if not _is_bool(d.get("present")):
        errs.append("present must be bool")
    tile = d.get("tile")
    if tile is not None and not (isinstance(tile, int) and not isinstance(tile, bool) and 0 <= tile <= 3):
        errs.append("tile must be int 0-3 or null")
    bbox = d.get("bbox_hint")
    if bbox is not None:
        if not (isinstance(bbox, list) and len(bbox) == 4 and all(_is_num(x) for x in bbox)):
            errs.append("bbox_hint must be [x1,y1,x2,y2] numbers or null")
    if not _is_conf(d.get("confidence")):
        errs.append("confidence must be a number in [0,1]")
    return errs


def validate_anchor_confirm(d: dict[str, Any]) -> list[str]:
    """CP3: {match: bool, actual_label: str|null, confidence: 0-1}."""
    errs: list[str] = []
    if not _is_bool(d.get("match")):
        errs.append("match must be bool")
    if not _opt_str(d.get("actual_label")):
        errs.append("actual_label must be str|null")
    if not _is_conf(d.get("confidence")):
        errs.append("confidence must be a number in [0,1]")
    return errs


def validate_frontier_select(d: dict[str, Any], *, n_choices: int = 5) -> list[str]:
    """CP5: {choice: 1-n, reason: str}."""
    errs: list[str] = []
    choice = d.get("choice")
    if not (isinstance(choice, int) and not isinstance(choice, bool) and 1 <= choice <= n_choices):
        errs.append(f"choice must be int 1-{n_choices}, got {choice!r}")
    if not isinstance(d.get("reason", ""), str):
        errs.append("reason must be str")
    return errs


# --------------------------------------------------------------------------- repair loop

#: A repair callable takes (raw_reply, errors) and returns a fresh raw reply string.
RepairFn = Callable[[str, list[str]], str]


def parse_with_repair(
    raw: str,
    validate: Callable[[dict[str, Any]], list[str]],
    repair: RepairFn | None = None,
) -> tuple[dict[str, Any] | None, list[str]]:
    """Decode+validate ``raw``; on failure run one repair round; else give up.

    Returns ``(obj, [])`` on success or ``(None, errors)`` after an exhausted single
    repair — the caller then applies its deterministic fallback. Mirrors the parse
    ladder's "one repair round then floor" contract. Never raises; a decode exception is
    folded into the error list so the repair round (and ultimately the fallback) still
    runs.
    """
    obj, errors = _try_decode(raw)
    if obj is not None and not errors:
        errors = validate(obj)
        if not errors:
            return obj, []
    if repair is None:
        return None, errors or ["invalid and no repair callable"]
    try:
        raw2 = repair(raw, errors)
    except Exception as exc:  # noqa: BLE001 — a dead repair provider -> deterministic fallback
        return None, errors + [f"repair call failed: {exc!r}"]
    obj2, errors2 = _try_decode(raw2)
    if obj2 is not None and not errors2:
        errors2 = validate(obj2)
        if not errors2:
            return obj2, []
    return None, errors2 or ["invalid after one repair"]


def _try_decode(raw: str) -> tuple[dict[str, Any] | None, list[str]]:
    try:
        return loads(raw), []
    except Exception as exc:  # noqa: BLE001 — malformed JSON becomes a repairable error
        return None, [f"could not decode JSON: {exc!r}"]
