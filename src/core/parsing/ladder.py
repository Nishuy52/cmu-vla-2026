"""Checkpoint-1 parse ladder: LLM tiers with validation + one repair round, regex floor.

parse() tries each injected ChatFn in order (api -> api2 -> local), schema-validating
every reply and granting each tier one repair round; a total-time cap (default 45 s,
measured via the injected Clock) is enforced between tiers and before repairs. The
deterministic regex tier is the floor: parse() never raises and always returns a Plan
that validates.

Ledger interaction: the fsm module owns the checkpoint call ledger (core.fsm.budget.
CallLedger). This module duck-types it defensively so a missing or foreign ledger can
never break parsing:

* If the ledger exposes a callable ``allow``, the ladder calls ``allow("parse")`` before
  attempting any LLM tier. When it returns False the ledger has decided the checkpoint's
  call budget / floor-reserve window is spent, so the ladder skips all API tiers and goes
  straight to the deterministic regex floor (the ledger reserves the tail for the floor
  path).
* If the ledger exposes a callable ``record``, it is invoked once per attempted tier with
  the real CallLedger signature ``record("parse", duration_seconds, tier)`` — duration is
  measured across that tier's LLM work via the injected Clock, tier is the ladder tier
  name (``api`` / ``api2`` / ``local`` / ``regex``).

Any ledger exception (or a ledger with a different/absent interface) is swallowed:
bookkeeping and gating must never break parsing.
"""
from __future__ import annotations

from typing import Any, Sequence

from core.interfaces import Clock
from core.parsing.prompts import ChatFn, build_parse_messages, build_repair_messages
from core.parsing.regex_tier import parse_regex
from core.plan_schema import Plan

#: parse_tier stamps for the injected ChatFns, in ladder order (see plan_schema.Plan).
DEFAULT_TIER_NAMES: tuple[str, ...] = ("api", "api2", "local")

#: Default total wall-clock budget for the whole ladder, seconds.
DEFAULT_TIME_CAP_S: float = 45.0


def parse(
    question: Any,
    chat_fns: Sequence[ChatFn],
    clock: Clock,
    ledger: Any = None,
    *,
    time_cap_s: float = DEFAULT_TIME_CAP_S,
    tier_names: Sequence[str] = DEFAULT_TIER_NAMES,
) -> Plan:
    """Parse a question (str or interfaces.Question) into a valid Plan; never raises."""
    qtext: str = getattr(question, "text", question)
    t0 = clock.now()

    def _expired() -> bool:
        return clock.now() - t0 >= time_cap_s

    # Defensive gate: if the ledger reserves the floor window (allow("parse") is False),
    # skip every API tier and go straight to the deterministic regex floor.
    if _ledger_allows(ledger):
        for i, fn in enumerate(chat_fns):
            name = tier_names[i] if i < len(tier_names) else f"llm{i}"
            if _expired():
                break
            tier_t0 = clock.now()
            plan, errors, raw = _attempt(fn, build_parse_messages(qtext))
            if plan is not None and not errors:
                _record_tier(ledger, clock, tier_t0, name)
                return _stamp(plan, qtext, name)
            if not _expired():
                plan, errors, _ = _attempt(fn, build_repair_messages(qtext, raw, errors))
                if plan is not None and not errors:
                    _record_tier(ledger, clock, tier_t0, name)
                    return _stamp(plan, qtext, name)
            # tier exhausted (invalid + failed/skipped repair): log the attempt and fall on
            _record_tier(ledger, clock, tier_t0, name)

    regex_t0 = clock.now()
    plan = parse_regex(qtext)
    plan.parse_tier = "regex"
    _record_tier(ledger, clock, regex_t0, "regex")
    return plan


# --------------------------------------------------------------------- internals


def _attempt(fn: ChatFn, messages: list[dict[str, str]]) -> tuple[Plan | None, list[str], str]:
    """Run one chat call and decode/validate its reply: (plan|None, errors, raw reply)."""
    try:
        raw = fn(messages)
    except Exception as exc:  # noqa: BLE001 — a dead provider must not kill the ladder
        return None, [f"chat call failed: {exc!r}"], ""
    try:
        plan = Plan.from_json(_extract_json(raw))
    except Exception as exc:  # noqa: BLE001 — malformed JSON goes to the repair round
        return None, [f"could not decode Plan JSON: {exc!r}"], raw
    return plan, plan.validate(), raw


def _stamp(plan: Plan, question: str, tier: str) -> Plan:
    """Stamp provenance fields onto a successfully parsed Plan."""
    plan.parse_tier = tier
    plan.question_raw = question
    return plan


def _extract_json(raw: str) -> str:
    """Extract the first balanced JSON object from reply text (tolerates fences/preamble)."""
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


def _ledger_allows(ledger: Any) -> bool:
    """Duck-typed ``allow("parse")`` gate; defaults to True when unavailable/foreign.

    Returns False only when the ledger explicitly refuses the parse checkpoint, in which
    case the ladder skips API tiers and uses the regex floor. Any exception or a ledger
    without a callable ``allow`` is treated as "allowed" — gating must never break parsing.
    """
    allow = getattr(ledger, "allow", None)
    if not callable(allow):
        return True
    try:
        return bool(allow("parse"))
    except Exception:  # noqa: BLE001 — a foreign/broken ledger must not block parsing
        return True


def _record_tier(ledger: Any, clock: Clock, tier_t0: float, tier: str) -> None:
    """Record one attempted parse tier via the real CallLedger signature.

    Calls ``ledger.record("parse", duration_seconds, tier)`` where duration is measured
    from ``tier_t0`` to now via the injected clock. Best-effort: swallows every ledger/clock
    failure so bookkeeping never breaks parsing.
    """
    record = getattr(ledger, "record", None)
    if not callable(record):
        return
    try:
        duration = float(clock.now()) - float(tier_t0)
        record("parse", duration, tier)
    except Exception:  # noqa: BLE001 — bookkeeping must never break parsing
        pass
