"""Checkpoint-1 parse ladder: LLM tiers with validation + one repair round, regex floor.

parse() tries each injected ChatFn in order (api -> api2 -> local), schema-validating
every reply and granting each tier one repair round; a total-time cap (default 45 s,
measured via the injected Clock) is enforced between tiers and before repairs. The
deterministic regex tier is the floor: parse() never raises and always returns a Plan
that validates.

Every schema-valid Plan an LLM tier returns is run through ``core.parsing.normalize.
normalize_llm_plan`` before being stamped and returned — deterministic post-processing
for the three systematic semantic defects the local model exhibits (issues #47/#48/#49),
never applied to the regex floor's own output.

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
from core.parsing.normalize import normalize_llm_plan
from core.parsing.prompts import ChatFn, build_parse_messages, build_repair_messages
from core.parsing.regex_tier import parse_regex
from core.plan_schema import Plan

#: parse_tier stamps for the injected ChatFns, in ladder order (see plan_schema.Plan).
DEFAULT_TIER_NAMES: tuple[str, ...] = ("api", "api2", "local")

#: Default total wall-clock budget for the whole ladder, seconds.
#:
#: #213: raised from 45.0 -> 85.0 (+40 s). The ladder's own repair round already retries a
#: failed/timed-out tier once (same fn, a repair prompt) before falling through — but the
#: old 45 s cap gave a single tier only ~5 s of slack past its own worst-case retry cost
#: (2 x core.llm.timeout.DEFAULT_CALL_TIMEOUT_S = 40 s), so a live-cluster job (723601,
#: slots 5/7/8/9) that hit one slow-but-real 20 s API timeout had the retry itself skipped
#: by _expired() and fell straight to the floor with no plan bound — a whole 6-point
#: instruction_following question lost to one transient. 85 s comfortably covers one full
#: retried tier (40 s) plus a second configured tier's own retried attempt (40 s) before
#: the deterministic regex floor. 40 s of the 780 s live per-question budget is ~5%.
DEFAULT_TIME_CAP_S: float = 85.0


def parse(
    question: Any,
    chat_fns: Sequence[ChatFn],
    clock: Clock,
    ledger: Any = None,
    *,
    time_cap_s: float = DEFAULT_TIME_CAP_S,
    tier_names: Sequence[str] = DEFAULT_TIER_NAMES,
    log_fn: Any = None,
) -> Plan:
    """Parse a question (str or interfaces.Question) into a valid Plan; never raises.

    log_fn (#213): optional ``(level, message) -> None`` callback (same shape as
    core.fsm.controller's injected LogFn), called once per tier attempt/failure and once
    if every configured tier failed and the regex floor was used, so a live timeout/error
    that used to leave only a bare traceback (#213's original evidence: one grep-found
    line, no visibility into which tier fell through or why) is now loud in the adapter
    log. Best-effort: any exception from log_fn is swallowed, exactly like the ledger seam
    below — observability must never break parsing.
    """
    qtext: str = getattr(question, "text", question)
    t0 = clock.now()

    def _expired() -> bool:
        return clock.now() - t0 >= time_cap_s

    def _log(level: str, msg: str) -> None:
        if log_fn is None:
            return
        try:
            log_fn(level, msg)
        except Exception:  # noqa: BLE001 — logging must never break parsing
            pass

    # Defensive gate: if the ledger reserves the floor window (allow("parse") is False),
    # skip every API tier and go straight to the deterministic regex floor.
    attempted_any_tier = False
    if _ledger_allows(ledger):
        for i, fn in enumerate(chat_fns):
            name = tier_names[i] if i < len(tier_names) else f"llm{i}"
            if _expired():
                _log(
                    "warn",
                    f"parse ladder: time cap ({time_cap_s:g}s) reached before tier "
                    f"{name!r} could be attempted; skipping remaining tiers",
                )
                break
            attempted_any_tier = True
            tier_t0 = clock.now()
            plan, errors, raw = _attempt(fn, build_parse_messages(qtext))
            if plan is not None and not errors:
                _record_tier(ledger, clock, tier_t0, name)
                _log("info", f"parse ladder: tier {name!r} succeeded on first attempt")
                return _stamp(normalize_llm_plan(plan, qtext), qtext, name)
            _log(
                "warn",
                f"parse ladder: tier {name!r} first attempt failed ({errors!r}); retrying",
            )
            if not _expired():
                plan, errors, _ = _attempt(fn, build_repair_messages(qtext, raw, errors))
                if plan is not None and not errors:
                    _record_tier(ledger, clock, tier_t0, name)
                    _log("info", f"parse ladder: tier {name!r} succeeded on retry")
                    return _stamp(normalize_llm_plan(plan, qtext), qtext, name)
                _log(
                    "warn",
                    f"parse ladder: tier {name!r} retry also failed ({errors!r}); "
                    "falling through to the next tier",
                )
            else:
                _log(
                    "warn",
                    f"parse ladder: tier {name!r} exhausted; time cap reached before "
                    "the retry could run",
                )
            # tier exhausted (invalid + failed/skipped repair): log the attempt and fall on
            _record_tier(ledger, clock, tier_t0, name)

    regex_t0 = clock.now()
    plan = parse_regex(qtext)
    plan.parse_tier = "regex"
    _record_tier(ledger, clock, regex_t0, "regex")
    if attempted_any_tier:
        _log(
            "warn",
            "parse ladder: every configured API tier failed or timed out; "
            "question answered via the deterministic regex floor",
        )
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
