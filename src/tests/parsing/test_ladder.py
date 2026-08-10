"""Ladder mechanics with stubbed ChatFns: tier order, repair round, time cap, ledger."""
from __future__ import annotations

import json

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

# valid JSON but structurally invalid Plan (numerical without a target)
INVALID_PLAN_JSON = json.dumps(
    {"qtype": "numerical", "question_raw": QUESTION, "target": None, "route": [], "avoid": []}
)


class FakeClock:
    """Injected clock whose time only moves when the test says so."""

    def __init__(self, t: float = 0.0) -> None:
        self.t = t

    def now(self) -> float:
        return self.t


class LedgerStub:
    """Duck-typed ledger capturing every record(checkpoint, duration, tier) call."""

    def __init__(self, allow_parse: bool = True) -> None:
        self.events: list[tuple[str, float, str]] = []
        self._allow_parse = allow_parse

    def allow(self, checkpoint: str) -> bool:
        return self._allow_parse if checkpoint == "parse" else True

    def record(self, checkpoint: str, duration: float, tier: str) -> None:
        self.events.append((checkpoint, duration, tier))


class StubChat:
    """ChatFn stub replaying scripted replies and logging received messages."""

    def __init__(self, replies: list[str], clock: FakeClock | None = None, cost_s: float = 0.0):
        self.replies = list(replies)
        self.calls: list[list[dict[str, str]]] = []
        self.clock = clock
        self.cost_s = cost_s

    def __call__(self, messages: list[dict[str, str]]) -> str:
        self.calls.append(messages)
        if self.clock is not None:
            self.clock.t += self.cost_s
        if not self.replies:
            raise RuntimeError("stub exhausted")
        return self.replies.pop(0)


def test_first_tier_success_stamps_api():
    fn = StubChat([VALID_PLAN_JSON])
    plan = parse(QUESTION, [fn], FakeClock())
    assert plan.parse_tier == "api"
    assert plan.validate() == []
    assert plan.question_raw == QUESTION
    assert len(fn.calls) == 1


def test_fenced_json_and_preamble_tolerated():
    fn = StubChat(["Here you go:\n```json\n" + VALID_PLAN_JSON + "\n```"])
    plan = parse(QUESTION, [fn], FakeClock())
    assert plan.parse_tier == "api"
    assert plan.target is not None and plan.target.noun == "pillow"


def test_repair_round_fixes_invalid_output():
    fn = StubChat([INVALID_PLAN_JSON, VALID_PLAN_JSON])
    plan = parse(QUESTION, [fn], FakeClock())
    assert plan.parse_tier == "api"
    assert len(fn.calls) == 2
    # repair prompt must carry the validation errors and the previous output
    repair_user = fn.calls[1][-1]["content"]
    assert "requires target" in repair_user
    assert INVALID_PLAN_JSON in repair_user


# -------------------------------------------------- schema-guided repair prompt (#45)


def _plan_json_with_pred(pred: str) -> str:
    return json.dumps(
        {
            "qtype": "object_reference",
            "question_raw": QUESTION,
            "target": {
                "noun": "beer bottle",
                "raw": "beer bottle",
                "attributes": [],
                "clauses": [{"pred": pred, "anchors": [{"noun": "couch", "raw": "couch"}]}],
            },
            "route": [],
            "avoid": [],
            "notes": "",
        }
    )


def test_repair_prompt_surfaces_offending_field_value_and_valid_preds():
    # "closer_to" is not a valid Pred and not an accepted synonym (unlike furthest_from,
    # which core.plan_schema now normalizes before this ever reaches validation) — it must
    # still fail, and the repair prompt built from that failure must name the field, the
    # bad value, and the schema's actual valid values (not a raw exception repr).
    bad_reply = _plan_json_with_pred("closer_to")
    fn = StubChat([bad_reply, VALID_PLAN_JSON])
    parse(QUESTION, [fn], FakeClock())
    assert len(fn.calls) == 2
    repair_user = fn.calls[1][-1]["content"]
    assert "'pred'" in repair_user
    assert "'closer_to'" in repair_user
    assert "farthest_from" in repair_user  # valid value, proves the list isn't truncated/stale
    assert "closest_to" in repair_user


def test_local_stub_furthest_from_repair_round_gets_schema_guidance():
    """Regression per issue #45's exact repro shape, using LocalStub's scripted replies: a
    first reply with an invalid/unrecognised pred, and a repair-round reply. The repair
    prompt handed back to the model must list 'farthest_from' among the valid values."""
    from core.llm.providers import LocalStub

    # Use a value that is NOT one of the accepted exact-synonym aliases (issue: enum
    # synonym normalization fixed 'furthest_from' itself to parse directly), so the repair
    # round in this test still actually fires and exercises the repair-prompt content.
    first_reply = _plan_json_with_pred("furthest_frm")  # typo'd, not an alias
    repair_reply = VALID_PLAN_JSON
    stub = LocalStub([first_reply, repair_reply])
    plan = parse(QUESTION, [stub], FakeClock(), tier_names=("local",))
    assert plan.parse_tier == "local"
    assert len(stub.calls) == 2
    repair_messages = stub.calls[1]
    repair_user = repair_messages[-1]["content"]
    assert "'pred'" in repair_user
    assert "'furthest_frm'" in repair_user
    assert "farthest_from" in repair_user


def test_falls_through_tiers_in_order():
    bad = StubChat(["garbage", "still garbage"])
    good = StubChat([VALID_PLAN_JSON])
    plan = parse(QUESTION, [bad, good], FakeClock(), tier_names=("api", "api2"))
    assert plan.parse_tier == "api2"
    assert len(bad.calls) == 2  # first attempt + one repair round
    assert len(good.calls) == 1


def test_all_tiers_fail_regex_floor():
    bad1 = StubChat(["nope", "nope"])
    bad2 = StubChat(["nope", "nope"])
    plan = parse(QUESTION, [bad1, bad2], FakeClock())
    assert plan.parse_tier == "regex"
    assert plan.validate() == []
    assert plan.qtype.value == "numerical"


def test_chatfn_exception_is_contained():
    def exploding(_messages):
        raise ConnectionError("network dark")

    good = StubChat([VALID_PLAN_JSON])
    plan = parse(QUESTION, [exploding, good], FakeClock())
    assert plan.parse_tier == "api2"


def test_no_chat_fns_goes_straight_to_regex():
    plan = parse(QUESTION, [], FakeClock())
    assert plan.parse_tier == "regex"
    assert plan.validate() == []


def test_time_cap_skips_llm_tiers():
    clock = FakeClock()
    fn = StubChat([VALID_PLAN_JSON], clock=clock)
    clock.t = 100.0  # already past cap relative to t0? t0 is read first: advance after
    # t0 = 100.0; make the cap already consumed by jumping the clock via a wrapper
    slow_probe = StubChat([VALID_PLAN_JSON])

    class JumpyClock(FakeClock):
        def __init__(self):
            super().__init__(0.0)
            self.reads = 0

        def now(self) -> float:
            self.reads += 1
            return 0.0 if self.reads == 1 else 60.0  # t0=0, everything after t=60

    plan = parse(QUESTION, [slow_probe, fn], JumpyClock(), time_cap_s=45.0)
    assert plan.parse_tier == "regex"
    assert slow_probe.calls == [] and fn.calls == []


def test_time_cap_between_tiers():
    clock = FakeClock()
    slow_bad = StubChat(["garbage", "garbage"], clock=clock, cost_s=30.0)  # 60 s total
    never_reached = StubChat([VALID_PLAN_JSON])
    plan = parse(QUESTION, [slow_bad, never_reached], clock, time_cap_s=45.0)
    assert plan.parse_tier == "regex"
    assert len(slow_bad.calls) >= 1
    assert never_reached.calls == []


def test_time_cap_skips_repair_round():
    clock = FakeClock()
    slow_bad = StubChat(["garbage", VALID_PLAN_JSON], clock=clock, cost_s=50.0)
    plan = parse(QUESTION, [slow_bad], clock, time_cap_s=45.0)
    assert plan.parse_tier == "regex"
    assert len(slow_bad.calls) == 1  # repair was skipped by the cap


def test_ledger_receives_events():
    ledger = LedgerStub()
    bad = StubChat(["garbage", "garbage"])
    parse(QUESTION, [bad], FakeClock(), ledger)
    tiers = [tier for (_cp, _dur, tier) in ledger.events]
    # one event for the attempted api tier, one for the regex fallback
    assert tiers == ["api", "regex"]
    assert all(cp == "parse" for (cp, _dur, _tier) in ledger.events)


def test_broken_ledger_never_breaks_parse():
    class BrokenLedger:
        def record(self, checkpoint, duration, tier) -> None:
            raise RuntimeError("ledger down")

    plan = parse(QUESTION, [StubChat([VALID_PLAN_JSON])], FakeClock(), BrokenLedger())
    assert plan.parse_tier == "api"


def test_question_object_accepted():
    from core.interfaces import Question

    plan = parse(Question(text=QUESTION, t_received=0.0), [], FakeClock())
    assert plan.question_raw == QUESTION


# ------------------------------------------------------------------ #213 timeout retry/floor


class TimeoutOnceThenSucceedChat:
    """Simulates a live API tier: TimeoutError on the first call, then a valid reply."""

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls: list[list[dict[str, str]]] = []

    def __call__(self, messages: list[dict[str, str]]) -> str:
        self.calls.append(messages)
        if len(self.calls) == 1:
            raise TimeoutError("chat call exceeded 20s timeout")
        return self.reply


class AlwaysTimeoutChat:
    """Simulates a wedged provider: every call raises TimeoutError."""

    def __init__(self) -> None:
        self.calls: list[list[dict[str, str]]] = []

    def __call__(self, messages: list[dict[str, str]]) -> str:
        self.calls.append(messages)
        raise TimeoutError("chat call exceeded 20s timeout")


def test_timeout_once_then_succeeds_via_retry():
    """(a) A tier that times out once and then succeeds must produce a plan through the
    ladder's own retry (repair-round) mechanism -- NOT fall through to the next tier or
    the regex floor -- and the tier it succeeded on must be recorded."""
    fn = TimeoutOnceThenSucceedChat(VALID_PLAN_JSON)
    ledger = LedgerStub()
    plan = parse(QUESTION, [fn], FakeClock(), ledger)
    assert plan is not None
    assert plan.parse_tier == "api"
    assert plan.validate() == []
    assert len(fn.calls) == 2  # first attempt (timed out) + retry (succeeded)
    tiers = [tier for (_cp, _dur, tier) in ledger.events]
    assert tiers == ["api"]  # the api tier is what's recorded as the winner


def test_timeout_always_falls_through_to_regex_floor():
    """(b) A tier that times out on every attempt must still yield a non-None, valid plan
    via the deterministic regex floor -- the FSM must always receive a plan, never None."""
    fn = AlwaysTimeoutChat()
    ledger = LedgerStub()
    plan = parse(QUESTION, [fn], FakeClock(), ledger)
    assert plan is not None
    assert plan.parse_tier == "regex"
    assert plan.validate() == []
    assert len(fn.calls) == 2  # first attempt + one retry, both timed out
    tiers = [tier for (_cp, _dur, tier) in ledger.events]
    assert tiers == ["api", "regex"]  # the exhausted api tier AND the regex floor recorded


def test_fast_path_unchanged_by_213():
    """(c) A tier that succeeds on the first call is untouched: one call, tier "api", no
    retry machinery invoked."""
    fn = StubChat([VALID_PLAN_JSON])
    plan = parse(QUESTION, [fn], FakeClock())
    assert plan.parse_tier == "api"
    assert plan.validate() == []
    assert len(fn.calls) == 1


def test_log_fn_called_on_retry_and_success():
    """#213 item 3: each tier attempt/failure is logged loudly via the injected log_fn."""
    events: list[tuple[str, str]] = []

    def log_fn(level: str, msg: str) -> None:
        events.append((level, msg))

    fn = TimeoutOnceThenSucceedChat(VALID_PLAN_JSON)
    parse(QUESTION, [fn], FakeClock(), log_fn=log_fn)
    assert any(lvl == "warn" and "retrying" in msg for lvl, msg in events)
    assert any(lvl == "info" and "succeeded on retry" in msg for lvl, msg in events)


def test_log_fn_called_on_regex_fallback():
    events: list[tuple[str, str]] = []

    def log_fn(level: str, msg: str) -> None:
        events.append((level, msg))

    fn = AlwaysTimeoutChat()
    parse(QUESTION, [fn], FakeClock(), log_fn=log_fn)
    assert any("regex floor" in msg for _lvl, msg in events)


def test_log_fn_exception_never_breaks_parse():
    def broken_log(level: str, msg: str) -> None:
        raise RuntimeError("logging is down")

    fn = AlwaysTimeoutChat()
    plan = parse(QUESTION, [fn], FakeClock(), log_fn=broken_log)
    assert plan is not None and plan.parse_tier == "regex"


def test_result_always_roundtrips():
    plan = parse(QUESTION, [StubChat([VALID_PLAN_JSON])], FakeClock())
    again = Plan.from_json(plan.to_json())
    assert again.validate() == []


# ------------------------------------------------- real CallLedger integration (defect 2)


def test_real_call_ledger_records_one_event_per_attempted_tier():
    """Wire the production CallLedger through parse: it must receive record('parse', dur,
    tier) once per attempted tier via the real 3-arg signature, not a swallowed TypeError."""
    from core.fsm.budget import BudgetState, CallLedger

    clock = FakeClock()
    ledger = CallLedger(BudgetState(clock))
    ledger._budget.latch(0.0)  # remaining() = 600 s, well above the floor reserve

    bad = StubChat(["garbage", "garbage"])  # api tier fails parse + repair -> regex floor
    plan = parse(QUESTION, [bad], clock, ledger)

    assert plan.parse_tier == "regex"
    log = ledger.log()
    # one attempted api tier + the regex floor, both recorded under checkpoint "parse"
    assert [r.tier for r in log] == ["api", "regex"]
    assert all(r.checkpoint == "parse" for r in log)
    assert ledger.count("parse") == 2


def test_real_call_ledger_disallow_routes_straight_to_regex():
    """When the real ledger's allow('parse') is False (floor window reserved), the ladder
    must skip all API tiers and use the regex floor without touching any ChatFn."""
    from core.fsm.budget import BudgetState, CallLedger, LEDGER_RESERVE_S
    from core.interfaces import QUESTION_BUDGET_S

    clock = FakeClock()
    budget = BudgetState(clock)
    budget.latch(0.0)
    # push elapsed past the point where remaining() < LEDGER_RESERVE_S so allow() is False
    clock.t = QUESTION_BUDGET_S - LEDGER_RESERVE_S + 1.0
    ledger = CallLedger(budget)
    assert ledger.allow("parse") is False

    never = StubChat([VALID_PLAN_JSON])
    plan = parse(QUESTION, [never], clock, ledger)

    assert plan.parse_tier == "regex"
    assert never.calls == []  # API tier never attempted
    # only the regex floor was recorded
    assert [r.tier for r in ledger.log()] == ["regex"]


def test_real_call_ledger_records_measured_duration():
    """Duration passed to the real ledger is measured across the tier via the clock."""
    from core.fsm.budget import BudgetState, CallLedger

    clock = FakeClock()
    ledger = CallLedger(BudgetState(clock))
    ledger._budget.latch(0.0)

    slow = StubChat([VALID_PLAN_JSON], clock=clock, cost_s=3.0)
    parse(QUESTION, [slow], clock, ledger)

    api_rec = next(r for r in ledger.log() if r.tier == "api")
    assert api_rec.duration == 3.0
