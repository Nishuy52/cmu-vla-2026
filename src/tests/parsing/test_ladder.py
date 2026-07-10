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
