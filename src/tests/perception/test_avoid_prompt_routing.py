"""Issue #108: avoid-anchor nouns route into the full GDINO caption.
Issue #173: avoid-anchor nouns route into the short question caption too.

``_plan_nouns`` (core.heads.explore_step) feeds ``refresh_prompt``'s ``question_nouns``
argument -- both the short question-noun-only caption and (as the top priority slice of)
the full question+vocab caption. Before #108 there was no way to route a noun into the
full caption without it also reaching the short one, so ``plan.avoid`` anchors were
dropped from both passes entirely (see the pre-#108 ``_plan_nouns`` docstring, issue #95).

#108 fixed that by adding a ``full_only_nouns`` category that reaches the full caption
only. That routing was live-measured (#173) to fail in practice: the full caption runs
every 3rd tick at a higher box threshold and its ~100-phrase length dilutes any one
phrase's decode score toward zero (detector.py's own probe: a 117-phrase caption decodes
zero 'teapot' where a 2-phrase caption decodes it in 194/211 keyframes) -- an avoid
anchor routed there alone got zero raw detections across two full IF runs. #173 makes
``core.heads.factory.HeadState.bind`` ALSO fold avoid nouns into ``question_nouns``, so
they reach the short, low-threshold, every-tick pass, while the full-caption route stays
exactly as #108 left it (additive, not replaced).

This file covers:
* ``core.perception.detector.refresh_prompt``'s ``full_only_nouns`` category (unchanged
  by #173: a caller can still route a noun into the full caption only if it explicitly
  chooses to, by not also including it in ``question_nouns``).
* ``core.heads.explore_step._plan_avoid_nouns`` walks ``plan.avoid`` recursively via
  ``core.plan_walk.iter_avoid_anchors`` (including nested disambiguators).
* ``core.heads.factory.HeadState.bind`` wires avoid nouns into BOTH the short caption
  (``question_nouns``, #173) and the full caption (``full_only_nouns``, #108).
* Priority order in the budgeted full caption is question nouns (now including avoid
  nouns) > avoid-only priority slice > standing vocab.
"""
from __future__ import annotations

from core.heads.explore_step import _plan_avoid_nouns, _plan_nouns
from core.heads.factory import HeadState, _STANDING_VOCAB_NOUNS
from core.interfaces import QType
from core.perception.detector import FakeDetector, build_gdino_prompt, refresh_prompt
from core.perception.scene_index import BasicSceneIndex
from core.plan_schema import Anchor, AvoidSpec, Clause, LegKind, Plan, Pred, RouteLeg


def _instruction_plan(route: list[RouteLeg], avoid: list[AvoidSpec] | None = None) -> Plan:
    return Plan(
        qtype=QType.INSTRUCTION_FOLLOWING,
        question_raw="go",
        route=route,
        avoid=avoid or [],
    )


def _kept_phrases(prompt: str) -> set[str]:
    return {p.strip() for p in prompt.removesuffix(" .").split(" . ") if p.strip()}


# ------------------------------------------------------------- refresh_prompt: full_only_nouns


def test_refresh_prompt_full_only_nouns_reach_full_caption():
    fake = FakeDetector()
    refresh_prompt(fake, ["door"], ["sofa"], full_only_nouns=["fireplace"])
    assert "fireplace" in _kept_phrases(fake.prompt)


def test_refresh_prompt_full_only_nouns_absent_from_short_caption():
    fake = FakeDetector()
    refresh_prompt(fake, ["door"], ["sofa"], full_only_nouns=["fireplace"])
    assert "fireplace" not in _kept_phrases(fake.question_prompt)
    assert fake.question_prompt == "door ."


def test_refresh_prompt_full_only_nouns_default_is_empty_noop():
    # Omitting the kwarg must reproduce pre-#108 behaviour exactly.
    fake = FakeDetector()
    refresh_prompt(fake, ["door"], ["sofa"])
    assert fake.prompt == "door . sofa ."
    assert fake.question_prompt == "door ."


def test_refresh_prompt_full_only_nouns_deduped_against_question_nouns():
    # A noun that is BOTH a question noun and an avoid noun must not be double-counted
    # or accidentally leak into the short caption via the full_only_nouns path.
    fake = FakeDetector()
    refresh_prompt(fake, ["door"], ["sofa"], full_only_nouns=["door"])
    assert fake.prompt == "door . sofa ."
    assert fake.question_prompt == "door ."


# ------------------------------------------------------------------- _plan_avoid_nouns


def test_plan_avoid_nouns_walks_between_and_near():
    p = _instruction_plan(
        [RouteLeg(kind=LegKind.GOTO, anchors=[Anchor(noun="door")])],
        avoid=[
            AvoidSpec(near=Anchor(noun="fireplace")),
            AvoidSpec(between=[Anchor(noun="sofa"), Anchor(noun="table")]),
        ],
    )
    assert set(_plan_avoid_nouns(p)) == {"fireplace", "sofa", "table"}


def test_plan_avoid_nouns_recurses_into_nested_disambiguator():
    # "avoid near the fireplace next to the rug" -- the nested "rug" must be walked too
    # (core.plan_walk.iter_avoid_anchors, landed with #95, reserved for exactly this).
    p = _instruction_plan(
        [RouteLeg(kind=LegKind.GOTO, anchors=[Anchor(noun="door")])],
        avoid=[
            AvoidSpec(
                near=Anchor(
                    noun="fireplace",
                    disambiguator=Clause(pred=Pred.NEAR, anchors=[Anchor(noun="rug")]),
                )
            )
        ],
    )
    assert set(_plan_avoid_nouns(p)) == {"fireplace", "rug"}


def test_plan_avoid_nouns_empty_when_no_avoid_specs():
    p = _instruction_plan([RouteLeg(kind=LegKind.GOTO, anchors=[Anchor(noun="door")])])
    assert _plan_avoid_nouns(p) == []


def test_plan_avoid_nouns_none_plan_is_empty():
    assert _plan_avoid_nouns(None) == []


def test_plan_nouns_still_excludes_avoid_anchors():
    # _plan_nouns itself stays avoid-free even after #173 -- it is also the affinity/
    # frontier-bias noun source (core.heads.explore_step), which must NOT gain avoid-
    # noun bias. #173 routes avoid nouns into the detector's question caption by
    # appending _plan_avoid_nouns' output onto _plan_nouns' return value at the
    # HeadState.bind call site (core.heads.factory), not by changing this function.
    p = _instruction_plan(
        [RouteLeg(kind=LegKind.GOTO, anchors=[Anchor(noun="door")])],
        avoid=[AvoidSpec(near=Anchor(noun="fireplace"))],
    )
    assert set(_plan_nouns(p)) == {"door"}


# --------------------------------------------------------- HeadState.bind wiring (factory)


def test_headstate_bind_routes_avoid_nouns_into_both_captions():
    # Issue #173 (was: "...into_full_caption_only", asserting avoid nouns stayed OUT of
    # .question_prompt -- the #108 routing this pins away from). Live measurement
    # (#173) found an avoid anchor routed into the full caption alone gets zero raw
    # detections across 94-132 full-caption ticks in real IF runs (the ~100-phrase full
    # caption dilutes recall toward zero), while the same phrase present in a question
    # caption scores 175 raw detections. HeadState.bind now folds avoid nouns into the
    # short question caption too, so the detector is actually asked to look for them at
    # the low threshold, every tick -- not just every 3rd tick in a diluted caption.
    scene = BasicSceneIndex([])
    fake = FakeDetector()
    state = HeadState(scene=scene, detector=fake)
    plan = _instruction_plan(
        [RouteLeg(kind=LegKind.GOTO, anchors=[Anchor(noun="door")])],
        avoid=[AvoidSpec(near=Anchor(noun="fireplace"))],
    )
    state.bind(plan)
    assert "fireplace" in _kept_phrases(fake.prompt)
    assert "fireplace" in _kept_phrases(fake.question_prompt)
    assert "door" in _kept_phrases(fake.question_prompt)


def test_headstate_bind_avoid_between_nouns_reach_question_caption():
    # Issue #173 regression, mirroring the chinese_room live case: a `between` avoid
    # clause naming two anchors ("avoiding the path between the chair and the folding
    # screen") -- both nouns must reach the question-caption phrase list, and the full
    # caption must still carry them (#108's route, left additive/unchanged).
    scene = BasicSceneIndex([])
    fake = FakeDetector()
    state = HeadState(scene=scene, detector=fake)
    plan = _instruction_plan(
        [RouteLeg(kind=LegKind.GOTO, anchors=[Anchor(noun="tea table")])],
        avoid=[AvoidSpec(between=[Anchor(noun="chair"), Anchor(noun="folding screen")])],
    )
    state.bind(plan)
    question_phrases = _kept_phrases(fake.question_prompt)
    full_phrases = _kept_phrases(fake.prompt)
    assert {"chair", "folding screen"} <= question_phrases
    assert {"chair", "folding screen"} <= full_phrases
    assert "tea table" in question_phrases


# ------------------------------------------------------ priority order under the budget


def test_build_gdino_prompt_priority_question_over_full_only_over_vocab():
    # When the budget binds, question nouns must survive first, avoid (full_only)
    # nouns second, and generic vocab nouns are the first to be dropped.
    dropped: list[str] = []
    prompt = build_gdino_prompt(
        ["target"],
        ["full_only_a", "full_only_b"] + ["vocab_a", "vocab_b"],
        max_tokens=1,
        dropped_out=dropped,
    )
    kept = _kept_phrases(prompt)
    assert kept == {"target"}
    # Everything else (both full_only and vocab nouns) dropped at this budget --
    # confirms question nouns strictly outrank both.
    assert set(dropped) == {"full_only_a", "full_only_b", "vocab_a", "vocab_b"}


def test_build_gdino_prompt_tight_budget_keeps_full_only_drops_vocab():
    # Same priority proof as above but with an explicit, deterministic token_estimator
    # (1 token per phrase) so the exact cut line is easy to reason about: 2 tokens of
    # budget fits the question noun + one avoid noun and nothing from generic vocab.
    def _fixed_estimator(text: str) -> int:
        return len([p for p in text.split(" . ") if p.strip()])

    prompt = build_gdino_prompt(
        ["door"],
        ["fireplace", "sofa", "table", "lamp", "rug", "window"],
        max_tokens=2,
        token_estimator=_fixed_estimator,
    )
    kept = _kept_phrases(prompt)
    assert kept == {"door", "fireplace"}
    assert "sofa" not in kept


def test_refresh_prompt_rescues_a_105_overflow_tail_noun_via_avoid_priority():
    # Issue #105 measured that, un-budgeted, the alphabetic tail of the 116-noun
    # standing vocab (incl. "vase") is silently dropped -- 293 real tokens vs a
    # 256-token limit. Routing "vase" through full_only_nouns (as an avoid anchor
    # would) places it ahead of the rest of the standing vocab in priority, so it must
    # now survive the budget even though plain standing-vocab nouns after it may not.
    fake = FakeDetector()
    refresh_prompt(fake, (), _STANDING_VOCAB_NOUNS, full_only_nouns=("vase",))
    kept = _kept_phrases(fake.prompt)
    assert "vase" in kept


def test_build_gdino_prompt_reports_which_vocab_nouns_full_only_displaces():
    # Direct measurement of the displacement this issue's brief asks for: with "vase"
    # prepended ahead of the standing vocab (mirroring what refresh_prompt does for
    # full_only_nouns), "vase" itself must never again be among the dropped, and
    # whatever standing-vocab noun(s) that costs (if any, under whichever token
    # estimator is active in this environment) must be genuine standing-vocab nouns,
    # never a duplicate of "vase" and never spuriously reported. See this task's final
    # report for the exact displaced set measured against the real tokenizer.
    from core.heads.factory import _STANDING_VOCAB_NOUNS

    baseline_dropped: list[str] = []
    build_gdino_prompt((), list(_STANDING_VOCAB_NOUNS), dropped_out=baseline_dropped)

    with_avoid_dropped: list[str] = []
    build_gdino_prompt(
        (), ["vase"] + list(_STANDING_VOCAB_NOUNS), dropped_out=with_avoid_dropped
    )

    # "vase" no longer among the dropped (it now has priority) -- the load-bearing
    # assertion for issue #108.
    assert "vase" not in with_avoid_dropped
    newly_displaced = set(with_avoid_dropped) - set(baseline_dropped)
    assert newly_displaced.isdisjoint({"vase"})
    assert newly_displaced <= set(_STANDING_VOCAB_NOUNS)
