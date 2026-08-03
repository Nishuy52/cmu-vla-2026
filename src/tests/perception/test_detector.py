"""Detector seam: FakeDetector routing/scripting, GDINO prompt + lazy-import error."""
from __future__ import annotations

import builtins
import json

import numpy as np
import pytest

from core.perception.detector import (
    Detection,
    DEFAULT_GDINO_ANSWER_MIN_OBS,
    DEFAULT_GDINO_ANSWER_MIN_SCORE,
    DEFAULT_GDINO_QUESTION_BOX_THRESHOLD,
    DEFAULT_GDINO_VOCAB_PASS_CADENCE,
    ENV_GDINO_ANSWER_MIN_OBS,
    ENV_GDINO_ANSWER_MIN_SCORE,
    ENV_GDINO_CHECKPOINT_PATH,
    ENV_GDINO_CONFIG_PATH,
    ENV_GDINO_DEVICE,
    ENV_GDINO_MODEL_ID,
    ENV_GDINO_PRECISION,
    ENV_GDINO_QUESTION_BOX_THRESHOLD,
    ENV_GDINO_VOCAB_PASS_CADENCE,
    ENV_RAW_DETECTION_DUMP_PATH,
    FakeDetector,
    GATE_ACCEPTED,
    GATE_NO_LIDAR_CLUSTER,
    GDINO_BACKOFF_BASE_S,
    GDINO_BACKOFF_CAP_S,
    GDINO_BACKOFF_DEGRADE_N,
    GDINO_MODEL_ID,
    GDINO_REQUIRED_INSTALLS,
    ELIGIBLE,
    INELIGIBLE_BOTH,
    INELIGIBLE_MALFORMED,
    INELIGIBLE_N_OBS,
    INELIGIBLE_SCORE,
    answer_eligibility_reason,
    answer_min_obs,
    answer_min_score,
    dump_raw_detections,
    dump_prompt_diagnostics,
    ENV_PROMPT_DUMP_PATH,
    is_answer_eligible,
    GroundingDinoDetector,
    _norm_cxcywh_to_tile_xyxy,
    build_gdino_prompt,
    refresh_prompt,
    _eviction_safe_vocab_order,
    CROSS_TILE_NMS_IOU_THRESHOLD,
    suppress_cross_tile_duplicates,
    CAPTION_JOIN_MARKER,
    FALLBACK_NOUN_MATCH_MIN_WORDS,
    caption_nouns,
    resolve_alignment_fallback,
)
from core.perception.tiling import DEFAULT_TILE_HFOV, DEFAULT_TILE_VFOV


class _FakeClock:
    """Manually-advanced monotonic clock stand-in (mirrors mocks.mock_io.FakeClock)."""

    def __init__(self, start: float = 0.0) -> None:
        self._t = float(start)

    def __call__(self) -> float:
        return self._t

    def advance(self, dt: float) -> float:
        self._t += float(dt)
        return self._t


def _tiles(n=4):
    return [np.zeros((10, 10, 3), dtype=np.uint8) for _ in range(n)]


# ------------------------------------------------------------------ Detection


def test_detection_center_and_foot():
    d = Detection(tile_id=0, bbox_xyxy=(10, 20, 30, 60), label="sofa", score=0.5)
    assert d.center_xy == (20.0, 40.0)
    assert d.foot_xy == (20.0, 60.0)


# ------------------------------------------------------------------ FakeDetector


def test_fake_detector_routes_to_tiles():
    dets = [
        Detection(0, (0, 0, 1, 1), "a", 0.5),
        Detection(2, (0, 0, 1, 1), "b", 0.5),
        Detection(0, (2, 2, 3, 3), "c", 0.5),
    ]
    fake = FakeDetector(dets)
    out = fake(_tiles(4))
    assert len(out) == 4
    assert [d.label for d in out[0]] == ["a", "c"]
    assert out[1] == []
    assert [d.label for d in out[2]] == ["b"]


def test_fake_detector_fixed_repeats():
    fake = FakeDetector([Detection(0, (0, 0, 1, 1), "x", 0.9)])
    a = fake(_tiles())
    b = fake(_tiles())
    assert [d.label for d in a[0]] == ["x"]
    assert [d.label for d in b[0]] == ["x"]  # same every call


def test_fake_detector_script_advances():
    script = [
        [Detection(0, (0, 0, 1, 1), "f1", 0.9)],
        [Detection(1, (0, 0, 1, 1), "f2", 0.9)],
    ]
    fake = FakeDetector(script=script)
    o0 = fake(_tiles())
    o1 = fake(_tiles())
    o2 = fake(_tiles())  # exhausted -> repeats last frame
    assert [d.label for d in o0[0]] == ["f1"]
    assert [d.label for d in o1[1]] == ["f2"]
    assert [d.label for d in o2[1]] == ["f2"]


def test_fake_detector_ignores_out_of_range_tile():
    fake = FakeDetector([Detection(9, (0, 0, 1, 1), "z", 0.5)])
    out = fake(_tiles(4))
    assert all(t == [] for t in out)


# ------------------------------------------------------------------ GDINO prompt


def test_build_gdino_prompt_dedupes_and_orders():
    prompt = build_gdino_prompt(["sofa", "window"], ["window", "potted plant", "sofa"])
    assert prompt == "sofa . window . potted plant ."


def test_build_gdino_prompt_empty():
    assert build_gdino_prompt([], []) == ""


def test_build_gdino_prompt_lowercases():
    assert build_gdino_prompt(["Sofa"], []) == "sofa ."


# ------------------------------------------------------- GDINO prompt token budget (#105)

#: Ground truth measured directly against the real GroundingDINO text encoder
#: (bert-base-uncased): ``build_gdino_prompt((), _STANDING_VOCAB_NOUNS)`` -- the
#: un-budgeted 116-noun standing vocab caption -- tokenizes to exactly this many real
#: wordpiece tokens, 37 over the 256 (``max_text_len``) limit the loaded GroundingDINO
#: config enforces. Pinned here as a fixture constant so the calibration tests below
#: never need ``transformers`` (absent from the ``src/`` test venv) to stay honest.
MEASURED_STANDING_VOCAB_REAL_TOKENS = 293

#: The 12 nouns the real tokenizer measurement showed get silently amputated by the
#: pre-fix unbudgeted caption (alphabetical-tail overflow past the 256-token limit).
MEASURED_OVERFLOW_TAIL_NOUNS = (
    "trash can", "tray", "tv", "tv cabinet", "tv remote", "vase",
    "wall lamp", "wardrobe", "wardrobe door", "water cooler", "whiteboard", "window",
)


def _standing_vocab_nouns():
    # Read-only import of the live 116-noun standing vocab -- this module owns no
    # part of factory.py, it only observes the vocab it ships to reproduce the #105
    # measurement against the real, current vocab (not a hand-copied snapshot that
    # could drift out of sync with it).
    from core.heads.factory import _STANDING_VOCAB_NOUNS

    return _STANDING_VOCAB_NOUNS


def test_heuristic_token_estimate_never_underestimates_measured_vocab():
    # Requirement: the dependency-free fallback heuristic must OVER-estimate, never
    # under-estimate, relative to the real tokenizer -- pinned against the one real
    # measurement we have (293 real tokens for the full un-budgeted vocab caption).
    # Tests the heuristic function directly (not the real/heuristic-selecting
    # `_default_token_estimate` seam) so this calibration check means the same thing
    # regardless of whether `transformers` happens to be importable in the test env.
    from core.perception.detector import _heuristic_token_estimate

    prompt = build_gdino_prompt((), _standing_vocab_nouns(), max_tokens=10**9)
    estimate = _heuristic_token_estimate(prompt)
    assert estimate >= MEASURED_STANDING_VOCAB_REAL_TOKENS
    # ...and not wildly so -- a heuristic pessimistic enough to discard half the
    # vocab under-uses the real 256-token budget just as badly as overflowing it.
    assert estimate <= 2 * MEASURED_STANDING_VOCAB_REAL_TOKENS


def test_build_gdino_prompt_full_vocab_stays_within_default_budget():
    from core.perception.detector import DEFAULT_GDINO_MAX_TEXT_TOKENS, _default_token_estimate

    prompt = build_gdino_prompt((), _standing_vocab_nouns())
    assert _default_token_estimate(prompt) <= DEFAULT_GDINO_MAX_TEXT_TOKENS


# --------------------------------------------- real-tokenizer / heuristic seam (#105)


@pytest.fixture(autouse=True)
def _reset_tokenizer_probe_cache(monkeypatch):
    # `_probe_real_tokenizer` caches its result at module scope (by design -- it must
    # not reconstruct a tokenizer on every detection-path call). Reset that cache
    # around every test in this file so tests that fake the probe's outcome, or that
    # assert on how many times it constructs, do not leak state between each other or
    # into unrelated tests.
    import core.perception.detector as detector_mod

    monkeypatch.setattr(detector_mod, "_cached_real_tokenizer", None)
    yield
    monkeypatch.setattr(detector_mod, "_cached_real_tokenizer", None)


def test_probe_real_tokenizer_falls_back_when_transformers_unavailable(monkeypatch):
    import builtins

    from core.perception.detector import _probe_real_tokenizer

    real_import = builtins.__import__

    def _blocked_import(name, *args, **kwargs):
        if name == "transformers" or name.startswith("transformers."):
            raise ImportError("simulated: transformers not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _blocked_import)
    assert _probe_real_tokenizer() is None


def test_default_token_estimate_falls_back_when_probe_returns_none(monkeypatch):
    import core.perception.detector as detector_mod

    monkeypatch.setattr(detector_mod, "_probe_real_tokenizer", lambda: None)
    # With no real tokenizer, the estimate must match the heuristic exactly (proves
    # the fallback path is actually taken, not just "some number").
    text = "sofa . window ."
    assert detector_mod._default_token_estimate(text) == detector_mod._heuristic_token_estimate(text)


def test_default_token_estimate_uses_real_tokenizer_when_probe_succeeds(monkeypatch):
    import core.perception.detector as detector_mod

    monkeypatch.setattr(detector_mod, "_probe_real_tokenizer", lambda: (lambda text: 999))
    # A fake "real" counter returning a fixed sentinel proves the seam actually wires
    # through to whatever `_probe_real_tokenizer` resolves, in preference over the
    # heuristic (which would return something in the tens, not 999, for this text).
    assert detector_mod._default_token_estimate("sofa . window .") == 999


def test_default_token_estimate_degrades_to_heuristic_if_real_counter_raises(monkeypatch):
    import core.perception.detector as detector_mod

    def _broken_counter(text):
        raise RuntimeError("simulated: tokenizer call failed")

    monkeypatch.setattr(detector_mod, "_probe_real_tokenizer", lambda: _broken_counter)
    text = "sofa . window ."
    # Must not raise, and must degrade to the same value the heuristic alone gives.
    assert detector_mod._default_token_estimate(text) == detector_mod._heuristic_token_estimate(text)


def test_probe_real_tokenizer_caches_and_does_not_reconstruct_per_call(monkeypatch):
    import core.perception.detector as detector_mod

    calls = []

    def _fake_loader():
        calls.append(1)
        return lambda text: len(text)

    # Simulate a successful first load by seeding the cache the same way a real,
    # successful `_probe_real_tokenizer()` call would -- then prove later calls read
    # the cache rather than rebuilding.
    monkeypatch.setattr(detector_mod, "_cached_real_tokenizer", _fake_loader())
    first = detector_mod._probe_real_tokenizer()
    second = detector_mod._probe_real_tokenizer()
    assert first is second
    # Sanity: the fake counter itself still behaves like a counter.
    assert first("abc") == 3


def test_probe_real_tokenizer_logs_once(monkeypatch, caplog):
    import logging

    import core.perception.detector as detector_mod

    calls = []

    class _FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(name):
            calls.append(name)

            class _Tok:
                def __call__(self, text):
                    return {"input_ids": [0] * len(text.split())}

            return _Tok()

    fake_transformers = type("_FakeModule", (), {"AutoTokenizer": _FakeAutoTokenizer})
    monkeypatch.setitem(__import__("sys").modules, "transformers", fake_transformers)

    with caplog.at_level(logging.INFO, logger="core.perception.detector"):
        detector_mod._probe_real_tokenizer()
        detector_mod._probe_real_tokenizer()
        detector_mod._probe_real_tokenizer()

    assert len(calls) == 1  # tokenizer constructed exactly once, not per probe call
    info_logs = [r for r in caplog.records if r.levelno == logging.INFO and "tokenizer loaded" in r.message]
    assert len(info_logs) == 1


def _kept_phrases(prompt: str) -> set[str]:
    return {p.strip() for p in prompt.removesuffix(" .").split(" . ") if p.strip()}


def test_build_gdino_prompt_question_anchor_nouns_survive_full_vocab_truncation():
    # The exact #105 regression: vase/tv/window/water cooler are in the alphabetic
    # overflow tail and were silently dropped from EVERY pass pre-fix. As question
    # nouns (recall-priority), they must survive even though the vocab pass still
    # overflows the budget.
    question_nouns = ["vase", "tv", "window", "water cooler"]
    prompt = build_gdino_prompt(question_nouns, _standing_vocab_nouns())
    kept_phrases = _kept_phrases(prompt)
    for noun in question_nouns:
        assert noun in kept_phrases


def test_build_gdino_prompt_question_nouns_never_dropped_for_vocab():
    # A budget of 1 -- below even a single CLS/SEP pair, real or heuristic-estimated --
    # leaves zero room for any vocab noun under either estimator, so every single one
    # must be dropped before a single question noun is.
    question_nouns = ["sofa", "window", "potted plant"]
    dropped: list[str] = []
    prompt = build_gdino_prompt(
        question_nouns,
        _standing_vocab_nouns(),
        max_tokens=1,
        dropped_out=dropped,
    )
    kept_phrases = _kept_phrases(prompt)
    for noun in question_nouns:
        assert noun in kept_phrases
    # window/sofa/potted plant are already in the prompt as question nouns, so they
    # are de-duplicated out of the vocab pass entirely (never counted as "dropped");
    # every other vocab noun must be dropped at this budget.
    expected_dropped = set(_standing_vocab_nouns()) - set(question_nouns)
    assert set(dropped) == expected_dropped


def test_build_gdino_prompt_over_budget_question_nouns_kept_and_logged(caplog):
    # If even the question nouns alone blow the budget, they are still kept in
    # full (never silently trimmed) but the condition is logged loudly.
    import logging

    question_nouns = ["sofa", "window", "potted plant", "refrigerator"]
    with caplog.at_level(logging.ERROR, logger="core.perception.detector"):
        prompt = build_gdino_prompt(question_nouns, (), max_tokens=1)
    kept_phrases = _kept_phrases(prompt)
    for noun in question_nouns:
        assert noun in kept_phrases
    assert any("question nouns alone" in rec.message for rec in caplog.records)


def test_build_gdino_prompt_drop_is_reported_not_silent(caplog):
    import logging

    dropped: list[str] = []
    with caplog.at_level(logging.WARNING, logger="core.perception.detector"):
        prompt = build_gdino_prompt((), _standing_vocab_nouns(), dropped_out=dropped)
    # Some nouns must actually have been dropped for this test to mean anything.
    assert dropped
    assert set(MEASURED_OVERFLOW_TAIL_NOUNS).issubset(set(dropped))
    kept_phrases = _kept_phrases(prompt)
    for noun in dropped:
        assert noun not in kept_phrases
    assert any("dropping" in rec.message for rec in caplog.records)


def test_build_gdino_prompt_no_drop_when_vocab_fits_budget():
    dropped: list[str] = []
    prompt = build_gdino_prompt(["sofa"], ["window", "potted plant"], dropped_out=dropped)
    assert dropped == []
    assert prompt == "sofa . window . potted plant ."


# ---------------------------------- disambiguator vocab prioritisation (issue #91) ----


#: Issue #91: the training corpus's own noun phrases for questions whose disambiguator
#: reads zero live-detection hits (VERIFIED EVIDENCE in the #91 task) -- the exact
#: sentences are "...closest to the pyramid candle holder" (livingroom_1),
#: "...near the jar" (japanese_room), "...closest to the map wall decal" (office_1).
_91_DISAMBIGUATOR_QUESTIONS = (
    (
        "Go to the potted plant closest to the pyramid candle holder and stop at the "
        "vase between the TV and the door."
    ),
    "Go near the small table with a vase on it and then to the flowers near the jar.",
    "How many computer monitors are on the table closest to the map wall decal?",
)


def test_refresh_prompt_disambiguator_nouns_reach_the_short_caption():
    # Issue #91 acceptance: the caption for a question includes that question's own
    # disambiguator nouns (not just its primary target). Exercises the real regex
    # parser + _plan_nouns recursion (#95) end-to-end, not a hand-built noun list, so a
    # regression in either upstream piece would show up here too.
    from core.heads.explore_step import _plan_nouns
    from core.parsing.regex_tier import parse_regex

    expectations = {
        _91_DISAMBIGUATOR_QUESTIONS[0]: "pyramid candle holder",
        _91_DISAMBIGUATOR_QUESTIONS[1]: "jar",
        _91_DISAMBIGUATOR_QUESTIONS[2]: "map wall decal",
    }
    for question, disambiguator in expectations.items():
        plan = parse_regex(question)
        question_nouns = _plan_nouns(plan)
        assert disambiguator in question_nouns, (question, question_nouns)

        det = FakeDetector()
        refresh_prompt(det, question_nouns, _standing_vocab_nouns())
        # Short (question-only) caption -- the pass #42 calibrated for actual recall of
        # rare/small classes, never diluted by the ~100-phrase standing vocab.
        assert disambiguator in _kept_phrases(det.question_prompt)
        # Full caption -- question nouns are never dropped for vocab nouns, so it must
        # survive there too even though the vocab pass alone overflows the budget.
        assert disambiguator in _kept_phrases(det.prompt)


def test_refresh_prompt_reprioritizes_vocab_tier_disambiguators_to_front():
    # Issue #91 follow-up: reprioritisation now only reorders the TAIL that already
    # doesn't survive the token-budget cut (core.perception.detector.
    # _eviction_safe_vocab_order), instead of the whole vocab list, so it can never
    # evict a survivor (see
    # test_refresh_prompt_disambiguator_priority_never_evicts_any_prior_survivor).
    # Disambiguator nouns that never needed rescuing (already outside the dropped
    # tail) must still reach the caption unconditionally, same as before.
    from core.perception.vocab import DISAMBIGUATOR_PRIORITY_NOUNS

    vocab = _standing_vocab_nouns()
    dropped_before: list[str] = []
    build_gdino_prompt((), vocab, dropped_out=dropped_before)
    dropped_before_set = set(dropped_before)

    det = FakeDetector()
    refresh_prompt(det, (), vocab)
    kept_phrases = _kept_phrases(det.prompt)

    for noun in DISAMBIGUATOR_PRIORITY_NOUNS:
        if noun in vocab and noun not in dropped_before_set:
            assert noun in kept_phrases, (noun, "evicted from the vocab-only full caption")

    # The overflow-tail disambiguators (already in the dropped tail before any
    # reorder) DO get pulled to the very front of that tail -- the reorder still
    # happens -- but under the current heuristic-estimator headroom (3 tokens; the
    # cheapest of them, "wall decal", costs 4 to add) neither is actually cheap
    # enough to be rescued into the caption. That is the honest, reported
    # budget-arithmetic trade-off documented on
    # core.perception.detector._eviction_safe_vocab_order: eviction-safety costs the
    # #91 promotion its practical effect against THIS estimator/vocab pairing rather
    # than costing high-value nouns (microwave/mirror/monitor) their spot.
    overflow_disambiguators = [n for n in DISAMBIGUATOR_PRIORITY_NOUNS if n in dropped_before_set]
    assert overflow_disambiguators, "fixture drift: expected >=1 disambiguator noun in the dropped tail"
    for noun in overflow_disambiguators:
        assert noun not in kept_phrases, (
            noun, "unexpectedly rescued into the caption -- update this test's comment"
        )


def test_refresh_prompt_disambiguator_priority_does_not_evict_high_value_standing_nouns():
    # Issue #91 explicit anti-regression: pulling the disambiguator-priority nouns
    # forward must not knock a common target/anchor noun (chair, sofa, table, potted
    # plant, pillow, vase, cabinet, bed, window, tv -- all used as the PRIMARY target or
    # anchor of multiple training questions) out of the vocab-only full caption, under
    # the SAME (default heuristic, worst-case) estimator this repo's test venv uses.
    high_value_nouns = {
        "chair", "sofa", "table", "potted plant", "pillow", "vase", "cabinet", "bed",
        "window", "tv", "picture", "lamp", "bowl",
    }
    dropped_before: list[str] = []
    build_gdino_prompt((), _standing_vocab_nouns(), dropped_out=dropped_before)

    det = FakeDetector()
    refresh_prompt(det, (), _standing_vocab_nouns())
    kept_after = _kept_phrases(det.prompt)

    newly_evicted = {n for n in high_value_nouns if n in dropped_before} - set()
    # (sanity: at least confirms the fixture is meaningful -- some of these nouns
    # already get dropped pre-fix under the heuristic estimator; the real assertion
    # below is that the fix does not make that WORSE for any of them.)
    for noun in high_value_nouns:
        was_kept_before = noun not in dropped_before
        if was_kept_before:
            assert noun in kept_after, (
                noun, "was in the full caption before reprioritisation, evicted after"
            )


def test_refresh_prompt_disambiguator_priority_never_evicts_any_prior_survivor():
    # Issue #91 correctness follow-up: an independent re-measurement found the
    # committed whole-list reorder (core.perception.vocab.prioritize_vocab_nouns
    # applied directly to the full standing vocab) evicts THREE nouns under the
    # heuristic estimator that were not evicted before -- microwave, mirror, monitor
    # -- none named in the #91 overflow-tail evidence, and monitor/mirror are
    # high-frequency/anchor classes (issue #94, #132) worse to lose than the
    # disambiguator gain. This pins the general invariant (not just those three
    # names): reprioritisation must NEVER evict a noun that survived the SAME
    # token-budget cut before it was reordered. Must FAIL against the direct
    # ``prioritize_vocab_nouns(vocab_nouns)`` reorder and pass once refresh_prompt
    # instead reorders only the already-dropped tail
    # (core.perception.detector._eviction_safe_vocab_order).
    vocab = _standing_vocab_nouns()

    dropped_before: list[str] = []
    build_gdino_prompt((), vocab, dropped_out=dropped_before)
    survived_before = {n for n in vocab if n not in set(dropped_before)}

    det = FakeDetector()
    refresh_prompt(det, (), vocab)
    kept_after = set(det.prompt.removesuffix(" .").split(" . "))

    newly_evicted = {n for n in survived_before if n not in kept_after}
    assert not newly_evicted, (
        "reprioritisation evicted noun(s) that survived the budget cut before it: "
        f"{sorted(newly_evicted)}"
    )
    # sanity: the three nouns the independent re-measurement actually named must be
    # among the confirmed survivors (i.e. the fixture is meaningful, not vacuous).
    assert {"microwave", "mirror", "monitor"} <= survived_before


def test_refresh_prompt_full_only_nouns_still_outrank_reprioritised_vocab_tier():
    # full_only_nouns (issue #108's avoid-anchor breadth) must still sit ahead of the
    # reprioritised vocab tier in the rendered caption -- reprioritisation only reorders
    # WITHIN the vocab tier; it must never let a vocab noun (even a priority one)
    # outrank a full_only noun.
    from core.perception.vocab import DISAMBIGUATOR_PRIORITY_NOUNS

    priority_noun = DISAMBIGUATOR_PRIORITY_NOUNS[0]
    det = FakeDetector()
    refresh_prompt(
        det, (), [priority_noun, "ball"], full_only_nouns=["avoid target noun"]
    )
    phrases = det.prompt.removesuffix(" .").split(" . ")
    assert phrases[0] == "avoid target noun"
    assert priority_noun in phrases[1:]


def test_build_gdino_prompt_custom_token_estimator_is_honoured():
    # Injecting an estimator that treats every rendered candidate as huge should drop
    # every vocab noun (still keeping question nouns) -- proves the seam is real, not
    # just decorative, and that no hard dependency on a real tokenizer exists.
    dropped: list[str] = []
    prompt = build_gdino_prompt(
        ["sofa"],
        ["window", "potted plant"],
        token_estimator=lambda text: 10**9,
        dropped_out=dropped,
    )
    assert prompt == "sofa ."
    assert dropped == ["window", "potted plant"]


# ------------------------------------------------------------------ GDINO stub


def test_gdino_constructs_without_torch():
    # constructing must not import torch or raise
    det = GroundingDinoDetector(["sofa"], ["window"])
    assert "sofa" in det.prompt and "window" in det.prompt


def test_gdino_call_raises_clear_install_error(monkeypatch):
    """Calling without torch/groundingdino raises ImportError listing the pip installs."""
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch" or name.startswith("groundingdino"):
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    det = GroundingDinoDetector(["sofa"])
    with pytest.raises(ImportError) as exc:
        det([np.zeros((4, 4, 3), dtype=np.uint8)])
    msg = str(exc.value)
    for pkg in GDINO_REQUIRED_INSTALLS:
        assert pkg in msg
    assert "pip install" in msg
    assert "ubuntu_setup.md" in msg


# ------------------------------------------------------------------ GDINO real __call__ (Gate 4)
# The real forward-pass dispatch (_call_batched / _call_per_tile / _ensure_model) needs
# torch + groundingdino + baked weights, which per the Gate-4 environment rules must never
# be imported/loaded by a test run on this box (the sim owns the GPU). These tests instead
# cover every torch-free piece: the empty-input short circuits, the env/constructor
# precedence for every deploy-time knob, the device/precision resolution against a duck-
# typed stand-in for ``torch``, the actionable errors when weight paths are unset, and the
# pure box-space math that maps a model box back to tile pixels (the actual "map detections
# back through the tiling transform" logic). Real batched-vs-per-tile dispatch against
# real weights is listed in the post-recording checklist (nothing here loads torch).


def test_gdino_call_empty_tiles_short_circuits():
    det = GroundingDinoDetector(["sofa"])
    assert det([]) == []


def test_gdino_call_empty_prompt_short_circuits_without_torch():
    # No question nouns yet (freshly booted, no question latched) -> nothing to ground.
    # Must return per-tile empty lists WITHOUT ever needing torch/groundingdino installed.
    det = GroundingDinoDetector()  # no nouns -> prompt == ""
    assert det.prompt == ""
    out = det(_tiles(4))
    assert out == [[], [], [], []]


def test_gdino_model_id_precedence(monkeypatch):
    monkeypatch.delenv(ENV_GDINO_MODEL_ID, raising=False)
    assert GroundingDinoDetector().model_id == GDINO_MODEL_ID  # module default
    monkeypatch.setenv(ENV_GDINO_MODEL_ID, "env/model-id")
    assert GroundingDinoDetector().model_id == "env/model-id"  # env overrides default
    assert GroundingDinoDetector(model_id="ctor/model-id").model_id == "ctor/model-id"  # ctor wins


def test_gdino_precision_precedence_and_default(monkeypatch):
    monkeypatch.delenv(ENV_GDINO_PRECISION, raising=False)
    assert GroundingDinoDetector().precision == "auto"
    monkeypatch.setenv(ENV_GDINO_PRECISION, "fp32")
    assert GroundingDinoDetector().precision == "fp32"
    assert GroundingDinoDetector(precision="fp16").precision == "fp16"  # ctor wins over env


def test_gdino_device_env_honoured(monkeypatch):
    monkeypatch.delenv(ENV_GDINO_DEVICE, raising=False)
    assert GroundingDinoDetector()._device_pref is None
    monkeypatch.setenv(ENV_GDINO_DEVICE, "cpu")
    assert GroundingDinoDetector()._device_pref == "cpu"
    assert GroundingDinoDetector(device="cuda")._device_pref == "cuda"  # ctor wins over env


class _FakeCuda:
    def __init__(self, available: bool) -> None:
        self._available = available

    def is_available(self) -> bool:
        return self._available


class _FakeTorch:
    """Bare duck-typed stand-in for the one thing _resolve_device touches: cuda.is_available()."""

    def __init__(self, cuda_available: bool) -> None:
        self.cuda = _FakeCuda(cuda_available)
        self.float32 = "float32"  # only used past a successful _ensure_model (backoff tests)


def test_resolve_device_prefers_explicit_pref():
    det = GroundingDinoDetector(device="cpu")
    assert det._resolve_device(_FakeTorch(cuda_available=True)) == "cpu"


def test_resolve_device_falls_back_to_cuda_availability():
    det = GroundingDinoDetector()
    assert det._resolve_device(_FakeTorch(cuda_available=True)) == "cuda"
    assert det._resolve_device(_FakeTorch(cuda_available=False)) == "cpu"


def test_resolve_half_fp16_and_fp32_explicit():
    assert GroundingDinoDetector(precision="fp16")._resolve_half("cpu") is True
    assert GroundingDinoDetector(precision="fp32")._resolve_half("cuda") is False


def test_resolve_half_auto_by_device():
    det = GroundingDinoDetector(precision="auto")
    assert det._resolve_half("cuda") is True  # half on CUDA: 8 GB dev box / 10-14 GB eval box
    assert det._resolve_half("cpu") is False  # full on CPU


class _FakeModel:
    """Duck-typed stand-in for a loaded groundingdino model: records .to()/.eval()/.half()
    calls in order so tests can assert the device move happens before precision is applied."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def to(self, device):
        self.calls.append(f"to:{device}")
        return self

    def eval(self):
        self.calls.append("eval")
        return self

    def half(self):
        self.calls.append("half")
        return self


def test_ensure_model_moves_to_device_never_halves(tmp_path):
    # groundingdino-py's load_model(..., device="cuda") leaves the returned model on CPU
    # (issue #41); _ensure_model must move it explicitly. Weights are NEVER hard-.half()ed
    # (mixed float32 buffers inside GroundingDINO break under a halved model) — fp16 comes
    # from _forward_ctx's autocast instead, so resolved_half only arms that context.
    cfg = tmp_path / "gdino.cfg.py"
    cfg.write_text("# fake config\n")
    ckpt = tmp_path / "gdino.pth"
    ckpt.write_bytes(b"\x00")
    det = GroundingDinoDetector(
        config_path=str(cfg), checkpoint_path=str(ckpt), device="cuda", precision="auto",
    )
    fake_model = _FakeModel()
    model = det._ensure_model(_FakeTorch(cuda_available=True), lambda *a, **k: fake_model)
    assert model is fake_model
    assert "to:cuda" in fake_model.calls
    assert "half" not in fake_model.calls
    assert det._resolved_half is True  # arms the fp16 autocast forward context


class _FakeAutocastTorch(_FakeTorch):
    """Fake torch recording autocast constructions (returned ctx is a no-op)."""

    def __init__(self, cuda_available: bool) -> None:
        super().__init__(cuda_available)
        self.autocast_calls: list[dict] = []
        self.float16 = "float16"

    def autocast(self, device_type, dtype):
        self.autocast_calls.append({"device_type": device_type, "dtype": dtype})
        import contextlib

        return contextlib.nullcontext()


def test_forward_ctx_autocast_only_on_cuda_half(tmp_path):
    cfg = tmp_path / "gdino.cfg.py"
    cfg.write_text("# fake config\n")
    ckpt = tmp_path / "gdino.pth"
    ckpt.write_bytes(b"\x00")
    det = GroundingDinoDetector(
        config_path=str(cfg), checkpoint_path=str(ckpt), device="cuda", precision="auto",
    )
    fake_torch = _FakeAutocastTorch(cuda_available=True)
    det._ensure_model(fake_torch, lambda *a, **k: _FakeModel())
    with det._forward_ctx(fake_torch):
        pass
    assert fake_torch.autocast_calls == [{"device_type": "cuda", "dtype": "float16"}]

    det_cpu = GroundingDinoDetector(
        config_path=str(cfg), checkpoint_path=str(ckpt), device="cpu", precision="auto",
    )
    fake_torch_cpu = _FakeAutocastTorch(cuda_available=False)
    det_cpu._ensure_model(fake_torch_cpu, lambda *a, **k: _FakeModel())
    with det_cpu._forward_ctx(fake_torch_cpu):
        pass
    assert fake_torch_cpu.autocast_calls == []  # nullcontext on CPU/full precision


def test_ensure_model_no_half_on_cpu(tmp_path):
    cfg = tmp_path / "gdino.cfg.py"
    cfg.write_text("# fake config\n")
    ckpt = tmp_path / "gdino.pth"
    ckpt.write_bytes(b"\x00")
    det = GroundingDinoDetector(
        config_path=str(cfg), checkpoint_path=str(ckpt), device="cpu", precision="auto",
    )
    fake_model = _FakeModel()
    model = det._ensure_model(_FakeTorch(cuda_available=False), lambda *a, **k: fake_model)
    assert model is fake_model
    assert "to:cpu" in fake_model.calls
    assert "half" not in fake_model.calls


def test_resolve_config_path_from_ctor_and_env(monkeypatch, tmp_path):
    monkeypatch.delenv(ENV_GDINO_CONFIG_PATH, raising=False)
    cfg = tmp_path / "gdino.cfg.py"
    cfg.write_text("# fake config\n")
    assert GroundingDinoDetector(config_path=str(cfg))._resolve_config_path() == str(cfg)
    monkeypatch.setenv(ENV_GDINO_CONFIG_PATH, str(cfg))
    assert GroundingDinoDetector()._resolve_config_path() == str(cfg)


def test_resolve_config_path_missing_raises_actionable_error(monkeypatch):
    monkeypatch.delenv(ENV_GDINO_CONFIG_PATH, raising=False)
    det = GroundingDinoDetector()
    # No baked config path and (in this venv) no installed groundingdino package to fall
    # back on -> a clear, actionable RuntimeError, not a bare AttributeError/ImportError.
    with pytest.raises(RuntimeError) as exc:
        det._resolve_config_path()
    msg = str(exc.value)
    assert "config_path" in msg
    assert ENV_GDINO_CONFIG_PATH in msg
    assert "ubuntu_setup.md" in msg


def test_resolve_checkpoint_path_from_ctor_and_env(monkeypatch, tmp_path):
    monkeypatch.delenv(ENV_GDINO_CHECKPOINT_PATH, raising=False)
    ckpt = tmp_path / "gdino.pth"
    ckpt.write_bytes(b"\x00")
    assert GroundingDinoDetector(checkpoint_path=str(ckpt))._resolve_checkpoint_path() == str(ckpt)
    monkeypatch.setenv(ENV_GDINO_CHECKPOINT_PATH, str(ckpt))
    assert GroundingDinoDetector()._resolve_checkpoint_path() == str(ckpt)


def test_resolve_checkpoint_path_missing_raises_actionable_error(monkeypatch):
    monkeypatch.delenv(ENV_GDINO_CHECKPOINT_PATH, raising=False)
    det = GroundingDinoDetector()
    with pytest.raises(RuntimeError) as exc:
        det._resolve_checkpoint_path()
    msg = str(exc.value)
    assert "checkpoint_path" in msg
    assert ENV_GDINO_CHECKPOINT_PATH in msg
    assert "baked into the Docker image" in msg  # offline-capable rule, never fetched at runtime


# ------------------------------------------------------------------ box-space math (tiling map-back)


def test_norm_cxcywh_to_tile_xyxy_center_box():
    # A box centred in a 480x640 tile spanning half its width/height.
    x0, y0, x1, y1 = _norm_cxcywh_to_tile_xyxy(0.5, 0.5, 0.5, 0.5, tile_w=480, tile_h=640)
    assert x0 == pytest.approx(120.0)
    assert x1 == pytest.approx(360.0)
    assert y0 == pytest.approx(160.0)
    assert y1 == pytest.approx(480.0)


def test_norm_cxcywh_to_tile_xyxy_full_frame():
    x0, y0, x1, y1 = _norm_cxcywh_to_tile_xyxy(0.5, 0.5, 1.0, 1.0, tile_w=100, tile_h=200)
    assert (x0, y0, x1, y1) == pytest.approx((0.0, 0.0, 100.0, 200.0))


def test_norm_cxcywh_to_tile_xyxy_scales_independently_per_axis():
    # A box that is a different fraction of width vs height lands correctly on each axis
    # independently — this is the actual math that stands in for "no inverse-resize needed".
    x0, y0, x1, y1 = _norm_cxcywh_to_tile_xyxy(0.25, 0.75, 0.1, 0.2, tile_w=200, tile_h=100)
    assert x0 == pytest.approx(40.0)
    assert x1 == pytest.approx(60.0)
    assert y0 == pytest.approx(65.0)
    assert y1 == pytest.approx(85.0)


# ------------------------------------------------------------------ box clamping (verifier refutation)


def test_norm_cxcywh_to_tile_xyxy_centered_case_unchanged():
    # Sanity baseline for the clamp tests below: a fully-inside box is untouched.
    x0, y0, x1, y1 = _norm_cxcywh_to_tile_xyxy(0.5, 0.5, 0.5, 0.5, tile_w=640, tile_h=640)
    assert (x0, y0, x1, y1) == pytest.approx((160.0, 160.0, 480.0, 480.0))


def test_norm_cxcywh_to_tile_xyxy_edge_hugging_clamps_to_bounds():
    # cx=0, cy=0, w=0.5, h=0.5 on a 640x640 tile: unclamped affine gives (-160,-160,160,160)
    # — a box hugging the top-left corner whose far edge legitimately sits inside the tile.
    # Real GDINO detections at tile edges produce exactly this shape; downstream pixel
    # indexing must never see a negative coordinate.
    x0, y0, x1, y1 = _norm_cxcywh_to_tile_xyxy(0.0, 0.0, 0.5, 0.5, tile_w=640, tile_h=640)
    assert (x0, y0, x1, y1) == pytest.approx((0.0, 0.0, 160.0, 160.0))
    assert 0.0 <= x0 <= x1 <= 640.0
    assert 0.0 <= y0 <= y1 <= 640.0


def test_norm_cxcywh_to_tile_xyxy_fully_outside_degenerates_safely():
    # Entirely outside normalised [0, 1] space on both axes: every raw corner clamps to the
    # same tile edge, collapsing to a zero-area box AT the bound rather than an inverted
    # (x1 < x0) or out-of-bounds one.
    x0, y0, x1, y1 = _norm_cxcywh_to_tile_xyxy(-1.0, -1.0, 0.2, 0.2, tile_w=640, tile_h=640)
    assert (x0, y0, x1, y1) == pytest.approx((0.0, 0.0, 0.0, 0.0))
    assert x0 <= x1 and y0 <= y1  # never inverted

    # Fully outside past the high edge too.
    x0, y0, x1, y1 = _norm_cxcywh_to_tile_xyxy(2.0, 2.0, 0.2, 0.2, tile_w=640, tile_h=640)
    assert (x0, y0, x1, y1) == pytest.approx((640.0, 640.0, 640.0, 640.0))
    assert x0 <= x1 and y0 <= y1


# ------------------------------------------------------------------ prompt refresh (issue #34)


def test_fake_detector_has_empty_prompt_by_default():
    assert FakeDetector().prompt == ""


def test_refresh_prompt_updates_fake_detector_in_place():
    fake = FakeDetector()
    assert fake.prompt == ""
    new_prompt = refresh_prompt(fake, ["chair", "table"], ["sofa"])
    assert new_prompt == "chair . table . sofa ."
    assert fake.prompt == "chair . table . sofa ."  # mutated in place, same instance


def test_refresh_prompt_updates_gdino_detector_in_place():
    det = GroundingDinoDetector()  # boot-time construction: no question latched yet
    assert det.prompt == ""
    refresh_prompt(det, ["lamp"], [])
    assert det.prompt == "lamp ."


def test_refresh_prompt_none_detector_is_noop():
    assert refresh_prompt(None, ["chair"], []) is None


def test_refresh_prompt_detector_without_prompt_attr_is_noop():
    def plain_detector(tiles):
        return [[] for _ in tiles]

    assert refresh_prompt(plain_detector, ["chair"], []) is None


# ------------------------------------------------------------------ dual-caption detection (issue #42)


def test_gdino_builds_short_question_only_prompt_at_construction():
    det = GroundingDinoDetector(["teapot"], ["table", "sofa"])
    assert det.prompt == "teapot . table . sofa ."
    assert det.question_prompt == "teapot ."  # no vocab nouns in the short caption


def test_fake_detector_has_empty_question_prompt_by_default():
    assert FakeDetector().question_prompt == ""


def test_refresh_prompt_updates_fake_detector_question_prompt_too():
    fake = FakeDetector()
    refresh_prompt(fake, ["teapot"], ["table"])
    assert fake.prompt == "teapot . table ."
    assert fake.question_prompt == "teapot ."


def test_refresh_prompt_updates_gdino_detector_both_captions():
    det = GroundingDinoDetector()  # boot-time: no question latched yet
    assert det.prompt == "" and det.question_prompt == ""
    refresh_prompt(det, ["teapot"], ["table", "chair"])
    assert det.prompt == "teapot . table . chair ."
    assert det.question_prompt == "teapot ."


def test_question_box_threshold_precedence(monkeypatch):
    monkeypatch.delenv(ENV_GDINO_QUESTION_BOX_THRESHOLD, raising=False)
    assert GroundingDinoDetector().question_box_threshold == DEFAULT_GDINO_QUESTION_BOX_THRESHOLD
    monkeypatch.setenv(ENV_GDINO_QUESTION_BOX_THRESHOLD, "0.18")
    assert GroundingDinoDetector().question_box_threshold == 0.18
    assert GroundingDinoDetector(question_box_threshold=0.4).question_box_threshold == 0.4  # ctor wins


def test_question_and_vocab_thresholds_are_explicit_and_split(monkeypatch):
    # Issue #91: the two box thresholds are separate, explicitly-named config — the
    # question-pass (recall) floor default is lower than the vocab-pass default, and
    # the vocab-pass default is the value the whole detector used before #42/#91 (0.35,
    # unchanged by this issue).
    monkeypatch.delenv(ENV_GDINO_QUESTION_BOX_THRESHOLD, raising=False)
    det = GroundingDinoDetector()
    assert DEFAULT_GDINO_QUESTION_BOX_THRESHOLD == 0.18
    assert det.box_threshold == 0.35  # vocab default, untouched
    assert det.question_box_threshold == DEFAULT_GDINO_QUESTION_BOX_THRESHOLD
    assert det.question_box_threshold < det.box_threshold


def test_vocab_pass_cadence_precedence(monkeypatch):
    monkeypatch.delenv(ENV_GDINO_VOCAB_PASS_CADENCE, raising=False)
    assert GroundingDinoDetector().vocab_pass_cadence == DEFAULT_GDINO_VOCAB_PASS_CADENCE
    monkeypatch.setenv(ENV_GDINO_VOCAB_PASS_CADENCE, "5")
    assert GroundingDinoDetector().vocab_pass_cadence == 5
    assert GroundingDinoDetector(vocab_pass_cadence=1).vocab_pass_cadence == 1  # ctor wins


def _dual_pass_detector(tmp_path, **kwargs) -> GroundingDinoDetector:
    """A GroundingDinoDetector whose model 'loads' successfully via a stubbed
    ``_lazy_import`` (no torch/groundingdino needed) so ``__call__``'s dual-pass
    orchestration can be exercised end to end; ``_dispatch_pass`` itself is left real —
    callers stub it directly to record/assert per-pass arguments without needing a real
    forward pass."""
    cfg = tmp_path / "gdino.cfg.py"
    cfg.write_text("# fake config\n")
    ckpt = tmp_path / "gdino.pth"
    ckpt.write_bytes(b"\x00")
    det = GroundingDinoDetector(
        config_path=str(cfg), checkpoint_path=str(ckpt), device="cpu", **kwargs,
    )
    fake_model = _FakeModel()
    det._lazy_import = lambda: (_FakeTorch(cuda_available=False), lambda *a, **k: fake_model, None)
    return det


def test_dual_pass_merges_question_and_vocab_detections(tmp_path):
    det = _dual_pass_detector(
        tmp_path, question_nouns=["teapot"], vocab_nouns=["table"], vocab_pass_cadence=1,
    )
    calls: list[tuple[str, float]] = []

    def fake_dispatch(torch, model, predict_fn, tiles, device, dtype, prompt, box_threshold):
        calls.append((prompt, box_threshold))
        label = "teapot" if prompt == det.question_prompt else "table"
        return [[Detection(0, (0, 0, 1, 1), label, 0.9)] if i == 0 else [] for i in range(len(tiles))]

    det._dispatch_pass = fake_dispatch
    out = det(_tiles(4))

    # Both passes ran (cadence=1 -> vocab pass due on the very first tick) and their
    # tile-0 detections were unioned, not overwritten.
    assert calls == [
        (det.question_prompt, det.question_box_threshold),
        (det.prompt, det.box_threshold),
    ]
    assert sorted(d.label for d in out[0]) == ["table", "teapot"]
    assert out[1] == out[2] == out[3] == []


def test_question_pass_uses_its_own_lower_threshold(tmp_path):
    det = _dual_pass_detector(
        tmp_path, question_nouns=["teapot"], vocab_nouns=[], question_box_threshold=0.22,
    )
    assert det.box_threshold == 0.35  # module default, unchanged
    seen: list[float] = []

    def fake_dispatch(torch, model, predict_fn, tiles, device, dtype, prompt, box_threshold):
        seen.append(box_threshold)
        return [[] for _ in tiles]

    det._dispatch_pass = fake_dispatch
    det(_tiles(2))
    assert 0.22 in seen  # question pass ran with its own threshold, not the vocab 0.35


def test_marginal_score_admitted_for_question_noun_cut_for_vocab_noun(tmp_path):
    """Issue #91 synthetic case: an identical marginal raw score (0.18) clears the lower
    question-pass box threshold but is cut by the higher vocab-pass box threshold — the
    exact mechanism that was silently dropping GT disambiguator classes (candle holder,
    wall decal, ...) whose true score sits in this band. ``fake_dispatch`` mirrors the
    real ``_decode_batch_item`` filtering rule (``max_logits > box_threshold``, i.e. a
    score at or below the threshold is never turned into a Detection) so this exercises
    the intended semantics, not just that two different threshold values get passed down
    (that plumbing is already covered by test_question_pass_uses_its_own_lower_threshold)."""
    RAW_SCORE = 0.18
    det = _dual_pass_detector(
        tmp_path,
        question_nouns=["candle holder"],
        vocab_nouns=["chair"],
        question_box_threshold=0.15,  # < 0.18: admits the marginal score
        vocab_pass_cadence=1,
    )
    assert det.box_threshold == 0.35  # > 0.18: cuts the identical marginal score

    def fake_dispatch(torch, model, predict_fn, tiles, device, dtype, prompt, box_threshold):
        if RAW_SCORE <= box_threshold:
            return [[] for _ in tiles]
        label = "candle holder" if prompt == det.question_prompt else "chair"
        return [[Detection(0, (0, 0, 1, 1), label, RAW_SCORE)] if i == 0 else [] for i in range(len(tiles))]

    det._dispatch_pass = fake_dispatch
    out = det(_tiles(2))
    labels = [d.label for d in out[0]]
    assert "candle holder" in labels  # question-pass threshold (0.15) admits the 0.18 score
    assert "chair" not in labels  # vocab-pass threshold (0.35) cuts the identical 0.18 score


def test_vocab_pass_runs_only_every_nth_tick(tmp_path):
    det = _dual_pass_detector(
        tmp_path, question_nouns=["teapot"], vocab_nouns=["table"], vocab_pass_cadence=3,
    )
    vocab_pass_ticks: list[int] = []

    def fake_dispatch(torch, model, predict_fn, tiles, device, dtype, prompt, box_threshold):
        if prompt == det.prompt:
            vocab_pass_ticks.append(1)
        return [[] for _ in tiles]

    det._dispatch_pass = fake_dispatch
    for _ in range(6):
        det(_tiles(2))
    # cadence=3 over 6 ticks (0..5) -> due on ticks 0 and 3 -> exactly 2 vocab passes.
    assert len(vocab_pass_ticks) == 2


def test_question_only_tick_is_a_single_forward(tmp_path):
    det = _dual_pass_detector(
        tmp_path, question_nouns=["teapot"], vocab_nouns=["table"], vocab_pass_cadence=3,
    )
    dispatch_calls = {"n": 0}

    def fake_dispatch(torch, model, predict_fn, tiles, device, dtype, prompt, box_threshold):
        dispatch_calls["n"] += 1
        return [[] for _ in tiles]

    det._dispatch_pass = fake_dispatch
    det(_tiles(2))  # tick 0 -> vocab due -> two forwards
    assert dispatch_calls["n"] == 2
    det(_tiles(2))  # tick 1 -> vocab not due -> one forward (question only)
    assert dispatch_calls["n"] == 3


def test_empty_question_prompt_still_runs_vocab_pass_alone(tmp_path):
    # No question nouns latched but a standing vocab prompt somehow present (edge case,
    # e.g. constructed directly with vocab_nouns and empty question_nouns): the question
    # pass is skipped (empty caption, nothing to ground) but the vocab pass still runs on
    # its own cadence tick.
    det = _dual_pass_detector(tmp_path, question_nouns=[], vocab_nouns=["table"], vocab_pass_cadence=1)
    assert det.question_prompt == ""
    assert det.prompt == "table ."
    calls: list[str] = []

    def fake_dispatch(torch, model, predict_fn, tiles, device, dtype, prompt, box_threshold):
        calls.append(prompt)
        return [[] for _ in tiles]

    det._dispatch_pass = fake_dispatch
    det(_tiles(2))
    assert calls == ["table ."]


def test_both_prompts_empty_still_short_circuits_without_torch():
    det = GroundingDinoDetector()  # no question latched -> both prompts empty
    assert det.prompt == "" and det.question_prompt == ""
    assert det(_tiles(3)) == [[], [], []]


# ------------------------------------------------------------------ load backoff (issue #39)
# A permanently-failing model load used to re-run the full load_model cascade every
# perception tick (5 Hz), pegging adapter CPU and starving other subscribers. These tests
# drive GroundingDinoDetector with a fake clock + a stubbed ``_lazy_import`` returning a
# controllable (always-failing or always-succeeding) load_model_fn, so no torch/groundingdino
# install is needed.


def _backoff_detector(tmp_path, clock, load_model_fn):
    cfg = tmp_path / "gdino.cfg.py"
    cfg.write_text("# fake config\n")
    ckpt = tmp_path / "gdino.pth"
    ckpt.write_bytes(b"\x00")
    det = GroundingDinoDetector(
        ["sofa"], config_path=str(cfg), checkpoint_path=str(ckpt), device="cpu", clock=clock,
    )
    det._lazy_import = lambda: (_FakeTorch(cuda_available=False), load_model_fn, None)
    return det


class _FailingLoad:
    """Counts calls; always raises."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        raise RuntimeError("boom: weights corrupt")


def test_backoff_empty_detections_during_cooldown_without_exception(tmp_path):
    clock = _FakeClock()
    load = _FailingLoad()
    det = _backoff_detector(tmp_path, clock, load)
    tiles = _tiles(4)

    out = det(tiles)  # first attempt fails

    assert out == [[], [], [], []]
    assert load.calls == 1
    assert det._consecutive_load_failures == 1


def test_backoff_schedule_no_reload_before_cooldown_then_reload_after(tmp_path):
    clock = _FakeClock()
    load = _FailingLoad()
    det = _backoff_detector(tmp_path, clock, load)
    tiles = _tiles(4)

    det(tiles)
    assert load.calls == 1  # first attempt

    clock.advance(0.5)  # < 1s backoff after first failure
    det(tiles)
    assert load.calls == 1  # still cooling down, no reload attempted

    clock.advance(0.5)  # now at the 1s boundary
    det(tiles)
    assert load.calls == 2  # cooldown elapsed -> retried (and failed again)

    clock.advance(1.9)  # < 2s backoff after second failure
    det(tiles)
    assert load.calls == 2

    clock.advance(0.1)  # now at the 2s boundary
    det(tiles)
    assert load.calls == 3


def test_backoff_success_resets_failure_state(tmp_path):
    clock = _FakeClock()
    load = _FailingLoad()
    det = _backoff_detector(tmp_path, clock, load)
    tiles = _tiles(4)

    det(tiles)
    clock.advance(GDINO_BACKOFF_BASE_S)
    det(tiles)
    assert det._consecutive_load_failures == 2

    fake_model = _FakeModel()
    clock.advance(2 * GDINO_BACKOFF_BASE_S)
    det._lazy_import = lambda: (_FakeTorch(cuda_available=False), lambda *a, **k: fake_model, None)
    det._call_batched = lambda torch, model, tiles, device, dtype: [[] for _ in tiles]
    det(tiles)

    assert det._consecutive_load_failures == 0
    assert det._next_retry_at == 0.0
    assert det._model is fake_model

    # No cooldown after a success: the very next call re-runs immediately.
    load2 = _FailingLoad()
    det._lazy_import = lambda: (_FakeTorch(cuda_available=False), load2, None)
    det._model = None  # force a fresh load attempt
    det(tiles)
    assert load2.calls == 1


def test_backoff_single_log_per_transition(tmp_path, caplog):
    clock = _FakeClock()
    load = _FailingLoad()
    det = _backoff_detector(tmp_path, clock, load)
    tiles = _tiles(4)

    with caplog.at_level("WARNING", logger="core.perception.detector"):
        for _ in range(GDINO_BACKOFF_DEGRADE_N):
            det(tiles)
            wait = det._next_retry_at - clock()
            if wait > 0:
                clock.advance(wait)

    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    errors = [r for r in caplog.records if r.levelname == "ERROR"]
    assert len(warnings) == GDINO_BACKOFF_DEGRADE_N - 1
    assert len(errors) == 1
    assert "degraded" in errors[0].message
    assert load.calls == GDINO_BACKOFF_DEGRADE_N


def test_backoff_degrades_to_cap_after_n_failures_no_further_logging(tmp_path, caplog):
    clock = _FakeClock()
    load = _FailingLoad()
    det = _backoff_detector(tmp_path, clock, load)
    tiles = _tiles(4)

    with caplog.at_level("WARNING", logger="core.perception.detector"):
        for _ in range(GDINO_BACKOFF_DEGRADE_N):
            det(tiles)
            wait = det._next_retry_at - clock()
            if wait > 0:
                clock.advance(wait)
        # 5th failure just logged the single ERROR transition; delay is now capped.
        assert det._consecutive_load_failures == GDINO_BACKOFF_DEGRADE_N

        caplog.clear()
        # One more failure past the degrade point: still capped, no new log record.
        det(tiles)
        assert det._next_retry_at - clock() == pytest.approx(GDINO_BACKOFF_CAP_S)
        assert caplog.records == []


# --------------------------------------------------------------------- issue #43a eligibility


class _Rec:
    """Minimal duck-typed stand-in for InstanceRecord (n_obs + score only)."""

    def __init__(self, n_obs: int, score: float) -> None:
        self.n_obs = n_obs
        self.score = score


def test_answer_eligibility_defaults():
    assert DEFAULT_GDINO_ANSWER_MIN_OBS == 2
    assert DEFAULT_GDINO_ANSWER_MIN_SCORE == pytest.approx(0.30)
    assert answer_min_obs() == DEFAULT_GDINO_ANSWER_MIN_OBS
    assert answer_min_score() == pytest.approx(DEFAULT_GDINO_ANSWER_MIN_SCORE)


def test_answer_eligible_at_both_floors():
    # Exactly at the floor on both axes -> eligible (>=, not >).
    assert is_answer_eligible(_Rec(n_obs=2, score=0.30)) is True


def test_ineligible_single_observation():
    # n_obs == 1 (the issue's headline case): ineligible even with a strong score.
    assert is_answer_eligible(_Rec(n_obs=1, score=0.95)) is False


def test_ineligible_below_score_floor():
    # n_obs sufficient but peak score just under the 0.30 answer floor (still clears the
    # 0.18 recall floor used for index/exploration — that floor is untouched).
    assert is_answer_eligible(_Rec(n_obs=5, score=0.29)) is False


def test_eligible_well_above_both_floors():
    assert is_answer_eligible(_Rec(n_obs=10, score=0.9)) is True


def test_ineligible_missing_attributes_not_raising():
    class _Bare:
        pass

    assert is_answer_eligible(_Bare()) is False


def test_answer_min_obs_env_override(monkeypatch):
    monkeypatch.setenv(ENV_GDINO_ANSWER_MIN_OBS, "5")
    assert answer_min_obs() == 5
    assert is_answer_eligible(_Rec(n_obs=4, score=0.9)) is False
    assert is_answer_eligible(_Rec(n_obs=5, score=0.9)) is True


def test_answer_min_score_env_override(monkeypatch):
    monkeypatch.setenv(ENV_GDINO_ANSWER_MIN_SCORE, "0.5")
    assert answer_min_score() == pytest.approx(0.5)
    assert is_answer_eligible(_Rec(n_obs=10, score=0.4)) is False
    assert is_answer_eligible(_Rec(n_obs=10, score=0.5)) is True


def test_answer_min_obs_malformed_env_falls_back_to_default(monkeypatch):
    monkeypatch.setenv(ENV_GDINO_ANSWER_MIN_OBS, "not-a-number")
    assert answer_min_obs() == DEFAULT_GDINO_ANSWER_MIN_OBS


def test_answer_min_score_malformed_env_falls_back_to_default(monkeypatch):
    monkeypatch.setenv(ENV_GDINO_ANSWER_MIN_SCORE, "not-a-number")
    assert answer_min_score() == pytest.approx(DEFAULT_GDINO_ANSWER_MIN_SCORE)


# ------------------------------------------------------------------ answer_eligibility_reason
# (issue #84 gate observability: the offline battery's GT mocks are born n_obs=3, always
# clearing both floors, so this breakdown is never exercised end-to-end offline -- these
# tests exercise it directly at the unit level, covering every branch.)


def test_eligibility_reason_eligible():
    assert answer_eligibility_reason(_Rec(n_obs=2, score=0.30)) == ELIGIBLE
    assert is_answer_eligible(_Rec(n_obs=2, score=0.30)) is True


def test_eligibility_reason_n_obs_only():
    assert answer_eligibility_reason(_Rec(n_obs=1, score=0.95)) == INELIGIBLE_N_OBS


def test_eligibility_reason_score_only():
    assert answer_eligibility_reason(_Rec(n_obs=5, score=0.29)) == INELIGIBLE_SCORE


def test_eligibility_reason_both():
    assert answer_eligibility_reason(_Rec(n_obs=1, score=0.1)) == INELIGIBLE_BOTH


def test_eligibility_reason_malformed():
    class _Bare:
        pass

    assert answer_eligibility_reason(_Bare()) == INELIGIBLE_MALFORMED


def test_is_answer_eligible_logs_rejection_reason(monkeypatch, caplog):
    """Issue #84: a gate rejection is no longer a silent boolean -- it logs the
    breakdown at DEBUG, so a live run's debug log shows WHY an instance lost the
    answer."""
    import logging

    caplog.set_level(logging.DEBUG, logger="core.perception.detector")
    assert is_answer_eligible(_Rec(n_obs=1, score=0.9)) is False
    assert any("n_obs_below_floor" in r.message for r in caplog.records)


# ------------------------------------------------------------------ run_caption_pass (issue #86)
# The remote-offload seam (core.perception.remote_detector.RemoteDetector, served on the
# other end by tools/cluster/gdino_server.py) calls this directly, one caption pass per
# request. These tests stub _lazy_import/_ensure_model the same way the #39 backoff tests do
# above, so no torch/groundingdino install is needed.


def test_run_caption_pass_empty_tiles_short_circuits_without_import(monkeypatch):
    det = GroundingDinoDetector()
    called = {"n": 0}

    def fake_lazy_import():
        called["n"] += 1
        raise AssertionError("must not be called for empty tiles")

    det._lazy_import = fake_lazy_import
    assert det.run_caption_pass([], "sofa .", 0.35) == []
    assert called["n"] == 0


def test_run_caption_pass_empty_caption_short_circuits_without_import():
    det = GroundingDinoDetector()
    det._lazy_import = lambda: (_ for _ in ()).throw(AssertionError("must not import"))
    tiles = _tiles(3)
    assert det.run_caption_pass(tiles, "", 0.35) == [[], [], []]
    assert det.run_caption_pass(tiles, "   ", 0.35) == [[], [], []]


def test_run_caption_pass_delegates_to_dispatch_pass(tmp_path):
    det = _dual_pass_detector(tmp_path, question_nouns=[], vocab_nouns=[])
    det._resolved_device = "cpu"
    captured: dict = {}

    def fake_dispatch(torch, model, predict_fn, tiles, device, dtype, prompt, box_threshold):
        captured["prompt"] = prompt
        captured["box_threshold"] = box_threshold
        captured["device"] = device
        return [[Detection(0, (0, 0, 1, 1), "teapot", 0.9)]]

    det._dispatch_pass = fake_dispatch
    out = det.run_caption_pass(_tiles(1), "teapot .", 0.22)
    assert captured["prompt"] == "teapot ."
    assert captured["box_threshold"] == pytest.approx(0.22)
    assert captured["device"] == "cpu"
    assert [d.label for d in out[0]] == ["teapot"]


def test_run_caption_pass_swaps_and_restores_text_threshold(tmp_path):
    det = _dual_pass_detector(tmp_path, question_nouns=[], vocab_nouns=[])
    det._resolved_device = "cpu"
    det.text_threshold = 0.25
    seen: list[float] = []

    def fake_dispatch(torch, model, predict_fn, tiles, device, dtype, prompt, box_threshold):
        seen.append(det.text_threshold)
        return [[]]

    det._dispatch_pass = fake_dispatch
    det.run_caption_pass(_tiles(1), "teapot .", 0.22, text_threshold=0.4)
    assert seen == [0.4]
    assert det.text_threshold == 0.25  # restored after the pass


def test_run_caption_pass_restores_text_threshold_on_exception(tmp_path):
    det = _dual_pass_detector(tmp_path, question_nouns=[], vocab_nouns=[])
    det._resolved_device = "cpu"
    det.text_threshold = 0.25

    def raising_dispatch(*args, **kwargs):
        raise RuntimeError("boom")

    det._dispatch_pass = raising_dispatch
    with pytest.raises(RuntimeError):
        det.run_caption_pass(_tiles(1), "teapot .", 0.22, text_threshold=0.4)
    assert det.text_threshold == 0.25  # restored even though the pass raised


def test_run_caption_pass_no_text_threshold_leaves_it_unchanged(tmp_path):
    det = _dual_pass_detector(tmp_path, question_nouns=[], vocab_nouns=[])
    det._resolved_device = "cpu"
    det.text_threshold = 0.25
    seen: list[float] = []

    def fake_dispatch(torch, model, predict_fn, tiles, device, dtype, prompt, box_threshold):
        seen.append(det.text_threshold)
        return [[]]

    det._dispatch_pass = fake_dispatch
    det.run_caption_pass(_tiles(1), "teapot .", 0.22)
    assert seen == [0.25]
    assert det.text_threshold == 0.25


def test_run_caption_pass_load_failure_raises_and_does_not_touch_backoff(tmp_path):
    det = _dual_pass_detector(tmp_path, question_nouns=[], vocab_nouns=[])
    det._lazy_import = lambda: (
        _FakeTorch(cuda_available=False),
        _FailingLoad(),
        None,
    )
    assert det._consecutive_load_failures == 0
    with pytest.raises(RuntimeError):
        det.run_caption_pass(_tiles(1), "teapot .", 0.22)
    # Loud failure for the server, not the tick backoff (issue #39 counters untouched).
    assert det._consecutive_load_failures == 0
    assert det._next_retry_at == 0.0


# ------------------------------------------------------------------ #84 raw detection dump


def test_dump_raw_detections_is_noop_without_env_var(monkeypatch, tmp_path):
    monkeypatch.delenv(ENV_RAW_DETECTION_DUMP_PATH, raising=False)
    det = Detection(tile_id=0, bbox_xyxy=(0, 0, 10, 10), label="window", score=0.4)
    dump_raw_detections([(det, GATE_ACCEPTED, 1)], keyframe_idx=0)
    assert list(tmp_path.iterdir()) == []


def test_dump_raw_detections_writes_jsonl_record(monkeypatch, tmp_path):
    out = tmp_path / "raw.jsonl"
    monkeypatch.setenv(ENV_RAW_DETECTION_DUMP_PATH, str(out))
    accepted = Detection(tile_id=0, bbox_xyxy=(0, 0, 10, 20), label="sofa", score=0.42)
    gated = Detection(tile_id=1, bbox_xyxy=(5, 5, 15, 25), label="window", score=0.28)
    dump_raw_detections(
        [(accepted, GATE_ACCEPTED, 3), (gated, GATE_NO_LIDAR_CLUSTER, None)],
        keyframe_idx=7,
    )
    assert out.exists()
    lines = out.read_text().strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["keyframe_idx"] == 7
    assert record["total_detections"] == 2
    assert record["by_class"]["sofa"] == {"total": 1, "accepted": 1, "gated": 0}
    assert record["by_class"]["window"] == {"total": 1, "accepted": 0, "gated": 1}
    dets = {d["label"]: d for d in record["detections"]}
    assert dets["sofa"]["gate"] == GATE_ACCEPTED
    assert dets["sofa"]["instance_id"] == 3
    assert dets["window"]["gate"] == GATE_NO_LIDAR_CLUSTER
    assert dets["window"]["instance_id"] is None


def test_dump_raw_detections_appends_across_calls(monkeypatch, tmp_path):
    out = tmp_path / "raw.jsonl"
    monkeypatch.setenv(ENV_RAW_DETECTION_DUMP_PATH, str(out))
    det = Detection(tile_id=0, bbox_xyxy=(0, 0, 10, 10), label="chair", score=0.5)
    dump_raw_detections([(det, GATE_ACCEPTED, 1)], keyframe_idx=0)
    dump_raw_detections([(det, GATE_ACCEPTED, 1)], keyframe_idx=1)
    assert len(out.read_text().strip().splitlines()) == 2


def test_dump_raw_detections_never_raises_on_bad_path(monkeypatch):
    monkeypatch.setenv(ENV_RAW_DETECTION_DUMP_PATH, "/dev/null/nonexistent/raw.jsonl")
    det = Detection(tile_id=0, bbox_xyxy=(0, 0, 10, 10), label="chair", score=0.5)
    dump_raw_detections([(det, GATE_ACCEPTED, 1)], keyframe_idx=0)  # must not raise


def test_dump_raw_detections_empty_records_still_writes_record(monkeypatch, tmp_path):
    out = tmp_path / "raw.jsonl"
    monkeypatch.setenv(ENV_RAW_DETECTION_DUMP_PATH, str(out))
    dump_raw_detections([], keyframe_idx=2)
    record = json.loads(out.read_text().strip())
    assert record["total_detections"] == 0
    assert record["by_class"] == {}


# --------------------------------------------------------------------------- issue #131: cross-tile NMS


def test_suppress_cross_tile_duplicates_empty_input():
    assert suppress_cross_tile_duplicates([]) == []
    assert suppress_cross_tile_duplicates([[], [], [], []]) == [[], [], [], []]


def test_suppress_cross_tile_duplicates_single_detection_is_a_noop():
    det = Detection(tile_id=0, bbox_xyxy=(10, 10, 50, 50), label="chair", score=0.4)
    out = suppress_cross_tile_duplicates([[det], [], [], []])
    assert out == [[det], [], [], []]


def test_same_tile_duplicate_boxes_collapse_to_the_higher_score():
    """The dominant real-world case (#131 evidence): the question pass and the
    (cadenced) vocab pass both fire on the same tile and both find the same physical
    object, producing two near-identical boxes in ONE tile. The union must keep only
    the higher-scoring one."""
    strong = Detection(tile_id=0, bbox_xyxy=(100.0, 200.0, 180.0, 300.0), label="chair", score=0.55)
    weak = Detection(tile_id=0, bbox_xyxy=(102.0, 198.0, 178.0, 302.0), label="chair", score=0.31)
    out = suppress_cross_tile_duplicates([[strong, weak], [], [], []])
    assert out == [[strong], [], [], []]


def test_cross_tile_duplicate_boxes_collapse_in_the_shared_angular_frame():
    """A duplicate spanning two neighbouring tiles' overlap band must be recognised as
    one object even though the two boxes live in different tiles' pixel spaces (and
    are nowhere near each other as raw pixel coordinates). Uses an explicit hfov wider
    than each tile's angular spacing (:data:`DEFAULT_TILE_HFOV` gives the shipped
    tiling zero nominal overlap — see the module's #131 comment) so the seam-overlap
    band this suppression exists for is actually present, exercising the cross-tile
    path rather than degenerating to same-tile behaviour."""
    hfov = np.deg2rad(100.0)
    strong = Detection(tile_id=0, bbox_xyxy=(0.0, 270.0, 20.0, 370.0), label="chair", score=0.9)
    weak = Detection(tile_id=1, bbox_xyxy=(455.0, 270.0, 470.0, 370.0), label="chair", score=0.6)
    out = suppress_cross_tile_duplicates(
        [[strong], [weak], [], []], n_tiles=4, hfov=hfov, vfov=DEFAULT_TILE_VFOV,
    )
    assert out == [[strong], [], [], []]


def test_distinct_bearings_both_survive():
    """Two same-label boxes at clearly different bearings (e.g. two separate chairs
    across the room) must NOT be merged — this is the false-suppression risk the
    issue explicitly calls out (a near object and a far object along a similar
    bearing must not collide with a near-object-vs-near-object duplicate)."""
    left = Detection(tile_id=0, bbox_xyxy=(10.0, 300.0, 60.0, 400.0), label="chair", score=0.5)
    right = Detection(tile_id=0, bbox_xyxy=(400.0, 300.0, 460.0, 400.0), label="chair", score=0.45)
    out = suppress_cross_tile_duplicates([[left, right], [], [], []])
    assert out == [[left, right], [], [], []]


def test_different_labels_never_compete_even_at_identical_boxes():
    a = Detection(tile_id=0, bbox_xyxy=(100.0, 200.0, 180.0, 300.0), label="chair", score=0.5)
    b = Detection(tile_id=0, bbox_xyxy=(100.0, 200.0, 180.0, 300.0), label="table", score=0.9)
    out = suppress_cross_tile_duplicates([[a, b], [], [], []])
    assert out == [[a, b], [], [], []]


def test_near_far_objects_at_the_same_bearing_are_not_conflated():
    """The issue's headline risk: a near object and a far object at (almost) the same
    bearing subtend very different angular extents, so real depth-distinct instances
    must survive even when centred on the same ray. A far/small box entirely nested
    inside a near/large box's angular extent stays below the geometry-derived 0.75
    threshold once the size difference is large enough (see the threshold's
    docstring: ~1.3x depth ratio already caps IoU near 0.6)."""
    near = Detection(tile_id=0, bbox_xyxy=(150.0, 150.0, 350.0, 450.0), label="chair", score=0.6)
    far = Detection(tile_id=0, bbox_xyxy=(230.0, 270.0, 270.0, 330.0), label="chair", score=0.4)
    out = suppress_cross_tile_duplicates([[near, far], [], [], []])
    assert out == [[near, far], [], [], []]


def test_three_way_duplicate_cluster_keeps_only_the_top_score():
    a = Detection(tile_id=0, bbox_xyxy=(100.0, 200.0, 180.0, 300.0), label="lamp", score=0.3)
    b = Detection(tile_id=0, bbox_xyxy=(101.0, 199.0, 179.0, 301.0), label="lamp", score=0.5)
    c = Detection(tile_id=0, bbox_xyxy=(99.0, 201.0, 181.0, 299.0), label="lamp", score=0.2)
    out = suppress_cross_tile_duplicates([[a, b, c], [], [], []])
    assert out == [[b], [], [], []]


def test_threshold_is_configurable_and_defaults_to_the_module_constant():
    strong = Detection(tile_id=0, bbox_xyxy=(100.0, 200.0, 180.0, 300.0), label="chair", score=0.55)
    weak = Detection(tile_id=0, bbox_xyxy=(140.0, 240.0, 220.0, 340.0), label="chair", score=0.3)
    # Overlapping but not near-identical: below the default (conservative) threshold,
    # both survive; a much lower threshold suppresses the weaker one.
    default_out = suppress_cross_tile_duplicates([[strong, weak], [], [], []])
    assert default_out == [[strong, weak], [], [], []]
    loose_out = suppress_cross_tile_duplicates(
        [[strong, weak], [], [], []], iou_threshold=0.1,
    )
    assert loose_out == [[strong], [], [], []]
    assert CROSS_TILE_NMS_IOU_THRESHOLD == 0.75


def test_groundingdino_call_dedupes_before_returning(monkeypatch):
    """End-to-end wiring check: :meth:`GroundingDinoDetector.__call__` runs the
    question pass then hands the merged union through
    :func:`suppress_cross_tile_duplicates` before returning it — a duplicate from the
    question pass must not reach the caller twice."""
    det = GroundingDinoDetector(question_nouns=["chair"])

    class _FakeModel:
        pass

    det._model = _FakeModel()
    det._resolved_device = "cpu"
    det._resolved_half = False

    def fake_dispatch_pass(torch, model, predict_fn, tiles, device, dtype, prompt, box_threshold):
        strong = Detection(tile_id=0, bbox_xyxy=(100.0, 200.0, 180.0, 300.0), label="chair", score=0.55)
        weak = Detection(tile_id=0, bbox_xyxy=(102.0, 198.0, 178.0, 302.0), label="chair", score=0.31)
        return [[strong, weak], [], [], []]

    class _FakeTorch:
        float32 = "float32"

    monkeypatch.setattr(det, "_dispatch_pass", fake_dispatch_pass)
    monkeypatch.setattr(det, "_lazy_import", lambda: (_FakeTorch, None, None))
    monkeypatch.setattr(det, "_ensure_model", lambda torch, load_model_fn: det._model)

    tiles = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(4)]
    out = det(tiles)
    assert sum(len(t) for t in out) == 1
    assert out[0][0].score == 0.55


# --------------------------------------------------- prompt diagnostic dump (#145) ----


def test_dump_prompt_diagnostics_is_noop_without_env_var(monkeypatch, tmp_path):
    monkeypatch.delenv(ENV_PROMPT_DUMP_PATH, raising=False)
    dump_prompt_diagnostics(
        tag="question_latch",
        question_prompt="sofa .",
        vocab_prompt="sofa . window .",
        dropped_vocab_nouns=["vase"],
    )
    assert list(tmp_path.iterdir()) == []


def test_dump_prompt_diagnostics_writes_jsonl_record(monkeypatch, tmp_path):
    out = tmp_path / "prompt.jsonl"
    monkeypatch.setenv(ENV_PROMPT_DUMP_PATH, str(out))
    dump_prompt_diagnostics(
        tag="question_latch",
        question_prompt="sofa .",
        vocab_prompt="sofa . window .",
        dropped_vocab_nouns=["vase", "tray"],
    )
    assert out.exists()
    lines = out.read_text().strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["tag"] == "question_latch"
    assert record["question_prompt"] == "sofa ."
    assert record["vocab_prompt"] == "sofa . window ."
    assert record["dropped_vocab_nouns"] == ["vase", "tray"]
    assert record["n_dropped_vocab_nouns"] == 2
    assert "wall_time" in record


def test_dump_prompt_diagnostics_appends_across_calls(monkeypatch, tmp_path):
    out = tmp_path / "prompt.jsonl"
    monkeypatch.setenv(ENV_PROMPT_DUMP_PATH, str(out))
    dump_prompt_diagnostics(
        tag="boot_prime", question_prompt="", vocab_prompt="sofa .", dropped_vocab_nouns=[]
    )
    dump_prompt_diagnostics(
        tag="question_latch", question_prompt="sofa .", vocab_prompt="sofa .",
        dropped_vocab_nouns=[],
    )
    assert len(out.read_text().strip().splitlines()) == 2


def test_dump_prompt_diagnostics_never_raises_on_bad_path(monkeypatch):
    monkeypatch.setenv(ENV_PROMPT_DUMP_PATH, "/dev/null/nonexistent/prompt.jsonl")
    dump_prompt_diagnostics(
        tag="question_latch", question_prompt="sofa .", vocab_prompt="sofa .",
        dropped_vocab_nouns=[],
    )  # must not raise


def test_refresh_prompt_dumps_diagnostics_when_env_var_set(monkeypatch, tmp_path):
    # Integration: refresh_prompt (the shared local+remote seam) fires the dump exactly
    # once per call, reporting the SAME dropped-noun set build_gdino_prompt actually
    # dropped -- the exact signal #145's evidence comment says was missing.
    out = tmp_path / "prompt.jsonl"
    monkeypatch.setenv(ENV_PROMPT_DUMP_PATH, str(out))
    det = FakeDetector()
    refresh_prompt(det, ["sofa"], _standing_vocab_nouns())
    lines = out.read_text().strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["tag"] == "question_latch"
    assert "sofa" in record["vocab_prompt"]
    assert record["n_dropped_vocab_nouns"] == len(record["dropped_vocab_nouns"])
    assert record["n_dropped_vocab_nouns"] > 0
    for noun in record["dropped_vocab_nouns"]:
        assert noun not in record["vocab_prompt"].removesuffix(" .").split(" . ")


def test_refresh_prompt_boot_prime_tag_when_no_question_nouns(monkeypatch, tmp_path):
    out = tmp_path / "prompt.jsonl"
    monkeypatch.setenv(ENV_PROMPT_DUMP_PATH, str(out))
    det = FakeDetector()
    refresh_prompt(det, (), _standing_vocab_nouns())
    record = json.loads(out.read_text().strip().splitlines()[0])
    assert record["tag"] == "boot_prime"


def test_refresh_prompt_no_dump_io_without_env_var(monkeypatch, tmp_path):
    monkeypatch.delenv(ENV_PROMPT_DUMP_PATH, raising=False)
    det = FakeDetector()
    refresh_prompt(det, ["sofa"], _standing_vocab_nouns())
    assert list(tmp_path.iterdir()) == []


# ------------------------------------------- oversized-vocab question-noun invariant --


def test_build_gdino_prompt_question_nouns_survive_arbitrarily_oversized_vocab():
    # Issue #145's design goal, generalised past the CURRENT 116-noun standing vocab:
    # a question's own anchor/disambiguator nouns must reach the composed prompt no
    # matter how large the general vocabulary grows -- not just under today's measured
    # size. A synthetic 2000-noun vocab (each one deliberately long, so every single
    # one is expensive) makes the point independent of any real vocab's specifics.
    question_nouns = ["potted plant", "cabinet", "hookah", "tray", "trash can"]
    huge_vocab = [f"synthetic-filler-noun-number-{i:04d}" for i in range(2000)]
    dropped: list[str] = []
    prompt = build_gdino_prompt(question_nouns, huge_vocab, dropped_out=dropped)
    kept_phrases = _kept_phrases(prompt)
    for noun in question_nouns:
        assert noun in kept_phrases
    # sanity: the fixture is meaningful -- the huge vocab actually overflows the budget.
    assert dropped
    for noun in question_nouns:
        assert noun not in dropped


def test_refresh_prompt_question_nouns_survive_arbitrarily_oversized_vocab():
    # Same invariant, exercised through the real refresh_prompt seam (the shared
    # local+remote entry point) rather than build_gdino_prompt directly.
    question_nouns = ["potted plant", "cabinet"]
    huge_vocab = [f"synthetic-filler-noun-number-{i:04d}" for i in range(2000)]
    det = FakeDetector()
    refresh_prompt(det, question_nouns, huge_vocab)
    kept_phrases = _kept_phrases(det.prompt)
    for noun in question_nouns:
        assert noun in kept_phrases


# ---------------------------------------- cost-aware disambiguator ordering (#145) ----


def test_eviction_safe_vocab_order_is_cost_aware_when_headroom_exists():
    # Issue #145: build_gdino_prompt stops trying every subsequent vocab noun the
    # instant ONE fails to fit -- so which priority noun is tried first, among several
    # that don't all fit in the available headroom, determines how many actually make
    # it in. This constructs a synthetic scenario with exactly enough headroom for the
    # CHEAPEST of two priority nouns but not the more expensive one, and confirms the
    # cheap one is rescued -- which the old fixed-discovery-order reorder could not
    # guarantee (a fixed order might have tried the expensive one first and rescued
    # nothing).
    from core.perception.vocab import DISAMBIGUATOR_PRIORITY_NOUNS

    cheap, expensive = "jar", "pyramid candle holder"
    assert cheap in DISAMBIGUATOR_PRIORITY_NOUNS and expensive in DISAMBIGUATOR_PRIORITY_NOUNS

    def cost_estimator(text: str) -> int:
        # word-count based: 1 token/word, so "jar" (1) is far cheaper than
        # "pyramid candle holder" (3), and a filler noun costs exactly 1.
        return len(text.split())

    # A vocab whose survivor prefix leaves headroom for exactly ONE extra word-token:
    # one filler noun that survives, then both priority nouns in the FIXED historical
    # order (expensive first) so the un-cost-aware path would try (and fail on) the
    # expensive one first and never even attempt the cheap one.
    vocab = ["filler", expensive, cheap]
    # Room for "filler" (2 word-tokens) plus " . jar" (2 more: separator + the word) --
    # i.e. exactly enough for the cheap noun once it is tried, but not for "filler" plus
    # " . pyramid candle holder" (4 more word-tokens: separator + 3 words).
    budget = cost_estimator(f"filler . {cheap} .")

    dropped: list[str] = []
    prompt = build_gdino_prompt(
        (), vocab, max_tokens=budget, token_estimator=cost_estimator, dropped_out=dropped,
    )
    kept_phrases = _kept_phrases(prompt)
    assert "filler" in kept_phrases
    # Un-reordered (fixed discovery order tries `expensive` before `cheap`): confirms
    # the fixture is meaningful -- without reordering, the cheap noun would be starved.
    assert expensive not in kept_phrases and cheap not in kept_phrases

    reordered = _eviction_safe_vocab_order(
        (), (), vocab, max_tokens=budget, token_estimator=cost_estimator,
    )
    dropped2: list[str] = []
    prompt2 = build_gdino_prompt(
        (), reordered, max_tokens=budget, token_estimator=cost_estimator, dropped_out=dropped2,
    )
    kept2 = _kept_phrases(prompt2)
    assert "filler" in kept2
    assert cheap in kept2, "cost-aware reorder must rescue the cheaper priority noun"
    assert expensive not in kept2, "still not enough headroom for the expensive one"


def test_eviction_safe_vocab_order_still_never_evicts_a_survivor_when_cost_aware():
    # The core #91 safety invariant must hold under the new cost-aware ordering too.
    vocab = _standing_vocab_nouns()
    dropped_before: list[str] = []
    build_gdino_prompt((), vocab, dropped_out=dropped_before)
    survived_before = {n for n in vocab if n not in set(dropped_before)}

    reordered = _eviction_safe_vocab_order((), (), vocab)
    dropped_after: list[str] = []
    build_gdino_prompt((), reordered, dropped_out=dropped_after)
    survived_after = {n for n in reordered if n not in set(dropped_after)}

    assert survived_before <= survived_after


# ------------------------------------------------------------------ #172: caption-label fallback


def test_caption_nouns_splits_a_rendered_gdino_caption():
    caption = "magazine . ottoman . potted plant . dressing table ."
    assert caption_nouns(caption) == ["magazine", "ottoman", "potted plant", "dressing table"]


def test_caption_nouns_empty_caption_is_empty_list():
    assert caption_nouns("") == []
    assert caption_nouns("   ") == []


def test_resolve_alignment_fallback_single_noun_caption_is_unambiguous():
    # A one-noun caption needs no recovery signal at all -- it is the only candidate.
    assert resolve_alignment_fallback("teapot .") == "teapot"
    assert resolve_alignment_fallback("teapot .", recovery_phrase="") == "teapot"


def test_resolve_alignment_fallback_maps_partial_recovery_phrase_to_matching_noun():
    # Issue #172: the exact live pattern -- a marginal (~0.19) detection whose full
    # text_threshold decode came up empty, but a coarser recovery decode surfaced the
    # word "potted", which shares a word with exactly one of the caption's four nouns.
    caption = "magazine . ottoman . potted plant . dressing table ."
    assert resolve_alignment_fallback(caption, recovery_phrase="potted") == "potted plant"
    assert resolve_alignment_fallback(caption, recovery_phrase="plant") == "potted plant"


def test_resolve_alignment_fallback_no_plausible_noun_drops():
    caption = "magazine . ottoman . potted plant . dressing table ."
    # No recovery signal at all -- multi-noun caption, nothing to disambiguate with.
    assert resolve_alignment_fallback(caption) is None
    assert resolve_alignment_fallback(caption, recovery_phrase="") is None
    # A recovery phrase that shares no word with any caption noun.
    assert resolve_alignment_fallback(caption, recovery_phrase="xyzzy") is None


def test_resolve_alignment_fallback_never_returns_text_containing_the_join_marker():
    caption = "magazine . ottoman . potted plant . dressing table ."
    for recovery_phrase in ("", "potted", "xyzzy", "table dressing"):
        result = resolve_alignment_fallback(caption, recovery_phrase=recovery_phrase)
        if result is not None:
            assert CAPTION_JOIN_MARKER not in result


# ---- local path (GroundingDinoDetector): _decode_batch_item / _call_per_tile wiring


class _FakePosmap:
    """Stand-in for a torch bool tensor: only carries which threshold produced it, so
    the fake ``get_phrases_from_posmap`` below can tell the text_threshold decode
    attempt apart from the box_threshold (coarse recovery) one."""

    def __init__(self, thresh: float) -> None:
        self.thresh = thresh


class _FakeQueryVec:
    """Stand-in for one query's (ntok,) row of ``item_logits``."""

    def __init__(self, peak: float) -> None:
        self._peak = peak

    def __gt__(self, thresh):
        return _FakePosmap(thresh)


class _FakeMaxLogits:
    """Stand-in for ``item_logits.max(dim=1)``'s first element (per-query peak)."""

    def __init__(self, peaks: list[float]) -> None:
        self._peaks = peaks

    def __gt__(self, thresh):
        return _FakeBoolVec([p > thresh for p in self._peaks])

    def __getitem__(self, i):
        return self._peaks[i]

    def __float__(self):
        return float(self._peaks[0])


class _FakeBoolVec(list):
    def nonzero(self):
        return _FakeIdxHolder([i for i, v in enumerate(self) if v])


class _FakeIdxHolder:
    def __init__(self, idxs: list[int]) -> None:
        self._idxs = idxs

    def flatten(self):
        return self

    def tolist(self):
        return self._idxs


class _FakeItemLogits:
    """Stand-in for the (nq, ntok) ``item_logits`` tensor _decode_batch_item receives.
    Only ``max(dim=1)`` and per-query ``__getitem__`` are exercised."""

    def __init__(self, peaks: list[float]) -> None:
        self._peaks = peaks

    def max(self, dim):
        assert dim == 1
        return _FakeMaxLogits(self._peaks), None

    def __getitem__(self, i):
        return _FakeQueryVec(self._peaks[i])


def _install_fake_get_phrases_from_posmap(monkeypatch, text_thresh: float, box_thresh: float, coarse_phrase: str):
    """Fake ``groundingdino.util.utils.get_phrases_from_posmap``: empty phrase at the
    query's own ``text_threshold`` decode (the alignment-failure case #172 fixes), a
    caller-supplied coarse recovery phrase at the lower ``box_threshold`` decode."""
    import sys
    import types

    def fake_get_phrases(posmap, tokenized, tokenizer):
        if posmap.thresh == text_thresh:
            return ""
        assert posmap.thresh == box_thresh
        return coarse_phrase

    monkeypatch.setitem(
        sys.modules, "groundingdino.util.utils",
        types.SimpleNamespace(get_phrases_from_posmap=fake_get_phrases),
    )


def _decode_batch_item_detector(tmp_path, *, box_threshold: float, text_threshold: float, prompt: str):
    import types

    det = _dual_pass_detector(tmp_path, question_nouns=caption_nouns(prompt))
    det.prompt = prompt
    det.box_threshold = box_threshold
    det.text_threshold = text_threshold
    det._model = types.SimpleNamespace(tokenizer=lambda text: {"input_ids": []})
    return det


def test_decode_batch_item_marginal_score_maps_to_matching_noun(monkeypatch, tmp_path):
    """Issue #172, local path: a query scoring 0.19 -- inside #91's admitted
    [box_threshold, text_threshold) band -- decodes an empty phrase at text_threshold.
    _decode_batch_item must retry at the (lower) box_threshold, recover a coarse
    phrase, and resolve it to the caption's single matching noun -- never the caption
    itself."""
    caption = "magazine . ottoman . potted plant . dressing table ."
    _install_fake_get_phrases_from_posmap(
        monkeypatch, text_thresh=0.25, box_thresh=0.18, coarse_phrase="potted",
    )
    det = _decode_batch_item_detector(tmp_path, box_threshold=0.18, text_threshold=0.25, prompt=caption)

    item_logits = _FakeItemLogits([0.19])
    item_boxes = [(0.5, 0.5, 0.1, 0.1)]
    tile = np.zeros((100, 100, 3), dtype=np.uint8)

    dets = det._decode_batch_item(0, tile, item_logits, item_boxes)

    assert len(dets) == 1
    assert dets[0].label == "potted plant"
    assert dets[0].score == pytest.approx(0.19)
    assert CAPTION_JOIN_MARKER not in dets[0].label


def test_decode_batch_item_no_plausible_noun_is_dropped(monkeypatch, tmp_path):
    caption = "magazine . ottoman . potted plant . dressing table ."
    _install_fake_get_phrases_from_posmap(
        monkeypatch, text_thresh=0.25, box_thresh=0.18, coarse_phrase="",
    )
    det = _decode_batch_item_detector(tmp_path, box_threshold=0.18, text_threshold=0.25, prompt=caption)

    item_logits = _FakeItemLogits([0.19])
    item_boxes = [(0.5, 0.5, 0.1, 0.1)]
    tile = np.zeros((100, 100, 3), dtype=np.uint8)

    dets = det._decode_batch_item(0, tile, item_logits, item_boxes)

    assert dets == []  # dropped -- never the raw caption


def _call_per_tile_detector(tmp_path, *, prompt: str):
    import contextlib

    det = _dual_pass_detector(tmp_path, question_nouns=caption_nouns(prompt))
    det.prompt = prompt
    det._to_tensor = lambda tile, torch, device, dtype: "fake_tensor"
    det._forward_ctx = lambda torch: contextlib.nullcontext()
    return det


class _FakeBoxesTensor:
    def __init__(self, rows):
        self._rows = rows

    def tolist(self):
        return self._rows


class _FakeScoresTensor:
    def __init__(self, vals):
        self._vals = vals

    def tolist(self):
        return self._vals


def test_call_per_tile_single_noun_caption_recovers_without_logits(tmp_path):
    # The per-tile predict() fallback has no per-token logits to recover a coarse
    # phrase from, but a one-noun caption needs none -- still never drops a legitimate
    # single-target detection.
    det = _call_per_tile_detector(tmp_path, prompt="teapot .")

    def fake_predict_fn(model, image, caption, box_threshold, text_threshold, device):
        return _FakeBoxesTensor([[0.5, 0.5, 0.1, 0.1]]), _FakeScoresTensor([0.19]), [""]

    per_tile = det._call_per_tile(
        torch=None, model=None, predict_fn=fake_predict_fn,
        tiles=[np.zeros((100, 100, 3), dtype=np.uint8)], device="cpu", dtype="float32",
    )
    dets = per_tile[0]
    assert len(dets) == 1
    assert dets[0].label == "teapot"


def test_call_per_tile_multi_noun_caption_with_no_recovery_signal_drops(tmp_path):
    caption = "magazine . ottoman . potted plant . dressing table ."
    det = _call_per_tile_detector(tmp_path, prompt=caption)

    def fake_predict_fn(model, image, caption, box_threshold, text_threshold, device):
        return _FakeBoxesTensor([[0.5, 0.5, 0.1, 0.1]]), _FakeScoresTensor([0.19]), [""]

    per_tile = det._call_per_tile(
        torch=None, model=None, predict_fn=fake_predict_fn,
        tiles=[np.zeros((100, 100, 3), dtype=np.uint8)], device="cpu", dtype="float32",
    )
    assert per_tile[0] == []  # dropped -- never the raw caption
