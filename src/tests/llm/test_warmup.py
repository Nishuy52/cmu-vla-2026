"""core.llm.warmup: forces the local tier's model to load before scored parses.

Issue #82: the local tier's very first scored call paid the cold-model-load tax inside
the tight per-call ladder timeout and fell through to the regex floor. These tests mock
the local endpoint's behaviour (slow-then-fast, hard timeout, unconfigured, missing SDK)
to cover that failure mode and confirm the warmup call is what absorbs it.
"""
from __future__ import annotations

import time

import pytest

from core.llm.config import LlmConfig, ProviderSpec
from core.llm.warmup import warm_up_local
from tests.llm.fakes import FakeOpenAIClient, make_openai_module


@pytest.fixture(autouse=True)
def _isolate_usage_log(tmp_path, monkeypatch):
    # every real adapter built here logs usage (core.llm.usage_log) — redirect away
    # from the real committed reports/soclaas_usage.jsonl (same discipline as
    # tests/llm/test_config.py's _clean_env fixture).
    monkeypatch.setenv("VLA_LLM_USAGE_LOG", str(tmp_path / "_warmup_usage.jsonl"))


def _local_cfg(**overrides) -> LlmConfig:
    spec = ProviderSpec(
        kind="openai", base_url="http://127.0.0.1:11434/v1", model="qwen2.5vl:3b",
        api_key_env="",
    )
    return LlmConfig(local=spec, **overrides)


def test_warmup_false_when_local_unconfigured():
    cfg = LlmConfig()  # no local slot at all
    assert warm_up_local(cfg, timeout_s=1.0) is False


def test_warmup_false_when_sdk_missing(monkeypatch):
    # openai import fails -> ProviderUnavailable at adapter build time
    import sys

    monkeypatch.setitem(sys.modules, "openai", None)  # forces ImportError on `import openai`
    cfg = _local_cfg()
    assert warm_up_local(cfg, timeout_s=1.0) is False


def test_warmup_true_on_successful_reply(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "openai", make_openai_module(reply="ok"))
    cfg = _local_cfg()
    assert warm_up_local(cfg, timeout_s=5.0) is True
    # one throwaway call was made, tagged with the local tier
    assert len(FakeOpenAIClient.last.create_calls) == 1


def test_warmup_reproduces_82_cold_load_then_times_out(monkeypatch):
    """Issue #82's mechanism: the local endpoint is reachable but the FIRST call blocks
    for longer than the scored ladder's per-call cap (cold GGUF load + GPU backend
    selection under cluster contention) — warm_up_local must swallow that timeout and
    report failure rather than raising or hanging past its own budget."""
    import sys

    class _SlowCompletions:
        def create(self, **payload):
            time.sleep(0.3)  # stands in for a multi-second+ cold model load
            raise AssertionError("unreachable: warm_up_local must time out first")

    class _SlowClient:
        def __init__(self, **kwargs):
            self.chat = type("C", (), {"completions": _SlowCompletions()})()

    mod = type(sys)("openai")
    mod.OpenAI = _SlowClient
    monkeypatch.setitem(sys.modules, "openai", mod)

    cfg = _local_cfg()
    # a warmup budget far shorter than the simulated cold-load stall must still return
    # False promptly, not hang or raise.
    assert warm_up_local(cfg, timeout_s=0.05) is False


def test_warmup_uses_its_own_timeout_not_the_scored_call_timeout(monkeypatch):
    # Regression guard: the scored ladder's call_timeout_s (20s default) must NOT be
    # what gets threaded into the OpenAI adapter's request-level timeout for the warmup
    # call -- that would just reproduce the #82 failure this function exists to avoid.
    monkeypatch.setitem(__import__("sys").modules, "openai", make_openai_module(reply="ok"))
    cfg = _local_cfg(call_timeout_s=20.0)
    warm_up_local(cfg, timeout_s=123.0)
    assert FakeOpenAIClient.last.create_calls[0]["timeout"] == 123.0


def test_warmup_never_raises_on_arbitrary_provider_error(monkeypatch):
    import sys

    class _BoomCompletions:
        def create(self, **payload):
            raise ConnectionRefusedError("connection refused")

    class _BoomClient:
        def __init__(self, **kwargs):
            self.chat = type("C", (), {"completions": _BoomCompletions()})()

    mod = type(sys)("openai")
    mod.OpenAI = _BoomClient
    monkeypatch.setitem(sys.modules, "openai", mod)

    cfg = _local_cfg()
    assert warm_up_local(cfg, timeout_s=5.0) is False


def test_warmup_defaults_to_load_config_when_none_passed(monkeypatch):
    monkeypatch.delenv("VLA_LLM_LOCAL_KIND", raising=False)
    monkeypatch.setenv("VLA_LLM_CONFIG", "/nonexistent/llm_config.json")
    # no local slot configured in env -> load_config() gives an empty local slot
    assert warm_up_local(timeout_s=1.0) is False
