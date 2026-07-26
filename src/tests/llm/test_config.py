"""Config precedence (env over file), build_chat_fns ordering + skipping."""
from __future__ import annotations

import json

import pytest

from core.llm.config import (
    LlmConfig,
    ProviderSpec,
    build_chat_fns,
    build_chat_fns_with_tiers,
    load_config,
)
from core.llm.providers import LocalStub

# env var names this module manipulates — cleared before each test for isolation
_ENV_KEYS = [
    "VLA_LLM_CONFIG",
    "VLA_LLM_CALL_TIMEOUT_S",
]
for slot in ("PRIMARY", "SECONDARY", "LOCAL"):
    for suffix in ("KIND", "BASE_URL", "MODEL", "API_KEY_ENV", "STUB_REPLY"):
        _ENV_KEYS.append(f"VLA_LLM_{slot}_{suffix}")


@pytest.fixture(autouse=True)
def _clean_env(tmp_path, monkeypatch):
    for k in _ENV_KEYS + [
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "VLA_LOCAL_API_KEY",
        "SOCLAAS_API_KEY",
    ]:
        monkeypatch.delenv(k, raising=False)
    # point config at a nonexistent path by default so no stray repo file leaks in
    monkeypatch.setenv("VLA_LLM_CONFIG", "/nonexistent/llm_config.json")
    # any real OpenAIChatAdapter built here logs usage (core.llm.usage_log) — redirect
    # away from the real committed reports/soclaas_usage.jsonl.
    monkeypatch.setenv("VLA_LLM_USAGE_LOG", str(tmp_path / "_default_usage.jsonl"))


def _write_file(tmp_path, data: dict) -> str:
    p = tmp_path / "llm_config.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return str(p)


def test_empty_env_and_no_file_gives_all_none():
    cfg = load_config()
    assert cfg.slots() == [None, None, None]
    assert build_chat_fns(cfg) == []


def test_env_only_configures_slots(monkeypatch):
    monkeypatch.setenv("VLA_LLM_PRIMARY_KIND", "openai")
    monkeypatch.setenv("VLA_LLM_PRIMARY_BASE_URL", "http://p/v1")
    monkeypatch.setenv("VLA_LLM_PRIMARY_MODEL", "gpt-x")
    cfg = load_config()
    assert cfg.primary is not None
    assert cfg.primary.kind == "openai"
    assert cfg.primary.base_url == "http://p/v1"
    assert cfg.secondary is None and cfg.local is None


def test_file_only_configures_slots(tmp_path, monkeypatch):
    path = _write_file(
        tmp_path,
        {"primary": {"kind": "anthropic", "model": "claude-x"}},
    )
    monkeypatch.setenv("VLA_LLM_CONFIG", path)
    cfg = load_config()
    assert cfg.primary is not None and cfg.primary.kind == "anthropic"
    assert cfg.primary.model == "claude-x"


def test_env_overrides_file_field_by_field(tmp_path, monkeypatch):
    path = _write_file(
        tmp_path,
        {"primary": {"kind": "openai", "base_url": "http://file/v1", "model": "file-model"}},
    )
    monkeypatch.setenv("VLA_LLM_CONFIG", path)
    monkeypatch.setenv("VLA_LLM_PRIMARY_MODEL", "env-model")  # overrides just model
    cfg = load_config()
    assert cfg.primary.model == "env-model"  # env wins
    assert cfg.primary.base_url == "http://file/v1"  # file value survives for unset field


def test_call_timeout_env_over_file(tmp_path, monkeypatch):
    path = _write_file(tmp_path, {"call_timeout_s": 5})
    monkeypatch.setenv("VLA_LLM_CONFIG", path)
    assert load_config().call_timeout_s == 5.0
    monkeypatch.setenv("VLA_LLM_CALL_TIMEOUT_S", "12.5")
    assert load_config().call_timeout_s == 12.5


def test_api_key_read_from_named_env_var(monkeypatch):
    monkeypatch.setenv("MY_SECRET_KEY", "sk-live-123")
    spec = ProviderSpec(kind="openai", model="m", base_url="u", api_key_env="MY_SECRET_KEY")
    assert spec.api_key() == "sk-live-123"


def test_default_key_env_names_per_slot(monkeypatch):
    monkeypatch.setenv("VLA_LLM_PRIMARY_KIND", "openai")
    monkeypatch.setenv("VLA_LLM_PRIMARY_BASE_URL", "u")
    monkeypatch.setenv("VLA_LLM_PRIMARY_MODEL", "m")
    monkeypatch.setenv("VLA_LLM_SECONDARY_KIND", "anthropic")
    monkeypatch.setenv("VLA_LLM_SECONDARY_MODEL", "m2")
    cfg = load_config()
    assert cfg.primary.api_key_env == "OPENAI_API_KEY"
    assert cfg.secondary.api_key_env == "ANTHROPIC_API_KEY"


def test_build_chat_fns_order_and_skip():
    cfg = LlmConfig(
        primary=ProviderSpec(kind="stub", stub_reply="P"),
        secondary=None,  # skipped
        local=ProviderSpec(kind="stub", stub_reply="L"),
    )
    fns = build_chat_fns(cfg)
    assert len(fns) == 2  # secondary skipped, order preserved
    # each stub replies with its own canned text -> proves order primary, local
    assert fns[0]([{"role": "user", "content": "x"}]) == "P"
    assert fns[1]([{"role": "user", "content": "x"}]) == "L"


def test_build_chat_fns_skips_broken_slot():
    # openai slot missing base_url/model -> ProviderUnavailable at build -> skipped
    cfg = LlmConfig(
        primary=ProviderSpec(kind="openai", model="", base_url=None),
        secondary=ProviderSpec(kind="stub", stub_reply="S"),
    )
    fns = build_chat_fns(cfg)
    assert len(fns) == 1
    assert fns[0]([{"role": "user", "content": "x"}]) == "S"


def test_build_chat_fns_wraps_with_timeout():
    # a stub replies fine within the (large) timeout; wrapper is transparent
    cfg = LlmConfig(primary=ProviderSpec(kind="stub", stub_reply="ok"), call_timeout_s=5.0)
    fns = build_chat_fns(cfg)
    assert fns[0]([{"role": "user", "content": "x"}]) == "ok"


def test_build_adapter_threads_call_timeout_into_openai_request(monkeypatch):
    # Issue #46: the resolved config's call_timeout_s must reach the OpenAI adapter's
    # request-level timeout, not just the outer thread-based backstop.
    from tests.llm.fakes import FakeOpenAIClient, make_openai_module

    monkeypatch.setitem(__import__("sys").modules, "openai", make_openai_module(reply="ok"))
    cfg = LlmConfig(
        primary=ProviderSpec(
            kind="openai", base_url="http://host/v1", model="gpt-x", api_key_env=""
        ),
        call_timeout_s=9.0,
    )
    fns = build_chat_fns(cfg)
    fns[0]([{"role": "user", "content": "x"}])
    assert FakeOpenAIClient.last.create_calls[0]["timeout"] == 9.0


def test_build_chat_fns_with_tiers_only_local_configured():
    # issue #44 repro: only the local slot is configured -> its tier name must be
    # "local", not "api" (which is what index-0 would get from the gapped fn list).
    cfg = LlmConfig(local=ProviderSpec(kind="stub", stub_reply="L"))
    pairs = build_chat_fns_with_tiers(cfg)
    assert [name for name, _ in pairs] == ["local"]
    assert pairs[0][1]([{"role": "user", "content": "x"}]) == "L"


def test_build_chat_fns_with_tiers_skips_gap_correctly():
    # secondary skipped (None) -> primary/local keep their true tier names, no shift.
    cfg = LlmConfig(
        primary=ProviderSpec(kind="stub", stub_reply="P"),
        secondary=None,
        local=ProviderSpec(kind="stub", stub_reply="L"),
    )
    pairs = build_chat_fns_with_tiers(cfg)
    assert [name for name, _ in pairs] == ["api", "local"]


def test_build_chat_fns_with_tiers_skips_broken_slot_keeps_tier_alignment():
    # primary broken (skipped at build) -> secondary keeps its true tier "api2".
    cfg = LlmConfig(
        primary=ProviderSpec(kind="openai", model="", base_url=None),
        secondary=ProviderSpec(kind="stub", stub_reply="S"),
    )
    pairs = build_chat_fns_with_tiers(cfg)
    assert [name for name, _ in pairs] == ["api2"]


def test_build_chat_fns_matches_build_chat_fns_with_tiers_fns():
    # build_chat_fns is defined in terms of build_chat_fns_with_tiers; each call builds
    # fresh adapter/timeout-wrapper closures, so compare behavior, not identity/equality.
    cfg = LlmConfig(
        primary=ProviderSpec(kind="stub", stub_reply="P"),
        local=ProviderSpec(kind="stub", stub_reply="L"),
    )
    plain = build_chat_fns(cfg)
    tiered = build_chat_fns_with_tiers(cfg)
    assert len(plain) == len(tiered) == 2
    msg = [{"role": "user", "content": "x"}]
    assert [fn(msg) for fn in plain] == [fn(msg) for _, fn in tiered] == ["P", "L"]


def test_default_primary_is_soclaas_when_key_present(monkeypatch):
    # issue: SoCLaaS is the DEFAULT primary tier now — no VLA_LLM_PRIMARY_* needed,
    # only the key env var itself.
    monkeypatch.setenv("SOCLAAS_API_KEY", "sk-soclaas-live")
    cfg = load_config()
    assert cfg.primary is not None
    assert cfg.primary.kind == "openai"
    assert cfg.primary.base_url == "https://soclaas-api.comp.nus.edu.sg/v1"
    assert cfg.primary.model == "qwen3.6:35b"
    assert cfg.primary.api_key_env == "SOCLAAS_API_KEY"
    assert cfg.primary.api_key() == "sk-soclaas-live"


def test_default_primary_absent_without_key_degrades_like_before(monkeypatch):
    # no SOCLAAS_API_KEY, no other override -> primary stays unconfigured, exactly the
    # pre-existing "empty environment" behavior (ladder falls straight to local/regex).
    cfg = load_config()
    assert cfg.primary is None
    assert cfg.slots() == [None, None, None]
    assert build_chat_fns(cfg) == []


def test_explicit_primary_override_wins_over_soclaas_default(monkeypatch):
    monkeypatch.setenv("SOCLAAS_API_KEY", "sk-soclaas-live")
    monkeypatch.setenv("VLA_LLM_PRIMARY_KIND", "anthropic")
    monkeypatch.setenv("VLA_LLM_PRIMARY_MODEL", "claude-explicit")
    cfg = load_config()
    assert cfg.primary is not None
    assert cfg.primary.kind == "anthropic"
    assert cfg.primary.model == "claude-explicit"


def test_explicit_primary_field_override_wins_field_by_field_over_soclaas_default(monkeypatch):
    # env still wins per-field even while the SoCLaaS default is active for the rest.
    monkeypatch.setenv("SOCLAAS_API_KEY", "sk-soclaas-live")
    monkeypatch.setenv("VLA_LLM_PRIMARY_MODEL", "qwen-custom")
    cfg = load_config()
    assert cfg.primary is not None
    assert cfg.primary.kind == "openai"  # from the SoCLaaS default
    assert cfg.primary.base_url == "https://soclaas-api.comp.nus.edu.sg/v1"  # default survives
    assert cfg.primary.model == "qwen-custom"  # explicit env wins


def test_file_primary_entry_wins_over_soclaas_default(tmp_path, monkeypatch):
    monkeypatch.setenv("SOCLAAS_API_KEY", "sk-soclaas-live")
    path = _write_file(
        tmp_path, {"primary": {"kind": "anthropic", "model": "claude-from-file"}}
    )
    monkeypatch.setenv("VLA_LLM_CONFIG", path)
    cfg = load_config()
    assert cfg.primary.kind == "anthropic"
    assert cfg.primary.model == "claude-from-file"


def test_malformed_json_file_is_ignored(tmp_path, monkeypatch):
    p = tmp_path / "llm_config.json"
    p.write_text("{not valid json", encoding="utf-8")
    monkeypatch.setenv("VLA_LLM_CONFIG", str(p))
    cfg = load_config()  # must not raise
    assert cfg.slots() == [None, None, None]
