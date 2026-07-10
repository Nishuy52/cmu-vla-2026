"""Config precedence (env over file), build_chat_fns ordering + skipping."""
from __future__ import annotations

import json

import pytest

from core.llm.config import (
    LlmConfig,
    ProviderSpec,
    build_chat_fns,
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
def _clean_env(monkeypatch):
    for k in _ENV_KEYS + ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "VLA_LOCAL_API_KEY"]:
        monkeypatch.delenv(k, raising=False)
    # point config at a nonexistent path by default so no stray repo file leaks in
    monkeypatch.setenv("VLA_LLM_CONFIG", "/nonexistent/llm_config.json")


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


def test_malformed_json_file_is_ignored(tmp_path, monkeypatch):
    p = tmp_path / "llm_config.json"
    p.write_text("{not valid json", encoding="utf-8")
    monkeypatch.setenv("VLA_LLM_CONFIG", str(p))
    cfg = load_config()  # must not raise
    assert cfg.slots() == [None, None, None]
