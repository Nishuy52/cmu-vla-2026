"""Adapter wire-format tests: message split, image parts, lazy-import errors.

No network and no real SDK: the fake ``openai`` / ``anthropic`` modules are injected into
``sys.modules`` so the adapters' lazy imports resolve to recording clients.
"""
from __future__ import annotations

import base64
import json

import pytest

from core.llm.providers import (
    AnthropicChatAdapter,
    LocalStub,
    OpenAIChatAdapter,
    ProviderUnavailable,
)
from tests.llm.fakes import (
    FakeAnthropicClient,
    FakeOpenAIClient,
    make_anthropic_module,
    make_openai_module,
)

MESSAGES = [
    {"role": "system", "content": "SYS-PROMPT"},
    {"role": "user", "content": "Question: how many chairs?"},
]

PNG_BYTES = b"\x89PNG\r\n\x1a\n" + b"\x00" * 16
JPEG_BYTES = b"\xff\xd8\xff" + b"\x00" * 16


@pytest.fixture
def fake_openai(monkeypatch):
    mod = make_openai_module(reply='{"ok": true}')
    monkeypatch.setitem(__import__("sys").modules, "openai", mod)
    return mod


@pytest.fixture
def fake_anthropic(monkeypatch):
    mod = make_anthropic_module(reply='{"ok": true}')
    monkeypatch.setitem(__import__("sys").modules, "anthropic", mod)
    return mod


# ---------------------------------------------------------------------------- OpenAI


def test_openai_forwards_messages_verbatim(fake_openai):
    adapter = OpenAIChatAdapter("http://host/v1", "gpt-x", "sk-key")
    out = adapter(MESSAGES)
    assert out == '{"ok": true}'
    client = FakeOpenAIClient.last
    # base_url + key threaded into the client constructor
    assert client.init_kwargs["base_url"] == "http://host/v1"
    assert client.init_kwargs["api_key"] == "sk-key"
    payload = client.create_calls[0]
    assert payload["model"] == "gpt-x"
    # OpenAI keeps the system message as a role in the array (no split)
    assert payload["messages"] == MESSAGES
    assert payload["temperature"] == 0.0


def test_openai_vision_attaches_image_parts(fake_openai):
    adapter = OpenAIChatAdapter("http://host/v1", "gpt-x", "sk-key")
    adapter.vision_chat(MESSAGES, [PNG_BYTES, JPEG_BYTES])
    payload = FakeOpenAIClient.last.create_calls[0]
    # system message untouched; last user message becomes a content-part list
    assert payload["messages"][0] == {"role": "system", "content": "SYS-PROMPT"}
    parts = payload["messages"][1]["content"]
    assert parts[0] == {"type": "text", "text": "Question: how many chairs?"}
    assert parts[1]["type"] == "image_url"
    png_b64 = base64.b64encode(PNG_BYTES).decode("ascii")
    assert parts[1]["image_url"]["url"] == f"data:image/png;base64,{png_b64}"
    # media type sniffed per-image: second is jpeg
    assert parts[2]["image_url"]["url"].startswith("data:image/jpeg;base64,")


def test_openai_missing_sdk_raises_clear_error(monkeypatch):
    # ensure no fake openai is present
    monkeypatch.setitem(__import__("sys").modules, "openai", None)
    adapter = OpenAIChatAdapter("http://host/v1", "gpt-x", "sk-key")
    with pytest.raises(ProviderUnavailable) as ei:
        adapter(MESSAGES)
    assert "openai" in str(ei.value).lower()


# --------------------------------------------------------------------------- Anthropic


def test_anthropic_hoists_system_out_of_messages(fake_anthropic):
    adapter = AnthropicChatAdapter("claude-x", "ak-key")
    out = adapter(MESSAGES)
    assert out == '{"ok": true}'
    client = FakeAnthropicClient.last
    assert client.init_kwargs["api_key"] == "ak-key"
    payload = client.create_calls[0]
    # system is a top-level param, NOT a message role
    assert payload["system"] == "SYS-PROMPT"
    assert payload["model"] == "claude-x"
    assert all(m["role"] != "system" for m in payload["messages"])
    assert payload["messages"] == [{"role": "user", "content": "Question: how many chairs?"}]


def test_anthropic_vision_uses_base64_source_blocks(fake_anthropic):
    adapter = AnthropicChatAdapter("claude-x", "ak-key")
    adapter.vision_chat(MESSAGES, [JPEG_BYTES])
    payload = FakeAnthropicClient.last.create_calls[0]
    assert payload["system"] == "SYS-PROMPT"
    blocks = payload["messages"][0]["content"]
    assert blocks[0] == {"type": "text", "text": "Question: how many chairs?"}
    img = blocks[1]
    assert img["type"] == "image"
    assert img["source"]["type"] == "base64"
    assert img["source"]["media_type"] == "image/jpeg"
    assert img["source"]["data"] == base64.b64encode(JPEG_BYTES).decode("ascii")


def test_anthropic_base_url_optional(fake_anthropic):
    # without base_url, the client is constructed without that kwarg
    AnthropicChatAdapter("claude-x", "ak-key")(MESSAGES)
    assert "base_url" not in FakeAnthropicClient.last.init_kwargs
    # with base_url, it is threaded through
    AnthropicChatAdapter("claude-x", "ak-key", base_url="http://proxy")(MESSAGES)
    assert FakeAnthropicClient.last.init_kwargs["base_url"] == "http://proxy"


def test_anthropic_missing_sdk_raises_clear_error(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "anthropic", None)
    adapter = AnthropicChatAdapter("claude-x", "ak-key")
    with pytest.raises(ProviderUnavailable) as ei:
        adapter(MESSAGES)
    assert "anthropic" in str(ei.value).lower()


# ---------------------------------------------------------------------------- LocalStub


def test_local_stub_replays_then_repeats_last():
    stub = LocalStub(["a", "b"])
    assert stub(MESSAGES) == "a"
    assert stub(MESSAGES) == "b"
    assert stub(MESSAGES) == "b"  # exhausted -> repeats last
    assert len(stub.calls) == 3


def test_local_stub_records_vision_images():
    stub = LocalStub("only")
    stub.vision_chat(MESSAGES, [PNG_BYTES])
    assert stub.image_calls == [[PNG_BYTES]]
    assert stub(MESSAGES) == "only"


def test_local_stub_rejects_empty_replies():
    with pytest.raises(ValueError):
        LocalStub([])


def test_vision_without_user_message_raises(fake_openai):
    adapter = OpenAIChatAdapter("http://host/v1", "gpt-x", "sk-key")
    with pytest.raises(ProviderUnavailable):
        adapter.vision_chat([{"role": "system", "content": "x"}], [PNG_BYTES])
