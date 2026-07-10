"""In-process fake SDK modules for the OpenAI / Anthropic adapters.

The adapters import their SDK lazily inside the call (``import openai`` / ``import
anthropic``). These fakes are injected into ``sys.modules`` under those names so the
import resolves to a fake client that records the exact request payload and returns a
scripted reply — no network, no real SDK installed. Shape mirrors the slices of each SDK
the adapters actually touch (``chat.completions.create`` / ``messages.create`` and their
response objects).
"""
from __future__ import annotations

import types
from typing import Any


# ------------------------------------------------------------------ OpenAI-compat fake


class _OAMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _OAChoice:
    def __init__(self, content: str) -> None:
        self.message = _OAMessage(content)


class _OAResponse:
    def __init__(self, content: str) -> None:
        self.choices = [_OAChoice(content)]


class FakeOpenAIClient:
    """Records init kwargs + create() calls; returns a scripted reply."""

    #: Class-level sinks so the test can read what the adapter sent without holding the
    #: instance (the adapter constructs the client itself).
    last: "FakeOpenAIClient | None" = None
    reply: str = "{}"

    def __init__(self, **kwargs: Any) -> None:
        self.init_kwargs = kwargs
        self.create_calls: list[dict[str, Any]] = []
        type(self).last = self

        outer = self

        class _Completions:
            def create(self, **payload: Any) -> _OAResponse:
                outer.create_calls.append(payload)
                return _OAResponse(type(outer).reply)

        self.chat = types.SimpleNamespace(completions=_Completions())


def make_openai_module(reply: str = "{}") -> types.ModuleType:
    """Build a fake ``openai`` module whose ``OpenAI`` returns a recording client."""
    FakeOpenAIClient.reply = reply
    FakeOpenAIClient.last = None
    mod = types.ModuleType("openai")
    mod.OpenAI = FakeOpenAIClient  # type: ignore[attr-defined]
    return mod


# ---------------------------------------------------------------------- Anthropic fake


class _AnthTextBlock:
    def __init__(self, text: str) -> None:
        self.type = "text"
        self.text = text


class _AnthResponse:
    def __init__(self, text: str) -> None:
        self.content = [_AnthTextBlock(text)]


class FakeAnthropicClient:
    last: "FakeAnthropicClient | None" = None
    reply: str = "{}"

    def __init__(self, **kwargs: Any) -> None:
        self.init_kwargs = kwargs
        self.create_calls: list[dict[str, Any]] = []
        type(self).last = self

        outer = self

        class _Messages:
            def create(self, **payload: Any) -> _AnthResponse:
                outer.create_calls.append(payload)
                return _AnthResponse(type(outer).reply)

        self.messages = _Messages()


def make_anthropic_module(reply: str = "{}") -> types.ModuleType:
    """Build a fake ``anthropic`` module whose ``Anthropic`` returns a recording client."""
    FakeAnthropicClient.reply = reply
    FakeAnthropicClient.last = None
    mod = types.ModuleType("anthropic")
    mod.Anthropic = FakeAnthropicClient  # type: ignore[attr-defined]
    return mod
