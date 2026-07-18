"""Concrete ChatFn / VisionChatFn adapters for the parse ladder.

Each adapter turns the parser's provider-agnostic message list
(``[{'role': 'system'|'user'|'assistant', 'content': str}, ...]``, the shape produced by
``core.parsing.prompts.build_parse_messages``) into one provider's wire format, calls the
provider, and returns the reply as plain text. Nothing here imports a provider SDK at
module load: each SDK is imported lazily *inside the call*, and a missing dependency
raises a clear ``ProviderUnavailable`` only when that adapter is actually invoked — so
the ladder can be built with providers whose SDKs are not installed (they simply fail on
call and the ladder falls through to the next tier), and the test suite runs with no SDK
installed at all.

Two callable shapes:

* ``ChatFn = Callable[[list[dict]], str]`` — the text-only signature the ladder consumes
  unchanged (defined in ``core.parsing.prompts``).
* ``VisionChatFn = Callable[[list[dict], list[bytes]], str]`` — a second variant taking
  raw image bytes (PNG/JPEG) alongside the messages, for the multimodal checkpoints
  (detector-miss recovery, frontier selection). Images are attached to the final user
  message using each provider's own image-part encoding.

Adapters (all constructed from a resolved ``ProviderSpec``):

* ``OpenAIChatAdapter`` — OpenAI-compatible ``/chat/completions``. Covers OpenAI proper,
  Gemini's OpenAI-compat endpoint, and local servers (llama.cpp, vLLM, Ollama) that speak
  the same schema — you just point ``base_url`` at them.
* ``AnthropicChatAdapter`` — Anthropic Messages API (system prompt is a top-level field,
  not a message; image parts use base64 source blocks).
* ``LocalStub`` — no network; returns canned replies for tests and the dark-network floor.
"""
from __future__ import annotations

import base64
from typing import Callable, Sequence

from core.parsing.prompts import ChatFn

#: Multimodal variant of ChatFn: messages plus a list of raw image byte blobs (PNG/JPEG).
VisionChatFn = Callable[[list[dict[str, str]], list[bytes]], str]

#: Default sampling knobs — parsing wants determinism, not creativity.
DEFAULT_MAX_TOKENS: int = 1024
DEFAULT_TEMPERATURE: float = 0.0


class ProviderUnavailable(RuntimeError):
    """Raised when an adapter is called but its SDK / config is missing.

    The message names the missing piece and how to fix it. The ladder treats this like
    any other provider failure (its ``_attempt`` catches ``Exception``) and falls through
    to the next tier, so a misconfigured provider degrades rather than crashes.
    """


def _split_system(messages: Sequence[dict[str, str]]) -> tuple[str, list[dict[str, str]]]:
    """Split a message list into (concatenated system text, non-system messages).

    Anthropic takes the system prompt as a top-level parameter rather than a role, and
    several OpenAI-compat local servers behave better with a single leading system
    message, so both adapters route through this. Multiple system messages are joined
    with blank lines; order of the remaining messages is preserved.
    """
    system_parts: list[str] = []
    rest: list[dict[str, str]] = []
    for m in messages:
        if m.get("role") == "system":
            system_parts.append(m.get("content", ""))
        else:
            rest.append({"role": m["role"], "content": m.get("content", "")})
    return "\n\n".join(p for p in system_parts if p), rest


def _guess_image_media_type(data: bytes) -> str:
    """Sniff PNG vs JPEG from magic bytes; default to PNG.

    Enough for the two formats the perception layer emits (keyframe tiles / panorama
    crops). Providers only need a correct top-level image media type.
    """
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    return "image/png"


# --------------------------------------------------------------------- OpenAI-compatible


class OpenAIChatAdapter:
    """OpenAI-compatible ``/chat/completions`` adapter (OpenAI, Gemini-compat, local).

    Uses the ``openai`` SDK's ``base_url`` override so one adapter serves every server
    that speaks the OpenAI chat schema. Text messages pass through as-is; for the vision
    variant, images are appended to the last user message as ``image_url`` parts with a
    base64 ``data:`` URI (the OpenAI multimodal content-part convention).
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str,
        *,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        call_timeout_s: float | None = None,
    ) -> None:
        self.base_url = base_url
        self.model = model
        self._api_key = api_key
        self.max_tokens = max_tokens
        self.temperature = temperature
        #: Issue #46: request-level timeout threaded into the openai SDK's own
        #: ``timeout=`` kwarg on every ``create()`` call (NOT just the client
        #: constructor). The SDK enforces this via the underlying httpx request
        #: timeout, so a timed-out call actually CLOSES the HTTP connection —
        #: which makes a local OpenAI-compat server (llama.cpp/vLLM) see the
        #: client disconnect and abort generation, instead of continuing to
        #: burn GPU/compute on a call nobody is waiting for anymore. This is a
        #: second, request-scoped enforcement layer on top of (not a
        #: replacement for) the outer thread-based ``wrap_call_timeout``
        #: backstop in core.llm.timeout, which still guards non-network hangs.
        self.call_timeout_s = call_timeout_s

    def _client(self):
        try:
            import openai  # lazy: only needed when this adapter is actually called
        except ImportError as exc:  # pragma: no cover - exercised via monkeypatch in tests
            raise ProviderUnavailable(
                "openai SDK not installed; `pip install openai` to use the "
                f"OpenAI-compatible provider (model={self.model!r}, base_url={self.base_url!r})"
            ) from exc
        return openai.OpenAI(base_url=self.base_url, api_key=self._api_key)

    def _create(self, messages: list[dict]) -> str:
        kwargs: dict = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        if self.call_timeout_s is not None:
            # Request-level timeout (issue #46): the openai SDK forwards this to
            # httpx as the per-request timeout, so it closes the connection on
            # expiry rather than merely raising client-side after the fact.
            kwargs["timeout"] = self.call_timeout_s
        resp = self._client().chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""

    def chat(self, messages: list[dict[str, str]]) -> str:
        """ChatFn entrypoint: plain text messages -> reply text."""
        return self._create([dict(m) for m in messages])

    def vision_chat(self, messages: list[dict[str, str]], images: list[bytes]) -> str:
        """VisionChatFn entrypoint: attach images to the final user message."""
        return self._create(_openai_attach_images(messages, images))

    # a bound method already satisfies the ChatFn / VisionChatFn Callable signatures
    __call__ = chat


def _openai_attach_images(
    messages: Sequence[dict[str, str]], images: Sequence[bytes]
) -> list[dict]:
    """Return messages with images attached to the last user message (OpenAI parts).

    The last user message's string content becomes a ``[{type:text}, {type:image_url}...]``
    content-part list; every other message is copied unchanged. Raises if there is no user
    message to attach to.
    """
    out: list[dict] = [dict(m) for m in messages]
    last_user = _last_index(out, "user")
    if last_user is None:
        raise ProviderUnavailable("vision call has no user message to attach images to")
    text = out[last_user].get("content", "")
    parts: list[dict] = [{"type": "text", "text": text}]
    for data in images:
        media = _guess_image_media_type(data)
        b64 = base64.b64encode(data).decode("ascii")
        parts.append(
            {"type": "image_url", "image_url": {"url": f"data:{media};base64,{b64}"}}
        )
    out[last_user] = {"role": "user", "content": parts}
    return out


# ------------------------------------------------------------------------- Anthropic


class AnthropicChatAdapter:
    """Anthropic Messages API adapter.

    The system prompt is hoisted to the top-level ``system`` parameter (Anthropic does not
    accept a ``system`` role in the messages array). Images use base64 ``source`` blocks
    appended to the last user message's content list.
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        *,
        base_url: str | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
    ) -> None:
        self.model = model
        self._api_key = api_key
        self.base_url = base_url
        self.max_tokens = max_tokens
        self.temperature = temperature

    def _client(self):
        try:
            import anthropic  # lazy: only needed when this adapter is actually called
        except ImportError as exc:  # pragma: no cover - exercised via monkeypatch in tests
            raise ProviderUnavailable(
                "anthropic SDK not installed; `pip install anthropic` to use the "
                f"Anthropic provider (model={self.model!r})"
            ) from exc
        kwargs = {"api_key": self._api_key}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        return anthropic.Anthropic(**kwargs)

    def _create(self, system: str, messages: list[dict]) -> str:
        resp = self._client().messages.create(
            model=self.model,
            system=system,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )
        # content is a list of blocks; concatenate the text blocks
        return "".join(
            getattr(block, "text", "") for block in resp.content
        )

    def chat(self, messages: list[dict[str, str]]) -> str:
        """ChatFn entrypoint: split off the system prompt, forward the rest as text."""
        system, rest = _split_system(messages)
        return self._create(system, [dict(m) for m in rest])

    def vision_chat(self, messages: list[dict[str, str]], images: list[bytes]) -> str:
        """VisionChatFn entrypoint: attach images to the final user message."""
        system, rest = _split_system(messages)
        return self._create(system, _anthropic_attach_images(rest, images))

    __call__ = chat


def _anthropic_attach_images(
    messages: Sequence[dict[str, str]], images: Sequence[bytes]
) -> list[dict]:
    """Return non-system messages with images attached to the last user message.

    Anthropic content blocks: text -> ``{type:text, text}``, image ->
    ``{type:image, source:{type:base64, media_type, data}}``.
    """
    out: list[dict] = [dict(m) for m in messages]
    last_user = _last_index(out, "user")
    if last_user is None:
        raise ProviderUnavailable("vision call has no user message to attach images to")
    text = out[last_user].get("content", "")
    blocks: list[dict] = [{"type": "text", "text": text}]
    for data in images:
        media = _guess_image_media_type(data)
        b64 = base64.b64encode(data).decode("ascii")
        blocks.append(
            {
                "type": "image",
                "source": {"type": "base64", "media_type": media, "data": b64},
            }
        )
    out[last_user] = {"role": "user", "content": blocks}
    return out


# ---------------------------------------------------------------------------- LocalStub


class LocalStub:
    """Offline canned-response ChatFn/VisionChatFn for tests and the dark-network floor.

    Replays ``replies`` in order; when exhausted it repeats the last reply (so a single
    canned Plan JSON serves any number of calls). Records every received message list and
    image batch for assertions. Never touches the network.
    """

    def __init__(self, replies: Sequence[str] | str) -> None:
        if isinstance(replies, str):
            replies = [replies]
        if not replies:
            raise ValueError("LocalStub needs at least one canned reply")
        self._replies = list(replies)
        self._i = 0
        self.calls: list[list[dict[str, str]]] = []
        self.image_calls: list[list[bytes]] = []

    def _next(self) -> str:
        reply = self._replies[min(self._i, len(self._replies) - 1)]
        self._i += 1
        return reply

    def chat(self, messages: list[dict[str, str]]) -> str:
        self.calls.append([dict(m) for m in messages])
        return self._next()

    def vision_chat(self, messages: list[dict[str, str]], images: list[bytes]) -> str:
        self.calls.append([dict(m) for m in messages])
        self.image_calls.append(list(images))
        return self._next()

    __call__ = chat


def _last_index(messages: Sequence[dict], role: str) -> int | None:
    """Index of the last message with the given role, or None."""
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == role:
            return i
    return None
