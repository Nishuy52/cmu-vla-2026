"""LLM provider layer: config -> ordered ChatFn ladder + per-call timeout backstop.

Turns environment / ``llm_config.json`` into the ``chat_fns`` list ``ladder.parse``
consumes (``build_chat_fns``), with SDK imports kept lazy so the core suite runs with no
provider SDK installed and no keys anywhere in code.
"""
from core.llm.config import (
    LlmConfig,
    ProviderSpec,
    build_chat_fns,
    load_config,
)
from core.llm.providers import (
    AnthropicChatAdapter,
    LocalStub,
    OpenAIChatAdapter,
    ProviderUnavailable,
    VisionChatFn,
)
from core.llm.timeout import (
    DEFAULT_CALL_TIMEOUT_S,
    call_with_timeout,
    with_timeout,
)

__all__ = [
    "AnthropicChatAdapter",
    "DEFAULT_CALL_TIMEOUT_S",
    "LlmConfig",
    "LocalStub",
    "OpenAIChatAdapter",
    "ProviderSpec",
    "ProviderUnavailable",
    "VisionChatFn",
    "build_chat_fns",
    "call_with_timeout",
    "load_config",
    "with_timeout",
]
