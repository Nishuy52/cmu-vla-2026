"""Load provider configuration and build the ladder's ``chat_fns`` list.

Configuration comes from two sources, merged with **environment variables winning** over
an optional ``llm_config.json`` at the repo root. This lets the committed file (if any)
carry non-secret defaults (kinds / base_urls / model names / *which env var holds the
key*) while the actual keys and any host-specific overrides stay in the environment. No
API key is ever read from the JSON file or hard-coded here — a spec names the env var
(``api_key_env``) that holds its key, and the key is read from that env var lazily.

Three provider slots, consumed by the ladder in this fixed order (architecture §1 row 7,
dual-provider failover -> local VLM):

    primary  -> ladder tier "api"
    secondary-> ladder tier "api2"
    local    -> ladder tier "local"

Any slot may be left unconfigured; ``build_chat_fns`` skips empty slots and returns only
the configured adapters, preserving order. A completely empty config yields an empty list,
which the ladder handles by going straight to the deterministic regex floor.

Environment variables
----------------------
Per slot ``SLOT`` in {``PRIMARY``, ``SECONDARY``, ``LOCAL``}:

    VLA_LLM_<SLOT>_KIND        "openai" | "anthropic" | "stub"   (required to enable slot)
    VLA_LLM_<SLOT>_BASE_URL    endpoint base URL (OpenAI-compat / local servers; optional
                               for anthropic which defaults to the SDK's base URL)
    VLA_LLM_<SLOT>_MODEL       model id (required for openai/anthropic)
    VLA_LLM_<SLOT>_API_KEY_ENV name of the env var that HOLDS the api key (indirection;
                               the key itself is never named here). Default per slot:
                               PRIMARY->OPENAI_API_KEY, SECONDARY->ANTHROPIC_API_KEY,
                               LOCAL->VLA_LOCAL_API_KEY.
    VLA_LLM_<SLOT>_STUB_REPLY  (kind=stub only) canned reply text for LocalStub.

Global:

    VLA_LLM_CALL_TIMEOUT_S     per-call timeout seconds for every adapter (default 20).
    VLA_LLM_CONFIG             path to the JSON config file (default: <repo>/llm_config.json).

JSON file (all fields optional; env overrides field-by-field)::

    {
      "call_timeout_s": 20,
      "primary":   {"kind": "openai",    "base_url": "...", "model": "...",
                    "api_key_env": "OPENAI_API_KEY"},
      "secondary": {"kind": "anthropic", "model": "...", "api_key_env": "ANTHROPIC_API_KEY"},
      "local":     {"kind": "openai",    "base_url": "http://localhost:8000/v1",
                    "model": "qwen2.5-vl-7b", "api_key_env": "VLA_LOCAL_API_KEY"}
    }

No keys in this module, no keys in ``llm_config.json`` (commit only the indirection).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.llm.providers import (
    AnthropicChatAdapter,
    LocalStub,
    OpenAIChatAdapter,
    ProviderUnavailable,
)
from core.llm.timeout import DEFAULT_CALL_TIMEOUT_S, with_timeout
from core.parsing.prompts import ChatFn

#: Ladder slots in ladder order. Names map onto ladder.DEFAULT_TIER_NAMES via build order.
SLOTS: tuple[str, ...] = ("primary", "secondary", "local")

#: Default env var name holding each slot's key (indirection only — never the key itself).
_DEFAULT_KEY_ENV: dict[str, str] = {
    "primary": "OPENAI_API_KEY",
    "secondary": "ANTHROPIC_API_KEY",
    "local": "VLA_LOCAL_API_KEY",
}


@dataclass(frozen=True)
class ProviderSpec:
    """Resolved (env + file merged) description of one provider slot."""

    kind: str  # "openai" | "anthropic" | "stub"
    model: str = ""
    base_url: str | None = None
    api_key_env: str = ""
    stub_reply: str = ""  # kind == "stub": canned reply

    def api_key(self) -> str:
        """Read the key from the named env var (empty string if unset/unnamed)."""
        if not self.api_key_env:
            return ""
        return os.environ.get(self.api_key_env, "")


@dataclass
class LlmConfig:
    """Ordered provider slots + global knobs, built from env and/or JSON file."""

    primary: ProviderSpec | None = None
    secondary: ProviderSpec | None = None
    local: ProviderSpec | None = None
    call_timeout_s: float = DEFAULT_CALL_TIMEOUT_S

    def slots(self) -> list[ProviderSpec | None]:
        """The three slots in ladder order (primary, secondary, local)."""
        return [self.primary, self.secondary, self.local]


# --------------------------------------------------------------------- file + env merge


def _repo_root() -> Path:
    """Repo root = two levels above this file (src/core/llm -> src -> repo)."""
    return Path(__file__).resolve().parents[3]


def _load_json(path: Path) -> dict[str, Any]:
    """Load the optional JSON config; missing file or empty path -> empty dict."""
    if not path or not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _env(name: str) -> str | None:
    """Read an env var, treating empty/whitespace as unset."""
    val = os.environ.get(name)
    if val is None:
        return None
    val = val.strip()
    return val or None


def _resolve_slot(slot: str, file_slot: dict[str, Any]) -> ProviderSpec | None:
    """Merge one slot's file dict with env overrides (env wins); None if not enabled.

    A slot is enabled iff a ``kind`` is present from either source. Each field is taken
    from the env var when set, else the file, else a per-slot default.
    """
    prefix = f"VLA_LLM_{slot.upper()}_"

    def pick(field_name: str, env_suffix: str, default: str | None = None) -> str | None:
        env_val = _env(prefix + env_suffix)
        if env_val is not None:
            return env_val
        file_val = file_slot.get(field_name)
        if file_val is not None:
            return str(file_val)
        return default

    kind = pick("kind", "KIND")
    if not kind:
        return None

    return ProviderSpec(
        kind=kind.lower(),
        model=pick("model", "MODEL", "") or "",
        base_url=pick("base_url", "BASE_URL", None),
        api_key_env=pick("api_key_env", "API_KEY_ENV", _DEFAULT_KEY_ENV.get(slot, "")) or "",
        stub_reply=pick("stub_reply", "STUB_REPLY", "") or "",
    )


def load_config(config_path: str | os.PathLike | None = None) -> LlmConfig:
    """Build an :class:`LlmConfig` from ``llm_config.json`` + environment (env wins).

    ``config_path`` overrides the file location; otherwise ``VLA_LLM_CONFIG`` env var,
    else ``<repo>/llm_config.json``. A wholly unconfigured environment yields a config
    whose slots are all ``None`` (ladder -> regex floor).
    """
    path = Path(
        config_path
        or _env("VLA_LLM_CONFIG")
        or (_repo_root() / "llm_config.json")
    )
    file_cfg = _load_json(path)

    timeout_env = _env("VLA_LLM_CALL_TIMEOUT_S")
    if timeout_env is not None:
        try:
            call_timeout_s = float(timeout_env)
        except ValueError:
            call_timeout_s = DEFAULT_CALL_TIMEOUT_S
    else:
        call_timeout_s = float(file_cfg.get("call_timeout_s", DEFAULT_CALL_TIMEOUT_S))

    return LlmConfig(
        primary=_resolve_slot("primary", _as_dict(file_cfg.get("primary"))),
        secondary=_resolve_slot("secondary", _as_dict(file_cfg.get("secondary"))),
        local=_resolve_slot("local", _as_dict(file_cfg.get("local"))),
        call_timeout_s=call_timeout_s,
    )


def _as_dict(v: Any) -> dict[str, Any]:
    return v if isinstance(v, dict) else {}


# ------------------------------------------------------------------- adapter construction


def _build_adapter(spec: ProviderSpec, call_timeout_s: float = DEFAULT_CALL_TIMEOUT_S):
    """Construct the raw (un-timeout-wrapped) adapter for one spec.

    Raises ``ProviderUnavailable`` for an unknown kind or a spec missing required fields.
    Note: this does NOT import or require the provider SDK — that stays lazy inside the
    adapter's call — so an OpenAI adapter can be built with the SDK absent and only fails
    when the ladder actually invokes it.

    ``call_timeout_s`` is threaded into the OpenAI adapter's request-level timeout
    (issue #46) so the configured call timeout closes the connection on expiry, not just
    the outer thread-based ``wrap_call_timeout``/``with_timeout`` backstop applied below.
    """
    kind = spec.kind
    if kind == "stub":
        return LocalStub(spec.stub_reply or "{}")
    if kind == "openai":
        if not spec.base_url or not spec.model:
            raise ProviderUnavailable(
                "openai provider needs both base_url and model "
                f"(got base_url={spec.base_url!r}, model={spec.model!r})"
            )
        return OpenAIChatAdapter(
            spec.base_url, spec.model, spec.api_key(), call_timeout_s=call_timeout_s
        )
    if kind == "anthropic":
        if not spec.model:
            raise ProviderUnavailable("anthropic provider needs a model")
        return AnthropicChatAdapter(spec.model, spec.api_key(), base_url=spec.base_url)
    raise ProviderUnavailable(f"unknown provider kind {kind!r}")


def build_chat_fns(config: LlmConfig) -> list[ChatFn]:
    """Turn a config into the ordered ``chat_fns`` list for ``ladder.parse``.

    Order is [primary, secondary, local], skipping any ``None`` slot. Each adapter is
    wrapped with the per-call timeout backstop. Slots that raise ``ProviderUnavailable``
    during construction (bad kind / missing required field) are skipped with their slot
    left out, so one broken slot never blanks the whole ladder.

    NOTE: this list alone does not carry which ``SLOTS`` entry each fn came from — when a
    slot is skipped, positions shift, so zipping this against ``ladder.DEFAULT_TIER_NAMES``
    by index mislabels ``parse_tier`` (issue #44). Callers that need correct tier stamps
    (i.e. production, not tests that always configure a contiguous prefix of slots) should
    use :func:`build_chat_fns_with_tiers` and pass its tier names through to
    ``ladder.parse(..., tier_names=...)``.
    """
    return [fn for _, fn in build_chat_fns_with_tiers(config)]


def build_chat_fns_with_tiers(config: LlmConfig) -> list[tuple[str, ChatFn]]:
    """Like :func:`build_chat_fns`, but pairs each fn with its actual ``SLOTS`` name.

    Skipped (``None`` or unbuildable) slots leave no gap — the returned list is exactly
    the configured/buildable slots in order, each tagged with the slot name it actually
    came from (``"primary"`` / ``"secondary"`` / ``"local"``, i.e. ``ladder`` tier names
    ``"api"`` / ``"api2"`` / ``"local"`` respectively — ``SLOTS`` and
    ``ladder.DEFAULT_TIER_NAMES`` are positionally aligned). This is the fix for issue #44:
    callers pass ``[name for name, _ in ...]`` as ``ladder.parse``'s ``tier_names`` so the
    recorded ``parse_tier`` reflects the slot that actually answered, not its position in
    the (possibly gapped) fn list.
    """
    from core.parsing.ladder import DEFAULT_TIER_NAMES

    pairs: list[tuple[str, ChatFn]] = []
    for i, spec in enumerate(config.slots()):
        if spec is None:
            continue
        try:
            adapter = _build_adapter(spec, config.call_timeout_s)
        except ProviderUnavailable:
            continue
        tier_name = DEFAULT_TIER_NAMES[i] if i < len(DEFAULT_TIER_NAMES) else SLOTS[i]
        pairs.append((tier_name, with_timeout(adapter, config.call_timeout_s)))
    return pairs
