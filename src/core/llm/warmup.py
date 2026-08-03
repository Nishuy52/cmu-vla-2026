"""Best-effort local-tier warmup: force the local model into VRAM before scored parses.

Issue #82 diagnosis
--------------------
Cluster live runs consistently showed the local (ollama) tier falling through to the
regex floor the moment it was actually the tier being exercised (every batch since the
SoCLaaS primary tier was wired in answers ``tier=api`` first and the local tier is never
reached at all — see the per-run ``verify_batch_*_usage_*.jsonl`` logs, which contain zero
``"tier": "local"`` records across every batch from 27 Jul onward). The original failure
(livingroom_1, 20 Jul) shows the local tier's parsing window running 1.3s -> 21.4s, i.e.
almost exactly ``core.llm.timeout.DEFAULT_CALL_TIMEOUT_S`` (20s), before falling through —
a timeout, not an instant import/config error (those fail in ~1-3ms, as seen in the
`openai SDK not installed` records from before that dependency was installed).

The cluster's ollama server logs (``verify_batch_*_ollama_*.log``) explain the timing:
the TCP port opens (`"Listening on 127.0.0.1:...`") immediately, but ollama then spends
3.5-24s (varies with node/GPU contention) discovering that its bundled ``cuda_v13``
library rejects the shared TITAN V (`compute capability not in compiled architectures`,
cc=700) before it falls back to ``cuda_v12`` and reports the GPU as usable
(`"inference compute"`). The sbatch's own readiness probe only waits for the port to open,
not for that backend-selection step, let alone for the ~3GB ``qwen2.5vl:3b`` GGUF weights
to actually be read from disk and copied into VRAM — ollama does that lazily, on the FIRST
inference request. So whichever call reaches the local tier first pays: GPU backend
selection (if not yet finished) + full cold model load + first-token generation, all
inside the ladder's tight per-call ``call_timeout_s`` (20s) — comfortably enough to blow
that budget under cluster GPU-sharing contention, exactly matching the observed ~20s
timeout-then-fallthrough.

Once warm, local-tier latency is well inside the 20s cap (Gate-4 smoke: 7-10s; the API
tier, sharing the same 20s ladder cap and a comparable model class, logs a 7.0s mean /
13.2s max across batch 699009 — see ``reports/``), so the fix is to pay the cold-load tax
once at boot, before any scored question can reach the ladder, not to loosen the per-call
timeout that protects every other call from a genuinely wedged provider.

This module builds ONLY the local slot's adapter — bypassing the per-call timeout wrapper
built for scored calls, which is far too tight for a cold load — and fires one throwaway
chat call with a generous, boot-time-only budget. Best-effort: any failure (unconfigured
local slot, missing SDK, connection refused, timeout) is swallowed and reported via the
return value, never raised — a failed warmup must not block the boot sequence, and the
ladder's own regex floor remains the safety net if the local tier is genuinely
unreachable.
"""
from __future__ import annotations

from core.llm.config import LlmConfig, _build_adapter, load_config
from core.llm.providers import ProviderUnavailable
from core.llm.timeout import call_with_timeout

#: Cold GGUF weight load (disk read + VRAM copy) for a small vision checkpoint under
#: shared-GPU contention can run well past the 20s scored-call timeout (see module
#: docstring). This is a ONE-TIME boot budget, not a per-question one, so it can afford
#: to be generous — ollama's own model load timeout defaults to 5 minutes
#: (``OLLAMA_LOAD_TIMEOUT``), this sits comfortably under that.
DEFAULT_WARMUP_TIMEOUT_S = 240.0

#: Minimal chat turn — small enough to add negligible tokens/latency once the model is
#: warm, non-empty because some OpenAI-compat servers reject an empty user message.
_WARMUP_MESSAGES: list[dict[str, str]] = [{"role": "user", "content": "ready"}]


def warm_up_local(
    config: LlmConfig | None = None, *, timeout_s: float = DEFAULT_WARMUP_TIMEOUT_S
) -> bool:
    """Fire one throwaway chat call at the local slot to force its model into VRAM.

    Returns ``True`` on a successful reply, ``False`` for any failure (unconfigured local
    slot, missing SDK, connection refused, timeout) — never raises. Call this once, after
    the local ollama server reports its port open and before any scored parse can reach
    the ladder (the cluster sbatch's ollama boot block does exactly this).

    The adapter is built with ``call_timeout_s=timeout_s`` (the warmup budget), NOT the
    resolved config's ``call_timeout_s`` (the scored ladder's 20s cap) — the OpenAI
    adapter threads its ``call_timeout_s`` into the SDK's own request-level timeout, so
    passing the scored cap here would just reproduce the exact failure this function
    exists to prevent.
    """
    cfg = config if config is not None else load_config()
    if cfg.local is None:
        return False
    try:
        adapter = _build_adapter(cfg.local, timeout_s, tier_name="local")
    except ProviderUnavailable:
        return False
    try:
        call_with_timeout(lambda: adapter(_WARMUP_MESSAGES), timeout_s)
    except Exception:
        return False
    return True
