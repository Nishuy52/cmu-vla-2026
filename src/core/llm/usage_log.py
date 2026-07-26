"""Per-call LLM usage logging: one JSON object per call, appended to a durable log.

Every OpenAI-compatible provider call (any tier: SoCLaaS primary, a local Ollama tier,
or any other ``kind=openai`` slot) records one JSON line via :func:`record_usage` to a
committed, append-only JSONL file — durable across sessions so usage/cost can be
tracked over the life of the project, not just within one run. The log is tagged by
``tier`` (``"api"``/``"api2"``/``"local"``) so SoCLaaS vs local usage stays comparable.

Path resolution: ``VLA_LLM_USAGE_LOG`` env var if set, else ``<repo>/reports/
soclaas_usage.jsonl``. The file is APPENDED to, never truncated or rotated here — that
is a deliberate accumulate-across-sessions design, matching how ``reports/`` already
holds other committed evidence artifacts in this repo.

Logging usage must never break the actual LLM call: any failure while resolving the
path, writing the file, or formatting the record is swallowed by the caller (see
``core.llm.providers.OpenAIChatAdapter``), never raised up through the ChatFn.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

#: One process-wide lock — JSONL append is a single line write, but guard against
#: concurrent callers (e.g. multiple ladder tiers/threads) interleaving partial lines.
_LOCK = threading.Lock()

#: Standard Python logging (not the ROS node logger — this module has no rclpy
#: dependency and adapters/providers must stay usable from plain tests). The adapter
#: process's stdout/stderr (which is what "the adapter log" means for every run in this
#: repo — see e.g. tools/cluster/live_run/cluster_verify_run.sbatch's `> ...adapter.log
#: 2>&1` redirection) captures this via the root logger's default stderr handler.
_LOGGER = logging.getLogger("core.llm.usage")


def _repo_root() -> Path:
    """Repo root = two levels above this file (src/core/llm -> src -> repo)."""
    return Path(__file__).resolve().parents[3]


def usage_log_path() -> Path:
    """Resolve the usage log path: ``VLA_LLM_USAGE_LOG`` env var, else the default.

    Default: ``<repo>/reports/soclaas_usage.jsonl`` (committed, accumulates across
    sessions — never truncated by this module).
    """
    override = os.environ.get("VLA_LLM_USAGE_LOG", "").strip()
    if override:
        return Path(override)
    return _repo_root() / "reports" / "soclaas_usage.jsonl"


def record_usage(
    *,
    tier: str,
    model: str,
    ok: bool,
    latency_ms: float,
    usage: dict[str, Any] | None = None,
    error: str | None = None,
    log_path: str | os.PathLike | None = None,
) -> dict[str, Any]:
    """Build one usage record, append it as a JSON line, and return it.

    ``usage`` is the raw ``{prompt_tokens, completion_tokens, total_tokens}`` dict (any
    missing key is recorded as ``None``); pass ``None`` when the provider returned no
    usage object at all (e.g. some local servers omit it) — the record still carries
    tier/model/ok/latency so local vs primary calls stay comparable even without token
    counts. ``log_path`` overrides :func:`usage_log_path` (tests use this to avoid
    touching the real committed log).
    """
    record: dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "tier": tier,
        "model": model,
        "ok": ok,
        "latency_ms": round(latency_ms, 1),
        "prompt_tokens": (usage or {}).get("prompt_tokens"),
        "completion_tokens": (usage or {}).get("completion_tokens"),
        "total_tokens": (usage or {}).get("total_tokens"),
    }
    if error:
        record["error"] = error

    path = Path(log_path) if log_path is not None else usage_log_path()
    line = json.dumps(record, sort_keys=True)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with _LOCK:
            with path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
    except OSError as exc:  # pragma: no cover - defensive: logging must never crash a call
        _LOGGER.warning("usage log write failed (%s): %s", path, exc)

    _LOGGER.info(summary_line(record))
    return record


def summary_line(record: dict[str, Any]) -> str:
    """One-line human-readable summary of a usage record, for the adapter's own log."""
    bits = [f"tier={record.get('tier')}", f"model={record.get('model')}", f"ok={record.get('ok')}"]
    total = record.get("total_tokens")
    if total is not None:
        bits.append(
            f"tokens={record.get('prompt_tokens')}+{record.get('completion_tokens')}={total}"
        )
    bits.append(f"latency_ms={record.get('latency_ms')}")
    if record.get("error"):
        bits.append(f"error={record['error']!r}")
    return "llm usage: " + " ".join(bits)
