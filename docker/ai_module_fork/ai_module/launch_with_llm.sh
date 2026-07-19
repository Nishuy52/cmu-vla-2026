#!/usr/bin/env bash
# launch_with_llm.sh — start the baked Ollama server, pre-warm the model, then exec our ROS node.
#
# Phase 3 (docs/local_llm_plan.md "Phase 3 — live integration, IN-IMAGE serving"). This is the
# SUBMISSION launch path: wired as the ai_module service's `command:` in
# docker/ai_module_fork/docker/compose.yml / compose_gpu.yml, in place of launching
# `ros2 launch vla_ai_module ai_module.launch.py` directly. The tail of this script `exec`s that
# exact command, so once the LLM step is done (or skipped) the node comes up identically to the
# pre-Phase-3 launch.
#
# Fork-shaped: lives at docker/ai_module_fork/ai_module/launch_with_llm.sh in THIS repo, synced
# by sync_to_fork.sh to <fork>/ai_module/launch_with_llm.sh (sibling of ai_module/docker/ and
# ai_module/src/), COPYed into the image by the Dockerfile at /opt/vla/launch_with_llm.sh.
#
# Degrade-not-block contract: the parse ladder's local tier is optional (a deterministic regex
# floor always backstops it — docs/local_llm_plan.md "Why this is cheap to do"), so this script
# must never hold up node startup for long. Budget: the connectivity wait + pre-warm together are
# bounded well under ~30s (cold load measured 7-10s in Phase 0; the two per-step timeouts below
# default to 15s + 20s so the worst case is bounded and still leaves the ~5 Hz control loop's
# first tick unblocked past that). Any failure or timeout here logs a clear warning and falls
# through to the ros2 launch anyway — this script only exits non-zero if the final `exec` itself
# fails, never because of an LLM warm-up problem alone.
set -uo pipefail   # deliberately NOT -e: individual LLM warm-up steps are allowed to fail; the
                   # script must keep going and still exec the ros2 launch at the end.

# Native ollama endpoints (health/generate) vs the OpenAI-compat `/v1` path core/llm/config.py's
# VLA_LLM_LOCAL_BASE_URL points at (src/core/llm/config.py) — same host:port, different base path.
OLLAMA_NATIVE_URL="http://${OLLAMA_HOST:-127.0.0.1:11434}"
MODEL="${VLA_LLM_LOCAL_MODEL:-qwen2.5vl:3b}"
READY_TIMEOUT_S="${VLA_LLM_WARMUP_READY_TIMEOUT_S:-15}"
# 60s default: measured dev-box cold load is ~25-40s (2026-07-19, RTX 4060 Laptop);
# 20s produced a spurious "prewarm failed" warning while the load finished anyway.
WARMUP_TIMEOUT_S="${VLA_LLM_WARMUP_TIMEOUT_S:-60}"

log() { echo "[launch_with_llm] $*" >&2; }

log "starting ollama serve (OLLAMA_MODELS=${OLLAMA_MODELS:-unset}, OLLAMA_HOST=${OLLAMA_HOST:-unset}, OLLAMA_CONTEXT_LENGTH=${OLLAMA_CONTEXT_LENGTH:-unset}, OLLAMA_KEEP_ALIVE=${OLLAMA_KEEP_ALIVE:-unset})"
ollama serve >/tmp/ollama_serve.log 2>&1 &
OLLAMA_PID=$!

# Wait for /api/version, bounded by READY_TIMEOUT_S (1 Hz poll — coarse is fine, this only runs
# once at boot).
ready=0
elapsed=0
while [ "${elapsed}" -lt "${READY_TIMEOUT_S}" ]; do
  if curl -fsS -m 2 "${OLLAMA_NATIVE_URL}/api/version" >/dev/null 2>&1; then
    ready=1
    break
  fi
  if ! kill -0 "${OLLAMA_PID}" 2>/dev/null; then
    log "WARNING: ollama serve (pid ${OLLAMA_PID}) exited while waiting for /api/version — see /tmp/ollama_serve.log; continuing degraded (local LLM tier unavailable, ladder falls back to the regex floor / other configured tiers)"
    break
  fi
  sleep 1
  elapsed=$((elapsed + 1))
done

if [ "${ready}" -eq 1 ]; then
  log "ollama up after ${elapsed}s — pre-warming model ${MODEL}"
  if timeout "${WARMUP_TIMEOUT_S}" curl -fsS -m "${WARMUP_TIMEOUT_S}" \
       "${OLLAMA_NATIVE_URL}/api/generate" \
       -d "{\"model\":\"${MODEL}\",\"prompt\":\"hi\",\"stream\":false}" \
       >/tmp/ollama_prewarm.log 2>&1; then
    log "prewarm OK: model ${MODEL} loaded"
  else
    log "WARNING: prewarm of ${MODEL} failed or timed out after ${WARMUP_TIMEOUT_S}s — see /tmp/ollama_prewarm.log; continuing degraded (the first live call pays the cold-load cost instead; the ladder's own 20s per-call cap still bounds it)"
  fi
else
  log "WARNING: ollama did not answer /api/version within ${READY_TIMEOUT_S}s — continuing degraded (local LLM tier unavailable, ladder falls back to the regex floor / other configured tiers)"
fi

log "handing off to ros2 launch (ollama server pid ${OLLAMA_PID} left running in the background)"
exec /bin/bash -lc "source /opt/vla/ws/install/setup.bash && ros2 launch vla_ai_module ai_module.launch.py"
