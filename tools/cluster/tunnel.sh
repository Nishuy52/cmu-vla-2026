#!/usr/bin/env bash
# Laptop-side SSH tunnel to the cluster dev-offload servers (issue #86). Dev-only, NEVER
# the submission path. Start the servers first (tools/cluster/servers.sh start), then run
# this to open the tunnel and print the env exports that point core/ at them. Remember to
# 'tools/cluster/servers.sh stop' when you're done — the allocation bills whether or not
# you're actively tunnelled in.
set -euo pipefail

SSH_ALIAS="xlogin"
GDINO_LOCAL_PORT=8765
OLLAMA_LOCAL_PORT=11434

read_addr() {
  # Missing file (no such job / server not up yet) -> empty string, not an error.
  ssh "$SSH_ALIAS" "cat \$HOME/$1 2>/dev/null" || true
}

port_listening() {
  ss -ltn 2>/dev/null | grep -q "127.0.0.1:$1[[:space:]]" || \
    ss -ltn 2>/dev/null | grep -q ":$1[[:space:]]"
}

gdino_addr="$(read_addr gdino_server.addr)"
ollama_addr="$(read_addr ollama_server.addr)"

if [[ -z "$gdino_addr" && -z "$ollama_addr" ]]; then
  echo "[tunnel.sh] neither server has an addr file on the cluster — nothing to tunnel." >&2
  echo "[tunnel.sh] start them first: tools/cluster/servers.sh start" >&2
  exit 1
fi

l_flags=()

if [[ -n "$gdino_addr" ]]; then
  gdino_node="${gdino_addr%%:*}"
  gdino_port="${gdino_addr##*:}"
  if port_listening "$GDINO_LOCAL_PORT"; then
    echo "[tunnel.sh] 127.0.0.1:$GDINO_LOCAL_PORT already listening — gdino tunnel already up, skipping."
  else
    l_flags+=(-L "${GDINO_LOCAL_PORT}:${gdino_node}:${gdino_port}")
  fi
else
  echo "[tunnel.sh] gdino_server.addr missing — gdino server is down, skipping its tunnel." >&2
fi

if [[ -n "$ollama_addr" ]]; then
  ollama_node="${ollama_addr%%:*}"
  ollama_port="${ollama_addr##*:}"
  if port_listening "$OLLAMA_LOCAL_PORT"; then
    echo "[tunnel.sh] 127.0.0.1:$OLLAMA_LOCAL_PORT already listening — ollama tunnel already up, skipping."
  else
    l_flags+=(-L "${OLLAMA_LOCAL_PORT}:${ollama_node}:${ollama_port}")
  fi
else
  echo "[tunnel.sh] ollama_server.addr missing — ollama server is down, skipping its tunnel." >&2
fi

if [[ ${#l_flags[@]} -gt 0 ]]; then
  echo "[tunnel.sh] opening tunnel: ssh -f -N ${l_flags[*]} $SSH_ALIAS"
  ssh -f -N "${l_flags[@]}" "$SSH_ALIAS"
fi

echo
echo "[tunnel.sh] export these in your shell:"
echo
if [[ -n "$gdino_addr" ]]; then
  echo "export VLA_DETECTOR=remote"
  echo "export VLA_REMOTE_DETECTOR_URL=http://127.0.0.1:${GDINO_LOCAL_PORT}"
fi
if [[ -n "$ollama_addr" ]]; then
  # Exact env names from src/core/llm/config.py's "local" slot. An openai-kind slot with
  # an empty/unset api_key_env target reads "" for the key, which the openai SDK client
  # accepts fine (it only raises for api_key=None) — so no API-key var is needed here.
  echo "export VLA_LLM_LOCAL_KIND=openai"
  echo "export VLA_LLM_LOCAL_BASE_URL=http://127.0.0.1:${OLLAMA_LOCAL_PORT}/v1"
  echo "export VLA_LLM_LOCAL_MODEL=qwen2.5vl:3b"
fi
echo
echo "[tunnel.sh] when done: tools/cluster/servers.sh stop"
