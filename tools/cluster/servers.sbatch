#!/bin/bash
# ONE combined Slurm job hosting BOTH dev-offload servers (GroundingDINO + ollama) on a
# single GPU allocation (issue #86) — halves fairshare cost vs. two separate jobs, since
# SwinB fp32 (~2-3GB) and qwen2.5vl:7b (~6-7GB) both fit on one 24GB Titan RTX (gpu
# partition GRES `nv`).
#
# FAIRSHARE COSTS (bake into every submit decision — IDLE ALLOCATION BILLS THE SAME AS
# BUSY, there is no discount for a server sitting there unused):
#   CPU               0.5 / unit-hr
#   GPU nv  (Titan RTX, partition gpu)       4.0 / hr   <- default, cheapest, use this
#   GPU nv  (partition gpu-long)             6.0 / hr
#   GPU a100-40                              10 / 15    (never for convenience)
#   GPU h100                                 12 / 30    (never for convenience)
# Fairshare decays with a 10-day half-life, so a wasted allocation is not "free later" —
# it is the most expensive thing this project bills.
#
# Default = one working session: 3h on the cheap `gpu` partition (its hard cap). For a
# longer campaign, override at submit time rather than editing this file:
#   sbatch -p gpu-long -t 8:00:00 servers.sbatch
# Do NOT switch to a100/h100 for convenience — `nv` is the only GPU either server needs.
#
# NEVER leave this running unused. Teardown is one command from the laptop:
#   tools/cluster/servers.sh stop      (or: ssh xlogin scancel <jobid>)
# A running-but-unused server is a defect, not a convenience.
#
# Submit from $HOME on the cluster (tools/cluster/servers.sh does this for you) so the
# -o log below lands at ~/vla_servers.out, not wherever sbatch happened to be invoked.
#SBATCH -J vla-servers
#SBATCH -p gpu
#SBATCH -t 3:00:00
#SBATCH --gpus=nv:1
#SBATCH -x xgpe0,xgpe2,xgpe6
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -o vla_servers.out

set -euo pipefail

VLA_ROOT="${VLA_CLUSTER_ROOT:-$HOME/vla}"

source "$HOME/venv-gpu/bin/activate"
export PYTHONPATH="$VLA_ROOT/src:${PYTHONPATH:-}"

# GDINO server config: respect any pre-set env (e.g. a submit-time override), else the
# pre-baked cluster weights/config paths.
export GDINO_CONFIG_PATH="${GDINO_CONFIG_PATH:-$HOME/models/gdino/GroundingDINO_SwinB_cfg.py}"
export GDINO_CHECKPOINT_PATH="${GDINO_CHECKPOINT_PATH:-$HOME/models/gdino/groundingdino_swinb_cogcoor.pth}"
export GDINO_PRECISION="${GDINO_PRECISION:-fp32}"

# GPU nodes are SHARED: another job (any user) may already hold a well-known port on
# this host — an ollama on 11434 bit us on xgpe5 ("bind: address already in use", and
# the health poll then false-positived against the FOREIGN server). Pick the first free
# port at or above the preferred one; the addr files carry the actual host:port, and
# tunnel.sh parses the port from there, so nothing downstream assumes the defaults.
pick_port() {
  local p="$1"
  while ss -ltnH 2>/dev/null | awk '{print $4}' | grep -q ":$p\$"; do
    p=$((p + 1))
  done
  echo "$p"
}
GDINO_PORT="$(pick_port "${GDINO_SERVER_PORT:-8765}")"
OLLAMA_PORT="$(pick_port "${OLLAMA_SERVER_PORT:-11434}")"

# ollama server config.
export OLLAMA_HOST="0.0.0.0:$OLLAMA_PORT"
export OLLAMA_MODELS="$HOME/ollama-models"
export OLLAMA_KEEP_ALIVE=60m
mkdir -p "$OLLAMA_MODELS"

GDINO_ADDR_FILE="$HOME/gdino_server.addr"
OLLAMA_ADDR_FILE="$HOME/ollama_server.addr"
NODE="$(hostname)"

cleanup() {
  rm -f "$GDINO_ADDR_FILE" "$OLLAMA_ADDR_FILE"
  # Kill whichever child(ren) are still alive — a half-dead allocation (one server up,
  # one crashed) must never keep billing; take the whole job down with it.
  kill "$OLLAMA_PID" "$GDINO_PID" 2>/dev/null || true
  wait "$OLLAMA_PID" "$GDINO_PID" 2>/dev/null || true
}
trap cleanup EXIT

echo "[servers.sbatch] starting ollama serve on $NODE:$OLLAMA_PORT"
"$HOME/ollama/bin/ollama" serve &
OLLAMA_PID=$!

echo "[servers.sbatch] starting gdino_server.py on $NODE:$GDINO_PORT"
python "$VLA_ROOT/tools/cluster/gdino_server.py" --port "$GDINO_PORT" &
GDINO_PID=$!

# Bounded poll: wait for each server's own health endpoint before writing its addr file
# (a stale/never-up addr file is worse than a slow-to-appear one — the laptop side treats
# a missing file as "down" and a present one as "ready to tunnel to").
poll_health() {
  local url="$1" tries=60
  while (( tries > 0 )); do
    if curl -fsS -m 2 "$url" >/dev/null 2>&1; then
      return 0
    fi
    tries=$((tries - 1))
    sleep 5
  done
  return 1
}

if poll_health "http://127.0.0.1:$GDINO_PORT/health"; then
  echo "$NODE:$GDINO_PORT" > "$GDINO_ADDR_FILE"
  echo "[servers.sbatch] gdino_server ready: $(cat "$GDINO_ADDR_FILE")"
else
  echo "[servers.sbatch] gdino_server never became healthy; aborting job" >&2
  exit 1
fi

if poll_health "http://127.0.0.1:$OLLAMA_PORT/api/version"; then
  echo "$NODE:$OLLAMA_PORT" > "$OLLAMA_ADDR_FILE"
  echo "[servers.sbatch] ollama ready: $(cat "$OLLAMA_ADDR_FILE")"
else
  echo "[servers.sbatch] ollama never became healthy; aborting job" >&2
  exit 1
fi

# If EITHER server dies, exit (the trap above tears the sibling down too) rather than
# bill a half-dead allocation for the rest of the walltime.
wait -n "$OLLAMA_PID" "$GDINO_PID"
exit 1
