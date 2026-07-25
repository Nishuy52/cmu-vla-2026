#!/usr/bin/env bash
# Laptop-side lifecycle for the cluster dev-offload servers (issue #86: GroundingDINO +
# a local LLM tier, offloaded off the wedge-prone laptop GPU onto the NUS SoC cluster).
#
# Fairshare etiquette: the cluster bills ALLOCATED time, not busy time — a server sitting
# idle costs exactly as much as one doing real work. Only `start` when you're about to
# iterate, and `stop` the moment you're done; a running-but-unused server is a defect.
# `start` submits one combined job (both servers, one GPU) on the cheap `gpu` partition
# by default (3h cap, GRES `nv` — see tools/cluster/servers.sbatch for the exact costs);
# pass extra sbatch flags through for a longer/different allocation, e.g.:
#   tools/cluster/servers.sh start -p gpu-long -t 8:00:00
#
# Usage:
#   tools/cluster/servers.sh start [extra sbatch args...]
#   tools/cluster/servers.sh stop
#   tools/cluster/servers.sh status
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SSH_ALIAS="xlogin"
JOB_NAME="vla-servers"

usage() {
  echo "usage: $0 {start|stop|status} [extra sbatch args for 'start']" >&2
  exit 1
}

cmd_start() {
  echo "[servers.sh] syncing src/ + tools/ to $SSH_ALIAS:~/vla/ ..."
  rsync -a --delete "$REPO_DIR/src" "$REPO_DIR/tools" "$SSH_ALIAS:~/vla/"

  echo "[servers.sh] submitting $JOB_NAME (gpu partition unless overridden)..."
  local sbatch_out
  sbatch_out="$(ssh "$SSH_ALIAS" "cd \$HOME && sbatch $* ~/vla/tools/cluster/servers.sbatch")"
  echo "$sbatch_out"
  echo
  echo "[servers.sh] job submitted. This allocation bills fairshare from now until it"
  echo "[servers.sh] ends (walltime cap) or you tear it down — run '$0 stop' when done."
  echo "[servers.sh] check readiness with '$0 status', then 'tools/cluster/tunnel.sh'."
}

cmd_stop() {
  echo "[servers.sh] cancelling $JOB_NAME job(s) for \$USER on the cluster..."
  ssh "$SSH_ALIAS" "squeue -u \"\$USER\" -h -n $JOB_NAME -o %i | xargs -r scancel"
  echo "[servers.sh] done. Verify with '$0 status'."
}

cmd_status() {
  echo "[servers.sh] queue:"
  ssh "$SSH_ALIAS" "squeue -u \"\$USER\" -n $JOB_NAME" || true
  echo
  echo "[servers.sh] gdino_server.addr:"
  ssh "$SSH_ALIAS" "cat ~/gdino_server.addr 2>/dev/null" || echo "  (down — no addr file)"
  echo "[servers.sh] ollama_server.addr:"
  ssh "$SSH_ALIAS" "cat ~/ollama_server.addr 2>/dev/null" || echo "  (down — no addr file)"
  echo
  echo "[servers.sh] while running: gpu partition nv = 4.0 fairshare units/hr (gpu-long:"
  echo "[servers.sh] 6.0/hr) — billed whether or not a request is in flight. Stop when done."
}

[[ $# -ge 1 ]] || usage
subcmd="$1"
shift
case "$subcmd" in
  start) cmd_start "$@" ;;
  stop) cmd_stop "$@" ;;
  status) cmd_status "$@" ;;
  *) usage ;;
esac
