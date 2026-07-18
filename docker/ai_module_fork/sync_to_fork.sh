#!/usr/bin/env bash
# sync_to_fork.sh — stage our repo's payload into the challenge fork's ai_module/ tree (H7).
#
# The submission fork may modify ONLY ai_module/ (upstream_notes.md gotcha 12). Our code lives
# in THIS repo under src/ and docker/ai_module_fork/docker/Dockerfile. This script copies both
# into the fork's ai_module/ directory in the exact shape the fork build expects:
#
#     <fork>/ai_module/docker/Dockerfile   <- docker/ai_module_fork/docker/Dockerfile (this repo)
#     <fork>/ai_module/src/core/           <- src/core/        (this repo)
#     <fork>/ai_module/src/ros_adapter/    <- src/ros_adapter/ (this repo)
#
# src/tests/ is DELIBERATELY EXCLUDED: the runtime image only needs `core` + `ros_adapter`
# (the adapter node imports neither pytest nor the test tree), so tests only bloat the image on a
# size-budgeted eval host (Simply NUC i9, upstream_notes.md §5). Run the test suite from THIS repo
# on Ubuntu (ubuntu_setup.md §3) — not from inside the fork/image. pytest is still pip-installed in
# the image for optional in-container smoke, but no test *files* are shipped.
#
# Usage:
#     ./docker/ai_module_fork/sync_to_fork.sh <path-to-fork-checkout>
# where <path-to-fork-checkout> is the root of the cloned challenge fork (the dir that CONTAINS
# ai_module/). The build/validate step then runs from inside that fork:
#     cd <fork>/docker && docker compose up --build      (see docker/ai_module_fork/README.md)
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <path-to-fork-checkout>  (the dir containing ai_module/)" >&2
  exit 2
fi

FORK_ROOT="$1"
# Resolve this repo's root from the script location (…/docker/ai_module_fork/sync_to_fork.sh).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

AI_MODULE="$FORK_ROOT/ai_module"
if [[ ! -d "$AI_MODULE" ]]; then
  echo "error: $AI_MODULE does not exist — is '$FORK_ROOT' the fork root (must contain ai_module/)?" >&2
  exit 1
fi

echo "repo   : $REPO_ROOT"
echo "fork   : $FORK_ROOT"
echo "target : $AI_MODULE"

# 1) Dockerfile -> ai_module/docker/Dockerfile
mkdir -p "$AI_MODULE/docker"
cp "$SCRIPT_DIR/docker/Dockerfile" "$AI_MODULE/docker/Dockerfile"
echo "synced : ai_module/docker/Dockerfile"

# 2) src/ (core + ros_adapter, NO tests) -> ai_module/src/
#    rsync with --delete keeps the fork's src/ an exact mirror of ours minus tests; falls back to
#    cp if rsync is unavailable.
mkdir -p "$AI_MODULE/src"
if command -v rsync >/dev/null 2>&1; then
  rsync -a --delete \
    --exclude 'tests/' \
    --exclude '__pycache__/' \
    --exclude '.pytest_cache/' \
    --exclude '*.pyc' \
    "$REPO_ROOT/src/" "$AI_MODULE/src/"
else
  rm -rf "$AI_MODULE/src"/*
  for d in core ros_adapter; do
    cp -r "$REPO_ROOT/src/$d" "$AI_MODULE/src/$d"
  done
  # prune caches that cp copied
  find "$AI_MODULE/src" -type d \( -name '__pycache__' -o -name '.pytest_cache' \) -prune -exec rm -rf {} +
  find "$AI_MODULE/src" -type f -name '*.pyc' -delete
fi
echo "synced : ai_module/src/  (core + ros_adapter; tests excluded)"

echo "done. Next: cd '$FORK_ROOT/docker' && docker compose up --build"
