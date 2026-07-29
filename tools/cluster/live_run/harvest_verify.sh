#!/bin/bash
# Harvest a cluster verify run (single OR batch) into reports/cluster_verify/<job>/,
# print the verdict block(s), and (if ros bag(s) were captured) score them with the
# SAME scorers as the offline gt_battery. Usage:
#   tools/cluster/live_run/harvest_verify.sh <jobid>
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../../.." && pwd)
J=${1:?jobid}
OUT=$REPO/reports/cluster_verify/$J
mkdir -p "$OUT"

# Detect single-question vs batch naming — a job only ever produces one of
# the two ($SLURM_JOB_ID is unique per allocation, so no ambiguity).
IS_BATCH=0
if ssh xlogin "test -f ~/verify_batch_${J}.out"; then
  IS_BATCH=1
fi

if [ "$IS_BATCH" -eq 1 ]; then
  echo "== batch job $J"
  scp -q "xlogin:verify_batch_${J}.out" "$OUT/verify_batch.out" 2>/dev/null || true
  # Per-slot logs are named *_<log>_<slot>.{log,jsonl} — pull everything for
  # this job in one shot rather than enumerating slots (the matrix that
  # produced them lives on the remote, not here).
  scp -q "xlogin:verify_batch_${J}_*.log" "xlogin:verify_batch_${J}_*.jsonl" "$OUT/" 2>/dev/null || true
  # debug dirs: verify_batch_${J}_debug_<slot>/ -> $OUT/debug/<slot>/
  mkdir -p "$OUT/debug"
  for d in $(ssh xlogin "ls -d ~/verify_batch_${J}_debug_* 2>/dev/null" || true); do
    slot=${d#*verify_batch_${J}_debug_}
    rsync -az "xlogin:$d/" "$OUT/debug/$slot/" 2>/dev/null || true
  done
  # ros bag captures: ~/verify_batch_${J}_captures/<slot>/<scene>/<qdir>/bag
  rsync -az "xlogin:verify_batch_${J}_captures/" "$OUT/captures/" 2>/dev/null || true
  echo "== harvested to $OUT"
  sed -n '/^== slot .* monitor rc/,/^SLOT /p' "$OUT/verify_batch.out" 2>/dev/null | head -400
else
  echo "== single-question job $J"
  # Prefer the job-specific stdout (concurrent jobs share ~/verify_run.out and
  # would clobber it); fall back to the shared name for older runs.
  scp -q "xlogin:verify_run_${J}.out" "$OUT/verify_run.out" 2>/dev/null \
    || scp -q "xlogin:verify_run.out" "$OUT/verify_run.out" || true
  scp -q "xlogin:verify_run_${J}_adapter.log" "xlogin:verify_run_${J}_ollama.log" \
        "xlogin:verify_run_${J}_launch.log" "xlogin:verify_run_${J}_bag.log" \
        "xlogin:verify_run_${J}_usage.jsonl" "$OUT/" 2>/dev/null || true
  rsync -az "xlogin:verify_run_${J}_debug/" "$OUT/debug/" 2>/dev/null || true
  # ros bag capture: ~/verify_run_${J}_captures/<scene>/<qdir>/bag  ->  $OUT/captures/...
  rsync -az "xlogin:verify_run_${J}_captures/" "$OUT/captures/" 2>/dev/null || true
  echo "== harvested to $OUT"
  sed -n '/== monitor rc/,$p' "$OUT/verify_run.out" 2>/dev/null | head -60
fi

# ---- score the captured bag(s), if any (host tool: needs the repo venv + GT data) ----
# Single-job layout:  captures/<scene>/<qdir>/bag        (2 levels above bag)
# Batch-job layout:   captures/<slot>/<scene>/<qdir>/bag (3 levels above bag,
#   the extra <slot> level lets the same (scene, qdir) repeat across a batch
#   without collision; score_live_run only inspects the immediate two parent
#   dir names — <qdir> then <scene> — so the extra outer level is transparent
#   to it as long as we hand it the exact <scene>/<qdir> run dir, which both
#   globs below do).
shopt -s nullglob
run_dirs=()
for bag in "$OUT"/captures/*/*/bag "$OUT"/captures/*/*/*/bag; do
  [ -d "$bag" ] || continue
  rd="$(dirname "$bag")"
  # #127: cluster_verify_batch.sbatch drops a DEGRADED sentinel next to the bag
  # when the sim endpoint (:10000) never came up for that slot — such a run
  # must never enter the scored aggregate even though a bag exists.
  if [ -f "$rd/DEGRADED" ]; then
    echo "== skipping $rd — marked DEGRADED (sim endpoint never came up, #127)"
    continue
  fi
  run_dirs+=("$rd")
done
if [ ${#run_dirs[@]} -eq 0 ]; then
  echo "== no ros bag captured (older run, or recorder failed — see the *_bag*.log)"
  exit 0
fi
# #149: a bare `python` resolves to whatever's on PATH — outside the repo
# venv that's /usr/bin/python, which imports tools.score_live_run fine (so
# `--help` looks healthy) but dies at runtime on the first bag read with
# `ModuleNotFoundError: No module named 'rosbags'`. The old `|| echo` then
# swallowed that per-run, so the loop ran to completion and the script
# exited 0 having written zero scores — a 30-question sweep harvested
# "successfully" with nothing to show for it (#149). Resolve an explicit
# interpreter instead of trusting PATH: honour PYTHON= if the caller set
# it, else the repo venv, else fall back to PATH `python` (and let the
# loud pre-flight check below catch the case where none of that works).
SCORE_PYTHON=${PYTHON:-}
if [ -z "$SCORE_PYTHON" ] && [ -x "$REPO/.venv/bin/python" ]; then
  SCORE_PYTHON="$REPO/.venv/bin/python"
fi
SCORE_PYTHON=${SCORE_PYTHON:-python}

if ! command -v "$SCORE_PYTHON" >/dev/null 2>&1 && [ ! -x "$SCORE_PYTHON" ]; then
  echo "== FATAL: scoring cannot run at all — interpreter '$SCORE_PYTHON' not found."
  echo "   (is the repo venv built at $REPO/.venv? or set PYTHON=<interpreter>.)"
  echo "== ${#run_dirs[@]} captured run(s) were harvested but NONE could be scored."
  exit 1
fi
if ! "$SCORE_PYTHON" -c "import rosbags" >/dev/null 2>&1; then
  echo "== FATAL: scoring cannot run at all — '$SCORE_PYTHON' has no 'rosbags' module."
  echo "   (is the repo venv built at $REPO/.venv? activate it, or set PYTHON=<interpreter>.)"
  echo "== ${#run_dirs[@]} captured run(s) were harvested but NONE could be scored."
  exit 1
fi

echo "== scoring ${#run_dirs[@]} captured run(s) with tools/score_live_run.py ($SCORE_PYTHON)"
score_failures=0
for rd in "${run_dirs[@]}"; do
  # each run dir is captures/<scene>/<qdir>; scores.md/json land under captures/
  if ! "$SCORE_PYTHON" -m tools.score_live_run "$rd" --out "$OUT/captures"; then
    echo "   score_live_run failed for $rd (are GT scenes present? see the error above)"
    score_failures=$((score_failures + 1))
  fi
done
if [ "$score_failures" -gt 0 ]; then
  echo "== WARNING: $score_failures/${#run_dirs[@]} run(s) failed to score — scores.md is incomplete"
fi
[ -f "$OUT/captures/scores.md" ] && { echo "== scores"; cat "$OUT/captures/scores.md"; }
if [ "$score_failures" -gt 0 ]; then
  exit 1
fi
