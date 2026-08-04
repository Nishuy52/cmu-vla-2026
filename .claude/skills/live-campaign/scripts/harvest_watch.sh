#!/bin/bash
# Harvest watcher for a chained verify-batch campaign (live-campaign skill).
#
# Usage: copy, set the four variables, run with run_in_background: true.
# For each job in order: wait for a terminal Slurm state, record the
# per-job assertions, then harvest. One line per event goes to $LOG —
# point a persistent Monitor at it with a filter like:
#   grep -E "harvested rc=|state=(FAILED|CANCELLED|TIMEOUT)|DEGRADED|src_drift_files=[1-9]"
#
# The assertions written per job:
#   state=...            terminal Slurm state (COMPLETED expected)
#   attempted='...'      the VERIFY-BATCH-DONE line (full count expected)
#   success=N            SLOT SUCCESS count
#   degraded=N           DEGRADED sentinel count
#   src_drift_files=N    content-diff file count vs the pinned tree (0 expected;
#                        nonzero means the measured code changed mid-campaign)

JOBS="CHANGE_ME e.g. 712650 712651 712652"
PIN="CHANGE_ME path to the detached worktree pinned at the measured commit"
REPO="/home/jason/cmu_ws/cmu-vla-2026"
LOG="CHANGE_ME absolute log path"

source "$REPO/.venv/bin/activate"
cd "$REPO"
for J in $JOBS; do
  echo "[$(date +%H:%M)] wait $J" >> "$LOG"
  while :; do
    st=$(timeout 60 ssh xlogin "sacct -j $J -X -n -o State 2>/dev/null" 2>/dev/null | head -1 | tr -d ' ')
    [ -n "$st" ] && [ "$st" != "RUNNING" ] && [ "$st" != "PENDING" ] && break
    sleep 300
  done
  att=$(timeout 60 ssh xlogin "grep -o 'VERIFY-BATCH-DONE.*' ~/verify_batch_$J.out" 2>/dev/null)
  ok=$(timeout 60 ssh xlogin "grep -c '^SLOT .*SUCCESS' ~/verify_batch_$J.out" 2>/dev/null)
  dg=$(timeout 60 ssh xlogin "grep -c DEGRADED ~/verify_batch_$J.out" 2>/dev/null)
  drift=$(timeout 120 rsync -rcn --exclude=__pycache__ --exclude=.pytest_cache \
    "$PIN/src/" xlogin:vla/src/ 2>/dev/null | grep -vE "^$|/$" | wc -l)
  echo "[$(date +%H:%M)] $J state=$st attempted='$att' success=$ok degraded=$dg src_drift_files=$drift" >> "$LOG"
  timeout 2400 "$REPO/tools/cluster/live_run/harvest_verify.sh" "$J" >> "$LOG" 2>&1
  echo "[$(date +%H:%M)] $J harvested rc=$?" >> "$LOG"
done
echo "[$(date +%H:%M)] ALL JOBS HARVESTED" >> "$LOG"
