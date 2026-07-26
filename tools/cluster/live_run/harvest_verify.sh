#!/bin/bash
# Harvest a cluster verify run into reports/cluster_verify/<job>/ and print the
# verdict block. Usage: tools/cluster/live_run/harvest_verify.sh <jobid>
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../../.." && pwd)
J=${1:?jobid}
OUT=$REPO/reports/cluster_verify/$J
mkdir -p "$OUT"
scp -q "xlogin:verify_run.out" "$OUT/" || true
scp -q "xlogin:verify_run_${J}_adapter.log" "xlogin:verify_run_${J}_ollama.log" \
      "xlogin:verify_run_${J}_launch.log" "$OUT/" 2>/dev/null || true
rsync -az "xlogin:verify_run_${J}_debug/" "$OUT/debug/" 2>/dev/null || true
echo "== harvested to $OUT"
sed -n '/== monitor rc/,$p' "$OUT/verify_run.out" 2>/dev/null | head -60
