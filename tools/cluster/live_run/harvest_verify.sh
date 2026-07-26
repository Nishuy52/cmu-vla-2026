#!/bin/bash
# Harvest a cluster verify run into reports/cluster_verify/<job>/, print the
# verdict block, and (if a ros bag was captured) score it with the SAME scorers
# as the offline gt_battery. Usage: tools/cluster/live_run/harvest_verify.sh <jobid>
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../../.." && pwd)
J=${1:?jobid}
OUT=$REPO/reports/cluster_verify/$J
mkdir -p "$OUT"
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

# ---- score the captured bag(s), if any (host tool: needs the repo venv + GT data) ----
shopt -s nullglob
run_dirs=()
for bag in "$OUT"/captures/*/*/bag; do
  [ -d "$bag" ] && run_dirs+=("$(dirname "$bag")")
done
if [ ${#run_dirs[@]} -eq 0 ]; then
  echo "== no ros bag captured (older run, or recorder failed — see verify_run_${J}_bag.log)"
  exit 0
fi
echo "== scoring ${#run_dirs[@]} captured run(s) with tools/score_live_run.py"
for rd in "${run_dirs[@]}"; do
  # each run dir is captures/<scene>/<qdir>; scores.md/json land under captures/
  python -m tools.score_live_run "$rd" --out "$OUT/captures" \
    || echo "   score_live_run failed for $rd (is the host venv active? are GT scenes present?)"
done
[ -f "$OUT/captures/scores.md" ] && { echo "== scores"; cat "$OUT/captures/scores.md"; }
