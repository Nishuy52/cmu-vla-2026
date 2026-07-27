#!/bin/bash
# One-command BATCH cluster verification leg: submits ONE Slurm allocation
# that answers every (scene, qtype, question) tuple for the requested scenes
# sequentially (tools/cluster/live_run/cluster_verify_batch.sbatch), so the
# ~12h nv-queue wait is paid once instead of once per question. Use
# push_and_verify.sh (singular) when you only need one question.
#
# Prereq: a live ssh ControlMaster window to the jump host (`ssh xlogin true`
# once, password typed interactively). Then:
#
#   tools/cluster/live_run/push_and_verify_batch.sh [scene1,scene2,...]
#
# scenes default: livingroom_1,office_1. Every numerical / object_reference /
# instruction_following question in the repo's questions.json for each
# requested scene becomes one matrix line (scene_dir = ~/scenes/<scene> — the
# uploaded scene, matching what push_and_verify.sh does for a single scene;
# never the baked scene, so the batch always answers the scene it claims to).
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../../.." && pwd)
SCENES_CSV=${1:-livingroom_1,office_1}
KIT=$REPO/tools/cluster/live_run
MATRIX=$(mktemp)
trap 'rm -f "$MATRIX"' EXIT

# QTYPES=inst[,nume,obje] restricts which question types enter the matrix, so a
# single allocation can sweep ONE type across MANY scenes (max scene diversity per
# ~175min job) instead of all types across few scenes. Default: all three.
export QTYPES=${QTYPES:-inst,nume,obje}
# QMAX=<n> caps questions per (scene, qtype) — QMAX=1 takes just the first, which
# is what a broad cross-scene sweep wants.
export QMAX=${QMAX:-99}

echo "== build matrix locally from questions.json (qtypes=$QTYPES qmax=$QMAX)"
IFS=',' read -ra SCENES <<< "$SCENES_CSV"
python3 - "$MATRIX" "${SCENES[@]}" <<'PYEOF'
import json
import sys

out_path = sys.argv[1]
wanted = sys.argv[2:]

sys.path.insert(0, ".")
import tools  # noqa: F401 -- inserts <repo>/src onto sys.path
from core.runner import gt_battery as GB

QTYPE_TO_QDIR = {
    "numerical": "nume",
    "object_reference": "obje",
    "instruction_following": "inst",
}

with open(GB.DEFAULT_QUESTIONS, encoding="utf-8") as fh:
    data = json.load(fh)
by_scene = {e["scene"]: e["questions"] for e in data}

missing = [s for s in wanted if s not in by_scene]
if missing:
    sys.exit(f"unknown scene(s) in questions.json: {missing} (have: {sorted(by_scene)})")

import os
keep_qdirs = {t.strip() for t in os.environ.get("QTYPES", "inst,nume,obje").split(",") if t.strip()}
qmax = int(os.environ.get("QMAX", "99"))

lines = []
for scene in wanted:
    questions = by_scene[scene]
    scene_dir = f"~/scenes/{scene}"
    for qtype, qdir in QTYPE_TO_QDIR.items():
        if qdir not in keep_qdirs:
            continue
        for q in questions.get(qtype, [])[:qmax]:
            q = " ".join(q.split())  # collapse newlines/whitespace -> one line
            lines.append(f"{scene}|{qdir}|{scene_dir}|{q}")

with open(out_path, "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines) + "\n")

print(f"{len(lines)} question(s) across {len(wanted)} scene(s)")
PYEOF
echo "== matrix ($(wc -l < "$MATRIX") lines):"
cat "$MATRIX"

echo "== sync ai_module src -> cluster ~/vla/src"
rsync -az --delete "$REPO/src/" xlogin:vla/src/

echo "== ensure scenes on cluster"
for SCENE in "${SCENES[@]}"; do
  if ! ssh xlogin "test -d ~/scenes/$SCENE"; then
    echo "   uploading data/unity_scenes_ros2/$SCENE (~300MB)"
    rsync -az "$REPO/data/unity_scenes_ros2/$SCENE/" "xlogin:scenes/$SCENE/"
  fi
done

echo "== sync kit scripts + matrix"
scp -q "$KIT"/cluster_verify_batch.sbatch "$KIT"/question_pub.py "$KIT"/live_monitor.py \
       "$KIT"/topic_probe.py xlogin:
scp -q "$MATRIX" xlogin:verify_batch_matrix.txt

echo "== submit"
# Optional: chain behind an already-queued job, e.g. AFTER=698999 (or a full
# spec like afterany:698999). Verify runs must never overlap — they drive the
# SAME udocker container/scene overlay on NFS home — so when another run is
# already queued, chain rather than submit a competing job. The dependency
# queues immediately (accruing age/priority in a congested queue) and only
# starts once the predecessor terminates, for any exit status.
DEP_ARG=""
if [ -n "${AFTER:-}" ]; then
  case "$AFTER" in *:*) DEP_SPEC="$AFTER" ;; *) DEP_SPEC="afterany:$AFTER" ;; esac
  DEP_ARG="--dependency=$DEP_SPEC"
  echo "   chaining: $DEP_ARG"
fi
JOB=$(ssh xlogin "cd ~ && sbatch $DEP_ARG --export=ALL ~/cluster_verify_batch.sbatch" | grep -oE '[0-9]+')
echo "SUBMITTED batch job $JOB (scenes=$SCENES_CSV, $(wc -l < "$MATRIX") question(s))"
echo "   watch:   ssh xlogin tail -f '~/verify_batch_${JOB}.out'"
echo "   harvest: tools/cluster/live_run/harvest_verify.sh $JOB"
