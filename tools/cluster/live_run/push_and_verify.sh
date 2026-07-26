#!/bin/bash
# One-command cluster verification leg (issues #83/#84/#88/#82/#80).
# Prereq: a live ssh ControlMaster window to the jump host (`ssh xlogin true`
# once, password typed interactively). Then:
#
#   tools/cluster/live_run/push_and_verify.sh [scene] ["question"] [qdir]
#
# scene default: livingroom_1 (the #83 failing scene; uploaded from
# data/unity_scenes_ros2/<scene> if not already on the cluster).
# question default: the livingroom_1 inst GT question (the live-baseline one).
# qdir default: inst — question type for the bag-capture layout the scorer
#   expects (nume|obje|inst -> numerical|object_reference|instruction_following).
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../../.." && pwd)
SCENE=${1:-livingroom_1}
QUESTION=${2:-"Go to the potted plant closest to the pyramid candle holder and stop at the vase between the TV and the door."}
QDIR=${3:-inst}
KIT=$REPO/tools/cluster/live_run

echo "== sync ai_module src -> cluster ~/vla/src"
rsync -az --delete "$REPO/src/" xlogin:vla/src/
echo "== sync kit scripts"
scp -q "$KIT"/cluster_verify_run.sbatch "$KIT"/question_pub.py "$KIT"/live_monitor.py "$KIT"/topic_probe.py xlogin:
echo "== ensure scene on cluster"
if ! ssh xlogin "test -d ~/scenes/$SCENE"; then
  echo "   uploading data/unity_scenes_ros2/$SCENE (~300MB)"
  rsync -az "$REPO/data/unity_scenes_ros2/$SCENE/" "xlogin:scenes/$SCENE/"
fi
echo "== submit"
JOB=$(ssh xlogin "cd ~ && QUESTION=\"$QUESTION\" SCENE=\"$SCENE\" QDIR=\"$QDIR\" SCENE_DIR=~/scenes/$SCENE sbatch --export=ALL ~/cluster_verify_run.sbatch" | grep -oE '[0-9]+')
echo "SUBMITTED job $JOB (scene=$SCENE qdir=$QDIR)"
echo "   watch:   ssh xlogin tail -f '~/verify_run.out'"
echo "   harvest: tools/cluster/live_run/harvest_verify.sh $JOB"
