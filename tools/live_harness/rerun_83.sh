#!/usr/bin/env bash
# #83 Phase-2: paired same-session captures (host-node, instrumented), wedge-aware.
set -u
H=/home/jason/cmu_ws/cmu-vla-2026/tools/live_harness
BASE=/home/jason/cmu_ws/cmu-vla-2026
declare -A QUESTIONS=(
  [livingroom_1]="Go to the potted plant closest to the pyramid candle holder and stop at the vase between the TV and the door."
  [office_1]="Go to the potted plant furthest from the projector screen then stop at the water cooler near the window."
)
for SCENE in livingroom_1 office_1; do
  R=$BASE/reports/issue83_live_captures/${SCENE}_inst_v3
  mkdir -p "$R"
  bash $H/scene_swap.sh $SCENE || { echo "SWAP_FAILED $SCENE"; exit 3; }
  cd $BASE
  VLA_EXPLORE_DEBUG_DIR=$R VLA_DETECTOR=grounding_dino RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
    bash tools/run_host_node.sh > $R/host_node.log 2>&1 &
  HN=$!
  sleep 45
  docker exec iros2026_system bash -c "export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp; source /opt/ros/*/setup.bash 2>/dev/null; timeout 20 ros2 topic pub --once --qos-durability transient_local --qos-reliability reliable /challenge_question std_msgs/msg/String '{data: \"${QUESTIONS[$SCENE]}\"}'" >> $R/host_node.log 2>&1
  WEDGED=0
  for i in $(seq 1 16); do
    LINE=$(nvidia-smi --query-gpu=pstate,clocks.sm,utilization.gpu --format=csv,noheader)
    echo "$(date +%T) $LINE" >> $R/clocks.log
    case "$LINE" in *210\ MHz*100\ %*) WEDGED=$((WEDGED+1));; *) WEDGED=0;; esac
    if [ $WEDGED -ge 3 ]; then echo "WEDGE_DETECTED_ABORT $SCENE" >> $R/clocks.log; break; fi
    sleep 15
  done
  kill $HN 2>/dev/null; sleep 3; kill -9 $HN 2>/dev/null || true
  echo "CAPTURE_END $SCENE samples=$(wc -l < $R/explore_debug.jsonl 2>/dev/null || echo 0) wedged=$WEDGED"
done
docker stop iros2026_system >/dev/null 2>&1
echo RERUN83_PHASE2_DONE
