#!/bin/bash
#SBATCH --job-name=verify-run
#SBATCH -p gpu
#SBATCH --gpus=nv:1
#SBATCH -x xgpe0,xgpe2,xgpe6
#SBATCH -t 30
#SBATCH -c 16
#SBATCH --mem=48G
#SBATCH -o /home/c/cyuhsin/verify_run_%j.out

# Instrumented verification run (issues #83/#84/#88/#82/#80): the live-run kit
# plus (a) explore-debug + instance-index dumps, (b) the local LLM tier served
# by ollama with its serve log captured per-run, (c) summary greps that decide
# each issue's verdict straight from the .out. Parameters (export before
# sbatch, or edit here):
#   QUESTION    challenge question (default: livingroom_1 inst GT question)
#   SCENE_DIR   optional path to a downloaded scene dir whose unity environment
#               build should replace the baked one (restart-based swap; the
#               baked scene is used when unset)
#   SCENE       scene NAME for the capture layout (default: basename of
#               SCENE_DIR, else "baked"). Only labels the bag output path.
#   QDIR        question-type dir for the capture layout — nume|obje|inst
#               (default: inst). Matches tools/score_live_run.py QDIR_TO_QTYPE.
#
# Records a ros bag of the driven trajectory + published answers into
# ~/verify_run_${J}_captures/$SCENE/$QDIR/bag so tools/score_live_run.py can
# produce a real IF-rubric / IoU / numerical-match score, not just the
# diagnostic verdict greps below (harvest_verify.sh pulls it back).
set -u
ROOT=~/.udocker/containers/64af1b4a-7bad-3be3-984b-bdccca76ee40/ROOT
WS=$ROOT/home/docker/autonomy_stack_mecanum_wheel_platform
XR=~/xstack/root
J=$SLURM_JOB_ID
QUESTION=${QUESTION:-"Go to the potted plant closest to the pyramid candle holder and stop at the vase between the TV and the door."}
SCENE_DIR=${SCENE_DIR:-}
SCENE=${SCENE:-$(basename "${SCENE_DIR:-baked}")}
QDIR=${QDIR:-inst}

# ---------- proven v4 base env ----------
export PATH=$ROOT/opt/ros/jazzy/bin:$PATH
export PYTHONPATH=$ROOT/opt/ros/jazzy/lib/python3.12/site-packages:$ROOT/usr/lib/python3/dist-packages:$ROOT/usr/lib/python3.12/dist-packages
export LD_LIBRARY_PATH=$ROOT/opt/ros/jazzy/lib:$ROOT/opt/ros/jazzy/lib/x86_64-linux-gnu:$ROOT/usr/lib/x86_64-linux-gnu:$ROOT/lib/x86_64-linux-gnu:$ROOT/usr/lib
export AMENT_PREFIX_PATH=$ROOT/opt/ros/jazzy
for d in "$WS"/install/*/; do
  pkg=$(basename "$d"); [ "$pkg" = "COLCON_IGNORE" ] && continue; [ -d "$d" ] || continue
  export AMENT_PREFIX_PATH="$d:${AMENT_PREFIX_PATH}"
  [ -d "$d/lib" ] && export LD_LIBRARY_PATH="$d/lib:$LD_LIBRARY_PATH"
  [ -d "$d/lib/python3.12/site-packages" ] && export PYTHONPATH="$d/lib/python3.12/site-packages:$PYTHONPATH"
done
for link in "$WS"/install/*/lib/python3.12/site-packages/*.egg-link; do
  [ -f "$link" ] || continue
  tgt=$(head -1 "$link"); [ -d "$ROOT$tgt" ] && export PYTHONPATH="$ROOT$tgt:$PYTHONPATH"
done
export ROS_DISTRO=jazzy RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_LOG_DIR=/tmp/roslog-$J ROS_HOME=/tmp/roshome-$J
mkdir -p "$ROS_LOG_DIR" "$ROS_HOME"
export CYCLONEDDS_URI='<CycloneDDS><Domain><General><Interfaces><NetworkInterface address="127.0.0.1" multicast="true"/></Interfaces></General></Domain></CycloneDDS>'

echo "== verify-run $J on $(hostname) $(date +%T)  question: $QUESTION"

# ---------- optional scene swap (restart-based: swap BEFORE Unity starts) ----------
# Mirrors the proven laptop flow (tools/live_harness/scene_swap.sh): overlay-copy the
# scene's whole tree (environment/ + map files) over the baked unity dir — no delete,
# exactly like the docker-cp overlay.
UNITY_DIR=$WS/src/base_autonomy/vehicle_simulator/mesh/unity
MODEL=$UNITY_DIR/environment/Model.x86_64
if [ -n "$SCENE_DIR" ]; then
  SWAP_MODEL=$(find "$SCENE_DIR" -maxdepth 3 -name 'Model.x86_64' | head -1)
  if [ -n "$SWAP_MODEL" ]; then
    SCENE_ROOT=$(dirname "$(dirname "$SWAP_MODEL")")
    rsync -a "$SCENE_ROOT/" "$UNITY_DIR/"
    echo "SCENE: overlaid $SCENE_ROOT -> $UNITY_DIR"
  else
    echo "SCENE: no Model.x86_64 under $SCENE_DIR — using baked scene"
  fi
else
  echo "SCENE: baked"
fi

# ---------- X + Unity ----------
mkdir -p /tmp/.X11-unix 2>/dev/null
$XR/usr/bin/Xvfb :99 -screen 0 1280x720x24 -xkbdir $XR/usr/share/X11/xkb -noreset 2>/dev/null &
XVFB_PID=$!; sleep 3
export DISPLAY=:99
chmod +x "$MODEL" 2>/dev/null
$XR/opt/VirtualGL/bin/vglrun -ld $XR/usr/lib -d egl0 "$MODEL" -force-glcore -screen-width 1280 -screen-height 720 > ~/verify_run_${J}_unity.log 2>&1 &
UNITY_PID=$!; sleep 8
kill -0 $UNITY_PID 2>/dev/null && echo "UNITY: alive" || { echo "UNITY: DEAD"; }

# ---------- autonomy stack ----------
timeout 1380 ros2 launch vehicle_simulator system_simulation.launch > ~/verify_run_${J}_launch.log 2>&1 &
LAUNCH_PID=$!
echo "stack launched; settling 45s"; sleep 45
(ss -ltn 2>/dev/null || true) | grep -q ":10000" && echo "ENDPOINT: :10000 up" || echo "ENDPOINT: MISSING"

# ---------- ollama (local LLM tier, issue #82 retest) ----------
# shared node: 11434 may already be bound (697648 lesson) — probe upward
OPORT=11434
while (ss -ltn 2>/dev/null || true) | grep -q ":$OPORT "; do OPORT=$((OPORT+1)); done
export OLLAMA_HOST=127.0.0.1:$OPORT OLLAMA_MODELS=~/ollama-models
~/ollama/bin/ollama serve > ~/verify_run_${J}_ollama.log 2>&1 &
OLLAMA_PID=$!
for i in $(seq 1 30); do
  (ss -ltn 2>/dev/null || true) | grep -q ":$OPORT " && break; sleep 1
done
(ss -ltn 2>/dev/null || true) | grep -q ":$OPORT " && echo "OLLAMA: up on $OPORT" || echo "OLLAMA: MISSING"

# ---------- ai_module adapter (venv torch/GDINO + ROOT rclpy) ----------
source ~/venv-gpu/bin/activate
export PYTHONPATH=~/vla/src:$PYTHONPATH
export VLA_DETECTOR=grounding_dino GDINO_DEVICE=cuda GDINO_PRECISION=fp16
# cap torch/OpenMP CPU threads — with all cores claimed, Unity's frame encode starves
# and the camera stream collapses (v7/v8 evidence: 2Hz -> 0.1Hz once GDINO loads)
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export GDINO_CONFIG_PATH=~/models/gdino/GroundingDINO_SwinB_cfg.py
export GDINO_CHECKPOINT_PATH=~/models/gdino/groundingdino_swinb_cogcoor.pth
export VLA_CAMERA_COMPRESSED=1
# issues #83/#84/#89: per-run explore-debug + instance-index dumps
export VLA_EXPLORE_DEBUG_DIR=~/verify_run_${J}_debug
mkdir -p "$VLA_EXPLORE_DEBUG_DIR"
export VLA_INSTANCE_DUMP_PATH=$VLA_EXPLORE_DEBUG_DIR/instance_index.jsonl
# issue #84: raw pre-gate detection dump (split "never proposed" vs "gated out")
export VLA_RAW_DETECTION_DUMP_PATH=$VLA_EXPLORE_DEBUG_DIR/raw_detections.jsonl
# SoCLaaS primary tier is now core/llm/config.py's DEFAULT primary (permanent, not a
# cluster-only override) — the key just needs to be present in the environment; the
# ladder fills base_url/model/api_key_env itself. Key issued via soclaas-portal, lives
# only in ~/.bashrc — extracted here, never in the repo. Ladder degrades
# primary -> local -> regex if the gateway is unreachable or the key is absent.
eval "$(grep -h 'export SOCLAAS_API_KEY' ~/.bashrc | head -1)" || true
# per-call LLM usage (tokens/latency/tier/ok/error) -> job-specific, harvestable log.
# This is what tells us WHETHER SoCLaaS actually served the parse or the api tier fell
# through to local/regex (and why) — the 697761 fall-through had no such record.
export VLA_LLM_USAGE_LOG=~/verify_run_${J}_usage.jsonl
if [ -n "${SOCLAAS_API_KEY:-}" ]; then
  echo "SOCLAAS: primary tier enabled by default (qwen3.6:35b); usage -> $VLA_LLM_USAGE_LOG"
  # fail-fast reachability probe from THIS compute node (697761 fell to regex silently)
  code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 8 \
    https://soclaas-api.comp.nus.edu.sg/v1/models -H "Authorization: Bearer $SOCLAAS_API_KEY" 2>/dev/null)
  echo "SOCLAAS: compute-node gateway probe HTTP $code (200 = reachable from this node)"
else
  echo "SOCLAAS: no key in ~/.bashrc — primary tier off, ladder falls to local/regex"
fi
# issue #82: local tier via ollama (same 3b model as the laptop plan for parity)
export VLA_LLM_LOCAL_KIND=openai
export VLA_LLM_LOCAL_BASE_URL=http://127.0.0.1:$OPORT/v1
export VLA_LLM_LOCAL_MODEL=qwen2.5vl:3b
export VLA_LOCAL_API_KEY=local-dummy
python -c "from ros_adapter.adapter_node import main; main()" > ~/verify_run_${J}_adapter.log 2>&1 &
ADAPTER_PID=$!
sleep 25
kill -0 $ADAPTER_PID 2>/dev/null && echo "ADAPTER: alive" || { echo "ADAPTER: DEAD — log tail:"; tail -15 ~/verify_run_${J}_adapter.log; }
grep -m1 -iE "perception|detector|boot" ~/verify_run_${J}_adapter.log || true
grep -m1 -iE "ladder configured" ~/verify_run_${J}_adapter.log || true

# ---------- ros bag capture (tools/score_live_run.py input) ----------
# Record the driven trajectory (/state_estimation) + the question + both
# published-answer topics into <scene>/<qdir>/bag/ so the run can be scored
# offline with the SAME scorers as the gt_battery. Started BEFORE the question
# is published (settle so the recorder is subscribed first) and stopped after
# the monitor returns; irrelevant answer topics for this qtype simply record 0
# messages, which is exactly what score_live_run's completeness check expects.
BAGDIR=~/verify_run_${J}_captures/$SCENE/$QDIR
mkdir -p "$BAGDIR"; rm -rf "$BAGDIR/bag"  # ros2 bag record refuses an existing dir
# setsid so BAG_PID is a process-group leader — lets us signal the recorder AND its
# rosbag2 child as a group at stop time (a bare SIGINT to the `ros2` wrapper does not
# reliably reach the child, which is why `wait` hung the whole job to the SLURM wall).
setsid ros2 bag record -o "$BAGDIR/bag" \
  /state_estimation /challenge_question /numerical_response /selected_object_marker \
  > ~/verify_run_${J}_bag.log 2>&1 &
BAG_PID=$!
sleep 3
kill -0 $BAG_PID 2>/dev/null && echo "BAG: recording -> $BAGDIR/bag" || { echo "BAG: recorder DEAD — log tail:"; tail -10 ~/verify_run_${J}_bag.log; }

# ---------- question + monitor ----------
python3 ~/question_pub.py "$QUESTION" > /dev/null 2>&1 &
QPUB_PID=$!
echo "question publishing; monitor up (deadline 780s)"
python3 ~/live_monitor.py 780 2>&1
MONITOR_RC=$?

# Stop the recorder — SIGINT the whole process group so rosbag2 finalizes the mcap,
# but BOUNDED so it can never hang the job (the old `wait $BAG_PID` blocked forever
# when the child didn't finalize -> 25min hang -> SLURM wall -> truncated bag).
kill -INT -"$BAG_PID" 2>/dev/null || kill -INT "$BAG_PID" 2>/dev/null
for i in $(seq 1 45); do kill -0 "$BAG_PID" 2>/dev/null || break; sleep 1; done
kill -KILL -"$BAG_PID" 2>/dev/null; kill -KILL "$BAG_PID" 2>/dev/null
echo "== bag capture"; timeout 30 ros2 bag info "$BAGDIR/bag" 2>&1 | grep -E "Messages:|Topic:|Duration:" || echo "BAG: no clean info (harvest reindex+convert will recover)"

# ---------- per-issue verdict summary ----------
# live_monitor only latches numerical/marker answers; an IF answer is a waypoint.
# The FSM's own answer transition in the adapter log is the authoritative signal.
grep -q "transition:answer->" ~/verify_run_${J}_adapter.log && MONITOR_RC=0
echo "== monitor rc=$MONITOR_RC"
echo "== adapter log tail"; tail -25 ~/verify_run_${J}_adapter.log
echo "== #82 parse tier"; grep -E "parsed: tier=|ladder" ~/verify_run_${J}_adapter.log | tail -5
echo "== #82 ollama serve tail"; tail -5 ~/verify_run_${J}_ollama.log
echo "== #82/SoCLaaS LLM usage log (per-call tokens/tier/ok/error)"; tail -12 "$VLA_LLM_USAGE_LOG" 2>/dev/null || echo "no usage log (no LLM call logged — tier fell through before any api/local attempt)"
echo "== #84 raw detections (proposed-vs-gated by class, last keyframe)"; tail -1 "$VLA_RAW_DETECTION_DUMP_PATH" 2>/dev/null | python3 -c "import json,sys; d=json.loads(sys.stdin.readline() or '{}'); print(json.dumps(d.get('by_class',{})))" 2>/dev/null || echo "no raw-detection dump"
echo "== #83 explore debug (last 3 records: pocket/free/reroot/status)"
for f in "$VLA_EXPLORE_DEBUG_DIR"/explore_debug_*.jsonl; do
  [ -f "$f" ] || continue
  tail -3 "$f" | python3 -c "
import json,sys
for line in sys.stdin:
    d=json.loads(line)
    c=d.get('costmap',{})
    li=d.get('live_instances',{})
    print('t=%.0f status=%s pocket=%s free=%s reroot=%s n_frontiers=%d instances=%s keyframes=%s' % (
        d.get('question_clock_t',0), d.get('status'), c.get('reachable_pocket_cells'),
        c.get('free'), d.get('bfs_reroot','n/a'), len(d.get('frontier_candidates',[])),
        li.get('total_instances'), d.get('keyframes_processed')))
"
done
echo "== #84 instance classes (last record)"
for f in "$VLA_EXPLORE_DEBUG_DIR"/explore_debug_*.jsonl; do
  [ -f "$f" ] || continue
  tail -1 "$f" | python3 -c "
import json,sys
d=json.loads(sys.stdin.readline())
print(json.dumps(d.get('live_instances',{}).get('by_class',{})))
"
done
echo "== #88 waypoint cadence + #80 pre-60s waypoints (analyze WAYPOINT lines above)"
grep -cE "^WAYPOINT" ~/verify_run_${J}.out 2>/dev/null || true
echo "== grounded or floor?"; grep -E "answer_from_floor|answer_from_head|published int answer|instances" ~/verify_run_${J}_adapter.log | tail -5
echo "== gpu"; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
kill $QPUB_PID $ADAPTER_PID $OLLAMA_PID $LAUNCH_PID $UNITY_PID $XVFB_PID ${BAG_PID:-} 2>/dev/null
sleep 2
[ $MONITOR_RC -eq 0 ] && echo "VERIFY-RUN-DONE: SUCCESS" || echo "VERIFY-RUN-DONE: rc=$MONITOR_RC"
