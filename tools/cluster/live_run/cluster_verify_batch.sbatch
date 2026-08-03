#!/bin/bash
#SBATCH --job-name=verify-batch
#SBATCH -p gpu
#SBATCH --gpus=nv:1
#SBATCH -x xgpe0,xgpe2,xgpe6
#SBATCH -t 175
#SBATCH -c 16
#SBATCH --mem=48G
#SBATCH -o /home/c/cyuhsin/verify_batch_%j.out

# Batch verification run: ONE Slurm allocation answers MULTIPLE (scene, qdir,
# question) tuples sequentially, so the ~12h nv-queue wait is amortized across
# a whole matrix instead of paid once per question (tools/cluster/live_run/
# cluster_verify_run.sbatch is the single-question flow this refactors; keep
# that file untouched — it stays the reference for a lone-question rerun).
#
# Matrix input: ~/verify_batch_matrix.txt, one line per question:
#   scene|qdir|scene_dir|question
# scene_dir may be empty (baked scene). qdir is nume|obje|inst, matching
# tools/score_live_run.py's QDIR_TO_QTYPE. Built locally by
# push_and_verify_batch.sh and scp'd alongside this script.
#
# Each matrix line gets a FULL relaunch (Xvfb/Unity/stack/ollama/adapter) —
# no attempt to keep Unity warm across questions (unproven optimization, not
# worth the risk of cross-question state leakage). Between questions every
# process from the previous question is killed and DISPLAY :99 / port :10000
# / the ollama port are confirmed free (bounded wait-loop) before the next
# relaunch, so each question starts from the same clean-slate state the
# single-question sbatch always started from.
#
# A single bad question (dead Unity, adapter crash, monitor timeout) must
# never abort the whole batch: run_one's body is invoked inside a guarded
# call in the main loop, and a RETURN trap inside run_one guarantees its
# teardown fires no matter how the function exits (normal end, `return` on
# an early error, ...). The loop always advances to the next matrix line.
#
# Per-question output layout (all keyed by a unique slot id so concurrent-run
# artifacts never clobber across questions in the SAME job):
#   SLOT=<i>_<scene>_<qdir>                       (i = 0-based matrix line index)
#   ~/verify_batch_${J}_captures/$SLOT/$SCN/$QD/bag   <- ros2 bag (scorer input;
#     NOTE the extra $SLOT level vs the single script's
#     captures/$SCENE/$QDIR/bag — that's what lets the SAME (scene, qdir) repeat
#     across matrix lines without collision, while still handing the scorer an
#     exact captures/<scene>/<qdir>/bag leaf for GT lookup + QDIR_TO_QTYPE.)
#   ~/verify_batch_${J}_debug_${SLOT}/             <- explore-debug + instance dumps
#   ~/verify_batch_${J}_usage_${SLOT}.jsonl         <- per-call LLM usage log
#   ~/verify_batch_${J}_{unity,launch,ollama,adapter,bag}_${SLOT}.log
set -u
ROOT=~/.udocker/containers/64af1b4a-7bad-3be3-984b-bdccca76ee40/ROOT
WS=$ROOT/home/docker/autonomy_stack_mecanum_wheel_platform
XR=~/xstack/root
J=$SLURM_JOB_ID
WALL_MIN=${WALL_MIN:-175}   # keep in sync with #SBATCH -t (wall-guard budget)
MATRIX=${MATRIX:-~/verify_batch_matrix.txt}
UNITY_DIR=$WS/src/base_autonomy/vehicle_simulator/mesh/unity
MODEL=$UNITY_DIR/environment/Model.x86_64
CAP_ROOT=~/verify_batch_${J}_captures

echo "== verify-batch $J on $(hostname) $(date +%T)  matrix: $MATRIX"
if [ ! -f "$MATRIX" ]; then
  echo "FATAL: matrix file $MATRIX not found"; exit 1
fi

# ---------- proven v4 base env (done ONCE for the whole allocation) ----------
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

# ai_module adapter deps + SoCLaaS key: independent of any single question,
# loaded once here so run_one only sets the per-question knobs.
source ~/venv-gpu/bin/activate
export PYTHONPATH=~/vla/src:$PYTHONPATH
export VLA_DETECTOR=grounding_dino GDINO_DEVICE=cuda GDINO_PRECISION=fp16
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export GDINO_CONFIG_PATH=~/models/gdino/GroundingDINO_SwinB_cfg.py
export GDINO_CHECKPOINT_PATH=~/models/gdino/groundingdino_swinb_cogcoor.pth
export VLA_CAMERA_COMPRESSED=1
eval "$(grep -h 'export SOCLAAS_API_KEY' ~/.bashrc | head -1)" || true
if [ -n "${SOCLAAS_API_KEY:-}" ]; then
  echo "SOCLAAS: primary tier enabled by default (qwen3.6:35b)"
  code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 8 \
    https://soclaas-api.comp.nus.edu.sg/v1/models -H "Authorization: Bearer $SOCLAAS_API_KEY" 2>/dev/null)
  echo "SOCLAAS: compute-node gateway probe HTTP $code (200 = reachable from this node)"
else
  echo "SOCLAAS: no key in ~/.bashrc — primary tier off, ladder falls to local/regex"
fi
export VLA_LLM_LOCAL_KIND=openai
export VLA_LLM_LOCAL_MODEL=qwen2.5vl:3b
export VLA_LOCAL_API_KEY=local-dummy

mkdir -p "$CAP_ROOT"

# ---------- per-question teardown: kill everything, wait for ports to free ----------
teardown_one() {
  local i busy
  kill "${QPUB_PID:-}" "${ADAPTER_PID:-}" "${OLLAMA_PID:-}" "${LAUNCH_PID:-}" \
       "${UNITY_PID:-}" "${XVFB_PID:-}" 2>/dev/null
  kill -KILL -"${BAG_PID:-}" 2>/dev/null; kill -KILL "${BAG_PID:-}" 2>/dev/null
  for i in $(seq 1 30); do
    busy=0
    (ss -ltn 2>/dev/null || true) | grep -q ":10000 " && busy=1
    [ -n "${OPORT:-}" ] && (ss -ltn 2>/dev/null || true) | grep -q ":$OPORT " && busy=1
    [ -f /tmp/.X11-unix/X99 ] && busy=1
    [ "$busy" -eq 0 ] && break
    sleep 1
  done
  # Belt-and-braces: anything still alive by name, one more sweep.
  pkill -9 -f "Model.x86_64" 2>/dev/null || true
  pkill -9 -f "Xvfb :99" 2>/dev/null || true
  pkill -9 -f "ollama serve" 2>/dev/null || true
  pkill -9 -f "ros_adapter.adapter_node" 2>/dev/null || true
  # ros2 launch may fail to reap a stack node (localPlanner/pathFollower/bridge...);
  # a survivor holds :10000 or keeps publishing into the shared DDS domain, poisoning
  # every later question. All stack binaries live under the workspace path.
  pkill -9 -f "autonomy_stack_mecanum_wheel_platform" 2>/dev/null || true
  rm -f /tmp/.X11-unix/X99 2>/dev/null || true
  (ss -ltn 2>/dev/null || true) | grep -q ":10000 " \
    && echo "TEARDOWN: WARNING :10000 STILL BUSY — next question will likely fail" \
    || echo "TEARDOWN: done (ports/display released)"
}

# The recorder never finalizes on SIGINT (issue #114) — it is SIGKILLed above with the
# mcap payload intact but no metadata.yaml, so `rosbags` cannot open it. reindex+convert
# is deterministic (18/18 recovered), so do it HERE and leave a readable bag behind
# instead of making every harvest a manual two-step.
finalize_bag() {
  local bag=$1
  [ -d "$bag" ] || return 0
  if [ -f "$bag/metadata.yaml" ] && timeout 30 ros2 bag info "$bag" >/dev/null 2>&1; then
    echo "BAG: clean, no recovery needed"; return 0
  fi
  echo "BAG: no metadata (recorder did not finalize, #114) — reindex+convert"
  timeout 120 ros2 bag reindex -s mcap "$bag" >/dev/null 2>&1
  local tmp="${bag}.recovered"
  rm -rf "$tmp"
  printf 'output_bags:\n  - uri: %s\n    storage_id: mcap\n    all_topics: true\n' "$tmp" > "/tmp/bagconv_$$.yaml"
  if timeout 300 ros2 bag convert -i "$bag" -o "/tmp/bagconv_$$.yaml" >/dev/null 2>&1 && [ -f "$tmp/metadata.yaml" ]; then
    rm -rf "$bag"; mv "$tmp" "$bag"
    echo "BAG: recovered in-job"
  else
    echo "BAG: RECOVERY FAILED — harvest must recover manually"; rm -rf "$tmp"
  fi
  rm -f "/tmp/bagconv_$$.yaml"
}

# ---------- run one (scene, qdir, question) tuple end to end ----------
run_one() {
  local SCN=$1 QD=$2 SDIR=$3 QUESTION=$4 SLOT=$5
  # SDIR arrives as DATA from the matrix file, so a leading `~` is NOT expanded by
  # bash (only assignment/word expansion does that) — `find "~/scenes/x"` fails and the
  # overlay is silently skipped, leaving the PREVIOUS scene baked in while the run is
  # still labelled/scored as this scene. Expand it explicitly.
  SDIR=${SDIR/#\~/$HOME}
  # `i` MUST be local: this function's ollama/bag wait-loops use `for i in $(seq ...)`,
  # and without this they clobber the CALLER's matrix-line counter — 699009 lost a bag
  # when two questions collided on the same SLOT name and the batch miscounted 10 as 7.
  local i
  trap teardown_one RETURN
  echo "== slot $SLOT: scene=$SCN qdir=$QD scene_dir=${SDIR:-<baked>} question: $QUESTION"

  # ---- optional scene overlay (restart-based: swap BEFORE Unity starts) ----
  if [ -n "$SDIR" ]; then
    local SWAP_MODEL SCENE_ROOT
    SWAP_MODEL=$(find "$SDIR" -maxdepth 3 -name 'Model.x86_64' | head -1)
    if [ -n "$SWAP_MODEL" ]; then
      SCENE_ROOT=$(dirname "$(dirname "$SWAP_MODEL")")
      rsync -a "$SCENE_ROOT/" "$UNITY_DIR/"
      echo "SCENE: overlaid $SCENE_ROOT -> $UNITY_DIR"
    else
      echo "SCENE: no Model.x86_64 under $SDIR — using currently-baked scene"
    fi
  else
    echo "SCENE: baked"
  fi

  # ---- X + Unity ----
  mkdir -p /tmp/.X11-unix 2>/dev/null
  $XR/usr/bin/Xvfb :99 -screen 0 1280x720x24 -xkbdir $XR/usr/share/X11/xkb -noreset 2>/dev/null &
  XVFB_PID=$!; sleep 3
  export DISPLAY=:99
  chmod +x "$MODEL" 2>/dev/null
  $XR/opt/VirtualGL/bin/vglrun -ld $XR/usr/lib -d egl0 "$MODEL" -force-glcore -screen-width 1280 -screen-height 720 \
    > ~/verify_batch_${J}_unity_${SLOT}.log 2>&1 &
  UNITY_PID=$!; sleep 8
  kill -0 $UNITY_PID 2>/dev/null && echo "UNITY: alive" || { echo "UNITY: DEAD"; return 1; }

  # ---- autonomy stack ----
  timeout 900 ros2 launch vehicle_simulator system_simulation.launch > ~/verify_batch_${J}_launch_${SLOT}.log 2>&1 &
  LAUNCH_PID=$!
  echo "stack launched; settling 45s"; sleep 45
  # ENDPOINT_OK gates the final verdict below (#127): a missing :10000 means the
  # autonomy stack never came up, so nothing driven for this question can be
  # trusted even if the adapter still runs its full FSM and "answers" — the slot
  # must land DEGRADED, not SUCCESS, and be kept OUT of the scoring aggregate
  # (see the DEGRADED sentinel dropped next to BAGDIR below, and
  # harvest_verify.sh, which skips any run dir carrying that sentinel).
  local ENDPOINT_OK=1
  (ss -ltn 2>/dev/null || true) | grep -q ":10000" && echo "ENDPOINT: :10000 up" || { echo "ENDPOINT: MISSING"; ENDPOINT_OK=0; }

  # ---- ollama (local LLM tier) ----
  OPORT=11434
  while (ss -ltn 2>/dev/null || true) | grep -q ":$OPORT "; do OPORT=$((OPORT+1)); done
  export OLLAMA_HOST=127.0.0.1:$OPORT OLLAMA_MODELS=~/ollama-models
  ~/ollama/bin/ollama serve > ~/verify_batch_${J}_ollama_${SLOT}.log 2>&1 &
  OLLAMA_PID=$!
  for i in $(seq 1 30); do
    (ss -ltn 2>/dev/null || true) | grep -q ":$OPORT " && break; sleep 1
  done
  (ss -ltn 2>/dev/null || true) | grep -q ":$OPORT " && echo "OLLAMA: up on $OPORT" || echo "OLLAMA: MISSING"
  export VLA_LLM_LOCAL_BASE_URL=http://127.0.0.1:$OPORT/v1

  # ---- warm the local model (issue #82) ----
  # "OLLAMA: up" above only means the TCP port is open — ollama still has to pick a
  # working CUDA backend (this node's TITAN V is cc=700, too old for ollama's bundled
  # cuda_v13 build; the fallback probe to cuda_v12 alone costs 3.5-24s per
  # verify_batch_*_ollama_*.log "skipping CUDA device" -> "inference compute" gaps) and
  # then, on the FIRST inference request, actually read the ~3GB qwen2.5vl:3b GGUF
  # weights off disk into VRAM — ollama does that lazily. Left alone, that whole cold
  # tax lands inside the scored ladder's tight per-call timeout
  # (core.llm.timeout.DEFAULT_CALL_TIMEOUT_S = 20s), which is exactly what blew the
  # original livingroom_1 failure's ~20s parse-to-regex window. Fire one throwaway
  # chat call now, on a generous boot-time-only budget (core.llm.warmup), so the local
  # tier is already warm by the time a real question can reach it. Best-effort: a
  # failed/timed-out warmup is only logged, never fatal — the ladder's regex floor is
  # still the safety net.
  #
  # Only worth paying when the local tier can actually be reached FIRST: with
  # SOCLAAS_API_KEY set the api tier always answers before local ever gets tried (every
  # batch run since 27 Jul shows zero "tier":"local" usage-log records), so warming it
  # would only slow this run's batch throughput for no benefit — skip in that case so
  # the api path is never slowed by this fix.
  if [ -z "${SOCLAAS_API_KEY:-}" ]; then
    timeout 250 python -c "
from core.llm.warmup import warm_up_local
print('WARMUP_OK' if warm_up_local() else 'WARMUP_FAILED')
" > ~/verify_batch_${J}_ollama_warmup_${SLOT}.log 2>&1
    grep -q WARMUP_OK ~/verify_batch_${J}_ollama_warmup_${SLOT}.log \
      && echo "OLLAMA: warmup ok (model loaded)" \
      || echo "OLLAMA: warmup failed — first scored call may still cold-load"
  else
    echo "OLLAMA: warmup skipped (SoCLaaS primary tier active; local tier not on this run's answering path)"
  fi

  # ---- ai_module adapter (per-question dump paths) ----
  export VLA_EXPLORE_DEBUG_DIR=~/verify_batch_${J}_debug_${SLOT}
  mkdir -p "$VLA_EXPLORE_DEBUG_DIR"
  export VLA_INSTANCE_DUMP_PATH=$VLA_EXPLORE_DEBUG_DIR/instance_index.jsonl
  export VLA_RAW_DETECTION_DUMP_PATH=$VLA_EXPLORE_DEBUG_DIR/raw_detections.jsonl
  # issues #102/#98: resolved-Plan and per-leg relaxation dumps. Paths are DERIVED from
  # VLA_EXPLORE_DEBUG_DIR (never a standalone new variable) so they land inside the
  # already-harvested debug dir — see #119, where both emitted zero files across all 45
  # captures because nothing in the repo ever set their env vars.
  # Gates: core/heads/factory.py:280 (VLA_PLAN_DUMP_PATH, also needs state.plan)
  #        core/heads/instruction.py:124 (VLA_LEG_RELAX_DUMP_PATH, IF questions only)
  export VLA_PLAN_DUMP_PATH=$VLA_EXPLORE_DEBUG_DIR/resolved_plan.jsonl
  export VLA_LEG_RELAX_DUMP_PATH=$VLA_EXPLORE_DEBUG_DIR/relaxation_audit.jsonl
  # issue #145: composed GDINO prompt(s) + token-budget-dropped vocab nouns, once per
  # question latch (plus once at boot priming) -- see core/perception/detector.py's
  # dump_prompt_diagnostics. Same #119 discipline: derived from VLA_EXPLORE_DEBUG_DIR.
  export VLA_PROMPT_DUMP_PATH=$VLA_EXPLORE_DEBUG_DIR/prompt_diagnostics.jsonl
  export VLA_LLM_USAGE_LOG=~/verify_batch_${J}_usage_${SLOT}.jsonl
  python -c "from ros_adapter.adapter_node import main; main()" > ~/verify_batch_${J}_adapter_${SLOT}.log 2>&1 &
  ADAPTER_PID=$!
  sleep 25
  kill -0 $ADAPTER_PID 2>/dev/null && echo "ADAPTER: alive" || { echo "ADAPTER: DEAD — log tail:"; tail -15 ~/verify_batch_${J}_adapter_${SLOT}.log; return 1; }
  grep -m1 -iE "perception|detector|boot" ~/verify_batch_${J}_adapter_${SLOT}.log || true
  grep -m1 -iE "ladder configured" ~/verify_batch_${J}_adapter_${SLOT}.log || true

  # ---- ros bag capture (extra $SLOT level vs single-run so the scorer's
  # required captures/<scene>/<qdir>/bag leaf stays exact while multiple
  # slots reuse the same (scene, qdir) across the batch without clobbering) ----
  local BAGDIR=$CAP_ROOT/$SLOT/$SCN/$QD
  # Clear any stale DEGRADED sentinel at the SAME point the old bag is cleared
  # (#127 gap 2): SLOT already bakes in the loop index (${i}_${SCENE}_${QDIR}),
  # so BAGDIR is unique per matrix line within one job and nothing left by an
  # earlier slot should ever be sitting here — but that uniqueness is an
  # invariant of SLOT's naming, not of this function, so clear defensively
  # rather than trust it silently forever. Order matters: clear BEFORE the
  # fresh per-slot write below, never after.
  mkdir -p "$BAGDIR"; rm -rf "$BAGDIR/bag" "$BAGDIR/DEGRADED"
  # Sentinel lives next to (not inside) the bag dir so it never confuses
  # ros2 bag's own metadata/reindex handling; harvest_verify.sh checks for it.
  [ "$ENDPOINT_OK" -eq 0 ] && : > "$BAGDIR/DEGRADED"
  setsid ros2 bag record -o "$BAGDIR/bag" \
    /state_estimation /challenge_question /numerical_response /selected_object_marker \
    > ~/verify_batch_${J}_bag_${SLOT}.log 2>&1 &
  BAG_PID=$!
  sleep 3
  kill -0 $BAG_PID 2>/dev/null && echo "BAG: recording -> $BAGDIR/bag" || { echo "BAG: recorder DEAD — log tail:"; tail -10 ~/verify_batch_${J}_bag_${SLOT}.log; }

  # ---- question + monitor ----
  python3 ~/question_pub.py "$QUESTION" > /dev/null 2>&1 &
  QPUB_PID=$!
  echo "question publishing; monitor up (deadline 780s)"
  python3 ~/live_monitor.py 780 2>&1
  local MONITOR_RC=$?

  # ---- robust bounded bag stop (setsid process-group SIGINT, escalating,
  # NEVER a blocking `wait` — a bare `wait $BAG_PID` can hang the whole
  # allocation to the SLURM wall if rosbag2's child doesn't finalize) ----
  kill -INT -"$BAG_PID" 2>/dev/null || kill -INT "$BAG_PID" 2>/dev/null
  for i in $(seq 1 45); do kill -0 "$BAG_PID" 2>/dev/null || break; sleep 1; done
  kill -KILL -"$BAG_PID" 2>/dev/null; kill -KILL "$BAG_PID" 2>/dev/null
  finalize_bag "$BAGDIR/bag"
  echo "== bag capture"
  timeout 30 ros2 bag info "$BAGDIR/bag" 2>&1 | grep -E "Messages:|Topic:|Duration:" \
    || echo "BAG: no clean info (harvest reindex+convert will recover)"

  # ---- per-question verdict summary ----
  grep -q "transition:answer->" ~/verify_batch_${J}_adapter_${SLOT}.log && MONITOR_RC=0
  echo "== slot $SLOT monitor rc=$MONITOR_RC"
  echo "== slot $SLOT adapter log tail"; tail -25 ~/verify_batch_${J}_adapter_${SLOT}.log
  echo "== slot $SLOT parse tier"; grep -E "parsed: tier=|ladder" ~/verify_batch_${J}_adapter_${SLOT}.log | tail -5
  echo "== slot $SLOT LLM usage"; tail -12 "$VLA_LLM_USAGE_LOG" 2>/dev/null || echo "no usage log"
  # #127: a missing sim endpoint outranks MONITOR_RC/transition:answer-> — the
  # adapter can run a full FSM and "answer" against a stack that never came
  # up (final pose stuck at spawn), so that must never read as SUCCESS.
  if [ "$ENDPOINT_OK" -eq 0 ]; then
    echo "SLOT $SLOT: DEGRADED (ENDPOINT MISSING — sim stack never came up)"
  elif [ $MONITOR_RC -eq 0 ]; then
    echo "SLOT $SLOT: SUCCESS"
  else
    echo "SLOT $SLOT: rc=$MONITOR_RC"
  fi
  return 0
}

# ---------- main matrix loop ----------
# Each line failing must not abort the batch: run_one's own body already
# returns non-zero on a hard failure (dead Unity/adapter) instead of exiting
# the script, and the guard below logs+continues regardless. `set -u` (not
# `set -e`) is in effect for the whole file, matching the single-question
# sbatch, so a nonzero return here never aborts the loop on its own — the
# explicit guard is belt-and-braces documentation of that invariant.
i=0
while IFS='|' read -r SCENE QDIR SCENE_DIR QUESTION || [ -n "${SCENE:-}" ]; do
  [ -z "${SCENE:-}" ] && continue
  case "$SCENE" in \#*) continue ;; esac
  # Wall-guard: an inst question costs the full 780s monitor + ~4min overhead, so a
  # 15-question sweep can exceed -t. Stop cleanly with captures intact rather than
  # being SIGKILLed mid-question (which truncates that bag and skips the verdict).
  if [ $SECONDS -gt $(( WALL_MIN*60 - 1200 )) ]; then
    echo "WALL-GUARD: ${i} question(s) done, $(( SECONDS/60 ))min elapsed of ${WALL_MIN} — stopping before the wall"
    break
  fi
  SLOT="${i}_${SCENE}_${QDIR}"
  if run_one "$SCENE" "$QDIR" "$SCENE_DIR" "$QUESTION" "$SLOT"; then
    echo "== slot $SLOT: run_one returned OK"
  else
    echo "== slot $SLOT: run_one FAILED (rc=$?) — continuing to next question"
  fi
  i=$((i+1))
done < "$MATRIX"

echo "== gpu"; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
echo "VERIFY-BATCH-DONE: $i question(s) attempted"
