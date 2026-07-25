#!/bin/bash
#SBATCH --job-name=live-run
#SBATCH -p gpu
#SBATCH --gpus=nv:1
#SBATCH -x xgpe0,xgpe2,xgpe6
#SBATCH -t 25
#SBATCH -c 16
#SBATCH --mem=48G
#SBATCH -o /home/c/cyuhsin/live_run.out

set -u
ROOT=~/.udocker/containers/64af1b4a-7bad-3be3-984b-bdccca76ee40/ROOT
WS=$ROOT/home/docker/autonomy_stack_mecanum_wheel_platform
XR=~/xstack/root
J=$SLURM_JOB_ID
QUESTION="How many chairs are in the room?"

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

echo "== live-run $J on $(hostname) $(date +%T)  question: $QUESTION"

# ---------- X + Unity ----------
mkdir -p /tmp/.X11-unix 2>/dev/null
$XR/usr/bin/Xvfb :99 -screen 0 1280x720x24 -xkbdir $XR/usr/share/X11/xkb -noreset 2>/dev/null &
XVFB_PID=$!; sleep 3
export DISPLAY=:99
MODEL=$WS/src/base_autonomy/vehicle_simulator/mesh/unity/environment/Model.x86_64
chmod +x "$MODEL" 2>/dev/null
$XR/opt/VirtualGL/bin/vglrun -ld $XR/usr/lib -d egl0 "$MODEL" -force-glcore -screen-width 1280 -screen-height 720 > ~/live_run_${J}_unity.log 2>&1 &
UNITY_PID=$!; sleep 8
kill -0 $UNITY_PID 2>/dev/null && echo "UNITY: alive" || { echo "UNITY: DEAD"; }

# ---------- autonomy stack ----------
timeout 1380 ros2 launch vehicle_simulator system_simulation.launch > ~/live_run_${J}_launch.log 2>&1 &
LAUNCH_PID=$!
echo "stack launched; settling 45s"; sleep 45
(ss -ltn 2>/dev/null || true) | grep -q ":10000" && echo "ENDPOINT: :10000 up" || echo "ENDPOINT: MISSING"

# ---------- ai_module adapter (venv torch/GDINO + ROOT rclpy) ----------
source ~/venv-gpu/bin/activate
export PYTHONPATH=~/vla/src:$PYTHONPATH
export VLA_DETECTOR=grounding_dino GDINO_DEVICE=cuda GDINO_PRECISION=fp16
# cap torch/OpenMP CPU threads — with all cores claimed, Unity's frame encode starves
# and the camera stream collapses (v7/v8 evidence: 2Hz -> 0.1Hz once GDINO loads)
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export GDINO_CONFIG_PATH=~/models/gdino/GroundingDINO_SwinB_cfg.py
export GDINO_CHECKPOINT_PATH=~/models/gdino/groundingdino_swinb_cogcoor.pth
# compressed camera straight into the adapter (raw 3.7MB frames can't survive
# unprivileged CycloneDDS here; the ~100KB jpeg stream can — VLA_CAMERA_COMPRESSED
# decodes in-process and re-enters the exact _on_image path)
export VLA_CAMERA_COMPRESSED=1
BRIDGE_PID=""
python -c "from ros_adapter.adapter_node import main; main()" > ~/live_run_${J}_adapter.log 2>&1 &
ADAPTER_PID=$!
sleep 25
kill -0 $ADAPTER_PID 2>/dev/null && echo "ADAPTER: alive" || { echo "ADAPTER: DEAD — log tail:"; tail -15 ~/live_run_${J}_adapter.log; }
grep -m1 -iE "perception|detector|boot" ~/live_run_${J}_adapter.log || true

# ---------- question + monitor ----------
python3 ~/question_pub.py "$QUESTION" > /dev/null 2>&1 &
QPUB_PID=$!
echo "question publishing; monitor up (deadline 780s)"
python3 ~/live_monitor.py 780 2>&1
MONITOR_RC=$?

echo "== monitor rc=$MONITOR_RC"
echo "== adapter log tail"; tail -25 ~/live_run_${J}_adapter.log
echo "== gpu"; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
echo "== screenshot"; $XR/usr/bin/xwd -root -display :99 -out ~/live_run_${J}_screen.xwd 2>/dev/null && echo saved || true

echo "== bridge log tail"; tail -3 ~/live_run_${J}_bridge.log 2>/dev/null
echo "== grounded or floor?"; grep -E "answer_from_floor|answer_from_head|published int answer|instances" ~/live_run_${J}_adapter.log | tail -5
kill $QPUB_PID $BRIDGE_PID $ADAPTER_PID $LAUNCH_PID $UNITY_PID $XVFB_PID 2>/dev/null
sleep 2
[ $MONITOR_RC -eq 0 ] && echo "LIVE-RUN-DONE: SUCCESS" || echo "LIVE-RUN-DONE: rc=$MONITOR_RC"
