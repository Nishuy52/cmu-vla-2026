#!/bin/bash
#SBATCH --job-name=sim-smoke
#SBATCH -p gpu
#SBATCH --gpus=nv:1
#SBATCH -x xgpe0,xgpe2,xgpe6
#SBATCH -t 15
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -o /home/c/cyuhsin/sim_smoke.out

set -u
ROOT=~/.udocker/containers/64af1b4a-7bad-3be3-984b-bdccca76ee40/ROOT
WS=$ROOT/home/docker/autonomy_stack_mecanum_wheel_platform
XR=~/xstack/root

# ---- bare-metal ROS env from the extracted sim image (proven: rclpy + 336 pkgs)
export PATH=$ROOT/opt/ros/jazzy/bin:$PATH
export PYTHONPATH=$ROOT/opt/ros/jazzy/lib/python3.12/site-packages:$ROOT/usr/lib/python3/dist-packages:$ROOT/usr/lib/python3.12/dist-packages
# $ROOT/usr/lib (non-multiarch) carries libOpenNI.so.0 + libarmadillo.so.12 (v2 casualties)
export LD_LIBRARY_PATH=$ROOT/opt/ros/jazzy/lib:$ROOT/opt/ros/jazzy/lib/x86_64-linux-gnu:$ROOT/usr/lib/x86_64-linux-gnu:$ROOT/lib/x86_64-linux-gnu:$ROOT/usr/lib
export AMENT_PREFIX_PATH=$ROOT/opt/ros/jazzy

# ---- overlay: every package prefix in the isolated colcon install tree
for d in "$WS"/install/*/; do
  pkg=$(basename "$d")
  [ "$pkg" = "COLCON_IGNORE" ] && continue
  [ -d "$d" ] || continue
  export AMENT_PREFIX_PATH="$d:${AMENT_PREFIX_PATH}"
  [ -d "$d/lib" ] && export LD_LIBRARY_PATH="$d/lib:$LD_LIBRARY_PATH"
  [ -d "$d/lib/python3.12/site-packages" ] && export PYTHONPATH="$d/lib/python3.12/site-packages:$PYTHONPATH"
  [ -d "$d/bin" ] && export PATH="$d/bin:$PATH"
done
# strip trailing colons
export AMENT_PREFIX_PATH="${AMENT_PREFIX_PATH%:}"

# ---- develop-mode (egg-link) packages: their site-packages entry is a pointer to a
# build-time absolute path that only exists under $ROOT — resolve and prefix it
for link in "$WS"/install/*/lib/python3.12/site-packages/*.egg-link; do
  [ -f "$link" ] || continue
  tgt=$(head -1 "$link")
  [ -d "$ROOT$tgt" ] && export PYTHONPATH="$ROOT$tgt:$PYTHONPATH"
done

export ROS_DISTRO=jazzy
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export ROS_LOG_DIR=/tmp/roslog-$SLURM_JOB_ID
export ROS_HOME=/tmp/roshome-$SLURM_JOB_ID
mkdir -p "$ROS_LOG_DIR" "$ROS_HOME"
# Same-host stack: force CycloneDDS onto loopback — the compute fabric drops multicast,
# so default-interface discovery finds nothing (v3 symptom: nodes up, zero topics visible)
export CYCLONEDDS_URI='<CycloneDDS><Domain><General><Interfaces><NetworkInterface address="127.0.0.1" multicast="true"/></Interfaces></General></Domain></CycloneDDS>'

echo "== node: $(hostname)  $(date +%T)"

# ---- headless X + GPU GL
mkdir -p /tmp/.X11-unix 2>/dev/null
$XR/usr/bin/Xvfb :99 -screen 0 1280x720x24 -xkbdir $XR/usr/share/X11/xkb -noreset 2>/dev/null &
XVFB_PID=$!
sleep 3
export DISPLAY=:99

# ---- Unity sim (baked stock scene) under VirtualGL EGL
MODEL=$WS/src/base_autonomy/vehicle_simulator/mesh/unity/environment/Model.x86_64
chmod +x "$MODEL" 2>/dev/null
$XR/opt/VirtualGL/bin/vglrun -ld $XR/usr/lib -d egl0 "$MODEL" -force-glcore -screen-width 1280 -screen-height 720 > /tmp/unity-$SLURM_JOB_ID.log 2>&1 &
UNITY_PID=$!
sleep 8
kill -0 $UNITY_PID 2>/dev/null && echo "UNITY: alive" || echo "UNITY: DEAD"

# ---- autonomy stack + bridge (logs straight into home so nothing is lost on a hang)
timeout 500 ros2 launch vehicle_simulator system_simulation.launch > ~/sim_smoke_launch.log 2>&1 &
LAUNCH_PID=$!
echo "launch started, settling 60s"
sleep 60

echo "== endpoint listening?"
(ss -ltn 2>/dev/null || netstat -ltn 2>/dev/null) | grep -E ":10000" || echo "   :10000 NOT LISTENING"
echo "== unity bridge status"
grep -iE "connect|ros" /tmp/unity-$SLURM_JOB_ID.log 2>/dev/null | tail -3
cp /tmp/unity-$SLURM_JOB_ID.log ~/sim_smoke_unity.log 2>/dev/null

# rclpy-based probe — no ros2 CLI, no daemon, hard-bounded runtime
echo "== rclpy probe (graph + 12s rates)"
timeout 90 python3 ~/topic_probe.py 12 < /dev/null 2>&1 | grep -vE "^$"
echo "== gpu"
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
echo "== launch log tail"
tail -12 ~/sim_smoke_launch.log

kill $LAUNCH_PID $UNITY_PID $XVFB_PID 2>/dev/null
timeout 20 ros2 daemon stop < /dev/null > /dev/null 2>&1
sleep 2
echo SIM-SMOKE-DONE
