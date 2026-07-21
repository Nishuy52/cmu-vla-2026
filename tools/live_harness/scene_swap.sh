#!/usr/bin/env bash
# Clean scene swap: container restart (clean tree), scene install, sim launch, scan assert.
set -u
SCENE="$1"
U=/home/docker/autonomy_stack_mecanum_wheel_platform/src/base_autonomy/vehicle_simulator/mesh/unity
DISPLAY=:1 xhost + >/dev/null 2>&1
docker restart iros2026_system >/dev/null 2>&1 || docker start iros2026_system >/dev/null 2>&1
sleep 8
docker exec iros2026_system rm -rf $U/environment
docker cp /home/jason/cmu_ws/cmu-vla-2026/data/unity_scenes_ros2/$SCENE/$SCENE/. iros2026_system:$U/
docker exec -u root iros2026_system bash -c "chown -R docker:docker $U && chmod +x $U/environment/Model.x86_64"
docker exec -d iros2026_system bash -c 'export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp; cd /home/docker/autonomy_stack_mecanum_wheel_platform && ./system_simulation.sh > /tmp/sim_launch.log 2>&1'
SETTLE=75; [ "$SCENE" = "arabic_room" ] && SETTLE=180
sleep $SETTLE
C=$(docker exec iros2026_system bash -c 'pgrep -cf "[M]odel.x86_64"')
[ "$C" = "1" ] || { echo "SCENE_LAUNCH_BAD count=$C"; exit 3; }
F=$(bash -c 'export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp; source /opt/ros/jazzy/setup.bash; timeout 30 ros2 topic hz /registered_scan 2>&1 | grep -c "average rate"' || true)
[ "$F" = "1" ] || { echo "SCAN_NOT_FLOWING $SCENE"; exit 4; }
echo "SCENE_READY $SCENE"
