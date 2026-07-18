#!/usr/bin/env bash
# Fast-iteration runner: launch OUR adapter node natively on the host against
# the sim container's ROS graph. Edit src/ -> rerun this (seconds), instead of
# sync_to_fork + docker build (minutes). Setup once via tools/setup_host_node.sh.
# The sim (iros2026_system) must be running; do NOT also run the containerised
# ai_module — two adapters on one graph will fight over the question latch.
set -euo pipefail

HOST_DIR="$HOME/vla_host"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

[[ -f "$HOST_DIR/env.sh" ]] || { echo "run tools/setup_host_node.sh first"; exit 1; }
set +u  # ROS setup.bash and venv activate are not `set -u`-clean
source /opt/ros/jazzy/setup.bash
source "$HOST_DIR/env.sh"
source "$HOST_DIR/venv/bin/activate"
set -u
export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

echo "adapter node starting (VLA_DETECTOR=$VLA_DETECTOR, RMW=$RMW_IMPLEMENTATION); ctrl-C to stop"
exec python -c "from ros_adapter.adapter_node import main; main()"
