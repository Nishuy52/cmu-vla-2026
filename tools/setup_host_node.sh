#!/usr/bin/env bash
# One-time host setup for the FAST ITERATION loop: run the ai_module ROS node
# natively on this machine against the sim container's ROS graph (host
# networking + CycloneDDS), instead of rebuilding the docker image per change.
# Docker remains the truth at checkpoints — see ubuntu_setup.md §8b.
#
# Run WITHOUT sudo (it sudo-prompts once for the apt part):
#   bash tools/setup_host_node.sh
#
# The python env mirrors the image's pins (docker/ai_module_fork/docker/
# Dockerfile is the single source of truth): numpy==1.26.4,
# transformers==4.57.6 (issue #38), torch/torchvision default CUDA wheels,
# groundingdino-py. Weights + HF cache are copied OUT of the built
# docker-ai_module image rather than re-downloaded.
set -euo pipefail

HOST_DIR="$HOME/vla_host"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ---------- 1. ROS 2 Jazzy (sudo; skipped if already installed) ----------
if [[ ! -d /opt/ros/jazzy ]]; then
  echo "==> Installing ROS 2 Jazzy (ros-base) + CycloneDDS RMW (sudo)..."
  sudo apt-get update
  sudo apt-get install -y software-properties-common curl
  sudo add-apt-repository -y universe
  sudo curl -fsSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
    -o /usr/share/keyrings/ros-archive-keyring.gpg
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo "$UBUNTU_CODENAME") main" \
    | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
  sudo apt-get update
  sudo apt-get install -y ros-jazzy-ros-base ros-jazzy-rmw-cyclonedds-cpp python3-venv
else
  echo "==> /opt/ros/jazzy present, skipping ROS install"
fi

# ---------- 2. Pinned venv mirroring the image ----------
echo "==> Creating venv at $HOST_DIR/venv (system-site-packages for rclpy)..."
mkdir -p "$HOST_DIR"
python3 -m venv --system-site-packages "$HOST_DIR/venv"
CONSTRAINTS="$HOST_DIR/pip-constraints.txt"
printf 'numpy==1.26.4\ntransformers==4.57.6\n' > "$CONSTRAINTS"
PIP_CONSTRAINT="$CONSTRAINTS" "$HOST_DIR/venv/bin/pip" install --no-cache-dir \
  torch torchvision "transformers==4.57.6" groundingdino-py \
  "rosbags>=0.11" pillow
"$HOST_DIR/venv/bin/python" - <<'EOF'
import torch, transformers
assert transformers.__version__ == "4.57.6", transformers.__version__
from transformers import BertModel
assert hasattr(BertModel, "get_head_mask"), "transformers pin regressed (#38)"
print(f"venv OK: torch {torch.__version__}, cuda={torch.cuda.is_available()}, transformers {transformers.__version__}")
EOF

# ---------- 3. Weights + HF cache out of the built image ----------
docker_cmd() {  # direct docker if the group applies, else sg fallback
  if docker ps >/dev/null 2>&1; then docker "$@"; else sg docker -c "docker $*"; fi
}
if [[ ! -f "$HOST_DIR/weights/groundingdino/groundingdino_swinb_cogcoor.pth" ]]; then
  echo "==> Extracting GDINO weights + HF cache from docker-ai_module:latest..."
  CID=$(docker_cmd create docker-ai_module:latest)
  docker_cmd cp "$CID:/opt/vla/weights" "$HOST_DIR/weights"
  docker_cmd cp "$CID:/opt/vla/hf_cache" "$HOST_DIR/hf_cache"
  docker_cmd rm "$CID" > /dev/null
else
  echo "==> Weights already extracted, skipping"
fi

# ---------- 4. Env file consumed by run_host_node.sh ----------
cat > "$HOST_DIR/env.sh" <<EOF
# Sourced by tools/run_host_node.sh — host-native ai_module environment.
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
export GDINO_CONFIG_PATH=$HOST_DIR/weights/groundingdino/GroundingDINO_SwinB_cfg.py
export GDINO_CHECKPOINT_PATH=$HOST_DIR/weights/groundingdino/groundingdino_swinb_cogcoor.pth
export HF_HOME=$HOST_DIR/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLA_DETECTOR=\${VLA_DETECTOR:-grounding_dino}
# Local LLM slot (host Ollama, ubuntu_setup §8a) — uncomment to enable:
# export VLA_LLM_LOCAL_KIND=openai
# export VLA_LLM_LOCAL_BASE_URL=http://localhost:11434/v1
# export VLA_LLM_LOCAL_MODEL=qwen2.5vl:3b
# export VLA_LOCAL_API_KEY=local
EOF

echo "==> Done. Start the sim containers as usual, then run: bash tools/run_host_node.sh"
