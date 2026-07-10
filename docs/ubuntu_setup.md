# Ubuntu Setup Guide — Phase 2 Machine

*The complete, ordered install procedure for the native Ubuntu box (the machine you reinstall when
home). Target: run the full challenge sim loop exactly as evaluation will. Written 10 Jul 2026;
tick the checkboxes as you go and note deviations in `LOG.md`.*

## 0. OS install

- [ ] **Ubuntu 24.04 LTS (Noble), 64-bit** — the challenge stack is built for it; do not substitute 22.04/25.x.
- [ ] During install: enable third-party drivers so the NVIDIA driver installs cleanly.
- [ ] Disk: ≥ 150 GB free for the repo + Docker images + Unity scene binaries + sample data.
- [ ] `sudo apt update && sudo apt full-upgrade -y && sudo reboot`

## 1. NVIDIA driver + verify GPU

```bash
sudo ubuntu-drivers install          # installs the recommended driver
sudo reboot
nvidia-smi                           # must show your GPU before proceeding
```

## 2. Docker + NVIDIA Container Toolkit

```bash
# Docker Engine (official repo, not snap)
sudo apt install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] \
  https://download.docker.com/linux/ubuntu noble stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list
sudo apt update && sudo apt install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
sudo usermod -aG docker $USER && newgrp docker   # docker without sudo

# NVIDIA Container Toolkit (GPU inside containers)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update && sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker

docker run --rm --gpus all ubuntu nvidia-smi    # verify GPU visible in a container
```

## 3. This workspace

```bash
sudo apt install -y git git-lfs python3.12-venv
git clone <your-remote-or-copy> ~/vla            # or rsync the Windows workspace over
cd ~/vla
python3 -m venv .venv && source .venv/bin/activate
pip install numpy pytest
cd src && pytest -q                              # the whole core suite must pass on Linux
```

If copying from Windows rather than cloning: copy the repo folder EXCLUDING `.venv/`, `data/`,
`upstream/` (re-fetch upstream fresh below — Windows checkouts can mangle line endings/symlinks).

## 4. Challenge repo + submodule

```bash
cd ~/vla/upstream 2>/dev/null || mkdir -p ~/vla/upstream && cd ~/vla/upstream
git clone https://github.com/Yuxin916/CMU-VLN-Challenge-2026.git CMU-VLN-Challenge-2026
cd CMU-VLN-Challenge-2026
git -c url."https://github.com/".insteadOf="git@github.com:" submodule update --init --recursive
```

(The `insteadOf` rewrite avoids needing a GitHub SSH key for the submodule; on this repo the
submodule is pinned to `81035e9`.)

## 5. Challenge Docker environment + Unity scenes

Follow `upstream/CMU-VLN-Challenge-2026/docker/README.md` exactly — summary of what it does
(details in `docs/upstream_notes.md` §4–5):

- [ ] Pull the challenge system image and the `ai_module` base image as instructed in `docker/`.
- [ ] Download the **training scene binaries** from the Google Drive folder linked in the upstream
      README ("training environments") — one Unity `Model.x86_64` per scene.
- [ ] Install a scene: place its files under
      `autonomy_stack_mecanum_wheel_platform/src/base_autonomy/vehicle_simulator/mesh/unity/environment/`
      and mark `Model.x86_64` executable (`chmod +x`). Swap scenes by swapping this folder's contents.
- [ ] Launch: the two-container compose stack (`system` + `ai_module`, host networking,
      CycloneDDS; Unity bridge on port 10000). First smoke test: `system_simulation.sh` with the
      dummy ai_module — the robot should sit in the scene and RVIZ should show the panorama,
      lidar, and terrain map.
- [ ] Env var sanity: `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp` must be consistent in BOTH containers
      or topic discovery silently fails (upstream gotcha 13).
- [ ] Verify the loop with the ordered Tier-2 smoke tests (containers → scene render → sensor
      liveness → actuation → dummy round-trip → relaunched dry-run) in `docs/sim_verification.md`,
      which extends this section with PASS/FAIL criteria per step.

## 6. Sample data for the replay harness

```bash
pip install gdown
python -m gdown --folder \
  "https://drive.google.com/drive/folders/1xaatyLeIKLTh_oRzkyd7F1G6tkRPbFtm" \
  -O ~/vla/data/sample_real_robot --remaining-ok
```

(Real-robot sample bags; play back with `system_bagfile.sh` or parse offline with the `rosbags`
pip package — no ROS needed for the latter.)

## 7. Our ai_module into the challenge stack (integration milestone)

1. `src/ros_adapter/` (written in Phase 2) is a rclpy node package: subscribes the six allowed
   topics → fills `core.interfaces` dataclasses → runs `core.fsm.QuestionController.tick()` at
   ~5 Hz → publishes via `RobotIO`. Keep it ~300 lines; zero logic in the adapter.
2. Mount/copy `src/` into the `ai_module` container per upstream's `ai_module/` build layout;
   replace the dummy launch entry with the adapter node.
3. Perception model weights (detector, local VLM fallback) are baked into the image at build time
   — the eval host must be assumed offline-capable (architecture §1 row 7). Keep the image under
   the size the eval machine tolerates (upstream README notes a Simply NUC i9 host).
   Before the Docker build, set the parse-provider env vars (`VLA_LLM_PRIMARY_*` / `_SECONDARY_*` /
   `_LOCAL_*` and the key vars they name, e.g. `OPENAI_API_KEY`) or place a keyless `llm_config.json`
   at the repo root — see `core/llm/config.py` for the full var list. Unset = ladder runs local/regex only.
4. Verify against a full training scene with the 10-minute clock before any submission
   (`docs/master_plan.md` Phase 2/3).

## 8. GPU sizing note

Evaluation (sim round) runs our container on the organisers' machine — plan VRAM for the RTX 4090
spec (architecture §6 targets ≈10–14 GB peak). If your home GPU is smaller, develop with the
detector in half precision / smaller variants and validate the full config on the SoC cluster
(`docs/soc_cluster_guide.md`) or accept slower local runs.

## Troubleshooting quick refs

| Symptom | Fix |
|---|---|
| Containers can't see each other's topics | `RMW_IMPLEMENTATION` mismatch — set CycloneDDS in both (gotcha 13) |
| Robot ignores waypoints | Publishing `/way_point` directly instead of `/way_point_with_heading` (gotcha 11), or waypoint too far — keep ≤2.5 m (gotcha 14) |
| Unity scene black/no sensors | `Model.x86_64` not executable, or bridge port 10000 blocked |
| GPU absent in container | re-run `nvidia-ctk runtime configure`, restart docker; `--gpus all` on run |
| Marker scored 0 despite correct object | wrong frame (must be `map`) or extents not full-size (gotchas 5–6) |
