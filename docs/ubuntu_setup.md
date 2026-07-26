# Ubuntu Setup Guide — Phase 2 Machine

*The complete, ordered install procedure for the native Ubuntu box (the machine you reinstall when
home). Target: run the full challenge sim loop exactly as evaluation will. Written 10 Jul 2026;
tick the checkboxes as you go and note deviations in `LOG.md`.*

## 0. OS install

- [x] **Ubuntu 24.04 LTS (Noble), 64-bit** — the challenge stack is built for it; do not substitute 22.04/25.x.
- [x] During install: enable third-party drivers so the NVIDIA driver installs cleanly.
- [x] Disk: ≥ 150 GB free for the repo + Docker images + Unity scene binaries + sample data.
- [ ] `sudo apt update && sudo apt full-upgrade -y && sudo reboot`

## 1. NVIDIA driver + verify GPU

```bash
sudo ubuntu-drivers install          # installs the recommended driver
sudo reboot
nvidia-smi                           # must show your GPU before proceeding
```

**RTD3 wedge hardening (required on this box — RTX 4060 Laptop):** the GPU
recurrently wedges into P8 @ 210 MHz under runtime power management and
never recovers without a reboot (see issue #86 and the 2026-07-20 live
baseline). After any driver install/upgrade run:

```bash
sudo tools/gpu_hardening/install.sh
```

Idempotent; applies the sysfs pin immediately and persists it. It installs
the `zz-nvidia-disable-rtd3.conf` modprobe option (needs one reboot if the
driver was already loaded), the udev pin, removes gpu-manager's
`/etc/u-d-c-nvidia-runtimepm-override` trigger, and enables a
`nvidia-pm-pin.service` oneshot ordered after gpu-manager (which otherwise
rewrites `power/control` to `auto` at every boot). Re-run after driver
package upgrades — they recreate the override flag. Verify:

```bash
cat /sys/bus/pci/devices/0000:01:00.0/power/control     # expect: on
grep DynamicPowerManagement /proc/driver/nvidia/params  # expect: 0
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

**Post-reboot note:** after a full reboot (not just `newgrp docker` in the current
shell), the `docker` group membership from `usermod -aG docker $USER` is picked up by
new login sessions automatically — plain `docker ...` commands work with no wrapper.
The `sg docker -c '...'` wrapper is only needed as a fallback in a **stale session**
that predates the group add (e.g. a long-lived autonomous/agent session started before
the reboot) — use it there instead of restarting that session.

```bash
sudo apt install -y git git-lfs python3.12-venv
git clone <your-remote-or-copy> ~/vla            # or rsync the Windows workspace over
cd ~/vla
python3 -m venv .venv && source .venv/bin/activate
pip install numpy pytest pytest-xdist rosbags pypdf   # rosbags = pure-Python ROS 2 bag read/write for the offline replay harness (no ROS needed); pytest-xdist parallelises the full gate; pypdf = tools/extract_pdf_answers.py (GT answer-key extraction from the upstream questions.pdf, dev-only)
cd src && pytest -m "" -n auto                    # the whole core suite must pass on Linux (full gate — plain `pytest` runs only the fast tier; see src/README.md "Test tiers")
```

If copying from Windows rather than cloning: copy the repo folder EXCLUDING `.venv/`, `data/`,
`upstream/` (re-fetch upstream fresh below — Windows checkouts can mangle line endings/symlinks).

**As-built (18 Jul 2026):** the repo actually lives at `~/cmu_ws/cmu-vla-2026` with a symlink
`~/vla -> ~/cmu_ws/cmu-vla-2026`, so every `~/vla` path in this guide works unchanged. The §0–§3
apt installs are scripted in `~/cmu_ws/setup_env.sh` (idempotent, re-runnable). ROS 2 Jazzy
(`ros-jazzy-desktop` + `ros-dev-tools`) is also installed natively and sourced in `~/.bashrc` —
not required by the Docker-based challenge stack, but useful for host-side `ros2 topic` debugging
against the host-networked containers.

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

- [x] Pull the challenge system image and the `ai_module` base image as instructed in `docker/`.
- [x] Download the **training scene binaries** from the Google Drive folder linked in the upstream
      README ("training environments") — one Unity `Model.x86_64` per scene.
- [x] Install a scene: place its files under
      `autonomy_stack_mecanum_wheel_platform/src/base_autonomy/vehicle_simulator/mesh/unity/environment/`
      and mark `Model.x86_64` executable (`chmod +x`). Swap scenes by swapping this folder's contents.
- [x] Launch: the two-container compose stack (`system` + `ai_module`, host networking,
      CycloneDDS; Unity bridge on port 10000). First smoke test: `system_simulation.sh` with the
      dummy ai_module — the robot should sit in the scene and RVIZ should show the panorama,
      lidar, and terrain map.
- [x] Env var sanity: `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp` must be consistent in BOTH containers
      or topic discovery silently fails (upstream gotcha 13).
- [ ] Verify the loop with the ordered Tier-2 smoke tests (containers → scene render → sensor
      liveness → actuation → dummy round-trip → relaunched dry-run) in `docs/sim_verification.md`,
      which extends this section with PASS/FAIL criteria per step.

**As-built findings (18 Jul 2026, first Ubuntu bring-up):**

- **Scene-source trap:** the correct scene zips are the ones the 2026 upstream README links —
  Drive folder `unity_env_models` (`1nki_xoFKX1bYr8m7qiGRQelwnQ7EKVYc`), binaries built Nov 2024.
  A near-identical 2025-era folder `cmu_vla_challenge_unity_environments_ros1` (binaries Nov 2023)
  circulates in search results and is **protocol-incompatible** with the ROS2 `ros_tcp_endpoint`:
  symptom is a reconnect loop of `Connection from 127.0.0.1` + `JSONDecodeError` in
  `handle_syscommand` and silent sensor topics, while `/state_estimation` (host-side odometry)
  still streams at 200 Hz. Check `Model.x86_64`'s file date if in doubt.
  `tools/fetch_unity_scenes.sh` downloads the correct set per-file via gdown (works on Ubuntu;
  the 10 Jul gdown failure was folder-mode/Windows-specific). Scenes live in
  `~/vla/data/unity_scenes_ros2/<scene>/<scene>/`.
- **Scene install is `docker cp`**, not a mount: the system container is a prebuilt image with the
  scene baked in at `.../mesh/unity/` (shipped default = livingroom_3). Swap:
  `docker exec iros2026_system rm -rf $U/environment`, `docker cp <scene>/<scene>/. iros2026_system:$U/`,
  then `docker exec -u root ... chown -R docker:docker $U && chmod +x $U/environment/Model.x86_64`
  (cp'd files arrive root-owned). Survives `docker restart`, lost on container re-create.
- **Gotcha 13 has a sharper form:** the containers set `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp`
  only in `~/.bashrc`, so **non-interactive `docker exec bash -c` shells skip it** and run FastDDS
  — discovery still works (topics/subscriber counts look right) but no data crosses containers
  (FastDDS shared-memory transport, separate `/dev/shm`). Export it explicitly in every exec, or
  use `bash -ic`. After changing middleware, restart both containers — stale daemons/nodes from
  the wrong RMW break node creation.
- `office_building_1/2` ship with **empty `object_list.txt`** in both scene sets — they are
  platform extras, not among the 15 training scenes.
- On the RTX 4060 Laptop, topic rates run below contract with RVIZ rendering
  (`/camera/image` ~3.7 Hz vs ~10, terrain ~3.9 vs ~5; scan ~4–5 Hz) — tracked as a GitHub issue.

## 6. Sample data for the replay harness

```bash
pip install gdown
python -m gdown --folder \
  "https://drive.google.com/drive/folders/1xaatyLeIKLTh_oRzkyd7F1G6tkRPbFtm" \
  -O ~/vla/data/sample_real_robot --remaining-ok
```

(Real-robot sample bags; play back with `system_bagfile.sh` or parse offline with the `rosbags`
pip package — no ROS needed for the latter.)

The offline path is `core/replay/` (bag → core dataclasses → `ReplayRobotIO` → the full
`QuestionController`). Because a full bag can be ~3 GB and may live on a remote Linux box, distil
it to compact portable fixtures on the cluster and transfer only those:

```bash
python -m core.replay.fixtures extract ~/vla/data/sample_real_robot/<bag> ~/vla/data/fixtures/<name>
# then load with core.replay.load_fixtures(dir) -> a store ReplayRobotIO drives
```

The distillation writes one compressed `.npz` per movement-gated keyframe (pano uint8 +
decimated scan/terrain float32 + odom) plus a schema-versioned `index.json`, shrinking a
multi-GB bag to a few hundred MB that move freely between machines.

## 7. Our ai_module into the challenge stack (integration milestone)

1. `src/ros_adapter/` (written in Phase 2) is a rclpy node package: subscribes the six allowed
   topics → fills `core.interfaces` dataclasses → runs `core.fsm.QuestionController.tick()` at
   ~5 Hz → publishes via `RobotIO`. Keep it ~300 lines; zero logic in the adapter.
2. Mount/copy `src/` into the `ai_module` container per upstream's `ai_module/` build layout;
   replace the dummy launch entry with the adapter node.
3. Perception model weights (detector, local VLM fallback) are baked into the image at build time
   — the eval host must be assumed offline-capable (architecture §1 row 7). Keep the image under
   the size the eval machine tolerates (upstream README notes a Simply NUC i9 host).
   The open-vocab detector is a GroundingDINO-class model (`core/perception/detector.py`
   `GroundingDinoDetector`, lazy-imported); install its deps into the image and pre-download the
   weights so nothing is fetched at run time.
   **DONE 18 Jul — baked into the fork Dockerfile** (`docker/ai_module_fork/docker/Dockerfile`),
   as-built facts:
   - Deps layer: `torch torchvision transformers groundingdino-py` under
     `PIP_CONSTRAINT numpy==1.26.4` (opencv≥4.12 would drag numpy 2 and break apt scipy; pip
     backtracks to opencv 4.11). Image grows 2.58 → **6.76 GB**.
   - Weights: `GDINO_MODEL_ID` (grounding-dino-base) = **Swin-B** →
     `groundingdino_swinb_cogcoor.pth` (938 MB, GitHub release v0.1.0-alpha2) at
     `GDINO_CHECKPOINT_PATH=/opt/vla/weights/groundingdino/...pth`, plus the matching
     `GroundingDINO_SwinB_cfg.py` copied out of the package to `GDINO_CONFIG_PATH` (without it
     the detector falls back to the SwinT config → state-dict mismatch).
   - **Hidden second fetch:** groundingdino constructs a `bert-base-uncased` tokenizer/encoder
     via HF at model build — baked (~441 MB) into `HF_HOME=/opt/vla/hf_cache`, then
     `HF_HUB_OFFLINE=1` + `TRANSFORMERS_OFFLINE=1` set so runtime resolves only from the baked
     cache (fails loud, never hangs on network).
   - A build-time assertion layer verifies checkpoint size, SwinB config marker
     (`swin_B_384_22k`), and the BERT snapshot.
   Until these are present the detector raises a clear ImportError and the offline path falls back
   to the scripted `FakeDetector` (tests only).
   - **Dual-caption detection (issue #42, added 18 Jul):** the live ~117-phrase
     question+vocab caption decodes zero `teapot`-class detections at any threshold
     (caption dilution); a short question-noun-only caption recovers them. Every
     detection tick now grounds that short caption at its own, lower threshold
     (`GDINO_QUESTION_BOX_THRESHOLD`, default `0.25`) in addition to the original
     question+vocab caption, which now runs at reduced cadence
     (`GDINO_VOCAB_PASS_CADENCE`, default every 3rd tick — it only feeds scene-index
     breadth, not target recall). Both are optional env overrides; unset = the module
     defaults above.
   Before the Docker build, set the parse-provider env vars (`VLA_LLM_PRIMARY_*` / `_SECONDARY_*` /
   `_LOCAL_*` and the key vars they name, e.g. `OPENAI_API_KEY`) or place a keyless `llm_config.json`
   at the repo root — see `core/llm/config.py` for the full var list. Unset = ladder runs local/regex only.
4. Verify against a full training scene with the 10-minute clock before any submission
   (`docs/master_plan.md` Phase 2/3).

## 7a. Build & run OUR ai_module (Phase-2 adapter artifacts) — FORK-SHAPED packaging (H7)

*These artifacts (`src/ros_adapter/`, `docker/ai_module_fork/`) were written on Windows and are
**untested drafts** — expect to fix small things on first Ubuntu build; anything surprising is
flagged "confirm on Ubuntu" in the files.*

**Packaging shape (H7 / red-team F4).** The submission is a **fork** in which only `ai_module/`
may be modified, and the upstream compose builds the `ai_module` service from **context
`../ai_module` using `ai_module/docker/Dockerfile`** (`docs/upstream_notes.md` §5B, gotcha 12).
Our fork-shaped Dockerfile + sync scripts live in `docker/ai_module_fork/`. The old
`docker/ai_module/` (repo-root build context) is **deprecated — do not build from it.**

**The packaging gate is now: `docker compose up --build` from a clean clone of the FORK
succeeds.** Building our repo directly does not count. Exact commands:

```bash
# 0. clean clone of the FORK (the submitted repo) — NOT our dev repo
git clone <fork-remote> /tmp/fork-clean

# 1. stage our payload (src/ minus tests + the fork-shaped Dockerfile) into ai_module/
#    from a checkout of THIS repo:
~/vla/docker/ai_module_fork/sync_to_fork.sh /tmp/fork-clean
#    -> writes /tmp/fork-clean/ai_module/docker/Dockerfile and /tmp/fork-clean/ai_module/src/

# 2. edit the fork's docker/compose.yml ai_module service to launch OUR node. Phase 3
#    (in-image Ollama bake, docs/local_llm_plan.md): the command is the launch wrapper,
#    not a bare ros2 launch — it starts the baked ollama serve + pre-warm before exec'ing
#    the same ros2 launch tail:
#    build.context: ../ai_module ; build.dockerfile: docker/Dockerfile
#    command: /bin/bash -lc "/opt/vla/launch_with_llm.sh"
#    environment: RMW_IMPLEMENTATION=rmw_cyclonedds_cpp  (re-assert; OVERRIDES the Dockerfile ENV)
#                 VLA_DETECTOR=grounding_dino  (issue #55: with no environment: block at all, this
#                 was unset -> adapter_node.py defaulted to VLA_DETECTOR=none -> offline stub ->
#                 SUBMISSION-BLOCKER line, failing the perception pre-submit check below)
#                 OLLAMA_KEEP_ALIVE=${OLLAMA_KEEP_ALIVE:-5m}  (short dev-box default; see below)
#    env_file: ../.env.llm  (optional; VLA_LLM_* keys)
#    Full snippet in docker/ai_module_fork/README.md; ready-made files at
#    docker/ai_module_fork/docker/compose.yml and compose_gpu.yml (copy-paste, not auto-synced —
#    gotcha 12, they live outside ai_module/).

# 3. build + bring the stack up exactly as evaluators do:
xhost +
cd /tmp/fork-clean/docker && docker compose -f compose.yml up --build -d
```

**As-built gotchas for step 3 (19 Jul, first full in-container run):**

- **`DISPLAY` propagates at `docker compose up` time** from the invoking shell into the
  system container. A shell without `DISPLAY` (cron, detached scripts, agent sessions)
  creates a container whose RVIZ dies with the Qt "xcb" platform error and whose sensor
  pipeline never comes up. Find the desktop display first (`who` shows `(:N)`), then
  `DISPLAY=:N docker compose ... up -d --force-recreate system`. Scene re-install is
  needed after every container re-create (docker cp is per-container state, §5).
- **Disk hygiene: run `docker builder prune -f --keep-storage 12GB` after every image
  build.** Build cache grows ~10 GB per full build generation; three generations filled
  the 190 GB volume mid-build on 19 Jul (ENOSPC during export).
- **The controller holds a 60 s ORIENT (in-place sweep) window after question latch
  before the first waypoint** — no `/way_point_with_heading` traffic for the first
  minute is CORRECT behavior, not a hang. When watching with `ros2 topic echo`,
  remember the message is `Pose2D` (fields `x:/y:/theta:`).
- **One question per adapter process** (gotcha 1): a second publish is dropped with a
  SECOND QUESTION DROPPED error; `docker restart iros2026_ai_module` for a fresh run.
  After the 600 s budget expires the controller goes DONE and stops publishing.

**Phase-3 bake (in-image Ollama, `docs/local_llm_plan.md` Phase 3 — authored-pending-build,
`b507eb9`):** the Dockerfile now also bakes an Ollama standalone server (tarball pinned
`OLLAMA_VERSION=v0.32.1`, non-CUDA/non-CPU runner variants pruned) plus the `qwen2.5vl:3b`
model blobs, so the local LLM tier is served entirely in-container at eval (no host
Ollama at eval — organisers run only this container). **Measured built size (19 Jul,
first real build): 28.1 GB total** — the old "6.76 GB baseline +5 GB" was an
authored-pending-build estimate; the real layers are base image 9.4 + ros-jazzy-desktop
3.4 + torch/CUDA pip 5.3 + GDINO weights 1.4 + Ollama runtime/model 3.2 + assorted
sub-GB layers, all load-bearing. The bake needs network at *build* time (Ollama
release + model registry, same class of build-time dependency as the GDINO weights
fetch) even though the resulting image is offline at *runtime*. Ordered post-build
verification (size delta, boot-log grep sequence, curl smoke, in-container ladder
conformance, VRAM coexistence with GDINO, and the clean-clone gate below) is
`reports/local_llm_phase3/BUILD_AND_VERIFY.md` — run it after every Phase-3 image build
before trusting the result.

~~Resolve the three **confirm-on-Ubuntu flags** marked in the Dockerfile as you go~~ **All three
resolved 18 Jul** (details in the Dockerfile's updated comments): (1) CycloneDDS cross-container —
works with the compose env entry; (2) PEP 668 — flag required, AND the base image ships with **no
pip at all** (Dockerfile now bootstraps `python3-pip` as root; pip installs split in two because
pytest's `pluggy>=1.5` collides with the apt-owned 1.4.0 and a blanket `--ignore-installed` would
break apt's numpy/scipy); (3) colcon symlink layout — works as drafted, with `--chown` on COPY and
the overlay `source` line in `/home/docker/.bashrc` (not root's — same trap class as gotcha 13).
Also fixed on first Ubuntu boot: `adapter_node.py` assigned `self._clock`, shadowing rclpy `Node`'s
own `_clock` → RecursionError; renamed `self._robotio_clock`. Tier 2.7 PASSED 18 Jul (twice,
independently verified): question latched, 5 Hz exploration waypoints, exactly one legal integer
on `/numerical_response` at latch+~219 s.

**First smoke test:** run the ordered Tier-2 checks in `docs/sim_verification.md`, especially
**Tier 2.7** (our-module round-trip — publish a challenge question, confirm the adapter latches
it and emits a legal answer on the matching topic). Keys unset ⇒ the parse ladder runs
local/regex only (offline).

## 8. GPU sizing note

Evaluation (sim round) runs our container on the organisers' machine — plan VRAM for the RTX 4090
spec (architecture §6 targets ≈10–14 GB peak). If your home GPU is smaller, develop with the
detector in half precision / smaller variants and accept slower local runs. *(The SoC cluster
fallback formerly noted here is NOT available — constraint confirmed 18 Jul 2026.)*

## 8a. Local LLM serving (dev bridge until API keys; see docs/local_llm_plan.md)

No-sudo user-space install (the systemd installer needs root; this doesn't):

```bash
mkdir -p ~/ollama && cd ~/ollama
curl -fsSL -O https://github.com/ollama/ollama/releases/latest/download/ollama-linux-amd64.tar.zst
tar --use-compress-program=unzstd -xf ollama-linux-amd64.tar.zst   # → bin/ lib/
# Serve with the models dir kept inside ~/ollama and an 8k context —
# REQUIRED: the CP2 4-tile call shape is ~4.2k tokens and 400s on the 4096 default.
OLLAMA_MODELS=~/ollama/models OLLAMA_CONTEXT_LENGTH=8192 OLLAMA_KEEP_ALIVE=30m \
  ~/ollama/bin/ollama serve &
~/ollama/bin/ollama pull qwen2.5vl:3b   # live-regime model (3.2 GB)
~/ollama/bin/ollama pull qwen2.5vl:7b   # offline-regime model (6.0 GB)
```

Wire into the module via the existing local provider slot (zero code changes):
`VLA_LLM_LOCAL_KIND=openai`, `VLA_LLM_LOCAL_BASE_URL=http://localhost:11434/v1`,
`VLA_LLM_LOCAL_MODEL=qwen2.5vl:3b`, and a dummy `VLA_LOCAL_API_KEY=local` (host
networking → localhost reachable from the containers; re-assert env in compose
like RMW, gotcha 13). Phase-0 latency table and model verdicts:
`docs/local_llm_plan.md` §"Phase 0 results".

**SDK requirement (Phase 1, 18 Jul):** `core/llm/providers.py` imports the `openai`
package lazily inside the call, so the fast test tier needs nothing installed — but
to actually DRIVE a configured `openai`-kind slot (local or a real cloud primary/
secondary) the SDK must be present: `pip install -e '.[llm]'` from `src/` (or
`pip install openai` directly). Conformance/battery tooling
(`tools/llm_conformance.py`, `tools/llm_parse_battery.py`) needs this installed.
Fixed in the ai_module image itself (issue #56, 19 Jul): the Dockerfile's constrained
pip layer now installs `openai>=1.0` alongside torch/transformers/groundingdino-py,
matching `src/pyproject.toml`'s `llm` extra — the baked `VLA_LLM_LOCAL_KIND=openai`
slot no longer hits `ProviderUnavailable (ModuleNotFoundError: openai)` at runtime.

**Submission requirement:** the host install above is the DEV loop only. The
submission image serves the model IN-CONTAINER (organisers run only our
container at eval) — Ollama binary + baked `qwen2.5vl:3b` blobs are baked into the
ai_module Dockerfile alongside the GDINO weights, with `ollama serve` +
pre-warm in the launch wrapper. **Authored 19 Jul (`b507eb9`), verification
pending the first post-reboot build** — spec: `docs/local_llm_plan.md` Phase 3;
Dockerfile/compose implications are reflected in §7a; ordered post-build
verification: `reports/local_llm_phase3/BUILD_AND_VERIFY.md`.

## 8b. Fast iteration: host-native ai_module node (containers at checkpoints only)

For adapter/perception iteration, run OUR node natively against the sim
container's graph (host networking + CycloneDDS make this transparent) instead
of rebuilding the image per change — seconds per cycle instead of minutes:

```bash
bash tools/setup_host_node.sh   # once; sudo-prompts for the ROS 2 Jazzy apt install
bash tools/run_host_node.sh     # per iteration, with the sim containers up
```

Rules that keep this safe:
- **The image is the truth.** The host venv mirrors the image pins
  (numpy==1.26.4, transformers==4.57.6 — single source of truth is the fork
  Dockerfile's constraint list; update BOTH when either changes) and the
  weights/HF cache are extracted FROM the built image, not re-downloaded.
- **Checkpoint discipline:** any gate exit, battery result you intend to
  trust, or submission MUST re-run in the container (sync_to_fork → compose
  build → §7a clean-clone gate). Host runs are scratch evidence only.
- Never run the host node and the containerised ai_module simultaneously
  (both would latch the question).

## 8c. Cluster detector/LLM offload (dev only)

The RTX 4060 Laptop has a hard power-management wedge (§1, issue #86): the GPU can
lock at P8/210 MHz the moment GroundingDINO loads its weights, and only a reboot
recovers it — which kills whatever live sim run was in progress. Issue #82 separately
found the local Ollama tier contending for the same VRAM. As an opt-in workaround,
the detector and the local-LLM tier can instead be served from the NUS SoC Slurm
cluster (`docs/soc_cluster_guide.md`) over SSH tunnels, while Unity + the ROS stack
keep running on this machine unchanged.

**This is a dev iteration tool only. Submission builds NEVER use this path** — the
scored image always runs the local, baked-in `GroundingDinoDetector` (§7) and the
in-image Ollama bake (§8a); nothing about the remote offload is part of the fork.

Workflow:

```bash
tools/cluster/servers.sh start          # rsyncs src/+tools/ to the cluster, sbatches the combined GPU job
tools/cluster/servers.sh status         # poll until both addr files exist on the cluster
tools/cluster/tunnel.sh                 # opens the SSH tunnels, prints the exports below
```

Paste the printed exports into the shell running the host node (`tools/run_host_node.sh`,
§8b) or the container env:

| Var | Value |
|---|---|
| `VLA_DETECTOR` | `remote` |
| `VLA_REMOTE_DETECTOR_URL` | `http://127.0.0.1:8765` |
| `VLA_REMOTE_DETECTOR_TIMEOUT_S` | `10.0` (default; override if the cluster is loaded) |
| `VLA_LLM_LOCAL_KIND` | `openai` |
| `VLA_LLM_LOCAL_BASE_URL` | `http://127.0.0.1:11434/v1` |
| `VLA_LLM_LOCAL_MODEL` | `qwen2.5vl:3b` (parity with the baked image default; `qwen2.5vl:7b` is also pulled on the cluster for A/B) |

No API key needed for the local slot — an empty key is fine against Ollama's openai
adapter.

**SoCLaaS primary tier (added 26 Jul 2026):** NUS SoC's free OpenAI-compatible
gateway (`https://soclaas-api.comp.nus.edu.sg/v1`; NUS network/VPN). The challenge
brief explicitly allows online LLM APIs at eval time. Issue a key with
`soclaas-portal issue` on any SoC host (interactive; stores `SOCLAAS_API_KEY` in
`~/.bashrc` — never commit it). Enable as the primary parse tier with:
`VLA_LLM_PRIMARY_KIND=openai`, `VLA_LLM_PRIMARY_BASE_URL=<gateway>/v1`,
`VLA_LLM_PRIMARY_MODEL=qwen3.6:35b`, `VLA_LLM_PRIMARY_API_KEY_ENV=SOCLAAS_API_KEY`.
The ladder falls through primary → local → regex when the gateway is unreachable,
so enabling it is safe everywhere (wired in tools/cluster/live_run/
cluster_verify_run.sbatch; model list per key via `GET /v1/models`).

**Perception dispatch (issue #88):** the adapter runs `PerceptionPipeline.process()`
on a dedicated worker thread by default (`core.perception.async_pipeline.
AsyncPerceptionWorker`, latest-frame-wins) so a slow detector forward (remote or
local GDINO) never stalls the 5 Hz tick/waypoint loop. Set `VLA_PERCEPTION_SYNC=1`
to force the old inline (synchronous) behaviour — debug only, reintroduces the
cadence stall; never set it for a live/eval run.

| Var | Value |
|---|---|
| `VLA_PERCEPTION_SYNC` | unset/`0` (default: threaded); `1` forces synchronous debug mode |
| `VLA_EXPLORE_DEBUG_DIR` | unset (default: no-op); dir for per-run explore/frontier JSONL dumps (#83) |
| `VLA_INSTANCE_DUMP_PATH` | unset (default: no-op); JSONL path for periodic + answer-time instance-index dumps (#84/#89) |
| `VLA_INSTANCE_DUMP_INTERVAL_S` | periodic dump throttle, default 10 |

**Addr-file handshake:** the sbatch job picks its own ports at start (shared GPU
nodes can already have something bound on 8765/11434 — the job probes upward for the
first free port) and writes the live `host:port` to `~/gdino_server.addr` and
`~/ollama_server.addr` on the cluster only after each server answers its own health
endpoint; both files are removed on job exit. `tunnel.sh` reads them over SSH before
opening the local port-forwards, so always run `servers.sh status` until both files
show up before `tunnel.sh` — an early tunnel attempt just fails cleanly, it doesn't
wedge anything.

Any network failure (cluster down, tunnel dropped, timeout) degrades `RemoteDetector`
to empty detections for that tick — it never crashes the adapter loop — but a dead
remote detector still starves the run of real boxes, so treat `servers.sh status` as
load-bearing before trusting a session's results.

**Stop the moment you're done** — `tools/cluster/servers.sh stop` (cancels only
`vla-servers` jobs; safe to run even if nothing is up). Fairshare bills the whole
allocation while it sits idle, not just active compute.

Override example for a longer session (defaults are `-p gpu -t 3:00:00`):

```bash
tools/cluster/servers.sh start -p gpu-long -t 8:00:00
```

**Measured latencies (26 Jul 2026, laptop -> xgpe5 Titan RTX via tunnel):**
`/health` RTT 101.5 ms warm (288.3 ms cold); one-tile `/detect` ("chair .",
812x451 render JPEG) 206.9 ms warm (398.3 ms cold); ollama `/v1/models`
69.8 ms. A full RemoteDetector dual-pass tick (question + vocab = two
POSTs) measured 641 ms, a question-only tick 203 ms — comfortably inside
the 10 s default timeout, but plan for ~2-5 detection ticks/s, not the
local GPU's rate.

## Troubleshooting quick refs

| Symptom | Fix |
|---|---|
| Containers can't see each other's topics | `RMW_IMPLEMENTATION` mismatch — set CycloneDDS in both (gotcha 13) |
| Robot ignores waypoints | Publishing `/way_point` directly instead of `/way_point_with_heading` (gotcha 11), or waypoint too far — keep ≤2.5 m (gotcha 14) |
| Unity scene black/no sensors | `Model.x86_64` not executable, or bridge port 10000 blocked |
| GPU absent in container | re-run `nvidia-ctk runtime configure`, restart docker; `--gpus all` on run |
| Marker scored 0 despite correct object | wrong frame (must be `map`) or extents not full-size (gotchas 5–6) |
