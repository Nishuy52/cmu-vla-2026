# VLA ai_module image — our rclpy adapter + pure-Python core, replacing the dummy_vlm node.
#
# FORK-SHAPED (H7). This file is the exact artifact that is dropped into the challenge
# fork at  ai_module/docker/Dockerfile , built with build context = ai_module/  — the
# same shape the upstream compose uses for the dummy service
# (docs/upstream_notes.md §5 "B) ai_module/docker/": `build.sh` -> `docker build -f
# docker/Dockerfile .` with context ai_module/, so `COPY src/...` resolves under ai_module/).
# Submission rules allow modifying ONLY ai_module/ in the fork (upstream_notes.md gotcha 12),
# so the Dockerfile and everything it COPYs must live inside ai_module/ and reference nothing
# above it. That is why the context is ai_module/ (NOT the repo root) and every COPY path is
# relative to ai_module/.
#
# In OUR repo this file is staged under docker/ai_module_fork/ ; our repo's src/ is synced
# into the fork's ai_module/src/ by docker/ai_module_fork/sync_to_fork.{ps1,sh} before build.
# See docker/ai_module_fork/README.md.
#
# UNTESTED DRAFT (Phase 2), authored on Windows. Build/validate on Ubuntu per
# docs/ubuntu_setup.md §7a. Base image tag is taken verbatim from upstream docker/compose.yml
# (service ai_module builds FROM this) — docs/upstream_notes.md §5.
FROM zhangjicmu/ubuntu24_ros:ai_module

# --- Middleware: both containers MUST agree on CycloneDDS or topic discovery silently fails
#     (docs/upstream_notes.md gotcha 13; upstream notes.txt). Baked here so the node inherits it.
#     WARNING: a compose `environment:` (or `env_file:`) entry for RMW_IMPLEMENTATION OVERRIDES
#     this ENV at run time. Our compose snippet re-asserts rmw_cyclonedds_cpp for exactly this
#     reason (docker/ai_module_fork/README.md); if anyone sets a different RMW there, this line
#     is silently ignored and discovery fails the same way F2 does (no topics, no question).
#     confirm on Ubuntu (flag 1 of 3): topics discover across the two containers with this RMW.
ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

# --- Our workspace lives under /opt/vla. src/ carries both `core` (pure Python) and
#     `ros_adapter` (the rclpy node). Put src/ on PYTHONPATH so `import core` / `import
#     ros_adapter` resolve without a pip install of the workspace.
ENV VLA_ROOT=/opt/vla
ENV PYTHONPATH=/opt/vla/src:${PYTHONPATH}

# --- Python deps for the core + adapter. rclpy and the ROS message packages come from the
#     base image; these are the pure-Python extras core needs.
#
#     PEP 668 (--break-system-packages): the Noble (Ubuntu 24.04) base marks the system Python
#     as "externally managed", so a bare `pip install` FAILS with
#     "error: externally-managed-environment" unless the base image pre-provisioned a venv.
#     We deliberately install into the system interpreter with --break-system-packages rather
#     than creating a venv, because:
#       (a) the base image's rclpy / ROS message packages live on the SYSTEM Python; a venv
#           would either lose them (no --system-site-packages) or fork the environment in a way
#           that complicates `ros2 run` / colcon's Python resolution — more moving parts on the
#           riskiest gate, not fewer;
#       (b) the container is single-purpose and disposable (fresh process per launch, gotcha 1),
#           so "polluting" the system site-packages costs nothing here;
#       (c) it keeps `import core` working from PYTHONPATH without a venv activation step in CMD.
#     confirm on Ubuntu (flag 2 of 3): if the base image already exposes a writable venv on PATH
#     (e.g. /opt/ros brings one), --break-system-packages is a harmless no-op and can be dropped;
#     if `pip install` succeeds WITHOUT the flag, the base is not externally-managed and the flag
#     is unnecessary. Left in as the defensive default so the build does not die at this layer.
#
#     torch + groundingdino are the perception stack — heavy. Gate 4 (docs/ubuntu_setup.md §7
#     step 3): installed below with default CUDA (cu12x) wheels, weights baked in the layer that
#     follows so nothing is fetched at run time.
#
#     CONFIRMED on Ubuntu (flag 2 of 3): the base image's default runtime user is the non-root
#     `docker` user, and it has NO pip binary at all (`pip`/`pip3`: command not found — the base
#     image never installs python3-pip, not even a PEP-668-restricted one). `apt-get install
#     python3-pip` requires root, so we switch to `USER root` for the install layers (the `docker`
#     user has passwordless sudo, but `USER root` is simpler for RUN layers) and switch back to
#     `USER docker` below before CMD so the runtime user matches the base image's own default.
#     Once python3-pip is present, `--break-system-packages` IS required and is NOT a no-op:
#     tested `pip install numpy` (as both root and the `docker` user) without the flag and both
#     fail with "error: externally-managed-environment" (PEP 668, Noble marks the system
#     interpreter externally-managed regardless of caller uid). Keep the flag.
USER root
RUN apt-get update -qq && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends python3-pip curl && \
    rm -rf /var/lib/apt/lists/*
#     Second CONFIRMED-on-Ubuntu finding: `pytest>=8` needs `pluggy>=1.5`, but the base image
#     already has `python3-pluggy` 1.4.0 installed via apt/dpkg (pulled in automatically by
#     ros-jazzy-ament-cmake-pytest -> python3-pytest 7.4.4). A single `pip install numpy rosbags
#     pytest` tries to upgrade that dpkg-owned pluggy in place and fails with "Cannot uninstall
#     pluggy 1.4.0, RECORD file not found" (pip refuses to remove a package it did not install).
#     A blanket `--ignore-installed` "fixes" that but also forces numpy to be reinstalled/upgraded
#     to the latest (2.x), which breaks the apt-provided scipy (`numpy<1.28.0,>=1.21.6` — used by
#     other ROS packages in the base image). Split into two invocations instead: numpy/rosbags
#     first (numpy>=1.26 is already satisfied by apt's 1.26.4, so this layer never touches it),
#     then pytest on its own with `--ignore-installed` scoped to just pytest's dependency graph
#     (pygments/packaging/iniconfig/pluggy) — numpy is not in that graph so it stays untouched.
RUN pip install --no-cache-dir --break-system-packages \
      "numpy>=1.26" \
      "rosbags>=0.11"
RUN pip install --no-cache-dir --break-system-packages --ignore-installed \
      "pytest>=8"
#     Gate 4: uncommented. Default PyPI wheels for torch/torchvision on linux are CUDA-enabled
#     (cu12x) and cover both the 4090 eval box and an 8 GB 4060 dev box, so no --index-url/CUDA
#     pin is needed. `transformers` is already a transitive dep of groundingdino-py
#     (requirements.txt: torch/torchvision/transformers/addict/yapf/timm/numpy/opencv-python/
#     supervision==0.6.0/pycocotools) but is named explicitly here because the very next layer
#     imports it directly to pre-fetch BERT.
#
#     Third CONFIRMED-on-Ubuntu finding (Gate 4 build, 18 Jul): same RECORD-file failure class as
#     the pluggy note above, this time via numpy. groundingdino-py pulls opencv-python, and
#     opencv-python >=4.12 requires numpy>=2 on py3.12 — so an unconstrained install tries to
#     upgrade the dpkg-owned numpy 1.26.4 in place and dies with "Cannot uninstall numpy 1.26.4,
#     RECORD file not found" (and even if it could, numpy 2.x breaks the apt scipy pinned
#     numpy<1.28 — see the pluggy note). Fix: a PIP_CONSTRAINT pinning numpy==1.26.4, which makes
#     the resolver keep the apt numpy untouched and backtrack opencv-python to 4.11.0.86 (verified
#     on PyPI: its py3.12 requirement is numpy>=1.26.0, satisfied by 1.26.4; pycocotools and
#     supervision==0.6.0 have no numpy upper pin; torch/transformers/timm don't pin numpy>=2).
#     Gate 4 build (18 Jul), issue #38: the unpinned `transformers` line above resolved a
#     version that dropped `BertModel.get_head_mask`, which groundingdino-py's vendored
#     BertModelWarper still calls -> AttributeError on every detector load. Pin
#     transformers==4.57.6 (verified on Ubuntu to still expose get_head_mask and load
#     cleanly against groundingdino-py 0.4.0) to keep the resolver off the broken range.
RUN printf 'numpy==1.26.4\ntransformers==4.57.6\n' > /tmp/pip-constraints.txt && \
    PIP_CONSTRAINT=/tmp/pip-constraints.txt pip install --no-cache-dir --break-system-packages \
      torch \
      torchvision \
      transformers==4.57.6 \
      groundingdino-py && \
    rm /tmp/pip-constraints.txt

# --- Weight baking (Gate 4 item 1: "bake into the image (offline-capable rule)").
#
#     Investigated groundingdino-py==0.4.0 (the giswqs/GroundingDINO fork on PyPI — the exact
#     package GDINO_REQUIRED_INSTALLS names) to find out what `groundingdino.util.inference.
#     load_model()` (the function GroundingDinoDetector._lazy_import pulls in) actually needs on
#     disk, since nothing about weights is obvious from the import alone:
#       1. `load_model(config_path, checkpoint_path, device)` takes LOCAL file paths for both a
#          plain-Python model config and a finetuned .pth checkpoint. The configs for both size
#          variants SHIP INSIDE THE PIP PACKAGE already (site-packages/groundingdino/config/
#          GroundingDINO_SwinT_OGC.py and GroundingDINO_SwinB_cfg.py) — nothing to bake there.
#          The checkpoint is NOT bundled and NOT auto-downloaded by the package or by load_model;
#          it only exists as a GitHub Release asset. GDINO_MODEL_ID in core/perception/detector.py
#          is "IDEA-Research/grounding-dino-base" (HF-hub-style "base" size) -> the Swin-B variant,
#          config GroundingDINO_SwinB_cfg.py + checkpoint groundingdino_swinb_cogcoor.pth
#          (confirmed reachable at the URL below, 938_057_991 bytes, release v0.1.0-alpha2; the
#          SwinT asset 404s under that same tag, SwinB is the one that's actually current there).
#       2. Both configs set text_encoder_type = "bert-base-uncased" — real BERT, not a
#          GroundingDINO-specific weight. groundingdino/util/get_tokenlizer.py calls
#          `AutoTokenizer.from_pretrained(...)` and `BertModel.from_pretrained(...)` at model
#          construction time, i.e. a genuine HuggingFace Hub fetch, cached under HF_HOME. (The
#          Swin backbone itself has NO such fetch: build_swin_transformer() only random-inits the
#          module — the finetuned .pth's state_dict overwrites backbone + transformer + BERT
#          projection in one `load_state_dict` call, so there is no separate ImageNet-pretrained
#          backbone download to bake for the Swin-B path used here.)
#     So two different baking mechanisms, both landing under /opt/vla so the runtime user can read
#     them: a plain file download for the checkpoint, and transformers' own from_pretrained() (not
#     a hand-rolled snapshot_download) for BERT, so the on-disk cache layout is exactly what the
#     real runtime from_pretrained() call will resolve.
#     Single RUN on purpose: fetches + chown MUST share one layer. A separate `RUN chown -R`
#     after the downloads re-copies every weight file into a fresh overlay layer (chown = metadata
#     change = whole-file copy-up), which was measured to duplicate 1.38GB in the first Gate-4
#     build. Fused like this the weights exist in exactly one layer, already docker-owned.
ENV HF_HOME=/opt/vla/hf_cache
ENV GDINO_CHECKPOINT_PATH=/opt/vla/weights/groundingdino/groundingdino_swinb_cogcoor.pth
#     GDINO_CONFIG_PATH must be baked too (detector.py reads ENV_GDINO_CONFIG_PATH): its
#     _resolve_config_path() fallback when the env var is unset is the package-bundled
#     GroundingDINO_SwinT_OGC.py — the SwinT config — which state-dict-mismatches the Swin-B
#     checkpoint above at load time. Copy the package's SwinB config to a stable path next to the
#     checkpoint (resolved from the installed package at build time, not a hardcoded
#     site-packages path) and point the env var at it.
ENV GDINO_CONFIG_PATH=/opt/vla/weights/groundingdino/GroundingDINO_SwinB_cfg.py
RUN mkdir -p "${HF_HOME}" "$(dirname "${GDINO_CHECKPOINT_PATH}")" && \
    python3 -c "from transformers import AutoTokenizer, BertModel; AutoTokenizer.from_pretrained('bert-base-uncased'); BertModel.from_pretrained('bert-base-uncased')" && \
    python3 -c "import groundingdino, os, shutil; src = os.path.join(os.path.dirname(groundingdino.__file__), 'config', 'GroundingDINO_SwinB_cfg.py'); shutil.copyfile(src, os.environ['GDINO_CONFIG_PATH'])" && \
    curl -fSL --retry 3 -o "${GDINO_CHECKPOINT_PATH}" "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth" && \
    chown -R docker:docker /opt/vla/hf_cache /opt/vla/weights
# Build-time assertion (no GPU, no model construction — plain file checks) that both baked
# artifacts actually landed before the layer is allowed to succeed.
#     The config check asserts on the backbone marker line, the field that actually
#     distinguishes the two bundled configs (SwinB: backbone = "swin_B_384_22k"; SwinT:
#     backbone = "swin_T_224_1k") — guarding against the SwinT-config/SwinB-weights mismatch.
RUN python3 -c "\
import glob, os; \
ckpt = os.environ['GDINO_CHECKPOINT_PATH']; \
ckpt_size = os.path.getsize(ckpt); \
assert ckpt_size > 900_000_000, f'groundingdino checkpoint too small: {ckpt_size} bytes at {ckpt}'; \
cfg = os.environ['GDINO_CONFIG_PATH']; \
assert os.path.isfile(cfg), f'missing baked gdino config at {cfg}'; \
cfg_text = open(cfg).read(); \
assert 'swin_B_384_22k' in cfg_text, f'{cfg} is not the Swin-B config (no swin_B_384_22k backbone marker) — SwinT config + SwinB checkpoint mismatches at load'; \
hub = os.path.join(os.environ['HF_HOME'], 'hub'); \
snaps = glob.glob(os.path.join(hub, 'models--bert-base-uncased', 'snapshots', '*')); \
assert snaps, f'no bert-base-uncased snapshot under {hub}'; \
snap = snaps[0]; \
assert os.path.isfile(os.path.join(snap, 'config.json')), f'missing config.json in {snap}'; \
assert os.path.isfile(os.path.join(snap, 'vocab.txt')), f'missing vocab.txt in {snap}'; \
weight_files = glob.glob(os.path.join(snap, '*.safetensors')) + glob.glob(os.path.join(snap, 'pytorch_model.bin')); \
assert weight_files, f'no bert weight file under {snap}'; \
print(f'weights OK: gdino checkpoint {ckpt_size} bytes; swinB config at {cfg}; bert snapshot {snap} weight file(s) {weight_files}')"
# Runtime offline guard: force transformers/huggingface_hub to resolve ONLY from the HF_HOME cache
# baked above and never attempt a network call at run time (docs/ubuntu_setup.md §7 item 3 — the
# eval host is assumed offline-capable). Set AFTER the prefetch layers above, which still need the
# network. If the cache is ever incomplete this turns a would-be silent hang into a loud
# "not found locally" error instead.
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

# --- Copy the workspace source. Build context is ai_module/ (the fork tree; see header), so
#     `src/` here resolves to ai_module/src/ — our repo's src/ synced in by sync_to_fork.*.
#     Tests are NOT synced into the fork (see README rationale), so `src/` is core + ros_adapter.
#     --chown keeps the tree owned by the `docker` runtime user (matches the base image's own
#     ownership model) rather than leaving it root-owned from the `USER root` block above.
COPY --chown=docker:docker src/ /opt/vla/src/

# --- Build the ament_python package so `ros2 launch vla_ai_module ...` resolves. The package
#     manifest lives in src/ros_adapter/. We build it into a colcon workspace at /opt/vla/ws.
#     CONFIRMED on Ubuntu (flag 3 of 3): the colcon symlink layout below works as drafted —
#     `colcon build --packages-select vla_ai_module` finds src/ros_adapter/package.xml through
#     the symlink and builds cleanly; colcon does NOT try to build `core` (it is plain
#     PYTHONPATH, not discovered as an ament package, exactly as intended) and does not complain.
#     ament (install/setup.bash) and `core` on PYTHONPATH coexist without conflict.
WORKDIR /opt/vla/ws
RUN mkdir -p src && ln -s /opt/vla/src/ros_adapter src/ros_adapter && \
    /bin/bash -c "source /opt/ros/jazzy/setup.bash && \
      colcon build --packages-select vla_ai_module" && \
    chown -R docker:docker /opt/vla/ws
# NOTE: base image's `docker` user gets RMW_IMPLEMENTATION + ROS sourcing from
# /home/docker/.bashrc (docs/ubuntu_setup.md gotcha 13's "only in ~/.bashrc" trap applies to the
# `docker` user specifically, not root) — append our workspace overlay to the SAME file. The
# original draft appended to /root/.bashrc, which the non-root runtime user never sources.
RUN echo "source /opt/vla/ws/install/setup.bash" >> /home/docker/.bashrc

# --- Local LLM serving (Phase 3, docs/local_llm_plan.md "Phase 3 — live integration"). Bakes
#     an Ollama standalone server + the qwen2.5vl:3b blobs into the image, mirroring the GDINO
#     weights-bake pattern above: fetched at build time, offline at runtime, build-time
#     assertions guard the bake. This is the SUBMISSION path — the eval host runs ONLY our
#     container, so the model must be served IN-container; the host Ollama install
#     (docs/ubuntu_setup.md §8a) remains dev-loop convenience only and is not used at eval.
#
#     PINNED VERSION (do not float to "latest"): the linux/amd64 release asset format changed
#     to .tar.zst as of v0.32 (docs/local_llm_plan.md "Phase 0 results" confirms this for
#     v0.32.1 on this box) — pinning avoids a rebuild silently picking up a differently-named or
#     differently-compressed asset under the "latest" alias. zstd is not in the base image (the
#     GDINO layers above only needed curl) so it is apt-installed here.
ENV OLLAMA_VERSION=v0.32.1
RUN apt-get update -qq && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends zstd && \
    rm -rf /var/lib/apt/lists/*
#     Extract + prune non-CUDA runner variants in one fused layer (chown included, same
#     copy-up-avoidance reasoning as the GDINO weights RUN above). The tarball ships
#     lib/ollama/<backend> subtrees for every accelerator Ollama supports (rocm, vulkan, a
#     cpu fallback, and one or more cuda_v* runners); we only need the CUDA runner that matches
#     the driver on both the dev 4060 and the eval 4090 (Phase-0 measured cuda_v13 selected
#     against driver 13.2) plus the cpu fallback (so the container degrades instead of
#     hard-failing to start if a GPU is ever unavailable at runtime). Everything else is deleted.
#     confirm on Ubuntu (flag 4 of 4, Phase 3): this build cannot run here (GPU wedged pending
#     reboot, per session constraint) — the `find`/`ls` line below prints the actual extracted
#     tree to the build log specifically so the first post-reboot build can eyeball it BEFORE
#     trusting the prune glob; if the runner directory names differ from cuda_v13/cpu*, fix the
#     `case` pattern before relying on this layer.
RUN mkdir -p /opt/vla/ollama && \
    curl -fSL --retry 3 -o /tmp/ollama.tar.zst \
      "https://github.com/ollama/ollama/releases/download/${OLLAMA_VERSION}/ollama-linux-amd64.tar.zst" && \
    tar --use-compress-program=unzstd -xf /tmp/ollama.tar.zst -C /opt/vla/ollama && \
    rm /tmp/ollama.tar.zst && \
    echo "=== extracted ollama tree (confirm-on-Ubuntu: verify before trusting the prune) ===" && \
    find /opt/vla/ollama -maxdepth 3 | sort && \
    if [ -d /opt/vla/ollama/lib/ollama ]; then \
      cd /opt/vla/ollama/lib/ollama && \
      for d in */; do \
        d="${d%/}"; \
        case "$d" in \
          cuda_v13*|cpu*) echo "keeping runner: $d" ;; \
          *) echo "pruning runner: $d"; rm -rf "$d" ;; \
        esac; \
      done; \
    else \
      echo "NOTE: no lib/ollama runner dir found at the expected path — layout may differ for ${OLLAMA_VERSION}, nothing pruned (confirm on Ubuntu)"; \
    fi && \
    chown -R docker:docker /opt/vla/ollama
RUN test -x /opt/vla/ollama/bin/ollama || (echo "ollama binary missing after extract at /opt/vla/ollama/bin/ollama" && exit 1)
#     Model blobs. Pattern per Phase 3 spec: start `ollama serve` backgrounded against the baked
#     OLLAMA_MODELS dir, wait for it to come up, `ollama pull` the model, then stop the server —
#     leaves the manifest+blobs on disk for the runtime server to load with no network call.
#     Deviation from the literal `kill %1` shorthand in the Phase 3 spec: `%1` is a bash
#     "current job" job-control reference, and non-interactive `RUN` shells (Docker uses
#     `/bin/sh -c`, i.e. dash — no job control regardless) don't reliably support it. Instead:
#     background `ollama serve` inside a `( … & echo $! > pidfile )` subshell so the PID is
#     captured reliably without depending on how the outer `&&`/`&` chain parses (a bare
#     `cmd1 && cmd2 &` backgrounds the WHOLE and-list as one job, not just cmd2 — `$!` right
#     after that would not reliably be ollama's own PID), then `ollama pull`, capture its exit
#     status, kill+wait the captured PID, chown, and re-raise the pull's exit status so a failed
#     pull still fails the build instead of silently leaving no model baked. Flag this whole
#     block as confirm-on-Ubuntu along with the rest.
ENV OLLAMA_MODELS=/opt/vla/ollama_models
ENV OLLAMA_CONTEXT_LENGTH=8192
ENV OLLAMA_HOST=127.0.0.1:11434
ENV PATH=/opt/vla/ollama/bin:${PATH}
RUN mkdir -p "${OLLAMA_MODELS}" && \
    ( ollama serve & echo $! > /tmp/ollama.pid ) && \
    sleep 5 && \
    ollama pull qwen2.5vl:3b; \
    PULL_STATUS=$?; \
    OLLAMA_PID="$(cat /tmp/ollama.pid)"; \
    kill "${OLLAMA_PID}" 2>/dev/null; \
    wait "${OLLAMA_PID}" 2>/dev/null; \
    rm -f /tmp/ollama.pid; \
    chown -R docker:docker "${OLLAMA_MODELS}"; \
    exit "${PULL_STATUS}"
# Build-time assertion (plain file/size checks, no GPU, no server) that the model actually landed
# — same "fail loud at build time, not silently at runtime" discipline as the GDINO checks above.
# confirm on Ubuntu: the manifest path below follows Ollama's documented on-disk layout
# (manifests/registry.ollama.ai/library/<model>/<tag>); verify against the pinned v0.32.1 tree.
RUN python3 -c "\
import os; \
models_dir = os.environ['OLLAMA_MODELS']; \
manifest = os.path.join(models_dir, 'manifests', 'registry.ollama.ai', 'library', 'qwen2.5vl', '3b'); \
assert os.path.isfile(manifest), f'missing ollama manifest at {manifest} — confirm-on-Ubuntu: check the actual on-disk layout under {models_dir}'; \
blobs_dir = os.path.join(models_dir, 'blobs'); \
assert os.path.isdir(blobs_dir), f'missing blobs dir at {blobs_dir}'; \
total = sum(os.path.getsize(os.path.join(blobs_dir, f)) for f in os.listdir(blobs_dir) if os.path.isfile(os.path.join(blobs_dir, f))); \
gb = total / (1024**3); \
assert 2.5 <= gb <= 4.0, f'qwen2.5vl:3b blob total {gb:.2f} GB outside sane range [2.5, 4.0] GB'; \
print(f'ollama model OK: manifest at {manifest}; blobs {gb:.2f} GB')"
# Long default keep_alive (baked default; the eval 4090 has headroom to hold the model
# resident) — the dev box override (short keep_alive, since GDINO + Unity + the 3B model
# together are tight on 8 GB VRAM) is set in compose, not here; see
# docker/ai_module_fork/docker/compose.yml and docs/local_llm_plan.md Phase 3 "VRAM note".
ENV OLLAMA_KEEP_ALIVE=60m
# Local-slot wiring for core/llm/config.py (src/core/llm/config.py's VLA_LLM_LOCAL_* indirection,
# docs/ubuntu_setup.md §8a): baked so the local tier works out of the box with no compose env
# needed, matching "offline at runtime" — compose may still override VLA_LLM_LOCAL_MODEL etc. if
# a different local model is ever swapped in. VLA_LOCAL_API_KEY is a dummy value (Ollama's
# OpenAI-compatible endpoint does not check it) — never a real credential.
ENV VLA_LLM_LOCAL_KIND=openai
ENV VLA_LLM_LOCAL_BASE_URL=http://127.0.0.1:11434/v1
ENV VLA_LLM_LOCAL_MODEL=qwen2.5vl:3b
ENV VLA_LOCAL_API_KEY=local-dummy-key
USER docker

# --- LLM parse-provider keys are NEVER baked into the image. They are passed at run time via
#     the environment (compose `env_file:` or `environment:`). core/llm/config.py reads:
#       VLA_LLM_PRIMARY_KIND / _MODEL / _BASE_URL / _API_KEY_ENV   (+ the named key var,
#         e.g. OPENAI_API_KEY)
#       VLA_LLM_SECONDARY_* (default key var ANTHROPIC_API_KEY)
#       VLA_LLM_LOCAL_*     (default key var VLA_LOCAL_API_KEY)
#       VLA_LLM_CALL_TIMEOUT_S, VLA_LLM_CONFIG
#     Unset => the ladder runs local/regex only (no network). To wire keys, add to the
#     compose ai_module service:
#         env_file: [../.env.llm]        # gitignored; holds OPENAI_API_KEY=... etc.
#     or a keyless llm_config.json at the repo root (indirection only; no keys in the file).
#     See docker/ai_module_fork/README.md.

# --- Launch wrapper (Phase 3): starts the baked ollama serve, pre-warms qwen2.5vl:3b bounded to
#     ~30s, then execs the same overlay-source + ros2-launch tail the old CMD used directly (see
#     docker/ai_module_fork/ai_module/launch_with_llm.sh for the full contract). Build context is
#     ai_module/ (see file header), so this resolves to ai_module/launch_with_llm.sh — synced in
#     by sync_to_fork.sh alongside src/ and docker/Dockerfile.
COPY --chown=docker:docker launch_with_llm.sh /opt/vla/launch_with_llm.sh
RUN chmod +x /opt/vla/launch_with_llm.sh

WORKDIR /opt/vla
# Entry: run the LLM launch wrapper, which execs the overlay-source + ros2-launch tail once
# ollama is up (or the warm-up budget expires — see launch_with_llm.sh). compose overrides this
# with the same command (see README.md); kept here so `docker run` alone brings the node up.
CMD ["/bin/bash", "-lc", "/opt/vla/launch_with_llm.sh"]
