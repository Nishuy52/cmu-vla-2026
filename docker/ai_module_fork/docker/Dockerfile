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

WORKDIR /opt/vla
# Entry: source the overlay then launch. compose overrides this with the launch command
# (see README.md); kept here so `docker run` alone brings the node up.
CMD ["/bin/bash", "-lc", "source /opt/vla/ws/install/setup.bash && ros2 launch vla_ai_module ai_module.launch.py"]
