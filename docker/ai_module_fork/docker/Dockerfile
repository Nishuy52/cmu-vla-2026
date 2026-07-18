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
#     torch + groundingdino are the perception stack — heavy, and gated on the weights decision.
#     Phase-2: uncomment after weights decision (docs/ubuntu_setup.md §7 step 3; use CUDA wheels
#     for the 4090 eval box and pre-download the GroundingDINO weights so nothing is fetched at
#     run time).
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
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends python3-pip && \
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
# Phase-2: uncomment after weights decision --
# RUN pip install --no-cache-dir --break-system-packages torch torchvision groundingdino-py

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
