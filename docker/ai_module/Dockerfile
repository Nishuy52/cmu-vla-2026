# VLA ai_module image — our rclpy adapter + pure-Python core, replacing the dummy_vlm node.
#
# ============================================================================================
# DEPRECATED (H7 / F4). DO NOT BUILD FROM THIS FILE.
#
# This Dockerfile uses a REPO-ROOT build context (`COPY src/` resolves against the repo root).
# That shape CANNOT slot into the submission fork, where only ai_module/ may be modified and the
# upstream compose builds the ai_module service from context ai_module/ using
# ai_module/docker/Dockerfile (docs/upstream_notes.md §5B, gotcha 12).
#
# The fork-shaped replacement lives at:  docker/ai_module_fork/  (Dockerfile + sync scripts +
# README). Use that. This file is kept ONLY as a historical record of the repo-root approach.
# ============================================================================================
#
# UNTESTED DRAFT (Phase 2), authored on Windows. Build/validate on Ubuntu per
# docs/ubuntu_setup.md §7a. Base image tag is taken verbatim from upstream docker/compose.yml
# (service ai_module builds FROM this) — docs/upstream_notes.md §5.
FROM zhangjicmu/ubuntu24_ros:ai_module

# --- Middleware: both containers MUST agree on CycloneDDS or topic discovery silently fails
#     (docs/upstream_notes.md gotcha 13; upstream notes.txt). Baked here so the node inherits it.
ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

# --- Our workspace lives under /opt/vla. src/ carries both `core` (pure Python) and
#     `ros_adapter` (the rclpy node). Put src/ on PYTHONPATH so `import core` / `import
#     ros_adapter` resolve without a pip install of the workspace.
ENV VLA_ROOT=/opt/vla
ENV PYTHONPATH=/opt/vla/src:${PYTHONPATH}

# --- Python deps for the core + adapter. rclpy and the ROS message packages come from the
#     base image; these are the pure-Python extras core needs.
#     torch + groundingdino are the perception stack — heavy, and gated on the weights
#     decision. Phase-2: uncomment after weights decision (docs/ubuntu_setup.md §7 step 3;
#     use CUDA wheels for the 4090 eval box and pre-download the GroundingDINO weights so
#     nothing is fetched at run time).
RUN pip install --no-cache-dir \
      "numpy>=1.26" \
      "rosbags>=0.11" \
      "pytest>=8"
# Phase-2: uncomment after weights decision --
# RUN pip install --no-cache-dir torch torchvision groundingdino-py

# --- Copy the workspace source. Build context is the repo root (see docker/ai_module/README.md),
#     so these paths resolve. Copy src/ and the ament package files.
COPY src/ /opt/vla/src/

# --- Build the ament_python package so `ros2 launch vla_ai_module ...` resolves. The package
#     manifest lives in src/ros_adapter/. We build it into a colcon workspace at /opt/vla/ws.
#     confirm on Ubuntu: colcon layout below assumes src/ros_adapter/package.xml is discoverable;
#     if colcon complains about `core` not being a ROS package, that is expected — core is a
#     plain pip/PYTHONPATH module, not an ament package, and is intentionally NOT built here.
WORKDIR /opt/vla/ws
RUN mkdir -p src && ln -s /opt/vla/src/ros_adapter src/ros_adapter && \
    /bin/bash -c "source /opt/ros/jazzy/setup.bash && \
      colcon build --packages-select vla_ai_module"
RUN echo "source /opt/vla/ws/install/setup.bash" >> /root/.bashrc

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
#     See docker/ai_module/README.md.

WORKDIR /opt/vla
# Entry: source the overlay then launch. compose overrides this with the launch command
# (see README.md); kept here so `docker run` alone brings the node up.
CMD ["/bin/bash", "-lc", "source /opt/vla/ws/install/setup.bash && ros2 launch vla_ai_module ai_module.launch.py"]
