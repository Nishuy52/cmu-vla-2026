# docker/ai_module — our ai_module image

**UNTESTED DRAFT (Phase 2).** Authored on the Windows dev box (no ROS/Docker). Build and
validate on Ubuntu — see `docs/ubuntu_setup.md` §7a. All commands below are the intended
recipe; treat any surprise as a "confirm on Ubuntu" item.

This image replaces the upstream **dummy** `ai_module` (the `dummy_vlm` C++ node) with our
rclpy adapter (`src/ros_adapter/adapter_node.py`, package `vla_ai_module`) wrapping the
pure-Python `core` reasoning stack.

- Base image: `zhangjicmu/ubuntu24_ros:ai_module` (verbatim from upstream `docker/compose.yml`).
- Our built image tag: `iros2026/ai_module:latest` (matches the tag the upstream
  `ai_module/docker/build.sh` produces, so nothing downstream needs renaming).

## Build

Build context is the **repo root** (so `COPY src/` in the Dockerfile resolves):

```bash
cd <repo-root>
docker build -t iros2026/ai_module:latest -f docker/ai_module/Dockerfile .
```

## Tag & push (Docker Hub — submission flow)

The challenge is submitted as a public GitHub repo (fork of upstream, `ai_module/` changed) +,
if the image changed, a pushed Docker Hub image whose link is submitted
(upstream README:154; `docs/upstream_notes.md` §5). Tag to your Docker Hub namespace and push:

```bash
docker login
docker tag iros2026/ai_module:latest <dockerhub-user>/cmu-vla-ai-module:latest
docker push <dockerhub-user>/cmu-vla-ai-module:latest
```

Keep the image within the eval host's size budget (a Simply NUC i9 — `docs/upstream_notes.md`
§5). The commented `torch`/`groundingdino` layer in the Dockerfile is the big one; only
uncomment it once the weights decision is made and pre-downloaded (`docs/ubuntu_setup.md` §7).

## Swap the dummy for our node in the compose stack

Upstream `docker/compose.yml` builds the `ai_module` service `FROM ../ai_module` using
`ai_module/docker/Dockerfile` and (per `docker/README.md`) launches
`ros2 launch dummy_vlm dummy_vlm.launch`. To run **our** node instead, point that service at
this Dockerfile and our launch. Two options:

**A. Edit the upstream compose `ai_module` service** (submission path — `ai_module/` is the
only editable tree, so for the real submission our files live under the fork's `ai_module/`):

```yaml
  ai_module:
    build:
      context: ..                       # repo root, so COPY src/ resolves
      dockerfile: docker/ai_module/Dockerfile
    container_name: iros2026_ai_module
    network_mode: host
    environment:
      - RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
    env_file:
      - ../.env.llm                     # optional, gitignored: OPENAI_API_KEY=... etc.
    command: >
      bash -lc "source /opt/vla/ws/install/setup.bash &&
                ros2 launch vla_ai_module ai_module.launch.py"
    stdin_open: true
    tty: true
```

**B. Prebuilt image** — replace `build:` with `image: <dockerhub-user>/cmu-vla-ai-module:latest`
and keep the same `command:` / `env_file:` / `environment:`.

The `system` service is unchanged (`docs/upstream_notes.md` gotcha 12: never touch the system
container). Bring the stack up exactly as before:

```bash
xhost +
cd <upstream>/docker && docker compose -f compose.yml up --build -d
```

## LLM keys — never baked

Keys are passed at run time only, never into the image. Provide them via `env_file`
(shown above) or `environment:`. `core/llm/config.py` reads `VLA_LLM_PRIMARY_*` /
`VLA_LLM_SECONDARY_*` / `VLA_LLM_LOCAL_*` (each names the env var that holds its key, e.g.
`OPENAI_API_KEY`), plus `VLA_LLM_CALL_TIMEOUT_S` and `VLA_LLM_CONFIG`. With nothing set the
ladder falls back to the local/regex path (fully offline). A keyless `llm_config.json` at the
repo root can carry non-secret defaults (kinds/models/base_urls) — commit only the indirection.

## First smoke test

After the stack is up, run the ordered Tier-2 checks in `docs/sim_verification.md` — in
particular **Tier 2.7** (our-module round-trip: publish a question, confirm the node latches
it and emits a legal answer on the matching topic). See `docs/ubuntu_setup.md` §7a.
