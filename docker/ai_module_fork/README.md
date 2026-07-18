# docker/ai_module_fork — fork-shaped ai_module packaging (H7)

**UNTESTED DRAFT (Phase 2).** Authored on the Windows dev box (no ROS/Docker). Build and
validate on Ubuntu — see `docs/ubuntu_setup.md` §7a. All commands below are the intended
recipe; treat any surprise as a "confirm on Ubuntu" item.

## Why this exists (the F4/H7 defect)

The old `docker/ai_module/Dockerfile` was written for a **repo-root build context** (`COPY src/`
resolving against the repo root). But the challenge is submitted as a **fork** in which only the
`ai_module/` directory may be modified (`docs/upstream_notes.md` gotcha 12, §5B), and the upstream
compose builds the `ai_module` service **from context `../ai_module` using `ai_module/docker/Dockerfile`**
(`docs/upstream_notes.md` §5B — the dummy's `build.sh` runs `docker build -f docker/Dockerfile .`
with context `ai_module/`). A repo-root-context Dockerfile living at `docker/ai_module/` in our
repo **cannot slot into that shape** — the submitted fork would carry a Dockerfile that references
paths above `ai_module/`, and the evaluators' `docker compose up --build` would fail.

This directory holds the **fork-shaped** artifact: a Dockerfile written for an `ai_module/` build
context, plus scripts that stage our payload into a fork checkout in the exact required layout.

## The fork layout this produces

After running the sync script against a fork checkout, the fork's `ai_module/` looks like:

```
<fork>/
  ai_module/
    docker/
      Dockerfile        <- docker/ai_module_fork/docker/Dockerfile (this repo), context = ai_module/
    launch_with_llm.sh  <- docker/ai_module_fork/ai_module/launch_with_llm.sh (this repo)
    src/
      core/             <- src/core/        (this repo)
      ros_adapter/      <- src/ros_adapter/ (this repo)   (ament pkg: vla_ai_module)
    ...                 (upstream's other ai_module/ files, untouched)
  docker/
    compose.yml         <- apply manually from docker/ai_module_fork/docker/compose.yml (this repo)
    compose_gpu.yml     <- apply manually from docker/ai_module_fork/docker/compose_gpu.yml (this repo)
  system/               <- upstream; NEVER modified (gotcha 12)
```

`src/tests/` is **not** synced — see "Sync approach" below. `compose.yml`/`compose_gpu.yml` are
**not** synced by the script either (they live outside `ai_module/`, so per gotcha 12 they're a
manual step — see "Point the compose `ai_module` service at our Dockerfile" below); the versions
under `docker/ai_module_fork/docker/` are the copy-pasteable source for that manual edit, not a
build-time dependency of the image.

## Sync approach (and why tests are excluded)

Our source of truth stays in THIS repo (`src/`, and the Dockerfile here). We do not maintain a
second copy by hand; the sync scripts mirror our payload into the fork:

- `sync_to_fork.ps1` — Windows (staging only; no build possible on Windows).
- `sync_to_fork.sh`  — Ubuntu (staging, then build with compose).

Both copy `docker/ai_module_fork/docker/Dockerfile` → `<fork>/ai_module/docker/Dockerfile` and
`src/{core,ros_adapter}` → `<fork>/ai_module/src/`, and **exclude `src/tests/`** (plus
`__pycache__`, `.pytest_cache`, `*.pyc`).

**Why exclude tests:** the runtime image only needs `core` + `ros_adapter` — the adapter node
imports neither pytest nor the test tree. Shipping `src/tests/` only inflates a size-budgeted eval
image (Simply NUC i9, `docs/upstream_notes.md` §5). The test suite is run from THIS repo on Ubuntu
(`docs/ubuntu_setup.md` §3: `cd src && pytest …`), which is the correct gate — not from inside the
fork or the image. `pytest` is still pip-installed in the image for optional in-container smoke,
but no test *files* are shipped.

## Dockerfile design decisions

- **Build context = `ai_module/`** (not repo root). Every `COPY` path is relative to `ai_module/`
  so the file is legal inside the fork's editable tree and matches how upstream builds the dummy.
- **PEP 668 → `--break-system-packages`** (not a venv). The Noble base marks the system Python
  externally-managed, so a bare `pip install` fails. We install into the system interpreter with
  `--break-system-packages` rather than a venv because the base image's rclpy / ROS message
  packages live on the system Python; a venv would either lose them or fork the environment and
  complicate colcon / `ros2 run` Python resolution — more risk on the riskiest gate. The container
  is single-purpose and disposable, so system site-packages "pollution" costs nothing. See the
  in-file comment (flag 2 of 3) for the drop-the-flag condition.
- **`RMW_IMPLEMENTATION=rmw_cyclonedds_cpp` baked as ENV, with a warning** that a compose
  `environment:`/`env_file:` entry OVERRIDES it. Our compose snippet re-asserts it deliberately.
- **colcon symlink layout + the three "confirm on Ubuntu" flags** are preserved as visible
  in-file comments: (1) cross-container discovery under this RMW, (2) `--break-system-packages`
  necessity, (3) colcon symlink layout + `ament`/`core`-on-PYTHONPATH coexistence.

## Point the compose `ai_module` service at our Dockerfile

Upstream `docker/compose.yml` builds the `ai_module` service `FROM ../ai_module` using
`ai_module/docker/Dockerfile` and (per `docker/README.md`) launches
`ros2 launch dummy_vlm dummy_vlm.launch`. After syncing, the fork's Dockerfile at
`ai_module/docker/Dockerfile` is already ours; edit the service `command` (and re-assert RMW /
optional keys) so it launches our node:

```yaml
  ai_module:
    build:
      context: ../ai_module              # fork-shape: context IS ai_module/ (COPY src/ resolves)
      dockerfile: docker/Dockerfile      # i.e. ai_module/docker/Dockerfile
    container_name: iros2026_ai_module
    network_mode: host
    environment:
      - RMW_IMPLEMENTATION=rmw_cyclonedds_cpp   # re-assert: OVERRIDES the Dockerfile ENV
    env_file:
      - ../.env.llm                      # optional, gitignored: OPENAI_API_KEY=... etc.
    command: >
      bash -lc "source /opt/vla/ws/install/setup.bash &&
                ros2 launch vla_ai_module ai_module.launch.py"
    stdin_open: true
    tty: true
```

The `system` service is unchanged (`docs/upstream_notes.md` gotcha 12: never touch the system
container).

## Definition of done (packaging gate)

Packaging is done ONLY when **`docker compose up --build` from a clean clone of the FORK**
succeeds and our node comes up. Building our repo directly does not count. Exact commands
(Ubuntu-gated):

```bash
# 0. clean clone of the FORK (the submitted repo), not our dev repo
git clone <fork-remote> /tmp/fork-clean && cd /tmp/fork-clean

# 1. stage our payload into ai_module/ (from a checkout of THIS repo)
/path/to/this-repo/docker/ai_module_fork/sync_to_fork.sh /tmp/fork-clean

# 2. build + bring the stack up exactly as evaluators do
xhost +
cd /tmp/fork-clean/docker
docker compose -f compose.yml up --build -d

# 3. verify our node came up and answers (Tier 2.7, docs/sim_verification.md)
```

For the real submission the synced `ai_module/` is committed into the fork and pushed; the sync
scripts are the reproducible way to regenerate it from this repo, not a run-time dependency of the
image.

## LLM keys — never baked

Keys are passed at run time only, never into the image (compose `env_file`/`environment`).
`core/llm/config.py` reads `VLA_LLM_PRIMARY_*` / `_SECONDARY_*` / `_LOCAL_*` (each names the env
var holding its key, e.g. `OPENAI_API_KEY`), plus `VLA_LLM_CALL_TIMEOUT_S` / `VLA_LLM_CONFIG`.
Unset ⇒ the ladder runs local/regex only (offline).

## Relationship to the old `docker/ai_module/`

`docker/ai_module/` is **deprecated** — see its README stub. Its Dockerfile is kept (not deleted)
as a historical record of the repo-root-context approach; do not build from it. All new work and
the submission flow use this directory.
