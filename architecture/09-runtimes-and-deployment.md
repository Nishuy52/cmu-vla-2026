# 09 - Runtimes and deployment

The execution surfaces the pure-Python `core` package plugs into: the ROS
adapter, the submission Docker image, host-native iteration, GPU hardening
for the dev box, the offline replay harness, and the two-tier test gate.

## ELI10

The brain (`core/`) can run three different ways: replaying a saved
recording with no robot at all (for fast iteration and unit tests), talking
to the real simulated robot straight from the host machine (fast to
restart, for day-to-day development), or packed into the one Docker
container the actual competition will run (slow to rebuild, but the only
form that counts on eval day). All three now work, and the full container
path has been proven end-to-end at least once — a real question got asked,
the robot drove, and it produced a real answer with a picture of the
correct object.

## The ROS adapter — the one ROS-dependent module

[`src/ros_adapter/adapter_node.py`](../src/ros_adapter/adapter_node.py)
(1040 lines) is, by its own docstring (around line 1), "the ONLY
ROS-dependent module in this repo" — the single seam between the pure-Python
`core` package and ROS 2. Dependency direction is one-way: `ros_adapter`
imports `core`; `core` never imports `rclpy` or anything ROS-specific, so
the `pytest` suite over `core` stays entirely ROS-free.

It does five things (docstring, lines 8-19): subscribes the six allowed
system-output topics; converts each message to the frozen `core` dataclasses
by reusing the same converters `core.replay.bag_reader` uses for offline
replay (rclpy message layouts match what those converters already read);
latches the latest converted value per topic behind a lock (rclpy executor
callbacks write, a 5 Hz timer reads); drives
`QuestionController.tick(self)` at 5 Hz with the node itself standing in as
the `RobotIO`; and publishes the three answer topics (Marker / Pose2D /
Int32). An optional debug-viz path (off by default, zero overhead when
disabled) republishes the tracked instance map and a breadcrumb path for
RVIZ.

**Perception wiring is env-gated** (`VLA_DETECTOR`, around lines 404-430).
With `VLA_DETECTOR=none` (the default) the node keeps an empty
`BasicSceneIndex([])` and answers come from the FSM floors only — a loud
`SUBMISSION-BLOCKER` log line fires so this can never pass a smoke test
silently (around line 438). With `VLA_DETECTOR=grounding_dino`, a
`PerceptionPipeline` is constructed and its live index (mutated in place as
frames fuse) becomes the index the controller sees
(`self._scene_index = self._perception.index`, around line 428) — this is
the path Gate 4 (below) actually exercised.

**Gate 4 — proven end-to-end in-container (commit `0fbd534`).** The clean-
container run recorded in
[`reports/gate4_final/evidence.log`](../reports/gate4_final/evidence.log)
shows: the launch wrapper's `prewarm OK: model qwen2.5vl:3b loaded` line,
the adapter's own `LLM ladder configured` boot line, a published question
("Find the teapot on the table."), 44 waypoints driven ending near
`(0.629, -4.814)`, and a `/selected_object_marker` capture for `dining
table` with a real, non-degenerate scale (`2.98 x 2.99 x 0.978`) — i.e. a
genuine detection-backed answer, not a floor default. The log also shows
GPU memory cycling `P8 → P0/P3` under load and back down, and 2 Ollama
calls recorded — consistent with the checkpoint-budgeted LLM ladder
actually firing rather than falling through to the regex floor.

## Packaging: `docker/ai_module_fork/` and the 28 GB image

The submission is a **fork** in which only `ai_module/` may be modified;
the upstream compose builds the `ai_module` service from context
`../ai_module` using `ai_module/docker/Dockerfile`
([`docs/upstream_notes.md`](../docs/upstream_notes.md) §5B). Our
fork-shaped Dockerfile and sync tooling live in
[`docker/ai_module_fork/`](../docker/ai_module_fork/) (`docker/Dockerfile`,
`docker/compose.yml`, `docker/compose_gpu.yml`,
`ai_module/launch_with_llm.sh`, `sync_to_fork.sh`/`.ps1`); the old
repo-root-context `docker/ai_module/` is deprecated. The packaging gate is
`docker compose up --build` succeeding from a **clean clone of the fork**,
not from building this dev repo directly.

**Measured image size (commits `418bb89`, `5c9e941`, 19 Jul): ~28 GB
total** — a correction of an earlier authored-pending-build estimate
("6.76 GB baseline + 5 GB" ≈ 11-12 GB) that predates the actual build. The
real layer breakdown, per
[`docs/ubuntu_setup.md`](../docs/ubuntu_setup.md) §7a (around line 273):
base image 9.4 GB + ros-jazzy-desktop 3.4 GB + torch/CUDA pip 5.3 GB +
GroundingDINO weights 1.4 GB + Ollama runtime/model 3.2 GB, plus assorted
sub-GB layers — all load-bearing (the `cuda_v12` Ollama runner variant is
correctly pruned; `cuda_v13*`/`cpu*` are kept). Investigate only if a
rebuild deviates from ~28 GB by more than 2 GB either direction. The bake
needs network access at *build* time (the Ollama release tarball + model
registry, same class of dependency as the GroundingDINO weight fetch) even
though the resulting image is fully offline at runtime.

**Ordered post-build verification:**
[`reports/local_llm_phase3/BUILD_AND_VERIFY.md`](../reports/local_llm_phase3/BUILD_AND_VERIFY.md)
is the runbook to execute after every Phase-3 image build, in order: (a)
image-size delta check against the ~28 GB bar; (b) the boot-log grep
sequence (`starting ollama serve` → `ollama up after Ns` → `prewarm OK` →
`vla_ai_module up: subscribed 6 topics`, proving the launch wrapper's
`exec` handoff succeeded); (c) an in-container curl smoke test, both
text-only and base64-JPEG vision, against the baked
`/v1/chat/completions` endpoint; (d) an in-container ladder-conformance
check (2-3 questions, asserting `parse_tier == 'local'` not `'regex'`); (e)
a VRAM/coexistence check confirming GDINO (~3.6 GB) and the local 3B model
(~3.2 GB) can hold memory simultaneously without an OOM; (f) the §7a
clean-clone packaging gate itself, re-run from a genuinely fresh clone to
prove reproducibility. Each step documents its own PASS bar and the most
likely failure mode plus fix.

As-built gotchas recorded from the first full run (`docs/ubuntu_setup.md`
§7a, around line 250): `DISPLAY` must propagate from an actual desktop
session at `docker compose up` time or RVIZ dies with a Qt "xcb" error;
`docker builder prune -f --keep-storage 12GB` is needed after every image
build or the build-cache volume fills (measured ~10 GB growth per
generation); the controller's 60 s ORIENT window before the first waypoint
is correct behaviour, not a hang; and the adapter accepts exactly one
question per process (a second publish is dropped, requiring a container
restart for a fresh run).

## Host-first iteration (standing rule 5, added 18 Jul 2026)

Per the workspace's standing rules, everything runs natively on the dev
machine by default — the host-node loop for the adapter/perception layer,
the host venv for `core`, host Ollama for LLM work. Docker builds/runs are
reserved for key checkpoints only: gate exits, battery results meant to be
trusted, the §7a packaging gate, and submissions — rebuilds are too slow
for tight iteration.

[`tools/run_host_node.sh`](../tools/run_host_node.sh) and
[`tools/setup_host_node.sh`](../tools/setup_host_node.sh) implement this
(`docs/ubuntu_setup.md` §8b, around line 352): `setup_host_node.sh` is a
one-time ROS 2 Jazzy apt install (needs sudo); `run_host_node.sh` runs the
adapter natively against the sim containers' ROS graph on each iteration —
host networking + CycloneDDS make cross-process discovery transparent, so
this is seconds per cycle instead of minutes for a rebuild. Three rules
keep it safe from drifting out of sync with the container: **the image is
the truth** (the host venv mirrors the image's exact pins —
`numpy==1.26.4`, `transformers==4.57.6` — with the fork Dockerfile as the
single source of truth for both, and model weights/HF cache extracted FROM
the built image rather than re-downloaded); **checkpoint discipline** (any
gate exit, trusted battery result, or submission MUST be re-run inside the
container — `sync_to_fork` → `compose build` → the §7a clean-clone gate —
because host runs are scratch evidence only); and **never run the host
node and the containerised `ai_module` simultaneously** (both would latch
the same question).

`run_host_node.sh` (21 lines) is deliberately thin: it sources
`~/vla_host/env.sh` (written by `setup_host_node.sh`, one-time), then ROS
2 Jazzy's `setup.bash` and the host venv, sets `PYTHONPATH` to this repo's
`src/`, and execs `python -c "from ros_adapter.adapter_node import main; main()"`
— the exact same `main()` entry point the container runs, just invoked
directly against the host's Python instead of through a `ros2 launch`
inside a container. The script's own header comment repeats the mutual-
exclusion rule as a warning: running this alongside the containerised
`ai_module` means two adapters fighting over the same question latch.

## GPU hardening (dev box only)

[`tools/gpu_hardening/`](../tools/gpu_hardening/) addresses a recurring
GPU wedge on the dev machine's RTX 4060 Laptop, where the GPU can lock at
the P8 power state (~210 MHz) under a software power cap and never recover
without a reboot. `install.sh` (commit `9cd11ec`) installs an
`/etc/modprobe.d/` override disabling NVIDIA's RTD3 runtime power
management, plus a udev rule pinning `power/control` to `on` for the PCI
device, and runs `update-initramfs -u`; changes apply at the next reboot.
Commit `1003fdb` renamed the conf file to `zz-nvidia-disable-rtd3.conf` so
it alphabetically outsorts (and overrides) the NVIDIA package's own
`nvidia-runtimepm.conf` in the same directory — a plain non-prefixed
filename lost that ordering fight. Verification after reboot:
`cat /sys/bus/pci/devices/0000:01:00.0/power/control` should read `on`, and
`nvidia-smi -q -d PERFORMANCE | grep -A4 'Clocks Event Reasons'` should show
no software-power-cap throttle reason. This is dev-box-only maintenance —
it does not touch anything shipped in the submission image.

## Replay: offline harness over recorded bags

[`src/core/replay/`](../src/core/replay/) lets the full
`core.fsm.QuestionController` run against recorded data with no ROS
installation at all, using the pure-Python `rosbags` package. Three
modules: `bag_reader.py` (`BagSource` plus a converter registry —
`image_to_pano`, `odom_to_state`, `pointcloud_to_lidar`,
`pointcloud_to_terrain`, `string_to_question` — these are the SAME
converters `adapter_node.py` reuses, so replay and the live adapter agree
on message-to-dataclass conversion by construction); `replay_io.py`
(`ReplayRobotIO`, `ReplayClock`, `MessageStore` — a `RobotIO`
implementation that serves recorded messages instead of live ROS topics);
and `fixtures.py` (`extract_fixtures`/`load_fixtures`, distilling large
bags down into compact, machine-portable `.npz` fixtures for repeatable
test/battery runs without carrying full bag files around).

## The two-tier test gate

Per [`src/README.md`](../src/README.md) "Test tiers": the suite splits into
a **fast** default tier and a **slow** tier of full-controller simulations
and multi-scene batteries (marked `@pytest.mark.slow`). While iterating,
plain `pytest` from `src/` runs the fast tier only (`-m "not slow"` is
baked into `addopts`) — about 40 s, no individual test over ~2 s. At
milestones — before committing, before opening a PR — `pytest -m ""` from
`src/` runs everything (~6 min serial), or `pytest -m "" -n auto` with the
optional `dev` extra installed (`pytest-xdist`) in parallel (~2.5 min);
`-n` is deliberately kept out of `addopts` so single-test debugging stays
serial and readable. The slow tier is dominated by simulated FSM ticking
and batteries; several integration cases compress the FSM's time-budget
gates through a scaled clock (`tests/integration/_scaled.py`, mirroring
`core.runner.single._ScaledClock`) so only the budget/watchdog timing sees
amplified time — structural assertions (states, answers, driven geometry)
stay unchanged while tick counts collapse roughly 20x.

## References

- [`src/ros_adapter/adapter_node.py`](../src/ros_adapter/adapter_node.py)
- [`docker/ai_module_fork/`](../docker/ai_module_fork/)
- [`reports/gate4_final/evidence.log`](../reports/gate4_final/evidence.log)
- [`reports/local_llm_phase3/BUILD_AND_VERIFY.md`](../reports/local_llm_phase3/BUILD_AND_VERIFY.md)
- [`docs/ubuntu_setup.md`](../docs/ubuntu_setup.md) §7a, §8, §8a, §8b
- [`tools/run_host_node.sh`](../tools/run_host_node.sh),
  [`tools/setup_host_node.sh`](../tools/setup_host_node.sh)
- [`tools/gpu_hardening/`](../tools/gpu_hardening/)
- [`src/core/replay/`](../src/core/replay/)
- [`src/README.md`](../src/README.md) "Test tiers"
- Sibling docs: [07-navigation-and-exploration.md](07-navigation-and-exploration.md),
  [10-gaps-and-risks.md](10-gaps-and-risks.md)
