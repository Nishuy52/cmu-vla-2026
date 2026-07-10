# Sim Verification Runbook — "is everything actually working?"

*Written 10 Jul 2026. The ordered, environment-by-environment way to prove the stack works —
from what runs on Windows **today** up to a full challenge sim loop on native Ubuntu. Each step
gives the exact command, the PASS signal, and the most likely fix on FAIL. Run tiers in order;
a later tier assumes the earlier ones pass.*

The three tiers map to the three machines in the plan:

| Tier | Environment | What it proves | Blocked-on |
|------|-------------|----------------|------------|
| **0** | Windows, now (no ROS/sim) | The pure-Python `core/` is correct and the offline loop closes | nothing — do now |
| **1** | WSL2 (optional stretch) | ROS 2 graph + containers come up headless | Windows 11 + WSL2 + Docker Desktop |
| **2** | native Ubuntu 24.04 | The real closed-loop challenge sim end-to-end | Ubuntu box (`docs/ubuntu_setup.md`) |

Cross-references: topic contract + rates = `docs/upstream_notes.md` §3; launch mechanics = §4;
docker = §5; waypoint plumbing = §5b; numbered gotchas = §7. Exact upstream commands are quoted
from `upstream/CMU-VLN-Challenge-2026/docker/README.md` and the compose files.

---

## TIER 0 — Windows, available now (no ROS, no simulator)

Everything here runs against mocks/replays under `src/`; no ROS import anywhere in `core/`
(`src/README.md`). These are the checks that gate every code change today.

### 0.1 Core test suite

```powershell
cd src
python -m pytest -q
```

- **PASS:** all tests green — currently **`244 passed`** (measured 10 Jul 2026). The count grows as
  modules land (`core/parsing`, real `core/perception`, `ros_adapter`), so treat the number as a
  high-water mark, not a fixed target: it should only ever go **up** between sessions, never down.
- **FAIL — `ModuleNotFoundError: No module named 'core'`:** you ran pytest from the repo root, not
  from `src/`. `pythonpath = ["."]` is set in `src/pyproject.toml` `[tool.pytest.ini_options]` and
  `src/tests/conftest.py` prepends `src/` to `sys.path` — both are relative to `src/`, so pytest
  must be invoked from there.
- **FAIL — collection error / import error in one module:** a module was edited outside its owner
  boundary or an interface changed. `core/interfaces.py` is frozen v1 (`src/README.md`); if a test
  fails there, the contract was broken and the change must be reverted or taken to architecture
  review, not patched in the test.
- **FAIL — `numpy` missing:** the venv is not active or deps not installed —
  `pip install numpy pytest` (`docs/windows_workplan.md` "Local environment setup").

### 0.2 Synthetic end-to-end smoke (no simulator)

Drives `MockRobotIO` over a `SyntheticScene` through the real `QuestionController` FSM and asserts
one answer is published on the correct sink. This exercises the core loop
(intake → parse → orient → explore → verify → answer, `src/core/fsm/controller.py`) with the
external work stubbed, exactly as the ROS adapter will call it — but with zero ROS. Save as
`src/tests/test_smoke_e2e.py` (or paste into a scratch file run with `PYTHONPATH=. python`):

```python
"""Synthetic end-to-end smoke: core loop closes with no simulator."""
from core.mocks.mock_io import MockRobotIO, FakeClock
from core.mocks.synthetic_scene import SyntheticScene
from core.fsm.controller import QuestionController, WorldView, StabilitySignal
from core.interfaces import IntAnswer


def test_synthetic_end_to_end_numerical():
    # Deterministic scene, three "sofa" boxes → the numerical answer is 3.
    scene = SyntheticScene(7)
    for i in range(3):
        scene.place_box("sofa", 1.0 + i, 1.0, 0.8, 0.8, 0.6)

    clock = FakeClock()
    io = MockRobotIO(scene, clock)
    io.set_question("How many sofas are in the room?")   # as the 1 Hz republish would deliver

    # Stub the injected heavy work; the FSM itself is under test.
    qc = QuestionController(
        parse=lambda q: object(),                        # a non-None "plan"
        explore=lambda io, plan, w: None,                # one no-op step per tick
        verify=lambda io, plan, w: IntAnswer(value=3),   # head produces the answer
        probe=lambda io: WorldView(                       # stable count → early-answer gate opens
            stability=StabilitySignal(winner_margin=1.0, min_contrib_n_obs=3)
        ),
    )

    while not qc.answer_published:                        # ~5 Hz tick; clock advances 0.2 s/tick
        qc.tick(io)
        clock.advance(0.2)

    assert io.ints == [IntAnswer(value=3)]                # published exactly once, right sink
    assert qc.state.value == "done"
```

```powershell
cd src
python -m pytest tests/test_smoke_e2e.py -q
```

- **PASS:** `1 passed`. The FSM latched the question, ran to `DONE`, and published one `IntAnswer(3)`
  into `MockRobotIO.ints` — the closed loop works without a simulator. (Verified running against the
  current tree on 10 Jul 2026.)
- **FAIL — never leaves the loop / `answer_published` stays False:** the numerical early-answer gate
  never opened. That gate needs `StabilitySignal(winner_margin >= 0.25, min_contrib_n_obs >= 3)`
  (`core/fsm/controller.py`, `StabilitySignal.stable`); if you weaken the stub below that, the FSM
  correctly waits for the budget/watchdog path instead — advance the clock past the explore budget
  or raise the margin.
- **FAIL — answer on the wrong sink (`io.markers`/`io.waypoints`):** the qtype was mis-inferred.
  `_infer_qtype` keys off the question text (`controller.py`); "how many …" must route to
  `QType.NUMERICAL`. Keep the "How many" prefix or pass an explicit `qtype` on the plan.
- **FAIL — `ImportError` on `mock_io`/`synthetic_scene`:** run from `src/` (same path rule as 0.1).

### 0.3 Replay-data check (real-robot sample bags)

The offline replay harness feeds recorded sensor data through `core/` without ROS. The bags are
parsed with the **`rosbags`** pip package — a pure-Python bag reader, no ROS install needed
(`docs/windows_workplan.md` task 4).

```powershell
python -c "import rosbags, sys; print('rosbags', rosbags.__version__)"
Get-ChildItem -Recurse data\sample_real_robot
```

- **PASS:** `rosbags` imports **and** `data/sample_real_robot/` contains the sample bag(s) plus
  `data_view.rviz2` (the folder should hold `system_ros2.zip` unpacked and the rviz config).
- **FAIL — `data/sample_real_robot/` is empty:** this is the **expected state right now**. The
  automated `gdown` pull returns 0-byte files on every mode (observed 10 Jul 2026), so the data must
  be **manually downloaded in a browser** from the Drive folder
  (`https://drive.google.com/drive/folders/1xaatyLeIKLTh_oRzkyd7F1G6tkRPbFtm`) into
  `data/sample_real_robot/` — see `docs/master_plan.md` (the "USER ACTION: manually download sample
  data" checkbox). This is **not** blocking for `core/` development; only the replay harness needs it.
- **FAIL — `ModuleNotFoundError: rosbags`:** `pip install rosbags`
  (`docs/windows_workplan.md` "Local environment setup").

---

## TIER 1 — WSL2 (optional stretch; attempt, do not depend on it)

Worth one time-boxed evening on Windows 11 (`docs/windows_workplan.md` "Stretch: WSL2"). The goal is
only to bring the ROS 2 graph and the two challenge containers up **headless** — building images,
running nodes, replaying bags. **Unity sim rendering is NOT expected to work** under WSLg/WSL2 GPU
passthrough; if it stutters or crashes, stop — the repo offers no WSL support and evaluation is
native Linux anyway (`docs/windows_workplan.md` "Unproven: Unity sim rendering performance under
WSLg"). Do not spend Tier-1 time chasing the sim window.

### 1.1 Install WSL2 + Ubuntu 24.04

```powershell
wsl --install -d Ubuntu-24.04
```

- **PASS:** after reboot, `wsl -l -v` shows `Ubuntu-24.04` at `VERSION 2`.
- **FAIL — installs as VERSION 1:** `wsl --set-default-version 2` then reinstall the distro.

### 1.2 Docker Desktop, WSL2 backend + GPU

Install Docker Desktop and enable the WSL2 backend (Settings → Resources → WSL integration →
Ubuntu-24.04). Then, from inside the Ubuntu WSL shell:

```bash
docker run --rm --gpus all ubuntu nvidia-smi
```

- **PASS:** `nvidia-smi` prints your GPU table from inside the container (same check as the native
  Ubuntu box, `docs/ubuntu_setup.md` §2). This confirms GPU passthrough works end-to-end.
- **FAIL — `could not select device driver "" with capabilities: [[gpu]]`:** GPU passthrough not
  wired — update the Windows NVIDIA driver (WSL support is in the standard driver), enable GPU in
  Docker Desktop, restart Docker Desktop. If it still fails, GPU under WSL is optional at this tier —
  headless CPU checks below still run.

### 1.3 Pull the challenge images

Image tags are fixed in the upstream compose files
(`upstream/CMU-VLN-Challenge-2026/docker/compose.yml:3,19–23`). Quoted exactly:

- system image: `zhangjicmu/ubuntu24_ros:cmu_vla_challenge_simulation`
- ai_module base image: `zhangjicmu/ubuntu24_ros:ai_module`

```bash
docker pull zhangjicmu/ubuntu24_ros:cmu_vla_challenge_simulation
docker pull zhangjicmu/ubuntu24_ros:ai_module
```

- **PASS:** both pulls complete; `docker images` lists both tags. (Compressed size is large — the
  system image carries the full autonomy stack; allow disk + time.)
- **FAIL — manifest not found / auth error:** confirm the exact tag against
  `docker/compose.yml` (do not guess a tag — if it has changed upstream since this was written,
  **confirm on first run** against the current compose file rather than inventing one).

### 1.4 Bring the compose stack up (headless) and check the ROS graph

Per `upstream/CMU-VLN-Challenge-2026/docker/README.md` ("Run Docker Containers"). No `xhost +` /
RVIZ needed for headless checks. From `upstream/CMU-VLN-Challenge-2026/docker`:

```bash
docker compose -f compose.yml up --build -d          # CPU/headless variant
docker exec -it iros2026_system bash
# inside the system container:
ros2 topic list
```

- **PASS:** both containers report `Up` (`docker compose ps` shows `iros2026_system` and
  `iros2026_ai_module`, names from `compose.yml:4,23`), and `ros2 topic list` runs without error.
  Before any sim launch the list is sparse (mostly `/rosout`, `/parameter_events`) — that is fine;
  this step only proves the ROS 2 middleware is alive in-container.
- **FAIL — `ros2: command not found`:** the container shell has not sourced the ROS setup; open a
  fresh `bash` (the image `.bashrc` sources it) or `source /opt/ros/jazzy/setup.bash`.

### 1.5 Build + run the dummy ai_module (headless)

```bash
docker exec -it iros2026_ai_module bash
# inside the ai_module container:
ros2 launch dummy_vlm dummy_vlm.launch
```

- **PASS:** the `dummyVLM` node starts and idles waiting for `/challenge_question`
  (`docs/upstream_notes.md` §2). In another `iros2026_ai_module` shell, `ros2 node list` shows
  `/dummyVLM`. The dummy needs no camera/lidar (it only reads odometry + question, §2), so it runs
  fine with no sim.
- **FAIL — `package 'dummy_vlm' not found`:** the module image did not rebuild `dummy_vlm`. The
  compose `ai_module` service builds from `../ai_module` with `--build`
  (`docs/upstream_notes.md` §4–5); re-run `docker compose ... up --build -d`.

### 1.6 CycloneDDS env-var sanity (both containers)

The two containers only discover each other's topics if the RMW matches (`notes.txt`;
`docs/upstream_notes.md` gotcha 13). Check in **each** container:

```bash
echo "$RMW_IMPLEMENTATION"    # run in iros2026_system AND in iros2026_ai_module
```

- **PASS:** both print `rmw_cyclonedds_cpp`. Confirm cross-container visibility with the dummy up:
  from the system container, `ros2 topic list` should include the dummy's advertised topics
  (e.g. `/way_point_with_heading`, `/numerical_response`).
- **FAIL — empty or `rmw_fastrtps_cpp`, or the two `ros2 topic list`s disagree:** RMW mismatch.
  `export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp` in both and relaunch. This is the classic silent
  "containers can't see each other" failure (`docs/ubuntu_setup.md` troubleshooting table).

**Not expected to work under WSL2:** the Unity `Model.x86_64` render window and any sensor stream
that depends on it (`/camera/image`, `/registered_scan`, terrain maps). Those need native Ubuntu
(Tier 2). Do not treat their absence under WSL as a failure.

---

## TIER 2 — native Ubuntu 24.04 (the real ladder)

This extends `docs/ubuntu_setup.md` §5. Prereqs: §1–§4 of that guide done (NVIDIA driver, Docker +
NVIDIA Container Toolkit, this workspace, the challenge repo + submodule) and a training scene
binary installed per §5. Run the smoke tests **in order** — each assumes the previous passed.
Commands are quoted from `upstream/CMU-VLN-Challenge-2026/docker/README.md` and the `system_*.sh`
scripts unless noted.

### 2.1 Containers up (compose)

```bash
xhost +
cd upstream/CMU-VLN-Challenge-2026/docker
docker compose -f compose_gpu.yml up --build -d       # GPU host; use compose.yml if no GPU
docker compose -f compose_gpu.yml ps
```

- **PASS:** two services `Up`: `iros2026_system` and `iros2026_ai_module` (`compose.yml:4,23`;
  `docker/README.md` "Run Docker Containers"). Both are `network_mode: host` and share the ROS graph
  (`docs/upstream_notes.md` §4).
- **FAIL — GPU device error on `compose_gpu.yml`:** NVIDIA Container Toolkit not configured —
  `sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker`, verify with
  `docker run --rm --gpus all ubuntu nvidia-smi` (`docs/ubuntu_setup.md` §2). Fall back to
  `compose.yml` only to confirm non-GPU bring-up.

### 2.2 Scene installed + sim window renders

Launch base autonomy inside the system container (this spawns the Unity binary + RVIZ):

```bash
docker exec -it iros2026_system bash
# inside the container:
/home/docker/autonomy_stack_mecanum_wheel_platform/system_simulation.sh
```

`system_simulation.sh` runs the Unity binary
`./src/base_autonomy/vehicle_simulator/mesh/unity/environment/Model.x86_64 &` then `sleep 3`, then
the `system_simulation.launch` ros2 launch + RVIZ (`system_simulation.sh:8–12`;
`docs/upstream_notes.md` §4).

- **PASS:** the Unity sim window renders the scene, and RVIZ (`vehicle_simulator.rviz`) opens showing
  the robot sitting in the room with the panorama image, lidar points, and terrain map
  (`docs/ubuntu_setup.md` §5 first-smoke-test bullet). The robot is stationary — the base config
  starts no exploration planner (`docs/upstream_notes.md` §5b), so no motion yet is correct.
- **FAIL — Unity window black / no sensors:** `Model.x86_64` is missing or not executable. The scene
  binary is **not in git** — it is downloaded per-scene and dropped into
  `.../mesh/unity/environment/` (`docs/upstream_notes.md` §4, U3); `chmod +x Model.x86_64`
  (`docs/ubuntu_setup.md` §5). Also check the Unity↔ROS bridge port 10000 is not blocked
  (`ros_tcp_endpoint`, `system_simulation.launch:57–65`; `docs/ubuntu_setup.md` troubleshooting).
- **FAIL — no RVIZ window:** `xhost +` was not run before compose up (`docker/README.md`).

### 2.3 Sensor liveness

With 2.2 up, in another system-container shell check each system-output topic streams at its
contract rate (`docs/upstream_notes.md` §3, `README.md:107–111`):

```bash
ros2 topic hz /camera/image        # panorama 1920x640
ros2 topic hz /registered_scan
ros2 topic hz /terrain_map
ros2 topic hz /terrain_map_ext
ros2 topic hz /state_estimation
```

- **PASS:** measured rates land near the contract:

  | Topic | Type | Expected rate |
  |-------|------|---------------|
  | `/camera/image` | `sensor_msgs/Image` | ~10 Hz |
  | `/registered_scan` | `sensor_msgs/PointCloud2` | ~5 Hz |
  | `/terrain_map` | `sensor_msgs/PointCloud2` | ~5 Hz |
  | `/terrain_map_ext` | `sensor_msgs/PointCloud2` | ~5 Hz |
  | `/state_estimation` | `nav_msgs/Odometry` | ~100–200 Hz |

  (Rates/types from `docs/upstream_notes.md` §3; `/camera/image` is the decompressed republish from
  `sim_image_repub`, `system_simulation.launch:67–79`.)
- **FAIL — `/camera/image` silent but `/camera/image/compressed` present:** `sim_image_repub` did not
  start; check the launch node list (`system_simulation.launch:67–79`).
- **FAIL — all topics silent:** the Unity bridge is not connected (port 10000) or the sim did not
  finish starting — the `sleep 3` in `system_simulation.sh` may be too short on a slow host; wait
  and re-check. If still dead, revisit 2.2.
- **FAIL — terrain topics silent but scan is live:** `terrain_analysis` / `terrain_analysis_ext`
  crashed; both subscribe to `/registered_scan` + `/state_estimation`
  (`docs/upstream_notes.md` §3) — confirm those are flowing first.

### 2.4 Actuation round-trip (waypoint → robot moves)

Publish a single reachable waypoint ~1.5 m ahead of the robot on the **allowed** actuation topic and
watch it drive there. The output type is `geometry_msgs/msg/Pose2D`; heading is ignored this year, so
`theta: 0.0` (`docs/upstream_notes.md` §3, gotcha 4). (Substitute x/y ~1.5 m from the robot's current
`/state_estimation` position; the example assumes the robot near the origin.)

```bash
ros2 topic pub --once /way_point_with_heading geometry_msgs/msg/Pose2D "{x: 1.5, y: 0.0, theta: 0.0}"
```

- **PASS:** the robot drives to roughly (1.5, 0.0) in the `map` frame and stops. The path runs
  `/way_point_with_heading` → `waypointConverter` → `/way_point` → `localPlanner`/`pathFollower` →
  `/cmd_vel` (`docs/upstream_notes.md` §5b); the local planner does the collision-avoidance, so a
  slightly-off waypoint still works (it is snapped into traversable area,
  `waypointConverter.cpp:196–246,305–309`; gotcha 10).
- **FAIL — robot does not move at all:** you likely published `/way_point` directly instead of
  `/way_point_with_heading`. `/way_point` is a **base-autonomy internal** topic and is not the AI
  module's actuation channel (`docs/upstream_notes.md` gotcha 11, §5b) — publish only
  `/way_point_with_heading`.
- **FAIL — robot lurches then gets stuck at a dead end:** the waypoint was too far. The stack warns
  that a distant waypoint can strand the vehicle (`autonomy_stack .../README.md:55`;
  `docs/upstream_notes.md` gotcha 14) — keep goals incremental and ≤ ~2.5 m ahead
  (`docs/ubuntu_setup.md` troubleshooting; `core/interfaces.py` `WaypointCmd` docstring).

### 2.5 Dummy ai_module round-trip (question → answer)

Bring the dummy up (`ros2 launch dummy_vlm dummy_vlm.launch` in `iros2026_ai_module`, per 1.5) and
send a question. The dummy branches purely on the leading text of the string
(`docs/upstream_notes.md` §2, `dummyVLM.cpp:279–307`). Use the object-reference branch — it is the
clearest single round-trip (marker + one waypoint). Command quoted from
`upstream/CMU-VLN-Challenge-2026/docker/README.md`:

```bash
ros2 topic pub --once /challenge_question std_msgs/msg/String \
  "{data: 'Find teal pillow on the sofa farthest from the window'}"
```

- **PASS:** the dummy publishes a `visualization_msgs/Marker` CUBE on `/selected_object_marker` (the
  hard-coded "sofa" object, highlighted blue in RVIZ) **and** one `Pose2D` on
  `/way_point_with_heading` at the object centre; the robot heads toward it
  (`docs/upstream_notes.md` §2, `dummyVLM.cpp:288`, `pubObjectMarker`/`pubObjectWaypoint`).
  Confirm with `ros2 topic echo --once /selected_object_marker`. Sanity-check the other branches too:
  a `"How many …"` question → a random `std_msgs/Int32` (1–10) on `/numerical_response`
  (`dummyVLM.cpp:292–294`, genuinely random — no counting); any other text → a sequence of nav
  waypoints (`dummyVLM.cpp:297–301`).
- **FAIL — no marker/answer at all:** the question topic type or name is wrong — it must be
  `std_msgs/msg/String` on `/challenge_question` (`docs/upstream_notes.md` §2 topic table). Confirm
  the dummy node is up (`ros2 node list` → `/dummyVLM`).
- **FAIL — marker appears but RVIZ scores/overlays look off:** the dummy advertises the marker topic
  **without** a leading slash (`selected_object_marker`, `dummyVLM.cpp:262`) while the README/eval
  use `/selected_object_marker`; under the default namespace these resolve to the same name
  (`docs/upstream_notes.md` §2 / gotcha 5) — echo the fully-qualified `/selected_object_marker` to
  confirm.

### 2.6 Full-question dry-run with evaluation-style relaunch

Simulate one scored question the way evaluation does: **relaunch the whole stack for the question**
(no state carries across questions — the system is relaunched per language command,
`docs/upstream_notes.md` gotcha 1, `README.md:162`), then observe for the full budget.

```bash
# tear down and bring the stack back up fresh (no cross-question memory)
cd upstream/CMU-VLN-Challenge-2026/docker
docker compose -f compose_gpu.yml down
docker compose -f compose_gpu.yml up -d
# relaunch base autonomy (2.2) + dummy (2.5), then publish ONE question and observe
```

- **PASS:** from a cold relaunch the stack reaches sensor liveness (2.3), the single published
  question is answered once, and the robot's behaviour is consistent with that answer over a
  **10-minute** observation window — the per-question hard budget, timed from system startup and
  including re-exploration (`docs/upstream_notes.md` gotchas 1–2, `README.md:162,180–182`). The
  question is republished at 1 Hz; a correct module latches the first receipt and ignores the rest
  (gotcha 3) — watch that the answer is published exactly once, not re-emitted each second.
- **FAIL — behaviour depends on the previous question:** something cached map/scene state across the
  relaunch. Nothing may persist between questions (gotcha 1); a full `down`/`up` (not just a node
  restart) is the reliable reset.
- **NOTE:** the real timing/scoring is enforced by the closed-source `challenge_evaluation_node`,
  which is not in this repo (`docs/upstream_notes.md` §4 "Timing / eval hooks", U4) — so this
  dry-run is a **behavioural** rehearsal of the 10-minute loop, not the scored measurement. Treat the
  10 minutes as the budget to design against; the exact scoring math is **confirm on first run** with
  the organizers' node.

### 2.7 Our module swap-in — placeholder (forward reference)

*Deferred until `src/ros_adapter/` exists (Phase 2, `docs/ubuntu_setup.md` §7).* When it lands, this
step replaces the dummy launch entry with the adapter node in the `ai_module` container and re-runs
2.5–2.6 against **our** module. Checklist to fill in then:

- [ ] `src/ros_adapter/` rclpy node builds inside `iros2026_ai_module` (colcon, per the `ai_module/`
      Dockerfile layout, `docs/upstream_notes.md` §5) and replaces the `dummy_vlm` launch entry
      (`docs/ubuntu_setup.md` §7 step 2).
- [ ] Adapter subscribes only the six allowed system-output topics and publishes only the three
      allowed answer topics (`docs/upstream_notes.md` §3; gotchas 8, 11).
- [ ] Object-reference question → our `/selected_object_marker` box scores against GT overlap
      (right `map` frame, full extents — gotchas 5–6; `docs/ubuntu_setup.md` troubleshooting).
- [ ] Numerical question → our `/numerical_response` (not random, unlike the dummy).
- [ ] Instruction-following question → our `/way_point_with_heading` sequence follows the route and
      stops at the goal within the 10-minute budget.
- [ ] Full training-scene run under the 10-minute clock before any submission
      (`docs/ubuntu_setup.md` §7 step 4).

---

## Known-good state checklist

Re-run this table after **any** environment change (driver update, image re-pull, scene swap,
adapter rebuild) to confirm nothing silently regressed. Tier tells you which machine each row needs.

| # | Component | Verify command | Expected (PASS) |
|---|-----------|----------------|-----------------|
| K0.1 | Core suite | `cd src && python -m pytest -q` | `244 passed` (grows over time; never fewer) |
| K0.2 | Synthetic loop | `python -m pytest tests/test_smoke_e2e.py -q` | `1 passed` — one `IntAnswer` published, FSM `done` |
| K0.3 | Replay deps + data | `python -c "import rosbags"` ; list `data/sample_real_robot` | import OK; sample bag(s) + `data_view.rviz2` present (manual download) |
| K1.1 | WSL GPU passthrough | `docker run --rm --gpus all ubuntu nvidia-smi` | GPU table prints (WSL) |
| K1.2 | Images pulled | `docker images` | `...:cmu_vla_challenge_simulation` + `...:ai_module` listed |
| K1.3 | Containers up | `docker compose ps` | `iros2026_system` + `iros2026_ai_module` `Up` |
| K1.4 | ROS graph alive | `docker exec -it iros2026_system ros2 topic list` | runs; `/rosout` etc. present |
| K1.5 | RMW consistent | `echo "$RMW_IMPLEMENTATION"` in both containers | both `rmw_cyclonedds_cpp` |
| K1.6 | Dummy node | `ros2 node list` (ai_module) | `/dummyVLM` present |
| K2.1 | Sim renders | run `system_simulation.sh` | Unity window + RVIZ show robot, pano, lidar, terrain |
| K2.2 | Camera stream | `ros2 topic hz /camera/image` | ~10 Hz |
| K2.3 | Registered scan | `ros2 topic hz /registered_scan` | ~5 Hz |
| K2.4 | Terrain maps | `ros2 topic hz /terrain_map` ; `/terrain_map_ext` | ~5 Hz each |
| K2.5 | Odometry | `ros2 topic hz /state_estimation` | ~100–200 Hz |
| K2.6 | Actuation | `ros2 topic pub --once /way_point_with_heading geometry_msgs/msg/Pose2D "{x: 1.5, y: 0.0, theta: 0.0}"` | robot drives ~1.5 m and stops |
| K2.7 | Dummy round-trip | publish a `Find …` question on `/challenge_question` | marker on `/selected_object_marker` + object waypoint |
| K2.8 | Relaunch isolation | `docker compose down && up`, re-ask one question | fresh answer, no cross-question state, ~10-min loop |

*Rows above the first Tier-2 line run on Windows/WSL; the Tier-2 rows need the native Ubuntu sim.*
