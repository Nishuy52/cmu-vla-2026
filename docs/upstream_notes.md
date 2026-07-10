# Upstream Reference — CMU VLN Challenge 2026

**Date:** 10 Jul 2026
**Scope:** Canonical internal reference for the upstream challenge repo. Read this instead of re-reading upstream code.
**Source repo (read-only):** `upstream/CMU-VLN-Challenge-2026/`
All paths below are relative to that repo root unless stated otherwise.

**Submodule note:** the autonomy-stack submodule
`autonomy_stack_mecanum_wheel_platform/` is pinned to
`160000 commit 81035e9e4190826b7458c711f08cb64f8f9e64ac` and is **now fully checked out**
locally. Claims about the base autonomy stack below are now sourced directly from the submodule
code (`autonomy_stack_mecanum_wheel_platform/...`), not just the top README. Sub-paths in this
doc that start at `src/...` or `system_*.sh` are relative to the submodule root
`autonomy_stack_mecanum_wheel_platform/`. The only content still un-inspectable is the Unity
scene binary itself (`.../mesh/unity/environment/Model.x86_64`, downloaded per-scene, not in git)
and the closed-source `challenge_evaluation_node`.

---

## 1. Repo top-level map

| Path | What it is |
|------|-----------|
| `README.md` | Master challenge spec: task, I/O contract, scoring, timing, FAQ. Primary contract source. |
| `LICENSE` | BSD-family license. |
| `notes.txt` | Two-line ops note: install `ros-jazzy-rmw-cyclonedds-cpp`, export `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp`; use the two zips to build the two docker images. |
| `.gitmodules` | Declares the one submodule (autonomy stack), branch `iros_workshop`, git-SSH URL. |
| `ai_module/` | **The only folder teams may modify.** Contains the dummy VLM ROS2 package + its own docker build. Our replacement target. |
| `autonomy_stack_mecanum_wheel_platform/` | Submodule (pinned `81035e9`, checked out): base navigation/autonomy system. Sub-dirs: `src/base_autonomy/` (local_planner, terrain_analysis[_ext], sensor_scan_generation, vehicle_simulator, waypoint_converter, visualization_tools), `src/exploration_planner/tare_planner`, `src/route_planner/far_planner`, `src/slam`, `src/utilities`, plus nine `system_*.sh` launch scripts at its root. |
| `docker/` | Top-level compose stack that runs the `system` container (autonomy sim) + `ai_module` container together. |
| `questions/` | Per-scene challenge questions: `questions.json` (all 15 scenes) plus one folder per scene with a `questions.pdf` and two target-trajectory `.ply` files. |
| `figures/` | README images only: `scenes.png`, `simulator.png`, `system.png`. |

---

## 2. The dummy `ai_module`

### File list

| File | Language / role |
|------|-----------------|
| `ai_module/src/dummy_vlm/src/dummyVLM.cpp` | C++ ROS2 (rclcpp) node — the entire dummy logic (310 lines). |
| `ai_module/src/dummy_vlm/CMakeLists.txt` | `ament_cmake` build; single executable `dummyVLM`. |
| `ai_module/src/dummy_vlm/package.xml` | Package `dummy_vlm`, format 3, maintainer Ji Zhang. |
| `ai_module/src/dummy_vlm/launch/dummy_vlm.launch` | XML launch; starts the one node with 3 params. |
| `ai_module/src/dummy_vlm/data/object_list.txt` | Single hard-coded object record (the "sofa"). |
| `ai_module/src/dummy_vlm/data/waypoints.ply` | 2-vertex canned waypoint path for nav questions. |
| `ai_module/src/dummy_vlm/data/question.txt` | Unused-at-runtime sample string `Go to the sofa`. |
| `ai_module/docker/Dockerfile` | Rebuilds only `dummy_vlm` on top of the prebuilt `ai_module` image. |
| `ai_module/docker/build.sh` | `docker build -t iros2026/ai_module:latest`. |
| `ai_module/docker/run.sh` | Standalone `docker run` with X11 + optional GPU. |

Package name is `dummy_vlm`; the executable/node name is `dummyVLM` (`package.xml:2`,
`CMakeLists.txt:2,45`, `dummyVLM.cpp:246`).

### Node structure (`dummyVLM.cpp`)

Single node `dummyVLM`, created in `main` (`:245–246`). Not a multi-node/composable setup —
plain `spin_some` polling loop at 100 Hz (`rclcpp::Rate rate(100)` `:274`; inner nav loop also
100 Hz `:130,165`). Params declared/read `:248–254`:
`waypoint_file_dir`, `object_list_file_dir`, `waypointReachDis` (default `1.0`).

Depends (CMakeLists `:11–19`): `rclcpp std_msgs sensor_msgs nav_msgs geometry_msgs tf2 tf2_ros
tf2_geometry_msgs visualization_msgs`. Note `sensor_msgs` / `message_filters` are linked/declared
but the dummy never actually subscribes to camera/lidar — it only uses odometry + question.

### Topic contract (dummy node)

| Direction | Topic | Type | QoS depth | Code ref |
|-----------|-------|------|-----------|----------|
| Sub | `/state_estimation` | `nav_msgs/msg/Odometry` | 5 | `:256`, handler `:230–234` |
| Sub | `/challenge_question` | `std_msgs/msg/String` | 5 | `:257`, handler `:237–241` |
| Pub | `/way_point_with_heading` | `geometry_msgs/msg/Pose2D` | 5 | `:259` |
| Pub | `selected_object_marker` | `visualization_msgs/msg/Marker` | 5 | `:262` |
| Pub | `/numerical_response` | `std_msgs/msg/Int32` | 5 | `:265` |

QoS is just the integer depth (default reliable/volatile profile) — no explicit `QoSProfile`.
**Gotcha:** the marker topic is created *without* a leading slash (`"selected_object_marker"`
`:262`) whereas the README documents `/selected_object_marker`. Under a default namespace these
resolve to the same fully-qualified name, but keep it in mind. The other two pubs and both subs
use explicit leading slashes.

### How the dummy forms its "answers"

The dummy does **no perception**. On startup it reads two files (`readWaypointFile` `:38`,
`readObjectListFile` `:89`) and then branches purely on the leading text of the question string
(`main` loop `:279–307`):

- **Question starts with `Find`/`find`** (`:288`): publish the hard-coded object as a CUBE Marker
  (`pubObjectMarker` `:179`) **and** send one waypoint at the object center
  (`pubObjectWaypoint` `:169`, sets `theta=0`).
- **Question starts with `How many`/`how many`** (`:292`): delete any marker, then publish a
  **random int `(rand()%10)+1`** i.e. 1–10 (`:294`) on `/numerical_response`. This is genuinely
  random — no counting.
- **Anything else** (treated as instruction-following/navigation, `:297–301`): delete marker,
  then `pubPathWaypoints` (`:127`) walks the canned `waypoints.ply` list, publishing the next
  waypoint each time the vehicle gets within `waypointReachDis` (1.0 m) of the current one,
  driven by `/state_estimation` position (`:150–162`).

After handling, `question.clear()` (`:304`) so it waits for the next one. The object record
(`object_list.txt`) format is:
`ID  midX midY midZ  L W H  heading  "label"` (parsed `:99–122`; label is the quoted trailing
string, spaces allowed). Marker uses `frame_id="map"`, `type=CUBE`, `ns=label`, `id=objID`,
pose from mid-point + RPY(0,0,heading), `scale = (L,W,H)`, color blue `a=0.5` (`:183–204`).

### How the dummy is launched

`ros2 launch dummy_vlm dummy_vlm.launch` (`docker/README.md`). The launch file
(`dummy_vlm.launch`) sets `waypoint_file_dir` → `$(find-pkg-share dummy_vlm)/data/waypoints.ply`,
`object_list_file_dir` → `.../data/object_list.txt`, `waypointReachDis=1.0`.

---

## 3. Test-time I/O contract

Authoritative table: `README.md:105–111` (system → AI) and `:122–126` (AI → system).
**Only these system-output topics may be used at test time** (`README.md:114`).

### System outputs (available to the AI module)

| Message | Topic | Type | Rate | Frame | Notes | Ref |
|---------|-------|------|------|-------|-------|-----|
| 360 camera image | `/camera/image` | `sensor_msgs/msg/Image` | 10 Hz | `camera` | **1920×640**, 360° HFOV, 120° VFOV (panoramic). README writes "1920/640 resolution". | `README.md:107` |
| Registered scan | `/registered_scan` | `sensor_msgs/msg/PointCloud2` | 5 Hz | `map` | Lidar registered into the map frame by state estimation. | `README.md:108` |
| Sensor scan | `/sensor_scan` | `sensor_msgs/msg/PointCloud2` | 5 Hz | `sensor_at_scan` | Raw lidar in the per-scan sensor frame. | `README.md:109` |
| Local terrain map (5 m) | `/terrain_map` | `sensor_msgs/msg/PointCloud2` | 5 Hz | `map` | Terrain-analysis output, 5 m around vehicle. | `README.md:110` |
| Local terrain map (20 m) | `/terrain_map_ext` | `sensor_msgs/msg/PointCloud2` | 5 Hz | `map` | Extended terrain map, 20 m around vehicle. | `README.md:110` |
| Sensor pose / odometry | `/state_estimation` | `nav_msgs/msg/Odometry` | 100–200 Hz | map → sensor | Vehicle pose; dummy reads `pose.pose.position.x/y`. | `README.md:111`, `dummyVLM.cpp:230–234` |

### Terrain map message type & semantics (RESOLVED — from submodule source)

- **Type / frame / rate / range:** `sensor_msgs/PointCloud2` of `pcl::PointXYZI`, `map` frame,
  ~5 Hz. `/terrain_map` covers 5 m around the vehicle; `/terrain_map_ext` covers 20 m.
  Source: `terrainAnalysis.cpp:267,683–684` (publishes `/terrain_map`, frame_id `"map"`);
  `terrainAnalysisExt.cpp:227,559` (publishes `/terrain_map_ext`, frame_id `"map"`).
- **Per-point `intensity` = obstacle height above the local ground, in metres.**
  `terrainAnalysis.cpp:606` sets `terrainCloudElev->points[..].intensity = disZ;` where
  `disZ = point.z − planarVoxelElev[...]` (`:596–597`) = the point's height above the estimated
  ground-plane elevation of its planar voxel. So **intensity ≈ 0 → ground/traversable, larger
  intensity → taller obstacle.** No-data / edge cells (when `noDataObstacle` is on) are stamped
  `intensity = vehicleHeight` (`:659`) to mark them as blocking.
- **How the stack thresholds it (the traversability rule you can reuse):** the `waypointConverter`
  splits the terrain cloud with a single cutoff — `intensity < obstacleHeightThre` → traversable
  area, else obstacle (`waypointConverter.cpp:305–309`). In terrainAnalysis the internal obstacle
  cutoff is `obstacleHeightThre = 0.2` m (`terrainAnalysis.cpp:53`); the converter's own default
  is `obstacleHeightThre = 0.05` m (`waypointConverter.cpp:41`) but it is overridden by the launch
  param. **Practical takeaway for us: treat a `/terrain_map` point with `intensity` below ~0.1–0.2 m
  as floor, above as obstacle.**
- **Resolution / voxelization:** raw scan is downsized at `scanVoxelSize = 0.05` m; ground
  estimation runs on a planar voxel grid of `planarVoxelSize = 0.2` m over a 51×51 window; terrain
  is accumulated in `terrainVoxelSize = 1.0` m voxels over a 21×21 window
  (`terrainAnalysis.cpp:36,65,68,73,74`). `terrain_analysis_ext` uses `terrainVoxelSize = 2.0` m
  (`terrainAnalysisExt.cpp:58`). Other useful defaults: `vehicleHeight = 1.5` m, `minRelZ = −1.5`,
  `maxRelZ = 0.2` (`terrainAnalysis.cpp:57,60,61`).
- Both terrain nodes subscribe to `/registered_scan` (PointCloud2) + `/state_estimation` and also
  a `/joy` + a clearing topic (`/map_clearing` for base, `/cloud_clearing` for ext)
  (`terrainAnalysis.cpp:259–265`, `terrainAnalysisExt.cpp:217–225`).
- Consistent with `README.md:114`: no separate traversability grid or GT-semantics topic is
  published at test time; traversability is inferred from these terrain clouds' intensity.

### AI module outputs (the three answer topics)

| Message | Topic | Type | Scored question type | Ref |
|---------|-------|------|----------------------|-----|
| Numerical response | `/numerical_response` | `std_msgs/msg/Int32` | Numerical (/1) | `README.md:98,126,173` |
| Selected object marker | `/selected_object_marker` | `visualization_msgs/msg/Marker` | Object Reference (/2) | `README.md:99,125,174` |
| Waypoint with heading | `/way_point_with_heading` | `geometry_msgs/msg/Pose2D` | Instruction-Following (/6) | `README.md:100,124,175` |

Semantics observed from the dummy (the reference producer):

- **`/numerical_response`** — `Int32.data` = the integer answer (`dummyVLM.cpp:220–227`).
- **`/selected_object_marker`** — `visualization_msgs/Marker` with:
  `header.frame_id = "map"`, `header.stamp = now`, `ns = object label`, `id = objID`,
  `action = ADD` (and `DELETE` to clear, `:207–218`), `type = CUBE`,
  `pose.position = (midX, midY, midZ)`, `pose.orientation = quaternion from RPY(0,0,heading)`,
  `scale = (L, W, H)` = bounding-box extents, color blue `a=0.5` (`:183–204`). Evaluation scores
  by **overlap of this box with the GT object box** (`README.md:174`); the box **center is also
  used as a navigation waypoint** (`README.md:53`).
- **`/way_point_with_heading`** — `geometry_msgs/Pose2D` with `x`, `y` in the `map` frame and
  `theta` = heading (radians, standard REP-103 CCW-from-+x). **The README explicitly says to
  "neglect the heading for this year's challenge"** (`README.md:100`), and the dummy sets
  `theta=0` for object waypoints (`:175`) / copies the file heading (0) for nav waypoints
  (`:143,160`). So for our module, heading can be published as 0. For instruction-following the
  answer is a **sequence** of Pose2D waypoints advanced as the robot reaches each (dummy pattern
  `:127–167`; scoring is over the actual followed trajectory, `README.md:175`).

**Frame note:** all three outputs are consumed in `map`. Waypoints out of traversable area are
snapped into it by the system (`README.md:118`). Numerical responses are **not** used to drive the
robot — only read by the evaluation node (`README.md:118`).

### Question input topic

| Message | Topic | Type | Rate | Ref |
|---------|-------|------|------|-----|
| Challenge question | `/challenge_question` | `std_msgs/msg/String` | **1 Hz (republished)** | `README.md:164–168` |

The evaluation node publishes **one** question per launch, **repeatedly at 1 Hz** (`README.md:168`).
The dummy latches the first non-empty string and clears it after answering (`dummyVLM.cpp:282,304`),
so re-publishes after that are ignored by the dummy — our module must tolerate the same string
arriving every second.

---

## 4. Launch & run mechanics

- **Two containers, one compose stack** (`docker/compose.yml`, `docker/README.md`):
  - `iros2026_system` — image `zhangjicmu/ubuntu24_ros:cmu_vla_challenge_simulation`; the
    Unity simulator + base autonomy stack. `privileged`, `network_mode: host`, X11 mounts,
    `/dev/input` + `/dev/bus/usb` mounted (joystick/USB).
  - `iros2026_ai_module` — built from `../ai_module` using `ai_module/docker/Dockerfile`;
    `network_mode: host`. Shares the ROS2 graph with the system via host networking.
- **Bring up:** `cd docker && docker compose -f compose.yml up --build -d`
  (GPU variant `compose_gpu.yml`). `xhost +` first for RVIZ (`docker/README.md`).
- **Start base autonomy** (inside system container):
  `/home/docker/autonomy_stack_mecanum_wheel_platform/system_simulation.sh` (`docker/README.md`).

#### The `system_*.sh` launch scripts (RESOLVED — submodule root)

All nine live at the submodule root. Each `source ./install/setup.bash`, then differ as below.
The `simulation` scripts additionally spawn the Unity binary
`./src/base_autonomy/vehicle_simulator/mesh/unity/environment/Model.x86_64 &` + `sleep 3` before
the ros2 launch, then start an RVIZ config (`system_simulation.sh:8–12`).

| Script | ROS2 launch it runs | Planner started | RVIZ config |
|--------|--------------------|-----------------|-------------|
| `system_simulation.sh` | `vehicle_simulator system_simulation.launch` | **none** (base autonomy only) | `vehicle_simulator.rviz` |
| `system_simulation_with_exploration_planner.sh` | `..._with_exploration_planner.launch` | **TARE** | `tare_planner_ground.rviz` |
| `system_simulation_with_route_planner.sh` | `..._with_route_planner.launch` | **FAR** | `far_planner default.rviz` |
| `system_bagfile*.sh` | `system_bagfile*.launch` | none / TARE / FAR | as above | (replays a recorded bag instead of launching Unity; runs SLAM `arise_slam_mid360`) |
| `system_real_robot*.sh` | `system_real_robot*.launch` | none / TARE / FAR | as above | (real robot: starts Livox `mid360` driver + SLAM instead of Unity) |

**What `system_simulation.launch` starts** (`src/base_autonomy/vehicle_simulator/launch/system_simulation.launch:140–149`):
`local_planner`, `terrain_analysis`, `terrain_analysis_ext`, `ros_tcp_endpoint`
(the Unity↔ROS bridge, port 10000 — `:57–65`), `sim_image_repub` (decompresses
`/camera/image/compressed` → `/camera/image`, `:67–79`), `vehicle_simulator`,
`sensor_scan_generation`, **`waypoint_converter`**, `visualization_tools`, `joy`.
**Note it does NOT start TARE or FAR** — the base config is what the challenge runs (see the
"Exploration & route planners" section for why this matters).

- **Start the AI module** (inside ai_module container):
  `ros2 launch dummy_vlm dummy_vlm.launch`.

#### Scene selection & the Unity bridge (RESOLVED)

- **The scene IS the Unity binary.** `system_simulation.sh` launches
  `src/base_autonomy/vehicle_simulator/mesh/unity/environment/Model.x86_64`. That per-scene
  executable is downloaded (from the Google-Drive scene models) and dropped into
  `.../mesh/unity/environment/`; it is **not** in git (the folder ships only `readme.txt`;
  `Model.x86_64` is absent locally). To change scene you replace the `environment/` contents with
  a different scene's build (`README.md:92`, `.../mesh/unity/readme.txt`). The `environment/`
  folder also carries the scene's `map.ply`, `object_list.txt`, `traversable_area.ply`,
  `AssetList.csv`, `Dimensions.csv`, `Categories.csv` (per `mesh/unity/readme.txt`) — but note
  those GT files are **not provided at challenge test time**.
- **`world_name` launch arg = `unity`** by default in `system_simulation.launch`
  (`:24`, `DeclareLaunchArgument('world_name', default_value='unity')`). It is passed to
  `visualization_tools` (`:111–113`). Inside `vehicle_simulator.launch` a separate `world_name`
  defaults to `garage` (`:56`) and is only used to pick a `world/<world_name>.world` collision
  file (`:10–13`) — the actual visuals/semantics come from the Unity binary, not the `.world`.
- **Camera resolution (1920×640, 360°/120°) is fixed inside the Unity build**, not in any ROS
  launch param — `sim_image_repub` only republishes/decompresses, it does not resize
  (`system_simulation.launch:67–79`). So the resolution figures come from `README.md:107`.
- **Sim ↔ ROS bridge:** it is **not** raw DDS from Unity. The Unity sim talks to ROS via the
  `ros_tcp_endpoint` node (`ROS-TCP-Connector`), `ROS_IP=0.0.0.0`, `ROS_TCP_PORT=10000`
  (`system_simulation.launch:57–65`). ROS-side traffic then uses **CycloneDDS**
  (`notes.txt`: install `ros-jazzy-rmw-cyclonedds-cpp`, export
  `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp`). Both containers must agree on RMW; host networking
  carries the DDS traffic.
- **Timing / eval hooks:** not present in this repo's visible code. The 10-minute-per-question
  timer and scoring live in the closed-source `challenge_evaluation_node`
  (`README.md:162–182`). **UNRESOLVED by design** — source is not public.

---

## 5. Docker

Two independent docker layers:

**A) Top-level `docker/` (runs system + ai together)**
- `compose.yml` / `compose_gpu.yml` — services `system` and `ai_module` (see §4).
- GPU flag = `deploy.resources.reservations.devices` with `driver: nvidia, count: all,
  capabilities: [gpu]` (compose_gpu only).
- `docker/README.md` — install Docker / Nvidia Container Toolkit, clone with
  `--recurse-submodules`, bring the stack up, launch both halves, and example
  `ros2 topic pub` commands for the three question types.

**B) `ai_module/docker/` (builds just our module)**
- Base image `zhangjicmu/ubuntu24_ros:ai_module` (prebuilt workspace).
  `Dockerfile` copies `src/dummy_vlm` into `/home/docker/ai_module/src/dummy_vlm`, then
  `colcon build --packages-select dummy_vlm` (Release) and appends
  `source /home/docker/ai_module/install/setup.bash` to `.bashrc`. Only `dummy_vlm` is rebuilt.
- `build.sh` → `docker build -t iros2026/ai_module:latest -f docker/Dockerfile .`
  (build context is `ai_module/`, so `COPY src/dummy_vlm` resolves).
- `run.sh` → standalone `docker run` with `--gpus all` when `nvidia-smi` exists, `--privileged`,
  `--network=host`, X11 env (`DISPLAY`, `QT_X11_NO_MITSHM`, `XAUTHORITY`), volume mounts for
  `/tmp/.X11-unix` and `/etc/localtime`. Runs `xhost +` first.

**Image names/tags:**
`zhangjicmu/ubuntu24_ros:cmu_vla_challenge_simulation` (system),
`zhangjicmu/ubuntu24_ros:ai_module` (ai base),
`iros2026/ai_module:latest` (our built image).

**Submission / evaluation run:** submit a public GitHub repo (fork this one); only `ai_module/`
may change; if the image changed, push it to Docker Hub and submit the link (`README.md:154`).
Evaluators pull the repo/image and run it against 3 held-out Unity scenes exactly as the local
compose stack does (`README.md:162`). Docker size limit depends on the eval machine (a Simply NUC
i9) (`README.md:200`). GPU on the real-robot final round: RTX 4090 (`README.md:141`).

---

## 5b. Exploration & route planners (TARE / FAR) — can we drive them?

**Bottom line: at challenge test time the built-in global planners are NOT running.** The
challenge launches the *base* config (`system_simulation.sh` → `system_simulation.launch`), which
starts `local_planner` + terrain analysis + **`waypoint_converter`** but **neither TARE nor FAR**
(`system_simulation.launch:140–149`; no planner in the `add_action` list). TARE and FAR only come
up under the separate `*_with_exploration_planner` / `*_with_route_planner` scripts+launches
(`system_simulation_with_exploration_planner.launch:154` adds `start_tare_planner`;
`..._with_route_planner.launch:154` adds `start_far_planner`). `docker/README.md` tells teams to
run plain `system_simulation.sh`. **So we must supply our own exploration/waypoint logic — the
robot only moves toward waypoints we publish.**

### The waypoint plumbing (this is the interface we actually use)

```
ai_module  --/way_point_with_heading (Pose2D)-->  waypointConverter
waypointConverter  --/way_point (PointStamped, "map")-->  localPlanner
localPlanner  --/path (nav_msgs/Path)-->  pathFollower  --/cmd_vel-->  vehicle
```

- `waypointConverter` (`waypoint_converter/src/waypointConverter.cpp`):
  - **Sub** `/way_point_with_heading` `geometry_msgs/Pose2D` (`:381`) — our output.
  - **Sub** `/state_estimation` (`:379`), **Sub** `/terrain_map` (`:383`).
  - **Pub** `/way_point` `geometry_msgs/PointStamped`, frame `map` (`:388–390`) — feeds the local
    planner. Also **Pub** `/speed` `Float32` (`:395`, default `speed = 0.875` m/s `:51`),
    `/way_point_reached` `Float32` (`:385`), `/traversable_area`, `/navigation_boundary`.
  - It re-projects the waypoint each cycle and, if `waypointTravAdj` is on, snaps an
    out-of-traversable waypoint to the nearest traversable terrain point (the "waypoints get
    snapped in" behaviour, `:196–246`, `:305–309`). This is the exact mechanism behind
    `README.md:118`.
- `localPlanner` (`local_planner/src/localPlanner.cpp`): **Sub** `/way_point`
  `geometry_msgs/PointStamped` (`:616`) + `/state_estimation` (`:608`); **Pub** `/path`
  `nav_msgs/Path` (`:629`). `pathFollower.cpp` subscribes `/path`, `/speed`, `/stop`, `/joy`
  and publishes `/cmd_vel` `geometry_msgs/TwistStamped` (`:290–302`). This local collision-avoiding
  follower IS running in the base config and is what makes our waypoints move the robot.

### TARE (exploration planner) — topics, and why we can't just invoke it

`tare_planner` (`exploration_planner/tare_planner/src/sensor_coverage_planner/sensor_coverage_planner_ground.cpp`):
- **Trigger:** `Sub` `/exploration_start` `std_msgs/Bool` (`:392`, param `sub_start_exploration_topic_`
  default `/exploration_start` `:27–28`). In the stack this Bool is fired by the RVIZ
  "Resume Navigation to Goal" button (`README.md:99`), not an allowed test-time topic.
- **Output:** `Pub` `/way_point` `geometry_msgs/PointStamped` (param `pub_waypoint_topic_` default
  `/way_point` `:51,451–452,653`) — i.e. TARE drives the robot by emitting the *same* `/way_point`
  that the local planner consumes. It also publishes `global_path`, `local_path`,
  `exploration_path` (`nav_msgs/Path`, `:439–450`) and `/exploration_finish` Bool (`:453`).
- Subscribes to `/registered_scan`, `/terrain_map`, `/terrain_map_ext`, `/state_estimation`,
  navigation-boundary polygons (`:397–429`).
- **Relevance to us:** TARE would frontier-explore for us, but (a) it is not launched in the
  challenge config, (b) it is started via `/exploration_start` (an RVIZ/interactive trigger, not a
  System-Output topic we are allowed to use at test time), and (c) it needs the OR-Tools binary.
  We cannot rely on it. If we want autonomous coverage we implement our own frontier logic and
  emit `/way_point_with_heading`.

### FAR (route planner) — topics

`far_planner` (`route_planner/far_planner/src/far_planner/far_planner.cpp`):
- **Trigger:** `Sub` `/goal_point` `geometry_msgs/PointStamped` (`:30`) — set by the RVIZ
  "Goalpoint" button (`utilities/goalpoint_rviz_plugin/src/goalpoint_tool.cpp:42`,
  `README.md:86`).
- **Output:** `Pub` `/way_point` `geometry_msgs/PointStamped` (`:34`) + `/far_reach_goal_status`
  `std_msgs/Bool` (`:43`). Same story: FAR drives via `/way_point`, so it also is not running and
  is triggered by an interactive topic.
- FAR is a visibility-graph global route planner (goal-directed, not coverage). Not applicable
  unless a future config launches it.

### Verdict for the ai_module implementer

- We **cannot** hand exploration to the built-in planners at test time — they aren't in the base
  launch and their triggers (`/exploration_start`, `/goal_point`) are interactive, not in the
  allowed System-Outputs list. **Build our own exploration/frontier planner.**
- Our only actuation channel is `/way_point_with_heading` (Pose2D) → `waypointConverter` →
  `/way_point` → `localPlanner`/`pathFollower`. The local planner already does terrain-aware
  collision avoidance and speed control for us; we just feed reachable 2-D goal points.
- **Do NOT publish directly to `/way_point`** to try to bypass the converter: `/way_point` is a
  base-autonomy internal topic and is not in the allowed test-time output list (`README.md:122–126`
  lists only `/way_point_with_heading`, `/selected_object_marker`, `/numerical_response`).

---

## 6. `questions/` folder

- **Layout:** one folder per scene named exactly by the scene slug (e.g. `questions/arabic_room/`)
  containing `questions.pdf` (human-readable questions + answer images) and two target-trajectory
  point clouds `trajectory_q4.ply`, `trajectory_q5.ply` (the two instruction-following questions'
  reference paths). All 15 scenes' questions also live consolidated in `questions/questions.json`.
- **`trajectory_q*.ply`** = ascii PLY, `element vertex N`, `property float x/y/z`; dense sampled
  path points (e.g. `arabic_room/trajectory_q4.ply` has 792 vertices at z≈0.75). These are GT
  trajectories for the two instruction-following questions (indexed q4, q5).
- **`questions.json` schema:** a top-level JSON **array** of 15 objects, each:
  ```json
  {
    "scene": "<scene_slug>",
    "questions": {
      "numerical": ["<one question>"],
      "object_reference": ["<q1>", "<q2>"],
      "instruction_following": ["<q1>", "<q2>"]
    }
  }
  ```
  So each scene has **5 questions**: 1 numerical, 2 object_reference, 2 instruction_following —
  matching the /1, /2, /6 scoring buckets and the q4/q5 trajectory files (q1=numerical,
  q2/q3=object_reference, q4/q5=instruction_following, by position).
- **Verbatim example record** (`questions/questions.json`, first element):
  ```json
  {
    "scene": "arabic_room",
    "questions": {
      "numerical": ["How many sofas are below a window?"],
      "object_reference": ["Find the pillow closest to the book on the stool.", "Find the wall lamp that is between a door frame and a window."],
      "instruction_following": ["Go near the stool under the picture and stop at the small table farthest from the columns.", "First, go to the potted plant furthest from the hookah, then take the path between the two columns, and stop at the tray on the table."]
    }
  }
  ```
- **The 15 training scenes** (from `questions.json`): `arabic_room`, `chinese_room`,
  `home_building_1`, `home_building_2`, `hotel_room_1`, `hotel_room_2`, `japanese_room`,
  `livingroom_1..4`, `loft`, `office_1`, `office_2`, `studio`. Most are single rooms; the
  `home_building_*` are multi-room. **3 additional scenes are held out for test evaluation** and
  are NOT in this repo (`README.md:37,80`). Total challenge scene count = 18.

---

## 7. Gotchas for the `ai_module` implementer

1. **Relaunch per question.** The whole system is **relaunched for every language command**;
   no scene memory carries over (`README.md:162`). Our module cannot cache a map across questions
   — each launch is a fresh 10-minute exploration + answer.
2. **10-minute hard budget per question**, timed from system startup (includes re-exploration).
   Overrun = penalty; finishing early = tiebreak bonus (`README.md:180–182`).
3. **Question republished at 1 Hz.** The single question arrives every second until the run ends
   (`README.md:168`). Latch-and-clear like the dummy (`dummyVLM.cpp:282,304`) or dedupe.
4. **Heading is ignored this year.** Publish `Pose2D.theta` as 0 for `/way_point_with_heading`;
   only x,y matter (`README.md:100`).
5. **Marker topic name has no leading slash in the dummy** (`selected_object_marker`,
   `dummyVLM.cpp:262`) but README uses `/selected_object_marker`. Publish on the fully-qualified
   `/selected_object_marker` to be safe (matches README + eval).
6. **Frames:** everything the planner consumes is in `map`. Camera is its own `camera` frame,
   raw lidar is `sensor_at_scan`, registered scan + terrain maps are already in `map`. Odometry
   is map→sensor. Waypoints/markers must be `map`-frame.
7. **Camera is a 1920×640 panorama** (360° H / 120° V), 10 Hz — not a pinhole. Any VLM must handle
   the equirectangular-style wide image, not a normal FOV frame (`README.md:107`).
8. **No traversability grid, no GT semantics** at test time (`README.md:114`). Only the 6 output
   topics in §3 are legal inputs; terrain must be inferred from `/terrain_map(_ext)` point clouds.
9. **Terrain map is a PointCloud2 of XYZI, not an OccupancyGrid.** `intensity` = obstacle height
   above local ground in metres; floor ≈ 0, obstacle > ~0.1–0.2 m
   (`terrainAnalysis.cpp:606`, `waypointConverter.cpp:305`). Use it directly for traversability.
10. **Waypoints get snapped into traversable area** by `waypointConverter` if you send an
    infeasible one (`waypointConverter.cpp:196–246,305–309`; `README.md:118`) — a waypoint
    reaching the object works even if slightly off.
11. **Built-in TARE/FAR planners are OFF in the challenge config** — the base
    `system_simulation.launch` doesn't start them, and their triggers (`/exploration_start`,
    `/goal_point`) aren't allowed test-time topics. Roll your own exploration; actuate only via
    `/way_point_with_heading`. Do **not** publish `/way_point` directly (internal, not allowed).
    See section 5b.
12. **Only `ai_module/` is editable for submission.** Do not touch the system container or
    submodule; if you change the docker image, push it and submit the Docker Hub link
    (`README.md:154`).
13. **Middleware is CycloneDDS + a Unity ROS-TCP bridge** on port 10000
    (`system_simulation.launch:57–65`). Keep `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp` consistent or
    the two containers won't discover each other (`notes.txt`).
14. **Waypoints should be near the vehicle.** The stack warns that a waypoint set too far can make
    the vehicle get stuck at a dead end (`autonomy_stack .../README.md:55`) — emit incremental
    reachable goals, not one distant target.

---

## UNRESOLVED items

- **U1. RESOLVED** — terrain-map intensity = obstacle height above ground in metres; traversable
  cutoff ~0.1–0.2 m (`terrainAnalysis.cpp:596–606`, `waypointConverter.cpp:305`,
  `terrainAnalysis.cpp:53`). See "Terrain map message type & semantics".
- **U2. RESOLVED** — all nine `system_*.sh` scripts and `system_simulation.launch` node lists
  documented in section 4. Base sim starts local_planner + terrain analysis (×2) +
  ros_tcp_endpoint + sim_image_repub + vehicle_simulator + sensor_scan_generation +
  waypoint_converter + visualization_tools + joy; no TARE/FAR.
- **U3. RESOLVED (with one residual)** — scene = per-scene Unity binary `Model.x86_64` dropped in
  `.../mesh/unity/environment/`; `world_name=unity` launch arg
  (`system_simulation.launch:24`). Residual: the Unity binary is downloaded, not in git, so how
  the *evaluators* swap the 3 held-out scene binaries in is inferred (replace `environment/`
  contents) — the swap script itself is not in the repo.
- **U4.** `challenge_evaluation_node` internals — timing enforcement, scoring math, how/when it
  reads the three answer topics. Source is deliberately private (`README.md:162`). Still
  unresolvable.
- **U5.** Whether `/way_point_with_heading` heading, if ever re-enabled, is degrees or radians —
  `waypointConverter` treats `Pose2D.theta` as **radians** (fed straight into trig, e.g.
  `waypointConverter.cpp:176–194`), so if re-enabled it is radians; but README says neglect it and
  the dummy sends 0.
- **U6.** The camera 1920×640 / 360°×120° figures are set inside the closed Unity build, not in
  any ROS param — taken from `README.md:107` (cannot be code-verified without the Unity binary).
