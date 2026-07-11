> Imported 2026-07-11 from a parallel research stream. See
> `docs/prior_art/README.md` for provenance and conflict rules.

# I/O contract - verified against the cloned dev kit

Source: Yuxin916/CMU-VLA-Challenge-2026, inspected 2026-07-11 from
a shallow clone (README.md, ai_module/src/dummy_vlm, docker/,
questions/questions.json). Written pre-merge as an independent
source-inspection pass; it corroborates this repo's
docs/challenge_brief.md and docs/upstream_notes.md, which stay
authoritative for build/process. Anything still unverified is
flagged UNVERIFIED.

## The one thing that was unknown: question-type signaling

**There is no question-type label.** A single question arrives as a
raw `std_msgs/msg/String` on `/challenge_question`, published by the
hidden `challenge_evaluation_node` at 1 Hz, one question per system
launch. The dummy model infers the type purely from the string
prefix:

- starts with `Find` / `find` -> object reference
- starts with `How many` / `how many` -> numerical
- anything else -> instruction-following

This is a reference heuristic in the dummy, NOT a guaranteed
contract - real questions may not fit these prefixes (e.g. the
japanese_room object-reference questions in questions.json start
with "The lantern..." / "The red pillow...", which the prefix rule
would misroute to instruction-following). Our classifier should be
an LLM/robust parse over the question text, not the prefix trick.

## The other unknown: exploration-done / answer-ready signaling

There is **no explicit "I'm done exploring, here's my answer"
signal**. The module simply publishes to the appropriate response
topic whenever it is ready, within the 10-minute budget:

- Numerical / object reference: publish once.
- Instruction-following: stream `Pose2D` waypoints one at a time,
  closed-loop against `/state_estimation` - advance to the next
  waypoint when the vehicle gets within `waypointReachDis` (1.0 m
  in the dummy) of the current one. "Done" = last waypoint reached.

The evaluation node scores from the actual robot trajectory / final
published values; timing runs from system startup and includes
re-exploration.

## Inputs (allowed at test time)

Live streams, no pre-built map, no ground-truth semantics, no
traversable-area data:

| Topic | Type | Rate | Frame |
| --- | --- | --- | --- |
| `/challenge_question` | std_msgs/String | 1 Hz | - |
| `/camera/image` | sensor_msgs/Image | 10 Hz | camera (1920x640, 360 HFOV / 120 VFOV) |
| `/registered_scan` | sensor_msgs/PointCloud2 | 5 Hz | map |
| `/sensor_scan` | sensor_msgs/PointCloud2 | 5 Hz | sensor_at_scan |
| `/terrain_map` | sensor_msgs/PointCloud2 | 5 Hz | map (5 m) |
| `/terrain_map_ext` | sensor_msgs/PointCloud2 | 5 Hz | map (20 m) |
| `/state_estimation` | nav_msgs/Odometry | 100-200 Hz | map->sensor |

More topics exist in the simulator but only these are legal at test
time. Ground-truth semantics and traversable area are NOT provided
this year (a change from prior years).

## Outputs (one per question type)

| Question type | Topic | Type | Notes |
| --- | --- | --- | --- |
| Numerical | `/numerical_response` | std_msgs/Int32 | read by eval node only; not used for navigation |
| Object reference | `/selected_object_marker` | visualization_msgs/Marker | CUBE marker; scored by bbox overlap. Marker center doubles as a nav waypoint |
| Instruction-following | `/way_point_with_heading` | geometry_msgs/Pose2D | heading ignored this year; stream waypoints |

Marker details from the dummy: `type=CUBE`, `action=ADD`,
`frame_id=map`, `pose` = object center + heading (as a quaternion
from RPY), `scale` = L/W/H. Object reference also publishes a
`Pose2D` waypoint at the object center so the robot navigates to it
(README: "the center point of the bounding box marker will be used
as a waypoint"). Note: the dummy creates the marker publisher with
the relative name `selected_object_marker` (no leading slash) while
docs say `/selected_object_marker`; both resolve identically in the
default namespace, but keep the leading slash to be safe.

## Scoring + question distribution (corroborates question_analysis.md)

Every one of the 15 dev scenes has the same mix (verified from
questions.json): **1 numerical + 2 object reference + 2
instruction-following = 5 questions/scene.**

Per-type max: numerical /1, object reference /2, instruction /6.
If the 3 test scenes mirror this distribution (README says the test
questions are "similar to those provided" but does not guarantee
the exact mix - UNVERIFIED), the max is:

`(1*1) + (2*2) + (2*6) = 17 points/scene * 3 scenes = 51 points`

The "45 points max" figure in docs/challenge_brief.md was an
earlier estimate and is likely wrong; treat 51 as the working
number, contingent on the test-scene mix.

## Runtime / how to actually run it

- Two Docker containers, `--network=host`, sharing the ROS2 graph:
  - `iros2026_system` - prebuilt image
    `zhangjicmu/ubuntu24_ros:cmu_vla_challenge_simulation`
    (simulator + fixed autonomy stack), needs privileged + X11
    (`/tmp/.X11-unix`, `DISPLAY`, `xhost +`) for RVIZ/Unity.
  - `iros2026_ai_module` - built locally from
    `ai_module/docker/Dockerfile`, contains our model.
- GPU path uses `compose_gpu.yml` (nvidia device reservation);
  a CPU-only `compose.yml` exists but Unity rendering wants a GPU.
- Eval box: 16-core i9, 32 GB RAM, RTX 4090, Ubuntu 24.04, ROS
  Jazzy, cyclonedds RMW (`RMW_IMPLEMENTATION=rmw_cyclonedds_cpp`).
- Manual smoke test: `ros2 topic pub --once /challenge_question
  std_msgs/msg/String "{data: 'Find the ...'}"`.

**Cannot run end-to-end on macOS** (needs Linux + ROS
Jazzy + X11 + ideally an Nvidia GPU for the Unity sim). The live
sim run is deferred to a Linux+GPU machine; everything above is
from source inspection, which is the load-bearing half of Phase 0b.

## What the AI module owns (submission surface)

Only `ai_module/` may change; folder structure must be preserved.
The dummy is a C++ ament/ROS2 package (`dummy_vlm`) but nothing
forces C++ - a Python `rclpy` node is fine as long as topics and
message types match and the launch/Dockerfile are updated. Object
list format used by the dummy marker builder:
`id x y z L W H heading "label"` (space-separated).

## Still UNVERIFIED (needs the live sim / organizer confirmation)

- Exact test-scene question mix (assumed 1/2/2 -> 51 max).
- Whether questions always fit the prefix heuristic (assume not).
- Instruction-following scoring internals (constraint ordering,
  forbidden-area detection) - eval node source is not public.
- Exact time bonus/penalty formula around the 10-min limit.
- Whether the object marker must carry the correct semantic label
  string to score, or only geometric overlap (dummy sets `ns` to
  the label but scoring is described as overlap-based).
