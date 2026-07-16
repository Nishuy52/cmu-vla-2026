# CMU VLA Challenge 2026 — Brief

*Compiled 10 Jul 2026 from [github.com/Yuxin916/CMU-VLN-Challenge-2026](https://github.com/Yuxin916/CMU-VLN-Challenge-2026) and [ai-meets-autonomy.com/cmu-vln-challenge](https://www.ai-meets-autonomy.com/cmu-vln-challenge).*

## Deadlines (all midnight AoE)

| Date | Event |
|---|---|
| Apr 15 2026 | Registration opened |
| **Jul 15 2026** | **Registration deadline — REGISTER NOW** |
| **Aug 15 2026** | **Submission deadline** |
| Later 2026 | Phase 2: real-robot eval (top teams, remote access); top-3 present at IROS workshop, cash prizes (AlphaZ sponsor) |

## The Task

A wheeled robot (mecanum platform) in unknown indoor Unity scenes receives a natural-language question on ROS topic `/challenge_question` (String, republished at 1 Hz) and must respond within **10 minutes per question** (exploration + answering combined, timed from system startup; system relaunches per question — no state carries over). Early finish earns bonus points (tie-break); overtime is penalised. *(Wording corrected 2026-07-14 against the upstream README "Timing" section — "Each question has a total time limit of 10 minutes for exploration and question answering combined"; an earlier revision here said "per scene". The per-question reading, which `architecture.md` §5 already budgets for, is confirmed correct.)*

Three question types:

| Type | Example | Output topic / msg | Score |
|---|---|---|---|
| Numerical | "How many blue chairs between the table and the wall?" | `/numerical_response` — `Int32` | 0–1 |
| Object reference | "Find the potted plant on the kitchen island closest to the fridge" | `/selected_object_marker` — `visualization_msgs/Marker` (bounding box) | 0–2 |
| Instruction-following | "Take the path near the window to the fridge" | `/way_point_with_heading` — `geometry_msgs/Pose2D` waypoints | 0–6 |

Instruction-following is worth 6× a numerical question — prioritise it.

## Allowed I/O at test time (only these topics)

- **In:** 360° camera image (1920×640 @ 10 Hz), 3D lidar point cloud (5 Hz), terrain map, odometry, `/challenge_question`
- **Out:** waypoints (Pose2D), object marker, Int32 answer

Manual waypoints / teleoperation prohibited. **LLMs/VLMs/online APIs explicitly allowed** (provide access tokens at runtime if needed) — an API-driven reasoning module is legal.

## Environment & stack

- Ubuntu 24.04 + **ROS 2 Jazzy** + Unity sim, all packaged in Docker (`docker/` folder in repo)
- Eval machine spec: i9 16-core, 32 GB RAM, RTX 4090
- 18 Unity scenes (indoor residential/corporate): **15 training** (downloadable, Google Drive links in repo README) + 3 held-out test
- Questions for the 15 training scenes ship in `questions/` (PDF + JSON)
- Base autonomy stack (`autonomy_stack_mecanum_wheel_platform/`) already handles SLAM/pose, terrain analysis, collision avoidance, and waypoint following — **we only build the brain** (`ai_module/`)

## Submission

Fork the repo, modify **only `ai_module/`**, push custom Docker image to Docker Hub, submit repo link via Google Form. **Multiple submissions allowed; highest score counts** → submit early, iterate.

## Key resources

- Dummy model to replace: `ai_module/src` (ROS 2 Python node)
- **VLA-3D dataset** ([HaochenZ11/VLA-3D](https://github.com/HaochenZ11/VLA-3D)): 7.6K scenes, 9M+ referential statements — includes the 15 training scenes; training data for grounding models
- Prior art referenced by organisers: **SORT3D** (LLM spatial reasoning over 3D object maps), OpenEQA, Room-Across-Room (RxR)
- 2025 leaderboard exists on the challenge site — study what won last year
