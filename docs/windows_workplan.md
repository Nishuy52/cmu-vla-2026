# Windows Workplan — what to do before Ubuntu is available

*The sim needs Ubuntu 24.04 + ROS Jazzy + Docker. Windows cannot run the full loop natively, but ~70% of the winning work is sim-independent. This doc maximises that.*

## Design principle that makes Windows work possible

**Split `ai_module` into an OS-independent core and a thin ROS adapter.**

```
ai_module/
  core/            # pure Python — runs anywhere, unit-testable on Windows
    perception/    # detection, lidar-camera fusion, 3D object mapping
    scene_graph/   # objects + spatial relations
    reasoning/     # question parsing, LLM calls, answer synthesis
    exploration/   # frontier policy over terrain map + odometry
    interfaces.py  # dataclasses mirroring the ROS messages (Image, PointCloud2, Pose2D...)
  ros_adapter/     # ROS 2 node: subscribes topics -> core dataclasses -> publishes answers
                   # written on Windows, only *executed* on Ubuntu/cluster
```

Everything in `core/` is developed and tested on Windows against mock/recorded data. The adapter is ~100 lines and gets validated in Phase 2.

## Fully Windows-viable tasks (do now)

1. **Study the upstream clone** (`upstream/CMU-VLN-Challenge-2026`) — reading code needs no ROS.
2. **Question analysis** — `questions/*.json` for 15 scenes; pure Python.
3. **VLA-3D dataset work** — download, parse, build few-shot examples / fine-tuning data; pure Python + PyTorch (CPU or your local GPU if any).
4. **Core pipeline development** — perception/scene-graph/reasoning modules with pytest; mock the sensor inputs with samples from the repo's real-robot sample-data Drive folder (bag files can be parsed with `rosbags` pip package — no ROS install needed).
5. **Prompt engineering** — the LLM reasoning module is API calls; iterate on Windows with saved scene-graph fixtures.
6. **Offline eval harness** — score answers against training-scene ground truth.
7. **Cluster work over SSH** — training jobs, Linux smoke-tests of the core (see `soc_cluster_guide.md`). The cluster is your Linux machine until you're home.
8. **All documentation.**

## Stretch: WSL2 (attempt, don't depend on it)

Worth one time-boxed evening. Windows 11 + WSL2 (Ubuntu 24.04) + Docker Desktop gives you ROS Jazzy containers, and WSLg + NVIDIA GPU passthrough *sometimes* runs GUI/GPU sim workloads.

```powershell
wsl --install -d Ubuntu-24.04    # then install Docker Desktop with WSL2 backend
```

- Likely works: building the container, running ROS nodes headless, `colcon build` of ai_module, replaying bags.
- Unproven: Unity sim rendering performance under WSLg. If it stutters or crashes, stop — the repo offers no WSL support and eval is native Linux anyway.
- If it *does* work, Phase 2 effectively starts early.

## Blocked until Ubuntu (park these)

- Full closed-loop sim runs, timing measurements against the 10-min budget
- Final Docker image build/verification exactly-as-eval
- Real waypoint-following behaviour tuning

## Local environment setup (Windows, now)

```powershell
winget install Python.Python.3.12 Git.Git    # if missing
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install numpy opencv-python torch rosbags pytest anthropic open3d
```
