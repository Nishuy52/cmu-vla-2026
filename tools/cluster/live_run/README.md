# Full-stack live runs on the SoC cluster

Runs the ENTIRE live loop — Unity sim (GPU-rendered headless) + ros_tcp_endpoint +
autonomy stack + our ai_module with real GroundingDINO — in one Slurm GPU allocation,
zero laptop involvement. First grounded answer achieved 2026-07-26 (job 697519).

Dev-only tooling: nothing here ships in the challenge fork.

## One-time cluster setup (already done for cyuhsin; recipe for a rebuild)

1. `~/xstack/root`: Xvfb + VirtualGL + deps, extracted from debs with `dpkg-deb -x`
   (no root). VirtualGL deb from GitHub releases; X debs via `apt-get download` on a
   COMPUTE node (the mirror is unreachable from login nodes).
2. Sim image extracted via udocker (`udocker pull/create zhangjicmu/ubuntu24_ros:cmu_vla_challenge_simulation`)
   — used only as a FILE TREE (`~/.udocker/containers/<id>/ROOT`); no container
   runtime at run time (AppArmor blocks userns; fakechroot hangs rclpy).
3. `~/venv-gpu` (torch cu126 + groundingdino-py + cv2, numpy==1.26.4),
   `~/models/gdino/` (SwinB cfg + checkpoint), `~/vla/` (rsync of src/ + tools/).

## Per-run

```
scp cluster_live_run.sbatch question_pub.py live_monitor.py <user>@xlogin:~/
ssh xlogin sbatch '~/cluster_live_run.sbatch'      # edit QUESTION= inside first
```

Outputs: `~/live_run.out` (monitor + verdict), `~/live_run_<job>_{adapter,launch,unity}.log`,
`~/live_run_<job>_screen.xwd`. `cluster_sim_smoke.sbatch` is the stack-only variant
(no ai_module) for sim-health checks; `topic_probe.py` is the rclpy graph/rate probe.

## Hard-won constraints (violate any of these and the run silently degrades)

- **CycloneDDS must be pinned to loopback** (`CYCLONEDDS_URI` in the sbatch): the
  compute fabric drops multicast; default discovery finds nothing.
- **Raw camera frames (3.7 MB) cannot cross DDS here** — unprivileged kernel socket
  buffers drop the fragments. The adapter must run `VLA_CAMERA_COMPRESSED=1`
  (in-process jpeg decode; commit fcbdea0). The stock `sim_image_repub` starves for
  the same reason.
- **Cap torch CPU threads** (`OMP_NUM_THREADS=4`) and allocate ≥16 cores, or Unity's
  frame encoding starves and the camera stream collapses once GDINO loads (v7/v8
  regression: 2 Hz → 0.1 Hz).
- **GDINO_PRECISION=fp16** on the shared card.
- **Never probe with the ros2 CLI** in scripts (daemon inherits pipes → hangs);
  use rclpy probes (`topic_probe.py` / `live_monitor.py`).
- **Exclude broken GPU nodes** `-x xgpe0,xgpe2,xgpe6`; `nv` GRES only (fairshare:
  4.0/hr vs 10–30 for A100/H100; idle allocations bill identically — tear down).

## Verification variant

`cluster_verify_run.sbatch` = the live run plus per-issue instrumentation:
`VLA_EXPLORE_DEBUG_DIR` dumps (issue #83 pocket/frontier + #84 instance index),
an in-job `~/ollama serve` with captured log + the local LLM tier enabled
(issue #82 retest; qwen2.5vl:3b for laptop parity), optional `SCENE_DIR` swap
(restart-based, resolved before Unity starts), and a verdict-summary block at
the end of `~/verify_run.out` (parse tier, last pocket/reroot records, instance
class counts, waypoint cadence). Harvest the `~/verify_run_<job>_debug/` JSONLs
alongside the logs; `tools/live_harness/compare_explore_debug.py` reads them.

## Known quality gaps (filed)

- #88 in-executor GDINO starves the tick loop (~0.8 Hz effective vs 5 Hz)
- #89 numerical instance explosion on grounded live runs (78–141 vs single-digit truth)

Scene swap: replace `.../vehicle_simulator/mesh/unity/environment/` inside the
extracted ROOT with a scene from `~/scenes/` (fetch via gdown per-file IDs in
tools/fetch_unity_scenes.sh), then rerun.
