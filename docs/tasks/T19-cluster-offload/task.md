# T19 — cluster-offload

**Started:** 25 Jul 2026
**Intent:** Add an opt-in, dev-only path that offloads the GroundingDINO
detector and the local-LLM tier to the NUS SoC Slurm cluster over SSH
tunnels, so live sim iteration on the RTX 4060 Laptop no longer has to
trigger the power-management wedge or contend with the local LLM for VRAM.
Unity + the ROS stack stay local throughout; only detection and local-LLM
inference move off-box.

## Context

- Issue #86: the RTX 4060 Laptop's GPU recurrently wedges into P8 @ 210 MHz
  under runtime power management, and the wedge reliably fires the moment
  GroundingDINO loads its weights — killing whatever live sim run was in
  progress, recoverable only by a reboot.
- Issue #82: separately diagnosed the local Ollama LLM tier contending for
  the same VRAM as the detector on this box.
- Fairshare-aware amendment (adopted before implementation): the cluster job
  must be start-on-demand / torn-down-after, not left resident — a single
  combined GPU allocation runs both servers together (SwinB fp32 ~1.3 GiB +
  qwen2.5vl:7b ~7 GB comfortably fit one 24 GB Titan RTX), and all one-time
  downloads (weights, ollama binary, model pulls) run on CPU partitions, not
  billed GPU time.
- Submission builds never use this path — the scored image keeps running the
  local `GroundingDinoDetector` (`docs/ubuntu_setup.md` §7) and the in-image
  Ollama bake (§8a) unchanged.

## Acceptance criteria

- [x] `src/core/perception/remote_detector.py`: `RemoteDetector` mirrors the
      GDINO dual-pass scheduling (issue #42) and the issue #39 backoff
      exactly; any network failure degrades to empty detections for that
      tick, never crashes the adapter loop.
- [x] `src/ros_adapter/adapter_node.py` `make_detector`: opt-in
      `VLA_DETECTOR=remote` branch, reading `VLA_REMOTE_DETECTOR_URL`
      (required) and `VLA_REMOTE_DETECTOR_TIMEOUT_S` (default `10.0`).
- [x] `tools/cluster/gdino_server.py`: cluster-side single-caption-pass HTTP
      server (`GET /health`, `POST /detect`) reusing
      `GroundingDinoDetector.run_caption_pass` directly, fp32 default on the
      24 GB Titan RTX.
- [x] `tools/cluster/servers.sbatch`: one combined GPU job running both the
      gdino server and `ollama serve` on a single `nv` allocation, with
      dynamic port selection and addr-file handshake (`~/gdino_server.addr`,
      `~/ollama_server.addr`), written only after each server answers its
      health endpoint and removed on job exit; either server dying kills the
      sibling and exits rather than billing a half-dead allocation.
- [x] `tools/cluster/servers.sh` (laptop): `start` (rsync + sbatch, extra
      sbatch args pass through), `stop` (scancels only `vla-servers` jobs),
      `status`.
- [x] `tools/cluster/tunnel.sh` (laptop): reads the addr files over SSH,
      idempotently opens the SSH port-forwards, prints the ready-to-paste
      env exports.
- [x] Tests green: fast tier (`pytest`, `src/`) and the `tools/` tier
      (`pytest tools` from repo root).
- [x] `docs/ubuntu_setup.md` updated in the same session (per CLAUDE.md
      standing rule 3) with the workflow, env vars, and the explicit
      submission-never-uses-this rule.
- [x] Live end-to-end verified against the real cluster allocation, with
      measured latencies recorded, then torn down.

## Notes

- **25 Jul:** task opened; implementation scoped as above. Cluster staging
  begun on CPU partitions per the fairshare amendment:
  - Job 697303 (env setup): FAILED — PyPI `torchvision` wheel is ABI-mismatched
    against the `torch` cu126 build already staged in `~/venv-gpu`, surfacing
    at import as `operator torchvision::nms does not exist`.
  - Job 697310 (fix): installed `torchvision==0.28.0+cu126` from the cu126
    index instead of plain PyPI — SETUP-OK.
  - Jobs 697304 / 697311 (ollama install): both hit 404s — the release URL
    pattern this repo's install notes assume (`.tgz` asset) no longer exists
    for the pinned version; `ollama` v0.32.3 ships
    `ollama-linux-amd64.tar.zst` instead.
  - Job 697315: ollama install fixed to pull the `.tar.zst` asset — OK.
  - Job 697316: `qwen2.5vl:3b` and `qwen2.5vl:7b` pulled on a CPU allocation
    (no GPU billed for the download).
  - Job 697313 (GPU verify, xgpe5): SwinB fp32 load in 7.5 s, 1.26 GiB VRAM —
    matches the sizing assumption above.
  - Job 697320 (xgpe6 probe): CUDA init failure — node confirmed broken,
    excluded going forward alongside the pre-existing `xgpe0`/`xgpe2`
    exclusions in `servers.sbatch`'s `-x` list.
- **25/26 Jul (server deploy):** two bugs found and fixed live while
  bringing the combined job up on real hardware:
  - Warmup tile was 32x32, too small for GroundingDINO's `topk(900)`
    proposal selection — failed with "selected index k out of range".
    Fixed: warmup tile enlarged to 480x640 (matches a real caption-pass
    frame size).
  - Port 11434 collided with a foreign, already-running process on shared
    node xgpe5, and the job's own health poll was a false positive against
    that foreign server (it also answers on 11434, just not ours). Fixed:
    dynamic free-port probing (>= 8765 for gdino, >= 11434 for ollama) with
    the actual bound port recorded in the addr files, so `tunnel.sh` always
    forwards to the port the job actually opened, not an assumed default.
- **26 Jul (live E2E verification, then teardown):** deploy job 697332 on
  xgpe5 (gpu partition, nv:1): gdino server (SwinB fp32, model load 21.0 s)
  on 8765, ollama on 11435 (dynamic pick — 11434 held by a foreign
  process). Through the laptop tunnel: `/health` 101.5 ms warm / 288.3 ms
  cold; one-tile `/detect` "chair ." on the livingroom_1 render (812x451)
  206.9 ms warm / 398.3 ms cold with a sane detection (label `chair`,
  score 0.473); ollama `/v1/models` 69.8 ms listing both qwen2.5vl tags.
  Live `RemoteDetector` smoke against the same server: dual-pass tick
  (question+vocab, two POSTs) 641 ms / 8 detections (chair, sofa, lamp,
  table), question-only tick 203 ms, backoff clean. Fresh-context verifier
  pass on the code: CONFIRMED (scheduling/backoff parity, additive-only
  detector.py diff, wire parity, suites green). Servers torn down
  immediately after verification (fairshare); addr files auto-removed by
  the job's exit trap. Restart on demand: `tools/cluster/servers.sh start`.
