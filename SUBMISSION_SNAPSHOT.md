# Submission snapshot

This file records the state of the branch used to build the competition
submission image, and the exact commands to build, smoke-test, and submit it.

## Pinned tree

- Commit: `253cf3c785b66f2ecc063acfd17c6e3eec8a266a`
  ("log: record the vehicle-wedge diagnosis and its fixes", 8 Aug 2026)
- Branch: `chore/submission-docker`
- Worktree: `/home/jason/cmu_ws/wt-docker`

This commit carries every verified fix through the vehicle-wedge stack
(issue #205/#207/#208 and the surrounding batch). The perception stack
(GroundingDINO detector, in-image Ollama LLM tier) is enabled in the
packaging — see the audit below.

## Packaging audit (10 Aug 2026)

Checked the fork-shaped packaging (`docker/ai_module_fork/`) against this
tree:

- `docker/ai_module_fork/docker/Dockerfile` copies the whole `src/`
  tree (`COPY --chown=docker:docker src/ /opt/vla/src/`), so this week's
  new files (`core/nav/` additions, `core/perception/fusion.py` changes,
  `ros_adapter/frame_watchdog.py`, `adapter_node.py` changes) are already
  packaged — no Dockerfile edit was needed for file coverage.
- `sync_to_fork.sh` mirrors the same tree (rsync, tests excluded) into a
  fork checkout; same result confirmed by a manual dry run of its steps.
- Full import scan of `src/core/` and `src/ros_adapter/` found no new
  third-party package beyond what the Dockerfile already installs
  (`torch`, `torchvision`, `transformers==4.57.6`, `groundingdino-py`,
  `openai>=1.0`, `numpy==1.26.4` pin, `rosbags>=0.11`). `cv2` and `PIL`
  are transitive (from `groundingdino-py`/`torchvision`) and resolved
  cleanly in a live dependency check (below).
- Perception (`torch`/`groundingdino`) is enabled, not commented out.
  `VLA_DETECTOR=grounding_dino` is set in both compose files
  (`docker/ai_module_fork/docker/compose.yml`,
  `compose_gpu.yml`), the GDINO weights and BERT tokenizer cache are
  baked at build time, and `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1`
  keep runtime offline. The in-image Ollama server + `qwen2.5vl:3b`
  blobs are baked for the local LLM tier.
- Env-var audit (grep `os.environ`/`os.getenv` across `src/`): every var
  the adapter and core read is either baked as a Dockerfile `ENV`
  (`RMW_IMPLEMENTATION`, `GDINO_CONFIG_PATH`, `GDINO_CHECKPOINT_PATH`,
  `HF_HOME`, `VLA_LLM_LOCAL_*`, `OLLAMA_*`), set in compose
  (`VLA_DETECTOR`), or an optional debug/dump var that defaults to a
  no-op when unset (`VLA_EXPLORE_DEBUG_DIR`, `VLA_*_DUMP_PATH`,
  `VLA_PERCEPTION_SYNC`). No var the scored path needs is missing.
- LLM keys (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `SOCLAAS_API_KEY`) are
  never baked, as designed — passed at run time only via `../.env.llm`
  (optional; the ladder degrades to local Ollama, then regex, if unset).
- Fixed one stale comment: `src/ros_adapter/setup.py` pointed at the
  deprecated `docker/ai_module/Dockerfile` path; corrected to
  `docker/ai_module_fork/docker/Dockerfile`.
- Fixed one stale "confirm on Ubuntu" flag in the Dockerfile (flag 4 of
  4, the Ollama runner prune): the 19 Jul build already confirmed it
  (measured ~28.1 GB image, expected runners kept); the comment still
  read "cannot run here, GPU wedged". Marked CONFIRMED with the
  measured evidence.
- No dependency, weight, or env-var gap found beyond the two comment
  fixes above — the packaging was already current for this tree.

## Verification performed (no full GPU run needed)

1. **File coverage**: every path the Dockerfile `COPY`s
   (`src/`, `launch_with_llm.sh`, `docker/Dockerfile`) exists in the
   tree; `src/tests/` and caches are excluded from the sync, confirmed
   by listing the staged tree.
2. **Dependency resolution**: `pip download --no-deps` against the
   exact pinned/constrained spec (`numpy==1.26.4`,
   `transformers==4.57.6` constraint; `torch`, `torchvision`,
   `groundingdino-py`, `openai>=1.0`, `rosbags>=0.11`, `pytest>=8`)
   resolved and downloaded cleanly from PyPI — no broken pin.
3. **Weight URLs**: `HEAD` requests against the GroundingDINO checkpoint
   release asset and the Ollama `v0.32.1` linux-amd64 tarball both
   returned `200` with content-length matching the Dockerfile's own
   size assertions (938 057 991 bytes GDINO; 1 435 963 408 bytes Ollama).
4. **Full CPU build — attempted, not completed in this session.** Ran
   `docker build -f docker/Dockerfile -t submission-audit-ai_module:test .`
   from a staged fork-shaped context (this tree's `src/` + Dockerfile
   synced into a scratch `ai_module/` directory, exactly what
   `sync_to_fork.sh` produces — the base image `FROM` line and the
   `ENV` lines resolved from cache instantly, confirming the base image
   itself is still cached locally). The build did **not** reuse the
   layer cache from the 19 Jul build past that point: the very first
   `RUN apt-get update -qq` inside the container was still running
   after 8+ minutes, where it normally completes in well under a
   minute. `curl`/`pip download` run directly on the host (outside a
   container, verification steps 2–3 above) completed normally in the
   same window, so the slow path is specific to network egress from
   inside a build container on this box, not a general outage. Given
   that one early, normally-trivial layer alone exceeded 8 minutes, the
   full bake (torch/groundingdino/GDINO weights/Ollama+model, ~7 GB of
   downloads) would not finish inside the 3000 s budget. Stopped the
   build cleanly (`TaskStop` on the background task; confirmed no
   `docker build`/`buildx`/`runc` processes remained and no partial
   image was tagged) rather than let it run unbounded.
   **Command for the main session to run instead** (same staging
   recipe, generous timeout, ideally on a shell with faster container
   egress than this sandbox):
   ```bash
   rm -rf /tmp/fork-stage
   mkdir -p /tmp/fork-stage/ai_module/docker /tmp/fork-stage/ai_module/src
   cp /home/jason/cmu_ws/wt-docker/docker/ai_module_fork/docker/Dockerfile \
     /tmp/fork-stage/ai_module/docker/Dockerfile
   cp /home/jason/cmu_ws/wt-docker/docker/ai_module_fork/ai_module/launch_with_llm.sh \
     /tmp/fork-stage/ai_module/launch_with_llm.sh
   chmod +x /tmp/fork-stage/ai_module/launch_with_llm.sh
   rsync -a --delete --exclude 'tests/' --exclude '__pycache__/' \
     --exclude '.pytest_cache/' --exclude '*.pyc' \
     /home/jason/cmu_ws/wt-docker/src/ /tmp/fork-stage/ai_module/src/
   cd /tmp/fork-stage/ai_module
   sg docker -c 'docker build -f docker/Dockerfile -t vla-ai_module:submission-audit .'
   ```
   Context dir: `/tmp/fork-stage/ai_module` (build context = that
   directory, matching the fork-shaped `ai_module/` layout). Expected
   duration: minutes if the local build cache from a prior full build
   is honored (only the `COPY src/` layer onward re-executes); up to
   ~60–90 minutes on a cold cache at the ~1.5 MB/s egress observed on
   this box, dominated by the GDINO checkpoint (938 MB), the Ollama
   tarball (1.4 GB), and the `qwen2.5vl:3b` blobs (~3.2 GB) — all three
   URLs were confirmed live and correctly sized by a direct `curl -I`
   in this session (see above), so a slow-but-successful build is the
   expected outcome, not a broken pin.

## One-command build

From a clean clone of the submission fork (the repo pushed to GitHub
with only `ai_module/` modified):

```bash
# from a checkout of THIS repo (chore/submission-docker or main once merged)
./docker/ai_module_fork/sync_to_fork.sh /path/to/fork-clone
cd /path/to/fork-clone/docker
cp /path/to/this-repo/docker/ai_module_fork/docker/compose_gpu.yml compose_gpu.yml   # or compose.yml for no GPU reservation
sg docker -c 'docker compose -f compose_gpu.yml build ai_module'
```

(`sg docker -c '...'` is only needed in a stale shell session that
predates the `docker` group add — see `docs/ubuntu_setup.md` §2. A
fresh login shell runs plain `docker compose ...`.)

## One-command smoke test

```bash
xhost +
cd /path/to/fork-clone/docker
DISPLAY=:<N> docker compose -f compose_gpu.yml up -d   # <N> from `who`
docker compose ps    # both iros2026_system and iros2026_ai_module Up
ros2 topic pub /challenge_question std_msgs/String "data: 'find the red chair'" --once
# watch for /way_point_with_heading / /numerical_response / /selected_object_marker
# traffic, and confirm no "SUBMISSION-BLOCKER" line in the ai_module boot log
docker logs iros2026_ai_module 2>&1 | grep -i "submission-blocker" || echo "no blocker lines"
```

Full ordered checks: `docs/sim_verification.md` Tier 2.7.

## Remaining manual steps to submit

1. Push this branch (or merge it to `main`) to `origin`
   (`github.com/Nishuy52/cmu-vla-2026`, private dev repo).
2. Fork `github.com/Yuxin916/CMU-VLN-Challenge-2026` on GitHub (public,
   per the upstream README's submission section) if not already done.
3. Run `sync_to_fork.sh` against a checkout of the public fork, apply
   `docker/ai_module_fork/docker/compose.yml` / `compose_gpu.yml` at
   the fork's `docker/compose.yml` (gotcha 12 — these two files live
   outside `ai_module/`, so they are a manual copy, not auto-synced),
   commit, and push to the fork.
4. Build the image on the Ubuntu machine and push it to Docker Hub
   (`docker tag ... && docker push ...`) — required because the base
   `ai_module` build installs packages beyond the upstream dummy.
5. Run the Gate 5 scored dry-run (`docs/phase2_playbook.md`) on at
   least one training scene end to end and confirm the pre-submit
   checklist: `instances_tracked > 0`, `VLA_DETECTOR` set to a real
   detector, no `SUBMISSION-BLOCKER` line in the boot log. This needs
   the live sim + GPU and was not re-run in this packaging session.
6. Submit the fork repo link (and the Docker Hub image link, since the
   image installs extra packages) via the Google Form linked in
   `docs/challenge_brief.md`. Multiple submissions are allowed; the
   highest score counts, so submit early and iterate before the 15 Aug
   AoE deadline.
