# Submission Docker export guide

This guide takes the repository from a commit on `main` to a submitted
competition entry. Follow the steps in order. Each step names its
machine: HOST is the Ubuntu machine with Docker and the GPU
(`docs/ubuntu_setup.md` sets it up). Steps 1 and 2 work on any machine
with git.

The deadline is 15 Aug 2026, AoE. Multiple submissions are allowed and
the highest score counts. Submit early, then iterate.

## 1. Choose the commit

Build from the head of `main` after its full test gate passed.

Do not build from commit `253cf3c` or from the `chore/submission-docker`
branch head. That tree carries the #211 starvation regression: the
detector runs from boot without the pre-latch throttle, the tick thread
starves, and the vehicle barely explores. The fix landed in `5786dd4`.
Every commit from `5786dd4` forward is safe on this point.

Record the chosen commit hash. It names the submission.

```bash
cd /home/jason/cmu_ws/cmu-vla-2026
git fetch origin && git checkout main && git pull
git log -1 --format='%H %s'
```

## 2. Prepare the public fork

The organizers pull a public fork of the upstream repository in which
only `ai_module/` differs. One-time setup:

1. Fork `github.com/Yuxin916/CMU-VLN-Challenge-2026` on GitHub.
2. Clone the fork next to this repository.

Sync this repository's code into the fork:

```bash
./docker/ai_module_fork/sync_to_fork.sh /path/to/fork-clone
```

The script mirrors `src/` into the fork's `ai_module/` and excludes
tests and caches.

Copy the compose files by hand. They live outside `ai_module/`, so the
sync script does not carry them (upstream gotcha 12):

```bash
cp docker/ai_module_fork/docker/compose.yml     /path/to/fork-clone/docker/compose.yml
cp docker/ai_module_fork/docker/compose_gpu.yml /path/to/fork-clone/docker/compose_gpu.yml
```

Commit and push the fork. Confirm the fork's diff against upstream
touches only `ai_module/` and the two compose files.

## 3. Build the image (HOST)

```bash
cd /path/to/fork-clone/docker
docker compose -f compose_gpu.yml build ai_module
```

Notes:

- **Add `--network=host` if the build cannot resolve host names.** On
  this machine the build container gets no working DNS: `apt-get
  update` fails with "Temporary failure resolving archive.ubuntu.com"
  after about 12 minutes, and the build stops with exit code 100. The
  host network namespace has working DNS, so
  `docker build --network=host -f docker/Dockerfile -t <tag> .`
  proceeds normally. This needs no daemon change and no restart.
  Measured 11 Aug 2026. An earlier session read the same symptom as a
  slow network; it is a name-resolution failure.
- In a shell session that predates the docker group membership, prefix
  with `sg docker -c '...'` (`docs/ubuntu_setup.md` section 2).
- A cold-cache build downloads about 7 GB: the GroundingDINO
  checkpoint (938 MB), the Ollama tarball (1.4 GB), and the
  `qwen2.5vl:3b` blobs (about 3.2 GB). At slow egress this takes 60 to
  90 minutes. A warm cache rebuilds in minutes because only the
  `COPY src/` layer onward re-executes.
- The image bakes the perception stack (torch, GroundingDINO with
  weights, offline HF caches) and the local LLM tier (Ollama server
  plus model blobs). The build fails loudly on a size or checksum
  mismatch; the asset URLs were verified live on 10 Aug 2026.
- API keys are never baked. The optional `../.env.llm` file supplies
  `OPENAI_API_KEY` or similar at run time. Without keys, the parse
  ladder degrades to the local Ollama tier and then to the regex tier.
- Expected image size: about 28 GB. Confirm free disk first
  (`df -h /var/lib/docker`).

## 4. Smoke test (HOST)

```bash
xhost +
cd /path/to/fork-clone/docker
DISPLAY=:<N> docker compose -f compose_gpu.yml up -d   # <N> from `who`
docker compose ps    # both iros2026_system and iros2026_ai_module show Up
```

Then, in a shell with ROS 2 sourced:

```bash
ros2 topic pub /challenge_question std_msgs/String "data: 'find the red chair'" --once
```

Pass criteria, in order:

1. `docker logs iros2026_ai_module 2>&1 | grep -ci submission-blocker`
   prints 0.
2. The boot log shows the detector on a real tier: `VLA_DETECTOR` is
   `grounding_dino`, not a mock.
3. Traffic appears on one answer topic: `/way_point_with_heading`,
   `/numerical_response`, or `/selected_object_marker`.
4. The log shows `instances_tracked > 0` after exploration starts.

The full ordered checklist is `docs/sim_verification.md` Tier 2.7. For
a scored end-to-end rehearsal on a training scene, run the Gate 5
dry-run in `docs/phase2_playbook.md`.

## 5. Push the image (HOST)

The image installs packages beyond the upstream dummy, so the
submission includes a Docker Hub image, not only the fork.

```bash
docker tag <image-id> <dockerhub-user>/vla-ai_module:<commit-hash>
docker push <dockerhub-user>/vla-ai_module:<commit-hash>
```

Use the commit hash from step 1 as the tag. A tag that names the
commit makes every submission reproducible.

## 6. Submit

Submit via the Google Form linked in `docs/challenge_brief.md`:

1. The public fork URL.
2. The Docker Hub image reference.

Record the submission in `LOG.md`: the commit hash, the image tag, the
date, and the smoke-test result.

## 7. Troubleshooting

| Symptom | Cause and action |
|---|---|
| `apt-get update` inside the build runs for minutes | Container network egress is slow on this network. The build completes; it is slow, not broken. Do not cancel on time alone. |
| `permission denied` on the docker socket | The shell predates the group add. Use `sg docker -c '...'` or open a fresh login shell. |
| The build fails on a size assertion | An upstream asset changed. Re-check the URL with `curl -I`, then update the Dockerfile assertion deliberately. |
| No answer-topic traffic in the smoke test | Read the ai_module boot log first. A `SUBMISSION-BLOCKER` line names the missing piece. |
| The vehicle does not explore in the dry-run | Confirm the built commit is `5786dd4` or later (step 1). Earlier trees carry the #211 regression. |

## Related documents

- `SUBMISSION_SNAPSHOT.md` — the 10 Aug packaging audit that verified
  file coverage, dependency pins, weights, and env vars. Its pinned
  commit predates the #211 fix; step 1 of this guide supersedes that
  pin.
- `docs/ubuntu_setup.md` — HOST machine setup from bare OS.
- `docs/phase2_playbook.md` — the gate ladder including the scored
  dry-run.
- `docs/sim_verification.md` — the full verification checklist.
