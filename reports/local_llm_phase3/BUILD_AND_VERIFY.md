# Phase 3 build & verify runbook — in-image Ollama bake

Post-build verification for the `docker-ai_module:latest` image built from
`/tmp/fork-clean/docker/compose_gpu.yml` (fork-shaped packaging, `docs/ubuntu_setup.md`
§7a). Run these steps **in order** right after `docker compose -f compose_gpu.yml build
ai_module` (or `up --build`) completes. Docker now works without the `sg docker -c`
wrapper post-reboot (`docs/ubuntu_setup.md` §8b note) — plain `docker` commands below.

This is authored-pending-build: the Dockerfile/launch wrapper/compose files landed in
commit `b507eb9` (WIP, interrupted by a host reboot) and have not yet been run on
Ubuntu. Treat every PASS bar here as the acceptance criterion for the first real
post-reboot build, not as an already-confirmed result.

Container/image names follow the existing convention (`docs/sim_verification.md`):
image `docker-ai_module:latest` (or whatever tag `compose_gpu.yml build` assigns —
confirm with `docker images` if compose names it differently), container
`iros2026_ai_module`.

## (a) Image size delta check

```bash
docker images --format '{{.Repository}}:{{.Tag}}\t{{.Size}}' | grep ai_module
```

**PASS (updated 19 Jul, first real build): ~28 GB total** — the original bar below was
authored-pending-build on a wrong 6.76 GB baseline; measured layers are base 9.4 +
ros-jazzy-desktop 3.4 + torch/CUDA pip 5.3 + GDINO 1.4 + Ollama+3B 3.2 (all
load-bearing; cuda_v12 runner correctly pruned, cuda_v13+cpu kept). Investigate only if
a rebuild deviates from ~28 GB by >2 GB in either direction.

~~**PASS:** size is the pre-bake Gate-4 baseline (**6.76 GB**, `docs/ubuntu_setup.md` §7)~~
**plus roughly 5 GB** (Ollama binary/libs after the rocm/vulkan/cpu-variant prune, minus
the kept `cuda_v13*`+`cpu*` runners, plus the `qwen2.5vl:3b` blobs, ~3.2 GB per the
Dockerfile's own blob-size assertion) — i.e. **~11-12 GB** total. Compare against the
`docs/local_llm_plan.md` Phase 3 estimate ("+5-7 GB... on the 20.3 GB image" — that
estimate predates the 7B disqualification and the actual 6.76 GB Gate-4 baseline; the
number that matters is *this build's* delta, not the stale estimate).

**Likely failure / fix:**
- Delta far above ~7 GB → the runner-prune `case` in the Dockerfile (`cuda_v13*|cpu*`)
  didn't match the actual extracted directory names for `OLLAMA_VERSION=v0.32.1`. Check
  the build log for the `=== extracted ollama tree ===` block the Dockerfile prints
  (flag 4 of 4 in the file) and fix the `case` pattern to match what's really under
  `/opt/vla/ollama/lib/ollama/`.
- Delta near 0 / image unchanged → build used a cached layer from before the Phase-3
  additions; `docker compose build --no-cache ai_module` and re-check.

## (b) Boot `ai_module` alone and check the log sequence

```bash
cd /tmp/fork-clean/docker
docker compose -f compose_gpu.yml up -d ai_module
sleep 20
docker logs iros2026_ai_module
```

Grep the exact wordings emitted by `docker/ai_module_fork/ai_module/launch_with_llm.sh`
and `src/ros_adapter/adapter_node.py`, in this order:

1. **Ollama serve started** — `[launch_with_llm] starting ollama serve (OLLAMA_MODELS=...`
   ```bash
   docker logs iros2026_ai_module | grep -F '[launch_with_llm] starting ollama serve'
   ```
2. **`/api/version` reachable in-container** — the script's own readiness line, either
   the success path or the timeout warning:
   ```bash
   docker logs iros2026_ai_module | grep -E '\[launch_with_llm\] (ollama up after|WARNING: ollama did not answer /api/version)'
   ```
   **PASS** requires the `ollama up after Ns` form, not the `WARNING` form.
3. **Pre-warm line**:
   ```bash
   docker logs iros2026_ai_module | grep -E '\[launch_with_llm\] (prewarm OK|WARNING: prewarm of)'
   ```
   **PASS** requires `prewarm OK: model qwen2.5vl:3b loaded`.
4. **Adapter boot line** (`adapter_node.py`'s own startup log, proves the launch
   wrapper's tail `exec`'d the real node):
   ```bash
   docker logs iros2026_ai_module | grep -F 'vla_ai_module up: subscribed 6 topics'
   ```

**PASS criteria:** all four lines present, in that order, with the `ollama up`/`prewarm
OK` (not the WARNING) variants, and the adapter line present (proves the `exec` handoff
at the end of `launch_with_llm.sh` succeeded).

**Likely failure / fix:**
- No `starting ollama serve` line at all → the wrapper never ran; check `command:` in
  `compose_gpu.yml` still points at `/opt/vla/launch_with_llm.sh` and that
  `chmod +x` landed (Dockerfile `RUN chmod +x /opt/vla/launch_with_llm.sh`).
- `WARNING: ollama did not answer /api/version` → `ollama serve` crashed or is slow;
  `docker exec iros2026_ai_module cat /tmp/ollama_serve.log` for the real error (common
  cause: `OLLAMA_MODELS` dir not owned by the `docker` runtime user — check the
  Dockerfile's `chown -R docker:docker "${OLLAMA_MODELS}"` step actually ran).
- `WARNING: prewarm of ... failed` → `docker exec iros2026_ai_module cat
  /tmp/ollama_prewarm.log`; degrade-not-block means the node still comes up, but the
  first live call pays a cold-load cost — re-run step (c) to confirm the model answers
  at all even if pre-warm failed.
- No adapter line → `exec /bin/bash -lc "source /opt/vla/ws/install/setup.bash && ros2
  launch vla_ai_module ai_module.launch.py"` failed; check `docker logs` for a colcon/
  ROS launch error above where the adapter line should be.

## (c) In-container curl smoke: text + vision

Text-only `/v1/chat/completions` (OpenAI-compat wire core/llm/config.py's local slot
targets):

```bash
docker exec iros2026_ai_module curl -fsS http://127.0.0.1:11434/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen2.5vl:3b","messages":[{"role":"user","content":"Reply with the single word: pong"}],"stream":false}'
```

**PASS:** HTTP 200, JSON body with a non-empty `choices[0].message.content`.

Base64-JPEG vision call (use any small JPEG already in the container, or copy one in
first — do NOT rely on network access at this step, the image is meant to be offline):

```bash
docker exec iros2026_ai_module bash -lc '
IMG_B64=$(python3 -c "
import base64, glob
# any baked/available JPEG works for a wire-format smoke test; this is not a quality
# check, just proves the multimodal content-array path round-trips.
f = next(iter(glob.glob(\"/opt/vla/**/*.jpg\", recursive=True)), None)
print(base64.b64encode(open(f, \"rb\").read()).decode() if f else \"\")
")
curl -fsS http://127.0.0.1:11434/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"qwen2.5vl:3b\",\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"What is in this image, one word?\"},{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/jpeg;base64,${IMG_B64}\"}}]}],\"stream\":false}"
'
```

If no JPEG exists in the container, `docker cp` a small tile from
`data/calibration/pano_livingroom_1.png` (converted to JPEG) in before this step —
same fixture `docs/local_llm_plan.md` Phase 0 used.

**PASS:** HTTP 200, JSON body with non-empty `choices[0].message.content` (content
need not be correct — this step is wire-format conformance, not vision quality;
quality is the Phase-2 battery's job).

**Likely failure / fix:**
- Connection refused → ollama isn't listening on `127.0.0.1:11434` inside the
  container; re-check step (b)'s readiness line, and confirm `OLLAMA_HOST` wasn't
  overridden to a different bind address by a stray compose `environment:` entry.
- HTTP 400 on the vision call, mentioning context length → `OLLAMA_CONTEXT_LENGTH`
  didn't take effect (baked default is 8192 in the Dockerfile — `docs/local_llm_plan.md`
  "Config requirement discovered"); `docker exec iros2026_ai_module env | grep
  OLLAMA_CONTEXT_LENGTH` to confirm it's still 8192 and wasn't clobbered by compose.
- 404 model not found → the bake's `ollama pull qwen2.5vl:3b` didn't land under the
  baked `OLLAMA_MODELS` dir the running server is pointed at; re-check
  `docker exec iros2026_ai_module env | grep OLLAMA_MODELS` matches the Dockerfile's
  `/opt/vla/ollama_models`.

## (d) In-container ladder conformance (2-3 questions)

`PYTHONPATH` per the Dockerfile is `/opt/vla/src` (`ENV PYTHONPATH=/opt/vla/src:${PYTHONPATH}`,
already set in the image — no extra flag needed unless invoking a bare `python3` without
inheriting the image env, e.g. via `docker exec -e`). The baked local-slot env
(`VLA_LLM_LOCAL_KIND=openai`, `VLA_LLM_LOCAL_BASE_URL=http://127.0.0.1:11434/v1`,
`VLA_LLM_LOCAL_MODEL=qwen2.5vl:3b`, `VLA_LOCAL_API_KEY=local-dummy-key`) is also already
set — `load_config()` picks it up with no extra env needed:

```bash
docker exec iros2026_ai_module python3 -c "
import time
from core.llm.config import build_chat_fns_with_tiers, load_config
from core.parsing import ladder

cfg = load_config()
pairs = build_chat_fns_with_tiers(cfg)
names = [n for n, _ in pairs]
fns = [f for _, f in pairs]

class WallClock:
    def now(self):
        return time.monotonic()

questions = [
    'Go to the kitchen.',
    'Is there a red chair near the sofa?',
    'How many doors are in this room?',
]
for q in questions:
    plan = ladder.parse(q, fns, WallClock(), tier_names=names)
    print(q, '->', plan.parse_tier, plan)
"
```

**PASS:** all 2-3 questions print `parse_tier == 'local'` (not `regex`), and each
`plan` is a schema-valid `Plan` object (no exception raised — `ladder.parse` never
raises, so a `regex` tier stamp with a plausible-looking Plan is the actual failure
signal to look for, not an exception).

**Likely failure / fix:**
- `parse_tier == 'regex'` for every question, with the underlying exception (if you
  drop the ladder's own swallow-and-fall-through and call `build_chat_fns_with_tiers`
  directly) being `ProviderUnavailable: ... ModuleNotFoundError: openai` →
  **RESOLVED by issue #56's fix**: the Dockerfile's constrained pip layer previously
  never installed the `openai` package even though `VLA_LLM_LOCAL_KIND=openai` is
  baked, so `core.llm.providers` raised `ProviderUnavailable` and the local slot was
  silently skipped by `build_chat_fns_with_tiers`. The Dockerfile now installs
  `openai>=1.0` alongside torch/transformers/groundingdino-py (matching
  `src/pyproject.toml`'s `llm` extra) — confirm this class of failure is gone by
  running `docker exec iros2026_ai_module python3 -c "import openai; print(openai.__version__)"`
  before re-running this step.
- `parse_tier == 'regex'` for every question with `openai` importable → the local slot
  never got a schema-valid reply within the ladder's one repair round; `docker exec
  iros2026_ai_module python3 -c "from core.llm.config import load_config; print(load_config().local)"`
  to confirm the slot resolved at all (kind/base_url/model non-empty). If it resolved
  but still falls through, this mirrors the Phase-1 conformance harness
  (`tools/llm_conformance.py`, not run here since it lives outside this scope) — check
  `/tmp/ollama_serve.log` for request errors during the call.
- Import error on `core.llm.config` / `core.parsing.ladder` → `PYTHONPATH` isn't
  reaching this shell; confirm with `docker exec iros2026_ai_module python3 -c "import
  sys; print(sys.path)"` — should include `/opt/vla/src`. Pass `-e
  PYTHONPATH=/opt/vla/src` explicitly to `docker exec` if it's missing (some `docker
  exec` invocations don't inherit the image's `ENV` the way `docker run`/the container's
  own process does — confirm on Ubuntu).

## (e) VRAM / coexistence check with the sim + GDINO running

```bash
cd /tmp/fork-clean/docker
docker compose -f compose_gpu.yml up -d   # system (Unity/sim) + ai_module together
sleep 30
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
```

**PASS:** total used VRAM is consistent with **GDINO ~3.6 GB + local 3B model ~3.2
GB** coexisting on one GPU (plus Unity's own allocation) — i.e. no OOM, no process
killed, `nvidia-smi` lists both a `python3`-class process (GDINO/torch) and an
`ollama`-class process holding memory simultaneously.

**Dev-box keep-alive note:** `compose_gpu.yml` defaults `OLLAMA_KEEP_ALIVE=5m` (short —
dev box is an 8 GB RTX 4060 Laptop shared with Unity + RVIZ + GDINO, so an idle-but-
resident model must release VRAM between LLM calls). This is correct for THIS check on
the dev box; do not "fix" a short idle-unload gap you observe here — it's intentional.
The eval box (RTX 4090, 24 GB) should override to the image's own baked default at
invocation: `OLLAMA_KEEP_ALIVE=60m docker compose -f compose_gpu.yml up --build -d`
(see the header comment in `compose_gpu.yml`).

**Likely failure / fix:**
- OOM / a process killed → on the 8 GB dev box this is expected if `OLLAMA_KEEP_ALIVE`
  was overridden long by mistake; confirm it's still the 5m default. On the eval-shaped
  24 GB box this would be a real regression — file a GitHub issue per repo standing
  rule 6.
- `ollama`-class process absent from `nvidia-smi` even though step (b)/(c) passed →
  the model unloaded due to `OLLAMA_KEEP_ALIVE` expiring between steps (5m default);
  re-run a fresh curl smoke (step c) immediately before this check to force a reload,
  then re-check `nvidia-smi` within the keep-alive window.

## (f) §7a clean-clone packaging gate

The definition-of-done gate (`docker/ai_module_fork/README.md` "Definition of done",
`docs/ubuntu_setup.md` §7a) — run from a **fresh** clean clone, not the one already
built above, to prove reproducibility:

```bash
git clone <fork-remote> /tmp/fork-clean-2 && cd /tmp/fork-clean-2
/path/to/this-repo/docker/ai_module_fork/sync_to_fork.sh /tmp/fork-clean-2
xhost +
cd /tmp/fork-clean-2/docker
docker compose -f compose_gpu.yml up --build -d
```

**PASS:** `docker compose up --build` exits 0 and `docker compose ps` shows both
`iros2026_system` and `iros2026_ai_module` in a running state; re-run step (b)'s grep
sequence against this fresh container as the final confirmation.

**Likely failure / fix:**
- Build fails fetching the Ollama tarball or the `qwen2.5vl:3b` blobs → the bake needs
  network at build time (unlike runtime, which is offline — see the Dockerfile's
  `HF_HUB_OFFLINE`/`OLLAMA` bake-vs-serve split); confirm the build host has outbound
  network access to `github.com` and the Ollama model registry. This is a real
  constraint difference from the GDINO weights bake (also network-at-build) — nothing
  new to fix, just confirm connectivity before blaming the Dockerfile.
- `sync_to_fork.sh` errors "does not exist — is ... the fork root" → ran it against the
  wrong path; it must be pointed at the directory that *contains* `ai_module/`, not
  `ai_module/` itself.
