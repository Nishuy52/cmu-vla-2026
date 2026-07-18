# Local-LLM bridge plan (API keys unavailable until early August)

Written 18 Jul 2026. Constraint set: no cloud LLM keys until ~Aug 1-5; no SoC
cluster; only local hardware = RTX 4060 Laptop (8 GB VRAM, shared with
Unity + RVIZ + GroundingDINO during live runs), 20 cores / 32 GB RAM.
MVS submission target Aug 3 (phase2_playbook), final deadline Aug 15.

## Why this is cheap to do

The provider layer already supports it end-to-end (recon 18 Jul):

- `core/llm/config.py` defines a third **"local" slot** alongside
  primary/secondary. `VLA_LLM_LOCAL_KIND=openai` +
  `VLA_LLM_LOCAL_BASE_URL` + `VLA_LLM_LOCAL_MODEL` (+ dummy
  `VLA_LOCAL_API_KEY`) turns any OpenAI-compatible server (Ollama /
  llama.cpp-server / vLLM) into a live tier. **Zero code changes.**
- The parse ladder tries tiers in order with one repair round each, a
  20 s per-call timeout, a 45 s total cap, and a guaranteed regex floor —
  a weak/slow local model can degrade answers but can never crash or hang
  the pipeline.
- Vision checkpoints (CP2 4-tile miss recovery, CP3 anchor confirm, CP5
  frontier select) send JPEG bytes as base64 data-URIs on the OpenAI wire;
  CP1 parse and CP4 verification are text-only. Worst case ≈9 calls per
  question, typical 3-5.

So the bridge = stand up a local server + pick models that fit the VRAM
budget + measure which checkpoints the small model actually helps.

## Model strategy (8 GB VRAM, shared)

Two regimes, two model sizes:

| Regime | GPU state | Model | Serving |
|---|---|---|---|
| **Live sim runs** | Unity + GDINO resident (~3.5-5 GB) | small VLM ~3-4B Q4 (first pick: `qwen2.5vl:3b`, alt `gemma3:4b`) — ~3 GB | Ollama, on-demand load |
| **Offline dev** (parse battery, prompt work, checkpoint replay vs recorded panos — sim NOT running) | GPU free | 7-8B VLM Q4 (first pick: `qwen2.5vl:7b`) — ~6 GB | Ollama |

- Offline regime is where most of the pre-August work happens (parse
  quality, prompt iteration, checkpoint evaluation against the recorded
  bags' real panoramas) — it gets the better model.
- Live regime only needs to prove wiring + latency + VRAM coexistence;
  the 3B is enough for that, and August's cloud models replace it for
  quality.
- Exact model tags verified at Phase 0 (availability moves); fallback
  serving path if Ollama's vision wire misbehaves: llama.cpp `llama-server`
  with GGUF, same OpenAI wire.

## Phases

### Phase 0 — serving setup (host-level; ~1 h)
1. Install Ollama; pull the 3B + 7B VLM tags.
2. Wire smoke: `curl` `/v1/chat/completions` text-only, then with a
   base64 JPEG (use a tile from `data/calibration/pano_livingroom_1.png`).
3. Latency baseline per shape: parse-sized text call; 1-tile vision;
   4-tile vision; full-pano vision — each vs the 20 s per-call cap, on
   both models, GPU idle vs GPU busy (sim running).
4. Update `docs/ubuntu_setup.md` (standing rule 3): Ollama install, model
   pulls, env var block.
   Exit: all four call shapes return under the cap on the chosen models.

### Phase 1 — conformance + config (~1 session)
1. Export the local-slot env; run the existing provider wire tests plus a
   small live-endpoint conformance script (JSON-mode discipline: does the
   model return parseable Plan JSON at temperature 0; repair-round rate).
2. Decide `max_tokens`/timeout overrides if needed (env-only).
3. Commit an `llm_config.json` example (no secrets) for the local slot.
   Exit: `parse()` through the real ladder returns schema-valid Plans from
   the local tier on a sample of training questions.

### Phase 2 — offline value measurement (1-2 sessions; the meat)
1. **Parse battery**: run all 75 training questions through the ladder
   (local 7B) vs the regex floor; score with the existing gt_battery /
   numerical-yardstick machinery. Keep LLM parse only where it beats the
   floor (per-QType decision, not global).
2. **Vision checkpoints offline**: replay CP2/CP3/CP5 prompts against
   real tiles/panos extracted from the recorded bags (perception tooling
   already crops tiles) with known GT from `object_list.txt`: hit/false
   rates per checkpoint per model size.
3. Output: a per-checkpoint enable matrix for the local regime (e.g.
   "CP1 on, CP4 on, CP2 on only with 7B, CP5 off") + measured latency
   ledger entries.
   Exit: documented matrix + battery deltas committed to reports/.

### Phase 3 — live integration, IN-IMAGE serving (~1-2 sessions; after
### Gate-4 live checklist)

**Requirement (user, 18 Jul): the local model must be served inside the
ai_module container, not from the host** — at eval the organisers run only
our container, so a host Ollama does not exist there. The host install
(Phase 0) remains the dev-loop convenience; the submission path is:

1. Bake serving into the image (Dockerfile, mirroring the GDINO-weights
   pattern: fetched at build, offline at runtime): Ollama standalone
   binary+libs (prune rocm/mlx variants, keep cuda) + the `qwen2.5vl:3b`
   model blobs pulled at build time into a baked `OLLAMA_MODELS` dir.
   Estimated +5-7 GB on the 20.3 GB image — matters for the Docker Hub
   push (Gate-0 residual).
2. Launch wrapper: start `ollama serve` (OLLAMA_CONTEXT_LENGTH=8192, long
   keep-alive) before/alongside the node, pre-warm the model at boot
   (Phase-0 cold load 7-10 s), then the adapter. Local-slot env
   (`VLA_LLM_LOCAL_*` → `http://localhost:11434/v1`) in the compose
   environment (re-assert like RMW, gotcha 13).
3. One live question per QType on the 3B with GDINO resident: VRAM watch,
   ledger latencies, no watchdog-floor regressions.
   Exit: live run with `llm=True` served ENTIRELY in-container, ledger
   latencies inside budget, no OOM, and the §7a clean-clone packaging
   gate still passes.

### Phase 4 — August switchover (when keys arrive)
1. Fill primary/secondary slots with the real keys; the baked local tier
   demotes to tier-3 failover exactly as the ladder intends — now also
   covering eval-day network hiccups (degraded-quality instead of
   regex-floor). Re-run the Phase-2 battery with cloud models to
   re-decide the checkpoint matrix.
2. Revisit image size vs Docker Hub limits if the push is painful
   (options: tighter quant, or drop the bake and accept regex-floor on
   network loss).

## Phase 0 results (18 Jul 2026)

Ollama v0.32.1 standalone (no sudo: tarball → `~/ollama`, `ollama serve` as a
user process; GitHub-releases asset is now `.tar.zst`), CUDA detected on the
4060 (cuda_v13 runner vs driver 13.2). `qwen2.5vl:3b` (3.2 GB) and
`qwen2.5vl:7b` (6.0 GB) pulled.

- **Config requirement discovered:** default served context auto-sizes to
  4096, and the CP2 4-tile shape alone is ~4.2k tokens → HTTP 400. Serve with
  `OLLAMA_CONTEXT_LENGTH=8192` (all four shapes fit).
- Latency vs the 20 s per-call cap (cold = includes model load; warm = mean
  of 3; measured while the sim stack + a pathological CPU-burning detector
  retry loop (#38/#39) were running — treat as worst-case-busy, re-baseline
  7B on a quiet box):

| model | parse | tile1 | tile4 | pano |
|---|---|---|---|---|
| qwen2.5vl:3b warm | 1.8 s | 1.0 s | 1.3 s | 1.1 s |
| qwen2.5vl:7b warm | 1.8 s | **21.6 s (over)** | 5.2 s | **91.3 s (over)** |

- **Verdict:** 3B passes every shape with 10× headroom even on a loaded box —
  live regime confirmed feasible. 7B text-parse is fine; its vision shapes
  spill to CPU under GPU contention — usable offline only if a quiet-box
  re-baseline comes in under cap, else drop to 3B everywhere or try a
  mid-size alternative. Cold loads (7-10 s for 3B) mean the live config
  should pre-warm at boot and use a long `OLLAMA_KEEP_ALIVE`.

## Risks / mitigations

- **VRAM OOM live** (worst risk): measured in Phase 0.3/3.2 before
  relying on it; mitigation = 3B live model, `keep_alive` unload between
  checkpoints if needed, CP2 (4 images) disabled live if prefill spikes.
- **Small-model JSON indiscipline**: ladder's repair round + regex floor
  already bound the damage; measured in Phase 1.
- **Latency vs 20 s cap on busy GPU**: measured explicitly in Phase 0.3;
  mitigation = shrink max_tokens, disable the slow checkpoint shapes.
- **Throwaway-work worry**: none of this is throwaway — the local slot is
  the permanent tier-3 failover, the conformance script and checkpoint
  matrix rerun unchanged against cloud models in August, and the Phase-4
  bake decision reuses the whole setup.
