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

### Phase 3 — live integration (~1 session; after Gate-4 live checklist)
1. Add the local-slot env to the compose `ai_module` service (host
   networking → `http://localhost:11434/v1` reachable from container;
   re-assert like RMW per gotcha 13/§7a).
2. One live question per QType on the 3B with GDINO resident: VRAM watch
   (`nvidia-smi` logging), ledger latencies, no watchdog-floor regressions.
   Exit: live run with `llm=True` completes with ledger latencies inside
   budget and no OOM.

### Phase 4 — August switchover (when keys arrive)
1. Fill primary/secondary slots with the real keys; local demotes to
   tier-3 failover exactly as the ladder intends. Re-run the Phase-2
   battery with cloud models to re-decide the checkpoint matrix.
2. **Decision item (flag now, decide then):** bake the quantized local
   model + server into the submission image as an eval-day dark-network
   fallback tier (+2-4 GB image size; organizers provide tokens at
   runtime, but a network hiccup then costs a whole question — the local
   tier turns that into degraded-quality instead of regex-floor).

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
