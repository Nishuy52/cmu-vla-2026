# Checkpoint Protocol Design — v1.0

*11 Jul 2026. Design for the five LLM/VLM checkpoints from architecture §3. Each checkpoint is a
bounded, timeout-guarded call with a structured output contract, a deterministic fallback, and a
ledger cap. Implementation lives in `core/checkpoints/`; prompts are fixture-tested against stub
ChatFns (no network in tests). The 46 parse goldens pattern applies: every checkpoint gets golden
input→output fixtures.*

## Shared rules

- All outputs are JSON with a fixed schema; one repair round on validation failure, then the
  deterministic fallback. Never free text into the answer path.
- Vision calls send JPEG-encoded crops/tiles via `core.llm.providers` vision variants; text calls
  serialise scene facts as compact tables. Token budgets: ≤2k in / ≤200 out per call (latency ≈
  cost ≈ variance — keep calls small).
- Every call: `ledger.allow(name)` gate before, `ledger.record(name, dur, tier)` after; hard
  timeout from `VLA_LLM_CALL_TIMEOUT_S` (default 20 s).
- Determinism note for replay/tests: checkpoint callables are injected into the heads (already the
  seam) — production wiring binds them to real ChatFns, tests to scripted stubs.

## CP2 — Detector-miss recovery (vision)

**Trigger** (from heads): a plan-critical noun has 0 instances after coverage threshold (explore
budget ≥60% spent) — at most once per question.
**Input:** the 4 current tiles (downscaled ≤640px) + the noun and its raw surface form.
**Prompt core:** "You are assisting a robot searching a room. Does this image contain a <noun>
(also written '<raw>')? Answer JSON: {\"present\": bool, \"tile\": 0-3|null, \"bbox_hint\":
[x1,y1,x2,y2] in tile pixels|null, \"confidence\": 0-1}." One tile per message-part, numbered.
**On present:** cast bbox_hint through fusion (same path as detections, min_points relaxed to 3)
→ provisional instance (n_obs=1, score=confidence×0.5) → targeted re-navigation toward it.
**Fallback (absent/timeout/invalid):** proceed to the resolve fallback ladder unchanged.
**Risk note:** a hallucinated "present" costs a detour; cap: provisional instances never satisfy
the ≥3-obs early-answer gate on their own.

## CP3 — Anchor confirmation (vision, IF only)

**Trigger:** arrival at each sub-goal (≤3 per question — ledger cap).
**Input:** the tile crop centred on the anchor's projected image location (project centroid via
`tiling.map_ray_to_camera` inverse) + anchor noun/attributes.
**Prompt core:** "The robot believes the object centred in this crop is a <noun with attributes>.
JSON: {\"match\": bool, \"actual_label\": str|null, \"confidence\": 0-1}."
**On mismatch (confidence ≥0.6):** demote the instance's score, re-run resolve for this anchor,
re-plan the leg to the runner-up if one exists; else continue (the original grounding stands —
one mismatch report is weaker evidence than the tracked map).
**Fallback:** continue on the map's belief. Never blocks the drive.

## CP4 — Pre-answer verification (text; the SORT3D compositional-slip counter)

**Trigger:** before committing OR answers and IF terminal answers (ledger cap 1).
**Input:** question verbatim + the winning candidate + per-clause `PredResult` explanations from
the toolbox's pass_matrix + the runner-up summary + `plan.notes` (the parser's ambiguity flags —
indefinite articles, attachment ambiguity — this is where they get consumed).
**Prompt core:** "Question: <q>. Selected: instance #<id> (<label>, <attrs>, at <xyz>). Computed
facts per clause: <table: clause | passed | explanation>. Runner-up: #<id2> (<facts>). The parser
noted: <notes>. Check EVERY requirement in the question against the computed facts, including any
the clause list may have MISSED. JSON: {\"verdict\": \"confirm\"|\"runner_up\"|\"neither\",
\"missed_constraint\": str|null, \"reason\": str}."
**On runner_up:** swap to runner-up (once). **On neither + missed_constraint:** if time ≥90 s,
re-run resolve with the missed constraint appended as a text-matched clause; else keep winner.
**Fallback:** keep the toolbox winner (deterministic answer stands — CP4 only ever improves on
clause-slip, the documented failure mode).
**This is the highest-value checkpoint** (it guards the 2-pt and 6-pt types); implement first.

## CP5 — Frontier selection (vision, rare)

**Trigger:** scene proved multi-room (≥2 disconnected explored regions or area > threshold) AND
ahead of budget AND ledger allows (cap 1).
**Input:** current panorama with numbered discs rendered at the top-5 frontier directions
(project frontier centroids to columns via azimuth) + the question.
**Prompt core:** "Robot exploring to answer: <q>. Numbered directions mark unexplored areas.
JSON: {\"choice\": 1-5, \"reason\": str}." **Fallback:** geometric top frontier.

## Self-consistency (IF final commit, optional)

Not a separate checkpoint: CP4 run ×3 (temperature default) with majority vote, only when
remaining ≥90 s and ledger allows. Ships after single-call CP4 proves itself in sim.

## Wiring map

| Checkpoint | Head seam (already injected) | New module |
|---|---|---|
| CP2 | `explore` fallback path | `core/checkpoints/miss_recovery.py` |
| CP3 | `anchor_confirm` callable in `heads.factory` | `core/checkpoints/anchor_confirm.py` |
| CP4 | `llm_verify` callable in `heads.factory` | `core/checkpoints/verification.py` |
| CP5 | `affinity_fn`/frontier hook | `core/checkpoints/frontier_select.py` |

Each module: `build(chat_fns, ledger, clock, cfg) -> callable` matching the head seam's signature,
plus prompt constants and JSON schema validators. Goldens in `core/checkpoints/fixtures/`.
