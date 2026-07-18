# T13 — Local-LLM Phase 1 conformance + Phase 2 parse battery

**Started/finished:** 2026-07-18/19 (single session) · **Source of scope:**
`docs/local_llm_plan.md` Phase 1 (conformance + config) and the parse half of
Phase 2 (offline value measurement).

## Intent

Prove the local-LLM slot (`VLA_LLM_LOCAL_*` env + `core.llm.config` /
`core.parsing.ladder`) works end-to-end against a live Ollama server, and
measure whether the local 7B model's parse actually beats the deterministic
regex floor on the 75 training questions — per QType, not globally.

## What was built

- `tools/llm_conformance.py` — Phase 1 conformance tool. Loads the local slot
  via `core.llm.config.load_config()` / `build_chat_fns()` exactly as
  `ros_adapter/adapter_node.py` does, drives a deterministic ~10-question
  sample through the REAL `core.parsing.ladder.parse()`, and reports
  schema-valid-via-LLM rate, repair-round usage, regex-floor fallback, and
  latency. PASS bar: >=80% schema-valid via the LLM tier. Unit tests for the
  pure helpers in `tools/tests/test_llm_conformance.py`.
- `tools/llm_parse_battery.py` — Phase 2 (parse half) battery. Runs all 75
  training questions (loaded from the same `questions.json` `gt_battery`
  reads) through (a) the regex floor directly and (b) the real ladder with
  only the local slot wired, diffs the two Plans structurally (qtype / target
  noun / route-leg / avoid counts), and scores qtype accuracy against the
  `questions.json` key (real ground truth, not a proxy). Supports resumable
  chunked runs (`--offset/--limit/--append`) and a `--report-only` rebuild
  mode — needed because the local server was badly contended this session
  (see below) and a full 75-question pass at realistic latencies exceeds a
  single bounded tool invocation. Unit tests in
  `tools/tests/test_llm_parse_battery.py`.
- `src/pyproject.toml` — added an `llm` optional-dependency group
  (`openai>=1.0`); the SDK is imported lazily by `core/llm/providers.py` so
  the fast test tier needed nothing installed, but driving a real `openai`-kind
  slot does.
- `docs/ubuntu_setup.md` §8a — documented the new `llm` install group.

## Results

- **Phase 1 conformance** (`reports/local_llm_phase1/conformance.md`, 10-question
  sample, qwen2.5vl:7b): **FAIL**, 6/10 schema-valid via the LLM tier (bar
  8/10). Two concrete, repairable-but-not-repaired model failure modes found:
  predicate surface-form left uncanonicalized (`furthest_from` instead of the
  schema enum `farthest_from`) and an anchor mis-bound as a second `closest_to`
  argument instead of a nested disambiguator. Both persisted through the
  ladder's one repair round because the repair prompt surfaces a raw Python
  exception string, not the valid enum values — filed as issue #45.
- **Phase 2 parse battery** (`reports/local_llm_phase2/parse_battery.md`, all
  75 questions): confounded — the local LLM tier was only actually reached on
  2/75 questions. Midway through this session the box carried the FULL sim
  stack (Unity + ROS nodes, not merely a background image build), and
  `ollama` server logs showed generation collapsing to ~1.9 tokens/s under
  that contention, so most calls timed out on both the first attempt and the
  repair round regardless of the per-call timeout value. The report's
  "Environment note" section documents this from the run's own latency
  distribution and explicitly recommends a re-run on a quiet box before
  trusting any KEEP-FLOOR call. Per-QType recommendation as measured this
  session: **KEEP FLOOR for all three QTypes** — not because the LLM is
  disproven, but because it was unusable within budget under this
  contention, and even the less-confounded Phase 1 sample fell short of the
  PASS bar on two identified, plausibly-fixable failure modes.
- No Ollama/OpenAI-wire compatibility gap was found — the `/v1/chat/completions`
  wire worked correctly against Ollama's OpenAI-compat endpoint (confirmed via
  direct curl + the `OpenAIChatAdapter`), so no adapter change in
  `src/core/llm/providers.py` was made (task's Phase 1 step 3 condition
  ("provably an Ollama-compat gap") was not met).

## Issues filed

- **#44** — `ladder.parse` mislabels `parse_tier` as `"api"` (not `"local"`)
  when only the local slot is configured, because `build_chat_fns` skips
  unconfigured slots but the default `tier_names` is positional over the
  fixed 3-slot order. Affects production wiring
  (`ros_adapter/adapter_node.py:663` never passes `tier_names`), corrupting
  provenance + ledger bookkeeping in exactly the local-only regime this
  work stands up.
- **#45** — repair-round error messages are raw exception reprs, not
  schema-guided, so a small model repeats the same enum/binding mistake on
  repair; root-caused with a live example from the Phase 1 sample.
- **#46** — `OpenAIChatAdapter`/`with_timeout` timeouts are client-side only;
  an abandoned call keeps running against Ollama's single-concurrency
  (`-np 1`) server, so under contention the client's own "give up" budget
  doesn't free the server slot and can cascade into delaying every later
  question in the same run.

## Deviations from the plan doc

None to the plan's scope. Both conformance and battery tools were run with
the local slot only (per Phase 1's framing — primary/secondary stay
unconfigured pre-August). The Phase 2 vision-checkpoint half
(`docs/local_llm_plan.md` Phase 2 item 2) is explicitly out of scope for this
task (parse half only) and untouched.

## Follow-ups (not done here)

- Re-run `tools/llm_parse_battery.py` on a quiet box (no sim) for an
  unconfounded Phase-2 read; the tool is already resumable/chunk-capable for
  this.
- Consider #45's fix (schema-guided repair errors) as a cheap lever before
  the re-run — could plausibly close some of the Phase-1 conformance gap
  without touching the model or examples.
- Phase 0 vision latency re-baseline (7B over-cap on busy box) was already
  flagged before this task; still open.

## Post-reboot clean rerun (2026-07-19, runbook step 2)

Both prior results above were invalidated (7B model used for Phase 1
instead of the required 3B; Phase 2 confounded by full sim-stack
contention — see `reports/local_llm_phase2/parse_battery.confounded.md`).
This is the authoritative clean rerun on the healthy post-reboot box
(GPU P0/2505 MHz under load, Ollama v0.32.1, qwen2.5vl:3b fully
GPU-resident, 459cebe's schema-guided-repair + enum-synonym fixes live).

- **Phase 1 conformance**: **still FAIL, 7/10** (bar 8/10) — up from the
  pre-fix 6/10 baseline noted in `LOG.md`, but short of the bar. Stable
  across 2 independent runs (same 3 questions fail both times). Two root
  causes, both rule violations in the model's output (not adapter/schema
  bugs — repair correctly steers on format issues, just not on wrong
  content): (A) flat two-clause target instead of a nested disambiguator
  for "X on Y closest/near to Z" phrasing (rule 4), and (B) fabricated
  `avoid` clauses synthesized from unrelated anchors, occasionally
  additionally schema-malformed. Full raw-output analysis in
  `reports/local_llm_phase1/conformance.md` "Residual failure mode
  analysis".
- **Phase 2 parse battery**: unconfounded, all 75 questions,
  `reports/local_llm_phase2/parse_battery.md`. LLM tier reached on 58/75
  (77%); tg rate 80-88 t/s (mean 84.3, from `~/ollama/serve.log`), healthy.
  Agreement with the regex floor: numerical 93%, object_reference 80%,
  instruction_following 63%. 18/75 divergences, grouped into the same two
  root causes as Phase 1 plus a route-leg-kind mislabeling pattern
  (`"go near X"` tagged `via_near` instead of `goto`, rule 6) and one
  cosmetic plural/singular miss. **Recommendation: KEEP FLOOR for all
  three QTypes** — the floor is 100% valid + 100% qtype-correct at zero
  latency on every question in the battery, and every non-cosmetic
  divergence is a case where the floor is right and the LLM is
  systematically wrong on a specific, identifiable construct (not random
  noise an ensemble/voting scheme would average out).
- **Issues filed**: #47 (goto/via_near mislabeling), #48 (flattened
  stacked relative clauses), #49 (spurious avoid fabrication) — all three
  are model-output content defects flagged for adjudication (few-shot
  prompt additions are the likely fix), not adapter or ladder bugs;
  no `src/` changes made per this task's constraint.
