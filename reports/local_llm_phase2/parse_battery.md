# Local-LLM Phase 2 parse battery (2026-07-18T16:56:51+00:00)

model=qwen2.5vl:7b base_url=http://localhost:11434/v1 temperature=0.0 n_questions=75

Ground truth for **qtype** is the `questions.json` key each question is filed under (authoritative, not a proxy). Target-noun / route-shape agreement between the regex floor and the LLM-ladder Plan is a structural diff, not independently GT-scored — see divergence examples below for manual judgement of which side is right.

## Environment note (computed from this run's own latencies)

The local LLM tier was actually reached (`parse_tier == "local"`) on 2/75 questions (3%); 73/75 questions triggered the ladder's repair round. Latency clusters (rounded seconds x count, top 5): ~16s x51, ~40s x22, ~15s x1, ~10s x1.

**Caveat:** most questions never got a real LLM reply — the dominant latency bands above line up with the per-call timeout (client gave up), not with genuine generation. This run coincided with the full sim stack (Unity + ROS nodes) actively running on the same GPU/CPU, not merely a background image build, and `ollama` server logs showed generation collapsing to ~1.9 tokens/s under that contention — this is a busier box than `docs/local_llm_plan.md`'s Phase 0 baseline. Treat the per-QType agreement/accuracy numbers above as a floor-fallback-dominated result, NOT a measurement of the 7B model's parsing quality. See `reports/local_llm_phase1/conformance.md` (a smaller sample run earlier in the same session, before contention worsened) for a less-confounded read: 6/10 schema-valid via the LLM tier, with two concrete failure modes identified (predicate-enum canonicalization, clause anchor-count) — see GitHub issue #45. **Recommendation: re-run this battery on a quiet box (no sim) before treating any KEEP-FLOOR call below as final.**

## Per-QType summary

| QType | n | Floor qtype acc | LLM qtype acc | Floor valid | LLM valid | Agreement | LLM used local tier | Mean lat (s) | Median lat (s) | Max lat (s) | Recommendation |
|---|---|---|---|---|---|---|---|---|---|---|---|
| numerical | 15 | 100% | 100% | 100% | 100% | 100% | 0% | 24.007 | 16.009 | 40.009 | KEEP FLOOR (LLM agrees almost always; no measured upside, added latency) |
| object_reference | 30 | 100% | 100% | 100% | 100% | 93% | 7% | 22.179 | 16.008 | 40.009 | KEEP FLOOR (LLM agrees almost always; no measured upside, added latency) |
| instruction_following | 30 | 100% | 100% | 100% | 100% | 100% | 0% | 23.207 | 16.008 | 40.013 | KEEP FLOOR (LLM agrees almost always; no measured upside, added latency) |

## Divergences (floor vs LLM Plan differ)

2/75 question(s) diverge. Examples:

| Scene | QType | LLM tier | Diffs | Question |
|---|---|---|---|---|
| chinese_room | object_reference | local | n_target_clauses: floor=1 llm=2 | Find the bowl on the table closest to the folding sc... |
| chinese_room | object_reference | local | n_target_clauses: floor=1 llm=2 | Find the pillow on the chair that is closest to the TV. |

## Errors (ladder raised despite its never-raises contract)

(none)
