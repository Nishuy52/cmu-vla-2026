# Local-LLM Phase 2 parse battery (2026-07-18T17:04:14+00:00)

model=qwen2.5vl:7b base_url=http://localhost:11434/v1 temperature=0.0 n_questions=75

Ground truth for **qtype** is the `questions.json` key each question is filed under (authoritative, not a proxy). Target-noun / route-shape agreement between the regex floor and the LLM-ladder Plan is a structural diff, not independently GT-scored — see divergence examples below for manual judgement of which side is right.

## Per-QType summary

| QType | n | Floor qtype acc | LLM qtype acc | Floor valid | LLM valid | Agreement | LLM used local tier | Mean lat (s) | Median lat (s) | Max lat (s) | Recommendation |
|---|---|---|---|---|---|---|---|---|---|---|---|
| numerical | 15 | 100% | 100% | 100% | 100% | 100% | 7% | 35.478 | 40.008 | 40.011 | KEEP FLOOR (LLM agrees almost always; no measured upside, added latency) |
| object_reference | 30 | 100% | 100% | 100% | 100% | 93% | 10% | 35.558 | 40.008 | 40.022 | KEEP FLOOR (LLM agrees almost always; no measured upside, added latency) |
| instruction_following | 30 | 100% | 100% | 100% | 100% | 97% | 10% | 38.034 | 40.008 | 40.011 | KEEP FLOOR (LLM agrees almost always; no measured upside, added latency) |

## Divergences (floor vs LLM Plan differ)

3/75 question(s) diverge. Examples:

| Scene | QType | LLM tier | Diffs | Question |
|---|---|---|---|---|
| arabic_room | instruction_following | local | route_kinds: floor=['goto', 'goto'] llm=['via_near', 'goto'] | Go near the stool under the picture and stop at the ... |
| chinese_room | object_reference | local | n_target_clauses: floor=1 llm=2 | Find the bowl on the table closest to the folding sc... |
| chinese_room | object_reference | local | n_target_clauses: floor=1 llm=2 | Find the pillow on the chair that is closest to the TV. |

## Errors (ladder raised despite its never-raises contract)

(none)

> **STATUS (19 Jul, 01:45):** superseded twice — first by sim-load contention,
> then by the GPU power-state wedge (LOG 19 Jul entry). The clean 3B rerun is
> step 2 of the post-reboot runbook; no recommendation from this file should
> be acted on.
