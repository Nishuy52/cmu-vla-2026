# Local-LLM Phase 2 parse battery (2026-07-19T09:10:32+00:00)

model=qwen2.5vl:3b base_url=http://127.0.0.1:11500/v1 temperature=0.0 n_questions=75

Ground truth for **qtype** is the `questions.json` key each question is filed under (authoritative, not a proxy). Target-noun / route-shape agreement between the regex floor and the LLM-ladder Plan is a structural diff, not independently GT-scored — see divergence examples below for manual judgement of which side is right.

## Environment note (computed from this run's own latencies)

The local LLM tier was actually reached (`parse_tier == "local"`) on 62/75 questions (83%); 16/75 questions triggered the ladder's repair round. Latency clusters (rounded seconds x count, top 5): ~3s x15, ~4s x14, ~5s x9, ~6s x7, ~7s x6.


## Per-QType summary

| QType | n | Floor qtype acc | LLM qtype acc | Floor valid | LLM valid | Agreement | LLM used local tier | Mean lat (s) | Median lat (s) | Max lat (s) | Recommendation |
|---|---|---|---|---|---|---|---|---|---|---|---|
| numerical | 15 | 100% | 100% | 100% | 100% | 100% | 87% | 4.684 | 3.553 | 13.585 | KEEP FLOOR (LLM agrees almost always; no measured upside, added latency) |
| object_reference | 30 | 100% | 100% | 100% | 100% | 97% | 83% | 4.713 | 4.047 | 10.053 | KEEP FLOOR (LLM agrees almost always; no measured upside, added latency) |
| instruction_following | 30 | 100% | 100% | 100% | 100% | 97% | 80% | 11.178 | 9.012 | 33.688 | KEEP FLOOR (LLM agrees almost always; no measured upside, added latency) |

## Divergences (floor vs LLM Plan differ)

2/75 question(s) diverge. Examples:

| Scene | QType | LLM tier | Diffs | Question |
|---|---|---|---|---|
| hotel_room_2 | object_reference | local | target_noun: floor='flower' llm='flowers' | Find the flowers near the window. |
| livingroom_1 | instruction_following | local | n_route_legs: floor=2 llm=3; route_kinds: floor=['goto', 'goto'] llm=['goto', 'corridor_between', 'goto'] | Go to the potted plant closest to the pyramid candle... |

## Errors (ladder raised despite its never-raises contract)

(none)
