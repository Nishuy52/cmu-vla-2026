# Local-LLM Phase 2 parse battery (2026-07-18T17:45:47+00:00)

model=qwen2.5vl:3b base_url=http://localhost:11434/v1 temperature=0.0 n_questions=75

Ground truth for **qtype** is the `questions.json` key each question is filed under (authoritative, not a proxy). Target-noun / route-shape agreement between the regex floor and the LLM-ladder Plan is a structural diff, not independently GT-scored — see divergence examples below for manual judgement of which side is right.

## Environment note (computed from this run's own latencies)

The local LLM tier was actually reached (`parse_tier == "local"`) on 58/75 questions (77%); 20/75 questions triggered the ladder's repair round. Latency clusters (rounded seconds x count, top 5): ~2s x25, ~3s x15, ~5s x12, ~4s x7, ~10s x4.


## Per-QType summary

| QType | n | Floor qtype acc | LLM qtype acc | Floor valid | LLM valid | Agreement | LLM used local tier | Mean lat (s) | Median lat (s) | Max lat (s) | Recommendation |
|---|---|---|---|---|---|---|---|---|---|---|---|
| numerical | 15 | 100% | 100% | 100% | 100% | 93% | 87% | 2.698 | 2.001 | 6.89 | KEEP FLOOR (LLM agrees almost always; no measured upside, added latency) |
| object_reference | 30 | 100% | 100% | 100% | 100% | 80% | 77% | 3.108 | 2.2 | 9.542 | KEEP FLOOR (no clear LLM upside measured; see divergence examples) |
| instruction_following | 30 | 100% | 100% | 100% | 100% | 63% | 73% | 6.366 | 4.953 | 16.516 | KEEP FLOOR (no clear LLM upside measured; see divergence examples) |

## Divergences (floor vs LLM Plan differ)

18/75 question(s) diverge. Examples:

| Scene | QType | LLM tier | Diffs | Question |
|---|---|---|---|---|
| arabic_room | instruction_following | local | route_kinds: floor=['goto', 'goto'] llm=['via_near', 'goto'] | Go near the stool under the picture and stop at the ... |
| chinese_room | object_reference | local | n_target_clauses: floor=1 llm=2 | Find the pillow on the chair that is closest to the TV. |
| chinese_room | instruction_following | local | route_kinds: floor=['goto', 'goto'] llm=['via_near', 'goto'] | Go near the potted plant on the table and stop at th... |
| home_building_1 | numerical | local | n_target_clauses: floor=1 llm=2 | How many pillows are on the sofa under the pictures? |
| home_building_2 | object_reference | local | n_target_clauses: floor=1 llm=2 | Find the lamp on the nightstand that has the photo o... |
| home_building_2 | instruction_following | local | route_kinds: floor=['goto', 'goto'] llm=['via_near', 'goto'] | Go near the magazine on the ottoman, then go to the ... |
| hotel_room_1 | instruction_following | local | route_kinds: floor=['goto', 'corridor_between', 'goto'] llm=['via_near', 'corridor_between', 'goto'] | First, go near the bedside table closest to the benc... |
| hotel_room_2 | object_reference | local | target_noun: floor='flower' llm='flowers' | Find the flowers near the window. |
| hotel_room_2 | instruction_following | local | n_route_legs: floor=3 llm=4; route_kinds: floor=['goto', 'corridor_between', 'goto'] llm=['goto', 'corridor_between', 'via_near', 'goto'] | First, go to the picture closest to the door, then t... |
| livingroom_1 | instruction_following | local | route_kinds: floor=['goto', 'corridor_between', 'goto'] llm=['via_near', 'corridor_between', 'goto'] | First, go near the lamp closest to the black chair, ... |
| livingroom_2 | instruction_following | local | n_avoid: floor=0 llm=1 | Go to the microwave on the kitchen counter and then ... |
| livingroom_3 | object_reference | local | n_target_clauses: floor=1 llm=2 | Find the potted plant near the books on the cabinet. |
| livingroom_3 | instruction_following | local | route_kinds: floor=['goto', 'via_near', 'goto'] llm=['via_near', 'via_near', 'goto'] | First, go near the stool, then take the path near th... |
| livingroom_4 | instruction_following | local | route_kinds: floor=['goto', 'goto', 'goto'] llm=['via_near', 'goto', 'goto'] | First, go near the fireplace, then go to the window ... |
| loft | instruction_following | local | route_kinds: floor=['goto', 'via_near', 'goto'] llm=['via_near', 'via_near', 'goto'] | Go near the fireplace, pass by the stairs, then stop... |
| office_1 | object_reference | local | n_target_clauses: floor=1 llm=2 | Find the paper cup on the table closest to the proje... |
| office_1 | instruction_following | local | n_avoid: floor=0 llm=1 | Go to the potted plant furthest from the projector s... |
| office_2 | object_reference | local | n_target_clauses: floor=1 llm=2 | Find the box on the cabinet that is closest to the w... |

## Errors (ladder raised despite its never-raises contract)

(none)

## GPU / token-generation sanity (from `~/ollama/serve.log`)

37/37 model layers offloaded to GPU (fully resident, no CPU spill). Across this run's
`print_timing` lines (n=67 decode segments): **tg rate min 80.89, mean 84.34, max
88.37 tok/s** — solidly inside the 40-100 t/s "healthy" band, consistent with the
post-reboot P0/2505 MHz clock state. No offload warnings, no errors in the log for this
run's window.

## Divergence analysis (18/75, grouped by root cause)

Each `agree=False` row is a *structural* diff (clause count / route-leg kind / target
noun form), not independently GT-scored — judged here by re-deriving which side matches
the parser's own rule spec (`src/core/parsing/prompts.py` rules 4 and 6) and, for the
`n_target_clauses` cases, by direct re-invocation of the LLM chat_fn to inspect the full
Plan JSON (not just the summary diff).

**A — `via_near` vs `goto` on non-terminal "go near X" legs (9/18: arabic_room IF,
chinese_room IF, home_building_2 IF, hotel_room_1 IF, hotel_room_2 IF, livingroom_1 IF,
livingroom_3 IF, livingroom_4 IF, loft IF).** Rule 6 is explicit: `"go to / go near /
stop at / stop by / then to" -> kind "goto"`; `"take the path near X" / "pass by X" ->
kind "via_near"`. The floor correctly tags "go near X" as `goto` in every one of these
9 cases; the LLM mislabels it `via_near`, i.e. it conflates "go near" with "pass by".
This is not cosmetic: `src/core/heads/instruction.py`'s leg-geometry step computes a
different target point per kind (`"goto -> anchor centroid projected to free space;
via_near -> a point near the anchor"`), so the wrong kind changes where the robot
actually drives for that leg, not just a label. **Floor is right, LLM is
systematically wrong** on this phrasing.

**B — flat two-clause target instead of a nested disambiguator (6/18: chinese_room OR,
home_building_1 numerical, home_building_2 OR, livingroom_3 OR, office_1 OR,
office_2 OR).** Rule 4: a relation that picks *which instance* of an anchor goes in
that anchor's `disambiguator`, nested. Manually re-running `Find the paper cup on the
table closest to the projector screen.` through the same chat_fn confirms the pattern:
floor produces one clause, `on(table)`, with `closest_to(projector screen)` correctly
nested inside the `table` anchor's `disambiguator` — i.e. "the cup on THE TABLE THAT'S
CLOSEST TO THE SCREEN". The LLM instead emits two flat peer clauses on the cup itself,
`[on(table), closest_to(projector screen)]` — structurally "a cup that is (a) on some
table and (b) closest to the screen", which is a different (and, with more than one
table in a scene, wrong) grounding: it lets `closest_to` disambiguate the *cup* instead
of the *table*. Both are schema-valid (hence this never shows up as a `FAIL` here, only
in Phase-1 conformance where a 3-anchor variant of the same template broke the anchor
count), but the semantics genuinely differ — **floor is right, LLM is wrong**, and this
is the same root cause as Phase-1 conformance's Mode A.

**C — spurious `avoid` clause fabricated from unrelated anchors (2/18: livingroom_2 IF,
office_1 IF).** Neither question contains "avoid"/"without" language. Re-running `Go to
the potted plant furthest from the projector screen then stop at the water cooler near
the window.` shows the LLM inventing `avoid: [{"between": ["projector screen",
"window"]}]` by pairing the anchor from leg 1's `furthest_from` disambiguator with the
anchor from leg 2's `near` disambiguator — two anchors that never appear together in
the source text. A fabricated forbidden corridor is not just "different", it actively
changes navigation (the costmap gets a hard avoid capsule stamped for a corridor the
question never asked to avoid). **Floor is right, LLM is wrong** — same root cause as
Phase-1 conformance's Mode B, just landing on a schema-valid (still wrong) Plan instead
of a validation failure here.

**D — cosmetic plural/singular target-noun normalization (1/18: hotel_room_2 OR,
`flower` vs `flowers`).** Rule 3 asks for canonical singular in `noun` with the surface
form kept in `raw`; the floor complies, the LLM left the plural in `noun`. Downstream
grounding is typo/synonym tolerant (`core.geometry.toolbox.resolve`), so this is very
unlikely to change which object gets grounded — the one divergence that is genuinely
low-stakes.

## Recommendation (per QType, reasoning)

**numerical: KEEP FLOOR.** 15/15 valid + qtype-correct on both sides; only 1
divergence, and it is Mode B (`home_building_1`, "How many pillows are on the sofa
under the pictures?") — exactly the flattened-clause defect that changes which
instance is being counted against. No latency or capability upside measured (floor is
free and 100% correct on this sample); the one divergence is a correctness risk, not a
wash.

**object_reference: KEEP FLOOR.** 80% agreement; every divergence but one is Mode B
(flat clause instead of nested disambiguator) — a real grounding-semantics bug, not
noise. Floor is already 100% valid + 100% qtype-correct on all 30 questions at zero
LLM latency; the LLM tier adds ~2-9.5s per question for no measured accuracy gain and a
recurring correctness risk on any target with two stacked relative clauses (a common
question shape in this corpus — "X on Y closest/near to Z").

**instruction_following: KEEP FLOOR.** Only 63% agreement, the lowest of the three
QTypes, driven by Mode A (`via_near`/`goto` route-kind errors, which change the actual
drive geometry per `core/heads/instruction.py`) and Mode C (fabricated `avoid`
corridors, which would corrupt the costmap with a forbidden zone the question never
asked for). Both are systematic misreadings of the parser's own leg/avoid rules, not
occasional slips — 9 and 2 of the 18 divergences respectively land here. Floor is
already 100% valid + 100% qtype-correct at zero latency; the LLM tier is both slower
(mean 6.4s, up to 16.5s) and carries the two most operationally dangerous defect modes
found in this whole battery.

**Overall:** the local 3B LLM tier is available (77% of calls reach it without falling
to the regex floor) and its raw token throughput is healthy (84 t/s mean), so the
*infrastructure* (Ollama, adapter, ladder, schema-guided repair, enum-synonym fixes
from 459cebe) is working as designed. The remaining gap is the model's own parsing
judgment on two specific constructs — non-terminal "go near X" phrasing and stacked
relative clauses — which the deterministic regex floor already handles correctly and
for free. Recommendation stands: **keep the regex floor as primary for all three
QTypes**; the LLM tier remains valuable as the ladder's second-chance path for
question shapes the regex floor cannot parse at all (novel phrasing outside the floor's
patterns), not as a routine replacement for the floor's steady-state behavior.
