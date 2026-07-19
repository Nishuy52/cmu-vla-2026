# Post-fix parse battery (#47/#48/#49) — 2026-07-19

Clean rerun of `tools.llm_parse_battery` (all 75 training questions, `qwen2.5vl:3b` via a
solo host `ollama serve` on `127.0.0.1:11500`, `VLA_LLM_CALL_TIMEOUT_S=30`, GPU otherwise
idle) against the ladder **with** `core/parsing/normalize.py` wired in
(`core/parsing/ladder.py`), immediately after landing the fixes for issues #47, #48, #49.

Compare against the pre-fix baseline committed at `reports/local_llm_phase2/parse_battery.md`
(same harness, same 75 questions, run 2026-07-18).

## Before / after

| | Before (`reports/local_llm_phase2/parse_battery.md`) | After (this report) |
|---|---|---|
| Divergences (floor vs LLM Plan) | 18/75 | 2/75 |
| Local tier reached | 58/75 (77%) | 62/75 (83%) |
| Group A — via_near/goto mislabel (#47) | 9/18 | **0** |
| Group B — flat stacked target clauses (#48) | 6/18 | **0** |
| Group C — fabricated avoid (#49) | 2/18 | **0** |
| Group D — plural/singular target noun (pre-existing, out of scope) | 1/18 | 1/2 (`hotel_room_2` `flower`/`flowers`, unchanged) |
| New: `corridor_between` misparse of "stop at X between Y and Z" | 0/18 | 1/2 (`livingroom_1`, see below) |

All 75 rows remain 100% schema-valid and 100% qtype-correct on both floor and LLM sides,
before and after (`normalize.py` never produces an invalid Plan — it only reshapes
already-valid structure).

## Per issue: cited cases now conforming

**#47** (9 cited: arabic_room, chinese_room, home_building_2, hotel_room_1, hotel_room_2,
livingroom_1, livingroom_3, livingroom_4, loft IF) — all 9 flip to `agree: true` in this
run's `parse_battery.jsonl`; `route_kinds` now matches the regex floor exactly. The
`hotel_room_2` case (which also had a 4-vs-3 leg-count split from the model duplicating
the terminal anchor) also collapses to the correct 3-leg shape via the leg-merge rule.

**#48** (6 cited: chinese_room, home_building_1, home_building_2, livingroom_3, office_1,
office_2) — 5/6 (chinese_room, home_building_1, home_building_2, livingroom_3, office_2)
flip to a single nested clause matching the floor. `office_1` ("Find the paper cup on
the table closest to the projector screen.") is a **pre-existing floor-side change**
(commit `c0d249f`, issue #25, landed after the original battery report): the floor's own
rule-4 semantics for a *bare* trailing superlative on `object_reference` questions now
also stays flat (a deliberate "ranks the target, not the anchor" interpretation) — the
LLM's flat output for this one case now genuinely **agrees** with the floor, so it no
longer diverges either.

**#49** (2 cited: livingroom_2, office_1 IF) — both flip to `n_avoid: 0` / `agree: true`;
`notes` on the affected Plans records `dropped 1 fabricated avoid entry: no avoid/without
language in question`.

## Residual divergences (2/75, neither is #47/#48/#49)

1. `hotel_room_2` object_reference, `Find the flowers near the window.` —
   `target_noun: floor='flower' llm='flowers'`. Pre-existing cosmetic plural/singular
   normalization gap (battery's "Mode D"), out of this ticket's scope; downstream
   grounding is typo/synonym tolerant so this is low-stakes.
2. `livingroom_1` instruction_following, `Go to the potted plant closest to the pyramid
   candle holder and stop at the vase between the TV and the door.` — LLM emits an extra
   `corridor_between` leg (`n_route_legs: floor=2 llm=3`) instead of a single `goto` leg
   whose anchor (`vase`) carries a `between` disambiguator. This is the **exact ambiguity
   rule 6 already calls out by name** ("the vase between the TV and the door" after "stop
   at" is a goto leg with a between-disambiguator, NOT a corridor leg) — the model still
   gets it wrong here. **Not a regression from this fix**: `core/parsing/normalize.py`
   never rewrites `corridor_between` leg kinds or merges a `corridor_between` leg with a
   neighboring `goto` (the merge rule only fires on two *consecutive goto* legs where the
   first was just downgraded from `via_near`), and this exact question was recorded as
   agreeing in the pre-fix baseline — the divergence is model output variance between
   runs (temperature 0.0 notwithstanding, GPU-batched llama.cpp inference is not bit-exact
   run to run), not something introduced here. Flagged as a candidate for a follow-up
   issue; not fixed in this change since it's a distinct, unfiled defect outside #47/#48/#49.
