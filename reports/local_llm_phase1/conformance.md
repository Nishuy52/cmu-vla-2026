# Local-LLM Phase 1 conformance (2026-07-18T17:36:45+00:00)

model=qwen2.5vl:3b base_url=http://localhost:11434/v1 temperature=0.0 n=10

**Result: FAIL** — 7/10 schema-valid via the LLM tier (bar: >= 80%), 3 repair round(s) used, 3 fell to the regex floor.
Mean LLM latency: 4.803 s.

| Scene | QType | Tier | Valid-via-LLM | Repair used | Floor | Calls | Latency (s) | Question |
|---|---|---|---|---|---|---|---|---|
| studio | object_reference | local | True | False | False | 1 | 2.944 | Find the beer bottle furthest from the couch. |
| livingroom_4 | object_reference | local | True | False | False | 1 | 1.814 | Find the fossil decoration closest to the phone. |
| hotel_room_2 | instruction_following | local | True | False | False | 1 | 5.301 | First, go to the picture closest to the door, then take t... |
| home_building_1 | object_reference | local | True | False | False | 1 | 1.81 | Find the clock on the TV cabinet. |
| livingroom_3 | object_reference | local | True | False | False | 1 | 2.257 | Find the vase between the cabinet and the stool. |
| office_1 | instruction_following | regex | False | True | True | 2 | 9.665 | Go to the potted plant furthest from the projector screen... |
| livingroom_1 | object_reference | regex | False | True | True | 2 | 5.113 | Find the pillow on the sofa that is closest to the windows. |
| arabic_room | object_reference | local | True | False | False | 1 | 1.794 | Find the pillow closest to the book on the stool. |
| office_2 | instruction_following | regex | False | True | True | 2 | 15.478 | First, go to the trash can near the cabinet, then go to t... |
| office_2 | object_reference | local | True | False | False | 1 | 1.856 | Find the computer monitor closest to the cabinet with a p... |

**Stability check:** a full independent rerun (same seed=0, same env) reproduced the
identical 7/10 with the identical 3 failing questions (`office_1`/potted-plant,
`livingroom_1`/pillow, `office_2`/trash-can) — the score is stable, not a one-off
sampling fluke, even though within-question resampling is not perfectly deterministic
at temperature 0 (see `office_1` note below).

## Residual failure mode analysis (bar not met: 7/10 < 8/10)

Both root causes below are **rule-4 / rule-of-avoid violations in the model's output**,
not adapter/schema bugs — the ladder, repair prompt, and enum-synonym fixes from 459cebe
are working as designed (schema-guided repair correctly steers the second attempt,
and it recovers 0 of these 3 because the underlying content is wrong, not just
mis-formatted). Raw model JSON captured via a direct re-invocation of the same
`chat_fn`/`build_parse_messages`/`build_repair_messages` path the ladder uses (not a
separate re-implementation).

### Mode A — flat multi-anchor clause instead of a nested disambiguator (rule 4)

`Find the pillow on the sofa that is closest to the windows.` Per rule 4 ("a relation
that picks WHICH instance of an anchor... goes in that anchor's disambiguator, nesting
as deep as needed"), `closest_to windows` should disambiguate *which sofa*, nested
inside the sofa anchor. The model instead puts both `sofa` and `window` as two peer
anchors of a single `closest_to` clause:

```
"clauses": [{"pred": "closest_to", "anchors": [
  {"noun": "sofa", ...}, {"noun": "window", "raw": "windows", ...}
], "negated": false}]
```
-> validator: `target.clauses[0] closest_to needs 1 anchor(s)`. The repair round is
given the exact error and the raw reply, but on both the original run and a follow-up
manual repro it reproduced essentially the same flat structure (adding a spurious
inner `disambiguator` on the `window` anchor rather than moving `closest_to` to
disambiguate `sofa`), so it stays invalid and falls to the floor. This is the same
defect pattern documented independently in the Phase-2 battery's `n_target_clauses:
floor=1 llm=2` divergences (6/18 divergences there) — see
`reports/local_llm_phase2/parse_battery.md`.

### Mode B — spurious "avoid" clause fabricated from unrelated anchors

`Go to the potted plant furthest from the projector screen then stop at the water
cooler near the window.` The question never says "avoid" or "without going near"
anything, yet the model fabricates an `avoid: [{"between": ["projector screen",
"window"], "near": null}]` entry out of two anchors that appear in *different*,
unrelated disambiguators (the first leg's `furthest_from` anchor and the second leg's
`near` anchor) — first attempt also folds `water cooler` and `window` into one flat
2-anchor `goto` step (`route[1] goto needs 1 anchor(s)`), consistent with Mode A. The
repair round is non-deterministic here: it fixed the `route[1]` anchor count on one
manual repro (yielding a fully valid Plan with a still-bogus `avoid` clause the
validator doesn't reject — it's schema-legal, just wrong) but reproduced the original
`route[1]` error on the actual conformance run and a second rerun, so this question is
genuinely borderline / on the edge of the ladder's one-repair-round budget rather than
deterministically unfixable. `First, go to the trash can near the cabinet, then go to
the folder on the cabinet closest to the whiteboard, and finally, to the door near the
exit sign.` shows the same fabrication: an `avoid` entry synthesized from the
`cabinet`/`closest_to whiteboard` disambiguator, first with both `between` and `near`
set (`avoid[0] must set exactly one of between/near`), then after repair with
`between` still short one anchor (`avoid[0].between needs exactly 2 anchors`) — two
repair-round attempts, still invalid both times.

**Assessment:** neither mode is a formatting slip fixable by tighter repair
instructions — Mode A is a template the model applies whenever a target/anchor has two
stacked relative clauses (attach both as peers instead of nesting), and Mode B is the
model inventing "avoid" semantics from anchors it has already used elsewhere in the
Plan, unprompted. Both would need prompt-level few-shot examples specifically covering
"two relative clauses on one target/anchor" and "do not emit avoid unless the text says
avoid/without" to close — flagged for adjudication, not fixed here per the no-src-edits
constraint.
