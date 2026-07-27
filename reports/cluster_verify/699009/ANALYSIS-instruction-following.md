# Live IF analysis — cluster runs 698999 + 699009 (27 Jul 2026)

First live IF scores with the LLM tier actually serving (SoCLaaS, `tier=api`).
Method: re-scored each bag with `tools/score_live_run.py`, then instrumented
`gt_battery._if_rubric_geometry` + `scoring._is_pass_by_leg` to dump per-leg goals,
resolved GT instance ids, and closest approach of the driven (object-frame) path.
Cross-checked against `instance_index.jsonl`, `raw_detections.jsonl`, adapter FSM logs.

## Headline

| run | scene | question | rubric | legs in order | Fréchet | cov@1m |
|---|---|---|---|---|---|---|
| 698999 | livingroom_1 | Q0 plant→vase | **0.500** | 1/2 | 6.17 | 0.00 |
| 699009 slot 4 | livingroom_1 | Q0 (**same text**) | **0.000** | 0/2 | 4.80 | 0.00 |
| 699009 slot 2 | livingroom_1 | Q1 lamp→corridor→cabinet | 0.333 | 2/3 | 8.52 | 0.41 |
| 699009 slot 3 | office_1 | Q0 plant→water cooler | 0.000 | 0/2 | 6.68 | 0.32 |
| 699009 slot 7 | office_1 | Q1 plant→corridor→bench | 0.333 | 2/3 | 4.68 | 0.63 |

Correction vs the first console read: slot 3 = 0.000 and slot 7 = 0.333 (the
"2/3 legs, coverage 0.63" best result is slot 7).

## Variance root cause (0.500 vs 0.000, identical scene+question)

Leg goals are computed from the GT-perfect index and are **bit-identical** between
the two runs — only the driven trajectory differs. Closest approaches:

```
698999: leg1 closest 1.587 m  (tolerance 1.746) -> PASS
slot 4: leg1 closest 2.072 m  (tolerance 1.746) -> MISS
```

Both closest approaches occur early, during the orientation sweep — **the credited
leg in 698999 is incidental**, not a deliberate arrival. Final resting poses differ
entirely between runs ((-3.74, 2.63) vs (-0.83, -2.82)), so the two runs' internal
plans genuinely disagree about where to drive.

**Mechanism — a perception vocabulary gap, not detector jitter on the target class.**
The question's disambiguating anchors are "pyramid candle holder" and "TV". Across
the entire session, in BOTH runs:

```
grep -ic candle  raw_detections.jsonl  -> 0
by_class union: no "candle holder", no "tv"/"television"  (either run)
```

The detector never produces these classes at all. `InstructionHead._ranked_anchor`
(`src/core/heads/instruction.py:390-460`) then relaxes the clause via
`toolbox.resolve()`'s ladder (drop_relation / category_only / drop_disambiguator)
and falls back to a same-label salience tie-break over 8 live "potted plant" and 12
live "vase" candidates. That tie-break is sensitive to which candidates happen to be
in the live map at answer time — which varies run to run. **One mechanism explains
both the wrong terminal legs and the run-to-run variance.**

## Per-leg failure taxonomy (all 5 runs)

| leg pattern | outcome | cause |
|---|---|---|
| First leg, VIA_NEAR / non-terminal GOTO | reached in every run | PASS-BY tolerance is generous (~2.0-2.4 m) — weak evidence of good grounding |
| GOTO with a **missing-class** relational disambiguator | missed every time | grounding: anchor class never detected (candle holder, projector screen case) |
| Corridor legs | closest approach 0.196 m / 1.108 m but **threading fails** | navigation geometry — path passes near the gate midpoint but segments never cross the gate line. Already tracked as **#77** |
| Terminal STOP with missing-class disambiguator ("vase between the TV and the door", "water cooler near the window", "bench closest to the map wall decal") | missed 4 of 5 | same vocabulary gap. Worst: office Q0 water cooler, 5.008 m — live "water cooler" candidates sit nowhere near GT, i.e. outright mislabeling |
| Terminal STOP where disambiguator IS detected | still missed (2.394 m, 2.857 m) | likely navigation/routing — resolved candidates cluster in a different part of the room than GT |

## Best result (slot 7, office_1 Q1 — 2/3 legs, cov 0.63)

Leg 0's disambiguator ("shelf") **is** a detected class → deliberate 0.251 m approach.
Leg 1 routes cleanly between two mapped tables (1.108 m). Leg 2 fails because "map
wall decal" is never detected (0 hits both office runs) — same mechanism again.

## Exploration ruled OUT

`explore_debug` covers only the 60 s orientation sweep; within it `bfs_reroot` behaves
correctly and consistently (pocket 16→44→1799 cells, reroot true while degenerate then
false). Not a contributor to the variance or the leg misses.

## Ranked cost to live IF score

1. **Missing-class relational disambiguators (dominant)** — 4 of 6 non-corridor legs.
   Detector never emits candle holder / TV / window / map wall decal. Causes both wrong
   terminal legs and the run-to-run variance.
2. **Corridor threading precision** — systemic, both corridor legs, full leg-equivalent
   penalty each. Tracked as #77.
3. **Generous PASS-BY tolerance masks first-leg quality** — "3/3 first legs passed" is
   not evidence of good grounding.
4. **Navigation/routing misses** where the GT target sits outside the explored region.

## Proposed fixes (proposals only)

- Vocabulary/prompt coverage audit for the GDINO seam, targeting GT disambiguator
  classes that never appear as detections: candle holder, TV/television, window,
  wall decal.
- Emit the relaxation-ladder step per answered leg (a `relaxation_step` flag beside
  `instance_index.jsonl`) so "this leg is an arbitrary tie-break" is visible live
  instead of reconstructed after the fact.
- Feed "disambiguator anchor never seen" back into route/answer timing — keep exploring
  for the missing class before committing the fallback resolve.
- File the vocabulary gap as its own issue, distinct from #77 (gate-crossing geometry).
