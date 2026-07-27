# Findings index — live cluster runs 698999 + 699009 (27 Jul 2026)

Every defect detected and every proposed fix from the three analyses and the
pre-submit verification, mapped to its tracking issue. Nothing stays as prose only.
**All claims below were independently verified before filing** (agent findings were
treated as inputs, not conclusions — one agent claim was initially mis-reproduced by a
wrong JSON path and re-verified against the correct schema before it was trusted).

## Instruction-following (the score driver — 70.6% of points)

| # | Finding | Verified how | Issue |
|---|---|---|---|
| 1 | Missing-class relational disambiguators (candle holder, TV, window, map wall decal never detected) — dominant cost; explains BOTH wrong terminal legs and the 0.500-vs-0.000 variance | `grep -ic candle raw_detections.jsonl` → 0 in all runs; by_class union has no tv/television | **#91** |
| 2 | Corridor-leg threading fails despite 0.196 m / 1.108 m closest approach — full leg penalty each | per-leg instrumentation of `scoring._is_pass_by_leg` | **#77** (comment) |
| 3 | PASS-BY tolerance (~2.0-2.4 m) too generous to discriminate grounding quality; the credited 0.500 leg was an incidental sweep graze | tolerances + closest approaches dumped per leg | **#100** |
| 4 | Legs missed where GT target sits outside the explored region (lower confidence, not separable from #91 for one of the two cases) | live candidate positions vs GT positions | **#103** |
| 5 | Relaxation-ladder fallback invisible in live artifacts | code read: `instruction.py:96-106, 390-460` | **#98** |
| 6 | Head commits an anchor resolve even when the disambiguator class was never seen | same | **#99** |
| 7 | `_plan_nouns` never recurses into `anchor.disambiguator`, so nested nouns reach the detector prompt only by luck | `explore_step.py:615` vs `plan_schema.py:49` | **#95** |

## Numerical

| # | Finding | Verified how | Issue |
|---|---|---|---|
| 8 | Relation clause did not restrict at all — published 28 == the entire tracked chair census (GT 8) | `by_class['chair']`=28 == published answer; GT=11 scene-wide | **#93** |
| 9 | Instance duplication: 28 tracked chairs → 14 spatial clusters (0.5 m union-find) vs 11 GT | union-find over dumped instance positions | **#89** (comment) |
| 10 | Severe UNDER-segmentation: 112 accepted monitor detections → 1 tracked instance; z 1.86 m vs GT 1.06 m | raw_detections gate counts + instance dump + GT loader | **#94** |
| 11 | Resolved `Plan` not captured anywhere — cannot distinguish "clause resolved then ignored" from "never resolved" | artifact inventory | **#102** |
| 12 | `instance_index` lacks per-instance AABBs — blocks offline replay of `on()`/`near()` | dump schema read | **#101** |

## Object-reference

| # | Finding | Verified how | Issue |
|---|---|---|---|
| 13 | All live obje runs score `n/a` — scorer-side GT resolution gap, NOT a pipeline failure (markers were published). Three distinct causes: nested-disambiguator not recursed; tie-break margin missed by 0.045; referential-statement corpus gap | `scoring.py:566/644/691`; `_anchor_agrees` confirms all nouns bridge | **#92** |
| 14 | Published marker falls 1.65 m outside office_1's annotated GT extent — frame-fit suspicion | marker centroid vs GT aabb bounds | **#104** |

## Tooling / harness

| # | Finding | Verified how | Issue |
|---|---|---|---|
| 15 | `score_live_run` default `--out` silently overwrites the committed 20-Jul baseline | `score_live_run.py:740`; observed and restored this session | **#96** |
| 16 | `_merge_with_existing` keys by `(scene, qdir)` → drops rows when a scene has 2 questions of one type (9 scored → 6 retained) | `score_live_run.py:691,695` | **#97** |
| 17 | LLM tiers blocked by missing `openai` SDK in the cluster venv — every parse fell to regex | usage log: both tiers fail at 1.6 ms, import-time error | **#82** (fixed, comment) |

## Fixed in-session (no issue per standing rule 8)

- Bag-recorder stop hung the job to the SLURM wall (`wait $BAG_PID`) → `setsid` + bounded escalating stop — `46c96ba`
- `run_one` `for i` without `local i` clobbered the matrix counter → slot collision, lost capture — `14e3980`
- Batch: SDIR tilde never expanded (scene overlay silently skipped — **would have invalidated all 45 questions**) — `183c7aa`
- Batch: shared matrix filename → all chained jobs would read the last matrix — `183c7aa`
- Batch: inst sweep overruns the wall → wall-guard + split — `183c7aa`
- Batch: teardown missed autonomy-stack binaries → stack pkill + busy warning — `183c7aa`
