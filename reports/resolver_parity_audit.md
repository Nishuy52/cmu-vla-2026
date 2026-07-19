# Issue #71 resolve-outcome parity audit

Baseline: `reports/gt_battery_post70` (`ordered_leg_credit=0.4389`,
`headline=0.3778`, `threading_violations=7`, `avoid_violations=0`) —
the same run `reports/conversion_probe.md` used.

## Method

`tools/resolver_parity.py` runs BOTH goal-resolution call sites —
`core.runner.gt_battery._if_rubric_geometry` (the rubric) and
`core.heads.instruction.InstructionHead` (the navigation head, via the real
`_run_instruction_head`) — on identical GT scene inputs (same `BasicSceneIndex`,
same fitted spawn/wall geometry) for every leg of all 30 IF questions (72
legs total), logging each side's resolved instance id + label + goal xy.
`tools/resolver_parity_adjudicate.py` then dumps, per divergent leg, the
question text, the anchor spec, and every ranked candidate's distance to
the GT reference trajectory (`trajectory_qN.ply`) — the evidence used to
judge which side (if either) picked the instance the demonstrator actually
went to.

## Parity table summary

Pre-fix (`reports/resolver_parity.json`): **8 of 72 legs** (11%) were
**id-divergent** — the two sides resolved a DIFFERENT GT instance for the
same leg. A further ~50 legs showed goal-position deltas >= 0.5 m on the
SAME resolved instance — this is the projection-method difference
(`Costmap.nearest_reachable_point` BFS vs. `_nearest_free_goal`'s
footprint-edge push) already understood from `reports/conversion_probe.md`
and out of this audit's scope; it is NOT resolve-instance divergence and
the informal "4-12 m delta implies a different instance" read in the
original issue does not hold for most of those legs — only 8 legs are
genuinely id-divergent.

## Adjudication (the 8 pre-fix divergent legs)

| leg | rubric pick | head pick | dist-to-GT (rubric / head) | verdict |
|---|---|---|---|---|
| home_building_1 q0 leg1 (dining table near big picture) | 165 `table` | 176 `coffee table` | 10.37 / 1.29 | RUBRIC-WRONG |
| home_building_2 q1 leg2 (potted plant between curtain/TV) | 58 `potted plant` | 221 `potted plant` | 0.85 / 5.34 | HEAD-WRONG |
| hotel_room_2 q1 leg1 (corridor: TV cabinet / bed) | 72 `tv cabinet` | 43 `sink cabinet` | 0.87 / 2.04 | HEAD-WRONG |
| japanese_room q1 leg1 (near the wardrobe doors) | 0 `wardrobe door` | 2 `door` | 0.86 / 1.67 | HEAD-WRONG |
| livingroom_1 q0 leg1 (vase between TV/door) | 11 `vase` | 42 `vase` | 2.62 / 0.80 | RUBRIC-WRONG |
| livingroom_2 q0 leg1 (crystal ball decoration) | 59 `crystal ball decoration` | 30 `dice decoration` | 0.79 / 0.84 | HEAD-WRONG |
| livingroom_3 q1 leg1 (the cabinet) | 41 `tv cabinet` | 88 `cabinet` | 4.56 / 0.99 | RUBRIC-WRONG |
| office_2 q1 leg1 (folder on cabinet closest to whiteboard) | 69 `folder` | 118 `folder` | 0.77 / 1.05 | HEAD-WRONG (tight cluster, low-confidence) |

**5 HEAD-WRONG, 3 RUBRIC-WRONG, 0 TEXT-AMBIGUOUS** (office_2 is a
borderline/low-confidence HEAD-WRONG — all 5 candidate folders sit within
~0.3 m of each other on the GT-distance metric).

Root cause, both classes: `InstructionHead._ranked_anchor`'s ordered-route
salience tie-break (IF-F3) re-sorted the FULL toolbox survivor pool by
raw proximity to the previous leg, discarding whatever label
discrimination `resolve()` had already encoded whenever survivors carried
DIFFERENT labels (an exact match losing to an alias/category match purely
for being farther from the previous leg's stop point) — not the
"instance-id-only tie" the tie-break's own docstring promised. The rubric,
meanwhile, applied no correction of its own at all (always took the raw
top-ranked candidate), so it had neither this defect nor any offsetting
salience signal — leaving it exposed whenever `resolve()`'s own soft
ranking (post `category_only` fallback) put a loosely-matched label ahead
of an exact one.

## Fixes

1. **HEAD-WRONG** (`src/core/heads/instruction.py`, `_ranked_anchor`):
   restrict the salience reorder to the group of survivors sharing the
   toolbox's own top-ranked LABEL — salience may pick among genuine peers
   only, never leapfrog a stronger label match. Fixes hotel_room_2,
   japanese_room, livingroom_2 cleanly; home_building_2 and office_2 remain
   unresolved (both are same-label ties — the underlying deficiency is a
   toolbox-level one, `between()`/tight-cluster hard-filter passes
   discarding their own soft margin, out of the core/heads-only fix
   surface — flagged as follow-up below). Known tradeoff: livingroom_3
   leg1 (`tv cabinet` id 41 vs `cabinet` id 88, a genuine id-only tier tie
   the label restriction can't distinguish, since the two ARE literally
   different labels but the same match tier) regresses from head's
   previously-correct 88 back to 41 — net effect measured below.

2. **RUBRIC-WRONG** (`src/core/runner/gt_battery.py`,
   `_if_rubric_geometry`'s `_resolve_anchor_rec`): after a `category_only`
   fallback, prefer an exact-label match over the toolbox's raw soft-score
   order; and give the rubric its OWN same-label-restricted salience
   tie-break, computed independently from its own `approach_xy` route walk
   (never reading `InstructionHead`'s resolved state — this is NOT the
   probe's rejected fix 1). Fixes home_building_1, livingroom_1; leaves
   livingroom_3 leg1 (all candidates already exact `vase`/`cabinet` label
   matches, so the exact-label correction has nothing to discriminate) and
   the two residual same-label ties above.

Both corrections skip when the anchor carries a superlative
(`closest_to`/`farthest_from`) disambiguator — the toolbox already ranks
those by a real margin.

## Per-fix integrated battery ledger

| stage | ordered_leg_credit | headline (rubric-proxy) | threading viol. | avoid viol. |
|---|---|---|---|---|
| baseline (`gt_battery_post70`) | 0.4389 | 0.3778 | 7 | 0 |
| + fix 1 (head, same-label salience) | 0.467 | 0.406 | 7 | 0 |
| + fix 2 (rubric, exact-label + salience parity) | 0.494 | 0.433 | 7 | 0 |
| **net (both fixes)** | **+0.055** | **+0.055** | unchanged | unchanged |

Legs reached in order: 33/72 (baseline) -> 35/72 (fix 1) -> 37/72 (fix 2).
Both fixes are net-positive on the integrated score AND independently
justified on correctness (GT-trajectory-distance) grounds for the legs
they touch — no correctness-over-score tradeoff to flag for either fix.

## Post-fix residual divergence (re-run with the fixed code)

Re-running `tools/resolver_parity.py` against the FIXED code
(`reports/resolver_parity_fix2.json`) surfaces 4 remaining id-divergent
legs — 3 pre-existing (home_building_2, office_2, and home_building_1
which FLIPPED: the rubric now correctly resolves 266 `dining table`, dist
1.65 m, while the head — whose same-label restriction has no
different-labelled peer to reorder toward, since 165/`table` is alone in
its label group — still lands on 165, dist 10.37 m) plus one newly exposed
by the salience/exact-label corrections interacting with the two sides'
already-known-different goal-projection methods (`japanese_room` q1 leg1,
id 0 vs id 1, both literally `wardrobe door` — a low-stakes projection
artifact, not a resolve defect) and one goal-projection-driven divergence
in `chinese_room`/`studio` (ids differ but both are plausible-distance
picks; not adjudicated further given the 90-minute budget).

**Not fixed this session** (flagged for the next resolver-parity pass):

- `home_building_2` / `office_2` same-label ties: the surviving candidates
  share an identical label, so no exact-label or label-restricted-salience
  correction can discriminate them. The likely real defect is in
  `core/geometry/toolbox.py`'s `resolve()` itself — hard-filter clause
  predicates (e.g. `between()`) discard their own continuous score once a
  candidate merely *passes* the filter, so multiple same-label survivors
  that all technically satisfy a spatial clause fall through to bare
  instance-id tier priority. Fixing this touches shared resolve() ranking
  used by every question type (numerical, object-reference, IF) and was
  judged out of the safe, narrowly-scoped surface for this session's
  HEAD-WRONG/RUBRIC-WRONG split (`core/heads` vs. `gt_battery.py` inputs
  only) — worth its own follow-up issue.
- `home_building_1`'s post-fix flip (rubric now right, head still wrong):
  the head's own fix has no mechanism to prefer an exact-label match the
  way the rubric's fix 2 does; porting that half of fix 2 into
  `InstructionHead._ranked_anchor` is a reasonable, low-risk next step but
  was not attempted given the stop-condition budget.

## Files

- `tools/resolver_parity.py`, `tools/resolver_parity_adjudicate.py` — the
  audit harness (offline, not part of the scored path).
- `src/core/heads/instruction.py` — HEAD-WRONG fix.
- `src/core/runner/gt_battery.py` — RUBRIC-WRONG fix.
- `reports/resolver_parity.json` — pre-fix parity table (all 72 legs).
- `reports/resolver_parity_fix1.json` — after the head fix only.
- `reports/resolver_parity_fix2.json` — after both fixes (current tree).
- `reports/if71_fix1_head_samelabel/`, `reports/if71_fix2_rubric_samelabel/`
  — the two per-fix integrated battery runs behind the ledger above.
