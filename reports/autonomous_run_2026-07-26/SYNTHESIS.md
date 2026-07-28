# Synthesis — what to fix to make the live run actually work

From the adversarially-verified diagnosis of 45 live questions (15 scenes x 3 types).
**15 findings confirmed, 18 refuted.** Ranked by effect on the ROBOT's behaviour, not on
measurement validity.

Live: IF 0.367 · numerical 0.133 · object_reference 0.005 → offline 0.800 / 1.000 / 1.000.

## The one-line diagnosis

**Perception is the whole gap, and it is a quality problem, not a tuning problem.**
Replaying `counting()` over the recorded answer-time index reproduces our wrong answers
**13/13**; the same parser and same counting code over the GT index scores **15/15**.
Parser, counting, scoring, and navigation are all exonerated for numerical.

The live index fails in two directions at once:
- **Under-coverage**: median tracked-vs-GT instance count **0.45x** (home_building_1: 85 of 432).
- **Over-production within detected classes**: chair 23 vs 6, pillow 30 vs 4, picture 37 vs 9,
  cup 14 vs 2 — spatially SPREAD (only 1–22% of pairs within 0.5 m), boxes inflated 1.1–4.5x.

A count question therefore sees far too many of the thing it asks about, inside a map that
is mostly empty.

## Ranked plan — by expected points per unit effort

### 1. Degenerate AABBs break the geometry predicates  [confirmed, MEDIUM sev, SMALL effort]
**7.4% of tracked instances (67/910) have an extent below 2 cm on at least one axis, and
5.5% (50/910) have an XY footprint area of exactly zero. 66 of the 67 are degenerate on a
footprint-relevant axis (X or Y).**

A zero-footprint box can never satisfy `on()` (footprint IoM), and behaves pathologically in
`near()`/`between()`. Every relation-qualified question — which is most of them — is exposed.
This is bounded, mechanically testable, and independent of the harder fusion work.
**Fix:** enforce a per-axis minimum extent at instance construction (the class dimension
priors already provide a floor via `prior.min_ext`).

### 2. `_eval_clause` masks a passing anchor with a failing one  [confirmed, LOW sev, TINY effort]
`toolbox.py:889-895` (and `:881-887` for BETWEEN) selects the **highest-scoring** anchor
result instead of **any passing** one, contradicting its documented existential semantics at
`:867-871`. `_filter_and` (`:1165-1177`) then reads only `.passed` of that single result, so a
failing-but-high-scoring anchor **masks** a genuinely passing one. Exploitable today by `on()`,
which deliberately decouples score from passed.
**Fix:** select among passing anchors. A few lines, with a regression test.

### 3. Colour attributes are never populated  [confirmed, MEDIUM sev, SMALL-MED effort]
`tracker.py:159-169` and `scene_index.py:450-459` build every `InstanceRecord` with
`color_bins=()`, `caption=""`, and no live component ever fills them (`ColoredVoxelMap` is
debug-viz only). A colour clause is therefore a filter that can never match, and `counting()`
returns a **confident 0** rather than abstaining. Costs 2 numerical questions outright
(home_building_2 "red pillows" truth 2 → 0; loft "black pillows" truth 2 → 0).
Note `resolve()` (object_reference) already relaxes an unmatched attribute — only the
counting path hard-zeros.
**Fix (cheap):** make an unmatchable colour attribute relax rather than zero the count —
aligns counting with the resolve path. **Fix (proper):** populate colour on the live path.

### 4. Instance generation quality  [confirmed cause, LARGE effort, HIGHEST points]
The under-coverage + over-production described above. This is the real prize and the real
risk: it means fusion/tracking work, not filtering.
**Do not retry these — both measured and refuted (#123):**
- clamping tracked extents to class priors → correct **1 → 1** of 15
- evidence gating (`min_obs` x score sweep) → mean|err| 4.93 → 2.13 but correct only **1 → 2** of 15
**Acceptance test that exists today:** replay `counting()` over
`reports/cluster_verify/699819/debug/<slot>/instance_index.jsonl` (`tag=answer_time`) — an
offline A/B on real live data, no cluster run needed.

## Enabler, not a points win — do it so we can TELL if the above worked

### 5. The scorer frame is spurious (#124)  [confirmed, HIGH sev]
`object_list.txt` matches the VLA-3D CSV id-for-id across all 15 scenes (mean Δ 0.003–0.045 m),
so the correct sim→object transform is the **identity** — yet we fit and apply up to **154°**
of rotation and **10.7 m** of translation, from a single endpoint correspondence that depends
on a resolver guess. Under the fitted frame, **0%** of arabic_room's driven path lies on any
floor (identity: 100%).

It costs **no points** (identity rescoring moves the obje mean 0.0048 → 0.0032) — but every
live object_reference IoU and every IF trajectory number is currently an **invalid
measurement**. Fix it not for score, but because without it we cannot tell whether items 1–4
helped. **Fix:** use identity when `object_list.txt` matches; keep the trajectory fit only as
a fallback gated on a free-space check (arabic_room 0.00 and home_building_1 0.11 would both
be correctly rejected today).

### 6. Observability plumbing (#119) — already fixed, unvalidated
Both dumps were gated on env vars nothing set. Now exported from `VLA_EXPLORE_DEBUG_DIR` in
both sbatch scripts. Caveat: the relaxation audit is instruction-following ONLY, so 30 of 45
questions emit zero records by design.

## Explicitly ruled out — do not spend effort here

- **"Instance fusion explodes one object into 100+ instances"** — refuted. Scene-wide we track
  **fewer** instances than GT (0.45x), not more.
- **AABB clamping** and **evidence gating** — measured, no benefit (#123).
- **#94 "partially worked"** — refuted; the pre/post delta must not be attributed to it.
- **#93 relation gate** — inert on livingroom_1 (chair/table/vase all first seen in the same
  keyframe), so it cannot explain that overcount.
- **Parser / counting / scoring / navigation for numerical** — exonerated by the controlled swap.

## Still unverified — do not act on these yet

Their verifiers died on the usage limit: the #107 tie-break verdict; "vehicle wedges 0.2–2.4 m
short of the commanded waypoint in 12/15 IF runs"; "loft's zero is a harness crash, not an
algorithm failure"; "three of the four perfect scenes are target-next-to-start". The wedge and
the harness crash are **run-reliability** items and worth verifying first if the goal is "when
it runs, it works".
