# Numerical-tier live-run diagnosis — run 699009

Scope: why the two live numerical answers are wrong, with numbers, using the
freshly harvested cluster artifacts. Diagnose-only; no `src/` changes made.

Evidence used (all under `reports/cluster_verify/699009/`):
- `debug/0_livingroom_1_nume/{instance_index,raw_detections}.jsonl`
- `debug/2_office_1_nume/{instance_index,raw_detections}.jsonl`
- `verify_batch_699009_adapter_0_livingroom_1_nume.log`
- `verify_batch_699009_adapter_2_office_1_nume.log`
- `captures/2_office_1_nume/office_1/nume/bag` (`/state_estimation` odometry,
  read via `ros2 bag info` / `rosbag2_py`)
- GT: `core.groundtruth.loader.load_scene("data/vla3d/Unity/<scene>")`,
  `docs/gt_answers_numerical.json`
- Code: `src/core/geometry/toolbox.py`, `src/core/heads/numerical.py`,
  `src/core/heads/explore_step.py`, `src/core/perception/scene_index.py`,
  `src/core/perception/detector.py`, `src/core/parsing/vocab.py`,
  `src/core/parsing/fixtures/office_1_numerical_1.json` (golden parse for the
  exact office question, useful as ground truth for what the Plan *should*
  look like).

---

## 1. Livingroom overcount — 28 vs GT 8 ("chairs near the table with a vase on it")

### The headline number
`instance_index.jsonl` last record: `by_class["chair"] = 28`, `total_instances = 92`.
The published answer was **28** — i.e. **the count equals the entire tracked
"chair" class census, with no reduction at all.** That equality is the single
strongest fact in this section: a relation filter that did anything would have
had to, by coincidence, keep literally every tracked chair.

### (a) Label fragmentation — ruled out
`raw_detections.jsonl` label histogram (1224 raw "chair"-family detections):

| label | gate=accepted | gate=no_lidar_cluster |
|---|---|---|
| `chair` | 963 | 261 |
| `chair table` | 6 | 1 |

No `office chair` / `chair leg` variants appear anywhere in this run. The 7
`chair table` detections are folded into the `chair` bucket by
`labels_foldable()` (`src/core/perception/scene_index.py:228`): token set
`{chair}` is a subset of `{chair, table}`, so the fold fires and `by_class`
shows a single `chair` entry (confirmed: no separate `chair table` key in
`by_class`). **Fragmentation contributes ~0 to the overcount in this run.**

### (b) Duplicate instance tracking — confirmed, ~2–2.5x contributor
GT (`load_scene("data/vla3d/Unity/livingroom_1")`) has **11 physical chairs**
total (9 `chair` + 2 `deck chair`), of which exactly **8** sit near the one
table (GT id 84, centroid `(-0.083, -6.555, 0.360)`) that carries the vase
(GT id 71, centroid `(-0.083, -6.590, 0.845)`, directly above the table —
this is unambiguously "the table with a vase on it"; the scene's other two
tables, ids 60/102, carry no vase). Distance from table 84 to each candidate
chair:

| chair id | pos | dist to table 84 |
|---|---|---|
| 8 | (-0.132,-5.919) | 0.64 m |
| 54 | (-0.838,-5.936) | 0.98 m |
| 61 | (0.574,-7.114) | 0.86 m |
| 64 | (1.098,-6.534) | 1.18 m |
| 74 | (-1.315,-6.556) | 1.24 m |
| 81 | (-0.152,-7.144) | 0.59 m |
| 90 | (0.575,-5.910) | 0.92 m |
| 98 | (-0.826,-7.175) | 0.97 m |

→ **8 chairs**, matching GT's answer exactly. The other 3 GT chairs (id 22 at
`y=1.70`, and 2 `deck chair`s at `y=-10.4/-11.9`) are 3+ m away — correctly
excluded were a filter to run.

Clustering the 28 **tracked** chair centroids (union-find, 0.5 m linkage)
collapses them to only **14 spatial clusters**:
```
3  [15, 66, 79]        2  [16, 18]           1  [23]   1  [25]
1  [27]                1  [29]               1  [31]
5  [67, 70, 74, 99, 102]                      6  [71, 96, 97, 101, 108, 112]
2  [72, 116]           1  [73]               1  [94]
2  [98, 113]           1  [103]
```
Restricting to just the tracked instances that fall inside the real
"near-table-84" footprint (x∈[-2.4,1.3], y∈[-7.2,-5.9]) gives **18 tracked
chair instances** against **8 real chairs there** — a ~2.25x local
duplication rate. Averaged over the whole scene, 28 tracked / 11 real
physical chairs ≈ 2.5x. This is the tracker/NMS layer re-minting a new
`instance_id` for re-observations of the same physical chair instead of
re-associating them (chair spacing here, 0.6–1.3 m, is inside what should be
one footprint radius) — a duplicate-tracking defect, not a fragmentation one.

### (c) Relation filter not restricting — confirmed, primary driver
`counting()` (`src/core/geometry/toolbox.py:1210`) is a strict AND-filter:
noun → attributes → every non-superlative, non-scope clause. Its output
equals the tracked class total (28) here, which only happens if
`target.clauses` was effectively empty for this query at answer time — i.e.
either the parse never attached a `NEAR(table[disambiguator=WITH(vase)])`
clause to the `chair` target, or the clause was attached but its own anchor
resolution emptied out with no filtering fallback (unlike the *nested*
disambiguator-drop fallback documented in §2 below, which is scoped to an
anchor's own disambiguator, not to a clause whose anchor itself can't be
resolved). We cannot recover the actual runtime `Plan` object — it isn't
captured in `instance_index.jsonl`, `raw_detections.jsonl`, the adapter log,
or the ROS bag (`/challenge_question`, `/numerical_response`,
`/state_estimation` are the only 3 topics recorded) — so this is inferred
from the exact-equality evidence, not directly observed. It is the dominant
driver by magnitude: even after fully deduplicating (b)'s ~2.5x tracking
inflation (28 → ~11 real chairs), an answer of 11 is still 3 too high versus
GT's 8; only a working relation filter closes that last gap.

### Cross-cutting (both questions)
Adapter log: `parsed:tier=api` (not `local`/`regex`), and the terminal event
trace reads `...answer:published:IntAnswer via answer_state...answered` — no
`answer_from_floor` anywhere in either log. **Both answers were committed by
the LLM-tier pipeline through the normal answer path, not a withhold/floor
fallback.**

---

## 2. Office undercount — 0 vs GT 6 ("computer monitors on the table closest to the map wall decal")

### Detector did propose the target class
`raw_detections.jsonl`: **112 "computer monitor" detections, gate=accepted
for all 112 (0 gate=no_lidar_cluster)**. Perception recall on the monitor
class itself was not the blocker.

### But they collapsed to 1 tracked instance, not 6
`instance_index.jsonl` last record: `by_class["computer monitor"] = 1`
(`total_instances=26`), the sole survivor being:
```
id 25, label "computer monitor", position [2.138, -2.364, 1.856],
score 0.66, n_obs 110, answer_eligible True
```
GT (`office_1`) has 6 real monitors, all tightly clustered at
`x∈[1.81,3.79], y∈[-2.01,-1.64], z≈1.065` on table id 53
(`x=2.86,y=-1.81`); the other GT table (id 50, `x=-1.84,y=-1.88`) has none.
112 accepted raw detections across the run reduced to a **single** tracked
instance — severe under-segmentation of 6 physically-adjacent objects into
one blob, which alone caps any correct answer at 1, not 6. That is a
separate defect from the eventual "0".

### Why the published answer is 0, not 1
The exact question has a golden parse fixture
(`src/core/parsing/fixtures/office_1_numerical_1.json`):
```
target: computer monitor
clauses: [ ON(table[disambiguator = CLOSEST_TO(map wall decal)]) ]
```
`grep -c decal debug/2_office_1_nume/raw_detections.jsonl` → **0**. "map wall
decal" was never once detected in this run's 161 keyframes, despite being
queried: it's present in `_STANDING_VOCAB_NOUNS`
(`src/core/parsing/vocab.py:73`, `("map","wall","decal") -> "map wall decal"`),
so it was in the GDINO text prompt regardless of any question-noun-extraction
gap (see the latent bug noted below).

By design, an unresolvable disambiguator anchor is **dropped, not
zero-filled**: `_apply_disambiguator` (`toolbox.py:786-830`), when
`sub_anchor_recs` (the decal) is empty, logs a `drop_disambiguator` audit
entry and returns `cands` unchanged (i.e. falls back to "any table"). So the
missing decal, by itself, should NOT have zeroed the count — the `ON(table)`
clause should then have been evaluated against **both** tracked tables.

That means the actual zero has to come from the `on()` geometric test
(`toolbox.py:328`) failing for the lone tracked monitor instance against
*every* tracked table candidate. The strongest signal for this: the tracked
monitor's `z = 1.856 m` is **~0.8 m above** the real GT monitor height
(`≈1.065 m`) — consistent with a bad depth/lidar-triangulated height for a
detection that was, per the evidence above, a merged blob of 6 small,
closely-spaced, likely distantly-viewed objects. `on()`'s vertical gate
requires the candidate's bottom to sit within
`[b.zmin + on_upper_span_frac*b_height, b.ztop + on_top_tol]` of a supporting
table's own AABB; an ~0.8 m height error is large enough to plausibly miss
that band for any real table. **This is our best-supported explanation, not
a proven one** — `instance_index.jsonl` records only `position`, not AABBs,
so `on()`'s actual inputs for this tick cannot be replayed from the captured
debug artifacts.

### Corroborating (not proven): exploration never reached the target area
Decoding the run's own bag (`ros2 bag info` + `rosbag2_py` read of
`/state_estimation`, 51,576 odometry messages) gives the robot's traveled
footprint: `x∈[-0.53, 0.72]`, `y∈[-4.91, 0.28]` — a corridor roughly
**1.25 m × 5.2 m**. GT places the target table/monitor cluster at
`x≈2.9–3.8` and the decal at `x≈6.0`. If the robot's odometry frame is
roughly aligned with the GT world frame (start pose ≈ origin, not
independently re-verified in this pass), the robot's explored footprint
never came close to either the correct table or the decal within the 210 s
explore budget — which would explain both the zero decal detections and why
the 6 real monitors, seen only from range if at all, fused into one
badly-localized blob. Flagged as **strongly corroborating, not confirmed**,
since exact frame alignment wasn't independently re-derived here.

### A latent (here non-causal) prompt-vocabulary bug
`_plan_nouns()` (`src/core/heads/explore_step.py:615-624`) walks
`plan.target.clauses[*].anchors[*].noun` but does **not** recurse into
`anchor.disambiguator`. For this exact question the nested anchor ("map wall
decal") happened to already be in `_STANDING_VOCAB_NOUNS`
(`core/parsing/vocab.py`), so the GDINO prompt included it anyway and this
bug is not the cause of the observed zero. But for any question whose
*nested* disambiguator anchor noun is rare/novel and NOT already in the
standing vocab, this gap would silently starve the GDINO prompt of that
noun, producing exactly the same failure mode by construction rather than by
luck. Worth fixing regardless of this incident.

---

## Ranked root causes and proposed fixes

**Livingroom (28 vs 8):**

1. **Relation clause not applied at count time (primary).** Count equals the
   raw class total exactly. *Fix:* have `counting()` unconditionally emit,
   per tick, how many hard clauses it evaluated and how many survivors each
   clause left (it already computes `CountResult.audit` for disambiguator
   drops and `.explanations` for the *empty*-count case — extend that to log
   even when `count > 0`, e.g. into a sibling `count_audit.jsonl`), so "0
   hard clauses ran on a NEAR query" is visible in the artifacts instead of
   requiring the reconstruction done in this report. Add a regression
   fixture: `TargetSpec(noun="chair", clauses=[NEAR(table[disambig=WITH(vase)])])`
   against a scene with 3 tables (only one vased) must return only the
   chairs near that one table.
2. **Duplicate instance tracking (secondary, ~2–2.5x compounding).** 28
   tracked chairs collapse to 14 spatial clusters at 0.5 m linkage (18
   tracked vs 8 real in the near-vase-table zone alone). *Fix:* tighten /
   scale-adapt the track-association radius for compact, densely-arranged
   furniture classes (mirror `near_thresh()`'s scale-adaptive footprint-diag
   logic when deciding whether a new detection re-associates to an existing
   tracked instance, rather than minting a new id), and/or add a periodic
   same-class re-clustering/merge pass before instances are surfaced in
   `by_class`.
3. **Label fragmentation — ruled out.** `labels_foldable()` correctly folds
   `chair table` into `chair`; no separate fragment class exists in this
   run's `by_class`. No fix needed; flagged only to close out the checklist.

**Office (0 vs 6):**

1. **Under-segmentation of the monitor cluster (112 accepted detections → 1
   tracked instance).** *Fix:* size/footprint-aware NMS for small,
   tightly-packed objects of the same class (don't let 6 similarly-labeled
   detections separated by their own footprint size collapse to one
   instance); flag/quarantine instances whose implied bounding volume from
   fused detections is much larger than the class's `dimension_priors.py`
   entry instead of silently keeping one degenerate merged instance.
2. **`on()` geometry rejecting the sole surviving instance against every
   table (best-supported mechanism for the actual "0", not proven).** *Fix:*
   sanity-clip/flag instance z-estimates that are wildly inconsistent with a
   class's known height prior (here off by ~0.8 m) before they reach
   relation evaluation, and capture per-instance AABBs (not just centroid) in
   the debug `instance_index.jsonl` dump so a future incident like this is
   directly replayable offline instead of inferred.
3. **Exploration coverage likely too narrow (corroborating).** Robot's own
   trajectory spans only ~1.25 m × 5.2 m vs. the scene's real object spread
   (target cluster + decal ~5–8 m away). *Fix:* investigate why the 210 s
   explore budget produced such a small `reachable_pocket` for this scene (a
   frontier-selection or costmap issue independent of this report's scope);
   confirming this needs an independently re-derived frame alignment between
   robot odometry and GT world coordinates, which this pass did not do.
4. **`_plan_nouns()` disambiguator-recursion gap (latent, non-causal here).**
   *Fix:* make `_plan_nouns` (`src/core/heads/explore_step.py:615`) recurse
   into `anchor.disambiguator` (and any of its own nested anchors) so every
   noun that gates eligibility, however deeply nested, is guaranteed a slot
   in the GDINO prompt — currently it is included only by the coincidence
   that "map wall decal" is already in `_STANDING_VOCAB_NOUNS`.

**Cross-cutting:** both answers (livingroom 28, office 0) were produced by
the LLM API parse tier (`parsed:tier=api`) and published through the normal
`answer_state`/`published int answer` path in both adapter logs — neither
run fell back to a floor/withhold answer.
