# Perception study — where the instance index actually loses

**28 Jul 2026.** Measured against scene ground truth on banked live evidence:
15 scenes x 4 jobs (`reports/cluster_verify/{699819,701116,701117,701118,701119}`),
using `data/vla3d/Unity/<scene>/<scene>_object_result.csv` as GT in the **identity
frame** (per #124).

Prior work established *that* perception is the whole numerical gap (#120) and that
post-hoc filtering cannot fix it (#123, two refuted fixes). This study asks *which
stage* loses what, and quantifies each loss separately for the first time.

---

## Headline — three separable losses

| loss | size | where |
|---|---|---|
| **Naming** | 21.8 points of recall | we localise an object but call it the wrong thing |
| **Duplication** | 5.8x | one GT object becomes ~6 tracked instances |
| **Coverage** | 70% of countable GT | nothing tracked anywhere near the object |

### The naming loss is the cheapest and was previously invisible

Recall against countable GT objects (structural classes — `wall`, `ceiling`,
`floor`, `unknown`, `air vent`, `light switch`, `window frame`, `door frame`,
`column`, `carpet` — excluded; tolerance `min(1.0 m, max(0.4 m, half the GT
diagonal))`):

| | pooled, 1579 countable GT objects |
|---|---|
| GT object has a **same-label** instance within tolerance | **7.9%** |
| GT object has **any** instance within tolerance | **29.7%** |
| **gap attributable to naming** | **21.8 points** |

We put something in roughly the right place for ~30% of countable objects, and
name it correctly in barely a quarter of those cases.

Real alias pairs recovered from the confusion table (GT label -> our label):
`potted plant -> plant` (9), `door frame -> "door door frame"` (11),
`door -> "door door frame"` (9), `wine bottle -> bottle` (5),
`flowers -> flower` (3). Genuine misclassifications also appear
(`chair -> table` 5, `vase -> tv cabinet` 4, `cup -> pillow` 3) and those are not
recoverable by aliasing.

**Caveat, stated plainly:** the 29.7% "any-label" figure is a permissive upper
bound — with ~6 instances per object scattered around, some hits are
coincidental. An earlier version of this measurement reported 39.8%; that number
was inflated by large structural GT objects (`wall`, `ceiling`) whose
half-diagonal tolerance swallowed unrelated instances, and is superseded. True
localisation recall lies between 7.9% and 29.7%.

---

## The funnel — 142k proposals become 2.9k instances

Pooled over 60 slots / 9,196 keyframes carrying both `raw_detections.jsonl` and
`instance_index.jsonl`:

```
142,088 proposals  ->  109,771 accepted (23% gated)  ->  2,929 instances
                                        = 37.5 accepted detections per instance
```

Association is already folding hard. It fails anyway because its input is
inflated and its gates are mis-sized.

### 1. Over-proposal at the 2D stage (#131)

Maximum proposals of a class in a **single keyframe**, versus how many exist in
the **whole scene**:

| scene | class | max in one keyframe | GT in scene |
|---|---|---|---|
| office_1 | window | 19 | 1 |
| hotel_room_2 | window | 14 | 1 |
| livingroom_1 | table | 13 | 1 |
| home_building_2 | lamp | 30 | 3 |

`detector.py:759-760` and `:1050-1051` document the assumption that "downstream
fusion/association already tolerates overlapping detections". Tiles overlap ~10
degrees by construction (`tiling.py:65`). **The measurement refutes the
assumption** — there is no cross-tile NMS, and every redundant box is lifted
independently into its own frustum cluster, producing a different centroid and a
different AABB for the same object.

### 2. The association gate is smaller than the duplicate spacing (#128)

`_assoc_gate()` (`tracker.py:178-195`) sizes the radius as `0.5 * class_diagonal`
clamped to `[0.25, 0.75]` m (`tracker.py:108-110`). For furniture over ~1.5 m the
ceiling binds, so two views of *different parts of the same object* can never
associate:

| scene | class | live | GT | gate m | median NN m |
|---|---|---|---|---|---|
| livingroom_3 | tv cabinet | 26 | 1 | 0.75 | 1.07 |
| hotel_room_1 | bed | 18 | 1 | 0.75 | 1.17 |
| home_building_2 | sofa | 17 | 1 | 0.75 | 1.22 |
| home_building_1 | table | 7 | 1 | 0.75 | 2.80 |

**32 of 38 class-scene pairs over-producing by >=3x have median nearest-neighbour
spacing greater than the gate.** This explains #123's "spatially spread
duplicates": they are spread *because the gate cannot reach across the object*.

### 3. A second blocker rejects co-located duplicates (#130)

Not everything is distance. chinese_room carries 20 `floor` instances at median
nearest-neighbour distance **0.00 m**; livingroom_1 has 16 `table` instances at
0.64 m against a 0.75 m gate. Something rejected those merges independent of
distance. Leading suspect is the extent veto (`_match_plausible`,
`tracker.py:198-225`, 1.3x class typical extent) interacting with already-inflated
boxes — inflation blocks the merge, the block spawns a new instance. **Not
confirmed**; needs per-rejection instrumentation under replay.

### 4. Structural classes are tracked as countable objects (#129)

`window` (12,985 proposals) and `floor` (8,967) are the 2nd and 5th most-proposed
labels overall. GT annotates 1 floor per scene; we track up to 20
(chinese_room). A floor box's frustum sweeps the room and the cluster chains
across the floor plane to the walls — measured consequence: **56% of live
instances have an AABB face within 5 cm of the scene-wide extremum, versus 16% of
GT boxes.** Instances terminate on the room shell 3.5x more often than real
objects do.

---

## What this changes about the plan

- The ranked plan in `reports/autonomous_run_2026-07-26/SYNTHESIS.md` put
  "instance generation quality" as one large, risky item. It decomposes into four
  bounded, independently testable defects: #131, #128, #130, #129 — plus a naming
  fix that sits outside fusion entirely and is the cheapest point per unit effort.
- #131 is upstream of the rest: fixing it reduces the input every other fix must
  cope with, so it should land first and the others be re-measured after.
- Neither refuted fix from #123 is revisited here. Both operated on the final
  index; all five findings above operate on instance *generation*, as #123
  recommended.

## Reproduction

Analysis scripts are transient (scratchpad). The durable instrument is
`tools/perception_replay.py` + `tools/perception_eval.py` on branch
`feat/perception-replay-harness`, which runs the real pipeline over
`data/sim_bags/` and scores any instance index — live or replayed — against scene
GT. Every number above is recomputable from committed evidence under
`reports/cluster_verify/` plus `data/vla3d/Unity/`.

## Run-quality note

Jobs 701116+701117 (15 instruction-following runs) scored mean **0.400**, against
the 0.367 baseline of the 20 Jul sweep. 701118/701119 (numerical, object
reference) harvested but produced no `scores.md` — their captured bags need the
#114 reindex/convert recovery. Their debug dumps are intact and were used above.

---

# Corrections — appended 28 Jul, later the same day

Two claims above are wrong and one is refuted. The original text is left intact so
the record shows what was believed and on what basis.

## 1. The tile-overlap mechanism in "1. Over-proposal at the 2D stage" is WRONG

That section says tiles "overlap ~10 degrees by construction (`tiling.py:65`)" and
frames the over-proposal as cross-tile double counting. Verified numerically:

```
DEFAULT_N_TILES   = 4
DEFAULT_TILE_HFOV = 90.0 deg
spacing (360/4)   = 90.0 deg
true overlap      = 0.0 deg
```

**The shipped tiles do not overlap at all.** `DEFAULT_SEAM_OVERLAP = 10 deg` is a
dangling constant referenced nowhere in `src/`, and the `tile_specs` docstring's
"~10deg for the 4x90deg default" is arithmetically false. Filed as #136.

The real sources of redundant proposals are (a) the dual-pass union — the question
caption and the vocab caption both re-detect the same object on a vocab-pass tick,
unioned with no dedupe — and (b) multiple boxes per object within a single pass.

## 2. The 5.8x duplication figure is INFLATED

It conflates two different things:

- genuine duplicate instances of one object, and
- **legitimate multi-part detection of a large object that GT annotates as one.**

GT annotates large structures coarsely. office_1's single `window` object measures
**7.81 x 0.51 x 2.10 m** — an entire window wall. `hotel_room_2 window` is 2.92 m;
`livingroom_1 table` is 2.09 m. Detecting such a thing as many parts is the detector
working correctly against a coarse annotation, not a perception defect.

The duplication problem is real — `home_building_2` has three ~0.5 m `lamp` objects
and 23 proposals in one keyframe, which granularity cannot explain — but the
headline multiplier overstates it and should not be quoted as 5.8x without this
caveat. It has not been re-derived with granularity controlled for.

**This does not weaken #128, it sharpens it.** For a counting question, collapsing
an 8 m window wall's parts into ONE instance is exactly the required behaviour, and
a correctly-sized association gate is the mechanism that would do it.

## 3. #131 (cross-tile NMS) is REFUTED as a fix

Swept over 199,589 banked proposals / 10,310 keyframes:

| IoU | removed | office_1 `window` | hb_2 `lamp` | chinese `chair` (must stay plural) | hotel_1 `pillow` (must stay plural) |
|---|---|---|---|---|---|
| 0.75 | 3.3% | 18 -> **18** | 23 -> **23** | 12 -> 8 | 16 -> 14 |
| 0.55 | 5.1% | 18 -> **18** | 23 -> 22 | 12 -> 8 | 16 -> 14 |
| 0.35 | 6.9% | 18 -> **18** | 23 -> 21 | 12 -> 8 | 16 -> **11** |

`office_1 window` does not move at any threshold — those boxes have essentially zero
mutual overlap, confirming they are distinct regions of the wall rather than
duplicates. Lowering the threshold buys nothing on the pathological cases while
steadily damaging the legitimate ones. Kept on `fix/131-cross-tile-nms` as optional
3.3% hygiene, not a scoring fix.

## 4. Replay baseline now exists

`tools/perception_replay.py` + `tools/perception_eval.py` (branch
`feat/perception-replay-harness`, commit `d22633f` — take that commit, NOT the branch
tip, which carries an unrelated unverified fusion change, #133). Full baseline replay
of `office_1_q1` on unmodified perception:

```
GT 112 · instances 370 · found 28 · recall 25.0% · duplication 13.21x
inflation 1.39x · shell adjacency 28.4% (GT 9.8%)
worst: picture 131 (GT 0) · book 43 (GT 3) · table 25 (GT 2) · floor 23 (GT 1)
```

Index at `reports/perception_replay/office_1_q1.jsonl`. Note it processes 1240
keyframes against a live run's ~146, so these numbers are comparable **only to
another replay of the same bag**, never to live-run figures.

`picture 131` against a GT count of **0** for that class in office_1 is the naming
loss from section 1 of this study showing up in replay: the objects are being seen,
under a label the scene's GT never uses.

## 5. Amendment to correction 2 — the granularity caveat was itself over-cautious

Correction 2 above says the duplication figure is inflated by GT annotating large
structures coarsely. **Measured, that is not materially true of the 3D instance
duplication.** Recomputed over the 15 banked answer-time indexes with structural
classes excluded, and then restricted to GT objects small enough that "the detector
saw parts of one big thing" cannot apply:

| basis | GT | found | recall | instances | duplication |
|---|---|---|---|---|---|
| structural excluded, all GT sizes | 1379 | 113 | 8.2% | 611 | **5.4x** |
| structural excluded, GT diagonal < 1.5 m | 1069 | 72 | 6.7% | 370 | **5.1x** |

Duplication stays at ~5.1x on objects where granularity is not a possible
explanation. So the two effects are separate and both real:

- **Granularity explains the 2D proposal counts** — 18 `window` boxes on an 8 m
  window wall annotated as one object. That is why #131's NMS could not touch them.
- **Granularity does NOT explain the 3D instance duplication.** ~5x duplication
  survives every control applied.

The headline duplication number should be quoted as **~5.1-5.4x with structural
classes excluded**, and it is a genuine defect, not an annotation artifact.

## 6. Gate-rejection instrumentation — what actually blocks a merge

`associate()` instrumented to tally which check rejects each candidate pair, over a
full `office_1_q1` replay:

```
label     107,586  (88.6%)
distance   13,067  (10.8%)
extent        855  ( 0.7%)
```

- **#130 is REFUTED.** The extent veto blocks 0.7% of candidate pairs. The proposed
  "inflation -> veto -> new instance -> more inflation" cycle does not exist.
- **The 88.6% label share is NOT a defect** — it is the expected background of
  genuinely different classes being compared. Checked directly: among instance pairs
  within 1.0 m of each other, only **1.0%** are label variants sharing a token
  (27.5% same-label, 71.5% unrelated classes). Label fragmentation hurts counting
  and GT matching; it is not the association blocker.

Same-label instance pairs within 2.0 m, bucketed by the **real** gate
(`clip(0.5 * ||prior.typ_ext||, 0.25, 0.75)`):

| bucket | pairs | share |
|---|---|---|
| within gate, unmerged anyway | 63 | 8.4% |
| **gate < d <= 1.5 m — #128's territory** | **462** | **61.9%** |
| 1.5-2.0 m | 221 | 29.6% |

**Method note:** the gate column in #128's original evidence table was computed with
the wrong attribute name (`typical_extents`; the real field is `ClassPrior.typ_ext`),
so that probe silently fell back to GT diagonals as a proxy. Redone against the real
computation the conclusion holds — the 0.75 m ceiling clamps every large class
(`floor` typ_diag 10.50, `counter` 3.29, `bookcase` 2.84, `shelf` 2.62, `table` 1.57)
— but the original table was not measuring what it claimed to.
