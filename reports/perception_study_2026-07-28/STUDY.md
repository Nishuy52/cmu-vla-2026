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
