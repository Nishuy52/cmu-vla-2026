# Training Question Set — Analysis

**Date:** 10 Jul 2026
**Source:** `upstream/CMU-VLN-Challenge-2026/questions/questions.json` (all 15 training scenes)
**Method:** Every statistic below was computed programmatically by parsing the JSON. Counts were not estimated by hand. Scoring weights are taken verbatim from the challenge `README.md` ("Question Types and Initial Scoring").

---

## 1. Dataset Overview

The entire training question set lives in a **single file**, `questions/questions.json`. The per-scene folders contain only a rendered `questions.pdf` (visualization / answer images) and two ground-truth trajectory point clouds (`trajectory_q4.ply`, `trajectory_q5.ply`, corresponding to the two instruction-following questions). **There is no per-scene question JSON** — the top-level file is authoritative.

### Totals

| Metric | Value |
|---|---|
| Scenes | 15 |
| Total questions | 75 |
| Numerical | 15 |
| Object-reference | 30 |
| Instruction-following | 30 |

Every scene has an **identical, uniform layout**: exactly `1` numerical + `2` object-reference + `2` instruction-following (the parser found only one distinct count-tuple `(1, 2, 2)` across all 15 scenes). This means there are no per-scene structural surprises to plan for — the test set of 3 held-out scenes will almost certainly follow the same 1/2/2 shape (5 questions per scene, as the README states).

### Per-scene breakdown

All 15 scenes (`arabic_room`, `chinese_room`, `home_building_1`, `home_building_2`, `hotel_room_1`, `hotel_room_2`, `japanese_room`, `livingroom_1`–`livingroom_4`, `loft`, `office_1`, `office_2`, `studio`) contribute the same 1 + 2 + 2 split.

### JSON schema

The file is a JSON array; each element is one scene object:

```json
{
  "scene": "<scene_name>",
  "questions": {
    "numerical": [ "<string>" ],
    "object_reference": [ "<string>", "<string>" ],
    "instruction_following": [ "<string>", "<string>" ]
  }
}
```

Each question is a **bare natural-language string**. There are no attribute fields, no target-object IDs, no answer keys, and no bounding boxes embedded in the JSON. The only two keys per scene entry are `scene` and `questions`; the only three keys inside `questions` are the three type names.

### Verbatim examples (one per type, from `arabic_room`)

- **numerical:** `"How many sofas are below a window?"`
- **object_reference:** `"Find the pillow closest to the book on the stool."`
- **instruction_following:** `"Go near the stool under the picture and stop at the small table farthest from the columns."`

---

## 2. Score-Weighted Breakdown

Scoring per the README: numerical `/1` (0 or 1, exact-match `Int32`), object-reference `/2` (0–2, bounding-box IoU with ground truth), instruction-following `/6` (0–6, path-constraint adherence in correct order, partial credit possible).

| Type | Questions | % of questions | Points each | Subtotal points | % of points |
|---|---|---|---|---|---|
| Numerical | 15 | 20.0% | 1 | 15 | **5.9%** |
| Object-reference | 30 | 40.0% | 2 | 60 | **23.5%** |
| Instruction-following | 30 | 40.0% | 6 | 180 | **70.6%** |
| **Total** | **75** | 100% | — | **255** | 100% |

**The scoring is severely asymmetric.** Instruction-following is 40% of the questions but **70.6% of the points**. Numerical questions are 20% of the count but under 6% of the points. The point-optimal engineering priority is therefore, in order: instruction-following ≫ object-reference ≫ numerical. A pipeline that perfectly answered every numerical question and nothing else would score 6% of the total; one that only handled instruction-following could reach 71%.

Object-reference is the second lever and shares most of its machinery with instruction-following (both need object grounding + spatial-relation resolution + a waypoint at the target). Numerical is the odd one out — its output is an integer, not a waypoint — and is the lowest-value target.

---

## 3. Spatial-Relation Inventory

Frequency of relation words/phrases, counted with word-boundary matching across all questions, split by type. (`on` counts the preposition in phrases like "on the table"; `path between` / `path near` are counted as distinct compound relations in addition to the bare prepositions.)

| Relation | Numerical | Object-ref | Instruction | **Total** |
|---|---|---|---|---|
| on | 11 | 17 | 20 | **48** |
| closest to | 1 | 18 | 18 | **37** |
| near | 1 | 4 | 28 | **33** |
| between | 0 | 4 | 13 | **17** |
| with | 2 | 2 | 7 | **11** |
| path between | 0 | 0 | 10 | **10** |
| above | 3 | 1 | 1 | **5** |
| farthest from | 0 | 1 | 3 | **4** |
| furthest from | 0 | 2 | 2 | **4** |
| path near | 0 | 0 | 4 | **4** |
| below | 1 | 1 | 1 | **3** |
| under | 1 | 0 | 1 | **2** |

**Top-5 spatial relations overall:** `on` (48), `closest to` (37), `near` (33), `between` (17), `with` (11).

Notes with design weight:

- **Proximity superlatives dominate the graded types.** `closest to` appears 18× in object-reference and 18× in instruction-following, plus `farthest`/`furthest from` (6× combined per type). Resolving "closest / farthest" requires a metric distance comparison across candidate objects, not a binary relation — this is the single most load-bearing relation for the high-value types.
- **`near` is the instruction-following workhorse** (28 of its 33 occurrences). Instruction paths are specified almost entirely by proximity anchors ("go near X", "path near Y").
- **`between` splits two ways.** As an object-reference/anchor relation ("the lantern between the vase and the stone decoration") it is a ternary spatial predicate. As `path between` (10× in instruction-following) it is a *corridor* constraint — the robot must physically pass between two objects, a different geometric primitive.
- **Support relations** (`on`, `above`, `below`, `under`, `with X on it`) are frequent and often *nested* ("the bowl on the table closest to the folding screen" — `on` then `closest to`).
- **No `left of` / `right of` / `in front of` / `behind` / `facing` appear anywhere** in the training set (all counted 0). Egocentric directional relations are absent — see §5.

---

## 4. Attribute Inventory

Attributes are used **sparingly**. Most disambiguation is spatial, not attributive.

### Colors (total 5 mentions across the whole set)

| Color | Count |
|---|---|
| red | 2 |
| black | 2 |
| blue | 1 |

### Sizes (3 mentions)

| Size | Count |
|---|---|
| small | 2 |
| big | 1 |

### Materials (2 mentions)

| Material | Count |
|---|---|
| stone | 1 |
| crystal | 1 |

Colors appear in `home_building_2` ("red pillows"), `loft` ("black pillows", "blue chair"), `japanese_room` ("red pillow"). Sizes: "small table", "big picture", "small table". Materials: "stone decoration", "crystal ball". A handful of other descriptive modifiers appear as noun-phrase heads rather than separable attributes (e.g. "folding screen", "potted plant", "framed records", "calligraphy paintings", "wooden"-type words did not occur).

**Implication:** attribute grounding (color/size/material classifiers) is a minor tie-breaker, not a primary matcher. The dominant identity signal is the **object category noun** plus its **spatial relation** to anchors.

### Distinct object nouns (114 noun-like tokens)

Full distinct set (singular/plural variants kept as written):

```
ball, bed, bedside, beer, bench, book, bookcase, books, bottle, bowl, box,
cabinet, calligraphy, can, candle, chair, chairs, clock, coffee, columns,
computer, cooler, couch, counter, crystal, cup, cups, curtain, decal,
decoration, dining, display, door, doors, dressing, easel, elephant, exit,
fan, figurine, file, fireplace, floor, flowers, folder, folding, fossil,
frame, framed, guitar, holder, hookah, horse, jar, kettle, kitchen, knife,
lamp, lantern, ledge, magazine, map, microwave, mirror, monitor, monitors,
nightstand, ottoman, painting, paintings, paper, phone, photo, photos,
picture, pictures, pillow, pillows, plant, plants, potted, projector,
pyramid, rack, records, refridgerator, remote, round, screen, shelf, sign,
soccer, sofa, sphere, stairs, stone, stool, suitcase, sushi, table, tables,
tea, trash, tray, vase, vases, wall, wardrobe, water, whiteboard, window,
windows
```

Most frequent object heads: `table` (25), `cabinet` (21), `potted plant` (13/12), `window` (11), `vase` (10), `picture` (10), `sofa` (8), `chair` (8), `pillow(s)` (12), `lamp` (6). The vocabulary is broad (100+ categories) and includes fine-grained / rare items ("hookah", "sushi", "calligraphy painting", "fossil decoration", "sphere decoration", "framed records", "map wall decal") that a fixed closed-set detector will likely miss — open-vocabulary grounding is required. Note `refridgerator` is misspelled in the source data (kept verbatim); the grounding module should be robust to source typos.

---

## 5. Linguistic Structure

| Metric | Numerical | Object-reference | Instruction-following |
|---|---|---|---|
| Avg words / question | 8.7 | 10.3 | 22.4 |
| Word range (min–max) | 7–14 | 6–16 | 15–33 |
| Avg spatial relations / question | 1.33 | 1.67 | 3.60 |
| Multi-constraint (≥2 relations) | 4/15 = 27% | 17/30 = **57%** | 30/30 = **100%** |

- **Numerical** questions are short, usually a single relation ("How many pillows are on the bed?"). A quarter carry a nested second constraint ("...on the sofa **under the pictures**", "...on the table **closest to** the map wall decal").
- **Object-reference** is majority multi-constraint (57%). The canonical hard form chains a support relation with a proximity superlative: "the bowl **on the table** **closest to** the folding screen", "the speaker **on the TV cabinet** **closest to** the potted plant **on the TV cabinet**". Resolution requires selecting a candidate set by one relation, then ranking by a second.
- **Instruction-following** is **always** multi-constraint (100%, avg 3.6 relations, up to 33 words). Each is a *sequence* of sub-goals with its own anchor relations, frequently mixing a path constraint ("take the path between X and Y") with ordered goal stops.

### View-dependent / egocentric phrasing

**None found.** The parser searched for `to your left`, `to your right`, `if facing`, `your left`, `your right`, `in front of you`, `facing`, `left of`, `right of`, `behind you` — **zero matches**. All spatial relations in the training set are **allocentric** (object-to-object: near/on/between/closest), not viewer-relative. This is a meaningful simplification: the reasoning module does **not** need to model the robot's heading or a canonical viewing direction to resolve any training question. (Caveat: this is an observation about the 15 training scenes only; the 3 held-out test scenes could introduce egocentric phrasing, but nothing in the training distribution demands it.)

### Temporal / ordering words (instruction-following)

| Word / phrase | Count |
|---|---|
| stop at | 20 |
| then | 19 |
| and stop | 17 |
| first | 12 |
| avoiding | 2 |
| finally | 1 |
| avoid | 1 |
| pass by | 1 |
| stop by | 1 |

Ordering is explicit and pervasive: `first … then … (finally)` sequencing plus a terminal `stop at`. Negative/forbidden-path constraints (`avoid` / `avoiding`, 3 total) are rare but scored — the README explicitly penalizes passing through forbidden regions, so the 3 avoidance cases carry real risk. `pass by` and `stop by` are single-occurrence phrasings the parser must treat as synonyms of pass-through and terminal-goal respectively.

---

## 6. Numerical Questions

**Ground-truth answers do NOT ship in the JSON.** Confirmed programmatically: each scene entry has only the keys `scene` and `questions`; each numerical entry is a bare question string with no answer field. The correct integers must be inferred from scene geometry (or read from the per-scene `questions.pdf` answer images, which are outside this JSON analysis and were not parsed). **No answer distribution can be computed from the JSON** — there are no answers in it to distribute.

All 15 numerical questions verbatim:

1. How many sofas are below a window?
2. Count the number of chairs with pillows on them.
3. How many pillows are on the sofa under the pictures?
4. How many red pillows are on the sofa?
5. How many pillows are on the bed?
6. How many pictures are above the bed?
7. How many calligraphy paintings are above the display ledge?
8. How many chairs are near the table with a vase on it?
9. How many cups are on the coffee table?
10. How many photos are on the TV cabinet?
11. How many pillows are on a sofa?
12. How many black pillows are on the sofa?
13. How many computer monitors are on the table closest to the map wall decal?
14. How many potted plants are on a table?
15. How many framed records are above the couch?

### Zero-count and high-count risks

- **Zero-count risk:** Every question presupposes the object exists. Since scoring is exact-match (0 or 1), a wrong integer scores 0 with no partial credit — including guessing `1` when the true answer is `0`. A robust counter must be able to return `0`. However, all 15 are phrased as "how many are…" with an implied non-empty set, so a true answer of 0 is unlikely but not impossible.
- **High-count risk:** Counts are almost certainly small (pillows on a bed, cups on a table, monitors on a table). The output is `Int32`; realistic answers are in roughly the 0–8 range. Over-counting from double-detections (e.g. a detector splitting one pillow into two) is the main failure mode, not large magnitudes.
- **Indefinite-article ambiguity (4 questions):** "on **a** sofa", "on **a** table", "below **a** window", "near the table with **a** vase" use an indefinite article, implying "any such object" — the count should sum over *all* qualifying anchors, not a single specific one. Contrast with the 11 questions using "**the** sofa / the bed / the coffee table", which presume a unique salient anchor. The counter must distinguish "count across all matching anchors" (indefinite) from "count on the one anchor" (definite).
- **Nested-constraint counting (4 questions):** #3, #8, #13 require first resolving a spatial sub-clause ("the table closest to the map wall decal") *then* counting on it — these reuse the object-reference resolution stack.

---

## 7. Instruction-Following: Constraint Decomposition

Five representative examples decomposed into ordered constraints. Each is either a **path constraint** (how to move / a corridor to traverse or avoid) or a **goal** (a location to reach / stop at). Order is significant and scored.

**A. `arabic_room` q1** — "Go near the stool under the picture and stop at the small table farthest from the columns."
1. GOAL: go near [stool, disambiguated by *under the picture*]
2. GOAL (terminal): stop at [small table, *farthest from* the columns]
- No temporal keyword beyond implicit sequencing via "and stop"; two ordered goals, no path/avoid constraint. Nested attribute (`small`) + superlative (`farthest from`) on the final goal.

**B. `arabic_room` q2** — "First, go to the potted plant furthest from the hookah, then take the path between the two columns, and stop at the tray on the table."
1. GOAL: go to [potted plant, *furthest from* the hookah] — keyword **First**
2. PATH: take the path *between* the two columns (corridor constraint) — keyword **then**
3. GOAL (terminal): stop at [tray, *on* the table]
- Explicit 3-step `First … then … and stop` ordering; mixes a goal, a corridor path, and a terminal goal. "the two columns" is a counted-pair anchor.

**C. `chinese_room` q2** — "First, go near the tea table with the elephant figurine on it, then stop at the table with the horse figurine on it, avoiding the path between the chair and the folding screen."
1. GOAL: go near [tea table, with the elephant figurine *on it*] — **First**
2. GOAL (terminal): stop at [table, with the horse figurine *on it*] — **then**
3. NEGATIVE PATH: **avoiding** the path *between* the chair and the folding screen (forbidden corridor)
- Contains a forbidden-region constraint applied to the whole traversal — scored with a penalty if violated. Anchors disambiguated by "with X on it" contents.

**D. `livingroom_2` q2** — "First, go to the chair near the window, then stop at the soccer ball near the couch, avoiding the path between the TV and the tea table."
1. GOAL: go to [chair, *near* the window] — **First**
2. GOAL (terminal): stop at [soccer ball, *near* the couch] — **then**
3. NEGATIVE PATH: **avoiding** the path *between* the TV and the tea table
- Same shape as C: two proximity-anchored goals plus one avoid-corridor. Confirms the avoid-pattern is a recurring template.

**E. `loft` q2** — "Go near the fireplace, pass by the stairs, then stop at the sphere decoration on the cabinet."
1. GOAL: go near [fireplace]
2. PATH: **pass by** the stairs (waypoint / traversal constraint, not a terminal stop)
3. GOAL (terminal): stop at [sphere decoration, *on* the cabinet] — **then**
- "pass by" is a mid-path waypoint the trajectory must include but not stop at — distinct from "stop at". The scorer checks both that the waypoint is hit and that the order (fireplace → stairs → cabinet) is preserved.

**General template:** every instruction-following question decomposes into an **ordered list of 2–4 sub-goals**, where each sub-goal is either (a) reach/near an anchor, (b) traverse a `path between/near X` corridor, (c) `pass by` a waypoint, or (d) `avoid` a forbidden corridor. The final element is always a terminal `stop at / stop by`. Correct *ordering* and *avoidance* are explicitly penalized if wrong, so the output waypoint sequence must preserve sequence and route around forbidden regions.

---

## 8. Design Implications

Ranked by point impact (instruction-following = 71% of points, object-reference = 24%, numerical = 6%):

- **[Highest value] Sequential path planner with ordered sub-goals.** Instruction-following is 180 of 255 points. The pipeline must parse a question into an **ordered** list of sub-goals, ground each anchor object, generate a **waypoint sequence** that visits them in order, and emit `Pose2D` waypoints on `/way_point_with_heading`. Order violations are penalized — sequencing is not optional.
- **[Highest value] Corridor ("path between/near X") and avoidance geometry.** Must physically route the trajectory *between* two grounded objects, *near* one object, and *around* forbidden `avoid` corridors. This is a geometric primitive distinct from point-goal navigation and appears in the majority of instruction-following questions (`path between` 10×, `near` 28×, `avoid`/`avoiding` 3×, `pass by` 1×).
- **[High value] Open-vocabulary object grounding.** 114 distinct nouns including rare/fine-grained categories the JSON never enumerates. A fixed closed-set detector will miss items; use open-vocabulary detection. Shared by object-reference and instruction-following (~94% of points combined). Must tolerate source typos (e.g. `refridgerator`).
- **[High value] Metric proximity ranking ("closest / farthest from").** The single most load-bearing relation for graded types (`closest to` 37×, `farthest/furthest from` 8×). Requires computing candidate-to-anchor distances and selecting the extremum — not a boolean predicate. Powers both object-reference disambiguation and instruction-following anchor selection.
- **[High value] Nested / multi-relation resolution.** 57% of object-reference and 100% of instruction-following questions chain ≥2 relations ("the bowl **on the table** **closest to** the folding screen"). Need a resolver that filters a candidate set by one relation, then ranks by another. Reused by the 4 nested numerical questions.
- **[Medium] Support-relation reasoning (`on`, `above`, `below`, `under`, `with X on it`).** Vertical/support relations (`on` 48×) and "with X on it" contents are common anchor disambiguators. Requires height/containment reasoning, not just 2D proximity.
- **[Medium] Object-reference marker output.** Emit a `visualization_msgs/Marker` bounding box on `/selected_object_marker`; scored by IoU (0–2), so box *tightness and localization* matter, not just picking the right object. Exactly one correct object exists per question.
- **[Lowest value] Integer counting with definite/indefinite anchor logic.** Numerical is only 6% of points and exact-match (no partial credit). Must count over the correct anchor set — distinguishing "on **a** sofa" (all matching anchors) from "on **the** sofa" (one salient anchor) — and be able to return small integers including 0. Guard against double-detection over-counts. Lowest engineering priority.
- **[Can defer] Attribute classifiers (color/size/material).** Only 10 total attribute mentions across 75 questions; a minor tie-breaker, not a primary matcher. Egocentric/view-dependent reasoning is **not required** by any training question (zero occurrences) — do not invest in robot-heading-relative spatial resolution unless the held-out test set introduces it.
