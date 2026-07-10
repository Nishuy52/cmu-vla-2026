# Proposal A — Deterministic-First Modular Pipeline

**Date:** 10 Jul 2026
**Target:** CMU VLA Challenge 2026 (submission deadline 15 Aug 2026)

---

## 0. Design Philosophy

Every operation whose correctness can be checked by a unit test is executed by
deterministic, verifiable code. Language models appear at exactly two points, and
only two: (1) translating the natural-language question into a structured query
(a small JSON DSL), and (2) optionally captioning object image crops to provide
attribute text. **No LLM or VLM ever performs raw spatial math, distance
comparison, or counting.** Counts come from an instance-tracked object map;
distances come from float arithmetic over map-frame coordinates; corridors come
from computational geometry.

Why this stance is the right bet for this challenge specifically:

- **The relation vocabulary is closed and tiny.** Programmatic analysis of all 75
  training questions shows the graded relations are `on`, `closest to`, `near`,
  `between`, `farthest/furthest from`, `above/below/under`, `with X on it`, plus
  the path forms `path between`, `path near`, `avoid(ing)`, `pass by`, `stop at`.
  Zero egocentric relations (`left of`, `facing`, etc.) occur. A dozen geometric
  predicates cover the entire distribution. A closed predicate set is exactly the
  case where hand-written geometry beats learned or prompted reasoning.
- **The organizers themselves documented the failure mode we are avoiding.** The
  challenge's organizing lab reports that LLMs given raw coordinates resolve
  "left of the bed" by picking the smallest x-coordinate; their fix (SORT3D — see
  §8) is a deterministic spatial-function toolbox that the LLM *calls* rather than
  reasoning about coordinates itself. We adopt that division of labor and push it
  further: in our design the LLM does not even orchestrate tool calls at answer
  time — it emits a declarative query *once*, and a deterministic resolver
  executes it. This removes LLM stochasticity from the answer path entirely.
- **Offline testability.** Because the resolver is pure code over an object map,
  the entire answer path can be regression-tested on Windows, without ROS, without
  a simulator, and without spending API tokens: feed synthetic or recorded object
  maps + the 75 training questions, assert outputs. LLM parsing is testable the
  same way (question string in, JSON out, compare against a hand-labeled parse
  set). This matters because our development machine has no ROS/sim until an
  Ubuntu reinstall (§7).
- **Debuggability under a no-retry scoring regime.** Every wrong answer has a
  discoverable cause: bad detection, bad track association, bad parse, or a bug
  in a geometric predicate. Each is independently observable in logs.

The cost of this stance — brittleness to question phrasings outside the DSL —
is mitigated by the LLM parser (which normalizes phrasing, typos like
"refridgerator", and synonyms into the fixed schema) and bounded by the evidence
that challenge questions are generated from the same heuristic relation templates
as the VLA-3D dataset (§8).

---

## 1. Pipeline Overview

The system is a single ROS 2 node (thin adapter) wrapping a pure-Python core.
It honors the exact test-time I/O contract: inputs are only the six allowed
topics; the *only* actuation channel is `/way_point_with_heading`
(`geometry_msgs/Pose2D`, map frame, `theta=0` — heading is ignored this year),
which feeds the base stack's `waypointConverter → localPlanner → pathFollower`
chain. We never touch internal topics such as `/way_point`.

```
 ALLOWED INPUTS (only these six)                       ALLOWED OUTPUTS (only these three)
 /challenge_question  String @1Hz ─┐                  ┌─> /numerical_response      Int32
 /state_estimation    Odom 100-200Hz┐│                │┌─> /selected_object_marker  Marker (CUBE, map frame)
 /camera/image        1920x640 @10Hz││                ││┌─> /way_point_with_heading Pose2D (x,y,theta=0)
 /registered_scan     PC2 map @5Hz  ││                │││
 /terrain_map         PC2 XYZI @5Hz ││                │││
 /terrain_map_ext     PC2 XYZI @5Hz ││                │││
        │                           ││                │││
        v                           vv                │││
 ┌─────────────────┐   ┌──────────────────────┐       │││
 │ OCCUPANCY &     │   │ QUESTION LATCH+DEDUP │       │││
 │ FRONTIER MAP    │   │ (1Hz republish-safe) │       │││
 │ 0.10 m grid from│   └─────────┬────────────┘       │││
 │ terrain XYZI    │             v                    │││
 │ (I<0.15=free,   │   ┌──────────────────────┐       │││
 │  I>=0.15=obst)  │   │ LLM QUESTION PARSER  │       │││
 └───────┬─────────┘   │ NL -> JSON query DSL │       │││
         │             │ (schema-validated,   │       │││
         │             │  retry w/ local      │       │││
         │             │  fallback parser)    │       │││
         │             └─────────┬────────────┘       │││
         v                       │                    │││
 ┌─────────────────┐             │                    │││
 │ EXPLORATION     │             │                    │││
 │ PLANNER (ours — │             │                    │││
 │ TARE/FAR are OFF│             │                    │││
 │ at test time):  │             │                    │││
 │ frontier select │             │                    │││
 │ + A* on grid    │             │                    │││
 │ + breadcrumb    │─────────────┼────────────────────┘││  (waypoints stepped <=2.5 m
 │   waypoints     │             │                      ││   ahead of vehicle, always)
 └───────┬─────────┘             │                      ││
         │ drives coverage       │                      ││
         v                       v                      ││
 ┌─────────────────┐   ┌──────────────────────┐         ││
 │ PERCEPTION      │   │ DETERMINISTIC        │         ││
 │ pano -> 4 tiles │   │ RESOLVER             │         ││
 │ open-vocab 2D   │──>│ executes DSL against │─────────┘│  (marker for object-ref)
 │ det + 2D track  │   │ OBJECT MAP with pure │──────────┘  (Int32 for numerical)
 │ + lidar fusion  │   │ geometry: on/near/   │
 │ -> 3D OBJECT MAP│   │ between/closest/     │─┐
 │ {id,label,      │   │ count/corridor/avoid │ │
 │  caption,AABB,  │   └──────────────────────┘ │
 │  pts,conf,hits} │                            v
 └─────────────────┘   ┌──────────────────────────────┐
                       │ WAYPOINT SEQUENCER (instr-   │
                       │ following): ordered subgoals │
                       │ -> A* paths honoring corridor│
                       │ gates + avoid-region costmap │──> /way_point_with_heading
                       │ -> breadcrumb Pose2D stream  │    (Pose2D, theta=0)
                       └──────────────────────────────┘
```

Control flow per run (system relaunches per question; no cross-question state):

1. On startup, begin frontier exploration immediately — do not wait for the
   question (it arrives at 1 Hz from the evaluation node; latch the first
   non-empty string, dedupe subsequent republishes).
2. As soon as the question is latched, send it to the LLM parser (network call
   overlaps with exploration — zero wasted time).
3. Perception runs continuously on keyframes during exploration, building the
   3D object map incrementally.
4. A **resolver tick** runs every ~5 s: try to execute the parsed DSL against
   the current object map. If it succeeds with sufficient confidence
   (§4), answer and stop; otherwise exploration continues, optionally biased
   toward finding the missing object categories.
5. Hard time gates (§4) force a best-effort answer before the 10-minute clock.

---

## 2. Exploration Policy (ours — the built-in planners are off)

At test time the challenge launches the *base* autonomy configuration only:
`local_planner`, terrain analysis, `waypoint_converter` — **neither TARE nor FAR
is running**, and their triggers (`/exploration_start`, `/goal_point`) are
interactive topics not in the allowed input list. The robot moves only toward
waypoints we publish. We therefore implement our own exploration planner.

### 2.1 Data structures over the XYZI terrain map

- **Occupancy grid** `G`: 2D array, 0.10 m cells, map frame, grown dynamically
  (start 40×40 m centered on origin, expand on demand). Each cell ∈
  {UNKNOWN, FREE, OBSTACLE}. Updated at 5 Hz from `/terrain_map_ext` (20 m
  radius): a terrain point with `intensity < 0.15` m marks its cell FREE, with
  `intensity ≥ 0.15` m marks OBSTACLE (this mirrors the base stack's own
  traversability cutoff — `waypointConverter` splits the terrain cloud on
  `obstacleHeightThre`, and `terrainAnalysis` uses 0.2 m internally). OBSTACLE
  observations win ties against FREE; cells decay back to their latest
  observation (no long-term Bayesian filtering needed — the terrain map is
  already temporally filtered upstream). `/terrain_map` (5 m, finer) refreshes
  the near field with priority.
- **Inflated costmap** `C`: `G` with obstacles dilated by the vehicle radius
  + margin (≈0.35 m total), plus soft cost near obstacles. Used by A*.
- **Frontier set** `F`: FREE cells 8-adjacent to UNKNOWN cells, clustered by
  connected components; clusters smaller than 5 cells (0.5 m) discarded.
  Recomputed at 1 Hz on the grid region that changed.
- **Visit mask**: cells within 3 m of any past vehicle pose (from
  `/state_estimation`), used to penalize re-exploration.

### 2.2 Algorithm

Greedy utility-based frontier selection (nearest-frontier with a small
information-gain term — deliberately simple, fully deterministic):

```
score(cluster) = expected_new_area(cluster)            # unknown cells within 4 m of centroid
                 / (path_len(vehicle -> centroid) + 3) # A* distance on C, metres
                 * category_bias(cluster)              # 1.0 default; >1 if question
                                                       # names an unfound category and a
                                                       # low-confidence detection of it was
                                                       # glimpsed in that direction
```

Pick the max-score cluster, plan A* on `C` from the vehicle cell to the
cluster centroid (goal snapped to nearest FREE cell). `category_bias` is the
only semantic influence and is computed from our own detector output — never
from an LLM judgment.

### 2.3 Waypoint emission (waypoints must stay near the vehicle)

The base stack warns that far-away waypoints strand the vehicle at dead ends.
We therefore never publish the frontier itself. A **breadcrumb follower** walks
the A* path: publish the path point ≤ 2.5 m ahead of the current vehicle pose
(line-of-sight-checked on `C`); when `/state_estimation` shows the vehicle
within 0.8 m of the active breadcrumb, advance to the next. Replan A* at 1 Hz
or on new obstacle cells intersecting the path. If the vehicle makes < 0.3 m
progress over 8 s, mark the current frontier cluster blacklisted for 60 s and
reselect. (The `waypointConverter` will also snap slightly-infeasible waypoints
into traversable area for us — a safety net, not something we rely on.)

### 2.4 Stopping criteria

Exploration stops when the first of these fires:

1. **Answer-ready:** the resolver tick succeeds with confidence (§4).
2. **Coverage complete:** frontier set empty (or only blacklisted clusters) —
   the scene is fully observed; answer from the final map.
3. **Phase budget expired:** the per-question-type exploration budget (§4)
   runs out; answer best-effort from the current map.

Scenes are mostly single rooms (the 15 training scenes: 13 single-room, 2
multi-room `home_building_*`); full coverage typically needs 1–3 minutes, so
criterion 2 is the common case, giving the resolver a complete map well inside
the budget.

---

## 3. Per-Question-Type Strategy

Point distribution across the training set (uniform 1/2/2 layout per scene):
instruction-following = **70.6 %** of points, object-reference = 23.5 %,
numerical = 5.9 %. Engineering priority follows exactly this order. All three
types share the same perception + parse + resolver machinery; they differ only
in the final output stage.

### 3.0 Shared: LLM question parser → query DSL

One LLM call converts the question to schema-validated JSON. The DSL (frozen
early, unit-tested against all 75 training questions with hand-labeled parses):

```json
{
  "type": "numerical | object_reference | instruction_following",
  "steps": [                      // instruction-following only; ordered
    {"kind": "goal|terminal_goal", "target": <ref>},
    {"kind": "corridor",  "between": [<ref>, <ref>]},
    {"kind": "path_near", "target": <ref>},
    {"kind": "pass_by",   "target": <ref>},
    {"kind": "avoid_corridor", "between": [<ref>, <ref>]}
  ],
  "target": <ref>,                // object_reference / numerical count target
  "anchor_quantifier": "definite|indefinite"   // numerical: "the sofa" vs "a sofa"
}
// <ref> = {"category": "pillow", "attributes": ["red"], "relations": [
//            {"rel": "on|near|closest_to|farthest_from|between|above|below|with_on_it",
//             "anchors": [<ref>, ...]} ]}   — relations nest recursively
```

Parser rules: canonicalize typos/synonyms ("refridgerator"→"refrigerator",
"furthest"→"farthest", "stop by"→terminal_goal, "pass by"→pass_by), map
"first/then/finally" to step order, map "avoiding the path between X and Y" to
`avoid_corridor`. The response is validated against a JSON Schema; on failure,
one retry with the validation error in-context; on second failure, a
deterministic regex/keyword fallback parser (coarse but covers the dominant
templates) takes over. The parse is done **once**; the LLM is not consulted
again during the run.

### 3.0b Shared: deterministic relation predicates

Pure functions over object-map entries `{label, caption, AABB(center, extents,
yaw≈0), points, confidence, n_observations}`; all thresholds are constants
tuned offline on the 15 training scenes' ground-truth object lists:

- `on(a, b)`: horizontal AABB overlap of `a`'s footprint with `b`'s footprint
  ≥ 50 % of `a`'s footprint, AND `a.z_min ∈ [b.z_max − 0.10, b.z_max + 0.25]`.
- `above/below/under(a, b)`: horizontal overlap (or centroid within inflated
  footprint) with the appropriate vertical ordering and a gap allowed
  (wall pictures above a bed etc. — no contact requirement).
- `near(a, b)`: closest 3D distance between AABBs ≤ max(1.2 m, 0.6 ×
  max footprint diagonal) — scale-adaptive, since "near" a sofa spans farther
  than "near" a cup.
- `closest_to / farthest_from(candidates, anchor)`: argmin/argmax of
  centroid-to-centroid (or box-to-box) distance. Metric comparison, exact.
- `between(a, b, c)`: `a`'s centroid within the capsule (stadium) of radius
  max(0.5 m, half the anchor gap × 0.4) around segment `centroid(b)–centroid(c)`,
  and projection parameter t ∈ (0.15, 0.85).
- `with_on_it(b, a)`: inverse of `on` — anchor `b` selected because `a` is on it.
- `count(<ref>)`: resolve the ref's candidate set, return its cardinality —
  counting is set-cardinality over tracked instances, never a model's guess.
- Nested refs resolve inner-first: filter candidates by each relation in turn,
  apply superlatives (`closest_to`) last as a ranking over survivors.
- **Not-found branch:** if a ref resolves to ∅, return an explicit NOT_FOUND
  (drives continued exploration or the fallback ladder in §6) rather than a
  hallucinated pick.

### 3.1 Instruction-following (6 pts each, 70.6 % of points — first priority)

Every training instruction decomposes into 2–4 ordered sub-goals: goals
("go near X", "stop at Y"), corridors ("take the path between A and B"),
pass-bys, and forbidden corridors ("avoiding the path between C and D").
Ordering and avoidance are explicitly scored. Execution:

1. Resolve every `<ref>` in `steps` against the object map (deterministic).
2. **Avoid-corridors:** compute the corridor capsule between the two anchors,
   stamp its cells OBSTACLE in a question-specific copy of costmap `C'` for the
   entire run. This makes avoidance a *planning invariant*, not a behavior —
   A* physically cannot route through it. (3 occurrences in training, but the
   penalty for violating one is real; the geometric cost of supporting it is a
   few dozen lines.)
3. **Goals / pass-bys:** target point = nearest FREE cell to the anchor's
   footprint, at a standoff of 0.6–1.0 m ("near"/"stop at" both score by
   trajectory proximity; we stop just outside the object's inflated footprint).
4. **Corridor ("path between A and B"):** insert a mandatory via-point at the
   midpoint of the A–B gap (snapped to FREE), and additionally stamp soft
   high cost outside the corridor capsule for that leg only, so A* prefers
   passing through the gate rather than around.
   "Path near X" → via-point in FREE space at ~0.8 m from X on the side
   consistent with the previous/next goal (choose the side minimizing total
   path length — deterministic tie-break).
5. Concatenate legs in step order, plan each leg with A* on `C'`, and stream
   the whole trajectory through the breadcrumb follower (§2.3). At the terminal
   goal, stop publishing waypoints (the scorer evaluates the followed
   trajectory; an early-finish signal is implicit in reaching the last goal and
   going quiet).

If a step's ref is NOT_FOUND after coverage is complete, degrade gracefully:
skip an unresolvable *pass-by/corridor* (partial credit preserved for order of
the resolvable goals) but never skip a *terminal* goal — for that, fall back to
the best category-only match (§6).

### 3.2 Object-reference (2 pts each, 23.5 %)

Resolve `target` to a single object; publish a `visualization_msgs/Marker`
CUBE on `/selected_object_marker`: `frame_id="map"`, pose = instance-cloud
centroid, scale = tight AABB extents of the fused instance point cloud (IoU
against the GT box is the score — box *tightness* matters, so we fit the box
to accumulated lidar points, trimmed at the 2nd/98th percentile per axis to
shed fusion outliers). Multi-constraint refs (57 % of these questions chain ≥2
relations, e.g. "the bowl on the table closest to the folding screen") are
handled natively by the nested resolver. Before publishing, a deterministic
verification pass re-checks every clause against the chosen object and demotes
to the next-ranked candidate on any hard failure — this targets the
"satisfied one constraint, dropped the other" failure mode the organizers
document for LLM-side resolution (§8).

### 3.3 Numerical (1 pt each, 5.9 % — lowest priority, near-free given the map)

`count()` over tracked instances. Two specific behaviors from the training
analysis: (a) **definite vs indefinite anchors** — "pillows on **the** sofa"
counts on the single best-matching anchor (most salient = most observations /
best relation fit), "pillows on **a** sofa" sums over *all* matching anchors;
(b) the counter may return **0**. Over-counting from duplicate tracks is the
main risk; the tracker's cross-frame association plus a final 3D
non-maximum-suppression merge (IoU > 0.3 between same-label instances → merge)
guards it. Publish `Int32`. Attributes ("red pillows", 5 color mentions in the
whole training set) are matched against the caption text by keyword — nearly
absent in the distribution, so no dedicated attribute classifier is built.

---

## 4. Time-Budget Management (10-minute clock, early-answer bonus)

The clock starts at system launch and includes exploration. Early finish earns
bonus points; overtime is penalized. Budget (t = seconds since our node is up):

| Phase | Window | Behavior |
|---|---|---|
| Boot + latch | t 0–10 | Models pre-loaded in the Docker image at build time and warmed in `__init__`; start exploring on first terrain map, don't wait for the question. |
| Parse | on latch | LLM parse in a worker thread, overlapped with exploration; timeout 20 s → one retry → regex fallback. Total parse path ≤ 45 s worst case, usually ~5 s. |
| Explore + resolve | until answer-ready | Resolver tick every 5 s. Answer as soon as: coverage complete, or the query resolves with margin (winner beats runner-up by ≥ 25 % on the deciding metric AND every referenced instance has ≥ 3 observations). |
| Soft gate | t = 300 (num/obj-ref), t = 240 (instr-following) | Stop *exploring for a better answer*; commit to best current resolution. Instruction-following gates earlier because it still needs driving time to execute the trajectory. |
| Execution reserve (instr-following) | ≥ 240 s | At 0.875 m/s planner speed and single-room scale, a 3-goal route is ~1–2 min; the reserve doubles that for replans/stalls. |
| Hard gate | t = 540 | Numerical/object-ref: publish best-effort answer (fallback ladder §6) — never publish nothing. Instruction-following: skip any unreached non-terminal sub-goal and drive directly (shortest safe path) to the terminal goal — terminal arrival is worth more than mid-path fidelity at this point. |
| Ceiling | t = 570 | Freeze all output; better to bank a partial answer than take the overtime penalty. |

Early-answer strategy: for numerical and object-reference, the whole answer is
one message — publish the moment answer-ready fires (often at coverage-complete,
~2–3 min in single-room scenes) and go idle, harvesting the bonus. For
instruction-following the bonus accrues from finishing the drive early, so the
lever is starting execution early: begin driving the route as soon as all
*terminal-goal* refs resolve, resolving remaining pass-by refs opportunistically
en route (perception keeps running while driving).

---

## 5. Concrete Model Choices (eval machine: i9 16-core, 32 GB RAM, RTX 4090 24 GB)

| Role | Model | Precision | VRAM | Latency (per call, 4090) |
|---|---|---|---|---|
| Open-vocab 2D detector | Grounding DINO (SwinT-OGC) with the question's categories + a fixed ~120-noun indoor prompt list (from the training-set noun inventory) | FP16 | ~3.5 GB | ~120 ms / tile |
| Instance segmentation (for clean lidar-mask fusion) | MobileSAM (box-prompted on detections) | FP16 | ~1 GB | ~15 ms / box |
| 2D cross-frame tracker | ByteTrack-style IoU/Kalman association (pure NumPy re-implementation) | CPU | 0 | ~1 ms |
| Per-object captioner (optional attribute text) | Qwen2.5-VL-Instruct-3B, best crop per track, once per stable track | INT4 (AWQ) | ~4 GB | ~250 ms / object |
| Question parser LLM | GPT-4o-class API (rules explicitly allow online APIs; token provided at runtime). Local fallback: Qwen2.5-7B-Instruct INT4 (~6 GB, loaded lazily only if the API fails) | — | 0 (API) | 2–5 s, once per run |

Peak steady-state VRAM ≈ 8.5 GB (detector + SAM + captioner), ≈ 14.5 GB if the
local LLM fallback loads — comfortable margin on 24 GB. The LLM is a **single
call per run** on the parse path only; a total API outage degrades us to the
regex fallback parser, not to failure — this is the payoff of keeping the LLM
out of the answer loop.

### Panorama handling (1920×640, 360° H × 120° V equirectangular)

- Process **keyframes**, not the 10 Hz stream: a frame every 0.5 m of travel or
  20° of yaw (from `/state_estimation`), ~1–2 Hz while moving.
- Split each keyframe into **4 yaw tiles of 120° with 15° overlap** (each tile
  resampled to ~800×640). At 5.33 px/°, gnomonic (pinhole) reprojection per tile
  removes most equirectangular distortion at detector input; detections in
  overlap zones are deduplicated by angular NMS before tracking.
- **Geometry never comes from the camera.** For each 2D mask, project the
  time-nearest `/registered_scan` points (already in map frame; transform to
  camera via `/state_estimation` pose and the fixed sensor extrinsic) into the
  panorama's azimuth/elevation model; points falling inside the mask, after
  range-based cluster filtering (reject background bleed-through by splitting
  on the dominant depth mode), form the instance's 3D points. Fused per track
  across frames → centroid, trimmed AABB, observation count. The camera
  contributes labels and captions only; lidar contributes all coordinates.
  (Rationale in §8: the organizing lab's own experiments show geometric
  inference from wide-FOV imagery generalizes poorly.)
- Per-keyframe perception budget: 4 tiles × 120 ms + SAM + fusion ≈ 0.7 s —
  keeps up with the keyframe rate with headroom.

---

## 6. Failure Modes and Mitigations

| # | Failure mode | Mitigation |
|---|---|---|
| 1 | Detector misses a rare category ("hookah", "sphere decoration", "framed records") | Question categories are injected verbatim (plus canonicalized form) into the detection prompt every keyframe once parsed; category-biased frontier scoring steers coverage; fallback ladder below. |
| 2 | Duplicate tracks inflate a count | ByteTrack association + final 3D same-label NMS merge; require ≥ 2 observations before an instance is countable. |
| 3 | One physical object split across two tiles / revisits | Angular-overlap dedup at detection time; 3D merge at map level. |
| 4 | LLM mis-parses the question | Schema validation + one repair retry; frozen DSL regression-tested against all 75 training parses; regex fallback parser. |
| 5 | API unreachable at eval time | Single-call design; local Qwen2.5-7B fallback; regex parser as final tier. Answer path needs no network at all. |
| 6 | Referenced object never found (typo, unseen, mis-detected) | Explicit NOT_FOUND branch → fallback ladder: (a) relax attributes; (b) relax the weakest relation (keep superlatives, drop `near` thresholds); (c) category-only nearest match; (d) numerical → answer the count of the unfiltered category (never abstain — 0 points either way, a guess dominates). |
| 7 | Ambiguous anchor ("the sofa" with several sofas) | Definite-article salience rule: pick the anchor maximizing (relation fit, observation count); for counts this only shifts which anchor is counted on — logged for post-hoc analysis. |
| 8 | Robot stalls / dead-end (waypoint too ambitious) | Breadcrumbs capped at 2.5 m with line-of-sight check; progress watchdog (0.3 m / 8 s) → frontier blacklist + replan; `waypointConverter` snap as last-ditch safety net. |
| 9 | Avoid-corridor makes the terminal goal unreachable | Detect A* failure on `C'`; the corridor block is shrunk stepwise (capsule radius −20 % per step) until a path exists — a narrow violation risk beats a guaranteed non-arrival. |
| 10 | Clock pressure | Hard gates in §4 fire unconditionally from a monotonic timer thread; every question type has a defined ≤ 570 s best-effort output. |
| 11 | Held-out scenes introduce egocentric relations ("on your left") | Not in any training question; if the parser emits an unknown relation, map it to `near` on the same anchors (graceful degradation) and log. Deliberately not engineered further — see Weaknesses. |

### Weaknesses of this approach (honest assessment)

- **Closed-DSL brittleness.** If the 3 held-out scenes contain phrasings whose
  *semantics* (not just wording) fall outside the DSL — genuinely novel relation
  types, view-dependent references, negated attributes — the deterministic
  resolver has no reasoning to fall back on. A big-VLM agent could plausibly
  wing such cases; we degrade to approximations. Our bet is distributional: the
  questions are generated from the same relation-template machinery as the
  training set, where these forms have zero occurrences.
- **Threshold sensitivity.** `on`, `near`, `between` are threshold predicates;
  values tuned on 15 training scenes may miscalibrate on held-out room scales
  (e.g. a large office where "near" spans 2 m). Scale-adaptive thresholds
  mitigate but don't eliminate this. An LLM judging "is this near?" from an
  image might occasionally get borderline cases right where a fixed threshold
  is wrong.
- **Perception is the single point of failure and is upstream of everything.**
  The design pushes all model risk into detection/tracking. A systematically
  missed or mislocalized category poisons counts, markers, and waypoints alike,
  and no downstream language reasoning can repair it. (Mitigations #1/#6 reduce
  but do not remove this.)
- **No semantic exploration value model.** Frontier scoring uses geometry plus
  a weak detector-driven bias — no learned "kitchens are that way" prior (cf.
  VLFM, §8). In these small single-room scenes coverage is cheap so this costs
  little; in a large held-out multi-room scene we may spend more of the budget
  exploring than a semantically-guided policy would.
- **Pragmatic implicature is unhandled by construction.** The organizers report
  failures like "the rightmost pillow" implicitly meaning "on the bed"; a
  literal geometric resolver will make exactly this class of error. The parser
  prompt instructs the LLM to make such implicatures explicit in the DSL, which
  moves the problem to the one place we do use a model — but it is a known,
  accepted residual risk.
- **Single-sample parse.** We deliberately avoid self-consistency voting over
  multiple LLM parses (cost/latency: negligible, actually — the real reason is
  keeping the pipeline single-path and debuggable). A rare unlucky parse is
  caught only by schema validation, which checks form, not meaning.

---

## 7. Implementation Plan (today = 10 Jul; submission 15 Aug 2026)

Constraint: development machine is Windows with no ROS/simulator until an
Ubuntu reinstall. Architecture answer: a **pure-Python core** (`vla_core/`) with
zero ROS imports — all interfaces are plain dataclasses (`OccupancyGrid`,
`ObjectMap`, `ParsedQuery`, `WaypointPlan`) and NumPy arrays — plus a **thin
ROS 2 adapter node** (`ai_module/`) that only converts messages ↔ dataclasses
and hosts the timers. Everything in `vla_core` runs and is tested on Windows;
the adapter is a few hundred lines written when Ubuntu lands.

| Week | Dates | Deliverables | Environment |
|---|---|---|---|
| 1 | Jul 10–16 | **Register (deadline Jul 15).** Freeze the DSL; hand-label parses for all 75 training questions; LLM parser + schema validation + regex fallback; parser regression suite green. Relation-predicate library + unit tests on synthetic object maps. | Windows, pure Python |
| 2 | Jul 17–23 | Resolver end-to-end on *ground-truth* object lists from the training scene packages (`object_list.txt` per scene — available in training scene downloads): all 30 object-ref + 15 numerical questions answered offline; measure accuracy, tune thresholds. Occupancy-grid + frontier + A* + breadcrumb logic with a scripted 2D kinematic stub simulating `/state_estimation` + synthetic XYZI terrain clouds. | Windows |
| 3 | Jul 24–30 | Perception offline: tile splitter, Grounding DINO + MobileSAM + tracker + lidar-fusion math on recorded/downloaded scene imagery and point clouds; captioner integration. Waypoint sequencer incl. corridor gates + avoid costmap, tested in the 2D stub against the 30 instruction-following questions using GT object positions + the provided `trajectory_q4/q5.ply` references as scoring proxies. **Ubuntu reinstall by Jul 30.** | Windows → Ubuntu |
| 4 | Jul 31–Aug 6 | ROS adapter node; Docker image (models baked in); full stack against the real sim on training scenes. First end-to-end timed runs. **First submission by Aug 6** (multiple submissions allowed; highest score counts — bank a score early). | Ubuntu + sim |
| 5 | Aug 7–13 | Iterate on timed-run telemetry: threshold tuning, stall handling, budget gates; per-scene dress rehearsals across all 15 scenes; failure-log triage. Second/third submissions. | Ubuntu + sim |
| Buffer | Aug 14–15 | Freeze, final Docker push, final submission form. | — |

Risk to the plan: the Ubuntu reinstall slipping. Mitigation: weeks 1–3 are
100 % OS-independent by design; even a Jul 30 → Aug 3 slip leaves ~10 sim days,
and the 2D stub keeps waypoint logic testable meanwhile.

---

## 8. Citations & Originality

This proposal was written from first principles against the challenge's
published interface and question set; the following prior work informed
specific, individually cited design decisions. No prose or code from any
source is reproduced here.

1. **SORT3D** — Nader Zantout, Haochen Zhang, Pujith Kachana, Jinkai Qiu, Ji
   Zhang, Wenshan Wang, *"SORT3D: Spatial Object-centric Reasoning Toolbox for
   Zero-shot 3D Grounding Using Large Language Models,"* arXiv:2504.18684, 2025.
   <https://arxiv.org/abs/2504.18684>. Borrowed concepts: the division of labor
   in which deterministic spatial functions, not the LLM, compute geometric
   relations; per-object VLM captioning of the best crop for attribute text;
   the documented LLM raw-coordinate failure mode motivating §0. Our design
   diverges by removing the LLM from answer-time orchestration entirely
   (single up-front parse instead of chain-of-thought tool calling).
   **License note: the SORT3D GitHub repository carries no license. No code
   from that repository is used, referenced during implementation, or adapted
   in any form — concepts only, cited here, re-implemented clean-room from the
   paper's description.**
2. **Transcrib3D** — *"Transcrib3D: 3D Referring Expression Resolution through
   Large Language Models,"* OpenReview 7j3sdUZMTF, 2024.
   <https://openreview.net/forum?id=7j3sdUZMTF>. Borrowed concept: expressing
   the referring expression as an executable/declarative filter that
   deterministic code evaluates over transcribed 3D detections — the direct
   ancestor of our parse-once-then-execute DSL and of counting as
   set-cardinality.
3. **VLFM** — Naoki Yokoyama et al., *"VLFM: Vision-Language Frontier Maps for
   Zero-Shot Semantic Navigation,"* ICRA 2024, arXiv:2312.03275.
   <https://arxiv.org/abs/2312.03275>. Considered and deliberately *not*
   adopted for frontier scoring (we use geometric utility + a detector-driven
   category bias instead of a VLM value map); cited as the source of the
   frontier-scoring idea our `category_bias` term is a deterministic
   simplification of, and acknowledged in §6 Weaknesses.
4. **3D-Mem** — *"3D-Mem: 3D Scene Memory for Embodied Exploration and
   Reasoning,"* CVPR 2025, arXiv:2411.17735. <https://arxiv.org/abs/2411.17735>.
   Borrowed concept: stop exploring when accumulated scene memory suffices to
   answer — the shape of our answer-ready stopping criterion (§2.4, §4).
5. **ConceptGraphs** — Qiao Gu et al., *"ConceptGraphs: Open-Vocabulary 3D
   Scene Graphs for Perception and Planning,"* ICRA 2024, arXiv:2309.16650.
   <https://arxiv.org/abs/2309.16650>. Supporting evidence for incremental
   open-vocabulary object-centric maps as the substrate for language grounding.
6. **OpenEQA** — Arjun Majumdar et al., *"OpenEQA: Embodied Question Answering
   in the Era of Foundation Models,"* CVPR 2024. <https://open-eqa.github.io/>.
   Evidence that frontier-VLM baselines fail specifically on counting and
   spatial questions — the empirical basis for routing counts through a tracked
   instance map (§3.3) rather than a VLM.
7. **VLA-3D** — Haochen Zhang et al., *"VLA-3D: A Dataset for 3D Semantic Scene
   Understanding and Navigation,"* arXiv:2411.03540.
   <https://arxiv.org/abs/2411.03540>. The challenge's stated training /
   question-generation resource; its closed relation vocabulary underwrites the
   closed-DSL bet (§0) and provides offline evaluation fixtures (§7 week 2).
8. **Grounding DINO** — Shilong Liu et al., *"Grounding DINO: Marrying DINO
   with Grounded Pre-Training for Open-Set Object Detection,"* ECCV 2024,
   arXiv:2303.05499. Detector choice (§5).
9. **ByteTrack** — Yifu Zhang et al., *"ByteTrack: Multi-Object Tracking by
   Associating Every Detection Box,"* ECCV 2022, arXiv:2110.06864. Association
   scheme re-implemented independently in NumPy (§5); concept cited.
10. **MobileSAM** — Chaoning Zhang et al., *"Faster Segment Anything: Towards
    Lightweight SAM for Mobile Applications,"* arXiv:2306.14289. Segmentation
    choice (§5). **Qwen2.5-VL** — Qwen Team, Alibaba, *"Qwen2.5-VL Technical
    Report,"* arXiv:2502.13923. Captioner choice (§5).
11. Challenge interface facts (topics, message types, scoring weights, timing,
    planner availability, terrain-map semantics) are taken from the official
    challenge repository *CMU-VLN-Challenge-2026* (github.com/Yuxin916/
    CMU-VLN-Challenge-2026) and its `autonomy_stack_mecanum_wheel_platform`
    submodule, and from the challenge site (ai-meets-autonomy.com).

**Originality statement.** The following elements are, to the author's
knowledge, original to this proposal: the single-parse declarative DSL with
schema-validated repair and a regex fallback tier (no answer-time LLM); the
avoid-corridor-as-costmap-invariant and corridor-gate via-point construction
for `path between` (§3.1); the definite/indefinite anchor-quantifier counting
rule (§3.3); the phase-gated time-budget schedule with an execution reserve
and terminal-goal triage at the hard gate (§4); and the Windows-first
pure-core/thin-adapter delivery plan (§7). All borrowed concepts are cited
inline above; no code is reused from any unlicensed repository.
