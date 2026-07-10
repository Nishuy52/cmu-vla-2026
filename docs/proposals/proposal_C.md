# Proposal C — Expected-Score-Maximisation Design

**Date:** 10 Jul 2026
**Target:** CMU VLA Challenge 2026, submission deadline 15 Aug 2026 (AoE)
**Team reality:** one developer, assisted by code-generation tooling; currently on Windows (no ROS/sim until an Ubuntu reinstall); NUS SoC GPU cluster reachable over SSH but optional; eval machine is i9 16-core / 32 GB RAM / RTX 4090.

---

## 0. Design philosophy

This proposal does not start from an architecture. It starts from the scoring function, computes
where the points are, prices each candidate subsystem in developer-days, and only then commits to
an architecture — plus, just as importantly, an engineering **schedule** with explicit cut-lines.
Two structural facts of this competition dominate every design decision:

1. **The score distribution is severely skewed.** Instruction-following is 70.6% of available
   points; numerical is 5.9%. A subsystem that adds 5% accuracy on instruction-following is worth
   ~9 points; the same 5% on numerical is worth 0.75 points. Effort must follow that gradient.
2. **Multiple submissions are allowed and the highest score is kept.** The correct strategy is
   therefore not "build the best system by Aug 15" but "have a scoring system on the leaderboard
   as early as possible, then ratchet upward." Every week of the schedule ends in a state that
   could be submitted. Risk is bounded by construction, not by hope.

Everything below follows from these two facts.

---

## 1. The expected-points model

### 1.1 The scoring math

Per the challenge README and the training question set (75 questions across 15 scenes, uniform
1 numerical + 2 object-reference + 2 instruction-following per scene; the 3 held-out test scenes
will almost certainly follow the same shape, i.e. **15 test questions, 51 test points**):

| Type | Qs (train/test) | Pts each | Train pts | % of pts | Output | Scoring |
|---|---|---|---|---|---|---|
| Instruction-following | 30 / 6 | 0–6 | 180 | **70.6%** | `Pose2D` waypoint sequence | partial credit on trajectory: ordered sub-goal adherence, corridor constraints, avoid-region penalties |
| Object-reference | 30 / 6 | 0–2 | 60 | **23.5%** | `Marker` bounding box | GT-box overlap; **the box center doubles as a nav goal** |
| Numerical | 15 / 3 | 0–1 | 15 | **5.9%** | `Int32` | exact match, no partial credit |

Plus: **10-minute hard budget per question** (system relaunched per question — no map carries
over), early finish earns a tiebreak bonus, overtime is penalised.

Three second-order consequences the architecture must respect:

- **Partial credit makes instruction-following the safest investment, not just the largest.**
  A trajectory that nails 2 of 3 sub-goals still scores. Exact-match numerical is all-or-nothing.
  Expected points per unit of accuracy is therefore even more skewed toward IF than the raw 70.6%.
- **Object-reference and instruction-following share ~90% of their machinery** (open-vocab
  grounding + spatial-relation resolution + a metric object map). Numerical shares the grounding
  stack too (count = filtered query over the same object map). One grounding core serves all
  three types; there is no reason to build three pipelines.
- **The Marker's center being a legal nav goal** means object-reference gets navigation "for
  free" once the box is chosen — and conversely, a merely *approximate* box still earns overlap
  credit. Box tightness is a second-order refinement, picking the right object is first-order.

### 1.2 Subsystem → points table (the actual prioritisation instrument)

Effort is in solo developer-days assuming code-generation assistance. "Points unlocked" is my
expected marginal test-set contribution (out of 51), estimated from training-set question
structure (`question_analysis`-style decomposition) and published accuracy numbers for the
component class (SORT3D, VLFM, OpenEQA baselines — citations in §8).

| # | Subsystem / capability | Points unlocked (E[Δ], /51) | Effort (days) | Pts/day | Rank |
|---|---|---|---|---|---|
| 1 | **Waypoint actuation + question latch + always-answer scaffolding** (publish *something* on the right topic for every type, never time out silently) | ~4–6 (floor: partial IF credit for driving at a plausibly-grounded first anchor; 50/50 marker guesses; modal integer guess) | 2 | ~2.5 | 1 |
| 2 | **Metric object map**: open-vocab 2D detection on panorama crops → lidar-fused 3D boxes → cross-frame instance merge | ~10–14 (prerequisite multiplier for everything; alone it converts guesses into grounded answers) | 6 | ~2.0 | 2 |
| 3 | **Spatial-relation toolbox** (deterministic geometry: near / on / between / closest-farthest / above-below / corridor / avoid) + LLM tool-calling resolver | ~12–16 (this is where IF and OR points actually get earned; organizers' own ablations show LLMs fail raw-coordinate geometry) | 5 | ~2.8 | 1–2 |
| 4 | **Sequential IF executor**: parse instruction → ordered sub-goals → waypoint sequence with corridor midpoints and avoid-region detours | ~8–10 (converts grounded anchors into the trajectory the scorer actually grades) | 4 | ~2.2 | 2 |
| 5 | **Frontier exploration sized to room scale** (coverage-greedy, terrain-map-driven) | ~5–7 (without it, unseen objects = unanswerable; most scenes are single rooms so cost is low) | 3 | ~2.0 | 2 |
| 6 | **Early-answer / confidence policy** (bank the early-finish bonus; stop exploring when the answer is stable) | ~1–3 direct + tiebreak value | 1 | ~2.0 | 3 |
| 7 | Numerical counting with definite/indefinite anchor logic + de-dup | ~1.5–2 (of 3 available) | 2 | ~0.9 | 4 |
| 8 | Per-object VLM captioning (color/material attributes) | ~1–2 (only ~10 attribute mentions in 75 training questions; but +11.6% on hard/view-dep statements in SORT3D's ablation hedges the unseen scenes) | 2 | ~0.8 | 4 |
| 9 | Box-tightness refinement (per-instance point-cloud extent fitting for IoU) | ~1–1.5 | 2 | ~0.6 | 5 |
| 10 | Self-consistency voting (3 LLM samples, majority) on 6-pt questions only | ~1–2 (SORT3D reports ±6% run variance) | 1 | ~1.5 | 3 |
| 11 | Fine-tuning a grounding model on VLA-3D | ~0–2 (zero-shot toolbox systems already match supervised SOTA on Sr3D-style language, which is what the challenge generator emits) | 10+ | ~0.1 | CUT |
| 12 | Dense per-point language features (AM-RADIO-style), learned VO, end-to-end VLN policy | ~0 within this budget | 15+ | ~0 | CUT |

**Reading of the table.** Rows 1–6 are the whole game: roughly 40–50 of 51 expected points for
~21 developer-days. Rows 7–10 are polish (≈6 days for ~5 points) taken only if the schedule
holds. Rows 11–12 are research-shaped work with competition-shaped costs; they are cut before
the project starts, which is why the SoC GPU cluster is *optional* in this design — nothing
critical-path trains anything.

**Cheap points, named explicitly:**

- *The floor is not zero.* A system that always publishes a marker on its best object guess,
  always publishes an integer (the training-set modal count region is 2–4; publish the map count
  or 3 if blind), and always drives toward its best-grounded first anchor collects partial credit
  structurally. Row 1 exists so no question ever scores 0 for reasons of plumbing.
- *Object-reference is the highest points-per-difficulty type.* 2 points for one grounded box,
  scored by overlap (forgiving), sharing all machinery with IF. Expect to convert OR at a higher
  rate than IF and treat it as the reliability backbone of the score.
- *The early-finish bonus is nearly free* once a confidence policy exists (row 6): it is one
  state-machine transition, not a subsystem.

---

## 2. Pipeline overview

### 2.1 The I/O contract we must honor

At test time the AI module may consume **only**: `/camera/image` (1920×640 360° panorama,
10 Hz), `/registered_scan` (PointCloud2, map frame, 5 Hz), `/sensor_scan`, `/terrain_map` (5 m,
XYZI where intensity = obstacle height; <~0.15 m ⇒ traversable), `/terrain_map_ext` (20 m),
`/state_estimation` (Odometry, 100–200 Hz), plus `/challenge_question` (String, republished at
1 Hz — must be latched/deduped). It may emit **only**: `/way_point_with_heading` (Pose2D,
theta=0 this year), `/selected_object_marker` (Marker, map frame, CUBE), `/numerical_response`
(Int32).

Crucially, the built-in TARE exploration planner and FAR route planner are **not launched** in
the challenge configuration, and their triggers (`/exploration_start`, `/goal_point`) are not
legal test-time topics. Exploration is entirely ours, and the *only* actuation channel is
`/way_point_with_heading` → the stack's `waypointConverter` → local planner → path follower.
The converter snaps infeasible waypoints into traversable area, and the stack warns against
far-away waypoints (dead-end risk) — so we emit **incremental, nearby goals** (≤3–4 m steps).

### 2.2 Architecture

The core is a **pure-Python, ROS-free library** (`vla_core`) developed and unit-tested on
Windows against recorded/synthetic fixtures; a **thin ROS 2 adapter node** (~300 lines) does
nothing but message conversion and topic plumbing. This is forced by the environment (no ROS on
the dev machine until the Ubuntu reinstall) but is also the right call on merits: every
subsystem is testable without the simulator, which is the scarce resource on this schedule.

```
                        ┌──────────────────────────────────────────────────────────┐
                        │                    ai_module container                     │
   /challenge_question ─┼─► [question latch/dedupe]                                 │
        (String,1Hz)    │        │                                                  │
                        │        ▼                                                  │
   /camera/image ───────┼─► ┌─────────────┐    ┌──────────────────────────────┐    │
   (1920x640 pano,10Hz) │   │ PERCEPTION  │    │  REASONER (LLM API, allowed)  │    │
                        │   │ pano→4 crops│    │  1. parse Q → typed plan      │    │
   /registered_scan ────┼─► │ open-vocab  │    │     (ordered sub-goals /      │    │
        (PC2,5Hz)       │   │ det+seg     │    │      referent / count filter) │    │
                        │   │ lidar fuse  │    │  2. tool-calls into ────────┐ │    │
   /state_estimation ───┼─► │ 3D instance │    │     SPATIAL TOOLBOX         │ │    │
      (Odom,100-200Hz)  │   │ merge/track │    │     (deterministic Python:  │ │    │
                        │   └──────┬──────┘    │      near/on/between/       │ │    │
   /terrain_map(_ext) ──┼─► ┌──────▼──────┐    │      closest/farthest/      │◄┘    │
        (XYZI,5Hz)      │   │ WORLD MODEL │◄───│      corridor/avoid)        │      │
                        │   │ object map  │    └───────────┬─────────────────┘      │
                        │   │ {id,label,  │                │                        │
                        │   │  box,conf,  │        ┌───────▼────────┐               │
                        │   │  best-crop} │        │  EXECUTIVE FSM │               │
                        │   │ + trav grid │        │ EXPLORE⇄ANSWER │               │
                        │   │ + frontiers │        │ budget clock   │               │
                        │   └─────────────┘        │ early-answer   │               │
                        │                          └──┬────┬────┬───┘               │
                        │                             │    │    │                   │
                        │      /way_point_with_heading│    │    │/numerical_response│
                        │            (Pose2D) ◄───────┘    │    └──► (Int32)        │
                        │      /selected_object_marker ◄───┘                        │
                        │            (Marker, map frame)                            │
                        └──────────────────────────────────────────────────────────┘
                                  ▼ (system container, provided — not ours)
                    waypointConverter → localPlanner → pathFollower → /cmd_vel
```

**Division of labor (deliberate, organizer-validated):** the camera is semantics-only, the lidar
is geometry-only, the LLM reasons *sequentially over text* and never computes geometry from raw
coordinates — all spatial predicates are deterministic Python functions the LLM invokes. This
split is the central lesson of the organizers' own SORT3D work (Zantout et al., arXiv:2504.18684;
see §8) and their documented failure mode ("left of" → LLM picks min-x). We re-implement the
concept clean-room (the SORT3D repo carries no license — no code reuse).

**Executive FSM** (one instance per launch, since each question is a fresh 10-minute run):

```
BOOT → LATCH_QUESTION → PARSE (LLM, ~5 s) → EXPLORE ⇄ GROUND → ACT → ANSWERED → (idle)
                                              ▲ frontier loop     │
                                              └── confidence gate ┘
```

- `PARSE` classifies the type from the question text (Find… / How many… / else) and emits a
  typed plan: referent spec (OR), count filter (Num), or ordered sub-goal list (IF).
- `EXPLORE` runs the frontier policy (§3), continuously feeding perception; grounding is
  re-attempted every few seconds against the growing object map.
- `ACT` per type: publish marker; publish integer; or step through the waypoint sequence,
  advancing on proximity (dummy-node pattern: next waypoint when within ~1 m), inserting
  corridor midpoints and avoid-region detours.
- A **watchdog** guarantees an answer before the budget expires regardless of state (§4.5).

---

## 3. Exploration policy — sized to the scoring reality

**Does full-coverage exploration ever pay? Mostly no.** 12 of the 15 training scenes are single
rooms; a 360° camera + 360° lidar sees most of a room from 2–4 viewpoints. Full coverage is only
plausibly necessary in the two `home_building_*` multi-room scenes (and whatever multi-room
scenes the test set holds). The expected-score framing makes the tradeoff explicit: an extra
3 minutes of exploration buys points only if the answer's referent hasn't been observed yet, and
costs the early-finish bonus plus overtime risk. So exploration is **lazy, question-conditioned,
and budgeted per type**:

**Phase 1 — orientation sweep (0:00–1:00).** Immediately publish a short spiral/rotation pattern
of nearby waypoints (2–3 m steps) to sweep the local room. With the panorama camera this alone
typically observes >70% of a single-room scene. Perception runs from the first frame; the
question is parsed in parallel during the sweep.

**Phase 2 — targeted frontier exploration (until per-type budget).** Standard frontier
extraction from a 2D traversability grid rasterised from `/terrain_map_ext` (intensity threshold
~0.15 m; cells beyond sensed extent = unknown; frontier = free∧adjacent-to-unknown). Frontiers
are scored by `expected_information_gain / travel_cost`, with a **semantic bonus** for frontiers
whose adjacent panorama sector shows high open-vocab detector affinity to the question's
mentioned nouns — a lightweight, detector-based rendition of VLFM's semantic frontier value maps
(Yokoyama et al., arXiv:2312.03275). Waypoints toward the chosen frontier are emitted in ≤3 m
increments (stack guidance: distant waypoints risk dead-end lock).

**Phase 3 — verification approach (opportunistic).** Once a candidate referent is grounded but
low-confidence, drive one waypoint to within ~2 m of it for a close re-observation before
answering. Skipped when confidence is already high (this is where the early-answer policy bites).

**Per-type exploration budgets** (hard cutoffs; after cutoff the FSM answers with best-available):

| Type | Explore budget | Rationale |
|---|---|---|
| Numerical | ≤3.5 min | must see *all* qualifying anchors (indefinite-article questions sum over anchors), but worth only 1 pt — never let it burn the clock |
| Object-reference | ≤4 min | needs referent + its anchor objects; usually one room |
| Instruction-following | ≤4.5 min, interleaved | needs *all* sub-goal anchors, but execution itself takes 2–4 min of driving; exploration and execution interleave — begin executing sub-goal 1 as soon as it's grounded while continuing to scan for later anchors en route |

The IF interleaving is the single most important time decision: waiting for a complete map
before moving wastes the fact that driving toward sub-goal 1 *is* exploration with a 360°
sensor suite.

**Stopping rule** (3D-Mem-inspired "explore while answer-relevant memory is insufficient",
Yang et al., arXiv:2411.17735, recast as a cheap heuristic): stop exploring when every noun
phrase in the parsed question has ≥1 grounded instance with detector confidence above threshold
*and* the last 45 s of exploration added no new instance of any mentioned category — or when the
type budget expires, whichever is first.

---

## 4. Per-question-type strategy (with the early-answer policy made explicit)

### 4.0 The early-answer decision rule

The bonus/penalty structure means the module needs an explicit stop-vs-continue policy, not a
vibe. The rule, uniform across types:

> **Answer now if** (a) the toolbox resolver returns a unique referent whose margin over the
> runner-up exceeds a threshold (distance-ratio ≥1.5 for closest/farthest predicates; unique
> candidate after filtering otherwise), **and** (b) 45 s of additional observation has not
> changed the answer. **Otherwise keep exploring until the type budget, then answer with the
> current best regardless.** Never pass; never overrun.

Rationale: the bonus is a tiebreak/small additive term while a wrong answer on a 6-pointer costs
up to 6 — so the policy is asymmetric by type. For **numerical** (1 pt), answer aggressively the
moment counts stabilise: the upside of more exploring is capped at 1 point. For **IF** (6 pts),
never cut exploration short to bank a bonus if any sub-goal anchor is ungrounded — the bonus can
never outweigh a dropped sub-goal. Object-reference sits between: answer early on unique
high-margin referents (common — most OR questions have exactly one plausible candidate once the
anchor relation is applied), explore to budget on multi-distractor cases.

### 4.1 Instruction-following (6 pts × 6 test questions — the business)

Training-set structure (100% multi-constraint, avg 3.6 relations, template: ordered 2–4
sub-goals ending in a terminal "stop at", with occasional `path between X and Y` corridors,
`pass by` waypoints, and 3 instances of `avoiding the path between…`):

1. **Parse** (LLM): question → JSON list of steps, each `{kind: GOAL|CORRIDOR|PASS_BY|AVOID,
   referent: <noun phrase + relations>, order: n, terminal: bool}`. Deterministic schema,
   validated; re-prompt once on schema failure, fall back to a regex/keyword splitter
   (First/then/and stop/avoiding) if the LLM output is still malformed.
2. **Ground each referent** through the toolbox (§4.2 machinery, shared).
3. **Compile to waypoints**: GOAL → free-space point ~1 m from the object on its traversable
   side; CORRIDOR → the midpoint of the segment between the two anchors (plus approach/exit
   points collinear with the corridor axis so the trajectory demonstrably passes *through*);
   PASS_BY → a via-point within ~1.5 m, not a stop; AVOID → inflate a forbidden capsule between
   the two anchors into the traversability grid and let A* on the grid route around it for
   *every* leg, since avoidance scopes the whole traversal.
4. **Execute** with the proximity-advance loop; theta always 0.
5. **Partial-credit discipline**: if step k's referent cannot be grounded by budget, do NOT
   abort — substitute the best-guess instance of the bare category (or skip to step k+1 if
   nothing matches) and continue. Ordered partial credit means a 4/6 beats a frozen robot's 0.

### 4.2 Object-reference (2 pts × 6)

1. LLM parses the referent into `{category, attribute?, relations: [(predicate, anchor), …]}` —
   57% of training OR questions chain ≥2 relations ("the bowl **on** the table **closest to**
   the folding screen"), so resolution is filter-then-rank: apply relation 1 to cut the
   candidate set, rank by relation 2's metric, take the extremum.
2. All predicates are deterministic: `on` = XY-overlap + base-above-top within ε; `near/closest/
   farthest` = 3D centroid distances; `between` = projection onto the anchor–anchor segment;
   `above/below/under` = vertical interval tests. The LLM only chooses *which* tools to call, in
   what order — one in-context worked example in the prompt (chain-of-thought tool use, the
   SORT3D pattern).
3. **Cheap verification pass** (targets SORT3D's own documented failure mode of satisfying one
   clause and dropping another): after selection, a single LLM call — "does object #k satisfy
   EVERY clause of the question? yes/no per clause" — using the toolbox's computed predicate
   values as evidence; on "no", take the runner-up. One extra API round-trip for the 2- and
   6-point types only.
4. Publish the Marker: map frame, CUBE, pose = fused instance centroid, scale = 3D extent of the
   instance's merged lidar points (padded ~10%; slight over-size is safer than under-size for an
   overlap metric). Marker box center is simultaneously a legal nav goal — for OR questions we
   also drive one waypoint toward it, which costs seconds and hedges the case where evaluation
   rewards proximity.

### 4.3 Numerical (1 pt × 3 — floor-protection only)

Counting is a **deterministic filter over the object map**, never a VLM asked to count in an
image (OpenEQA's headline finding: frontier-model VLMs are bad at counting; and Transcrib3D's
lesson: let the LLM write the filter, let code execute it). The LLM emits a predicate
(category + relation + anchor scope); Python counts matching instances. Definite vs indefinite
anchors ("on **the** sofa" vs "on **a** sofa") switch between counting on the single resolved
anchor vs summing over all qualifying anchors. Instance merge (cross-frame association by 3D
IoU + label) is the over-count defence. Able to return 0, but if the map is empty of the target
category at budget expiry, publish the small-integer prior (2) rather than 0 — training-set
phrasing presupposes existence.

### 4.4 Panorama handling

The 1920×640 equirectangular strip is split into **4 pinhole-ish crops of 480×640 (90° HFOV
each)** with ~10° overlap, processed round-robin at ~2 Hz effective per sector (detector runs at
~8 crops/s on the 4090; that's the full ring every ~0.5 s, far above the 5 Hz lidar it fuses
with). Detections are mapped back to azimuth via crop origin; lidar association: project
registered-scan points into the crop using the known camera model, take the segmentation-masked
points' 3D centroid/extent. Distortion at crop edges is tolerable at 90° HFOV; we do not attempt
any geometry *from* the image (the organizers' own thesis line documents wide-FOV visual
geometry failing to generalise — lidar owns geometry).

### 4.5 Watchdog (the floor, again)

An independent timer thread answers unconditionally at T-30 s with best-available: current best
marker; current count (or prior); for IF, whatever waypoints remain unexecuted are flushed as a
final direct-to-terminal-goal sequence. Silence is the only unforgivable failure.

---

## 5. Concrete model choices, VRAM and latency

Eval GPU is an RTX 4090 (24 GB). Budget:

| Component | Model | Precision | VRAM | Latency | Notes |
|---|---|---|---|---|---|
| Open-vocab detection | GroundingDINO (SwinT) | fp16 | ~3 GB | ~90 ms/crop | prompt = question nouns + a fixed ~60-category indoor list; community consensus choice, also the organizers' |
| Segmentation (for lidar masking) | SAM 2 / MobileSAM (small) | fp16 | ~2 GB | ~40 ms/box | box-prompted only on kept detections |
| Attribute captioning (stretch, row 8) | Qwen2.5-VL-7B-Instruct | int8/AWQ | ~9 GB | ~0.6 s/crop | best-crop per instance only, lazily, only for categories the question mentions with an attribute |
| Reasoner | GPT-4o-class API (LLM/VLM APIs explicitly allowed; token provided at runtime) | — | 0 | 2–6 s/call | 3–6 calls per question ≈ ≤30 s total, negligible vs the 10-min budget |
| Local reasoner fallback | Qwen2.5-7B-Instruct | int8 | shares captioner budget | ~4 s/call | shipped in the image; auto-switch if two consecutive API calls fail — hedges eval-machine connectivity |

Peak concurrent VRAM ≈ 3+2+9 ≈ **14 GB**, comfortably inside 24 GB with CUDA overhead. Without
the captioner (the MVS configuration) it is ~6 GB. CPU/RAM: object map is a few hundred
instances of centroids+extents+one JPEG crop each — trivial. The perception loop's steady state
(4-crop ring @ ~0.5 s/revolution, fused at 5 Hz lidar rate) comfortably outruns a 0.875 m/s
robot.

API cost/latency risk is bounded: calls are few, small (the scene is serialised as a compact
text object list `{id, label, cx, cy, cz, size, caption?}`, not images), and each has the local
fallback. Self-consistency voting (row 10) triples reasoner calls on IF questions only — still
<1 min of the budget.

---

## 6. Failure modes, mitigations — and honest weaknesses

### 6.1 Failure modes and mitigations

| Failure | Likelihood | Mitigation |
|---|---|---|
| Referent never observed (occlusion, missed frontier, multi-room test scene bigger than training) | Med | semantic frontier bonus steers toward question nouns; per-type budget then best-guess answer; watchdog floor |
| Detector misses rare category ("hookah", "framed records", misspelled "refridgerator") | Med | prompt GroundingDINO directly with the question's noun phrases (open-vocab prompts absorb typos and rare nouns better than closed sets); category-level fallback to nearest embedding match |
| Double-counting / split instances → wrong Int32 | Med | 3D-IoU + label instance merge; costs at most 1 pt when it fails |
| LLM logic slip: satisfies one clause, drops another | Med-high | deterministic toolbox + per-clause verification pass (§4.2.3); the two known SORT3D failure classes get explicit regression tests |
| LLM run-to-run variance (±6% reported) | Med | temperature 0; self-consistency ×3 on IF |
| Malformed LLM output (schema break) | Med | strict JSON schema + one re-prompt + regex fallback parser |
| API unreachable on the eval machine | Low-med | local 7B fallback baked into the image; auto-switch |
| Robot stuck (dead-end, waypoint snapped somewhere odd) | Med | ≤3 m waypoint increments; progress monitor: if displacement <0.3 m over 20 s, re-route via the frontier grid; last resort, waypoint back toward last-known-good pose |
| Avoid-region violated en route (scored penalty) | Low (3/30 train questions) | forbidden capsule inflated into the A* grid for all legs, not just adjacent ones |
| Overtime penalty | Low | budgets + watchdog make overrun structurally impossible short of a process hang; the launch script supervises and restarts the node once |
| Docker/DDS integration surprises (CycloneDDS RMW mismatch, marker-topic slash discrepancy) | Med early, low late | integration is scheduled in week 1 *after* the Ubuntu reinstall, not week 5; publish on fully-qualified `/selected_object_marker`; pin `RMW_IMPLEMENTATION=rmw_cyclonedds_cpp` |

### 6.2 Weaknesses of this approach (honest)

- **Score-first engineering deliberately underinvests in robustness for the 3 unseen scenes.**
  The budgets, priors (small-count integers, single-room sweep-first exploration), and the
  decision to cut attribute/egocentric machinery are all fitted to the *training* distribution.
  If the held-out scenes are systematically harder — larger multi-room layouts, egocentric
  phrasing ("to your left", zero training occurrences), deliberately imperfect references in the
  IRef-VLA style, higher distractor counts — several of my expected-point estimates degrade
  together, because they share the same distributional assumption. This is the design's
  correlated risk, and it is accepted knowingly: with 26 developer-days, hedging against a
  distribution shift I cannot observe costs points against the distribution I can.
  Partial hedges taken: open-vocab (not closed-set) detection, budgets that degrade to
  best-guess rather than to silence, and the captioning stretch goal (whose main measured value
  is precisely on hard/view-dependent statements).
- **No dialogue, no true ambiguity handling.** If a test question is genuinely ambiguous, the
  system picks the toolbox-margin winner. The organizers' ambiguity taxonomy suggests they think
  about this; a wrong-but-confident answer is this design's response, mitigated only by the
  verification pass.
- **The reasoner is a rented brain.** API dependence is a real single point of failure at eval
  time; the local-7B fallback is materially weaker at multi-step parsing, and its quality gap on
  IF questions is untested until week 3.
- **One developer, hard deadline, sim only arrives mid-schedule.** The first two weeks produce
  code that has never touched ROS or Unity. The pure-core/thin-adapter pattern is designed to
  make that safe, but integration surprises (timing, frames, QoS) concentrate into week 3 —
  which is why the MVS gate is placed there and not later.
- **Zero-shot ceiling.** By cutting fine-tuning, we accept whatever ceiling the
  toolbox-plus-frontier-LLM pattern has (~60–75% grounding accuracy on hard statements in
  published results). A team that fine-tunes on VLA-3D's 9.7M statements *and* executes well
  could out-ground us. My bet — the expected-score bet — is that within this time budget,
  execution reliability (never time out, never score 0 for plumbing reasons, always collect
  partial credit) dominates grounding-accuracy deltas. That bet can be wrong.

---

## 7. Build schedule to 15 Aug (with cut-lines)

~5.5 weeks. Registration (deadline **15 Jul**) is done in week 0. Every week ends submittable.

**Minimum Viable Submission (MVS), defined now:** rows 1+2+3-lite+5-lite — question latch +
typed answer scaffolding + panorama detection with lidar fusion into an object map + a
closest/near/on/between-only toolbox with LLM parsing + sweep-then-frontier exploration + the
watchdog. No captioner, no verification pass, no corridor geometry (corridors degrade to a
midpoint via-point), counts straight off the map. Estimated expected score ≈ 55–65% of the
full design's. **This ships end of week 3 regardless of anything.**

| Week | Dates | Deliverable | Gate / cut-line |
|---|---|---|---|
| 0 | Jul 10–13 | **Register (hard deadline Jul 15).** Ubuntu reinstall + Docker + sim + one training scene running; record rosbags (camera/lidar/terrain/odom) for Windows-side fixtures if reinstall is quick, else pull straight to the Ubuntu box. Repo scaffold: `vla_core` (pure Python) + adapter skeleton. | If reinstall stalls: develop `vla_core` on Windows against synthetic fixtures; reinstall is only truly blocking for week 3. |
| 1 | Jul 14–20 | `vla_core` offline: panorama splitter, GroundingDINO+SAM2 wrapper, lidar projection/fusion, instance map with 3D-IoU merge — all unit-tested on recorded frames. Traversability rasteriser + frontier extractor from terrain-map fixtures. | Cut first: SAM2 (fall back to detection-box lidar cropping). |
| 2 | Jul 21–27 | Toolbox (all predicates of §4.2) + LLM parse/resolve loop with the one-shot tool-use prompt; batch-evaluated **offline against all 75 training questions** using maps built from recorded exploration bags — this offline harness is the project's main accuracy instrument. IF compiler (goals → waypoint sequences, corridor midpoints, avoid-capsule A*). | Cut first: avoid-capsule A* (degrade to leg-wise straight routing — costs ≤3 training questions' penalty exposure). |
| 3 | Jul 28–Aug 3 | **Integration week.** ROS 2 adapter, FSM, watchdog; full closed-loop runs on ≥5 training scenes; Docker image built and pushed. **SUBMIT MVS by Aug 3.** | Non-negotiable gate. If integration eats the week, everything from week 2 not yet wired (verification pass, corridor A*) waits — MVS ships anyway. |
| 4 | Aug 4–10 | Score-driven hardening, in pts/day order: early-answer policy tuning on the clock; IF interleaved explore-execute; verification pass; per-type budget calibration across all 15 scenes ×5 questions; stuck-detection; numerical definite/indefinite logic. **Submit v2 ~Aug 9.** | Cut first: numerical logic (1 pt exposure). Then self-consistency voting. |
| 5 | Aug 11–15 | Stretch, only if v2 is stable: Qwen2.5-VL captioner (+ hard-statement hedge); box-extent refinement; self-consistency on IF; full 75-question timed regression; **final submission Aug 13–14**, one day of slack before AoE deadline. | Cut everything here without regret — week 5 is bonus territory by design. |

**Cut order across the whole project** (first cut → last): row 12/11 (pre-cut) → captioner →
box refinement → self-consistency → numerical anchor logic → avoid-A* → verification pass →
corridor geometry → **never cut**: watchdog, scaffolding, object map, toolbox core, frontier
exploration, MVS gate.

---

## 8. Citations & originality

Borrowed concepts, with sources; everything below is concept-level adoption with clean-room
implementation, and no prose is copied from any source.

- **SORT3D** — Nader Zantout, Haochen Zhang, Pujith Kachana, Jinkai Qiu, Ji Zhang, Wenshan Wang,
  *"SORT3D: Spatial Object-centric Reasoning Toolbox for Zero-shot 3D Grounding Using Large
  Language Models,"* 2025, arXiv:2504.18684, <https://arxiv.org/abs/2504.18684>. Borrowed: the
  perception→captions→LLM-filter→deterministic-spatial-toolbox→sequential-LLM division of labor;
  the "LLM must not do coordinate geometry" principle; the one-shot tool-use prompting pattern;
  the +11.6% captioning-ablation figure used in row 8's estimate. **The SORT3D GitHub repository
  (github.com/nzantout/SORT3D) has no license: no code is reused from it — concepts only,
  re-implemented from scratch.** Related organizer material: H. Zhang, *"Object-Centric Grounding
  for Deployable and Interactive Vision-Language Navigation Agents,"* CMU-RI-TR-25-86, 2025; and
  P. Kachana, *"Advancing 3D Semantic and Geometric Reasoning,"* CMU-RI-TR-25-18, 2025 (source of
  the wide-FOV visual-geometry warning behind §4.4).
- **Transcrib3D** — Jiading Fang, Xiangshan Tan, Shengjie Lin, Igor Vasiljevic, Vitor Guizilini,
  Hongyuan Mei, Rares Ambrus, Gregory Shakhnarovich, Matthew Walter, *"Transcrib3D: 3D Referring
  Expression Resolution through Large Language Models,"* IROS 2024,
  <https://openreview.net/forum?id=7j3sdUZMTF>. Borrowed: LLM-writes-the-filter /
  code-executes-it counting (§4.3); scene-as-text serialisation.
- **VLFM** — Naoki Yokoyama, Sehoon Ha, Dhruv Batra, Jiuguang Wang, Bernadette Bucher,
  *"VLFM: Vision-Language Frontier Maps for Zero-Shot Semantic Navigation,"* ICRA 2024,
  arXiv:2312.03275, <https://arxiv.org/abs/2312.03275>. Borrowed: semantic scoring of frontiers
  by question relevance (implemented here as a detector-affinity bonus rather than BLIP-2 value
  maps).
- **3D-Mem** — Yuncong Yang, Han Yang, Jiachen Zhou, Peihao Chen, Hongxin Zhang, Yilun Du,
  Chuang Gan, *"3D-Mem: 3D Scene Memory for Embodied Exploration and Reasoning,"* CVPR 2025,
  arXiv:2411.17735, <https://arxiv.org/abs/2411.17735>. Borrowed: the explore-until-answer-
  relevant-memory-suffices stopping principle (§3), simplified to a heuristic.
- **OpenEQA** — Arjun Majumdar et al., *"OpenEQA: Embodied Question Answering in the Era of
  Foundation Models,"* CVPR 2024, <https://open-eqa.github.io/>. Borrowed: the empirical finding
  that frontier VLMs count poorly, motivating map-based deterministic counting.
- **VLA-3D** — Haochen Zhang, Nader Zantout, Pujith Kachana, Zongyuan Wu, Ji Zhang, Wenshan
  Wang, *"VLA-3D: A Dataset for 3D Semantic Scene Understanding and Navigation,"* 2024,
  arXiv:2411.03540, <https://arxiv.org/abs/2411.03540>. Used as: the prior on question style
  (its relation vocabulary generated the challenge questions) and as offline eval fixtures;
  explicitly *not* used for fine-tuning (row 11 cut). Companion: IRef-VLA, arXiv:2503.17406
  (source of the imperfect-reference risk noted in §6.2).
- **2025 result data point** — CMU MRSD newsletter (<https://labs.ri.cmu.edu/mrsd-news/articles/>):
  team "CopyPasta" (Paruchuri, Gupta, Singh, Adhar) placed 3rd worldwide in the 2025 challenge
  using Gemini 2.5 Pro plus a ROS state machine. Used as evidence that a strong-API-model-plus-
  state-machine design is competitive, informing this proposal's hybrid (API reasoner, but with
  the deterministic toolbox the organizers' ablations say pure-LLM approaches lack).
- **GroundingDINO / SAM 2 / ByteTrack-style association / Qwen2.5-VL** are used as off-the-shelf
  published models under their respective licenses (Apache-2.0 for GroundingDINO and Qwen;
  SAM 2 under Meta's license), cited here as tooling: Liu et al., *"Grounding DINO,"* ECCV 2024,
  arXiv:2303.05499; Ravi et al., *"SAM 2,"* 2024, arXiv:2408.00714; Zhang et al., *"ByteTrack,"*
  ECCV 2022, arXiv:2110.06864; Bai et al., *"Qwen2.5-VL,"* 2025, arXiv:2502.13923.

**Original to this proposal:** the expected-points model and pts/day prioritisation (§1.2); the
per-type exploration budgets and the interleaved explore-execute policy for instruction-
following (§3); the explicit asymmetric early-answer rule (§4.0); the watchdog answer-floor and
"the floor is not zero" scaffolding analysis; the corridor approach/exit-point compilation and
whole-traversal avoid-capsule handling (§4.1); the MVS definition and the cut-order schedule
(§7). No published work found in the surveyed literature optimises for a time-budgeted,
partial-credit, relaunch-per-question scoring function — that optimisation layer is the
contribution of this design.

---

*End of Proposal C.*
