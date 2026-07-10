# Proposal B — Frontier-VLM Agentic System

**Date:** 10 Jul 2026
**Target:** CMU VLA Challenge 2026 (submission deadline 15 Aug 2026, AoE)
**Design stance:** Lean maximally on a frontier multimodal LLM API at runtime. The panorama plus
lightweight geometric context go into a multimodal agent loop that decides where to move and what
to answer. No custom-trained models, minimal bespoke perception machinery, and only a thin
deterministic assist where geometry genuinely cannot be delegated (waypoint feasibility off the
XYZI terrain map; back-projecting a VLM's image pick into a map-frame bounding box).

The rules explicitly permit online LLM/VLM APIs at test time (access tokens supplied at runtime).
The 2025 3rd-place team, CMU MRSD "CopyPasta" (Paruchuri, Gupta, Singh, Adhar — MRSD newsletter,
labs.ri.cmu.edu/mrsd-news/articles/), bet exactly this way: Gemini 2.5 Pro for language and
embodied spatial reasoning coordinated by a ROS state machine, and it earned a podium finish and a
cash prize. This proposal is the 2026-grade version of that bet: a stronger model generation, a
tighter agent loop, snapshot memory, and deterministic guardrails at precisely the points where
2025-era evidence says raw VLMs fail (counting, metric geometry).

---

## 1. System overview and agent loop

### 1.1 Contract recap (what we may touch)

- **Inputs (only these six):** `/camera/image` (1920×640 equirect panorama, 360°H × 120°V, 10 Hz),
  `/registered_scan` (PointCloud2, map frame, 5 Hz), `/sensor_scan` (PointCloud2, sensor frame,
  5 Hz), `/terrain_map` (XYZI, 5 m radius, 5 Hz), `/terrain_map_ext` (XYZI, 20 m radius, 5 Hz),
  `/state_estimation` (Odometry, 100–200 Hz), plus `/challenge_question` (String, republished
  at 1 Hz — latch and dedupe).
- **Outputs (only these three):** `/way_point_with_heading` (Pose2D, θ=0 this year — the *sole*
  actuation channel; waypointConverter → localPlanner → pathFollower does collision avoidance for
  us), `/selected_object_marker` (Marker CUBE, map frame, scored by ground-truth box overlap),
  `/numerical_response` (Int32, exact match).
- **Run shape:** system relaunches per question; no cross-question memory; 10-minute hard clock
  from startup with an early-finish bonus and an overtime penalty. Built-in TARE/FAR planners are
  NOT running and their triggers are not legal topics — all exploration is on us, one Pose2D at a
  time. Waypoints outside traversable area get snapped in by the stack; keep hops short (the stack
  itself warns that far waypoints can wedge the vehicle in dead ends).

### 1.2 Architecture

Two processes inside the `ai_module` container:

1. **ROS adapter (thin, dumb, ~300 lines):** subscribes to the six topics, maintains the latest
   panorama + terrain clouds + pose in shared state, latches the question, publishes the three
   answer topics, and tracks "waypoint reached" (distance < 0.9 m or 20 s stall timeout). It never
   makes decisions.
2. **Agent core (pure Python, ROS-free, testable on Windows):** the VLM agent loop plus the thin
   deterministic assists. Talks to the adapter through a narrow `RobotIO` interface (get_snapshot,
   get_terrain, get_pose, send_waypoint, publish_answer) so the identical core runs against
   recorded data on a laptop.

```
                    /challenge_question (1 Hz, latched once)
                                  |
                                  v
 +--------------------------- ROS ADAPTER (rclpy) ----------------------------+
 |  /camera/image  /registered_scan  /terrain_map(_ext)  /state_estimation    |
 |        |                |                 |                  |             |
 |        v                v                 v                  v             |
 |   latest pano     depth-by-azimuth   traversability     pose (map)         |
 |                     lookup table      grid (XYZI,                          |
 |                                       I<0.15m=floor)                       |
 +---------------------------------|-------------------------|---------------+
                                   |  RobotIO interface      ^
                                   v                         |  Pose2D waypoints,
 +------------------------- AGENT CORE (pure Python) --------|---------------+
 |                                                           |               |
 |  [PLAN once] question -> VLM -> typed task plan           |               |
 |     (ordered sub-goals, corridor/avoid constraints,       |               |
 |      count spec, or target-object spec)                   |               |
 |                                                           |               |
 |  [LOOP, ~15-35 steps]                                     |               |
 |   1. Snapshot: pano (annotated) + pose + step budget      |               |
 |   2. Deterministic assist proposes K feasible waypoints   |               |
 |      (frontier + object-approach candidates, drawn ON     |               |
 |      the panorama as numbered markers)                    |               |
 |   3. VLM agent call: {system prompt, task plan, snapshot  |               |
 |      memory digest, current annotated pano, candidates}   |               |
 |      -> one action:                                       |               |
 |         MOVE(k)         -> feasibility-check -> waypoint -+               |
 |         LOOK(defer)     -> wait 1 s, re-snapshot          |               |
 |         MARK(bbox2d)    -> lidar back-projection -> Marker box            |
 |         COUNT_TALLY(..) -> update instance ledger                         |
 |         ANSWER(int)     -> /numerical_response                            |
 |         SUBGOAL_DONE    -> advance task plan                              |
 |         DONE            -> stop (early-finish bonus)                      |
 |   4. Snapshot memory: store keyframe (pose, thumbnail,                    |
 |      VLM's own one-line scene note, object sightings)                     |
 |                                                                           |
 |  [WATCHDOG] clock-aware forcing function + offline fallback               |
 +---------------------------------------------------------------------------+
        |                          |                         |
        v                          v                         v
 /way_point_with_heading   /selected_object_marker   /numerical_response
```

The intelligence lives in one place: the VLM call in step 3. Everything else is bookkeeping,
rendering, or a feasibility filter.

### 1.3 The two deterministic assists (and why they're unavoidable)

Honesty clause: two things cannot be delegated to the VLM because they are metric operations on
data the VLM never sees at full fidelity.

- **Waypoint feasibility / candidate generation.** The `/terrain_map(_ext)` XYZI clouds encode
  per-point obstacle height in intensity (floor ≈ 0, obstacle above ~0.1–0.2 m — this is the
  exact threshold the stack's own waypointConverter uses). We rasterize the ext map into a 0.2 m
  2-D grid each step and (a) reject/snap any VLM-chosen motion into free space, (b) generate the
  K candidate waypoints the VLM chooses among: frontier cells (free cells adjacent to
  never-observed cells), doorway/gap midpoints, and approach points ~1 m short of any
  previously-sighted object relevant to the task. ~200 lines of numpy. The VLM still *chooses*;
  code only guarantees the choice is physically executable.
- **Marker back-projection.** The Marker is scored by overlap with a ground-truth 3-D box —
  center *and extents* matter. A VLM cannot emit metric extents from a picture. So the VLM's
  output for object reference is a 2-D box on the equirectangular panorama; code converts pixel
  columns/rows to azimuth/elevation rays (equirectangular mapping is a linear pixel→angle map, no
  calibration model needed), gates `/registered_scan` points inside that angular frustum, clusters
  by range (simple 1-D range histogram + DBSCAN-lite), and fits an axis-aligned box to the
  dominant cluster. The VLM picks *which* object; lidar supplies *where and how big*.

Everything else — language parsing, sub-goal sequencing, deciding what has been seen, deciding
where to look next, counting judgment, when to stop — is the VLM's job.

### 1.4 Question-topic handling

The evaluation node republishes the single question at 1 Hz for the whole run. The adapter latches
the first non-empty string, and ignores identical re-publishes (the dummy node's latch-and-clear
pattern, generalized). Waypoint θ is always published as 0 (README: heading neglected this year).
The marker is published on fully-qualified `/selected_object_marker` (the dummy omits the leading
slash; we don't copy that quirk). Answers are re-published at 1 Hz once decided, so a late-joining
evaluator subscriber cannot miss a single volatile message.

---

## 2. Exploration policy — how a VLM explores with planners off

TARE/FAR are not running and cannot be triggered legally, so exploration is the agent loop itself.
The policy is **VLM-scored movement proposals over the terrain map**, with **snapshot memory** for
persistence — an adaptation of two published patterns:

- **VLFM** (Yokoyama, Ha, Batra, Wang, Kim — "VLFM: Vision-Language Frontier Maps for Zero-Shot
  Semantic Navigation," ICRA 2024, arXiv:2312.03275, https://arxiv.org/abs/2312.03275) showed that
  scoring geometric frontiers by vision-language relevance to the goal text beats uninformed
  frontier exploration for zero-shot object navigation. We keep the idea but replace the BLIP-2
  cosine-similarity value map with the frontier VLM itself: candidates are drawn as numbered
  markers directly onto the panorama (each frontier direction is a pixel column — a free gift of
  the equirectangular geometry), and the model is asked "which numbered direction most plausibly
  leads toward ⟨the thing the task needs next⟩, given what you can see down each one?" This is
  strictly more informed than an embedding dot product: the model sees the actual hallway, the
  actual doorway glow, the actual furniture.
- **3D-Mem** (Yang et al. — "3D-Mem: 3D Scene Memory for Embodied Exploration and Reasoning,"
  CVPR 2025, arXiv:2411.17735, https://arxiv.org/abs/2411.17735) contributes the memory shape and
  the stopping rule: keep a compact set of *memory snapshots* (keyframes with poses covering
  observed regions) plus *frontier snapshots* (views of unexplored boundaries), and stop exploring
  when the memory is sufficient to answer. Our version: every executed MOVE stores
  {pose (x, y), 512×170 pano thumbnail, the VLM's own one-sentence note ("dining area: table with
  4 chairs, two windows on east wall"), and named object sightings with approximate map
  coordinates from the depth-by-azimuth lookup}. The running digest fed to each step is text +
  the last 2 full panoramas + up to 4 recalled thumbnails, so context stays bounded (§5.3).

Concrete loop behavior:

- **Step 0 (orientation spin, no API):** publish 2–3 short waypoints in a tight triangle around
  the start pose while the first API call is in flight, guaranteeing a clean 360° of registered
  scan and a settled terrain map. Costs ~15 s, pays for itself in map quality.
- **Candidate set each step:** up to K=8 — top frontier gaps by information gain (unknown-area
  behind the gap, computed on the terrain grid), plus approach points for task-relevant objects
  already sighted, plus "hold position and look" as candidate 0.
- **Directed vs. coverage mode:** the task plan (§3) tells the loop what it is hunting for. While
  any required anchor object is un-sighted, frontier candidates dominate (coverage). Once all
  anchors for the current sub-goal are sighted, approach candidates dominate (servoing). This is
  VLFM's goal-conditioning without any trained value model.
- **Scene-scale prior:** 13 of 15 training scenes are single rooms; only `home_building_*` are
  multi-room. The prompt tells the model this ("most scenes are one room — prefer confirming from
  the current room before committing to a doorway"), which suppresses the classic VLM wanderlust
  failure and protects the early-finish bonus.
- **Stopping:** 3D-Mem-style sufficiency check is folded into every step — the model must output
  a `confidence_to_answer` field; the loop shifts from exploration to answering when it reports
  the memory suffices, or when the watchdog forces it (§4).

---

## 3. Per-question-type strategy (weighted by points)

Verified point split across the training distribution: instruction-following **70.6%** of points
(30 questions × 6), object reference **23.5%** (30 × 2), numerical **5.9%** (15 × 1). Engineering
priority follows the points, in that order.

### 3.1 Instruction-following (6 pts each, 70.6% — the whole ballgame)

Every training instruction decomposes into an **ordered list of 2–4 sub-goals**: reach/near an
anchor, traverse a `path between X and Y` / `path near X` corridor, `pass by` a waypoint, or
`avoid` a forbidden corridor, terminated by a `stop at`. Ordering and avoidance are explicitly
scored. All relations are **allocentric** — zero egocentric phrasings (left/right/facing) exist in
the training set, so no viewer-frame machinery is built.

- **Plan once, up front.** The first API call converts the question into a typed JSON plan:
  `[{goal, anchor_spec, disambiguator}, {corridor, between:[A,B]}, ..., {stop_at, ...}]` plus a
  global `avoid` list. This call is text-only, cheap, and validated by a JSON-schema check with
  one retry. The plan is fixed; the loop executes it sub-goal by sub-goal.
- **Anchor grounding is visual, in-loop.** For each sub-goal the agent hunts the anchor via §2,
  confirms it visually ("is the object at marker 3 the potted plant *furthest from the hookah*?
  You have seen 3 potted plants; their positions and the hookah's position are: …"), and then
  moves to the approach candidate. Superlatives (`closest/farthest`, 36% of instruction anchors)
  are resolved by handing the VLM the *sighting ledger with map coordinates* — the model compares
  distances we compute and print, it never estimates metric distance from pixels. This is the one
  lesson we import from the organizers' own thesis line (H. Zhang, CMU-RI-TR-25-86): raw-coordinate
  spatial math inside an LLM is unreliable; arithmetic is printed for it, judgment is left to it.
- **`path between A and B` corridors (10/30 questions):** once both anchors are sighted with map
  coordinates, code inserts the midpoint of segment A–B as an intermediate waypoint, preceded by a
  staging waypoint on the near side, so the executed trajectory demonstrably threads the gap. The
  VLM confirms the gap is traversable from the panorama before commit.
- **`avoid the path between A and B` (3/30, penalty-scored):** the deterministic grid marks a
  capsule (segment A–B dilated by ~0.8 m) as forbidden; candidate waypoints inside it are never
  offered, and straight-line legs crossing it are re-routed via a trivial A* on the terrain grid.
  This is a guardrail, not intelligence — the penalty for one careless leg is too high to leave
  to sampling noise.
- **Terminal stop:** on the final `stop at`, drive to the approach point, then stop publishing
  waypoints and (if confident overall) signal DONE for the early bonus.
- **Partial credit strategy:** sub-goals are executed even at moderate confidence — visiting the
  right two of three anchors in order still scores. The watchdog (§4) reserves enough clock to
  reach the terminal anchor even if mid-plan grounding stays uncertain.

### 3.2 Object reference (2 pts each, 23.5%)

Same hunting machinery, different terminal action. When the agent believes it faces the target,
it emits `MARK` with a tight 2-D box on the panorama; back-projection (§1.3) produces the map-frame
CUBE Marker. Two accuracy boosters, both cheap:

- **Get close before marking.** A 2° azimuth error at 6 m is 21 cm of center error; at 1.5 m it is
  5 cm. The loop always closes to <2.5 m before MARK.
- **Extent sanity check.** The prompt carries a table of typical object dimensions ("pillow ≈
  0.5×0.4×0.15 m") and the model is shown the lidar-fitted extents for a yes/no plausibility pass;
  implausible fits trigger one re-look from a different bearing. The center of the fitted box is
  double-checked against the depth-by-azimuth table to catch see-through-window ghost clusters.
- **57% of these questions are multi-constraint** ("the bowl on the table closest to the folding
  screen") — handled identically to instruction anchors: candidate set + printed coordinates +
  VLM chain-of-thought verification that *every* clause is satisfied (the compositional-slip
  failure mode documented for SORT3D — see §8 — is checked for explicitly with a final "verify
  each clause" pass).

### 3.3 Numerical (1 pt each, 5.9%) — the documented VLM weakness, addressed head-on

OpenEQA (Majumdar et al., "OpenEQA: Embodied Question Answering in the Era of Foundation Models,"
CVPR 2024, https://open-eqa.github.io/) documents that even frontier multimodal models lag humans
badly on counting; naive "count the pillows in this image" is a known loser. We do not pretend the
2026 model generation has fixed this. Mitigations, in order of leverage:

1. **Count from multiple viewpoints, tally in a ledger, never from one frame.** Each COUNT_TALLY
   action makes the VLM enumerate instances *individually with panorama pixel positions* ("pillow
   1 at (612, 300), pillow 2 at (700, 310)…"). Code converts each to an approximate map coordinate
   (azimuth + lidar range) and merges into an instance ledger with a 0.4 m merge radius. Multiple
   viewpoints add and confirm entries; the final Int32 is **the ledger's cardinality, not a number
   the VLM utters**. This moves de-duplication — the actual hard part — from the model's attention
   to a distance check.
2. **Anchor-scoped counting.** All 15 training numericals count *on/above/near an anchor* ("on the
   sofa under the pictures"). The plan resolves the anchor first (reusing §3.2 machinery), and only
   sightings whose map coordinates fall inside the anchor's dilated footprint (or above it, for
   on/above relations — lidar gives height) enter the ledger. Definite vs. indefinite article
   matters ("on **the** sofa" = one salient anchor; "on **a** sofa" = union over all sofas) — the
   plan JSON carries an `anchor_scope: one|all` flag.
3. **Self-consistency at trivial cost.** The final tally prompt is sampled 3× (temperature 0.7,
   majority vote) — worth ~15 s on a 1-point question only because it happens once, at the end.
4. **Zero is a legal answer.** Exact-match scoring means a confident 0 beats a guessed 1 when the
   set is empty; the prompt says so explicitly.
5. **Budget honesty:** numerical is 1 point. If the anchor hasn't been found by T+7 min, answer
   the ledger's best guess (or the mode of 1–3 for the question's object class as a desperation
   prior) and take the early-exit. No 1-point question is allowed to consume overtime penalty.

---

## 4. Time and latency budget

**Clock:** 600 s per question, from system startup. Early finish = bonus; overtime = penalty.

**Per-step latency model** (frontier multimodal API, one 1920×640 panorama + ~3k text tokens in,
~400 tokens out, thinking mode off/low):

| Component | Typical | Worst |
|---|---|---|
| Snapshot render + candidate gen (local) | 0.3 s | 1 s |
| API round trip (agent step) | 4–8 s | 20 s (retry) |
| Robot travel per hop (2–4 m @ 0.875 m/s + planner) | 3–6 s | 15 s (stall) |

Travel and inference are **pipelined**: the next API call is fired the moment a waypoint is
published, using the panorama captured just before departure; the action is executed on arrival.
Effective step time ≈ max(travel, API) ≈ **8–12 s/step**.

**Step budget:** startup/bringup + first question latch ~20 s; orientation spin 15 s; planning
call 8 s; reserve 60 s for terminal answering (marker refinement or tally votes) and 30 s safety
margin → ~470 s of loop time ≈ **35–55 agent steps** available, of which a single-room scene
typically needs 10–20. We are not step-starved; we are *variance*-exposed (one 20 s retry costs
two steps).

**Watchdog schedule (hard, code-enforced, per question type):**

| Clock | Instruction-following | Object ref | Numerical |
|---|---|---|---|
| T+0:35 | plan fixed, loop running | same | same |
| T+5:00 | must be ≥ sub-goal 2; else skip forward | must have candidate sighting; else widen search | anchor found; else widen |
| T+7:00 | abandon un-grounded mid anchors, drive remaining plan geometrically | MARK best candidate now | answer ledger/prior now |
| T+8:30 | drive directly to best-guess terminal anchor, stop there | answered | answered |
| T+9:20 | freeze: final waypoint published, DONE | — | — |

**Early-answer policy:** the model's per-step `confidence_to_answer` triggers answering as soon as
it crosses threshold two steps running. Expected finish on single-room scenes: 3–5 min, banking
the bonus on most questions rather than optimizing the last point on the hardest one.

**Connectivity failure (a stated challenge risk — the eval site's WiFi/API access is not
guaranteed-perfect):** layered fallback, all local and dependency-free:

1. Every API call: 15 s timeout, 2 retries with jittered backoff, then failover to a *second
   provider* (two independent frontier APIs configured; both tokens supplied at runtime as the
   rules allow). Dual-provider failover is the single highest-value reliability line in this
   design.
2. If both providers are dark > 60 s: enter **offline degraded mode** — pure geometric frontier
   exploration (largest-gap heuristic on the terrain grid) to keep the robot productively mapping,
   so that when connectivity returns the snapshot memory is richer, not staler.
3. If connectivity never returns: emit the honest floor — numerical: 2 (distribution mode
   fallback); object ref: box on the largest unexplained lidar cluster near the question's noun
   if any sighting was ledgered pre-outage, else no marker; instruction: visit ledgered anchor
   coordinates in plan order geometrically. Degraded, not zero.

---

## 5. Concrete choices

### 5.1 Model

Primary: **a frontier multimodal API model of the Gemini 2.5 Pro / GPT-5 class** (final selection
by a July bake-off on the 15 training scenes' recorded data — the harness in §7 makes the model a
config string). Selection criteria, in order: (1) grounded 2-D pointing/boxing quality on
equirectangular crops, (2) p95 latency at ~4k-token multimodal prompts, (3) instruction adherence
to a strict JSON action schema, (4) cost. Secondary/failover: a different provider's frontier
model with the same action schema (the loop is provider-agnostic by construction). A small-fast
tier of the same family handles the per-step scene notes and tally enumeration where full
reasoning is wasted.

### 5.2 Panorama feeding

The native 1920×640 image is sent **whole, once per step**, downscaled to 1536×512 (equirect
mapping preserved; the model receives an explicit legend: "x-pixel ∝ azimuth: column 0 = rear,
768 = front, wraps; y-pixel ∝ elevation"). Rationale: agentic look-around decisions need the full
360° context, and modern long-context multimodal models handle a ~1.5k-wide image in one shot.
Tiles are used in exactly two places:

- **MARK refinement:** after the coarse pick, a 3× upscaled crop (~600×600) around the candidate
  is sent for the tight 2-D box — box tightness feeds directly into Marker overlap score.
- **COUNT_TALLY:** the anchor region crop at full native resolution, so small objects (cups,
  photos) aren't lost to downscaling.

Candidate waypoints, sub-goal state, and ledgered object positions are **drawn onto the panorama**
(numbered discs, colored per type) before sending — visual grounding of choices outperforms
coordinate lists for picking, and the numbers make the action space discrete and parse-proof.

### 5.3 Context management across steps

No ever-growing chat transcript. Each step is a **fresh, statelessly-assembled prompt**:

```
[system: role, contract, action schema, object-size table, scene-scale prior]   ~1.2k tok (cached)
[task plan JSON + per-sub-goal status]                                          ~0.3k
[memory digest: one line per keyframe + object ledger with map coords]          ~0.5k, capped
[last action + result ("moved to 4, reached; stall=no")]                        ~0.1k
[images: current annotated pano (+ up to 4 recalled thumbnails on request)]     ~1.5–3k
```

≈ 4–6k input tokens/step, with the system block served from provider-side prompt caching. The
memory digest is the 3D-Mem-style compression: the *model's own* one-line notes plus code-computed
coordinates, never raw history. A RECALL action lets the model pull any keyframe thumbnail back
into the next step's prompt when it wants to re-inspect something it noted earlier.

### 5.4 Cost per question

Typical instruction-following run: 1 plan call (~2k in/0.5k out) + ~18 loop steps × (5k in of
which 1.2k cached / 0.4k out) + 2 refinement crops + 3 tally votes ≈ **~100k input + ~9k output
tokens**. At mid-2026 frontier-API pricing (order of $1.25–2.50 per M input, $8–15 per M output),
that is **≈ $0.20–0.60 per question**, i.e. **under ~$10 for a full 15-question evaluation** and
roughly $50–150 for the entire development season's regression runs against recorded data (which
mostly use the small-fast tier at ~10× cheaper). API cost is negligible against the engineering
time it replaces; latency, not price, is the binding constraint.

---

## 6. Failure modes and mitigations

| # | Failure | Mitigation |
|---|---|---|
| 1 | API outage / throttling mid-run | Dual-provider failover, 15 s timeouts, offline degraded exploration, honest-floor answers (§4) |
| 2 | VLM stochasticity — same scene, different action | Temperature 0 for action selection; self-consistency voting only on terminal answers; watchdog makes any trajectory converge to *an* answer |
| 3 | Malformed action JSON | Strict schema validation, one re-ask with the error message, then deterministic default (best frontier) |
| 4 | Counting under/over-shoot | Ledger cardinality with metric de-dup, anchor scoping, multi-view confirmation, 3-vote tally (§3.3) |
| 5 | Marker extents wrong (VLM has no metric sense) | Lidar frustum back-projection supplies extents; typical-dimension sanity table; close-range re-look (§3.2) |
| 6 | Compositional slip (satisfies one clause, drops another) | Final verify-every-clause pass before MARK/ANSWER; printed coordinate table for superlatives |
| 7 | Wanderlust / premature doorway commits | Scene-scale prior in prompt; frontier info-gain ordering; watchdog milestones |
| 8 | Robot stall / waypoint wedged | 20 s stall detector → back-off waypoint 1 m toward last pose → replan; hops capped at 4 m |
| 9 | Avoid-corridor violation by a careless leg | Code-enforced forbidden capsule + A* re-route — never left to the model (§3.1) |
| 10 | Question latch race / republished question | Latch-first-nonempty + dedupe; answers republished at 1 Hz once decided |
| 11 | Anchor never found | Watchdog degradation ladder: widen search → geometric best-guess → partial-credit execution |
| 12 | Equirect distortion confuses the model near image poles | 120° VFOV limits pole area; MARK always re-confirmed on an undistorted-ish local crop |

### Weaknesses of this approach (honest)

- **Runtime API dependence is existential.** Every mitigation in §4 reduces but cannot eliminate
  it: if the eval network is down for the full 10 minutes, this design scores near-floor on that
  question while a fully-onboard pipeline (SORT3D-style) would be unimpaired. This is the bet's
  fundamental risk and it is not hedgeable from inside the bet.
- **Stochasticity means score variance.** SORT3D's authors report ±6%-level run-to-run variance
  for LLM-based grounding; an agent loop compounds sampling across 15–30 decisions. Temperature 0
  and watchdogs bound the tail, but two submissions of identical code will not score identically.
  (Mitigation at the meta level: the challenge allows multiple submissions, highest counts.)
- **Geometric imprecision on the Marker answer.** Overlap scoring rewards centimeter-true extents.
  Our extents come from a lidar frustum cluster fit — decent, but a purpose-built detector +
  instance segmentation + tracked 3-D fusion would beat it on tightness, especially for partially
  occluded or thin objects (wall lamps, photos) where lidar returns are sparse. Wall-mounted and
  small objects are this design's weakest scored output.
- **Latency eats steps.** 8–12 s per decision is fine for 1–2 room scenes; a large multi-room
  test scene with a far-flung anchor chain could genuinely run out of steps where an onboard
  frontier planner sampling at 5 Hz would not.
- **Documented VLM counting weakness** (OpenEQA) is mitigated, not solved — the ledger converts it
  into a pointing + de-dup problem, but enumerated pointing on cluttered anchors (a sofa heaped
  with pillows) still fails in ways a tracked-instance perception stack fails less.
- **Opacity.** When a run goes wrong, the "why" is a prompt/sampling artifact, not an inspectable
  module boundary. We buy debuggability back with full per-step logging (prompt, image, action,
  state) and offline replay, but root-causing remains harder than in a modular pipeline.
- **Prompt overfitting to the training distribution.** The scene-scale prior, allocentric-only
  assumption, and object-size table are tuned on 15 training scenes; the 3 held-out scenes could
  break any of them (e.g., introducing egocentric phrasing, which zero training questions use).
  Priors are phrased as soft preferences, never hard gates, for exactly this reason.

---

## 7. Implementation plan (today = 10 Jul → 15 Aug deadline)

Current constraint: development machine is Windows; no ROS 2 / Unity sim until an Ubuntu reinstall.
This stance is *unusually compatible* with that constraint: the system is ~90% prompt + loop + light
geometry, all pure Python, all testable on Windows against recorded data. The ROS adapter is a thin
shell written last.

**Pattern:** pure-Python `agent_core/` (zero ROS imports) + `ros_adapter/` (rclpy, ~300 lines)
implementing the same `RobotIO` interface as the offline `ReplayIO` harness.

| Dates | Milestone | Environment |
|---|---|---|
| Jul 10–12 | Register (deadline Jul 15!). Repo skeleton: `RobotIO` interface, terrain-grid rasterizer + candidate generator, equirect annotation renderer, unit tests on synthetic clouds | Windows |
| Jul 13–17 | Offline replay harness: feed recorded panoramas/terrain/pose (from bagfiles once available; until then, static scene captures + questions.json). Plan-call prompt + JSON schema; agent-step prompt v1; provider abstraction with dual-API failover | Windows |
| Jul 18–24 | Full loop against replayed data on all 15 training scenes' question set (75 questions). Model bake-off (primary vs. failover, latency/pointing/cost). Counting ledger + marker back-projection with recorded scans. Per-question-type watchdogs | Windows |
| Jul 25–28 | **Ubuntu reinstall.** Stand up Docker compose stack + Unity scenes; write + wire the ROS adapter; first closed-loop runs in sim | Ubuntu |
| Jul 29–Aug 5 | Closed-loop iteration on all 15 scenes; timing calibration (real travel + API latency); stall/edge handling; **first submission pushed** (multiple submissions allowed — highest score counts, so submit the moment it beats the dummy) | Ubuntu |
| Aug 6–12 | Hardening: connectivity-failure drills (yank network mid-run), variance measurement (3 seeds × 15 scenes), prompt regression suite, held-out-style self-test (hide 3 scenes during tuning) | Ubuntu |
| Aug 13–15 | Freeze, final Docker push, final submission with margin before AoE deadline | Ubuntu |

Risk buffer: the only Ubuntu-gated work is the adapter + closed-loop timing (~1.5 weeks allocated
for what is ~4 days of work). If the reinstall slips, everything through Jul 24 still proceeds,
and the adapter contract is already pinned by the upstream dummy node's topic table.

---

## 8. Citations & originality

Borrowed concepts, cited at point of use and summarized here:

- **VLFM — Vision-Language Frontier Maps.** N. Yokoyama, S. Ha, D. Batra, J. Wang, B. Kim,
  ICRA 2024, arXiv:2312.03275, https://arxiv.org/abs/2312.03275. Borrowed: semantic scoring of
  geometric frontiers conditioned on the language goal (§2). Our variant replaces the trained
  value map with direct VLM choice over annotated panorama candidates.
- **3D-Mem — 3D Scene Memory for Embodied Exploration and Reasoning.** Y. Yang et al., CVPR 2025,
  arXiv:2411.17735, https://arxiv.org/abs/2411.17735. Borrowed: snapshot-based scene memory
  (memory + frontier snapshots) and the answer-when-memory-sufficient stopping rule (§2, §5.3).
- **OpenEQA.** A. Majumdar et al., "OpenEQA: Embodied Question Answering in the Era of Foundation
  Models," CVPR 2024, https://open-eqa.github.io/. Borrowed: the documented finding that frontier
  multimodal models are weak at counting/spatial EQA — the direct motivation for the instance
  ledger in §3.3.
- **SORT3D — Spatial Object-centric Reasoning Toolbox.** N. Zantout, H. Zhang, P. Kachana,
  J. Qiu, J. Zhang, W. Wang (CMU RI), 2025, arXiv:2504.18684, https://arxiv.org/abs/2504.18684.
  Cited **in contrast**: SORT3D is the organizers' modular pipeline (open-vocab detection +
  tracked 3-D instances + captioning + deterministic relation toolbox). This proposal takes the
  opposite bet — a frontier VLM in the loop — but imports two of its evidence-backed lessons:
  never let a language model do raw-coordinate arithmetic (we print computed distances instead,
  §3.1), and expect compositional-constraint slips (we add a verify-every-clause pass, §3.2). Its
  GitHub repository carries **no license; no code from it is used or adapted** — concepts only,
  cited here.
- **CopyPasta (2025 3rd place).** CMU MRSD team — S. Paruchuri, I. Gupta, P. Singh, D. Adhar —
  per the MRSD program newsletter (https://labs.ri.cmu.edu/mrsd-news/articles/): Gemini 2.5 Pro
  for language/spatial reasoning + a ROS state machine, 3rd worldwide in the 2025 CMU-VLA
  Challenge. Borrowed: the existence proof that the frontier-VLM-plus-state-machine bet is
  competitive on this exact challenge (§0, §1). Their architecture details beyond the newsletter
  blurb are not public; everything past "big model + state machine" here is independent design.
- **Transcrib3D.** J. Fang et al., OpenReview: https://openreview.net/forum?id=7j3sdUZMTF.
  Borrowed in spirit: keeping the final numeric answer deterministic (our ledger cardinality)
  rather than a language-model utterance (§3.3).
- **Challenge platform facts** (topic contract, terrain-map XYZI semantics, planner availability,
  scoring weights, question distribution) are from the official CMU-VLA-Challenge 2026 repository
  and its base autonomy stack (github.com/Yuxin916/CMU-VLN-Challenge-2026;
  ai-meets-autonomy.com/cmu-vla-challenge), verified against the shipped source.

**Originality statement.** The agent-loop design — annotated-panorama discrete action space,
stateless per-step prompt assembly with a coded memory digest, the anchor-scoped counting ledger,
lidar-frustum marker back-projection, the per-question-type watchdog ladder, and the dual-provider
failover scheme — is original to this proposal. No prose in this document is copied from any
source; all third-party ideas are identified and cited above.

---

*End of Proposal B — 10 Jul 2026.*
