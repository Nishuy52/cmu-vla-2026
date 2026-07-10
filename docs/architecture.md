# System Architecture — v1.0 (MERGED)

*10 Jul 2026. Synthesised from three independently developed proposals and a structured cross-critique
(full record: `docs/proposals/proposal_{A,B,C}.md`, `critique_{A,B,C}.md`). Supersedes draft v0.2.
Every externally sourced concept is cited inline; see §9 for the consolidated attribution statement.*

---

## 0. The bet, in one paragraph

A deterministic geometric backbone (tracked 3D instance map from lidar + open-vocabulary 2D detection;
all counting, distance, and corridor math in verifiable code) with **bounded, checkpointed language
reasoning**: an LLM parses the question into a typed plan, is re-consulted only at defined recovery
and verification checkpoints — never inside the actuation loop — and a watchdog guarantees every
question emits a legal answer before the clock. Scheduling follows an expected-points ledger with a
minimum-viable-submission gate. This is the deterministic-first frame of Proposal A, disciplined by
Proposal C's scoring math, with Proposal B's re-observation capability admitted in bounded form at
exactly the two places the debate showed it pays: recovery from detector misses and pre-answer
verification.

## 1. Debate adjudication — the contested points and their resolutions

| # | Contested point | Resolution | Rationale |
|---|---|---|---|
| 1 | LLM placement: once-only parse (A) vs per-step agent (B) vs few tool-calls (C) | **Checkpointed reasoning.** Parse at T0; re-consult only on defined events: grounding failure, pre-answer per-clause verification, and one anchor-confirmation per IF sub-goal arrival. Never for movement selection. | B's attack on A landed (one valid-but-wrong parse silently loses a 6-pt question; schema checks form, not meaning) — but A's and C's attack on B also landed (15–35 sequential API calls put network variance inside the 540 s scored window). Event-triggered checkpoints give mid-run recovery at ~3–6 calls/question, not 15–35. |
| 2 | Exploration intelligence | Deterministic frontier planner (occupancy from terrain XYZI, A*, breadcrumb waypoints ≤2.5 m) + question-noun detector-affinity frontier bias. One optional VLM frontier-selection call (numbered candidates drawn on the panorama, per B) permitted only when scene proves multi-room. | Movement must sample at the 5 Hz terrain rate with zero network (critique_A B1). A's own concession: its `category_bias` is weaker than semantic steering (critique_A §5); C's detector-affinity term is the deterministic-compatible fix. B's numbered-marker panorama technique is adopted for the *rare bounded* call because a discrete action space is parse-proof (critique_C §3). |
| 3 | Corridor/avoid geometry: costmap invariant (A) vs early cut (C) | **In the MVS, non-negotiable.** Avoid-capsules stamped hard into the per-question costmap; corridor gates as mandatory via-segments with a threading check (trajectory must cross the gate segment, not merely touch a midpoint). | Both A and B independently attacked C's cut-order here — the penalty-scored primitives of the 70.6% question type cannot be deferred. C's author did not defend the cut in its non-negotiables. Consensus. |
| 4 | A's capsule-shrink recovery when avoid blocks all routes | **Rejected** (self-defeating — reintroduces the violation it prevents, critique_C A3). Replacement: capsule stays hard; accept longer routes; if provably unreachable, stop at the nearest legal point to the goal and answer from there. A deliberate least-bad choice, never a silent geometry edit. | C's attack was correct and A conceded no defense. |
| 5 | Marker extents | Lidar-only geometry: multi-frame tracked instance clouds, per-axis 2nd/98th-percentile trimmed AABB (A), plus B's object-dimension sanity table (reject a 2 m "pillow"). Camera never contributes metric extents. | Unanimous post-debate: B conceded its single-frustum fit is second-best on the IoU-scored type (23.5% of points); Kachana thesis warns against wide-FOV visual geometry (organizer_playbook A.2). |
| 6 | Detector-miss recovery | When a question-critical noun has zero detections after coverage: one bounded VLM pass over stored keyframe tiles ("do you see an X? which tile?"), then targeted re-navigation. | B's strongest structural point: A "has no fallback that *sees*" (critique_B A1). This admits it at one checkpoint without per-step exposure. 114-noun open vocabulary makes detector misses a when, not an if. |
| 7 | API posture | Dual-provider failover → local quantised VLM (Qwen2.5-VL-7B-class) baked into the Docker image → regex-tier parse. The full answer path must produce a legal (degraded) answer with the network dark. | B's API dependence was judged existential by both rivals and conceded by B. A's outage-robustness is the keep; dual-provider failover is B's cheap, stolen-by-everyone improvement. |
| 8 | Early answer / time budget | C's asymmetric per-type policy + A's concrete gate: answer numericals aggressively once counts are stable (winner-margin ≥25% AND ≥3 observations per instance); never bank an early-finish bonus on an IF question with an ungrounded sub-goal. Watchdog floor at T−30 s: always a marker, always an integer, always drive at the best anchor. "Silence is the only unforgivable failure." | C's floor framing was every critic's keep; A's min-observation condition fixes C's under-specified margin rule (conceded by C). |
| 9 | Prioritisation instrument | C's expected-points ledger governs the build order and cut lines, with error-bars discipline: rows whose E[Δ] is a guess are marked as such and cut decisions near a boundary get re-estimated after sim results. | B's attack (unfalsifiable self-estimates driving irreversible cuts) tempers but does not replace the ledger — an explicit, revisable ledger still beats implicit taste (critique_C NN2). |
| 10 | IF execution | C's interleaved explore-execute: drive toward sub-goal 1 while scanning for later anchors en route; 360° sensing makes travel double as exploration. | Uncontested in debate; tightest clock use on 70.6% of points. |

## 2. Pipeline

```
                     ┌────────────────────────────────────────────────────────────┐
 /challenge_question │ T0 PARSE (checkpoint 1): LLM → typed plan JSON              │
 (String, 1 Hz)  ───►│   dual-API → local-VLM → regex ladder; schema-validated     │
                     │   {type, targets[], anchors[], relations[], corridors[],    │
                     │    avoids[], ordering}                                      │
                     └──────────────┬─────────────────────────────────────────────┘
                                    ▼
 /camera/image ───► PERCEPTION (continuous, deterministic)          EXPLORATION/EXECUTION
 (1920×640,10Hz)     4 gnomonic tiles → open-vocab detector          (deterministic, 5 Hz)
                     (question-noun-injected prompts) + light        occupancy grid from
 /registered_scan ─► segmentation → cross-frame instance             terrain XYZI (<0.15 m
 (PC2, 5Hz)          tracking → lidar-fused 3D instance map          = free) → frontier
                     {id, class, caption?, centroid, trimmed         scoring (geometry +
 /terrain_map(_ext)─►AABB, n_obs, per-axis percentiles}              noun-affinity bias) →
 (XYZI, 5Hz)                        │                                A* on per-question
                                    ▼                                costmap (avoid-capsules
 /state_estimation ─► SCENE GRAPH + SPATIAL TOOLBOX                  = hard obstacles;
 (Odom)              deterministic predicates: on/near/closest/      corridor gates = via-
                     between/above/…; scale-adaptive thresholds      segments) → breadcrumb
                     (near = max(1.2 m, 0.6×footprint diag))         waypoints ≤2.5 m
                                    │                                        │
                                    ▼                                        ▼
                     ANSWER HEADS (deterministic core, checkpointed LLM)   /way_point_with_heading
                     • numerical: set-cardinality over tracked instances     (Pose2D, θ=0)
                       via LLM-built filter executed as code
                     • object-ref: toolbox ranking → per-clause verification
                       (checkpoint) → trimmed AABB + dimension sanity table
                       → /selected_object_marker (Marker, map frame)
                     • instruction-following: ordered sub-goals, interleaved
                       explore-execute, anchor confirmation on arrival
                       (checkpoint), corridor threading check
                     WATCHDOG: T−30 s floor answer on every type; all LLM
                     calls timeout-bounded; dark-network degradation ladder
```

Split implementation: pure-Python `core/` behind a `RobotIO` dataclass interface (Windows-developable,
unit-testable, replay-harness-driven) + a thin rclpy adapter (~300 lines) validated after the Ubuntu
reinstall. Actuation only via `/way_point_with_heading`; only the six allowed input topics; relaunch
per question (no cross-question state).

## 3. Checkpoint budget (the LLM/VLM call ledger per question)

| Checkpoint | Trigger | Calls | Fallback if dark |
|---|---|---|---|
| 1. Parse | question received | 1 (+1 retry) | local VLM → regex tier |
| 2. Detector-miss recovery | question noun absent after coverage threshold | 0–1 | skip; fallback ladder relaxes filters |
| 3. Anchor confirmation (IF only) | arrival at each sub-goal | 0–3 | trust the map |
| 4. Pre-answer verification | before committing OR/IF answer | 1 | deterministic clause re-check only |
| 5. Frontier selection (rare) | scene proves multi-room AND clock ahead of budget | 0–1 | geometric frontier score |
| Optional: self-consistency ×3 | IF final commit, if ≥90 s spare | 0–2 | single sample |

Worst case ≈ 9 calls/question, typical 3–5, hard timeout on each; every path terminates in the
watchdog floor. Estimated API cost well under $0.10/question at these volumes.

## 4. Per-type strategies (points-weighted)

- **Instruction-following (180/255 pts):** deepest machinery. Plan → ordered legs; corridor gates as
  mandatory via-segments with threading verification; avoid-capsules as costmap invariants (hard);
  interleaved explore-execute; anchor confirmation checkpoint on each arrival; early-finish only when
  every sub-goal is grounded with ≥3 observations. Tolerant matching for typos ("refridgerator").
- **Object-reference (60/255 pts):** toolbox candidate ranking (allocentric relations only, per the
  verified question distribution) → per-clause verification → trimmed-AABB Marker in `map` frame,
  dimension-sanity-gated; Marker center doubles as the navigation goal.
- **Numerical (15/255 pts):** capped effort. LLM builds the filter once; deterministic set-cardinality
  over tracked, NMS-deduplicated instances; answer aggressively at count stability; modal-integer
  floor at watchdog.

## 5. Time budget per question (600 s)

1. 0–60 s: orientation sweep (in-place rotation; panorama + lidar seed the map) — parse runs concurrently.
2. Exploration/execution window with per-type soft budgets (numerical ≈ 210 s, OR ≈ 240 s, IF ≈ 270 s
   before answer-path pressure), continuously trading against the early-finish bonus per the
   asymmetric policy (§1 row 8).
3. T−90 s: forced best-effort answer assembly begins if not already answering.
4. T−30 s: watchdog publishes the floor answer unconditionally.

## 6. Concrete stack

| Function | Choice | Notes |
|---|---|---|
| Detection | GroundingDINO-class open-vocab detector, question-noun prompt injection | ~4–6 GB VRAM |
| Segmentation (light) | MobileSAM/SAM2-class for mask-tightened crops | optional below cut-line |
| Tracking | ByteTrack-style association in NumPy over detections + lidar centroids | prevents double-count |
| Captions (stretch) | Qwen2.5-VL-7B on best crop per instance | +11% grounding evidence, cuttable |
| Local reasoning fallback | quantised Qwen2.5-VL-7B-class in the Docker image | dark-network floor |
| Cloud reasoning | two frontier multimodal APIs, failover | checkpoints only |
| Panorama handling | 4× gnomonic 90° tiles for detection; full strip only for the rare frontier-selection call | thin-object recall watched at tile seams |

Peak VRAM ≈ 10–14 GB — fits the RTX 4090 eval spec with margin.

## 7. Schedule (merged; every week ends submittable)

| Window | Deliverable |
|---|---|
| Jul 10–24 (Windows) | `core/` modules + replay harness against recorded bags; toolbox + planner unit tests; prompt/parse fixtures. USER: register by Jul 15. |
| ~Jul 25–Aug 2 (Ubuntu) | ROS adapter, sim integration, calibration on 15 training scenes. |
| **Aug 3 — MVS gate** | Minimum viable submission banked: parse ladder + instance map + toolbox + frontier planner + **corridor/avoid geometry** + watchdog floor. |
| Aug 4–13 | Ratchet: verification checkpoints, anchor confirmation, captioner, self-consistency, multi-room frontier call — in expected-points order, re-estimated against sim scores. |
| Aug 13–15 | Freeze, final submission. |

Cut order under schedule pressure (first→last): captioner → self-consistency → VLM frontier call →
anchor confirmation → detector-miss recovery. **Never cut:** watchdog floor, toolbox, corridor/avoid
geometry, parse fallback ladder.

## 8. Accepted residual risks

1. Distribution shift on the 3 held-out scenes (thresholds, budgets, and priors are fitted to the
   training distribution) — partially hedged by scale-adaptive thresholds, checkpoint recovery, and
   open-vocab detection; accepted knowingly.
2. Detector recall on fine-grained nouns remains the single largest failure surface; checkpoint 2 is
   mitigation, not cure.
3. Egocentric or implicature-laden phrasing absent from training but possible at test — the
   verification checkpoint is the only line of defense; monitored, not solved.

## 9. Attribution & originality

Concepts adopted from published work, cited here and at point of use: object-centric text scene
representation and deterministic spatial toolbox invoked by an LLM — SORT3D (Zantout, Zhang, Kachana,
Qiu, Zhang, Wang; [arXiv:2504.18684](https://arxiv.org/abs/2504.18684)); LLM-generated filters
executed as code — Transcrib3D ([OpenReview 7j3sdUZMTF](https://openreview.net/forum?id=7j3sdUZMTF));
semantic frontier scoring — VLFM (Yokoyama et al., [arXiv:2312.03275](https://arxiv.org/abs/2312.03275));
snapshot-memory / answer-when-sufficient stopping — 3D-Mem ([arXiv:2411.17735](https://arxiv.org/abs/2411.17735));
open-vocab 3D scene graphs — ConceptGraphs ([arXiv:2309.16650](https://arxiv.org/abs/2309.16650));
VLM counting weakness — OpenEQA (Majumdar et al., CVPR 2024, [site](https://open-eqa.github.io/));
dataset & question provenance — VLA-3D ([arXiv:2411.03540](https://arxiv.org/abs/2411.03540));
detector/tracker/captioner components — GroundingDINO, SAM/MobileSAM, ByteTrack, Qwen2.5-VL (cited in
the proposals). The 2025 third-place configuration is known only from a CMU MRSD newsletter item and
is cited in `organizer_playbook.md`.

**License note:** the SORT3D repository carries no license. No code is taken from it — all components
here are clean-room implementations of *cited concepts*, plus original work: the checkpointed-reasoning
call ledger, corridor threading verification, the asymmetric early-answer policy with min-observation
gates, the watchdog floor framing, and the expected-points scheduling instrument (developed in
`docs/proposals/`, this repo).
