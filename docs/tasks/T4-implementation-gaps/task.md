# Task: Top-5 holes in the implementation vs the research

## Intent
Compare this repo's implementation against the verified
findings of the parallel research stream (its io-contract,
challenge-rules, problem-brief, prior-art/*, vla-3d docs) and
surface the top discrepancies: where the code/docs are MISSING
something the research found, or CONTRADICT it. Deliverable is this ranked list so the
team can decide what to fix before the sim comes online.

## Context
- The implementation: full pure-Python core (642 tests), GT
  eval harness, FSM, 3 answer heads, ROS adapter, 20+ docs.
  Substantial and mostly aligned with the research - these are
  holes in an
  otherwise solid build, not a teardown.
- Method: 4 parallel comparison agents (I/O, scoring+qtypes,
  prior-art vs architecture, VLA-3D dataset), then main-thread
  verification of the two load-bearing "unconfirmed" claims.
- One agent-flagged hole was FALSIFIED on verification and
  dropped (see Notes): object-reference does publish both marker
  and waypoint (controller.py:350-354). Kept only what survived.
- Source-of-truth caveat: some research docs still carry
  UNVERIFIED flags (scoring internals, test-scene qtype mix);
  where the implementation is better-evidenced, that is noted.

## The top 5 holes (ranked by scoring impact)

### 1. "near" threshold contradicts the organizers' own
### question-generation formula - CONTRADICTION, highest impact
- Research finding: VLA-3D questions were generated with
  `near` = `euclidean(center_a, center_b) < 0.01 *
  region_volume` - room-volume-scaled. The dossier calls out by
  name that "a fixed-radius 'near' will disagree with how the
  source questions were generated, especially between small and
  large rooms."
  research: docs/prior-art/vla-3d.md:193-198, :415-419
- The code: `near_thresh = max(1.2, 0.6 * footprint_diag(b))` -
  object-size-scaled and room-BLIND. A toolbox comment claims
  these are "spec-mandated values"; sibling thresholds in the
  same struct are self-labeled "(invented)".
  impl: src/core/geometry/toolbox.py:40-48, :137-140, :201-208;
  design in docs/architecture.md:58, docs/calibration.md:35-36
- Why it matters: "near" feeds numerical counts, object-ref
  selection, AND instruction-following "near X" constraints - a
  systematic, scene-size-correlated loss across ALL three
  rubric categories, worst in the largest scenes. The comment
  claims spec-mandated; it contradicts the actual spec on file.

### 2. `target_index` == `object_id` id-space assumption is
### unverified - OMISSION / potentially critical
- Research finding: the referential-statements schema has a
  `target_index` field, but nothing asserts it shares the id
  space of the object CSV's `object_id` / scene-graph
  `object_id`.
  research: docs/prior-art/vla-3d.md:364-377
- The code: casts `target_index` straight to int and joins it
  against `instance_id` (built from `object_id`) with no
  validation.
  impl: src/core/groundtruth/scoring.py:417, :524
- Why it matters: if the language generator indexes differently
  (per-region, filtered, etc.), EVERY object-reference IoU
  score silently grounds against the wrong object - and the
  numerical GT is circular on top of it, so nothing catches it.
  Neither side has verified this; it is a live
  gap, not a settled fact.

### 3. Instruction-following scorer optimizes trajectory
### similarity, not the constraint/order/avoidance rubric -
### CONTRADICTION (code vs its own doc + the research)
- Research finding: IF is scored 0-6 with penalties for wrong
  constraint order, unmet constraints, and passing through
  forbidden areas. IF = ~70% of all points - it is where
  ranking is won.
  research: docs/challenge-rules.md:65-66; weight in prior-art.md
- The code: `score_instruction_following` computes only
  discrete Frechet distance + 1.0 m path-coverage vs ONE
  recorded GT trajectory. No order check, no unmet-constraint
  check, no forbidden-area penalty. This repo's own
  question_analysis.md:56 describes the official rubric the
  same way - so the code disagrees with its own doc.
  impl: src/core/groundtruth/scoring.py:26-31, :807-874
- Why it matters: if this proxy steers engineering priority, it
  optimizes "look like the one reference path" instead of
  "satisfy the constraints in order" - the wrong objective on
  the highest-value question type. (The scorer is honestly
  self-labeled a proxy, not the official scorer - but the
  objective mismatch is the risk.)

### 4. Prior-art docs stale/false on 2025 1st & 2nd place, and
### the "forbidden GT channel" insight never reached the design
### docs - CONTRADICTION + OMISSION
- Research finding: full 2025 leaderboard resolved - 1st NROS
  (44.26), 2nd ReasonX (34.58), 3rd CopyPasta (30.98), 4th
  URL-KAIST (22.80), with per-team dossiers. Critically: 2 of
  the top 3 (ReasonX, CopyPasta) leaned on the 2025 sim's
  ground-truth `/object_markers` topic - which the 2026 I/O
  list does NOT include - so their reported approaches
  overstate how much ports forward.
  research: docs/prior-art.md:367-408; prior-art/2025-1st-nros.md,
  2025-2nd-reasonx.md:279-295, 2025-3rd-copypasta.md:302-313
- The docs here: `docs/prior_art.md:84-86` still says 1st/2nd
  "not found... no arXiv reports yet"; `architecture.md:165-166`
  still calls CopyPasta "known only from a newsletter item" -
  both false as of the same day. The correction DOES exist in
  this repo (organizer_playbook.md:115-137) but was never folded
  into architecture.md / master_plan.md, and no schedule item
  reacts to it.
- Why it matters: anyone reading the design docs believes the
  winners are unknown and that the open-vocab perception stack
  faces the same bar as teams that got a free GT-semantics pass
  - miscalibrating effort on the single largest failure
  surface.

### 5. GT numerical harness encodes wrong VLA-3D relation
### semantics for the highest-frequency statement types -
### OMISSION cluster
- The numerical "ground truth" is self-consistency (the
  resolver re-run over its own scene index), so any semantic
  error in the resolver is invisible - which makes these matter
  more, not less.
  impl: src/core/groundtruth/scoring.py:7-16, :298-346
- (a) `between` / `hanging_on` relations: the research doc says
  these are
  `[id1, id2]` PAIR-shaped scene-graph entries, distinct from
  the flat `object_id -> [ids]` shape. `_scene_graph_count`
  flattens every relation uniformly and does `str(a)` per
  element - a pair-list element stringifies to garbage that
  matches no object, silently zeroing those counts.
  research: docs/prior-art/vla-3d.md:359-363;
  impl: src/core/groundtruth/scoring.py:265-295 (confirm vs real
  scene-graph JSON before treating as certain)
- (b) closest/farthest: the research doc says keep top-3 and
  SUPPRESS the
  statement when consecutive-rank distances are within
  `ordered_thres=0.2` of object size. The code's note treats it
  as
  plain single nearest/farthest, no suppression gate. These are
  the two largest statement-type counts in the dataset.
  our: docs/prior-art/vla-3d.md:212-226, :420-428
- (c) big/small: the research doc uses relative
  `largest_face_area` with a >1.2x gap to the next same-class
  object; `_size_token`
  uses fixed absolute-volume cutoffs (0.5 / 0.02 m^3).
  research: docs/prior-art/vla-3d.md:300-310;
  impl: src/core/groundtruth/loader.py:51-52, :165-172

## Verified clean: no disallowed GT semantics at test time

Checked directly (this is the exact rule that sank 2 of the
2025 top-3), and the build is clean. Recording the full trace
because it is load-bearing for hole #4's framing and is the kind
of thing a reviewer will (rightly) want to re-confirm on Ubuntu.

The distinction that matters:
- DEV-TIME ground truth (offline scoring against the VLA-3D
  dataset in a battery runner) is ALLOWED and expected - it is
  how you grade yourself offline with no sim.
- TEST-TIME ground truth (the running module reading GT scene
  semantics, or subscribing a GT topic like the 2025
  `/object_markers`, to produce its answer) is DISALLOWED - the
  2026 I/O list omits every GT channel. This is precisely what
  ReasonX (2nd) and CopyPasta (3rd) leaned on in 2025 (see hole
  #4), and it does not port forward.

Evidence the GT code is quarantined to the dev harness:
- Import graph: `core.groundtruth` (loader + scoring) is
  imported ONLY by the dev-time runners and the test suite -
  `src/core/runner/battery.py`, `src/core/runner/gt_battery.py`,
  and `src/tests/**`. Nothing else.
- The runtime path has ZERO references to `groundtruth`,
  `object_markers`, or any GT topic: grep across
  `src/ros_adapter/`, `src/core/fsm/`, `src/core/heads/`, and
  `src/core/perception/` returns nothing.

Evidence the runtime resolves against perception, not GT:
- The answer heads resolve against a source-agnostic
  `SceneIndex` abstraction (`src/core/heads/factory.py:110-144`,
  `numerical.py:57-74`, `object_ref.py:76-92`) - they do not
  know or care whether the index came from perception or GT, so
  the runtime can (and does) feed the non-GT one.
- The ROS node builds its index from the PERCEPTION module, not
  the loader: `src/ros_adapter/adapter_node.py:77` imports
  `BasicSceneIndex` from `core.perception.scene_index`, and
  `:190` sets `self._scene_index = BasicSceneIndex([])`, fed to
  `build_callables(self._scene_index)` at `:255`.
- GT-derived indexes enter ONLY through the dev-side runner
  chain: `runner/single.py:_derive_scene_index` (`:98-105`)
  picks explicit > `io.scene` > empty, and `gt_battery.py` is
  what passes a GT-backed `io.scene`. The ROS adapter never
  calls that path.

The honest caveat (why this is "clean by design", not yet
"proven end-to-end"):
- The runtime is clean partly because the live-perception
  replacement is NOT built yet. `adapter_node.py:186-190`
  wires an EMPTY `BasicSceneIndex([])` with an explicit Phase-2
  TODO: "The heads resolve against the live SceneIndex. Phase 2
  wires the real perception map here... swap `BasicSceneIndex([])`
  for the live perception scene index once core/perception is
  fused into this node." Until then the FSM floor still emits a
  legal (but uninformed) answer.
- So the separation is confirmed by architecture and by the
  import graph, but the perception path that REPLACES the empty
  index is unimplemented - there is no GT leakage wired in, and
  none can sneak in without importing `core.groundtruth` into a
  runtime module (currently nothing does). Re-confirm on Ubuntu
  when core/perception is fused in: the invariant to hold is
  "no `core.groundtruth` import anywhere under `ros_adapter/`,
  `fsm/`, `heads/`, `perception/`."

Bottom line: architecturally the OPPOSITE of the 2025 2nd/3rd
approach - GT is a grading tool, not an answer source. This
strengthens hole #4: the design correctly forbids the channel;
the gap there is only that the docs never state how much harder
that makes the perception job than it was for the teams that
got the free pass.

## Runners-up (real, lower impact - not in the top 5)
- "10 min per SCENE" (challenge_brief.md:16) vs the research's
  "10 min per QUESTION" (challenge-rules.md:20-23) -
  operationally different budget; the brief's own parenthetical
  is self-contradictory.
  Resolve against the dev-kit README.
- `/challenge_question` subscribed RELIABLE + TRANSIENT_LOCAL
  (adapter_node.py:171) on an unverified assumption; if the
  hidden evaluator publishes VOLATILE, QoS won't match and the
  module silently gets zero questions. Needs live-sim check.
- Region-heading prose note says heading "unused/artifact"
  (vla3d_notes.md:192-196) but the doc's own sample rows show
  pi/2,
  3pi/2 - the CODE is correct (loader reads the heading), only
  the note is wrong.
- `/sensor_scan` input topic (research io-contract.md:54) is
  unused and the adapter docstring miscounts inputs as "six" not seven
  - legal (subset), but misleading to contributors.
- NROS-vs-exploration gap: 1st place named a MULTI-MODAL
  frontier-scoring mechanism; the design deliberately keeps VLM
  frontier calls rare/bounded. A deliberate tradeoff, but never
  weighed against the actual winner's differentiator.

## Gaps in the RESEARCH docs surfaced by the comparison (bonus)
- No ground-truth numerical answers ship in the released
  question JSON at all (question_analysis.md:193) - the
  research's "exact match" framing implies an answer key that
  does not
  exist.
- Dev kit ships per-scene `trajectory_q4/q5.ply` reference
  trajectories for IF questions - undocumented on the research
  side.
- `/challenge_question` (String, ~1 Hz) is missing from the
  research I/O inputs table.
- The per-scene qtype mix here (1 numerical : 2 object-ref :
  2 IF) is programmatically verified across all 15 training
  scenes - stronger than the research's UNVERIFIED hedge (test
  scenes still open).

## Acceptance Criteria
- [x] Top-5 holes identified, each with research-side and
      implementation-side citations and a contradiction-vs-omission label.
- [x] At least the two load-bearing claims verified in source
      (not left as agent speculation).
- [x] Falsified candidate(s) dropped, not silently kept.
- [ ] (Optional next) confirm #2 id-space and #5(a) pair-shape
      against the actual VLA-3D JSON on a machine that has it.

## Artifacts
- This file (the ranked list). No code changed.

## Stage Gates
| Gate | State | Evidence / notes |
| --- | --- | --- |
| Task | APPROVED | top-5 gap analysis requested as a task |
| Spec | N/A | analysis-only deliverable |
| Plan | N/A | 4-agent compare + verify, done |

## Todo
- [x] 4 parallel comparison agents (I/O, scoring, prior-art,
      VLA-3D)
- [x] Verify obj-ref dual-publish (FALSIFIED - dropped) and
      "near" formula (CONFIRMED wrong) in source
- [x] Adjudicate + write ranked top 5
- [ ] Decide with team which holes to fix; if #2/#5a confirmed
      against real JSON, they jump priority

## Status re-check (2026-07-11, after the research-import PR)

Re-verified each hole against the current tree (origin/main
unmoved at 5c8601d; the import branch adds docs only, no code):

- #1 near threshold: OPEN - toolbox.py:40, :137 unchanged.
- #2 target_index id-space: OPEN - scoring.py:204 still joins
  unvalidated; needs the real VLA-3D JSON check.
- #3 IF scorer objective: OPEN - scoring.py:807 still the
  Frechet + coverage proxy only.
- #4 stale 2025 docs: PARTIALLY FILLED by the import -
  prior_art.md and organizer_playbook.md now carry the full
  leaderboard and the CopyPasta corrections. Still open:
  folding the GT-markers-banned implication into
  architecture.md / master_plan.md and reacting to it in the
  schedule.
- #5 GT relation semantics: OPEN - _scene_graph_count
  (scoring.py:230) and _size_token (loader.py:165) unchanged.

## Notes
- 2026-07-11: created. 4-agent compare against
  the local clone. Dropped one agent-flagged hole
  (object-ref never navigates) after verifying
  controller.py:350-354 publishes BOTH marker and waypoint -
  the concern was a scope artifact (agent didn't read the
  controller). Confirmed "near" is object-size-scaled in
  toolbox.py:137-140, contradicting our region-volume finding.
- Framing: the build is strong (642 tests, honest self-labeled
  proxies, most of the dossier lessons already applied). These
  5 are the highest-leverage divergences from the research, not
  a verdict on the codebase.
- 2026-07-11 (later): re-checked all five holes after the
  research-import PR landed on the branch; #4 partially filled
  (docs), #1/#2/#3/#5 unchanged in code. See "Status re-check"
  above.
- 2026-07-11 (later): traced the runtime GT question directly
  (does the build read disallowed GT semantics at test time?).
  Answer: no - GT is quarantined to the dev harness; the ROS
  runtime feeds a perception-sourced (currently empty)
  SceneIndex, never the loader. Full evidence + the Phase-2
  caveat in the new "Verified clean: no disallowed GT semantics
  at test time" section above.
- 2026-07-14: #2 and #5a CLOSED against the real VLA-3D JSON
  (all 15 scenes, instance-level dump in the numerical-count
  diagnosis task record). #2: every referential target_index
  exists as an object_id - the id join is sound (caveat: 5-25%
  of statements carry a target_class that drifts from the
  object's raw_label). #5a: `between` IS pair-shaped
  ([id,id] pairs) - research right; `hanging_on` is a FLAT
  scalar list - research wrong on that half; the scorer's
  uniform flattening garbles between pairs (latent, no current
  numerical question routes through it). #1 (near) got a data
  point: the only near-driven battery question passed with the
  current adaptive form - demoted from "proven wrong" to
  "sweep dimension". See
  docs/tasks/T5-numerical-count-diagnosis/docs/diagnosis.md.
