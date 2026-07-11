> Imported 2026-07-11 from a parallel research stream. See
> `docs/prior_art/README.md` for provenance and conflict rules.

# Urban Robotics Lab @ KAIST (4th place, 2025) - the
most complete public pipeline, and the lowest scorer of
the four finalists

One-liner: this team shipped by far the most componentry
of any 2025 entry found in this survey (full scene-graph
perception, active grounding, frontier exploration,
keyframe memory, an "avoid X" region-blocker) - yet placed
4th of 4 finalists with final score equal to its
preliminary score (22.80, no real-world-round gain). Code
reading below turns up concrete, evidence-grounded reasons
this might be so: an unsandboxed single-point-of-failure
LLM call, internally inconsistent/duplicated scaffolding, a
narrow per-scene detector class list despite vendoring a
broad one, and a ground-truth-semantics fusion path that
2026's rules would forbid outright.

## What it is

- Repo: https://github.com/url-kaist/Vision-Language-Autonomy
  (default branch `master`). Cloned shallow (`--depth 1`)
  for this pass; only one commit is visible locally, a
  squashed merge `5bbab60` by author `dazory
  <ds.hong@kaist.ac.kr>`, dated 2025-09-16 21:01 +0900,
  message "Merge branch 'master' of
  https://github.com/url-kaist/Vision-Language-Autonomy" -
  consistent with the prior recon note of "19 commits, all
  2025-09-16, initial commit from old repo" (the fuller
  history was not refetched here; only branch is `master`,
  no tags).
- Contributors named in prior recon: Dasol Hong
  (ds.hong@kaist.ac.kr, matches the commit author),
  Jeewon Kim, Taeyun Kim.
- This is a different KAIST team from the non-finalist
  "KAIST-ISE" entry already covered in
  `docs/prior_art/2025_kaist_ise.md` (Gemini-segmented
  scene graph, KAIST + Ewha students, did not place). Do
  not conflate the two - only this Urban Robotics Lab team
  is a top-4 finalist.
- `ai_module/src/vlm/` is the orchestrator package: a thin
  C++ `manager.cpp`/`manager.h` node plus `launch/main.launch`
  and `launch/benchmark.launch`, wiring together `sem`,
  `visual_grounding`, and `user_interface` (confirmed via
  `package.xml` build/exec depends). It is not itself a
  model server - actual GPT-4o/YOLO-World calls live inside
  `sem` and `task_planner`/`visual_grounding`.
- Root `README.md` is the stock 2025 dev-kit README
  (question-type examples, scoring breakdown, 10-minute
  limit), not a KAIST-authored writeup - this team left no
  prose description of its own approach anywhere in the
  repo.

## Architecture

Question in -> `task_planner` (GPT-4o decomposes the raw
question string into an `Entity`/`Edge`/`Task`/`Plan`
graph, and the LLM itself sets a `type` flag: 0 = find,
1 = count, 2 = instruction-following) -> `task_planner`
picks and `roslaunch`es one of three `visual_grounding`
launch files (`vg.launch` for find/count,
`vg_if_two.launch`/`vg_if_one.launch` for
instruction-following depending on whether constraints
exist) -> `visual_grounding` (either `active_grounder.py`
for find/count or `visual_follower.py` for
instruction-following, both subclassing the same
1950-line `base_visual_grounder.py` engine) pulls the
live scene graph from `sem` over ROS services, drives the
`exploration` package's frontier/coverage exploration
strategy, and issues its own GPT-4o vision calls over
captured keyframes -> a type-specific answer topic is
published (`Int32`, object `Marker`, or a continuous
`/active_waypoints` steering stream).

Two structural findings cut across this flow:

- `sem` ships **two independent scene-graph builder
  nodes** selected purely by which launch file is run:
  `scene_graph_node` (YOLO-World detection only,
  `sem.launch`/`sg.launch`) and `gt_graph_node`
  (YOLO-World fused with simulator ground-truth object
  markers, given priority over vision, `gt.launch`). See
  Perception below - this is the single most consequential
  finding in this dossier.
- `task_planner` exists as **two non-matching
  implementations**: a C++ ROS node
  (`src/task_planner.cpp`) whose service client targets a
  service literally named `"get_subplans"`, and the actual
  wired path, a Python service
  (`scripts/task_planner_server.py`) advertising
  `/task_planner/request_subplans` - the names do not
  match, and only the Python path is invoked by
  `launch.sh`.

## Components

### Perception

- Real detection path is inline in each `sem/app/*_node`
  file, not the standalone `object_detect.py` (which uses
  plain Ultralytics `YOLO`+`SAM` and is not imported by any
  live node - dead code). The wired path uses vendored
  `third_party/mmyolo` plus a custom `yolo_world/` package
  (`config/yolo-world_config_{l,m,x}.py`, selected at
  runtime by `config.json`'s `YOLO_MODEL_SIZE`), 2D
  box -> SAM (`mobile_sam.pt`) mask refine -> lidar-point
  selection inside the mask, projected via an
  equirectangular 360-degree panorama mapping
  (`scan2pixels_*`, matching the robot's 360 camera), not a
  classic pinhole reprojection.
- **Correction to prior recon**: the class taxonomy is
  *not* actually ScanNet200 at runtime. `data/
  scannet200_classes.txt` (200 lines) and its colors JSON
  exist in the tree but a repo-wide grep found zero code
  references to them - unwired. The live class set is
  either the 162-item `MATTERPORT_LABELS_160` default or a
  small (7-50 item) hand list from a
  per-scenario file (`sem/config/00824.json`,
  `homebuilding2.json`, `livingroom1.json`, `real.json`).
  YOLO-World's own base config still declares the stock
  1203-class LVIS / 80-class COCO numbers
  (`num_classes=1203`, `num_training_classes=80`,
  `yolo-world_config_l.py`), further confirming this is
  stock YOLO-World reconfigured per-scene, not a genuinely
  broad open-vocabulary deployment.
- `object_merge.py` associates detections across frames by
  tracking-ID continuity first, then a centroid-distance +
  3D-bbox-overlap + containment gate
  (`dist < 0.8` and class-match/overlap thresholds), with
  confidence taking the max across merges and a separate
  periodic (`every_n_iter`) re-merge pass for stragglers.
  Persistent per-point identity is additionally kept in a
  voxel hash map keyed on rounded 3D coordinates.
- `room_segment.py`'s OpenCV watershed room-splitting
  pipeline is a standalone offline tool (own `argparse`
  `__main__`, no cross-references found) - the live nodes
  instead label rooms at runtime via CLIP zero-shot
  classification of the current camera frame against text
  prompts like `"a photo of living room"`
  (`jackal_sg_node`). Two independent, unreconciled
  approaches to the same problem.
- `gt_sg_node` (361 lines) is a pure ground-truth logging
  tool: it subscribes unconditionally to `/object_markers`
  (`MarkerArray`, simulator ground truth) and dumps matched
  keyframes + a `gt_objects.json` - a dataset-collection
  helper, not part of the question-answering path.
- `gt_graph_node` (2485 lines, launched by `gt.launch` with
  `config/simulation_gt.json`) is the consequential one: it
  runs a ground-truth ingestion path (`from_GT`, matching
  markers into the lidar cloud via a GT-labeled semantic
  image color match) **unconditionally alongside** the
  YOLO-World path, every loop iteration, with no code-level
  toggle to disable it. GT objects are recorded with
  hardcoded `"yolo_conf": 1.0` confidence and are treated
  as authoritative: when a YOLO-derived object spatially
  coincides with a GT one, the YOLO node is deleted and
  folded into the GT node (log line: `"$$$$$$ GT ...
  already exists! removing yolo node ..."`).
  `config/simulation_gt.json` wires this node to
  `/camera/semantic_image` and `/semantic_scan` - a lidar
  stream the simulator itself already semantically labels,
  not raw sensor data.

### Map / scene representation

- `sem`'s output schema: `ObjectNode.msg` (id, xyzrgb
  points, center, min/max corners, class name/id,
  confidence), batched in `ObjectNodes.msg`, served by
  `/get_object_node_info` (`ObjectNodeFromClass.srv`).
- `snapshot/` (C++) defines a keyframe = one RGB image +
  one annotated RGB image + a list of detected 3D objects
  (`MapObject.msg`: id, class, 3D center/min/max bbox, 2D
  bbox), persisted as JSON
  (`/ws/external/vis/snapshots/snapshots.json`). Trigger is
  purely distance-based - a hardcoded 3.0 m threshold
  against `/state_estimation` odometry
  (`snapshot.cpp:42`), not tied to detections, frontiers, or
  the question. **No confirmed consumer** of the stored
  snapshot JSON or its `SnapshotData` service was found
  anywhere in `visual_grounding` (a repo-wide grep for the
  service name and the JSON path outside `snapshot/` itself
  returned nothing) - flagged as "not found," not
  "confirmed dead," since the query-side code may live
  outside what this pass covered.
- `exploration/occupancy_map.py`'s `TerrainMapBuilder`
  rebuilds a 50 m x 50 m, 0.1 m/px grid via 1080 raycasts
  per `/state_estimation` callback against the
  `/traversable_area` point cloud (default topic name;
  see Takeaways for why this specific topic name matters
  for 2026).
- `traversable_authority/` holds the traversable area as an
  XY point set with three services
  (`/traversable/apply_bbox`, `/traversable/apply_segment`,
  `/traversable/reset`) that remove/restore points to
  implement "avoid X" instructions, republishing on
  `/traversable_area_filtered`. Confirmed consumers:
  `astar_planner_py.py` and `coverage_path.py` both
  subscribe to the *filtered* topic. `occupancy_map.py`
  (which drives frontier detection), however, defaults its
  `~terrain_topic` param to the **raw**, unfiltered
  `/traversable_area` - so blocked regions constrain the
  path planner but not, by default, the frontier explorer's
  candidate targets.

### Reasoning LLM

- `task_planner_main.py` builds a single `LlmClient`
  against `OPENAI_API_KEY0` only. `config.json` declares
  `NUM_CLIENTS: 3` and three keys, but a repo-wide grep for
  `NUM_CLIENTS` in Python finds no references anywhere -
  the parallel-client fan-out implied by that config key is
  dead configuration, not exercised code.
- The prompt (`task_planner/scripts/prompt.txt`, paired
  with a hardcoded system message in
  `task_planner_main.py:40`) demands GPT-4o return a
  complete, directly-executable Python function
  (`def create_subtasks_by_LLM():`), with a strict object
  schema (unique entity IDs even for repeated names,
  `attr`/`exclude_attr` always `dict[str, list[str]]`,
  relations as `Edge(name=, source_id=, target_ids=[...])`,
  `'avoid'` entities routed to `plan.constraints`) and four
  worked few-shot examples.
- **This response is `exec()`'d directly** with no
  sandboxing
  (`task_planner_main.py`'s `create_subtasks()`): on
  failure it prints the raw response and bare
  `raise`s - no retry, no schema-repair, no fallback plan
  is ever constructed. This is the single least-robust
  call site in the whole pipeline (contrast with
  `visual_grounding`'s retry wrapper below).
- The LLM also decides the question-type flag inline
  (`plan.type` = 0/1/2), but `Plan.msg` has **no `type`
  field** - `task_planner_server.py` branches on it locally
  to pick a `visual_grounding` launch file, then it is
  silently dropped when the `Plan` is serialized to the ROS
  response, so no downstream consumer of
  `GetSubplansResponse.plan` can recover the classification
  from the message itself.
- `visual_grounding`'s own GPT-4o calls
  (`vlms/loaders/client.py`, `vision_client.py`) are
  genuinely vision-capable (base64-encoded images + text in
  one chat message) and are wrapped in real exponential
  backoff: 6 retries, 0.8 s base delay, 20 s cap, gated by
  a concurrency semaphore (max 2 concurrent LLM calls) -
  markedly more defensive than the task-planning call site.
- Prompting here is one unified template system
  (`system_instruction.py`'s `SYSTEM_INSTRUCTION` dict,
  keyed by `<type>_<action>`), not fully separate prompts
  per question type. It includes an explicit
  disambiguation protocol: pick exactly one "green" bbox
  candidate if confidently matched; if only ambiguous
  matches exist, return a "red" reference bbox instead, a
  signal meaning "move closer / explore more" rather than
  guessing. A second, separate prompt pair
  (`PATH_GENERATION`/`PATH_EVALUATION`) drives
  instruction-following: the local planner selects up to
  three grid points in strict "Green -> Red -> Blue" color
  order, and a separate "mobility evaluator" call later
  judges mission completion as a boolean.

### Exploration

- `frontier_mapping.py` is actually a CLIP-driven,
  VLFM-style semantic value map, not plain
  nearest-frontier: frontier cells are BFS-clustered, then
  PCA-reduced to 1-4 representative waypoints per cluster,
  and ranked by a running-average CLIP segment-score
  "value" projected along rays into free space, refreshed
  on a 1 s timer.
- `coverage_path.py`'s live strategy is boundary/contour
  following (OpenCV contours + A* stitching between
  points), not boustrophedon. A TSP-based alternative
  (`python_tsp` exact DP solver) exists in the same file
  but is confirmed dead code - never called from the live
  path.
- `multi_goal_planner.py` (a multi-target Dijkstra meant to
  rank several candidate goals cheaply) is confirmed
  imported but never instantiated anywhere - dead code.
- `astar_planner_py.py` is a standard 8-connected A*
  (Euclidean heuristic), with zero obstacle inflation by
  default (`~inflate_radius` defaults to 0.0 m) and an
  O(grid-size) score-array initialization on every single
  planning call, regardless of path length.
- The stopping condition is external to `exploration`
  itself and mixed: default question-directed frontier
  chase, falling back to full boundary coverage after a
  stall (no frontier target for 5 s and <3 m moved); a hard
  stop on an external `/success` `Bool` (published once
  grounding is confident); and, specifically for
  instruction-following questions, exploration stops as
  soon as a **single** frontier is reached, not full
  coverage (`object_navigation.py`, "Frontier reached
  during instruction_following ... Stopping exploration").
  `exploration_coverage.py`'s 85% coverage threshold only
  stops file-logging telemetry, not the planner.

### Output handling

- **Numerical**: not a single LLM-emitted integer. Each
  independent VLM call over a keyframe returns a
  `target_ids` list; `len(target_ids)` becomes one vote,
  folded across many calls into an `AggregatedResult`
  confidence aggregator; the eventual `Int32` on
  `/numerical_response` is this aggregated count.
- **Object-reference**: the VLM returns exactly one
  `target_id` per call; the final `Marker` bbox comes from
  that entity's existing 3D detection, refined via
  `scipy.optimize.minimize` against every keyframe that saw
  it - a geometric refinement step, not a fresh grounding
  call at answer time.
- **Instruction-following**: there is no single final
  waypoint-list message. Waypoints are streamed
  continuously as `MarkerArray`s on `/active_waypoints`,
  paired with `/exploration_strategy` mode toggles
  (`vg_first` / `semantic_frontier` / `geometric_frontier`)
  that the exploration package consumes in real time.
  Mission completion is a boolean event
  (`mission_completed_event`) set by a separate VLM
  "mission evaluator" call over trajectory history, not a
  discrete returned path.
- A commit(?)-level bug, flagged for completeness: `active_
  grounder.py` defines a **module-level `ANSWER_TYPE`
  dict with `find`/`count` inverted** relative to the
  correct mapping actually used in
  `base_visual_grounder.py` - it appears unreferenced
  elsewhere (dead but wrong), a small but telling sign of
  unreviewed code.

## Key numbers

- 2025 final leaderboard (per `docs/prior_art/README.md`,
  Wayback-verified screenshot): NROS 44.26 (1st), ReasonX
  34.58 (2nd), CopyPasta 30.98 (3rd), **Urban Robotics Lab
  @ KAIST 22.80 (4th)**. Its final score equals its
  preliminary score exactly - the real-world round added
  nothing, unlike CopyPasta (22.99 -> 30.98).
- `ai_module/src/config.json`, quoted in full (8 keys, the
  entire file):
  `NUM_CLIENTS: 3`, `OPENAI_API_KEY0/1/2` (placeholder
  strings), `MODEL_NAME: "gpt-4o"`,
  `VG_LOG_DIR: "/ws/external/vis"`, `MAX_WORKERS: 3`,
  `YOLO_MODEL_SIZE: "x"`, `YOLO_CONFIDENCE: 0.3`. No
  timeouts, no map/exploration parameters, no
  question-routing config live here - those live scattered
  across individual node files/launch args instead.
- `task_planner`'s own (Korean) README states an average
  measured planning time of **8.40 s per question**, dated
  2025-03-01 - roughly six months before the September
  submission window, and covering only the decomposition
  stage, not end-to-end exploration+grounding time.
- `visual_grounding` LLM retry: 6 attempts, 0.8 s base
  delay, 20 s max delay, exponential backoff with jitter,
  gated by a concurrency semaphore of 2.
- Answer-readiness gate (`base_visual_grounder.py`):
  commit an answer once `best_confidence > thres_high`
  AND `exploration_status == 'no_frontier'`, **or**
  regardless of confidence once remaining time drops to
  30 s of the 10-minute budget.
- Snapshot trigger: hardcoded 3.0 m odometry-displacement
  threshold. Occupancy grid: 0.1 m/px, 50 m x 50 m buffer,
  5 m observation radius (10 m specifically for
  instruction-following questions).

## Strengths and weaknesses for our setting

Strengths:

- By component count this is the most mature public 2025
  pipeline found in this survey (also noted in
  `docs/prior_art/README.md`): a real, continuous active-grounding
  loop that keeps gathering VLM evidence as exploration
  proceeds, rather than answering off one shot; genuine
  retry/backoff and concurrency control on the
  `visual_grounding` LLM call path; and a clean modular ROS
  service boundary between perception (`sem`), reasoning
  (`task_planner`), and grounding
  (`visual_grounding`) that maps well onto our own dev
  kit's package-per-concern layout.
- The evidence-aggregation-before-answering pattern (many
  independent keyframe-level VLM calls, only committing
  once confidence is high AND exploration is exhausted, or
  the clock is nearly out) and the explicit
  green/red-bbox "confident match vs. need to look closer"
  disambiguation signal are reusable design ideas
  independent of this team's placing.

Weaknesses - offered as evidence-grounded hypotheses for
the low score, not proven causes (the actual submitted
launch configuration and any post-hoc scoring notes were
not found anywhere in the repo or online):

1. **Ground-truth-semantics risk.** `gt_graph_node` fuses
   simulator ground-truth object markers with priority over
   vision detections, wired to `/object_markers`,
   `/camera/semantic_image`, and an already-labeled
   `/semantic_scan`. The 2025 dev kit README vendored in
   this very repo states teams may choose to use published
   GT semantics but "methods that do not use the published
   ground-truth semantics will be scored differently." If
   the team's actual submission used `gt.launch` (unknown -
   the repo does not record which launch file was
   submitted), their score may reflect a different rubric
   from vision-only entries, independent of raw capability.
2. **Single point of failure in decomposition.** GPT-4o's
   raw generated Python is `exec()`'d with no retry, no
   sandboxing, and no fallback plan on failure - unlike
   every other LLM call site in the repo. One malformed
   completion or transient API error here can zero out an
   entire question inside the 10-minute budget with no
   recovery path.
3. **Internally inconsistent scaffolding.** Two
   non-matching `task_planner` node implementations
   (mismatched service names); `Plan.msg` missing the
   `type` field the server branches on; the inverted
   `ANSWER_TYPE` dict in `active_grounder.py`;
   `RequestSubgraph.srv` returning an opaque
   `string graph # TODO` instead of structured data. These
   read as signs of a codebase assembled under time
   pressure (one squashed commit, one date) rather than
   exercised end-to-end.
4. **Narrow, per-scene-tuned classes despite a broad
   vendor.** Full `mmyolo` plus a 200-class ScanNet200 list
   are vendored, but never wired up - the live class set is
   a small (7-50 item) hand list per training scene, or a
   162-item Matterport default. This looks tuned to the
   known training scenes rather than genuinely
   open-vocabulary, a real generalization risk against the
   3 held-out test scenes.
5. **Possibly-unwired snapshot memory.** No consumer of the
   persisted keyframe/`SnapshotData` service was found in
   `visual_grounding` - if this holds beyond this pass's
   coverage, a nontrivial subsystem may be dead weight
   rather than contributing to answers.
6. **Frontier/traversable-authority mismatch.** The path
   planner (`astar_planner_py.py`, `coverage_path.py`)
   honors "avoid X" blocked regions via the filtered
   traversable topic, but the frontier explorer
   (`occupancy_map.py`) defaults to the raw, unfiltered
   topic - a plausible mechanism for exactly the
   "passing through forbidden areas" penalty our own
   `docs/challenge_brief.md` notes for instruction-following
   scoring.
7. **Compressed timeline.** All commits land in a single
   squashed 2025-09-16 merge, matching the pattern also
   found in the unrelated non-finalist KAIST-ISE entry
   (`docs/prior_art/2025_kaist_ise.md`, 36 commits across
   2025-09-15/16) - consistent with a rushed final assembly
   that would explain both the inconsistencies above and a
   score that did not move at all in the real-world round.

## Takeaways for our 2026 module

- **Most actionable finding**: our 2026 dev kit
  (`docs/challenge_brief.md`) states plainly, "Only listed
  sensor topics may be used at test time. No ground-truth
  semantics, no traversable-area data." Two things in this
  repo map directly onto that sentence: (a) `gt_graph_node`'s
  entire GT-fusion path is exactly the ground-truth-semantics
  use the 2026 rules now forbid outright, not merely
  discouraged - this part of the codebase would be a rules
  violation if reused as-is; (b) the whole exploration
  stack (`occupancy_map.py`, `coverage_path.py`,
  `astar_planner_py.py`, `traversable_authority`) is built
  on a topic literally named `/traversable_area`, which does
  not appear in our 2026 I/O contract
  (`docs/io_contract_crosscheck.md` lists `/terrain_map` and
  `/terrain_map_ext` instead). This is an inference from
  topic-name/rule-text correspondence, not a confirmed fact
  - worth a direct check once our 2026 base-autonomy stack
  is running: if `/traversable_area` simply is not
  published in 2026, none of this exploration code can be
  reused unmodified; it would need rebuilding against
  `/terrain_map`.
- The evidence-aggregation-before-answering gate (commit
  only when confident AND exploration-exhausted, or time is
  nearly out) is a reusable pattern for our own
  explore-vs-answer tradeoff, independent of this team's
  score.
- The green/red (confident-match vs. move-closer)
  disambiguation convention is a clean way to make an
  "I need more evidence" signal explicit in an LLM's output
  schema - worth considering for our own object-reference
  prompts.
- Do not copy the `exec()`-of-raw-LLM-code pattern in
  `task_planner` under any circumstance - unsandboxed
  execution of model output with zero retry and zero
  fallback is a concrete anti-pattern to avoid regardless of
  this team's placing; a parsed/validated JSON schema with
  retry is strictly safer and no more implementation effort.
- Component count does not predict score: this is the
  clearest case in our survey so far
  (per `docs/prior_art/README.md`) where the most componentry
  scored lowest of the four finalists. The visible
  in-code predictors are integration consistency
  (duplicated/mismatched node pairs, dropped fields) and a
  possible ground-truth dependency the 2026 rules have since
  closed off - not raw feature count.
- Both KAIST entries surveyed so far - this Urban Robotics
  Lab team and the unrelated non-finalist KAIST-ISE team
  (`docs/prior_art/2025_kaist_ise.md`) - independently
  converged on "geometric/hand-engineered scene graph +
  GPT-family LLM reasoning." Combined with NROS's stated
  capability split, this is now a second and third
  confirmation that this shape is the default competent
  2025 entry, not a differentiator on its own.

## Sources

- Repo (cloned shallow, `--depth 1`, for this dossier):
  https://github.com/url-kaist/Vision-Language-Autonomy
- Root `README.md` inside that repo (2025 dev-kit
  boilerplate: question examples, scoring breakdown,
  10-minute limit, ground-truth-semantics
  choice-and-scoring clause) - read directly, not a
  separate URL.
- `system/unity/README.md` inside that repo (base-autonomy
  stack description, Ubuntu 20.04 + ROS Noetic).
- `docker/README.md` inside that repo (Nvidia Container
  Toolkit setup, eval-machine reference).
- Contributor email observed in-repo commit metadata:
  ds.hong@kaist.ac.kr (Dasol Hong, commit author "dazory").
- This repo's own prior findings, cross-referenced
  throughout: `docs/prior_art/README.md` (KAIST Urban Robotics Lab
  paragraph, verified 2025 leaderboard table, "no
  real-world points added" note), `docs/challenge_brief.md`
  (2026 I/O contract, ground-truth/traversable-area ban),
  `docs/io_contract_crosscheck.md` (2026 legitimate topic list),
  `docs/prior_art/2025_kaist_ise.md` (the unrelated
  non-finalist KAIST team, for disambiguation).
