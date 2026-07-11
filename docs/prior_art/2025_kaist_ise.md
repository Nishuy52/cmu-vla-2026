> Imported 2026-07-11 from a parallel research stream. See
> `docs/prior_art/README.md` for provenance and conflict rules.

# KAIST-ISE team (2025 entry) - Gemini-segmented scene
graph + GPT LLM reasoning, non-finalist

One-liner: a participating, non-finalist 2025 entry that
built a full three-task pipeline (Gemini 2.5 Flash for
open-vocabulary segmentation, a hand-engineered geometric
scene-graph builder, GPT-5/GPT-4o for reasoning) in a very
short window (all 36 commits land within two days,
2025-09-15 to 2025-09-16); the scene-graph relation-rule
thresholds are the most directly reusable artifact here.

This team does **not** appear on the 2025 finalist
leaderboard (NROS 44.26, ReasonX 34.58, CopyPasta 30.98,
URL-KAIST 22.80) - it is a competing entry that did not
place among finalists, not a top-4 team.

## What it is

- Repo: https://github.com/CMU-VLA-KAIST-ISE/CMU-VLA-Challenge-2025
  (org "CMU-VLA-KAIST-ISE", repo name matches the 2025
  challenge).
- It is a fork; GitHub's recorded fork parent (confirmed
  via `git remote -v` after `gh repo clone`) is
  https://github.com/HaochenZ11/CMU-VLA-Challenge - the
  2025 official dev kit - not the `Yuxin916` 2026 dev kit
  named in this repo's own `CLAUDE.md`. Worth remembering:
  the 2026 dev kit (Yuxin916) and the 2025 dev kit
  (HaochenZ11) are different repos.
- 36 commits ahead of upstream, all authored between
  2025-09-15 and 2025-09-16 (two calendar days per
  `git log upstream/main..main --format='%ad' --date=short`).
  This reads as a late, compressed build rather than a
  sustained research effort.
- Contributors (from commit authorship in the fork):
  - 이동하, `max107304@gmail.com`.
  - mummy / mummy-alive, `mummyee@ewhain.net` - Ewha
    Womans University domain.
  - Wonhyeok / Wonhyeok316, `choewonhyeog3@gmail.com`;
    the `interaction_manager` package's `package.xml`
    maintainer is listed as `wonhyeok316@kaist.ac.kr`,
    tying this contributor to KAIST.
  - The `gemini_api` package's `package.xml` maintainer
    is `yeonu.seo@example.com`, attributed to "AAILAB" -
    a fourth, unconfirmed name/lab not in the three given
    above. Unverified who "AAILAB" is or whether Yeonu Seo
    is a fourth team member or a template leftover.
  - All other package.xml maintainers (`dummy_vlm_py`,
    `color_scan_generation`) still say "Ji Zhang" - the
    original CMU dev-kit author - meaning those two
    packages were carried over largely unmodified from the
    upstream skeleton (see Architecture).

## Architecture

All added code lives under `ai_module/src/` as five ROS
(catkin) packages. None of this existed in the upstream
dev kit - the diff against `upstream/main` for `ai_module/`
is 61 files changed, 15527 insertions, 311 deletions, i.e.
the team built the whole reasoning module from a near-empty
scaffold.

Dispatch flow, reconstructed from
`ai_module/src/interaction_manager/src/interaction_manager.py`,
`launch_pipeline.py`, and the `.launch` files:

1. On startup, `interaction_manager.py` immediately
   publishes `/launch_type = 0`, which
   `launch_pipeline.py` (`LaunchOnEvent._handle`) maps to
   launching `leo_vlm/launch/gpt_llm.launch` (the GPT-5
   node, package name `leo_vlm`, node name `gpt_llm`).
2. `interaction_manager.py` builds a fixed classifier
   prompt ("strict single-digit classifier... 1 =
   numerical, 2 = object_reference, 3 =
   instruction_following") and publishes it to
   `/user_query`, which the GPT-5 node answers on
   `/llm_response`.
3. Depending on the digit returned, `/question_type` is
   published (1 or 2 - both trigger an immediate
   `/shutdown` after publish; 3 does not shut down
   because task 3 keeps running).
4. `LaunchOnEvent._handle` maps question type to launch
   sets:
   - type 1 or 2 (numerical / object-reference):
     `color_scan_generation.launch`,
     `planning_node_1_2.launch` (grid_node, tsp_node,
     explore_node - full-coverage exploration),
     `fusion_pipeline_SG_node_merge.launch` (Gemini scene
     graph node), `llm_prompt_builder.launch`.
   - type 3 (instruction-following):
     `color_scan_generation.launch`,
     `a_star_point_nav.launch` (grid_node,
     a_star_node_nav_node, tsp_node),
     `fusion_pipeline_SG_node_merge_task3.launch` (a
     second, near-duplicate Gemini node), `plan_task_3.launch`
     (a self-contained GPT-4o planner/executor).
   - a fourth launch type (`4`, `gpt_llm_4o.launch`) is
     wired in `launch_pipeline.py` but nothing in the
     repository ever publishes `Int32(data=4)` to
     `/launch_type` - see Dead code below.
5. Sensor fusion (`fusion_pipeline_SG_node_merge*.py`)
   subscribes to the 360 camera image and the two color
   point-cloud topics, detects "robot has stopped" via
   odometry (and a TF-based fallback poller), then on
   each stop calls Gemini once to segment the current
   frame, projects lidar points into the panoramic image
   to get 3D centers/AABBs per detected object, colorizes
   each object from the image pixels, and folds the result
   into a running, ID-tracked scene graph that is merged
   across stops (see Components).
6. The merged scene graph (simple: id/label/center/aabb +
   relations; "rich": adds NYU/NYU40 class ids, color
   histogram, volume) is republished on latched topics;
   `dummy_vlm_py/scripts/llm_prompt_builder.py` (type 1/2)
   or `plan_task_3.py` (type 3) turn it into the final LLM
   prompt.
7. Output is published on the exact topics named in this
   repo's `docs/challenge_brief.md`: `/numerical_response`
   (`std_msgs/Int32`), `/selected_object_marker`
   (`visualization_msgs/Marker`, a CUBE built from the
   object's AABB/OBB), `/way_point_with_heading`
   (`geometry_msgs/Pose2D`, from the A* nodes).

## Components

### Perception (segmentation + 3D lift)

- File:
  `ai_module/src/gemini_api/src/fusion_pipeline_SG_node_merge.py`
  (type 1/2 path) and
  `fusion_pipeline_SG_node_merge_task3.py` (type 3 path,
  ~90% identical code, forked rather than shared).
- Detector: **Gemini 2.5 Flash** (`google-genai` SDK,
  `genai.Client`), not a locally-run detector - all object
  proposals come from a hosted VLM call, not ground-truth
  simulator markers. This means their "own detector" really
  means "Gemini's open-vocabulary segmentation", not a
  trained model of their own.
- Two-stage prompting per stop event:
  1. One-time (memoized in a `global ob_list`) call: "Extract
     object names from QUESTION" - turns the challenge
     question into a comma-separated allow-list of nouns.
  2. Per-frame call: `Segment objects in the image that in
     {ob_list} (up to strictly 5)` (task 1/2 variant) - each
     result is `{box_2d, mask (base64 PNG), label}` via a
     structured `response_schema=list[SegItem]`.
  - In the task3 file the segmentation prompt additionally
    asks for a `rel_score` field per object (an inline 0-100
    relevance score baked into the same JSON schema) and a
    concurrent multi-worker retry harness
    (`_seg_with_timeout` with `ThreadPoolExecutor`, 4 workers
    x 4 attempts) plus a `backup_segmenter` fallback that is
    referenced but never assigned anywhere in the class - a
    latent `AttributeError` bug in the timeout/retry path if
    all workers fail (see Strengths/weaknesses).
  - The task-1/2 (`_node_merge.py`, non-task3) file instead
    computes relevance via a *separate* third Gemini call - a
    `RelevanceScore` structured-output prompt scoring
    `information_completeness`, `object_coverage`,
    `coverage_extent` (each 0-10, explicitly told to be
    "conservative") - and averages the three into a 0-10
    `rel_score` used only to pick which saved frames get
    passed to the final GPT-5 prompt (top-K by score, see
    below). Two different rel_score schemes exist in the two
    files for essentially the same purpose - inconsistent,
    unlikely both were finished.
  - A very large hardcoded `OBJECTS_UNITY` vocabulary list
    (roughly 350 Unity-scene object names, e.g. "display
    ledge", "TV cabinet", "Arabic jar") appears in both
    files but is **commented out** in the currently-active
    prompt path - the live segmentation prompt is the
    bare `Segment objects in the image that in {ob_list}...`
    line, not label-constrained by this vocabulary. Confirms
    a design that was tried (closed-vocabulary label
    constraint against Unity's actual asset names) and later
    abandoned/disabled in favor of open-ended labeling.
- 3D lift: lidar points are transformed into the panoramic
  camera frame via `scan2pixels_mecanum()` - an equirectangular
  projection (`u = W/(2*pi) * atan2(-y,x) + W/2`,
  `v = H/(2*pi/3) * atan(-z/horiDis) + H/2`, i.e. a 360x120-deg
  FOV assumption baked into the constant `2*pi/3`), then
  points falling inside a detected mask are kept to build a
  per-object 3D point cluster, from which an AABB, center, and
  a top-down XY "footprint" rectangle are derived.
- Color: per-object pixel colors are converted to HSV and
  bucketed into 11 named colors (`COLOR_NAMES`, see
  `_classify_basic_colors`); thresholds are: black `v<0.20`,
  white `v>0.90 and s<0.10`, gray `s<0.20` (excluding
  black/white), then hue bands on the chromatic remainder -
  red `h<=15 or h>=345`, orange `15<h<=45`, yellow
  `45<h<=65`, green `65<h<=170`, cyan `170<h<=200`, blue
  `200<h<=255`, purple `255<h<=290`, brown a `v<0.6,s>0.2`
  override inside the orange band. Top-3 colors by pixel-count
  share are kept per object with their percentage.

### Scene graph and relation rules (directly reusable)

- File: `ai_module/src/gemini_api/src/scene_graph_builder.py`
  - a small, dependency-free, fully deterministic geometric
    rule engine (`build_scene_graph`), no learned component
    at all. Inputs are per-object `id, label, center (xyz),
    aabb {min,max}, footprint (xmin,ymin,xmax,ymax
    top-down)`; output is `{frame, timestamp, objects,
    relations}` where `relations` is a dict keyed by
    relation name, each mapping `str(object_id) -> [related
    ids]` (or, for `between`, `[[b_id, c_id], ...]`).
- Relation rules, all pairwise over every (i, j) with
  `i != j` (parametrized by a `SceneGraphParams`
  dataclass; two different call sites use two different
  parameter sets - see below):
  - **near**: Euclidean distance between centers
    `< near_dist`.
  - **above / below**: vertical center gap
    `dz = center_j.z - center_i.z`; `above` if
    `dz > above_gap` AND horizontal footprint overlap
    ratio `r > overlap_ratio`; `below` if
    `dz < -above_gap` and same overlap condition. Overlap
    ratio is `intersection_area / min(area_i, area_j)`
    (falls back to `area_i+area_j` if either area is ~0).
  - **on** (j is "on" i): let `top_i` = max z of i's AABB,
    `bot_j` = min z of j's AABB, `gap = bot_j - top_i`;
    true if `-0.02 <= gap <= on_gap` AND overlap ratio
    `r > max(0.5, overlap_ratio)` - i.e. "on" requires much
    higher horizontal overlap (>=50%) than "above/below".
  - **left_of / right_of** and **in_front_of / behind**:
    computed in the *robot's* frame, not the map frame -
    object centers are translated by robot (x,y) and
    rotated by `-yaw` first. Then lateral offset
    `rel[1] > lr_thresh` -> left_of, `< -lr_thresh` ->
    right_of; forward offset `rel[0] > fb_thresh` ->
    in_front_of, `< -fb_thresh` -> behind. Both thresholds
    default 0.20 m (in the online node) or 0.05 m (in the
    export_merged_rich params, see next point) - i.e. these
    relations are **robot-pose-dependent**, not static
    object-object facts, and will change value as the
    robot moves even with no object motion. Notably, in the
    merged/rich-export path `robot_xyyaw` is hardcoded to
    `[0,0,0]` (see `SGMerger.export_merged_simple/rich`),
    so in that path left/right/front/behind collapse to
    "relative to the map-frame origin", not the robot's
    actual pose - almost certainly an unintended
    simplification/bug, since the online scene-graph
    consumer (`llm_prompt_builder.py`) explains these
    relations to the LLM as if they were robot-relative.
  - **between**: for anchor object i, scan all pairs
    (b, c) of *other* objects, project i's center A onto
    the segment BC in top-down XY, compute the lateral
    (perpendicular) distance `lat` from A to that
    projection; keep the best (smallest-`lat`) up to
    `max_between_pairs_per_anchor` (default 2) pairs where
    `lat < between_lat_thresh` and the projection parameter
    `t` is strictly inside `(0, 1)` (i.e. A must sit between
    B and C along the line, not off one end).
- Two threshold profiles exist for the same builder
  function, both magic numbers hand-tuned, no calibration
  data or ablation given:
  - Online node defaults (`OneShotFusionNode.__init__`,
    overridable via ROS params): `near_dist=0.8`,
    `overlap_ratio=0.2`, `on_gap=0.15`, `above_gap=0.20`,
    `lr_thresh=0.20`, `fb_thresh=0.20`,
    `between_lat_thresh=0.30`, `max_between_pairs_per_anchor=2`.
  - `export_merged_rich`'s inline params (used for the
    *merged* rich scene graph specifically):
    `near_dist=0.75`, `overlap_ratio=0.05`, `on_gap=0.25`,
    `above_gap=0.35`, `lr_thresh=0.05`, `fb_thresh=0.05`,
    `between_lat_thresh=0.5`, `max_between_pairs_per_anchor=2`.
  - These are meaningfully different (e.g. overlap_ratio
    0.2 vs 0.05, lr/fb 0.20 vs 0.05 m) with no comment
    explaining the divergence - looks like drift between
    two code paths rather than a deliberate design choice.
- Object identity across stops (`SGMerger`, same file):
  greedy nearest-match by (a) same label, (b) 2D IoU of
  the XY bounding rectangle `>= iou_thresh` (default 0.25),
  (c) center distance `<= center_dist` (default 0.75 m),
  scored by `iou - 0.1*center_dist` and taking the best
  match; unmatched detections spawn a new persistent
  track ID. On a repeat match, the code **does not fuse**
  geometry/color, it only bumps a `seen` counter and
  timestamp - so a track's AABB/color is frozen at first
  observation, never refined by later, possibly better
  views.
- Raw-label-to-NYU/NYU40 mapping (`_NYU_MAP`) is a tiny,
  hand-written dict (chair, sofa, table, tv, plant, desk,
  cabinet, shelf, bed, monitor, books, door, window,
  unknown - about 16 entries); anything else maps to
  `nyu_id/nyu40_id = -1` and passes the raw label through
  unchanged as both nyu_label and nyu40_label. Not a real
  NYU40 classifier, just a lookup table covering common
  furniture nouns.

### Reasoning LLMs (Gemini/GPT split)

- Split is task-type-based, not capability-based:
  - **Gemini 2.5 Flash** (`google-genai`) is used
    exclusively for perception - segmentation and the
    relevance-scoring calls inside the fusion node. It
    never produces the final answer.
  - **GPT-5** (`ai_module/src/leo_vlm/src/gpt_llm.py`,
    OpenAI Responses API, `model="gpt-5"` by default, ROS
    node/topic `gpt_llm`) does two jobs: (1) the
    question-type classifier prompt from
    `interaction_manager.py`, and (2) the final
    numerical/object-reference answer prompt from
    `llm_prompt_builder.py`, both over `/user_query` ->
    `/llm_response`. For the final-answer call it also
    attaches up to 3 images pulled from `/tmp/rel` (a
    plain filesystem directory, not a ROS topic) sorted by
    a relevance score parsed out of the filename via
    regex `_(\d{1,3})\.png$` - i.e. cross-process
    image hand-off happens through the filesystem, not
    message passing.
  - **GPT-4o** is used in exactly one place:
    `ai_module/src/interaction_manager/src/plan_task_3.py`,
    which makes its own direct `OpenAI(...)` client calls
    (`model="gpt-4o"`, hardcoded, not the shared `leo_vlm`
    node) for: (a) parsing the instruction into
    waypoint/between/avoid/sequence directives, (b) picking
    a target object per waypoint from the partial scene
    graph, (c) a vision check ("is the object inside the
    red/blue box really X?") that re-renders the last 360
    image with an OpenCV-drawn quadrilateral and sends it
    back to GPT-4o for a True/False confirmation before
    committing to a move.
  - `ai_module/src/leo_vlm/src/gpt_llm_4o.py` is a
    **second**, separate GPT-4o ROS node (topics
    `/user_query_4o`, `/user_image_4o`) with its own launch
    file (`gpt_llm_4o.launch`) and its own dispatch code
    path (`launch_type == 4`) in `launch_pipeline.py` - but
    nothing in the repository ever publishes
    `Int32(data=4)` to `/launch_type`. This node is dead:
    present, wired for dispatch, never triggered.
- Prompting style throughout favors long, highly
  structured, few-shot, strict-output-format prompts
  (explicit "OUTPUT FORMAT (STRICT)" blocks, JSON schemas
  with worked examples, explicit synonym-normalization
  rules e.g. "nightstand" -> "bedside_table", "couch" ->
  "sofa") rather than short zero-shot prompts - a fairly
  mature prompt-engineering style even though the
  underlying geometry/fusion code has visible seams.

### Exploration strategy

Two independent, non-shared exploration implementations
exist for the two task groups:

- **Type 1/2 (numerical / object-reference)**:
  `ai_module/src/planning_node/src/explore_node.py`
  (`CoverHeuristicNode`, ROS node `uncovered_area_node`).
  Full occupancy-grid coverage exploration: builds an
  incrementally-updated visibility set via 8-octant
  recursive shadowcasting (`covered_points` /
  `_scan_octant`, a classic FOV/shadowcasting algorithm)
  from the robot's current grid cell, capped at
  `max_sight_distance = 20.0` cells; frontiers are cells
  adjacent to "seen-free" cells that are themselves
  unseen; the next target is chosen by
  `find_best_frontier_adaptive`, a weighted sum of an
  info-gain term (how many new cells would become visible)
  and a distance term (Gaussian-like falloff around a
  "preferred distance" of `10x grid_size`), with weights
  that shift from favoring info-gain+distance equally
  early (`covered_ratio<0.5`) toward favoring distance
  almost exclusively late (`covered_ratio>=0.75`, weights
  0.1/0.9). Stops when `coverage_ratio > 0.9999` or after a
  hardcoded `stop_time = 60*9` seconds (9 minutes) - close
  to but under the 10-minute-per-question budget in the
  challenge (see this repo's
  `docs/challenge_brief.md`/`docs/challenge_brief.md`).
- **Type 3 (instruction-following) fallback**:
  `plan_task_3.py` defines its own inline
  `random_exploration` class (MST-based, not
  shadowcasting): builds a minimum spanning tree over the
  A* node graph (`/mst_edges_marker`), roots it at the
  highest-degree node closest to the robot, and does a
  DFS-with-backtracking traversal order; when a "go to X"
  step can't be resolved from the current partial scene
  graph, it advances to the next unvisited node in that
  MST traversal (or the nearest unvisited node if the
  traversal position can't be found), while pruning
  nodes near an `avoid` list. Used only when GPT-4o's
  target-selector prompt returns `"random_explore"`.
  A **near-duplicate** copy of this same class also
  exists as a standalone module,
  `ai_module/src/interaction_manager/scripts/random_exploration.py`,
  imported by `scripts/__init__.py` - but the package's
  `CMakeLists.txt` `catkin_install_python` list only
  installs `interaction_manager.py`, `launch_pipeline.py`,
  `path.py` (not `plan_task_3.py`, not anything from
  `scripts/`), and no other file in the repo imports the
  `scripts` package. This copy looks like **dead code**: an
  earlier/parallel version of the class that was later
  inlined into `plan_task_3.py` directly rather than
  imported, and never deleted.

### Output handling

- Numerical (`/numerical_response`, `std_msgs/Int32`):
  `llm_prompt_builder.py` parses GPT-5's final response as
  a bare integer and publishes it directly; on publish it
  also wipes `/tmp/rel` (the relevance-image staging dir)
  and fires `/shutdown`.
- Object-reference (`/selected_object_marker`,
  `visualization_msgs/Marker`): GPT-5 returns a single
  object id (bare integer); the id is looked up in the last
  merged (non-rich) scene graph to get its `center` and
  AABB `min/max`, and a `Marker.CUBE` is built directly from
  that AABB (axis-aligned, `yaw=0` always - there is a
  dead/unused `obb_from_corners_xyyaw_no_numpy` helper that
  would compute a true oriented box + yaw from the *rich*
  scene graph's 8-corner `bbox`, gated behind a hardcoded
  `rich_not = True` flag that always takes the AABB branch
  instead, so the OBB code path is written but never
  taken).
- Instruction-following (`/way_point_with_heading`,
  `geometry_msgs/Pose2D`): produced by
  `planning_node/scripts/planning_node/a_star_algorithm.py`'s
  `A_star.move()` (not read in detail for this dossier;
  called from both `explore_node.py` and `plan_task_3.py`
  with a `(current_pose, target_xyz)` pair and returns
  `True` once the goal is reached, in a `rospy.Rate(5)`
  polling loop). `plan_task_3.py` additionally reprojects
  each candidate object's 8-corner bbox into the 2D
  panoramic image (`image_coordinate()`, the same
  equirectangular projection math as the fusion node) to
  draw a debug quad and to let GPT-4o visually confirm
  "is the thing in this box really the target" before
  committing to the waypoint - a same-model
  propose-then-verify loop, not just propose.

## Key numbers

- 36 commits ahead of the 2025 dev-kit fork parent
  (https://github.com/HaochenZ11/CMU-VLA-Challenge), all
  dated 2025-09-15 to 2025-09-16 (2 calendar days).
- `ai_module/` diff vs upstream: 61 files changed, +15527 /
  -311 lines - the whole reasoning module is new code, not
  a light patch.
- Not on the finalist leaderboard (NROS 44.26, ReasonX
  34.58, CopyPasta 30.98, URL-KAIST 22.80) - no score for
  this team is available from that leaderboard; none found
  elsewhere during this pass (unverified whether any public
  score exists at all).
- Segmentation cap: "up to strictly 5" objects per Gemini
  call (both fusion-node variants).
- Coverage-exploration time budget: hardcoded 540 s (9 min)
  stop condition, under the per-question 10-minute budget
  in `docs/challenge_brief.md`.
- Object-track merge thresholds: IoU >= 0.25, center
  distance <= 0.75 m (see Components/scene graph above for
  the full relation-threshold table).

## Strengths and weaknesses for our setting

Strengths:
- The scene-graph relation rules
  (`scene_graph_builder.py`) are a compact, dependency-free,
  well-isolated ~170-line module with clear geometric
  semantics per relation and tunable thresholds - the
  single most directly portable artifact in this codebase
  for our own object-reference/numerical reasoning, whether
  we take it as-is or as a reference design to critique and
  improve (especially the near/on/above/below/between
  logic, which is generic and simulator-agnostic).
- The equirectangular lidar-to-panorama projection
  (`scan2pixels_mecanum` / `image_coordinate`) is a useful
  worked example of going from the 360 camera + lidar setup
  in our challenge to per-pixel/per-object 3D grounding,
  assuming our sensor geometry matches (same 1920x640,
  360x120-deg FOV assumption - verify against our actual
  dev-kit camera spec before reusing verbatim).
- Frontier/shadowcasting coverage exploration
  (`explore_node.py`) is a solid, well-commented reference
  implementation of full-map coverage under a time budget,
  with an adaptive info-gain/distance weighting scheme that
  is easy to reason about and reuse.
- The propose-then-verify pattern in `plan_task_3.py`
  (crop/box the candidate object back into the source image
  and ask the VLM "is this really X" before committing) is
  a cheap, generalizable way to reduce grounding
  hallucination that we should consider regardless of which
  models we use.

Weaknesses / traps to avoid:
- Two scene-graph threshold profiles for the same relation
  types with no documented rationale (see Components) -
  a caution against copying "the" thresholds without first
  checking there is only one canonical set.
- The robot-frame left/right/front/behind relations are
  computed with `robot_xyyaw` hardcoded to `[0,0,0]` in the
  merged-graph export path (`SGMerger.export_merged_rich` /
  `export_merged_simple`), silently breaking the intended
  robot-relative semantics that `llm_prompt_builder.py`'s
  prompt tells the LLM to assume - a subtle bug worth
  testing for explicitly if we build something similar.
- Track fusion never updates existing tracks' geometry/color
  after the first observation - first-seen AABB is treated
  as final, which will be wrong whenever the first partial
  view under- or over-estimates object extent.
- Relies entirely on a hosted VLM (Gemini) for every
  detection with no local fallback - single point of
  failure/latency/cost dependency, and the `backup_segmenter`
  referenced in the retry path of the task-3 fusion node is
  never actually instantiated (latent bug, would raise
  `AttributeError` if ever reached).
- The commented-out `OBJECTS_UNITY` closed-vocabulary
  scheme suggests they tried and abandoned constraining
  Gemini's labels to the simulator's actual asset
  vocabulary - worth understanding why before assuming
  open-vocabulary labeling alone is sufficient for scoring
  against ground truth.
- Meaningful dead code exists in this codebase (the
  `gpt_llm_4o.py` node/launch/dispatch path,
  `interaction_manager/scripts/random_exploration.py`) -
  a reminder to trace actual wiring (launch files +
  publisher/subscriber topic names), not file presence,
  when judging what any team's system "does".

## Takeaways for our 2026 module

- Treat `scene_graph_builder.py`'s relation definitions as
  a strong starting draft for our own spatial-relation
  vocabulary (near/above/below/on/left_of/right_of/
  in_front_of/behind/between), but re-derive thresholds
  against our own object scale distribution rather than
  reusing their numbers verbatim, and pick one threshold
  profile (their two disagree by 4x on some parameters).
- A stop-detect -> single-shot VLM segmentation -> lidar
  lift -> ID-tracked merge -> LLM-over-scene-graph pipeline
  is a reasonable, implementable shape for the
  numerical/object-reference tasks and matches the
  intended I/O contract in our own
  `docs/challenge_brief.md`; the propose-then-verify
  crop-and-reask step for object reference is worth
  carrying forward independent of model choice.
- For instruction-following, MST-based frontier traversal
  plus box-perimeter "avoid" corridors
  (`create_avoid_path_between_objects` /
  `create_avoid_near_objects`) is a workable pattern for
  turning a natural-language route description into
  waypoint sequences with obstacle avoidance, though this
  team's version is tightly coupled to their own PSG schema
  and would need real integration work, not a drop-in
  reuse.
- This is prior-art context from the pre-merge research
  pass - no architecture decision should be drawn from
  this dossier alone; weigh takeaways against
  `docs/architecture.md` and `docs/master_plan.md`.

## Sources

- Repo (primary source for all code claims above):
  https://github.com/CMU-VLA-KAIST-ISE/CMU-VLA-Challenge-2025
  - cloned locally and inspected directly; all file paths
    cited above are relative to `ai_module/src/` in that
    repo.
- Fork parent / 2025 official dev kit (per
  `git remote -v` after `gh repo clone`):
  https://github.com/HaochenZ11/CMU-VLA-Challenge
- This repo's own challenge-context docs, cross-checked for
  the I/O contract and time budget:
  `docs/challenge_brief.md`, `docs/io_contract_crosscheck.md`.
- 2025 finalist scores (NROS 44.26, ReasonX 34.58,
  CopyPasta 30.98, URL-KAIST 22.80) as given in this task's
  brief; not independently re-verified in this pass -
  treat as caller-supplied context, not sourced here.
- No external (non-repo) sources were fetched for this
  dossier; everything above is derived from static
  inspection of the cloned repository (code, `git log`,
  `git diff --stat`, `.launch` XML, `package.xml`).
