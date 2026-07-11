> Imported 2026-07-11 from a parallel research stream. See
> `docs/prior_art/README.md` for provenance and conflict rules.

# ReasonX (2nd place, 2025) - Gemini planner +
RoboRefer pixel-grounding, riding the 2025 dev kit's
ground-truth object-marker topic

Compiled 2026-07-11 from a shallow clone of
https://github.com/Yuxin916/ReasonX_SGTeam (all
branches, depth 1 per branch tip). ReasonX is the only
2025 top-2 team with public code (per
docs/prior_art/README.md's leaderboard resolution: sim
prelim 31.42, final 34.58, 2nd of 4). This doc reads
the code directly rather than repeating the earlier
docs/prior_art/README.md summary; treat this as the deep-dive
companion to that entry.

## What it is

- Joint NTU + NUS team (Singapore) - NOT purely NUS as
  first assumed. Members per Meeting_Minutes.md
  (2025-07-13): Yuxin, Chen Jie, Haoruo; commit emails
  include e1010693@u.nus.edu and
  caiyuxin001220@gmail.com. See "Team identity
  (targeted follow-up)" below for resolved identities.
- Repo: https://github.com/Yuxin916/ReasonX_SGTeam.
  Most-active branch `vlm_baseline` (52 commits ahead
  of the upstream CMU-VLA-Challenge fork point; tip
  commit `e7cbe57` "updated"). This is the branch
  analyzed below.
- It is a fork of the 2025 CMU-VLA-Challenge dev kit
  (Ubuntu 20.04, ROS Noetic, Unity sim - see the
  repo's own top-level README.md, which is the 2025
  challenge README, not a ReasonX-authored doc). Their
  submission lives entirely under
  `ai_module/src/vln_module`, replacing the dummy C++
  `dummy_vlm` package (kept for reference under
  `original_ai_module/`).
- `Meeting_Minutes.md` (2025-07-13) records a broader
  plan than what shipped: Chen Jie was to work on
  "object refer segmentation" and question whether to
  use ground-truth bounding boxes at all; Haoruo was to
  build a real-time occupancy map, a scene graph along
  the trajectory, and optionally YOLO + SAM2 detection;
  Yuxin was to train an end-to-end waypoint-prediction
  model. None of that made it into `vlm_baseline` - see
  "Branch comparison" below.

## Architecture

- Three separate processes, not one node graph: ROS
  (rosmaster + the `vln_module` package), a Flask "VLM
  server" (`vlm_server.py`, conda env `rosenv`, talks
  to Gemini), and a separate RoboRefer API server
  (`RoboRefer/API/api.py`, conda env `roborefer`, port
  25547). `launch.sh` boots all three in sequence,
  including a first-run RoboRefer env setup step.
- Only three ROS nodes are actually wired, per
  `ai_module/src/vln_module/launch/vln_agent.launch`:
  `challenge_agent_node.py` (the main loop),
  `pixel_to_waypoint_node.py`, and
  `navigation_watcher_node.py`. Everything else under
  `scripts/` - `box_visualizer_node.py`,
  `marker_annotater.py` (its projection math is
  duplicated inline in `challenge_agent_node.py`
  instead of being reused as a node),
  `relative_waypoint_converter.py`,
  `image_saver_node.py`, and the whole
  `scripts/vlm_wp_planner/` subfolder (`test_llm.py`,
  `visualize_pixel.py`, a duplicate
  `image_saver_node.py`) - is a debug/visualization aid
  or dev scratch file that is never launched.
- Core loop is a synchronous "reason-act" cycle: get
  the latest 360 panorama -> ask Gemini for one JSON
  action -> if `navigation`, crop the panorama to the
  chosen third and call RoboRefer for a pixel -> project
  that pixel to a 3D floor waypoint via the known camera
  model -> wait for `/waypoint_reached` -> repeat. There
  is no persistent map or scene graph; the only state
  carried between steps is the Gemini chat history plus
  a live dict of currently-visible object markers.

## Components

### Perception

- ReasonX owns no object detector. Object identity and
  counting are delegated entirely to the CMU dev kit's
  own `/object_markers` topic: a ROS `MarkerArray` of
  ground-truth-labeled `CUBE` markers within 2 m of the
  vehicle, broadcast at 5 Hz. This topic was an ALLOWED
  test-time input in the 2025 dev kit - confirmed by the
  repo's own top-level README.md, "System Outputs"
  table: "Ground-Truth Semantics | ... object labels and
  bounding boxes within 2m around the vehicle | 5Hz |
  `/object_markers`". The broadcaster is
  `system/unity/src/bbox_calculation/src/bboxBroadcast.cpp`,
  which reads a per-scene, pre-baked
  `object_list_with_labels.txt` (id, x, y, z, L, W, H,
  heading, quoted label) and republishes only the
  entries within `broadcastDisThre = 2.0` m of the
  vehicle's current (x, y).
- `challenge_agent_node.py`'s `object_markers_callback`
  just buckets incoming markers into
  `nearby_objects_data: Dict[class_name, List[Marker]]`
  - no filtering, dedup, or confidence logic.
- IMPORTANT cross-check: our own
  `docs/io_contract_crosscheck.md` states that ground-truth
  semantics are NOT provided in the 2026 dev kit ("a
  change from prior years"), flagged there as read from
  the 2026 dev-kit README rather than independently
  re-verified in this pass. If that holds, ReasonX's
  entire object-identification strategy has no
  equivalent input to key off in 2026 (see "Strengths
  and weaknesses" below).

### Reasoning LLM/VLM

- Gemini 2.5 Pro via the `google-genai` SDK
  (`vlm_planner.py`), one persistent chat session per
  mission (`self.chat = self.client.chats.create(...)`).
  Structured output is enforced with Pydantic
  (`VLMResponse`: `type` in `{navigation, end}`,
  `reasoning`, `image_division` in
  `{left, center, right}`, `subgoal_description`,
  optional `number`, optional `subgoal_states` list).
- A question-type classifier is a plain regex prefix
  match, duplicated verbatim in both
  `challenge_agent_node.py` and `vlm_planner.py`:
  `^(find|locate|point to|what is|which is|get|bring
  me)` -> object reference; `how many` -> numerical;
  else -> instruction-following. This is the same
  fragile prefix-heuristic pattern our own
  `docs/io_contract_crosscheck.md` warns against for the 2026 dummy
  model (real questions may not fit the prefixes).
- For instruction-following questions only, a second,
  separate one-shot Gemini 2.5 Pro call
  (`_initialize_subgoals`) decomposes the instruction
  into an ordered JSON list of subgoal strings before
  any navigation happens, converting path phrases like
  "take the path between X and Y" into "free space
  between X and Y" targets.
- A second consolidated call, `find_target_in_list`,
  asks Gemini whether the question's primary target
  (distinguishing it from secondary/context objects)
  appears in the live list of nearby marker class names,
  returning a Pydantic `FoundObjectResponse`
  (`is_present`, `target_name`, `reasoning`).

### Grounding / pixel-level ("exploration" and object
pointing)

- RoboRefer-8B-SFT (HF `Zhoues/RoboRefer-8B-SFT`,
  fetched by `download_ckpts.sh` via `git lfs`), served
  over its own Flask API. The `RoboReferClient` in
  `challenge_agent_node.py` sends an image crop plus a
  text description with a hard-coded suffix demanding
  the answer as `[(x1, y1)]` normalized coordinates, and
  denormalizes the first point returned.
- Depth Anything V2 ViT-Large
  (`depth_anything_v2_vitl.pth`, from HF
  `depth-anything/Depth-Anything-V2-Large`) is
  downloaded as a RoboRefer server dependency, but the
  actual client payload always sets `"enable_depth": 0`
  - so depth conditioning exists in the stack but is
  disabled at every call site read in this pass. This
  looks like a downloaded-but-unused (dead) capability,
  not a wired one.
- Navigation subgoals are always phrased as "Point to
  the free area near/at/between <subgoal text>" before
  being sent to RoboRefer - i.e. the grounding target is
  deliberately walkable free space, not the object
  footprint itself, so the returned pixel becomes a
  waypoint rather than an object mask/box.

### "Map" / scene representation

- None, beyond the live `nearby_objects_data` dict and
  Gemini's own chat history (which implicitly carries
  whatever spatial memory the model infers from prior
  turns' images). No occupancy grid, voxel map, or
  persistent scene graph is built at runtime - despite
  that being explicitly on Haoruo's task list in
  `Meeting_Minutes.md`; that work exists only on the
  unmerged `dev-occupancy-map` branch (see "Branch
  comparison").

### Output handling

- Numerical: parsed from Gemini's structured `number`
  field. If missing, a scripted reprompt offers Gemini
  two explicit JSON shapes (finalize with a number, or
  navigate for a better view); if that still fails, a
  regex + word-to-number (`zero`..`twenty`) text
  extraction over the reasoning string is the last-resort
  fallback (`_try_extract_number_from_text`).
- Object reference: at Gemini's `end`, `find_target_in_list`
  matches the question's primary target against the live
  marker-class names. Exactly one marker of that class ->
  publish it directly. Multiple markers of the same class
  -> RoboRefer is called again on the full question text
  (cropped to whichever division Gemini said contains the
  target), and the marker whose 3D position projects
  (via the same panoramic camera model, reimplemented
  inline from `marker_annotater.py`) closest in pixel
  space to RoboRefer's point is chosen.
- Instruction-following: Gemini tracks its own subgoal
  completion via the `subgoal_states` field it returns
  each turn; once every subgoal is marked complete, the
  code force-flips the response `type` to `"end"`
  client-side.
- Waypoint math
  (`pixel_to_waypoint_node.py`): raycasts from the
  panoramic camera model (1920x640 px, 360 deg HFOV,
  120 deg VFOV) through the robot's current pose
  (quaternion from `/state_estimation`) down to the
  ground plane (`z = 0`); if the ray points up or level
  it nudges the pixel row down by 10 px, and if that
  still fails it forces a small downward z-component and
  renormalizes - a hand-tuned patch for a real failure
  mode (looking at the horizon/ceiling produces an
  unusable ray).
- `navigation_watcher_node.py` declares a waypoint
  "reached" only once both linear speed < 0.05 m/s and
  angular speed < 0.1 rad/s hold continuously for 3.0 s
  - not a pure distance check. A `goal_zone_distance =
  0.5` m constant is defined but its actual distance
  comparison is commented out in the current code (dead
  threshold).

## Key numbers

- 2025 scores: sim prelim 31.42, final (real robot)
  34.58, 2nd of 4 teams (source: our own
  `docs/prior_art/README.md` leaderboard resolution, not
  re-derived here).
- Camera model: 1920x640 px, 360 deg HFOV, 120 deg VFOV.
  Each reason-act step sends 4 images to Gemini per
  call: the full panorama plus 3 independent 640x640
  left/center/right crops.
- `max_reprompts = 100` in `challenge_agent_node.py`
  before falling back to a scripted "point to the free
  area in the center" exploration action - a large
  ceiling relative to the 10-min/question budget; a
  stuck agent could in principle spend most of the
  budget reprompting before this trips.
- `/object_markers` broadcast radius: 2 m, at 5 Hz
  (`bboxBroadcast.cpp`, `broadcastDisThre = 2.0`,
  `broadcastRate = 5.0`).
- RoboRefer server: port 25547. VLM Flask server: port
  5000, `localhost` only (single-machine assumption in
  the client code, `vlm_client.py`).
- HTTP timeouts: 60 s for `start_mission` and
  `get_next_action`, 45 s for `find_target`
  (`vlm_client.py`); 20 s for the RoboRefer call
  (`challenge_agent_node.py`). No visible instrumentation
  anywhere in the code we read that tracks elapsed
  mission time against the 10-minute limit.

## Strengths and weaknesses for our setting

Strengths:

- End-to-end simplicity: a single JSON-schema contract
  with an off-the-shelf frontier VLM (Gemini 2.5 Pro),
  no training, no owned perception model, no map to
  build or debug. Fast to build and iterate, and
  consistent with the challenge's "no restriction on
  LLMs/VLMs/APIs" rule (per the repo's own README FAQ).
- RoboRefer gives sub-object, pixel-precise pointing
  ("free area near X") rather than reasoning only over
  bounding-box centroids, which should generalize better
  to cluttered/adjacent-object scenes than a
  centroid-only waypoint.
- Real engineering hardening against a flaky/format-
  drifting LLM: Pydantic-validated structured output,
  layered reprompt strategies (targeted reprompt ->
  generic fallback reprompt -> scripted exploration
  fallback), and a text-extraction safety net for
  numerical answers.

Weaknesses, specifically for our 2026 setting:

- The entire object-reference and numerical pipeline is
  built on `/object_markers` - an ALLOWED-IN-2025,
  ground-truth, 2 m-radius-limited semantic topic. Our
  own `docs/io_contract_crosscheck.md` states ground-truth
  semantics are NOT provided in the 2026 dev kit ("a
  change from prior years") - itself UNVERIFIED against
  the live 2026 dev-kit source in that doc. If that
  holds, `nearby_objects_data` would always be empty in
  2026 and `handle_object_reference_end` would loop into
  `reprompt_and_continue` indefinitely (up to
  `max_reprompts = 100`) with no path to success. This is
  the single biggest portability risk in this design, and
  the one fact most worth re-verifying against the actual
  2026 dev kit before drawing further conclusions from
  this repo.
- No persistent spatial memory: each Gemini call sees
  only the current 4 images plus chat history. Avoiding
  loops or recalling "the room I saw 3 minutes ago"
  depends entirely on the LLM's own (unverified) spatial
  reasoning over a growing image-heavy chat context, not
  on any structured map.
- Depth Anything V2 is downloaded and wired into the
  RoboRefer server but disabled at every call site
  (`enable_depth: 0`) - a whole model is dead weight at
  inference time. Unclear whether this was a deliberate
  simplification (time pressure, or depth hurt/didn't
  help results) or an oversight; either way it means any
  "depth-aware grounding" claim in the toolset doesn't
  actually apply at runtime.
- Two network hops per reason-act step (Gemini API +
  local RoboRefer server), each with a multi-second-to-
  minute timeout budget, against a 10-minute/question
  wall clock, with no time-budget check visible in the
  code. A single slow Gemini call could consume a
  meaningful fraction of the budget unnoticed.
- Division-based navigation (independent left/center/right
  thirds of the panorama, RoboRefer called on one
  640x640 crop at a time) discards full-panorama context
  at the exact moment a pixel is chosen, and structurally
  cannot point at something straddling a division
  boundary.
- The question-type classifier is a plain regex prefix
  match duplicated in two files - the same fragile
  pattern our own `docs/io_contract_crosscheck.md` flags as
  unreliable for 2026 (real questions, e.g. the
  `japanese_room` object-reference questions in that
  doc's example, don't reliably start with the expected
  prefix words).

## Team identity (targeted follow-up, 2026-07-11)

Resolved by a dedicated identity pass (full detail and
negative-result checklist in the task-local research notes).

- Team lead: **Cai Yuxin** (Yuxin916), PhD student at the
  AutoMan lab, NTU Singapore (advisor Prof. Chen Lv;
  co-supervised by Dr. Wei-Yun Yau, A*STAR I2R). Her homepage
  states directly: "Our team ReasonX won the 2nd Place in CMU
  Vision-Language Autonomy Challenge and presented our work at
  IROS 2025" - an independent corroboration of the leaderboard.
  Source: https://yuxin916.github.io/
- "Haoruo" is plausibly **Zhang Haoruo**, NTU AutoMan PhD
  student (same lab/advisor; https://ezhanghz.github.io/ ) -
  lab-roster match, not self-reported; treat as probable.
- "Chen Jie" is plausibly **Jie Chen**, PhD student, NUS MARMoT
  Lab (advisor Guillaume Sartoretti; co-author with Cai on
  ImagiNav 2026) - probable, not certain. J1dan (the second
  committer, e1010693@u.nus.edu) remains unidentified.
- No talk recording and no dedicated ReasonX paper exist
  (checked YouTube, workshop site, arXiv, Cai's full Scholar
  list). Closest methodological sibling: CL-CoTNav
  (arXiv 2504.09000, same lab, zero-shot object-goal nav with
  hierarchical VLM chain-of-thought) - NOT confirmed as the
  ReasonX method paper.

## Takeaways for our 2026 module

- Do not port the `/object_markers` reliance as-is.
  Whatever we build for object identification (numerical
  and object-reference) must come from our own
  perception, since GT semantics are stated as removed
  for 2026 - but verify this claim against the actual
  cloned 2026 dev-kit source before finalizing our
  perception design; ReasonX's failure mode if the
  assumption is wrong either way is instructive for how
  fragile a GT-topic dependency is.
- The pattern "reason over a few image crops -> point
  with a referring-expression grounding model -> project
  the pixel to a floor waypoint" is worth keeping as a
  component even without GT markers, provided it is paired
  with our own detector/segmenter to produce the candidate
  object list that RoboRefer currently gets for free from
  the dev kit's marker topic.
- Reusable, provider-agnostic engineering patterns worth
  copying regardless of our perception choice: Pydantic-
  validated structured LLM output; a numeric-answer
  reprompt-then-text-extraction fallback chain; and a
  "stationary for N continuous seconds" waypoint-reached
  check rather than a pure distance threshold (which
  ReasonX's own commented-out `goal_zone_distance` shows
  they moved away from).
- If we use any referring-expression/grounding model with
  an optional depth-conditioning path, decide and log
  whether it is enabled and measure its effect - do not
  silently ship a disabled capability the way this repo
  did with Depth Anything V2.
- Given the 10-min/question budget and the complete
  absence of time instrumentation in this codebase, our
  module should have an explicit clock/deadline check;
  this is a concrete gap in ReasonX's implementation we
  should not repeat.
- The gap between `Meeting_Minutes.md`'s stated ambitions
  (scene graph, occupancy map, YOLO/SAM2, a trained
  end-to-end waypoint model) and what actually shipped in
  `vlm_baseline` (GT markers + an off-the-shelf VLM + an
  off-the-shelf grounder) is a useful scoping signal for
  our own Phase 2: a small, tightly-integrated, mostly
  off-the-shelf pipeline reached 2nd place, while more
  ambitious from-scratch perception/mapping work (visible
  on their unmerged branches) did not make it into the
  submitted system in time.

## Sources

- Repo (cloned and read directly, not from a hosted
  view): https://github.com/Yuxin916/ReasonX_SGTeam,
  branch `vlm_baseline`, tip commit `e7cbe57` "updated".
- Top-level `README.md` (2025 challenge rules; "System
  Outputs" table including the GT `/object_markers` row;
  evaluation/timing/scoring sections; FAQ on LLM/API
  usage).
- `ai_module/src/vln_module/README.md` (ReasonX-authored
  setup guide, architecture summary, model paths).
- `ai_module/src/vln_module/scripts/challenge_agent_node.py`
- `ai_module/src/vln_module/scripts/pixel_to_waypoint_node.py`
- `ai_module/src/vln_module/scripts/navigation_watcher_node.py`
- `ai_module/src/vln_module/scripts/relative_waypoint_converter.py`
  (present but unwired - not in `vln_agent.launch`)
- `ai_module/src/vln_module/scripts/box_visualizer_node.py`
  (present but unwired)
- `ai_module/src/vln_module/src/vln_module/utils/vlm_planner.py`
- `ai_module/src/vln_module/src/vln_module/utils/vlm_client.py`
- `ai_module/src/vln_module/src/vln_module/utils/vlm_server.py`
- `ai_module/src/vln_module/src/vln_module/utils/vln_data_interface.py`
- `ai_module/src/vln_module/src/vln_module/utils/marker_annotater.py`
  (unwired as a standalone node; its projection math is
  reimplemented inline in `challenge_agent_node.py`)
- `ai_module/src/vln_module/launch/vln_agent.launch`
  (defines exactly which 3 nodes run)
- `launch.sh`, `download_ckpts.sh`, `.gitmodules`
  (RoboRefer submodule at
  https://github.com/Zhoues/RoboRefer.git; checkpoint
  sources HF `Zhoues/RoboRefer-8B-SFT` and HF
  `depth-anything/Depth-Anything-V2-Large`)
- `system/unity/src/bbox_calculation/src/bboxBroadcast.cpp`
  (source of `/object_markers`: reads a per-scene GT
  `object_list_with_labels.txt`, broadcasts markers
  within 2 m of the vehicle at 5 Hz)
- `Meeting_Minutes.md` (2025-07-13, team roles and
  planned-vs-shipped scope)
- Branch comparison (`git diff --stat` between
  `origin/vlm_baseline` and each of `origin/dev`,
  `origin/dev-object-detection`,
  `origin/dev-occupancy-map`, `origin/main`, run
  2026-07-11 against the shallow clone): confirms the
  object-detection and occupancy-map branches carry
  thousands of lines never merged into the shipped
  branch.
- Our own repo, for the 2025 leaderboard and the 2026
  I/O-contract cross-check:
  docs/prior_art/README.md and docs/io_contract_crosscheck.md (both in
  this repo).
