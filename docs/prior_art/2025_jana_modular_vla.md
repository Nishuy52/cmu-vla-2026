> Imported 2026-07-11 from a parallel research stream. See
> `docs/prior_art/README.md` for provenance and conflict rules.

# Jana et al. modular VLA framework (2025 entry) -
a published-paper entry whose code is a relabeled fork
of a 2024 team's submission

Compiled 2026-07-11. This is a participating 2025 CMU VLA
Challenge entry, NOT one of the four confirmed 2025
finalists (NROS 44.26, ReasonX 34.58, CopyPasta 30.98,
URL-KAIST 22.80 - see `docs/prior_art/README.md` section 1). Its
significance is that it is the only 2025-era entry with a
public, citable paper, so it is a useful window into what a
real (non-winning) submission looks like end to end.

## What it is

Paper: Anindya Jana, Snehasis Banerjee, Arup Sadhu, Ranjan
Dasgupta, "A Modular Vision-Language-Action Robotics
Framework for Indoor Environments," arXiv:2606.31144v1
[cs.RO], posted 30 Jun 2026.
Source: https://arxiv.org/html/2606.31144v1
Abstract page (venue/date metadata):
https://arxiv.org/abs/2606.31144

The arXiv "Comments" field gives the venue: "IEEE IROS 2025
Workshop on Generative AI for Robotics and Smart
Manufacturing." No author institutional affiliation is
stated anywhere in the paper body. The only acknowledgment
is a generic thanks to the challenge organizers - no lab or
company name. Note the timeline: the described system
competed in 2025 (workshop presentation implied by the
venue string) but the arXiv write-up was not posted until
mid-2026, about 9 months after the workshop.

Code repo: https://github.com/anindya-jana/CMU-VLA-Challenge
- itself a fork of the official dev kit
(HaochenZ11/CMU-VLA-Challenge, confirmed via `gh repo view
--json isFork,parent`). The most active branch is
`ai-moudle-update-fast-navigation` (sic - typo in the branch
name), 21 commits ahead of the shared history with `main`.
That branch was cloned and read directly for this dossier;
`main` does not contain the working `ai_module` code at all
(`run.py` etc. only exist on the two dev branches).

## Architecture

Paper Figure 3 (per headless-fetch extraction of the HTML) shows
two parallel pipelines feeding a shared semantic voxel map,
gated by a query classifier:

1. Autonomous exploration - full-coverage waypoint sweep
   over a 2D grid built from the traversable-area point
   cloud.
2. Semantic mapping - listens to an "upstream object
   detection node" and folds each detection's embedding,
   3D position, and scale into a semantic voxel map, time-
   boxed to 500 seconds (falls back to a partial map on
   timeout).
3. Query processing - an LLM (Gemini 2.0 Flash) classifies
   the incoming question into numerical / object-reference
   / instruction-following, then a per-type prompt grounds
   the question in the map.
4. Action generation - the LLM's answer is parsed into the
   type-specific output (integer, RViz marker, or a plan of
   Goto/Between/Avoidbetween/StopAt primitives that is
   executed against the grid).

The code's actual control flow (`ai_module/src/dummy_vlm/
src/run.py`) is a four-state machine: `INITIALIZE ->
MAPPING -> ASK_QUESTION -> ANSWERING`, looping back to
`ASK_QUESTION` after each answer. Mapping is a one-shot,
pre-question, full-coverage sweep - there is no re-
exploration once questions start, matching the paper's
description but diverging from strategies (e.g. CopyPasta,
url-kaist per `docs/prior_art/README.md`) that explore
incrementally per question.

## Components

### Perception (what actually produces semantics)

This is the single most important thing to get right about
this entry, so it gets its own subsection below ("GT-marker
verdict"). Short version: object semantics come from the
simulator's `/object_markers` topic (a `MarkerArray` the
sim publishes with ground-truth object poses, scales, and
class names in the `ns` field), not from any image-based
detector. OwlViT (`google/owlvit-base-patch32`) is loaded
and used, but only as a **text encoder**
(`OwlViTModel.get_text_features`) over each marker's class-
name string - never as an object detector over camera
pixels.
File: `map/buildVoxel.py` (`SemanticVoxelMap.marker_callback`,
`process_marker`, `compute_clip_embeddings`).

### Map

A `VoxelizedPointcloud` (voxel size 0.15 m, mean feature
pooling) built on `torch_geometric`'s voxel_grid /
consecutive_cluster / scatter primitives. Each GT marker
becomes one point with a CLIP-space text-embedding feature,
a 3D position, and a scale. Localization at query time
(`voxel_map/voxel_map_localizer.py`,
`VoxelMapLocalizer.localize_AonB`) embeds the query noun(s),
computes cosine alignment against every stored point, and
for "A near/on B" queries takes the closest pair of top-k
matches by Euclidean distance. A separate 2D occupancy grid
(`map/gridMap.py`, 0.25 m cells) built from
`/traversable_area` is used for waypoint planning and for
converting localized 3D points into grid coordinates fed to
the LLM prompt.

### Reasoning LLM

Only Gemini is wired into the live pipeline
(`vla/queryVLM.py`): `gemini-2.0-flash` via the raw
`generativelanguage.googleapis.com` REST endpoint, with a
second hardcoded Gemini API key tried if the first request
fails. There is no DeepSeek (or any other) fallback in the
code path actually invoked by `run.py` /
`handleQuestions.py` - see discrepancies below. Both
`queryVLM.py` and an unused experimental file
(`src/temp/a.py`) contain live-looking, hardcoded API keys
committed to a public repository; this dossier does not
reproduce the literal key strings.

Three hand-written prompt templates
(`vla/queryGenerator.py`) do the real reasoning work, each
with a one-shot chain-of-thought example:
- Numerical: reason over grid-coordinate overlap between
  the target object(s) and the anchor object, then "Answer:
  <int>".
- Object reference: reason over candidate grid coordinates
  for all matches of the two nouns, then "Answer: (x,y)".
- Instruction-following: decompose the instruction into an
  ordered list drawn from
  `Goto(x,y) / Between((x1,y1),(x2,y2)) /
  Avoidbetween((x1,y1),(x2,y2)) / StopAt(x,y)`.

Query classification (`vla/handleQuestions.py`,
`process_question`) is itself a fourth, separate LLM call
with its own prompt reusing the challenge's own example
questions almost verbatim.

### Exploration

Full-coverage strategy (`map/planner.py`): sample candidate
viewpoints spaced by a minimum gap (tuned to 1.25 m, see
Key numbers) that are far enough from obstacles (distance-
transform threshold), connect nearby candidates with A*-
style shortest-path edges, solve an approximate TSP over
that graph (`networkx.approximation
.traveling_salesman_problem`) to order the tour, then reduce
the A* cell path to turn points only. Falls back to a home
waypoint if no candidate points exist, and falls back to
node order if TSP fails - both defensive fallbacks read as
post-hoc robustness patches rather than original design.

### Output handling

`vla/postProcess.py` parses the LLM's free-text answer per
type:
- Numerical: regex-free, just takes the text after the
  first line starting with `Answer:` and returns it as a
  **string** - it is never published to any ROS topic in
  this codebase (confirmed: no `Int32` publisher anywhere
  under `ai_module/`, via `grep -rn "Int32"`). The answer
  only reaches a local Flask/Socket.IO web page
  (`run.py`'s `answering()` emits it over `socketio`).
- Object reference: regex-parses `Answer: (x,y)`, maps the
  grid coordinate back to the stored global point/scale,
  and publishes a `visualization_msgs/Marker` on
  `/selected_object_marker` for 5 seconds
  (`vla/publisher.py`, `publish_object_marker`).
- Instruction-following: parses the ordered primitive list,
  mutates the traversability grid for `Avoidbetween`, runs
  a Theta*-flavored A* between successive waypoints, and
  streams `geometry_msgs/Pose2D` waypoints on
  `/way_point_with_heading` closed-loop against
  `/state_estimation` (`map/wpNav.py`,
  `WaypointController`).

## Key numbers

All from the paper (headless-fetch extraction of
https://arxiv.org/html/2606.31144v1, section citations as
given by the tool):
- Exploration/mapping time budget: 500 seconds hard limit;
  proceeds with a partial voxel map on timeout (Section
  VI-B). Matches the code's `timeout_seconds = 500` in
  `run.py`'s `create_voxel_map`.
- Exploration tuning (Section VI-E): initial trials took
  about 8 minutes 42 seconds; after tuning, about 4 minutes
  17 seconds - "a 50% reduction." Two of the tuning knobs
  named: waypoint spacing widened from 0.6 m to 1.25 m, and
  arrival threshold loosened from 0.65 m to 1.0 m. (In the
  code read here, `Planner.min_gap = 1.25` matches the
  tuned spacing value; the arrival-threshold constant lives
  in `WaypointController.__init__(threshold=1.0)`, also
  matching the tuned value - the untuned 0.6/0.65 values are
  not present in this snapshot, consistent with the paper
  describing an already-applied optimization.)
- Voxel size increased to 0.15 m for lower resolution /
  lower compute (Section VI-E2) - matches
  `VoxelizedPointcloud(voxel_size=0.15)` in
  `map/buildVoxel.py`.
- Example outputs (Section VII-B, no scores/ranking
  reported anywhere in the paper): numerical "how many
  chairs are there" -> "12"; object reference "Find the
  potted plant on the file cabinet" -> coordinates
  [1.15, 1.26, 1.33].
- Reported limitations (Section VIII): reliance on online
  VLM APIs (latency), and an assumption of a static
  environment.

No accuracy, precision, or leaderboard numbers are reported
anywhere in the paper - it is a system-description paper
with worked examples, not an evaluation paper.

## Paper vs code discrepancies

1. **Perception source.** The paper's Section VI-B
   describes an "upstream object detection node" processing
   "real-time camera feed[s]," which reads as an image-based
   open-vocabulary detector. The code instead subscribes
   directly to the simulator's ground-truth
   `/object_markers` `MarkerArray` and uses OwlViT purely as
   a **text** encoder over each marker's `ns` (class-name)
   field - no image ever reaches OwlViT's vision tower in
   this pipeline. See "GT-marker verdict" below for full
   reasoning and the direct code citation.
2. **LLM fallback.** The paper states "Deepseek v3 is
   employed as a fallback API in case of failures with the
   primary service." The actual pipeline invoked from
   `run.py` (`vla/queryVLM.py`) has no DeepSeek call at all
   - its only fallback is a second, hardcoded Gemini API
   key. DeepSeek (`deepseek/deepseek-chat-v3.1:free`) only
   appears in an unrelated, unused experimental file,
   `src/temp/a.py` (and its near-duplicate `temp/a copy.py`),
   which fan out to three OpenRouter models
   (DeepSeek, Kimi-K2, Qwen3) as a side test and is never
   imported by anything the state machine calls.
3. **Numerical output channel.** Nothing in the paper claims
   an Int32 ROS publish, but this matters against the actual
   challenge contract: this codebase never publishes a
   numerical answer to any ROS topic - it only shows the
   text in a local web UI. Whatever the current dev kit
   expects (our own `docs/io_contract_crosscheck.md` records
   `/numerical_response` `std_msgs/Int32` for the 2026 dev
   kit), this entry's numerical path would not satisfy it as
   written.
4. **Venue/affiliation.** The paper body itself states no
   institutional affiliation; the venue is recoverable only
   from the arXiv "Comments" metadata field, not the visible
   text headless fetch extracted from the HTML body.

## GT-marker verdict (the check this dossier was asked to make)

**Confirmed: this 2025 entry uses ground-truth object
markers, not a real detector on images - it is a direct
code descendant of the 2024 Anand Singh entry, and the
paper's "upstream object detection node" framing overstates
what the code does.**

Evidence, in order of strength:

- The Jana repo (`ai-moudle-update-fast-navigation` branch)
  and Anand Singh's 2024 repo
  (https://github.com/AnandSingh-0619/CMU-VLA-Challenge,
  branch `anand_dev`) have **byte-identical** files at
  `ai_module/src/dummy_vlm/src/map/gridMap.py`,
  `vla/handleQuestions.py`, `vla/postProcess.py`,
  `vla/publisher.py`, `vla/queryGenerator.py`, and
  `voxel_map/voxel.py` (confirmed with `diff`, 0 differing
  lines on each). The remaining files
  (`run.py`, `map/buildVoxel.py`, `map/planner.py`,
  `map/wpNav.py`, `vla/queryVLM.py`,
  `voxel_map/voxel_map_localizer.py`) differ only by small,
  explainable edits: swapping the default LLM backend from
  `model_name="mistral"` to `"gemini"`, adding a 500-second
  navigation timeout and an `VLA_SKIP_VOXEL` debug escape
  hatch, and adding an offline/local-cache fallback path for
  loading OwlViT.
- Anand's original `run.py` opens with the comment header
  `# Author: Anand Mohan Singh` describing exactly this
  state machine; that header is absent from the Jana repo's
  otherwise near-identical `run.py`, but the diff shows
  nothing else at the top of the file changed besides that
  comment removal.
- Both `SemanticVoxelMap.marker_callback` (map/buildVoxel.py)
  implementations subscribe to `/object_markers` and read
  `marker.ns` as the object's class name, feeding only that
  string into OwlViT's `get_text_features` - in both repos.
  Neither repo's perception code ever touches
  `sensor_msgs/Image` or any camera topic. GitHub's own fork
  graph shows both repos as independent forks of the shared
  upstream dev kit (`HaochenZ11/CMU-VLA-Challenge`), not of
  each other - so this is not a GitHub "fork" relationship,
  but the file-level evidence above leaves little doubt the
  `ai_module` code was copied from the 2024 submission into
  the 2025 one.
- This matches the independent finding already on file in
  `docs/prior_art/README.md` that CopyPasta (2025, 3rd place) also
  used "the PROVIDED ground-truth object markers in sim" for
  two of its three question types - i.e. GT markers were a
  known, exploitable shortcut available to 2025 teams, and
  at least two entries (this one and its 2024 ancestor) built
  their entire semantic layer on that shortcut rather than
  on any learned detector.

Caveat: this verdict describes the **2025 fork's specific
implementation**, not a claim about what the underlying
`/object_markers` topic represents in the current 2026 dev
kit. Our own `docs/io_contract_crosscheck.md` (verified against the
2026 dev kit clone) states ground-truth semantics are "NOT
provided this year (a change from prior years)" and does not
list `/object_markers` or `/traversable_area` among the
legal 2026 input topics at all. If that holds, this entire
class of solution (GT-marker-driven semantics, one-shot
`/traversable_area`-based exploration) is architecturally
inapplicable to 2026 regardless of the paper/code gap - it
was already riding a shortcut specific to earlier years.

## Takeaways for our 2026 module

- Do not copy this entry's perception approach wholesale:
  even setting aside the paper/code gap, it depends on two
  topics (`/object_markers`, `/traversable_area`) that our
  own `docs/io_contract_crosscheck.md` says are not legal 2026 inputs.
  Our perception layer has to do real open-vocabulary
  detection over `/camera/image` and build traversability
  from `/registered_scan` / `/terrain_map`, not assume a
  ground-truth object feed.
- The "OwlViT as text encoder over class names" pattern is a
  toy shortcut, not evidence OwlViT-family detectors don't
  work for this task - it says nothing about OwlViT's actual
  open-vocab image-detection quality, which we would need to
  evaluate separately if we wanted to use it as a real
  detector.
- The output side is a more useful reference: publishing an
  RViz `Marker` for object-reference and streaming
  `Pose2D` waypoints closed-loop against `/state_estimation`
  for instruction-following both match our own
  `docs/io_contract_crosscheck.md` contract's shape (topic names/types
  are 2026-specific and would need updating, but the
  mechanics - marker-with-scale-as-bbox, closed-loop
  waypoint streaming with a reach-distance threshold - are
  reusable design patterns). The numerical path is the one
  piece not even wired to a ROS topic here and should not be
  copied at all.
- The prompt templates (`vla/queryGenerator.py`) are a
  decent starting point for one-shot chain-of-thought
  prompts per question type, and are worth reading directly
  even though the geometry they reason over ("grid
  coordinates," bounding corners) assumes the same GT-marker
  scale/position data we cannot assume we'll have.
  https://github.com/anindya-jana/CMU-VLA-Challenge/blob/ai-moudle-update-fast-navigation/ai_module/src/dummy_vlm/src/vla/queryGenerator.py
- Full-coverage exploration via TSP-ordered, distance-
  transform-filtered viewpoint sampling
  (`map/planner.py`) is architecture-agnostic and could
  inform our own exploration strategy regardless of the
  perception-source issue - just note it depends on having a
  traversability grid, which we will have to build ourselves
  from live scan data rather than from `/traversable_area`.
- Treat the paper's specific numeric claims (500 s budget,
  8:42 -> 4:17 exploration tuning, 0.15 m voxels) as evidence
  of realistic engineering constraints worth planning around
  (our own time budget is 10 minutes/question, so a multi-
  minute one-shot exploration phase is a real tradeoff to
  budget for), not as endorsed target numbers for a
  materially different sensing setup.
- Hardcoded live API keys committed to a public repo (in
  both `vla/queryVLM.py` and the unused `temp/a.py`) is a
  concrete reminder to keep credentials out of any code we
  publish or fork from.

## Sources

- Paper (HTML, primary source for architecture/components/
  key numbers): https://arxiv.org/html/2606.31144v1
- Paper abstract page (venue metadata, submission date):
  https://arxiv.org/abs/2606.31144
- Code repo (default branch):
  https://github.com/anindya-jana/CMU-VLA-Challenge
- Code repo, most-active branch (cloned and read directly
  for this dossier):
  https://github.com/anindya-jana/CMU-VLA-Challenge/tree/ai-moudle-update-fast-navigation
- 2024 ancestor codebase (cloned and diffed directly for
  the GT-marker verdict):
  https://github.com/AnandSingh-0619/CMU-VLA-Challenge/tree/anand_dev
- Shared upstream dev kit (both repos are forks of this):
  https://github.com/HaochenZ11/CMU-VLA-Challenge
- This repo's own verified 2026 I/O contract, used for the
  "not a legal 2026 input" and Int32-output claims:
  `docs/io_contract_crosscheck.md`
- This repo's prior-art survey, used for the 2025 finalist
  scores and the CopyPasta GT-marker cross-check:
  `docs/prior_art/README.md`
