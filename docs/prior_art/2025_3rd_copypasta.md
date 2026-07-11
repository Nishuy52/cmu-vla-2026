> Imported 2026-07-11 from a parallel research stream. See
> `docs/prior_art/README.md` for provenance and conflict rules.

# CopyPasta (3rd place, 2025) - Gemini-in-a-ROS-state-machine,
leaning hard on the sim's ground-truth object markers for two of
three question types, with a live-panorama Reason-Act loop for the
third.

Compiled 2026-07-11 from: the team's IROS 2025 talk (manual
captions, read in full), the actual submitted `ai_module` source
on GitHub (read directly, not just skimmed), the MRSD newsletter
blurb, and a team member's site. Every claim below is cited; items
that could not be confirmed are marked "unverified."

## What it is

Team CopyPasta - four CMU MRSD (Class of 2026) students,
Sreeharsha Paruchuri, Ishita Gupta, Parth Singh, and Daksh Adhar -
placed 3rd at the 2025 CMU VLA Challenge (final score 30.98, sim
prelim 22.99; see `docs/prior_art/README.md` for the full leaderboard).
Source: https://labs.ri.cmu.edu/mrsd-news/articles/ ,
https://sreeharshaparuchur1.github.io/

Their system routes each incoming question through a Gemini-based
classifier, then to one of three dedicated per-type solvers, all
implemented as Python ROS nodes inside `ai_module/src`. For
Numerical and Object-Reference questions they explore the scene
just enough to harvest the simulator's ground-truth
`/object_markers`, then hand the harvested markers plus the raw
question to an LLM with prompts that embed explicit spatial-math
formulas. For Instruction-Following they do not explore first (no
GT markers, no map) and instead run a live Reason-Act loop over
the streaming 360 panorama, asking Gemini 2.5 Pro to point at a
region of the current frame each step and converting that pixel
into a waypoint via camera-ray geometry.

The real method code is public - this supersedes the "only saw
gemini env scripts, didn't go deep" note in earlier prior-art
research. Repo:
https://github.com/parths5/CMU-VLA-Challenge , default branch
`team-copypasta-submission`
(https://github.com/parths5/CMU-VLA-Challenge/tree/team-copypasta-submission),
repo description "Achieved 3rd Place at IROS 2025."

## Architecture

As presented in the talk (four conceptual stages, all "reasoning
tasks powered by Gemini 2.5 Pro" per the speaker): **Input
Handler** (Gemini parses the query into goal objects + constraints
and routes by question type) -> **Explorer** (scene exploration
for T1/T2; immediate real-time nav for T3) -> **Solver**
(dedicated per-type reasoning agent) -> **Verifier** (a
closed-loop check meant to cut hallucination). Talk:
https://www.youtube.com/watch?v=kAPltAaRPk4

As actually implemented in code, it is a leaner 5-state ROS state
machine in `main_launcher.py`: `INITIALIZE` ->
`QUESTION_TYPE_DECISION` -> `EXPLORING` (only for
Numerical/Object-Reference) or straight to `ANSWERING_QUESTION`
(for Instruction-Following, skipping exploration) -> `SHUTDOWN`.
Source:
https://github.com/parths5/CMU-VLA-Challenge/blob/team-copypasta-submission/ai_module/src/main_launcher.py

Two discrepancies worth flagging explicitly:

- **No standalone "Verifier" file or class was found** by
  searching the repo (`gh api search/code?q=verif` returned
  nothing in-repo). The diagram's verifier box appears to
  correspond to ad hoc checks inlined in the solvers - e.g. T3's
  reprompt-and-retry loop (`max_reprompts = 100`) and
  `/waypoint_reached` gating - rather than a discrete module.
  Unverified: whether a separate verifier existed in an earlier,
  unpublished code state.
- **Model usage is not uniformly "2.5 Pro at every stage"** as the
  talk states. See Components > Reasoning LLM below for the
  code-level breakdown, which shows flash models doing most of the
  work and 2.5 Pro reserved for T3.

A second, unused launcher variant, `main_launcher_t3.py`, exists in
the repo: a "T3 Enhanced" state machine that runs exploration and
GT-marker collection for *all* three question types, including
Instruction-Following. The actual `roslaunch` file wires in
`main_launcher.py`, not the T3-enhanced one -
confirmed by decoding
`ai_module/src/launch/dummy_vlm.launch`, which invokes
`type="main_launcher.py"`. So the system that was actually scored
is the one where T3 gets no GT markers and no pre-exploration,
matching the talk's claim. Source (launch file, base64-decoded via
`gh api`):
https://github.com/parths5/CMU-VLA-Challenge/blob/team-copypasta-submission/ai_module/src/launch/dummy_vlm.launch

All `ai_module` code landed in one squashed commit, `feac7bd0`
("final sub - docker might need a small fix."), dated
2025-09-16 - one day after the 2025-09-15 submission deadline noted
in `docs/prior_art/README.md`. There is no earlier, granular commit
history for the solver logic itself (only README/infra commits
precede it), so the team's iterative development process cannot be
reconstructed from git history - only the final artifact is
visible. Source:
https://github.com/parths5/CMU-VLA-Challenge/commit/feac7bd0

## Components

**Perception.** For T1/T2, perception is entirely the simulator's
provided `/object_markers` (ground-truth 3D bounding boxes with
label namespaces) - no learned detector runs at all. The talk
states color is recovered by projecting each marker's 3D centroid
onto the 360 image and sampling a pixel patch, since the markers
carry no color attribute themselves. In the T1/T2 solver code
actually read, however, the color values passed into the LLM
prompt come directly from the ROS `Marker.color.r/g/b` field (the
marker's own render color), not from a visibly separate
pixel-sampling step. **Unverified**: whether the pixel-projection
step happens upstream inside `map/object_marker_collector.py`
(only partially read here) or whether the marker's own render
color is being reused as a proxy for real-world object color -
worth a closer read of that file before relying on this technique.
For T3, perception is the live 360 panorama only (no markers, no
map), consumed frame-by-frame by the VLM planner.

**Map / scene representation.** T1/T2 keep no persistent scene
graph - each solve call gets the full flat list of collected
markers as JSON and reasons over raw coordinates plus
prompt-embedded geometry rules (see Reasoning LLM below). For
exploration coverage they build a 0.4 m grid from
`/traversable_area`, with a 0.15 m safety buffer via binary
dilation and 0.2 m minimum clearance from non-traversable cells.
T3 keeps no explicit map either - the talk describes "iteratively
building a semantic scene graph" while exploring for
Instruction-Following, but the wired-up code path
(`main_launcher.py` + `t3_solver.py`) does not explore first at
all; it reasons directly over the live panorama each step, with no
persistent structure carried between reason-act steps beyond
`subgoal_states`. **Unverified**: how or whether the described
semantic scene graph is actually built and used, given the deployed
launcher skips exploration for T3.

**Reasoning LLM - actual model usage per stage (code-level,**
**not talk-level):**

| Stage | Model configured in code |
|---|---|
| Question-type classifier | `gemini-1.5-flash` (early version) -> `gemini-2.0-flash-001` via raw REST call, temperature 0 |
| T1 numerical solver | `gemini-2.0-flash-001` via REST |
| T2 object-reference solver | `gemini-2.0-flash-001` via REST |
| T3 VLM planner | `gemini-2.5-pro` (primary), `gemini-1.5-flash` (secondary/lightweight) |

Sources (all in `ai_module/src`):
`question_parser/question_parser.py`,
`t1_numerical_solver/t1_solver.py`,
`t2_selected_object_marker_solver/t2_solver.py`,
`t3_instruction_following_solver/vlm_planner.py` - all on
https://github.com/parths5/CMU-VLA-Challenge/tree/team-copypasta-submission/ai_module/src

This contradicts the talk's framing that "all the reasoning tasks
are powered by Gemini 2.5 Pro" - in the submitted code, only T3
actually defaults to 2.5 Pro; classification, counting, and
object-reference selection all ran on flash-tier models. Plausible
reading (unverified): flash models were a late swap for latency
and Gemini API rate-limit reasons - the talk separately mentions
"we have to be wary of the Google Gemini API rate limit" for T3,
suggesting rate-limit pressure was a live concern across the team.

Both T1 and T2 prompts embed near-identical explicit mathematical
definitions of spatial-relation words, matching the talk's claim
almost verbatim. From the T1 prompt (numerical solver): 2D
Euclidean distance `d_xy(A,B) = sqrt((Ax-Bx)^2 + (Ay-By)^2)`;
ABOVE/BELOW via a z-threshold; ON via a z-band plus an xy
proximity cap; NEAR via an adaptive threshold derived from the
median bounding-box diagonal of all scene objects (clamped to
[0.4 m, 1.2 m], scaled by 0.8); CLOSEST/FARTHEST via min/max
`d_xy`; BETWEEN via an axis-aligned bounding rectangle over two
reference points with a small margin. The T1 prompt also
explicitly instructs the model to treat an unverifiable attribute
(e.g. a color word absent from any marker's namespace) as a hard
filter, driving the count to zero rather than guessing. Source:
`ai_module/src/t1_numerical_solver/t1_solver.py`. The T2 prompt
(`t2_selected_object_marker_solver/prompts/obj_reference_prompt.py`)
uses the same formula set inside a six-phase protocol (parse ->
filter candidates -> compute spatial metrics -> weighted
constraint-satisfaction scoring, 40% object-type match / 50%
spatial constraints / 10% attributes -> verify uniqueness ->
report), with an explicit tie-break order (score, then distance to
the primary reference, then exact type match, then lowest marker
ID).

**Exploration.** `navigator/planner.py` implements full-coverage
grid exploration: candidate viewpoints spaced by a minimum 0.8 m
gap on the 0.4 m grid, A* pathfinding between them (Manhattan
heuristic, 4-connected neighbors only - no diagonal moves), and a
TSP approximation to order the visiting sequence (constants
observed in code: `TSP_THRESHOLD_DISTANCE=12.0`,
`MAX_WAYPOINTS=95`, `MIN_DISTANCE_BETWEEN_WAYPOINTS=1.5`,
`LOOP_DETECTION_MULTIPLIER=1.5` to bias against revisiting loops).
This matches the talk's description of an A* planner "with
heuristics to minimize loops" and "dynamic timeouts to reach a
waypoint," tuned to finish exploration under the 4-minute budget,
with full coverage plans generated for 14 of 17 training scenes
(timing out on larger office-style scenes and tight corners, per
the team's own account).

**Output handling.** All three question types publish on exactly
the ROS topics specified by the dev-kit contract: T1 ->
`std_msgs/Int32` on `/numerical_response`; T2 ->
`visualization_msgs/Marker` on `/selected_object_marker` (one of
the collected GT markers, selected by the LLM); T3 ->
`geometry_msgs/Pose2D` on `/way_point_with_heading`, emitted one
waypoint at a time inside the Reason-Act loop and gated by
`/waypoint_reached`. `t3_instruction_following_solver/vlm_planner.py`
defines a Pydantic-validated `VLMResponse` schema (`type`:
navigation|end; `image_division`: left|center|right, splitting the
panorama into three regions for the model to choose a target
region; `subgoal_description`; normalized `pixel_coordinates`
in [0,1] returned directly by the same call, explicitly to "skip a
separate grounding call"; `subgoal_states`). Simple
"X then Y"-style instructions are split into subgoals by regex,
without an API call, before the loop starts.
`pixel_to_waypoint_converter.py` turns a normalized pixel into a 3D
waypoint by computing an azimuth/elevation ray from the known
360-camera FOV (HFOV 360 deg, VFOV 120 deg, 1920x640 image),
rotating that ray into the map frame with the robot's live
odometry quaternion, and forcing the ray downward (adjusting the
pixel row, or clamping the ray's z-component) if it would otherwise
point up or level with the horizon.

Two competing T3 implementations exist side by side in the repo,
with the choice made explicit by inline comments in
`question_solver_master/question_solver.py`: the deployed one
(`t3_instruction_following_solver/t3_solver.py`, commented
`## TODO: This is good`) does the panorama + VLM-pointing loop
described above; an abandoned alternative
(`t3_solver_sad/t3_solver_object_markers.py`, commented
`## TODO: This is bad`) leaned on `/object_markers` instead and was
not used.

## Key numbers

- Final leaderboard score: 30.98 (3rd of 4 finalists); sim prelim
  22.99 - the largest sim-to-real gain of the four finalists, per
  the leaderboard already resolved in `docs/prior_art/README.md`. Source:
  https://web.archive.org/web/20251118230729/https://www.ai-meets-autonomy.com/cmu-vla-challenge
- Self-reported internal per-type success (from the talk, not an
  official rubric score): 80% success disambiguating
  different-colored instances of the same object class
  (Numerical); 70% correct object-marker selection
  (Object-Reference); 35.2% "all sub-goals reached"
  (Instruction-Following) - by far the hardest of the three even
  for a team that placed 3rd overall.
- Exploration budget: under 4 minutes of the 10-minute
  question-answer budget; full coverage plans succeeded in 14 of
  17 training scenes; generated viewpoints were within a 2 m
  threshold of most object markers.
- Real-robot final evaluation: CMU AirLab, Pittsburgh, on hardware
  matching the dev-kit spec - 16-core i9 CPU, 32 GB RAM, RTX 4090
  GPU, Velodyne VLP-16 lidar, Insta360 camera. The team states the
  final test scene's object variety was "drastically different"
  from anything tested in training, and that T3
  (Instruction-Following) specifically underperformed there,
  hypothesized as a training/deployment scene distribution shift -
  notable since the overall score still rose sharply in the
  real-world round (22.99 -> 30.98), implying T1/T2 carried the
  gain while T3 lagged.
- Team: Sreeharsha Paruchuri, Ishita Gupta, Parth Singh, Daksh
  Adhar (CMU MRSD, Class of 2026). Acknowledged in the talk: Dr.
  Jean Oh, Dr. Wenshan Wang, and other CMU Robotics Institute
  organizers, plus sponsor AlphaZ.
- The member site links slides
  (https://drive.google.com/file/d/1yZwuDYyb8nZ7pP3vIkd65N3o2MwtKy7-/view)
  and a certificate
  (https://drive.google.com/file/d/1Zgg9Okdk3pesOiHJPdLvRPj8GqbYgQnK/view)
  - **not opened/verified here**; the site's one-line project
  summary was fetched but the linked Drive files were not.

## Strengths and weaknesses for our setting

Strengths:

- The prompt-embedded mathematical definitions of spatial-relation
  words (near/on/above-below/closest-farthest/between, with
  concrete formulas and an adaptive "near" threshold) is a
  training-free way to stop the LLM from freestyling over raw
  coordinates - the same lesson SORT3D reaches via tool-calling
  instead of prompting (see `docs/prior_art/README.md` section 3). Two
  independent teams converging on "don't let the LLM eyeball
  geometry" is a strong signal this is worth copying regardless of
  our own perception stack.
- Cheap, low-temperature classifier call before committing to a
  heavier per-type solver keeps latency and Gemini rate-limit
  exposure down under a hard 10-minute clock.
- The full-coverage grid + A* + TSP exploration planner, with
  dynamic per-waypoint timeouts so the robot is never stuck
  waiting on one unreachable point, is directly reusable
  independent of whether we rely on GT markers.
- T3's Reason-Act loop over the live panorama, with a
  schema-validated (Pydantic) VLM response and pixel-pointing
  folded into the same call (no separate grounding round-trip), is
  an efficient design for exactly the no-map, live-sensor situation
  our own module will face if the 2026 dev kit does not expose
  ground-truth semantics.
- The team's own retrospective is unusually candid about where it
  underperformed (T3, real-world round) and names a plausible cause
  (train/test scene distribution shift) - a concrete, checkable
  failure mode to design against rather than a vague "it's hard."

Weaknesses / risks, specifically as a template for us:

- Two of three question types (Numerical, Object-Reference) do
  zero learned object detection and depend entirely on the
  simulator's ground-truth `/object_markers` topic. The dev-kit
  README itself notes "teams have a choice whether to use the
  ground-truth semantics... methods that do not use the published
  ground-truth semantics will be scored differently" - worth
  re-verifying against the 2026 dev kit whether this channel exists
  at all, and under what scoring, before assuming it as a fallback.
  If it is unavailable or penalized in 2026, two-thirds of
  CopyPasta's pipeline has no perception fallback to fall back on.
- The talk's claim of uniform "Gemini 2.5 Pro at every stage"
  does not match the submitted code (flash-tier models for
  classification, T1, and T2; 2.5 Pro only for T3) - a reminder to
  verify self-reported architecture claims against code wherever
  code is available, rather than taking conference-talk framing at
  face value.
- No discrete verifier module was found; the four-box
  Input-Handler/Explorer/Solver/Verifier diagram is a
  simplification of what is really a linear ROS state machine with
  ad hoc retry loops. Treat it as a narrative aid, not a literal
  architecture to replicate.
- Hardcoded Gemini API keys are committed in plaintext across
  multiple files (`question_parser.py`, `t1_solver.py`,
  `t2_solver.py`, and again as an exported environment variable in
  `launch_module.sh`) - a credential-hygiene lapse worth avoiding
  in our own repo.
- T3's only error-recovery mechanism is a bare
  reprompt-until-`max_reprompts=100` retry loop - the team itself
  names "exploring error recovery in more detail" as unfinished
  future work. Given Instruction-Following is worth 6 of the 9
  rubric points (versus 1 for Numerical and 2 for
  Object-Reference, per `docs/challenge_brief.md`), this is the
  area where their own weakest link carries the most scoring
  weight.
- The Manhattan-only, 4-connected A* in the exploration planner
  (no diagonal moves) is a minor inefficiency worth noting if we
  build our own coverage planner - not a correctness bug, but an
  easy win to not repeat.

## Takeaways for our 2026 module

1. Verify early, against the actual 2026 dev kit, whether
   ground-truth `/object_markers` are provided in the real-robot
   final round and how their use is scored. CopyPasta's Numerical
   and Object-Reference solvers only function because that channel
   existed in 2025 sim training scenes - this is the single
   highest-leverage unresolved fact for deciding whether we need
   our own detector from day one.
2. Reuse the "prompt-embedded mathematical definitions of spatial
   relations" pattern outright - their exact near/on/above-below/
   closest-farthest/between formulas are a solid starting point,
   whether we feed the LLM ground-truth markers or our own
   detections.
3. Reuse-worthy engineering patterns regardless of perception
   source: a cheap/fast classifier model ahead of dispatch;
   per-question-type dedicated solvers and prompts; structured,
   schema-validated VLM output for navigation-style questions;
   dynamic per-waypoint timeouts so exploration or navigation never
   stalls; and rule-based (no-API) subgoal splitting for simple
   "then"-chained instructions to save latency and rate-limit
   budget.
4. Treat Instruction-Following as the priority investment, not an
   afterthought. It is worth 6 of 9 rubric points, it was
   CopyPasta's weakest self-reported metric (35.2%) even while
   placing 3rd overall, and their own retrospective names it as
   where the real-world round hurt most.
5. Do not copy their T3 error-recovery as-is (bare
   reprompt-until-100 loop) - this is a named, unaddressed gap even
   in a 3rd-place system; build a more principled recovery policy
   (e.g. re-explore, back off to a nearer intermediate waypoint, or
   an explicit failure signal) instead.
6. Keep secrets out of git history in our own repo - a concrete,
   avoidable defect visible in the public CopyPasta repo.
7. If deeper confidence on the color-extraction question matters
   later, re-clone the repo and read
   `ai_module/src/map/object_marker_collector.py` in full (only
   partially inspected here) to settle whether the talk's
   pixel-projection color technique is actually implemented there
   or whether the solvers simply reuse the marker's own render
   color.

## Sources

- Talk (manual captions, read in full):
  https://www.youtube.com/watch?v=kAPltAaRPk4
- Repo (default branch `team-copypasta-submission`):
  https://github.com/parths5/CMU-VLA-Challenge ,
  https://github.com/parths5/CMU-VLA-Challenge/tree/team-copypasta-submission
- Final submission commit (all `ai_module` code, squashed):
  https://github.com/parths5/CMU-VLA-Challenge/commit/feac7bd0
- Launch file confirming which state machine is actually wired up:
  https://github.com/parths5/CMU-VLA-Challenge/blob/team-copypasta-submission/ai_module/src/launch/dummy_vlm.launch
- Key solver/prompt files (all under
  `ai_module/src` on the branch above):
  `question_parser/question_parser.py`,
  `question_parser/gemini_classifier.py`,
  `question_solver_master/question_solver.py`,
  `t1_numerical_solver/t1_solver.py`,
  `t2_selected_object_marker_solver/t2_solver.py`,
  `t2_selected_object_marker_solver/prompts/obj_reference_prompt.py`,
  `t3_instruction_following_solver/t3_solver.py`,
  `t3_instruction_following_solver/vlm_planner.py`,
  `t3_instruction_following_solver/pixel_to_waypoint_converter.py`,
  `t3_instruction_following_solver/vln_data_interface.py`,
  `navigator/planner.py`, `map/object_marker_collector.py`,
  `main_launcher.py`, `main_launcher_t3.py`
- MRSD newsletter blurb: https://labs.ri.cmu.edu/mrsd-news/articles/
- Team member site: https://sreeharshaparuchur1.github.io/
- 2025 leaderboard (context, already resolved in
  `docs/prior_art/README.md`):
  https://web.archive.org/web/20251118230729/https://www.ai-meets-autonomy.com/cmu-vla-challenge
- Dev-kit README (I/O contract, scoring, timing - fetched from the
  same repo, matching `docs/challenge_brief.md`):
  https://github.com/parths5/CMU-VLA-Challenge/blob/team-copypasta-submission/README.md
