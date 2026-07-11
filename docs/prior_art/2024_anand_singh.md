> Imported 2026-07-11 from a parallel research stream. See
> `docs/prior_art/README.md` for provenance and conflict rules.

# Anand Singh 2024 entry - GT-marker
voxel map + Mistral-over-coordinates

One-liner: the 2024 CMU VLA Challenge entry
presented at the IROS 2024 workshop by Anand
Singh is a 4-state ROS pipeline that does one
up-front full-coverage sweep, builds a semantic
map from the sim's ground-truth object markers
(OwlViT used only as a text encoder, not a
detector), and hands raw grid coordinates to
Mistral Large to answer all three question
types - a clean, readable reference
implementation of the "LLM reasons over raw
coordinates" pattern, with several verified
robustness gaps.

Whether this entry won or placed at 2024 is
UNVERIFIED. The IROS 2024 workshop page lists
five unrelated invited-speaker talks plus one
separate "CMU VLA Challenge Presentation" by
Anand Singh, titled "CMU VLA Challenge Method,"
and says only that the workshop "featured the
CMU Vision-Language-Autonomy Challenge,
presenting 'first stage of results' with top
participant presentations." That does not say
1st, 2nd, or 3rd place - it is accurately
described only as "the 2024 entry that was
presented at the workshop."
Source: https://www.ai-meets-autonomy.com/iros-workshop-2024

## What it is

- Talk: Anand Singh, "CMU VLA Challenge
  Method," IROS 2024 workshop, October 14 2024,
  Abu Dhabi UAE, 9:00 AM-1:00 PM GST, Room 17.
  Video: https://youtu.be/-bIMTsnbuoY
  A later pass DID manage to download this
  video and read its slide frames (no
  transcript - see "From the talk video" section
  below); the "machine-untranscribable" framing
  below is now partly outdated. Content in the
  rest of this "What it is" through "Takeaways"
  sections still comes entirely from reading the
  code, as originally written.
  Source: https://www.ai-meets-autonomy.com/iros-workshop-2024
- Code: https://github.com/AnandSingh-0619/CMU-VLA-Challenge
  branch `anand_dev`, commit
  `90dce46188198d49480f68f4672ddf8991403636`,
  message "Working code for submission,"
  authored 2024-09-16T02:16:09Z, 19 files
  changed. Verified directly via `gh api` against
  the branch head and the commit object - dates
  and message match exactly.
- The repo is a fork of
  https://github.com/HaochenZ11/CMU-VLA-Challenge
  (the official challenge/dev-kit repo, confirmed
  via the GitHub API `parent` field), i.e. this
  is a real, in-format submission built by editing
  only `ai_module/` per the challenge's own
  submission instructions.
- The pinned commit's own README (fetched at that
  exact commit, so this is the 2024-era rule text,
  not a later revision) states the 2024 rules
  directly - see "Key numbers" below for the
  scoring and time-budget text, since it differs
  from what we have been assuming for 2026.
  Source: https://github.com/AnandSingh-0619/CMU-VLA-Challenge/blob/90dce46188198d49480f68f4672ddf8991403636/README.md

## Architecture

Single ROS node (`run.py`) running a 4-state
machine - INITIALIZE -> MAPPING -> ASK_QUESTION
-> ANSWERING - alongside a Flask + Flask-SocketIO
web GUI on port 16552 (auto-opened in a browser
tab). The GUI streams log lines over a socket and
accepts a typed question; a "Go to Home" button
publishes waypoint (0,0) directly, bypassing the
state machine.

Flow:
1. INITIALIZE subscribes to `/traversable_area`
   and blocks until the first occupancy grid is
   built.
2. MAPPING always rebuilds a fresh voxel map
   (`res = "no"` is hardcoded in `mapping()`, so
   the `load_stored_map()` pickle-cache code path
   is dead - every run re-explores from scratch
   even though a `voxel_map.pkl` save/load path
   exists). It computes ONE static full-coverage
   waypoint tour up front (`planner.py`), then
   drives it start-to-finish while a
   `/object_markers` subscriber
   (`buildVoxel.py`) passively accumulates
   ground-truth objects into a voxel map as the
   robot happens to pass them. There is no
   re-planning, no time-boxing, and no early-exit
   logic anywhere in this sweep - it runs to
   completion or it does not run.
3. ASK_QUESTION just idles (`rospy.sleep(1)`)
   waiting for a GUI submission.
4. ANSWERING classifies the question type,
   extracts and localizes objects of interest,
   builds a type-specific prompt, gets one more
   Mistral Large response, and post-processes it
   into the ROS output contract (Int32-shaped
   string, a `/selected_object_marker` Marker, or
   a `/way_point_with_heading` sequence), then
   returns to ASK_QUESTION.

Two independent occupancy/pose subsystems exist
side by side with no shared state: the coverage
`Planner` + one `WaypointController` instance
drive the mapping sweep, while `QuestionHandler`
holds its own separate `WaypointController`
instance (a second `/state_estimation`
subscriber) purely to read the robot's current
pose during instruction-following. Both read the
same 0.25 m grid resolution and the same
`origin`/sign convention consistently across
files, which is a real point in the code's favor.

## Components

**Semantics.** `buildVoxel.py`'s
`SemanticVoxelMap` subscribes to
`/object_markers`, which the challenge README
labels explicitly as "Ground-Truth Semantics ...
object labels and bounding boxes within 2m
around the vehicle" - i.e. this entry uses the
sim's ground-truth object detector directly and
ships no learned detector of its own. For each
NEW marker id (deduped by id, so a moving/
re-published marker is only embedded once),
`marker.ns` (its class-name string, e.g.
"chair") is embedded with OwlViT
(`google/owlvit-base-patch32`)
`get_text_features` - OwlViT here is used ONLY
as a CLIP-style text encoder over class-name
strings, never on an image; there is no
reference to `/camera/image` anywhere in these
nine files, so no per-object color or visual
attribute is ever extracted despite the sensor
suite providing a 360 camera. The resulting
L2-normalized text embedding becomes the
point's "feature" in a `VoxelizedPointcloud`
(voxel size 0.05 m, mean feature pooling,
`voxel_map/voxel.py` - vendored, unmodified
utility code).

**Map.** `map/gridMap.py`'s `GridMapHandler`
rebuilds a full 2D binary occupancy grid from
scratch on every `/traversable_area` callback
(0.25 m cells, y-axis sign-flipped, origin =
observed min-x/min-y). Rebuilding from scratch
per message is only correct because the README
describes this topic as containing "the
traversable area of the entire environment" (a
cumulative snapshot, not an incremental patch) -
confirmed from the pinned README text, not
assumed.
Source: https://github.com/AnandSingh-0619/CMU-VLA-Challenge/blob/90dce46188198d49480f68f4672ddf8991403636/README.md
The `/traversable_area` subscription registered
in `initialize()` is never torn down, so the
grid can keep changing in the background during
ASK_QUESTION/ANSWERING - `handleQuestions.py`
re-fetches `get_grid_map()` at question time
rather than freezing the map used for planning,
which is a structural risk (unverified at
runtime here) if new pointcloud messages keep
arriving mid-question.

**Exploration.** `map/planner.py`'s `Planner`
samples coverage points on a `min_gap=0.6 m`
stride, keeps only cells at least
`threshold_distance=0.2 m` from any obstacle
(via `scipy.ndimage.distance_transform_edt`),
links candidate points within `8.0` grid cells
(2.0 m at 0.25 m resolution) into a graph
weighted by 4-connected A* path length, and
solves an open-path TSP over that graph
(`networkx.approximation.traveling_salesman_
problem`). A `safety_buffer=0.15 m` obstacle-
dilation step (`add_safety_buffer`) is defined
but its call site is commented out
(`# self.add_safety_buffer()`) - so the safety
margin exists in the code but is never actually
applied. The tour always appends a return-to-
origin waypoint `(0,0)` at the end, duplicating
the GUI's separate "Go to Home" button. Timing
is measured (`t_end = time.time() - t_start`,
printed as "Got Waypoints in ...") but only to
stdout - never logged to the GUI, never checked
against a time budget.

**Reasoning LLM.** `vla/queryVLM.py`'s
`QueryVLM` supports three backends
(`query_gemini`, `query_chatgpt`,
`query_mistral`); only Mistral Large
(`mistral-large-latest`, via the `mistralai`
SDK) is wired up live (`model_name="mistral"`
hardcoded in `run.py`). The Gemini and ChatGPT
paths carry literal placeholder strings as API
keys (`"Your API Key: DO NOT USE"`,
`"your_chatgpt_api_key"`) - not real leaked
credentials, just unused dead code. The Mistral
path's key is also a literal placeholder
(`"Your API Key: USE THIS"`), so the committed
code is not runnable without someone filling in
a real key. Every call sleeps 2 s first
(apparent manual rate-limit backoff). A single
question triggers at least 3 sequential Mistral
calls: (1) classify question type, (2) extract
objects of interest, (3) the type-specific
answer - each with its own 2 s sleep plus network
latency, before any of the type-specific
post-processing runs.

**Localization.** `voxel_map/
voxel_map_localizer.py`'s `VoxelMapLocalizer`
loads OwlViT a SECOND, independent time (not
shared with `buildVoxel.py`'s copy). Its
`localize_AonB(A, B, ...)` supports two modes:
if `B` is empty, `find_alignment_for_A`
computes cosine similarity between the query's
text embedding and every voxel feature and
returns ALL points above `threshold` (plural,
despite the singular-looking variable names);
if `B` is a real anchor object, it instead
finds the single `A`-candidate closest in raw
L2 distance to any top-k `B`-candidate ("A near
B"). In practice `handleQuestions.py.
localize_objects()` ALWAYS calls
`localize_AonB(obj, "", k_A=10, k_B=50,
threshold=0.85, data_type='xyz')` for every
object of interest - `B` is always `""`. That
means the entire "A near B" geometric-anchoring
branch is dead code at runtime: every spatial
relation in every question ("closest to,"
"between," "near the window") is resolved
purely by the downstream LLM reading a flat list
of candidate grid coordinates, with zero
geometric pre-filtering in code. This is a
directly verified instance of the "LLM over raw
coordinates" fragility called out as a design
risk, not an inference from behavior.
A related latent bug: the same function's
`data_type == 'r3d'` branch references a
variable `target` that is only ever assigned in
the (dead, at runtime) `B`-non-empty branch;
called with `B=""` and `data_type='r3d'` this
would raise `UnboundLocalError`. It never
surfaces because every live call passes
`data_type='xyz'`, which skips that block
entirely - the bug is real in the source but
inert given how the function is actually called.

**Output handling.** `vla/postProcess.py` (465
lines) dispatches on query type:
- Numerical: scans response lines for one
  starting `"Answer:"` and returns the trailing
  substring **as a raw string, never cast to
  int** - so the Int32 output contract is not
  actually enforced in code; any non-numeric
  trailing text would pass through unchanged.
- Object reference: regex-extracts an `(x, y)`
  integer grid pair from `"Answer: (x,y)"`, then
  requires an EXACT integer match against one of
  the grid coordinates that were serialized into
  the prompt earlier (`find_global_and_scale`).
  If the LLM's coordinates do not exactly match
  (rounding, reordering, any drift), the lookup
  returns `None` and the very next line calls
  `.get()` on it with no None-check - a
  confirmed crash path, not a hypothetical one.
- Instruction following: splits the LLM's answer
  into a 4-primitive DSL - `Goto(x,y)`,
  `Between((x1,y1),(x2,y2))`,
  `Avoidbetween((x1,y1),(x2,y2))`,
  `StopAt(x,y)` - matched by case-sensitive
  substring checks (`"Goto" in action`, etc.).
  `Avoidbetween` rasterizes a Bresenham line
  between the two given points and zeroes out
  that cell plus its 8 neighbors in a working
  copy of the grid, biasing subsequent pathing
  away from it - a real, if blunt, mechanism.
  `Between` just averages the two given integer
  coordinates; no cross-check against the
  localized-object list. A `"#"`-prefixed
  subgoal-splitting feature exists in
  `execute_plan` but the instruction-following
  prompt never asks the LLM to emit `#` headers,
  so that branch is dead code at runtime.
  Path stitching between waypoints uses a
  DIFFERENT pathfinder (`theta_star`, 8-
  connected with line-of-sight shortcutting)
  than the plain 4-connected A* used during
  up-front coverage planning - two unrelated
  pathfinding implementations coexist with no
  shared code.
  Final execution reports "Instruction following
  completed" vs "incomplete" based on a
  `complete_flag`, but `WaypointController.run()`
  (in `map/wpNav.py`, fetched from the repo since
  it was not among the nine saved files)
  unconditionally sets `self.success_flag = True`
  after its loop finishes, even for waypoints
  that individually timed out and were skipped
  (30 s per-waypoint timeout in
  `publish_waypoints`). So "incomplete" is
  effectively unreachable dead-letter text - a
  partial, timed-out run is reported as complete.

## Prompts (summarized)

All three prompts live in `vla/queryGenerator.py`
and share a structure: environment X/Y grid
bounds, a one-shot worked example, the serialized
`format_localized_points()` block (per-object
grid-coordinate centers, plus a bounding box only
when both X and Y scale exceed 0.6 m), the
question text, and an instruction to use chain-
of-thought reasoning before a strict answer line.

- **Classification prompt** (in
  `handleQuestions.py.process_question`, not
  `queryGenerator.py`): defines the three
  question types with the challenge's own
  example questions, and asks for EXACTLY one of
  `numerical` / `object reference` /
  `instruction following`. Parsing is strict
  set-membership after `.replace('*','').strip().
  lower()` - any other phrasing causes the whole
  pipeline to silently produce no answer (no
  retry, no fallback).
- **Object-extraction prompt**
  (`handleQuestions.py.
  extract_objects_of_interest`): asks for a
  singular-form, comma-separated list of objects
  of interest, in original spelling. Parsed by a
  bare `.split(',')` - an object description
  containing its own comma (e.g. "gray, L-shaped
  couch") would silently mis-split.
- **Numerical query**: one-shot example is a
  sofa/pillow overlap-counting problem; asks for
  `Reasoning: ...` then `Answer: <single
  integer>`.
- **Object-reference query**: one-shot example
  shows combinatorial reasoning over 6 candidate
  plant locations and 6 candidate book locations
  to pick the one nearest a cabinet, entirely in
  grid-coordinate text; asks for
  `Answer: (x,y)` - the unique grid coordinate of
  the target object only, no bounding box in the
  expected answer format even though bounding
  boxes were provided as context.
- **Instruction-following query**: defines an
  action DSL as
  `action_list[Goto(x,y), Between((x1,y1),
  (x2,y2)), Avoidbetween((x1,y1),(x2,y2),
  StopAt(x,y)]` - this bracket is malformed in
  the actual source (a missing closing paren
  after `(x2,y2)` and no closing paren for
  `Avoidbetween` before the list closes) - a
  verified typo in the prompt text itself, not a
  transcription error here. The worked example
  output only demonstrates `Goto`, `Between`, and
  `StopAt` - `Avoidbetween` is named in the DSL
  list but never shown in an example, and the
  downstream parser matches it by exact-case
  substring, so an LLM that renders it as
  `AvoidBetween` (capital B) would silently fail
  to match.

## Key numbers

- Grid resolution: 0.25 m, consistent across
  `planner.py`, `gridMap.py`,
  `handleQuestions.py`, `postProcess.py`.
- Coverage waypoint stride (`min_gap`): 0.6 m.
- Safety buffer: 0.15 m, defined but unused (call
  site commented out).
- Minimum clearance to keep a coverage waypoint:
  0.2 m from nearest obstacle.
- TSP graph-linking radius: 8.0 grid cells = 2.0 m.
- Voxel size: 0.05 m (`VoxelizedPointcloud`
  default).
- Semantic match threshold: cosine similarity
  > 0.85 (object localization).
- Top-k candidates: `k_A=10`, `k_B=50` (`k_B`
  unused at runtime since `B` is always `""`).
- Bounding-box emission threshold: object scale
  > 0.6 m in both X and Y.
- Waypoint arrival threshold: 0.65 m default
  (`WaypointController`, used during the
  full-coverage sweep) vs 0.8 m (used by the
  separate `publish_waypoints` helper invoked
  from `postProcess.execute_plan`) - two
  different arrival tolerances for two phases of
  the same system.
- Per-waypoint navigation timeout: 30 s (skip and
  move on).
- Object-marker publish duration for the final
  selected object: 5 s at 10 Hz.
- Web GUI port: 16552.
- LLM: `mistral-large-latest`; hardcoded 2 s
  sleep per call; >= 3 sequential calls per
  question (classify, extract objects,
  type-specific answer).
- Commit: `90dce46188198d49480f68f4672ddf8991403636`,
  "Working code for submission," 2024-09-16
  02:16:09 UTC, 19 files changed.
- `postProcess.py`: exactly 465 lines (verified
  with `wc -l` on the saved copy).

2024-era rules, from the README at this exact
commit (differs from what we have been assuming
for 2026 - flagging explicitly, not reconciled
here):
- Scoring: Numerical /1 (exact match), Object
  Reference /1 (marker center within some X-Y
  radius of ground truth), Instruction-Following
  /3 (trajectory score with partial credit). This
  is the SAME 1:1:3 ratio as the /1:/2:/6 scoring
  already recorded for 2025 in
  `docs/prior_art/README.md`, just at half scale -
  instruction-following dominates the point total
  in both years.
- Time budget: "a total time limit for the
  combined exploration and question-answering
  given one scene and one language statement...
  3 minutes for most scenes and longer for the
  scenes with multiple rooms." This reads as a
  budget per SCENE (covering exploration plus
  presumably the scene's questions), not
  explicitly "10 min per question" as this repo's
  other docs currently assume for 2026. Whether
  the 2026 dev kit uses the same per-scene framing
  or a genuinely different per-question framing is
  UNVERIFIED here - `docs/challenge_brief.md`
  already flags the 2026 I/O contract as needing
  verification against the cloned dev kit, and
  this is one more concrete thing to check when
  that happens.
  Source: https://github.com/AnandSingh-0619/CMU-VLA-Challenge/blob/90dce46188198d49480f68f4672ddf8991403636/README.md

## Strengths and weaknesses for our setting

Strengths:
- End-to-end, actually working, and legible - a
  full state machine, mapping, semantic map, LLM
  routing, and per-type output handling all
  present and internally consistent on
  coordinate conventions (0.25 m resolution and
  the same y-sign flip everywhere).
- The coverage planner (distance-transform
  clearance filtering, KD-tree neighbor linking,
  TSP tour, redundant-waypoint pruning) is a
  reasonable, self-contained piece of engineering
  independent of the LLM/semantics choices.
- Cheap to build: OwlViT-as-text-encoder plus
  ground-truth markers avoids running any actual
  object detector, and the localization match is
  a simple cosine threshold.

Weaknesses, most relevant to us first:

1. **LLM reasons over raw coordinates with no
   geometric pre-filtering.** Confirmed at the
   code level, not inferred: the localizer's
   "A near B" anchoring branch is written but
   never called (`B` is always `""`), so every
   spatial relation - "closest to," "between,"
   "near the window" - is resolved purely by an
   LLM reading a flat text list of grid
   coordinates and bounding boxes. This is
   exactly the fragility SORT3D-style prior art
   warns about (see `docs/prior_art/README.md` section
   3) and this entry is a clean, verified
   instance of the anti-pattern rather than the
   mitigation.
2. **Ground-truth-marker dependence, zero visual
   perception.** All semantics come from
   `/object_markers`; OwlViT never touches an
   image. No color or visual-attribute
   extraction exists anywhere, despite the
   sensor suite providing a 360 camera and
   despite the challenge's own example questions
   ("blue chairs," "orange chair," "black trash
   cans") and the VLA-3D dataset's own emphasis
   on per-object dominant color (see
   `docs/prior_art/README.md` section 2). If the 2026
   dev kit withholds ground-truth markers at test
   time (unverified either way here), this
   architecture has no fallback.
3. **Up-front, static, unbounded full-coverage
   mapping.** The entire environment is swept
   ONCE, planned as a single static TSP tour,
   before any question is even asked, with no
   time budget check, no early termination, and
   no re-planning as new area is discovered. Per
   the pinned 2024 README, exploration and
   question-answering SHARE a combined budget as
   short as 3 minutes for most scenes - a single
   unbounded full-coverage sweep is a direct risk
   of running out the clock before ever answering
   a question, especially for multi-room scenes.
   Whether the actual 2026 budget is per-scene or
   per-question (see "Key numbers" above), an
   up-front full sweep with no time-awareness is
   fragile either way.
4. **Silent failure modes verified in code, not
   hypothesized.** Numerical answers are never
   cast to int; a non-exact-match object
   reference lookup returns `None` and the next
   line crashes on it; a mis-classified question
   type produces no answer with no retry; a
   partially-timed-out instruction-following run
   is still reported "completed" because
   `success_flag` is unconditionally set `True`.
   None of these are edge-case speculation - each
   is a specific, cited line of code.
5. **Case-sensitive, exact-string parsing of LLM
   output throughout.** The DSL parser matches
   `"Goto"`, `"Between"`, `"Avoidbetween"`,
   `"StopAt"` as exact-case substrings; the
   object-reference parser requires an exact
   integer-tuple match against previously-
   serialized coordinates; the object-extraction
   step is a bare comma-split. Any LLM formatting
   drift (very plausible across model versions,
   temperatures, or provider swaps) breaks these
   silently rather than gracefully.
6. **Duplicated, inconsistent subsystems.** Two
   separate `WaypointController` instances (two
   `/state_estimation` subscribers), two separate
   OwlViT model loads, two unrelated pathfinding
   implementations (plain A* for coverage
   planning vs theta-star for instruction
   execution), and two different waypoint-arrival
   thresholds (0.65 m vs 0.8 m) for what is
   conceptually one navigation capability. None
   of this is fatal, but it is waste and a source
   of subtle behavioral mismatch between mapping-
   time and answer-time navigation.
7. **Not runnable as committed.** The only live
   LLM backend's API key is a literal placeholder
   string; someone must edit source to add a real
   Mistral key before this code executes at all.

## Takeaways for our 2026 module

- Do not repeat the "return a text-serialized
  coordinate list and let the LLM do all spatial
  reasoning" pattern for relations like "between,"
  "closest to," or "near" - this entry shows
  exactly how that looks in a complete system,
  including the specific crash/silent-failure
  points it creates downstream. Prefer geometric
  pre-filtering (candidate shortlists via actual
  distance/containment computations) before
  handing anything to an LLM, consistent with
  SORT3D's and the KAIST-ISE team's approach
  already noted in `docs/prior_art/README.md`.
- If we plan to lean on any ground-truth
  semantics topic at test time, build and test a
  fallback path now - do not assume it will be
  available in the 2026 dev kit, and do not let
  color/attribute extraction depend entirely on
  a class-name string with no image ever
  consulted.
- Treat "explore fully, then answer" as a
  concrete anti-pattern to avoid by design: budget
  exploration time explicitly, prefer incremental/
  adaptive exploration (closer to SayNav's
  "plan on partial map, refine" loop, already
  flagged as relevant in `docs/prior_art/README.md`
  section 3) over a single static up-front TSP
  sweep with no clock awareness.
- Whatever output-parsing scheme we use for LLM
  responses, make it typed and defensively
  validated (cast/validate the integer answer,
  handle a failed object lookup instead of
  crashing, treat "no valid classification" and
  "waypoint timed out" as real failure states
  rather than silently reporting success) - this
  entry is a working demonstration of what
  happens when that validation is skipped.
- The exact 2024 time-budget wording ("3 minutes
  for most scenes... combined exploration and
  question-answering") is worth reconciling
  against the 2026 dev kit once cloned - it may
  mean our working "10 min/question" assumption
  in `docs/prior_art/README.md` and elsewhere needs a
  citation update or a per-scene reframing.

## From the talk video (frames/transcript)

A follow-up pass used the `/watch` skill and
succeeded where the earlier yt-dlp attempt (noted
above) failed. Frames were obtained; no
transcript was obtained (no audio content below).

**Download.** yt-dlp 2026.07.04 with its default
(web) client still fails exactly as before -
"This video is not available" on plain
`yt-dlp <url>`, or "Only images are available
for download" once metadata succeeds. Trying
other player clients found one that works:
`--extractor-args "youtube:player_client=android"`
exposes format `18` (mp4 muxed, 640x360, 25 fps,
avc1/mp4a, ~13 MB, duration 09:18) - a real,
playable stream. Other clients tried (`ios`,
`web`, `mweb`, `web_embedded`, `tv`,
`web_creator`, `android_vr`) each failed
differently (images-only, PO-token required,
DRM-protected, sign-in required, or "video not
available") - this looks like a general YouTube-
side extraction quirk (SABR-only streaming /
PO-token gating), not something specific to this
video. No native captions came back on any
client.

**Transcript.** Still unavailable - no
GROQ_API_KEY/OPENAI_API_KEY configured, so the
Whisper fallback was skipped (`--no-whisper`).
Everything below is read from ~35 slide frames
(ffmpeg scene-change detection plus frames pulled
at ~25 s intervals across the full 9:18 runtime,
some at boosted resolution to read small text/
tables), not from spoken audio.

**New facts from slides, organized by topic:**

- **Presenter affiliation is Georgia Tech, not
  CMU.** Title slide reads "Anand Singh,
  anandsingh@gatech.edu" with the same GitHub
  link already recorded above. The on-slide talk
  title is "AI Meets Autonomy: Vision, Language,
  and Autonomous Systems - IROS 2024 Workshop,"
  which differs from the "CMU VLA Challenge
  Method" title recorded from the workshop page
  above - almost certainly the same talk under
  two different titles in two places, but
  UNVERIFIED which is authoritative.
- **Talk-level pipeline name: "Planning ->
  Mapping -> Query."** This is the slide deck's
  own three-stage framing, distinct from (but
  structurally consistent with) the code's
  4-state INITIALIZE/MAPPING/ASK_QUESTION/
  ANSWERING machine described above - the deck is
  a simplified talk-level view, not a literal
  relabeling.
- **Planning sub-pipeline, explicit on slide:**
  Top-Down Map -> "Divide into traversable grid"
  -> "TSP to find shortest path" -> "A* local
  Planner" -> "Smoothed Path" (shown with an
  actual grid-map-plus-path plot). This confirms
  the code-derived coverage-planner description
  above and adds the author's own framing: TSP is
  explicitly called the GLOBAL planner, A* the
  LOCAL planner.
- **Mapping sub-pipeline, explicit on slide:**
  "Ground Truth semantics (<2m objects)" ->
  "Information (coordinates, scale, name)" ->
  "CLIP Embeddings" -> "Semantic Memory (Voxel
  Map)." Slide footnote: "[1] Liu, P., Orru, Y.,
  Paxton, C., Shafiullah, N.M.M. and Pinto, L.,
  2024. OK-Robot: What really matters in
  integrating open-knowledge models for
  robotics." Flagging a naming point, not
  resolving it: the slide says "CLIP Embeddings,"
  while the code-derived note above says
  `buildVoxel.py` calls OwlViT
  (`google/owlvit-base-patch32`) `get_text_
  features`. OK-Robot's own published pipeline
  pairs OWL-ViT (detection) with CLIP
  (embeddings), and OwlViT's text tower is
  CLIP-derived, so this may just be the talk
  simplifying "OwlViT-as-text-encoder" to "CLIP
  Embeddings" for a slide audience - UNVERIFIED
  either way; recording the discrepancy as-is.
- **No LLM brand named anywhere on slides.**
  Every Approach/Query slide says generically
  "LLM Response" or "Prompt Engineering" - Mistral
  is never shown visually. The code-level
  "Mistral Large" fact above stands (it comes
  from reading `run.py`/`queryVLM.py` directly);
  it is simply not corroborated by the deck.
- **Instruction-following worked example, more
  literal than the code comments alone:** slide
  shows the raw LLM response format as
  `Goto(13,43)`, `Goto(22,28)`, `StopAt(28,28)`
  (comma-separated integers) for the example
  query "Take the path near the TV and go to the
  pillow farthest from the lamp," followed by a
  separate "Grounding LLM Response" box listing
  rules resembling "Identify p1, add p1 to
  waypoint list," "add the endpoint to waypoint
  list," "add all in-between points as new
  waypoints," "add p3 to waypoint list" (table
  partly hard to read at 640x360 source
  resolution - treat as paraphrase, not verbatim
  quote). This feeds a **Theta\* algorithm** as
  local planner for instruction-following -
  matching the code's `theta_star` usage exactly,
  and confirming (from the author himself) that
  Theta* is deliberately different from the
  coverage-time A*. Slide cites "[2] I. Singh et
  al., 'ProgPrompt: Generating Situated Robot
  Task Plans using Large Language Models,' 2023
  IEEE ICRA" as the explicit inspiration for the
  Goto/Between/Avoidbetween/StopAt DSL - a
  citation not previously recorded from the code
  alone.
- **Worked examples shown on slides:**
  - Numerical: "How many photos are on the TV
    cabinet?" -> LLM Response: `2`.
  - Object Referencing: "Find the potted plant
    near the books on the cabinet." -> LLM
    Response: "Object 'potted plant' is located at
    coordinates: [5.67, 0.59, 1.17]" - three
    floats (x, y, z). UNVERIFIED whether this is
    illustrative/rounded differently from what the
    code's regex parser (`Answer: (x,y)`, integer
    2D pair, per the code-level notes above)
    actually consumes at runtime, or whether the
    real prompt/parser also has a z-including
    variant not present in the saved source files.
  - Instruction Following: "Take the path near the
    TV and go to the pillow farthest from the
    lamp" - shown executing live in the demo video
    (see below).
- **Evaluation Results slide - exact numbers, not
  previously recorded anywhere in this repo.**
  Evaluated across 3 scenes ("Scene 1 with 1 room;
  Scene 2 and 3 with 2 rooms each"):
  - Numerical Query (max score = 1 per scene):
    Scene 1 = 0, Scene 2 = 0, Scene 3 = 1.
  - Object Referencing Query (max score = 2 per
    question, 2 questions/scene): Scene 1 = 2, 2;
    Scene 2 = 0, 2; Scene 3 = 2, 2.
  - Instruction Following Query (slide literally
    prints "Ques 1 max score= 4" then "Ques 1 max
    score= 6" again - almost certainly a slide
    typo for "Ques 2 max score= 6"; not
    re-verified against any second source):
    Scene 1 = 0, 6; Scene 2 = 4, 6; Scene 3 = 4, 4.
  Treat these as a rough calibration point for
  "what a working but imperfect 2024 entry
  actually scored," not as a 2026 benchmark - the
  2026 scoring scale/rubric is still unverified
  against the dev kit per `docs/challenge_brief.md`.
- **Demo video (3 static screenshots only, no
  motion since this pass is frames-only):** the
  same "near the TV, stop at the pillow farthest
  from the lamp" instruction executing on a
  top-down occupancy map, with a moving yellow-arc
  field-of-view indicator, plus a live camera feed
  panel and what looks like a segmentation-style
  overlay panel alongside it. This visual overlay
  is consistent with, but not proof of, richer
  per-object visual processing than the code
  review found - it may be the base-autonomy
  stack's own output rather than anything this
  entry's `ai_module` computes; not reconciled
  here.
- **"Future Work & Improvements" slide (verbatim
  bullets, useful as the author's own risk
  register):**
  - "Color Information Limitations: Ground truth
    semantic sensor lacks color data. Hence
    numerical ques have lower success rates."
  - "Object Detection Range: Readings from the
    ground truth sensor are limited to within 2
    meters. Objects placed on shelves or deeper
    inside cupboards may not be detected, impacting
    accuracy." - directly confirms the code-derived
    "<2m objects" ground-truth radius above as a
    self-acknowledged limitation, not just an
    inferred sensor spec.
  - "Time Optimisation: The current approach uses
    TSP as a global planner (NP-hard) and A* as a
    local planner. Planning time can be reduced
    with optimized algorithms." - matches this
    dossier's own "no time-budget check" weakness
    above, now as a first-party admission.
  - "LLM Integration: Use offline LLMs for handling
    smaller responses more efficiently, enhancing
    system responsiveness." - the author's own
    stated intent to move off a hosted API (Mistral
    Large) toward local models, directly relevant
    to the "not runnable as committed" / API-key
    weakness noted above.

None of this contradicts the code-level findings
recorded above in any load-bearing way - it mostly
confirms, names, and cites them, and adds the
scoring numbers and future-work list that reading
code alone could not reveal.

## Sources

- Code repo (branch, commit, all component
  files): https://github.com/AnandSingh-0619/CMU-VLA-Challenge
  branch `anand_dev`, commit
  `90dce46188198d49480f68f4672ddf8991403636`
- Same commit's README (2024 scoring and time
  budget text, system I/O table):
  https://github.com/AnandSingh-0619/CMU-VLA-Challenge/blob/90dce46188198d49480f68f4672ddf8991403636/README.md
- Fork parent (official challenge/dev-kit repo):
  https://github.com/HaochenZ11/CMU-VLA-Challenge
- Talk (frames read via `/watch` in a later
  pass, downloaded with
  `yt-dlp --extractor-args
  "youtube:player_client=android"`; no transcript,
  no Whisper key configured):
  https://youtu.be/-bIMTsnbuoY
- Workshop page listing the talk (used only to
  confirm date/venue/talk title, not a placing):
  https://www.ai-meets-autonomy.com/iros-workshop-2024
- Cross-reference, this repo's own prior-art
  synthesis (SORT3D/ConceptGraphs/SayNav pattern,
  VLA-3D color emphasis, 2025 scoring numbers):
  `docs/prior_art/README.md`
