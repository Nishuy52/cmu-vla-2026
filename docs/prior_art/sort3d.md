> Imported 2026-07-11 from a parallel research stream. See
> `docs/prior_art/README.md` for provenance and conflict rules.

# SORT3D - zero-shot 3D grounding via an LLM
plus a heuristic spatial-relation toolbox

Compiled 2026-07-11 from the arXiv v2 full text
(https://arxiv.org/html/2504.18684v2, cross-checked
against the abstract page
https://arxiv.org/abs/2504.18684) and the public code
repo (https://github.com/nzantout/SORT3D, default
branch `humble-wheelchair`, fetched via `gh api`).
IROS 2025. Authors: Nader Zantout, Haochen Zhang
(equal contribution), Pujith Kachana, Jinkai Qiu,
Guofei Chen, Ji Zhang, Wenshan Wang - CMU (the same
group that publishes VLA-3D and IRef-VLA).

## What it is

SORT3D grounds free-form referring language ("the
nightstand to the right of the bed") to a specific
object in a live 3D scene, and optionally drives a
robot to it. It is zero-shot (no text-to-3D training
data) and needs only one in-context example. Two
pieces make it work: (1) captions from a 2D VLM
attached to each detected 3D object, giving the LLM
attributes (color, material, shape, affordance) that
pure geometry can't see; (2) a small library of
hand-coded geometric "tools" (find_near, find_left,
order_bottom_to_top, ...) that the LLM calls in
sequence instead of trying to do 3D spatial math
itself.

Relevance to us is high for a structural reason, not
just a topical one: SORT3D-Nav is built directly on
top of the same base-autonomy stacks our own project
already treats as reference systems - the wheelchair
platform is `jizhang-cmu/cmu_vla_challenge_unity`
(https://github.com/jizhang-cmu/cmu_vla_challenge_unity),
listed in our repo's own CLAUDE.md as the "prior-year
base system," and there is also a mecanum-wheel
variant on
`jizhang-cmu/autonomy_stack_mecanum_wheel_platform`
(https://github.com/jizhang-cmu/autonomy_stack_mecanum_wheel_platform).
SORT3D's `ai_module/` is therefore the closest
published example of exactly the kind of module we
are being asked to build: a reasoning layer dropped
on top of that same fixed autonomy stack, driven by
360 camera + 3D LiDAR + odometry. Source for the repo
layout claim:
https://github.com/nzantout/SORT3D (README, section
"Repository Structure").

No LICENSE file exists in the repo (`gh repo view`
reports `licenseInfo: null`, and the recursive file
tree has no LICENSE/COPYING entry). Treat the code as
reference material for re-implementation, not as a
license-cleared dependency, unless this is checked
directly with the authors.

## Pipeline

The system runs in five stages (paper section IV +
repo cross-check).

1. **Instance-level semantic mapping.** Open-vocab 2D
   detection via Grounding DINO plus instance
   segmentation via SAM 2, then LiDAR points are
   projected onto the semantic image and associated
   with instances; per-frame instance point clouds are
   tracked across frames using 3D proximity. For the
   ReferIt3D/IRef-VLA benchmarks this stage is skipped
   entirely and ground-truth ScanNet 3D boxes are used
   instead - the toolbox and LLM stages are identical
   either way. Source:
   https://arxiv.org/html/2504.18684v2 (section IV-A);
   detector confirmed in repo README under "Afterwards,
   install the following dependencies in the semantic
   mapping module" which pulls in Grounded-SAM-2
   (`pip install Grounded-SAM-2/grounding_dino` /
   `Grounded-SAM-2`),
   https://github.com/nzantout/SORT3D (README).

2. **2D captioning.** For each object instance, the
   best 2D view is the one with the highest CLIP
   similarity to the object's class label; that crop is
   captioned. Qwen2-VL-7B is used for benchmark
   evaluation, and a quantized Qwen2.5-VL-Instruct-3B is
   used on-robot "due to memory constraints." Caption
   prompt template (paper, quoted verbatim): "You are an
   AI model that describes the characteristics of a
   query object in an image. Describe the <<object>> in
   this image, using properties like color, material,
   shape, affordances, and other meaningful attributes.
   Provide the response in this format: 'The <<object
   name>> is <<color>>, <<material>>, <<shape>>'."
   Source: https://arxiv.org/html/2504.18684v2 (section
   IV-B).

3. **Two-stage LLM filtering** (Mistral Large 2). First
   query extracts referenced object nouns/modifiers from
   the command; second query matches those against the
   scene's object list and returns candidate IDs. Both
   prompts are in the repo verbatim at
   `ai_module/src/language_planner/language_planner/prompts/object_extraction.py`
   and are worth reading directly for the pragmatic
   in-context examples they use, e.g. "I'm hungry, where
   can I get food?" -> `["fridge"]` and "I finished
   drinking this soda, and I want to throw it out." ->
   `["soda", "trash can"]` - i.e. the filter LLM is
   explicitly taught to resolve indirect/implicit object
   references, not just literal nouns. Source:
   https://github.com/nzantout/SORT3D (path above,
   fetched 2026-07-11).

4. **Spatial reasoning via LLM + toolbox.** The
   candidate objects are handed, as a flat table, to a
   LangChain/LangGraph tool-calling agent
   (`llm_backend/langgraph_agent.py`) built as a small
   state machine: an `assistant` node calls the LLM
   (GPT-4o or Mistral Large 2), a `tools` node executes
   whichever toolbox function the LLM invoked, and an
   `instruct_retry` node nags the LLM if it replies with
   prose instead of a tool call. The loop ends when the
   LLM's message contains the literal string "done". A
   `notepad` pseudo-tool lets the LLM externalize
   step-by-step reasoning between tool calls (visible in
   the one in-context example, see "Models and prompts"
   below). The repo also contains an *actor-critic*
   variant of this graph
   (`build_actor_critic_graph`, with a
   `CriticStructuredOutput{approval, feedback}` model)
   where a second LLM critiques the first's reasoning -
   this is present in the code but is not described in
   the paper text extracted here, so its use in the
   published results is unconfirmed; flagging as
   repo-only/unverified. Source:
   https://github.com/nzantout/SORT3D,
   `ai_module/src/language_planner/language_planner/llm_backend/langgraph_agent.py`.

5. **Action parsing (navigation mode only).** In
   benchmark mode the agent ends by calling
   `pick_object(id)`. In live-navigation mode it instead
   calls `command_robot([(str, tuple), ...])`, where each
   tuple is `("go_near", (id,))` or `("go_between", (id1,
   id2))`; `go_near` places a waypoint at the closest
   point in free space to the target's centroid,
   `go_between` places one near the midpoint of two
   targets. These waypoints go to the fixed downstream
   planner - i.e. SORT3D never touches motion planning or
   obstacle avoidance itself, matching our own
   scope boundary. Source:
   https://arxiv.org/html/2504.18684v2 (section IV-D) and
   `tools.py`'s `CommandRobotSchema`/`command_robot`.

Note on a second, separate prompt path found only in
the repo: `prompts/planning.py` implements an older/
alternate direct-code-generation planner that skips
the toolbox entirely - it hands the LLM a flat `(id,
name, x, y)` list and two primitives
(`go_between`/`go_near`) and asks it to write a Python
`go()` function directly, with two full in-context
examples embedded in the prompt string. It is unclear
from the fetched material whether this is a
deprecated predecessor to the toolbox approach or an
alternate lightweight mode still exercised somewhere
in SORT3D-Nav; the paper text does not describe it.
Flagging as unverified/unclear rather than presenting
it as part of the published method.

## The spatial toolbox (function list + definitions)

Table I of the paper lists the toolbox at a
conceptual level; the actual implementation lives in
`ai_module/src/language_planner/language_planner/llm_backend/tools.py`
(`class AgentToolbox`). Both are combined below. All
positions are `(x, y, z)` centroids with **z as the
vertical/up axis** (confirmed because
`order_bottom_to_top` sorts on index 2, and
`find_above`/`find_below` compare z-extents) and x, y
as the horizontal ground plane. Object "size" for
ordering is the area of the object's largest face, not
volume - the paper states this explicitly: "more
intuitive than volume for flat objects." Objects
carry both an absolute internal ID and a
relabelled-to-the-LLM relative ID
(`object_id_map`/`inv_object_id_map`), which the paper
describes as the perception module's full output being
"partially hidden from the LLM."

Functions confirmed **implemented and wired into the
LLM's tool list** (from `AgentToolbox.__init__`):

- `find_between(target_name, first_anchor_name,
  second_anchor_name)` - view-independent. Builds a
  local frame whose x-axis runs from anchor1 to
  anchor2 (via a 2D rotation matrix derived from the
  anchor-anchor vector), rotates all three object
  boxes into that frame, then requires the target's
  center to fall strictly between the two anchors
  along that axis, with non-overlap between target and
  each anchor bounded by `overlap_thres = 0.3` and a
  symmetry check `|iom1 - iom2| < symmetry_thres =
  0.5`, plus a `distance_thres = 1` cap on each 1D gap.
  If those pass, it also requires actual footprint
  overlap-in-projection (`between_iom = 0.05`) with
  both anchors; if the target and anchors don't overlap
  in the horizontal plane it falls back to checking a
  vertical (z) between-ness with the same
  `overlap_thres`/`between_iom` logic. This is by far
  the most geometrically involved tool in the codebase.
- `find_near(target_name, anchor_id)` - the version
  actually registered as a tool is pure Euclidean
  distance ranking: for each candidate with a matching
  name, take 3D centroid distance to the anchor's
  centroid, sort ascending, and return the ordered ID
  list ("closest to furthest"). A second, unregistered
  implementation, `find_near_old`, exists in the file
  and does proximity-of-bounding-box-corners with a
  `near_thres = 1` threshold, but it is dead code - not
  in `self.tools` - so `find_near` in the shipped system
  is a plain distance sort, not a proximity threshold
  test.
- `find_left(target_name, anchor_id)` /
  `find_right(target_name, anchor_id)` - view-dependent.
  Takes the midpoint of anchor and target centroids,
  snaps it to the nearest point in the precomputed
  free-space point cloud (`self.freespace`), then forms
  unit vectors from that free-space point to the anchor
  and to the target (projected to the x-y plane) and
  takes their 2D cross product: positive -> left,
  negative -> right. This matches the paper's stated
  rationale for handling Nr3D-style statements with no
  explicit observer viewpoint: "we ... assume the
  relationship is unambiguous from any feasible viewing
  direction," using the nearest free-space point as a
  stand-in observer position.
- `order_left_to_right(target_name)` - same
  free-space-anchor construction as `find_left`/
  `find_right`, but computes the signed angle
  (`atan2`) of each target relative to that anchor
  point and sorts by angle.
- `find_above(target_name, anchor_name)` /
  `find_below(target_name, anchor_name)` - compares
  z-extents of the two boxes (target's min-z >=
  anchor's min-z for "above"; target's max-z < anchor's
  max-z for "below"), then requires either horizontal
  footprint overlap above `vertical_iom = 0.5`, or, if
  that fails, at least one pair of box corners within
  `near_thres = 1` of each other as a fallback "resting
  nearby" case.
- `order_bottom_to_top(target_name)` - sorts matching
  objects by raw z-centroid, ascending.
- `order_smallest_to_largest(target_name)` - sorts by
  `largest_face` field, ascending.
- `find_objects_near_room_corner(target_name)` -
  derives the four room corners from the min/max x
  and y extents of the raw registered point cloud
  (`self.pcl`), then flags an object if its centroid or
  any bbox corner is within `near_thres = 1` of one of
  those four corners.
- `notepad(thoughts)` - not spatial; a scratch-pad tool
  so the LLM can write out chain-of-thought between
  other tool calls; always returns "Noted and please
  proceed with the next step!".
- `command_robot(list_of_commands)` (live-navigation
  mode) / `pick_object(object_id)` (benchmark mode) -
  terminal action tools, see pipeline stage 5 above.

Functions present as **schemas but stubbed as `pass`
(non-functional no-ops)** in the fetched code:
`find_in_front_of`, `find_behind`,
`order_front_to_back`. This is an important caveat
for anyone assuming the released toolbox is complete:
front/behind relations, despite having LLM-facing
tool descriptions, do nothing in this version of the
code. If the CMU VLA Challenge's instruction-following
queries lean on "in front of"/"behind" phrasing, this
part of SORT3D cannot be borrowed as-is.

Other threshold constants defined in `__init__` but
not obviously consumed in the functions above:
`on_thres = 0.01`, `under_thres = 0.01` - present in
the class but their call sites were not found in the
fetched `tools.py`; flagging as unverified/dead or used
elsewhere in the codebase not fetched. None of these
thresholds carry explicit units in the source; given
the LiDAR/room scale elsewhere in the repo, meters is
the plausible but unverified assumption.

Object representation actually passed to the LLM (from
the in-context example in
`prompts/examples.py::get_tool_call_example_1`), 8
fields per row: `[absolute_id, relative_id, name,
caption, x, y, z, size]` - e.g. `["33", "5",
"refrigerator", "The refrigerator is white, made of
metal...", "64", "23", "9", "1.98..."]`. This is
richer than the simplified `{id, name, caption, cx, cy,
cz, size}` schema stated in the paper's prose, which
appears to describe it after collapsing the
absolute/relative ID pair to one `id` for readability.
Internally (`AgentToolbox.object_dict`), each object is
actually stored with more structure still: `name` is
itself a dict with `string` and `nyu_label` keys,
plus `image`, `centroid`, `dimensions` (length/width/
height), `heading` (yaw, used to build oriented boxes
via `get_bbox_coords_heading_xyzlwh`), and
`largest_face`.

## Models and prompts

- **Detector (real-world):** Grounding DINO + SAM 2
  (open-vocab detection + segmentation). Benchmark mode
  uses ScanNet ground-truth boxes instead. Source:
  https://arxiv.org/html/2504.18684v2 (section IV-A),
  repo README dependency list.
- **Captioner:** Qwen2-VL-7B (benchmark), quantized
  Qwen2.5-VL-Instruct-3B (on-robot). View selected by
  max CLIP similarity to the object's class label.
  Source: https://arxiv.org/html/2504.18684v2 (section
  IV-B).
- **Filter LLM:** Mistral Large 2, two sequential
  queries (extract references -> match to scene
  objects). Exact prompts, with worked pragmatic
  examples, are in
  `ai_module/src/language_planner/language_planner/prompts/object_extraction.py`.
- **Reasoning LLM:** GPT-4o (best accuracy) or Mistral
  Large 2 (open/cheaper alternative, close but slightly
  lower accuracy - see benchmark tables below). Source:
  https://arxiv.org/html/2504.18684v2 (section V) and
  repo README (`--model` argument of
  `language_planner_benchmark.py`, default `mistral`,
  paper's reported runs additionally use `gpt-4o`).
- **In-context strategy:** exactly one worked example
  is prepended to the tool-calling conversation, and the
  paper is explicit that it "does not have to be from a
  particular dataset" - i.e. it's a generic
  demonstration of toolbox usage, not a
  benchmark-specific few-shot example. The actual
  example shipped in the repo
  (`get_tool_call_example_1` in `prompts/examples.py`)
  walks through: query "Navigate to the windows in the
  kitchen" over an 11-object list -> `notepad` reasoning
  that a refrigerator is the best kitchen-anchor object
  -> `find_near(target_name="window", anchor_id=5)` ->
  tool result `[0, 1]` -> `notepad` picks window 0. The
  simplified example quoted in the paper text itself
  ("Find the computer near the desk with a printer on
  it" -> `find_below(desk, printer)` -> `find_near
  (computer, 2)`) differs from the exact one shipped in
  code, suggesting the paper's in-text example is
  illustrative rather than the literal in-context
  example used in runs; both are documented here so
  this isn't conflated as one artifact.
- **Agent runtime:** LangGraph state machine
  (`assistant` / `tools` / `instruct_retry` nodes, see
  Pipeline stage 4), not a single free-text prompt-and-
  parse call. An actor-critic variant exists in the
  repo but its use in the published results is
  unconfirmed (see Pipeline stage 4 caveat).
- **Open-source LLM note:** the paper states the
  "modular design ... allows for the LLMs to be easily
  replaced by smaller local models" as a forward-looking
  remark, not a tested result; no local/open-weight LLM
  substitution numbers are reported for the reasoning
  stage. Source: https://arxiv.org/html/2504.18684v2
  (section VI, Limitations).

## Key numbers (benchmarks, ablations, runtime)

All accuracy numbers below are as reported in the
paper on 200-sample subsets of Nr3D/Sr3D (repo ships
`data/referit3d/nr3d_test_200.csv` and
`sr3d_test_200.csv`, corroborating the "200-sample"
claim for those two; the repo does not ship an
IRef-VLA sample file locally - that data is pulled
from the separate IRef-VLA repo - so the exact IRef-VLA
eval-set size was not independently re-derived here
and should be treated as reported-only). Source for
all table numbers: https://arxiv.org/html/2504.18684v2
(section V, Tables II-V).

**Nr3D (view-dependent vs view-independent split):**

| Method | Overall | View-dep | View-ind |
|---|---|---|---|
| Transcrib3D (GPT-4o) | 65.6 | 63.3 | 66.7 |
| Transcrib3D (Mistral) | 63.8 | 57.1 | 66.7 |
| **SORT3D (GPT-4o)** | **62.0±1.2** | **56.6±0.0** | **64.3±1.7** |
| **SORT3D (Mistral)** | **61.6±0.3** | **59.4±0.9** | **62.6±0.9** |

(Supervised baselines score 62-65% overall for
reference; ZSVG3D/VLM-Grounder/CSVG are older
zero-shot baselines scoring 39-59%.)

**Sr3D:**

| Method | Overall | View-dep | View-ind |
|---|---|---|---|
| Transcrib3D (GPT-4o) | 96.5 | 88.9 | 96.9 |
| **SORT3D (Mistral)** | **92.0±0.7** | **90.9±0.0** | **92.2±0.8** |
| **SORT3D (GPT-4o)** | **92.0±0.0** | **95.5±0.0** | **91.6±0.0** |

**IRef-VLA:**

| Method | Overall | Easy | Hard |
|---|---|---|---|
| Transcrib3D (GPT-4o) | 77.5 | 82.5 | 57.5 |
| **SORT3D (GPT-4o)** | **71.8±1.8** | **71.0±1.3** | **75.0±3.5** |
| **SORT3D (Mistral)** | **69.0±0.7** | **70.0±0.8** | **65.0±0.0** |

Notable: SORT3D (GPT-4o) beats Transcrib3D (GPT-4o) by
a wide margin specifically on IRef-VLA's Hard split
(75.0 vs 57.5) despite trailing on Overall and Easy -
the authors call this "a large margin" improvement,
implying the toolbox helps most exactly when the
existing SOTA (Transcrib3D) is weakest, i.e. on harder
compositional/spatial statements.

**Ablation - effect of adding 2D captions (Table V,
Nr3D):**

| Setting | Overall | View-dep | View-ind |
|---|---|---|---|
| SORT3D w/o captions (GPT-4o) | 54.5 | 50.7 | 56.2 |
| SORT3D w/ captions (GPT-4o) | 60.5 | 56.6 | 62.3 |
| Improvement | +11.0 pt | **+11.6 pt** | +10.9 pt |

Captions give SORT3D a bigger lift (+11.0 pt overall)
than they give the Transcrib3D baseline (+4.3 pt
overall) on the same data, and the single largest
single-number effect in the whole ablation set is the
+11.6 pt gain on view-dependent statements - this
matches the number already flagged in our own
prior-art notes and is confirmed here directly from
the paper table. No other ablations (toolbox on/off,
number of in-context examples, detector choice) are
reported in the paper text extracted; that is a gap in
the paper's own evaluation, not something we failed to
find.

**Runtime/latency:** the paper reports **no explicit
per-stage timing, ms/query, or FPS numbers** anywhere
in the extracted text. What is reported is
hardware and a qualitative real-time claim: two robot
platforms, both running an onboard Intel NUC + RTX
4090 (16GB VRAM on the mecanum platform, 24GB on the
wheelchair), with captions "batch-generated only after
a user query is typed into the system to speed up
semantic mapping and decrease overall power
consumption." The repo's own hardware-requirements
section restates VRAM floors: 10GB for live semantic
mapping + live captioning, 7GB for ground-truth
semantics + live captioning, and calls out that a
WiFi connection is required on-robot to reach the
Mistral API. Given our own 10-minute-per-question
budget, the absence of latency numbers here means we
cannot borrow a timing budget from SORT3D and must
benchmark our own pipeline from scratch. Sources:
https://arxiv.org/html/2504.18684v2 (section V-C);
https://github.com/nzantout/SORT3D (README, "System
Requirements").

**Variance:** the paper explicitly flags that LLM
stochasticity "introduces variance between trials,
affecting grounding accuracy up to 6%" - the ± values
in the tables above are the authors' own reported
standard deviations across trials, not our estimate.

## Limitations

Explicitly admitted by the authors
(https://arxiv.org/html/2504.18684v2, section VI):

- **Internet/API dependency.** The system calls online
  LLM APIs, which the authors call reasonable for
  homes/offices but not for environments where such
  connectivity can't be assumed.
- **Thin evaluation set.** The authors themselves say
  "we acknowledge that our evaluation set is limited,"
  citing a lack of benchmarks that rigorously test
  online 3D referential grounding with diverse,
  attribute-rich natural language.
- **No real-time-interaction study.** They note that
  evaluating in a simulated environment with real-time
  interactions - multi-turn grounding, failure
  correction - would give deeper insight, but they
  haven't done it.
- **Two concrete failure cases** shown in the paper's
  Figure 4: (1) a spatial-filtering error where the
  model picks the desk closest to a window instead of
  the one near the whiteboard; (2) a pragmatics
  failure where the model picks the rightmost pillow
  literally, missing that the sentence implies "a
  pillow that is on the bed."
- **LLM-induced variance** up to 6% between trials
  (see above), an inherent limitation of using LLMs for
  grounding rather than a deterministic pipeline.

Additional limitations surfaced by our own repo
inspection, not stated by the authors:

- **Front/behind relations are unimplemented** in the
  released toolbox code (`find_in_front_of`,
  `find_behind`, `order_front_to_back` are all `pass`
  stubs with LLM-facing schemas but no logic). Anyone
  reusing this toolbox needs to write these themselves.
- **No LICENSE file** in the repo - reuse/redistribution
  status is unclear and should be confirmed with the
  authors before lifting code verbatim.
- **Dead/duplicate code** (`find_near_old`) suggests the
  released repo is a research snapshot rather than a
  hardened library; expect rough edges if depending on
  it directly rather than reading it as a reference.

## Takeaways for our 2026 module

- The toolbox's **function list and geometric
  definitions are directly reusable pseudocode**, not
  the LLM-agent scaffolding around them. We can
  reimplement `find_near`/`find_left`/`find_right`/
  `find_above`/`find_below`/`order_bottom_to_top`/
  `order_left_to_right`/`order_smallest_to_largest`/
  `find_between`/`find_objects_near_room_corner`
  ourselves in whatever language/stack we choose,
  independent of LangGraph or their specific LLMs -
  this is the single most portable artifact this paper
  offers us.
- SORT3D-Nav is layered on the **same base-autonomy
  family we already treat as reference**
  (`jizhang-cmu/cmu_vla_challenge_unity` and the
  mecanum variant), so its `ai_module/` directory
  layout, launch files, and the split between
  ground-truth-semantics mode vs live-semantic-mapping
  mode are a concrete structural template worth
  comparing against once we clone the actual CMU VLA
  dev kit, per our task's Phase 0 plan.
- **Captioning open-vocab detections is worth
  prioritizing early**: it is the one ablation the
  paper actually ran, and it produced the largest
  single gain in their whole evaluation (+11.6 pt on
  view-dependent Nr3D), specifically because raw
  geometry can't disambiguate "the red chair" from
  "the black chair" the way a caption can.
- **Don't assume front/behind relations come for
  free** - SORT3D's own released code doesn't implement
  them despite advertising the schemas; if our
  instruction-following queries need "in front of"/
  "behind", budget separate design/implementation time
  for that geometry (likely needs a robot- or
  object-heading-relative convention, unlike the
  view-independent tools here).
- **No real-time latency numbers exist to borrow** -
  we cannot infer whether a GPT-4o + Grounding DINO +
  SAM2 + captioner pipeline like this fits inside our
  10-minute/question budget from this paper alone; that
  has to be measured directly once we have a running
  pipeline.
- The **zero-shot, one-in-context-example design**
  directly validates the "no training data" constraint
  we are under - it is evidence from an
  organizer-adjacent group, on the same sensor rig,
  that an LLM-plus-heuristics pattern (rather than a
  trained grounding model) is viable under exactly our
  constraints, reinforcing the "boring baseline" already
  flagged as the going approach in
  `docs/prior_art/README.md`.
- Treat the released code as **algorithm reference
  only** given the missing LICENSE - don't vendor it
  into our repo without checking with the authors
  first.

## Sources

- Paper (full text, v2):
  https://arxiv.org/html/2504.18684v2
- Paper abstract page:
  https://arxiv.org/abs/2504.18684
- Code repo:
  https://github.com/nzantout/SORT3D
  (default branch `humble-wheelchair`; files fetched
  2026-07-11 via `gh api repos/nzantout/SORT3D/...`):
  - `README.md`
  - `ai_module/src/language_planner/language_planner/llm_backend/tools.py`
  - `ai_module/src/language_planner/language_planner/llm_backend/langgraph_agent.py`
  - `ai_module/src/language_planner/language_planner/prompts/object_extraction.py`
  - `ai_module/src/language_planner/language_planner/prompts/planning.py`
  - `ai_module/src/language_planner/language_planner/prompts/examples.py`
  - `ai_module/src/language_planner/language_planner/spatial_relations/bbox_utils.py`
  - repo metadata (`gh repo view nzantout/SORT3D --json
    description,licenseInfo,createdAt,pushedAt,stargazerCount`)
- Referenced base-autonomy repos (linked from the
  SORT3D README, cross-referenced against our own
  `docs/challenge_brief.md`/CLAUDE.md pointers):
  - https://github.com/jizhang-cmu/cmu_vla_challenge_unity
  - https://github.com/jizhang-cmu/autonomy_stack_mecanum_wheel_platform
  - https://github.com/gfchen01/semantic_mapping_with_360_camera_and_3d_lidar
- Prior internal note this dossier deepens:
  `docs/prior_art/README.md` (section citing SORT3D arXiv
  2504.18684 as an IROS 2025, CMU VLA-3D-authors-group
  paper).
