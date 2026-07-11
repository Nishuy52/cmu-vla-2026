# Questions

Running FAQ for this task. Newest first.

---

## Q7 (2026-07-11): Where did the "45 points max" figure come from,
and is the ambiguity actually resolved?

### Answer

Traced via the session transcripts (not derivable from docs alone).
45 was never in the README - it came from the very first WebFetch
summarization of the dev-kit GitHub page, before the repo was
cloned. That summary made two compounding errors: (1) it summed the
three per-type maxes (numerical /1, object ref /2, instruction /6)
into "9 points per question," as if one question could score on all
three axes instead of only its own type's axis; (2) it then
multiplied `9 * 5` using 5 as the test-set total instead of
questions-per-scene, instead of `9 * 15` (3 scenes * 5 q/scene) -
landing on 45 by coincidence of two wrong numbers, not one. That
number was written verbatim into the first draft of
docs/challenge-rules.md. Once the dev kit was actually cloned and
questions.json inspected directly (session f70fa724), the corrected
per-type math (1*1 + 2*2 + 2*6 = 17/scene, confirmed identical
across all 15 dev scenes) replaced it everywhere: 17 * 3 scenes =
51. All four docs (challenge-rules.md, io-contract.md, task.md,
this file's Q6) are already consistent on 51 as the working number.

### Decision / Follow-up

No doc changes needed - 45 doesn't appear as a live claim anywhere,
only as "the earlier estimate" being corrected. The one genuine
open item is still correctly flagged UNVERIFIED: whether the 3
held-out test scenes mirror the dev scenes' 1/2/2 distribution.
Lesson for later WebFetch-based extraction: don't trust summarizer
arithmetic on scoring/quantity questions - re-derive from primary
source data (here, questions.json) once available.

---

## Q6 (2026-07-11): How does the dev kit actually signal question
type and "exploration done"? (closes the Q3 open gap)

### Answer

Verified from the cloned dev kit (docs/io-contract.md). Question
type is NOT labeled: one raw std_msgs/String arrives on
/challenge_question at 1 Hz, one question per system launch. The
dummy model infers type from the string prefix only ("Find"->object
reference, "How many"->numerical, else->instruction-following).
That heuristic is brittle - real questions.json entries like "The
lantern between..." or "Count the number of..." break it - so our
classifier must parse the text (LLM), not the prefix. There is also
NO explicit exploration-done signal: the module just publishes to
the matching output topic when ready, within 10 min. Instruction-
following streams Pose2D waypoints closed-loop, advancing when the
vehicle is within ~1.0 m of the current one; "done" = last waypoint
reached. Also corrected scoring: every scene is 1 numerical + 2
object + 2 instruction, so max is 51 (not 45) if the test mix
matches dev.

### Decision / Follow-up

Q3 gap closed for design purposes. Remaining unknowns need the live
sim / organizers (test-scene mix, instruction scoring internals,
whether the marker's semantic label must match) - listed at the
bottom of docs/io-contract.md. Live end-to-end sim run deferred to
a Linux + ROS Jazzy + Nvidia GPU box.

---

## Q5 (2026-07-11): Can we circumvent the scene reset between
questions by persisting the map via an external API?

### Answer

Mechanically yes (network access is allowed, so state could be
shipped out and retrieved by fingerprinting the scene), but it is
rule circumvention, not a hack: "information collected from
previously exploring the scene is not retained" is a stated
evaluation condition - the reset is the test. Submissions are
reviewed (GitHub repo + Docker image, cash prizes, real-robot
phase), phone-home persistence is easy to spot, payoff is small
(amortized exploration across 5 questions/scene) vs. DQ risk, and
it doesn't transfer to the real-robot phase. Legitimate version of
the instinct: treat exploration efficiency as a scoring dimension -
question-conditioned exploration, offline priors from dev scenes /
VLA-3D baked into weights or prompts (general knowledge = legal;
scene-instance state across resets = not), fast incremental scene
graph rebuilt per question.

### Decision / Follow-up

Don't pursue. Exploration efficiency goes on the Phase 2 design
agenda. For any future borderline idea, email organizers for a
written OK first.

---

## Q4 (2026-07-11): Is JEPA suited to this problem's shape?

### Answer

No - different problem shape from the challenge's central
bottleneck. JEPA (I-JEPA/V-JEPA) is self-supervised: predict the
latent embedding of a masked/future patch or frame from context,
no language involved in pretraining. Native use (V-JEPA 2): latent-
space model-predictive control - roll the world model forward,
compare to a goal state, pick actions. This challenge's bottleneck
is grounding natural-language referring expressions and spatial
relations into a scene representation - a vision-language alignment
problem. JEPA latents aren't language-aligned like CLIP-family
embeddings; building that alignment is a research project, not a
drop-in. Narrow legitimate fit: the exploration/next-best-view
decision ("what's likely around this corner") is a video-dynamics-
prediction problem, i.e. JEPA's native shape - viable as an
exploration-policy signal, not as the grounding backbone.

### Decision / Follow-up

Confirms the existing parked-idea call in
docs/approach-roadmap.md (JEPA: stretch experiment vs. a working
baseline in Phase 4, never the foundation) with the underlying
"why." No action - stays parked until Phase 4.

---

## Q3 (2026-07-11): Is the robot blind between questions, or does
it already have scene data? What do inputs/outputs actually look
like end to end?

### Answer

Not blind, not pre-mapped. Per docs/challenge-rules.md the robot
gets a continuous live stream (360 camera 10 Hz, registered + raw
lidar 5 Hz, terrain maps 5 Hz, odometry 100-200 Hz) - no pre-built
point cloud, no ground-truth semantics, no traversable-area data.
Scene resets between questions (no memory carries over), so every
question starts exploration from zero - this is why "active
navigation to strategic viewpoints" is in the problem statement.
Output is three separate topics gated by question type, not one
pipe: numerical -> single Int32 on /numerical_response; object
reference -> a Marker (bounding box, "highlight") on
/selected_object_marker, scored by bbox overlap; instruction-
following -> repeated Pose2D waypoints on /way_point_with_heading,
streamed as the path is traced.

### Decision / Follow-up

Open gap not resolvable from docs alone: how the module learns
which question type is active and how exploration-vs-answer timing
is signaled. Needs Phase 0b (clone dev kit, run sim end-to-end) -
already on the task Todo.

---

## Q2 (2026-07-11): What are the problem constraints - speed,
technical / functional / non-functional requirements?

### Answer

Mostly not answerable from the public challenge page; it is
investigation work (Phase 0 of the roadmap), not speculation.
Load-bearing unknowns: evaluation metric per task type (partial
credit, distance thresholds, tie-breakers), time limit per
episode, compute environment at eval time, exact dev-kit API
contract, and whether closed-source model APIs (GPT/Claude) are
allowed at evaluation - that last fork decides between
"orchestrate frontier VLMs" and "everything runs locally".
Primary sources to chase: challenge rules / registration
materials, dev-kit README, 2025 phase results, organizer comms.

### Decision / Follow-up

Answer these before any architecture commitment. See
docs/approach-roadmap.md Phase 0.

---

## Q1 (2026-07-11): What are the data inputs?

### Answer

Per episode from the Unity sim: a natural language query plus
synthesized 360-degree camera and 3D LiDAR data. Outputs: terminal
text for numerical questions, waypoint coordinates for navigation
tasks (out-of-bounds waypoints auto-snapped to traversable areas).
Offline: the VLA-3D dataset - point clouds, object/region labels
with attributes, scene graphs, 9M+ referring statements, covering
the 15 dev scenes. Unknown details (exact message format, whether
question type is labeled, ground-truth access at dev time) need
the dev kit.

### Decision / Follow-up

Confirm exact I/O contract from the dev kit in Phase 0.
