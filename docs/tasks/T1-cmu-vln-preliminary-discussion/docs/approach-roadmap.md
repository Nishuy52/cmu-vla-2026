# Approach roadmap (advisor notes, 2026-07-11)

How to approach the challenge - process, not architecture.
Companion to the parked-ideas list below.

## Problem shape: episodic outer, sequential (POMDP) inner

Anchor framing (from discussion, 2026-07-11). The problem is
episodic and sequential at two different levels - the distinction
drives design:

- **Between questions -> episodic.** Each question is an
  independent episode with a hard reset: scene memory wiped, 10-min
  clock restarts, one question per system launch, nothing carries
  over (the reset is a stated eval condition, questions.md Q5). No
  cross-episode state to exploit -> RL-across-questions and
  persistent maps are off the table; every question pays full
  exploration cost.
- **Within a question -> sequential, a POMDP.** Partially observed
  (only what the 360 cam + LiDAR have covered so far) and
  sequential (each waypoint changes the next observation, which
  informs the next waypoint): explore -> observe -> update belief
  -> decide (move again or answer). This inner loop is exactly the
  "active navigation to strategic viewpoints" in the problem brief,
  and where the exploration-efficiency axis lives. Even the
  "answer-once" types (numerical, object-ref) have a sequential
  run-up before the single terminal output.

Consequence: the reset kills the outer sequence; partial
observability creates the inner one. SayNav's "plan on partial map,
refine as you observe more" fits the inner loop (see
docs/prior-art.md). Directly feeds the Phase 2 pipeline-topology
question below.

## Phases

1. **Phase 0 - Pin the problem** (days): get rules + dev kit, run
   the sim end-to-end with dummy outputs, answer the constraints
   list (metric, time limits, compute env, API contract, whether
   closed-source model APIs are allowed at eval).
2. **Phase 1 - Prior art** (parallel with 0): 2025 phase winners,
   VLA-3D paper, 2-3 scene-graph + LLM navigation papers
   (ConceptGraphs, SayNav, etc.). Don't design in a vacuum - VLN
   is a mature field.
3. **Phase 2 - Brainstorm / design session**: only now, grounded
   by 0 and 1. Output: one deliberately boring baseline
   architecture + the train-vs-compose decision.
4. **Phase 3 - Dumbest end-to-end baseline**: nonzero score on all
   three task types in the sim. Value = plumbing, real failure
   cases, a number to beat.
5. **Phase 4 - Error-analysis loop**: improvements ranked by
   observed failures, not idea appeal. Parked ideas re-enter here
   as hypotheses that must beat the baseline on the metric.

Discipline: between now and Phase 2, architecture ideas get
written down and parked, not pursued.

## Parked ideas (with evaluation)

- **Router + per-task-type specialist agents**: query
  classification is nearly trivial; the shared hard core of all
  three task types is grounding referring expressions into a
  spatial scene representation. Routing is a cheap Phase-4 add-on;
  not a foundation.
- **JEPA embeddings**: "encodes world semantics well" is
  defensible (V-JEPA 2); "latents translate directly into
  language" does not exist off the shelf - JEPA latents are not
  language-aligned (unlike CLIP-family), and building alignment is
  a research project. Stretch experiment vs. a working baseline
  in Phase 4, never the foundation.
- **Synthetic / augmented data**: only relevant if we
  train/fine-tune. If we compose pretrained components zero-shot,
  the 15 dev scenes are for evaluation, not training, and data
  volume is irrelevant. Decide train-vs-compose (Phase 2) first;
  data strategy falls out of it.
- **Exploration efficiency as a design axis** (Phase 2 agenda):
  scene resets between questions make per-question exploration a
  scoring dimension (10-min limit, time bonuses). Candidates:
  question-conditioned exploration, offline priors from dev
  scenes / VLA-3D baked into weights or prompts (legal), fast
  incremental scene graph. Cross-question state persistence via
  external APIs is circumvention - ruled out (questions.md Q5).
- **Likely-strong boring path** (for Phase 2 consideration):
  detect objects -> maintain 3D scene graph -> LLM/VLM reasons
  over it -> emit answer or waypoints. Matches what VLA-3D was
  built for and is well-published.

## Phase 2 design agenda (open questions, decide in the session)

Seeded now so nothing is lost; do NOT resolve before Phase 2.

- **Pipeline topology: parallel vs. staged** (raised in discussion
  2026-07-11). Only one question type is active per system launch
  (verified - one question/launch, type not labeled; see
  docs/io-contract.md), so the three type-specific handlers are
  mutually exclusive per episode. The open question is the internal
  structure and how state estimation + mapping are shared:
  - Shared front-end, parallel heads: run state estimation +
    exploration + object-centric map build once, then branch to a
    numerical / object-ref / instruction-following head. Simpler,
    reuses perception, but may over-explore for a cheap count.
  - Classify-first, staged/gated: classify the question, then run
    only the pipeline that type needs and let the type gate how
    much/what we explore. More time-efficient under the 10-min
    clock, but couples the classifier to exploration policy and
    fails harder when classification is wrong.
  Directly shapes - and is shaped by - the exploration-efficiency
  axis in the parked-ideas list above.
