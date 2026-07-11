> Imported 2026-07-11 from a parallel research stream. See
> `docs/prior_art/README.md` for provenance and conflict rules.

# SayNav - LLM plans over an incrementally-built scene
graph, refining the plan as more rooms are explored

One-liner: SayNav (ICAPS 2024) queries an LLM over a
subgraph of a 3D scene graph that grows as the agent
explores, re-planning whenever new rooms/objects appear,
and beats an oracle PointNav baseline by ~39 points SR
(95.35% vs 56.06%) on a ProcTHOR multi-object-search
benchmark - but every strong number depends on ground-
truth object/room positions, which we will not have.

## What it is

- Paper: "SayNav: Grounding Large Language Models for
  Dynamic Planning to Navigation in New Environments",
  Rajvanshi, Sikka, Lin, Lee, Chiu, Velasquez (SRI
  International). ICAPS 2024.
  arXiv: https://arxiv.org/abs/2309.04077
  Full text used for this dossier (ar5iv HTML):
  https://ar5iv.labs.arxiv.org/html/2309.04077
  Publisher record: https://ojs.aaai.org/index.php/ICAPS/article/view/31506
  Code: https://github.com/arajv/SayNav
- Problem framing, quoted from the intro: "Finding
  multiple target objects in a novel environment is a
  relatively easy task for a human but a daunting task
  for an autonomous agent."
- Task solved: multi-object navigation (MultiON) - given
  3 target object categories and no prior map, find one
  instance of each in an unseen house.
- Core idea in one loop: explore -> grow a 3D scene graph
  from what has been seen so far -> serialize a local
  subgraph to text -> ask an LLM for the next few
  navigation/look steps -> execute with a low-level point-
  goal planner -> repeat, re-querying the LLM as soon as
  new rooms/objects change what is knowable.

## Incremental scene graph

- Inputs claimed by the paper: RGBD images, semantic
  segmentation maps, and agent pose (location +
  orientation). 3D object coordinates are derived from
  segmented-object depth plus pose.
- Environment/sim: the paper's own benchmark is built on
  **ProcTHOR** (procedurally generated houses, "3 rooms to
  10 rooms", 108 object types, 132 episodes), which sits
  on **AI2-THOR** in the paper text. NOTE - unresolved
  conflict: the public code repo's README instead lists
  Habitat-sim + Habitat-Lab as the runtime dependency, not
  AI2-THOR directly. Not resolved in this pass; flagged as
  needing a hands-on check of the repo before assuming
  which simulator stack actually produced the paper's
  numbers.
  Repo: https://github.com/arajv/SayNav
- Graph has four abstraction levels: small objects, large
  objects, rooms, house. Every node carries a 3D
  coordinate. Edges encode topological relationships
  between levels (object-in-room, room-in-house, etc.).
- Room-type labeling is itself LLM-assisted: the paper
  states they "use LLMs to annotate and identify the
  spatial entity (room type) at the room level of the
  graph based on its connected objects at lower levels" -
  i.e. room semantics are inferred from what objects have
  been seen in it, not given directly.
- Growth model: the graph is only ever extended, never
  rebuilt - "the agent continuously expands the scene
  graph during exploration." This is exactly the partial-
  map property we care about: the graph at query time is
  always a strict subset of the true house.
- Important caveat (see "Assumptions" below): the paper's
  headline numbers use a **ground-truth (GT)** scene graph
  variant, i.e. object/room 3D positions taken directly
  from the simulator rather than estimated from noisy
  perception. A "VO" (visual-observation-derived) variant
  exists and is reported separately, with a large accuracy
  drop.

## Dynamic planning loop (prompts, refinement,
validation)

- Context management: SayNav never feeds the LLM the
  whole graph. It "extracts a subgraph from the full 3D
  scene graph" limited to "the local region centered
  around the current position of the agent," then
  converts that subgraph into a text prompt (pseudo-code-
  style function calls per the paper's Figure 4). This is
  the concrete lever for bounding prompt size as the
  explored area grows.
- Few-shot structure: the planning prompt includes "two
  in-context examples inside the prompt to constrain the
  LLM-generated plans," plus the current subgraph text,
  the list of not-yet-found target objects, and the
  current room's identity. The LLM is asked to emit step-
  by-step pseudocode using `navigate` and `look` actions
  with arguments.
- Re-planning trigger: unverified in exact mechanical
  detail (headless-fetch extraction of the paper's Algorithm 1
  was inconsistent across passes on line numbers - do not
  trust exact line references). What is consistently
  stated in the text across passes:
  - a `plan_needed` condition gates whether a fresh LLM
    call happens versus continuing execution of the
    current plan;
  - re-planning is triggered on: (a) looking around and
    updating the unfound-objects list / room-type
    identification, and (b) the previous plan failing or
    running out of feasible next steps;
  - the system is explicitly described as "robust to
    failures in the low-level planner, by making regular
    plan updates" - i.e. re-planning is the mechanism that
    absorbs low-level execution failures, not a separate
    recovery module.
- Feasibility gating: before/alongside planning, SayNav
  "computes the feasibility of completing the current goal
  based on the room type" - e.g. it can "skip the restroom
  when looking for a spoon." This feasibility check is
  LLM-based per the paper's Figure 6 prompt (not a hand-
  written lookup table), though this dossier could not
  independently confirm that Figure 6 content beyond the
  headless-fetch extraction - treat "LLM-based" as likely-
  correct but not hand-verified against the PDF figure
  itself.
- Plan validation: **this is the weakest-documented part
  of the paper for our purposes.** Repeated extraction
  passes found no separate validation stage described in
  prose - the generated plan appears to go straight to
  execution ("the LLM-generated plan is then executed by a
  pre-trained low-level planner"). The only checked thing
  found is a type dispatch (`if action.type == 'navigate'`
  / `'look'`) in the execution loop, which is plan
  *dispatch*, not plan *validation* in the sense of
  checking grounding against the scene graph before
  running it. Mark this explicitly: **we could not verify
  that SayNav has a true validation step separate from
  execution** - this may be a gap in the paper itself, or
  an extraction miss on our side; worth reading the actual
  PDF figures/algorithm block directly rather than relying
  on further automated fetches.
- Termination: when all doors/rooms have been explored
  without finding the remaining targets, the system
  returns a "Task Failed" state rather than looping
  forever.

## Key numbers

All numbers below are from the MultiON-style benchmark
the paper itself constructs on ProcTHOR (132 episodes,
3-10 rooms, 3 target object categories per episode, action
space = turn-left/turn-right/move-forward/stop/look-
around). Metrics: SR = success rate (all 3 objects found),
SPL = success weighted by path length, Kendall-Tau (tau) =
similarity between found-order and some reference order.

| Configuration | SR (%) | SPL | Kendall-tau |
|---|---|---|---|
| Oracle PointNav baseline (strong GT assumptions) | 56.06 | 0.49 | - |
| SayNav, GT graph + oracle nav (OrNav), GPT-3.5 | 95.35 | 0.43 | 0.70 |
| SayNav, GT graph + learned PointNav (PNav), GPT-3.5 | 80.62 | 0.32 | 0.72 |
| SayNav, visual-observation (VO) graph + OrNav, GPT-3.5 | 71.32 | 0.48 | 0.56 |
| SayNav, VO graph + PNav, GPT-3.5 | 60.32 | 0.34 | 0.62 |
| SayNav, GT graph + OrNav, GPT-4 | 93.93 | 0.46 | 0.76 |

- Headline claim: best SayNav configuration (95.35% SR)
  beats the 56.06% oracle-PointNav baseline by "more than
  8%" per the abstract's own framing - note the abstract's
  "8%" language is about relative framing in the paper's
  own words; the raw absolute gap in the table above is
  ~39 percentage points, so read the abstract's "8%" as
  the authors' chosen phrasing, not as 39 - treat this
  discrepancy as a flag to double check the abstract's
  precise wording against the PDF rather than either
  number found so far.
- Low-level PointNav policy (imitation-learned via DAgger,
  no pretraining, ~7x10^5 simulation steps, GroupNorm
  ResNet18 + 2-layer 512-hidden GRU): 84.5% success / 0.782
  SPL on unseen validation scenes for long-range point-
  goals; 98.5% success / 0.930 SPL for short-range, single-
  room point-goals.
- GT -> VO degradation: SR drops from 95.35% to 71.32%
  (OrNav case) when switching from ground-truth object
  positions to positions estimated from visual
  observation. Paper attributes this to inaccurate 3D
  position estimates from partial views and to a hard
  filter that drops objects under 20 pixels in the
  segmentation mask (small objects get missed entirely).
- LLM-as-memory ablation: replacing the scene graph's
  book-keeping with an LLM-managed conversational memory
  (LangChain "Conversational Chains") drops GT+OrNav SR to
  77.86% under GPT-3.5, but recovers to 93.93% under GPT-4
  - i.e. the explicit scene-graph memory is load-bearing
  more for weaker LLMs than for stronger ones, per this
  ablation.
- No comparison to a human baseline or to classical non-
  LLM baselines (e.g. frontier-based exploration) was
  found - the only baseline is the oracle PointNav row
  above.

## Assumptions that do not transfer to our setting

- **Ground truth object/room positions.** The paper's best
  numbers (95.35% SR) use a scene graph built from
  simulator ground truth, not estimated from sensors. Our
  challenge gives live sensors only, with scene resets and
  no privileged access to object 3D positions - we should
  expect something closer to SayNav's own VO-degraded
  numbers (71.32% SR case) as the realistic ceiling for a
  SayNav-style approach, not the GT numbers.
- **Oracle low-level navigation option (OrNav).** SayNav's
  best runs use an A*-oracle low-level navigator with
  access to the simulator's reachable-position graph. Our
  base-autonomy stack is fixed and presumably not an
  oracle; the PNav (learned, imitation-trained) rows are
  the more comparable regime, and even those assume a
  policy trained specifically on the target sim
  distribution (ProcTHOR/AI2-THOR or Habitat, whichever it
  turns out to be), not our Unity scenes.
  Repo: https://github.com/arajv/SayNav
- **Small, closed target set known up front.** MultiON
  gives 3 target object categories at episode start; our
  instruction-following task requires open-ended natural-
  language instructions parsed into waypoints, a strictly
  harder grounding problem than "navigate to any instance
  of category X."
- **Segmentation ground truth / near-ground-truth.**
  Semantic segmentation maps are treated as a given input.
  Our stack's actual detector quality/latency was not yet
  characterized when this was written and may be
  considerably noisier, especially for small
  objects - which is precisely the failure mode SayNav's
  own ablation already flags (sub-20px objects dropped).
- **No stated time budget per episode.** SayNav does not
  appear to optimize under anything like our 10-minute-
  per-question wall clock; its re-planning loop cost (LLM
  latency per query, roughly one call per room/decision
  point per the paper's description) is not benchmarked
  against a time limit anywhere found in this pass.
- **Single continuous exploration run per episode**, not
  reused across separate question types the way our three
  question types (numerical / object-reference /
  instruction-following) might need to share one scene
  understanding.
- **Simulator identity itself is unresolved** (AI2-THOR/
  ProcTHOR per the paper text vs Habitat-sim per the repo
  README) - flagged above, not resolved here.

## Takeaways for our 2026 module

- The reusable mechanism, independent of SayNav's specific
  numbers, is the **partial-map plan-refine loop**: keep a
  running structured representation of what has been
  explored (their scene graph), extract only the locally
  relevant slice for the current LLM call (their subgraph-
  around-current-position), and re-invoke the LLM whenever
  the slice changes materially (new room, new objects,
  plan failure) rather than committing to one long-horizon
  plan up front. This maps directly onto our instruction-
  following task's "path-level reasoning under partial
  maps" requirement called out as this task's motivation.
- Their four-level graph (small object / large object /
  room / house) is a reasonable starting schema to borrow
  for our own scene representation, provided we build it
  from our own (non-oracle) sensor stream rather than
  assuming ground truth.
- Their feasibility gate ("skip the restroom looking for a
  spoon") is a cheap, LLM-based commonsense prior worth
  replicating - it prunes search without needing a learned
  model, which suits our "no prior environment knowledge"
  constraint.
- The GT-vs-VO gap (95.35% -> 71.32%) is the single most
  important number to keep in mind when estimating our own
  likely performance from a SayNav-style design: expect
  the realistic version of this approach to land well
  below the paper's headline figure once real perception
  noise and no-oracle-navigation are factored in.
- The undocumented/thin "plan validation" step is a real
  gap worth designing around rather than copying blind -
  if we build a similar planner, we likely need our own
  explicit validation (e.g. checking a proposed waypoint
  target actually exists in the current scene graph slice
  before dispatching to the base-autonomy stack), since
  SayNav's paper does not clearly demonstrate one.
- Before leaning further on this paper, resolve the two
  flagged unknowns by reading the actual PDF/repo directly
  rather than further automated fetches: (1) AI2-THOR/
  ProcTHOR vs Habitat-sim as the true runtime, (2) whether
  a genuine plan-validation stage exists in Algorithm 1 or
  the appendix figures.

## Sources

- arXiv abstract: https://arxiv.org/abs/2309.04077
- Full text (ar5iv HTML, used for all quoted/paraphrased
  detail above): https://ar5iv.labs.arxiv.org/html/2309.04077
- ICAPS 2024 publisher record:
  https://ojs.aaai.org/index.php/ICAPS/article/view/31506
- Code repository (setup/dependency claims re: Habitat-
  sim vs AI2-THOR): https://github.com/arajv/SayNav
- SRI project page (not independently opened in this pass,
  found via search - recheck directly):
  https://www.sri.com/ics/computer-vision/saynav
