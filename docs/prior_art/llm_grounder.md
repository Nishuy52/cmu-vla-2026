> Imported 2026-07-11 from a parallel research stream. See
> `docs/prior_art/README.md` for provenance and conflict rules.

# LLM-Grounder - zero-shot LLM agent that calls an
open-vocab 3D grounding tool, then reasons over the
returned coordinates to pick a referent

One-liner: an LLM plays orchestrator over a 3D
open-vocab grounding tool (OpenScene or LERF),
decomposing the query and reasoning over returned
candidate coordinates/distances - zero-shot, no
training, but absolute accuracy stays well below
trained ScanRefer baselines.

## What it is

- Paper: "LLM-Grounder: Open-Vocabulary 3D Visual
  Grounding with Large Language Model as an Agent."
  Jianing Yang, Xuweiyi Chen, Shengyi Qian, Nikhil
  Madaan, Madhavan Iyengar, David F. Fouhey, Joyce
  Chai. arXiv:2309.12311, submitted 2023-09-21, single
  version (no v2/v3). Accepted ICRA 2024.
  https://arxiv.org/abs/2309.12311
- Project page (demo video, live demo link):
  https://chat-with-nerf.github.io/
- Code: https://github.com/sled-group/chat-with-nerf
  (repo is titled "Chat with NeRF", [ICRA 2024]).
- Core idea: instead of training a text-to-3D grounding
  model, use an off-the-shelf LLM (GPT-3.5/GPT-4) as an
  agent that (a) parses a referring expression into
  object + attributes + landmarks + spatial relations,
  (b) calls an existing open-vocab 3D grounding tool to
  fetch candidate object locations, and (c) reasons in
  text over the returned coordinates/distances to pick
  the final candidate. Zero-shot - no task-specific
  training at all.

## Agent loop and tools

Four-stage loop per turn, repeated until the LLM
decides it has an answer:

- **Observation** - summarize the query plus any
  feedback from prior tool calls.
- **Reasoning** - a scratchpad for high-level planning.
- **Plan** - concrete next steps: which tool to call,
  with what text, or which candidates to compare.
- **Self-critique** - reflect on the plan, correct it
  if needed, before executing.

Two tools are exposed to the LLM:

- **Target Finder** - takes a free-form noun phrase
  (e.g. "wooden chair") and returns candidate object
  bounding boxes as centroid + size (Cx, Cy, Cz, dX,
  dY, dZ) plus object volume for each candidate.
- **Landmark Finder** - takes a landmark text query and
  a spatial relation, returns landmark bounding boxes
  plus the Euclidean distance from each target
  candidate's centroid to the landmark centroid.

The query is decomposed by the LLM into: object
category/attributes (color, shape, material),
landmark(s) and their spatial relation to the target,
and simple noun phrases that get passed to the tools.
Object volumes and landmark distances are fed back as
plain numbers in the prompt; the LLM then "holistically
evaluates the situation, in terms of spatial relation
and commonsense" to select a candidate. The paper does
not publish the exact prompt templates.

Both tool backends (below) locate candidates the same
way: cosine similarity between a CLIP text embedding
and per-point/per-voxel CLIP features, then DBSCAN
clustering of high-similarity points into bounding
boxes.

- **OpenScene** - distills 2D CLIP features into a 3D
  point cloud (point-level, no per-scene training).
- **LERF** - encodes CLIP embeddings into a neural
  radiance field (NeRF-level).

Backend trade-off the paper reports: LERF gives
"weaker overall grounding capability" and, per the
authors, "too noisy" feedback for the LLM to reason
over - the LLM-added gain with LERF as backbone is
small (+2.5 pt Acc@0.25 with GPT-4). OpenScene gives
cleaner candidates and a bigger LLM gain (+4.1 pt
Acc@0.25 with GPT-4). Read: the LLM's added reasoning
value is bounded by how clean the underlying tool's
candidates already are - garbage in, garbage out even
with a strong LLM.

## Key numbers

ScanRefer validation set, 14 scenes, 998 text-3D pairs
(paper's Table I; percentages are accuracy):

| Method | Backbone | Agent | Acc@0.25 | Acc@0.5 |
|---|---|---|---|---|
| LERF alone | LERF | none | 4.4% | 0.3% |
| LLM-Grounder | LERF | GPT-4 | 6.9% (+2.5) | 1.6% (+1.3) |
| OpenScene alone | OpenScene | none | 13.0% | 5.1% |
| LLM-Grounder | OpenScene | GPT-3.5 | 14.3% (+1.3) | 4.7% (-0.4) |
| LLM-Grounder | OpenScene | GPT-4 | 17.1% (+4.1) | 5.3% (+0.2) |

Trained baselines, same benchmark, for honest context -
these are supervised on ~36k labeled ScanRefer samples,
not zero-shot, so not directly comparable, but they
show the absolute gap:

- ScanRefer (trained): 34.4% Acc@0.25, 20.1% Acc@0.5
- 3DVG-Transformer (trained): 41.5% Acc@0.25, 28.2%
  Acc@0.5

So the best zero-shot LLM-Grounder configuration
(OpenScene + GPT-4) reaches roughly half the Acc@0.25
and about a quarter of the Acc@0.5 of a fully supervised
2021-era baseline. This is honestly modest - the value
claim is "zero-shot with no training data," not "beats
trained models."

**No Nr3D or ReferIt3D numbers are reported anywhere in
the paper** - checked the single arXiv version (v1,
2023-09-21, no later revisions exist). Only ScanRefer
is evaluated. Anyone citing Nr3D numbers for this paper
would be conflating it with a different work (e.g.
SORT3D, which does report Nr3D/Sr3D numbers - see
Takeaways below).

Ablations (paper's Table II, Figs. 4-5):

- Low visual difficulty (0 distractor objects of the
  same class): LLM adds +4.3 to +6.0 pt Acc@0.25.
- High visual difficulty (>=1 distractor): LLM adds
  only +1.9 to +3.5 pt Acc@0.25 - harder disambiguation
  needs visual cues the LLM does not have.
- The LLM's added value vs. query complexity (noun
  count) is roughly quadratic-shaped: it helps most at
  moderate complexity and the advantage shrinks again
  at very high complexity.

**Latency and cost:** no quantitative numbers (no
seconds-per-query, no token counts, no dollar cost) are
given anywhere in the paper. The Limitations section
flags both qualitatively only (quoted below).

## Limitations

Verbatim from the paper's Limitations section (Sec. V):

> "Cost: Utilizing GPT-based models as the core
> reasoning agent can be computationally expensive,
> which may limit its deployment in resource-
> constrained environments. Latency: The reasoning
> process, due to the inherent latency of GPT models,
> can be slow. This latency could be a significant
> bottleneck for real-time robotic applications where
> rapid decision-making is crucial."

Other limitations discussed in the paper:

- The LLM agent "is blind" - it only ever sees numbers
  (coordinates, volumes, distances), never pixels or
  geometry directly, so harder instance disambiguation
  that needs fine visual cues is out of reach no matter
  how good the LLM's reasoning is.
- Bounding-box precision (the Acc@0.5, tighter-overlap
  metric) is "not correctable by an LLM" - if the
  underlying grounding tool's box is imprecise, no
  amount of LLM reasoning fixes the geometry.
- Noisy tool feedback (as with the LERF backend)
  degrades the LLM's added value - the agent cannot
  reason its way out of bad candidate data.

## Takeaways for our 2026 module

- Second confirmed data point (after SORT3D, see
  `docs/prior_art/README.md` section 3/4) that a zero-shot
  "LLM as orchestrator over a grounding tool" pattern
  is a real, working approach class for 3D referring -
  not a one-off. Reinforces the "detect -> structured
  candidates -> LLM reasons over them" baseline shape
  already flagged as likely-strong in this repo.
- Contrast with SORT3D is the useful lesson, not just
  the similarity: LLM-Grounder feeds the LLM raw
  numbers (centroid, size, volume, landmark distance)
  and lets it freely reason in text about spatial
  relations ("closest to", "between", etc.). SORT3D
  instead pushes spatial-relation computation into a
  fixed heuristic toolbox (find_near, find_left,
  order_bottom_to_top...) that the LLM calls but does
  not itself compute geometry for - the LLM only
  filters/orchestrates. SORT3D's own reported numbers
  (Nr3D 62.0%, Sr3D 92.0%, IRef-VLA 71.8%, per
  `docs/prior_art/README.md`) are far higher than
  LLM-Grounder's ScanRefer numbers above, which is
  consistent with (though not a controlled comparison
  of) the "don't let the LLM freestyle over raw
  coordinates" lesson already drawn from SORT3D.
- LLM-Grounder's own ablations support the same
  lesson from the inside: gains shrink as scene/query
  difficulty rises, and the LLM cannot fix box
  precision or noisy tool candidates - the LLM adds
  value only on top of an already-decent tool, it does
  not rescue a bad one.
- The honest absolute numbers (best case 17.1%/5.3%
  Acc@0.25/0.5 vs a trained baseline's 34.4%/20.1%)
  are a useful gut-check against over-trusting
  "zero-shot LLM agent" as a silver bullet - it is a
  workable pattern, not a high-accuracy one on its own,
  and our module will need either a stronger grounding
  backend, heuristic spatial tools (SORT3D-style), or
  both.
- Latency/cost caution is directly relevant to our
  10-minute-per-question budget with live sensors and
  scene resets: an iterative, multi-turn LLM agent loop
  per referring query (observe -> reason -> plan ->
  critique, possibly several rounds of tool calls) adds
  real per-query latency risk that the paper itself
  flags but never quantifies. Worth budgeting for and
  measuring directly rather than assuming it is cheap.

## Sources

- Paper (single version, v1, 2023-09-21):
  https://arxiv.org/abs/2309.12311
- Full text used for extraction (ar5iv HTML):
  https://ar5iv.labs.arxiv.org/html/2309.12311
- PDF: https://arxiv.org/pdf/2309.12311
- Project page (demo video, live demo link):
  https://chat-with-nerf.github.io/
- Code repo: https://github.com/sled-group/chat-with-nerf
- Video demo: https://youtu.be/eO-Vaf-1R1s
- Cross-reference, SORT3D contrast and repo-wide
  baseline pattern: `docs/prior_art/README.md` (sections 3-4)
