> Imported 2026-07-11 from a parallel research stream. See
> `docs/prior_art/README.md` for provenance and conflict rules.

# ConceptGraphs - open-vocabulary 3D scene graphs built
by fusing 2D foundation-model outputs across posed RGB-D
views, queried by an LLM at inference time

One-liner: ConceptGraphs turns a posed RGB-D video into an
open-vocabulary 3D scene graph (per-object point clouds +
CLIP features + LLaVA/GPT-4 captions + GPT-4-inferred
edges), then answers queries by serializing the graph to
JSON and asking GPT-4 to pick relevant nodes - a close
structural match to our "detect -> object-centric 3D map ->
LLM reasons over it" baseline, but built on assumptions
(known camera pose, dense multi-view RGB-D, offline/offboard
GPT-4 calls) that do not hold for our live 360-camera +
lidar + odometry rig.

Paper: Gu, Kuwajerwala, Morin, Jatavallabhula, Sen, Agarwal,
Rivera, Paul, Ellis, Chellappa, Gan, de Melo, Tenenbaum,
Torralba, Shkurti, Paull. "ConceptGraphs: Open-Vocabulary 3D
Scene Graphs for Perception and Planning." ICRA 2024,
pp. 5021-5028. arXiv:2309.16650,
https://ar5iv.labs.arxiv.org/html/2309.16650 (verified via
headless fetch, full text). Project page:
https://concept-graphs.github.io/. Code:
https://github.com/concept-graphs/concept-graphs.

## What it is

- Builds a 3D scene graph where each **node** is an object
  instance (3D point cloud + fused CLIP feature + natural-
  language caption + bounding box) and each **edge** is a
  natural-language spatial relationship between two objects
  (e.g. "a on b", "b in a"), inferred by an LLM rather than
  hand-coded rules.
- Explicitly open-vocabulary: no closed object-class list,
  no 3D-labeled training data. It leans entirely on 2D
  foundation models (SAM, CLIP, LLaVA, GPT-4) applied frame-
  by-frame, then fused geometrically in 3D.
- Positioned for both perception (semantic mapping, object
  retrieval) and planning (the graph is queryable by an LLM
  to support navigation/manipulation task planning).
- Real-robot demos exist (Clearpath Jackal UGV, Boston
  Dynamics Spot Arm) but are limited in scope - see below.

## Pipeline (association, fusion, captioning, edges)

Per-frame processing, then incremental 3D fusion across the
posed RGB-D sequence:

1. **Class-agnostic instance segmentation per frame**:
   Segment Anything (SAM) produces class-agnostic 2D masks;
   these are back-projected to 3D using depth + known camera
   pose to get a per-detection partial point cloud.
2. **Per-detection semantic feature**: a CLIP image encoder
   embeds the cropped detection region into a visual-
   language feature vector.
3. **Cross-view association (the merge decision)**: for each
   new detection `i` in the current frame, compare against
   each existing 3D object `j` already in the map using a
   combined geometric + semantic similarity:
   - Geometric term `phi_geo(i,j)`: the proportion of points
     in the new detection's point cloud that have a nearest
     neighbor in object j's point cloud within a distance
     threshold `delta_nn = 2.5 cm` (a nearest-neighbor
     overlap ratio, not IoU or Mahalanobis distance).
   - Semantic term `phi_sem(i,j)`: normalized cosine
     similarity between the new detection's CLIP feature
     `f_t,i` and the object's fused CLIP feature `f_o_j`,
     rescaled to [0,1] via `(f_t,i . f_o_j) / 2 + 1/2`.
   - Combined score: `phi(i,j) = phi_sem(i,j) + phi_geo(i,j)`.
   - Decision rule: match the new detection to the existing
     object with the *highest* combined score; if the best
     score is below association threshold `delta_sim = 1.1`,
     instantiate a *new* object node instead of merging.
     (Note: with both terms in [0,1], `1.1` means association
     requires meaningfully high scores on both geometry and
     semantics together - not one term alone.)
4. **Feature fusion on match**: when a detection is merged
   into object `j`, its CLIP feature is folded in via a
   running average weighted by the object's detection count:
   `f_o_j <- (n_o_j * f_o_j + f_t,i) / (n_o_j + 1)`, and
   `n_o_j` increments. Point clouds are similarly merged/
   accumulated (geometric map update via the fused point
   sets); the paper's public repo uses GradSLAM for the
   underlying 3D reconstruction/registration substrate
   (per README, exact role vs. externally supplied pose not
   fully disentangled - flagged as unverified below).
5. **Node captioning**: for each finalized 3D object, the
   system selects up to its **best 10 views** (multi-view
   selection, not all frames) and captions each with the
   open LVLM **LLaVA-7B** (tested against LLaVA-7B-v0 at a
   pinned commit; README warns later LLaVA versions "may
   require some adaptations"). The per-view prompt used is
   reported simply as **"describe the central object in the
   image."** These multiple per-view captions are then
   consolidated by **GPT-4**, which is prompted to
   "summarize the initial captions into a coherent and
   accurate final caption" for the object node.
6. **Edge inference**: candidate object pairs are pruned
   using 3D bounding-box proximity/IoU and reduced to a
   **minimum spanning tree** over the object graph (so GPT-4
   is not asked about every O(n^2) pair). For each retained
   pair, GPT-4 is given both objects' captions and 3D
   locations and asked to describe the likely spatial
   relationship as a short phrase (e.g. "a on b", "b in a"),
   with an accompanying free-text explanation.

Model roster used end-to-end: SAM (segmentation), CLIP
(per-detection embedding + fusion), LLaVA-7B (per-view
captioning), GPT-4 (caption consolidation + edge inference +
query answering). The README recommends GPT-4 specifically,
noting "GPT-3.5 often produces inconsistent results."

## Graph schema and querying

- **Node fields**: object id; fused 3D point cloud
  (`p_o_j`); fused CLIP semantic feature (`f_o_j`); final
  natural-language caption; 3D bounding-box extents and
  center.
- **Edge fields**: a natural-language relationship label
  (subject-predicate-object style, e.g. "backpack _may be
  stored in_ closet") plus a short GPT-4-generated
  explanation string. Edges are not from a fixed predicate
  vocabulary - they are open natural language, subject to the
  minimum-spanning-tree pruning above for which pairs get
  asked about at all.
- **Query-time answering**: the graph is **serialized to
  JSON** (ids, captions, locations, and edges) and handed to
  an LLM (GPT-4) inside a system prompt at inference time.
  This is prompt-stuffing the whole graph as text, not a
  tool-calling/function-calling loop over a live API - the
  paper's example system prompt asks the model to return a
  JSON object with fields `relevant_objects`,
  `query_achievable`, `final_relevant_objects`, and
  `explanation`. The chosen object's fused 3D pose is then
  handed off to a downstream task/motion pipeline (e.g. a
  navigation goal). Two query modalities are compared: pure
  CLIP-embedding nearest-neighbor retrieval (fast, no LLM
  call) vs. this LLM-mediated query (slower, handles
  negation/composition/affordance queries CLIP alone
  cannot).

## Key numbers (benchmarks, runtime, compute)

Scene-graph construction quality on Replica scenes:
- ConceptGraphs (CG): node precision **71%**, edge precision
  **88%**, discovers **43-60 objects per scene** (varies by
  scene).
- Ablation without some component, "CG-detect"-style variant
  (paper labels it **CG-D**): node precision **61%**, edge
  precision **91%**.

Open-vocabulary semantic segmentation on Replica:
- ConceptGraphs: mean accuracy (mAcc) **40.63**, frequency-
  weighted mIoU (F-mIoU) **35.95**, beating ConceptFusion's
  **24.16 mAcc** baseline while using a smaller memory
  footprint (per-object fused features vs. per-point dense
  features).

Object-retrieval top-1 accuracy by query type (range across
evaluated scenes/queries):
- Descriptive queries: CLIP-only **0.59-1.0**; LLM-mediated
  **0.61-1.0**.
- Affordance queries: CLIP-only **0.40-0.43**; LLM-mediated
  **0.57-1.0**.
- Negation queries: CLIP-only **0.00-0.26**; LLM-mediated
  **0.80-1.0**.
  (This negation gap is the paper's central argument for why
  an LLM query layer is needed on top of pure CLIP nearest-
  neighbor: CLIP embeddings cannot represent "not X" at all,
  collapsing to near-chance/near-zero, while GPT-4-mediated
  reasoning over captions recovers high accuracy.)

Runtime/compute: **not reported as per-frame or per-scene
timing anywhere found** - neither the paper text (via
headless-fetch extraction) nor the project page nor the GitHub
README gives latency numbers (fps, seconds/frame, or
seconds/scene) or a specific GPU model/VRAM figure. The
README only states it was tested with **CUDA 11.8 + PyTorch
2.0.1** as an example environment, not a hard requirement.
Robot demos describe LLaVA running **offboard on a desktop**
and GPT-4 accessed via the **remote API** - i.e. no onboard/
embedded inference is demonstrated, and no wall-clock number
for a query round-trip is given in any source checked. This
is a real gap for our purposes (10 min/question budget) and
should be treated as **unverified/unknown**, not assumed
fast.

## Limitations and adaptation cost for our sensor rig

Paper-stated limitations:
- "Node captioning incurs errors due to current limitations
  of LVLMs like LLaVA."
- "Occasionally misses small or thin objects and makes
  duplicate detections."
- Explicitly flags "computational and economic costs" from
  running multiple LVLM inferences per object plus one or
  more proprietary LLM (GPT-4) calls per query/edge.
- README separately warns of **domain gap**: applying the
  released pipeline to AI2-THOR-like environments, "the
  performance of ConceptGraphs may be worse than other
  datasets reported" - i.e. the Replica numbers above should
  not be assumed to transfer to a new Unity sim domain like
  ours without re-validation.

Required assumptions that our rig does not satisfy:
- **Known/accurate camera pose per frame.** The whole
  association step (`phi_geo`) depends on placing each new
  detection's point cloud into a shared global frame, which
  requires trustworthy extrinsics. The paper assumes RGB-D
  SLAM/SfM-quality pose; the public repo also references
  GradSLAM for reconstruction, but it is **unverified from
  the README alone** whether the released code can consume
  externally supplied pose (e.g. our odometry stream) versus
  needing to run its own SLAM/registration - this needs a
  direct code read of the repo (`slam` / `reconstruct`
  entry points), not inferred from the README text fetched
  here.
- **Depth per view.** Association and point-cloud fusion are
  built on RGB-D; we have 3D lidar, not per-pixel depth
  registered to the 360 camera. Adapting would require
  either (a) projecting lidar returns into each camera
  view to synthesize a depth image per frame, or (b)
  reworking `phi_geo` to compare lidar point-cloud clusters
  directly against camera-derived detections rather than
  RGB-D-derived point clouds - a nontrivial re-plumbing of
  the association step, not a drop-in swap.
  Comment: since our real sensor is lidar-native, projecting
  lidar into the camera frame is likely the lower-risk path
  in changed terms.
- **Dense multi-view observations, pose-registered ahead of
  time.** The paper's captioning step explicitly selects
  "the best 10 views" per object *after* the fact from a
  full posed sequence. Our setting is a live/online 10-min
  budget per question with scene resets - we cannot assume
  a pre-collected dense multi-view tour of an object exists
  before we need to answer. This pushes toward either (a)
  an online/incremental captioning trigger (caption as soon
  as enough views accumulate, not a full-scene batch pass),
  or (b) accepting lower per-object caption quality from
  sparser view counts - both are adaptations, not defaults
  the paper validates. The paper does not report an ablation
  on view count/sparsity, so the degradation curve below "10
  best views" is **unknown**, not just "worse" - could not
  find a stated minimum-viable view count.
- **Offboard/API-latency LLM calls per query and per edge.**
  Edge inference calls GPT-4 once per retained pair (after
  MST pruning) at *map-build* time, and query answering
  calls GPT-4 again at *query* time. Neither is shown to run
  within our onboard latency/compute envelope; both were
  demonstrated with desktop-class offboard compute and a
  cloud API, which is a reasonable analogue to "call our own
  LLM API from the robot's onboard compute over network,"
  but the paper gives no evidence this stays within a 10-
  minute total budget once mapping + reasoning + navigation
  are chained together.

## Takeaways for our 2026 module

- The overall shape - class-agnostic 2D segmentation, per-
  object multi-view feature fusion keyed on a combined
  geometric+semantic similarity score, LLM-generated
  captions/edges, LLM-mediated query answering over a
  serialized graph - is a strong existing precedent for our
  "detect -> object-centric 3D map -> LLM reasons over it"
  baseline (see `docs/prior_art/README.md`), not a departure from
  it. It is worth treating as the closest known reference
  implementation to adapt from rather than design from
  scratch.
- The concrete, reusable pieces: (1) a combined nearest-
  neighbor-overlap + CLIP-cosine similarity score with a
  single scalar threshold for the merge-vs-new-object
  decision (simple, tunable, no learned weights); (2) a
  running-average feature-fusion update (O(1) per new
  detection, no need to store per-view histories); (3)
  pruning candidate edges via bounding-box proximity + a
  minimum spanning tree before spending LLM calls on
  relationship inference (keeps GPT-4 edge-inference cost
  linear in object count, not quadratic).
- The CLIP-vs-LLM query gap (near-zero on negation queries
  for pure CLIP retrieval vs. 0.80-1.0 with an LLM in the
  loop) is direct evidence that our own query-answering stage
  should route through an LLM over serialized graph text
  (or equivalent), not rely on embedding-similarity lookup
  alone, especially for negation/composition-style natural-
  language instructions in our instruction-following task
  type.
- The biggest adaptation cost is the pose/depth assumption:
  our sensor rig (360 camera + 3D lidar + odometry, live,
  with scene resets) does not match ConceptGraphs' posed-
  RGB-D-video assumption. Before committing to this shape we
  would need to either fuse lidar-derived depth into each
  camera frame, or replace the RGB-D point-cloud association
  step with a lidar-cluster-based one - this is real
  engineering work, not configuration.
- Runtime/compute feasibility within our 10-min-per-question
  budget is an open question this paper does not answer -
  no latency numbers were found anywhere checked (paper,
  project page, or README). This should be flagged as a
  Phase 1/2 risk item to benchmark ourselves (e.g. GPT-4-
  class API round-trip time for edge inference + query
  answering, times expected object count) rather than assumed
  safe by analogy to this paper's offboard-desktop demos.

## Sources

- Paper (full text via ar5iv, confirms association/fusion
  formulas, thresholds, model roster, prompts, benchmark
  numbers, and stated limitations):
  https://ar5iv.labs.arxiv.org/html/2309.16650
- arXiv abstract/listing: https://arxiv.org/abs/2309.16650
- Project page (robot demo description, video links, CLIP-
  vs-LLM query framing):
  https://concept-graphs.github.io/
  - Main demo video: https://youtu.be/ie9VdL7wieY
  - Tutorial video: https://youtu.be/56jEFyrqqpo
- Code repository (dependency versions, SAM checkpoints,
  LLaVA/GPT-4 usage notes, AI2-THOR domain-gap caveat):
  https://github.com/concept-graphs/concept-graphs
- Bibliographic confirmation (venue, page numbers, full
  author list) via web search cross-reference:
  https://www.semanticscholar.org/paper/ConceptGraphs:-Open-Vocabulary-3D-Scene-Graphs-for-Gu-Kuwajerwala/174a3290ff0040e0f6a7a0b43bcd752c456350a0
  and https://pure.johnshopkins.edu/en/publications/conceptgraphs-open-vocabulary-3d-scene-graphs-for-perception-and-
- Cross-reference: this repo's own baseline framing that
  ConceptGraphs is being compared against:
  `docs/prior_art/README.md`
