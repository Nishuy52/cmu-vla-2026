> **Provenance:** Imported 2026-07-11 from an independent,
> parallel research stream (separate workspace, now folded in here).
> Compiled from primary sources: entrant code read directly,
> papers fetched in full, talk videos frame-read. Conflict rule:
> for build/process questions the team docs (master_plan,
> architecture, challenge_brief, upstream_notes) stay
> authoritative; for 2025-results facts these imports are the
> newer, primary-sourced record.

# CMU VLA / VLN Challenge - prior-art research

Design-informing research pass for a 2026 entry. Compiled
2026-07-11. Every claim is cited inline; the "Open questions"
section is honest about what could not be verified.

Spot-check (2026-07-11): the two load-bearing citations were
independently re-fetched and confirmed real - Jana et al. arXiv
2606.31144 ("A Modular Vision-Language-Action Robotics Framework
for Indoor Environments," submitted 2026-06-30, an explicit CMU
VLA Challenge entry) and SORT3D arXiv 2504.18684 (IROS 2025, by
the VLA-3D authors' group at CMU). The genuine gap below (2025
top-1/top-2 identities) is a real blocker, not a citation defect.

Scope reminder: our module is the high-level reasoning layer
(scene understanding + query grounding + waypoint/answer
emission). Base autonomy is fixed. Sensors are a 360 camera
(10 Hz) + 3D LiDAR (5 Hz) + odometry, streamed live, no pre-built
map, no ground-truth semantics, scene resets between questions,
10 min/question.

## Per-source dossiers (deep dives)

This file is the bird's-eye summary. Each substantial source has
a standalone deep-dive dossier under `docs/prior_art/`, written
from the primary sources (code read directly, papers fetched in
full, talk transcript/frames). Added 2026-07-11.

2025 finalists (leaderboard order):

- [2025_1st_nros.md](2025_1st_nros.md)
  - winner (44.26), HITSZ NROS lab. Scene-graph construction +
  multi-modal frontier-exploration scoring, per a recovered
  HITSZ news article; members known, no paper/repo yet.
- [2025_2nd_reasonx.md](2025_2nd_reasonx.md)
  - 2nd (34.58), NTU+NUS. Gemini 2.5 Pro JSON planner +
  RoboRefer-8B pixel pointing + camera-ray waypoints; leans on
  the 2025 GT /object_markers topic. Only top-2 team with code.
- [2025_3rd_copypasta.md](2025_3rd_copypasta.md)
  - 3rd (30.98), CMU MRSD. Gemini state machine; prompts embed
  mathematical spatial-relation definitions; full code + talk.
- [2025_4th_url_kaist.md](2025_4th_url_kaist.md)
  - 4th (22.80), KAIST URL. Most feature-complete stack
  (YOLO-World + scene graph + frontier exploration + GPT-4o) but
  a rushed, bug-ridden build - why componentry != placing.

Other entries:

- [2025_kaist_ise.md](2025_kaist_ise.md)
  - non-finalist. Gemini 2.5 Flash + GPT-5 over a rule-based
  scene graph; directly reusable relation-rule thresholds.
- [2025_jana_modular_vla.md](2025_jana_modular_vla.md)
  - non-finalist; the only published paper entry. GT markers +
  OwlViT-as-text-encoder; largely copied from the 2024 code.
- [2025_other_entries.md](2025_other_entries.md)
  - survey of six more real 2025 entries (TAMSxGalbot, a
  no-LLM VLFM entry, RelTR scene-graph pipeline, ...).
- [2024_anand_singh.md](2024_anand_singh.md)
  - the 2024 presented entry (Georgia Tech). Code + talk-slide
  frames; exact 2024 per-scene scores recovered.

Technique references:

- [sort3d.md](sort3d.md)
  - organizer-adjacent zero-shot grounding; full spatial-toolbox
  function list with exact thresholds (and its stub gaps).
- [vla_3d.md](vla_3d.md)
  - the question-generation spec: exact relation thresholds,
  closed 15-color vocabulary, disambiguation cascade, from the
  organizers' own generation code.
- [conceptgraphs.md](conceptgraphs.md)
  - open-vocab 3D scene graphs from posed RGB-D; association
  thresholds; adaptation cost for our lidar+360 rig.
- [saynav.md](saynav.md)
  - incremental scene graph + LLM re-planning loop; the
  partial-map planning pattern for instruction-following.
- [openmask3d.md](openmask3d.md)
  - open-vocab 3D instance segmentation; offline-only, poor fit
  live; Open-YOLO 3D noted as the faster alternative.
- [3dgraphllm.md](3dgraphllm.md)
  - trained reference; quantifies "relations help" (+2.3) and
  the GT-vs-predicted-segmentation drop (~11 pts).
- [llm_grounder.md](llm_grounder.md)
  - LLM-agent-over-grounding-tools; honest low numbers show why
  free-form LLM spatial reasoning underperforms a toolbox.

---

## 1. 2025 challenge results and top approaches

### What is confirmed

- The challenge is run by CMU's "AI Meets Autonomy" group and
  results are presented at the associated IROS workshop. 2024 was
  Abu Dhabi, 2025 was Hangzhou, 2026 is Pittsburgh (Oct 1, 2026).
  Source: https://www.ai-meets-autonomy.com/
- Starting 2025 the final evaluation round moved to a real robot;
  earlier rounds stay in simulation. Any LLM/VLM/online API is
  allowed. 2025 submissions are closed.
  Source: https://github.com/HaochenZ11/CMU-VLA-Challenge/
- Scoring (from the challenge repo): Numerical /1 (exact-match,
  binary), Object-reference /2 (bounding-box overlap with GT),
  Instruction-following /6 (trajectory accuracy + constraint
  adherence + ordering, partial credit). So instruction-following
  dominates the point total - it is worth as much as the other
  two combined times three.
  Source: https://github.com/HaochenZ11/CMU-VLA-Challenge/

### 2025 top teams (partial - this is the weak spot)

- **3rd place, 2025: team "CopyPasta"** - CMU MRSD students
  Sreeharsha Paruchuri, Ishita Gupta, Parth Singh, Daksh Adhar.
  Their described stack: 3D LiDAR + 360 vision + "advanced
  vision-language reasoning" for spatial/positional queries. No
  public writeup or repo was found; only a newsletter blurb.
  Source: https://labs.ri.cmu.edu/mrsd-news/articles/
- **1st and 2nd place, 2025: NOT FOUND.** The full leaderboard
  lives on the challenge site
  (https://www.ai-meets-autonomy.com/cmu-vla-challenge), which is
  a JavaScript SPA that a headless fetch could not render (returned 404 /
  truncated shell). Presentation recordings are said to be on the
  workshop site but were not machine-readable here. This should be
  re-checked by a human opening the page in a browser.

### 2024 pointer

- A talk titled "CMU VLA Challenge Method" by **Anand Singh** is
  on YouTube (https://youtu.be/-bIMTsnbuoY), linked from the IROS
  2024 workshop page. Likely a top-2024-team method talk; content
  not transcribable here. Worth watching manually.
  Source: https://www.ai-meets-autonomy.com/iros-workshop-2024

### A concrete published 2025-era entry (most useful find)

"A Modular Vision-Language-Action Robotics Framework for Indoor
Environments" (Jana, Banerjee, Sadhu, Dasgupta), arXiv 2606.31144,
is explicitly a CMU VLA Challenge submission and is the clearest
window we have into what a real entry looks like:

- Parallel perception + language pipelines. Perception does
  LiDAR 2D grid mapping + full-coverage exploration, and builds a
  **semantic voxel map** storing geometry plus per-object feature
  embeddings (not a classic node/edge scene graph).
- Detection/embedding: **OWL-ViT** (owlvit-base-patch32) for
  open-vocab detection + CLIP-style embeddings.
- Reasoning LLM: **Gemini 2.0 Flash**, with **DeepSeek v3** as
  fallback.
- Query router classifies numerical / object-reference /
  instruction-following, then: count via VLM response parsing;
  object-ref via embedding similarity -> 3D coords -> RViz marker;
  instruction-following via LLM plan -> edit grid traversability
  -> execute.
- Reports example Q&A only, no scores/ranking. Notable engineering
  detail: exploration time tuned from ~8:42 down to ~4:17, which
  matters directly given the 10-min budget.
  Source: https://arxiv.org/html/2606.31144v1

Takeaway: at least one real entry uses exactly the
detect -> spatial-semantic map -> LLM-router pattern, with an
open-vocab detector and a frontier LLM. That is strong evidence
the "boring baseline" is the going approach (see section 4).

### Update 2026-07-11: second research pass (what changed)

Four angles run in parallel: the fork network of the 2025 repo
(HaochenZ11/CMU-VLA-Challenge, 24 forks + GitHub repo/code
search), Wayback snapshots of the challenge site, arXiv/Semantic
Scholar, and news/blog/LinkedIn search in English, Chinese, and
Korean. Findings below supersede parts of the "partial"
subsection above and several section-5 open questions; the
original text is left in place as a record.

#### 1st place 2025: NROS Lab, HIT Shenzhen (self-reported)

- The Networked RObotics and Systems Lab (NROS) at Harbin
  Institute of Technology Shenzhen (director Chen Haoyao /
  陈浩耀) announced on 2025-10-13 that its team won the 2025
  challenge - specifically BOTH tracks ("双冠": champion of the
  simulation track and of the real-robot platform track), and
  presented at the IROS 2025 "AI Meets Autonomy" workshop
  (Hangzhou, 2025-10-24). Source:
  https://www.nrs-lab.com/2025/10/13/热烈祝贺nros实验室代表队斩获cmu-vision-language-autonomy挑战赛（cmu-vla-challenge/
- Method detail is thin: "independently designed a
  vision-language-action large-model algorithmic framework"
  covering NL-instruction scene understanding, object
  spatial-semantic recognition, and autonomous trajectory
  planning. No member names, no paper, no repo (the lab's
  GitHub org, https://github.com/HITSZ-NRSL , has no VLA repo).
- Caveat: single self-published source. No HITSZ press release,
  third-party report, or organizer-side confirmation was found
  despite targeted English + Chinese searching. Treat as
  probable but uncorroborated.

#### 2nd place 2025: still unidentified (verified negative)

- No source in English, Chinese, or Korean names a 2nd-place
  team - checked news, LinkedIn, GitHub repo/code/fork search,
  arXiv, Semantic Scholar, and candidate lab sites.
- The leaderboard exists but is machine-unreachable: the live
  2025 challenge page was taken down (404s; it was a Google
  Sites property), and Wayback snapshots render the page text
  but embed the leaderboard and workshop program as IMAGES on
  lh3.googleusercontent.com behind session-gated hotlink
  protection (HTTP 400/403 through every header/proxy variant
  tried). Best snapshot - heading says "2025 Leaderboard",
  final, Nov 2025:
  https://web.archive.org/web/20251118230729/https://www.ai-meets-autonomy.com/cmu-vla-challenge
  ACTION: open that snapshot in a real browser; the leaderboard
  image may render in-page even though direct fetch fails.

#### 3rd place upgraded: CopyPasta's method is now fully known

- Repo found: https://github.com/parths5/CMU-VLA-Challenge -
  repo description: "Achieved 3rd Place at IROS 2025" (Parth
  Singh's account; corroborates the newsletter blurb).
- Full method talk with manual captions: "IROS '25: CMU Vision
  Language Autonomy Challenge - 3rd place",
  https://www.youtube.com/watch?v=kAPltAaRPk4 (uploaded by team
  member Ishita Gupta). Method, from the transcript:
  - Reasoning LLM: Gemini 2.5 Pro at every reasoning stage.
  - Hierarchical four-level state machine: Input Handler (LLM
    parses the query into goal objects + constraints, routes by
    question type) -> Explorer -> Solver (dedicated per-type
    agents) -> Verifier (closed-loop check to cut
    hallucination).
  - Perception: for numerical + object-reference they used the
    PROVIDED ground-truth object markers in sim, projecting 3D
    centroids into the 360 image and sampling pixel patches to
    recover per-object color (the markers carry no color). For
    instruction-following (no GT access) they incrementally
    built their own semantic scene graph while exploring.
    NOTE: worth re-checking against the 2026 dev kit whether GT
    object markers are still provided - this contradicts our
    "no ground-truth semantics" assumption at least for the
    2025 sim training scenes.
  - Prompts embed explicit mathematical definitions of the
    spatial-relation words ("closest to", "on top", "between")
    - independently converges on SORT3D's "don't let the LLM
    freestyle over geometry" lesson, via prompting instead of
    tool calls.
  - Exploration: A*-based planner with loop-minimizing
    heuristics and dynamic waypoint timeouts; tuned to finish
    exploration in under 4 minutes; full coverage plans in
    14/17 training scenes (timed out on large office scenes).
  - Self-reported internal metrics: 80% success disambiguating
    same-class different-color instances (numerical), 70%
    correct object marker (object-reference), 35.2% "all
    sub-goals reached" (instruction-following) - note how much
    harder instruction-following is even for a placing team.
  - Final eval on a real robot at CMU AirLab, Pittsburgh, in a
    scene "drastically different" from training scenes.
    Hardware: 16-core i9, 32 GB RAM, RTX 4090; Velodyne VLP-16;
    Insta360.

#### Other identified 2025 teams (no placing claims found)

The 2025 fork network + repo search surfaced these real,
substantial entries. Fork activity clusters 2025-09-12 to
2025-10-09, bracketing the 2025-09-15 submission deadline.

- **KAIST Urban Robotics Lab** (standalone repo, 12 stars - the
  most mature pipeline found this pass):
  https://github.com/url-kaist/Vision-Language-Autonomy
  GPT-4o (3 parallel API clients) + YOLO-World-x over ScanNet200
  classes; frontier-based exploration (occupancy map, coverage
  path, A*, multi-goal planner); semantic mapping with object
  merge, room segmentation, and a scene graph; keyframe
  "snapshot" object memory (C++); active visual grounding; and
  a "traversable authority" service that blocks map regions to
  honor "avoid X" instructions. Jackal ROS nodes imply
  real-robot testing. Contributors are KAIST students (Dasol
  Hong, Jeewon Kim, Taeyun Kim). No placing stated anywhere;
  the lab site and Korean-language search are silent - any
  top-2 guess is pure speculation.
- **ReasonX / SGTeam** (NUS - members per in-repo minutes:
  Yuxin, Chen Jie, Haoruo):
  https://github.com/Yuxin916/ReasonX_SGTeam (52 commits ahead
  of upstream). Gemini as planner over panoramic images +
  RoboRefer-8B-SFT for spatial referring + Depth Anything V2 +
  pixel-to-waypoint camera projection.
- **KAIST-ISE team** (KAIST + Ewha):
  https://github.com/CMU-VLA-KAIST-ISE/CMU-VLA-Challenge-2025
  Gemini + GPT-4o ensemble; hand-engineered scene-graph builder
  computing near/above/below/on/left/right/in-front/behind/
  between from object AABBs via geometric rules (the SORT3D
  lesson again); random-exploration module.
- **TAMSxGalbot**:
  https://github.com/ShangQingLiu/CMU-VLA-Challenge
  Vendors NaVid-VLN-CE (video-based VLN-CE model) for
  instruction-following + OpenAI API for the rest; the public
  repo's ai_module still holds only the dummy placeholder, so
  their real pipeline is not public. Needs >72 GB RAM per its
  README.
- **Grounded-SAM-2 pipeline** (team unnamed):
  https://github.com/xiaofeifei-1/CMU-VLA-Challenge (branch
  tqf_dev). GroundingDINO + SAM2 vendored, CLIP-based
  captioner, cloud-image fusion into a semantic map, active
  mapping + waypoint navigation.
- **Jana et al. paper repo** (the known arXiv 2606.31144
  entry): https://github.com/anindya-jana/CMU-VLA-Challenge -
  voxel-map semantic localization + VLM query pipeline,
  matching the paper.

#### Context facts pinned this pass

- 2025 was sponsored by AlphaZ (https://alpha-z.ai/): cash
  prizes + a workshop talk for the top 3 (challenge repo README
  and FAQ). Results presented at the 2nd "AI Meets Autonomy"
  workshop, IROS 2025, Hangzhou, 2025-10-24, Room 210C.
- Organizers (from the archived challenge page): Haochen Zhang,
  Pranav Saxena, Ivan Lin, Zhixuan Liu, Nader Zantout, Pujith
  Kachana, Jean Oh, Sebastian Scherer, Ji Zhang, Wenshan Wang.
- arXiv negative result: no paper besides 2606.31144 claims to
  be a 2025 entry, and none states a placing (arXiv full-text
  search + all 22 papers citing VLA-3D checked). SORT3D's paper
  itself never mentions the challenge.
- Anand Singh's 2024 talk (youtu.be/-bIMTsnbuoY) remains
  untranscribable - yt-dlp (2026.07.04) gets no formats or
  captions for that video. His submission code was inspected
  instead (next subsection).

#### 2024 presented entry (Anand Singh): method from code

The talk is unwatchable by machine, but the submission code is
public and was read directly:
https://github.com/AnandSingh-0619/CMU-VLA-Challenge , branch
anand_dev, commit 90dce461 "Working code for submission"
(2024-09-16). As implemented:

- Four-state state machine (INITIALIZE -> MAPPING ->
  ASK_QUESTION -> ANSWERING). Mapping is a one-shot
  full-coverage sweep BEFORE any question handling: 0.25 m
  occupancy grid from /traversable_area, sampled waypoints
  ordered by A* distances into a coverage tour.
- Semantics ride on the GROUND-TRUTH /object_markers topic - no
  learned detector at all. OwlViT (owlvit-base-patch32) is used
  purely as a TEXT encoder: each marker's class name is embedded
  as its voxel feature, and query nouns are matched by cosine
  similarity (threshold 0.85) - synonym-tolerant text-to-text
  matching ("couch" vs "sofa").
- LLM: Mistral Large (Gemini 1.5 Flash and GPT-4 code paths
  exist but are unused). The LLM classifies the question type
  (prompt-based, not prefix heuristics), extracts objects of
  interest, then reasons over the localized objects serialized
  as grid coordinates + bounding-box corners; a 465-line
  post-processor parses LLM output into the Int32 / Marker /
  waypoint ROS messages.
- Read: the 2024 presented entry is "GT markers ->
  embedding-indexed object map -> LLM over raw coordinates" -
  exactly the LLM-over-coordinates pattern SORT3D-style prior
  art warns is fragile, and the second confirmed entry (after
  CopyPasta 2025) that leans on the sim's GT marker topic
  rather than its own detector.

#### Still unverified after this pass

- NROS 1st place: self-reported only; no organizer-side or
  third-party confirmation.
- 2nd place identity: unknown everywhere. Highest-leverage next
  step: a human opening the Wayback leaderboard snapshot above
  in a browser.
- url-kaist placement: unknown; its repo and lab publish no
  claim.
- Whether GT object markers (used by CopyPasta for two of the
  three question types) exist in the 2026 dev kit: unchecked.

### RESOLVED 2026-07-11: the full 2025 leaderboard

A human-browser screenshot of the archived leaderboard (the
image that was machine-blocked, see above) settles the placings
and scores. Saved copy:
[docs/assets/cmu_vla_2025_leaderboard.png](../assets/cmu_vla_2025_leaderboard.png)
(captured from the Wayback snapshot of
ai-meets-autonomy.com/cmu-vla-challenge, 2026-07-11).

| Team | Sim (prelim) | Final |
|---|---|---|
| NROS Team | 36.03 | 44.26 (1st) |
| ReasonX | 31.42 | 34.58 (2nd) |
| CopyPasta | 22.99 | 30.98 (3rd) |
| Urban Robotics Lab @ KAIST | 22.80 | 22.80 (4th) |

All four teams advanced to the real-world evaluation round.

Implications, superseding the subsections above:

- NROS's self-reported championship is CORROBORATED: top sim
  score and top final score (their "double champion" claim
  matches winning both columns).
- **2nd place = ReasonX** - the NUS team already surveyed above,
  with a public repo and extracted method (Gemini planner +
  RoboRefer-8B-SFT spatial referring + Depth Anything V2 +
  pixel-to-waypoint projection):
  https://github.com/Yuxin916/ReasonX_SGTeam
  The 2025 top-2 gap is now closed; ReasonX is the only top-2
  team with public code.
- Urban Robotics Lab @ KAIST placed 4th (url-kaist repo above);
  its final score equals its prelim score, suggesting no
  real-world points were added - consistent with the lab's
  silence about a placing.
- Score spread: 1st-to-2nd is a large gap (44.26 vs 34.58),
  and CopyPasta closed most of its sim deficit in the
  real-world round (22.99 -> 30.98) while KAIST added nothing -
  the real-robot round can reshuffle margins but did not change
  the ordering.
- We now have method detail for ALL four finalists: NROS (thin,
  announcement only), ReasonX (repo), CopyPasta (repo + full
  talk transcript), URL-KAIST (repo).

Remaining open after this: only the NROS technical details
(no paper/repo) and the exact scoring scale/rubric behind these
numbers.

---

## 2. VLA-3D dataset takeaways

Paper: "VLA-3D: A Dataset for 3D Semantic Scene Understanding and
Navigation," arXiv:2411.03540 (Nov 2024).
Sources: https://arxiv.org/abs/2411.03540 ,
https://github.com/HaochenZ11/VLA-3D

What it is and what it enables:

- Aggregates ~7.6K indoor 3D scenes (11.6K regions) from ScanNet,
  Matterport3D, HM3D, 3RScan, ARKitScenes, plus Unity scenes -
  i.e. it mixes real scans with the same Unity engine the
  challenge runs in.
- Ships processed point clouds, semantic object + room
  annotations, **explicit scene graphs**, navigable free-space
  annotations, and ~9.6M synthetic **referring statements**.
- Object attributes stored per node: semantic class (mapped to
  NYUv2 / NYU40), oriented bounding box (center, dims, heading),
  **top-3 dominant colors** with RGB + coverage percentages, and
  front-facing direction where applicable. 477 unique object
  classes referenced.
- Spatial relations: ~8 relation families - above/below, on, in,
  near (proximity threshold), closest/farthest (ordered by
  distance), and **between** (ternary, multi-anchor). 23.5M
  heuristically generated relations across the corpus.
- Referring statements are generated to be **view-independent,
  unique, and minimal**: attributes (color, size) are added only
  when needed to disambiguate. This is a deliberate design choice
  - the benchmark rewards spatial-relation reasoning over
  viewpoint-dependent cues.

Why this matters for us:

- The dataset's own representation - object OBBs + color/size
  attributes + a small closed set of spatial relations
  (near/between/closest/on/above...) - is essentially a spec for
  the scene graph our reasoning module should build at test time.
  The challenge questions are drawn from this same relation
  vocabulary, so mirroring it (especially "between", "closest to",
  "near") is the highest-leverage modeling decision.
- Colors matter: dominant-color-per-object is a first-class
  attribute here, and the sample questions ("blue chairs",
  "orange chair", "black trash cans") lean on it. A detector
  pipeline that does not extract per-object color will lose
  numerical/object-ref points.

Related benchmark from the same group: **IRef-VLA** (arXiv
2503.17406), an interactive referential-grounding benchmark with
deliberately imperfect language - relevant if we want to stress
test robustness. Source: https://huggingface.co/papers/2503.17406

---

## 3. Prior-art techniques

Each row: core idea | what it grounds/reasons over | zero-shot vs
trained | relevance to our module. Table cells are wrapped for
readable raw source.

| Technique | Core idea | Grounds / reasons over | Zero-shot vs trained | Relevance |
|---|---|---|---|---|
| **SORT3D** (arXiv 2504.18684, github.com/nzantout/SORT3D; organizer-referenced) | Decompose referential grounding into: open-vocab 3D detection -> 2D VLM captions for attributes -> LLM filters candidate objects -> LLM does spatial reasoning by emitting calls to a heuristic toolbox (find_near, find_left, order_bottom_to_top...). | A flat object table {id, name, caption, cx, cy, cz, size}; spatial logic delegated to rule-based functions, not raw coords. | **Zero-shot.** No text-to-3D training; needs one in-context example. GPT-4o / Mistral Large 2. | **Highest.** Same sensor rig (360 cam + LiDAR), runs real-time on two ground robots, targets object-goal nav in unseen scenes. Reads like the organizers' reference design. Nr3D 62.0%, Sr3D 92.0%, IRef-VLA 71.8% (200-sample sets). Captions add +11.6pt on view-dependent Nr3D. |
| **ConceptGraphs** (arXiv 2309.16650, concept-graphs.github.io) | Build an open-vocab 3D scene graph from posed RGB-D: instance-seg regions -> multi-view CLIP fusion per object -> LVLM captions objects + infers edges. LLM (GPT-4) queries the graph for planning. | Object-centric 3D map: CLIP embeddings + captions per node, inter-object edges. | **Zero-shot / open-vocab**, no finetuning. | High. Canonical "objects + edges + LLM" recipe. But assumes posed RGB-D and multi-view fusion - our exploration must produce enough views. |
| **SayNav** (arXiv 2309.04077, ICAPS) | LLM does dynamic high-level planning while **incrementally** building a 3D scene graph of the explored area; feeds the growing graph to the LLM to generate + continuously refine step-by-step nav plans; a low-level planner executes point-goals. | An incrementally-built scene graph of only the explored region. | Zero-shot LLM planning (uses a pretrained low-level planner). | High for the **exploration + instruction-following** side. Beats an oracle baseline by >8% success on MultiON. Its "plan under partial map, refine as you see more" loop matches our unknown-scene, 10-min constraint. |
| **OpenMask3D / OpenScene** (arXiv 2306.13631; openmask3d.github.io) | Open-vocab **3D instance segmentation**: class-agnostic masks -> per-mask multi-view CLIP feature aggregation, so you can query objects by free-form text (geometry, material, affordance). OpenScene is the point-level predecessor (distills CLIP into 3D point features). | 3D instance masks with CLIP features; open-vocab text queries. | **Zero-shot** open-vocab; no closed class set. | Medium-high. A candidate **perception front-end** for building the object set our scene graph needs - stronger long-tail recall than closed-set detectors, but heavier and needs good multi-view coverage. |
| **3DGraphLLM** (arXiv 2412.18450, ICCV 2025) | Learn a 3D scene-graph token representation (k-NN object selection + relation edges) fed into an LLM's embedding space, so the LLM sees explicit inter-object semantics, not just coords. | Learnable scene-graph tokens (objects + semantic relations). | **Trained** (learnable representation + LLM finetuning). | Medium. SOTA on ScanRefer (+4.4% Acc@0.5), Multi3DRefer, Scan2Cap. Shows explicit relations beat coords-only - but the training requirement and reliance on GT/segmented instances make it a research reference, not a plug-in for a live-sensor, no-training-data setting. |
| **LLM-Grounder** (arXiv 2309.12311) | LLM acts as an agent that decomposes a referring query and calls an open-vocab 3D grounding tool (e.g. OpenScene/LERF), then reasons over returned candidates. | Open-vocab grounding tool outputs, orchestrated by an LLM agent. | **Zero-shot**, no training. | Medium. Same agent-over-tools shape as SORT3D; useful as a second data point that zero-shot LLM-as-orchestrator works for 3D referring. |

Cross-cutting pattern: the strong, deployable systems (SORT3D,
ConceptGraphs, SayNav, LLM-Grounder) are all **zero-shot LLM
orchestration over an object-centric map**. The trained systems
(3DGraphLLM, and most ScanRefer-leaderboard models) score higher
on clean benchmarks but assume GT or high-quality segmented
instances and offline scenes - a poor fit for live sensors, no
training data, and scene resets.

---

## 4. The "boring strong baseline" assessment

Pattern: detect objects -> build a 3D scene graph (objects with
attributes + pairwise/anchored spatial relations) -> LLM/VLM
reasons over the graph in text -> emit integer / bounding-box /
waypoints.

**Is this what winning/published systems actually do? Yes, with
high confidence for the published ones.**

- SORT3D (organizer-referenced, same robot rig) is exactly this,
  minus the explicit edge set - it keeps a flat object table and
  pushes relations into an LLM-called heuristic toolbox.
- ConceptGraphs, SayNav, LLM-Grounder are all instances of it.
- The one confirmed CMU-VLA-Challenge writeup we found (Jana et
  al., 2606.31144) is this pattern with OWL-ViT + a semantic voxel
  map + Gemini 2.0 Flash routing.
- VLA-3D's own ground-truth representation *is* this scene graph,
  which strongly implies the questions are answerable by
  reconstructing it.

Caveat on "winning": we could NOT verify the exact methods of the
2025 1st/2nd place teams, so "winning systems do this" is inferred
from (a) the organizer reference design SORT3D and (b) one
published entry, not from reading the actual top-1 writeup.

### Known hard parts / failure modes

1. **Open-vocab detection quality is the ceiling.** If an object
   is missed or mis-classed, every downstream count and relation
   is wrong. Long-tail objects, clutter, and small objects hurt.
   Closed-set detectors are fast but miss vocabulary; OpenMask3D
   is open-vocab but heavy and coverage-dependent. This is the #1
   risk.
2. **Spatial-relation grounding, especially "between",
   "closest", "near".** Thresholds are ambiguous (how near is
   near?), and "between A and B" is ternary and geometry-sensitive.
   SORT3D's answer - hand-written heuristic functions the LLM
   calls - is more reliable than asking the LLM to reason over raw
   coordinates. Matching VLA-3D's exact relation definitions is
   the safe move.
3. **Viewpoint / exploration selection under a 10-min clock.**
   Scene resets every question, so exploration cost is paid every
   time. Multi-view CLIP fusion (ConceptGraphs/OpenMask3D) needs
   enough coverage; too little and object features are wrong, too
   much and you blow the time budget (recall Jana et al. spent
   real effort cutting exploration 8:42 -> 4:17). SayNav's
   incremental "plan on partial map, refine" loop is the relevant
   mitigation.
4. **Color/attribute extraction.** Questions hinge on color
   ("blue chairs") and VLA-3D treats dominant color as a
   first-class attribute; a pipeline without robust per-object
   color estimation silently loses points.
5. **Instruction-following is 6 of the points and the hardest.**
   It needs path-level reasoning (order, "near X", "avoid Y")
   translated to waypoints, not just object grounding. A
   scene-graph-of-objects alone does not capture paths; you likely
   need the free-space/traversability layer plus relational
   constraints, closer to SayNav than to a pure referring-grounder.
6. **Numerical exactness is binary.** Counting is all-or-nothing;
   one duplicate or missed instance = 0. Robust instance
   de-duplication across views matters more than for object-ref.
7. **LLM hallucination / format drift.** The LLM must emit a clean
   integer, a valid marker, or a waypoint list. Toolbox/function-
   calling (SORT3D style) constrains output better than free-form
   generation.

---

## 5. Open questions / what could not be found or verified

- **2025 1st and 2nd place teams, their methods, scores, repos.**
  Only 3rd place ("CopyPasta", CMU MRSD) was found, via a
  newsletter blurb with no technical detail or repo. The
  leaderboard page (ai-meets-autonomy.com/cmu-vla-challenge) is a
  JS SPA that did not render for automated fetch. ACTION: a human
  should open it in a browser and also check the workshop
  presentation recordings.
- **2024 winner specifics.** Anand Singh's "CMU VLA Challenge
  Method" talk (youtu.be/-bIMTsnbuoY) is likely the best primary
  source; not transcribable here. ACTION: watch it.
- **No public top-team GitHub repos located.** Teams fork
  HaochenZ11/CMU-VLA-Challenge and edit only `ai_module`; forks
  from top teams were not surfaced by search. ACTION: browse the
  fork network of that repo and of the 2026 dev kit
  (Yuxin916/CMU-VLA-Challenge-2026) directly on GitHub.
- **No official leaderboard with numeric scores** was found in any
  machine-readable form for either year - only "top 3" language.
  We cannot state the score gap between approaches.
- **Jana et al. (2606.31144) reports no scores or ranking**, so we
  know the architecture of one entry but not how well it placed.
- **Real-robot final round details** (2025+) - how the sim->real
  final is scored and weighted - were not pinned down beyond "the
  final round is on the real robot."

Bottom line: the technique landscape and the baseline pattern are
well-characterized and well-cited; the single genuine gap is the
identity and exact method of the 2025 top-1/top-2 finishers, which
is gated behind a non-scrapable results page and video talks.

Update 2026-07-11: several items above are now superseded by the
"Update 2026-07-11" subsection in section 1 - 1st place found
(NROS Lab, HITSZ; self-reported), CopyPasta's repo + full method
talk found, the 2025 fork network surveyed, and the Anand Singh
talk confirmed untranscribable. Still open: 2nd place identity,
organizer-side leaderboard (image-locked; needs a human browser
on the Wayback snapshot), and NROS corroboration.

Later same day: FULLY RESOLVED via a human-browser screenshot of
the archived leaderboard - see "RESOLVED 2026-07-11: the full
2025 leaderboard" in section 1. 1st NROS 44.26, 2nd ReasonX
34.58, 3rd CopyPasta 30.98, 4th Urban Robotics Lab @ KAIST
22.80. Per-source deep-dive dossiers live in docs/prior_art/.

---

## Source index

- Challenge / org: https://www.ai-meets-autonomy.com/ ,
  https://www.ai-meets-autonomy.com/iros-workshop-2024 ,
  https://www.ri.cmu.edu/research-group-to-host-cmu-vision-language-autonomy-challenge/
- Challenge repo: https://github.com/HaochenZ11/CMU-VLA-Challenge/
- 2025 3rd place blurb: https://labs.ri.cmu.edu/mrsd-news/articles/
- Published entry: https://arxiv.org/html/2606.31144v1
- VLA-3D: https://arxiv.org/abs/2411.03540 ,
  https://github.com/HaochenZ11/VLA-3D
- IRef-VLA: https://huggingface.co/papers/2503.17406
- SORT3D: https://arxiv.org/abs/2504.18684 ,
  https://github.com/nzantout/SORT3D
- ConceptGraphs: https://concept-graphs.github.io/ (arXiv 2309.16650)
- SayNav: https://arxiv.org/abs/2309.04077
- OpenMask3D: https://arxiv.org/abs/2306.13631 ,
  https://openmask3d.github.io/
- 3DGraphLLM: https://arxiv.org/abs/2412.18450 (ICCV 2025)
- LLM-Grounder: https://arxiv.org/abs/2309.12311

Added 2026-07-11 (second research pass):

- 2025 1st place (self-reported): https://www.nrs-lab.com/2025/10/13/热烈祝贺nros实验室代表队斩获cmu-vision-language-autonomy挑战赛（cmu-vla-challenge/
- NROS lab: https://www.nrs-lab.com/ ,
  https://github.com/HITSZ-NRSL
- CopyPasta repo: https://github.com/parths5/CMU-VLA-Challenge
- CopyPasta talk: https://www.youtube.com/watch?v=kAPltAaRPk4
- Final 2025 leaderboard snapshot (image-locked):
  https://web.archive.org/web/20251118230729/https://www.ai-meets-autonomy.com/cmu-vla-challenge
- 2025 team repos:
  https://github.com/url-kaist/Vision-Language-Autonomy ,
  https://github.com/Yuxin916/ReasonX_SGTeam ,
  https://github.com/CMU-VLA-KAIST-ISE/CMU-VLA-Challenge-2025 ,
  https://github.com/ShangQingLiu/CMU-VLA-Challenge ,
  https://github.com/xiaofeifei-1/CMU-VLA-Challenge ,
  https://github.com/anindya-jana/CMU-VLA-Challenge ,
  https://github.com/AnandSingh-0619/CMU-VLA-Challenge (2024)
