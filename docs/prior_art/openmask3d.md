> Imported 2026-07-11 from a parallel research stream. See
> `docs/prior_art/README.md` for provenance and conflict rules.

# OpenMask3D (+ OpenScene) - open-vocab 3D instance
segmentation via frozen mask proposals + multi-view CLIP,
computed once per fully-scanned scene, not incrementally

One-liner: OpenMask3D gives strong long-tail 3D instance
recall by pairing a class-agnostic Mask3D backbone with
per-mask multi-view CLIP features, but it is a post-hoc,
whole-scene batch method (5-10 min/scene on a single GPU,
needs the full posed RGB-D sequence + reconstructed point
cloud up front) - a poor fit to run continuously inside a
10-min live episode, though it (or a faster descendant) could
run once, after exploration, as a final grounding pass.

## What it is

- **OpenMask3D** - "Open-Vocabulary 3D Instance Segmentation"
  (arXiv 2306.13631, NeurIPS 2023). Project page:
  https://openmask3d.github.io/. Paper (ar5iv full text):
  https://ar5iv.labs.arxiv.org/html/2306.13631. Code:
  https://github.com/OpenMask3D/openmask3d.
- Goal: given a 3D scene, produce class-agnostic instance
  masks and, for each mask, a CLIP embedding - so any object
  instance can be retrieved later by an arbitrary free-form
  text query, not just a fixed closed set of class names.
- **OpenScene** (its point-level predecessor) - "3D Scene
  Understanding with Open Vocabularies" (arXiv 2211.15654).
  Paper: https://ar5iv.labs.arxiv.org/html/2211.15654.
  Project page: https://pengsongyou.github.io/openscene.
  It distills CLIP into **per-point** dense 3D features (no
  instance masks at all) so any 3D point can be scored
  against a free-form text query for zero-shot semantic
  segmentation. OpenMask3D's own paper uses OpenScene as a
  baseline and improves specifically on tail-category
  *instance* recall, since OpenScene only does dense
  per-point/semantic prediction, not object instances.

## Pipeline

**OpenMask3D, per scene:**

1. **Mask proposals (class-agnostic).** Reuses the frozen
   transformer mask-decoder of a pre-trained **Mask3D**
   model (sparse-conv MinkowskiUNet backbone + transformer
   decoder), but discards Mask3D's class predictions and
   confidence scores, keeping only binary 3D instance mask
   proposals. Non-contiguous masks are split with DBSCAN
   (epsilon=0.95) into spatially contiguous clusters.
2. **Input requirement.** Needs "a set of posed RGB-D images
   captured in a scene, along with the reconstructed scene
   point cloud" and known camera intrinsics/extrinsics - i.e.
   a scene that has already been fully captured and
   reconstructed. It subsamples "1 frame in every 10 frames"
   of the RGB-D sequence rather than using every frame.
3. **View selection per mask.** For each 3D instance mask,
   compute a visibility score per frame (visible-point count
   under the camera FOV with occlusion checking), normalize
   it, and keep the top `k_view` (default 5) best views for
   that instance.
4. **2D mask refinement.** In each selected view, run the
   **Segment Anything Model (SAM, ViT-H)** with a RANSAC-like
   sampling scheme (10 rounds x 5 sampled points) to get a
   clean 2D silhouette of the instance in that view, keeping
   the highest-confidence SAM mask.
5. **Multi-scale CLIP feature extraction.** For each refined
   2D mask, take `L=3` multi-level crops (expansion ratio
   0.1) and encode each with a **CLIP ViT-L/14 (336px)**
   visual encoder (768-d features).
6. **Aggregation.** Average-pool the CLIP features over all
   crop scales and all selected views to get one CLIP
   embedding per 3D instance mask.
7. **Query time.** At inference, a free-form text query is
   embedded with the CLIP text encoder and matched against
   all precomputed per-mask embeddings by cosine similarity -
   this step is the only part that is fast/interactive.

**OpenScene, per scene (for contrast):**

1. **2D fusion branch.** Per-pixel embeddings from a 2D
   vision-language segmentation model (e.g. OpenSeg/LSeg) are
   back-projected onto the 3D surface points using posed
   RGB-D images, then averaged per point across views.
2. **3D distillation branch.** A sparse 3D conv net
   (MinkowskiNet18A) is trained (cosine-similarity loss) to
   reproduce those fused 2D features from geometry alone, so
   it can run even where no 2D image covers a point.
3. **2D-3D ensemble.** At inference, for each text query, pick
   per-point whichever branch (2D-fused or 3D-distilled) gives
   the higher CLIP similarity.
4. Output is **dense per-point features**, not instance masks;
   it needs a mesh/point cloud + posed RGB images as input,
   same reconstruction-first assumption as OpenMask3D.

## Key numbers (benchmarks, runtime, memory)

- **ScanNet200 instance segmentation (AP / AP50 / AP25):**
  OpenMask3D 15.4 / 19.9 / 23.1, vs. fully-supervised
  closed-vocab Mask3D 26.9 / 36.2 / 41.4 (expected gap, since
  Mask3D is trained on ScanNet200 labels and OpenMask3D is
  zero-shot). On **tail categories only**, OpenMask3D reaches
  14.9 AP vs. OpenScene's best of 9.9 AP - the paper's central
  claim: better long-tail instance recall than a dense-feature
  open-vocab method.
- **Replica (AP / AP50 / AP25):** OpenMask3D 13.1 / 18.4 /
  24.2 vs. OpenScene 10.9 / 15.6 / 17.3.
- **Oracle-mask ablation:** with ground-truth masks instead of
  Mask3D proposals, OpenMask3D's CLIP-feature stage alone
  reaches 29.1 AP, and beats Mask3D by +15.0 AP on tail
  classes - i.e. a large share of the error budget is in mask
  *proposal* quality, not the CLIP-feature stage.
- **Runtime:** "computation of the mask-features of a ScanNet
  scene on a single GPU takes 5-10 minutes depending on the
  number of mask proposals and number of frames." Query time
  after that is real-time (~1-2 ms) since it's just a cosine
  similarity lookup. No explicit GPU memory number is given in
  the paper; the official repo
  (https://github.com/OpenMask3D/openmask3d) exposes an
  `OPTIMIZE_GPU_USAGE` flag for memory-constrained machines,
  noted as slower than the default path - implying default
  operation assumes non-trivial VRAM headroom (unspecified).
- **OpenScene runtime/scale (for contrast):** ~24h to train
  the 3D distillation branch, then ~0.1s inference per scene
  once trained; training used a single NVIDIA A100 (40GB) for
  ScanNet/Matterport3D and 4x A100 for nuScenes.
  Zero-shot semantic segmentation mIoU (OpenSeg variant):
  ScanNet 47.5, Matterport3D 42.6, nuScenes 42.1 - "noticeable
  gap to state-of-the-art fully-supervised approaches" (e.g.
  -11.6 mIoU vs SOTA on Matterport3D), but on a 160-class
  long-tail split on Matterport3D, OpenScene beats a
  fully-supervised MinkowskiNet on mAcc (23.1 vs 18.4).
- **Follow-up speed context - Open-YOLO 3D** (arXiv 2406.02548,
  ICLR 2025 oral): explicitly targets OpenMask3D's runtime as
  the bottleneck, attributing it to "heavy reliance on 3D clip
  features, which require computationally expensive 2D
  foundation models like Segment Anything (SAM) and CLIP for
  multi-view aggregation into 3D." Open-YOLO 3D reports "up to
  ~16x speedup" over OpenMask3D and 24.7 mAP on ScanNet200 val
  at 22 seconds/scene - i.e. a same-family method exists that
  is both faster and higher-scoring, by skipping the SAM
  refinement step and reusing class-agnostic 3D instance
  projections directly.
  (https://arxiv.org/abs/2406.02548)

## Fit for live, time-budgeted use

- **Not incremental.** OpenMask3D's mask module and CLIP
  aggregation both assume the *complete* posed RGB-D sequence
  and the *complete* reconstructed point cloud are already
  available - there is no mechanism in the paper or repo for
  adding a new frame's information into an existing mask/CLIP
  index without recomputing over the accumulated sequence.
  It is a batch, post-hoc method: capture everything, then run
  the pipeline once. (Same conclusion for OpenScene's 2D-fusion
  branch, which back-projects a full posed-image set onto a
  full point cloud before the 3D branch ever runs.)
- **Budget mismatch.** 5-10 minutes of GPU compute *after* the
  scene is fully captured does not fit inside a 10-min/question
  budget that also has to cover exploration, base-autonomy
  navigation, and answer generation - unless exploration itself
  is short and the mask/CLIP pass is treated as a single
  terminal step. Even then it consumes a large fraction of the
  remaining time before any query can be answered, and per-
  question scene resets in this challenge mean this cost is
  likely paid on every question, not once per environment
  (unclear from our current docs whether a scene reset also
  resets any previously built map/index - flag as an open
  question against `docs/challenge_brief.md`).
- **Coverage dependence.** Multi-view CLIP aggregation quality
  depends on `k_view` good, unoccluded views per instance
  existing in the captured trajectory. A robot that only
  partially explores an unknown scene before its time budget
  expires will under-cover many instances, and OpenMask3D has
  no fallback for masks with zero/poor views - this is a direct
  tension with "unknown indoor Unity scenes" + tight per-
  question time limits.
- **What adaptation would be needed** to use it at all:
  1. Run mask+CLIP computation once, after an exploration phase
     is judged "done enough," not continuously - i.e. treat it
     as a terminal grounding step, similar to how the paper
     itself uses it (offline, whole-scene).
  2. Prefer a faster same-family method for this setting -
     Open-YOLO 3D's ~16x speedup (22s vs 5-10 min) is much
     closer to fitting inside a live budget, at the cost of
     depending on its own (unverified in this pass) coverage
     assumptions.
  3. Cache across questions within the same scene instance if
     the harness allows it, so the expensive mask/CLIP pass is
     amortized over the numerical / object-reference /
     instruction-following questions asked in that scene rather
     than repeated per question - contingent on confirming scene
     reset semantics (open item, not yet verified against the
     cloned dev kit per `docs/challenge_brief.md`).
  4. Consider a cheaper closed-set or lighter open-vocab
     detector for the live/incremental loop, reserving an
     OpenMask3D-class model (if time allows) as a one-shot
     precision pass late in a question's time budget.

## Takeaways for our 2026 module

- OpenMask3D is best read as **evidence that class-agnostic
  mask proposal + multi-view CLIP retrieval measurably beats
  dense per-point open-vocab features (OpenScene) on long-tail
  *instance* recall** (14.9 vs 9.9 AP on ScanNet200 tail) - a
  useful data point if our object-reference task type needs
  to name/localize instances of long-tail object categories
  that a closed-set detector would miss
  (`docs/challenge_brief.md`).
- Its own oracle-mask ablation (29.1 AP with GT masks vs 15.4
  with Mask3D proposals) says the practical error bottleneck
  is *mask proposal quality*, not the CLIP-matching idea -
  so if we ever adopt this family, the leverage is in getting
  better/faster 3D instance masks (e.g. Open-YOLO 3D's route),
  not in re-deriving the multi-view CLIP aggregation step.
- As a **candidate perception front-end**, OpenMask3D itself is
  disqualified by runtime/architecture for the live loop as
  specified; a faster descendant (Open-YOLO 3D, ~22s/scene) is
  the more realistic starting point if we want this family's
  long-tail recall inside a 10-min episode.
- This is a **parked/candidate note only** - written during
  the pre-merge research pass, before the team architecture
  was adopted; nothing here should be read as committing to
  a mask-proposal + CLIP-retrieval design (see
  `docs/architecture.md` for what was actually adopted).
- Open question to carry into Phase 1/2 discussion: does a
  scene reset between questions also invalidate any
  incrementally-built map/index, which determines whether an
  expensive one-shot pass like this could ever be amortized
  across a scene's question set. Not yet verified against the
  cloned dev kit.

## Sources

- OpenMask3D paper (NeurIPS 2023), ar5iv full text:
  https://ar5iv.labs.arxiv.org/html/2306.13631
- OpenMask3D project page: https://openmask3d.github.io/
- OpenMask3D arXiv abstract/PDF:
  https://arxiv.org/pdf/2306.13631
- OpenMask3D official code + README (input requirements,
  checkpoints, `OPTIMIZE_GPU_USAGE` flag):
  https://github.com/OpenMask3D/openmask3d
- OpenScene paper, ar5iv full text:
  https://ar5iv.labs.arxiv.org/html/2211.15654
- OpenScene arXiv abstract: https://arxiv.org/abs/2211.15654
- OpenScene project page:
  https://pengsongyou.github.io/openscene
- Open-YOLO 3D paper (ICLR 2025 oral, speed/runtime
  comparison vs OpenMask3D): https://arxiv.org/abs/2406.02548
- Cross-reference, this repo's challenge I/O docs:
  `docs/challenge_brief.md`, `docs/io_contract_crosscheck.md`
