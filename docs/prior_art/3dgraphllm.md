> Imported 2026-07-11 from a parallel research stream. See
> `docs/prior_art/README.md` for provenance and conflict rules.

# 3DGraphLLM - trained scene-graph-token LLM; the
evidence we cite is "explicit relations beat coords-only,"
not a design to reuse

One-liner: 3DGraphLLM feeds an LLM a per-object token plus
a k=2 nearest-neighbor "subgraph" of (object, semantic
relation, object) triplets instead of raw coordinates, and
reports consistent gains over a coords-only baseline across
five 3D-QA/grounding benchmarks - but it requires LoRA
fine-tuning an LLM on ScanNet/3RScan data, a separately
trained relation classifier (VL-SAT), and per-object point
clouds from GT or predicted instance segmentation. We treat
it purely as evidence for a design principle (explicit
relations > coordinates alone), not as something we could
plug into our no-training, live-sensor, Unity-sim setting.

## What it is

- Paper: "3DGraphLLM: Combining Semantic Graphs and Large
  Language Models for 3D Scene Understanding," Tatiana
  Zemskova and Dmitry Yudin (AIRI, MIPT). arXiv 2412.18450,
  ICCV 2025. Abstract/versions:
  https://arxiv.org/abs/2412.18450. Full text (v1, Dec
  2024): https://arxiv.org/html/2412.18450v1. Full text
  (v3, camera-ready, Aug 2025):
  https://arxiv.org/html/2412.18450v3.
- Code: https://github.com/CognitiveAISystems/3DGraphLLM
  (listed in the paper; verified live via web search,
  "[ICCV 2025] 3DGraphLLM is a model that uses a 3D scene
  graph and an LLM to perform 3D vision-language tasks").
- Model weights: https://huggingface.co/wingrune/3DGraphLLM
- Task family: 3D referred-object grounding (ScanRefer,
  Multi3DRefer), dense scene captioning (Scan2Cap), and 3D
  visual question answering (ScanQA, SQA3D) - all on
  ScanNet/3RScan indoor scans, using an offline reconstructed
  point cloud with per-object instance masks as input.
- Baseline it builds on: Chat-Scene (Huang et al. 2024),
  which represents a scene as a flat list of per-object
  identifier tokens with 2D+3D features but no explicit
  relation tokens. 3DGraphLLM's central claim is that adding
  explicit relation edges on top of that same object-token
  scheme improves results.

## Method

- **Per-object features** (inputs: an object's point cloud,
  from GT or predicted instance segmentation):
  - 2D: DINOv2 features aggregated over masked multi-view
    images of the object -> `Z_2d` in R^1024.
  - 3D: Uni3D encoder (point cloud features pre-aligned to
    text descriptions) -> `Z_vp` in R^1024.
  - Relation (pairwise): VL-SAT, "a method for generating 3D
    semantic scene graphs from point clouds," pre-trained on
    GT-segmented data; 3DGraphLLM uses VL-SAT's latent
    graph-neural-network feature *before* its classification
    head (not the discrete relation label) -> `Z_e` in R^512
    per object pair.
- **Graph construction.** A complete pairwise graph over n
  objects would need `2n + 3n(n-1)` tokens (29,900 for
  n=100, matching Mask3D's fixed 100 proposals) - too many
  to fit an LLM context cheaply. Instead each object keeps
  only a subgraph of its k nearest neighbors, giving
  `2n + 3nk` tokens (800 for n=100, k=2). k=2 was chosen as
  "the best trade-off between performance gains and
  computational complexity across all three tasks" (an
  appendix sweep tried k in {0,1,2,4}, capped at 4 by GPU
  memory).
- **Token sequence per object i:** identifier token `<OBJi>`
  (added to the LLM vocabulary, following Chat-Scene) + its
  2D feature, then k triplets `(F_i^v, F_ij^e, F_j^v)`
  describing object i and its k nearest neighbors.
- **Projections.** Three trainable 3-layer MLPs (2D-object,
  3D-object, relation) map the above features into the LLM's
  token embedding space.
- **Dedup filters.** Because Mask3D always emits exactly 100
  segments, nearest-neighbor lists are full of near-duplicate
  proposals; the paper adds an NMS filter (IoU threshold
  0.99) and a minimum-distance filter (1 cm) to remove
  duplicate neighbors before building subgraphs.
- **Backbone LLM.** LLAMA3-8B-Instruct (primary) or
  Vicuna-1.5-7B (used in some ablations), fine-tuned with
  LoRA (rank 16).
- **Training recipe.** Two-stage: (1) pre-train the
  projection MLPs + LoRA-adapted LLM on GT-instance-
  segmentation data; (2) fine-tune on noisier predicted-
  segmentation data (Mask3D/OneFormer3D), "considering a
  scenario where ground-truth segmentation is unavailable."
  Batch size 8, 3 epochs, initial LR 5e-6, cosine annealing,
  on 4x A100 GPUs, "approximately 24 hours."
- **Training data.** ScanNet (1201 train / 312 val scans:
  ScanRefer, Multi3DRefer, Scan2Cap, ScanQA, SQA3D) plus
  3RScan (1175 train / 157 val scans: RioRefer for grounding,
  3RQA for QA; RioRefer + Nr3D reused as extra captioning
  references).

## Key numbers (incl. relations-ablation and
GT-vs-predicted-segmentation drop)

Camera-ready main comparison (v3, Table 2; ScanRefer
Acc@0.5, Multi3DRefer F1@0.5, Scan2Cap CIDEr@0.5, ScanQA
CIDEr, SQA3D EM):

| Method | ScanRefer A@0.5 | M3DRefer F1@0.5 | Scan2Cap C@0.5 | ScanQA C | SQA3D EM |
|---|---|---|---|---|---|
| Chat-Scene (Vicuna-7B, baseline) | 50.2 | 52.4 | 77.1 | 87.7 | 54.6 |
| 3DGraphLLM (Vicuna-1.5-7B) | 53.0 | 57.3 | 79.2 | 91.2 | 55.1 |
| 3DGraphLLM (LLAMA3-8B-Instruct) | 56.6 | 59.9 | 81.0 | 88.8 | 55.9 |

That is +6.4 Acc@0.5 and +7.5 F1@0.5 over Chat-Scene for the
LLAMA3 variant. Note: the original Dec-2024 preprint (v1)
reported a smaller gap - Chat-Scene 50.2 vs 3DGraphLLM-LLAMA3
54.6 Acc@0.5, i.e. **+4.4** - which is the figure commonly
quoted in summaries (including our own task brief); the
camera-ready v3 numbers above supersede it after further
tuning. Verified directly from both HTML sources.

**Relations-ablation (v3, Table 4)** - isolating relation
edges (k) from the GT-pretrain stage, same LLM/data
otherwise:

| LLM | Pretrain | k (edges) | Data | ScanRefer A@0.5 | M3DRefer F1@0.5 |
|---|---|---|---|---|---|
| Vicuna-1.5-7B | no | 0 | ScanNet | 50.2 | 52.4 |
| Vicuna-1.5-7B | no | 2 | ScanNet | 50.1 | 52.7 |
| Vicuna-1.5-7B | yes | 2 | ScanNet+3RScan | 53.1 | 57.3 |
| LLAMA3-8B | no | 0 | ScanNet | 52.0 | 55.1 |
| LLAMA3-8B | no | 2 | ScanNet | 54.3 | 57.3 |
| LLAMA3-8B | yes | 2 | ScanNet | 56.2 | 58.7 |
| LLAMA3-8B | yes | 2 | ScanNet+3RScan | 56.6 | 59.9 |

Reading this precisely: with LLAMA3 and no GT-pretrain
stage, adding k=2 relation edges alone moves ScanRefer
Acc@0.5 from 52.0 to 54.3 (+2.3) and Multi3DRefer F1@0.5
from 55.1 to 57.3 (+2.2) - relations help even without the
extra pretraining. But with Vicuna and no GT-pretrain,
relations alone barely move Acc@0.5 at all (50.2 -> 50.1,
i.e. -0.1) with only a marginal F1@0.5 gain (+0.3) - most of
Vicuna's headline improvement instead comes from the
GT-segmentation pretrain stage (50.1/52.7 -> 53.1/57.3). So
the "relations help" claim is solid for LLAMA3 in isolation,
but is confounded with training-pipeline choices for Vicuna.

**GT-vs-predicted-segmentation drop (v3, Table 5)** - all
rows use the full pipeline (GT-ScanNet pretrain), ScanRefer
Acc@0.5 / Multi3DRefer F1@0.5:

| Instance segmentation | k (edges) | Min. neighbor dist | Acc@0.5 | F1@0.5 |
|---|---|---|---|---|
| GT | 0 | - | 61.5 | 64.4 |
| GT | 2 | 0 cm | 66.9 | 69.9 |
| Mask3D | 0 | - | 52.0 | 55.1 |
| Mask3D | 2 | 0 cm | 55.6 | 58.2 |
| Mask3D + NMS | 2 | 0 cm | 55.7 | 58.6 |
| Mask3D + NMS | 2 | 1 cm | 56.2 | 58.7 |
| OneFormer3D | 0 | - | 50.0 | 52.8 |
| OneFormer3D | 2 | 0 cm | 52.8 | 55.8 |
| OneFormer3D + NMS | 2 | 1 cm | 54.6 | 57.2 |

Relations still help under predicted segmentation (Mask3D:
52.0 -> 56.2 Acc@0.5, +4.2; OneFormer3D: 50.0 -> 54.6,
+4.6), so the effect is not GT-only. But swapping GT for
predicted segmentation costs far more than relations gain
back: best GT config 66.9 Acc@0.5 / 69.9 F1@0.5 vs. best
predicted-segmentation config (Mask3D+NMS) 56.2 / 58.7 - a
drop of **10.7 Acc@0.5 points and 11.2 F1@0.5 points**
(about 16% relative), holding k=2 relations constant.
Instance-segmentation quality dominates the error budget
more than the relation-graph choice does.

## Why it does not fit our setting directly

- Requires LoRA fine-tuning an LLM plus training three
  projection MLPs on ScanNet/3RScan-specific
  vision-language QA data; our challenge allows LLM/VLM API
  calls at inference time but no task-specific training data
  in practice, so this training recipe cannot be reproduced
  or adapted for our target scenes.
- Depends on a separately trained relation classifier
  (VL-SAT) with a fixed relation ontology learned from
  ScanNet/3RScan - unverified whether its relation vocabulary
  or its point-cloud-only features transfer to arbitrary
  Unity indoor scenes and our three question types
  (numerical, object-reference, instruction-following); we
  cannot retrain it to adapt.
- Needs per-object point clouds up front, either GT instance
  segmentation (never available to us) or a batch run of a
  dedicated 3D instance-segmentation network (Mask3D /
  OneFormer3D) over an already-reconstructed point cloud -
  not something built incrementally from a live, moving
  robot's streaming sensors under a per-question time budget
  and scene resets.
- Its own best "no-GT" numbers (56.2 Acc@0.5, 58.7 F1@0.5)
  are still measured on ScanNet validation scans, not Unity
  sim scenes - domain shift to our actual evaluation
  environment is unverified either way.

## Takeaways for our 2026 module

- Primary value: solid evidence that giving an LLM explicit
  object-to-object relation edges (even just k=2 nearest
  neighbors) measurably beats a coordinates/identifier-only
  object-token baseline, across grounding, captioning, and
  QA, and the effect survives (attenuated) when instance
  segmentation is imperfect. This supports designing our
  scene representation/prompting around explicit relations
  (e.g. "on," "left of," "near") rather than raw coordinates
  alone, even without training.
- Caveat before leaning on this too hard: the relations-alone
  effect was clean for LLAMA3 but not for Vicuna without the
  GT-pretrain stage - some of the reported gain is entangled
  with a training pipeline we cannot replicate. Treat "explicit
  relations help" as directionally true, not as a number we can
  port (+2-6 points is model/pipeline-dependent, not a fixed
  constant).
- Segmentation/detection quality is the bigger lever: the
  GT-vs-predicted gap (~11 points) dwarfs the relations-vs-
  none gap (~2-5 points) in every row of Table 5. Since we
  will never have GT instances either, whatever
  detection/segmentation front-end we choose is likely to
  matter more for our final scores than how we encode
  relations in the prompt.
- Actionable, training-free analog worth considering later
  (Phase 2, not now): when prompting an LLM/VLM about a
  scene, provide each candidate object with a short list of
  nearby objects and a cheap, non-learned spatial-relation
  label (e.g. computed geometrically from bounding boxes/
  centroids) instead of only raw coordinates - borrowing the
  "structured relation triplet" idea as a prompt-formatting
  pattern, not the trained VL-SAT/LoRA machinery.
- This is a parked research reference only, from the
  pre-merge research pass - nothing here should be read as
  adopting a scene-graph-token design; see
  `docs/architecture.md` for the adopted design.

## Sources

- arXiv abstract (all versions):
  https://arxiv.org/abs/2412.18450
- Full text v1 (Dec 2024 preprint, source of the "+4.4
  Acc@0.5" figure): https://arxiv.org/html/2412.18450v1
- Full text v3 (Aug 2025, ICCV camera-ready, source of the
  main tables quoted above): https://arxiv.org/html/2412.18450v3
- Official code: https://github.com/CognitiveAISystems/3DGraphLLM
- Model weights: https://huggingface.co/wingrune/3DGraphLLM
- IEEE Xplore listing (ICCV 2025):
  https://ieeexplore.ieee.org/document/11444102/
- Cross-reference, this repo's challenge context:
  `docs/challenge_brief.md`
