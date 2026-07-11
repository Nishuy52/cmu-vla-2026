# Prior Art — Models & Approaches for VLA/VLN Challenge Problems

*Deep research pass, 10 Jul 2026 (Fable window). Sources linked inline; all arXiv/venue dates checked.*

## TL;DR — what this means for our build

1. **The organisers published their own solution.** [SORT3D](https://arxiv.org/abs/2504.18684) (Zantout, H. Zhang, Kachana, Qiu, J. Zhang, W. Wang — CMU RI, 2025) is a zero-shot LLM grounding + navigation system built on **the exact challenge robot** (360° cam + 3D lidar + the same autonomy stack). The [code](https://github.com/nzantout/SORT3D) explicitly integrates with "the CMU VLA challenge mecanum-wheeled platform" and ships Unity sim configs. Our architecture draft independently converged on the same shape — validate against it, reuse what's legal (⚠️ no explicit license in repo — check before copying code; concepts are fair game).
2. **Object-centric text scene graphs + LLM reasoning beats end-to-end 3D models in zero-shot settings** — this is now well-replicated (SORT3D, Transcrib3D, ConceptGraphs). No fine-tuning is likely needed to be competitive → cluster training becomes optional, not critical-path.
3. **The differentiator is not grounding accuracy, it's exploration + time budget.** Papers evaluate on fully-observed scenes; the challenge starts unknown with a 10-min clock. VLFM-style semantic frontier scoring is the best-published answer.
4. Two CMU MSR theses from the organising group are effectively the challenge playbook: [H. Zhang, "Object-Centric Grounding for Deployable and Interactive Vision-Language Navigation" (2025)](https://www.ri.cmu.edu/app/uploads/2025/09/HaochenZhang_MSR_Thesis.pdf) and [P. Kachana, "Advancing 3D Semantic and Geometric Reasoning" (2025)](https://www.ri.cmu.edu/app/uploads/2025/05/Advancing_3D_Semantic_and_Geometric_Reasoning__Thesis_-2.pdf). **Read both.**

---

## 1. The organisers' line of work (most load-bearing)

### SORT3D — Spatial Object-centric Reasoning Toolbox ([arXiv:2504.18684](https://arxiv.org/abs/2504.18684))

Four-stage pipeline, entirely zero-shot:

1. **Perception & attributes**: 2D detections → crops picked by CLIP similarity → **Qwen2-VL/Qwen2.5-VL captions** per object → objects as `{id, name, caption, cx, cy, cz, size}`. Online variant: Grounded-SAM-2 + GroundingDINO + **ByteTrack** for cross-frame instance tracking.
2. **Object filtering**: LLM (Mistral Large 2 default, GPT-4o supported) extracts mentioned types/attributes from the query, prunes the scene to candidates.
3. **Spatial reasoning toolbox**: deterministic geometric functions (`left_of`, `closest_to`, `between`, …) that *rank* candidates. Key trick for view-dependent relations ("to the left if facing the door"): anchor feasible viewpoints using **free traversable space**.
4. **LLM sequential reasoning**: chain-of-thought with ONE in-context example; the LLM calls toolbox functions instead of doing mental math; output parsed into waypoints.

Results: Nr3D 60.5% zero-shot (vs ZSVG3D 39%, VLM-Grounder 48%); Sr3D 92% (beats *supervised* 3D-VisTA 76.4%); VLA-3D hard statements 75% (vs Transcrib3D 57.5%). Ablation: adding 2D-VLM captions ≈ **+11%** — attribute richness is the single biggest lever.

Failure modes to design against: multi-constraint logic slips (satisfying one clause, ignoring the other) and pragmatic implicature ("rightmost pillow" → should infer *on the bed*). Also: ±6% run-to-run variance from LLM stochasticity — consider self-consistency voting on high-value questions.

Hardware: 7–10 GB VRAM for perception (fits far below the eval 4090); LLM via API (allowed by rules).

### Transcrib3D ([OpenReview](https://openreview.net/forum?id=7j3sdUZMTF))
Precursor idea: transcribe 3D detections (type, colour, position, extent) into pure text; LLM reasons with **interactive code generation** + self-corrected fine-tuning. Nr3D 70.2%/Sr3D 98.4% *on filtered statements* (not directly comparable). Takeaway we should steal: **LLM writes a filter program, deterministic code executes it** — kills hallucinated counts for numerical questions.

### VLA-3D dataset ([arXiv:2411.03540](https://arxiv.org/abs/2411.03540)) & IRef-VLA ([arXiv:2503.17406](https://arxiv.org/abs/2503.17406))
11.5K rooms, 23.5M object relations, 9.7M referential statements, scene graphs + free-space annotations; includes the 15 training scenes. IRef-VLA adds imperfect/ambiguous language handling. Use: offline eval fixtures for our grounding module; fine-tuning data only if zero-shot plateaus.

## 2. Zero-shot LLM navigation (adjacent VLN literature)

- **[VLFM](https://arxiv.org/abs/2312.03275)** (ICRA'24): frontier exploration where each frontier is scored by a **BLIP-2 cosine-similarity "value map"** between RGB and a target-object text prompt → drive to the most semantically promising frontier. SOTA zero-shot ObjectNav (HM3D/MP3D/Gibson), deployed on real Spot. → Our exploration policy: plain frontier exploration until question arrives, then question-conditioned VLFM-style scoring.
- **[ConceptGraphs](https://arxiv.org/abs/2309.16650)**: incremental open-vocab 3D scene graphs from RGB-D; hierarchical relations for language navigation. Validates our persistent-object-map design.
- **[Open-Nav](https://arxiv.org/abs/2409.18794)** (ICRA'25): zero-shot VLN-CE with *open-source* LLMs via spatial-temporal chain-of-thought — fallback evidence that local models can replace APIs if connectivity at eval is a concern.
- **[3D-Mem](https://arxiv.org/abs/2411.17735)** (CVPR'25): "memory snapshots" + frontier snapshots for exploration-aware EQA — the best pattern for *when to stop exploring*: explore while the answer-relevant memory is insufficient.

## 3. Embodied QA (the numerical questions)

- **[OpenEQA](https://open-eqa.github.io/)** (CVPR'24): 1,600+ questions over real scenes; baseline = frontier exploration + multimodal LLM answering. Headline finding: even GPT-4V-class models lag humans badly on **counting and spatial questions** — naive "ask a VLM the image" will lose; structured counting over a tracked object map (with ByteTrack-style instance de-dup across frames) is the way.
- **EXPRESS-Bench** ([arXiv:2503.11117](https://arxiv.org/html/2503.11117v1)): categorises EQA errors; counting reliability comes from explicit instance memory, not bigger VLMs. Reinforces: numerical answers = deterministic count over the scene graph, LLM only builds the filter.

## 4. Instruction-following with path constraints (the 6-point questions)

Thinnest published area — biggest opportunity to differentiate:

- **Task-oriented Sequential Grounding** ([arXiv:2408.04034](https://arxiv.org/pdf/2408.04034)): ground a *sequence* of objects/regions for multi-step instructions — matches "take the path near the window to the fridge" (constraint waypoint → goal waypoint).
- SORT3D's action parser already converts grounded anchors into ordered waypoints; H. Zhang's thesis covers the deployable version.
- Classic VLN (R2R/RxR, [1st-place RxR-Habitat report](https://arxiv.org/pdf/2206.11610)) is mostly end-to-end policy learning — **not** the right tool here (no training budget, different action space). The LLM-decomposition approach (constraint → anchor object → free-space waypoint near it → goal) fits the provided waypoint-following autonomy better.

## 5. Perception stack consensus (what everyone uses in 2025–26)

| Function | Consensus choice | Notes |
|---|---|---|
| Open-vocab detection | GroundingDINO / Grounded-SAM-2 | SORT3D uses exactly this |
| Instance tracking across frames | ByteTrack | prevents double-counting — critical for numericals |
| Attribute captioning | Qwen2.5-VL (7B fits eval GPU) | +11% grounding from captions |
| Panorama handling | split equirect into pinhole crops | standard practice |
| Lidar-camera fusion | project points into 2D masks → 3D centroid/extent | our draft already says this |

## 6. Gaps in the literature = where we can win

1. **Time-budgeted answering** (10-min clock, early-answer bonus): no paper optimises this. Confidence-triggered early stop + per-question-type exploration budgets is unpublished territory.
2. **Numerical robustness**: object-map counting with LLM-generated deterministic filters (Transcrib3D trick) beats every published VLM-counting baseline.
3. **Multi-constraint verification**: SORT3D's own failure mode — add a cheap LLM verification pass ("does the selected object satisfy EVERY clause?") before answering.
4. LLM stochasticity: self-consistency (3 samples, majority vote) on 6-point questions only — spend latency where points are.

## Reading list (priority order)

1. [SORT3D paper](https://arxiv.org/abs/2504.18684) + [code](https://github.com/nzantout/SORT3D) — the blueprint
2. [H. Zhang MSR thesis](https://www.ri.cmu.edu/app/uploads/2025/09/HaochenZhang_MSR_Thesis.pdf) — deployable grounding on this exact platform
3. [VLFM](https://arxiv.org/abs/2312.03275) — exploration policy
4. [3D-Mem](https://arxiv.org/abs/2411.17735) — when-to-stop-exploring
5. [Transcrib3D](https://openreview.net/forum?id=7j3sdUZMTF) — code-generation filtering for counts
6. [OpenEQA](https://open-eqa.github.io/) — QA failure modes
7. [Kachana thesis](https://www.ri.cmu.edu/app/uploads/2025/05/Advancing_3D_Semantic_and_Geometric_Reasoning__Thesis_-2.pdf), [VLA-3D](https://arxiv.org/abs/2411.03540), [IRef-VLA](https://arxiv.org/abs/2503.17406) — background + eval data

## Open question carried forward

- 2025 winning teams' write-ups: an MRSD team took 3rd worldwide ([CMU RI news](https://www.ri.cmu.edu/research-group-to-host-cmu-vision-language-autonomy-challenge/), [MRSD newsletter](https://labs.ri.cmu.edu/mrsd-news/articles/)); no arXiv technical reports found yet. Check the [2025 leaderboard page](https://www.ai-meets-autonomy.com/cmu-vla-challenge) and IROS-2025 workshop proceedings for team writeups when hardening the architecture.

---

## Update 11 Jul 2026 - open question resolved

The 2025-results question above is settled. The full 2025
leaderboard (with scores) was recovered via a Wayback snapshot of
the challenge site, and per-team method dossiers were written from
primary sources (entrant code read directly, papers fetched in
full, talk videos frame-read). Index + leaderboard:
`docs/prior_art/README.md`; dossiers under `docs/prior_art/`.

- 1st: NROS Lab, HIT Shenzhen - 44.26. Scene graph + multi-modal
  frontier-exploration scoring; method self-reported (Chinese
  sources only), no public code.
- 2nd: ReasonX, NTU+NUS - 34.58. Gemini 2.5 Pro JSON planner +
  RoboRefer-8B pixel pointing + camera-ray waypoints; public code.
- 3rd: CopyPasta, CMU MRSD - 30.98. Gemini state machine; public
  repo + full method talk (contra the note above that none was
  found - see `docs/prior_art/2025_3rd_copypasta.md`).
- 4th: URL-KAIST - 22.80. Most feature-complete finalist but a
  rushed, bug-ridden build - robustness beat componentry.

Key strategic finding: ground-truth object markers
(`/object_markers`) were legal and widely used in 2025 (ReasonX
grounding, CopyPasta for two of three question types). That topic
is absent from the 2026 allowed-I/O list (`docs/challenge_brief.md`),
so 2025 pipelines do not port directly - open-vocabulary
perception becomes the differentiator.
