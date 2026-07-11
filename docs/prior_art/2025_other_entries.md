> Imported 2026-07-11 from a parallel research stream. See
> `docs/prior_art/README.md` for provenance and conflict rules.

# Other 2025 entries (non-finalists) - survey

The 2025 CMU-VLA-Challenge finalist leaderboard was NROS 44.26,
ReasonX 34.58, CopyPasta 30.98, URL-KAIST 22.80 (see the sibling
`2025-1st-nros.md` etc. in this directory). Alongside those four,
GitHub search turns up several public forks of the dev kit
(https://github.com/HaochenZ11/CMU-VLA-Challenge) from teams that
entered but did not make the finalist cut. None of these publish a
paper or a scoring number, so everything below is read directly out
of their code: what detector/representation they use, whether an
LLM is in the loop, and how much of the pipeline is real versus
still the stock dev-kit placeholder. Repos were shallow-cloned into
a scratch directory and diffed/read file-by-file; findings below
cite exact paths. All push dates are from the GitHub API as of
2026-07-11.

## TAMSxGalbot (ShangQingLiu/CMU-VLA-Challenge)

Repo: https://github.com/ShangQingLiu/CMU-VLA-Challenge
(fork of HaochenZ11/CMU-VLA-Challenge, branch `main`, last push
2025-10-01).

The README ("Submission TAMSxGalbot") documents a heavyweight
two-container Docker setup (system container + a separate VLM
container), requires RAM > 72 GB, and asks users to put an
`OPENAI_API_KEY` in `~/.bashrc` (a shared key was also distributed
via Google Drive per the README - i.e. handed out informally, not
scoped per-team).

Verified: the public `ai_module`'s ROS-facing node,
`ai_module/src/dummy_vlm/src/dummyVLM.cpp`, is **byte-identical**
to the stock dev-kit placeholder (diffed against
HaochenZ11/CMU-VLA-Challenge directly - zero differences). It does
keyword matching on the question string ("Find..." -> mark+drive to
a pre-set object; "How many..." -> `rand() % 10 + 1`; else -> replay
a static waypoint file) and nothing else. The Python entry point
`ai_module/src/dummy_vlm/src/main.py` is an empty file (0 lines).
So the wired-up, running ai_module in this repo is genuinely the
unmodified dummy - the "reportedly still placeholder" claim is
confirmed, not just alleged.

The real R&D sits in a large, vendored, disconnected subtree:
`ai_module/src/dummy_vlm/src/navid_ws/NaVid-VLN-CE/` is a full copy
of the NaVid-VLN-CE codebase (NaVid = a video-LLM VLN-CE policy;
`navid/model/navid_arch.py`, `navid/model/language_model/
llava_navid.py`), plus a 1182-line `navid_agent.py` that acts as a
VLN-CE evaluation harness with a `RemoteNaVidAgentProxy(host=
"0.0.0.0", port=8000)` client and several `evaluate_agent_CMU*`
methods that additionally call OpenAI's `gpt-4`-class chat API
(`from openai import OpenAI`, `OPENAI_BASE_URL=https://api.openai.com/
v1`) - likely for auxiliary reasoning/parsing around the NaVid policy
output. A root-level `test_navid_service.py` is a smoke-test client
for the FastAPI NaVid service. None of this is imported by
`dummyVLM.cpp` or `main.py` - it looks like a NaVid-VLN-CE research
branch that was never bridged into the ROS ai_module before
submission.

## xiaofeifei-1 (branch tqf_dev)

Repo: https://github.com/xiaofeifei-1/CMU-VLA-Challenge, branch
`tqf_dev` (fork of HaochenZ11/CMU-VLA-Challenge, last push
2025-09-15).

This is the most complete non-finalist pipeline found. Real code
lives under `ai_module/src/dummy_vlm/src/dummy_vlm/` (package name
kept as `dummy_vlm` but contents fully replaced):

- `semantic_map/external/Grounded-SAM-2/` - a full vendored copy of
  the upstream Grounded-SAM-2 repo (GroundingDINO open-vocabulary
  detector + SAM2 segmenter, including their demo/tracking scripts
  and `LICENSE_sam2` / `LICENSE_groundingdino`).
- `semantic_map/cloud_image_fusion.py` (409 lines) - lidar-to-camera
  projection math (`scan2pixels`) that looks derived from the
  dev kit's own C++ lidar/camera fusion, reimplemented in Python/
  numpy, used to back-project detected 2D boxes into 3D.
- `semantic_map/semantic_map.py` (843 lines) - builds a persistent
  3D object map (`SingleObject`/`AdjacencyGraph`), with ByteTrack
  (`bytetrack.byte_tracker.BYTETracker`) for object re-identification
  across frames and a rerun-based visualizer.
- `captioner/models/captioning.py` - a captioning backend supporting
  several interchangeable VLMs (PaliGemma, Qwen2.5-VL, SigLIP,
  optional vLLM backend) plus `captioner/models/clip.py` for
  CLIP-based embeddings/similarity.
- `active_map/planner.py` (251 lines) and `active_map/wpNav.py` - a
  2D grid-map planner (KDTree + networkx + `scipy.ndimage` distance
  transform for obstacle inflation) and a waypoint controller
  publishing to `/way_point_with_heading`.
- `vla/queryVLM.py` - an LLM abstraction with `query_gemini`,
  `query_chatgpt`, `query_mistral`, and `query_doubao` methods.
  `query_doubao` (ByteDance's Doubao, OpenAI-compatible endpoint at
  `https://ark.cn-beijing.volces.com/api/v3`, model
  `doubao-1-5-pro-32k-250115`) is the one wired into
  `filter_objects()`, called from `vla/handleQuestions.py` to filter
  the semantic-map object list down to candidates matching the
  question. Note: this file has a **live-looking API key hardcoded
  in source** (`doubao_api_key = "82576321-..."`) - flagged here as
  a secrets-hygiene issue we should not repeat in our own repo, key
  value intentionally not reproduced in full above.

Net: real open-vocabulary 2D detector (GroundingDINO) + real
segmenter (SAM2) + persistent 3D semantic object map with tracking
+ a genuine LLM (Doubao, OpenAI/Gemini/Mistral as configurable
alternatives) doing object-candidate filtering, feeding a classic
grid-map path planner. This is architecturally close to what a
"detect + map + LLM-filter + plan" 2026 baseline could look like.

## aswarthm

Repo: https://github.com/aswarthm/CMU-VLA-Challenge (fork of
HaochenZ11/CMU-VLA-Challenge, last push 2026-02-01).

Uses a **ROS2 ament_python** package layout for its custom package
(`ai_module/src/dummy_vlm/{package.xml,setup.py,setup.cfg}`, plus a
`dummy_vlm/dummy_vlm/`, `dummy_vlm/srv`, `dummy_vlm/resource`
sub-structure) even though the stock dev kit and system are ROS1
Noetic/catkin - unusual, and it's unverified from the code alone
whether they run a ROS1/ROS2 bridge or ported the whole stack.

The interesting part is `ai_module/genai-server/app.py` (Flask
server) + `pydanticModels.py`: a native **Gemini 2.5 Pro
function-calling agent loop** (`from google import genai`,
`genai_client.models.generate_content(model="gemini-2.5-pro", ...)`
with `ToolConfig(function_calling_config=FunctionCallingConfig(
mode="ANY"))`). Pydantic models define the tool schema: e.g.
`navigate_to_point_fn` returns 5 candidate `[y, x]` pixel points
normalized to a 0-1000 scale (Gemini's native "pointing" convention),
`submit_final_count_fn`, `submit_final_object_reference_fn`,
`finish_instruction_following_fn`, and a verification tool
`verify_object_exists_fn`. The system prompt in `fn_call()`
explicitly instructs the model to explore before counting, to
navigate via pixel-pointing rather than object IDs directly, and to
treat "close proximity" (not just visibility) as landmark-visited
for instruction-following. There is no explicit open-vocabulary 2D
detector in this path - grounding is delegated entirely to
Gemini's native visual understanding on the raw camera image.

Supporting ROS2 nodes: `scripts/goalsnapper.py` builds a k-d tree
(`scipy.spatial.cKDTree`) over the `/traversable_area` point cloud
and snaps a proposed 3D goal to the nearest traversable point (with
a pull-in-factor to avoid landing exactly on/against an obstacle);
`scripts/depth_overlay.py` (not fully read, name implies depth-to-
pixel overlay for converting Gemini's 2D point + depth into a 3D
goal); `scripts/pointtoimage.py` is a small standalone debug script
with a hardcoded local path (`/home/aswarth/CMU-VLA-Challenge/...`)
for visually checking point-to-pixel scaling - clearly a dev
scratch tool, not part of the running pipeline.

## morigakuto (branch feat/cpu-only)

Repo: https://github.com/morigakuto/CMU-VLA-Challenge, branch
`feat/cpu-only` (fork of HaochenZ11/CMU-VLA-Challenge, last push
2026-04-02, single commit on the branch: "wip : using cpu without
GPU").

This is architecturally different from every other entry surveyed:
**no LLM at all**. `ai_module/src/uiap_ogn_ros/src/uiap_ogn_ros/
policy_node.py` wraps `vlfm.policy.reality_policies.
RealityITMPolicyV2` - i.e. VLFM (Vision-Language Frontier Maps), a
published zero-shot object-goal-navigation policy built on
GroundingDINO + MobileSAM + a BLIP-2-style image-text-matching (ITM)
score, with frontier-based exploration
(`compute_frontiers=True`) and a learned PointNav policy
(`pointnav_weights.pth`) for local motion. The `uiap-ogn` name
itself is not expanded anywhere in the repo (package.xml just says
"ROS wrappers for running uiap-ogn on the CMU VLA challenge
topics") - what "uiap" stands for is unverified.

Question understanding is a plain regex classifier, not an LLM:
`question_router.py` (37 lines) and the mirrored fallback in
`policy_node.py::_question_cb` pattern-match "how many" -> count,
"find"/"locate" -> find, "go to"/"navigate" -> navigate, and strip
the matched prefix to get an `object_goal` string, which is then
handed to VLFM as its target class the same way VLFM's original
ObjectNav benchmark does. Counting is a heuristic over VLFM's
internal object point-cloud cluster IDs
(`np.unique(np.round(cloud[:, -1], decimals=2))`), not a vision-
grounded count. The `feat/cpu-only` branch name and its one commit
message reflect an in-progress attempt to make `RealityITMPolicyV2`
run without CUDA; the code still warns
"CUDA not available; RealityITMPolicyV2 expects a GPU and may fail
on CPU" rather than guaranteeing CPU support.

## Brief mentions

**CINAPSLab/CMU_VLA**
(https://github.com/CINAPSLab/CMU_VLA, not a fork, last push
2025-09-16). More than a translation exercise: alongside a
`README_JP.md` it has its own `docs/` set (`function_calling.md`,
`LocalLLM.md`, `owlvit_finetuning_strategy.md`, `world_model_schema.md`,
`sensor_to_json_pipeline.md`) and a real `ai_module/src/cinaps_vlm/`
package (perception/exploration/reasoning submodules). Notably,
`ai_module/src/cinaps_vlm/CLEANUP_SUMMARY.md` records that they
built and then **deliberately deleted** an LLM-based QA node
(`llm_qa_node.py`, `llm_qa_node_fixed.py`, described as
"experimental"), keeping instead a rule-based `qa_node.py` with an
"enhanced synonym dictionary" for question parsing. They also
scoped (but the docs suggest did not necessarily finish) fine-tuning
OWL-ViT, an open-vocabulary 2D detector, on the VLA-3D dataset. This
repo also has its own `CLAUDE.md`, i.e. this team used Claude Code
for their own prep work too.

**yangjunwon1309/ai_module**
(https://github.com/yangjunwon1309/ai_module, not a fork, last push
2025-08-21). Broader than "a captioning package with sample crops":
it's a full catkin workspace with `depth_camera`, `dummy_vlm`
(unmodified stub, not inspected further), `gemini_API`,
`captioning`, `reltr_scene_graph`, and `planning_node` packages.
`src/reltr_scene_graph` vendors RelTR (a relation-transformer scene-
graph-generation model) as a ROS node producing per-frame scene
graphs that get merged across frames
(`captioning/output/merged_sg_*_with_captions.json`), then captioned
per-object crop (`captioning/src/captioning_node.py`, with sample
crops under `captioning/output/crops/`). `gemini_API/src/
answer_numerical.py` calls `google.generativeai` with model
`gemini-2.5-flash-lite` to answer numerical questions from the
merged, captioned scene graph, orchestrated by `gemini_API/scripts/
orchestrator.py`. So: scene-graph generation (RelTR) + captioning +
Gemini-flash-lite for numerical QA - a distinct scene-graph-first
approach worth knowing about even as only a brief entry here.

## Takeaways for our 2026 module

- Six-for-six real repos, but only three (xiaofeifei-1, aswarthm,
  yangjunwon1309) have an LLM/VLM actually wired into the running
  pipeline; morigakuto's VLFM entry has none at all, and CINAPSLab
  explicitly tried and reverted an LLM QA node in favor of a rule-
  based synonym matcher. Don't assume "use an LLM" is a foregone
  architectural choice among 2025 entrants - the finalists (per the
  sibling docs in this directory) skew LLM/scene-graph-driven, but a
  meaningful fraction of the field did not.
- Two broad detector/representation strategies recur: (a) 2D open-
  vocabulary detector + 3D back-projection into a persistent object
  map (xiaofeifei-1's GroundingDINO+SAM2+lidar fusion, morigakuto's
  VLFM GroundingDINO+MobileSAM+frontier map), versus (b) no explicit
  detector, grounding delegated to a large VLM's native pixel-
  pointing on raw images (aswarthm's Gemini 2.5 Pro function-calling
  loop). Both are viable; (b) is much less code but ties correctness
  entirely to the VLM's spatial grounding quality, which several
  papers say is inconsistent - worth stress-testing early if we go
  that route.
- The "keep a rule-based fallback for the ai_module ROS bridge and
  bolt the real intelligence on separately, then never actually
  wire it up" failure mode is real and visible in a submitted repo
  (ShangQingLiu/TAMSxGalbot: NaVid-VLN-CE + GPT-4 sat fully vendored
  and unused while the byte-identical stock dummy node was what
  actually ran). If we vendor a heavy external model, add an
  explicit integration-test step verifying the ROS node actually
  imports and calls it, not just that the vendored code exists in
  the tree.
- A hardcoded, real-looking LLM API key committed to a public
  GitHub repo (xiaofeifei-1) is a concrete, avoidable failure we
  should check for in our own repo before any push (secrets scan /
  gitignore of local config, never a literal key string in source).
- ROS1 vs ROS2 layout choices vary per team (aswarthm mixes ROS2
  ament_python packages into what the dev kit ships as ROS1/catkin)
  with no explanation in-repo of the bridging - a reminder to pin
  down the dev kit's actual ROS version early (per
  `docs/challenge_brief.md`'s open verification item) rather than
  assume.

## Sources

- https://github.com/ShangQingLiu/CMU-VLA-Challenge (README, branch
  `main`, `ai_module/src/dummy_vlm/src/dummyVLM.cpp`,
  `ai_module/src/dummy_vlm/src/main.py`,
  `ai_module/src/dummy_vlm/src/navid_ws/NaVid-VLN-CE/navid_agent.py`,
  `test_navid_service.py`)
- https://github.com/HaochenZ11/CMU-VLA-Challenge (upstream dev kit,
  used as the diff baseline for the stock `dummyVLM.cpp`)
- https://github.com/xiaofeifei-1/CMU-VLA-Challenge, branch
  `tqf_dev` (`ai_module/src/dummy_vlm/src/dummy_vlm/semantic_map/`,
  `.../captioner/`, `.../active_map/`, `.../vla/queryVLM.py`,
  `.../vla/handleQuestions.py`)
- https://github.com/aswarthm/CMU-VLA-Challenge
  (`ai_module/genai-server/app.py`,
  `ai_module/genai-server/pydanticModels.py`,
  `ai_module/src/dummy_vlm/scripts/goalsnapper.py`,
  `.../pointtoimage.py`)
- https://github.com/morigakuto/CMU-VLA-Challenge, branch
  `feat/cpu-only`
  (`ai_module/src/uiap_ogn_ros/src/uiap_ogn_ros/policy_node.py`,
  `.../question_router.py`, `.../package.xml`)
- https://github.com/CINAPSLab/CMU_VLA (`README_JP.md`,
  `ai_module/src/cinaps_vlm/CLEANUP_SUMMARY.md`,
  `docs/owlvit_finetuning_strategy.md`)
- https://github.com/yangjunwon1309/ai_module
  (`src/reltr_scene_graph/README.md`,
  `src/captioning/src/captioning_node.py`,
  `src/gemini_API/src/answer_numerical.py`,
  `src/gemini_API/scripts/orchestrator.py`)
