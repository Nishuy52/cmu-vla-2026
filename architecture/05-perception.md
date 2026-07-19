# 5. Perception

How raw panorama + lidar frames become the sparse 3D instance map the
answer heads resolve against, and which parts of that pipeline are real
today versus stubbed.

## ELI10

The robot doesn't build a photorealistic model of the room — it keeps a
list of boxes: "chair, seen 3 times, roughly here." Every frame, a
detector proposes 2D boxes on camera tiles, lidar turns each into a 3D
box, and a simple nearest-neighbour matcher decides whether that box is
one you've already met or a new one. There is no fancy motion model —
if the same-labelled box didn't move far since last time, it's the same
instance.

## Detector status: mock vs real

[src/core/perception/detector.py](../src/core/perception/detector.py) —
two implementations behind one `DetectorProtocol` (around line 61):

- `FakeDetector` (around line 70) — deterministic, hand-scripted boxes,
  used throughout the test suite and the structural battery. Not a
  model at all.
- `GroundingDinoDetector` (around line 358) — the real Phase-2 seam.
  Its `torch`/`groundingdino` imports are lazy (deferred to first
  `__call__`, per the module docstring around line 8), so constructing
  it never fails on a machine without the model dependencies; it only
  fails loudly at inference time if they're absent.

Which one runs is an environment switch, not a code branch: in
production, [src/ros_adapter/adapter_node.py](../src/ros_adapter/adapter_node.py)'s
`make_detector` (around line 135) reads `VLA_DETECTOR` — `none`
(default) returns `None` and the `PerceptionPipeline` is never even
constructed (the empty `BasicSceneIndex([])` stub stands in, and a
SUBMISSION-BLOCKER log fires); `grounding_dino` constructs the real
`GroundingDinoDetector`. **As of today's adapter wiring, the default is
`none`** — the scored pipeline's live perception is off by default and
answers come from FSM floors only unless an operator explicitly sets
`VLA_DETECTOR=grounding_dino`. The log line at
[adapter_node.py](../src/ros_adapter/adapter_node.py) around line 152
states this plainly: "real inference lands in Phase 2."

## Instance map

[src/core/perception/scene_index.py](../src/core/perception/scene_index.py) —
`BasicSceneIndex`, a flat list of `InstanceRecord`s with a typo/plural/
synonym-tolerant lookup ladder (`by_label` / `by_label_tiered`, module
docstring around line 9): exact canonical match, then a small hand-built
synonym table (`_SYNONYM_GROUPS`, around line 56 — `refrigerator`/
`fridge`, `sofa`/`couch`, `television`/`tv`, `picture`/`photo`), then
head-noun match ("beer bottle" → "bottle"), then edit-distance typo
match (length-scaled budget, disabled under 5 characters) as the last
resort only when the first three tiers are empty.

Fusion of a new detection into an existing instance
(`BasicSceneIndex.add`) merges same-label instances whose footprint 3D
IoU exceeds `MERGE_IOU = 0.3`: points concatenate, the AABB is
recomputed as a per-axis 2nd/98th-percentile trimmed box (outlier-
resistant), `n_obs` increments, `score` keeps the max.

## Tracking: nearest-neighbour, not anything fancier

[src/core/perception/tracker.py](../src/core/perception/tracker.py) —
stated plainly in the module docstring (line 1): this is greedy-nearest
cross-frame association, no Kalman filter, no learned re-ID, no motion
model. `associate()` (around line 130):

1. Build every (detection, existing instance) candidate pair that is
   **label-compatible** (`labels_compatible`, around line 64 — same
   canonical noun after folding through both `scene_index.normalize_label`
   and `core.parsing.vocab.NOUN_ALIASES`, so "television" associates
   with a "tv" instance) **and** within a fixed **centroid-distance
   gate** (`TrackerConfig.gate = 0.75` m, line 76).
2. Sort all candidate pairs by ascending distance; greedily match each
   side at most once (a detection or instance, once consumed, is never
   reconsidered).
3. Matched detections fuse into their target instance via
   `BasicSceneIndex.add`; unmatched detections mint new instances.

That's the entire "tracking" algorithm — one gate, one greedy sort, no
velocity or appearance state carried between frames.

Two auxiliary pieces layered on top of raw association:

- **Keyframe gate** (`PerceptionPipeline._is_keyframe`, around line
  249): perception runs only every `every_k` frames (default 1, i.e.
  every frame) OR after the vehicle has moved `min_translation` (0.5 m)
  OR turned `min_rotation` (30°) since the last processed frame
  (`KeyframeConfig`, around line 87).
- **Singleton decay** (`decay_singletons`, around line 177, issue
  H15(a)): an instance still at `n_obs == 1` that hasn't been
  re-observed within `decay_k` (default 5) *detection-bearing*
  keyframes since it was minted is pruned as a one-frame ghost.
  Confirmed tracks (`n_obs >= 2`) are never decayed regardless of age.
  The clock only advances on keyframes where the detector produced at
  least one detection — a keyframe where the detector saw nothing at
  all is not evidence against a singleton (cold start, occlusion), so
  it doesn't count against the decay budget (comment at
  [tracker.py](../src/core/perception/tracker.py) around line 290).

`PerceptionPipeline.process(pano, scan)` (around line 265) is the glue:
tiling → detector → fusion → `associate` → decay, behind the keyframe
gate, returning the list of instance ids touched that frame (empty list
if the frame was skipped).

## How perception feeds the heads

The heads never see raw frames — they resolve against whatever
`SceneIndex` they're handed. In production
([adapter_node.py](../src/ros_adapter/adapter_node.py) around line 415-
428), when a detector is configured the `PerceptionPipeline`'s **live**
index (`self._perception.index`) is the same object the controller's
`self._scene_index` points at, so it grows in place as frames fuse —
the heads (`core.geometry.toolbox.resolve`, `counting`) always read the
current accumulated state, never a snapshot. When no detector is
configured, `self._scene_index` stays the permanently-empty
`BasicSceneIndex([])` stub and every resolve/count call sees zero
instances — that's the "SUBMISSION-BLOCKER" state the adapter logs
about, and it is today's default.

For offline batteries, [src/core/runner/single.py](../src/core/runner/single.py)
takes the scene straight from the injected `RobotIO`'s `.scene`
attribute (a synthetic scene) or an explicit `scene_index` argument
rather than running the perception pipeline at all — the structural/GT
batteries described in chapter 8 mostly bypass `PerceptionPipeline`
entirely and hand the heads a fully-observed index directly (see
chapter 8's "fully observed" note on `gt_battery.py`).

## Scripted detections (dev aid, feeds the SAME pipeline code)

[src/core/perception/scripted.py](../src/core/perception/scripted.py) —
`ScriptedPanoDetector` replays hand-labelled 2D boxes
(`data/fixtures/jingfan_labels.json`, drawn on the raw equirectangular
panorama) converted into the exact tile-space `Detection` objects the
real pipeline expects (module docstring around line 1-13), so fusion
sees real lidar against real-if-approximate 2D boxes without needing
GroundingDINO weights. This is a `DetectorProtocol` implementation like
`FakeDetector` and `GroundingDinoDetector` — it runs through the
identical `PerceptionPipeline` code, just with a different detector
plugged in, so it is IN the scored code path's class hierarchy (it
could be wired via `VLA_DETECTOR`-style selection) but is not itself
one of the two detector choices the adapter's `make_detector` currently
selects between.

## Outside the scored pipeline: offline dev tooling

The following live under `tools/`, are excluded from the challenge fork
surface per the repo's own layout convention, and never run under the
600 s question budget, the checkpoint ledger, or any relaunch-per-
question constraint:

- **`tools/colored_cloud.py`** — offline colored dense point-cloud
  reconstruction. Accumulates a recorded session's raw lidar points and
  tints each by the panorama pixel it forward-projects onto (the mirror
  image of `fusion._points_in_frustum`'s bearing/elevation math,
  composed with `tiling`'s calibrated azimuth/elevation conventions —
  no sign conventions re-derived, all pulled from `tiling.AZIMUTH_SIGN`
  / `COLUMN0_YAW_OFFSET` / `ELEVATION_SIGN`), voxel-downsamples, writes
  a PLY a human opens in CloudCompare/MeshLab/Open3D. Documented v1
  limitations (module docstring): no occlusion handling (a per-scan
  z-buffer is the planned fix; misregistration ghosting is left visible
  deliberately as the diagnostic signal) and no motion deskew. CLI:
  `python -m tools.colored_cloud extract <bag_or_fixture_dir> <out.ply>`.
  The repo otherwise keeps only the sparse instance map
  (`scene_index.py`) and a 2D occupancy grid (height discarded) — this
  tool is the only way to look at the reconstructed geometry by eye.
- **`tools/llm_vision_checkpoint_replay.py`** — the vision-half battery
  companion to the parse battery (chapter 8): replays CP2/CP3/CP5
  against real recorded panoramas from
  `data/sim_bags/{japanese_room_q1,office_1_q1,livingroom_1_tour}`
  through `core.replay.bag_reader.BagSource`, calling the exact
  checkpoint entrypoints (`run_miss_recovery` / `run_anchor_confirm` /
  `run_frontier_select`) against a live local Ollama vision model
  (`qwen2.5vl:3b`-style, per module docstring around line 30). Ground
  truth comes from each scene's `object_list.txt` plus odom bearing
  math. Two of its three test surfaces (CP3's crop generation, CP5's
  frontier-disc overlay) note there is no upstream crop-generation or
  frontier-detector code yet, so this tool derives/supplies both itself
  "for format/sanity purposes only" — it is explicitly a diagnostic
  harness, not a stand-in for missing production code.

Both tools are excluded from the `src/` pytest suite (`tools/` has its
own `python -m pytest tools` invocation, per the repo's top-level
`CLAUDE.md` layout table) and neither is importable from anywhere in
`src/core/`.

## What I could not verify

- I did not run `GroundingDinoDetector` against live model weights —
  its status as "constructs cleanly, fails loudly only at inference
  without deps" is read from the code and docstrings, not from an
  actual GPU run in this session.
- I did not execute `tools/colored_cloud.py` or
  `tools/llm_vision_checkpoint_replay.py`; their behavior is described
  from their source and docstrings only.
