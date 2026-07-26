# Notes — issues #84 / #89 (live grounding coverage + instance explosion)

Worktree: `agent-aa0525cc4e0a4a99b`. Scope: `src/core/perception/{detector,tracker,scene_index}.py`,
`src/core/heads/factory.py` (prompt wiring only), tests.

## 1. Prompt dead zone (#84)

`GroundingDinoDetector` is constructed at boot (`ros_adapter/adapter_node.py:172`,
out of scope) with `question_nouns=()`/`vocab_nouns=()`, so `self.prompt`/
`self.question_prompt` stay `""` until `HeadState.bind()` first fires on plan latch.
Parsing (LLM ladder or regex) takes real wall-clock time, and exploration/perception
keyframes tick throughout that window, so every keyframe before latch ran the
detector fully blind — independent of GDINO quality.

Fix: `build_callables()` (`core/heads/factory.py`) now calls
`refresh_prompt(detector, (), _STANDING_VOCAB_NOUNS)` immediately after constructing
`HeadState`, i.e. at build time, before any plan exists. The detector grounds on the
standing 114-noun vocab from the very first keyframe; `HeadState.bind()` still fully
rebuilds the prompt (question nouns first, standing vocab after) the instant the plan
latches, unchanged from before. `question_prompt` stays `""` pre-latch (correct — no
question target is known yet, nothing for the short caption pass to search for).

Updated 4 assertions in `tests/heads/test_factory.py` that previously asserted
`build_callables()` leaves the prompt untouched (`fake.prompt == ""`); those asserted
the exact defect being fixed. Added a new regression test
(`test_build_callables_primes_detector_with_standing_vocab_before_latch`).

Status: **fixed**.

## 2. Answer-eligibility gate exercised + observable (#84)

The gate (`is_answer_eligible`, `core/perception/detector.py`) itself already had good
unit coverage; the issue is that the offline GT battery's mocks are born `n_obs=3`
(`core/runner/gt_battery.py:10-17`, out of scope — not touched, mock semantics
unchanged), so the gate's REJECTION path is never exercised end-to-end offline, and a
rejection was previously a silent `False` with no visibility anywhere.

Did NOT retune `DEFAULT_GDINO_ANSWER_MIN_OBS` (2) or `DEFAULT_GDINO_ANSWER_MIN_SCORE`
(0.30) — no held-out evidence exists for either value; per `docs/calibration.md`'s
generalization protocol this would need scene-level holdout evidence I have not
gathered. Evidence plan if a change is ever proposed: (a) pull `answer_eligible=false`
counts + reasons from the new `dump_instance_index` stream (below) across >=4 live
runs spanning >=4 scenes never used to pick the value, (b) compare against GT counts
for the same classes, (c) only then consider a threshold change, reported per rule 1
of the generalization protocol, not committed on a single run's numbers. Filed as a
follow-up item, not a code change, in this session.

What changed instead (structural, not tuning):
- Added `answer_eligibility_reason(record) -> str` (`core/perception/detector.py`):
  returns `ELIGIBLE` / `INELIGIBLE_N_OBS` / `INELIGIBLE_SCORE` / `INELIGIBLE_BOTH` /
  `INELIGIBLE_MALFORMED`. `is_answer_eligible` is now built on top of it and logs a
  DEBUG line (`core.perception.detector` logger) with the record id/label/n_obs/score
  and the reason on every rejection — no behavior change (same boolean returned), pure
  additive observability.
- Wired `answer_eligible` + `eligibility_reason` into every instance entry of the new
  `dump_instance_index` stream (item 5) — so a live run's dump shows exactly which
  instances would win an answer right now and why not, without log-scraping.
- Added direct unit tests for every `answer_eligibility_reason` branch
  (`tests/perception/test_detector.py`) plus a dump-level regression test
  (`test_dump_instance_index_surfaces_gate_rejection_reason`,
  `tests/perception/test_scene_index.py`) using an `n_obs=1` record — exactly the
  case the battery's `n_obs=3` mocks can never construct.

Status: **structural work done (gate exercised + observable); threshold tuning
deferred pending evidence, plan above**.

## 3. Label/vocab bridging + instance-explosion root cause (#84 item 3 / #89)

Found the actual root cause, shared by both issues: `tracker.associate()` decides a
detection matches an existing instance using the CENTROID distance gate (0.75 m) and
the FULL alias bridge (`canonical_for_match`, which folds `core.parsing.vocab.
NOUN_ALIASES` on top of `scene_index.normalize_label`'s own narrower 4-group
`_SYNONYM_MAP`). It then used to hand the matched detection to `index.add()`, which
INDEPENDENTLY RE-DERIVES the merge decision from `scene_index.normalize_label` ALONE
(missing any alias only known to `NOUN_ALIASES`, e.g. `refridgerator` vs
`refrigerator`) and a 3D AABB IoU > 0.3 test — a stricter, DIFFERENT criterion than the
centroid gate. Whenever the two disagreed — a label variant `NOUN_ALIASES` bridges but
`normalize_label` alone doesn't, or (far more commonly, live) ordinary pose-jitter
dropping a small object's per-keyframe AABB IoU below 0.3 despite the centroids
staying well inside the association gate — `add()` fell through to its "new instance"
branch, found the id association had already assigned was taken, and silently minted a
FRESH id. Every such disagreement is a duplicate instance for something already being
tracked: the dominant, structural cause of #89's 78-141 instance overcounts, and
simultaneously the #84 item-3 "label-variant ghosts" mechanism.

Fix: `BasicSceneIndex.merge_into(instance_id, rec)` (`core/perception/scene_index.py`)
fuses `rec` directly into the instance identified by `instance_id`, trusting the
caller's association decision unconditionally — no label/IoU re-derivation.
`tracker.associate()`'s matched branch now calls `index.merge_into(target.instance_id,
rec)` instead of `index.add(rec)`. The unmatched (new-instance) branch is untouched —
`index.add()` still runs its own IoU dedup there, which is a legitimate safety net
against same-frame duplicate detections that were never matched to anything by
association in the first place.

Tests added:
- `tests/perception/test_scene_index.py`: `merge_into` fuses by id even when
  disjoint (#89 regression) and even across label variants not in `_SYNONYM_MAP`
  (#84 regression), plus the fallback-to-`add` path when the id is genuinely absent.
- `tests/perception/test_tracker.py`:
  `test_associate_merges_across_label_variant_not_in_scene_index_synonym_map`
  (repro for `refridgerator`/`refrigerator` — would have spawned a 2nd `n_obs=1`
  ghost pre-fix) and `test_associate_merges_repeated_sightings_despite_jitter_
  defeating_iou` (the #89 synthetic repro: 5 sightings of one physical chair, tiny
  per-frame point clusters so consecutive AABBs never overlap by IoU, all centroids
  within the association gate — asserts one instance / `n_obs=5`, not 5 ghosts).

Status: **fixed** (both #84 item 3 and #89's dominant contributor share this one root
cause and this one fix).

## 4. #89 instance explosion

See item 3 above — same fix. No separate change needed; the synthetic jitter repro in
`test_tracker.py` is the reproduction the issue asked for.

## 5. Instrumentation

Added `dump_instance_index(index, tag, *, keyframes_processed=None, extra=None)` in
`core/perception/scene_index.py`, following the existing `core/heads/explore_debug.py`
opt-in-JSONL-dump pattern (unset env var == zero behavior/perf cost, not even the
`os.environ.get` matters beyond one dict lookup):

- **Env var**: `VLA_INSTANCE_DUMP_PATH` — JSONL output path. Unset (default) = no-op.
- **Env var**: `VLA_INSTANCE_DUMP_INTERVAL_S` — minimum seconds between PERIODIC dumps
  (default 10.0; only throttles the `tag="periodic"` caller below, an explicit call is
  never throttled).
- Record shape: `wall_time`, `tag`, `keyframes_processed`, `total_instances`,
  `by_class` (label -> count), `instances: [{id, label, position, score, n_obs,
  answer_eligible, eligibility_reason}, ...]`.

**Periodic** dump is wired into `PerceptionPipeline.process()`
(`core/perception/tracker.py`) — fires on every keyframe, throttled by the PanoFrame's
own `t` (not wall-clock, so it stays deterministic/testable), tagged `"periodic"`.

**Answer-time** dump: the natural shared seam across all qtypes is
`core/heads/factory.py`'s `_final_answer(state)` (called from the `verify` callable),
but per my ownership scope for that file ("prompt wiring only") I did not add a call
there. **Hook needed** (for you to integrate, one line, wherever you decide is the
right per-qtype or shared seam):

```python
from core.perception.scene_index import dump_instance_index
dump_instance_index(state.scene, tag="answer_time")
```

Natural call sites, in order of preference:
1. `core/heads/factory.py::_final_answer(state)`, right before it returns a non-None
   answer — one call point covers NUMERICAL/OBJECT_REFERENCE/INSTRUCTION_FOLLOWING.
2. If per-qtype granularity is wanted instead: `core/heads/numerical.py`,
   `core/heads/object_ref.py`, `core/heads/instruction.py` at their respective terminal
   answer points.
3. If instead you want it tied to the actual ROS publish (not just "verify() decided"):
   `ros_adapter/adapter_node.py`'s `publish_int`/marker/waypoint publish methods
   (~line 1104 for `publish_int`, ~1100 for the marker publish) — out of my ownership,
   not edited.

Tests: `tests/perception/test_scene_index.py` (`dump_instance_index` no-op/write/
append/never-raises/gate-reason-surfaced) and `tests/perception/test_tracker.py`
(`PerceptionPipeline` periodic dump wiring + throttle).

## Test-gate / determinism check

Fast tier (`pytest` from `src/`, default `-m "not slow"` via `pyproject.toml`
addopts): ran before and after these changes. Failure/error set is byte-identical
before/after (1 pre-existing flaky float-equality assertion in
`tests/runner/test_gt_battery.py::test_synthetic_scene_extra_wall_cells_mark_
terrain_obstacle`, plus ~35 pre-existing `ModuleNotFoundError: rosbags` errors in
`tests/replay/*` and `tests/parsing/test_regex_full_set.py` — confirmed pre-existing
via `git stash`/re-run, unrelated to `rosbags` not being installed in this
environment). No new failures introduced. No battery-score-relevant behavior change:
none of the merge_into/prompt-priming/observability changes alter offline battery
mocks (`gt_battery.py` untouched) or default thresholds.
