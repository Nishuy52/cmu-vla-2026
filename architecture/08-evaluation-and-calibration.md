# 8. Evaluation and Calibration

How the pipeline is scored offline before a submission — structural
health, real ground-truth accuracy, calibration sweeps, and the
generalization discipline that keeps tuning honest.

## ELI10

There's no answer key for the 75 training questions, so a first
harness just checks "did the machine produce something legal without
falling over" (`battery.py`). A second harness downloads the real
VLA-3D scene data, which DOES carry ground truth, and grades every
answer for real (`gt_battery.py`). A third harness (`cvsweep.py`) tries
many settings of the reasoning knobs, but never trusts a knob that only
looks good on the scenes it was tuned on — it always checks the knob
against scenes it wasn't allowed to see.

## `single.py` — run one question

[src/core/runner/single.py](../src/core/runner/single.py) —
`run_question` (module docstring, line 1) constructs the scene index +
heads + controller, injects the question, then ticks the
`QuestionController` on the io's *simulated* clock until DONE or a
wall-clock guard trips. Because the clock is simulated, a question with
the full 600 s sim budget completes in far under a second of real time.
Works over both `core.mocks.mock_io.MockRobotIO` (synthetic scene) and
`core.replay.replay_io.ReplayRobotIO` (fixtures/bags). `RunResult`
(around line 30) records the published answer, qtype, elapsed sim
seconds, states visited, checkpoint-call count, the full flight log,
and whether the floor path (not a head) produced the answer — every
other harness in this chapter is built by driving many `run_question`
calls and aggregating `RunResult`s.

## `battery.py` — structural health, no ground truth

[src/core/runner/battery.py](../src/core/runner/battery.py) — drives
all 75 training questions over a *synthetic* scene per scene name, built
by `core.runner.scenegen.build_scene_for` (one instance per mentioned
noun, 2+ for numerical-count targets so counting isn't degenerate).
Module docstring (line 1) is explicit: **this is not an accuracy
harness** — no GT answers exist for the training set. It scores
structural signals only: `answered_before_watchdog` (legal answer
before the 570 s floor), `answer_type_correct`, `parse_tier_used`,
`target_grounded`, `corridor_built`/`avoid_built` for IF, `sim_elapsed`,
`floor_used`. Output: `battery_report.md` + `battery_results.json`
under `reports/battery_<date>/`.

`core.runner.scenegen` ([src/core/runner/scenegen.py](../src/core/runner/scenegen.py))
is the scene-synthesis half: `nouns_in_text` (around line 22) extracts
every vocabulary noun from a question via a full n-gram sweep
(longest-phrase-first, left to right), and `build_scene_for` lays boxes
onto a seeded grid deterministically.

## `gt_battery.py` — real accuracy against VLA-3D ground truth

[src/core/runner/gt_battery.py](../src/core/runner/gt_battery.py) —
"Battery v2" (module docstring, line 1). Loads the actual VLA-3D scene
folders (matched by scene name to a subfolder of the same name) and
scores each question for real accuracy:

- **NUMERICAL** — exact-match count plus an independent second opinion.
- **OBJECT_REFERENCE** — 3D IoU of our resolved box vs the GT target
  box.
- **INSTRUCTION_FOLLOWING** — discrete Fréchet distance plus the
  fraction of the GT path within 1.0 m, alongside the headline rubric
  score (below).

The GT scene is *fully observed* — every GT object becomes an
`InstanceRecord` with `n_obs=3` (module docstring, line 11) — so
NUMERICAL/OBJECT_REFERENCE call the resolver directly against the GT
index with no exploration in the loop; INSTRUCTION_FOLLOWING drives the
**full pipeline** via `single.run_question` over a GT-mirrored
`MockRobotIO`, so the instruction head plans a real waypoint path
through a costmap built from GT geometry and the emitted waypoints are
scored against `trajectory_qN.ply`.

`aggregate()` (around line 1483) is the scoring rubric implementation:
for NUMERICAL, `true_accuracy` (against `docs/gt_answers_numerical.json`,
the PDF-extracted human answer key) is the primary yardstick;
`independent_agreement_rate` explicitly **excludes** `*_class_only` rows
from both numerator and denominator (a class-total count is not
independent evidence for a relation-filtered question — NUM-F6/F7,
comment around line 1497). For OBJECT_REFERENCE, `mean_iou` plus
threshold hit-rates at 0.25/0.5. For INSTRUCTION_FOLLOWING, the
headline is `mean_rubric_score` (the driven-trajectory rubric score,
not the Fréchet/coverage diagnostics) plus `mean_ordered_leg_credit` and
threading/avoid violation totals; unaligned scenes (scene-frame fit
residual too large) are split into `unaligned_scenes_data_confirmed`
(a documented GT-data defect — endpoints that can't be rigidly mapped)
vs `unaligned_scenes_unexplained` (which should be empty; a non-empty
value is a genuine resolution regression, per the meth-F11 comment
around line 1528).

A per-leg diagnostic probe (`_leg_probe_rows`, around line 831, added
for the #59 investigation — chapter 6) records, per leg: our resolved
goal and instance, the distance from our OWN driven path to that goal,
and the distance from our goal to the real GT reference trajectory —
splitting "our path missed the goal" from "our goal was wrong" as
distinct failure modes. This field is probe-only and never read by the
scoring path itself.

## `gt_leg_ceiling.py` — the reachability ceiling (T11)

[src/core/runner/gt_leg_ceiling.py](../src/core/runner/gt_leg_ceiling.py) —
answers "how good could the ordered-leg score possibly be, even with a
perfect resolver?" For every IF question, it resolves the scorer's
ordered leg goals exactly as `gt_battery._if_rubric_geometry` does, then
measures the GT REFERENCE trajectory's own minimum distance to each
(mapped into the object frame by the same per-scene rigid fit the
battery uses). A leg counts "reached" at `<= LEG_ARRIVAL_TOL_M` (0.8 m).
This makes the "GT reference reaches K/N leg goals" claim reproducible:
since even the canonical correct path can only score up to K/N, any
battery gap above K/N is a scorer/geometry property, not a pipeline
defect (module docstring).

## `cvsweep.py` — cross-validated threshold calibration

[src/core/runner/cvsweep.py](../src/core/runner/cvsweep.py) — leave-3-
scenes-out cross-validated search over the geometry `Thresholds` family
(module docstring, line 1): 5 folds × 3 held-out scenes over the 15
training scenes. The stated rationale (lines 10-18) is exactly the
repo's generalization concern: a single global argmax over all 15
scenes overfits — a threshold that flips one borderline predicate in
one scene can win with zero evidence it helps elsewhere. Each fold's
best config is frozen and evaluated on its own held-out scenes; the
final recommendation keeps only values with modal consensus across
folds, and flags parameters with no consensus `"unstable — keep
default"` rather than moving them on thin evidence.

The composite objective (lines 20-44) mirrors the actual challenge
rubric, not a proxy metric: NUMERICAL scores strict independent-count
agreement only over questions with real independent evidence (the same
`*_class_only`-exclusion discipline as `gt_battery.aggregate`);
OBJECT_REFERENCE scores IoU >= 0.25 over scoreable questions only;
INSTRUCTION_FOLLOWING scores `score_instruction_rubric` over the
*driven* trajectory on aligned questions only — replacing an earlier
planned-path-coverage@1m proxy the challenge rubric doesn't actually pay
for. Threshold injection reaches only the pipeline/subject under test
(scoring functions and the `InstructionHead` driving the trajectory),
never the measuring instrument — the GT target/independent count an
answer is checked against is threshold-independent (lines 46-55).
`generalization_gap()` (around line 750) reports train-score minus
held-out score directly in every sweep report.

## Driven-polyline densification fix (#58, commit 478ecef)

Not in `runner/` itself but consumed by every IF scorer described
above: [src/core/groundtruth/scoring.py](../src/core/groundtruth/scoring.py)
around line 1102. The v1 kinematic follower emits one pose per planned
waypoint with no densification, so a driven route that passed within
centimetres of a leg goal mid-segment could score "not reached" if both
bracketing waypoints happened to sit farther than `LEG_ARRIVAL_TOL_M`
(0.8 m) away. `_densify_polyline` (around line 1172) linearly
interpolates the driven trajectory to `ARRIVAL_RESAMPLE_STEP_M = 0.25`
m spacing before any arrival/threading/capsule check — every inserted
point lies exactly on its original segment, so segment-intersection-
based checks (threading, avoid capsules) see identical verdicts
before/after; only the arrival-distance check, which only ever looked at
polyline vertices, changes behavior. `score_instruction_rubric` (around
line 1229) calls this immediately after building the trajectory array
and before any per-leg check.

## `provenance.py` — self-describing results

[src/core/runner/provenance.py](../src/core/runner/provenance.py) —
`collect_provenance` stamps every battery/sweep results JSON with which
tool produced it, when, from which git commit, and against which
calibration snapshot (module docstring, line 1-6, explicitly framed as
closing the gap "a fabricated-attribution incident exploited"). It is a
hard non-raising boundary: any git/subprocess failure degrades the
affected field to `None` and appends a human-readable reason to
`"note"` rather than raising, so a broken git state can never take down
a battery run (line 8-10). Untracked files over 16 MiB are digested by
name+size only, never read into memory (`_MAX_UNTRACKED_DIGEST_BYTES`,
around line 32) — a stray PLY or rosbag left in the tree must not become
a stamping hazard.

## `core/calibration.py` — the tunable ledger

[src/core/calibration.py](../src/core/calibration.py) — single source
of truth for every tunable constant across `core/`, without refactoring
the owning modules (module docstring, lines 1-19). It **composes** the
config dataclasses that already exist (`geometry.Thresholds`,
`perception.fusion.FusionConfig`, `perception.tracker.TrackerConfig`/
`KeyframeConfig`) and **mirrors** the loose module-level constants that
don't have a dataclass home yet as new dataclasses (`NavTunables` around
line 46, plus a `BudgetTunables`). A pinning test
(`tests/test_calibration.py`) imports the live modules and asserts
`default_calibration()` still matches them, so drift between this
ledger and the code fails loudly rather than silently. `docs/
calibration.md` documents all 64 fields (18 geometry, 5 fusion, 2
tracker, 3 keyframe, 21 nav, 15 budget), whether each is wireable today
(constructor arg / function param) or still a bare module constant
needing a wiring change (11 of the 64) before a sweep can move it.

## The generalization protocol (`docs/calibration.md`)

[docs/calibration.md](../docs/calibration.md) around line 21, added 19
Jul 2026 in response to a user-raised overfitting concern: since
evaluation runs on held-out scenes and freshly generated questions, any
tuning decision that only looks good against the 75 training questions
is self-deception. Three standing rules apply to every tunable, prompt,
or method decision from that date forward:

1. **Scene-level holdout** — tune on a training-scene subset, report on
   >= 4 scenes never used for tuning (mirroring eval's own unseen-scene
   structure). No swept value is trusted on tuning-scene evidence alone.
   `cvsweep.py`'s 5-fold/3-held-out-scenes design is this rule
   operationalized.
2. **Spec over sample** — a value justified by the VLA-3D generator
   spec or upstream stack constants outranks one justified only by
   "improved the 75"; sample-only justifications are flagged in the
   calibration ledger's Evidence column and must still pass rule 1.
3. **Generated held-out questions** — the question-generator spec
   (`docs/prior_art/`) permits generating novel questions for a private
   test set; method-level A/B choices (e.g. an LLM parse-tier route
   decision) report on those, not only the shipped 75.

The doc names its own highest current fitting-risk items as of the last
update: the GroundingDINO question-pass threshold of 0.25 (single-scene
fit, `japanese_room` probe — flagged as needing rule-1 validation), and
the parser construction fixes #23/#25 (flagged acceptable under rule 2,
since they're generator-template-derived rather than sample-fit).

## Where evidence lands

Every battery/sweep run writes to a dated or issue-tagged directory
under `reports/` (git-tracked history, never overwritten) — for
example `reports/gt_battery_postT11_2026-07-14/`,
`reports/gt_battery_probe59/`, `reports/local_llm_phase2/`,
`reports/issue59_probe.md` plus its companion
`reports/issue59_leg_ceiling.json`. `provenance.py`'s stamp is what
makes each of those artifacts self-describing after the fact.

## What I could not verify

- I did not execute `gt_battery.py`, `cvsweep.py`, or
  `gt_leg_ceiling.py` myself in this session (they require the VLA-3D
  Unity ground-truth data root, which I did not confirm is present in
  this worktree) — the numbers I cite (e.g. the #59 probe's 52.8%
  ceiling, 12/72 reached) come from reading `reports/issue59_probe.md`
  directly, not from a fresh run.
- I did not confirm the current contents of `docs/gt_answers_numerical.json`
  or that it is up to date with the latest scene set; I only confirmed
  `gt_battery.py` reads it as the numerical answer key.
