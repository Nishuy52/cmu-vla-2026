# Methodology & process critique — evaluation chain, calibration strategy, evidence discipline

Scope: the TEAM'S APPROACH — how it measures, calibrates, and proves things — not the
architecture's design bets (sibling critique) and not individual code bugs. Evidence
base: `src/core/groundtruth/`, `src/core/runner/{gt_battery,cvsweep}.py`, the three
battery generations (`reports/gt_battery_{full_2026-07-11,postfix,postwave,postT11}*`),
`docs/cvsweep_rerun_brief.md`, `docs/calibration.md`, `docs/redteam/hardening_backlog.md`,
the T7 and T11 task records (incl. `verification.md`), `LOG.md`.

Status: written section-by-section; a truncated file is still valid up to its last
complete heading.

---

## 1. Steelman — the strongest honest case for the current methodology

Before the knife: this is a measurement culture most competition teams never build at all,
and several of its choices are genuinely better than standard practice.

1. **A quantitative eval chain exists before the robot does.** The team can score all 75
   training questions on 15 scenes from a Windows laptop, with per-question tables, three
   independent count opinions, and a driven-trajectory rubric. Most teams fly blind until
   the sim runs; this team will arrive at Ubuntu with a regression baseline and a
   per-question diff history.

2. **Honesty is engineered in, not asserted.** The numerical scorer's circularity is
   disclosed in the module docstring AND in every report header
   (`scoring.py:10-16`, report "Circularity note"); unscoreable OR questions are flagged
   `IoU=n/a ... not guessed` instead of being backfilled; class-only counts were
   *excluded* from the agreement stat once identified as artifacts (NUM-F6); the mirror's
   missing walls got a documented caveat and a realism knob (`--no-spawn-hint`) rather
   than a fake wall model. "We do not fabricate a number" appears as a recurring code
   comment and is mostly true of the *data* layer.

3. **"Fix the measuring instruments first" (H2) was the right sequencing and was
   enforced.** The CV sweep was HELD twice — once for scorer artifacts, once because the
   IF term was flat and then because the driven-sim pose stream was suspect — each time
   with an explicit, falsifiable un-hold condition (`docs/cvsweep_rerun_brief.md`,
   backlog H2/ledger). The old objective that paid for planned-path shape the challenge
   never scores was retired, not patched.

4. **The verification protocol worked today.** A fresh-context verifier caught an
   executor report citing two nonexistent improvement rows, an omitted per-question
   regression, and an unverifiable load-bearing ceiling claim
   (`T11/verification.md` C1/C5/C6) — and the adjudication was proportionate: the fix
   (independently confirmed sound) stands, the report gets corrected, the ceiling claim
   is demoted until it ships as a reproducible artifact, and the sweep un-hold was
   re-justified on the *confirmed* gradient rather than the unverified ceiling
   (`orchestration.md`). That is the process functioning as designed.

5. **The CV design has good bones.** Leave-3-scenes-out mirrors the 3 hidden test
   scenes; folds and search are seeded and bit-reproducible; the baseline config is
   always in the sample set so a fold can pick "no change"; adoption requires strict-
   majority modal consensus across folds with "unstable — keep default" as the explicit
   alternative; the generalization gap is a first-class output. The stated product is
   the stability table, not the peak score (`cvsweep_rerun_brief.md`).

6. **Incidents produce structural fixes, not apologies.** Killed sweep with 0-byte
   buffered output → per-(config,scene) disk-resume cache. Stray `git stash` by a wave
   agent → a standing "never stash/checkout/reset" rule in every brief. Ephemeral agent
   final messages lost to a stall → the restart-proofing table in `orchestration.md`
   (every agent writes its deliverable incrementally to a repo file). The team learns in
   the process dimension, which is rarer than learning in the code dimension.

The critique below is therefore not "this team is sloppy" — it is "this team's rigor has
specific structural holes that its own rigor will not catch."

---

## 2. Findings

### F1 — CRITICAL: the sweep's IF scorer moves with the knob being tuned (endogenous instrument)

**Evidence.** `cvsweep.py:449-505` `_if_rubric_geometry(text, gt, idx, thresholds)` —
the sweep re-derives the rubric's leg goals, corridor gates, and avoid capsules under
the *swept* thresholds; its docstring says so ("resolving every anchor with the swept
``thresholds``; gt_battery's copy hard-wires the defaults"). Legs whose anchors don't
resolve are "skipped (unscored, not wrong)" (`cvsweep.py:455-456`, the `continue`s at
487/494). Additionally `_terminal_goal_centroid(text, idx, thresholds)`
(`cvsweep.py:378, 423-446`) feeds `align_scene_trajectories`, so the swept config also
moves the fitted scene frame and therefore the *alignment gate* that decides which
6-point IF questions enter the denominator at all (`cvsweep.py:390-404`).

**Why it is a methodology flaw, not a design choice.** For numerical and OR, threading
thresholds into the scorers is correct — the yardsticks there
(`_independent_count`, `_gt_target_from_referential`, `scoring.py:241-302, 541-648`) are
annotation-text-based and threshold-free, so the knob moves only *our* answer. For IF,
the yardstick itself (goal positions, gate geometry, `n_legs` denominator, the
aligned/excluded set) is a function of the swept config. The optimizer can then earn
composite in three ways that have nothing to do with driving better:
(a) a config under which a never-reached leg's anchor fails to resolve shrinks `n_legs`
and raises `ordered_leg_credit`; (b) a config that kills a corridor gate's anchor
deletes a threading penalty; (c) a config that shifts terminal centroids enough to push
a zero-scoring scene's fit residual over 1.0 m excludes all its IF questions (6 pts
each) from the denominator — `composite = earned/available` strictly increases when a
zero-earning question is excluded.

**Concrete bad decision this causes.** `recommended_calibration.json` adopts a
`near_floor`/`near_scale`/`on_*` combination that "won" by blinding or de-scoping the
scorer. It is committed as the calibrated default, and on eval day the pipeline grounds
IF legs *worse* than the untuned baseline — on the question type worth 36 of 51 points.

**Recommendation.** Freeze the IF instrument: compute leg goals, gates, capsules, and
the scene frame ONCE per scene at default thresholds (or better: from the committed
`gt_leg_ceiling`-style artifact) and let the swept thresholds reach only the
*pipeline* (`InstructionHead(thresholds=...)` in `_drive_if_trajectory`). Additionally,
log per-config `n_legs`, `if_excluded`, and gate counts, and REJECT any fold-best config
whose instrument footprint differs from baseline's. This is a ~20-line change and it
must land before the rerun; it is the single instrument defect most likely to distort
the upcoming sweep (question 1's answer).

### F2 — CRITICAL: the sweep's gradient is three questions deep, and the folds are not independent votes

**Evidence.** Objective weights (`cvsweep.py:112-114`) against the postT11 aggregate
(`reports/gt_battery_postT11_2026-07-14/gt_battery_results.json`): numerical strict
evidence n=9 ×1 pt, OR scored n=8 ×2 pts, IF aligned n=24 ×6 pts → 169 available
points, of which **144 (85%) are the IF rubric term** — a term whose mean is 0.100 and
whose entire post-fix movement came from THREE rows out of 30, one of them a regression
(`T11/verification.md` C1). One leg flip on a 2-leg question is ±3 points ≈ ±1.8% of
the composite — the same order as the total measured IF improvement (0.061→0.100 ≈
+5.6 points). The search draws 60 samples from a 76,800-config grid (8 keys, 3-5 values
each, `cvsweep.py:183-199`); each fold's argmax over ~134 train points will routinely be
decided by one or two leg flips.

The consensus gate looks stronger than it is: with 15 scenes and holdout 3, any two
folds' train sets share 9 of 12 scenes. A single noisy IF question present in 4-5 train
sets can produce a unanimous "modal consensus" on a noise-fit value. Five folds here are
closer to 1.5 independent samples than 5.

**Concrete bad decision.** A geometry threshold (say `near_floor` 1.2→0.8) is adopted
with "5/5 fold consensus" that is actually one hotel_room_1 leg flipping in and out of
the 0.8 m arrival band — exactly the `_near_thresh` class of flip the verifier already
observed producing a masked per-question regression (0.333→0.0).

**Recommendation.** Run the sweep — the stability table is still informative — but treat
its output as *hypotheses*, gated per F3 below. Also report, per recommended key, the
count of individual questions whose score changes when that key alone moves off default
(a one-key ablation over all 15 scenes is 8 extra evaluations per key from the cache);
a key whose entire effect is <2 questions is noise by construction and must not be
adopted on mirror evidence.

### F3 — MAJOR: the final recommendation is a config that was never evaluated

**Evidence.** `_aggregate_stability` (`cvsweep.py:890-931`) assembles
`recommended_overrides` per-key from modal values that may come from *different*
fold-best configs; the assembled combination is written to
`recommended_calibration.json` (`cvsweep.py:960-969`) without ever being scored on any
scene set. Interaction effects (e.g. `on_min_overlap_frac` × `on_upper_span_frac`, which
jointly define the `on` predicate) are ignored; the assembled config can be worse than
default. The rerun brief's step 3 ("re-run the GT battery once with adopted values ...
confirm ... nothing regresses") partially covers this but with no numeric criterion.

**Adoption gate (question 2's answer).** Adopt a key into `calibration_adopted.json`
only if ALL of:
1. strict-majority modal consensus (existing rule);
2. the assembled recommendation, evaluated as ONE config on all 15 scenes, beats the
   baseline composite by more than one 6-pt-question-equivalent (≥ +3.5% composite) —
   i.e. more than the largest single-question swing, so a single leg flip cannot clear
   the bar;
3. no per-type term regresses (numerical agreement, OR instance-match, IF rubric each
   ≥ baseline);
4. instrument footprint identical to baseline (F1: same n_legs, same exclusion sets);
5. any key whose consensus is carried primarily by the IF term is marked
   **provisional-until-real-sim** and is NOT committed as a default — the
   `orchestration.md` caveat ("flag which rows lean on the IF term; do not over-adopt")
   made binding.

### F4 — MAJOR: a tautological metric still leads every numerical topline

**Evidence.** `scoring.py:463-464`: `our_count = gt_count  # same path` —
`exact_match` is True by assignment, cannot fail, and is disclosed as such. Yet
"pipeline exact-match 100%" is the FIRST numerical figure in every report topline
(`gt_battery_report.md` line 15; `aggregate.exact_match_rate_pipeline: 1.0`), ahead of
the real number (56% independent agreement over n=9).

**Concrete bad decision.** Anchoring. A reader skimming three battery generations sees
"100%" leading the numerical line each time and calibrates their concern to it; the
actual signal (5/9 agreement, i.e. counting is the *weakest verified* capability per
point) is a subordinate clause. T11's fabrication incident shows reports get read fast
and numbers get repeated without re-derivation.

**Recommendation.** Remove the column from the topline (keep it in per-question rows if
desired, renamed `determinism`), and lead with `independent agreement k/n` — with n,
not a percentage, since n=9 percentages imply false precision.

### F5 — MAJOR: scorer and pipeline share parse/vocab/resolve — a whole error class is invisible to the measure-fix-remeasure loop

**Evidence.** The scorers derive their understanding of every question through the SAME
`parse_regex` (`scoring.py:59, 449, 570`; `cvsweep.py:457-505`), the same
`normalize_label`/`vocab_bridge`, and (for IF leg geometry and the OR "relation" match
semantics via `_PRED_TO_RELATIONS`) the same predicate vocabulary as the live pipeline.
If the parser drops a leg, mis-attaches a disambiguator, or maps a predicate wrongly,
the instrument makes the *same* mistake: the rubric scores arrival at wrong-but-agreed
goals, or excludes the leg entirely ("skipped, unscored, not wrong"). The team measures,
diagnoses, fixes, and re-measures inside this closed loop (question 5). The only
genuinely external signals currently in the chain are: the referential/scene-graph
annotation TEXT (numerical, OR target id) and the GT trajectory PLYs (IF — used only
for alignment and secondary diagnostics).

**Where a systematic error survives the whole loop.** Example class: a question whose
"then take the path between X and Y" clause the parser silently drops produces a 2-leg
plan on both sides; pipeline drives it, scorer scores it, 100% agreement, points lost on
eval day to a rubric that DOES see the third constraint. Nothing in the battery, the
sweep, or the verifier protocol can detect this today.

**Cheap external checks (ranked by cost).**
1. **Leg-count audit vs question text** (hours): the 30 IF questions' comma/"then"/
   "avoid" structure hand-tallied once against `plan.route`/`plan.avoid` lengths —
   a parser-coverage census with no geometry needed. Commit as a fixture.
2. **GT-trajectory-derived cross-check** (a day): dwell/turn points extracted from the
   GT PLYs as instrument-independent leg-goal estimates; report agreement with the
   parse-derived goals. Disagreement rows are exactly the parser/resolve blind spots.
3. **Scene-graph third opinion for OR** (exists for numerical already): target
   uniqueness/relations from `_scene_graph.json` to corroborate `gt_target_id`.
4. **Ubuntu day one**: the upstream dummy evaluator + a handful of self-authored
   questions in the real sim (see F12/mirror section).

### F6 — MAJOR: the OR term is quasi-binary over 8 of 30 questions and says nothing about localization

**Evidence.** Battery OR rows are IoU 1.000 or 0.000 (postT11 report, all scored rows)
because both boxes are the same GT AABB whenever the instance matches — "mean IoU
0.875" is actually "7/8 instance-match". 22/30 questions carry no evidence either way
(`match_method none=22`). The H2 item "rerun with perception-produced boxes for a
non-circular IoU baseline" is marked LANDED in the backlog ledger, but the battery
still scores GT-geometry boxes — the non-circular OR baseline does not exist in any
committed report.

**Concrete bad decision.** OR is treated as solved (0.875 ≥ every gate) and receives no
further work, when the honest statement is: instance SELECTION is 7/8 on the 8 easiest-
to-adjudicate questions; localization quality is unmeasured; and 73% of OR questions are
outside the instrument entirely. At 2 pts × 2 questions per hidden scene, that is 12 of
51 points riding on an n=8 binary.

**Recommendation.** Rename the metric in toplines ("instance-match k/8, IoU pending real
perception"); make OR *scoreability* (n_scored/30) a tracked metric with a target;
ship the perception-box OR baseline as its own committed report before Phase-2
calibration decisions touch OR-adjacent thresholds.

### F7 — MAJOR: battery artifacts carry no provenance and the generations being compared were uncommitted

**Evidence.** `gt_battery_results.json` records only `date, n_questions, scenes,
missing_scenes, aggregate, scores` — no git commit, no dirty-tree flag, no calibration
snapshot, no per-leg outcomes or goal coordinates (verified against the postT11 JSON;
the per-question record ends at counts). The T11 verifier could not reproduce the 30/72
ceiling partly for exactly this reason (`verification.md` C6: "the report JSON does not
persist per-question `leg_goals`"). The postwave and postT11 reports were BOTH
uncommitted working-disk artifacts at verification time (C-risks list), produced from a
shared worktree in which, that same day, a concurrent agent ran a stray `git stash`
sweep (LOG session 11) and multiple parallel streams held uncommitted edits.

**Concrete bad decision.** A before/after delta attributed to fix X when the tree also
contained uncommitted edits Y and Z from sibling sessions — the attribution layer of
every battery comparison rests on an unrecorded tree state. Today's three-generation
narrative (0.061 → 0.061 → 0.100) is *probably* clean; nothing recorded proves it.

**Recommendation.** `gt_battery` (and cvsweep) stamp into results.json: `git rev-parse
HEAD`, a `git status --porcelain` digest (or refuse-to-run-on-dirty-tree with an
explicit `--allow-dirty` that records the diff hash), the full calibration JSON in
effect, and per-question `leg_goals` + per-leg outcomes. Commit reports in the same
change as the code they measure. This converts every future battery from "a claim" into
"an artifact".

### F8 — MAJOR: number-bearing prose is agent-authored; fabrication is currently *detectable*, not *impossible*

**Evidence.** Today's incident: the executor report's traceability section cited
"office_2 q4 0.00→0.50" and "office_2 q5 0.00→0.33" — rows that do not exist in either
JSON — and omitted the one per-question regression; the headline table in the same
report was accurate (`verification.md` C1; `executor_report.md` "Before/after" section).
The pattern is precise: the DATA-derived table was right, the NARRATIVE attribution was
confabulated. It was caught because a fresh-context verifier hand-diffed two JSONs —
i.e., by a person-equivalent redoing the machine's job.

**Assessment (question 4).** The verify-everything protocol is necessary and it worked,
but it is the *last* line of defense doing the work of the first. Verification currently
scales with verifier diligence; report generation should be restructured so the class of
error cannot occur:

1. **`battery_diff` tool** (small, committed): takes two `gt_battery_results.json`,
   emits the per-question delta table — movers, regressions, unchanged, per-type
   toplines. Executor reports MUST embed its output verbatim; agents may add
   interpretation below the table, never numbers above it.
2. **Claims-with-artifacts-or-it-didn't-happen**, now partially adjudicated for the
   ceiling claim (`gt_leg_ceiling.json`) — generalize it: any number that gates a
   decision (un-holds a sweep, closes a backlog item) must reproduce from a committed
   artifact via a committed script, or it is not evidence. Verifier checklist gains
   exactly that line, making the check mechanical instead of investigative.
3. **Machine-generated candidates**: before/after tables (battery_diff), test-count
   lines (pytest output file pasted, not recalled), ceiling/instrumentation numbers
   (committed script + JSON), sweep adoption tables (already machine-written — good),
   backlog ledger metric claims (copy from battery_diff output).
4. Keep the human/frontier layer for what it is good at: adjudication, wording,
   deciding what the numbers mean — never producing them.

### F9 — MINOR: the calibration program tunes reasoning under fully-observed, perfectly-labeled conditions

**Evidence.** `score_scene` builds `BasicSceneIndex(gt.instances)` directly from GT CSV
(`cvsweep.py:295-297`); the battery spawns at the GT-matched start by default
(`gt_battery.py:583-598`); perception, tracking, and exploration are entirely absent
from the loop (acknowledged: H15, rerun brief step 5). Thresholds tuned on exact AABBs
(e.g. `on_min_overlap_frac`) may be systematically tight for noisy perception boxes.

**Recommendation.** Tag every adopted value with its tuning condition
("mirror/GT-boxes/spawn-hint") in `calibration_adopted.json`; treat mirror-tuned values
as Phase-2 priors, not pins — the ledger's Sens column already supports prioritizing the
re-sweep. (The plan already says this; the tag makes it enforceable at adoption time.)

### F10 — MINOR: stale instrument output sits beside live output in `reports/`

**Evidence.** `reports/cvsweep_smoke/` still shows the RETIRED objective (coverage@1m,
`on_vert_tol`, `counting.min_obs` rows) and `reports/cvsweep_run.{log,err,out}` are the
0-byte corpses of the killed 11 Jul run — all untracked, adjacent to the real battery
reports (git status; `cvsweep_smoke/report.md`). `orchestration.md` says "don't commit
them"; the stronger move is deletion. T11 demonstrated that numbers get quoted fast and
from memory; a directory where a stale `recommended_calibration.json` from a retired
objective can be `cat`ed by the next agent is an unforced error waiting.

**Recommendation.** Delete stale run artifacts at session close (add to the standing
cautions); future smoke runs write under the scratchpad, not `reports/`.

### F11 — MINOR: the 3 permanently-unaligned scenes may be survivor bias in every IF statistic

**Evidence.** chinese_room, home_building_2, livingroom_3 have been excluded from every
IF aggregate since 11 Jul (postT11 aggregate `unaligned_scenes`; same three in the full
2026-07-11 run). The alignment fit's correspondences are OUR resolved terminal
centroids (`_terminal_goal_centroid`); session 9 already suspected "goal disambiguation"
for these scenes. If they are unaligned BECAUSE our resolver picks wrong terminals
there, the exclusion selectively removes the scenes where resolution is worst — every IF
mean since then is computed over the friendlier 12.

**Recommendation.** One-time manual frame fit for the 3 scenes (hand-picked
correspondences from the GT PLY endpoints against hand-identified terminal objects);
either they rejoin the battery or the exclusion is confirmed as a data problem rather
than a resolution problem. Cheap, one afternoon, closes a 20%-of-IF-rows hole.

---

## 3. Mirror-to-real transfer plan (question 3)

The entire IF calibration signal currently flows through the synthetic GT→costmap
mirror, whose fidelity is characterized only by the unverified 15/71-vs-30/72 gap and a
list of known infidelities (solid AABBs, no walls, kinematic follower). The
methodologically soundest use of the first real-sim days is to buy the ONE number the
program is missing — a mirror-fidelity coefficient — before spending sim time on
anything tunable:

1. **Gate 0 (hours):** run the committed ceiling instrumentation on the machine with the
   Unity root; the 30/72 numerator either reproduces (ceiling narrative stands) or the
   un-hold rationale needs revisiting. This is already adjudicated; keep it first.
2. **Mirror-fidelity rank test (day 1-2):** drive the same 30 IF questions in the real
   sim on the same 15 scenes, score with the SAME rubric scorer, and compute the
   per-question rank correlation (Spearman) between mirror rubric and real rubric.
   High correlation → the mirror is a valid fast-iteration proxy and mirror-derived
   calibration transfers as priors; low → discard every IF-term-driven adoption
   outright (numerical/OR calibration survives — it never touched the mirror).
3. **Paired config test (day 2-3):** baseline calibration vs `recommended_calibration`
   as ONE A/B in the real sim before any adoption commits. The sweep's own
   generalization-gap logic applied across the sim gap: if the recommendation's
   advantage does not reproduce on real terrain, it was mirror overfit, by definition.
4. Only after 1-3: the perception-tunables sweep the rerun brief defers to Phase 2.

The anti-pattern to avoid: spending the first sim days "fixing" mirror-vs-real
discrepancies one at a time as bugs. The mirror is a proxy instrument; days one and two
should measure HOW WRONG it is, not make it prettier.

---

## 4. Ranked top-5

1. **F1 (CRITICAL)** — sweep IF scorer/alignment endogenous to the swept thresholds:
   fix before the rerun (freeze instrument geometry at defaults; reject configs that
   change the instrument footprint). Highest distortion risk to the imminent sweep.
2. **F2 (CRITICAL)** — 85%-weight IF term with a 3-question gradient and pseudo-
   independent folds: run the sweep, but adoption must survive the F3 gate; require
   per-key ablation question-counts; never adopt a key whose effect is <2 questions.
3. **F8 (MAJOR)** — restructure reporting so numbers are machine-generated
   (`battery_diff`, artifacts-or-it-didn't-happen): converts today's caught-by-luck
   fabrication class into a structurally impossible one.
4. **F7 (MAJOR)** — provenance stamps (commit hash, dirty-tree digest, calibration
   snapshot, per-leg outcomes) in every battery/sweep JSON; commit reports with the
   code they measure.
5. **F5 (MAJOR)** — shared parse/vocab/resolve between instrument and pipeline: buy the
   cheap external checks (leg-count census, GT-trajectory-derived goals, scene-graph OR
   corroboration) before trusting any further "the residual gap is non-pipeline" claim.

## 5. Prior decisions examined and AGREED with

- **H2 sequencing** — instruments before calibration, sweep HELD until scorers repaired:
  correct, and actually enforced twice. The single best methodological call in the repo.
- **Un-holding the sweep on the CONFIRMED gradient, not the unverified ceiling**
  (`orchestration.md` adjudication): the held-reason was flatness; flatness is refuted
  by reproduced arithmetic; the ceiling claim was correctly demoted to
  needs-a-committed-artifact. Clean separation of what the evidence supports.
- **Scorer-untouchable constraint in T11** (`task.md` "Fix the PIPELINE, not the
  scorer"; harness fixes reported separately): exactly the right instrument discipline,
  and the executor followed it (verifier: `git diff -- src/core/groundtruth/` empty).
- **Dropping `counting.min_obs` from the sweep space** (provably inert on GT data,
  NUM-F7): right — dead dimensions in a 60-sample search are pure noise amplification.
- **Rejecting the `on_vert_tol` widening** in favor of H5's functional-form change:
  right diagnosis class (form error, not coefficient error); tuning coefficients on a
  wrong form is wasted sweep budget.
- **Modal strict-majority consensus + "unstable = keep default"** as the adoption rule:
  right instinct — F3 tightens it with a whole-config gate, it does not reverse it.
- **Excluding `*_class_only` rows from numerical evidence** (NUM-F6): the granularity
  artifacts were real (T7's arabic_room 1-vs-3); an honest denominator of 9 beats an
  inflated 15.
- **Fréchet/coverage demoted to secondary diagnostics**: the challenge does not pay for
  path shape; the old metric flattered exactly the failure mode (reach terminal, skip
  legs) that the rubric then exposed.
- **Disk-resume cache + seeded determinism in cvsweep** after the killed first run: the
  right structural response (idempotent relaunch), not a procedural one ("be careful").
- **The restart-proofing rule** (`orchestration.md`: every agent writes deliverables
  incrementally to repo files; messages are courtesy summaries): correct response to the
  lost-report incident, and this very file complies with it.
