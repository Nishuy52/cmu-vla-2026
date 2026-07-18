# Hardening Backlog — adjudicated red-team findings

**Date:** 14 Jul 2026 (review batch ran 12 Jul)
**Inputs:** the five facet reports in this directory — `attack_instruction_following.md` (IF), `attack_numerical.md` (NUM), `attack_object_ref_checkpoints.md` (OR), `attack_eval_day.md` (SYS), `dossier_deltas.md` (DD).
**Points frame:** eval = 3 hidden scenes × (1 numerical ×1 + 2 object-ref ×2 + 2 IF ×6) = **51 pts; 36 are IF**.
**Adjudication rules applied:** findings ranked by expected points impact (points at risk × probability), deduplicated across reports (shared root causes merged into one item), and split by *when they can land* (Windows-now vs Ubuntu-gate). Where two reports independently confirmed the same defect, probability was revised up. Every item cites its source finding IDs; nothing here is new analysis.

## The one-paragraph read

Four defects dominate everything else: (1) the `/challenge_question` QoS mismatch can zero the entire run before any code we care about executes (SYS-F2); (2) instruction-following questions — 71% of points — deadlock by construction because the IF head waits for grounding and grounding waits for motion (IF-F1 ≡ SYS-F3, found independently twice); (3) the resolve ladder's silent fallbacks are the *shared* root cause behind the counting over-counts, the wrong-instance markers, and the wrong IF terminal goals (NUM-F1, OR-F1, IF-F3) — one design decision ("always answer something") metastasized into all three question types; and (4) our geometry predicates use different functional forms from the VLA-3D generation code that produced the questions, so `on`/`between`/`under`/`with` disagree with ground-truth semantics no matter how their coefficients are tuned (DD-A1..A8, empirically corroborated by NUM-F2). Everything measurement-related (H2) must land before the CV sweep reruns, or the sweep tunes against artifacts (NUM-F7, DD §A pin-list).

---

## Status ledger (updated 14 Jul — implementation day 1)

| Item | Status |
|---|---|
| H1 QoS | **LANDED** `4bcf18b` (dual-sub, VOLATILE primary), verified |
| H2 scorers | **LANDED** `4bcf18b` (driven-trajectory rubric, strict counts; walls: no usable source, no-spawn-hint knob shipped instead), verified |
| H3 IF explore | **LANDED** `8a8c78d`, verified (deadlock broken end-to-end) |
| H4a/b ladder policies | **LANDED** `4bcf18b` + scope-clause follow-up, verified |
| H4c provisional terminal | **LANDED** `8a8c78d`, verified — gate inert until `budget_frac` seam wired (see H8) |
| H5 predicate forms | **LANDED** `0863294` per the T8-reconciled spec, verified (92.3% recall vs real GT on-edges) |
| H8 pack | **LANDED** (wave, 14 Jul): seam wiring (ladder api→api2→regex; local tier DESCOPED; checkpoints; budget_frac/remaining_s late-bound), injection-boundary timeouts, ledger admission bound. Open remainder: worker-thread FSM restructure (Ubuntu gate) |
| H6 perception gate | **LANDED** (wave): PerceptionPipeline seam behind `VLA_DETECTOR`, playbook gate line. Integration itself Ubuntu-gated |
| H10 vocab | **LANDED** `4bcf18b`, verified |
| IF-F6 pair anchors | **LANDED** `8a8c78d`, verified |
| H7 docker | **LANDED** (wave): fork-shaped `docker/ai_module_fork/` + sync scripts; validation = compose-up from clean fork clone at the Ubuntu gate |
| H9 checkpoints | **LANDED** (wave): CP4 keep-on-neither + real runner-up + clause-synthesizing re-resolve; CP2 bbox/conf validation + provisional-commit guard; CP3 cross-class-only demote; CP1 prompt ≤2k tokens; CP5 abstain |
| H11 nav/drive | **LANDED** (wave): replan consumption (cap 3), avoid lifecycle + edge-triggered tripwire, forced-assembly continue-drive, explore-goal guards/clamps, CP2 latch clear, free-space via, 1 Hz frontier throttle. +2 integration fixes: CP3 mid-tick demote follower-None guard; decay clock counts detection-bearing keyframes only |
| H12 dimension priors | **LANDED** (wave): data-derived 69-class table + clamp seam; markers in object_ref + floors routed through it |
| H13 overhead soften | **LANDED** (wave): soft cost via is_unknown seam, clone() hardens for corridors, stride-scaled gate. Soft-cost weight applies as UNKNOWN_COST_MULT until a planner seam (Phase-2) |
| H14 perf/hygiene | **LANDED** (wave): BFS vectorized ~10× (bit-identical, oracle-tested), CP1 trim, CP5 abstain, second-question policy, encoder assert. Reference BFS impl kept for sim bake-in, then delete |
| H15 counting hygiene | **LANDED** (wave): detection-bearing-keyframe ghost decay, answer-time obs gating, coverage_frac seam (unwired — connect to an exploration coverage signal when chosen) |
| cvsweep re-point | **LANDED** (wave): objective = rubric×6/strict×1/IoU×2, min_obs dropped, disk-resume cache, tests |
| IF driven-sim | **LANDED** (wave): closed-loop harness (rewind + bounded re-tick); rubric numbers now trustworthy |
| **NEW: IF intermediate-leg threading** | **LANDED (T11)**: rubric 0.061→0.100, ordered-leg credit 0.094→0.122, threading 9→8; numerical (56%) + OR IoU (0.875) unchanged. Root causes: (pipeline) leg goal projected to an unreachable pocket → snap to nearest pose-REACHABLE cell; "near" via 1.2 m standoff overshot the 0.8 m arrival band → vehicle-radius clearance; plural leg anchor mis-singularised ("vases"→"vas") → e-final-stem fix. (harness) GT→costmap mirror stamped solid floor-to-top AABBs (incl. tabletop objects) → z-aware terrain slab; wedged driven-sim padded to watchdog → net-progress stall guard. Report: `docs/tasks/T11-if-leg-threading/executor_report.md`. Residual gap is ceiling-bound: GT reference itself reaches only 30/72 leg goals within 0.8 m of the raw centroid (scorer centroid-tolerance) and the mirror over-approximates footprints (harness fidelity) — NOT scored-path bugs |
| CV sweep | **COMPLETE (15 Jul) — verdict: KEEP DEFAULTS.** Run after the instrument-endogeneity fix (`81f2a2a`) + green full gate. All 5 folds chose the baseline config as best; no parameter reached consensus; mean holdout 0.229±0.036, gap −0.010 (no overfit). The adjudicated adoption gate passes trivially (recommendation == baseline). Report: `reports/cvsweep_2026-07-15/`. Post-H5 defaults stand as the calibration; re-sweep deferred to real-sim (Phase 2) per adjudication |
| **NEW: PDF answer keys (numerical)** | **EXTRACTED (15 Jul)**: the per-scene `questions.pdf` text layer contains the numerical answers verbatim — `tools/extract_pdf_answers.py` → `docs/gt_answers_numerical.json` (committed, provenance-stamped). **True numerical accuracy 11/15 (73%)** vs the 56%-over-9 proxy; proxies shown unreliable in both directions (livingroom_3 indep=10 vs truth 2 — pipeline was right; chinese_room indep=1 vs truth 6). True failures: arabic_room 0≠2, loft 0≠2 (zero-count relation misses), home_building_1 11≠6, home_building_2 3≠2 (over-counts). **WIRED (T12, 17 Jul)** — see below |
| **NEW: T12 numerical yardstick + provenance + pre-Ubuntu queue** | **LANDED (17 Jul, branch `feat/t12-numerical-yardstick`)**, two verifier gates CONFIRMED, full gate ~1140 passed. **meth-F4/F6 topline honesty**: battery leads with TRUE accuracy k/15 (`c88fa62`); pipeline exact-match renamed determinism; OR headline = instance-match/scoreability. **meth-F7/F8**: `provenance.py` stamps (commit/dirty-digest/calibration snapshot/per-leg IF outcomes) in `gt_battery`+`cvsweep` results (`8e3e903`); `tools/battery_diff.py` is now the sole legal before/after table source (`fb70a90`). **Numerical 11/15→13/15**: `under()`/`below()` gained a wall-relative lateral branch (inverse-`above`, no footprint-IoM gate — the vertical mirror of the H5 `above()` fix) fixing arabic_room + home_building_1 (`c867fce`); the 11 passes held row-for-row. Two diagnosed-unfixed color-annotation gaps deferred to the sweep (loft black→gray binning, home_building_2 maroon-dominance) → issues **#11/#12**. **meth-F5 leg census** (`3c9b35c`): 0/30 IF questions drop a leg — no silent-dropped-leg problem; fixture is a tripwire. **arch-F9 no-spawn-hint** (`ef58b07`): IF rubric is spawn-artifact-sensitive (0.100↔0.189), read mirror IF relatively. **meth-F11 unaligned scenes** (`1f0c99e`): fallback candidate-search rejoined chinese_room + home_building_2 (aligned 24→28/30); livingroom_3 exclusion data-confirmed (frame-independent distance contradiction); friendly-ward bias hole closed. **NEW BUG → issue #13**: terminal-goal resolver mis-ranks the wrong object (chinese_room, home_building_2) — the frame-fit only works around it; the same `resolve()` drives OR + real-sim GOTO. Records: `docs/tasks/T12-numerical-yardstick/` (diagnosis.md, unaligned_scenes.md, postT11_to_T12.md). OR/IF image answer-keys still pending visual transcription (arch-F3 residual, user action) |

## Tier 0 — run-killers (fix before anything else; all cheap)

### H1. `/challenge_question` subscription QoS: match the publisher or receive nothing — **51 pts, p≈0.85, cost S, Windows-now**
Sources: SYS-F2. Subscriber requests TRANSIENT_LOCAL; the closed-source evaluator almost certainly publishes VOLATILE (the dummy does); incompatible durability = zero messages = zero questions = zero answers for the whole run, and the watchdog is downstream of question receipt so it cannot fire. **Verdict: accepted verbatim.** Fix: subscribe RELIABLE+VOLATILE (like the dummy), optionally dual-subscribe both durabilities into the same latch. Add `ros2 topic info -v` verification to the Ubuntu gate checklist. This was already flagged in T4 as a runner-up risk — it was under-ranked there; it is the single highest expected-loss item in the system.

### H2. Fix the measuring instruments before touching calibration — **governs all other decisions, cost S–M, Windows-now**
Sources: IF-F2, NUM-F6, NUM-F7, DD §A pin-vs-sweep, OR-F6 (battery IoU circularity).
- IF scorer: score a *driven* (simulated) trajectory on ordered per-leg arrival + `threading_check` + `capsule_violated` — the oracles exist and are only called from tests today; keep Fréchet/coverage as secondary diagnostics. Add walls to the mirror scene; add a no-spawn-hint variant.
- Count scorer: strict class equality for target matching (no substring/token fallback), exclude `*_class_only` rows from agreement stats, add an annotation-coverage column.
- OR: rerun the battery with perception-produced boxes (not GT geometry) for a non-circular IoU baseline.
- **CV sweep: HOLD the rerun** until H2 + H4 + H5 land (its counting objective currently rewards artifacts; `on_vert_tol`'s swept range cannot reach the real gaps; `counting.min_obs` is inert). The rerun brief's stability-table adoption rule survives unchanged.
**Verdict: accepted; sequenced first because every downstream verdict (incl. the sweep) depends on trusting these numbers.**

### H3. IF explore-while-ungrounded — **36 pts, p≈0.9, cost M, Windows-now**
Sources: IF-F1 ≡ SYS-F3 (independent double confirmation), architecture §4 row 10. For IF questions `ExploreHead` never explores and `InstructionHead` publishes nothing until *every* leg grounds — a self-sustaining deadlock on unseen scenes; `next_noun_affinity_target` (the designed mitigation) is dead code called only by a test. **Verdict: accepted verbatim; highest-value single code change in the repo.** Fix: for IF, delegate to the instruction head *and fall through to frontier exploration* (noun-affinity-biased, capsule-aware) whenever no waypoint was emitted this tick; run the orientation sweep for IF too; drive the longest grounded prefix of legs (partial credit is real — banking grounded legs scores). Acceptance test: "IF question + empty scene ⇒ waypoints still published within N ticks".

### H4. Resolve-ladder fallback policy: split by question type — **root cause across all three types, cost S–M, Windows-now**
Sources: NUM-F1 (+DD-A13), OR-F1, IF-F3, IF-F6.
- **Counting (S):** no relaxation rungs inside `counting()` — noun+attributes+clauses only; empty ⇒ 0 with the failed-clause explanation. 0 is a legal answer. (Fired on 6/15 training numericals; answered 31 where truth ≈ 2–5.)
- **Object-ref (M):** make `_resolve_anchor` honor nested disambiguators (filter/rank anchor candidates by their own clause before evaluating the outer clause; tie-break survivors by the nested metric, never by instance id). 57% of OR questions have the nested form; both battery wrong-instance results are exactly this; probe-reproduced.
- **IF (M):** relaxation-audited groundings are *provisional* — do not commit a terminal leg resolved via `drop_relation`/`category_only` while budget remains (ties into H3's continued exploration); replace index-order/lowest-id salience with a defensible salience; distinct-instance constraint for "the two X" pair anchors (probe-confirmed zero-width gate collapses the whole route to recovery).
**Verdict: accepted as a family; the ladder itself stays (it is right for "must publish something"), the *consumers* get type-appropriate policies.**

---

## Tier 1 — high (land before the Aug 3 MVS)

### H5. Geometry predicate functional forms: match the VLA-3D generation spec — **pins before sweep, cost M, Windows-now**
Sources: DD-A1..A8 + the T8 reconciliation appendix in `dossier_deltas.md` (T8's instance-level GT evidence supersedes two generator priors), NUM-F2 empirics. Reconciled forms to pin: `on` = footprint IoM-over-min ≥ 0.5 + anchor-larger gate + target bottom within the supporter's upper z-span (band bounds sweepable) [T8-C2/C3]; `above` = lateral-offset tolerance (XY centre within inflated anchor footprint) REPLACING the overlap gate [T8-C4 + NUM-F2 — do NOT add an IoM gate]; `between` = pin strict-betweenness (0<t<1) only, keep the capsule, defer symmetry/IoM to the sweep [T8-C5 narrowed]; `next_to`/"beside"/"adjacent to" alias to `near`; `with` ≡ inverse-`on`; `under` gains the class-gated tuck-under branch; size resolver = relative per-class largest-face + 1.2× gap [DD-A12 = T8-C6]; wire or drop the dead `above_gap_max`. Leave coefficients to the (post-H2) sweep. **Verdict: accepted — a mismatch with the code that generated the questions is a scoring bug, not a style choice.** Watch: `on` also feeds OR/IF anchoring — regression-check those goldens.

### H6. Perception wiring is a gated deliverable, not a comment — **~48 pts if unshipped, cost: seam S now / integration L at Ubuntu**
Sources: SYS-F1. The committed adapter answers every question from `BasicSceneIndex([])`; the pano is latched and never consumed; the Dockerfile's detector install is commented out. Acknowledged Phase-2 work — the *defect* is that nothing gates submission on it and every smoke test passes with the stub. **Verdict: accepted.** Fix now (Windows): the `PerceptionPipeline` seam in the adapter following the `runner/single.py` scripted pattern, a loud startup log when the index is the stub, and a phase2_playbook gate item "instances_tracked > 0 on a live scene before any submission". Integration itself stays Ubuntu-gated. *Wiring-time check (verifier note, 14 Jul): `normalize_label` does not fold underscores — if the real detector ever emits underscore-joined class names ("night_stand"), the vocab bridge silently misses them; add the normalization when the detector lands.*

### H7. Docker/fork packaging restructure to the upstream shape — **submission-integrity risk, cost M, restructure Windows-now**
Sources: SYS-F4. Our Dockerfile location/context cannot slot into the fork's required `ai_module/docker/Dockerfile` + `ai_module/` context; PEP 668 will likely break bare pip on the Noble base; the colcon-symlink layout is one of the three "confirm on Ubuntu" flags. **Verdict: accepted.** Restructure the layout now (it is text); add `--break-system-packages`/venv defensively; definition of done = `docker compose up --build` from a clean clone of the fork. Update `ubuntu_setup.md` in the same change (standing rule 3).

### H8. FSM/adapter robustness pack — **several small categorical-zero holes, cost S each, Windows-now**
Sources: SYS-F5, SYS-F6, SYS-F7, SYS-F8.
- Correct `controller.qtype` from `plan.qtype` after parse (a motion-verbed OR question otherwise gets a WaypointCmd floor on the wrong topic: categorical 0).
- Budget-skew hedge: gates to 480/540 s or launch params (evaluator's clock starts at system startup, not question receipt); measure real skew at the Ubuntu gate. The ns-vs-s unit-mismatch lead was chased and cleared — no bug; add the `use_sim_time` assert.
- Wire the LLM ladder + checkpoint seams in the adapter (today the entire reasoning tier is dead code on the eval path) — and make the dark-network local-VLM tier a deliberate decision: ship it or descope it and amend architecture §3. *Add `budget_frac` to that wiring list (verifier note, 14 Jul): without it the H4c provisional-terminal gate commits immediately in production — safe (never strands) but the withholding guard never fires until the seam is injected.*
- Checkpoints off the tick thread (worker + polled future, or MultiThreadedExecutor) and enforce `with_timeout` on every injected seam inside `build_callables`; admission bound = reserve + per-call worst case (also OR-F7's repair-outside-envelope).

### H9. Checkpoint protocol repairs (CP2/CP3/CP4) — **guards all 60 OR-equivalent pts; cost S–M, Windows-now**
Sources: OR-F4, OR-F5, OR-F8, OR-F9, SYS-F11.
- CP4: wire the rich verifier (or map `neither`→keep in the bool seam) — the deployed seam currently *inverts* the design fallback and swaps to an unverified candidate the model never saw; adopt the proposed prompt wording; make `missed_constraint` a parseable `<pred> <anchor>` and synthesize a real clause so the re-resolve stops being a no-op.
- CP2: validate bbox bounds/ordering against tile dims; confidence floor ≥ 0.5; provisional-only winners are never committed as markers without re-observation; clear the provisional waypoint on arrival/timeout/observation (it currently latches forever and replaces exploration).
- CP3: demote only on class-different `actual_label` + high confidence; synonyms confirm.

### H10. Vocabulary: one matching stack for live and scorer — **cost S, Windows-now**
Sources: OR-F2, OR-F3, NUM-F3, NUM-F4, DD-A11, DD-A12. Move `vocab_bridge` into `BasicSceneIndex.by_label` (it currently fixes only the scorer while the live matcher has 4 synonym groups vs ≥7 proven drift pairs); typo tier: short-circuit when exact/synonym matches exist, length-scaled distance budget, and **aliases never participate in fuzzy matching** ("yellow"≡"pillow" at distance 2 added 13 phantom pillows); colour bridge for question-colour → 15-bin scheme names; deterministic size resolver (largest-face + 1.2× rule).

---

## Tier 2 — medium (post-MVS hardening, still Windows-doable)

### H11. Nav/drive robustness pack — sources: IF-F4, IF-F5, IF-F7, IF-F8, SYS-F9, SYS-F10
Consume `replan_flag` (set-only today; stall = parked forever): rebuild costmap, re-stamp capsules, re-plan, cap ≈3/question. Avoid capsules: grounding is a route-build precondition, re-stamp on re-ground, runtime pose tripwire via `capsule_violated`. IF budget-exhaust: keep ticking the head until route completion or watchdog (never abandon breadcrumbs for one distant terminal waypoint — stranding + ordered-constraint loss). Exploration: guard against pre-odom (0,0)/t=0 anchoring; route frontier goals through A*+breadcrumbs (or clamp ≤2.5 m); cap unreachable-frontier scores. Via points: free-space-gradient placement instead of the fixed +1.2 m +x offset. Route re-commitment when a leg's resolved instance changes or geometry moves.

### H12. Marker extents: per-class dimension priors — sources: OR-F6
The architecture's "dimension sanity table" does not exist in code; trimmed under-boxes vs GT over-boxes will shed IoU cliffs at 0.25/0.5 once real perception is in the loop. Seed the table from VLA-3D stats; clamp to class-min, inflate the unobserved axis toward class-typical. Measure against H2's non-circular baseline.

### H13. Overhead-clearance layer: soften and validate — sources: SYS-F12
Make overhead soft-cost except in corridor threading; scale `min_points_per_cell` with decimation stride; validate the five jingfan-fitted tunables on ≥2 more scenes' bags at the Ubuntu gate; decide (or document) the exploration-side asymmetry.
**Validation DONE 18 Jul 2026** on 5 bags / 4 scenes (`tools/overhead_validation.py`, `reports/overhead_validation_2026-07-18/`): tunables hold cross-scene; residuals filed as #36 (sensor-height constant low for sim rig) and #37 (per-object overhang misses). Exploration-side asymmetry still open.

### H14. Performance + hygiene — sources: SYS-F13, SYS-F14, OR-F10
Benchmark the pure-Python hot loop on a realistic grid now (frontier BFS ~50–150 ms/tick estimated); vectorize if >50 ms; throttle frontier detection to 1 Hz; decimate odom conversion. Loud-log + deliberate policy on second-question text. Gate-checklist line: grep compose/launch override for `debug_viz`. Trim the CP1 parse system prompt under the 2k-token budget; startup assert that a network provider isn't paired with the `.npy` encoder; CP5 abstain option.

### H15. Eval-time counting hygiene (untested-by-battery risks) — sources: NUM-F8
The battery never exercised fusion/tracker (index built from CSV) — the perception-side counting risks are unexonerated: track decay for n_obs=1 ghosts, answer-time min_obs gating, coverage-gated early answers for numericals. Validate in the Phase-2 sim (this is the perception-tunables sweep the rerun brief correctly defers).

---

## Explicitly rejected / deferred

- **Widening `on_vert_tol` to make pillows pass (NUM-F2 alternative):** rejected — degrades `on` into `near` and damages OR/IF anchoring worth 16× the numerical points. The fix is H5's form change, not a bigger tolerance.
- **`in` predicate rework (DD-A10):** deferred — zero training occurrences; our looser form doubles as the room-membership test.
- **TRANSIENT_LOCAL dual-subscription as the *default* (H1):** the simple volatile profile is primary; dual-sub only if paranoia is cheap at implementation time.
- **Reopening any architecture-v1.0 structural decision:** no report produced evidence against the deterministic-backbone / checkpointed-LLM shape; the 2025 dossier confrontation (DD verdict) strengthens it — every GT-marker-dependent 2025 pipeline is structurally inapplicable in 2026.

## Open questions flagged for team adjudication (from DD §C)

- **DD-C1 — semantic frontier scoring:** the 2025 winner (NROS) and 4th place both put a multi-modal/CLIP signal *continuously in the frontier scorer*, vs our rare bounded CP5 call. Evidence is thin (NROS's method is a news-article phrase — see DD "could not verify"), so this is a hypothesis, not a sourced fact. Decide after H3 lands (it's moot while IF never explores): extend the deterministic noun-affinity bias with a semantic value term, or keep CP5-only.
- **DD-C6 — IF leg targets via free-space projection:** both top-2 2025 teams pointed at a pixel and projected to a floor waypoint; our marker centroid doubles as the nav goal and can sit inside furniture (interacts with OR-F6's front-shell boxes). Decide whether IF leg targets route through a free-space projection independent of the marker centroid.
- **Process (DD-B14):** add a secrets scan to the session-end push protocol — three 2025 teams committed live API keys to public repos.

## Sequencing (what blocks what)

1. **H1 + H8-qtype** (hours): remove the categorical-zero holes.
2. **H2** (scorers) — before any calibration or sweep decision.
3. **H3 + H4** (the deadlock + ladder policies) — before re-running the GT battery, since they change every measured number.
4. **H5** (predicate forms) — then re-run the GT battery once → **then** the CV sweep (coefficients on the new forms; drop `counting.min_obs`, add the new `on`/`between` params).
5. **H6 seam + H7 restructure + H9 + H10** in parallel with the above (independent files).
6. Tier 2 as capacity allows; H15 waits for the sim.
7. Ubuntu gate additions from this review: QoS verify (H1), skew measurement (H8), overhead validation (H13), `compose up --build` from clean fork (H7), stub-index submission guard (H6).

## Cross-cutting lesson

Three of the five reports converged on the same root cause from different directions: **silent fallbacks that keep the pipeline "always answering" also keep it confidently wrong** — the relaxation ladder (H4), the CP4 bool seam (H9), and the recovery path that masks parser bugs as graceful degradation (IF-F6) are the same pattern. The watchdog philosophy ("silence is the only unforgivable failure") is correct for *publishing*; it was over-applied to *resolution*. The design rule going forward: fallbacks must be visible in the audit trail and consumers must be able to distinguish a clean result from a relaxed one.
