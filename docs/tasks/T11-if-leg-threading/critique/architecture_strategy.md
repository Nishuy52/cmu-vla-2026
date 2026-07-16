# Architecture & Competitive Strategy — Adversarial Critique

*2026-07-14. Scope: the bets themselves — architecture v1.0, the calibration/eval
methodology, and the remaining-schedule strategy — not implementation bugs (T10
covered those). Evidence base: challenge_brief, question_analysis, architecture v1.0
+ proposals debate, T8 adjudication, T10 redteam + hardening_backlog ledger, T11
task/executor/verification records, `reports/gt_battery_post{fix,wave,T11}_2026-07-14/`,
phase2_playbook, organizer_playbook, competitor_forks.*

---

## 1. Steelman — the strongest honest case for the current strategy

Before attacking it, the case for this architecture as it stands:

1. **The core bet is organizer-validated.** The deterministic-geometry +
   checkpointed-LLM split is not a guess; it is the division of labor the
   organizing lab itself built (SORT3D), documented in two theses, evaluated on
   the same dataset (VLA-3D) that generates the challenge questions. Zhang's
   thesis explicitly documents the failure mode (LLMs doing raw-coordinate
   geometry) that the toolbox design avoids. No competitor intelligence suggests
   anyone has a better-validated frame.

2. **Effort allocation tracks the points, on the build side.** IF is 70.6% of
   points and got the deepest machinery: ordered legs, corridor gates as
   mandatory via-segments with threading verification, hard avoid-capsules,
   interleaved explore-execute, partial-credit banking. Numerical (5.9%) got
   capped effort. This is exactly what `question_analysis.md` §8 prescribes.

3. **The team found the 2026 discontinuity before building the wrong thing.**
   T8/prior_art established that 2 of the 2025 top-3 answered from the
   ground-truth `/object_markers` topic, absent in 2026 — so 2025 methods
   don't port and 2025 scores don't calibrate difficulty. The architecture was
   amended (§8) rather than left chasing last year's leaderboard. Matching
   geometry predicates to the VLA-3D *generator* semantics (H5, 92.3% recall vs
   real GT on-edges) is a genuine moat: the questions were made by that code.

4. **The process is self-correcting at a rate most teams never reach.** In four
   days: a five-facet red team (T10), a prioritized backlog with points-weighted
   adjudication, fifteen hardening items landed and independently verified, a
   scorer rebuilt to measure driven trajectories instead of flattering Fréchet
   shape, and a verifier that caught fabricated attribution rows in its own
   executor's report and refused an unverifiable ceiling claim. The measurement
   discipline (fix instruments before calibrating, H2-first sequencing) is the
   single most impressive strategic behavior in the record.

5. **Failure floors are engineered, not hoped for.** QoS run-killer fixed, watchdog
   floor on every type, parse fallback ladder, per-question relaunch honored.
   "Silence is the only unforgivable failure" is the right frame for a
   one-shot, no-retry scored contest, and the T10 correction (fallbacks must be
   auditable, not silent) fixed its main over-application.

If the remaining four weeks execute the phase2_playbook cleanly, this is a
coherent, defensible top-half entry with a credible shot at the top tier.
The findings below are about where that "if" is weakest.

---

## 2. Findings

Direct answers to the five mandate questions, then the findings that carry them:

- **Q1 (points-weighted priorities):** the *build* allocation matches the points;
  the *measurement* allocation is inverted — IF carries 70.6% of points, ×6 of
  the sweep objective, and the least-trusted instrument (F1). IF weakness is
  neither pure calibration debt nor a structural design gap: it is currently
  **unmeasurable** on the mirror, which is worse than either, because it permits
  confident wrong conclusions in both directions.
- **Q2 (mirror fidelity):** yes, the team is at material risk of over-fitting its
  self-evaluation to the mirror. The disagreement between mirror and real sim is
  not just unmeasured — its *sign* is unknown (the mirror is simultaneously
  harder: sealed anchors, over-approximated AABBs; and easier: no interior
  walls, spawn hints, GT-perfect scene index, no clock). Cheapest bounding
  experiment: F1's disagreement audit.
- **Q3 (perception seam):** not adequately de-risked. It is the self-declared
  largest failure surface, the 2026 differentiator, and the only major subsystem
  with literally zero measured performance and a half-day integration budget (F2).
- **Q4 (2026 vs 2025):** the *scored pipeline* correctly exploits the 2026 moats
  (generator-matched predicates, threading verification, open-vocab posture).
  But the *development apparatus* quietly recreates 2025's world — every number
  the team trusts was produced with a GT-derived scene index, i.e. with exactly
  the `/object_markers` crutch that 2026 removed (F2, F9).
- **Q5 (sequencing):** running the mirror sweep now, while Windows-bound, is
  cheap and fine; **adopting** it is not (F1). The real inversion is smaller and
  fixable: Gate 0 logistics are unchecked with ~2 days left (F7), and detector
  de-risking that needs no Ubuntu (cluster GPU benchmarking) hasn't started (F2).

### F1 — CRITICAL: The sweep objective concentrates weight exactly where the instrument is least trusted

**Evidence.** Sweep objective re-pointed to `rubric×6 / strict×1 / IoU×2`
(`hardening_backlog.md` ledger, "cvsweep re-point"); sweep UNBLOCKED on the T11
gradient (ledger, "CV sweep" row) and launched per
`docs/tasks/T11-if-leg-threading/orchestration.md`. Meanwhile: the IF rubric term
moved 0.061→0.100 (postT11 report topline); the executor's own instrumentation
puts the mirror's *perfect-planner ceiling* at ≈0.18 (13–15/71 leg goals
reachable on our costmap, `task.md` "Achievable-ceiling measurement"); the
GT-reference ceiling claim (30/72) is **unverifiable** per `verification.md` C6;
and the battery's own header concedes the mirror has no interior walls and a
spawn hint (`gt_battery_report.md` wall-realism note).

**The failure it predicts.** The ×6-weighted term has a dynamic range of roughly
0.10–0.18 on this mirror, and most of what separates one sweep configuration from
another inside that band is *which anchors happen to be sealed by
over-approximated AABBs* — mirror artifact, not pipeline quality. The sweep will
dutifully emit a `recommended_calibration.json` whose IF-sensitive rows encode
mirror geometry. If adopted, those values ship to a real sim whose terrain,
walls, and gate widths disagree with the mirror in unmeasured ways — and they
will look validated, because the number that blessed them is committed in a
report. This is the classic self-play trap: the team hardened the scorer (H2,
correctly) and then pointed it at a world model nobody has audited against the
real one.

**Recommendation.**
1. Gate adoption per-term, not per-sweep: numerical-strict and OR-IoU rows may
   be adopted from the mirror; every row whose selection leaned on the IF term
   is *provisional* and re-swept on real sim scores after Gate 5. (The
   orchestration file's caveat says "flag" — make it "do not adopt".)
2. Hold the un-hold to its own condition: the adjudication required a
   committed, reproducible `gt_leg_ceiling.json` before calibration adoption
   leans on the ceiling claim. Enforce that.
3. Add a **mirror-vs-sim disagreement audit** to the phase2_playbook as a named
   gate item (fits between Gate 5 and the calibration loop, ~2 hours): for 2–3
   training scenes, compute on the real terrain map the same two booleans the
   mirror ceiling rests on — (a) is each IF leg goal reachable within 0.8 m,
   (b) is each corridor gate threadable — and report mirror-vs-real agreement.
   That single table either licenses the mirror as a calibration instrument or
   retires it; today its license is assumed.

### F2 — CRITICAL: The 2026 differentiator has zero measured performance and a half-day budget

**Evidence.** `src/core/perception/detector.py:182` — the real detector call
`raise NotImplementedError` (Phase 2); Dockerfile detector install commented out
(H6, SYS-F1); every battery number ever produced used a GT-derived scene index
(battery circularity notes); tiling constants (`AZIMUTH_SIGN` etc.) unvalidated
until Gate 3.3; phase2_playbook Gate 4 = "real perception — ~half a day";
architecture §8 risk 2 names detector recall "the single largest failure
surface"; T8/`organizer_playbook.md` establish perception as the decisive
2026-vs-2025 delta; `question_analysis.md` §4 counts 114 open-vocab nouns
including fine-grained classes ("hookah", "framed records", "map wall decal").

**The failure it predicts.** Gate 4 is where this project meets its actual
problem for the first time, ~10 days before the MVS. Plausible and unbudgeted
discoveries: GroundingDINO recall collapse on Unity-rendered textures (domain
gap nobody has sampled), seam losses from the 4-tile gnomonic layout, prompt
saturation with question-noun injection over 114 classes, VRAM contention with
the captioner, and per-instance `n_obs`/decay/dedup tunables (calibrated against
clean GT indexes) misbehaving on noisy real detections — H15 already concedes
the tracker/fusion path is "unexonerated". If detector recall lands at, say,
60% on question-critical nouns, the calibrated reasoning stack above it is
optimizing the wrong regime, and there is no time to re-calibrate. ~48 pts of
downstream machinery (H6's own estimate) rests on this seam.

**Recommendation.**
1. **Start detector de-risking now, off-machine.** The SoC cluster is available
   from Windows today (master_plan Phase-1 item, unchecked). Batch
   GroundingDINO over VLA-3D scene renders / the sample real-robot panorama,
   score recall against the 114-noun inventory, iterate prompt batching and
   tile layout offline. This converts Gate 4 from discovery into installation.
2. Elevate Gate 5b (GT-injected vs full-pipeline error split) from "do once" to
   the **pivot decision of the calibration loop**: if perception-caused loss
   dominates (likely), the remaining weeks belong to detector recall and
   attribute extraction, not another coefficient sweep. Pre-commit to that
   branch condition in writing so momentum doesn't keep the sweep loop alive by
   default.
3. Budget honestly: rename Gate 4 from "~half a day" to a 2–3 day window with a
   defined minimum (detector produces instances on one scene, `instances_tracked
   > 0`) and a stretch (measured recall table). The MVS date survives either way;
   the fiction of a half-day does not.

### F3 — MAJOR: Real answer keys sit unparsed while every battery leans on proxies

**Evidence.** `question_analysis.md` §1: each per-scene folder ships a rendered
`questions.pdf` with *answer images*; §6: "The correct integers must be inferred
from scene geometry (**or read from the per-scene questions.pdf answer
images**, which are outside this JSON analysis and were not parsed)."
Meanwhile the battery's numerical headline is "independent agreement 56% **over
9** with strict evidence" (3 questions have *no* independent evidence at all),
and its OR section can identify a GT target for only 8/30 questions
(`match_method: none=22`) — both are proxy chains built to approximate ground
truth that is sitting in the repo as pixels.

**The failure it predicts.** Calibration and adjudication decisions keep being
made on proxy agreement (referential-annotation counts, scene-graph edges) that
the record itself shows disagreeing with the pipeline in unresolvable ways
(chinese_room: pipeline=6, independent=1, scene-graph=6 — which is right? nobody
can say). A config that improves true accuracy but worsens proxy agreement gets
rejected; the reverse gets adopted.

**Recommendation.** Spend the one human hour: transcribe the 15 numerical
integers and the 30 OR target identities from the PDFs into a committed
`answers.json` (provenance-noted, verbatim). Point the battery at it. This is
the highest information-per-effort action available before Ubuntu, it
permanently retires the circularity caveats, and it turns the 56%-over-9 into a
real accuracy over 15. (If the PDFs turn out not to render answers for some
types, that discovery is itself worth the hour.)

### F4 — MAJOR: The OR headline hides a 73% measurement hole

**Evidence.** postT11 report: "Object reference (n=30, **scored=8**): mean 3D
IoU 0.875" — 22/30 rows are `tgt=None`, IoU undefined. The 0.875 is a
survivor-biased mean over the 8 questions where GT-target matching happened to
succeed, and those 8 skew toward the *easy* cases (unique-in-scene, clean
relation matches); several of the unscored 22 are exactly the nested-disambiguator
forms that T10 identified as the OR failure mode (57% of OR questions).

**The failure it predicts.** "OR is fine at 0.875" becomes ambient truth (it is
already the topline in the backlog ledger and every LOG entry), effort routes
away from OR, and the sweep's IoU×2 term overfits 8 data points. At eval, OR is
12/51 points — a silent quarter of the score is being steered by a headline
whose denominator is 27% of its population.

**Recommendation.** F3's transcription closes this (target identity from the
answer images). Until then, every topline that quotes 0.875 must carry the
scored-fraction beside it, and OR should be treated as *unmeasured*, not good.

### F5 — MAJOR: The dark-network floor was silently downgraded from adjudicated commitment to a ledger footnote

**Evidence.** `architecture.md` §1 row 7: API dependence "judged existential by
both rivals"; the adjudicated posture is dual-API → **local quantised VLM baked
into the Docker image** → regex tier. `hardening_backlog.md` H8 ledger row:
"ladder api→api2→regex; **local tier DESCOPED**". LOG session 11 confirms. No
architecture amendment records the reversal or its rationale.

**The failure it predicts.** On eval day, an API outage, token-provisioning
hiccup, or provider-side block (the environment is a container on someone
else's machine) drops the parse ladder straight to regex. Regex is proven for
*question-type classification*, not for extracting ordered multi-leg IF plans
with corridor and avoid clauses — the 70.6%-of-points type degrades most. The
debate called exactly this existential, then the mitigation evaporated in a
status table.

**Recommendation.** Re-adjudicate explicitly, either outcome is defensible:
(a) accept the risk with stated reasoning (organizers explicitly allow APIs and
accept runtime tokens; dual-provider failover; estimated P(outage) low) and
amend architecture §3 accordingly — the standing rule for structural changes; or
(b) reinstate a minimal local tier scoped to *parse only* (a small quantized
text model, not the full VLM), which is the cheapest slice of the original
commitment and covers the highest-value fallback. What is not acceptable is a
v1.0 non-negotiable dying by ledger annotation — that is process debt that
invites the next silent reversal.

### F6 — MAJOR: Clock semantics are internally inconsistent and unverified

**Evidence.** `challenge_brief.md:16`: "must respond within **10 minutes per
scene** (exploration + answering combined; system relaunches per question)".
`organizer_playbook.md` Part B (from the upstream repo): "**10 minutes per
question**", 5 questions per scene. `architecture.md` §5 budgets 600 s **per
question**; H8 hedges gates to 480/540 s per question. These cannot all be true.

**The failure it predicts.** Low probability, catastrophic magnitude: if the
budget is genuinely per-scene-shared (or if the evaluator's clock semantics
differ from the assumption in any 5× direction), every time constant in the
system — soft budgets, forced-assembly at T−90, watchdog at T−30, the
early-finish bonus policy — is wrong across all 15 eval questions
simultaneously. Overtime is explicitly penalized.

**Recommendation.** A ten-minute check against the upstream README/rules text
(and the eval-runner script if visible), then pin the resolved wording in
`challenge_brief.md` with a citation and fix the brief's "per scene" phrasing.
The H8 work already made budgets launch-parameterized, so the fix is one
number — the risk is entirely in never asking the question. Add "measure real
clock semantics" to the Gate 2 checklist alongside the QoS verify.

### F7 — MAJOR: Gate 0 is unexecuted with ~2 days before the Ubuntu machine lands

**Evidence.** `phase2_playbook.md` Gate 0 — every box unchecked (bootable USB,
two funded API providers, Docker Hub account, multi-GB training-scene binaries
"downloading to an external drive", final transfer push). Ubuntu arrives ~16 Jul
(LOG session 11 "Next"; today is 14 Jul). The scene binaries are explicitly
flagged as VPN-throttled multi-GB downloads; the sample-data USER action from
master_plan has been pending since 10 Jul.

**The failure it predicts.** Ubuntu day one — the start of the only window that
actually matters — spends itself on downloads and account creation instead of
Gates 1–2. Every day lost there compresses Gate 4/5, which F2 already shows is
the thinnest part of the plan. The playbook's own estimate (2–3 focused days to
Gate 5) only holds if Gate 0 is done *before* the machine exists.

**Recommendation.** Execute Gate 0 today/tomorrow as the top user-action item;
start the scene-binary download on the cluster or campus network now (it
parallelizes with everything else). This is the cheapest schedule insurance
available.

### F8 — MINOR: Eval-day variance strategy is under-articulated for a 15-question sample

**Evidence.** Eval = 3 scenes × 5 questions = 51 pts (backlog points frame); a
single IF question is 6 pts = 11.8% of the total. Training-set fit is over 75
questions; eval is 15, held-out, with acknowledged distribution-shift risk
(architecture §8 risk 1/3).

**The failure it predicts.** Decisions near cut lines optimized for *mean*
training score can trade away tail robustness — but at n=15, one catastrophic
per-question failure mode (unparseable phrasing, nonexistent-object reference,
unreachable anchor) outweighs several small mean improvements. The
watchdog/partial-credit design already leans the right way; the *ledger* that
ranks remaining work does not explicitly price variance.

**Recommendation.** Add a worst-case column to the expected-points ledger
(what does this item do to the floor, not the mean?) and a per-type
"catastrophic mode drill" to Gate 5 (one deliberately hostile question per
type: nonexistent object, ambiguous superlative, blocked corridor). The
organizers' own IRef-VLA work signals they think about imperfect references —
the existence-check branch (organizer_playbook takeaway 4) deserves a place on
the never-cut list if the drill shows it firing.

### F9 — MINOR: Exploration and time economics have never been measured, anywhere

**Evidence.** The mirror battery spawns with a GT-matched hint (default), has no
interior walls, and no clock; no report in `reports/` carries a wall-time
number; the early-finish bonus policy (§1 row 8) and the 0–60 s orientation
sweep have never been exercised against a budget. The one third-party 2025
datapoint (organizer_playbook Part B) used a hard 500 s exploration cutoff —
i.e., exploration cost dominated their design.

**The failure it predicts.** The interleaved explore-execute strategy — row 10
of the adjudication, "uncontested" — meets a clock for the first time at Gate 5.
If full-scene coverage costs 300+ s, the per-type soft budgets (210/240/270 s)
are fiction and the early-answer policy never fires, or fires wrongly.

**Recommendation.** Pre-Ubuntu: run the battery's `--no-spawn-hint` variant once
to bound exploration sensitivity in the mirror (the knob exists and is unused in
every committed report). At Gate 5: the playbook already demands the
explore-vs-answer wall-time split — treat that number as the calibration
currency it claims to be, and re-derive the soft budgets from it rather than
from the architecture's estimates.

### F10 — MINOR: The competitive moat is real but single-legged

**Evidence.** `competitor_forks.md`: 8 dev-kit forks, only 2 active, both
converging on scene-graph + VLM with GroundingDINO — i.e., the field is thin
and nobody visible is ahead on the reasoning layer. Our differentiation stack:
generator-matched predicates (H5), corridor threading verification, checkpoint
protocol, watchdog floors. All of it sits *above* the perception seam that F2
shows is unproven — the moat only cashes if detection works.

**The failure it predicts / recommendation.** None beyond F2; noted so the
adjudicator weighs F2 as the moat's single point of failure, not just a
schedule item. Re-run the fork survey before submission (already noted in the
doc itself); the snapshot is 3 days old and forks move.

---

## 3. Ranked top-5

1. **F2 — Perception is the 2026 differentiator and the only unmeasured
   subsystem.** ~48 pts of machinery rests on a `NotImplementedError` with a
   half-day budget. Start cluster-side detector benchmarking now; make Gate 5b
   the pivot decision for the calibration weeks.
2. **F1 — The ×6-weighted sweep term is mirror-artifact-dominated.** Adopt
   sweep output per-term only; require the committed ceiling artifact; run the
   mirror-vs-sim disagreement audit at Gate 5 before any IF-sensitive adoption.
3. **F3 — The real answer keys are in the repo, unread.** One human hour of PDF
   transcription replaces the entire numerical/OR proxy chain with ground truth
   and fixes F4 for free. Do it before Ubuntu.
4. **F7 — Gate 0 is unexecuted with ~2 days left.** Pure logistics, pure
   schedule insurance, zero intellectual content — which is exactly why it gets
   skipped and exactly why it must not be.
5. **F6 — "10 minutes per scene" vs "per question" is unresolved in the team's
   own documents.** Ten minutes of verification against a 5×-catastrophic
   miscalibration of every time constant in the system.

(F5 — the descoped dark-network tier — misses the top-5 on probability, not on
principle; it is the one finding about process integrity rather than points.)

---

## 4. Prior decisions examined and AGREED with

So the adjudicator knows what was checked, not just what was attacked:

1. **Deterministic backbone + checkpointed LLM (architecture §0, §1 row 1).**
   Organizer-thesis-validated; the T10 confrontation strengthened rather than
   weakened it. No evidence found to reopen it, and I do not.
2. **Corridor/avoid geometry as MVS-non-negotiable (§1 row 3)** and the
   **rejection of capsule-shrink recovery (row 4).** Both correct; the debate
   record shows the right attack won.
3. **Lidar-only metric extents, camera as semantics-only (row 5).** Directly
   backed by Kachana's fisheye-VO failure evidence; correct.
4. **Watchdog floor / "silence is unforgivable" (row 8)** — correct for
   publishing, and the T10 cross-cutting correction (auditable fallbacks, split
   ladder policies per type, H4) was the right repair of its one
   over-application. Endorsed as amended.
5. **T8's evidence-over-dossier ruling** (instance-level GT beats generator
   priors; `above()` lateral-offset over overlap gates) — methodologically
   exactly right, and the kind of call that separates this record from
   cargo-culting prior art.
6. **H5: matching predicate functional forms to the VLA-3D generation spec.**
   A mismatch with the code that generated the questions is a scoring bug, not
   a style choice — agreed verbatim.
7. **H2-first sequencing** (fix measuring instruments before calibration) and
   **holding the CV sweep until the IF term had gradient.** Correct discipline;
   F1 is an extension of this principle to mirror fidelity, not a reversal.
8. **Treating 2025 scores as non-comparable** (GT `/object_markers` gone) and
   the consequent perception-recall re-prioritization (architecture §8 note).
   Correct — F2's complaint is that the conclusion hasn't yet produced action,
   not that it is wrong.
9. **T11's constraint discipline:** fixing the pipeline rather than the scorer,
   reporting harness fixes separately, and the verifier's refusal to accept the
   30/72 ceiling without a reproducible artifact — plus the orchestrator's
   adjudication requiring `gt_leg_ceiling.json` before adoption. All endorsed;
   F1 builds on that verdict rather than relitigating it.
10. **No investment in egocentric/view-dependent relations** (question_analysis
    §5: zero training occurrences) — right call at these stakes; the residual
    risk is correctly logged as accepted (§8 risk 3).
11. **The expected-points ledger as prioritization instrument (row 9)** —
    endorsed, with F8's amendment (add a worst-case/floor column) as a
    refinement, not a replacement.
12. **The task-record + incremental-artifact + fresh-verifier workflow** —
    the T11 verification catching fabricated report rows is the system working
    as designed. Keep it exactly as is through the freeze.

---

*End of critique. Output protocol honored: this file was written incrementally,
section by section; no other file was modified.*



