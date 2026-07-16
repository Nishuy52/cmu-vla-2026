# Adjudication — approach & architecture critique (2026-07-14)

Inputs: `architecture_strategy.md` (arch-F1..F10) and
`methodology_process.md` (meth-F1..F11), both frontier-tier critics with
disjoint mandates, plus `verification.md` (T11 verifier) and a direct
source check performed during adjudication. Verdicts below are binding
for orchestration; architecture-document amendments follow the standing
rule (explicit, dated, in `architecture.md`).

## Headline

Both critics independently endorsed the core bets (deterministic
backbone, H2 instruments-first sequencing, T8 evidence-over-dossier,
the task-record/verifier workflow) and independently converged on the
same two soft spots from opposite directions:

1. **The CV sweep cannot be trusted as currently built** — arch-F1
   (the ×6 IF term's dynamic range is mostly mirror artifact) and
   meth-F1/F2/F3 (the IF instrument is endogenous to the swept
   thresholds; the gradient is ~3 questions deep; the assembled
   recommendation is never evaluated as a config) are one finding in
   two dialects.
2. **The team's evidence apparatus quietly recreates 2025's world** —
   arch-F2/F9 (perception unmeasured, GT scene indexes everywhere)
   and meth-F5/F9 (instrument shares parse/vocab/resolve with the
   pipeline; calibration tuned under perfect labels).

Neither critic found grounds to reopen the architecture's structure.
The risk is concentrated in what the numbers are allowed to mean.

## Verdicts and dispositions

### Resolved during adjudication

- **arch-F6 (clock semantics) — RESOLVED, benign.** Checked against
  the upstream README "Timing" section directly: the limit is
  unambiguously **10 minutes per question** (exploration + answering,
  timed from system startup; overtime penalty; early-finish tie-break
  bonus). `architecture.md` §5 was already right; `challenge_brief.md`
  said "per scene" and has been corrected in place with a citation.
  No time constant changes.

### Accepted — acted on this session

- **meth-F1 (endogeneity) — ACCEPTED, CRITICAL, verified against
  `cvsweep.py` by the adjudicator before action.** Fix dispatched
  (executor #2): rubric geometry, terminal centroids, alignment frozen
  at default thresholds; swept thresholds reach only the driven
  pipeline; regression test pins instrument invariance across configs.
  **The sweep does not launch until this lands.**
- **meth-F3 + arch-F1 (adoption gate) — ACCEPTED AS BINDING.** Written
  into `orchestration.md`: modal consensus AND assembled-config
  evaluation beating baseline by > one 6-pt-question-equivalent AND no
  per-type regression AND identical instrument footprint AND
  IF-term-driven keys provisional-until-real-sim (not committed as
  defaults). meth-F2's per-key ablation question-count (reject keys
  whose whole effect is <2 questions) is added to the gate.
- **meth-F10 (stale artifacts) — ACCEPTED.** `reports/cvsweep_run.*`,
  `reports/cvsweep_smoke/` (retired objective) deleted at session
  close; standing caution updated from "don't commit" to "delete".

### Accepted — scheduled next (pre-Ubuntu agent work, in order)

1. **meth-F8 (machine-generated numbers) + meth-F7 (provenance).**
   A committed `battery_diff` tool whose output is the only legal
   source of before/after tables in reports; `gt_battery`/`cvsweep`
   stamp commit hash, dirty-tree digest, calibration snapshot, and
   per-question `leg_goals`/per-leg outcomes into results JSON.
   Rationale: today's fabricated-attribution incident was caught by
   verifier diligence; this makes the error class structurally
   impossible instead of detectable. Do before the next battery
   generation so the postT11→next comparison is born provenanced.
2. **meth-F5 cheap external check #1 (leg-count census)** — hand-tally
   the 30 IF questions' clause structure vs `plan.route`/`plan.avoid`
   once, commit as a fixture. Hours, closes the silent-dropped-leg
   class.
3. **arch-F9 / mirror knob** — one `--no-spawn-hint` battery run to
   bound exploration sensitivity (knob exists, never used in a
   committed report).
4. **meth-F11 (3 unaligned scenes)** — one-time manual frame fit;
   either the scenes rejoin the IF battery or the exclusion is
   confirmed as data, not resolution. Closes a 20%-of-IF-rows hole
   that currently biases every IF mean friendly-ward.
5. **meth-F4 + meth-F6 (topline honesty)** — drop pipeline-exact-100%
   from toplines (keep as `determinism` per-row), lead numerical with
   `agreement k/n`; rename OR headline to `instance-match k/8, IoU
   pending real perception`; track OR scoreability (n/30) as a metric.

### Accepted — user actions (cannot be delegated to agents)

- **arch-F3/F4 + meth-F6 (answer keys): the single highest
  information-per-hour item on the board.** Transcribe the numerical
  integers and OR target identities from the per-scene `questions.pdf`
  answer images into a committed `answers.json`. Replaces the entire
  proxy chain (56%-over-9, 8/30 scoreable) with ground truth.
- **arch-F7 (Gate 0): execute today/tomorrow.** USB, two funded API
  providers, Docker Hub, scene-binary downloads started on a fast
  network. Pure schedule insurance for the 16 Jul machine.
- **arch-F2 partial (cluster access):** SoC login works from Windows
  now; unblocks detector benchmarking (below).

### Accepted — strategy amendments (folded into plan docs next session)

- **arch-F2 (perception de-risking) — ACCEPTED, top strategic
  finding.** Start GroundingDINO recall benchmarking on the SoC
  cluster before Ubuntu; re-budget Gate 4 from half a day to a 2–3 day
  window with defined minimum/stretch outcomes; elevate Gate 5b
  (perception-vs-reasoning error split) to the explicit pivot decision
  of the calibration weeks, pre-committed in writing.
- **arch-F1.3 + meth-§3 (mirror-vs-real audit) — ACCEPTED.** Named
  Gate-5 items: (a) reachability/threadability booleans on real
  terrain for 2–3 scenes vs mirror; (b) same-rubric Spearman between
  mirror and real per-question scores; (c) baseline-vs-recommended
  A/B in the real sim before any adoption commits. First sim days
  measure how wrong the mirror is, not make it prettier.
- **arch-F8 (variance) — ACCEPTED as refinement.** Worst-case/floor
  column in the expected-points ledger; one hostile question per type
  as a Gate-5 drill.
- **arch-F10 (fork survey refresh) — ACCEPTED**, pre-submission
  checklist item.

### Deferred to the user (recommendation attached)

- **arch-F5 (descoped local-VLM tier).** The critic is right that a
  v1.0 "existential" mitigation must not die by ledger footnote.
  Recommendation: option (a) — accept the API-outage risk explicitly
  (organizers permit APIs; dual-provider failover exists; container
  runs on the organizers' network where a local 7B's latency/VRAM
  cost is real) and amend `architecture.md` §3 with that reasoning,
  PLUS harden the regex tier for multi-leg IF plans as the cheap
  partial mitigation (it currently covers classification well and
  plans poorly — and IF is where the ladder bottoming out hurts).
  Needs your sign-off since it reverses an adjudicated non-negotiable.

### Rejected / not pursued

- meth-F2's implied option of *not* running the sweep — rejected; the
  stability table is informative and the user directed the run. The
  gate (above) carries the safety instead.
- Any reopening of the deterministic-backbone / checkpointed-LLM
  split, corridor non-negotiables, lidar-metric camera-semantic
  split, or watchdog floors — both critics endorsed; no new evidence.

## Standing changes to how this repo works (from meth-F8, effective now)

Any number that gates a decision must reproduce from a committed
artifact via a committed script; agent-authored prose may interpret
numbers below a machine-generated table, never introduce them above
it. Verifier briefs gain this as a mechanical checklist line.
