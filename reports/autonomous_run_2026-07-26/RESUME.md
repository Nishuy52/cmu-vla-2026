# Cold-pickup — live-run improvement campaign

**Refresh this file at every milestone.** Everything here is committed; a dead session
loses nothing but the harvest step. Supersedes `RESUME-0430.md`.

## Operating loop (user directive, 28 Jul)

**Do NOT implement fixes in the main session.** The loop is:
1. **WAIT** for every verifier — a claim that looks dominant often gets refuted
   (12 of 26 verdicts were refutations, including the headline "instance explosion"
   narrative and an AABB-clamp hypothesis that measurement killed).
2. **SYNTHESISE** in the main session — rank by expected-points-per-effort.
3. **DELEGATE** one agent per bounded confirmed defect (goal, owned files,
   done-criteria, tests).
4. **RE-VERIFY** with a fresh-context verifier before claiming any fix works.

## Current state (28 Jul ~05:00)

### Live scoreboard — 45 questions, 15 scenes, all 3 types
| type | live | offline |
|---|---|---|
| instruction_following | **0.367** | 0.800 |
| numerical | **0.133** | 1.000 |
| object_reference | **0.005** | 1.000 |
| all 45 | **0.167** | — |

The harness is solid (45/45 SUCCESS, all `tier=api`). **The bottleneck is perception.**

### In flight
- **Cluster:** `701116` RUNNING (inst x8) -> `701117` (inst x7) -> `701118` (nume x15)
  -> `701119` (obje x15), chained. First run that captures the #102/#98 dumps.
  Harvest: `tools/cluster/live_run/harvest_verify.sh <job>` then recover bags
  (`ros2 bag reindex -s mcap` + `ros2 bag convert`) — these predate the #114 in-job fix.
- **Diagnosis workflow:** runId `wf_d5c85523-ce5`. Resume with
  `Workflow({scriptPath: ".../workflows/scripts/live-failure-diagnosis-wf_d5c85523-ce5.js", resumeFromRunId: "wf_d5c85523-ce5"})`
  — completed agents replay from cache. Synthesis was still pending at last check
  (30/35 agents done, 26 verdicts: 14 confirmed / 12 refuted).

### Next actions, in order
1. Collect the workflow synthesis (resume it if it died).
2. File an issue per confirmed finding not yet filed (rule: every defect AND every
   proposed fix gets its own verified issue; see `reports/cluster_verify/699009/FINDINGS-INDEX.md`).
3. Synthesise the ranked plan; then **delegate** fixes, one agent per defect.
4. Re-verify each fix against the replay harness before believing it.
5. Harvest 701116-701119 as they land.

## Verified facts — do not re-derive

- **Perception is 100% of the numerical gap (#120).** Controlled swap: same parser +
  same `counting()`, GT index -> **15/15**; live index -> reproduces our wrong answers
  **13/13**. Parser/counting/scoring/navigation exonerated.
- **Coverage is the under-appreciated half**: median tracked-vs-GT instance count is
  **0.45x** — we track fewer than half the instances GT has (home_building_1 85/432).
  There is NO scene-wide instance explosion.
- **Within detected classes we over-produce**: chair 23 vs 6, pillow 30 vs 4,
  picture 37 vs 9, cup 14 vs 2 — spatially SPREAD (only 1-22% of pairs within 0.5 m),
  boxes inflated **1.1-4.5x** (#123).
- **Two fixes tested on recorded snapshots and REFUTED (#123)** — do not retry:
  clamping tracked extents (correct 1->1 of 15); evidence gating (mean|err| 4.93->2.13
  but correct only 1->2 of 15, far too weak for the generalization protocol).
- **The replay harness is the right acceptance test**: replay `toolbox.counting()` over
  `reports/cluster_verify/699819/debug/<slot>/instance_index.jsonl` (`tag=answer_time`)
  to A/B a perception change offline against real live data before any cluster run.
- **obje answers are wrong, not just unscoreable (#118)** — 6 of 7 now-scoreable score
  exactly 0.000 IoU. The earlier all-`n/a` masked a real failure.
- **Colour attributes never populated live (#121)** — `color_bins=()`, `caption=""`;
  colour-qualified questions return a confident 0.

## Operational gotchas

- **Never `scancel` to apply a fix** — loses the queue place. Only runtime-loaded files
  update on a queued job (`~/vla/src`, `~/live_monitor.py`, `~/question_pub.py`); the
  sbatch body is frozen at submit. Use `scontrol hold`/`release` to sync `src` safely
  between jobs — a running job re-imports `src` per question, so syncing mid-run splits
  it across code versions.
- **Fable-verify batch scripts before submitting** — caught two silent run-invalidators
  (unexpanded `~` so the scene overlay never happened; one shared matrix filename so all
  chained jobs would read the last matrix).
- **Bags never finalize on SIGINT (#114)** — fixed in-job for future submissions only.
- **Schema trap**: `by_class`/`total_instances` are TOP-LEVEL in `instance_index.jsonl`,
  not under `live_instances`.
- **`score_live_run` needs an explicit `--out`** (#96) or it overwrites the committed
  20-Jul baseline; its merge key drops rows for repeated scene+qtype (#97).
- **Keep agent concurrency modest** — a 35-agent fan-out destroyed the usage limit.
- Dump env vars must be DERIVED from `VLA_EXPLORE_DEBUG_DIR`, never standalone (#119).

## Open issues: 24+ — key ones

#120 perception is the numerical gap · #123 inflation + both negative results ·
#118 obje answers wrong · #121 colour attributes · #91 missing-class disambiguators ·
#94 under-segmentation · #119 dump plumbing · #122 `_eval_clause` anchor selection ·
#77 corridor threading · #96/#97 score_live_run data loss

## Run-reliability verifications (28 Jul, both done)

- **#126 wedging — CONFIRMED, worse than claimed.** All **15/15** IF runs stop short of the
  last commanded waypoint by **0.16-2.43 m** (median ~1.0), with `tail_motion = 0.00 m`
  (fully stationary for the final quarter). 4 of 15 exceed the ~1.746 m terminal tolerance,
  so those legs fail on this alone. Likely the stock planner's <0.5 m clearance obstacle-stop.
- **#127 loft — claim REFUTED, but a real harness defect found.** `ENDPOINT: MISSING` yet the
  slot was recorded SUCCESS. However loft scores **0.000 OFFLINE too** (legs 0/1) and the
  adapter did not crash (clean FSM lifecycle, 19 publications) — so its zero is genuine and
  must NOT be excluded from the mean as a harness artifact.

## Fixes delegated (28 Jul ~05:20) — awaiting agent reports

Three worktree-isolated executors, each with the offline battery as a no-regression gate
(baseline IF 0.7444 / numerical 15/15 / obje 12/30 at IoU 1.000):
- **#126** terminal-waypoint standoff (confirm mechanism first, then fix)
- **#125** floor degenerate instance extents to the class prior — acceptance test is the
  offline replay over `reports/cluster_verify/699819/debug/<slot>/instance_index.jsonl`
- **#122** `_eval_clause` pick a passing anchor, not the highest-scoring one

Next: integrate each after a fresh-context verifier pass; do NOT merge on the executor's
own report.

## Fix campaign status (28 Jul, end of session)

MERGED + VERIFIED on main:
- **#122** `_eval_clause` picks a passing anchor, not the highest-scoring (`2e85b1f`).
  Moves no number today by design — banked for when the index improves.
- **#125** degenerate instance extents floored to the class prior (`17e6f9c`).
  Real effect **1/15 -> 2/15** (the implementer claimed 0->1; verification corrected it UP).
- **#126** terminal GOTO waypoint standoff (`afac7b9`) — **MERGED AS PARTIAL, NOT A FIX.**
  Raises clearance 0.1 -> only 0.2-0.3 m, short of the 0.45 m it targets and the ~0.5 m
  the planner needs. Merged as a strict no-regression improvement only.
- **replay A/B harness** `tools/replay_live_numerical.py` (`61208c6`).

### AUTHORITATIVE replay baselines (supersede all earlier hand-rolled figures)
```
raw recorded boxes    exact-match 1/15 (arabic_room)   mean|err| 4.933
degenerate_floor      exact-match 2/15 (+office_2)     mean|err| 4.867
```
The earlier 4.867/4.800 figures were WRONG — off by one on hotel_room_1. The harness is
faithful to the recorded data and the head's own logic (verified by calling `toolbox.on()`
directly: exactly 23 pillows pass, matching the harness). The 1-count gap vs the live run
is live-vs-snapshot skew, inherent to snapshot replay.

A/B a perception change with:
```
python -m tools.replay_live_numerical --variant-a raw --variant-b pkg.mod:my_transform
```

### PENDING — not merged
- **lateral cluster segmentation** in `fusion.py`, committed as `0b0ba0d` on branch
  `feat/perception-replay-harness`. UNVERIFIED. That branch also carries the harness
  (already cherry-picked to main) and a third tool from a concurrent session — do NOT
  merge the branch wholesale.
- **#132 clearance-field accounting gap** — requesting 0.45 m yields 0.2-0.3 m. Affects
  the long-shipped VIA_NEAR mechanism too, so every clearance-based goal placement has
  likely been landing ~0.2 m short. This, not #126, is the real wedge blocker.

### Corrections worth remembering
- The 4 wedging runs' terminal anchors are **compact** (arabic jar 0.23 m, trash can
  0.65 m, potted plant 0.31 m, mirror 0.34 m) — NOT large. The figures 2.43/2.34/2.21 m
  are WEDGE DISTANCES from #126, mis-transcribed into LOG.md as footprint diagonals.
- Three consecutive executors under-reported failing tests (one omitted 5). Always count
  failures independently; environment-caused failures must be stated, not omitted.
- Two agents ended turns waiting on background jobs. Briefs must forbid ending in a wait.
