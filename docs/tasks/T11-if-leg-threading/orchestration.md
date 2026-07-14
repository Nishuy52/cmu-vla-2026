# T11 orchestration state (resume file)

Purpose: if the orchestrating session hits a usage/context limit, a fresh
session resumes from THIS file (per the session protocol: LOG.md points
here). Update the checklist + "resume point" as each stage completes;
delete or archive this file when T11 wraps.

User instructions in force (given 2026-07-14, user asleep — run
autonomously, do not block on questions):

1. Fix runs via an executor agent on branch `fix/if-leg-threading` (cut
   from main `b0c2518`). Implementation stays on standard tiers — never
   frontier.
2. After the fix concludes + verifies: run the CV sweep (user explicitly
   authorized despite the shallow IF gradient — see caveat below).
3. Spawn Fable(frontier)-tier agents to orchestrate a CRITIQUE of the
   overall approach and architecture (allowed on frontier: review/
   adjudication). Distinct from T10's facet red-team: target the strategy
   itself — calibration methodology, eval trust chain, architecture v1.0
   bets — using post-hardening numbers as evidence.
4. Session close per protocol: LOG one-liners, task record, backlog
   ledger row, commit, push to origin, open PR.

## Pipeline checklist

- [x] Branch cut, task record + INDEX entry committed-to-tree
- [x] Executor dispatched (diagnosis -> fix -> tests -> battery)
- [x] Diagnosis complete (see task.md Notes — 3 signatures, common
      upstream cause: mirror stamps solid AABBs)
- [x] Fixes + regression tests in working tree; battery rerun done:
      `reports/gt_battery_postT11_2026-07-14/` — IF rubric 0.061->0.100,
      ordered-leg 0.094->0.122, violations 9->8; numerical 56% and
      OR 0.875 hold (no regression)
- [x] Executor final report received; commits `e7ce1bb` + `1a5549c` on
      branch; full write-up in `executor_report.md`. Full gate launched
      once, detached, result outstanding (executor wakes + relays when
      it lands; targeted subsets + fast tier were green: 1035 passed)
- [ ] Fresh-context VERIFIER pass — probe hardest: (a) "remaining IF
      gap is mirror-fidelity (harness) + scorer centroid property, not
      pipeline" incl. the load-bearing "GT reference trajectories reach
      only 30/72 leg goals within 0.8 m" claim; (b) the harness edits
      (overhang stamping, stall guard) don't flatter scores; that pair
      justifies un-holding the sweep
- [ ] CV sweep launched detached (`docs/cvsweep_rerun_brief.md`;
      objective already re-pointed: rubric x6 / strict x1 / IoU x2,
      disk-resume cache). Output: `reports/cvsweep_<date>/`
- [ ] Fable critique agents spawned + adjudicated into a written record
      (put outputs under `docs/tasks/T11-if-leg-threading/critique/` or
      `docs/redteam/` follow-on — adjudicator's choice, record where)
- [ ] LOG one-liner(s), ledger row in `docs/redteam/hardening_backlog.md`
      (threading item + sweep HELD status), commit, push, PR to main

## Resume point (update on every stage transition)

2026-07-14 ~22:15 SGT: executor DONE (report in `executor_report.md`;
commits `e7ce1bb`+`1a5549c`). IF rubric 0.061->0.100 confirmed-by-agent,
toplines hold. Full gate still running detached on the machine (single
final run; fast tier + targeted subsets were green). NEXT STAGE:
verifier dispatched — writes `verification.md` per-claim. After
CONFIRMED: launch CV sweep detached + Fable critique agents (see
checklist). If resuming cold: check `verification.md` for verdicts;
missing verdicts -> re-dispatch verifier for those claims only.

## Agent restart-proofing (applies to every agent in this pipeline)

Agent final messages are ephemeral (lost if the agent or the
orchestrating session dies — it already happened once this task). Rule:
**every agent writes its deliverable to a repo file incrementally and
commits it; the returned message is only a courtesy summary.** A dead
agent is re-dispatched with the same brief + a pointer to its partial
on-disk output; it continues, never restarts from zero.

Per-stage artifact paths + dead-agent recovery:

| Stage | On-disk deliverable | If found dead |
|---|---|---|
| Executor (fix) | `task.md` dated notes + `executor_report.md` (requested mid-run) + milestone commits on branch | Changes are the modified files in the tree; re-dispatch executor pointing at task.md notes + `git status` to finish remaining acceptance criteria |
| Verifier | `verification.md` in this folder — one section PER CLAIM, written as each probe completes (not one dump at the end) | Re-dispatch verifier for the claims missing a CONFIRMED/REFUTED verdict in the file |
| CV sweep | `reports/cvsweep_<date>/` — the re-pointed sweep has a DISK-RESUME CACHE; rerunning the same command resumes, it does not restart | Relaunch same command (see `docs/cvsweep_rerun_brief.md`); cache makes it idempotent |
| Fable critics | one draft file per critic under `critique/` in this folder, written section-by-section as they go | Re-dispatch only the critic(s) whose draft is missing/truncated |
| Adjudication | `critique/adjudication.md` | Re-run adjudication over the on-disk drafts (inputs are all files, not messages) |

## Standing cautions

- Stale untracked leftovers NOT to commit: `reports/cvsweep_run.*`,
  `reports/cvsweep_smoke/`, `reports/T5_colored_cloud/`.
- Never git stash/checkout/reset/clean in this shared tree.
- No AI/tooling attribution in commits or docs.
- Sweep caveat: IF term gradient is real but shallow (0.100) and
  x6-weighted — when adopting `recommended_calibration.json`, flag which
  rows lean on the IF term; do not over-adopt from a weak signal.
- Battery/sweep must run from THIS main tree (worktrees lack git-ignored
  `data/`).
