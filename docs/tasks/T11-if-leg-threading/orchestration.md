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
- [ ] Executor final report received; milestone commits made on branch
- [ ] Fresh-context VERIFIER pass — probe hardest the claim "remaining
      IF gap is mirror-fidelity (harness), not pipeline"; that claim
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

2026-07-14 ~20:45 SGT: executor (resumed once after an API stall) is in
final verification — full gate `pytest -m "" -n auto` + `tests/runner`
pass running. All code changes on disk uncommitted. If resuming cold:
check `git log origin/main..fix/if-leg-threading` and `git status`; if
no commits exist, the executor died pre-commit — its changes are the
modified files in the tree (instruction.py, synthetic_scene.py,
vocab.py, gt_battery.py + tests); re-dispatch an executor to finish
per `task.md` acceptance criteria before verifying.

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
