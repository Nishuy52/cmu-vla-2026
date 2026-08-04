---
name: live-campaign
description: Run a live cluster verification campaign for cmu-vla-2026 end to end — submit chained Slurm sweep jobs, watch them, harvest and score each group, compare against baselines with a matched scorer, diagnose failures from artifacts, file issues, dispatch fixes, verify, merge, push, and resubmit. Use this skill whenever the user asks to submit a sweep, benchmark on the cluster, harvest or analyse sweep results, measure a fix live, run an A/B against a baseline, or resume a running campaign. Also use it when a task notification reports that a sweep job finished.
---

# Live verification campaign

This skill runs the measure-diagnose-fix loop against the NUS SoC cluster.
Each rule below exists because its absence caused a measured failure once.
The reference files carry the detail; read them at the step that needs them.

Read `references/no-stall-orchestration.md` FIRST, in every session that
runs a campaign. It holds the operating doctrine: never end a turn purely
waiting, take over stalled agents fast, dispatch by dependency, resolve
gates instead of waiting on them, and keep state outside context so a
restart or a session migration loses nothing. The campaign only works at
speed because of those rules.

The campaign loop has five phases. Run them in order. Do not skip a phase.

1. Pre-flight and submit.
2. Watch.
3. Harvest and analyse each group when it lands.
4. Diagnose, file, and fix.
5. Verify, merge, push, resubmit.

## Phase 1 — pre-flight and submit

Read `references/cluster-mechanics.md` before your first submission in a
session. It holds the exact commands and the contamination rules.

Pre-flight checklist. Confirm every item before `sbatch`:

- `squeue -u cyuhsin` shows no verify job of ours, or you chain behind the
  last one with `afterany`. Two verify jobs must never run at once: they
  drive the same udocker container and scene overlay.
- The tree you measure is GATED (full `pytest -m ""` green at merge level)
  and PUSHED. Record the commit hash. It names the measurement.
- Pin the tree: `git worktree add --detach <pin-dir> <commit>`. Submit and
  sync only from the pin. Main moves while campaigns run; a moving tree
  once produced three jobs that measured different code.
- Sync `~/vla/src` from the pin, then prove it: a content-checksum dry run
  (`rsync -rcn`) must report zero differing files. The sbatch reads
  `~/vla/src` LIVE at run time, per question.
- Matrices: build per group, upload each under a UNIQUE name, and pass it
  with `--export=ALL,MATRIX=...`. For an A/B, diff the matrix against the
  baseline's matrix file byte for byte before you submit.
- If ANY job is running, submit matrix-only. Do not run
  `push_and_verify_batch.sh` and do not scp helper scripts: a running job
  re-reads `~/question_pub.py` and `~/live_monitor.py` on every question,
  and scp truncates in place.

Submit each group with `sbatch --dependency=afterany:<prev>`. After each
submission, read `squeue` and confirm the new job is PD with the correct
predecessor in its DEPENDENCY column before you submit the next. A
mis-parsed job id makes `afterany` satisfiable at once and two jobs then
run concurrently.

## Phase 2 — watch

Arm three layers. Each layer covers a failure of the one above it.

1. A harvest watcher: copy `scripts/harvest_watch.sh`, set the job list and
   log path, run it with `run_in_background`. It polls sacct, records the
   per-job assertions, and harvests each job when it ends.
2. A persistent Monitor that tails the watcher log and emits one event per
   `harvested rc=` line, plus lines that match `FAILED|CANCELLED|DEGRADED`
   and `src_drift_files=[1-9]`.
3. An hourly cron backstop (CronCreate, off-minute). Embed the FULL
   analysis plan in the cron prompt: job ids, log paths, baseline numbers,
   and the issue numbers to update. The cron prompt is what survives a
   process restart and context loss. Make it delete itself when every
   group is analysed.

Never end a turn waiting for a job. While jobs run, work the backlog:
diagnose earlier groups, harvest stalled agents, update issues.

## Phase 3 — harvest and analyse each group

Trigger: a Monitor event or the cron. For each landed group, in order:

1. Read the watcher log line first. Require
   `VERIFY-BATCH-DONE: <n> question(s) attempted` with the full count,
   SLOT SUCCESS = n, DEGRADED count known, and `src_drift_files=0`.
   A truncated or drifted group is not comparable; say so before any
   number leaves your hands.
2. Score with the CURRENT scorer, `PYTHONHASHSEED=0`, serialized, one
   `timeout` per run. Scoring and bag reading are main-session work; never
   give them to a subagent (contention, shared-output collisions, and
   bag-read hangs).
3. Compare question-matched, never mean-vs-mean. Re-score the baseline
   bags with the SAME scorer version first: the rubric changes over time,
   and a stale baseline mean once mis-stated an A/B by 0.02.
4. Exclude capture-flagged rows from means on BOTH sides, and print the
   flag column. A starved run once sat inside a reported mean unnoticed.
5. Judge deltas against noise: one question is 1/n of a group mean, run
   variance flips 3-5 questions per run at n=1, and scorer churn alone
   measured ~0.02. Leg counts aggregate 2x more events than question
   means; prefer them for direction. Read
   `references/analysis-rules.md` for the full comparison discipline.
6. Commit the group's scores and log entry (ASD-STE100), push.

## Phase 4 — diagnose, file, fix

Diagnose from artifacts, not memory. Per slot you have:
`instance_index.jsonl` (last record with `instances` = the index at answer
time), `resolved_plan.jsonl`, `prompt_diagnostics.jsonl`,
`raw_detections.jsonl`, the trajectory bag, and the FSM event line in
`verify_batch.out`. Cross-run tables beat single-run reads: a failure that
repeats at the same leg or the same coordinates across generations is a
mechanism; one that flips is run variance.

File one GitHub issue per defect, in ASD-STE100, with measured numbers and
artifact paths. An issue with a reproduction beats a report paragraph.

Dispatch fixes to executor agents with worktree isolation. Pack all pending
issues for one hot file into ONE agent contract. Briefs must demand:
targeted test suites only (the full gate runs once per merge batch, in the
main session), foreground commands under `timeout`, no ending a turn while
a background command runs, raw-JSONL validation only (no bag reads, no
live replay), an ASD-STE100 commit message, and no push or merge. Expect
agents to stall on background waits anyway: when a completion notice says
"waiting", read the worktree, run the targeted tests yourself, commit, and
move on.

## Phase 5 — verify, merge, push, resubmit

- Verification before merge is not optional, but scope it: a no-regression
  proof covers the slots the changed surface can affect, not the whole
  sweep. Prefer proofs by construction (the change only touches inputs in
  state X) plus a counted before/after table on real captured data.
- Batch independent branches; run ONE full gate on the integrated tree
  (`PYTHONHASHSEED=0`); push. Pushing auto-closes `Fixes #N` issues —
  check for wrong auto-closes (a commit that cites an issue without fixing
  it) and reopen with a comment.
- Resubmit the affected groups on the new pinned tree (phase 1 again). Keep
  the polluted group's results: identical questions on the pre-fix tree are
  the cheapest possible live A/B anchor.

## Resuming a campaign in a fresh session

A campaign survives session death and migration by design. To resume cold:

1. Read the memory file (`next-session-issue-queue`) for the campaign
   state: job ids, the pinned commit, baseline numbers, comparison rules.
2. Read the open issue list. It is the actionable queue and the record
   that outlives every session.
3. Probe ssh with a short timeout. If it hangs, the ControlMaster is dead:
   the cluster jobs are safe server-side; ask the user for one
   interactive `ssh xlogin true`, then continue.
4. Check watcher processes (`ps aux | grep harvest_`), read their logs for
   groups that landed unwatched, and analyse those first.
5. Re-arm the Monitors (they never survive a restart) and confirm the cron
   backstop exists; recreate it with the full embedded plan if not.
6. Keep going. Do not wait for the user to repeat the mandate: the open
   issues, the queued jobs, and the memory file ARE the mandate.

## Failure playbook

- Job runs but ssh hangs: the ControlMaster died. Jobs are safe server-side;
  ask the user to run `ssh xlogin true` once, interactively. Do not queue
  password prompts in the background.
- Mid-run rsync into `~/vla/src` detected (file mtimes newer than submit
  time): the group is contaminated even if the content now matches the pin.
  Discard it, state why, resubmit. Content equality now does not prove
  stability during the run.
- A group returns fewer questions than submitted: compare against the
  baseline on the matched subset only, and say so.
- An agent reports done but its worktree shows no commit: the worktree is
  the truth. Harvest it yourself.
