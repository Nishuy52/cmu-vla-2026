# No-stall orchestration

The campaign makes progress in every turn. These rules keep it moving.
They are user-taught, repeated across sessions, and non-negotiable.

## Never end a turn purely waiting

Before you end any turn, name what you wait for. Every wait needs three
parts: a primary trigger (a Monitor event or a task notification), a
backstop (the hourly cron), and a deadline. A wait without a deadline is
a stall. When a deadline passes, probe immediately and act on what you
find.

While anything runs, work the backlog in the same turn:

- Analyse an earlier group that has landed.
- Harvest a stalled agent's worktree.
- Build the next matrices or the next issue update.
- Update memory so a restart loses nothing.

"No work left" requires the mechanical sweep first: `gh issue list
--state open` cross-checked against active agents' owned surfaces. Every
unowned open issue on a free surface is dispatchable NOW.

## Agents: expect stalls, take over fast

- An agent that reports "waiting", "standing by", or "monitoring" is a
  STOPPED agent. Its monitor rarely survives; its wait is now yours.
- The worktree is the truth. An agent's "committed" claim with a dirty
  worktree and no commit is a false report; say so in its resume message.
- Takeover procedure: read the diff, run the targeted tests yourself,
  write the commit (ASD-STE100), merge in the next batch. This costs
  minutes; a third round-trip costs an hour.
- Resume an agent at most twice (SendMessage with the failure evidence
  and a precise contract). After two failed resumes, finish the work
  yourself or re-dispatch fresh with a tighter brief.
- Write the anti-stall demand into every brief: run commands in the
  FOREGROUND under `timeout`; never end a turn while a background command
  runs. Expect the demand to fail anyway; that is why the takeover
  procedure exists.

## Dispatch by dependency, not by eventual need

- Everything independent launches in the same turn, `run_in_background`,
  worktree-isolated, with disjoint file ownership.
- Hot files get packed contracts: all pending issues for one file go to
  ONE agent as separate commits. Sequential agents queueing on a file is
  a bottleneck you built yourself.
- Blocked work queues behind its surface owner and launches the moment
  the surface frees — the hand-off happens in the turn the owner lands.

## Resolve gates; do not wait on them

- A flaky gate is a defect: fix the test (CPU time instead of wall time),
  file it, and keep the pipeline moving with the known-flaky set named.
- Full gates run once per merge batch, main-session. An in-flight
  validation on one surface never holds merges on other surfaces.
- If something is truly unsolvable in the session, say so to the user
  explicitly and ask for an exception NOW. Silent parking is a stall.

## Approval boundaries

Work with an issue number proceeds without asking. Ask first only for:

- rubric or scoring-semantics changes right before a measured A/B,
- core FSM behaviour changes with no issue number,
- destructive or irreversible operations (cancelling running jobs,
  deleting data, force-pushes).

Offering a recommendation with the ask is required; a bare question is a
stall with extra steps.

## Resurrection: survive restarts and migration

Assume the session dies at any moment. Keep these alive outside context:

1. The memory file (`next-session-issue-queue`): campaign state, job ids,
   pinned commit, baseline numbers, comparison rules, open hazards.
2. The cron backstop prompt: embed the ENTIRE analysis plan — job ids,
   log paths, baseline means, issue numbers, and the self-delete
   condition. The cron prompt is the resurrection vector; write it so a
   context-free session can execute it verbatim.
3. The issue tracker: every finding filed, every fix referenced. The
   tracker outlives every session.
4. Watcher logs under the session scratchpad: a new session reads them to
   learn what landed while nobody watched.

After any restart: re-probe ssh (the ControlMaster dies with the host
session; ask the user for one interactive `ssh xlogin true` if probes
hang), confirm watcher processes with `ps`, re-arm Monitors (they never
survive), and re-read the memory file before touching anything.
