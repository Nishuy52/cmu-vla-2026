# Task: Merge the research stream into the team repo (branch + PR)

## Intent
Fold the parallel research stream (2025 results dossiers,
VLA-3D question spec, competitor forks, verified I/O contract)
into the team repo (github.com/Nishuy52/cmu-vla-2026) so agents
working there see the sources. Append-first: existing team
content takes priority, research content is added with
provenance; nothing is lost. Work on a new branch, open a PR
for review.

## Context
- Team repo: full implementation + 20+ docs, strict CLAUDE.md
  conventions (LOG.md entries, no AI attribution, snake_case
  docs, push at milestones).
- Research-side unique value: full 2025 leaderboard + 15
  dossiers (team docs said 1st/2nd "not found"), VLA-3D
  generation spec,
  competitor-fork snapshot, io-contract cross-check.
- Excluded per instruction: approach-roadmap, duplicate docs
  (problem brief, challenge rules). Tasks were initially
  excluded, then reversed mid-run (see Notes): imported
  verbatim under docs/tasks/ so all agents see them; the
  friction log rode along as part of T2.
- Plan approved in chat 2026-07-11 ("branch and pr is good pls
  do it urself. go all the way").

## Acceptance Criteria
- [ ] Branch `docs/import-2025-research` on the team repo with all
      imports + surgical appends, pushed.
- [ ] Every imported file carries a provenance header; internal
      links remapped to the repo's snake_case layout; no broken links
      to files that only exist in our repo.
- [ ] Existing team docs only appended to (dated update
      sections), never
      rewritten.
- [ ] CLAUDE.md table, README, LOG.md updated per the repo session
      protocol.
- [ ] PR opened; no AI attribution anywhere; no task codes.

## Artifacts
- Questions: questions.md (none yet)

## Stage Gates
| Gate | State | Evidence / notes |
| --- | --- | --- |
| Task | APPROVED | plan proposed + explicit go-ahead in chat |
| Spec | APPROVED | plan message in chat = spec; approved |
| Plan | APPROVED | same message; branch+PR explicitly OK'd |

## Todo
- [x] Clone the team repo locally, create branch
- [x] Map + rewrite internal links of dossiers (kebab->snake,
      drop refs to our-repo-only files)
- [x] Import 15 dossiers + index under docs/prior_art/
- [x] Import competitor_forks.md, io_contract_crosscheck.md
- [x] Append updates: prior_art.md, organizer_playbook.md,
      vla3d_notes.md, CLAUDE.md, README.md, LOG.md
- [x] Commit, push branch, open PR (#1, commit 84930f9)
- [x] Import tasks/ verbatim to docs/tasks/ (scope change)
- [x] CLAUDE.md task-workflow adapter (committed - initial
      local-only idea reversed; CLAUDE.md stays tracked)

## Notes
- 2026-07-11: created after plan approval; execution starts
  immediately.
- 2026-07-11: PR #1 opened
  (github.com/Nishuy52/cmu-vla-2026/pull/1).
- 2026-07-11: scope change (mid-run instruction): tasks now
  imported verbatim under docs/tasks/ so all agents see them.
  Task workflow in the team repo is adapted via a CLAUDE.md
  section (committed; first planned local-only via
  skip-worktree, then reversed - CLAUDE.md stays tracked):
  tasks live in docs/tasks/ (committed), LOG.md gets one-liner
  entries ONLY at task milestones (started/finished/wrapped
  up); detail stays in task files.
- 2026-07-11: sweep findings: neutralized tooling mentions
  (WebFetch -> headless fetch) and task-code/workflow refs in
  the polished doc imports; docs/tasks/ copies stay verbatim
  by explicit instruction.
