# Task: Architecture folder - explainer-style decomposition

## Intent

Produce a comprehensive root `architecture/` folder of Markdown
files (explainer format) describing the current implemented
system, replacing the earlier single-file attempt that was
dropped by rebase. Deepest coverage goes to the core runtime
loop: question intake, perception, answer submission, and time
budgeting.

## Context

- The single-file `ARCHITECTURE.md` (commit `a864e0b`, task
  record T5) was deliberately dropped via rebase; its content
  was source-verified the same day and is mined as raw material
  (extracted to the session scratchpad from the git object
  store).
- `docs/architecture.md` stays the historical 10 Jul design and
  debate record; the new folder documents the implementation as
  it exists.
- Explainer format: top ELI10, per-section Intent questions,
  "A simple way to view it", section-local link index;
  renderer-safe PlantUML, simple + de-risk tiers where earned.
- Repo conventions: hard-wrap ~65 cols, "-" not em dashes, no
  AI/tooling attribution, preview-safe absolute links with no
  line numbers inside link targets.
- Placement decision: root `architecture/` (mirrors the T5
  root-level precedent and the user's "/architecture" wording;
  avoids clashing with `docs/architecture.md`).

## Scope revision (2026-07-11, user feedback)

The user rejected the T5 monolith on all four counts: one giant
file, repetitive explainer boilerplate, too abstract, wrong
emphasis. Decisions:

- Write PURELY from source; the T5 text is not given to
  writers (its caveats become "verify in source" checks for
  the reviewer, not copied facts).
- Lean skeleton per file: one ELI10, technical meat, optional
  short design-rationale, one References list at the end. No
  per-section Intent questions / mental models / link indexes.
- Concreteness bar: exact constant names + values + defining
  file for every threshold; real identifiers; worked examples
  from LOG.md / reports/ where available.
- Rebalanced to 10 files; runtimes/testing/deployment condensed
  to one; six files serve the core loop.

## Acceptance Criteria

- [x] `architecture/` contains README index + 9 numbered docs
  (rebalanced per scope revision) covering context+contracts,
  core loop, time budgeting, parsing/checkpoints, perception,
  answer heads, nav/exploration, runtimes/testing/deployment,
  gaps.
- [x] Core-loop docs (02, 03, 06) go deeper than the T5
  monolith: tick mechanics, exact thresholds, per-question-type
  submission paths, timeline treatment.
- [x] Lean explainer structure per scope revision; claims
  distinguish implemented vs planned/unwired behavior - all T5
  caveats independently rediscovered from source, plus new
  findings (ledger caps, regex-only live parse, unfed CP4
  timer).
- [x] All local links resolve; PlantUML blocks are
  renderer-safe; markdown hard-wrapped; identity-free.
- [x] Fresh-context verification pass against current source
  (verdict: accurate and internally consistent).
- [ ] LOG.md one-liner, INDEX.md entry, CLAUDE.md layout row;
  commit + push. (Docs done; commit pending.)

## Artifacts

- Docs: `../../../architecture/`
- T5 raw material: scratchpad `t5-architecture.md` (session-local)
- Review: review.md

## Stage Gates

| Gate | State | Evidence / notes |
| --- | --- | --- |
| Task | APPROVED | Direct request: "please generate a comprehensive /architecture". |
| Spec | APPROVED | Section list follows the user's stated emphasis; structure proposed in-turn. |
| Plan | APPROVED | Autonomous session; generation is reversible docs work. |

## Todo

- [x] Create folder + dispatch writer agents (9 sonnet + 1 opus
  for gaps) with per-file briefs; all 10 files written.
- [x] README index written against the final file list.
- [x] Mechanical checks: link resolution (all resolve), em-dash
  scan (clean), PlantUML pairing (clean), absolute-path leak
  scan (clean); fixed 27 chat-style reference lines in
  03-time-budgeting.md to relative Markdown links.
- [x] Fresh-context verifier (opus) cross-checks claims vs
  source + cross-doc constant consistency + gaps coverage of
  cross-writer findings; fixes applied (see review.md).
- [x] LOG.md one-liner, CLAUDE.md layout row.
- [ ] Commit + push via commit-pr.

## Notes

- 2026-07-11: T5 commit found dangling (reflog: rebase dropped
  it); interpreted as "redo as folder". Content salvaged via
  `git show a864e0b:ARCHITECTURE.md`.
