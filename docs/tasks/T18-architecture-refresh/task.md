# T18 — architecture-refresh

**Started:** 19 Jul 2026
**Intent:** Recreate the `architecture/` as-built reference folder (concept
from the archived 11 Jul snapshot, branch
`archive/2026-07-11-architecture-snapshot`) against TODAY's main. The old
folder was judged stale (drifted line anchors, superseded claims: regex-only
parse path, un-executed adapter, pre-Ubuntu framing) and archived rather
than merged.

## Context

- `docs/architecture.md` stays the 10 Jul design/debate record; the new
  `architecture/` folder documents what is actually built and tested on
  main as of 19 Jul 2026 (post sessions 17–19, issues #47–#70 era).
- Structure follows the archived folder: navigable chapters, ELI10 +
  PlantUML overview in the README, file:line anchors, closing
  gaps-and-risks chapter grounded in the live issue tracker.

## Acceptance criteria

- [ ] `architecture/README.md` — index, ELI10, how-it-runs, PlantUML overview
- [ ] Chapters 01–10 (contracts, core loop, time budgeting, parsing,
      perception, answer heads, navigation, evaluation/calibration,
      runtimes/deployment, gaps/risks)
- [ ] Every file/line anchor verified against current main (spot-checked
      by an independent verification pass before merge)
- [ ] Gaps chapter cross-referenced with open GitHub issues
      (#61 #62 #63 #64 #67 #68 #69 #70) and the reverted #67 fix (bbb66bd)
- [ ] No AI/tooling attribution anywhere

## Notes

- 19 Jul: task opened; branch `docs/current-architecture` renamed to
  `archive/2026-07-11-architecture-snapshot` on origin. Chapter writing
  dispatched to executor tier in three isolated worktrees (A: README+01–03,
  B: 04–06+08, C: 07+09+10); integration + verifier pass owned by the
  main session.
