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

- [x] `architecture/README.md` — index, ELI10, how-it-runs, PlantUML overview
- [x] Chapters 01–10 (contracts, core loop, time budgeting, parsing,
      perception, answer heads, navigation, evaluation/calibration,
      runtimes/deployment, gaps/risks)
- [x] Every file/line anchor verified against current main (spot-checked
      by an independent verification pass before merge)
- [x] Gaps chapter cross-referenced with open GitHub issues
      (#61 #62 #63 #64 #67 #68 #69 #70) and the reverted #67 fix (bbb66bd)
- [x] No AI/tooling attribution anywhere

## Notes

- 19 Jul: task opened; branch `docs/current-architecture` renamed to
  `archive/2026-07-11-architecture-snapshot` on origin. Chapter writing
  dispatched to executor tier in three isolated worktrees (A: README+01–03,
  B: 04–06+08, C: 07+09+10); integration + verifier pass owned by the
  main session.
- 19 Jul (cont.): all 11 files delivered (3,006 lines) and integrated.
  Writer findings worth keeping: interfaces.py budget constants (510/570)
  vs budget.py effective defaults (480/540) called out explicitly;
  `_nearest_free_goal` lives in runner/gt_battery.py, not nav/; production
  perception defaults to `VLA_DETECTOR=none`; design-doc self-consistency
  pass is budget-only scaffolding, never implemented; ch10 reflects the
  #69 closure + ceiling-diagnosis caveat (460ea33) that merged mid-task.
  Independent verifier pass dispatched before commit.
- 19 Jul (close): verifier CONFIRMED — 50+ anchors and all material-claim
  clusters correct; single discrepancy (wrong module path in ch10's
  "where to change what" table, `core/groundtruth/` → `core/runner/`)
  fixed before commit. CLAUDE.md Layout table gained an `architecture/`
  row with a refresh-on-drift note. Task done.
