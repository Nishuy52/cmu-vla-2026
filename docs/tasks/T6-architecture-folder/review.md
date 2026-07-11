# Review: architecture/ folder

## Scope

Fresh-context verification of all 10 files in `architecture/`
against current source, plus mechanical convention checks.

## Process

- 10 writer agents, one per file, source-only (the rejected
  single-file draft was withheld from them); each reported its
  3 most load-bearing claims with file:line evidence, several
  ran the relevant test files to confirm behavior is live.
- Mechanical pass: link resolution (all resolve), em/en-dash
  scan, PlantUML pairing/safety, absolute-path and attribution
  scan, chat-style reference cleanup in 03-time-budgeting.md.
- Fresh-context verifier re-read source and reconciled every
  constant quoted in more than one file.

## Findings

- Cross-doc constants: all agree (600/510/570/60/45 s gates,
  explore budgets 210/240/270, MERGE_IOU 0.3, MODAL_COUNT 2,
  checkpoint caps).
- All ~10 spot-verified load-bearing claims confirmed in
  source, including tick ordering, answer latching, the two
  unreconciled qtype classifiers, the ledger enforcement gap,
  frontier A* bypass, and the double-paid orientation window.
- Coverage fixes: six source-verified findings from the
  per-file writers were added to 09-gaps-and-risks.md (ledger
  caps D6, regex-only live parse D7, tracker/spec divergence
  D8, unfed CP4 remaining_s, two declared-but-unread
  constants).
- Convention fixes: 23 em dashes in 05-perception.md; 27
  chat-style reference lines converted to relative links in
  03-time-budgeting.md.

## Residual

- "around line N" citations drift by +-1 in a couple of spots
  (accepted; style says line numbers are approximate).
- README keeps an overview-altitude simplification of the
  checkpoint caps and defers to 09 for the nuance.
- The docs report but cannot close the live-system gaps; ROS
  adapter, Docker, real detector, and sim QoS stay unverified
  until Ubuntu integration.
