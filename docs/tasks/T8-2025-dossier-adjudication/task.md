# Task: Adjudicate 2025 dossiers vs architecture v1.0

## Intent
Mine the imported 2025 dossiers (docs/prior_art/) for
threshold priors and named failure modes from NROS (1st),
ReasonX (2nd), CopyPasta (3rd); decide which conflict with
architecture v1.0 decisions; record keep/change/defer
verdicts append-only per the provenance rules in
docs/prior_art/README.md.

## Context
- Follow-up to the research-import PR (branch
  docs/import-2025-research).
- Two tiers: mining (mechanical extraction, delegated) and
  adjudication (judgment, main chat).
- Known conflict already flagged by the implementation-gaps
  record (T4 hole #4): 2 of the 2025 top 3 leaned on the GT
  `/object_markers` topic, absent from the 2026 I/O list, so
  their reported methods overstate what ports forward. The
  "perception job is categorically harder in 2026"
  implication must be folded into docs/architecture.md and
  docs/master_plan.md (currently absent there).
- Adjudication targets: docs/architecture.md decisions +
  src/core/geometry/toolbox.py constants.

## Acceptance Criteria
- [x] Comparison table: each dossier threshold prior / failure
      mode vs our constant / decision, cited both sides.
      -> docs/mining-table.md
- [x] Every conflict gets a keep/change/defer verdict with
      rationale, recorded append-only per provenance rules.
      -> docs/verdicts.md
- [x] The GT-markers-banned implication lands in
      architecture.md and master_plan.md. (architecture.md
      section 8 calibration note + section 9 stale-line fix;
      master_plan.md Phase 2 recalibration checkbox.)

## Artifacts
- Mining table: docs/mining-table.md
- Verdicts: appended to docs/architecture.md (or a verdicts
  section per provenance rules)

## Stage Gates
| Gate | State | Evidence / notes |
| --- | --- | --- |
| Task | APPROVED | scoped in handoff item 4; mining tier explicitly "scout-able now" |
| Spec | N/A | analysis + doc-fold deliverable |
| Plan | N/A | mine (delegated) -> adjudicate -> fold into docs |

## Todo
- [x] Dispatch mining scout over docs/prior_art/ + toolbox
      constants
- [x] Review mining table; adjudicate each conflict
      keep/change/defer -> docs/verdicts.md
- [ ] Fold GT-markers implication into architecture.md +
      master_plan.md (executor dispatched, verify diff)
- [x] Record verdicts append-only per provenance rules
- [ ] (follow-on) apply CHANGE verdicts C2/C6 in code together
      with the diagnosis D1-D4 fixes (single fix pass; new
      task when picked up)

## Notes
- 2026-07-14: created from handoff item 4; mining scout
  dispatched in parallel with the numerical-count diagnosis
  (T5).
- 2026-07-14: adjudication done. Notable: two dossier priors
  were REJECTED on instance-level GT evidence from the T5
  diagnosis (on() 1 cm band, above() overlap gate) - the
  generator's own formulas disagree with the questions'
  effective semantics. C2 (on overlap 0.5) and C6 (relative
  per-class size) ruled CHANGE; C1 (near form) deferred to the
  k-fold sweep; C5 (between) kept with a scorer-side pair-shape
  fix flagged.
