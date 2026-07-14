# T10 — Red-team design review (numbered T5 pre-merge; renumbered, id claimed by colored-cloud-tool) (Fable batch)

**Started:** 12 Jul 2026
**Status:** done (14 Jul 2026)

## Intent

Last day of the Fable subscription window (ends 12 Jul 11:59 PM PT). Spend the
remaining frontier quota on the highest-leverage judgment work: a structured
adversarial review of the whole system, now informed by real measured numbers
(GT battery, real-data replay), adjudicated into a ranked hardening backlog
that governs the post-Fable build weeks.

## Approach

Batch of five parallel frontier-model review agents, each attacking one facet
from a hostile stance, each writing one markdown deliverable under
`docs/redteam/`:

| # | Facet | Deliverable |
|---|---|---|
| 1 | Instruction-following pipeline (70.6% of points; weakest measured area) | `attack_instruction_following.md` |
| 2 | Numerical/counting (independent agreement 15–27%, over-count) | `attack_numerical.md` |
| 3 | Object-ref head + checkpoint prompts CP2–CP5 (live at eval, never reviewed) | `attack_object_ref_checkpoints.md` |
| 4 | Eval-day systems: watchdog, budgets, exploration, ROS adapter, docker | `attack_eval_day.md` |
| 5 | 2025 dossier adjudication: threshold priors + failure modes vs architecture v1.0 | `dossier_deltas.md` |

Main session then adjudicates all findings into
`docs/redteam/hardening_backlog.md` (ranked by expected points impact, with
executor-ready task specs for the top items).

## Acceptance criteria

- [x] Five attack/adjudication docs written, every claim carrying file:line or
      report evidence
- [x] Adjudicated `hardening_backlog.md`: ranked, deduplicated, each item with
      verdict (accepted/rejected + why), points-at-risk, and cost
- [x] No existing files modified by review agents; no tooling attribution
- [x] LOG one-liner + commit + push

## Notes

- 12 Jul: dispatched the five-agent batch. Heavy harness/API turbulence all
  day (permission-classifier outages surfacing as declined reads, three
  ECONNRESET waves, one 10-min stream stall, session limit at the very end):
  every agent needed 1–3 resumes. Anti-stall protocol that worked: treat
  denials as transient (retry ×2 then note-and-proceed) and write
  deliverables incrementally.
- 12 Jul: all five deliverables landed on disk. NUM, OR, IF, SYS complete;
  DD truncated after §A (agent died on the session limit before §B/§C).
- 14 Jul: recovery session. Verified all files; §A of DD complete and
  high-value (predicate-form mismatches vs the VLA-3D generation spec).
  §B/§C re-drafted by a standard-tier executor from the dossiers (Fable
  window closed; frontier re-run would burn credits), adjudicated in main
  session. Wrote `docs/redteam/hardening_backlog.md` (adjudication of all
  five reports: 4 dominant defects, tiered items H1–H15, sequencing incl.
  CV-sweep hold) + `docs/redteam/README.md` index.
- 14 Jul (implementation day 1): backlog execution began — batches A
  (H1/H2/H4a-b/H8-partial/H10, commit `4bcf18b`), B (H3/H4c/IF-F6,
  `8a8c78d`), H5 (`0863294`), each independently verified CONFIRMED.
  Merged main (tiering T9 + colored map + T7/T8 records); this task
  renumbered T5→T10 (pre-merge commit messages say "T5"). **T8-vs-DD
  verdict conflicts reconciled before H5 was coded** (appendix in
  dossier_deltas.md; the above() reversal was the big one — our own
  NUM-F2 empirics sided with T8 against our dossier agent). Post-fix
  GT battery rerun in progress; status ledger added to the backlog.
