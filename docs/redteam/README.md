# Red-team design review — 12–14 Jul 2026

Adversarial review batch over the whole system, run against real measured
baselines (GT battery, real-data replay, 2025 dossiers). Task record:
`docs/tasks/T10-redteam-review/`.

**Read `hardening_backlog.md` first** — it is the adjudicated, deduplicated,
ranked output; the five facet reports are the evidence base.

| File | Facet | Headline |
|---|---|---|
| `hardening_backlog.md` | **Adjudicated backlog** (start here) | 4 dominant defects; sequencing incl. CV-sweep hold |
| `attack_instruction_following.md` | IF pipeline (36/51 eval pts) | IF never explores (deadlock); GT scorer measures the wrong thing; 3 unaligned scenes = wrong terminal goals |
| `attack_numerical.md` | Counting | Over-count is counting-logic (~65%) + scorer artifacts (~35%), not perception; hold the CV sweep |
| `attack_object_ref_checkpoints.md` | OR head + CP2–CP5 | Nested disambiguators silently dropped; CP4 seam inverts the fallback rule |
| `attack_eval_day.md` | Eval-day systems | QoS mismatch can zero the whole run; perception unwired in adapter; docker shape mismatch |
| `dossier_deltas.md` | 2025 prior art vs architecture v1.0 | Predicate functional forms disagree with the VLA-3D generation spec (scoring bugs); architecture shape upheld |

Conventions: findings are cited as `<REPORT>-F<n>` / `DD-A<n>`; every claim
in the reports carries a file:line or report citation; "What I could not
verify" sections mark the honest edges.
