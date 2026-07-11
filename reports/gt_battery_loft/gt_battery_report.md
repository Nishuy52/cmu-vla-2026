# GT battery — REAL accuracy report (2026-07-11)

5 question(s) across 1 ground-truth scene(s): loft. Missing/skipped scenes: none.

> **Circularity note (numerical):** the primary `pipeline_gt` count is OUR resolver run over the ground-truth geometry, so exact-match validates *pipeline self-consistency*, not absolute truth. The `indep` column is an independent second opinion from the referential-statement annotations (distinct annotated target instances); `referential_class_only` means the count is relation-agnostic (coarser). Disagreements are the informative signal.

> **OBB->AABB note:** GT boxes are the axis-aligned hull of each object's oriented box (rotated corners, min/max), a strict over-approximation for non-axis-aligned objects — small IoU deficits on rotated targets are partly this, not localisation error.

## Topline (per type)

- **Numerical** (n=1): pipeline exact-match 100%; independent agreement 0% over 1 with a second opinion.
- **Object reference** (n=2, scored=0): mean 3D IoU n/a; IoU>=0.25 n/a; IoU>=0.5 n/a.
- **Instruction following** (n=2): mean discrete Frechet 6.021 m; mean coverage within 1.0 m 0%.


## Per-question

| Scene | Type | Metric(s) | Note | Question |
|---|---|---|---|---|
| loft | nume | count=11 pipeline_gt=11, indep=9(referential_class_only) | pipeline=11 vs independent=9 (disagreement) | How many black pillows are on the sofa? |
| loft | obje | IoU=n/a tgt=None(none) | no GT target matched; IoU undefined (flagged, not guessed) | The blue chair that is closest to the cup of coffee. |
| loft | obje | IoU=n/a tgt=None(ambiguous) | no GT target matched; IoU undefined (flagged, not guessed) | Find the potted plant between a vase and the cabinet... |
| loft | inst | Frechet=6.57m cover1m=0% (ours 128/gt 739) | probable frame offset: our-path vs GT-path centroids far apart; VLA-3D object frame and challenge trajectory frame are not aligned (no transform shipped) — read Frechet/coverage as diagnostic only | Go to the cup near the TV remote and avoid the path ... |
| loft | inst | Frechet=5.47m cover1m=0% (ours 280/gt 771) | probable frame offset: our-path vs GT-path centroids far apart; VLA-3D object frame and challenge trajectory frame are not aligned (no transform shipped) — read Frechet/coverage as diagnostic only | Go near the fireplace, pass by the stairs, then stop... |
