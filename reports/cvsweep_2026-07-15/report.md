# CV calibration sweep — reasoning thresholds (2026-07-15)

5-fold leave-3-out CV over 15 scenes, 60 random-search samples (seed 0). Cache: 2760 hits (0 from a prior on-disk run — resume) / 915 misses.

> **Objective (repaired, NUM-F7/H2):** points-weighted challenge composite — NUMERICAL STRICT independent (referential) count agreement (1pt; class-only / scene-graph rows excluded, not evidence), OBJECT_REFERENCE IoU>=0.25 on scoreable Qs (2pt), INSTRUCTION_FOLLOWING driven-trajectory RUBRIC score on aligned Qs (6pt; the retired planned-path coverage@1m is NOT scored). Normalised by available points; unscoreable / unaligned / no-strict-evidence questions excluded from both numerator and denominator.

## Fold scores

| Fold | Held-out scenes | Train | Holdout | Best config (diff vs default) |
|---|---|---|---|---|
| 0 | chinese_room, livingroom_4, livingroom_3 | 0.211 | 0.294 | (default) |
| 1 | hotel_room_2, loft, home_building_1 | 0.226 | 0.200 | (default) |
| 2 | home_building_2, livingroom_1, livingroom_2 | 0.214 | 0.241 | (default) |
| 3 | hotel_room_1, arabic_room, studio | 0.227 | 0.195 | (default) |
| 4 | office_1, japanese_room, office_2 | 0.220 | 0.216 | (default) |

- **Mean holdout:** 0.229 +/- 0.036
- **Mean train:** 0.219
- **Generalization gap (train - holdout):** -0.010

## Per-parameter stability (chosen value counts across folds)

| Param | Default | Value counts across folds | Recommendation |
|---|---|---|---|
| geometry.above_lateral_infl | 0.5 | 0.5×5 | 0.5 |
| geometry.in_containment_frac | 0.6 | 0.6×5 | 0.6 |
| geometry.near_floor | 1.2 | 1.2×5 | 1.2 |
| geometry.near_scale | 0.6 | 0.6×5 | 0.6 |
| geometry.next_to_gap | 0.75 | 0.75×5 | 0.75 |
| geometry.on_min_overlap_frac | 0.5 | 0.5×5 | 0.5 |
| geometry.on_top_tol | 0.15 | 0.15×5 | 0.15 |
| geometry.on_upper_span_frac | 0.25 | 0.25×5 | 0.25 |

## Final recommendation

No parameter reached strict-majority modal consensus across folds — every swept key is left at its default (the sweep found no value that robustly generalises; the defaults are the honest recommendation).

Parameters flagged `unstable — keep default` had no single value recur in a strict majority of folds — moving them would overfit one scene split.
