# Battery diff

## Provenance

### Before

- tool: gt_battery
- generated_utc: 2026-07-17T15:49:08.413861+00:00
- git_commit: 25fd955c8657
- git_dirty: True
- calibration_sha1: addd43cce497

### After

- tool: gt_battery
- generated_utc: 2026-07-17T21:59:10.905622+00:00
- git_commit: cbfd548e3bd7
- git_dirty: True
- calibration_sha1: addd43cce497

## Topline

| Key | Before | After | Δ |
|---|---|---|---|
| instruction_following.mean_coverage_1m_aligned_diag | 0.3968 | 0.3968 | +0.0000 |
| instruction_following.mean_frechet_m_aligned_diag | 5.4157 | 5.4157 | +0.0000 |
| instruction_following.mean_ordered_leg_credit | 0.1389 | 0.1389 | +0.0000 |
| instruction_following.mean_rubric_score | 0.1167 | 0.1167 | +0.0000 |
| instruction_following.n | 30 | 30 | +0.0000 |
| instruction_following.n_aligned | 28 | 28 | +0.0000 |
| instruction_following.n_scored | 30 | 30 | +0.0000 |
| instruction_following.n_unaligned_scenes | 1 | 1 | +0.0000 |
| instruction_following.total_avoid_violations | 0 | 0 | +0.0000 |
| instruction_following.total_threading_violations | 8 | 8 | +0.0000 |
| numerical.independent_agreement_rate | 0.5556 | 0.5556 | +0.0000 |
| numerical.n | 15 | 15 | +0.0000 |
| numerical.n_no_independent_evidence | 3 | 3 | +0.0000 |
| numerical.n_with_independent | 9 | 9 | +0.0000 |
| numerical.n_with_scenegraph | 11 | 11 | +0.0000 |
| numerical.n_with_true_answer | 15 | 15 | +0.0000 |
| numerical.pipeline_determinism_rate | 1.0000 | 1.0000 | +0.0000 |
| numerical.scenegraph_agreement_rate | 0.6364 | 0.6364 | +0.0000 |
| numerical.true_accuracy | 1.0000 | 1.0000 | +0.0000 |
| object_reference.iou_at_0p25 | 0.8750 | 1.0000 | +0.1250 |
| object_reference.iou_at_0p5 | 0.8750 | 1.0000 | +0.1250 |
| object_reference.match_method_breakdown.none | 22 | 24 | +2.0000 |
| object_reference.match_method_breakdown.relation | 7 | 5 | -2.0000 |
| object_reference.match_method_breakdown.unique | 1 | 1 | +0.0000 |
| object_reference.mean_iou | 0.8750 | 1.0000 | +0.1250 |
| object_reference.n | 30 | 30 | +0.0000 |
| object_reference.n_scored | 8 | 6 | -2.0000 |

## Changed rows

| Scene | Qtype | Question | Changed fields |
|---|---|---|---|
| chinese_room | object_reference | Find the bowl on the table closest to the folding screen. | gt_target_id: 71 -> null; iou: 1.0 -> null; match_method: "relation" -> "none"; our_target_id: 71 -> 91; target_source: "referential" -> "none" |
| home_building_2 | object_reference | Find the speaker on the TV cabinet closest to the potted plant on the TV cabinet. | gt_target_id: 91 -> null; iou: 0.0 -> null; match_method: "relation" -> "none"; our_target_id: 108 -> 122; target_source: "referential" -> "none" |
| hotel_room_1 | object_reference | Find the picture above the suitcase furthest from the floor. | our_target_id: 6 -> 19 |
| livingroom_3 | object_reference | Find the potted plant near the books on the cabinet. | our_target_id: 58 -> 31 |

## Added / removed rows

(none)
