# Battery diff

## Provenance

### Before

- tool: gt_battery
- generated_utc: 2026-07-16T19:18:42.901652+00:00
- git_commit: 1f0c99e9c594
- git_dirty: True
- calibration_sha1: 026ed8dfef31

### After

- tool: gt_battery
- generated_utc: 2026-07-17T15:49:08.413861+00:00
- git_commit: 25fd955c8657
- git_dirty: True
- calibration_sha1: addd43cce497

## Topline

| Key | Before | After | Δ |
|---|---|---|---|
| instruction_following.mean_coverage_1m_aligned_diag | 0.3844 | 0.3968 | +0.0124 |
| instruction_following.mean_frechet_m_aligned_diag | 5.6337 | 5.4157 | -0.2180 |
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
| numerical.true_accuracy | 0.8667 | 1.0000 | +0.1333 |
| object_reference.iou_at_0p25 | 0.8750 | 0.8750 | +0.0000 |
| object_reference.iou_at_0p5 | 0.8750 | 0.8750 | +0.0000 |
| object_reference.match_method_breakdown.none | 22 | 22 | +0.0000 |
| object_reference.match_method_breakdown.relation | 7 | 7 | +0.0000 |
| object_reference.match_method_breakdown.unique | 1 | 1 | +0.0000 |
| object_reference.mean_iou | 0.8750 | 0.8750 | +0.0000 |
| object_reference.n | 30 | 30 | +0.0000 |
| object_reference.n_scored | 8 | 8 | +0.0000 |

## Changed rows

| Scene | Qtype | Question | Changed fields |
|---|---|---|---|
| chinese_room | instruction_following | First, go near the tea table with the elephant figurine on it, then stop at the table with the horse figurine on it, avoiding the path between the chair and the folding screen. | coverage_1m: 0.2889 -> 0.6365; driven_n_poses: 10 -> 4002; frechet_m: 6.5433401039927155 -> 4.2747987939224386; leg_goals: [["goto", [-0.023072689144201186, -2.1974385433261934]], ["goto", [-0.023072689144201186, -2.1974385433261934]]] -> [["goto", [-0.023072689144201186, -2.1974385433261934]], ["goto", [6.228927551517312, -2.5931388830339324]]]; leg_outcomes: [{"i": 0, "kind": "goto", "goal": [-0.023072689144201186, -2.1974385433261934], "reached_in_order": false, "threaded": null}, {"i": 1, "kind": "goto", "goal": [-0.023072689144201186, -2.1974385433261934], "reached_in_order": false, "threaded": null}] -> [{"i": 0, "kind": "goto", "goal": [-0.023072689144201186, -2.1974385433261934], "reached_in_order": false, "threaded": null}, {"i": 1, "kind": "goto", "goal": [6.228927551517312, -2.5931388830339324], "reached_in_order": false, "threaded": null}]; our_n_waypoints: 10 -> 4002 |
| home_building_2 | instruction_following | Go near the magazine on the ottoman, then go to the potted plant on the dressing table. | driven_n_poses: 40 -> 242; frechet_m: 20.975638577964215 -> 17.139455497519506; leg_goals: [["goto", [1.2610146516641076, 4.45810909851061]], ["goto", [3.130286219107404, -0.6635941391844721]]] -> [["goto", [1.2610146516641076, 4.45810909851061]], ["goto", [-3.3362839432674747, 11.698064925936668]]]; leg_outcomes: [{"i": 0, "kind": "goto", "goal": [1.2610146516641076, 4.45810909851061], "reached_in_order": false, "threaded": null}, {"i": 1, "kind": "goto", "goal": [3.130286219107404, -0.6635941391844721], "reached_in_order": false, "threaded": null}] -> [{"i": 0, "kind": "goto", "goal": [1.2610146516641076, 4.45810909851061], "reached_in_order": false, "threaded": null}, {"i": 1, "kind": "goto", "goal": [-3.3362839432674747, 11.698064925936668], "reached_in_order": false, "threaded": null}]; our_n_waypoints: 40 -> 242 |
| home_building_2 | instruction_following | Take the path between the sofa and the coffee table and go to the kettle on the dining table, then go to the potted plant between the curtain and the TV. | leg_goals: [["corridor_between", [-0.06634096150392166, 7.51439952650719]], ["goto", [1.1735493261216094, 1.4193672086698286]], ["goto", [-3.803853304009978, -0.7176745735671239]]] -> [["corridor_between", [-0.06634096150392166, 7.51439952650719]], ["goto", [3.46397830448581, 11.027123219300286]], ["goto", [-3.803853304009978, -0.7176745735671239]]]; leg_outcomes: [{"i": 0, "kind": "corridor_between", "goal": [-0.06634096150392166, 7.51439952650719], "reached_in_order": false, "threaded": false}, {"i": 1, "kind": "goto", "goal": [1.1735493261216094, 1.4193672086698286], "reached_in_order": false, "threaded": null}, {"i": 2, "kind": "goto", "goal": [-3.803853304009978, -0.7176745735671239], "reached_in_order": false, "threaded": null}] -> [{"i": 0, "kind": "corridor_between", "goal": [-0.06634096150392166, 7.51439952650719], "reached_in_order": false, "threaded": false}, {"i": 1, "kind": "goto", "goal": [3.46397830448581, 11.027123219300286], "reached_in_order": false, "threaded": null}, {"i": 2, "kind": "goto", "goal": [-3.803853304009978, -0.7176745735671239], "reached_in_order": false, "threaded": null}] |
| home_building_2 | numerical | How many red pillows are on the sofa? | gt_count_pipeline: 3 -> 2; our_count: 3 -> 2; true_match: false -> true |
| livingroom_3 | object_reference | Find the potted plant near the books on the cabinet. | our_target_id: 31 -> 58 |
| loft | numerical | How many black pillows are on the sofa? | gt_count_pipeline: 0 -> 2; our_count: 0 -> 2; true_match: false -> true |

## Added / removed rows

(none)
