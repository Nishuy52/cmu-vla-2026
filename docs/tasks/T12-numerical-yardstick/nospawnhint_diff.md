# Battery diff

## Provenance

### Before

- tool: gt_battery
- generated_utc: 2026-07-16T18:26:28.764027+00:00
- git_commit: a7ce9c12f308
- git_dirty: True
- calibration_sha1: 026ed8dfef31

### After

- tool: gt_battery
- generated_utc: 2026-07-16T18:33:27.000686+00:00
- git_commit: 8f7333ebd816
- git_dirty: True
- calibration_sha1: 026ed8dfef31

## Topline

| Key | Before | After | Δ |
|---|---|---|---|
| instruction_following.mean_coverage_1m_aligned_diag | 0.3998 | 0.2657 | -0.1341 |
| instruction_following.mean_frechet_m_aligned_diag | 4.5593 | 5.3427 | +0.7834 |
| instruction_following.mean_ordered_leg_credit | 0.1222 | 0.2222 | +0.1000 |
| instruction_following.mean_rubric_score | 0.1000 | 0.1889 | +0.0889 |
| instruction_following.n | 30 | 30 | +0.0000 |
| instruction_following.n_aligned | 24 | 24 | +0.0000 |
| instruction_following.n_scored | 30 | 30 | +0.0000 |
| instruction_following.n_unaligned_scenes | 3 | 3 | +0.0000 |
| instruction_following.total_avoid_violations | 0 | 1 | +1.0000 |
| instruction_following.total_threading_violations | 8 | 7 | -1.0000 |
| numerical.independent_agreement_rate | 0.5556 | 0.5556 | +0.0000 |
| numerical.n | 15 | 15 | +0.0000 |
| numerical.n_no_independent_evidence | 3 | 3 | +0.0000 |
| numerical.n_with_independent | 9 | 9 | +0.0000 |
| numerical.n_with_scenegraph | 11 | 11 | +0.0000 |
| numerical.n_with_true_answer | 15 | 15 | +0.0000 |
| numerical.pipeline_determinism_rate | 1.0000 | 1.0000 | +0.0000 |
| numerical.scenegraph_agreement_rate | 0.7273 | 0.7273 | +0.0000 |
| numerical.true_accuracy | 0.7333 | 0.7333 | +0.0000 |
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
| arabic_room | instruction_following | First, go to the potted plant furthest from the hookah, then take the path between the two columns, and stop at the tray on the table. | coverage_1m: 0.3221 -> 0.0; driven_n_poses: 28 -> 9; frechet_m: 3.460417220479202 -> 7.415604175723193; our_n_waypoints: 28 -> 9 |
| arabic_room | instruction_following | Go near the stool under the picture and stop at the small table farthest from the columns. | coverage_1m: 0.2576 -> 0.0; driven_n_poses: 51 -> 46; frechet_m: 3.295542960561886 -> 6.476685527848722; our_n_waypoints: 51 -> 46 |
| chinese_room | instruction_following | First, go near the tea table with the elephant figurine on it, then stop at the table with the horse figurine on it, avoiding the path between the chair and the folding screen. | coverage_1m: 0.1921 -> 0.0794; driven_n_poses: 6 -> 4; frechet_m: 6.563625496878366 -> 6.1797161177970805; our_n_waypoints: 6 -> 4 |
| chinese_room | instruction_following | Go near the potted plant on the table and stop at the painting near the TV. | coverage_1m: 0.2651 -> 0.1279; driven_n_poses: 63 -> 9; frechet_m: 6.258435733885026 -> 7.16867509668462; our_n_waypoints: 63 -> 9 |
| home_building_1 | instruction_following | First, go to the nightstand with a clock on it, then take the path between the dining table and the picture, and stop at the trash can closest to the refridgerator. | coverage_1m: 0.0293 -> 0.0; driven_n_poses: 30 -> 238; frechet_m: 13.070148139806987 -> 16.22219567326353; our_n_waypoints: 30 -> 238 |
| home_building_1 | instruction_following | Go to the coffee table with the kettle on it and stop at the dining table near the big picture. | coverage_1m: 0.0846 -> 0.0; driven_n_poses: 185 -> 84; frechet_m: 16.21725165931761 -> 16.23189352193397; our_n_waypoints: 185 -> 84 |
| home_building_2 | instruction_following | Go near the magazine on the ottoman, then go to the potted plant on the dressing table. | coverage_1m: 0.0699 -> 0.1932; driven_n_poses: 45 -> 61; frechet_m: 8.367893899754813 -> 12.042294908585818; leg_outcomes: [{"i": 0, "kind": "goto", "goal": [1.2610146516641076, 4.45810909851061], "reached_in_order": false, "threaded": null}, {"i": 1, "kind": "goto", "goal": [3.130286219107404, -0.6635941391844721], "reached_in_order": false, "threaded": null}] -> [{"i": 0, "kind": "goto", "goal": [1.2610146516641076, 4.45810909851061], "reached_in_order": true, "threaded": null}, {"i": 1, "kind": "goto", "goal": [3.130286219107404, -0.6635941391844721], "reached_in_order": false, "threaded": null}]; n_legs_reached_in_order: 0 -> 1; ordered_leg_credit: 0.0 -> 0.5; our_n_waypoints: 45 -> 61; rubric_score: 0.0 -> 0.5 |
| home_building_2 | instruction_following | Take the path between the sofa and the coffee table and go to the kettle on the dining table, then go to the potted plant between the curtain and the TV. | coverage_1m: 0.0766 -> 0.2108; driven_n_poses: 195 -> 261; frechet_m: 20.398138688573358 -> 20.398527153784787; leg_outcomes: [{"i": 0, "kind": "corridor_between", "goal": [-0.06634096150392166, 7.51439952650719], "reached_in_order": false, "threaded": false}, {"i": 1, "kind": "goto", "goal": [1.1735493261216094, 1.4193672086698286], "reached_in_order": false, "threaded": null}, {"i": 2, "kind": "goto", "goal": [-3.803853304009978, -0.7176745735671239], "reached_in_order": false, "threaded": null}] -> [{"i": 0, "kind": "corridor_between", "goal": [-0.06634096150392166, 7.51439952650719], "reached_in_order": false, "threaded": false}, {"i": 1, "kind": "goto", "goal": [1.1735493261216094, 1.4193672086698286], "reached_in_order": true, "threaded": null}, {"i": 2, "kind": "goto", "goal": [-3.803853304009978, -0.7176745735671239], "reached_in_order": false, "threaded": null}]; n_legs_reached_in_order: 0 -> 1; ordered_leg_credit: 0.0 -> 0.3333; our_n_waypoints: 195 -> 261 |
| hotel_room_1 | instruction_following | First, go near the bedside table closest to the bench, then take the path between the TV and the bed to the picture closest to the TV. | coverage_1m: 0.7469 -> 0.6724; driven_n_poses: 54 -> 56; frechet_m: 1.6120538534674147 -> 1.8647683400332677; our_n_waypoints: 54 -> 56 |
| hotel_room_1 | instruction_following | Go to the bedside table closest to the window and stop at the chair closest to the TV. | coverage_1m: 0.9388 -> 0.9346; driven_n_poses: 61 -> 59; frechet_m: 1.228033273851248 -> 1.2294152006981334; our_n_waypoints: 61 -> 59 |
| hotel_room_2 | instruction_following | First, go to the picture closest to the door, then take the path between the TV cabinet and the bed, and stop by the curtain closest to the TV. | coverage_1m: 0.6012 -> 0.2982; driven_n_poses: 17 -> 19; frechet_m: 2.902868196367345 -> 4.233057873263294; leg_outcomes: [{"i": 0, "kind": "goto", "goal": [2.6144971939508657, -1.314291003592708], "reached_in_order": false, "threaded": null}, {"i": 1, "kind": "corridor_between", "goal": [-0.16876974959572938, 1.9038136454935441], "reached_in_order": false, "threaded": false}, {"i": 2, "kind": "goto", "goal": [-4.416262138073918, -0.8613001156648123], "reached_in_order": true, "threaded": null}] -> [{"i": 0, "kind": "goto", "goal": [2.6144971939508657, -1.314291003592708], "reached_in_order": false, "threaded": null}, {"i": 1, "kind": "corridor_between", "goal": [-0.16876974959572938, 1.9038136454935441], "reached_in_order": true, "threaded": false}, {"i": 2, "kind": "goto", "goal": [-4.416262138073918, -0.8613001156648123], "reached_in_order": true, "threaded": null}]; n_legs_reached_in_order: 1 -> 2; ordered_leg_credit: 0.3333 -> 0.6667; our_n_waypoints: 17 -> 19; rubric_score: 0.0 -> 0.3333 |
| hotel_room_2 | instruction_following | Go between the bench and the bed and stop at the lamp closest to the fireplace. | coverage_1m: 1.0 -> 0.9484; driven_n_poses: 4 -> 5; frechet_m: 0.6063179817919504 -> 2.3712952037671653; leg_outcomes: [{"i": 0, "kind": "corridor_between", "goal": [-0.43276481677540773, 1.403682685083295], "reached_in_order": false, "threaded": false}, {"i": 1, "kind": "goto", "goal": [0.8848910815122211, 1.4200463453290173], "reached_in_order": false, "threaded": null}] -> [{"i": 0, "kind": "corridor_between", "goal": [-0.43276481677540773, 1.403682685083295], "reached_in_order": true, "threaded": true}, {"i": 1, "kind": "goto", "goal": [0.8848910815122211, 1.4200463453290173], "reached_in_order": false, "threaded": null}]; n_legs_reached_in_order: 0 -> 1; n_threading_violations: 1 -> 0; ordered_leg_credit: 0.0 -> 0.5; our_n_waypoints: 4 -> 5; rubric_score: 0.0 -> 0.5 |
| japanese_room | instruction_following | Go near the small table with a vase on it and then to the flowers near the jar. | coverage_1m: 0.1917 -> 0.0; driven_n_poses: 53 -> 113; frechet_m: 4.745261923070459 -> 5.653166420821805; our_n_waypoints: 53 -> 113 |
| japanese_room | instruction_following | Go to the lantern closest to the fan decoration, then take the path near the wardrobe doors to the flowers on the display ledge. | coverage_1m: 0.2678 -> 0.258; driven_n_poses: 96 -> 77; frechet_m: 5.147718371239248 -> 4.5862861579942775; leg_outcomes: [{"i": 0, "kind": "goto", "goal": [1.3969999714249122, -0.5210001486425236], "reached_in_order": false, "threaded": null}, {"i": 1, "kind": "via_near", "goal": [0.1395487283988932, 4.485747208787377], "reached_in_order": false, "threaded": null}, {"i": 2, "kind": "goto", "goal": [-2.347529211138802, 4.857188684285708], "reached_in_order": false, "threaded": null}] -> [{"i": 0, "kind": "goto", "goal": [1.3969999714249122, -0.5210001486425236], "reached_in_order": false, "threaded": null}, {"i": 1, "kind": "via_near", "goal": [0.1395487283988932, 4.485747208787377], "reached_in_order": true, "threaded": null}, {"i": 2, "kind": "goto", "goal": [-2.347529211138802, 4.857188684285708], "reached_in_order": false, "threaded": null}]; n_legs_reached_in_order: 0 -> 1; ordered_leg_credit: 0.0 -> 0.3333; our_n_waypoints: 96 -> 77; rubric_score: 0.0 -> 0.3333 |
| livingroom_1 | instruction_following | First, go near the lamp closest to the black chair, then take the path between the sofa and the round tables, and stop at the cabinet with a picture above it. | coverage_1m: 0.3685 -> 0.0; driven_n_poses: 48 -> 3; frechet_m: 5.916194314433992 -> 9.208565933843639; our_n_waypoints: 48 -> 3 |
| livingroom_1 | instruction_following | Go to the potted plant closest to the pyramid candle holder and stop at the vase between the TV and the door. | coverage_1m: 0.1185 -> 0.0; driven_n_poses: 4002 -> 5; frechet_m: 4.678133123261484 -> 7.703526671984266; leg_outcomes: [{"i": 0, "kind": "goto", "goal": [1.781554902322501, -4.2988059739525815], "reached_in_order": false, "threaded": null}, {"i": 1, "kind": "goto", "goal": [-1.1499276868975108, 2.342958504160549], "reached_in_order": true, "threaded": null}] -> [{"i": 0, "kind": "goto", "goal": [1.781554902322501, -4.2988059739525815], "reached_in_order": true, "threaded": null}, {"i": 1, "kind": "goto", "goal": [-1.1499276868975108, 2.342958504160549], "reached_in_order": false, "threaded": null}]; our_n_waypoints: 4002 -> 5 |
| livingroom_2 | instruction_following | First, go to the chair near the window, then stop at the soccer ball near the couch, avoiding the path between the TV and the tea table. | coverage_1m: 0.333 -> 0.0; driven_n_poses: 4002 -> 70; frechet_m: 4.466674599374724 -> 5.679618867586609; n_avoid_violations: 0 -> 1; our_n_waypoints: 4002 -> 70 |
| livingroom_2 | instruction_following | Go to the microwave on the kitchen counter and then go to the crystal ball decoration on the shelf near the TV. | coverage_1m: 0.5992 -> 0.0; driven_n_poses: 45 -> 70; frechet_m: 2.8193598093109284 -> 4.1255647924637255; our_n_waypoints: 45 -> 70 |
| livingroom_3 | instruction_following | First, go near the stool, then take the path near the cabinet, and stop at the bowl on the table. | coverage_1m: 0.4215 -> 0.0931; frechet_m: 4.433155405884215 -> 3.576113636061884 |
| livingroom_3 | instruction_following | Take the path near the TV and go to the pillow farthest from the lamp. | coverage_1m: 0.3009 -> 0.0; driven_n_poses: 68 -> 46; frechet_m: 6.237712839956094 -> 3.5767318707605815; our_n_waypoints: 68 -> 46 |
| livingroom_4 | instruction_following | First, go near the fireplace, then go to the window closest to the bookcase, and stop at the chair farthest from the mirror. | coverage_1m: 0.4477 -> 0.4387; frechet_m: 2.5114539147782615 -> 2.494440331248057 |
| livingroom_4 | instruction_following | Go near the chair closest to the bookcase and stop at the table with the flowers on it. | coverage_1m: 0.5125 -> 0.5893; driven_n_poses: 4002 -> 44; frechet_m: 2.2409188973489167 -> 2.114640199546464; our_n_waypoints: 4002 -> 44 |
| loft | instruction_following | Go near the fireplace, pass by the stairs, then stop at the sphere decoration on the cabinet. | coverage_1m: 0.2685 -> 0.0623; driven_n_poses: 4002 -> 61; frechet_m: 7.255682879813292 -> 5.907479794189338; leg_outcomes: [{"i": 0, "kind": "goto", "goal": [-1.6460001385558578, 0.9700002957901653], "reached_in_order": false, "threaded": null}, {"i": 1, "kind": "via_near", "goal": [1.8161145557947749, -0.316929289683928], "reached_in_order": false, "threaded": null}, {"i": 2, "kind": "goto", "goal": [6.877113061233509, -1.6266811211170369], "reached_in_order": false, "threaded": null}] -> [{"i": 0, "kind": "goto", "goal": [-1.6460001385558578, 0.9700002957901653], "reached_in_order": false, "threaded": null}, {"i": 1, "kind": "via_near", "goal": [1.8161145557947749, -0.316929289683928], "reached_in_order": false, "threaded": null}, {"i": 2, "kind": "goto", "goal": [6.877113061233509, -1.6266811211170369], "reached_in_order": true, "threaded": null}]; n_legs_reached_in_order: 0 -> 1; ordered_leg_credit: 0.0 -> 0.3333; our_n_waypoints: 4002 -> 61; rubric_score: 0.0 -> 0.3333 |
| loft | instruction_following | Go to the cup near the TV remote and avoid the path near the cabinet. | coverage_1m: 0.0907 -> 0.0988; driven_n_poses: 4 -> 6; frechet_m: 7.751496475528258 -> 4.815231444810237; leg_outcomes: [{"i": 0, "kind": "goto", "goal": [6.203529128048337, 0.684117738247934], "reached_in_order": false, "threaded": null}] -> [{"i": 0, "kind": "goto", "goal": [6.203529128048337, 0.684117738247934], "reached_in_order": true, "threaded": null}]; n_legs_reached_in_order: 0 -> 1; ordered_leg_credit: 0.0 -> 1.0; our_n_waypoints: 4 -> 6; rubric_score: 0.0 -> 1.0 |
| office_1 | instruction_following | First, go near the potted plant on the shelf, then take the path between the two tables, and stop at the bench closest to the map wall decal. | coverage_1m: 0.8384 -> 0.7096; frechet_m: 1.3557188875782347 -> 1.471403548899476 |
| office_1 | instruction_following | Go to the potted plant furthest from the projector screen then stop at the water cooler near the window. | coverage_1m: 0.236 -> 0.3294; driven_n_poses: 46 -> 48; frechet_m: 4.189280263573234 -> 4.273247055731311; our_n_waypoints: 46 -> 48 |
| office_2 | instruction_following | First, go to the trash can near the cabinet, then go to the folder on the cabinet closest to the whiteboard, and finally, to the door near the exit sign. | coverage_1m: 0.6169 -> 0.368; driven_n_poses: 35 -> 48; frechet_m: 3.9890822301760447 -> 4.125031338535446; leg_outcomes: [{"i": 0, "kind": "goto", "goal": [2.891661531137774, -0.5330266165326596], "reached_in_order": true, "threaded": null}, {"i": 1, "kind": "goto", "goal": [5.903858093455036, 3.139529553765287], "reached_in_order": false, "threaded": null}, {"i": 2, "kind": "goto", "goal": [6.141392347445373, -1.7087482913623502], "reached_in_order": true, "threaded": null}] -> [{"i": 0, "kind": "goto", "goal": [2.891661531137774, -0.5330266165326596], "reached_in_order": false, "threaded": null}, {"i": 1, "kind": "goto", "goal": [5.903858093455036, 3.139529553765287], "reached_in_order": false, "threaded": null}, {"i": 2, "kind": "goto", "goal": [6.141392347445373, -1.7087482913623502], "reached_in_order": true, "threaded": null}]; n_legs_reached_in_order: 2 -> 1; ordered_leg_credit: 0.6667 -> 0.3333; our_n_waypoints: 35 -> 48; rubric_score: 0.6667 -> 0.3333 |
| office_2 | instruction_following | Go near the potted plant on the cabinet and stop at the window closest to the clock. | coverage_1m: 0.4611 -> 0.5476; driven_n_poses: 49 -> 46; frechet_m: 2.3952046960804636 -> 2.139379035188075; our_n_waypoints: 49 -> 46 |
| studio | instruction_following | First, go to the vase closest to the easel, then, take the path between the couch and the table and stop at the window closest to the couch. | coverage_1m: 0.1184 -> 0.0; driven_n_poses: 19 -> 9; frechet_m: 4.100034351680412 -> 4.414373700302494; our_n_waypoints: 19 -> 9 |
| studio | instruction_following | Go to the vases on the cabinet below the TV and stop at the guitar near the couch. | coverage_1m: 0.1449 -> 0.121; driven_n_poses: 46 -> 53; our_n_waypoints: 46 -> 53 |

## Added / removed rows

(none)
