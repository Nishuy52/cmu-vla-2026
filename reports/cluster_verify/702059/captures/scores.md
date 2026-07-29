# Live baseline scores (2026-07-29)

10 run(s) scored. Headline metric per type: numerical = TRUE-answer match (1.0/0.0), object_reference = 3D IoU of the live marker vs the GT target box, instruction_following = rubric-proxy score (same scorers as the offline `gt_battery`; see module docstring). `delta` = live - offline on the same headline metric for the same question.

**Offline column provenance (#139):** `Offline`/`offline` is NOT recomputed from each run's own bag — it is a fixed baseline snapshot read from `reports/gt_battery_main_post_carve/gt_battery_results.json` and looked up by (scene, qtype, question) only. It is identical across different live captures of the same question by design (the offline battery is a separate simulated-follower pipeline); it is not evidence that a change to the live driving/scoring had no effect.

| Scene | Type | Live | Offline | Delta | Capture issues | Note | Question |
|---|---|---|---|---|---|---|---|
| arabic_room | inst | 0.6667 | 0.0000 | 0.6667 |  | legs=2/3 | First, go to the potted plant furthest from the hookah, then take the  |
| arabic_room | inst | 0.5000 | 0.5000 | 0.0000 |  | legs=1/2 | Go near the stool under the picture and stop at the small table farthe |
| chinese_room | inst | 0.0000 | 0.5000 | -0.5000 |  | legs=1/2; avoid[0]: trajectory entered capsule at (3.04, 0.72) | First, go near the tea table with the elephant figurine on it, then st |
| chinese_room | inst | 0.5000 | 1.0000 | -0.5000 |  | legs=1/2 | Go near the potted plant on the table and stop at the painting near th |
| home_building_1 | inst | 0.0000 | 0.0000 | 0.0000 |  | legs=0/3; leg 1: threading: trajectory never crossed the gate segment | First, go to the nightstand with a clock on it, then take the path bet |
| home_building_1 | inst | 0.5000 | 0.0000 | 0.5000 |  | legs=1/2 | Go to the coffee table with the kettle on it and stop at the dining ta |
| home_building_2 | inst | 0.5000 | 1.0000 | -0.5000 |  | legs=1/2 | Go near the magazine on the ottoman, then go to the potted plant on th |
| home_building_2 | inst | 0.0000 | 0.6667 | -0.6667 |  | legs=0/3; leg 0: threading: trajectory never crossed the gate segment | Take the path between the sofa and the coffee table and go to the kett |
| hotel_room_1 | inst | 0.6667 | 1.0000 | -0.3333 |  | legs=2/3 | First, go near the bedside table closest to the bench, then take the p |
| hotel_room_1 | inst | 0.5000 | 1.0000 | -0.5000 |  | legs=1/2 | Go to the bedside table closest to the window and stop at the chair cl |

Runs with a capture-completeness issue: 0/10.
