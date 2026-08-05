# Live baseline scores (2026-08-05)

10 run(s) scored. Headline metric per type: numerical = TRUE-answer match (1.0/0.0), object_reference = 3D IoU of the live marker vs the GT target box, instruction_following = rubric-proxy score (same scorers as the offline `gt_battery`; see module docstring). `delta` = live - offline on the same headline metric for the same question.

**Offline column provenance (#139):** `Offline`/`offline` is NOT recomputed from each run's own bag — it is a fixed baseline snapshot read from `reports/gt_battery_main_post_carve/gt_battery_results.json` and looked up by (scene, qtype, question) only. It is identical across different live captures of the same question by design (the offline battery is a separate simulated-follower pipeline); it is not evidence that a change to the live driving/scoring had no effect.

| Scene | Type | Live | Offline | Delta | Capture issues | Note | Question |
|---|---|---|---|---|---|---|---|
| hotel_room_2 | inst | 0.6667 | 1.0000 | -0.3333 |  | legs=2/3 | First, go to the picture closest to the door, then take the path betwe |
| hotel_room_2 | inst | 0.5000 | 1.0000 | -0.5000 |  | legs=2/2; leg 0: threading: trajectory never crossed the gate segment | Go between the bench and the bed and stop at the lamp closest to the f |
| japanese_room | inst | 1.0000 | 1.0000 | 0.0000 |  | legs=2/2 | Go near the small table with a vase on it and then to the flowers near |
| japanese_room | inst | 1.0000 | 1.0000 | 0.0000 |  | legs=3/3 | Go to the lantern closest to the fan decoration, then take the path ne |
| livingroom_1 | inst | 1.0000 | 0.0000 | 1.0000 |  | legs=2/2; leg 1: gate degenerate (anchors' footprints overlap, width=0.013m < 0.80m) — threading unevaluable, not scored | First, go near the lamp closest to the black chair, then take the path |
| livingroom_1 | inst | 0.0000 | 1.0000 | -1.0000 |  | legs=0/1 | Go to the potted plant closest to the pyramid candle holder and stop a |
| livingroom_2 | inst | 0.5000 | 0.5000 | 0.0000 | keyframes_processed=63 is far below the job's median (148) — capture-completeness issue (#158) | legs=2/2; avoid[0]: trajectory entered capsule at (1.16, -3.42) | First, go to the chair near the window, then stop at the soccer ball n |
| livingroom_2 | inst | 1.0000 | 1.0000 | 0.0000 |  | legs=1/1 | Go to the microwave on the kitchen counter and then go to the crystal  |
| livingroom_3 | inst | 0.5000 | 0.6667 | -0.1667 |  | legs=1/2 | First, go near the stool, then take the path near the cabinet, and sto |
| livingroom_3 | inst | 1.0000 | 0.5000 | 0.5000 |  | legs=1/1 | Take the path near the TV and go to the pillow farthest from the lamp. |

Runs with a capture-completeness issue: 1/10.

## Mean headline by type

| Type | n | n_scored | n_excluded | mean_headline |
|---|---|---|---|---|
| numerical | 0 | 0 | 0 | n/a |
| object_reference | 0 | 0 | 0 | n/a |
| instruction_following | 10 | 10 | 0 | 0.7167 |
