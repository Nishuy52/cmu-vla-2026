# Live baseline scores (2026-08-05)

16 run(s) scored. Headline metric per type: numerical = TRUE-answer match (1.0/0.0), object_reference = 3D IoU of the live marker vs the GT target box, instruction_following = rubric-proxy score (same scorers as the offline `gt_battery`; see module docstring). `delta` = live - offline on the same headline metric for the same question.

**Offline column provenance (#139):** `Offline`/`offline` is NOT recomputed from each run's own bag — it is a fixed baseline snapshot read from `reports/gt_battery_main_post_carve/gt_battery_results.json` and looked up by (scene, qtype, question) only. It is identical across different live captures of the same question by design (the offline battery is a separate simulated-follower pipeline); it is not evidence that a change to the live driving/scoring had no effect.

| Scene | Type | Live | Offline | Delta | Capture issues | Note | Question |
|---|---|---|---|---|---|---|---|
| arabic_room | obje | 0.0579 | n/a | n/a |  |  | Find the pillow closest to the book on the stool. |
| arabic_room | obje | n/a | n/a | n/a |  | no GT target matched; IoU undefined (flagged, not guessed) | Find the wall lamp that is between a door frame and a window. |
| chinese_room | obje | 0.0000 | n/a | n/a |  |  | Find the bowl on the table closest to the folding screen. |
| chinese_room | obje | 0.2561 | n/a | n/a |  |  | Find the pillow on the chair that is closest to the TV. |
| home_building_1 | obje | 0.0000 | n/a | n/a |  |  | Find the bowl closest to the knife rack near the trash can. |
| home_building_1 | obje | 0.0000 | 1.0000 | -1.0000 |  |  | Find the clock on the TV cabinet. |
| home_building_2 | obje | 0.0000 | n/a | n/a | keyframes_processed=33 is far below the job's median (78) — capture-completeness issue (#158) |  | Find the lamp on the nightstand that has the photo on it. |
| home_building_2 | obje | 0.0000 | n/a | n/a |  |  | Find the speaker on the TV cabinet closest to the potted plant on the  |
| hotel_room_1 | obje | 0.0000 | 1.0000 | -1.0000 |  |  | Find the bedside table farthest from the window. |
| hotel_room_1 | obje | 0.0000 | n/a | n/a |  |  | Find the picture above the suitcase furthest from the floor. |
| hotel_room_2 | obje | 0.0000 | 1.0000 | -1.0000 |  |  | Find the flowers near the window. |
| hotel_room_2 | obje | 0.0000 | 1.0000 | -1.0000 |  |  | Find the picture closest to the bench. |
| japanese_room | obje | n/a | n/a | n/a |  | no GT target matched; IoU undefined (flagged, not guessed) | The lantern between the vase and the stone decoration that is closest  |
| japanese_room | obje | n/a | n/a | n/a |  | no GT target matched; IoU undefined (flagged, not guessed) | The red pillow closest to the sushi. |
| livingroom_1 | obje | n/a | n/a | n/a |  | no GT target matched; IoU undefined (flagged, not guessed) | Find the pillow on the sofa that is closest to the windows. |
| livingroom_1 | obje | 0.0000 | n/a | n/a |  |  | Find the vase on the cabinet below the picture. |

Runs with a capture-completeness issue: 1/16.

## Mean headline by type

| Type | n | n_scored | n_excluded | mean_headline |
|---|---|---|---|---|
| numerical | 0 | 0 | 0 | n/a |
| object_reference | 16 | 12 | 0 | 0.0262 |
| instruction_following | 0 | 0 | 0 | n/a |
