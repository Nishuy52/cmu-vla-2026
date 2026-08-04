# Live baseline scores (2026-08-04)

14 run(s) scored. Headline metric per type: numerical = TRUE-answer match (1.0/0.0), object_reference = 3D IoU of the live marker vs the GT target box, instruction_following = rubric-proxy score (same scorers as the offline `gt_battery`; see module docstring). `delta` = live - offline on the same headline metric for the same question.

**Offline column provenance (#139):** `Offline`/`offline` is NOT recomputed from each run's own bag — it is a fixed baseline snapshot read from `reports/gt_battery_main_post_carve/gt_battery_results.json` and looked up by (scene, qtype, question) only. It is identical across different live captures of the same question by design (the offline battery is a separate simulated-follower pipeline); it is not evidence that a change to the live driving/scoring had no effect.

| Scene | Type | Live | Offline | Delta | Capture issues | Note | Question |
|---|---|---|---|---|---|---|---|
| livingroom_2 | obje | n/a | n/a | n/a |  | no GT target matched; IoU undefined (flagged, not guessed) | Find the pillow on the sofa that is closest to the lamp. |
| livingroom_2 | obje | 0.2592 | n/a | n/a |  |  | Find the stool closest to the shelf near the TV cabinet. |
| livingroom_3 | obje | n/a | n/a | n/a |  | no GT target matched; IoU undefined (flagged, not guessed) | Find the potted plant near the books on the cabinet. |
| livingroom_3 | obje | 0.0000 | n/a | n/a |  |  | Find the vase between the cabinet and the stool. |
| livingroom_4 | obje | 0.0000 | n/a | n/a |  |  | Find the fossil decoration closest to the phone. |
| livingroom_4 | obje | n/a | n/a | n/a |  | no GT target matched; IoU undefined (flagged, not guessed) | Find the picture closest to a window. |
| loft | obje | 0.0000 | 1.0000 | -1.0000 | perception built zero instances (None keyframes processed) — capture-completeness failure (#158) |  | Find the potted plant between a vase and the cabinet with a TV on it. |
| loft | obje | n/a | n/a | n/a | perception built zero instances (None keyframes processed) — capture-completeness failure (#158) | no GT target matched; IoU undefined (flagged, not guessed) | The blue chair that is closest to the cup of coffee. |
| office_1 | obje | 0.0000 | n/a | n/a |  |  | Find the paper cup on the table closest to the projector screen. |
| office_1 | obje | 0.0500 | n/a | n/a |  |  | Find the potted plant on the file cabinet. |
| office_2 | obje | n/a | n/a | n/a |  | no GT target matched; IoU undefined (flagged, not guessed) | Find the box on the cabinet that is closest to the whiteboard. |
| office_2 | obje | n/a | n/a | n/a |  | no GT target matched; IoU undefined (flagged, not guessed) | Find the computer monitor closest to the cabinet with a phone on it. |
| studio | obje | 0.0000 | n/a | n/a |  |  | Find the beer bottle furthest from the couch. |
| studio | obje | 0.0000 | 1.0000 | -1.0000 |  |  | Find the vase closest to the guitar. |

Runs with a capture-completeness issue: 2/14.

## Mean headline by type

| Type | n | n_scored | n_excluded | mean_headline |
|---|---|---|---|---|
| numerical | 0 | 0 | 0 | n/a |
| object_reference | 14 | 8 | 0 | 0.0386 |
| instruction_following | 0 | 0 | 0 | n/a |
