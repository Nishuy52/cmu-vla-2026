# Live baseline scores (2026-08-10)

10 run(s) scored. Headline metric per type: numerical = TRUE-answer match (1.0/0.0), object_reference = 3D IoU of the live marker vs the GT target box, instruction_following = rubric-proxy score (same scorers as the offline `gt_battery`; see module docstring). `delta` = live - offline on the same headline metric for the same question.

**Offline column provenance (#139):** `Offline`/`offline` is NOT recomputed from each run's own bag — it is a fixed baseline snapshot read from `reports/gt_battery_main_post_carve/gt_battery_results.json` and looked up by (scene, qtype, question) only. It is identical across different live captures of the same question by design (the offline battery is a separate simulated-follower pipeline); it is not evidence that a change to the live driving/scoring had no effect.

| Scene | Type | Live | Offline | Delta | Capture issues | Note | Question |
|---|---|---|---|---|---|---|---|
| livingroom_4 | inst | 0.0000 | 1.0000 | -1.0000 |  | legs=0/3 | First, go near the fireplace, then go to the window closest to the boo |
| livingroom_4 | inst | 1.0000 | 1.0000 | 0.0000 |  | legs=1/1 | Go near the chair closest to the bookcase and stop at the table with t |
| loft | inst | 0.6667 | 1.0000 | -0.3333 |  | legs=2/3 | Go near the fireplace, pass by the stairs, then stop at the sphere dec |
| loft | inst | 0.0000 | 0.0000 | 0.0000 |  | legs=0/1 | Go to the cup near the TV remote and avoid the path near the cabinet. |
| office_1 | inst | 0.0000 | 1.0000 | -1.0000 |  | legs=1/3; leg 1: threading: trajectory needs >=2 (x,y) points | First, go near the potted plant on the shelf, then take the path betwe |
| office_1 | inst | 0.0000 | 1.0000 | -1.0000 |  | legs=0/2 | Go to the potted plant furthest from the projector screen then stop at |
| office_2 | inst | 0.3333 | 1.0000 | -0.6667 |  | legs=1/3 | First, go to the trash can near the cabinet, then go to the folder on  |
| office_2 | inst | 0.0000 | 1.0000 | -1.0000 |  | legs=0/2 | Go near the potted plant on the cabinet and stop at the window closest |
| studio | inst | 0.0000 | 1.0000 | -1.0000 |  | legs=0/3; leg 1: threading: trajectory never crossed the gate segment | First, go to the vase closest to the easel, then, take the path betwee |
| studio | inst | 0.0000 | 1.0000 | -1.0000 |  | legs=0/1 | Go to the vases on the cabinet below the TV and stop at the guitar nea |

Runs with a capture-completeness issue: 0/10.

## Mean headline by type

| Type | n | n_scored | n_excluded | mean_headline |
|---|---|---|---|---|
| numerical | 0 | 0 | 0 | n/a |
| object_reference | 0 | 0 | 0 | n/a |
| instruction_following | 10 | 10 | 0 | 0.2000 |
