# Live baseline scores (2026-07-28)

7 run(s) scored. Headline metric per type: numerical = TRUE-answer match (1.0/0.0), object_reference = 3D IoU of the live marker vs the GT target box, instruction_following = rubric-proxy score (same scorers as the offline `gt_battery`; see module docstring). `delta` = live - offline on the same headline metric for the same question.

| Scene | Type | Live | Offline | Delta | Capture issues | Note | Question |
|---|---|---|---|---|---|---|---|
| livingroom_2 | inst | 0.5000 | 1.0000 | -0.5000 |  | legs=1/2 | Go to the microwave on the kitchen counter and then go to the crystal  |
| livingroom_3 | inst | 0.0000 | 0.5000 | -0.5000 |  | legs=0/2 | Take the path near the TV and go to the pillow farthest from the lamp. |
| livingroom_4 | inst | 1.0000 | 1.0000 | 0.0000 |  | legs=2/2 | Go near the chair closest to the bookcase and stop at the table with t |
| loft | inst | 0.0000 | 0.0000 | 0.0000 |  | legs=0/1 | Go to the cup near the TV remote and avoid the path near the cabinet. |
| office_1 | inst | 0.0000 | 1.0000 | -1.0000 |  | legs=0/2 | Go to the potted plant furthest from the projector screen then stop at |
| office_2 | inst | 0.0000 | 1.0000 | -1.0000 |  | legs=0/2 | Go near the potted plant on the cabinet and stop at the window closest |
| studio | inst | 1.0000 | 1.0000 | 0.0000 |  | legs=2/2 | Go to the vases on the cabinet below the TV and stop at the guitar nea |

Runs with a capture-completeness issue: 0/7.
