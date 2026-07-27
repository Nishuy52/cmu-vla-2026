# Live baseline scores (2026-07-27)

1 run(s) scored. Headline metric per type: numerical = TRUE-answer match (1.0/0.0), object_reference = 3D IoU of the live marker vs the GT target box, instruction_following = rubric-proxy score (same scorers as the offline `gt_battery`; see module docstring). `delta` = live - offline on the same headline metric for the same question.

| Scene | Type | Live | Offline | Delta | Capture issues | Note | Question |
|---|---|---|---|---|---|---|---|
| livingroom_1 | inst | 0.5000 | 1.0000 | -0.5000 |  | legs=1/2 | Go to the potted plant closest to the pyramid candle holder and stop a |

Runs with a capture-completeness issue: 0/1.
