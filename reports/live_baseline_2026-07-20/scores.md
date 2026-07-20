# Live baseline scores (2026-07-20)

5 run(s) scored. Headline metric per type: numerical = TRUE-answer match (1.0/0.0), object_reference = 3D IoU of the live marker vs the GT target box, instruction_following = rubric-proxy score (same scorers as the offline `gt_battery`; see module docstring). `delta` = live - offline on the same headline metric for the same question.

| Scene | Type | Live | Offline | Delta | Capture issues | Note | Question |
|---|---|---|---|---|---|---|---|
| livingroom_1 | inst | 0.0000 | 1.0000 | -1.0000 | missing /challenge_question (0 messages) — question read from run.log | legs=0/2 | Go to the potted plant closest to the pyramid candle holder and stop a |
| livingroom_1 | nume | 0.0000 | 1.0000 | -1.0000 |  | live=2 true=8 | How many chairs are near the table with a vase on it? |
| livingroom_1 | obje | n/a | n/a | n/a | missing /challenge_question (0 messages) — question read from run.log | no GT target matched; IoU undefined (flagged, not guessed) | Find the vase on the cabinet below the picture. |
| office_1 | inst | 0.5000 | 1.0000 | -0.5000 |  | legs=1/2 | Go to the potted plant furthest from the projector screen then stop at |
| office_1 | obje | n/a | n/a | n/a |  | no GT target matched; IoU undefined (flagged, not guessed) | Find the potted plant on the file cabinet. |

Runs with a capture-completeness issue: 2/5.
