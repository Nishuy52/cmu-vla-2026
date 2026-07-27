# Live baseline scores (2026-07-27)

6 run(s) scored. Headline metric per type: numerical = TRUE-answer match (1.0/0.0), object_reference = 3D IoU of the live marker vs the GT target box, instruction_following = rubric-proxy score (same scorers as the offline `gt_battery`; see module docstring). `delta` = live - offline on the same headline metric for the same question.

| Scene | Type | Live | Offline | Delta | Capture issues | Note | Question |
|---|---|---|---|---|---|---|---|
| livingroom_1 | inst | 0.0000 | 1.0000 | -1.0000 |  | legs=0/2 | Go to the potted plant closest to the pyramid candle holder and stop a |
| livingroom_1 | nume | 0.0000 | 1.0000 | -1.0000 |  | live=28 true=8 | How many chairs are near the table with a vase on it? |
| livingroom_1 | obje | n/a | n/a | n/a |  | no GT target matched; IoU undefined (flagged, not guessed) | Find the pillow on the sofa that is closest to the windows. |
| office_1 | inst | 0.3333 | 1.0000 | -0.6667 |  | legs=2/3; leg 1: threading: trajectory never crossed the gate segment | First, go near the potted plant on the shelf, then take the path betwe |
| office_1 | nume | 0.0000 | 1.0000 | -1.0000 |  | live=0 true=6 | How many computer monitors are on the table closest to the map wall de |
| office_1 | obje | n/a | n/a | n/a |  | no GT target matched; IoU undefined (flagged, not guessed) | Find the potted plant on the file cabinet. |

Runs with a capture-completeness issue: 0/6.
