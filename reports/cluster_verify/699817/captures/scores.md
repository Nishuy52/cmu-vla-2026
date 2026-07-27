# Live baseline scores (2026-07-27)

8 run(s) scored. Headline metric per type: numerical = TRUE-answer match (1.0/0.0), object_reference = 3D IoU of the live marker vs the GT target box, instruction_following = rubric-proxy score (same scorers as the offline `gt_battery`; see module docstring). `delta` = live - offline on the same headline metric for the same question.

| Scene | Type | Live | Offline | Delta | Capture issues | Note | Question |
|---|---|---|---|---|---|---|---|
| arabic_room | inst | 0.0000 | 0.5000 | -0.5000 |  | legs=0/2 | Go near the stool under the picture and stop at the small table farthe |
| chinese_room | inst | 0.5000 | 1.0000 | -0.5000 |  | legs=1/2 | Go near the potted plant on the table and stop at the painting near th |
| home_building_1 | inst | 0.0000 | 0.0000 | 0.0000 |  | legs=0/2 | Go to the coffee table with the kettle on it and stop at the dining ta |
| home_building_2 | inst | 1.0000 | 1.0000 | 0.0000 |  | legs=2/2 | Go near the magazine on the ottoman, then go to the potted plant on th |
| hotel_room_1 | inst | 0.5000 | 1.0000 | -0.5000 |  | legs=1/2 | Go to the bedside table closest to the window and stop at the chair cl |
| hotel_room_2 | inst | 0.0000 | 1.0000 | -1.0000 |  | legs=1/2; leg 0: threading: trajectory never crossed the gate segment | Go between the bench and the bed and stop at the lamp closest to the f |
| japanese_room | inst | 1.0000 | 1.0000 | 0.0000 |  | legs=2/2 | Go near the small table with a vase on it and then to the flowers near |
| livingroom_1 | inst | 0.0000 | 1.0000 | -1.0000 |  | legs=0/2 | Go to the potted plant closest to the pyramid candle holder and stop a |

Runs with a capture-completeness issue: 0/8.
