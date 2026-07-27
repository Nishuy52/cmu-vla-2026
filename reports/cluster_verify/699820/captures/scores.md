# Live baseline scores (2026-07-28)

15 run(s) scored. Headline metric per type: numerical = TRUE-answer match (1.0/0.0), object_reference = 3D IoU of the live marker vs the GT target box, instruction_following = rubric-proxy score (same scorers as the offline `gt_battery`; see module docstring). `delta` = live - offline on the same headline metric for the same question.

| Scene | Type | Live | Offline | Delta | Capture issues | Note | Question |
|---|---|---|---|---|---|---|---|
| arabic_room | obje | 0.0000 | n/a | n/a |  |  | Find the pillow closest to the book on the stool. |
| chinese_room | obje | n/a | n/a | n/a |  | no GT target matched; IoU undefined (flagged, not guessed) | Find the bowl on the table closest to the folding screen. |
| home_building_1 | obje | 0.0000 | 1.0000 | -1.0000 |  |  | Find the clock on the TV cabinet. |
| home_building_2 | obje | 0.0000 | n/a | n/a |  |  | Find the lamp on the nightstand that has the photo on it. |
| hotel_room_1 | obje | 0.0000 | 1.0000 | -1.0000 |  |  | Find the bedside table farthest from the window. |
| hotel_room_2 | obje | 0.0000 | 1.0000 | -1.0000 |  |  | Find the flowers near the window. |
| japanese_room | obje | n/a | n/a | n/a |  | no GT target matched; IoU undefined (flagged, not guessed) | The lantern between the vase and the stone decoration that is closest  |
| livingroom_1 | obje | n/a | n/a | n/a |  | no GT target matched; IoU undefined (flagged, not guessed) | Find the vase on the cabinet below the picture. |
| livingroom_2 | obje | n/a | n/a | n/a |  | no GT target matched; IoU undefined (flagged, not guessed) | Find the stool closest to the shelf near the TV cabinet. |
| livingroom_3 | obje | n/a | n/a | n/a |  | no GT target matched; IoU undefined (flagged, not guessed) | Find the potted plant near the books on the cabinet. |
| livingroom_4 | obje | n/a | n/a | n/a |  | no GT target matched; IoU undefined (flagged, not guessed) | Find the picture closest to a window. |
| loft | obje | n/a | n/a | n/a |  | no GT target matched; IoU undefined (flagged, not guessed) | The blue chair that is closest to the cup of coffee. |
| office_1 | obje | 0.0000 | n/a | n/a |  |  | Find the potted plant on the file cabinet. |
| office_2 | obje | n/a | n/a | n/a |  | no GT target matched; IoU undefined (flagged, not guessed) | Find the computer monitor closest to the cabinet with a phone on it. |
| studio | obje | 0.0333 | 1.0000 | -0.9667 |  |  | Find the vase closest to the guitar. |

Runs with a capture-completeness issue: 0/15.
