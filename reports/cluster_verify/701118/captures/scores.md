# Live baseline scores (2026-07-28)

15 run(s) scored. Headline metric per type: numerical = TRUE-answer match (1.0/0.0), object_reference = 3D IoU of the live marker vs the GT target box, instruction_following = rubric-proxy score (same scorers as the offline `gt_battery`; see module docstring). `delta` = live - offline on the same headline metric for the same question.

| Scene | Type | Live | Offline | Delta | Capture issues | Note | Question |
|---|---|---|---|---|---|---|---|
| arabic_room | nume | 0.0000 | 1.0000 | -1.0000 |  | live=4 true=2 | How many sofas are below a window? |
| chinese_room | nume | 0.0000 | 1.0000 | -1.0000 |  | live=25 true=6 | Count the number of chairs with pillows on them. |
| home_building_1 | nume | 0.0000 | 1.0000 | -1.0000 |  | live=1 true=6 | How many pillows are on the sofa under the pictures? |
| home_building_2 | nume | 0.0000 | 1.0000 | -1.0000 |  | live=0 true=2 | How many red pillows are on the sofa? |
| hotel_room_1 | nume | 0.0000 | 1.0000 | -1.0000 |  | live=12 true=4 | How many pillows are on the bed? |
| hotel_room_2 | nume | 0.0000 | 1.0000 | -1.0000 |  | live=1 true=3 | How many pictures are above the bed? |
| japanese_room | nume | 0.0000 | 1.0000 | -1.0000 |  | live=1 true=3 | How many calligraphy paintings are above the display ledge? |
| livingroom_1 | nume | 0.0000 | 1.0000 | -1.0000 |  | live=13 true=8 | How many chairs are near the table with a vase on it? |
| livingroom_2 | nume | 0.0000 | 1.0000 | -1.0000 |  | live=6 true=2 | How many cups are on the coffee table? |
| livingroom_3 | nume | 0.0000 | 1.0000 | -1.0000 |  | live=0 true=2 | How many photos are on the TV cabinet? |
| livingroom_4 | nume | 0.0000 | 1.0000 | -1.0000 |  | live=5 true=6 | How many pillows are on a sofa? |
| loft | nume | 0.0000 | 1.0000 | -1.0000 |  | live=0 true=2 | How many black pillows are on the sofa? |
| office_1 | nume | 0.0000 | 1.0000 | -1.0000 |  | live=0 true=6 | How many computer monitors are on the table closest to the map wall de |
| office_2 | nume | 1.0000 | 1.0000 | 0.0000 |  | live=1 true=1 | How many potted plants are on a table? |
| studio | nume | 0.0000 | 1.0000 | -1.0000 |  | live=1 true=3 | How many framed records are above the couch? |

Runs with a capture-completeness issue: 0/15.
