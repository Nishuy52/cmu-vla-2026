# Battery structural-health report (2026-07-11)

75 question(s) across 15 scene(s), regex-tier parse only (offline / deterministic). No ground-truth answers exist — these are STRUCTURAL health signals, not accuracy.

## Aggregate

- Answered before watchdog: **100%**
- Answer type correct: **100%**
- Target grounded (>=1 candidate): **100%**
- Floor/watchdog answer used: **0%**
- IF corridor legs built: **30%** | IF avoid specs built: **10%**
- Parse tiers: {'regex': 75}

### Per qtype

| QType | n | Answered<WD | Type OK | Grounded | Floor |
|---|---|---|---|---|---|

| numerical | 15 | 100% | 100% | 100% | 0% |
| object_reference | 30 | 100% | 100% | 100% | 0% |
| instruction_following | 30 | 100% | 100% | 100% | 0% |

## Per-question

| Scene | QType | Answered<WD | Type OK | Grounded | Corridor | Avoid | Floor | Sim s | Q |
|---|---|---|---|---|---|---|---|---|---|
| arabic_room | nume | yes | yes | yes | no | no | no | 67.0 | How many sofas are below a window? |
| arabic_room | obje | yes | yes | yes | no | no | no | 243.0 | Find the pillow closest to the book on the stool. |
| arabic_room | obje | yes | yes | yes | no | no | no | 243.0 | Find the wall lamp that is between a door frame and a win... |
| arabic_room | inst | yes | yes | yes | no | no | no | 243.0 | Go near the stool under the picture and stop at the small... |
| arabic_room | inst | yes | yes | yes | yes | no | no | 273.0 | First, go to the potted plant furthest from the hookah, t... |
| chinese_room | nume | yes | yes | yes | no | no | no | 67.0 | Count the number of chairs with pillows on them. |
| chinese_room | obje | yes | yes | yes | no | no | no | 243.0 | Find the bowl on the table closest to the folding screen. |
| chinese_room | obje | yes | yes | yes | no | no | no | 243.0 | Find the pillow on the chair that is closest to the TV. |
| chinese_room | inst | yes | yes | yes | no | no | no | 243.0 | Go near the potted plant on the table and stop at the pai... |
| chinese_room | inst | yes | yes | yes | no | yes | no | 273.0 | First, go near the tea table with the elephant figurine o... |
| home_building_1 | nume | yes | yes | yes | no | no | no | 67.0 | How many pillows are on the sofa under the pictures? |
| home_building_1 | obje | yes | yes | yes | no | no | no | 243.0 | Find the clock on the TV cabinet. |
| home_building_1 | obje | yes | yes | yes | no | no | no | 243.0 | Find the bowl closest to the knife rack near the trash can. |
| home_building_1 | inst | yes | yes | yes | no | no | no | 273.0 | Go to the coffee table with the kettle on it and stop at ... |
| home_building_1 | inst | yes | yes | yes | yes | no | no | 273.0 | First, go to the nightstand with a clock on it, then take... |
| home_building_2 | nume | yes | yes | yes | no | no | no | 67.0 | How many red pillows are on the sofa? |
| home_building_2 | obje | yes | yes | yes | no | no | no | 243.0 | Find the lamp on the nightstand that has the photo on it. |
| home_building_2 | obje | yes | yes | yes | no | no | no | 243.0 | Find the speaker on the TV cabinet closest to the potted ... |
| home_building_2 | inst | yes | yes | yes | no | no | no | 273.0 | Go near the magazine on the ottoman, then go to the potte... |
| home_building_2 | inst | yes | yes | yes | yes | no | no | 273.0 | Take the path between the sofa and the coffee table and g... |
| hotel_room_1 | nume | yes | yes | yes | no | no | no | 67.0 | How many pillows are on the bed? |
| hotel_room_1 | obje | yes | yes | yes | no | no | no | 243.0 | Find the bedside table farthest from the window. |
| hotel_room_1 | obje | yes | yes | yes | no | no | no | 243.0 | Find the picture above the suitcase furthest from the floor. |
| hotel_room_1 | inst | yes | yes | yes | no | no | no | 273.0 | Go to the bedside table closest to the window and stop at... |
| hotel_room_1 | inst | yes | yes | yes | yes | no | no | 273.0 | First, go near the bedside table closest to the bench, th... |
| hotel_room_2 | nume | yes | yes | yes | no | no | no | 67.0 | How many pictures are above the bed? |
| hotel_room_2 | obje | yes | yes | yes | no | no | no | 243.0 | Find the flowers near the window. |
| hotel_room_2 | obje | yes | yes | yes | no | no | no | 243.0 | Find the picture closest to the bench. |
| hotel_room_2 | inst | yes | yes | yes | yes | no | no | 243.0 | Go between the bench and the bed and stop at the lamp clo... |
| hotel_room_2 | inst | yes | yes | yes | yes | no | no | 273.0 | First, go to the picture closest to the door, then take t... |
| japanese_room | nume | yes | yes | yes | no | no | no | 67.0 | How many calligraphy paintings are above the display ledge? |
| japanese_room | obje | yes | yes | yes | no | no | no | 243.0 | The lantern between the vase and the stone decoration tha... |
| japanese_room | obje | yes | yes | yes | no | no | no | 243.0 | The red pillow closest to the sushi. |
| japanese_room | inst | yes | yes | yes | no | no | no | 243.0 | Go near the small table with a vase on it and then to the... |
| japanese_room | inst | yes | yes | yes | no | no | no | 273.0 | Go to the lantern closest to the fan decoration, then tak... |
| livingroom_1 | nume | yes | yes | yes | no | no | no | 67.0 | How many chairs are near the table with a vase on it? |
| livingroom_1 | obje | yes | yes | yes | no | no | no | 243.0 | Find the vase on the cabinet below the picture. |
| livingroom_1 | obje | yes | yes | yes | no | no | no | 243.0 | Find the pillow on the sofa that is closest to the windows. |
| livingroom_1 | inst | yes | yes | yes | no | no | no | 273.0 | Go to the potted plant closest to the pyramid candle hold... |
| livingroom_1 | inst | yes | yes | yes | yes | no | no | 273.0 | First, go near the lamp closest to the black chair, then ... |
| livingroom_2 | nume | yes | yes | yes | no | no | no | 67.0 | How many cups are on the coffee table? |
| livingroom_2 | obje | yes | yes | yes | no | no | no | 243.0 | Find the stool closest to the shelf near the TV cabinet. |
| livingroom_2 | obje | yes | yes | yes | no | no | no | 243.0 | Find the pillow on the sofa that is closest to the lamp. |
| livingroom_2 | inst | yes | yes | yes | no | no | no | 213.0 | Go to the microwave on the kitchen counter and then go to... |
| livingroom_2 | inst | yes | yes | yes | no | yes | no | 273.0 | First, go to the chair near the window, then stop at the ... |
| livingroom_3 | nume | yes | yes | yes | no | no | no | 67.0 | How many photos are on the TV cabinet? |
| livingroom_3 | obje | yes | yes | yes | no | no | no | 243.0 | Find the potted plant near the books on the cabinet. |
| livingroom_3 | obje | yes | yes | yes | no | no | no | 243.0 | Find the vase between the cabinet and the stool. |
| livingroom_3 | inst | yes | yes | yes | no | no | no | 273.0 | Take the path near the TV and go to the pillow farthest f... |
| livingroom_3 | inst | yes | yes | yes | no | no | no | 273.0 | First, go near the stool, then take the path near the cab... |
| livingroom_4 | nume | yes | yes | yes | no | no | no | 67.0 | How many pillows are on a sofa? |
| livingroom_4 | obje | yes | yes | yes | no | no | no | 243.0 | Find the picture closest to a window. |
| livingroom_4 | obje | yes | yes | yes | no | no | no | 243.0 | Find the fossil decoration closest to the phone. |
| livingroom_4 | inst | yes | yes | yes | no | no | no | 243.0 | Go near the chair closest to the bookcase and stop at the... |
| livingroom_4 | inst | yes | yes | yes | no | no | no | 273.0 | First, go near the fireplace, then go to the window close... |
| loft | nume | yes | yes | yes | no | no | no | 67.0 | How many black pillows are on the sofa? |
| loft | obje | yes | yes | yes | no | no | no | 243.0 | The blue chair that is closest to the cup of coffee. |
| loft | obje | yes | yes | yes | no | no | no | 243.0 | Find the potted plant between a vase and the cabinet with... |
| loft | inst | yes | yes | yes | no | yes | no | 273.0 | Go to the cup near the TV remote and avoid the path near ... |
| loft | inst | yes | yes | yes | no | no | no | 243.0 | Go near the fireplace, pass by the stairs, then stop at t... |
| office_1 | nume | yes | yes | yes | no | no | no | 67.0 | How many computer monitors are on the table closest to th... |
| office_1 | obje | yes | yes | yes | no | no | no | 243.0 | Find the potted plant on the file cabinet. |
| office_1 | obje | yes | yes | yes | no | no | no | 243.0 | Find the paper cup on the table closest to the projector ... |
| office_1 | inst | yes | yes | yes | no | no | no | 273.0 | Go to the potted plant furthest from the projector screen... |
| office_1 | inst | yes | yes | yes | yes | no | no | 273.0 | First, go near the potted plant on the shelf, then take t... |
| office_2 | nume | yes | yes | yes | no | no | no | 67.0 | How many potted plants are on a table? |
| office_2 | obje | yes | yes | yes | no | no | no | 243.0 | Find the computer monitor closest to the cabinet with a p... |
| office_2 | obje | yes | yes | yes | no | no | no | 243.0 | Find the box on the cabinet that is closest to the whiteb... |
| office_2 | inst | yes | yes | yes | no | no | no | 243.0 | Go near the potted plant on the cabinet and stop at the w... |
| office_2 | inst | yes | yes | yes | no | no | no | 273.0 | First, go to the trash can near the cabinet, then go to t... |
| studio | nume | yes | yes | yes | no | no | no | 67.0 | How many framed records are above the couch? |
| studio | obje | yes | yes | yes | no | no | no | 243.0 | Find the vase closest to the guitar. |
| studio | obje | yes | yes | yes | no | no | no | 243.0 | Find the beer bottle furthest from the couch. |
| studio | inst | yes | yes | yes | no | no | no | 273.0 | Go to the vases on the cabinet below the TV and stop at t... |
| studio | inst | yes | yes | yes | yes | no | no | 273.0 | First, go to the vase closest to the easel, then, take th... |
