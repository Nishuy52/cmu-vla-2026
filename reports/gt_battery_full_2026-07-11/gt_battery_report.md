# GT battery — REAL accuracy report (2026-07-11)

75 question(s) across 15 ground-truth scene(s): arabic_room, chinese_room, home_building_1, home_building_2, hotel_room_1, hotel_room_2, japanese_room, livingroom_1, livingroom_2, livingroom_3, livingroom_4, loft, office_1, office_2, studio. Missing/skipped scenes: none.

> **Circularity note (numerical):** the primary `pipeline_gt` count is OUR resolver run over the ground-truth geometry, so exact-match validates *pipeline self-consistency*, not absolute truth. The `indep` column is an independent second opinion from the referential-statement annotations (distinct annotated target instances); `referential_class_only` means the count is relation-agnostic (coarser). Disagreements are the informative signal.

> **OBB->AABB note:** GT boxes are the axis-aligned hull of each object's oriented box (rotated corners, min/max), a strict over-approximation for non-axis-aligned objects — small IoU deficits on rotated targets are partly this, not localisation error.

## Topline (per type)

- **Numerical** (n=15): pipeline exact-match 100%; independent (referential) agreement 15% over 13; scene-graph agreement 27% over 15.
- **Object reference** (n=30, scored=6): mean 3D IoU 0.667; IoU>=0.25 67%; IoU>=0.5 67%. Match method: none=24, relation=6.
- **Instruction following** (n=30): aligned scenes score mean discrete Frechet 6.106 m; mean coverage within 1.0 m 30% (over 24 aligned Qs). 3 scene(s) unaligned (residual > 1 m, diagnostic-only): home_building_2, hotel_room_2, livingroom_3.


## Per-question

| Scene | Type | Metric(s) | Note | Question |
|---|---|---|---|---|
| arabic_room | nume | count=1 pipeline_gt=1, indep=3(referential_class_only), sg=3(scene_graph_class_only) | pipeline=1; independent=3; scene_graph=3 (disagreement) | How many sofas are below a window? |
| arabic_room | obje | IoU=n/a tgt=None(ambiguous/none) | no GT target matched; IoU undefined (flagged, not guessed) | Find the pillow closest to the book on the stool. |
| arabic_room | obje | IoU=n/a tgt=None(none/none) | no GT target matched; IoU undefined (flagged, not guessed) | Find the wall lamp that is between a door frame and ... |
| arabic_room | inst | Frechet=6.52m cover1m=26% fit=0.56m (ours 214/gt 792) | aligned via fitted scene transform (residual 0.56 m) | Go near the stool under the picture and stop at the ... |
| arabic_room | inst | Frechet=8.05m cover1m=36% fit=0.56m (ours 138/gt 829) | aligned via fitted scene transform (residual 0.56 m) | First, go to the potted plant furthest from the hook... |
| chinese_room | nume | count=6 pipeline_gt=6, indep=1(referential), sg=6(scene_graph) | pipeline=6; independent=1; scene_graph=6 (disagreement) | Count the number of chairs with pillows on them. |
| chinese_room | obje | IoU=0.000 tgt=71(referential/relation) | target matched via relation | Find the bowl on the table closest to the folding sc... |
| chinese_room | obje | IoU=n/a tgt=None(none/none) | no GT target matched; IoU undefined (flagged, not guessed) | Find the pillow on the chair that is closest to the TV. |
| chinese_room | inst | Frechet=6.21m cover1m=12% fit=0.66m (ours 216/gt 1509) | aligned via fitted scene transform (residual 0.66 m) | Go near the potted plant on the table and stop at th... |
| chinese_room | inst | Frechet=5.64m cover1m=19% fit=0.66m (ours 234/gt 630) | aligned via fitted scene transform (residual 0.66 m) | First, go near the tea table with the elephant figur... |
| home_building_1 | nume | count=31 pipeline_gt=31, indep=18(referential_class_only), sg=11(scene_graph) | pipeline=31; independent=18; scene_graph=11 (disagreement) | How many pillows are on the sofa under the pictures? |
| home_building_1 | obje | IoU=1.000 tgt=185(referential/relation) | target matched via relation | Find the clock on the TV cabinet. |
| home_building_1 | obje | IoU=n/a tgt=None(ambiguous/none) | no GT target matched; IoU undefined (flagged, not guessed) | Find the bowl closest to the knife rack near the tra... |
| home_building_1 | inst | Frechet=22.56m cover1m=8% fit=0.47m (ours 767/gt 733) | aligned via fitted scene transform (residual 0.47 m) | Go to the coffee table with the kettle on it and sto... |
| home_building_1 | inst | Frechet=20.09m cover1m=0% fit=0.47m (ours 1/gt 2015) | aligned via fitted scene transform (residual 0.47 m) | First, go to the nightstand with a clock on it, then... |
| home_building_2 | nume | count=10 pipeline_gt=10, indep=4(referential), sg=4(scene_graph) | pipeline=10; independent=4; scene_graph=4 (disagreement) | How many red pillows are on the sofa? |
| home_building_2 | obje | IoU=n/a tgt=None(none/none) | no GT target matched; IoU undefined (flagged, not guessed) | Find the lamp on the nightstand that has the photo o... |
| home_building_2 | obje | IoU=0.000 tgt=91(referential/relation) | target matched via relation | Find the speaker on the TV cabinet closest to the po... |
| home_building_2 | inst | Frechet=13.80m cover1m=7% fit=6.02m (ours 59/gt 916) | scene frame fit residual 6.02 m > 1.0 m — endpoints/spawn did not co-locate under a single rigid transform; Frechet/coverage diagnostic only | Go near the magazine on the ottoman, then go to the ... |
| home_building_2 | inst | Frechet=13.35m cover1m=0% fit=6.02m (ours 45/gt 1670) | scene frame fit residual 6.02 m > 1.0 m — endpoints/spawn did not co-locate under a single rigid transform; Frechet/coverage diagnostic only | Take the path between the sofa and the coffee table ... |
| hotel_room_1 | nume | count=4 pipeline_gt=4, indep=4(referential), sg=4(scene_graph) |  | How many pillows are on the bed? |
| hotel_room_1 | obje | IoU=n/a tgt=None(none/none) | no GT target matched; IoU undefined (flagged, not guessed) | Find the bedside table farthest from the window. |
| hotel_room_1 | obje | IoU=n/a tgt=None(none/none) | no GT target matched; IoU undefined (flagged, not guessed) | Find the picture above the suitcase furthest from th... |
| hotel_room_1 | inst | Frechet=4.38m cover1m=0% fit=0.20m (ours 26/gt 719) | aligned via fitted scene transform (residual 0.20 m) | Go to the bedside table closest to the window and st... |
| hotel_room_1 | inst | Frechet=4.45m cover1m=2% fit=0.20m (ours 36/gt 644) | aligned via fitted scene transform (residual 0.20 m) | First, go near the bedside table closest to the benc... |
| hotel_room_2 | nume | count=5 pipeline_gt=5, indep=3(referential), sg=3(scene_graph_class_only) | pipeline=5; independent=3; scene_graph=3 (disagreement) | How many pictures are above the bed? |
| hotel_room_2 | obje | IoU=n/a tgt=None(none/none) | no GT target matched; IoU undefined (flagged, not guessed) | Find the flowers near the window. |
| hotel_room_2 | obje | IoU=1.000 tgt=31(referential/relation) | target matched via relation | Find the picture closest to the bench. |
| hotel_room_2 | inst | Frechet=5.59m cover1m=0% fit=1.09m (ours 123/gt 155) | scene frame fit residual 1.09 m > 1.0 m — endpoints/spawn did not co-locate under a single rigid transform; Frechet/coverage diagnostic only | Go between the bench and the bed and stop at the lam... |
| hotel_room_2 | inst | Frechet=2.87m cover1m=0% fit=1.09m (ours 96/gt 845) | scene frame fit residual 1.09 m > 1.0 m — endpoints/spawn did not co-locate under a single rigid transform; Frechet/coverage diagnostic only | First, go to the picture closest to the door, then t... |
| japanese_room | nume | count=3 pipeline_gt=3, sg=4(scene_graph_class_only) | pipeline=3; scene_graph=4 (disagreement) | How many calligraphy paintings are above the display... |
| japanese_room | obje | IoU=n/a tgt=None(none/none) | no GT target matched; IoU undefined (flagged, not guessed) | The lantern between the vase and the stone decoratio... |
| japanese_room | obje | IoU=n/a tgt=None(none/none) | no GT target matched; IoU undefined (flagged, not guessed) | The red pillow closest to the sushi. |
| japanese_room | inst | Frechet=4.79m cover1m=18% fit=0.16m (ours 91/gt 339) | aligned via fitted scene transform (residual 0.16 m) | Go near the small table with a vase on it and then t... |
| japanese_room | inst | Frechet=7.78m cover1m=26% fit=0.16m (ours 325/gt 717) | aligned via fitted scene transform (residual 0.16 m) | Go to the lantern closest to the fan decoration, the... |
| livingroom_1 | nume | count=8 pipeline_gt=8, indep=9(referential), sg=8(scene_graph) | pipeline=8; independent=9; scene_graph=8 (disagreement) | How many chairs are near the table with a vase on it? |
| livingroom_1 | obje | IoU=n/a tgt=None(ambiguous/none) | no GT target matched; IoU undefined (flagged, not guessed) | Find the vase on the cabinet below the picture. |
| livingroom_1 | obje | IoU=n/a tgt=None(ambiguous/none) | no GT target matched; IoU undefined (flagged, not guessed) | Find the pillow on the sofa that is closest to the w... |
| livingroom_1 | inst | Frechet=4.63m cover1m=9% fit=0.93m (ours 36/gt 751) | aligned via fitted scene transform (residual 0.93 m) | Go to the potted plant closest to the pyramid candle... |
| livingroom_1 | inst | Frechet=1.34m cover1m=93% fit=0.93m (ours 79/gt 882) | aligned via fitted scene transform (residual 0.93 m) | First, go near the lamp closest to the black chair, ... |
| livingroom_2 | nume | count=1 pipeline_gt=1, indep=2(referential), sg=2(scene_graph) | pipeline=1; independent=2; scene_graph=2 (disagreement) | How many cups are on the coffee table? |
| livingroom_2 | obje | IoU=n/a tgt=None(none/none) | no GT target matched; IoU undefined (flagged, not guessed) | Find the stool closest to the shelf near the TV cabi... |
| livingroom_2 | obje | IoU=n/a tgt=None(ambiguous/none) | no GT target matched; IoU undefined (flagged, not guessed) | Find the pillow on the sofa that is closest to the l... |
| livingroom_2 | inst | Frechet=3.48m cover1m=44% fit=0.38m (ours 16/gt 791) | aligned via fitted scene transform (residual 0.38 m) | Go to the microwave on the kitchen counter and then ... |
| livingroom_2 | inst | Frechet=3.07m cover1m=35% fit=0.38m (ours 36/gt 928) | aligned via fitted scene transform (residual 0.38 m) | First, go to the chair near the window, then stop at... |
| livingroom_3 | nume | count=2 pipeline_gt=2, indep=10(referential), sg=9(scene_graph) | pipeline=2; independent=10; scene_graph=9 (disagreement) | How many photos are on the TV cabinet? |
| livingroom_3 | obje | IoU=n/a tgt=None(ambiguous/none) | no GT target matched; IoU undefined (flagged, not guessed) | Find the potted plant near the books on the cabinet. |
| livingroom_3 | obje | IoU=n/a tgt=None(none/none) | no GT target matched; IoU undefined (flagged, not guessed) | Find the vase between the cabinet and the stool. |
| livingroom_3 | inst | Frechet=4.29m cover1m=23% fit=1.81m (ours 18/gt 585) | scene frame fit residual 1.81 m > 1.0 m — endpoints/spawn did not co-locate under a single rigid transform; Frechet/coverage diagnostic only | Take the path near the TV and go to the pillow farth... |
| livingroom_3 | inst | Frechet=7.75m cover1m=16% fit=1.81m (ours 14/gt 548) | scene frame fit residual 1.81 m > 1.0 m — endpoints/spawn did not co-locate under a single rigid transform; Frechet/coverage diagnostic only | First, go near the stool, then take the path near th... |
| livingroom_4 | nume | count=7 pipeline_gt=7, indep=6(referential), sg=6(scene_graph) | pipeline=7; independent=6; scene_graph=6 (disagreement) | How many pillows are on a sofa? |
| livingroom_4 | obje | IoU=n/a tgt=None(none/none) | no GT target matched; IoU undefined (flagged, not guessed) | Find the picture closest to a window. |
| livingroom_4 | obje | IoU=n/a tgt=None(ambiguous/none) | no GT target matched; IoU undefined (flagged, not guessed) | Find the fossil decoration closest to the phone. |
| livingroom_4 | inst | Frechet=5.25m cover1m=17% fit=0.69m (ours 186/gt 521) | aligned via fitted scene transform (residual 0.69 m) | Go near the chair closest to the bookcase and stop a... |
| livingroom_4 | inst | Frechet=4.60m cover1m=37% fit=0.69m (ours 180/gt 889) | aligned via fitted scene transform (residual 0.69 m) | First, go near the fireplace, then go to the window ... |
| loft | nume | count=11 pipeline_gt=11, indep=9(referential_class_only), sg=8(scene_graph) | pipeline=11; independent=9; scene_graph=8 (disagreement) | How many black pillows are on the sofa? |
| loft | obje | IoU=n/a tgt=None(none/none) | no GT target matched; IoU undefined (flagged, not guessed) | The blue chair that is closest to the cup of coffee. |
| loft | obje | IoU=1.000 tgt=110(referential/relation) | target matched via relation | Find the potted plant between a vase and the cabinet... |
| loft | inst | Frechet=6.71m cover1m=9% fit=0.30m (ours 84/gt 739) | aligned via fitted scene transform (residual 0.30 m) | Go to the cup near the TV remote and avoid the path ... |
| loft | inst | Frechet=6.04m cover1m=27% fit=0.30m (ours 236/gt 771) | aligned via fitted scene transform (residual 0.30 m) | Go near the fireplace, pass by the stairs, then stop... |
| office_1 | nume | count=6 pipeline_gt=6, indep=6(referential), sg=12(scene_graph) | pipeline=6; independent=6; scene_graph=12 (disagreement) | How many computer monitors are on the table closest ... |
| office_1 | obje | IoU=n/a tgt=None(none/none) | no GT target matched; IoU undefined (flagged, not guessed) | Find the potted plant on the file cabinet. |
| office_1 | obje | IoU=n/a tgt=None(ambiguous/none) | no GT target matched; IoU undefined (flagged, not guessed) | Find the paper cup on the table closest to the proje... |
| office_1 | inst | Frechet=4.87m cover1m=47% fit=0.45m (ours 89/gt 1339) | aligned via fitted scene transform (residual 0.45 m) | Go to the potted plant furthest from the projector s... |
| office_1 | inst | Frechet=0.83m cover1m=100% fit=0.45m (ours 33/gt 458) | aligned via fitted scene transform (residual 0.45 m) | First, go near the potted plant on the shelf, then t... |
| office_2 | nume | count=1 pipeline_gt=1, indep=3(referential_class_only), sg=1(scene_graph) | pipeline=1; independent=3; scene_graph=1 (disagreement) | How many potted plants are on a table? |
| office_2 | obje | IoU=n/a tgt=None(none/none) | no GT target matched; IoU undefined (flagged, not guessed) | Find the computer monitor closest to the cabinet wit... |
| office_2 | obje | IoU=n/a tgt=None(none/none) | no GT target matched; IoU undefined (flagged, not guessed) | Find the box on the cabinet that is closest to the w... |
| office_2 | inst | Frechet=5.63m cover1m=13% fit=0.39m (ours 39/gt 694) | aligned via fitted scene transform (residual 0.39 m) | Go near the potted plant on the cabinet and stop at ... |
| office_2 | inst | Frechet=4.19m cover1m=27% fit=0.39m (ours 20/gt 1454) | aligned via fitted scene transform (residual 0.39 m) | First, go to the trash can near the cabinet, then go... |
| studio | nume | count=3 pipeline_gt=3, sg=1(scene_graph_class_only) | pipeline=3; scene_graph=1 (disagreement) | How many framed records are above the couch? |
| studio | obje | IoU=1.000 tgt=46(referential/relation) | target matched via relation | Find the vase closest to the guitar. |
| studio | obje | IoU=n/a tgt=None(ambiguous/none) | no GT target matched; IoU undefined (flagged, not guessed) | Find the beer bottle furthest from the couch. |
| studio | inst | Frechet=2.75m cover1m=72% fit=0.56m (ours 89/gt 752) | aligned via fitted scene transform (residual 0.56 m) | Go to the vases on the cabinet below the TV and stop... |
| studio | inst | Frechet=2.67m cover1m=51% fit=0.56m (ours 138/gt 566) | aligned via fitted scene transform (residual 0.56 m) | First, go to the vase closest to the easel, then, ta... |
