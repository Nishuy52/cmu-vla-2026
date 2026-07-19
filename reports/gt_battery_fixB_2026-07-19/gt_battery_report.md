# GT battery — REAL accuracy report (2026-07-19)

75 question(s) across 15 ground-truth scene(s): arabic_room, chinese_room, home_building_1, home_building_2, hotel_room_1, hotel_room_2, japanese_room, livingroom_1, livingroom_2, livingroom_3, livingroom_4, loft, office_1, office_2, studio. Missing/skipped scenes: none.

> **Yardstick note (numerical):** the PRIMARY yardstick is now `TRUE accuracy` — our count vs the human answer extracted from each scene's questions.pdf (`gt_answer_true`, source `questions_pdf_text`), matched by scene under a whitespace-insensitive question-text guard (a mismatch is left null, never mis-anchored). The `pipeline_gt` exact-match is DEMOTED to a self-consistency signal (our resolver over GT geometry — measures determinism, not truth). The `indep` column is a second opinion from the referential-statement annotations (distinct annotated target instances); `referential_class_only` means the count is relation-agnostic (coarser). Disagreements are the informative signal.

> **OBB->AABB note:** GT boxes are the axis-aligned hull of each object's oriented box (rotated corners, min/max), a strict over-approximation for non-axis-aligned objects — small IoU deficits on rotated targets are partly this, not localisation error.

> **IF headline is the rubric proxy (IF-F2):** the instruction-following headline is `rubric` — ordered per-leg arrival credit over the DRIVEN trajectory (a constant-speed kinematic follower over the planned breadcrumbs; a v1 simplification, no local-planner deviation / waypoint-snapping), minus one leg-equivalent penalty per corridor leg never threaded (`threading_check`) and per avoid capsule breached (`capsule_violated`). `legs=k/n` is ordered legs reached. Fréchet/coverage after `| diag:` are SECONDARY planned-path shape diagnostics only — never the headline (they measure shape similarity to the reference PLY, which the challenge rubric does not score).

> **Wall realism (IF-F2) — interior walls derived from `traversable_area.ply` (default on):** the VLA-3D region data ships only per-region AABBs (no door/passage geometry), so those can't source real walls — but each scene's `traversable_area.ply` (a floor-traversability point mesh in the sim/trajectory frame) can. Its coverage is rasterized to the mirror costmap's grid via the scene's fitted sim->object frame; grid cells inside the scene's outer boundary that the mesh does NOT cover (after a small dilation to bridge point-cloud sampling gaps) become OBSTACLE. A per-question row's note flags `interior walls unavailable` when the scene had no fitted frame or no `traversable_area.ply` to derive walls from — those fall back to the old boundary-only costmap. `--no-walls` reproduces the old behaviour unconditionally.

## Topline (per type)

- **Numerical** (n=15): TRUE accuracy 15/15 (answer key: questions.pdf); independent (referential) agreement 56% over 9 with strict evidence; scene-graph agreement 64% over 11; 3 question(s) had no independent evidence (class-only counts excluded from agreement).
- **Object reference** (n=30): instance-match 6/6 scored (scoreability 6/30); IoU pending real perception. Match method: none=24, relation=5, unique=1.
- **Instruction following** (n=30, scored=30): HEADLINE mean rubric-proxy score 0.150 (mean ordered-leg credit 0.161; 7 threading + 0 avoid violation(s) total). SECONDARY (diagnostic only): aligned planned-path mean Frechet 6.764 m, coverage-1m 38% over 28 aligned; 1 scene(s) unaligned: livingroom_3.


## Per-question

| Scene | Type | Metric(s) | Note | Question |
|---|---|---|---|---|
| arabic_room | nume | count=2 pipeline_gt=2, indep=3(referential_class_only), sg=3(scene_graph_class_only), ann_cov=3/3 | no independent evidence (relation-agnostic class-only counts only) | How many sofas are below a window? |
| arabic_room | obje | IoU=n/a tgt=None(ambiguous/none) | no GT target matched; IoU undefined (flagged, not guessed) | Find the pillow closest to the book on the stool. |
| arabic_room | obje | IoU=n/a tgt=None(none/none) | no GT target matched; IoU undefined (flagged, not guessed) | Find the wall lamp that is between a door frame and ... |
| arabic_room | inst | rubric=0.00 legs=0/2 thread_viol=0 avoid_viol=0 poses=47 | diag: Frechet=3.30m cover1m=39% |  | Go near the stool under the picture and stop at the ... |
| arabic_room | inst | rubric=0.00 legs=0/3 thread_viol=1 avoid_viol=0 poses=5 | diag: Frechet=3.46m cover1m=32% | 1 corridor leg(s) never threaded; leg 1: threading: trajectory never crossed the gate segment | First, go to the potted plant furthest from the hook... |
| chinese_room | nume | count=6 pipeline_gt=6, indep=1(referential), sg=6(scene_graph), ann_cov=6/6 | pipeline=6; independent=1; scene_graph=6 (disagreement) | Count the number of chairs with pillows on them. |
| chinese_room | obje | IoU=n/a tgt=None(none/none) | no GT target matched; IoU undefined (flagged, not guessed) | Find the bowl on the table closest to the folding sc... |
| chinese_room | obje | IoU=n/a tgt=None(none/none) | no GT target matched; IoU undefined (flagged, not guessed) | Find the pillow on the chair that is closest to the TV. |
| chinese_room | inst | rubric=0.50 legs=1/2 thread_viol=0 avoid_viol=0 poses=17 | diag: Frechet=6.26m cover1m=27% |  | Go near the potted plant on the table and stop at th... |
| chinese_room | inst | rubric=0.00 legs=0/2 thread_viol=0 avoid_viol=0 poses=4002 | diag: Frechet=4.27m cover1m=64% |  | First, go near the tea table with the elephant figur... |
| home_building_1 | nume | count=6 pipeline_gt=6, indep=18(referential_class_only), sg=11(scene_graph), ann_cov=18/18 | pipeline=6; scene_graph=11 (disagreement) | How many pillows are on the sofa under the pictures? |
| home_building_1 | obje | IoU=1.000 tgt=185(referential/relation) | target matched via relation | Find the clock on the TV cabinet. |
| home_building_1 | obje | IoU=n/a tgt=None(ambiguous/none) | no GT target matched; IoU undefined (flagged, not guessed) | Find the bowl closest to the knife rack near the tra... |
| home_building_1 | inst | rubric=0.00 legs=0/2 thread_viol=0 avoid_viol=0 poses=4002 | diag: Frechet=35.37m cover1m=8% |  | Go to the coffee table with the kettle on it and sto... |
| home_building_1 | inst | rubric=0.00 legs=0/3 thread_viol=1 avoid_viol=0 poses=102 | diag: Frechet=18.74m cover1m=4% | 1 corridor leg(s) never threaded; leg 1: threading: trajectory never crossed the gate segment | First, go to the nightstand with a clock on it, then... |
| home_building_2 | nume | count=2 pipeline_gt=2, indep=4(referential), sg=4(scene_graph), ann_cov=10/10 | pipeline=2; independent=4; scene_graph=4 (disagreement) | How many red pillows are on the sofa? |
| home_building_2 | obje | IoU=n/a tgt=None(none/none) | no GT target matched; IoU undefined (flagged, not guessed) | Find the lamp on the nightstand that has the photo o... |
| home_building_2 | obje | IoU=n/a tgt=None(none/none) | no GT target matched; IoU undefined (flagged, not guessed) | Find the speaker on the TV cabinet closest to the po... |
| home_building_2 | inst | rubric=0.00 legs=0/2 thread_viol=0 avoid_viol=0 poses=274 | diag: Frechet=19.33m cover1m=7% |  | Go near the magazine on the ottoman, then go to the ... |
| home_building_2 | inst | rubric=0.00 legs=0/3 thread_viol=1 avoid_viol=0 poses=274 | diag: Frechet=20.48m cover1m=4% | 1 corridor leg(s) never threaded; leg 0: threading: trajectory never crossed the gate segment | Take the path between the sofa and the coffee table ... |
| hotel_room_1 | nume | count=4 pipeline_gt=4, indep=4(referential), sg=4(scene_graph), ann_cov=4/4 |  | How many pillows are on the bed? |
| hotel_room_1 | obje | IoU=1.000 tgt=72(referential/relation) | target matched via relation | Find the bedside table farthest from the window. |
| hotel_room_1 | obje | IoU=n/a tgt=None(none/none) | no GT target matched; IoU undefined (flagged, not guessed) | Find the picture above the suitcase furthest from th... |
| hotel_room_1 | inst | rubric=0.50 legs=1/2 thread_viol=0 avoid_viol=0 poses=7 | diag: Frechet=3.17m cover1m=39% |  | Go to the bedside table closest to the window and st... |
| hotel_room_1 | inst | rubric=0.00 legs=1/3 thread_viol=1 avoid_viol=0 poses=7 | diag: Frechet=2.83m cover1m=41% | 1 corridor leg(s) never threaded; leg 1: threading: trajectory never crossed the gate segment | First, go near the bedside table closest to the benc... |
| hotel_room_2 | nume | count=3 pipeline_gt=3, indep=3(referential), sg=3(scene_graph_class_only), ann_cov=3/5 | annotation coverage 3/5 — class under-annotated, independent count deflated | How many pictures are above the bed? |
| hotel_room_2 | obje | IoU=1.000 tgt=38(unique_in_scene/unique) |  | Find the flowers near the window. |
| hotel_room_2 | obje | IoU=1.000 tgt=31(referential/relation) | target matched via relation | Find the picture closest to the bench. |
| hotel_room_2 | inst | rubric=0.00 legs=0/2 thread_viol=1 avoid_viol=0 poses=4 | diag: Frechet=0.61m cover1m=100% | 1 corridor leg(s) never threaded; leg 0: threading: trajectory never crossed the gate segment; interior walls unavailable (frame fit residual 0.97 m > 0.8 m wall-derivation gate); boundary-only costmap | Go between the bench and the bed and stop at the lam... |
| hotel_room_2 | inst | rubric=0.67 legs=2/3 thread_viol=0 avoid_viol=0 poses=17 | diag: Frechet=2.90m cover1m=60% | interior walls unavailable (frame fit residual 0.97 m > 0.8 m wall-derivation gate); boundary-only costmap | First, go to the picture closest to the door, then t... |
| japanese_room | nume | count=3 pipeline_gt=3, sg=3(scene_graph_class_only), ann_cov=0/3 | no independent evidence (relation-agnostic class-only counts only); annotation coverage 0/3 — class under-annotated, independent count deflated | How many calligraphy paintings are above the display... |
| japanese_room | obje | IoU=n/a tgt=None(none/none) | no GT target matched; IoU undefined (flagged, not guessed) | The lantern between the vase and the stone decoratio... |
| japanese_room | obje | IoU=n/a tgt=None(none/none) | no GT target matched; IoU undefined (flagged, not guessed) | The red pillow closest to the sushi. |
| japanese_room | inst | rubric=0.00 legs=0/2 thread_viol=0 avoid_viol=0 poses=71 | diag: Frechet=6.79m cover1m=19% |  | Go near the small table with a vase on it and then t... |
| japanese_room | inst | rubric=0.00 legs=0/3 thread_viol=0 avoid_viol=0 poses=126 | diag: Frechet=7.13m cover1m=27% |  | Go to the lantern closest to the fan decoration, the... |
| livingroom_1 | nume | count=8 pipeline_gt=8, indep=9(referential), sg=8(scene_graph), ann_cov=9/9 | pipeline=8; independent=9; scene_graph=8 (disagreement) | How many chairs are near the table with a vase on it? |
| livingroom_1 | obje | IoU=n/a tgt=None(ambiguous/none) | no GT target matched; IoU undefined (flagged, not guessed) | Find the vase on the cabinet below the picture. |
| livingroom_1 | obje | IoU=n/a tgt=None(ambiguous/none) | no GT target matched; IoU undefined (flagged, not guessed) | Find the pillow on the sofa that is closest to the w... |
| livingroom_1 | inst | rubric=0.50 legs=1/2 thread_viol=0 avoid_viol=0 poses=4002 | diag: Frechet=4.68m cover1m=12% | interior walls unavailable (frame fit residual 0.93 m > 0.8 m wall-derivation gate); boundary-only costmap | Go to the potted plant closest to the pyramid candle... |
| livingroom_1 | inst | rubric=0.00 legs=0/3 thread_viol=1 avoid_viol=0 poses=31 | diag: Frechet=1.58m cover1m=88% | 1 corridor leg(s) never threaded; leg 1: threading: trajectory never crossed the gate segment; interior walls unavailable (frame fit residual 0.93 m > 0.8 m wall-derivation gate); boundary-only costmap | First, go near the lamp closest to the black chair, ... |
| livingroom_2 | nume | count=2 pipeline_gt=2, indep=2(referential), sg=2(scene_graph), ann_cov=2/2 |  | How many cups are on the coffee table? |
| livingroom_2 | obje | IoU=n/a tgt=None(none/none) | no GT target matched; IoU undefined (flagged, not guessed) | Find the stool closest to the shelf near the TV cabi... |
| livingroom_2 | obje | IoU=n/a tgt=None(ambiguous/none) | no GT target matched; IoU undefined (flagged, not guessed) | Find the pillow on the sofa that is closest to the l... |
| livingroom_2 | inst | rubric=0.50 legs=1/2 thread_viol=0 avoid_viol=0 poses=47 | diag: Frechet=2.56m cover1m=66% |  | Go to the microwave on the kitchen counter and then ... |
| livingroom_2 | inst | rubric=0.00 legs=0/2 thread_viol=0 avoid_viol=0 poses=4002 | diag: Frechet=4.79m cover1m=36% |  | First, go to the chair near the window, then stop at... |
| livingroom_3 | nume | count=2 pipeline_gt=2, indep=10(referential), sg=9(scene_graph), ann_cov=10/9 | pipeline=2; independent=10; scene_graph=9 (disagreement) | How many photos are on the TV cabinet? |
| livingroom_3 | obje | IoU=n/a tgt=None(ambiguous/none) | no GT target matched; IoU undefined (flagged, not guessed) | Find the potted plant near the books on the cabinet. |
| livingroom_3 | obje | IoU=n/a tgt=None(none/none) | no GT target matched; IoU undefined (flagged, not guessed) | Find the vase between the cabinet and the stool. |
| livingroom_3 | inst | rubric=0.00 legs=0/2 thread_viol=0 avoid_viol=0 poses=68 | diag: Frechet=6.24m cover1m=30% | frame fit unaligned — DATA-CONFIRMED unfittable (meth-F11): GT q4/q5 terminal endpoints are 1.20 m apart but every pillow x bowl pairing is >= 3.27 m apart — a frame-independent distance contradiction (rigid two-point residual floor 1.04 m > 1.0 m gate). The q4 trajectory ends in the dining-chair corner (1.04 m from a chair, 1.33 m from the nearest pillow), not at any pillow; no rigid sim->object transform can map the endpoints onto the terminal objects.; interior walls unavailable (frame fit residual 1.81 m > 0.8 m wall-derivation gate); boundary-only costmap | Take the path near the TV and go to the pillow farth... |
| livingroom_3 | inst | rubric=0.00 legs=0/3 thread_viol=0 avoid_viol=0 poses=4002 | diag: Frechet=4.43m cover1m=42% | frame fit unaligned — DATA-CONFIRMED unfittable (meth-F11): GT q4/q5 terminal endpoints are 1.20 m apart but every pillow x bowl pairing is >= 3.27 m apart — a frame-independent distance contradiction (rigid two-point residual floor 1.04 m > 1.0 m gate). The q4 trajectory ends in the dining-chair corner (1.04 m from a chair, 1.33 m from the nearest pillow), not at any pillow; no rigid sim->object transform can map the endpoints onto the terminal objects.; interior walls unavailable (frame fit residual 1.81 m > 0.8 m wall-derivation gate); boundary-only costmap | First, go near the stool, then take the path near th... |
| livingroom_4 | nume | count=6 pipeline_gt=6, indep=6(referential), sg=6(scene_graph), ann_cov=6/6 |  | How many pillows are on a sofa? |
| livingroom_4 | obje | IoU=n/a tgt=None(none/none) | no GT target matched; IoU undefined (flagged, not guessed) | Find the picture closest to a window. |
| livingroom_4 | obje | IoU=n/a tgt=None(ambiguous/none) | no GT target matched; IoU undefined (flagged, not guessed) | Find the fossil decoration closest to the phone. |
| livingroom_4 | inst | rubric=0.00 legs=0/2 thread_viol=0 avoid_viol=0 poses=4002 | diag: Frechet=2.24m cover1m=51% |  | Go near the chair closest to the bookcase and stop a... |
| livingroom_4 | inst | rubric=0.33 legs=1/3 thread_viol=0 avoid_viol=0 poses=18 | diag: Frechet=3.30m cover1m=32% |  | First, go near the fireplace, then go to the window ... |
| loft | nume | count=2 pipeline_gt=2, indep=9(referential_class_only), sg=8(scene_graph), ann_cov=9/11 | pipeline=2; scene_graph=8 (disagreement); annotation coverage 9/11 — class under-annotated, independent count deflated | How many black pillows are on the sofa? |
| loft | obje | IoU=n/a tgt=None(none/none) | no GT target matched; IoU undefined (flagged, not guessed) | The blue chair that is closest to the cup of coffee. |
| loft | obje | IoU=1.000 tgt=110(referential/relation) | target matched via relation | Find the potted plant between a vase and the cabinet... |
| loft | inst | rubric=0.00 legs=0/1 thread_viol=0 avoid_viol=0 poses=2 | diag: Frechet=7.85m cover1m=9% |  | Go to the cup near the TV remote and avoid the path ... |
| loft | inst | rubric=0.00 legs=0/3 thread_viol=0 avoid_viol=0 poses=44 | diag: Frechet=7.26m cover1m=27% |  | Go near the fireplace, pass by the stairs, then stop... |
| office_1 | nume | count=6 pipeline_gt=6, indep=6(referential), sg=6(scene_graph), ann_cov=6/6 |  | How many computer monitors are on the table closest ... |
| office_1 | obje | IoU=n/a tgt=None(none/none) | no GT target matched; IoU undefined (flagged, not guessed) | Find the potted plant on the file cabinet. |
| office_1 | obje | IoU=n/a tgt=None(ambiguous/none) | no GT target matched; IoU undefined (flagged, not guessed) | Find the paper cup on the table closest to the proje... |
| office_1 | inst | rubric=0.50 legs=1/2 thread_viol=0 avoid_viol=0 poses=24 | diag: Frechet=4.87m cover1m=48% |  | Go to the potted plant furthest from the projector s... |
| office_1 | inst | rubric=0.33 legs=1/3 thread_viol=0 avoid_viol=0 poses=4002 | diag: Frechet=0.88m cover1m=100% |  | First, go near the potted plant on the shelf, then t... |
| office_2 | nume | count=1 pipeline_gt=1, indep=3(referential_class_only), sg=1(scene_graph), ann_cov=3/3 |  | How many potted plants are on a table? |
| office_2 | obje | IoU=n/a tgt=None(none/none) | no GT target matched; IoU undefined (flagged, not guessed) | Find the computer monitor closest to the cabinet wit... |
| office_2 | obje | IoU=n/a tgt=None(none/none) | no GT target matched; IoU undefined (flagged, not guessed) | Find the box on the cabinet that is closest to the w... |
| office_2 | inst | rubric=0.00 legs=0/2 thread_viol=0 avoid_viol=0 poses=49 | diag: Frechet=2.40m cover1m=46% |  | Go near the potted plant on the cabinet and stop at ... |
| office_2 | inst | rubric=0.67 legs=2/3 thread_viol=0 avoid_viol=0 poses=21 | diag: Frechet=5.24m cover1m=43% |  | First, go to the trash can near the cabinet, then go... |
| studio | nume | count=3 pipeline_gt=3, sg=1(scene_graph_class_only), ann_cov=0/3 | no independent evidence (relation-agnostic class-only counts only); annotation coverage 0/3 — class under-annotated, independent count deflated | How many framed records are above the couch? |
| studio | obje | IoU=1.000 tgt=46(referential/relation) | target matched via relation | Find the vase closest to the guitar. |
| studio | obje | IoU=n/a tgt=None(ambiguous/none) | no GT target matched; IoU undefined (flagged, not guessed) | Find the beer bottle furthest from the couch. |
| studio | inst | rubric=0.00 legs=0/2 thread_viol=0 avoid_viol=0 poses=46 | diag: Frechet=3.47m cover1m=14% |  | Go to the vases on the cabinet below the TV and stop... |
| studio | inst | rubric=0.00 legs=0/3 thread_viol=1 avoid_viol=0 poses=10 | diag: Frechet=3.65m cover1m=12% | 1 corridor leg(s) never threaded; leg 1: threading: trajectory never crossed the gate segment | First, go to the vase closest to the easel, then, ta... |


## Parse diagnostics

| Scene | Type | Question | Parse notes |
|---|---|---|---|

| arabic_room | nume | How many sofas are below a window? | indefinite article on 'window' (any matching instance qualifies) |
| arabic_room | obje | Find the wall lamp that is between a door frame and ... | indefinite article on 'door frame' (any matching instance qualifies); indefinite article on 'window' (any matching instance qualifies) |
| chinese_room | obje | Find the bowl on the table closest to the folding sc... | trailing superlative surfaced to head target |
| home_building_1 | inst | First, go to the nightstand with a clock on it, then... | indefinite article on 'clock' (any matching instance qualifies) |
| home_building_2 | obje | Find the speaker on the TV cabinet closest to the po... | trailing superlative surfaced to head target |
| hotel_room_1 | obje | Find the picture above the suitcase furthest from th... | trailing superlative surfaced to head target |
| japanese_room | inst | Go near the small table with a vase on it and then t... | indefinite article on 'vase' (any matching instance qualifies) |
| livingroom_1 | nume | How many chairs are near the table with a vase on it? | indefinite article on 'vase' (any matching instance qualifies) |
| livingroom_1 | inst | First, go near the lamp closest to the black chair, ... | indefinite article on 'picture' (any matching instance qualifies) |
| livingroom_4 | nume | How many pillows are on a sofa? | indefinite article on 'sofa' (any matching instance qualifies) |
| livingroom_4 | obje | Find the picture closest to a window. | indefinite article on 'window' (any matching instance qualifies) |
| loft | obje | Find the potted plant between a vase and the cabinet... | indefinite article on 'vase' (any matching instance qualifies); indefinite article on 'tv' (any matching instance qualifies) |
| office_1 | obje | Find the paper cup on the table closest to the proje... | trailing superlative surfaced to head target |
| office_2 | nume | How many potted plants are on a table? | indefinite article on 'table' (any matching instance qualifies) |
| office_2 | obje | Find the computer monitor closest to the cabinet wit... | indefinite article on 'phone' (any matching instance qualifies) |
