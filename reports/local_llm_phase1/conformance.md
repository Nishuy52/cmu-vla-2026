# Local-LLM Phase 1 conformance (2026-07-18T16:11:04+00:00)

model=qwen2.5vl:7b base_url=http://localhost:11434/v1 temperature=0.0 n=10

**Result: FAIL** — 6/10 schema-valid via the LLM tier (bar: >= 80%), 5 repair round(s) used, 4 fell to the regex floor.
Mean LLM latency: 14.751 s.

| Scene | QType | Tier | Valid-via-LLM | Repair used | Floor | Calls | Latency (s) | Question |
|---|---|---|---|---|---|---|---|---|
| studio | object_reference | regex | False | True | True | 2 | 7.883 | Find the beer bottle furthest from the couch. |
| livingroom_4 | object_reference | local | True | False | False | 1 | 2.526 | Find the fossil decoration closest to the phone. |
| hotel_room_2 | instruction_following | local | True | False | False | 1 | 7.226 | First, go to the picture closest to the door, then take t... |
| home_building_1 | object_reference | local | True | False | False | 1 | 2.444 | Find the clock on the TV cabinet. |
| livingroom_3 | object_reference | local | True | False | False | 1 | 3.259 | Find the vase between the cabinet and the stool. |
| office_1 | instruction_following | local | True | True | False | 2 | 12.319 | Go to the potted plant furthest from the projector screen... |
| livingroom_1 | object_reference | local | True | False | False | 1 | 4.566 | Find the pillow on the sofa that is closest to the windows. |
| arabic_room | object_reference | regex | False | True | True | 2 | 7.268 | Find the pillow closest to the book on the stool. |
| office_2 | instruction_following | regex | False | True | True | 2 | 50.008 | First, go to the trash can near the cabinet, then go to t... |
| office_2 | object_reference | regex | False | True | True | 2 | 50.01 | Find the computer monitor closest to the cabinet with a p... |
