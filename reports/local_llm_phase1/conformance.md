# Local-LLM Phase 1 conformance (2026-07-18T17:10:52+00:00)

model=qwen2.5vl:7b base_url=http://localhost:11434/v1 temperature=0.0 n=10

**Note (added post-run):** the box's dGPU was later found wedged in a P8/210 MHz,
8.5 W software power-cap state (only cleared by a host reboot; not load-related).
The schema-validity result below (8/10 PASS) is unaffected — it depends on the
model's output, not on clock speed — but the per-question **latency figures may not
be representative** of the GPU's normal clocked performance and should not be used
for latency comparisons until confirmed post-reboot.

**Result: PASS** — 8/10 schema-valid via the LLM tier (bar: >= 80%), 3 repair round(s) used, 2 fell to the regex floor.
Mean LLM latency: 5.862 s.

| Scene | QType | Tier | Valid-via-LLM | Repair used | Floor | Calls | Latency (s) | Question |
|---|---|---|---|---|---|---|---|---|
| studio | object_reference | local | True | False | False | 1 | 3.247 | Find the beer bottle furthest from the couch. |
| livingroom_4 | object_reference | local | True | False | False | 1 | 2.523 | Find the fossil decoration closest to the phone. |
| hotel_room_2 | instruction_following | local | True | True | False | 2 | 14.426 | First, go to the picture closest to the door, then take t... |
| home_building_1 | object_reference | local | True | False | False | 1 | 2.418 | Find the clock on the TV cabinet. |
| livingroom_3 | object_reference | local | True | False | False | 1 | 3.085 | Find the vase between the cabinet and the stool. |
| office_1 | instruction_following | local | True | False | False | 1 | 5.593 | Go to the potted plant furthest from the projector screen... |
| livingroom_1 | object_reference | local | True | False | False | 1 | 4.066 | Find the pillow on the sofa that is closest to the windows. |
| arabic_room | object_reference | regex | False | True | True | 2 | 6.352 | Find the pillow closest to the book on the stool. |
| office_2 | instruction_following | local | True | False | False | 1 | 8.617 | First, go to the trash can near the cabinet, then go to t... |
| office_2 | object_reference | regex | False | True | True | 2 | 8.288 | Find the computer monitor closest to the cabinet with a p... |
