# Vision-checkpoint replay: qwen2.5vl:3b vs recorded panoramas (19 Jul 2026)

Replay of the CP2/CP3/CP5 vision checkpoints through the real
OpenAIChatAdapter vision path against panoramas from three recorded bags
(japanese_room_q1, office_1_q1, livingroom_1_tour), tiled exactly as live,
with GT from each scene's object_list.txt. Tool:
tools/llm_vision_checkpoint_replay.py; raw: vision_checkpoints.jsonl (36
cases, dense stride-4 run; case counts are capped by the constructors'
visibility/distance criteria, not the requested Ns; a first thin run is
preserved as vision_checkpoints.thin.jsonl). Ollama healthy at ~87 tok/s
throughout; GPU unwedged (P0).

| Checkpoint | n | Latency p50/p90 | Result | Verdict |
|---|---|---|---|---|
| CP2 4-tile miss recovery | 15 | 23.9 s / 25.1 s | answered "absent" on ALL cases incl. 9 true positives (0% hit) | **OFF** — blind and over the 20 s cap |
| CP3 anchor confirm | 18 | 1.4 s / 4.7 s | correct crops: 4 confirm / 5 reject (44%); wrong crops: 7 reject / 2 confirm (78%) — 61% aggregate | **OFF** — vetoes real anchors more often than it confirms them |
| CP5 frontier select | 3 | 2.3 s | 3/3 format-valid choices; no quality GT, n too small | **OFF** — insufficient evidence (rare checkpoint) |

## Enable matrix — local regime (qwen2.5vl:3b), authoritative

| Checkpoint | Local 3B | Basis |
|---|---|---|
| CP1 parse | **OFF** (regex floor) | parse_battery.md: floor 100% valid/qtype-correct; 3B systematically wrong on rule-6/rule-4 constructs |
| CP2 miss recovery | **OFF** | this report |
| CP3 anchor confirm | **OFF** | this report |
| CP4 verification (text) | **OFF** | same content-fabrication family as CP1 (#47-#49); untested live path not worth the risk given CP1/CP3 evidence |
| CP5 frontier select | **OFF** | this report |

Net: **the local 3B contributes nothing measurable to the live pipeline
today.** Its remaining roles: (1) permanent tier-3 ladder failover
(harmless — the ladder's floor still wins), (2) scaffolding for the August
cloud-model re-run of BOTH batteries (same tools, same wire). Phase-4
decision flagged: the +7.7 GB in-image bake currently buys insurance only —
revisit at the Docker Hub push whether to keep it (better small models may
exist by then) or slim the image.

Caveats per the generalization protocol: all three replay scenes were also
detector-development scenes; n is small and capped by case construction; a
cloud-model rerun should regenerate cases on scenes unused for tuning.
