# System Architecture — DRAFT v0.1

*10 Jul 2026. First-pass design to be hardened during the Fable window after the upstream deep-dive and question analysis land. Everything here is provisional.*

## Overview

```
                    ┌─────────────────────────────────────────────┐
 /challenge_question│                AI MODULE                    │
 ───────────────────►  Question Parser ──► Strategy Selector      │
                    │        │                   │                │
 360° cam (10Hz) ───►  Perception          Exploration Policy ────►/way_point_with_heading
 lidar (5Hz) ───────►  (detect+fuse)       (frontier over         │
 terrain map ───────►      │                terrain+odom)         │
 odometry ──────────►  3D Object Map ──► Scene Graph              │
                    │                        │                    │
                    │                  LLM Reasoning ─────────────►/numerical_response (Int32)
                    │                  (scene graph + question) ──►/selected_object_marker
                    └─────────────────────────────────────────────┘
```

SORT3D-style: perception builds an explicit 3D object map with attributes; an LLM does the spatial/linguistic reasoning over a text serialisation of it. This plays to the rules (online APIs allowed) and avoids end-to-end VLA training we don't have time/compute for.

## Components

1. **Question parser / strategy selector** — classify into numerical / object-ref / instruction-following (LLM call or regex-assisted); extract target objects, attributes, spatial constraints. Drives how much exploration is needed.
2. **Exploration policy** — frontier exploration over the terrain map until (a) question targets are found with sufficient confidence, or (b) time budget for exploration expires. The 10-min clock is the real adversary: budget ≈ explore-until-confident with a hard cutoff, answer with best available info. Early-answer bonus argues for confidence-triggered early stop.
3. **Perception** — open-vocabulary detector (GroundingDINO / YOLO-World / OWLv2) on the 360° image (possibly split into 4–6 pinhole crops); project lidar points into detections for 3D centroids + extents; track/merge across frames into a persistent object map with attributes (class, colour, size, position).
4. **Scene graph** — objects + pairwise spatial relations (left/right/near/between/on) computed geometrically, serialised compactly for the LLM.
5. **Reasoning** — per question type:
   - *Numerical*: LLM counts over scene graph with attribute/relation filters; sanity-check with deterministic filter code (LLM proposes the filter, code executes it — reduces hallucinated counts).
   - *Object ref*: LLM selects object ID → publish its 3D bbox as Marker; navigate to it.
   - *Instruction-following*: LLM decomposes into ordered spatial waypoint constraints → grounded to map coordinates → waypoint sequence respecting path constraints ("take the path near the window").
6. **ROS adapter** — thin node wiring topics ↔ core dataclasses. See `windows_workplan.md` for the core/adapter split.

## Open questions (resolve during Fable window)

- [ ] What exactly does the terrain map topic provide? (upstream deep-dive)
- [ ] Does the dummy ai_module reveal expected Marker format details (frame, scale semantics)?
- [ ] 360° image: equirectangular? How to best crop for pinhole detectors?
- [ ] Latency budget: API LLM calls at 1–5 s each — how many calls fit in 10 min alongside exploration?
- [ ] Is any fine-tuning worth it (referring-expression grounding on VLA-3D) vs pure zero-shot + prompting? Decide by ~20 Jul; cluster training only if clearly justified.
- [ ] What did 2025's winners do? (leaderboard/writeups)
