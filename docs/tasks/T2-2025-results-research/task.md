# Task: Identify CMU VLA Challenge 2025 top-1/top-2 teams + methods

## Intent
Close the highest-value gap from the prior-art pass: who placed
1st and 2nd in the 2025 CMU VLA Challenge and what their technical
methods were (detector, map/scene-graph representation, reasoning
LLM, exploration strategy, repo if any). Append verified findings
to `docs/prior-art.md` section 1 without overwriting content.

## Context
- `docs/prior-art.md` section 1 + section 5 record the gap:
  3rd place "CopyPasta" (CMU MRSD: Paruchuri, Gupta, Singh,
  Adhar) confirmed via newsletter blurb only; top-1/top-2 unknown.
- Leaderboard page ai-meets-autonomy.com/cmu-vla-challenge is a
  JS SPA that plain WebFetch could not render - needs workaround
  (archive.org, underlying JSON/JS data, cache).
- `docs/competitor-forks.md` already surveyed the 2026 dev-kit
  fork network (Yuxin916/CMU-VLN-Challenge-2026), VLA-3D, and
  jizhang-cmu/cmu_vla_challenge_unity - but NOT the 2025 repo
  HaochenZ11/CMU-VLA-Challenge fork network.
- Known pointers: Anand Singh talk youtu.be/-bIMTsnbuoY (2024
  method), Jana et al. arXiv 2606.31144 (already captured).
- Output style: hard-wrap ~65 cols, "-" not em dashes, cite every
  URL, be explicit about what remains unverified.

## Acceptance Criteria
- [ ] Fork network of HaochenZ11/CMU-VLA-Challenge surveyed via
      gh CLI (forks, branches, ai_module diffs).
- [ ] Leaderboard SPA workaround attempted (archive, JSON, cache).
- [ ] IROS 2024/2025 workshop pages + YouTube talks mined.
- [ ] arXiv/Scholar swept beyond 2606.31144.
- [ ] Findings appended to docs/prior-art.md section 1, existing
      content preserved, every claim cited, unverified items
      flagged.

## Artifacts
- Docs: docs/ (agent raw findings if needed)

## Stage Gates
| Gate | State | Evidence / notes |
| --- | --- | --- |
| Task | APPROVED | Fully specified in the task prompt 2026-07-11 |
| Spec | APPROVED | N/A - research task, no design ambiguity |
| Plan | APPROVED | Attack angles enumerated in the task prompt |

## Todo
- [x] Survey HaochenZ11/CMU-VLA-Challenge fork network (gh CLI)
- [x] Break into leaderboard SPA (archive/JSON/cache workaround)
- [x] Mine IROS workshop pages + YouTube talk metadata/transcripts
- [x] Sweep arXiv/Scholar for further CMU VLA Challenge entries
- [x] Sweep CMU RI news / lab blogs / LinkedIn
- [x] Synthesize + verify; append to docs/prior-art.md section 1

## Todo (stage 2, added 2026-07-11)
- [x] Leaderboard resolved via a human browser screenshot (saved
      to docs/assets/cmu-vla-2025-leaderboard.png)
- [x] Friction log written (docs/research-friction-log.md)
- [x] 15 per-source dossiers written to docs/prior-art/
- [x] Targeted winner-aware pass: NROS Chinese-web deep-dive
      (HITSZ article recovered; members + techniques), ReasonX
      identity hunt (Cai Yuxin, NTU; merged into dossier),
      /watch on Anand talk (android-client bypass; frames read)
- [x] Dossier index added to docs/prior-art.md
- [x] NROS member-name footprint hunt: Liu Xiao = liuxiao916
      (first-party win confirmation on her homepage); no method
      paper/code exists anywhere public; fork channel exhausted

## Notes
- 2026-07-11: task created from a fully-specified prompt;
  running autonomously, no intake round needed.
- 2026-07-11 (stage 2): leaderboard resolved - 1st NROS 44.26,
  2nd ReasonX 34.58, 3rd CopyPasta 30.98, 4th URL-KAIST 22.80.
  Biggest strategic finding: GT semantics were legal in 2025
  and most entries depended on them; 2026 bans them, so 2025
  pipelines do not port. VLA-3D generation code = question
  spec (exact thresholds + 15-color vocab).
- 2026-07-11: DONE. 1st place = NROS Lab, HIT Shenzhen
  (self-reported, single source, both tracks). 2nd place =
  verified negative across EN/CN/KR web, GitHub, arXiv.
  3rd place (CopyPasta) upgraded: repo (parths5) + fully
  captioned method talk (Gemini 2.5 Pro, 4-level state
  machine, GT markers for T1/T2, own scene graph for T3).
  Six other 2025 team repos identified with methods; strongest
  unplaced entry = url-kaist (KAIST URL; GPT-4o + YOLO-World +
  scene graph + frontier exploration) - top-2 hypothesis
  checked and unsupported. Leaderboard image on Wayback is
  hotlink-blocked; human-browser follow-up needed. Findings
  appended to docs/prior-art.md section 1 + section 5 pointer +
  source index.
