# Task: Preliminary discussion on the CMU VLN Challenge problem

## Intent

Before any design or implementation work starts, align as a team
on what the CMU Vision-Language-Navigation Challenge actually asks
us to build. This task is the prep material for that discussion -
problem statement, worked examples, and the parts of the
competition process that shape the solution (simulation, I/O
format, evaluation). Explicitly excludes logistics (venue, dates,
deadlines, prizes) - those don't affect the technical approach.

## Context

Source: https://www.ai-meets-autonomy.com/cmu-vln-challenge

Full problem statement, worked examples (numerical / object
reference / instruction-following), and competition-process
specifics (Unity sim, I/O format, environments, evaluation,
VLA-3D dataset) are written up in repo-root
`docs/problem-brief.md` (durable project doc, not task-local).
Open discussion questions for the team to resolve are in this
task's `docs/problem-brief.md`.

Repo is a fresh, empty project (no git history, no prior
decisions) - this is the first task.

## Acceptance Criteria

- [x] Problem statement, examples, and competition-process
      specifics are captured in one place, accurate to the source
      page, with logistics deliberately excluded.
- [x] Open discussion questions are listed so the team knows what
      to resolve in the preliminary discussion.

## Artifacts

- Spec: n/a (no design ambiguity yet - discussion prep only)
- Plan: n/a
- Questions: questions.md
- Docs: docs/problem-brief.md (discussion questions); repo-root
  docs/problem-brief.md (durable problem brief)
- Review: n/a

## Stage Gates

| Gate | State | Evidence / notes |
| --- | --- | --- |
| Task | PENDING | drafted this turn, awaiting approval |
| Spec | N/A | not needed - no implementation being scoped yet |
| Plan | N/A | not needed - no implementation being scoped yet |

## Todo

- [x] Fetch and extract problem statement, examples, and
      competition-process specifics from the challenge page
- [x] Draft discussion questions to resolve as a team
- [ ] Hold the preliminary discussion and record decisions in
      Notes below (or spin off a design task if scope grows)
- [x] Phase 0a: find rules + dev kit, answer constraints list
      (captured in repo-root docs/challenge-rules.md)
- [~] Phase 0b: dev kit cloned + source-inspected; exact I/O
      contract, question-type signaling, and scoring distribution
      captured in docs/io-contract.md. Live end-to-end sim run
      DEFERRED - needs a Linux + ROS Jazzy + Nvidia GPU box
      (can't run on macOS).
- [x] Phase 1: prior-art pass done -> docs/prior-art.md.
      Scene-graph+LLM baseline validated by prior art (SORT3D,
      ConceptGraphs, SayNav, + one confirmed challenge entry Jana
      et al. 2606.31144). Two load-bearing citations spot-verified.
      Residual gap: 2025 top-1/top-2 identities (results page is a
      JS SPA - needs a human browser + workshop videos).

## Notes

- 2026-07-11: Migrated from an ad-hoc root-level `TASKS.md` into
  the task-workflow structure as requested.
- 2026-07-11: Advisor session on an initial thought dump (router
  agents, JEPA, synthetic data). Roadmap + parked-ideas
  evaluation captured in docs/approach-roadmap.md; answered
  questions logged in questions.md.
- 2026-07-11: Phase 0a done. Dev kit found
  (Yuxin916/CMU-VLA-Challenge-2026); rules captured in repo-root
  docs/challenge-rules.md. Key resolutions: any model/API allowed
  (network risk ours), 10 min/question, scene resets between
  questions, RTX 4090 eval box, ROS Jazzy topics fixed.
- 2026-07-11: Walked I/O model (live streaming sensors, no pre-
  built map, three gated output topics) and JEPA problem-shape fit
  end to end in discussion; logged as questions.md Q3/Q4. Flagged one
  open gap for Phase 0b: how question-type + exploration-done
  timing is actually signaled by the dev kit.
- 2026-07-11: Phase 0b (inspection half) done. Cloned dev kit,
  read dummy_vlm source + docker + questions.json. Closed the Q3
  gap: NO question-type label (dummy infers from string prefix;
  brittle - real questions like "The lantern..." break it), NO
  explicit exploration-done signal (just publish; instruction
  waypoints stream closed-loop within 1.0 m). Corrected scoring:
  every scene is 1 num / 2 obj / 2 instr -> 51 max (not 45) if
  test mirrors dev. Full contract in docs/io-contract.md. Live
  sim run deferred (needs Linux + GPU). Logged as Q6.
- 2026-07-11: Launched Phase 1 prior-art subagent (opus,
  background) -> docs/prior-art.md; runs in parallel, no file
  overlap with Phase 0b.
- 2026-07-11: Phase 1 landed. Key result: the "boring baseline"
  (detect -> object-centric 3D map -> LLM/VLM reasons -> emit) is
  the going approach - SORT3D (organizer-referenced, same rig),
  ConceptGraphs, SayNav, LLM-Grounder are all zero-shot LLM-over-
  map; one confirmed challenge entry (Jana et al.) uses OWL-ViT +
  semantic voxel map + Gemini 2.0 Flash router. Design levers:
  instruction-following is 6/9 of a scene's non-numerical points
  and hardest (needs traversability, closer to SayNav); mirror
  VLA-3D's relation set (between/closest/near) + per-object color;
  spatial relations via heuristic toolbox the LLM calls, not LLM-
  over-raw-coords. Spot-checked 2 citations - real. Only real gap:
  2025 top-1/top-2 methods (JS SPA page, needs human browser).
- 2026-07-11: Surveyed the dev-kit fork network (`gh api`) for
  competing-team activity -> docs/competitor-forks.md. 6/8 forks
  untouched; 2 active (jainanshu0912, kante2), both building the
  same scene-graph + VLM pipeline as our prior-art baseline pick -
  validates docs/approach-roadmap.md, no course change.
