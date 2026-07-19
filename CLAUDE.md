# CMU VLA Challenge 2026 — Workspace Instructions

Team repo for the CMU Vision-Language-Navigation Challenge 2026. Deadline: **15 Aug 2026** (submission; registered 10 Jul).

## Scope & isolation

- **This workspace is fully self-contained.** All project state lives here (docs/, LOG.md, src/); nothing about this project may be written to any other project's files or memory.
- If this session was started from a different working directory (e.g. the trading/Stonks workspace) and its instructions are in context: **those instructions do not apply here.** Trading rules, watchlists, and scan protocols are irrelevant to this project; do not let them shape behavior, and do not write VLA content into that workspace.
- Prefer starting sessions from this directory so only this file governs.

## Standing rules (given by the user; survive any session)

1. **Frontier-model usage is planning/orchestration only** — architecture, adjudication, review, delegation. All implementation (code, tests, docs, fixtures) runs on standard executor/mech-executor tiers; verification on the verifier role. Never pass a frontier-model override to an implementation agent.
2. **No AI/tooling attribution anywhere** — not in documents, not in commit messages, not in code comments.
3. **`docs/ubuntu_setup.md` must stay current** — any change affecting installation/deployment (dependency, model weight, env var, container) updates it in the same session.
4. **Commit and push (`origin` = github.com/Nishuy52/cmu-vla-2026, private) at every milestone and session end.**
5. **Host-first iteration (added 18 Jul 2026): run everything natively on
   this machine by default** — the host-node loop (`tools/run_host_node.sh`,
   ubuntu_setup §8b) for the adapter/perception, the host venv for core, the
   host Ollama for LLM work. Docker builds/runs ONLY at key checkpoints
   (gate exits, battery results to be trusted, §7a packaging gate,
   submissions) — rebuilds take too long for iteration. Host env pins must
   mirror the fork Dockerfile (single source of truth).
6. **Generalization protocol (added 19 Jul 2026): no tuning decision is
   trusted on training-sample evidence alone.** Scene-level holdout
   reporting, spec-over-sample precedence, and generated held-out
   questions for method comparisons — full rules in
   `docs/calibration.md` "Generalization protocol". Applies to every
   tunable, prompt, threshold, and A/B from here on.
7. **No-stall orchestration (added 19 Jul 2026 after repeated user callouts).**
   Never end a turn purely waiting: while any collector/agent runs, work the
   backlog (docs debt, cleanup, triage, next-step prep) or dispatch more.
   Never claim "no work left" without the mechanical sweep — `gh issue list
   --state open` cross-referenced against active agents' owned surfaces;
   every unowned open issue on a free surface is dispatchable NOW.
   Score-driving work (the IF chain, anything on the rubric) outranks
   ceremony (gates, evidence passes) in dispatch order — ceremony rides
   triggers. An agent that reports "waiting/pausing" is a STOPPED agent:
   take over its wait condition or re-message it with a finish-now
   directive; briefs must forbid agents from ending turns in a wait.
   Deadline every wait with an expected-arrival time and probe immediately
   when overdue.
8. **Detected issues become GitHub issues.** Any defect, failure, or
   open problem detected (by review, verification, battery/diagnosis
   runs, or agents) that is not fixed in the same session gets filed
   as a GitHub issue on `origin` (`gh issue create`) — one issue per
   defect, with file/line refs and reproduction context — so PRs can
   be opened against them and close them (`Fixes #N`). Known-defect
   lists in docs/reports still get written, but the issue tracker is
   the actionable queue.

## Session protocol (every session)

1. Read `LOG.md` (last entry = where we stopped) and `docs/master_plan.md` (current phase + checkboxes).
2. Do the work. Delegate execution to role subagents per the global orchestration policy. **Frontier-model (Fable-class) usage is restricted to planning, architecture, adjudication, and orchestration — never implementation.** All code/doc execution runs on standard executor/mech-executor tiers; verification on the verifier role. Never pass a frontier model override to an implementation agent (see `docs/claude_budget.md`). While iterating on `src/`, run the fast test tier (`pytest` from `src/`, ~40 s); it skips the slow full-controller sims and batteries.
3. Before ending: append a dated entry to `LOG.md` (what was done, decisions made, next step), tick any completed checkboxes in `master_plan.md`, **update `docs/ubuntu_setup.md` if anything changed that affects installing/running on the Ubuntu machine**, run the full test gate (`pytest -m ""` from `src/`, or `pytest -m "" -n auto` with the `dev` extra) before a milestone commit, commit, and **push to `origin` (private backup: github.com/Nishuy52/cmu-vla-2026)**.
4. Documents and commits carry no AI/tooling attribution.

## Layout

| Path | What |
|---|---|
| `docs/challenge_brief.md` | Challenge rules, deadlines, scoring, I/O contract |
| `docs/master_plan.md` | Phased plan with checkboxes — **the** source of truth for status |
| `docs/claude_budget.md` | Subscription/model-mix strategy (Fable window ends 12 Jul!) |
| `docs/windows_workplan.md` | What's doable on Windows now; core/adapter split design |
| `docs/soc_cluster_guide.md` | NUS SoC cluster: access, Slurm, GPUs, TODOs to verify |
| `docs/architecture.md` | System design v1.0 — merged from the 3-proposal debate (`docs/proposals/`), adjudication table + citations |
| `docs/ubuntu_setup.md` | **Ordered install guide for the Phase-2 Ubuntu machine** — OS → NVIDIA → Docker → challenge stack → our module. MUST be updated whenever any change affects installation/deployment (new dependency, model weight, env var, container change) |
| `docs/upstream_notes.md` | Distilled upstream repo internals (topic contract, launch mechanics, gotchas) |
| `docs/sim_verification.md` | "Is everything working?" runbook — three-tier ladder (Windows-now → WSL2 → native Ubuntu sim) with exact command + PASS/FAIL fix per step, and a known-good-state checklist |
| `docs/phase2_playbook.md` | **THE Ubuntu order-of-operations** — Gates 0–5 from fresh install to first scored dry-run, then the calibration loop to the Aug 3 MVS submission. Start here on Ubuntu day one |
| `docs/question_analysis.md` | Training-question taxonomy — verified stats (75 Q / 255 pts; IF = 70.6%) |
| `docs/prior_art.md` | Published-work survey: SORT3D blueprint, VLN/EQA literature, reading list |
| `docs/prior_art/` | Per-source deep-dive dossiers (imported 11 Jul from a parallel research stream): full 2025 leaderboard + per-team method dossiers from primary sources, SORT3D exact thresholds, VLA-3D question-generation spec. `README.md` inside is the index and carries the provenance/conflict rules |
| `docs/competitor_forks.md` | Dev-kit fork-network snapshot - who else is building for 2026, as of 11 Jul |
| `docs/io_contract_crosscheck.md` | Independent source-inspection of the dev-kit I/O contract - corroborates `challenge_brief.md`/`upstream_notes.md` (which stay authoritative) |
| `docs/organizer_playbook.md` | Organiser theses distilled + 2025 winner intel |
| `docs/proposals/` | Architecture debate record: 3 proposals + 3 critiques |
| `LOG.md` | Running session log — append, never rewrite history |
| `docs/tasks/` | Task records - one folder per task + `INDEX.md`; see "Task-record workflow" below |
| `upstream/` | Clone of the official challenge repo — **read-only reference, git-ignored**; never edit; `git pull` to refresh |
| `src/` | Our `ai_module`: pure-Python `core/` (geometry, nav, fsm, parsing, perception, mocks — see `src/README.md`) + `ros_adapter/` (Phase 2). Run tests: `python -m pytest` from `src/` |
| `tools/` | Offline dev/debug tools — **outside the challenge-fork surface**, never part of the scored pipeline. `colored_cloud.py`: colored point-cloud reconstruction from replay data → PLY (`python -m tools.colored_cloud extract <bag_or_fixtures> <out.ply>`). Tests: `python -m pytest tools` from repo root (not collected by the `src/` suite) |

## Task-record workflow (adapter, added 11 Jul 2026)

Non-trivial work is tracked as task folders (a convention that
arrived with the parallel research stream). In this repo the
workflow is adapted to fit the existing conventions instead of
replacing them:

- Task records live under `docs/tasks/T<N>-<slug>/` (committed, so
  every agent and teammate can read them). `docs/tasks/INDEX.md` is
  the one-line-per-task index; next task number = highest existing
  N + 1.
- Each task folder's `task.md` is the detailed record (intent,
  context, acceptance criteria, todo, dated notes). Detailed
  progress logging for a task goes THERE, not in `LOG.md`.
- `LOG.md` gets ONLY minimal one-liner entries at task milestones -
  started / finished / wrapped up a task, with a pointer to the
  task folder - never detailed task logs. Session entries unrelated
  to task records keep the existing format.
- Imported task folders (T1-T4) predate this repo merge and are
  kept verbatim as historical records: path references inside them
  use the pre-merge research-workspace layout (e.g.
  `docs/prior-art.md` there = `docs/prior_art/README.md` here,
  `docs/io-contract.md` = `docs/io_contract_crosscheck.md`).

## Hard constraints

- **Environment**: user is on Windows until an Ubuntu reinstall (date TBD). Do not attempt to run ROS/sim locally; develop the OS-independent `core/` with mocks instead (`docs/windows_workplan.md`).
- **Only `ai_module/` may be modified** in the eventual challenge fork. Design everything to slot in there.
- **Test-time I/O contract** (only allowed topics) is in `docs/challenge_brief.md` — do not design around any other topic.
- Times in planning docs: user is in Singapore (SGT = UTC+8); challenge deadlines are AoE.
- NUS SoC cluster work goes over SSH (`xlogin.comp.nus.edu.sg`); check the TODO items in the cluster guide on first connect.
