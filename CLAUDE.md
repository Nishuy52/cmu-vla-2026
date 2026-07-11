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

## Session protocol (every session)

1. Read `LOG.md` (last entry = where we stopped) and `docs/master_plan.md` (current phase + checkboxes).
2. Do the work. Delegate execution to role subagents per the global orchestration policy. **Frontier-model (Fable-class) usage is restricted to planning, architecture, adjudication, and orchestration — never implementation.** All code/doc execution runs on standard executor/mech-executor tiers; verification on the verifier role. Never pass a frontier model override to an implementation agent (see `docs/claude_budget.md`).
3. Before ending: append a dated entry to `LOG.md` (what was done, decisions made, next step), tick any completed checkboxes in `master_plan.md`, **update `docs/ubuntu_setup.md` if anything changed that affects installing/running on the Ubuntu machine**, commit, and **push to `origin` (private backup: github.com/Nishuy52/cmu-vla-2026)**.
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
| `docs/organizer_playbook.md` | Organiser theses distilled + 2025 winner intel |
| `docs/proposals/` | Architecture debate record: 3 proposals + 3 critiques |
| `LOG.md` | Running session log — append, never rewrite history |
| `upstream/` | Clone of the official challenge repo — **read-only reference, git-ignored**; never edit; `git pull` to refresh |
| `src/` | Our `ai_module`: pure-Python `core/` (geometry, nav, fsm, parsing, perception, mocks — see `src/README.md`) + `ros_adapter/` (Phase 2). Run tests: `python -m pytest` from `src/` |

## Hard constraints

- **Environment**: user is on Windows until an Ubuntu reinstall (date TBD). Do not attempt to run ROS/sim locally; develop the OS-independent `core/` with mocks instead (`docs/windows_workplan.md`).
- **Only `ai_module/` may be modified** in the eventual challenge fork. Design everything to slot in there.
- **Test-time I/O contract** (only allowed topics) is in `docs/challenge_brief.md` — do not design around any other topic.
- Times in planning docs: user is in Singapore (SGT = UTC+8); challenge deadlines are AoE.
- NUS SoC cluster work goes over SSH (`xlogin.comp.nus.edu.sg`); check the TODO items in the cluster guide on first connect.
