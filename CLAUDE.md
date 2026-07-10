# CMU VLA Challenge 2026 — Workspace Instructions

Team repo for the CMU Vision-Language-Navigation Challenge 2026. Deadline: **15 Aug 2026** (submission), registration by **15 Jul 2026**.

## Session protocol (every session)

1. Read `LOG.md` (last entry = where we stopped) and `docs/master_plan.md` (current phase + checkboxes).
2. Do the work. Prefer delegating volume work to subagents; keep frontier-model tokens for design and review (see `docs/claude_budget.md`).
3. Before ending: append a dated entry to `LOG.md` (what was done, decisions made, next step), tick any completed checkboxes in `master_plan.md`, and commit.

## Layout

| Path | What |
|---|---|
| `docs/challenge_brief.md` | Challenge rules, deadlines, scoring, I/O contract |
| `docs/master_plan.md` | Phased plan with checkboxes — **the** source of truth for status |
| `docs/claude_budget.md` | Subscription/model-mix strategy (Fable window ends 12 Jul!) |
| `docs/windows_workplan.md` | What's doable on Windows now; core/adapter split design |
| `docs/soc_cluster_guide.md` | NUS SoC cluster: access, Slurm, GPUs, TODOs to verify |
| `docs/architecture.md` | System design (draft — harden before coding) |
| `docs/upstream_notes.md` | (to be written) distilled upstream repo internals |
| `docs/question_analysis.md` | (to be written) training-question taxonomy |
| `docs/prior_art.md` | (to be written) SORT3D / 2025 winners / OpenEQA |
| `LOG.md` | Running session log — append, never rewrite history |
| `upstream/` | Clone of the official challenge repo — **read-only reference, git-ignored**; never edit; `git pull` to refresh |
| `src/` | (to be created) our `ai_module` core + ros_adapter |

## Hard constraints

- **Environment**: user is on Windows until an Ubuntu reinstall (date TBD). Do not attempt to run ROS/sim locally; develop the OS-independent `core/` with mocks instead (`docs/windows_workplan.md`).
- **Only `ai_module/` may be modified** in the eventual challenge fork. Design everything to slot in there.
- **Test-time I/O contract** (only allowed topics) is in `docs/challenge_brief.md` — do not design around any other topic.
- Times in planning docs: user is in Singapore (SGT = UTC+8); challenge deadlines are AoE.
- NUS SoC cluster work goes over SSH (`xlogin.comp.nus.edu.sg`); check the TODO items in the cluster guide on first connect.
