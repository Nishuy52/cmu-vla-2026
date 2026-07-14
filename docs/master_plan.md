# Master Plan — CMU VLA Challenge 2026

*Created 10 Jul 2026. Submission deadline 15 Aug 2026 (36 days). Update the checkboxes and Phase status as work lands; log every session in `LOG.md`.*

## Constraints driving the plan

1. **Fable 5 leaves Claude subscriptions 12 Jul 2026** → front-load all deep design/research work into the next 48h (see `claude_budget.md`).
2. **Registration closes 15 Jul** → user must register immediately.
3. **Stuck on Windows** until home → no native Ubuntu 24.04/ROS Jazzy; sim can't run locally yet (WSL2 attempt is a stretch goal). Maximise sim-independent work now (see `windows_workplan.md`).
4. **NUS SoC cluster** available for GPU training via SSH from Windows (see `soc_cluster_guide.md`).
5. 10-min-per-scene budget at eval → architecture must balance exploration time vs answering time.

## Phase 0 — Fable window (10–12 Jul) ★ CURRENT

Highest-leverage thinking while Fable is cheap. All outputs are docs/code in this repo.

- [x] Workspace + git init + docs skeleton (this commit)
- [x] **USER ACTION: register for the challenge** *(done 10 Jul — ahead of the 15 Jul deadline; subscription also upgraded to Max)*
- [x] Deep-dive the cloned upstream repo (`upstream/CMU-VLN-Challenge-2026`): dummy `ai_module` node, message flow, launch files, Docker setup → write `docs/upstream_notes.md` *(done 10 Jul incl. submodule: full topic contract, terrain XYZI semantics, TARE/FAR are OFF at test time → own exploration required, 14 implementer gotchas)*
- [x] Study `questions/` JSON for all 15 training scenes → question taxonomy + answer-format stats → `docs/question_analysis.md` *(done 10 Jul, independently verified: 75 Q / 255 pts, instruction-following = 70.6% of points, all-allocentric relations, ~zero attributes)*
- [x] Research prior art (SORT3D paper, 2025 leaderboard winners, OpenEQA baselines) → `docs/prior_art.md` *(done 10 Jul — SORT3D is the organisers' own system and integrates with our exact stack; zero-shot is competitive, fine-tuning demoted to optional)*
- [x] Write full system architecture → `docs/architecture.md` *(v1.0 merged 10 Jul from 3 independent proposals + cross-critique; full debate record in `docs/proposals/`)*
- [x] Scaffold `ai_module` core as OS-independent Python (mocked ROS interfaces) + unit tests *(done 10 Jul: full `src/core/` — contracts, mocks+synthetic scenes, geometry toolbox, nav/frontier planner, FSM/watchdog, parse ladder + 46 goldens; 263 tests green; adversarially verified, 2 defects found+fixed)*

## Phase 1 — Windows development (12 Jul – Ubuntu reinstall)

- [ ] Build the pure-Python core pipeline: perception → 3D object map → scene graph → LLM reasoning → 3 answer heads. Test against recorded/mock data on Windows.
- [x] Question-type classifier + parse ladder *(regex tier: 100% qtype on all 75 training questions; API/local tiers prompt-ready)* — per-type answer heads still to wire (resolve→verify→publish glue)
- [x] Exploration policy design (frontier-based, using terrain map + odometry) *(implemented in `core/nav/`: occupancy, frontier scoring w/ affinity hook, A* costmap with hard avoid-capsules, corridor pinch-threading, breadcrumbs, orientation sweep)*
- [ ] Mock evaluation harness: feed training-scene questions to the core, score offline
- [ ] **USER ACTION: manually download sample data** (browser: [Drive folder](https://drive.google.com/drive/folders/1xaatyLeIKLTh_oRzkyd7F1G6tkRPbFtm) → `data/sample_real_robot/`; `system_ros2.zip` + `data_view.rviz2`). Automated download fails (gdown 0-byte on all modes, 10 Jul) — needed for the replay harness, not blocking module development
- [ ] Stretch: WSL2 + Docker Desktop + GPU → try running the ROS Jazzy container headless; Unity sim rendering under WSLg is unproven
- [ ] Set up SoC cluster access (account, VPN, SSH, conda env) — works from Windows
- [ ] If fine-tuning is needed (e.g. grounding model on VLA-3D): prepare data + training scripts, run on cluster

## Phase 2 — Full sim loop (Ubuntu reinstall → ~5 Aug)

- [ ] Native Ubuntu 24.04 install, Docker + repo setup, run sim end-to-end
- [ ] Port the Windows-developed core into the real `ai_module` ROS node (thin adapter — designed for this from day one)
- [ ] Iterate on all 15 training scenes; measure score + wall-clock per question type
- [ ] Perception-difficulty recalibration: 2025 top-3 results
      leaned on the ground-truth /object_markers topic (absent in
      2026) - treat 2025 scores as non-comparable; prioritize
      detector recall + attribute extraction accordingly (see
      docs/prior_art/, architecture.md section 8).
- [ ] Tune the 10-minute budget: exploration cutoff, early-answer bonus strategy

## Phase 3 — Hardening + submission (5–15 Aug)

- [ ] Build + push Docker image; verify it runs exactly as eval will
- [ ] **Submit an early working version ASAP** (multiple submissions allowed, highest counts)
- [ ] Failure-mode sweep: ambiguous references, zero-count answers, unreachable waypoints
- [ ] Final submission before 15 Aug AoE

## Success criteria

- Minimum: a submitted, scoring system (beats the dummy)
- Target: competitive on instruction-following (6-point questions) — that's where ranking is won
- Stretch: top-tier → phase-2 real-robot invite
