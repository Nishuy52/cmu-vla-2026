# Session Log

Append-only. Newest entry last. Format: date (SGT), what was done, decisions, next step.

---

## 2026-07-10 — Workspace bootstrap

**Done:**
- Researched challenge (repo + official site) → `docs/challenge_brief.md`
- Researched Claude plan economics → `docs/claude_budget.md`. Critical: **Fable 5 leaves subscriptions 12 Jul**; recommended Max 5x upgrade before then, downgrade after 15 Aug.
- Researched NUS SoC cluster → `docs/soc_cluster_guide.md` (access steps verified from public guides; storage/partition TODOs flagged)
- Wrote `docs/master_plan.md` (4 phases), `docs/windows_workplan.md` (core/adapter split so ~70% of dev happens on Windows), `docs/architecture.md` (draft v0.1, SORT3D-style)
- Cloned official repo → `upstream/CMU-VLN-Challenge-2026` (git-ignored reference)
- git init + first commit

**Decisions:**
- SORT3D-style architecture (explicit 3D object map + LLM reasoning via API — allowed by rules) over end-to-end VLA training
- `ai_module` split: pure-Python `core/` (Windows-testable) + thin `ros_adapter/`
- Cluster reserved for optional fine-tuning; sim work waits for native Ubuntu (WSL2 = time-boxed stretch)

**Next step (Phase 0, Fable window — before 12 Jul):**
1. USER: register for the challenge (deadline 15 Jul) + decide on Max upgrade
2. Upstream deep-dive → `docs/upstream_notes.md`
3. Question JSON analysis → `docs/question_analysis.md`
4. Prior-art research → `docs/prior_art.md`
5. Harden `architecture.md`, then scaffold `src/core/`

---

## 2026-07-10 (session 2) — Prior-art deep research

**Done:**
- Deep research across published work → `docs/prior_art.md` (SORT3D method + numbers, Transcrib3D, VLA-3D/IRef-VLA, VLFM, 3D-Mem, OpenEQA, ConceptGraphs, Open-Nav, sequential grounding, perception-stack consensus table, reading list)
- Architecture bumped to v0.2 with prior-art validation note
- Upstream repo finished cloning → `upstream/CMU-VLN-Challenge-2026` ✓

**Key findings:**
- **SORT3D (arXiv:2504.18684) is the challenge organisers' own zero-shot system, already integrated with our exact autonomy stack + Unity sim.** It is the blueprint; no license in repo — concepts only until checked.
- Zero-shot object-map + LLM beats supervised 3D models → **fine-tuning demoted to optional**; SoC cluster no longer critical-path.
- Literature gaps we can exploit: 10-min time-budgeted exploration, deterministic count-filters (numericals), multi-constraint verification, self-consistency on 6-pt questions.
- 2025 3rd place was a CMU MRSD team; no public technical reports found — recheck IROS-25 workshop proceedings later.

**Next step:**
1. USER: register (by 15 Jul!) + Max decision
2. Read the two CMU MSR theses (links in prior_art.md reading list)
3. Upstream deep-dive → `docs/upstream_notes.md` (clone is now local)
4. Question JSON analysis → `docs/question_analysis.md`
5. Then scaffold `src/core/`

---

## 2026-07-10 (session 3) — Delegated deep-dives, verified

Work fanned out to role agents; results independently verified where fact-heavy.

**Done:**
- `docs/upstream_notes.md` — full upstream distillation incl. autonomy-stack submodule (fetched at pinned `81035e9` after fixing a broken shallow checkout). Highlights: relaunch-per-question, 1 Hz question republish, Marker scored by GT overlap + its center used as nav goal, Pose2D heading ignored (radians if ever enabled), terrain map = XYZI cloud with intensity = obstacle height (m), traversable < ~0.1–0.2 m, waypointConverter snaps infeasible waypoints, **TARE/FAR planners NOT launched at test time and their triggers aren't allowed topics → we implement our own exploration**, CycloneDDS + Unity TCP bridge on :10000, keep waypoints near the vehicle.
- `docs/question_analysis.md` — 75 Q / 255 pts; instruction-following **70.6% of points**; relations: on(48), closest-to(37), near(33), between(17); zero egocentric phrasing; only 10 attribute mentions; no GT answers in JSON (only trajectory PLYs); "refridgerator" typo. **All claims independently re-computed and CONFIRMED by a fresh-context verification pass.**
- `docs/organizer_playbook.md` — organiser theses distilled (toolbox-over-raw-LLM-geometry, +11.6-pt captioning gain, fisheye-VO failure warning → lidar for geometry / camera for semantics). 2025 3rd place: CMU MRSD "CopyPasta" (Gemini 2.5 Pro + ROS state machine); no public leaderboard/scores found (challenge site 404s, IROS-25 workshop not archived).

**Decisions:**
- Exploration is on us (built-in planners off at test time) — actuate only via `/way_point_with_heading`.
- Documentation and commits carry no tooling attribution (user rule, applies from session 3 onward).

**Next step:** independent architecture proposals by parallel frontier-model agents → structured critique → merged architecture v1.0 with full citations (plagiarism-safe: no unlicensed code reuse, concepts attributed).

---

## 2026-07-10 (session 4) — Architecture debate & merge

**Done:**
- Three independent architecture proposals written by parallel frontier-model agents, each from an opposed stance, none allowed to read the prior draft: `proposal_A` (deterministic-first), `proposal_B` (VLM-agentic), `proposal_C` (expected-score-first). Committed `c36ad8a`.
- Structured cross-critique round: each author attacked the rivals' best versions, listed steals, non-negotiables, concessions (`critique_{A,B,C}.md`).
- **Merged into `docs/architecture.md` v1.0** with an explicit adjudication table (10 contested points, resolution + rationale each).

**Key merge decisions:**
- Checkpointed reasoning: LLM at ~3–6 event-triggered checkpoints per question (parse, detector-miss recovery, IF anchor confirmation, pre-answer verification) — never in the actuation loop. Resolves A's no-recovery weakness without B's per-step API variance.
- Corridor/avoid geometry lives inside the MVS (unanimous post-debate); A's capsule-shrink recovery rejected as self-defeating.
- Lidar-only Marker extents (trimmed percentile AABB + dimension sanity table); camera is semantics-only.
- Answer path survives a dark network: dual-API → local quantised VLM in the image → regex tier; watchdog floor answer at T−30 s on every type ("silence is the only unforgivable failure").
- C's expected-points ledger + Aug 3 MVS gate govern the build; named cut order (never cut: watchdog, toolbox, corridor geometry, parse ladder).

**Attribution hygiene:** all borrowed concepts cited inline in every proposal + §9 of architecture.md; SORT3D repo is unlicensed → zero code reuse, clean-room implementations only; documents carry no tooling attribution.

**Next step (Phase 0 wrap / Phase 1 start):**
1. USER: register (by 15 Jul!) — last reminder before it's overdue
2. Scaffold `src/core/` to the v1.0 module map (interfaces + tests first)
3. Download training scenes + sample bags for the Windows replay harness
4. Set up SoC cluster access (optional now — fine-tuning was demoted)

---

## 2026-07-10 (session 5) — Core implementation sprint

Orchestrated build: frozen contracts written in the main session; five modules implemented by parallel role executors; independent adversarial verification; fixes; docs.

**Done (263 tests green, `python -m pytest` from `src/`):**
- `core/interfaces.py` + `core/plan_schema.py` — frozen contracts (RobotIO protocol, closed-predicate plan DSL with validation)
- `core/mocks/` + `core/perception/scene_index.py` — synthetic scene generator, MockRobotIO, typo/synonym-tolerant instance index with trimmed-AABB merge (45 tests)
- `core/geometry/` — full predicate toolbox with explanations + margins, resolve/fallback ladder, corridor gate + threading check, avoid capsules; thresholds centralised in one calibration dataclass (57 tests)
- `core/nav/` — occupancy from terrain XYZI, frontier scoring w/ affinity hook, A* (UNKNOWN at 3×), hard capsule stamps + nearest-reachable fallback (never relaxes), corridor pinch forcing, ≤2.5 m breadcrumbs, diamond orientation sweep (52 tests)
- `core/fsm/` — budget gates, checkpoint CallLedger (45 s floor reserve), always-ready floor answers, controller with watchdog overlay + flight recorder (60→79 tests)
- `core/parsing/` — prompt templates, tiered ladder (api→api2→local→regex, never raises), regex tier total over all 75 questions (100% qtype), vocab w/ typo map, **46 hand-parsed golden fixtures** (30+ tests)
- Verification pass found 2 real defects, both fixed + 19 regression tests: (1) MEDIUM — wrong-typed verify return latched publish before dispatch → total silence; dispatch now reports success before latch, failures fall to floor; (2) LOW — ladder→CallLedger signature mismatch silently no-opped the parse audit; now real signature + allow() gating
- Docs: `docs/ubuntu_setup.md` (full install runbook), `docs/sim_verification.md` (3-tier working-check ladder w/ verified image tags/commands), CLAUDE.md file map refreshed
- Sample-data download: all automated modes fail (0-byte) → manual browser download flagged in master_plan

**Rules adopted this session:** frontier-tier models plan/orchestrate only — implementation on standard executor tiers; install-affecting changes must update ubuntu_setup.md same-session; no tooling attribution anywhere.

**Next step:**
1. USER: register (15 Jul deadline!), manually download sample data
2. Wire the answer heads: resolve→verify→publish glue between parsing/geometry/fsm (the per-type strategies from architecture §4)
3. Replay harness over sample bags once data lands; then real perception (detector integration) behind SceneIndex
4. Ubuntu reinstall → Phase 2 (sim_verification.md Tier 2)

---

## 2026-07-10 (session 6) — Answer heads: pipeline answers end-to-end

**Done (317 tests green, full suite now ~7 min — integration tests simulate full 5 Hz runs):**
- `core/heads/` — the glue implementing architecture §4: factory builds the FSM's injected callables from a live SceneIndex + nav stack; numerical head (count-stability gating), object-ref head (resolve → per-clause verify → marker), instruction head (leg grounding, once-per-question avoid stamping, plan_through → breadcrumbs, interleaved explore-execute, terminal WaypointCmd), explore step (sweep → affinity-biased frontiers)
- `tests/integration/test_end_to_end.py` — full QuestionController runs on synthetic scenes: all 3 qtypes answer correctly; pathological scene → floor answer, never silence
- Adversarial verification round 2 found a **HIGH defect**: corridor+avoid combination where the capsule engulfs gate+start → the nearest-reachable fallback emitted a raw straight-line path that drove THROUGH the hard capsule (min dist 0.025 m). Fixed: recovery now A*-planned over the stamped costmap; breadcrumbs line-of-sight hardened against out-of-snapshot cells; verifier re-ran its own reproducers → CONFIRMED clean (min dist 2.36 m, answer still published). 5 regression tests added.
- Notable: absent-target numerical answers IntAnswer(0) (the true count), not the modal floor — 0 is correct for "how many unicorns"
- Fix agent died mid-run (auth error) but had completed its work; state audited (git + suite + verifier re-probe), nothing lost

**Known inert wart:** `BreadcrumbFollower.replan_flag` is write-only (never read) and the new no-LOS branch sets it eagerly — harmless today; revisit when wiring replanning at integration.

**Next step:**
1. USER: register (15 Jul!), manual sample-data download (browser)
2. Replay harness: parse sample bags (`rosbags`) → feed PanoFrame/LidarScan/TerrainPatch into the core; then real perception (open-vocab detector) behind SceneIndex — the last big Windows-doable pieces
3. Calibration pass over the geometry/nav tunables against training scenes (Phase 2)
4. Ubuntu reinstall → sim_verification.md Tier 2
