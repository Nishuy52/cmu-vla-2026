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

---

## 2026-07-10 (session 6, continued) — LLM layer; quota wall

**Done:**
- `core/llm/` — provider adapters (OpenAI-compat covers OpenAI/Gemini-compat/local servers; Anthropic Messages; LocalStub), env-driven failover config (`VLA_LLM_*` vars, keys indirected — see `core/llm/config.py` docstring), Windows-safe thread-based call timeout. 30 tests, SDK-free suite. **347 total green.** Committed `ecc5738`.
- **Session usage limit hit** (resets 11:10 PM SGT) mid-dispatch of the perception scaffold. Task spec preserved in `docs/next_task_perception.md` — next session: check whether the re-dispatched agent completed it (look for `core/perception/tiling.py` etc. + run suite); if absent, re-run the spec verbatim as an executor delegation, then delete the spec file and commit.

**Resume checklist for next session (in order):**
1. `git status` + full `pytest` (~7 min) — confirm 347+ green baseline
2. Perception scaffold per `docs/next_task_perception.md` (if not landed)
3. Replay harness once the USER's manual sample-data download exists in `data/sample_real_robot/`
4. Then: CLI question-runner utility, calibration prep, and Phase 2 per `master_plan.md`
5. Standing rules: frontier model orchestrates only; no tooling attribution; install-affecting changes update `ubuntu_setup.md`

---

## 2026-07-11 (session 7) — Windows core complete; backed up to GitHub

**Done (445 tests green, full suite 7.5 min):**
- USER: registered for the challenge + upgraded subscription (both hard deadlines cleared)
- `core/perception/` — gnomonic tiling (4×90° pinhole tiles, invertible pixel↔ray, calibration constants exposed: AZIMUTH_SIGN / COLUMN0_YAW_OFFSET / ELEVATION_SIGN; camera assumed level — Phase-2 item), lidar-frustum fusion (nearest range cluster, ±π wrap-safe), greedy tracker over BasicSceneIndex merge, FakeDetector + lazy GroundingDINO seam. 59 tests. *(Note: files landed inside the earlier "Registration complete" commit — agent finished writing before that sweep; message doesn't reflect content.)*
- `core/replay/` — bag→dataclass converters (PointField-offset-robust), ReplayRobotIO (full controller runs on recorded data), fixture distillation (measured 5.7× shrink; CLI `python -m core.replay.fixtures extract <bag> <out>`). `rosbags` added as hard dep. 39 tests. Deletes the need to move whole bags between machines — extract on the cluster, transfer fixtures only.
- Private backup remote created and pushed: github.com/Nishuy52/cmu-vla-2026 (session protocol now ends with a push)
- `docs/next_task_perception.md` obsolete → deleted this entry (task landed)

**Windows-side core is structurally COMPLETE**: contracts → parsing → geometry → nav → FSM → heads → LLM providers → perception → replay. Everything remaining is data-dependent (sample bags: user downloading via rclone over throttled VPN; cluster route recommended) or Ubuntu-dependent (real detector weights, ROS adapter, sim loop).

**Next session:**
1. If `data/sample_real_robot/` has the bags: extract fixtures, run the controller against real data, start calibration (thresholds in geometry.Thresholds, FusionConfig, nav tunables)
2. Else: CLI question-runner utility (`python -m core.run "<question>"` against synthetic/replay scenes) + offline scoring harness for the 75 training questions
3. SoC cluster: user may deliver `sinfo` / `quota -s` output → update soc_cluster_guide.md TODOs

**Next step:**
1. USER: register (15 Jul!), manual sample-data download (browser)
2. Replay harness: parse sample bags (`rosbags`) → feed PanoFrame/LidarScan/TerrainPatch into the core; then real perception (open-vocab detector) behind SceneIndex — the last big Windows-doable pieces
3. Calibration pass over the geometry/nav tunables against training scenes (Phase 2)
4. Ubuntu reinstall → sim_verification.md Tier 2

---

## 2026-07-11 (session 7, continued) — Design discussion: colored 3D reconstruction (debug tool)

**Done:**
- Extended Q&A walkthrough of the perception/nav architecture (lidar-vs-camera roles, how
  detection bboxes become 3D positions via `tiling.py` + `fusion.py`, occupancy vs instance-map
  distinction, why no dense 3D voxel map exists today).
- Evaluated and rejected: feeding raw/uncolored lidar voxel screenshots to a VLM for object ID
  (no color/texture — worse signal than the existing camera crop); building a live 3D voxel map
  for the real answer pipeline (sparse instance map + 2D costmap already cover it).
- Landed on a genuinely new, in-scope idea: an **offline colored point-cloud reconstruction
  tool** (forward-project accumulated lidar points onto the panorama to pick up RGB, voxel-
  downsample, export `.ply` for viewing in Open3D/CloudCompare) — pure dev/debug tooling, not
  part of the scored pipeline, so it should live outside `core/`/`ai_module` (proposed:
  top-level `tools/`).
- Task spec written to `docs/next_task_colored_reconstruction.md` for the next session to pick
  up (technique, plumbing over the existing replay harness, occlusion/volume caveats, resume
  checklist).

**Next step:**
1. Implement `docs/next_task_colored_reconstruction.md` (forward-projection function, voxel
   downsample, ASCII PLY writer, CLI over the replay harness, unit tests)
2. Otherwise continue the standing next-step list above (sample-data download, replay harness
   real-data run, calibration pass, Ubuntu reinstall)

---

## 2026-07-11 (session 8) — Colored-reconstruction research pass (3-agent fan-out)

**Done:**
- Pre-implementation research for the colored point-cloud tool: one codebase-recon agent +
  two web-research agents (colorization techniques; occlusion/voxel/format choices). Findings
  folded into `docs/next_task_colored_reconstruction.md` as a binding research addendum.

**Key outcomes (spec amendments):**
- Plan survives review; forward projection confirmed as pure composition of existing
  calibrated `tiling.py` functions; `LidarScan.points` already map-frame → apex subtraction
  only.
- **VFOV gate added** (spec omission): pano covers ±60° elevation only — out-of-band points
  must be dropped/flagged, never clamped.
- **Binary PLY** replaces ASCII (still zero-dep via numpy structured arrays; 3–5× smaller;
  `red/green/blue` as `uchar` — float color is a known viewer-breaking trap).
- **No color blending**: nearest-in-time pano, nearest-neighbor pixel; misregistration should
  stay visible (it's the diagnostic signal). Min-range cutoff ~0.75 m kills the near-field
  parallax tail (~30 px error at 1 m vs ~3–6 px past 5 m).
- **HPR rejected / z-buffer confirmed** as the eventual occlusion primitive; skipping occlusion
  first-pass stays OK (errors localized to silhouette edges; multi-view accumulation
  self-corrects). Depth-discontinuity skip is the best cheap mitigation.
- **Packed-int64 numpy voxel reduce** (min-offset before floor; `np.unique` + `bincount`)
  replaces the dict sketch; mean color first, median as follow-up.
- Closest prior art identified: **OmniColor** (arXiv:2404.04693) — same pipeline shape.
- Recon confirmed: `data/sample_real_robot/` still only a partial zip (validate on synthetic
  fixtures); tool tunables are CLI flags, not calibration-ledger entries; `tools/` dir does
  not exist yet.

**Next step:** implement per the amended spec (unchanged step list, addendum binding).

---

## 2026-07-11 (session 8, overnight autonomous run) — Real data + checkpoints wired

User asleep; ran autonomously per standing directive. **605 tests passed + 3 skipped** at close.

**Real-data milestone:**
- Sample download completed (3.16 GB) → unzipped → MCAP bag verified against the documented topic contract (123 s, 6 topics, expected rates)
- Fixture extraction on first contact: **zero converter failures**, 246 keyframes, 3.16 GB → 569 MB (5.6×). Fixtures at `data/fixtures/jingfan/` (git-ignored)
- Full controller replay against real data exposed a replay-only defect: bag (123 s) ends before answer gates (510/570 s) → runner surfaced an exploration waypoint as the "answer". Fixed: ReplayClock free-runs past end-of-data (frozen world) so gates fire; RunResult.answer now captures only the FSM's published answer; `--budget-scale` added for time-compressed replay. Repro now completes idle→…→answer→done with a correct MarkerBox (floor path — expected: no real detector on Windows).

**Checkpoint protocol (architecture §3) designed and implemented:**
- `docs/checkpoint_design.md` — full protocol: prompts, JSON schemas, fallbacks, ledger caps for CP2 (miss recovery), CP3 (anchor confirm), CP4 (pre-answer verification — highest value), CP5 (frontier select)
- `core/checkpoints/` — all four modules + golden fixtures, offline-tested (79 tests)
- Head seams widened to the full contracts (CP4 three-way verdict incl. missed-constraint re-resolve; CP3 demote-and-replan; CP2 provisional instances that can never satisfy the ≥3-obs gate; CP5 multi-room trigger) with backward compat via signature inspection (46 tests)

**Phase-2 deployment drafts (untested until Ubuntu, flagged inline):**
- `src/ros_adapter/` — complete rclpy node reusing the real-data-proven replay converters, ament package, launch file; `docker/ai_module/` — Dockerfile from the upstream base image + README with build/push/submission flow; ubuntu_setup.md §7a
- Three "confirm on Ubuntu" items flagged in-file: ament+core coexistence, scene-index/perception wiring into the node, colcon workspace layout

**Also:** qtype heuristic fixed (word-boundary regexes; "kitchen counter" no longer matches "count") + pinned against all 75 questions; `.gitignore` checkpoint-dir collision fixed.

**Next session:**
1. Wire PerceptionPipeline into the replay path with a scripted FakeDetector against the jingfan fixtures → first grounded real-data answers
2. USER: API keys (`VLA_LLM_*`) → live CP4/parse tiers; SoC cluster login (`sinfo`, `quota -s` → finish cluster guide)
3. Battery re-run + report diff after seam widening
4. Ubuntu: sim_verification Tier 2 + the three confirm-on-Ubuntu flags

---

## 2026-07-11 (session 8 finale) — First grounded real-data answers

- Exported real panoramas to PNG (scene: a CMU student lounge), hand-labeled 29 detections across 2 keyframes → `data/fixtures/jingfan_labels.json`
- `core/perception/scripted.py` + runner `--detections` flag: pano-space labels → tile-space detections → **real lidar frustum fusion** → tracked 3D instances feeding the heads
- **"How many white stools are in the room?" → 2** (5 labeled; 3 fused with ~300 lidar pts each, 2 adjacent merged IoU>0.3, 2 rejected min_points — sensible dedup); **"Find the folding chair closest to the yellow door." → MarkerBox(2.74, 1.27, 'chair')**; both via real heads, floor_used=False, 9 instances tracked
- Judgment call flagged: bag's first ~9 s has no registered scan → startup fallback fuses keyframe-0 against the earliest scan (vehicle stationary there — geometrically valid)
- **642 tests green.** The architecture's full chain is now demonstrated on real data: panorama → detections → fusion → instance map → spatial toolbox → published answer.

---

## 2026-07-11 (session 9) — Ground-truth evaluation: first real accuracy numbers

- VLA-3D Unity subset (all 18 scenes) downloaded after a truncation fight (curl+resume beats the boto3 script) and extracted (5 GB, `data/vla3d/Unity/`); dataset notes in `docs/vla3d_notes.md` (MIT license, schemas verified, scene names/nouns match challenge verbatim)
- GT harness: `core/groundtruth/` loader (OBB→AABB, colors→attributes) + honest per-type scorers + `gt_battery` runner; refined with statement-based target matching (exact/fuzzy/relation ladder), per-scene trajectory frame fitting (translation+optional yaw, residual-gated), multi-opinion counting
- **FULL 15-SCENE TOPLINE (75 questions):** numerical pipeline-exact 100% but independent agreement only 15–27% (we over-count vs annotations — top calibration target); object-ref scoreable on 6/30 (4 perfect IoU, 2 wrong-instance; 24 honest vocabulary-drift non-matches); instruction-following mean Fréchet 6.1 m, 30% coverage@1m on 12/15 aligned scenes (3 scenes unaligned — goal disambiguation suspect: home_building_2, hotel_room_2, livingroom_3)
- Suite green (~711 tests). Reports: `reports/gt_battery_full_2026-07-11/`

**Next:** k-fold CV sweep harness (leave-3-scenes-out) over high-sensitivity calibration params; OR scoreability via vocabulary bridging; investigate the 3 unaligned scenes + counting over-count.

---

## 2026-07-11 (research import) - parallel 2025-results research stream folded in

- Imported an independent parallel research stream on branch `docs/import-2025-research`: 15 per-source dossiers under `docs/prior_art/` (index + provenance/conflict rules: `docs/prior_art/README.md`), `docs/competitor_forks.md`, `docs/io_contract_crosscheck.md`, and the 2025 leaderboard screenshot (`docs/assets/cmu_vla_2025_leaderboard.png`)
- Resolves the standing "2025 1st/2nd unknown" gap: 1st NROS (HITSZ) 44.26, 2nd ReasonX (NTU+NUS) 34.58, 3rd CopyPasta (CMU MRSD) 30.98, 4th URL-KAIST 22.80; GT `/object_markers` reliance means most 2025 pipelines don't port to the 2026 I/O list
- Dated update sections appended to `prior_art.md`, `organizer_playbook.md`, `vla3d_notes.md`; discovery rows added to `CLAUDE.md` layout table and `README.md`; no existing team text rewritten
- Import references to the pre-merge workspace's roadmap/task files were remapped to `architecture.md`/`master_plan.md` equivalents so no links dangle

**Next:** review + merge the PR, then mine the dossiers for calibration targets (e.g. VLA-3D relation thresholds vs `core/geometry/toolbox.py` constants; CopyPasta/ReasonX failure modes as battery test cases)

---

## 2026-07-11 (task-record import) - milestone one-liners

- Wrapped up task: research import into this repo - record at `docs/tasks/T3-merge-into-team-repo/`
- Finished task: implementation-vs-research gap analysis (ranked top-5) - record at `docs/tasks/T4-implementation-gaps/`
- Task records now live under `docs/tasks/` (workflow adapter in `CLAUDE.md`); LOG gets one-liners only at task milestones from here on
## 2026-07-11 (session 9 close) — Overhead clearance; stopped on user instruction

- **Overhead-clearance layer** (`core/nav/occupancy.py` OverheadConfig + costmap integration): the base stack's terrain analysis discards lidar points above maxRelZ=0.2 (verified terrainAnalysis.cpp:172), so under-furniture floor reads FREE; our layer reads the unfiltered /registered_scan and blocks cells with points in the 0.25–1.2 m band. **Validated on real jingfan data: 73 cells of terrain-FREE floor under actual tabletops now blocked (+10.1 m² total)**. 5 new calibration tunables documented. 19 tests; suite green.
- RVIZ debug kit committed earlier this session (`d9ff785`): ai_module_debug.launch.py + config, gated instance-map/planned-path publishers, sim_verification §2.8.
- Vocab bridge raised scoreable OR questions 6→7 (from the cvsweep agent's pre-step, committed with its files when sweep lands).
- Playbook Gate 5b added: perception-vs-reasoning error split via dev-time GT semantics (2025's GT-semantics loophole is closed at test time in 2026 — only the six topics are legal).

**Still running detached (local compute, no quota):** the k-fold CV sweep (two python processes; writes to `reports/cvsweep_<date>/{report.md,results.json,recommended_calibration.json}` when done; cvsweep code itself is on disk uncommitted in `core/runner/cvsweep.py` + tests — commit with its results next session).

**Backlog for next session (not dispatched, per stop instruction):**
1. Read + commit the CV sweep results; adopt/record the recommended calibration
2. Test-suite tiering: @pytest.mark.slow markers + "-m not slow" default, budget-scale the integration tests, pytest-xdist; update session protocol (design agreed in-conversation 11 Jul)
3. Red-team design review over the GT-battery + sweep numbers → hardening backlog
4. Investigate: numerical over-count (independent agreement 15–27%), 3 unaligned IF scenes (goal disambiguation), 24 unscoreable OR questions

---

## 2026-07-11 — task milestones

- Started + finished task: colored point-cloud reconstruction debug tool (T5) —
  `tools/` package, validated on real jingfan data; record at
  `docs/tasks/T5-colored-cloud-tool/`. Spec file
  `docs/next_task_colored_reconstruction.md` deleted per its lifecycle note
  (decisions preserved in the task record; full text in git history). PR from
  `tool/colored-cloud`.
- Started + finished task: live colored voxel map + RViz robot-in-map debug
  view (T6) — record at `docs/tasks/T6-live-colored-map/`; stacked PR from
  `feat/live-colored-map`. Debug layer only; scored path untouched.

---

## 2026-07-14 (task milestones) - one-liners

- Finished task: GT-battery numerical count diagnosis - record at `docs/tasks/T7-numerical-count-diagnosis/` (headline: not a calibration problem - 4 resolver code defects + 3 scorer-side measurement artifacts; k-fold sweep gated on the fixes)
- Finished task: 2025-dossier adjudication vs architecture v1.0 - record at `docs/tasks/T8-2025-dossier-adjudication/` (7 conflict verdicts; two dossier priors rejected on instance-level GT evidence; GT-markers implication folded into `architecture.md` section 8 + `master_plan.md` Phase 2)

---

## 2026-07-12 (task milestone one-liner)

- Finished task: test-suite tiering (fast default tier + slow milestone gate, budget-scaled integration tests, pytest-xdist full gate) — record at `docs/tasks/T9-test-suite-tiering/`

---

## 2026-07-14 (session 10) — PR review round: #2–#5 all merged

- Reviewed and merged all four open PRs in stack order: #2 (T5 colored-cloud tool), #3 (T6 live colored map — retargeted to main after #2's branch deletion auto-closed it), #4 (T7/T8 diagnosis + adjudication docs), #5 (T9 test-suite tiering).
- Review method: parallel finder passes (correctness / removed-behavior / cross-file / reuse / conventions) + fresh-context verification per PR; findings posted as PR comments. Net: no merge blockers; 5 CONFIRMED non-blocking defects recorded on #2/#3 (all in debug-only paths — pano shape guard, NaN voxel-guard bypass, colored-map lock contract, `colored_cloud_period` 0.0 coercion, whole-scan abort on one out-of-range point) for a later hardening pass; scored answer path verified untouched.
- Conflict resolutions: #4 vs main (LOG/INDEX chronological interleave of T5/T6 with T7/T8); #5 vs main (same two files + task record renumbered T7 → T9, id was claimed by the merged numerical-count diagnosis).
- Post-merge verification on final main: fast tier 687 passed / 47 deselected in ~51 s (3 setup errors are this worktree's missing git-ignored `data/` fixtures, identical pre-merge — not regressions).

**Next:** session-9 backlog items 1, 3, 4 — read/commit the CV sweep results + adopt calibration, red-team design review over GT-battery + sweep numbers, and the over-count/unaligned-scene/vocab-drift investigations (T7's diagnosis now gates the sweep adoption).

---

## 2026-07-14 (session 11) — CP2 provisional-gate test reworked for H15b

- Coordination fix requested by the perception-hygiene stream (H15b / NUM-F8, uncommitted in sibling worktree `nostalgic-mccarthy-3bb3ea`): their answer-time n_obs gating in `NumericalHead.advance` (once a noun is established at n_obs>=3, n_obs==1 instances are excluded from the count) breaks `test_provisional_never_satisfies_early_answer_gate`, which asserted the old pinning mechanism; that file is owned by this stream.
- Split it into two contract-bridging tests in `src/tests/heads/test_explore_cp2.py`, green against BOTH the current head on this branch and the H15b head: (1) cold-start — all instances below the establish threshold, single-obs contributor still pins `min_contrib_n_obs < 3`, gate shut; (2) established class — mechanism-agnostic CP2 guarantee `stable ⇒ answer excludes the provisional` (pre-H15b holds via the shut gate, post-H15b via exclusion with count == 1).
- Verified: targeted file 12/12 pass; fast tier passes; H15b cross-check ran read-only against the sibling worktree's head (expected values confirmed: cold-start min=1/unstable; established stable with answer 1). Full gate has 5 `tests/runner/` failures + 3 `tests/parsing/test_regex_full_set.py` collection errors, all `FileNotFoundError` on this worktree's missing git-ignored `upstream/` clone — environmental, pre-existing, unrelated to this change.

**Next:** unchanged — session-9 backlog items 1, 3, 4 (see session-10 entry). The hygiene stream can now merge/rebase onto this branch's test.
