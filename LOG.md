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

## 2026-07-14 — task milestone (parallel stream, merged after session 10)

- Wrapped up task: red-team design review (five-facet frontier batch + adjudication) — record at `docs/tasks/T10-redteam-review/`, output at `docs/redteam/` (**start at `hardening_backlog.md`**: 4 dominant defects incl. a QoS run-killer and the IF exploration deadlock; CV-sweep rerun is now HELD until Tier-0 fixes + scorer repairs land — supersedes the standing "rerun cvsweep" next-step until then)

---

## 2026-07-14 (session 11) — Red-team backlog implementation day 1: H1–H5, H8/H10 partials

- Executed hardening backlog via agent batches, each independently verified CONFIRMED: **batch A** `4bcf18b` (H1 QoS dual-sub, H2 scorer rebuild [driven-trajectory IF rubric + strict counts], H4a/b ladder policies [+scope-clause follow-up: "in the room" is vacuous scoping, named rooms strict], H8 partial [qtype correction, 480/540 skew-hedged gates, sim-time guard, stub-index shout], H10 vocab tiers + colour bridge); **batch B** `8a8c78d` (H3 IF explore-while-ungrounded — the 36-pt deadlock is broken, empty scene ⇒ waypoints, full-controller verified; H4c provisional terminals; IF-F6 distinct pair anchors); **H5** `0863294` (predicate forms per the reconciled spec; 92.3% recall vs 819 real GT on-edges; T7-S2 scorer pair fix).
- Merged main `bb79133`: T9 tiering (fast 56 s / full gate 71 s — verifier passes now minutes not an hour), colored-map stack, T7/T8 records; redteam task renumbered T5→T10.
- **Cross-stream reconciliation before H5 (the judgment call of the day):** T8's instance-level-GT verdicts vs our dossier report's generator-code priors — evidence won; `above()` overlap gate REPLACED by lateral-offset (our own NUM-F2 sided with T8 against our dossier agent), `on()` upper-z-span, `between()` narrowed to strict-t. Appendix in `dossier_deltas.md`; backlog H5 rewritten pre-implementation.
- **Post-fix GT battery** (`reports/gt_battery_postfix_2026-07-14/`): numerical independent agreement **15–27% → 56%** (strict-evidence rows; sg agreement 73%; the 31-pillow answer is now 11, red-pillows 10→3 vs truth 4); object-ref scored 8/30, mean IoU 0.875, **the OR-F1 nested-disambiguator case now scores 1.000** (was 0.000); IF measured honestly for the first time — **rubric-proxy 0.061** (ordered-leg credit 0.094, 9 threading violations): the old Fréchet numbers flattered shape, the rubric exposes that driven legs rarely arrive in order. Suspicious harness artifact flagged: several IF rows show poses=2–30 (follower exhausts near-immediately in the mirror sim) — investigate the v1 kinematic driver before trusting IF absolute levels.
- Backlog status ledger added (H1–H5, H10, IF-F6 landed; H8/H6 partial; H7, H9, H11–H15 open; sweep still HELD pending objective re-point + this baseline).

**Next:** (1) investigate the IF driven-sim poses collapse (harness vs pipeline) — it gates reading the rubric; (2) H9 checkpoint repairs + H8 seam wiring (budget_frac, ladder); (3) re-point cvsweep objective and un-hold the sweep; (4) Ubuntu Thu 16 Jul → phase2_playbook Gate 0 with the new Ubuntu-gate checklist items from the backlog.

---

## 2026-07-14 (session 11, continued) — Backlog wave: H6–H15 + follow-ups all landed

- Nine-agent parallel wave (disjoint file ownership) + 2 follow-ups + FSM continue-drive: adapter seam wiring (perception `VLA_DETECTOR` switch, LLM ladder api→api2→regex with the local tier explicitly DESCOPED, checkpoint seams, budget_frac live, injection-boundary timeouts, ledger admission bound), CP2–CP5 checkpoint repairs (keep-on-neither, clause-synthesizing re-resolve, provisional-commit guard, cross-class-only demote, prompt ≤2k, abstain), nav/drive robustness (replan consumption, avoid lifecycle + edge-triggered tripwire, explore-goal guards, CP2 latch clear, free-space via, 1 Hz frontier throttle), dimension-priors marker clamp (69 data-derived classes; wired into object_ref + floors), detection-bearing-keyframe ghost decay + obs-gated counting, overhead soft-cost (via the A* unknown seam; clone() hardens for corridors), frontier BFS vectorized ~10× (bit-identical, oracle-tested), fork-shaped docker packaging + sync scripts, cvsweep objective re-point (rubric×6/strict×1/IoU×2, disk-resume cache) + tests, IF driven-sim made closed-loop (poses-collapse was a real harness bug — but fixing it proved rubric≈0 is genuine pipeline behavior), and a new FSM `DRIVE_OUT` state so IF keeps driving after ANSWER until route completion or the watchdog (verified: the publish latch means the floor can never overwrite the answer).
- Verification: two combined fresh-context verifier passes (wave CONFIRMED; DRIVE_OUT CONFIRMED with 14 independent probes) + full gate **1039 passed** (only the 8 known worktree data gates non-passing). Four integration defects found by the gates/verifiers and fixed in-session (CP3 mid-tick demote crash, decay-vs-single-pass semantics, tracker ledger drift, FSM answer-and-stop-at-270 s).
- Incident: a stray `git stash` by one wave agent swept the shared tree mid-wave; fully recovered (stash inventory → selective restore → owners re-applied), no work lost. Rule added to all agent briefs: never git stash/checkout/reset.
- **Post-wave battery** (`reports/gt_battery_postwave_2026-07-14/`): numerical 56% / OR 0.875 hold; **IF rubric 0.061 unchanged — intermediate-leg threading (routes reach terminals without passing within 0.8 m of intermediate leg goals) is confirmed as the top remaining pipeline item (36 pts)**. CV sweep stays deliberately HELD until it lands (flat IF term = no gradient for the ×6 weight). Backlog status ledger updated in `docs/redteam/hardening_backlog.md`.

**Next:** (1) IF intermediate-leg threading diagnosis + fix → battery → un-hold sweep; (2) Ubuntu Thu 16 Jul: phase2_playbook Gate 0 with the new gate items (QoS verify, skew measurement, compose-up from clean fork, stub-index guard); (3) merge branch to main via PR.

---

## 2026-07-14 (session 11) — Calibration-ledger drift check (tracker decay_k)

- Investigated the reported `test_field_counts_per_subsystem` failure (`tracker: 2 != 1`). Not reproducible on main-derived branches: the H15 `decay_k` change lives only in the concurrent red-team worktree (`claude/nostalgic-mccarthy-3bb3ea`, uncommitted). That session had already fixed the test pin (`tracker: 2`, `nav: 21`), the `tracker.decay_k` doc row, and the tracker section header; its calibration test passes 17/17.
- Residual drift was only the two hand-maintained header lines in that worktree's `docs/calibration.md`. Fixed the nav section header (20 → 21 fields). The wiring-summary line could not be edited from this session (permission boundary on another session's active worktree) and **still reads `59 fields total (… tracker 1 … nav 20 …) / wireable 45`; correct values are `61 total (tracker 2, nav 21) / wireable 47` (TODO stays 14; 47+14=61)** — H15 session: fold this one-line fix into your commit. Note the five `overhead_*` band rows in the nav table are OverheadConfig/module-constant documentation, not NavTunables fields, and are correctly excluded from all tallies.
- No code changes on this branch; no test-gate run needed (LOG-only commit).

**Next:** H15 session applies the wiring-summary line fix above and commits it with the rest of its calibration.md edits.

---

## 2026-07-14 (session 12) — CP2 provisional-gate test reworked for H15b

- Coordination fix requested by the perception-hygiene stream (H15b / NUM-F8, uncommitted in sibling worktree `nostalgic-mccarthy-3bb3ea`): their answer-time n_obs gating in `NumericalHead.advance` (once a noun is established at n_obs>=3, n_obs==1 instances are excluded from the count) breaks `test_provisional_never_satisfies_early_answer_gate`, which asserted the old pinning mechanism; that file is owned by this stream.
- Split it into two contract-bridging tests in `src/tests/heads/test_explore_cp2.py`, green against BOTH the current head on this branch and the H15b head: (1) cold-start — all instances below the establish threshold, single-obs contributor still pins `min_contrib_n_obs < 3`, gate shut; (2) established class — mechanism-agnostic CP2 guarantee `stable ⇒ answer excludes the provisional` (pre-H15b holds via the shut gate, post-H15b via exclusion with count == 1).
- Verified: targeted file 12/12 pass; fast tier passes; H15b cross-check ran read-only against the sibling worktree's head (expected values confirmed: cold-start min=1/unstable; established stable with answer 1). Full gate has 5 `tests/runner/` failures + 3 `tests/parsing/test_regex_full_set.py` collection errors, all `FileNotFoundError` on this worktree's missing git-ignored `upstream/` clone — environmental, pre-existing, unrelated to this change.
- Integrated into `main` via PR #6 (user-approved merge). Conflict vs main: LOG-only interleave with the session-11 calibration-drift entry from PR #7; this entry renumbered 11 → 12 (number claimed on main). No code overlap (#7 was LOG-only).

**Next:** unchanged — session-9 backlog items 1, 3, 4 (see session-10 entry). The hygiene stream now rebases/merges from main to pick up the bridged test.

---

*(Merge note, session 11: PR #6's bridged CP2 test variant was superseded by this branch's version, which was written and verified against the integrated H15b head; PR #7's wiring-summary line fix is applied in this merge commit.)*

---

## 2026-07-14 (session 13) - task milestone

- Started task: IF intermediate-leg threading fix (T11) - record at
  `docs/tasks/T11-if-leg-threading/`; branch `fix/if-leg-threading`.
  **If this session died mid-pipeline: resume from
  `docs/tasks/T11-if-leg-threading/orchestration.md`** (queue: executor
  fix -> verifier -> CV sweep -> Fable critique -> commit/push/PR).

## 2026-07-15 (session 13 close) - task milestones

- Finished task: IF intermediate-leg threading fix (T11) - rubric
  0.061->0.100, toplines held, verifier-hardened record; full detail at
  `docs/tasks/T11-if-leg-threading/` (executor_report, verification,
  gt_leg_ceiling artifact).
- Frontier critique of approach + architecture adjudicated -
  `docs/tasks/T11-if-leg-threading/critique/adjudication.md` (binding
  sweep-adoption gate; clock semantics resolved per-question from
  upstream README; cvsweep instrument-endogeneity fixed pre-launch;
  user decisions pending: answers.json transcription, Gate 0,
  local-VLM tier posture).
- CV sweep RUNNING since 15 Jul -> `reports/cvsweep_2026-07-15/`
  (disk-resume; BelowNormal priority; adoption gated per adjudication).
  Full gate 1090 passed / 5 skipped. Stale 11-Jul sweep artifacts
  deleted per adjudication meth-F10.

## 2026-07-15 (session 13 addendum)

- CV sweep COMPLETE - verdict: keep defaults (all 5 folds chose
  baseline; holdout 0.229+/-0.036, gap -0.010; adoption gate passed
  trivially). `reports/cvsweep_2026-07-15/`. Re-sweep deferred to
  real sim.
- Numerical GT answer keys extracted from questions.pdf TEXT layer
  (not images): `tools/extract_pdf_answers.py` ->
  `docs/gt_answers_numerical.json`. TRUE numerical accuracy 11/15
  (73%) vs 56%-over-9 proxy; proxies unreliable both directions.
  4 true failures = the numerical worklist. pypdf added to dev deps
  (ubuntu_setup updated). OR/IF keys are images - still need visual
  transcription.

---

## 2026-07-17 (session 14) - task milestones

- Started + finished task: T12 numerical yardstick, provenance, and the
  pre-Ubuntu queue - record at `docs/tasks/T12-numerical-yardstick/`
  (task.md, plan.md, diagnosis.md, unaligned_scenes.md, postT11_to_T12.md).
  Branch `feat/t12-numerical-yardstick`, spec+plan+subagent execution
  (8 tasks + 2 verifier gates), both gates CONFIRMED, full gate ~1140
  passed. Headlines: TRUE numerical accuracy 11/15 -> 13/15 (`under()`
  wall-relative branch fixes arabic_room + home_building_1; 11 passes
  held); every battery/sweep number now born-provenanced (meth-F7/F8)
  and diffable via `tools/battery_diff.py`; battery topline led by TRUE
  accuracy k/15 (meth-F4/F6); IF frame-alignment coverage 24 -> 28/30
  (meth-F11, friendly-ward bias hole closed, livingroom_3 exclusion
  data-confirmed); IF leg census 0/30 dropped (meth-F5); first committed
  `--no-spawn-hint` run (arch-F9, IF rubric shown spawn-artifact-sensitive).
- New standing rule 5 (CLAUDE.md `59f858e`): unfixed detected issues get
  filed as GitHub issues. Applied this session: **#10** (battery_diff
  test fixture stale key), **#11** (loft black-pillow count - no raw RGB
  in InstanceRecord), **#12** (home_building_2 red-pillow color-salience,
  sweep-deferred), **#13** (terminal-goal resolver mis-ranking - affects
  object-ref + real-sim GOTO, only worked around by the frame-fit).

**Next:** open PR to main for the T12 branch. Then the residual
adjudicated/user items: OR/IF image answer-key visual transcription
(arch-F3 residual), Gate 0 execution (arch-F7), local-VLM tier posture
(arch-F5), SoC cluster detector benchmarking (arch-F2), Ubuntu day-one
(phase2_playbook Gate 0). Issues #11/#12/#13 are the actionable code
follow-up queue (PRs to close them).

---

## 2026-07-17 (session 15) - PR #14 merged

- Reviewed and merged PR #14 (T12 branch) into main (`5b6e5ba`);
  review verdict approve - load-bearing claims (under() lateral branch
  as faithful inverse-above, answer-key guard, frame-fit fallback
  inertness on aligned scenes, non-raising provenance) all verified
  against code; targeted fast tests green (91 src + 33 tools).
  Polish-level review findings filed per standing rule 5: **#15**
  (write_report `cal` param never wired - provenance always stamps
  default calibration), **#16** (corrupt answer key indistinguishable
  from missing), **#17** (CLI or_iou summary vs instance-match report
  headline), **#18** (dirty_digest excludes untracked contents,
  docstring overclaims), **#19** (test-hygiene: unrelated
  _DATA_UNFITTABLE_IF_SCENES assertion tacked onto terminal-goal test).
- Housekeeping: removed 4 stale worktrees (3 detached `.claude/worktrees`
  + `vla-tiering`; all clean) and deleted 8 merged local branches
  (6 `claude/*` session branches, `feat/live-colored-map`,
  `test/suite-tiering`). Local state now just main.

**Next:** unchanged from session 14 - OR/IF image answer-key visual
transcription (arch-F3 residual), Gate 0 execution, issues #11-#13
(now plus #15-#19 polish queue) as the actionable code follow-ups.

## 2026-07-18 (session 16) - Ubuntu day-one: Gate 1 complete

- Machine bring-up on the fresh Ubuntu 24.04.4 install (phase2_playbook
  Gate 1 / ubuntu_setup §0-§4). All installs scripted in
  `~/cmu_ws/setup_env.sh` (outside the repo, idempotent).
- Deviations from the guide, recorded in §3 "As-built": repo at
  `~/cmu_ws/cmu-vla-2026`, symlink `~/vla` -> there (all doc paths work
  verbatim); native ROS 2 Jazzy installed in addition to the Docker
  stack (host-side debugging only).
- Removed a suspicious snap curl (publisher "aoilinux" - not the
  official curl publisher; likely typosquat, worth remembering the box
  had it) -> proper apt curl 8.5.0.
- Verified: NVIDIA driver 595.71 on the RTX 4060 Laptop (8 GB - below
  the 10-14 GB eval target; §8 half-precision posture applies, full
  config validates on the SoC cluster). Docker 29.6.2 + NVIDIA
  Container Toolkit; `docker run --gpus all ubuntu nvidia-smi` PASSES.
  Disk 158 GB free. §0 full-upgrade done via script but NO reboot yet -
  do one before Gate 2 sim work in case of kernel/driver updates.
- Upstream cloned + submodule at the expected pin `81035e9` (§4).
- venv + deps (numpy/pytest/pytest-xdist/rosbags/pypdf). **Full gate
  green on Linux first try: 1095 passed, 46 skipped, 78 s** (`pytest
  -m "" -n auto` from `src/`); tools suite 33/33. No path/line-ending
  surprises - the Gate 1 exit criterion is met.

**Next:** Gate 2 - training scene binaries (Drive folder in upstream
README; remember gdown failed on the sample-data folder 10 Jul, so
browser download likely needed), pull upstream images, stock-sim smoke
tests 2.1-2.5 per sim_verification.md. Gate 0 residuals: LLM API keys,
Docker Hub account.

## 2026-07-18 (session 16, continued) - Gate 2: stock sim + dummy round-trip PASS

- **Scene-source trap found and resolved.** First scene install used zips
  from the 2025-era Drive folder `cmu_vla_challenge_unity_environments_ros1`
  (Nov 2023 binaries) -> ros_tcp_endpoint reconnect loop + JSONDecodeError,
  sensors silent. Control test with the image's stock scene (identified as
  livingroom_3 by object_list md5) passed, isolating the fault to the scene
  build generation. Correct folder = `unity_env_models`
  (`1nki_...`, the one the 2026 README links; Nov 2024 builds). Full detail
  in ubuntu_setup §5 "As-built findings".
- **All 18 correct scenes fetched + extracted** to
  `data/unity_scenes_ros2/` via new `tools/fetch_unity_scenes.sh` (per-file
  gdown with pinned IDs; folder-mode gdown failure from 10 Jul does not
  apply per-file on Ubuntu). office_building_1/2 have empty
  object_list.txt in both sets - platform extras, not training scenes.
- **Tier 2.1-2.5 PASS** on livingroom_1 (Nov 2024 build): containers up;
  bridge clean (0 JSON errors); scan ~4-5 Hz / state_estimation 200 Hz;
  waypoint round-trip drives the robot ((0,0) -> stop near (1.5,0), short
  of target consistent with obstacle stop); dummy VLM cross-container
  round-trip answers numerical question (`data: 4`).
- **Sharper form of gotcha 13 hit in practice:** RMW env only in the
  containers' .bashrc -> non-interactive `docker exec` runs FastDDS;
  discovery looks fine but no data crosses containers. Documented in
  ubuntu_setup §5.
- **Residuals:** (1) topic rates below contract on this box (camera
  ~3.7 Hz vs ~10) - issue drafted, creation blocked by session
  permissions, FILE MANUALLY next session; (2) Tier 2.2 visual RVIZ/Unity
  window check not eyeballed yet (user to confirm windows render);
  (3) Tier 2.6+ (relaunched dry-run) pending; (4) host not rebooted since
  full-upgrade - do before next sim session.

**Next:** file the rates issue, eyeball RVIZ, Tier 2.6 relaunch dry-run,
then Gate 3 (our module replaces dummy) per phase2_playbook.

## 2026-07-18 (session 16, continued) - Gate 3.1-3.2 PASS: our module replaces the dummy

- Executor (sonnet subagent) staged the fork shape into /tmp/fork-clean
  (local clone only - no public GitHub fork yet, deliberate), ran
  sync_to_fork.sh, built `docker-ai_module` (9.46 GB; base ~9.4 GB),
  swapped it in for the dummy container.
- All three confirm-on-Ubuntu Dockerfile flags resolved; biggest find:
  the upstream ai_module base image has NO pip (bootstrap added).
  Critical adapter fix: `self._clock` shadowed rclpy Node's `_clock` ->
  RecursionError at boot; renamed `_robotio_clock`. Details in
  ubuntu_setup.md §7a (updated).
- **Tier 2.7 PASS**, then independently re-verified by a fresh verifier
  subagent (fresh question, module restart, attribute-collision audit,
  fork invariant `git status` clean outside ai_module/ + compose,
  fast tests 1054 passed / 32 skipped). Evidence in both agents' runs:
  question latched -> 5 Hz waypoints -> exactly one legal integer at
  latch+~219 s (explore budget 210 s + answer ticks). Master plan rows
  38-39 ticked.
- pilotfish v1.2.1 installed globally this session (~/.claude: 8 role
  agents, orchestration policy, model=best); executor+verifier
  frontmatter switched to sonnet per user. Effective next session.
- **New defect (decision pending):** empty scene index -> numerical head
  injects partial.count=0 which overrides MODAL_COUNT=2 floor
  (core/fsm/floors.py:147-148). If perception is dark at eval, count
  questions answer 0 instead of modal 2. Proposed: suppress head count
  when instances_tracked==0. Needs GitHub issue (gh issue create still
  blocked for the agent session - file manually with the rates issue).

**Next:** Gate 3.3 tiling-constant calibration against the sim panorama,
then Gate 4 (GroundingDINO weights + real detector + PerceptionPipeline
wiring). Gate 0 residuals unchanged: LLM API keys, Docker Hub account.

## 2026-07-18 (session 16, continued) - #31 fixed, #32 closed, Gate 3.3 PASS

- **#31 fixed and verified** (61f046e): NumericalHead withholds on empty
  index; executor found the defect fired through a SECOND channel
  (factory._final_answer verify path bypasses floors entirely) and
  closed both. 7 tests added; independent verifier CONFIRMED all claims
  incl. fresh e2e repro (empty scene -> 2, three-sofas+absent-noun -> 0).
- **#32 closed** (locking tests only): object_ref and instruction heads
  audited - both already withhold on zero evidence and defer to
  better-informed floors. Full gate 1105 passed. Audit surfaced a
  confidence-gating design question -> filed as **#33** (committable
  route prefix accepts n_obs=1 legs; needs expected-points adjudication).
- **Gate 3.3 PASS, no flips**: AZIMUTH_SIGN/-1, COLUMN0_YAW_OFFSET/pi,
  ELEVATION_SIGN/-1 all CONFIRMED against live livingroom_1 pano -
  7 GT objects spanning the ring, <2% column error; camera-level
  consistent. Evidence: docs/tiling_calibration.md +
  data/calibration/pano_livingroom_1.png. Sim publishes bgr8 (not rgb8);
  checked - image_to_pano already swaps to RGB, no defect.
- Issues #30/#31 filed this session (gh allow rules added to user +
  project settings; note: rules match direct `gh issue ...` commands,
  not scripts wrapping them). #31 auto-closed by commit; #32 by 70922e6.

**Next:** Gate 4 - real perception (GroundingDINO deps + weights baked
into the image, real detector __call__, PerceptionPipeline wired into
adapter_node, VRAM check vs 8 GB with half precision). Blockers to clear
before Gate 5: LLM API keys (user), Docker Hub account (user), reboot.

## 2026-07-18 (session 17) - sim bag recording: livingroom_1 validation bags

- **Two livingroom_1 bags recorded + harness-verified** into git-ignored
  `data/sim_bags/` (index README there; procedure documented as new
  sim_verification.md section "Recording validation bags from the sim"):
  - `livingroom_1_tour/` (319.5 s, 5.5 GB): 15-waypoint scripted tour,
    living-room coverage x[-1.4,1.3] y[-5.7,1.8], path 20.2 m.
  - `livingroom_1_tour_nw/` (449.6 s, 7.4 GB): NW-pocket tour + dining
    doorway probe, x[-1.65,0.96] y[-5.03,1.42], path 17.3 m.
  Both: all five contract topics convert through core.replay.BagSource
  for the full duration (camera/scan/terrain 3.7-3.85 Hz, odom 200 Hz).
- **Driver was scripted waypoints, not a module question run**: the
  session's permission classifier blocked container restarts/kills, the
  adapter latches only the first question per boot, and that question was
  already consumed - so Tier-2.4-style `/way_point_with_heading` tours
  drove coverage instead. Equivalent for the bags' consumers (replay
  harness, overhead-tunables validation).
- **Scene + planner intel from the stalls**: livingroom_1 has an interior
  wall at y=-4.8 (doorway x[-0.9,0.2]) separating a dining area (2.1x1.0
  table + 8 chairs); the stock local planner refuses goals with <~0.5 m
  clearance to low objects (zero cmd_vel obstacle-stop, truncated /path).
  Working recipe: clearance-check tour waypoints against object_list.txt
  (excluding flat entries); >=0.5 m clearance tracks reliably, the 0.75 m
  doorway strip threads at ~0.35 m.
- **Residuals:** (1) the 2 remaining validation scenes (redteam F12 wants
  >=3 total; jingfan + livingroom_1 = 2) need a scene swap + sim relaunch,
  blocked this session on restart permissions - either add a docker
  restart/exec-kill allow rule or swap manually per ubuntu_setup S5;
  (2) prior residuals unchanged (rates issue to file, RVIZ eyeball,
  reboot, LLM keys, Docker Hub).

**Next:** remaining two validation-scene bags after relaunch permission,
then Gate 4 - real perception (unchanged from session 16 Next).

## 2026-07-18 (session 16, continued) - Gate 4 build side COMPLETE (live tests deferred)

All Gate 4 work that can be done without the sim/GPU is done, verified,
and pushed (bag recording ran in a parallel session throughout; the
containers, topics, and GPU were never touched by this session's agents).

- **Detector implemented** (87fe3b7): real GroundingDinoDetector.__call__
  (lazy torch, batched 4-tile forward + per-tile fallback, precision
  constructor>env>default with fp16-on-CUDA), vision_encode.py JPEG
  encoder wired into CP2/CP3/CP5 seams. PerceptionPipeline seam was
  already wired (87c16b3). 37 tests. Verifier confirmed 5/6 claims and
  REFUTED unclamped box math -> fixed below.
- **Weights baked** (9c150c5): image 6.76 GB with SwinB checkpoint
  (938 MB) + matching SwinB config (GDINO_CONFIG_PATH - without it the
  SwinT fallback would state-dict-mismatch) + hidden bert-base-uncased
  fetch pre-baked (441 MB) with HF_HUB_OFFLINE/TRANSFORMERS_OFFLINE
  runtime guards; PIP_CONSTRAINT numpy==1.26.4 (opencv>=4.12 trap);
  build-time assertion layer. Running containers untouched - new image
  waits as docker-ai_module:latest.
- **#34 fixed + clamp** (this commit): prompt refresh in HeadState.bind
  (shared seam, fires once on plan latch, question nouns + 116-noun
  standing vocab); tile-box clamping with inversion reorder. 15 tests.
  Both independently verified (e2e repro incl. adversarial clamp cases).
- Full gate now **1157 passed, 46 skipped**.

**DEFERRED - post-recording live checklist (run when bag session done):**
1. Swap container to new image (docker compose up -d ai_module from
   /tmp/fork-clean/docker) - watch boot log for perception=True and
   GDINO paths resolving to the baked SwinB config/checkpoint.
2. GPU load test on the 4060 (cu13 torch wheels vs host driver 595.71).
3. fp16 VRAM measurement vs 8 GB (architecture §6: 10-14 GB on 4090).
4. Batched forward vs per-tile fallback against the REAL model (the
   captions-list batch call is the one unverified design assumption).
5. Offline check: from_pretrained resolves from baked cache network-less.
6. RVIZ smoke: object_reference question -> Marker on the right object.
7. Clean-clone packaging gate: docker compose up --build from a fresh
   fork clone (§7a definition of done).
8. VLA_DETECTOR=grounding_dino must be set in compose env for the swap
   (default none = stub index + SUBMISSION-BLOCKER boot line).

Gate 0 residuals: LLM API keys, Docker Hub account, host reboot.
Open issues: #30 (rates), #33 (confidence gating - needs adjudication).

## 2026-07-18 (session 17, continued) - remaining validation scenes recorded; F12 scene bar met

- User added docker allow rules (sg docker */docker restart/exec/cp) ->
  restarts unblocked; scene swap + relaunch cycle now fully scriptable
  (swap via docker cp per ubuntu_setup S5, relaunch via
  system_simulation.sh with DISPLAY=:1 + RMW exported).
- **Three more bags, all module-driven** (fresh ai_module boot + numerical
  question -> sweep + frontier exploration from spawn), all harness-
  verified full-duration:
  - `office_1_q1` (329.5 s, 4.29 Hz): coverage x[-0.36,5.29]
    y[-4.18,0.43], 22.2 m - the best eval-realistic bag so far.
  - `loft_q1` (329.5 s): degraded-rate specimen - loft renders at
    1.45 Hz on this box, exploration starved, robot stayed near spawn
    (7.6 m). Kept as low-rate stress case.
  - `japanese_room_q1` (329.6 s, 3.19 Hz): x[-2.12,1.20] y[0,3.32],
    9.7 m; low-furniture geometry.
- **Redteam F12 scene bar met**: overhead-tunables validation data now
  spans 4 distinct scenes (jingfan + livingroom_1 + office_1 +
  japanese_room). Actual validation run against these bags is the next
  step of the hardening backlog item.
- **Rate finding for the to-be-filed rates issue**: sensor rate is
  per-scene render cost (4.29 / 3.7-3.85 / 3.19 / 1.45 Hz across
  office_1 / livingroom_1 / japanese_room / loft) - all below the ~10 Hz
  contract; loft is pathological. Eval-machine (4090) presumably faster.
- Sim left with japanese_room installed. Container bag copies deleted
  after verified host copies (disk back to ~79G free... ~21 GB in
  data/sim_bags).

**Next:** run the overhead-tunables validation against the 4-scene bag
set (hardening backlog), file the rates issue with the per-scene table,
then Gate 4 - real perception (unchanged).

## 2026-07-18 (session 17, continued) - overhead multi-scene validation DONE (H13/F12)

- Executor built `tools/overhead_validation.py` (+20 tests, `pytest tools`
  50 passed; src/ untouched, fast tier clean): replays a bag through the
  live overhead wiring against 3 config variants in lock-step, judges the
  final OVERHEAD layer against object_list.txt (rotated-rect footprints)
  + traversable_area.ply. Ran all 5 sim bags -> reports/
  overhead_validation_2026-07-18/ (per-bag JSON + summary.md).
- **Verdict: jingfan-fitted tunables hold cross-scene.** Flags 1.2-3.7%
  of observed cells tracking furniture density; 0 false flags on open
  floor (caveat: that GT under-tests wall-mounted objects); known-
  overhang hit 26-45% aggregate on healthy scenes, bimodal per object.
- **Issues filed:** #36 vehicle_sensor_height 0.60 vs measured 0.75-0.78
  on every sim scene (fallback-only blast radius, fix before sparse-
  terrain runs); #37 overhang misses (livingroom_1 shelf 73 = 0% in both
  bags; loft misses confounded by 1.45 Hz rate -> #30). min_points_per_
  cell is the dominant sensitivity lever (3->2: +36..273 cells/bag).
- Issue-tracker cleanup while filing rates data: per-scene table + loft
  starvation evidence commented onto #30; dups #35/#28/#29 closed.
- calibration.md boxed caveat + backlog H13 flipped to validated-with-
  pointers. Exploration-side asymmetry remains open/deferred.

**Next:** Gate 4 live checklist (build side complete per session-16 entry:
swap docker-ai_module:latest in with VLA_DETECTOR=grounding_dino, GPU/VRAM
+ batched-forward + offline-cache checks, RVIZ smoke, clean-clone packaging
gate). Gate 0 residuals: LLM keys, Docker Hub, host reboot.

## 2026-07-18 (session 17, continued) - local-LLM bridge plan written

- New constraint from user: no cloud LLM keys until early August, and no
  SoC cluster. Plan: docs/local_llm_plan.md.
- Recon confirmed the provider layer needs ZERO code changes: the "local"
  slot (VLA_LLM_LOCAL_KIND=openai + BASE_URL + MODEL) speaks OpenAI wire
  incl. vision; ladder timeout/repair/regex-floor already bound a weak
  local model's damage.
- Strategy: Ollama serving two regimes on the 8 GB 4060 - 3-4B Q4 VLM for
  live runs (GPU shared with Unity+GDINO), 7-8B Q4 VLM for offline parse
  battery + checkpoint replay against recorded-bag panoramas. Phases:
  0 serving+latency baseline -> 1 conformance -> 2 offline value
  measurement (per-checkpoint enable matrix vs regex floor) -> 3 live
  integration (post Gate-4 checklist) -> 4 August switchover, local slot
  demotes to permanent tier-3 failover (+ decision item: bake local model
  into submission image as eval-day dark-network fallback).

**Next:** Gate 4 live checklist, then local-LLM Phase 0 (both need the
sim box; order per plan).

## 2026-07-18 (session 17, continued) - Gate 4 detector UNBLOCKED live; host-first loop adopted

- **New standing rule (user): host-first iteration** - everything runs
  natively on this box by default; docker only at key checkpoints
  (CLAUDE.md standing rule 5, ubuntu_setup S8b). Enablers landed:
  tools/setup_host_node.sh (ROS Jazzy check incl. separate CycloneDDS RMW
  gate, image-pinned venv, weights extracted FROM the image) and
  tools/run_host_node.sh. Proved out immediately: three fix-verify cycles
  in ~20 min that would each have been a 20-40 min image rebuild.
- **Gate 4 detector chain fixed through three layers**: #38 transformers
  pin 4.57.6 (get_head_mask removed at 5.0; executor-verified per-version)
  -> #41 model never moved off CPU by load_model -> fp16 via
  torch.autocast at forward time instead of .half() (GDINO's internals
  keep fp32-only islands; hard-half broke both forward paths). Fixed in
  2a8fc85 (+2 regression tests, detector suite 37 green).
- **Live evidence (host node, japanese_room)**: perception=True, ZERO
  perception errors across two full question runs (was 463/4min), real
  GPU inference, VRAM steady 3.7 GB peak alongside Unity -> fp16 fits the
  8 GB dev card with ~4.4 GB headroom. Robot explored and answered with a
  real non-stub marker.
- **New defect #42**: the grounded answer picked the wrong object
  (marker 5.4 m from teapot GT, nearest GT = column) - accuracy, not
  mechanics; queued for the calibration loop. #39 (retry backoff) also
  re-evidenced by the pre-fix error storm.
- Issues #38-#41 filed/closed this stretch; checklist steps 2/5/7 PASSed
  earlier by executor (CUDA cu13 OK, offline cache OK, packaging build OK
  with the #40 uncommitted-payload trap noted). The in-container
  checkpoint rerun awaits the still-running image rebuild.
- Local-LLM Phase 0 DONE (Ollama 0.32.1 user-space, qwen2.5vl 3b/7b,
  8k-context requirement found, latency table in local_llm_plan.md:
  3B passes all four call shapes with 10x headroom; 7B vision over-cap
  under load, re-baseline pending). In-image serving promoted to a
  Phase-3 REQUIREMENT (user): bake Ollama + 3B into the ai_module image.

**Next:** (1) in-container checkpoint rerun when the rebuild lands;
(2) #42 grounding accuracy + #39 backoff; (3) local-LLM Phase 1-2
(conformance + parse battery + checkpoint matrix vs recorded bags);
(4) Phase-3 in-image Ollama bake + S7a gate; (5) commit staged fork
payload (#40) before the pending host reboot.

## 2026-07-19 — Local-LLM Phase 1-2 (parse half) — task record T13

Started and wrapped T13 (`docs/tasks/T13-local-llm-phase1-2/task.md`):
Phase 1 conformance tool + 10-question sample (6/10 valid, below the 8/10
bar) and Phase 2 parse battery over all 75 training questions (confounded
by unexpected full sim-stack contention on the dev box — 2/75 reached the
LLM tier; re-run on a quiet box recommended). No Ollama wire quirk found,
so no adapter change. Issues #44 (parse_tier mislabeling on local-only
config), #45 (repair prompt not schema-guided), #46 (client-only timeout
vs Ollama's single-concurrency server) filed. Next: quiet-box Phase-2
re-run; Phase-2 vision-checkpoint half; Phase-3 in-image bake.

## 2026-07-19 (session 17, continued) - #42 fixed+verified, LLM fixes, GPU wedge -> reboot gate

- **#42 dual-caption + anchor-floor fixes landed (959a74f)**, offline-
  validated through the new code path (19/19 on-bearing teapot detections
  at the 0.25 question-pass threshold). Live rerun: mechanics perfect
  (0 errors, 3.6 GB VRAM, dual-pass on GPU), answer still wrong via a
  DIFFERENT path -> #43 filed (false-positive target instances at the
  recall threshold + exploration never gained the decisive vantage;
  needs #33-style expected-points adjudication + instrumented debug run
  = calibration loop item, not a tonight bug).
- **Env truths learned the hard way**: (1) sim sensor streams can wedge
  (Unity/bridge zombie - processes alive, topics dead) after hours +
  CPU storms; relaunch fixes. (2) qwen2.5vl:7b at 8k ctx NEVER fits the
  7.6 GiB card (llama-server offloads -> ~2 tok/s); 3B is the only
  local model for this box. (3) the dGPU itself wedged into P8/210 MHz
  w/ 8.5 W SW power cap (52 C, plugged, performance profile - unfixable
  in software) -> **HOST REBOOT REQUIRED**, everything GPU-bound gated
  on it. (4) The 4 h docker rebuild died without completing (image tag
  unchanged); pip-layer cache state unknown until the next build.
- **LLM ladder fixes** (agent-committed): #44 tier mislabel, #45 schema-
  guided repair prompts, enum synonym normalization (furthest_from->
  farthest_from etc.). Conformance was 6/10 pre-fix; clean post-fix
  rerun PENDING REBOOT (all latency data since ~00:00 invalid due to
  the wedge). #46 (client timeout leaves server generating) open as a
  Phase-3 design item.
- **Phase-3 in-image bake being AUTHORED now** (no build): Dockerfile
  Ollama v0.32.1 + baked 3B blobs + serve/prewarm launch wrapper +
  compose env, in the IN-REPO fork copy (docker/ai_module_fork - /tmp/
  fork-clean dies with the reboot; recreate via upstream clone +
  sync_to_fork.sh; its commit 6543178 was the #40 fix, redo after
  restage). Post-reboot runbook: reports/local_llm_phase3/
  BUILD_AND_VERIFY.md (authored by the same task).

**POST-REBOOT RUNBOOK (execute in order):**
1. Verify GPU unwedged: nvidia-smi clocks ~boost under load, then
   restart Ollama (ubuntu_setup §8a one-liner) + `ollama ps` sanity.
2. Rerun LLM conformance (expect >=8/10 post-fixes) + full 75-question
   parse battery on the 3B (tools/llm_conformance.py /
   tools/llm_parse_battery.py; quiet box) -> finalize per-QType
   recommendation in reports/local_llm_phase2/.
3. Vision-checkpoint replay (CP2/CP3/CP5 vs recorded-bag panos + GT) ->
   completes the Phase-2 enable matrix.
4. Restage fork (/tmp/fork-clean): upstream clone + sync_to_fork.sh +
   commit (redo #40's commit).
5. Rebuild docker-ai_module (now incl. #41/#42/#39 src + Phase-3 bake if
   its authoring landed) - schedule for an idle stretch, download may
   restart from zero. Then the IN-CONTAINER checkpoint: relaunch sim,
   teapot run, VRAM/batched/marker evidence, BUILD_AND_VERIFY.md steps,
   §7a clean-clone gate.
6. Then: #43/#33 calibration adjudication; Gate-0 residuals (LLM keys
   early Aug, Docker Hub account).

## 2026-07-19 (session 18) — post-reboot runbook step 2: T13 clean rerun

Executed runbook step 2 (T13, `docs/tasks/T13-local-llm-phase1-2/task.md`
"Post-reboot clean rerun" section) on the healthy post-reboot box (GPU
P0/2505 MHz, qwen2.5vl:3b fully GPU-resident, tg 80-88 t/s). Conformance:
still FAIL at 7/10 (bar 8/10; up from pre-fix 6/10), stable across 2 runs,
2 root causes with raw model output analyzed in
`reports/local_llm_phase1/conformance.md`. Parse battery: unconfounded,
all 75 questions in `reports/local_llm_phase2/parse_battery.md`; LLM
tier reached 77% of calls; agreement 93%/80%/63% (numerical/OR/IF);
recommendation KEEP FLOOR for all three QTypes, reasoning grounded in an
18-divergence root-cause analysis. Filed #47 (goto/via_near mislabel),
#48 (flattened stacked relative clauses), #49 (spurious avoid
fabrication) for adjudication — model-content defects, no src/ changes.
**Next:** step 3 (vision-checkpoint replay) per the runbook.
---

## 2026-07-18 (session 16) - task milestones

- Started + finished task: T13 issue-sweep - record at
  `docs/tasks/T13-issue-sweep/`. Branch `fix/issue-sweep`; all nine
  open issues (#10-#13, #15-#19) fixed, two verifier gates CONFIRMED,
  numerical TRUE accuracy 13/15 -> 15/15, full gate exit 0. Follow-up
  issues filed: #20 (pre-existing hb2 speaker OR mismatch), #21
  (anchor-tier latent caveat). PR to main opened closing all nine.
- Environment fix: git-ignored `upstream/` clone found empty (casualty
  of session-15 cleanup) - re-cloned Yuxin916/CMU-VLN-Challenge-2026
  (shallow); parsing tests green again. No ubuntu_setup.md change
  needed (no new deps; upstream clone step already documented).

**Next:** merge the T13 PR; then the residual queue unchanged from
session 15 - OR/IF image answer-key visual transcription (arch-F3),
Gate 0 execution, local-VLM tier posture (arch-F5), SoC cluster
detector benchmarking (arch-F2); #20/#21 now head the code follow-up
queue. New color knobs (colour_dominance_floor, dark_luma_max,
light_luma_min) join the CV-sweep surface.

---

## 2026-07-18 (session 16b) - task milestones

- Extended task T13 per user: #20 + #21 fixed on the same branch -
  record at `docs/tasks/T13-issue-sweep/` (root causes, adjudications,
  verifier evidence). #21: bare-noun anchor tier-merge (cbfd548).
  #20: diagnosis overturned the issue premise (GT=91 spurious, correct
  target 122); fixed as trailing-superlative parse surfacing + 
  superlative-aware GT matching (554f927), OR mean_iou 0.875 -> 1.0
  over 6 scored, numerical 15/15 + IF byte-identical. Combined
  verifier gate CONFIRMED. Battery reference rebaselined:
  `reports/gt_battery_T13b_2026-07-18`. New issue filed: #23
  (pre-existing leading-superlative parse quirk, zero corpus
  occurrences). Stale LibreOffice lock removed from
  data/vla3d/Unity/home_building_2/.

**Next:** PR #22 review (pr-review-toolkit), then merge - closes
#10-#13, #15-#21. Residual queue: #23, OR/IF image answer-key
transcription (arch-F3), Gate 0 execution, local-VLM tier (arch-F5),
SoC detector benchmarking (arch-F2).

- (session 16b cont.) PR #22 reviewed via pr-review-toolkit (5 agents):
  no merge blockers; follow-up polish applied (86f0c41, fast tier 1100
  passed); architectural findings filed as #24-#27. PR ready to merge.

## 2026-07-19 - task milestone (branch if1-walls, worktree)

- Started + finished T14 if-wall-realism: derived interior walls for
  the gt_battery IF mirror costmap from each scene's
  `traversable_area.ply` (flag-gated, `--no-walls` escape). Record at
  `docs/tasks/T14-if-wall-realism/`. Full battery re-run:
  `reports/gt_battery_walls_2026-07-19/`. Fast + full (`-m ""`) test
  gates green.

## 2026-07-19 (session cont.) - task milestone (branch if2-corridor, worktree)

- T15 if2-corridor-threading (issues #52 + #51). Fixed #52: tighter
  `WALL_FIT_MAX_RESIDUAL_M=0.8` wall-derivation gate (vs the 1.0 scoring
  gate); livingroom_1's marginal-fit flip reverts, IF headline
  0.100 -> 0.117 (`reports/gt_battery_fixA_2026-07-19/`), registered in
  docs/calibration.md. Then #51: per-leg traced 3 failing corridor
  questions BEFORE fixing (`reports/if2_corridor_trace_2026-07-19/`);
  issue's "one defect" premise overturned — found and fixed THREE
  distinct defects (rubric anchor-distinctness mismatch vs
  InstructionHead's own enforcement; resolver tier-priority loss
  letting an exact label match lose to a head-noun cousin by instance
  id — a real nav defect, not just scoring; pinch-corridor fallback
  never tried on a total direct-astar miss, and never able to open a
  real gate narrower than 2x vehicle_radius since it only ever ADDED
  blocking). Threading violations 9 -> 7 (target 0-2 NOT reached),
  headline 0.117 -> 0.150, numerical 15/15 + OR 6/6 unchanged
  throughout. Remaining 7 trace to a 4th, broader defect (GT
  obstacle-stamping fidelity for architectural "wall" labels) filed as
  #53, left #51 open (commented with status) rather than falsely
  closing it. Record at `docs/tasks/T15-if2-corridor-threading/`. Fast
  + full (`-m ""`) test gates green.

## 2026-07-19 (session 17, continued) - issue sweep to 5 open; IF 0.117->0.150 honest

- **Full issue sweep**: 26 open -> 5 (3 parked 3B-model defects for the
  August cloud re-test, #38/#30 close at the in-container checkpoint).
  Included discovering + merging the parallel session's unmerged PR-22
  branch (fixes #10-#21) before wave-1 agents finished duplicating it.
- **Post-sweep battery: zero regressions** (15/15 numerical, 6/6 OR
  scoreable, IF unchanged) - reports/gt_battery_postsweep_2026-07-19.
- **Generalization protocol** encoded as standing rule 6 + calibration.md
  section (user-raised overfitting concern): scene-level holdout,
  spec-over-sample, generated held-out questions. Highest-risk item
  flagged: the single-scene GDINO 0.25 question threshold.
- **IF improvement block (user-approved items 1-4) DONE**:
  IF-1 wall-aware offline costmap from traversable_area.ply (honest
  baseline 0.117->0.100, frame-fit-guarded; livingroom_3 confirmed
  data-unfittable); IF-3 pinch/unknown-cost seams wired (ledger
  updated); IF-4 leg-ordered anchor-seeking exploration (live-regime,
  offline no-op as expected; avoid-nouns consumption question -> #50);
  IF-2 became a trace-driven fix: #52 wall-fit gate 0.8 m (0.100->0.117)
  + three corridor defects fixed (rubric anchor-distinctness, resolver
  tie-break losing match-tier priority - a REAL nav bug, pinch fallback
  never retried/never able to open narrow gates) -> **IF 0.150, violations
  9->7**. Residual root cause = GT obstacle-stamping fidelity -> #53;
  #51 stays open, honestly.
- VLA-3D Unity groundtruth (2 GB) now local (data/vla3d/) - battery runs
  on this box.

**Next:** vision-checkpoint enable matrix (chunks running), in-container
checkpoint (closes Gate 4, #38, #30, verifies Phase-3 bake), then #53
(GT stamping fidelity) and #33/#43-family calibration items.

## 2026-07-19 (session 18) - T16 if53-obstacle-stamping: 3/7 legs traced to a
## room-scale AABB stamping bug, fixed; new #54 masked defect found; #53 status

- Traced all 7 remaining threading violations (fixB baseline) per-leg (see
  `docs/tasks/T16-if53-obstacle-stamping/task.md`). 3 (home_building_1,
  home_building_2, studio) trace to room-scale architectural GT AABBs
  ("wall"/"unknown"/"floor" aggregates spanning a large fraction of the
  room in both axes, floor-level) stamped as solid floor-to-ceiling
  obstacles by `_synthetic_from_gt`, sealing the corridor gate — the
  EXACT `home_building_2` "wall" id 126 instance #53's issue text
  named. 1 (hotel_room_2) is a genuinely different shape: a real
  duplicate-labelled "bed frame" furniture item at the gate (not an
  architectural over-stamp). 3 (arabic_room, hotel_room_1,
  livingroom_1) are pre-existing, distinct defects already correctly
  scoped out by #51/#53 (reachability, furniture crowding, degenerate
  bare-noun gate) — untouched.
- **Fixed** (`core/runner/gt_battery.py`): `_synthetic_from_gt` skips
  raw-solid stamping for a floor-level GT instance whose footprint
  spans > `ARCHITECTURAL_AABB_ROOM_FRACTION` (0.3) of its OWN scene's
  room bounds in BOTH x and y. Purely geometric, self-referential per
  scene, no label list. Swept over all 15 GT scenes: zero real
  furniture instance clears the bar (closest: hotel_room_2's bed
  frame, 0.28 on its short axis); every instance that does clear it is
  labelled wall/floor/unknown. 6 new tests
  (`src/tests/runner/test_gt_battery.py`).
- **IF headline unchanged (0.150, 7 violations)** — post-fix tracing
  (instrumented astar/path_crosses_gate/pinch wrapper) found the 3
  shape-(a) gates are no longer sealed by a stamped obstacle (fix
  works as intended) but a DIFFERENT, previously-MASKED planner defect
  now blocks all 3: `_pinch_costmap` has no exemption for the
  vehicle's own start position (can newly-block its own start cell
  when within `pinch_disc_m` of the gate) and its forced 1 m corridor
  band can seal the only real route through other furniture. Filed as
  **#54** with full trace evidence. Numerical 15/15 + OR 6/6
  unchanged; IF Frechet/coverage secondary diagnostics improved
  (6.764m->4.225m, 38%->47%) - `reports/gt_battery_if53_2026-07-19/`.
- Shape-(b) case (hotel_room_2) traced but not fixed — deferred
  (needs a shared rubric/planner helper + dedicated regression
  coverage, out of this session's budget).
- #53 left open with a status comment (root cause fixed and verified,
  headline metric didn't move because fixing it unmasked #54) rather
  than a false `Fixes #53`.

**Next:** #54 (pinch-corridor self-block/seal defect), then hotel_room_2's
deferred shape-(b) midpoint-nudge, then vision-checkpoint/#33/#43 items.

## 2026-07-19 (session 17, close) - second reboot gate; state saved

- **Checkpoint (clean run) verdicts**: in-image Ollama bake WORKS
  structurally (binds, serves text+vision in-container); §7a clean-clone
  gate PASS; two real packaging bugs found and FIXED (#55 compose env
  VLA_DETECTOR, #56 missing openai SDK - merged 8c9682d, fork synced);
  KEEP_ALIVE-60m coexistence OOM evidence recorded (compose now pins 5m
  dev override). #38/#30 closed on evidence (post-reboot rates 5.8 Hz =
  1.8x pre-reboot).
- **GPU RE-WEDGED under load hours after the reboot** - recurrent, not
  stale-driver. Standing recommendation: NVreg_DynamicPowerManagement=0
  modprobe hardening (user sudo) - can be installed after this reboot to
  apply at the next one. All checkpoint latency data under the wedge is
  suspect; structural results stand.
- Disk crisis mid-rebuild (93->99%): build cache 37.7 GB across 3 image
  generations. Pruned to 12 GB keep-storage + deleted the disqualified
  qwen2.5vl:7b (5.6 GB) + spent archives -> 29 GB free. RULE: run
  `docker builder prune --keep-storage 12GB` after every image build.
- #54 (pinch start-seal): fix drafted in worktree if54-pinch-seal
  (UNCOMMITTED planner.py changes - worktree survives reboot) but the
  fix itself HANGS tests/heads/test_instruction.py (~15th test) -
  deadlock/unbounded relax loop; the agent's own "stalled pytests" were
  this bug. Next session: fix non-termination (hard iteration cap +
  terminal fallback), then battery (target: 3 unsealed legs thread,
  IF > 0.150).
- Image rebuild with #55/#56 fixes was mid-flight at reboot - rerun from
  /tmp-restaged fork (upstream clone + sync_to_fork.sh + commit; cached
  layers make it fast on today's pipe).

**POST-REBOOT RUNBOOK v2:**
1. Verify pstate under load (P0-P3) BEFORE any measurement; optionally
   install the runtime-PM modprobe hardening first.
2. Restage /tmp/fork-clean (clone upstream -> sync_to_fork.sh -> commit
   -> commit packaging state), rebuild image, `builder prune` after.
3. Boot stack from compose (stock env now correct per #55), run
   BUILD_AND_VERIFY steps b-e clean: prewarm OK, ladder parse_tier=local,
   KEEP_ALIVE=5m coexistence (batched forward should hold), teapot run.
   -> closes tasks 6+8 / Gate 4 formally.
4. if54-pinch-seal worktree: fix the hang, battery, merge -> #54, then
   re-evaluate #51/#53 chain.
5. Then: #50 adjudication, LLM-route A/B (holdout rules), Aug 3 MVS prep.

## 2026-07-19 (session 17b, post-reboot) - Phase 3 verified; #54 closed; #59 found; THIRD WEDGE

- Hardening v1 SILENTLY OVERRIDDEN: Ubuntu's /lib/modprobe.d/nvidia-runtimepm.conf
  (=0x02) lexically outsorts nvidia-disable-rtd3.conf. Fixed with zz- prefix
  (1003fdb); NEEDS: sudo tools/gpu_hardening/install.sh rerun + reboot to arm.
- Fork restage v2 (correct procedure: upstream/CMU-VLN-Challenge-2026 clone +
  docker/ai_module_fork/sync_to_fork.sh + compose copy). Found+fixed #57
  (env_file hard-required broke clean-clone boot). Image rebuilt; 28.1 GB is
  CORRECT (runbook's 11-12 GB bar was a stale pre-build estimate).
- Phase 3 VERIFIED for real (task 8 closed): keep-alive 5m pin works, prewarm
  completes ~25-40s (wrapper cap 20->60s, dff9a57), ladder parses via baked
  LOCAL tier (ollama GIN log: adapter /v1/chat/completions 200s at latch).
  Stale "DESCOPED" log line fixed. NOTE: image predates dff9a57 - rebuild
  before next checkpoint run.
- #54 closed honestly (merge 7d5eb8e): bounded pinch relax in, hang never
  reproduced (was the wedged host), battery effect ZERO.
- **#58 found+fixed** (478ecef): IF rubric scored pose-wise arrival over
  undensified waypoints. Real latent bug, unit-proven - and STILL zero battery
  effect, which yielded **#59: routes miss GT leg goals by >0.8m even on plain
  gotos (credit 0.161; Frechet 4.17m/coverage 49% = right rooms, wrong stop
  points)**. IF headline 0.150 invariant across four code states. All evidence
  in reports/gt_battery_{if54_post,if54_densified,main_densified}_2026-07-19.
- **THIRD WEDGE mid-teapot-run** (P8/210MHz, SW power cap, all ollama calls 500
  at the 20s cap): this boot ran the overridden hardening. Teapot/Gate-4
  evidence INVALID; task 6 stays open, blocked on hardened reboot.
- USER DIRECTIVE: next session = fix #59 first, then remaining open issues
  (#53/#51 re-triage post-#59; #50 adjudication; #47-49 local-parse defects).
- Stack STOPPED at session end. Worktree if54-pinch-seal merged; cleanup of
  worktrees pending (2 stale agent worktrees at b507eb9 unchecked).

## 2026-07-19 (session 17c) - hardened GPU verified; in-container drive PROVEN; issue sweep

- Hardened reboot VERIFIED: DynamicPowerManagement=0 live, clocks ramp under
  load (P0/2490 under sim+module), no wedge through multiple full-stack runs.
- **In-container autonomy PROVEN end-to-end**: latch -> local-tier parse ->
  60s ORIENT -> EXPLORE_EXECUTE with /way_point_with_heading at ~0.58 Hz ->
  DONE at budget. The earlier "no waypoints" scare was OUR observation error
  (Pose2D grep shape + windows misaligned with the 60s ORIENT + post-budget
  checks) - no code defect. Diagnosed via 3 hot-patch cycles in the container
  (image untouched; final evidentiary run recreates the container clean).
- Coexistence evidence: 3B + GDINO + Unity + RVIZ resident at 7.2 GB stable,
  P0 sustained, prewarm 12-44s (fits new 60s cap), parse 7-10s under load;
  separately captured the 5m-keep-alive OOM-then-recover sequence with #39
  backoff working as designed.
- Issue sweep (parallel executors, disjoint worktrees):
  #59 FIXED+merged (resolver ranking; credit 0.161->0.172; remainder split
  into #61/#62), #50 adjudicated+dead API removed, #51 closed superseded,
  #57 fixed (env_file optional), #58 fixed earlier, #60 filed+scoped
  (silent seams + unobservable answer publication), #47-49 root-caused with
  normalize.py + 15 unit tests (battery rerun in flight, worktree
  llm-parse-47-49). Stale pre-reboot agent worktrees archived to
  stale-archive-* branches and removed.
- ubuntu_setup §7a as-built gotchas + BUILD_AND_VERIFY 28 GB size bar
  corrected (418bb89, 5c9e941).
- Pending to close task 6: final clean-container teapot run with armed
  answer-topic subscribers (script staged in scratchpad) - queued behind the
  #47-49 battery rerun for GPU. User directive recorded in memory: no-idle
  orchestration rules (never end a turn purely waiting; pausing agents are
  stopped agents; deadline+probe every wait).

## 2026-07-19 (session 17d) - GATE 4 CLOSED; issue sweep concluded

- **Gate 4 / task 6 CLOSED** on clean-container evidence (reports/gate4_final/
  evidence.log): unmodified image, hardened GPU, sim live -> prewarm OK ->
  local-tier parse (2/2 ollama 200s) -> 44 waypoints driven -> ANSWER MARKER
  published (ns "dining table", CUBE, map frame, full extents). VRAM trace
  shows the 8 GB time-sharing design working: 4.3 (3B) -> 7.5 P0 (GDINO+3B
  coexist, no OOM) -> 3.6 (3B released at 5m keep-alive). Semantic note:
  grounded the table, not the teapot on it - grounding accuracy (#42/#61
  lane), not packaging.
- #53 closed (scope exhausted; already fixed in 092178b, re-verified per-gate;
  remnant filed #63 - queued behind #61/#62's surface). #60 closed (injected
  logger: transitions, swallowed-seam tracebacks first+every-50th, answer
  publication + waypoint-count logging, events tail on DONE; merged e037990).
- Parse-battery hang caught by deadline-probe (40 min at 0% CPU, stacked host
  serves); killed, agent stood down cleanly and self-invalidated a
  contention-tainted run (timeout-cluster fingerprint). Clean rerun next.
- CLAUDE.md standing rule 7 added (no-stall orchestration) per user directive;
  same rules in global memory.

## 2026-07-19 (session 18) - T17 if66-arrival-stamping wrapped up

- Started/finished T17 (issue #66, worktree if66-arrival-stamping): traced
  all 26 #61/#62 "goal plausible, drive never arrives" legs; shipped a
  directional own-anchor-footprint push fix to `_nearest_free_goal`, rejected
  a door/door-frame pass-through candidate on negative evidence. IF headline
  0.1667 -> 0.2000, ordered-leg credit 0.1889 -> 0.2222, numerical/OR
  unchanged. See docs/tasks/T17-if66-arrival-stamping/task.md,
  reports/gt_battery_post66/.

## 2026-07-19 (session 19) - commit conventions codified

- User callout: commit-message naming was all over the place (~50 ad-hoc
  prefixes, 82 of ~200 commits unprefixed). Added "Commit conventions" to
  CLAUDE.md: `scope: subject` format, fixed scope vocabulary mapped to
  `src/core/` package names + process scopes, one-scope rule, legacy-prefix
  reading key. History left untouched; applies from here on.
- Branch conventions added to CLAUDE.md (`<type>/<slug>`, issue number
  first in slug; `main` only long-lived branch; harness branches pruned
  after harvest). Cleanup: deleted merged locals if59-goal-placement +
  2 worktree-agent-*; renamed stale-archive-agent-* ->
  archive/pre-reboot-draft-{a37b4db,a5997e8}. Remote prune of 3
  fully-merged origin branches (claude/x2, fix/issue-sweep) blocked by
  session permissions - left for manual `git push origin --delete`.
  origin docs/current-architecture kept (2 unmerged doc commits, 14 Jul).
- Wrapped up T18 architecture-refresh: as-built `architecture/` folder
  (README + ch 01-10) landed; 11 Jul snapshot branch archived. See
  docs/tasks/T18-architecture-refresh/task.md.

## 2026-07-19 (session 17e) - yardstick redesign landed; IF 0.378

- **#70 layer-1 redesign MERGED** (16dac66, full gate green): derived arrival
  tolerance 1.746 m (vehicle 0.4 + grid diag/2 + measured p95 fit residual
  1.2256 + 0.05 margin; every term cited in core/groundtruth/arrival.py,
  single source consumed by scorer AND instruction head), stop vs pass-by
  closest-approach semantics, ceiling probe reworked to per-leg segments.
- **IF headline trajectory today: 0.150 -> 0.167 (#59) -> 0.200 (#61/#66) ->
  0.378 (#70)**; ordered-leg credit 0.161 -> 0.439; honest ceiling now
  59/72 = 0.819 per-segment (0.861 whole-traj; #69's artifact ~4 pts).
- #69 closed as corrected-map negative result: ceiling classification's
  per-rule predictions were whole-trajectory artifacts (CAVEAT.md committed);
  resolver exonerated; one real zero-width-gate fix merged. #67 reverted on
  main after integration regression (+standalone/-integrated) - reopened,
  ruling delegated to the new probe; #68 subsumed by #70.
- Process hardening from today's failures (memory + CLAUDE.md rule 7):
  integrated-head measurement before close; heartbeat monitor (5 min) over
  all lanes; ownership sweep before any "no work left"; stall tally at 3.
- Parallel session T18 (architecture refresh) active on this repo - doc
  surface only, no src collision; pull-rebase before push is standing.
- DISPATCHED: integrated conversion+threading lane on the new yardstick
  (probe -> ranked fixes -> threading; stop at headline 0.55 / two dry
  fixes / 90 min). 0.8 verdict: inside the 0.819 ceiling with 0.019 slack -
  requires near-perfect conversion + zero penalties; re-verdict after this
  lane's probe.
## 2026-07-19 (session 20) - IF conversion probe (#71), no code shipped

- Fresh per-leg-segment probe of every non-in-order IF leg under the #70
  yardstick, off `reports/gt_battery_post70` (already HEAD's own evidence,
  credit 0.4389/headline 0.3778/tv=7 — no rerun needed). Full breakdown:
  reports/conversion_probe.md + .json. Cascade effects are negligible (1/39
  legs) — the ordered cursor never regresses on a miss, so ranking fixes by
  "legs unlocked incl. cascades" collapses to ranking by bucket size:
  structurally-unreachable 9/72, arrival-blocked 29/72 (the dominant, broad
  bucket — 13/15 scenes).
- Two fixes attempted for arrival-blocked, BOTH reverted after measurement
  (neither ships): (1) substituting `InstructionHead`'s own driven-toward
  goal for the rubric's independent goal recovered credit 0.44->0.79
  integrated, but per-leg diffs showed 4-12 m goal jumps — a different
  RESOLVED INSTANCE, not a projection refinement — rejected as a
  self-referential/gamed metric ("measures determinism, not truth"), same
  trap the codebase already flags for `pipeline_gt`. (2) Mirroring only
  `InstructionHead`'s salience tie-break in the rubric's own resolve had
  ~zero effect (credit unchanged, headline down on one more threading
  violation) — rejected, and it rules out tie-breaking as fix (1)'s
  explanation. Working conclusion: the residual gap is a genuine
  navigation-quality shortfall, not a scoring-geometry artifact. Stop
  condition (b) (two consecutive fixes fail to improve integrated numbers)
  reached; no further fix attempted this session.
- Issue #67 ruled OBSOLETE under the new tolerance (closed, comment with
  evidence) — the wide #70 tolerance band dwarfs `_nearest_free_goal`'s
  bounded push distance, so the raw-vs-pushed-goal failure mode #67 tracked
  cannot reproduce; remaining hotel_room_1 failures are arrival-blocked
  (#62), not goal-placement.
- Phase 3 threading trace: only 2/7 threading violations are actually
  attempted corridors (hotel_room_1 leg1, hotel_room_2 leg0 — arrive near
  the gate midpoint, never cross); the #64 nudge's tight quantization-only
  guard (miss < 1 grid cell) correctly does NOT engage for either (both miss
  by 6-12x that band) — no threading code defect traced. The other 5
  violations are corridors never approached at all (same arrival-blocked/
  structurally-unreachable legs already covered). No threading fix
  attempted.
- Filed #71 (rubric-vs-head anchor-resolve divergence, the unexplained
  4-12 m deltas from fix attempt 1 — a real resolve-outcome mismatch worth
  its own audit, separate from the tie-break already ruled out) and posted
  fresh-data comments on #61/#62 with the current bucket sizes. Integrated
  numbers UNCHANGED from `gt_battery_post70` (no code shipped). Next step:
  `core/nav` route-precision investigation for arrival-blocked (#62), and
  the #71 resolver-parity audit, before either bucket is touched again.

## 2026-07-19 — #72 drive-precision: BreadcrumbFollower Euclidean-lookahead fix

**Done (branch `if72-drive-precision`):**
- Rebuilt the arrival-blocked bucket from `gt_battery_post71`'s own
  `leg_probe`/`leg_outcomes` (post-#71 re-placement): 22 (later fell
  further as fixes landed) arrival_blocked / 11 structurally_unreachable /
  0 cascade, of 35 non-in-order legs of 72 total.
- Traced mechanism by comparing each arrival-blocked leg's OWN resolved
  navigation goal (`InstructionHead._legs[i].geom`) against the rubric's
  independent goal and the driven trajectory's closest approach: most
  legs' driven trajectory bottoms out almost exactly where the HEAD's own
  goal sits (resolver-instance divergence from the rubric's independent
  resolve — `core/geometry/toolbox` territory, frozen, #73's lane, not
  actionable here). A distinct sub-group had the head's own goal sitting
  CLOSE to the rubric goal (well inside tolerance) while the driven
  trajectory never got anywhere near it — a genuine navigation defect:
  `BreadcrumbFollower._select_crumb`'s "farthest LOS-clear point within
  `LOOKAHEAD_M`" bound used straight-line (Euclidean) distance from the
  CURRENT POSE to each candidate. On a route that runs out to a leg's own
  goal and then doubles back near its own earlier ground (e.g. returning
  toward a later corridor/leg), a point many real metres of travel away
  can sit Euclidean-close to the pose merely because the path folds back
  — the greedy selector then shortcuts straight past the entire
  out-and-back detour (and the leg goal at its tip), never actually
  driving there, even though the planned path correctly visited it.
- Fix 1 (`core/nav/planner.py` `astar`): stop the reconstructed path's
  final vertex at the goal CELL'S CENTRE when the requested continuous
  goal point was itself passable (unsnapped) — use the exact requested
  point instead. Targets the <0.1 m near-misses. In isolation: DRY (credit
  0.4944, headline 0.4333 — byte-identical to baseline; the near-miss
  legs' true bottleneck was the rubric-vs-head resolver divergence, not
  grid quantization). Kept staged, not reverted, because —
- Fix 2 (`core/nav/breadcrumbs.py` `BreadcrumbFollower._select_crumb`):
  bound the lookahead scan by arc length travelled ALONG the path from
  the follower's progress index (anchored on the pose's own remaining
  distance to `path[_idx]`, since `_idx` can legitimately lag/lead the
  pose), not by straight-line pose-to-candidate distance. Fixes the
  double-back-skip without touching the deliberate LOS-shortcut behaviour
  on genuinely open, non-looping stretches (arc length ≈ Euclidean there).
  Alone: credit 0.4944 → 0.5111, headline 0.4333 → 0.4500 (+2 legs
  reached: `hotel_room_2` leg0, `chinese_room` leg0). Combined with fix 1
  (now non-dry — the arc-length fix gets a `livingroom_4` leg close enough
  that the leftover cell-quantization slack decides it): credit 0.4944 →
  **0.5222**, headline 0.4333 → **0.4611** (+3 legs total), threading
  violations unchanged at 7, avoid violations unchanged at 0, numerical
  15/15 and object-reference 6/6 unchanged. Full fast tier green
  (`pytest -m ""` from `src/`).
- Residual: several arrival-blocked legs still show the same signature
  (head's own goal close to the rubric goal, driven trajectory never
  arrives) but with a bigger gap than fix 2 closes — traced to the SAME
  follower design gap at a more severe scale: `BreadcrumbFollower` has no
  concept of ordered-leg boundaries, so a short leg's own goal can still
  get skipped within a single lookahead window even after the arc-length
  correction, or via the separate (unfixed) `advance()` within/overshoot
  loop's blind Euclidean consumption from a stationary pose. A full fix
  needs the follower to be leg-boundary-aware (hard-stop each ordered leg
  before considering the next), which is a larger design change than this
  session's remaining budget could safely land and verify — filed as a
  follow-up issue rather than rushed.
- Reports: `reports/gt_battery_fix1_cellsnap` (fix 1 alone, dry, kept as
  evidence), `reports/gt_battery_fix2_arclookahead_only` (fix 2 alone),
  `reports/gt_battery_post72` (both fixes, final integrated state — new
  baseline for future #72/#73 work). Next step: the leg-boundary-aware
  follower redesign (filed issue), and the #71 resolver-parity audit for
  the resolver-divergence majority of this bucket (`core/geometry/toolbox`,
  #73's lane).

## 2026-07-20 (overnight, sessions 21+) - concurrent IF grind: 0.461 -> 0.539

- Max-concurrency overnight loop per user directive (3 write lanes + read-only
  probe; per-unit commits + incremental notes as limit-kill insurance;
  night heartbeat 15 min; per-beat issue intake sweep).
- **#74 leg-boundary awareness MERGED** (ec077cf): leg goals are
  waypoints-of-record with dwell-gated ceilings, existing stall watchdog
  reused. Credit 0.522->0.572, headline 0.461->0.511, zero regressions.
- **#75 salience gating MERGED** (95f48f8) as explicit correctness-over-
  score: parity divergences 3->1, both target legs to GT-correct instances;
  integrated 0.572->0.544 credit (two wrong-instance passes un-gamed; their
  arrival gaps transferred to #77). #73 landed earlier (parity 4->3).
- **#76 wall derivation: investigated-NEGATIVE** (evidence merged): the
  0.8 m residual gate is correctly calibrated (relaxation unblocks 0/6,
  regresses 3); home_building_1 x2 = genuine ~3 m mesh-coverage gap ->
  adjudicated structural. Probe-v2 merged (exact headline formula verified:
  rubric = max(0, credit - tv/n_legs)).
- **#77 dwell-release fix MERGED** (7c5582a): follower conflated the
  frame-fit credit tolerance (~1.75 m) with physical arrival; now uses its
  own REACH_M. Credit 0.544->0.600, headline 0.483->0.539, 4 legs, zero
  regressions. Integrated confirmation reports/gt_battery_main_post77a.
- Process: pgrep self-match bug bit a third time (dead wait-loop caught by
  user); bracket-pattern rule added to silent-death memory. One lane
  correctly reported and disregarded a mid-session prompt-injection-style
  framing of its own revert; verified state from disk.
- RUNNING: 77b gate pose-convergence lane (3 dwell-class corridors, traced
  lead: pose emission near dwell, not crumb selection).
- Trajectory: 0.150 -> 0.378 (#70) -> 0.433 (#71) -> 0.461 (#72) -> 0.511
  (#74) -> 0.483 (truer, #75) -> **0.539 (#77a)**. Ceiling 0.819.

## 2026-07-20 (overnight cont.) - threading unlocked: IF 0.589, tv 7->3

- **#79 MERGED** (77a73a9): goto/via_near pinch-overlay gate-crossing
  fallback (astar-None-only, usable_gate_point, 77b lookahead guard) +
  reinstated corridor crossing extension. Threading 7->3, credit 0.611,
  headline 0.539->0.589, zero regressions, full gate green, #54 guard
  intact. Integrated confirmation reports/gt_battery_main_post79
  (nume 15/15).
- #78 diagnosed (evidence merged, stays open, mechanism distinct): goal
  BFS floods the un-pinched costmap -> goto goals pocket-clamped after
  corridors; 6-leg blast radius; #79's probes proved the flagship case
  DISCONNECTED beyond the gate (structural) - residue re-verification in
  the 77c lane.
- Cross-lane relay worked as designed: #78's diagnosis redirected #79's
  design mid-flight (goal-side vs route-side), and #79's probes then
  refuted the goal-side half with geometry evidence - both recorded.
- DISPATCHED: 77c residual lane on the fresh post-79 table (blast-radius
  split, 3 remaining threading violations classification, structural
  ledger as key deliverable).
- Trajectory: 0.150 -> 0.378 -> 0.433 -> 0.461 -> 0.511 -> 0.483(truer)
  -> 0.539 -> **0.589**. tv 7->3. Ceiling 0.819 (structural ledger will
  refine the honest max).

## 2026-07-20 (overnight close-out) - IF 0.611; structural ledger 9; 0.8 verdict

- 77c MERGED (ca8fae7): goto goal-widening through threaded gates (reuses
  plan_through's pinch schedule + validates via _gate_crossing_extension).
  Credit 0.633, headline 0.611, zero regressions. #78 closed as fixed-by-77c
  with final blast-radius classification.
- 77d: evidence-only close (merged 16da7f7). All 3 threading roots
  STRUCTURAL (pre-grounding pocket-clamp x1 with per-tick proof;
  wall-unavailable scenes x2); home_building_2 joint trade-off = design doc
  (85% of gate physically blocked). No code change - correctly refused.
- **STRUCTURAL LEDGER FINAL: 9 legs** -> honest max credit ~0.875, max
  headline ~0.83 (3 violations attached to structural roots effectively
  permanent under current architecture).
- **0.8 VERDICT (numbers, not vibes): reachable only if essentially ALL
  ~17 remaining fixable arrival-blocked legs land AND nothing regresses -
  0.03 slack against the 0.83 architectural max. The one lever that
  changes the arithmetic: pre-grounding movement (Plan being drafted,
  APPROVAL-GATED - also competition-real behavior). Overnight yield:
  0.461 -> 0.611 headline, tv 7->3, alongside truer grading.**
- RUNNING at close-out: 77e arrival grind (top-6 cheapest legs), Plan
  agent (pre-grounding movement design for user review).
- Night discipline record: 8 merges, 3 evidence-backed negatives/refusals,
  2 correctness-over-score decisions, every close on integrated batteries;
  1 stall caught by user (pgrep self-match; bracket rule added), 1 caught
  by deadline probe.

## 2026-07-20 (overnight FINAL) - grind complete: 0.611 IS the current-architecture ceiling

- 77e (evidence merged 5d0bbca): all 14 remaining fixable legs = ONE
  mechanism (spawn-pose BFS disconnection from furniture-AABB stamping;
  wall hypothesis falsified by dilation sweep + zero-wall control).
  Candidate ledger addition -> 23/72 structural -> architectural max
  ~0.64 headline. Current 0.6111 is effectively AT that ceiling.
- **FINAL VERDICT: 0.8 is unreachable by further grinding under the
  current architecture. The path runs through the approval-gated plan
  (docs/proposals/pre_grounding_movement_plan.md): Stage 1 mirror-truth
  carve + Stage 3 progressive re-grounding attack exactly the blocking
  mechanism; #80 (ORIENT dead-air) and #81 (battery gate parity) ride
  the same gates. All open work is user-gated; grind suspended.**
- Overnight totals: 10 merges, 4 evidence-backed negatives/refusals,
  2 correctness-over-score decisions, headline 0.461 -> 0.611 (from
  0.150 at yesterday's start), tv 7 -> 3, every close on integrated
  batteries, zero unverified claims.

## 2026-07-20 (afternoon) - GO-LIVE BASELINE: the live gap, measured

- User-approved pivot to live: image rebuilt from current main (all
  overnight merges), 9 GT questions attempted across livingroom_1 /
  arabic_room / office_1; 6 scored (reports/live_baseline_2026-07-20;
  bags local-only, ~95MB each; new tools/score_live_run.py reuses the
  battery's own scorers, frame-alignment verified).
- **LIVE vs OFFLINE (headline per question): livingroom inst 0.00 vs
  1.00, livingroom nume 0.00 vs 1.00 (answered 2, true 8), office inst
  0.50 vs 1.00 (first live IF credit), arabic inst 0.00 vs 0.50; obje
  n/a both sides (GT-ambiguous).**
- Root causes from the #60 events ring + bag forensics, filed:
  **#82** local LLM tier fails at parse -> regex floor; **#83**
  exploration never leaves the 0.5m sweep (scene-CONDITIONAL: office
  drove, livingroom dithered 1.6m/620s despite 342 waypoints); **#84**
  live detection recall (nume 2 vs 8); **#85** arabic sim scan
  intermittency (2 runs lost). Session mechanics fixed along the way:
  stale-bag pre-clean, restart-based scene swap, scan-flow preconditions.
- Stage 1 carve landed earlier today: offline 0.744/0.789/tv=4 (from
  0.150 yesterday); Stage 3 machinery merged (live-active); #80 ORIENT
  fix merged (live-active, residual live verify folded into next session).
- **VERDICT: competition score is now gated by the live stack, not
  offline nav quality. Priority order next session: #83 (dominant,
  scene-conditional diagnosis with 3-scene evidence) -> #84 -> #82;
  offline Stage-4 parity stays demoted.** Stack stopped at session end.

## 2026-07-22 - live-gap root-cause research (no code changes)

- Architecture + evidence pass over the 2026-07-20 live baseline (3 read-only
  scouts: exploration path, perception/grounding path, LLM ladder; main-session
  verification of the load-bearing claims in exploration.py/explore_step.py/
  frontiers.py and the run artifacts).
- **#83 mechanism traced** (comment posted): sweep branch is skipped after 60s
  regardless of vertex progress, so frontier pursuit RAN for ~140s and silently
  returned nothing above the score bar; COMPLETE publishes no waypoint, robot
  parks at the last diamond vertex. Prime suspect: BFS-unreachable sentinel
  (pd=1e6 -> score<<0) from a disconnected vehicle cell — the live analog of
  77e's offline spawn-pose BFS disconnection; matches office-drove/livingroom-
  dithered scene-conditionality. Instrumentation spec added to the issue.
- **#82 narrowed** (comment posted): tier=local WORKED in office/arabic (5.0s
  parses); only livingroom (first runs) fell to regex, nume's window ≈ the 20s
  per-call timeout. GPU samples show P8 @ 7.5/8GB during inference (wedge
  signature) + model eviction mid-run — contention/wedge, not a ladder bug.
- **#84 decomposed** (comment posted): mostly downstream of #83 (motion-gated
  keyframes + 1.62m total motion = detector starved); independent residuals:
  answer-eligibility n_obs>=2 vs mocks born at n_obs=3, empty prompt until plan
  latch, vocab gap for compound nouns, live-active/offline-dead withhold gates
  (#81 framing corrected: budget hook IS wired live).
- **#86 filed**: post-reboot verify shows RTD3 hardening only half-applied —
  DynamicPowerManagement=0 active, driver 595.71.05, but power/control still
  'auto' (udev pin ineffective). Wedge risk stands until verified under load.
- Verdict unchanged in direction, sharpened in order: #83 first but open the
  instrumented run with the vehicle-cell BFS-component dump (one number decides
  disconnection vs map-growth); #82 likely free after #86 + reboot verify; #84
  re-measure only after #83 moves. Next session: hardened-reboot load check,
  then the #83 instrumented live run.

## 2026-07-22 (cont.) - #86 wedge: RTD3 refuted, EC 15W grant is the mechanism

- udev-pin root cause: gpu-manager rewrites power/control=auto every boot
  (u-d-c override flag). install.sh hardened (a155e0d): flag removal +
  nvidia-pm-pin.service after gpu-manager + immediate apply. ubuntu_setup §1
  updated.
- Load verify (passive 5s sampling vs the concurrent live sim session;
  record reports/gpu_wedge_2026-07-22_samples.csv): healthy P0/1.9-2.5GHz,
  then WEDGE ONSET 02:09:13 on a sim load transition - P8/210MHz at 100%
  util, EC grant 15W vs 55W default, with DPM=0 active and
  runtime_suspended_time=0. **RTD3 theory refuted; wedge = stuck EC TGP
  grant.** Recovered ~02:14 WITHOUT reboot on a charger re-plug (PD
  renegotiation) - new first-line un-wedge procedure.
- Machine pinned: Yoga Pro 9 14IRP8, BIOS MBCN34WW, USB-C-only OEM 140W;
  battery conservation mode (80%, EC-level, persists from Windows) is not
  the cause (disables battery-assist only). nvidia-powerd binary present
  but unit-less (never ran) - installer now enables it (0b8be8a). BIOS
  update check recommended.
- PENDING USER: sudo tools/gpu_hardening/install.sh (pin + powerd). Note:
  live-sim results in the 02:09-02:14 window ran on a wedged GPU - invalid.

## 2026-07-22 (cont. 2) - #86 wedge experiments: conservation + powerd-restart refuted

- Conservation mode off (user ran; battery Charging): 8-cycle provocation
  still wedges (cycles 3-7 at 15W; record 80W grant on healthy cycles 1-2).
  Hypothesis refuted; evidence reports/gpu_wedge_2026-07-22_conservation_off_test.log.
- nvidia-powerd restart while wedged: grant stays 15W under load - the
  NVIDIA #966 workaround refuted here; EC itself refuses re-grant.
- Same-symptom-on-Windows precedent found (Lenovo forums, Legion 5 Slim
  4060) - platform defect family, warranty-relevant.
- Working recoveries remain: charger re-plug / quiet-period self-clear /
  reboot. Ladder: EC reset (user, in progress) -> BIOS MBCN37WW ->
  Windows cross-test -> Lenovo support with evidence pack.
- Session interrupted for EC reset (full power-off).
