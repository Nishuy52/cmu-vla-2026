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
