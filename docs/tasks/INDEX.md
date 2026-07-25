# Task Index

- **T1** cmu-vln-preliminary-discussion — prep material (problem
  statement, examples, competition-process specifics) for the
  team's preliminary discussion on the CMU VLN Challenge.
- **T2** 2025-results-research — identify CMU VLA Challenge 2025
  top-1/top-2 teams and their methods; append to
  docs/prior-art.md section 1.
- **T3** merge-into-team-repo — fold the parallel research
  stream into the team repo (github.com/Nishuy52/cmu-vla-2026)
  on a branch + PR; append-first, provenance-tagged.
- **T4** implementation-gaps — top-5 discrepancies (missing /
  contradictory) between the implementation and the imported
  research findings; ranked, cited, one candidate falsified.
- **T5** colored-cloud-tool — offline colored point-cloud
  reconstruction debug tool (`tools/`): lidar → pano color →
  voxel downsample → PLY; validated on real jingfan data.
- **T6** live-colored-map — incremental colored voxel map
  (`core/perception/colored_map.py`) + debug-gated RViz
  PointCloud2 publisher: live "robot inside the colored map"
  view; debug layer only, scored path unchanged.
- **T7** numerical-count-diagnosis - per-question cause bucket
  and calibration-vs-code verdict for the GT-battery numerical
  count disagreements (13 of 15 questions).
- **T8** 2025-dossier-adjudication - mine 2025 dossier
  threshold priors / failure modes, adjudicate conflicts with
  architecture v1.0, fold GT-markers implication into design
  docs.
- **T9** test-suite-tiering — split the pytest suite into a fast
  default tier + slow milestone gate (`@pytest.mark.slow` +
  `-m "not slow"` addopts), budget-scale the structural
  integration tests (~363 s → ~16 s), add pytest-xdist for the
  parallel full gate. Fast ~38 s / full ~152 s (-n auto).
- **T10** redteam-review — five-facet adversarial design review
  (IF, counting, OR+checkpoints, eval-day systems, 2025 dossier
  adjudication) run as a parallel frontier-agent batch in the
  closing Fable window; adjudicated into
  docs/redteam/hardening_backlog.md.
- **T11** if-leg-threading — diagnose + fix IF intermediate-leg
  threading (rubric flat at 0.061; routes reach terminals without
  passing within 0.8 m of intermediate leg goals); gates the
  HELD CV sweep.
- **T12** numerical-yardstick — wire the extracted true answer key
  into the battery as the numerical yardstick (true accuracy k/15
  leads the topline, meth-F4/F6 honesty edits); provenance stamps +
  `tools/battery_diff.py` (meth-F7/F8); diagnose the 4 true numerical
  failures (11/15→13/15: `under()` wall-relative branch; 2 color gaps
  → issues #11/#12); IF leg-count census (meth-F5, 0/30 dropped);
  `--no-spawn-hint` run (arch-F9); unaligned-scene frame fit (meth-F11,
  aligned 24→28/30). New resolver-mispick bug → issue #13.
- **T13** local-llm-phase1-2 — Phase 1 conformance
  (`tools/llm_conformance.py`, 6/10 schema-valid vs 8/10 bar, two model
  failure modes found) + Phase 2 parse battery
  (`tools/llm_parse_battery.py`, all 75 questions; run confounded by full
  sim-stack contention — 2/75 reached the LLM tier; re-run on a quiet box
  recommended). No Ollama wire quirk found. Issues #44/#45/#46 filed.
- **T13** issue-sweep — clear all nine open GitHub issues
  (#10–#13, #15–#19) on branch `fix/issue-sweep`: battery-diff
  fixture schema, provenance untracked-content digest, gt_battery
  cal/answer-key/CLI polish + test split, anchored-disambiguator
  terminal-goal fix (#13, verifier-CONFIRMED), and the two color
  failures via ColorBin RGB + luminance/dominance salience knobs
  (#11/#12, numerical 13/15 → 15/15, OR/IF byte-identical,
  verifier-CONFIRMED). Follow-ups filed: #20, #21.
- **T14** if-wall-realism — derive interior walls for the IF mirror
  costmap from each scene's `traversable_area.ply` (mapped via the
  scene's fitted sim->object frame, flag-gated `--no-walls`); IF
  headline 0.117 → 0.100 (one flip, marginal-fit scene), numerical
  15/15 and OR 6/6 unchanged; livingroom_3 frame-fit-unfittable
  status re-confirmed (data defect, not a code bug).
- **T15** if2-corridor-threading (#52 + #51) — tighter wall-derivation
  frame-fit gate (0.8 m vs the 1.0 m scoring gate, #52) reverts
  livingroom_1's flip, headline 0.100 → 0.117. Then #51: traced 3
  failing corridor legs, overturned its "one defect" premise, fixed
  three distinct defects (rubric anchor-distinctness mismatch,
  resolver tier-priority loss, pinch-corridor never opening a real
  gate / never retried on a total miss); threading violations 9 → 7
  (target 0-2 not reached), headline 0.117 → 0.150, numerical 15/15 +
  OR 6/6 unchanged. Remaining 7 trace to a 4th, broader defect (GT
  obstacle-stamping fidelity) filed as #53; #51 left open with a
  status comment.
- **T16** if53-obstacle-stamping (#53) — traced all 7 remaining
  threading violations; 3 (home_building_1, home_building_2, studio)
  trace to room-scale architectural GT AABBs ("wall"/"unknown"/
  "floor" aggregates spanning a large fraction of the room in both
  axes) stamped as solid floor-to-ceiling obstacles, sealing the
  corridor gate. Fixed: `_synthetic_from_gt` skips raw-solid stamping
  for any floor-level GT instance whose footprint spans >30% of its
  OWN scene's room bounds in BOTH axes (geometry-derived,
  self-referential per scene; zero false positives swept over all 15
  scenes). IF headline stayed 0.150/7 threading violations —
  post-fix tracing found a DIFFERENT, previously-masked planner
  defect (`_pinch_costmap` self-blocking the vehicle's own start
  position / sealing the only real route) now blocks the same 3
  legs; filed as #54. Numerical 15/15 + OR 6/6 unchanged; IF
  Fréchet/coverage secondary diagnostics improved (6.764m→4.225m,
  38%→47%). Shape-(b) case (hotel_room_2, a real duplicate-labelled
  "bed frame" at the gate) traced but not fixed this session
  (deferred). #53 left open with a status comment (root cause fixed,
  headline metric didn't move — see #54).
- **T17** if66-arrival-stamping (#66) — traced all 26 "goal plausibly
  placed, drive never arrives" legs from the #61/#62 lane; dominant
  class (10/26) is a rubric goal-definition mismatch, not a stamping
  defect: `_nearest_free_goal` (#61) excluded the leg's OWN resolved
  anchor from its free-space push, leaving many GOTO/VIA_NEAR goals
  at the anchor's raw centroid — literally inside the anchor's own
  solid, stamped footprint, arrival-tolerance-unreachable by any real
  drive. Fixed: the anchor's own footprint is now pushed too, and
  DIRECTIONALLY (toward the side the route actually approaches from
  — the previous leg's goal / scene spawn — not a plain nearest-of-4-
  edges guess, which traced worse on a symmetric footprint). A second
  candidate fix (door/door-frame pass-through stamping, targeting the
  issue's other named bucket) was implemented, measured, and REJECTED
  — net negative on both credit and headline with zero corroborating
  gain across all 15 scenes. Result: ordered-leg credit 0.1889 →
  0.2222 (+17.6%), IF headline 0.1667 → 0.2000 (+20%), numerical
  15/15 + OR 6/6 byte-identical. One traced residual regression
  (hotel_room_1) and 10/26 unresolved legs (`_pinch_costmap`
  territory / #62 leg-goal-semantics territory) left for follow-up.
- **T18** architecture-refresh - as-built architecture/ folder (README + ch 01-10, 3,006 lines, verifier-confirmed anchors) vs 19 Jul main; 11 Jul snapshot archived as `archive/2026-07-11-architecture-snapshot`. DONE
- **T19** cluster-offload (#86, #82) — opt-in dev-only detector/LLM offload
  to the NUS SoC Slurm cluster over SSH tunnels (RemoteDetector +
  combined-GPU sbatch job + servers.sh/tunnel.sh), keeping Unity/ROS local;
  submission builds unaffected; live-verified end-to-end 26 Jul (detect RTT
  ~207 ms warm through the tunnel), servers start-on-demand via
  tools/cluster/servers.sh. DONE
