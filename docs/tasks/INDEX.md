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
