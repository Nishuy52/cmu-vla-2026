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
- **T7** test-suite-tiering — split the pytest suite into a fast
  default tier + slow milestone gate (`@pytest.mark.slow` +
  `-m "not slow"` addopts), budget-scale the structural
  integration tests (~363 s → ~16 s), add pytest-xdist for the
  parallel full gate. Fast ~38 s / full ~152 s (-n auto).
