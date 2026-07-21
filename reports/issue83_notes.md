# Issue #83 — live exploration starvation, scene-conditional

## Phase 1 — instrumentation (done)

Added `src/core/heads/explore_debug.py` (env-gated on `VLA_EXPLORE_DEBUG_DIR`,
unset = no-op) and wired one call into `ExploreHead._explore` (`src/core/heads/explore_step.py`),
throttled to ~10s of question-clock time. Each JSONL line records:
- costmap summary: grid shape, FREE/UNKNOWN/OBSTACLE cell counts, BFS-reachable
  "pocket" size (cells + m^2) from the current pose
- every clustered frontier blob (pre-filter) with accepted/reason
  (too_small_cluster / unreachable / score_below_min / accepted)
- the waypoint actually published this tick (or null)

Verified: fast test tier green (`pytest -q` from `src/`, 0 failures), manual
smoke test with a synthetic terrain patch confirms JSONL output shape and that
with the env var unset the code path is a single dict-lookup no-op.

Next: Phase 2 controlled live comparison (livingroom_1 vs office_1).
