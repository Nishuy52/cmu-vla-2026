# Issue #83 — live exploration starvation fix, working notes

Root cause (already diagnosed, not re-derived here): the vehicle cell's
BFS-reachable FREE component stays at 1 cell for the whole run on
livingroom_1 (`reports/issue83_live_captures/livingroom_1_inst_v3/explore_debug_83797.jsonl`).
Mechanism: `mark_pose` carves one cell/tick -> disconnected trail at
>=0.1 m/tick spacing; the terrain blind-spot annulus around a
furniture-dense spawn never paints FREE, so the trail never joins the
terrain-derived FREE mass; `_bfs_distances` then stamps every frontier
with the 1e6 sentinel; `ExplorationPolicy.step` returns COMPLETE; the
head parks for the rest of the window.

## Change 1 — footprint carve (src/core/nav/occupancy.py)

`mark_pose` now carves every cell within `VEHICLE_FOOTPRINT_RADIUS_M`
(0.25 m) of the pose FREE, not just the centre cell. Chosen deliberately
as a SEPARATE constant from `core.nav.costmap.VEHICLE_RADIUS_M` (0.4 m,
"half footprint + margin" — an obstacle-INFLATION radius tuned for
planning safety margin, not the raw physical footprint). Keeping them
distinct also avoids occupancy.py importing costmap.py (costmap.py
already imports occupancy.py).

Consecutive tick poses at >=0.1 m/tick now overlap into one connected
trail instead of disconnected singletons.

## Change 2 — degenerate-pocket BFS re-rooting (src/core/nav/frontiers.py)

`detect_frontiers` computes the vehicle-rooted pocket as before. If
`pocket_cells < DEGENERATE_POCKET_CELLS` (50 = 0.5 m^2, ~2 vehicle
footprints) AND the largest FREE component is
`>= DEGENERATE_POCKET_RATIO` (4x) the pocket, it re-runs `_bfs_distances`
rooted at the nearest cell (squared-euclidean) of that larger component
and uses those distances for scoring. The full-grid FREE clustering
(`_cluster(grid.state == FREE)`) is only run when the pocket is already
small — the common/healthy case (large pocket) skips it entirely, so the
fix costs nothing on the path that already works.

Observability: module-level `_LAST_CALL_INFO` / `last_call_info()`
records `bfs_reroot`, `pocket_cells`, `largest_component_cells` for the
most recent call (module-level "last call" diagnostics per the ratified
design, rather than threading a new field through every caller).

## Change 3 — no silent COMPLETE (src/core/heads/explore_debug.py)

`maybe_dump`'s record now carries `bfs_reroot`,
`bfs_reroot_pocket_cells`, `bfs_reroot_largest_component_cells` (read
from `frontiers.last_call_info()` right after the same-tick
`detect_frontiers` call that already built `frontier_candidates`). A
COMPLETE tick with `bfs_reroot: true` or a small pocket is now visible
in the debug dump instead of a silent park. No new event/log plumbing
was added — the debug dump already fires every tick regardless of
status and was the "existing lightweight hook" the ratified design
allowed reusing.

## Unplanned but required: test-fixture float32/lattice interaction

Implementing change 1 exposed a **pre-existing, unrelated** defect: the
mock `SyntheticScene.terrain_patch` (`src/core/mocks/synthetic_scene.py`)
samples an EXACT 0.1 m lattice; `TerrainPatch.points` is float32 (matches
the real ROS message dtype). float32(n * 0.1) rounds down by one ULP for
many `n` (e.g. n=7 -> 0.699999988 -> floor(x/cell_m) = 6, not 7),
colliding two adjacent lattice rows/cols into one occupancy cell and
leaving the other perpetually UNKNOWN. Result: a checkerboard-like
~47% spurious UNKNOWN fragmentation of FREE space in any dense
synthetic-scene grid (confirmed present identically before any of my
changes via `git stash`). This was invisible pre-fix because the old
single-cell `mark_pose` barely touched it and `detect_frontiers`
was ALREADY returning 100% frontiers unreachable in this exact
scenario (the very #83 symptom, reproduced locally) — the CP2 tests were
only passing because that all-unreachable state's arbitrary tie-break
fallback (`frontiers[0].xy`) happened to land far from the vehicle.
Post-fix, reachability is correctly computed, and a real (but
extremely close, well within `PROVISIONAL_ARRIVAL_TOL_M`) frontier at
the disc boundary was legitimately winning, tripping the documented
"already there -> clear immediately" SYS-F11 arrival case on the same
tick.

This is out of scope for the ratified 3 changes (root cause lives in a
module I don't own, `core/mocks/synthetic_scene.py`, and a general
occupancy.py float precision fix has a much bigger blast radius than
this task). Filed as a GitHub issue (see final report for number).
Fixed test-side only, in the 7 affected tests
(`tests/heads/test_explore_cp2.py`, `tests/heads/test_explore_robustness.py`):
their local `_ExploreIO.latest_terrain` now (a) jitters sample points off
the exact lattice and (b) clips terrain to a realistic sensor-range
radius around the vehicle (the raw scene is a single walled room with no
doorway out, so integrating its FULL terrain leaves zero frontiers
regardless of the lattice bug — an unrelated, second latent fixture gap
this also happened to surface).

## Battery-visible risk assessment (change 1)

`core/runner/gt_battery.py` does not call `ExploreHead`/`mark_pose`
directly for NUMERICAL/OBJECT_REFERENCE scoring (no `OccupancyGrid` use
there beyond a resolution-matching comment). It DOES drive
`InstructionHead` directly for IF questions
(`_drive_if_path`/`_drive_if_trajectory` -> `_run_instruction_head` ->
`InstructionHead(...)`), and `InstructionHead.advance` calls
`self.grid.mark_pose(...)` every tick
(`src/core/heads/instruction.py:292`) — the SAME grid `Costmap` wraps for
routing/scoring. So the wider footprint carve CAN shift IF-question
battery metrics (`if_rubric`, threading/avoid violations): cells within
0.25 m of the driven path that terrain classified OBSTACLE (previously
overridden FREE only at the single centre cell) are now overridden FREE
across the whole disc, which could very occasionally change routing
near tight corridors close to a wall. NUMERICAL/OBJECT_REFERENCE
questions are unaffected (no `ExploreHead`/`mark_pose` call in their
battery path). Recommended verification command (NOT run this session —
requires the Unity groundtruth root and is a multi-scene battery run):

```
cd src && python -m core.runner.gt_battery --groundtruth <unity_root>
```

(defaults for `--questions`/`--answers`/`--questions-dir` match the
checked-in challenge question set; compare `if_rubric`,
`if_thread_viol`, `if_avoid_viol` against the last known-good report
under `reports/`.)
