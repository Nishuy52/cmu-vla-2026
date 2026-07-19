# 07 - Navigation and exploration

Everything under `src/core/nav/` — the occupancy grid, the per-question
costmap, frontier detection, the exploration policy, breadcrumb following,
and the A* planner with its corridor-threading overlay. This is the "HOW do
we get there" layer; the answer heads (instruction-following in particular)
decide WHERE.

## ELI10

The robot keeps a graph-paper map of the floor it has actually seen (free,
blocked, or still blank). When it needs to go somewhere, it draws a
temporary "no-go" layer on top of that map for this one question only
(furniture to avoid, walls), finds a path with A*, then only ever tells the
real robot the next couple of metres of that path so it never gets sent on
a wild-goose chase to a point on the far side of a wall. Three genuinely
tricky bits went through real bugfix cycles: making a forced "walk through
this doorway" corridor not trap the robot at its own starting point (#54),
making the doorway-forcing math agree with the same math the scoring code
uses (#63/#64), and — reverted — trying to skip an unnecessary nudge when a
goal was already close enough (#67, undone the same day).

## The occupancy grid

[`src/core/nav/occupancy.py`](../src/core/nav/occupancy.py) is a growable
world-frame grid built from `/terrain_map(_ext)` XYZI point clouds. Cells
are 0.10 m (`CELL_M`, around line 28); classification is by the terrain
intensity channel: `intensity < FREE_MAX (0.15)` → FREE, else OBSTACLE
(`OccupancyGrid.integrate_patch`, around line 212). The grid starts as a
1x1 array and grows lazily: `_ensure_bounds` (around line 168) pads the
backing arrays by `GROW_PAD_CELLS` (8) whenever a patch falls outside
current bounds, shifting `origin_x`/`origin_y` so cell (0,0) stays anchored
to a stable world lattice across growth.

Per cell it tracks the max intensity ever observed there (`integrate_patch`,
around line 251), so one obstacle reading is never overwritten by a later
free reading in the same cell — an intentional one-way ratchet toward
OBSTACLE. A separate `observed` boolean mask (`mark_pose`, around line 345)
flags every cell within `OBSERVE_RADIUS_M` (8.0 m) of any past vehicle
pose; this is what the frontier code treats as "seen" — distinct from FREE,
since a cell can be observed-but-obstacle.

**Overhead clearance (issue #36 / the terrain-slab gap).** The upstream
`terrainAnalysis.cpp` filters `/registered_scan` to a thin slab
(`maxRelZ=0.2 m` above the vehicle) before publishing `/terrain_map`, so
anything overhanging — bar-height tabletops, shelves — is invisible to the
terrain cloud and reads as FREE floor underneath. `integrate_scan_overhead`
(around line 293) re-reads the raw scan and flags a separate boolean
`overhead` layer wherever scan points fall in a configurable clearance
band (`OverheadConfig.overhead_min/max`, 0.25-1.20 m above the local
ground, around line 74) with at least `min_points_per_cell` (3) hits — kept
apart from `state` so terrain-derived FREE/OBSTACLE classification stays
pure. Local ground per cell is either the terrain-derived `ground_z`, or,
before any terrain point has landed there, a runtime estimate: issue #36's
`ground_offset_estimate` property (around line 262) computes
`median(vehicle_z) - min(terrain point z)` from samples accumulated across
`integrate_patch` calls, warming up after `GROUND_OFFSET_WARMUP_PATCHES`
(5) samples and falling back to a fixed `vehicle_sensor_height` (0.60 m,
tuned for a jingfan-rig mount) until then. An env var
(`VLA_OVERHEAD_VEHICLE_SENSOR_HEIGHT_M`) can pin the fallback directly,
taking precedence over both.
`integrate_scan_overhead_decimated` (around line 387) is the per-tick
wiring helper: it clips the scan to the grid's current world bounds (never
grows the grid just to flag a far overhang) and decimates to
`OVERHEAD_SCAN_MAX_PTS` (12000) points with a deterministic stride, scaling
the effective `min_points_per_cell` threshold down proportionally so a
sparse-but-real edge doesn't silently drop below the noise gate after
decimation (the fix documented around line 421).

## The costmap: hard avoid vs. soft overhead

[`src/core/nav/costmap.py`](../src/core/nav/costmap.py)'s `Costmap` wraps
one `OccupancyGrid` snapshot per question with its own derived masks, so
stamping/inflation never mutates the shared grid other questions/the live
map use. Two separate blocking layers:

- **`base_blocked`** — OBSTACLE cells dilated by `VEHICLE_RADIUS_M` (0.4 m,
  around line 49), via `_inflate` (around line 115).
- **`capsule_blocked`** — hard avoid-capsule stamps from `stamp_capsule`
  (around line 138): every cell within `radius_m + vehicle_radius_m` of an
  avoid segment. Architecture-non-negotiable: capsules only ever grow for
  the lifetime of a question and are never relaxed, even if that makes a
  goal unreachable — `nearest_reachable_point` (around line 271) then
  returns the least-bad reachable cell instead, rather than shrinking the
  avoid zone.

**Overhead is SOFT by default**, not hard-blocked (module docstring, around
line 15-38; the change itself is a redteam-driven fix, H13/SYS-F12). A cell
the overhead layer flagged stays traversable but A* pays a steep multiplier
(`OVERHEAD_SOFT_COST_MULT = 4.0`, around line 57) to cross it — folded into
the same `is_unknown()` query A* already reads for its UNKNOWN-cell penalty
(`Costmap.is_unknown`, around line 181), so a false-positive overhead flag
next to the only door makes the route expensive rather than sealing it
shut entirely (the previous hard-block behaviour used to do exactly that,
per the docstring). **Exception:** `Costmap.clone()` (around line 196),
the only entry point the pinch-corridor overlay uses, hardens overhead back
into `base_blocked` — a forced corridor threaded UNDER furniture is not a
legal doorway, so corridor-threading specifically needs the old hard
treatment. `overhead_hard=True` at construction requests the same hardening
explicitly; `allow_overhead=True` ignores the layer altogether (terrain-only
opt-out).

`inflation_only_mask()` (around line 230) separates cells blocked only by
the vehicle-radius margin from cells with a real obstacle footprint
(`raw_blocked`) — this distinction is what lets the pinch-corridor overlay
(below) relax the inflation margin around a verified gate without ever
letting the vehicle drive through actual geometry.

## Frontiers

[`src/core/nav/frontiers.py`](../src/core/nav/frontiers.py) defines a
frontier cell as FREE and 4-adjacent to at least one UNKNOWN cell (or on
the array border, which counts as implicitly UNKNOWN) — `frontier_mask`,
around line 49. Frontier cells are 8-connected clustered
(`_cluster`, around line 68); clusters under `MIN_CLUSTER_SIZE` (5) are
discarded as noise. Each surviving cluster gets a centroid snapped to the
nearest actual cluster cell, then a score:

```
score = W_SIZE * size - W_DIST * path_distance + W_AFFINITY * affinity(xy)
```

(`detect_frontiers`, around line 165; weights around lines 28-30: size 1.0,
distance 0.5, affinity 4.0). `path_distance` is a grid-cell BFS distance
from the vehicle over FREE cells, computed by `_bfs_distances` (around line
92) as a vectorised boolean wavefront rather than a per-cell Python deque
— the docstring cites a measured 340 ms → 15-26 ms win on a 200x200 grid
from this change (H14/F13). `affinity` is an injected `(x, y) -> float`
callable for question-noun semantic bias; it defaults to constant 0 so the
base scoring is purely geometric.

## Exploration policy

[`src/core/nav/exploration.py`](../src/core/nav/exploration.py)'s
`ExplorationPolicy.step` (around line 87) is the decision loop the FSM
steps once per cycle: for the first `SWEEP_S` (60.0 s) it walks a 4-point
diamond of side `SWEEP_SIDE_M` (1.0 m) around the start pose
(`sweep_waypoints`, around line 48) — the vehicle cannot rotate in place
via `/way_point_with_heading` (heading is ignored this year), so driving
this short loop is the stand-in for a panoramic in-place spin, seeding the
occupancy grid from four headings. After the sweep (or immediately, if the
FSM sets `budget_state["force_frontier"]` under clock pressure) it calls
`detect_frontiers` and returns the best-scored frontier above
`MIN_FRONTIER_SCORE` (0.0); no frontier clearing the bar means
`ExplorationStatus.COMPLETE` (coverage saturated).

Scope note: this module (and the planner/breadcrumb layer below it)
underlies the instruction-following head's driving (`core/heads/instruction.py`
calls `plan_through` + `BreadcrumbFollower` directly). The exploration
policy itself is question-type-agnostic and is also driven by the
numerical/object-reference explore step (`core/heads/explore_step.py`),
but that caller currently publishes `ExplorationPolicy`'s raw waypoint
directly rather than routing it through `plan_through`/A* first — worth
knowing when reading a driven trajectory for those two question types, since
only the IF path gets the costmap's line-of-sight and overhead guarantees
on its own waypoints.

## Breadcrumbs

[`src/core/nav/breadcrumbs.py`](../src/core/nav/breadcrumbs.py)'s
`BreadcrumbFollower` turns a planned A* path into the near-vehicle waypoint
stream the FSM actually publishes at 5 Hz, because a distant waypoint can
strand the vehicle at a dead end. `_select_crumb` (around line 107) picks
the FARTHEST path point that is both within `LOOKAHEAD_M` (2.5 m) and in
clear line-of-sight (`line_of_sight`, around line 49, a Bresenham-style walk
over the costmap's own blocked mask). `advance` (around line 166) is called
per tick: it moves `_idx` forward past any path point the vehicle is within
`REACH_M` (0.8 m) of, has overshot, or the reached-signal covers.

**Issue #62 fix.** `_select_crumb` can hand back a crumb many indices ahead
of `_idx` on an open stretch (any LOS-clear point qualifies, regardless of
index gap) — normal. But on a path that loops back near itself (e.g. after
threading a corridor gate en route to a later leg), `_idx`'s local
neighbours can end up on an earlier, now-unrelated stretch the vehicle
never approaches again, so the within/overshoot check never fires and the
follower wedges forever (documented as "T-62 sig-1", around line 100). The
fix tracks `_last_crumb_idx` — the index of the last crumb actually
selected and LOS-verified — and fast-forwards `_idx` past it once the
vehicle demonstrably reaches it (around line 201), safe because it only
ever jumps to a crumb the follower itself already verified, never past an
unverified point.

Stall detection (`_is_stalled`, around line 234) raises `replan_flag` when
movement stays under `STALL_MOVE_M` (0.3 m) over `STALL_WINDOW_S` (10 s);
`core/heads/instruction.py` consumes this flag to trigger a replan (around
its line 760).

## The planner: A* and corridor threading

[`src/core/nav/planner.py`](../src/core/nav/planner.py)'s `astar` (around
line 59) is a standard 8-connected A* over a `Costmap` — octile heuristic
(`_heuristic`, around line 53), orthogonal cost 1 / diagonal cost `sqrt(2)`,
UNKNOWN (and soft-overhead) cells at `UNKNOWN_COST_MULT` (3.0x). Hard-blocked
cells are impassable. Start/goal are snapped to the nearest passable cell
if they land on a blocked one (`_snap_passable`, around line 115).

`plan_through` (around line 392) plans an ordered multi-leg route:
`goto`/`via_near` legs are plain A*; a `corridor_between` leg must
physically cross its gate segment, not just pass near the midpoint. This is
where the three issue fixes below live.

### Issue #63/#64: the shared usable-crossing / gate-nudge geometry

Two problems surfaced while threading corridor gates:

- **#63** — a genuine solid obstacle (not a stamping artifact) can sit
  exactly at a gate's raw midpoint: hotel_room_2 has a duplicate "bed
  frame" GT instance at the corridor midpoint between two anchors. A
  via-point placed there demands geometry no route can satisfy.
- **#64** — even with a clear midpoint, A*'s snapped goal cell can land the
  final path vertex up to half a grid cell short of the exact gate line
  (grid quantization), so the mandatory `path_crosses_gate` check fails
  even though the corridor was, in every practical sense, threaded.

Both are fixed through one shared helper, `usable_gate_point` in
[`src/core/geometry/primitives.py`](../src/core/geometry/primitives.py)
(around line 289): given a blocked-predicate callback, it slides the
crossing point away from the raw midpoint along the gate's own p0↔p1
axis — never off the verified anchor-to-anchor line — in bounded
increments (`slide_step_frac=0.1`, `max_slide_frac=0.45` of the gate
length) until the predicate clears or the sweep exhausts. The point of
sharing it: `core.geometry.toolbox.corridor_gate` (the **scoring**
geometry) and `plan_through` (**live planning**) both call the exact same
function with their own respective blocked-predicates
(`_raw_obstacle_blocked_xy` for planning, around planner.py line 263, which
tests the UNINFLATED `raw_blocked` mask — an inflation-only halo is not a
real obstruction the gate needs to route around), so the two surfaces can
never disagree about where a blocked gate's usable crossing point is (this
was previously the #51 mismatch class of bug). Commits: `08e76ed` (helper +
nav call-site), `cfc8f4b` (passes the scene index through so `corridor_gate`
engages it too), `b4a11eb` (the quantization-vs-already-threaded
distinction below).

For #64 specifically, `plan_through` (around line 476-510) distinguishes a
genuine quantization near-miss from an already-threaded gate on a later
route rebuild with a tight two-part condition: the miss must be smaller
than quantization can produce (under one grid cell — double
`_snap_passable`'s half-cell bound), AND the current position must NOT
already sit within that same one-cell band of the gate at leg start. Both
must hold before `_nudge_past_gate` (around line 238) is applied; either
failing leaves the target at the raw midpoint (the pre-#64, #54-safe
behaviour, unchanged) — a deliberately narrow condition because broadening
it once already re-broke the #54 guard below.

### Issue #54: pinch start-seal relax with bounded fallback (commit `0a22895`)

The pinch-corridor overlay (`_pinch_costmap`, around line 176) forces A*
through a narrow band around the gate by capsule-blocking everything else
within `PINCH_DISC_M` (3.0 m) of the gate midpoint. That disc is centred on
the GATE, with no regard for where the vehicle currently sits — if `cur`
sits within the disc but off the forced corridor line, the overlay would
newly stamp the vehicle's OWN cell impassable, and the forced-corridor A*
is dead-on-arrival before it can even leave (`outside_corridor` had no
notion of "already occupied"). The fix exempts a same-radius disc around
`start_xy` from the overlay's added blocking (around line 199-232) — safe
because it only ever widens what stays passable near the vehicle, never
narrows it; a pre-existing real obstacle under the vehicle's current cell
would already be a contradiction, since the vehicle is sitting there
unharmed.

`plan_through`'s outer loop (around line 511-533) additionally bounds a
**relaxation** cycle: round 0 uses the caller's own `pinch_disc_m`; if the
start-disc exemption still doesn't free a route (the vehicle's own local
inflation halo is wider than the disc), the disc widens by
`PINCH_RELAX_GROWTH` (1.75x) and retries, capped at
`MAX_PINCH_RELAX_ROUNDS` (4). Past the cap, `plan_through` falls back to
the pre-#54 behaviour — leg unreachable — rather than relaxing forever.

### `_nearest_free_goal` and the reverted #67 fix

`_nearest_free_goal` is NOT in the nav package — it lives in
[`src/core/runner/gt_battery.py`](../src/core/runner/gt_battery.py) (around
line 586) as part of the ground-truth scoring rubric, pushing a raw anchor
centroid off any floor-level obstacle footprint (including the anchor's
OWN footprint, per issue #66) it falls inside, mirroring what real
navigation (`core/heads/instruction.py`'s `_goto_point`/`_via_point`) does
by BFS-snapping onto free/reachable space instead of targeting a point
inside solid geometry.

**Issue #67 (tolerance-aware push, merged then reverted).** Commit
`1f7f8e8` added a guard: skip the push entirely when a footprint's own
half-diagonal is already within `LEG_ARRIVAL_TOL_M`, on the reasoning that
a small enough object's raw centroid is already provably close to its own
boundary, so pushing it anyway is needless churn that can (and, in a traced
hotel_room_1 chair-leg case, did) land the goal FARTHER from the real
approach than the raw centroid was (0.684 m → 1.02 m). Measured in
isolation on its own branch base this looked like a real win: ordered-leg
credit 0.2278 / headline 0.2056 vs. the pre-fix 0.2222 / 0.2000.

**It was reverted the same day, at commit `bbb66bd`.** The guard was
re-measured on integrated `main` — which by then had also picked up the
#63/#64 `usable_gate_point` call-site — and REGRESSED there to 0.1944 /
0.1722, worse than baseline, because of an interaction between the two
changes that was never diagnosed before the branch-local numbers were
trusted (issue #67's own reopening comment calls out that an earlier
"fixed" comment on the same issue was posted before the integrated numbers
were actually read). The revert restores `main` to the current committed
0.2222 / 0.2000 baseline described in `LOG.md`'s session 18 entry.

**As the code stands today (post-revert):** `_nearest_free_goal` (around
gt_battery.py line 586) has no tolerance-aware skip. It unconditionally
pushes any raw goal that falls inside a footprint+clearance box — the
directional own-anchor push from issue #66 is the only conditioning logic
present; issue #67's half-diagonal short-circuit does not exist in the
current file. Why this matters for anyone reading the function fresh: the
docstring's own reasoning about "push only if necessary" that a reader
might expect from the issue history is not there, deliberately, because
that guard measured worse once integrated with the rest of `main`. Issue
#67 is tracked OPEN on GitHub with this history attached; a re-fix is
explicitly deferred behind issue #70's layer-1 rubric redesign, which
rewrites this same arrival semantics — re-diagnosing the guard against the
#63/#64 gate-call-site interaction only makes sense once that redesign
lands (see [10-gaps-and-risks.md](10-gaps-and-risks.md)).

## References

- [`src/core/nav/occupancy.py`](../src/core/nav/occupancy.py)
- [`src/core/nav/costmap.py`](../src/core/nav/costmap.py)
- [`src/core/nav/frontiers.py`](../src/core/nav/frontiers.py)
- [`src/core/nav/exploration.py`](../src/core/nav/exploration.py)
- [`src/core/nav/breadcrumbs.py`](../src/core/nav/breadcrumbs.py)
- [`src/core/nav/planner.py`](../src/core/nav/planner.py)
- [`src/core/geometry/primitives.py`](../src/core/geometry/primitives.py) — `usable_gate_point`
- [`src/core/runner/gt_battery.py`](../src/core/runner/gt_battery.py) — `_nearest_free_goal`
- Issues: #54, #61, #62, #63, #64, #67 (GitHub, `gh issue view <N>`)
- Sibling docs: [09-runtimes-and-deployment.md](09-runtimes-and-deployment.md),
  [10-gaps-and-risks.md](10-gaps-and-risks.md)
