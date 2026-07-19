# Issue #77e — arrival-blocked pool: per-leg mechanism trace, single dominant cluster found

Baseline: `reports/gt_battery_main_post77c` (credit 0.6333, headline 0.6111, tv=3),
unchanged this session — **no `src/` fix landed** (see Conclusion).

## Rebuilt miss table

Every `instruction_following` leg with `reached_in_order=false` in the current
baseline, split by kind:

- `corridor_between` legs with `threaded=false` (3: `arabic_room` leg1,
  `home_building_1` leg1, `livingroom_1` leg1) are the `n_threading_violations`
  metric's own category, not part of the arrival-blocked pool.
- The remaining 22 `goto`/`via_near` misses. Excluding the named structural
  ledger legs that are still present as misses (7 of the ledger's 9 — the
  `hotel_room_2` leg now passes under the current baseline's tolerances) and
  the `home_building_2` deferred trade-off leg (1) leaves a **14-leg fixable
  pool** (script: `reports/issue77e_notes_work/build_miss_table.py`):

| excess (m) | scene | leg | question (truncated) |
|---|---|---|---|
| 0.0221 | studio | 1 | Go to the vases on the cabinet below the TV and stop at the guitar... |
| 0.0276 | arabic_room | 2 | First, go to the potted plant furthest from the hookah... (leg2) |
| 0.1838 | office_2 | 1 | Go near the potted plant on the cabinet and stop at the window... |
| 0.2150 | livingroom_4 | 0 | First, go near the fireplace, then go to the window... |
| 0.8338 | chinese_room | 0 | Go near the potted plant on the table and stop at the painting... |
| 1.6087 | home_building_2 | 1 | Go near the magazine on the ottoman, then go to the potted plant... |
| 1.6586 | home_building_2 | 2 | Take the path between the sofa and the coffee table... (leg2) |
| 1.9666 | hotel_room_1 | 0 | Go to the bedside table closest to the window... |
| 2.2641 | office_1 | 0 | Go to the potted plant furthest from the projector screen... |
| 2.4806 | chinese_room | 1 | First, go near the tea table with the elephant figurine... |
| 2.5674 | livingroom_2 | 1 | First, go to the chair near the window... |
| 4.1174 | arabic_room | 0 | Go near the stool under the picture... |
| 5.3525 | loft | 2 | Go near the fireplace, pass by the stairs... |
| 5.5254 | loft | 0 | Go to the cup near the TV remote... |

(Excluded as structural/deferred, not re-litigated: `livingroom_1` leg0,
`home_building_2` leg1-of-the-corridor-question, `livingroom_3` leg1/leg2,
`home_building_1` leg0/leg1-of-the-kettle-question, `arabic_room` leg0-of-the-
hookah-question, `home_building_1` leg0-of-the-nightstand-question.)

## Per-leg mechanism trace (top 6 by smallest excess, plus 8 more for
## cluster confirmation)

Method: reran each leg's grounding standalone (`InstructionHead._ground_one`),
compared the picked candidate's centroid to the rubric's true goal (is
selection correct?), then checked whether that candidate's cell is reachable
from the grounding-time pose via the costmap's own `reachable_mask` BFS (is
selection blocked?), then drove the actual controller and located the
closest approach to both the head's own resolved goal and the rubric goal.
Scripts: `reports/issue77e_notes_work/trace_leg.py`,
`reports/issue77e_notes_work/trace_leg2.py`.

**Critical methodology note surfaced by this trace**: the pool's "excess"
metric (`min_dist_driven_to_goal_m - tol_used`, the same definition
`legs_post79.json` uses) is measured against the **head's own resolved
goal**, not the rubric goal — arrival credit, however, is scored against the
rubric goal. When the head's own goal is a costmap-forced substitute far
from the true target, `excess` is near-zero (the vehicle trivially reaches
its own wrong goal) even though the leg is nowhere close to earning credit.
This is why "smallest excess" did **not** turn out to mean "cheapest to
fix" here — every one of the 6 cheapest-excess legs actually has a 1.5–3.5 m
gap to the true rubric goal, invisible to the excess ranking itself.

| scene | leg | candidate correct? (dist to rubric) | reachable from grounding pose? | head's own goal reached? | named mechanism |
|---|---|---|---|---|---|
| `studio` | 1 (guitar) | yes, 0.826 m | **no** (real+raw blocked, BFS-disconnected) | yes, dist 0.0 | grounding-pose BFS disconnection |
| `arabic_room` | 2 (tray) | yes, 0.366 m | **no** | yes, dist 0.0 | grounding-pose BFS disconnection |
| `office_2` | 1 (window) | yes, 0.400 m | **no** | yes, dist 0.0 | grounding-pose BFS disconnection |
| `livingroom_4` | 0 (fireplace) | yes, 0.670 m | **no** | close but leaves (0.11 en route) | grounding-pose BFS disconnection |
| `chinese_room` | 0 (potted plant) | yes, 0.620 m | **no** | close but leaves (0.70 en route) | grounding-pose BFS disconnection |
| `home_building_2` | 1 (potted plant) | yes, 0.570 m | **no** | yes, dist 0.0 | grounding-pose BFS disconnection |
| `home_building_2` | 2 (potted plant) | yes, 1.088 m | **no** | (not separately driven — same question chain) | grounding-pose BFS disconnection |
| `hotel_room_1` | 0 (bedside table) | yes, 0.841 m | **no** | — | grounding-pose BFS disconnection |
| `office_1` | 0 (potted plant) | yes, 0.860 m | **no** | — | grounding-pose BFS disconnection |
| `chinese_room` | 1 (table) | yes, 0.645 m | **no** | — | grounding-pose BFS disconnection |
| `livingroom_2` | 1 (crystal ball decoration) | yes, 0.085 m | **no** | — | grounding-pose BFS disconnection |
| `arabic_room` | 0 (stool) | yes, 0.675 m | **no** | — | grounding-pose BFS disconnection |
| `loft` | 2 (sphere decoration) | yes, 0.729 m | **no** | — | grounding-pose BFS disconnection |
| `loft` | 0 (fireplace) | yes, 0.367 m | **no** | — | grounding-pose BFS disconnection |

**All 14/14 legs in the pool cluster into one mechanism.** In every case the
head's own candidate-selection logic (`_ranked_anchor`/`_resolve_leg_anchors`)
already picks the *correct* instance (dist-to-rubric well under 1.1 m in
every row — this is not a salience/tie-break bug); the failure is entirely
downstream, in `_goto_point`'s `cm.reachable_mask(self._pose)` /
`cm.nearest_reachable_point`: the correct anchor's cell is not in the BFS-
reachable component from `self._pose` (a single pose fixed once at
`InstructionHead.__init__` and reused unchanged for every leg's grounding,
regardless of how many legs the plan has already "walked" by the time a
later leg is grounded — `src/core/heads/instruction.py:223`,`471-476`).
`nearest_reachable_point` then substitutes the nearest cell that *is*
reachable, which lands 1.5–3.5 m short of the true target — nowhere near
enough to earn arrival credit, but exactly on the substitute goal, which is
why the driven trajectory's distance to that substitute is ~0.

## Why this is not the previously-named "wall-unavailable costmap" /
## "pocket-clamp" mechanism (ruled out, not assumed)

The 3 already-adjudicated structural legs (`arabic_room`, `home_building_1`,
`livingroom_1` leg0, #77d) were attributed to derived-wall gaps (missing or
misplaced interior walls from `traversable_area.ply`). This session tested
that hypothesis directly against the new 14-leg cluster and **falsified
it as the cause here**:

- Sweeping the wall-derivation dilation parameter (`_WALL_DILATE_CELLS`,
  currently 2 cells / 0.2 m) from 2 up to 30 cells — enough to erase nearly
  all interior wall cells in every traced scene (down to 0 cells for
  `chinese_room` at dilate=30) — **never reconnects the anchor** in any of
  the 6 swept scenes (`reports/issue77e_notes_work` inline sweep, not
  checked into a standalone script — see session transcript for the exact
  loop; reproducible via `GB._derive_wall_cells(..., dilate_cells=N)` fed
  back through `GB._run_instruction_head`).
- A control run with `wall_cells=set()` (zero interior walls at all, only
  the outer room boundary + real furniture AABBs) on `chinese_room` still
  leaves the anchor unreachable, and the grounding-pose's own BFS component
  covers only ~15% of the costmap's passable cells (4821/31329).

The real blocking source is the **density of stamped furniture AABBs**
(`_synthetic_from_gt`'s `sc.place_box` per GT instance) collectively
sealing off large sub-regions of the mirrored scene from a single fixed
BFS origin — independent of wall derivation entirely. This refines (not
just re-confirms) the prior sessions' "wall-unavailable" framing: the
dominant residual mechanism is the synthetic costmap mirror's fidelity
ceiling around densely-furnished rooms, not specifically the wall-mesh
derivation step.

## Fix attempted: none

Per the brief's cluster-size ranking, this is the only cluster (14/14 traced
legs share it) — there is no second, smaller mechanism competing for
priority in this pool. No candidate fix survives the frozen-surface
constraint:

- Widening wall-derivation dilation (a `gt_battery.py`-local, non-frozen
  parameter) was swept and directly falsified as a lever (evidence above).
- Relaxing inflation only (the existing pinch-relax machinery, already
  reused for corridor-adjacent GOTO legs) cannot help either: the
  disconnection persists on `raw_blocked` (real object footprints, zero
  inflation) — pinch-relax only ever widens the *inflation* margin, by
  design (`nearest_reachable_point`'s own docstring: "capsules are NEVER
  relaxed"), and real footprint blocking isn't inflation.
- The only lever that would actually reach these targets is re-grounding
  each leg from the pose the vehicle will actually occupy after driving
  the preceding legs (progressive/pre-grounding re-flood), instead of the
  single fixed spawn-time BFS used for the whole route today. This is
  **the same architectural lever #77d already scoped as a design document,
  not a bounded implementation** ("pre-grounding movement... cross-cutting,
  out of bounded scope"), now confirmed to be the blocking factor for the
  *entire* remaining pool rather than one leg — which strengthens the case
  for eventually building it, but does not change the bounded-session
  cost/benefit calculus #77d already worked through. Implementing it here
  would mean restructuring `_ground_legs`'s single forward-pass shared path
  every leg in every scene goes through, well past what a single session's
  frozen-surface / full-gate-only-if-shared-path-changes budget supports.

No `src/` change was made. Per the brief's stop condition, this is reported
as a decisive, evidence-based **dry finding on the entire cluster** rather
than spending an integrated-battery cycle on a change the reachability
measurements already show cannot flip any leg.

## Integrated numbers (unchanged, no code touched)

| metric | value |
|---|---|
| `mean_ordered_leg_credit` | 0.6333 |
| `mean_rubric_score` (headline) | 0.6111 |
| `total_threading_violations` | 3 |
| `total_avoid_violations` | 0 |

## Structural ledger — candidate addition (not unilaterally added)

Following #77d's own precedent (evidentiary contribution, not a unilateral
table edit — the existing ledger table's scope is curated by the
maintainer), this session's finding is offered as a candidate row rather
than added directly:

| scene | legs | mechanism | evidence |
|---|---|---|---|
| (candidate, all 14 in this pool) | 14 | grounding-time-pose BFS disconnection from dense furniture-AABB costmap stamping (single fixed spawn pose used for every leg's `reachable_mask`, no pre-grounding re-flood as the route progresses) | this session: 14/14 traced legs show correct candidate selection (dist-to-rubric < 1.1 m) but BFS-unreachable anchor from the grounding pose, independent of wall-derivation dilation (swept 2->30 cells) and of inflation (persists on `raw_blocked`) |

If adopted, the honest arithmetic ceiling for this baseline (9 previously-
ledgered + 14 here = 23 structural legs across the 30 scored `goto`/
`via_near`/`corridor_between` legs) would need to be recomputed by the
maintainer against the full leg count; left to that judgment rather than
computed unilaterally here, matching #77d's stance on the same question.

## Files

- `reports/issue77e_notes_work/build_miss_table.py` — rebuilds the current
  miss table from `reports/gt_battery_main_post77c/gt_battery_results.json`,
  applying the named structural/deferred exclusions.
- `reports/issue77e_notes_work/trace_leg.py` — per-leg goal-resolution +
  driven-trajectory tracer (candidate anchor, head goal vs rubric goal,
  closest-approach diagnostics).
- `reports/issue77e_notes_work/trace_leg2.py` — deeper reachability check
  (candidate-selection correctness vs BFS-reachability of that candidate
  from the grounding pose; nearby-instance sanity check).

## Tests

No `src/` code changed this session — fast tier and full gate were not
re-run (nothing to verify). All diagnostic scripts under
`reports/issue77e_notes_work/` were run standalone against the current
`main`-derived worktree and their output inspected directly (see above);
they are outside `src/`'s test surface.
