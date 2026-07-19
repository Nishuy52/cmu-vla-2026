# Implementation Plan — Pre-Grounding Movement in the IF Pipeline

**Status: AWAITING USER APPROVAL — no stage may be implemented before its gate.**
Authored 20 Jul 2026 (overnight session); probes verified against main at
`72cbc64`-era code. Full provenance: issue #77 comment thread and
`reports/issue77d_notes.md` (Family A evidence this plan re-probed).

## 0. Decisive new evidence (changes the scope)

Re-probed the arabic_room case (read-only, reproducing
`reports/issue77d_notes_work/probe_leg0_detail.py`'s exact state):

| Probe | Result |
|---|---|
| BFS from spawn, inflated mirror costmap | 536-cell pocket; nearest reachable cell to anchor 7.33 m |
| BFS from spawn, raw uninflated mask | pocket only 1,513 cells; still 5.77 m — not an inflation pinch |
| BFS with derived-wall cells removed | 35,841 cells; anchor reachable to 0.05 m |
| GT reference trajectory vs the mirror | the GT path drives through **102 mirror-blocked cells** (36 derived-wall, 66 object-box: sofa 74, carpet 32, pillow 24, wall 8) |
| Carve a 0.4 m corridor along GT trajectories out of all blocked cells | pocket 536 → 23,817 cells; nearest reachable 2.50 m (leg tol 2.51) |

**Implication:** offline, the spawn pocket is sealed by mirror-fidelity
defects that the scene's own GT trajectory physically contradicts. The
battery's map is fully observed at tick 0 (`MockRobotIO.latest_terrain` →
`SyntheticScene.terrain_patch`, `src/core/mocks/synthetic_scene.py:284-317`;
ingested once in `InstructionHead._ingest_terrain`,
`src/core/heads/instruction.py:238-251`), so **no amount of simulated
movement can escape the pocket — a vehicle cannot drive out of its own BFS
component.** Pre-grounding movement alone recovers ~0 offline legs for this
class. The offline lever is a **mirror-truth constraint** (the mirror must
not block the scene's own GT trajectory). Pre-grounding movement remains the
correct **live-competition** lever (incremental sensing; UNKNOWN is
BFS-passable, `src/core/nav/costmap.py:60,265`), and the battery must mirror
it (#71 lesson). Stages below cover both, with honest expected recovery.

**Bycatch (competition-critical, pre-existing):** the ORIENT state is 60 s of
dead air. `_tick_orient` (`src/core/fsm/controller.py:285-288`) publishes
nothing and calls no explore step; the "orientation sweep" actually happens
after t=60 s (`src/core/heads/explore_step.py:194-195`). The robot is parked
for the first minute of every live question. Never visible offline (battery
bypasses the FSM).

## 1. Design answers

### (1) Where this belongs: the head, not the FSM. No new state.
The codebase's pattern for "don't act on weak grounding" is head-level
withhold gates polled from injected budget signals — #33's `min_n_obs` gate
(`_obs_gated_prefix_len`, `instruction.py:641-656`), H4c's provisional gate
(`_committable_prefix_len`, 617-639), IF-F5's avoid gate (696-701). A
withheld route already produces movement: `ExploreHead.advance` falls through
to noun-biased frontier exploration whenever the instruction head emits
nothing (`explore_step.py:151-160`). So pre-grounding movement = a new
goal-credibility withhold gate in that family + re-ground-improvement
rebuilds while driving. A GROUND_REFINE state duplicates owned machinery:
rejected. Rides along: ORIENT dead-time fix; DRIVE_OUT already re-ticks the
head (no FSM change for continuous re-grounding).

### (2) Budget interaction
Gate mirrors H4c override semantics exactly: withhold only while
`budget_frac() < PROVISIONAL_COMMIT_FRAC` (671-682), never past forced
assembly (658-669). Unconfigured hooks (None) = commit immediately — battery
and all existing tests byte-identical by default. IF explore budget 270 s
(`interfaces.py:269-273`), forced assembly 480 s (`budget.py:25`), watchdog
540 s: withholding feeds exploration inside the existing window, no new
carve-out. IF early-answer gate already returns False (`controller.py:369-371`).

### (3) Re-grounding semantics
`_ground_legs` already re-runs each tick from the current pose
(`instruction.py:225`); `_goto_point` re-floods from `self._pose` (471-476) —
fresh geometry is computed and thrown away because
`_maybe_build_or_extend_route` early-returns on an ungrown prefix (703-704).
Adoption rules: GOTO/VIA_NEAR legs only, index >= `_leg_progress`, not in
`_confirmed` (never rewrite an arrived leg, `_mark_arrivals` 950-957).
Trigger: fresh goal-clamp improves the committed snapshot by more than an
`ARRIVAL_TOL_M`-derived bar (parameter-free). Guards all reused:
`_keeps_corridor_threaded` via `_goto_point_pinch_relax` (527-534 →
`planner._gate_crossing_extension`/`_gate_extension_keeps_route_planned`,
`planner.py:378/304`); adopt only if `plan_through` succeeds (never swap into
`_recover_path`, 740-747 — needs a guarded `_build_route` variant). Bound:
`MAX_REGROUND_REBUILDS` (proposal 2), separate from
`MAX_REPLANS_PER_QUESTION=3` (93) so improvement rebuilds never starve
stall/no-LOS/capsule replans; rebuild resets follower progress exactly like
`_replan` (846-871).

### (4) Blast radius
Touched: `instruction.py` (goal_clamp_m field; gate 3 in
`_committable_prefix_len`; snapshot+improvement+guarded rebuild),
`controller.py` (`_tick_orient` only), `gt_battery.py` (Stages 1+4).
Untouched: planner.py, costmap.py, breadcrumbs.py (frozen), toolbox.py,
core/groundtruth (frozen), parse tier, `_if_rubric_geometry` resolver — no
anchor-selection change anywhere (the #71/#73/#75 convergence rule is not
implicated: we change when/where goals commit, never which instance resolves).

### (5) Offline-battery implications (#71 lesson)
- **Stage 1 (mirror truth):** carve the per-scene GT-trajectory corridor out
  of derived-wall + stamped-obstacle cells in
  `_synthetic_from_gt`/`_scene_wall_cells` (`gt_battery.py:141-181,
  273-303`). Measurement fidelity, not tuning: GT evidence only, uniform
  across scenes, physically-derived radius (0.4 = VEHICLE_RADIUS_M).
- **Stage 4 (behavior parity):** battery head currently gets
  `InstructionHead(plan=plan)` with budget hooks None (`gt_battery.py:418`) —
  silently disabling every withhold gate including the pre-existing #33/H4c
  ones (a parity gap worth its own issue). Wire synthetic hooks to the drive
  clock; allow bounded head re-ticks after full-prefix commit
  (`_drive_if_trajectory` loop stops at 507). Full sensing simulation
  explicitly out of scope.

## 2. Staged delivery (each stage approval-gated)

- **Stage 0 — evidence lock-in** (no src change): commit the probes as
  scripts; extend the GT-traj-vs-blocked-cells audit to all 15 scenes (sizes
  Stage 1; confirms no passing leg depends on a blocked mirror).
  **Gate G0: user approves Stage-1 scope from the scene table.**
- **Stage 1 — mirror-truth carve in gt_battery.py.** Expected: arabic_room
  leg0 flips or reaches tolerance boundary; its corridor+goto become
  attemptable (plausibly −1 threading violation); wall-unavailable scenes
  unaffected. Zero regressions required; full per-question diff.
  **Gate G1: user adjudicates carve as measurement fidelity (register in
  docs/calibration.md) — NOT tuning.**
- **Stage 2 — ORIENT dead-time fix** (`_tick_orient` ticks the explore
  callable). Battery byte-identical (bypasses FSM); sim trace shows
  waypoints before t=60 s. **Gate G2: review.**
- **Stage 3 — credibility withhold gate + improvement rebuilds** (the core
  head change; defaults preserve all current behavior). Unit tests (a)-(e)
  incl. the #77c corridor-guard regression case; full gate mandatory.
  **Gate G3: user approves from unit evidence + unchanged battery + live
  trace.**
- **Stage 4 — battery parity hooks** for Stage 3 (expected ~0 score
  movement; parity is the point). **Gate G4: keep-only-if-not-regressing.**

Stages 1 and 2 are independent; 3 benefits from 0; 4 requires 3.

## 3. Risk register

| Risk | Stage | Mitigation |
|---|---|---|
| Carve judged tuning-to-sample | 1 | GT-evidence-only, uniform, physical radius; explicit G1 adjudication + calibration.md registration; Stage-0 all-scenes table as holdout-style check |
| Carve reroutes a passing leg | 1 | Full per-question diff; revert on any regression |
| Withhold strands a sealed live pocket | 3 | Budget override commits at pressure — worst case equals today, delayed |
| Rebuild churn/oscillation | 3 | Derived threshold + hard cap + monotone improvement |
| Rebuild costs earned threading (#77c shape) | 3 | Corridor legs excluded; `_keeps_corridor_threaded` inherent; test (d) |
| Battery runtime blow-up | 4 | Bounded re-tick budget (`_DRIVE_HEAD_RETICK_BUDGET` pattern, 388) |
| #33/H4c offline enablement shifts baseline | 4 | Separately-diffed, separately-approvable toggle |

## 4. Open decisions (blocking the respective stages)

1. G1 adjudication: is GT-trajectory carving accepted as mirror measurement
   fidelity? (Probes: it is the ONLY mechanism that recovers arabic_room
   offline; movement cannot.)
2. Object-box carve mechanism: SyntheticScene exclusion-cell seam (touches
   core/mocks) vs runner-side carve (recommended; blast radius stays in
   gt_battery.py).
3. Credibility bar: ARRIVAL_TOL_M-derived (recommended, parameter-free) vs
   swept threshold (triggers generalization-protocol holdout machinery).
4. Stage-4 scope: parity-only hooks, or also enable the currently-dead
   #33/H4c battery gates (pre-existing parity gap — arguably its own issue).
5. ORIENT fix in this lane or as its own issue (independent and
   competition-critical either way).

## Critical files
- `src/core/heads/instruction.py` (617-639, 684-705, 450-549, 842-871)
- `src/core/runner/gt_battery.py` (141-181, 273-303, 391-424, 427-557)
- `src/core/fsm/controller.py` (285-288, 329-348, 243-253)
- `src/core/heads/explore_step.py` (141-160, 458-487 — read-only dependency)
- `src/core/nav/costmap.py` (238-296 — read-only)
