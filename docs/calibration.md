# Calibration Ledger

Single-source inventory of every tunable constant in `core/`, mirrored by
`core/calibration.py` (the `Calibration` dataclass) and pinned to the live modules
by `tests/test_calibration.py`. Phase-2 will sweep these against the sim; this table
is the sweep checklist.

**How to read it**
- **Field** — dotted key used by `apply_overrides` / `diff` / `to_json`
  (`subsystem.field`).
- **How wired** — how the value reaches its consumer *today*:
  - **constructor arg** — the owning class takes it as an `__init__` / dataclass
    field, so an override flows in by constructing with the swept value.
  - **function param** — a module function accepts it as a keyword with the constant
    as default; overridable by passing the swept value at the call site.
  - **module constant** — a bare module-level global read directly at call time with
    no injection seam. Moving it needs a code change → **wiring TODO (Phase 2)**.
- **Sensitivity** — H/M/L judgement of how much sweeping this field moves behaviour,
  from reading the consuming code. Guides sweep priority, not a hard claim.

**Wiring summary:** 61 fields total (geometry 15 · fusion 5 · tracker 2 · keyframe 3 ·
nav 21 · budget 15). **Wireable today (constructor arg or function param): 47.**
**Wiring TODO (module constant, needs a setter/param before a sweep can move it): 14.**

---

## geometry (`core.geometry.toolbox.Thresholds`) — 15 fields, all wireable

Composed live: `default_calibration().geometry is-equal DEFAULT_THRESHOLDS`. Every
predicate in `toolbox.py` takes `th: Thresholds = DEFAULT_THRESHOLDS`, so passing a
swept `Thresholds` at the call site wires all 15 with no code change.

| Field | Default | Unit | Controls | Evidence / source | How wired | Sens |
|---|---|---|---|---|---|---|
| geometry.near_floor | 1.2 | m | Floor of the scale-adaptive `near` radius: `max(near_floor, near_scale·diag)` | Spec-fixed "near" value | function param (`near`, `near_thresh`) | H |
| geometry.near_scale | 0.6 | — | Slope of `near` radius vs anchor footprint diagonal | Spec-fixed | function param | H |
| geometry.next_to_gap | 0.75 | m | Max AABB gap for the tight `next_to` (unrouted; `Pred.NEXT_TO`→`near`, DD-A5) | Spec-fixed "next_to" | function param | L |
| geometry.on_min_overlap_frac | 0.50 | frac | `on` footprint intersection-over-min gate (was 0.30 over-target; H5/T8-C2) | VLA-3D gen [V] IoM>0.5 | function param | M |
| geometry.on_upper_span_frac | 0.25 | frac | `on` upper z-band starts at `zmin + this·height` (support semantics, H5/T8-C3/D3) | VLA-3D gen (reconciled) | function param | M |
| geometry.on_top_tol | 0.15 | m | `on` allows `a.bottom` up to this far above b's AABB top | Invented (sweepable) | function param | M |
| geometry.in_containment_frac | 0.60 | frac | Min footprint fraction inside b for `in_` | Invented | function param | M |
| geometry.in_vert_slack | 0.10 | m | Z-span slack for `in_` vertical containment | Invented | function param | L |
| geometry.above_lateral_infl | 0.50 | m | `above` anchor-footprint inflation for the lateral-offset gate (replaces overlap, H5/T8-C4/D4) | VLA-3D evidence (reversed) | function param | M |
| geometry.under_iom_min | 0.50 | frac | `under`/`below` footprint IoM-over-min gate (both branches, DD-A7) | VLA-3D gen [V] IoM>0.5 | function param | M |
| geometry.under_tuck_tol | 0.15 | m | `under` tuck-under floor tolerance (`a.min_z ≤ b.zmin + this`) + strict-branch slack | VLA-3D `under_thres`=0.01 (sweepable) | function param | M |
| geometry.with_feature_pad | 0.30 | m | "near" pad for the `with_feature` RELAXATION rung (primary is `on(feature,a)`, DD-A6) | Invented | function param | L |
| geometry.avoid_inflate | 0.25 | m | Capsule/disc inflation for avoid geometry | Spec-fixed avoid inflation | function param (`avoid_capsule`) | H |
| geometry.superlative_margin_frac | 0.25 | frac | Early-answer winner-margin gate for superlatives | Invented | function param (carried on `Thresholds`; read by the answer path) | M |
| geometry.size_sep_gap | 1.20 | ratio | Size resolver: min largest-face-area ratio for a "small"/"big"/"largest" extreme (DD-A12) | VLA-3D gen 1.2× | function param (via `_attrs_match`) | M |

---

## fusion (`core.perception.fusion.FusionConfig`) — 5 fields, all wireable

Composed live: equals `DEFAULT_FUSION_CONFIG`. `fuse_detection(..., cfg=DEFAULT_FUSION_CONFIG)`
and `PerceptionPipeline(fusion_cfg=...)` both accept it → constructor/param wired.

| Field | Default | Unit | Controls | Evidence / source | How wired | Sens |
|---|---|---|---|---|---|---|
| fusion.min_points | 5 | count | Reject a detection below this many cluster points | Invented (docstring) | constructor arg + function param | H |
| fusion.angular_pad | `deg2rad(3.0)` ≈ 0.05236 | rad | Frustum padding beyond bbox rays | Invented | function param | M |
| fusion.depth_bin | 0.25 | m | Range-histogram bin width (declared; clustering uses `cluster_gap`) | Invented | function param | L |
| fusion.max_range | 15.0 | m | Ignore lidar points beyond this | Upstream lidar/terrain scale | function param | M |
| fusion.cluster_gap | 0.5 | m | Range gap splitting nearest cluster | Invented | function param | H |

---

## tracker (`core.perception.tracker.TrackerConfig`) — 2 fields, wireable

Composed live: equals `DEFAULT_TRACKER_CONFIG`. `associate(..., cfg=...)` and
`PerceptionPipeline(tracker_cfg=...)` accept it.

| Field | Default | Unit | Controls | Evidence / source | How wired | Sens |
|---|---|---|---|---|---|---|
| tracker.gate | 0.75 | m | Max centroid distance for a cross-frame association match | Invented (docstring says default 0.75) | constructor arg + function param | H |
| tracker.decay_k | 5 | keyframes | One-frame ghosts (n_obs==1) not re-observed within this many keyframes are pruned; confirmed tracks never decay; 0 disables (redteam H15a) | Invented, conservative | constructor arg | M |

---

## keyframe (`core.perception.tracker.KeyframeConfig`) — 3 fields, all wireable

Composed live: equals `DEFAULT_KEYFRAME_CONFIG`. `PerceptionPipeline(keyframe_cfg=...)`
accepts it; `_is_keyframe` reads it off the instance.

| Field | Default | Unit | Controls | Evidence / source | How wired | Sens |
|---|---|---|---|---|---|---|
| keyframe.every_k | 1 | frames | Process every K-th frame regardless of motion | Invented | constructor arg | M |
| keyframe.min_translation | 0.5 | m | Motion threshold that forces a keyframe | Invented | constructor arg | H |
| keyframe.min_rotation | `deg2rad(30.0)` ≈ 0.5236 | rad | Rotation threshold that forces a keyframe | Invented | constructor arg | H |

---

## nav (`core.calibration.NavTunables`) — 21 fields

New mirror dataclass; **not** composed from an existing config. Mixed wiring: the
class-level params (`Costmap`, `OccupancyGrid`, `ExplorationPolicy`, `BreadcrumbFollower`,
`detect_frontiers`) are wireable today; the planner globals and the occupancy grow-pad
are hard module constants.

| Field | Default | Unit | Controls | Evidence / source | How wired | Sens |
|---|---|---|---|---|---|---|
| nav.cell_m | 0.10 | m/cell | Occupancy grid resolution | `occupancy.CELL_M` | constructor arg (`OccupancyGrid.cell_m` default = `CELL_M`) | H |
| nav.free_max | 0.15 | m | Terrain intensity < this ⇒ FREE | `occupancy.FREE_MAX`; matches upstream `obstacleHeightThre`/`TerrainPatch.FREE_MAX` | constructor arg (`OccupancyGrid.free_max`) | H |
| nav.observe_radius_m | 8.0 | m | Lidar footprint radius for the observed mask | `occupancy.OBSERVE_RADIUS_M` | constructor arg (`OccupancyGrid.observe_radius_m`) | M |
| nav.grow_pad_cells | 8 | cells | Extra ring added when the grid grows | `occupancy.GROW_PAD_CELLS` | module constant — **wiring TODO (Phase 2)** (read directly in `_ensure_bounds`) | L |
| nav.vehicle_radius_m | 0.4 | m | Obstacle inflation radius | `costmap.VEHICLE_RADIUS_M` | constructor arg (`Costmap.vehicle_radius_m` default = const) | H |
| nav.overhead_soft_cost_mult | 4.0 | × | Intended A* penalty to cross a SOFT-overhead cell vs FREE (redteam H13). Default overhead is now a SOFT high-cost layer (`Costmap`), NOT a hard block: a false-positive overhead flag makes a route expensive, not unreachable, while a genuinely-blocked under-furniture route (no alternative) stays strongly avoided. The applied per-cell penalty on the A* path is `planner.UNKNOWN_COST_MULT` (the only per-cell cost seam A* reads without a planner change); this value records the intended weight and is the sweep handle once a Phase-2 planner seam lands. Corridor threading still HARDENS overhead (`Costmap.clone`). | `costmap.OVERHEAD_SOFT_COST_MULT` | constructor arg (`Costmap.overhead_soft_cost_mult` default = const) | M |
| nav.overhead_min | 0.25 | m | Overhead-clearance band lower edge (height above local ground; skip near-ground returns) | `occupancy.OverheadConfig.overhead_min` | dataclass field (`OverheadConfig`, passed to `integrate_scan_overhead`) | H |
| nav.overhead_max | 1.20 | m | Overhead-clearance band upper edge (skip walls/ceiling above furniture) | `occupancy.OverheadConfig.overhead_max` | dataclass field (`OverheadConfig`) | H |
| nav.overhead_min_points_per_cell | 3 | points | In-band scan points a cell needs before it flags OVERHEAD (noise reject). The decimation helper `integrate_scan_overhead_decimated` now scales this DOWN by the applied stride (`ceil(min_points/stride)`, floor 1) so a sparse-but-real overhang edge that passes at full density still passes after decimation (redteam H13 / SYS-F12). | `occupancy.OverheadConfig.min_points_per_cell` | dataclass field (`OverheadConfig`) | M |
| nav.vehicle_sensor_height | 0.60 | m | Fallback local-ground = `vehicle_z − this` when a cell has no terrain-derived ground z (jingfan: vehicle z ≈ 0.0, floor z ≈ −0.6) | `occupancy.OverheadConfig.vehicle_sensor_height` | dataclass field (`OverheadConfig`) | M |
| nav.overhead_scan_max_pts | 12000 | points | Per-tick decimation cap for the raw /registered_scan fed to the overhead layer | `occupancy.OVERHEAD_SCAN_MAX_PTS` | module constant (default arg to `integrate_scan_overhead_decimated`) | L |
| nav.min_cluster_size | 5 | cells | Frontier clusters below this are noise | `frontiers.MIN_CLUSTER_SIZE` | function param (`detect_frontiers`) | M |
| nav.w_size | 1.0 | — | Frontier score reward on cluster size | `frontiers.W_SIZE` | function param | M |
| nav.w_dist | 0.5 | per cell | Frontier score penalty on path distance | `frontiers.W_DIST` | function param | H |
| nav.w_affinity | 4.0 | — | Frontier score weight on injected semantic affinity | `frontiers.W_AFFINITY` | function param | H |
| nav.unknown_cost_mult | 3.0 | × | A* cost multiplier for an UNKNOWN cell | `planner.UNKNOWN_COST_MULT` | module constant — **wiring TODO (Phase 2)** (read in `astar`) | H |
| nav.pinch_disc_m | 3.0 | m | Radius of the local pinch overlay around a gate | `planner.PINCH_DISC_M` | module constant — **wiring TODO (Phase 2)** (`_pinch_costmap`) | M |
| nav.pinch_corridor_half_w_m | 0.5 | m | Half-width of the forced corridor through a gate | `planner.PINCH_CORRIDOR_HALF_W_M` | module constant — **wiring TODO (Phase 2)** | M |
| nav.sweep_s | 60.0 | s | Duration of the opening orientation sweep | `exploration.SWEEP_S` | constructor arg (`ExplorationPolicy.sweep_s`) | M |
| nav.sweep_side_m | 1.0 | m | Side length of the sweep diamond | `exploration.SWEEP_SIDE_M` | constructor arg + function param (`sweep_waypoints`) | L |
| nav.min_frontier_score | 0.0 | score | Frontiers below this aren't pursued | `exploration.MIN_FRONTIER_SCORE` | constructor arg (`ExplorationPolicy.min_frontier_score`) | M |
| nav.coverage_saturated_free_frac | 0.0 | frac | Reserved; frontier absence is the real completion gate | `exploration.COVERAGE_SATURATED_FREE_FRAC` | module constant — **wiring TODO (Phase 2)** (reserved / unused) | L |
| nav.lookahead_m | 2.5 | m | Farthest a crumb may sit ahead of the vehicle | `breadcrumbs.LOOKAHEAD_M`; upstream gotcha 14 (≤ ~2.5 m) | constructor arg (`BreadcrumbFollower.lookahead_m`) | H |
| nav.reach_m | 0.8 | m | Advance to next crumb within this distance | `breadcrumbs.REACH_M` | constructor arg | M |
| nav.stall_move_m | 0.3 | m | Movement below this over the window ⇒ stalled | `breadcrumbs.STALL_MOVE_M` | constructor arg | M |
| nav.stall_window_s | 10.0 | s | Stall observation window | `breadcrumbs.STALL_WINDOW_S` | constructor arg | M |

> **Overhead-clearance tunables — single-bag fit; multi-scene validation is an
> Ubuntu-gate item (redteam H13 / SYS-F12).** The five overhead tunables
> (`overhead_min`, `overhead_max`, `overhead_min_points_per_cell`,
> `vehicle_sensor_height`, `overhead_scan_max_pts`) plus the softening weight
> (`overhead_soft_cost_mult`) are fitted to the single jingfan bag. They are NOT
> validated on any other scene. Softening the layer to SOFT-cost by default (H13)
> makes a wrong flag cheap rather than route-killing, which lowers the risk of the
> single-bag fit — but the band/point-gate values still need validation on ≥2 more
> scenes' recorded bags at the Ubuntu gate before they are trusted (the hardening
> backlog lists this as an open Ubuntu-gate item). The exploration-side asymmetry
> (only IF routes use a Costmap; frontier exploration ignores overhead) is likewise
> flagged there and deferred.

---

## budget (`core.calibration.BudgetTunables`) — 15 fields

New mirror dataclass. The clock gates and per-QType exploration budgets are consumed
as bare module constants inside `BudgetState`/`interfaces`; the checkpoint caps ARE
wireable (the `CallLedger(caps=...)` argument). `explore_budget_map()` and
`checkpoint_max_map()` reconstruct the live `EXPLORE_BUDGET_S` / `CHECKPOINT_MAX` dicts
exactly (pinned in tests).

| Field | Default | Unit | Controls | Evidence / source | How wired | Sens |
|---|---|---|---|---|---|---|
| budget.question_budget_s | 600.0 | s | Total per-question wall budget | `interfaces.QUESTION_BUDGET_S`; architecture §3 | module constant — **wiring TODO (Phase 2)** (`BudgetState.remaining`) | H |
| budget.forced_assembly_s | 510.0 | s | T-90: begin best-effort answer assembly | `interfaces.FORCED_ASSEMBLY_S` | module constant — **wiring TODO (Phase 2)** (`forced_assembly`) | H |
| budget.watchdog_floor_s | 570.0 | s | T-30: publish floor answer unconditionally | `interfaces.WATCHDOG_FLOOR_S` | module constant — **wiring TODO (Phase 2)** (`watchdog_floor`) | H |
| budget.terrain_free_max | 0.15 | m | `TerrainPatch.FREE_MAX` traversability cutoff | `interfaces.TerrainPatch.FREE_MAX`; upstream `obstacleHeightThre=0.2` (conservative 0.15) | module constant — **wiring TODO (Phase 2)** (frozen dataclass class-var) | H |
| budget.orientation_s | 60.0 | s | In-place sweep window (`in_orientation`) | `fsm.budget.ORIENTATION_S`; architecture §5 step 1 | module constant — **wiring TODO (Phase 2)** | M |
| budget.ledger_reserve_s | 45.0 | s | Floor-reserve below which no discretionary checkpoint fires | `fsm.budget.LEDGER_RESERVE_S` | module constant — **wiring TODO (Phase 2)** (`CallLedger.allow` reads global) | H |
| budget.explore_budget_numerical_s | 210.0 | s | Soft exploration budget, NUMERICAL | `interfaces.EXPLORE_BUDGET_S[NUMERICAL]`; architecture §5 | module constant — **wiring TODO (Phase 2)** | M |
| budget.explore_budget_object_reference_s | 240.0 | s | Soft exploration budget, OBJECT_REFERENCE | `interfaces.EXPLORE_BUDGET_S[OBJECT_REFERENCE]` | module constant — **wiring TODO (Phase 2)** | M |
| budget.explore_budget_instruction_following_s | 270.0 | s | Soft exploration budget, INSTRUCTION_FOLLOWING | `interfaces.EXPLORE_BUDGET_S[INSTRUCTION_FOLLOWING]` | module constant — **wiring TODO (Phase 2)** | M |
| budget.cap_parse | 2 | calls | Hard call cap, `parse` checkpoint | `fsm.budget.CHECKPOINT_MAX["parse"]`; architecture §3 | constructor arg (`CallLedger(caps=...)`) | M |
| budget.cap_miss_recovery | 1 | calls | Hard call cap, `miss_recovery` | `CHECKPOINT_MAX["miss_recovery"]` | constructor arg | M |
| budget.cap_anchor_confirm | 3 | calls | Hard call cap, `anchor_confirm` | `CHECKPOINT_MAX["anchor_confirm"]` | constructor arg | M |
| budget.cap_verification | 1 | calls | Hard call cap, `verification` | `CHECKPOINT_MAX["verification"]` | constructor arg | H |
| budget.cap_frontier_select | 1 | calls | Hard call cap, `frontier_select` | `CHECKPOINT_MAX["frontier_select"]` | constructor arg | L |
| budget.cap_self_consistency | 2 | calls | Hard call cap, `self_consistency` | `CHECKPOINT_MAX["self_consistency"]` | constructor arg | M |

---

## Phase-2 wiring TODO (the 14 module-constant fields)

Before the sweep can move these, each needs an injection seam (constructor arg,
function param, or a module setter) added to its owning module — deliberately **not**
done now (the task forbids touching those modules). Grouped by owner:

- **`core/nav/planner.py`** — `unknown_cost_mult`, `pinch_disc_m`,
  `pinch_corridor_half_w_m` (read as globals in `astar` / `_pinch_costmap`).
- **`core/nav/occupancy.py`** — `grow_pad_cells` (`_ensure_bounds`).
- **`core/nav/exploration.py`** — `coverage_saturated_free_frac` (reserved / unused).
- **`core/interfaces.py`** — `question_budget_s`, `forced_assembly_s`,
  `watchdog_floor_s`, `terrain_free_max`, and the three `explore_budget_*` values.
- **`core/fsm/budget.py`** — `orientation_s`, `ledger_reserve_s` (both read as globals).

Everything else (geometry Thresholds, fusion/tracker/keyframe configs, the nav
class-parameter fields, and the six checkpoint caps) is injectable today by passing
the swept dataclass/args at the call site.
