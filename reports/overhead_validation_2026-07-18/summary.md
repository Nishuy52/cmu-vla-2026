# Overhead-clearance tunables: multi-scene validation (redteam H13 / SYS-F12)

Replays the five recorded sim bags in `data/sim_bags/` through the live
`OccupancyGrid` overhead-clearance wiring (`core.nav.occupancy.OverheadConfig`
+ `integrate_scan_overhead_decimated`, mirrored via
`tools/overhead_validation.py`) and checks the resulting `OVERHEAD` flags
against each scene's ground truth (`object_list.txt` + `traversable_area.ply`
under `data/unity_scenes_ros2/`). Addresses the open Ubuntu-gate item in
`docs/calibration.md` (boxed caveat under the nav table) and redteam F12
(`docs/redteam/attack_eval_day.md`): the five tunables were fitted to a single
real-robot bag (jingfan, not present on this machine) and had never been
checked against any other scene.

Per-bag detail: `office_1_q1.json`, `japanese_room_q1.json`,
`livingroom_1_tour.json`, `livingroom_1_tour_nw.json`, `loft_q1.json` (this
directory).

## Cross-scene table

| Bag | Scene | Sensor height m (delta) | Overhead cells (flagged/observed) | False flags (n/flagged) | Known-overhang hit rate | Sensitivity delta (shifted/minpts2) |
|---|---|---|---|---|---|---|
| loft_q1 (degraded-rate) | loft | 0.75 (+0.15) | 464/23278 (2.0%) | 0/464 (0.0%) | 3.0% | -23 / +144 |
| office_1_q1 | office_1 | 0.75 (+0.15) | 1308/35101 (3.7%) | 0/1308 (0.0%) | 44.8% | -12 / +273 |
| japanese_room_q1 | japanese_room | 0.78 (+0.18) | 346/28897 (1.2%) | 0/346 (0.0%) | n/a | +1 / +36 |
| livingroom_1_tour | livingroom_1 | 0.78 (+0.18) | 874/34658 (2.5%) | 0/874 (0.0%) | 33.3% | +3 / +173 |
| livingroom_1_tour_nw | livingroom_1 | 0.78 (+0.18) | 791/33518 (2.4%) | 0/791 (0.0%) | 26.5% | +2 / +170 |

`loft_q1` is the degraded-rate specimen (sensors at 1.45 Hz, robot stayed near
spawn, 7.6 m path per `data/sim_bags/README.md`) -- its numbers are weaker
evidence (small observed/GT footprint) and are included but should be read as
a stress case, not a coverage case.

## Findings

### 1. Sensor height: `vehicle_sensor_height=0.60` under-estimates on every sim scene

Measured `median(vehicle_z) - floor_z` is **0.75-0.78 m** on all five bags
(office_1/loft ~0.75 m, japanese_room/livingroom_1 ~0.78 m) against the
configured **0.60 m** -- a consistent **+0.15 to +0.18 m delta**, all in the
same direction. The configured value was fit to the jingfan real-robot rig;
these sim-recorded bags run a visibly taller virtual sensor stack. Practical
impact is small in this data because the fallback (used only when a cell has
no terrain-derived `ground_z` yet) fires rarely -- 0, 1, 1, 10, 13 flagged
cells out of 346-1308 across the five bags used the fallback; the rest had
real terrain `ground_z` by the time they flagged. So the mis-calibrated
constant is real and consistently wrong by ~0.15-0.18 m on sim scenes, but its
observed blast radius here is small because terrain coverage catches up
quickly. It would matter more on a bag with sparser terrain coverage (a fast
tour, or a scene with more occluded floor) where more cells stay on the
fallback for longer.

Recommendation: bump `vehicle_sensor_height` to ~0.75 m (or make it
scene/bag-configurable) the next time this constant is revisited, and note
that the jingfan real-robot rig and the sim sensor mount are not
interchangeable heights.

### 2. Overhead-cell fraction: 1.2-3.7% of observed cells, scales with furniture density

`japanese_room` (low-furniture per its bag note) flags the fewest (1.2%),
`office_1` (shelves, desks, monitors, cabinets) the most (3.7%). Consistent
with the layer picking up genuine furniture, not noise.

### 3. False-flag rate: 0/flagged on all five scenes (open-floor GT and
furniture-footprint GT are almost disjoint by construction)

Every single flagged cell on every bag falls inside the GT overhang-object
footprint set (objects whose vertical extent intersects the
[floor+0.25, floor+1.20] band, footprint inflated by 1 cell) -- **zero**
flagged cells land on the rasterized `traversable_area.ply` open-floor set
that isn't also GT furniture. Spot-checked on `office_1_q1`: the open-floor
cell set (3255 cells) and the GT-footprint cell set (3619 cells) overlap in
only 6 cells total -- `traversable_area.ply` already excludes most furniture
footprints from the navigable mesh, so this metric is structurally biased
toward 0 whenever flags line up with furniture (which they do here). Read
this as "no evidence of false positives on truly open floor in these five
bags," not as "the false-flag rate is provably always 0" -- a scene with
wall-mounted objects over open floor (sconces, shelves, window sills, per the
F12 evidence list) would be a sharper test and none of the five bags
happened to drive near one at in-band height with the overhead layer
observing it.

### 4. Known-overhang hit rate: 0-99% per object, wide variance, several genuine misses

Aggregate hit rate (fraction of a named table/desk/shelf/cabinet's *observed*
footprint cells that got flagged) ranges 26-45% on office_1/livingroom_1, only
3% on loft, and is undefined (no qualifying objects) on japanese_room (its
three "table" objects are low traditional tables, tops at 0.23-0.29 m above
floor -- below the 0.4 m known-overhang classification threshold used here,
not a code defect, a genuine scene characteristic matching the bag README's
"low-furniture geometry" note).

Per-object hit rates are bimodal rather than uniformly moderate: `office_1`
shelf id 51 hits 98.8% (83/84 footprint cells flagged) while shelf id 0 (same
scene, same object type) hits only 53.3%; `livingroom_1` shelf id 73 hits
**0%** (0/39 observed footprint cells ever flagged, both livingroom bags
agree). `loft` is the weakest overall: 5 of 7 known-overhang objects hit 0%,
including a `TV cabinet` and two `table`s with fully-observed footprints (28
and 88 cells) that the vehicle drove past but the overhead layer never
flagged once. Two `loft` objects report a physically implausible `top_z` of
about 3.7 m (ids 22 and 73, both named "table") -- almost certainly a
scene-authoring artifact (a mis-tagged light fixture or duplicate mesh, not a
replay bug -- the raw `object_list.txt` values are used as-is); their hit
rates (0% and 20%) are not meaningful evidence either way and should be
excluded if this metric is used to tune the band.

The loft weak spots plus shelf-73 are exactly the missed-overhead failure
mode F12 warns about (band/point-gate too strict for some real geometry) --
worth a closer look at the `min_points_per_cell` sensitivity below before
concluding the band itself is wrong; the miss could be scan-density-driven
(loft's 1.45 Hz degraded rate means far fewer in-band returns per cell over
the same dwell time) rather than a band-placement problem.

### 5. Sensitivity: `min_points_per_cell` dominates; band-shift correction is second-order

Re-running the same decimated scan stream through `min_points_per_cell=2`
(vs. default 3) adds **36-273 more flagged cells** across the five bags (a
+8-59% swing on top of the default count) -- by far the largest lever of the
two tested. Correcting `vehicle_sensor_height` to the measured value (see
finding 1) moves the flagged count by only **-23 to +3** cells -- noise-level
by comparison, and not even monotonic in sign (loft goes down, the rest go
up slightly) because the correction only changes cells that were on the
fallback ground estimate at the moment they were scored during replay, not
just the ones still on it at the end (a cell can accumulate its in-band count
before its own terrain patch ever arrives). The takeaway: if the goal is
catching more genuine overhangs (loft's misses above), lowering
`min_points_per_cell` is the higher-leverage knob of the two calibrated
here -- but per finding 3/4 it is not obviously buying correctness margin
against false positives on this data (false-flag rate stayed 0 in the main
run; not independently re-checked against GT at `min_points_per_cell=2`,
since that scan wasn't part of the false-flag/hit-rate metrics -- flagged
counts only).

## Defects / limitations noticed in the overhead layer during this exercise

- **No code defect found in `core/nav/occupancy.py` itself.** The layer
  behaved as documented on all five bags; `integrate_patch` /
  `integrate_scan_overhead_decimated` / `mark_pose` ran to completion with no
  exceptions, NaNs, or shape errors across ~65-90k odom messages and
  ~350-1700 terrain/scan messages per bag.
- **`vehicle_sensor_height=0.60` is measurably wrong for the sim sensor rig**
  (finding 1) -- low practical impact today (fallback rarely used) but worth
  fixing before it matters on a sparser-terrain bag.
- **Known-overhang misses on `loft`** (finding 4) are consistent with F12's
  "missed overhead -> stranding" risk and are plausibly a symptom of loft's
  degraded 1.45 Hz sensor rate (fewer in-band returns per dwell) rather than
  a band-placement bug -- flagged for follow-up with `min_points_per_cell`
  tuning rather than as a confirmed defect.
- **`object_list.txt` data quality**: two `loft` objects named "table" carry
  `top_z ~= 3.7 m` (ids 22, 73) -- almost certainly scene-authoring noise, not
  a parser bug (values read directly from the file); flagged here so anyone
  reusing `loft`'s GT for calibration excludes those two rows.
- **False-flag metric is structurally close to 0 on this GT** (finding 3) --
  not a defect, but a methodology caveat: `traversable_area.ply` already
  excludes most furniture footprints, so this specific check under-tests
  wall-mounted-object false positives (the F12 evidence's specific worry
  about sconces/shelves/sills). None of the five bags drove near one of
  those at observing range.

## Reproduce

```
python -m tools.overhead_validation run data/sim_bags/<bag> data/unity_scenes_ros2/<scene>/<scene> \
    --out reports/overhead_validation_2026-07-18/<bag>.json \
    --summary-md reports/overhead_validation_2026-07-18/summary_table.md
```

Tests: `python -m pytest tools` (GT rasterization helpers, `tools/tests/test_overhead_validation.py`).
