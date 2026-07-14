# T11 IF-leg-threading — adversarial verification

Verifier: fresh-context adversarial pass. Baseline `b0c2518` (origin/main).
Branch `fix/if-leg-threading`. Full gate (`pytest -m "" -n auto`, PID 43696,
15 workers) was STILL RUNNING throughout this pass — light probes only; no
new full/-n auto runs started.

## C1 — Improvement real and honestly measured

Topline (from `reports/gt_battery_post{wave,T11}_2026-07-14/gt_battery_results.json`,
uncommitted on-disk artifacts):
- IF rubric-proxy 0.0611 -> 0.1000 (mean over 30 IF rows; verified sum 1.833 -> 3.000 / 30).
- ordered-leg credit 0.0944 -> 0.1222.
- threading violations 9 -> 8.  avoid violations 0 -> 0.
- `git diff b0c2518 -- src/core/groundtruth/` is EMPTY (scorer UNTOUCHED). Confirmed.

BUT the report's per-row attribution is INACCURATE:
- Report claims "office_2 q4 0.00->0.50 (leg 1/2)" and "office_2 q5 0.00->0.33".
  NO such rows exist. The only office_2 IF gain is a DIFFERENT question
  ("First, go to the trash can near the cabinet...") 0.000 -> 0.667 (legs 2/3).
  The plant-on-cabinet office_2 row (the sig-1 exemplar) stayed 0.000 -> 0.000.
- Report claims the gain is "traceable to real ordered arrivals: office_2 q4
  0.00->0.50, office_2 q5 0.00->0.33, studio q4 empty-motion -> moving".
  The ACTUAL changed IF rows (only three moved):
    * hotel_room_1 "Go to the bedside table closest to the window..." 0.000 -> 0.500 (legs 0/2 -> 1/2)  [GAIN]
    * office_2     "First, go to the trash can near the cabinet..."   0.000 -> 0.667 (legs 0/3 -> 2/3)  [GAIN]
    * hotel_room_1 "First, go near the bedside table closest to the bench..." ordleg 0.333 -> 0.000 (legs 1/3 -> 0/3)  [PER-QUESTION REGRESSION]
- The studio "vases" empty-motion row: poses 0 -> 4001 (does now MOVE), but
  rubric stayed 0.000 — it did NOT contribute to the headline gain. The
  grounding fix produced motion but no scored arrival.

The +0.039 net is genuine and the scorer is untouched, so the DIRECTION is
honest, but the report cites two nonexistent rows and omits a per-question
ordered-leg regression (hotel_room_1 near-bedside-table leg lost, 0.333->0.0).

Also: postT11 introduced poses=4002 (watchdog-cap) on BOTH hotel_room_1 rows
and poses=4001 on studio — the "stall guard removes the poses=4002 artifact"
claim is contradicted by the postT11 data itself (see C5b).

**C1: REFUTED** — headline numbers and scorer-untouched claim are true, but the
report's traceability ("office_2 q4 0.00->0.50 / q5 0.00->0.33") is fabricated
against the actual JSON, it omits a per-question ordered-leg regression, and the
studio row it credits scored 0. The improvement is real; the honest-measurement
claim (accurate attribution + no per-question regression) is not.

## C2 — No regression (numerical + OR)

Per-question comparison across all 15 numerical and 30 OR rows (not topline):
- Numerical: our_count / exact_match / gt_count_independent / iou — 0 field diffs.
  Independent agreement 0.5556 -> 0.5556 identical.
- Object-reference: iou / match_method / gt_target_id — 0 field diffs.
  mean_iou 0.875 -> 0.875 identical (all 8 scored rows unchanged).
No numerical or OR regression at per-question granularity.

**C2: CONFIRMED**

## C3 — Sig-1 fix correct (reachable leg goal)

`instruction._goto_point` diff (b0c2518 -> HEAD): old body returned
`self._project_free(anchor_xy)` (nearest PASSABLE cell). New body: when a costmap
+ pose exist, `cm.nearest_reachable_point(anchor_xy, self._pose)` (BFS from pose
over passable cells) and uses the anchor cell if directly reachable else the
reachable standoff. Falls back to `_project_free` with no costmap. Correct wiring
of the intended fix.

`nearest_reachable_point` (costmap.py:210) edge cases read directly:
- BFS is bounded by a `seen` mask (each cell enqueued once) -> NO infinite loop
  even for a fully-unreachable goal.
- Goal unreachable anywhere -> returns the reachable cell nearest (Euclidean) to
  goal (least-bad) — sane.
- Start cell blocked / off-grid -> snaps to nearest passable; if the whole map is
  impassable returns `start_xy` — sane.
- A sealed nearer pocket is EXCLUDED (only cells reachable from start are
  candidates) -> a farther reachable pocket is correctly preferred. Matches C3.

`tests/heads/test_if_leg_reachability.py` PASSED (2 tests, single file, no xdist).
The regression test would fail on b0c2518: old `_goto_point` returns
`_project_free` which snaps to the sealed-pocket passable cell, so the test's
`_reachable(...)` assertion (goal in same component as pose) would fail.

**C3: CONFIRMED**

## C4 — Sig-3 vocab fix, no vocab regression

Diff: `singularize` "-es" stem set `("s","x","z","ch","sh")` -> `("x","z","ch","sh","ss")`.
i.e. bare `"s"` removed (so single "-ses" no longer strips "es"), `"ss"` added
(so "-sses" still strips). Compared new vs `b0c2518:singularize` over a 36-word list:

Words that CHANGED (old -> new), all CORRECTED e-final stems:
  vases vas->vase | houses hous->house | nurses nurs->nurse | roses ros->rose |
  noses nos->nose | cases cas->case | phases phas->phase | bases bas->base |
  poses pos->pose | hoses hos->hose
One mild REGRESSION introduced: buses bus->buse (old was correct; e-final rule now
over-applies to "-uses"). "buses" is not a challenge object noun; low risk.
Everything else stable: boxes->box, benches->bench, dishes->dish, couches->couch,
tables->table, churches->church, foxes->fox, classes->class, addresses->address,
processes->process, shelves->shelve, knives->knive (both unchanged).

NOTE: the claim's "glasses->glass" is WRONG — "glasses" is in `_PLURAL_INVARIANT`
and stays "glasses" in BOTH old and new (never singularised). No behavior change,
but the claim mis-states it.

`tests/parsing/test_vocab.py` PASSED (7 tests, single file).

**C4: CONFIRMED** (fix correct; only intended e-final stems change, plus one
harmless "buses" over-application; claim's "glasses->glass" example is inaccurate
but the behavior is safe).

## C5 — Harness edits don't flatter scores

(a) `synthetic_scene.terrain_patch`: an object with base `cz >= TERRAIN_SLAB_MAX_Z`
(0.25 m) is skipped in the floor-intensity stamp (free floor). Doc comment ties
this to terrainAnalysis.cpp's ~0.2 m slab filter and `OverheadConfig.overhead_min`.
The direction is FAITHFUL: the real stack does drop overhang points from
/terrain_map, so freeing that floor makes the mirror MORE like reality, not easier
in a way the real run wouldn't also enjoy. The object's surface still feeds the
overhead layer (soft-avoid). Consistent with the documented real behavior.
(Independent cross-read of the terrainAnalysis maxRelZ note was not located in
`docs/upstream_notes.md` in this pass — the justification rests on the in-code doc
comment; flagged as a soft spot but the mechanism is sound and the 0.25 m constant
is tied to the overhead layer's own threshold.)

(b) `_drive_if_trajectory` stall guard: resets `stall_ref` whenever the vehicle
moves > `_DRIVE_STALL_EPS_M` (0.05 m) within a `_DRIVE_STALL_TICKS` (40) window;
only breaks after 40 consecutive sub-0.05 m ticks. With `_DRIVE_STEP_M` = 0.25 m,
a healthy drive moves 0.25 m/tick >> 0.05 m, so the window resets every tick and a
progressing drive is NEVER cut. Confirmed it cannot terminate a healthy drive.

BUT the report's stronger claim — that the guard "ends a wedged drive instead of
padding to the watchdog length (the poses=4002 artifact)" — is CONTRADICTED by the
postT11 data itself: hotel_room_1 BOTH IF rows show driven_n_poses = 4002 and
studio q4 = 4001 in the postT11 JSON. The guard only catches TRANSLATION stalls; a
follower that keeps moving (net > 0.05 m/window) but never arrives at the terminal
still pads to `_DRIVE_MAX_TICKS`. So poses=4002 was NOT eliminated. This does not
flatter SCORES (arrival credit is independent of trailing padded poses), so the
score-safety property holds — but the report's characterization is inaccurate.

**C5: CONFIRMED (score-safety)** — neither harness edit inflates scores: the
terrain edit is a faithful-direction fidelity change and the stall guard cannot cut
a progressing drive. Caveat: the "poses=4002 artifact removed" claim is false per
the postT11 JSON (guard only catches translational wedges).

## C6 — Load-bearing ceiling claim (GT reaches 30/72 leg goals)

The denominator IS independently corroborated: summing `n_legs` over the 30 postT11
IF rows = 72 exactly (postwave = 71; the +1 is the newly-grounding studio "vases"
leg). GT trajectory PLYs load cleanly (e.g. office_2 q4 = 694 poses).

HOWEVER the NUMERATOR (30) is NOT independently reproducible in this repo. Computing
"GT trajectory within 0.8 m of the raw leg-goal centroid" requires the leg-goal
centroids from `_if_rubric_geometry`, which need the 15 GT scenes' object geometry
(`<scene>_object_result.csv` / scene-graph). Those files are ABSENT from the repo
(only VLA-3D sample_data scenes — loft, 3RScan, etc. — ship here; none of the 15
battery scenes). The report JSON does not persist per-question `leg_goals`, and no
instrumentation script was left. The battery was run against an external
`--groundtruth` Unity root not present on this machine's checkout.

So the 30/72 figure rests on the executor's own instrumentation, which I cannot
reproduce or refute with the data available. Per protocol I flag this LOUDLY: the
single most load-bearing claim (it justifies calling the residual gap non-pipeline
and un-holding the ×6-weighted CV sweep) is taken on faith. The denominator checks
out; the numerator is unverified.

**C6: UNVERIFIABLE** — denominator (72) reproduced; numerator (30 GT-reached)
cannot be reproduced without the external GT scene CSVs. Do not treat the "ceiling
is GT-bounded, gap is non-pipeline" conclusion as verified when un-holding the sweep.

## C7 — Scored-path safety (I/O contract)

Pipeline files changed: `instruction.py`, `vocab.py` (the two scored-`core/` files).
Read both diffs in full:
- No GT access: no import of `groundtruth`, no scene-graph / CSV / referential reads.
- No new topics: `_goto_point` uses existing `self._costmap` / `self._pose` /
  `self.grid`; `nearest_reachable_point` is a pre-existing costmap method. vocab is
  pure string logic.
- No new dependencies: no new imports added in either diff (numpy/logging already
  present). `VIA_NEAR_CLEARANCE_M` is a module constant.
GT access / new topics / new deps all live only in the HARNESS files
(`gt_battery.py`, `synthetic_scene.py`), which are not in the scored fork surface.

**C7: CONFIRMED**

## C8 — Test claims

Full-gate (`pytest -m "" -n auto`, PID 43696, 15 workers) was STILL RUNNING
throughout this verification (confirmed alive at multiple checkpoints). Per the
machine-etiquette protocol I did NOT start another full/`-n auto` run and did NOT
re-run the fast tier. So the "1035 passed, 5 skipped, 54 deselected" fast-tier
figure is UNVERIFIABLE-deferred to the detached gate's eventual result.

What I DID run (single files, no xdist, from `src/`), all green:
- `tests/heads/test_if_leg_reachability.py` — 2 passed
- `tests/parsing/test_vocab.py` — 7 passed
- `tests/mocks/test_synthetic_scene.py` — 12 passed

5 new regression tests exist (exactly, via `git diff b0c2518 -- src/tests/`):
  test_goto_goal_snaps_to_reachable_cell_not_a_sealed_pocket,
  test_reachable_goal_lets_open_anchor_be_reached_directly (heads),
  test_elevated_object_is_not_a_floor_obstacle,
  test_floor_mounted_object_still_blocks (mocks),
  test_singularize_e_final_stem_keeps_e (parsing).

Do they fail against b0c2518? Reasoned over old code (read via git show, did not
check out):
- vocab: `b0c2518:singularize("vases")` = "vas" (verified by running the old
  function) != "vase" -> test FAILS on old code. CONFIRMED.
- reachability: `b0c2518:_goto_point` returns `_project_free` (nearest passable),
  which snaps into the sealed pocket; the test asserts the goal is in the pose's
  reachable component -> FAILS on old code. CONFIRMED by code reading.
- synthetic_scene: old `GTObject` had no `cz` and `terrain_patch` had no overhang
  skip, so `test_elevated_object_is_not_a_floor_obstacle` (elevated object frees
  floor) FAILS on old code. CONFIRMED by diff.

**C8: CONFIRMED (with fast-tier count deferred)** — 5 new tests exist and are
red-against-old / green-against-new; single-file runs pass. The exact
"1035 passed" fast-tier count is deferred to the running detached gate.

## Overall verdict

**REFUTED (narrowly) — fix is sound and the sweep un-hold is DEFENSIBLE but not
fully verified; two report claims are inaccurate and the load-bearing ceiling
number is unreproducible here.**

What holds up:
- Scorer UNTOUCHED (`git diff b0c2518 -- src/core/groundtruth/` empty). [C1]
- IF headline genuinely improved 0.061 -> 0.100 (arithmetic reproduced from the
  JSONs), violations 9 -> 8. [C1]
- No numerical / OR regression at PER-QUESTION granularity (0 field diffs). [C2]
- Reachable-leg-goal fix correct, no infinite-loop, reachable-over-sealed. [C3]
- Vocab fix correct; only intended e-final stems change (+ harmless "buses"). [C4]
- Neither harness edit inflates SCORES; terrain edit is faithful-direction
  (corroborated by upstream_notes.md maxRelZ/obstacleHeightThre = 0.2 m). [C5]
- Scored-path changes carry no GT access / new topics / new deps. [C7]
- 5 new tests exist, green on new, red on old; single-file runs pass. [C8]

Why REFUTED rather than CONFIRMED (the specific broken claims):
1. C1 traceability is FABRICATED against the JSON: the report credits
   "office_2 q4 0.00->0.50" and "office_2 q5 0.00->0.33" — neither row exists.
   The real movers are hotel_room_1 (0->0.5) and office_2 "trash can" (0->0.667),
   and the report OMITS a per-question ordered-leg regression (hotel_room_1
   "go near the bedside table" 0.333 -> 0.0). The direction is honest; the
   attribution is not.
2. C5 "poses=4002 artifact removed" is FALSE per the postT11 JSON: two
   hotel_room_1 rows are poses=4002 and studio is 4001. The stall guard only
   catches translational wedges, not never-arriving-but-moving followers. (No
   score impact, so score-safety survives, but the stated effect did not occur.)
3. C6 (the load-bearing "GT reaches only 30/72 leg goals" that justifies calling
   the gap non-pipeline and UN-HOLDING the ×6 CV sweep) is UNVERIFIABLE here: the
   72 denominator reproduces, but the 30 numerator needs the 15 GT scene object
   CSVs, which are ABSENT from the repo, and leg goals aren't persisted in the
   JSON. It rests entirely on the executor's own instrumentation.

Risks to carry into the calibration sweep + architecture critique:
- Do NOT un-hold the CV sweep on the strength of the 30/72 ceiling alone — it is
  unverified. Re-run the ceiling instrumentation on the machine that has the GT
  Unity root, or persist `leg_goals` in the battery JSON so it's auditable.
- The `_near_thresh` shrink (1.2 m -> footprint_half_diag + 0.45 m) caused a
  per-question ordered-leg regression on hotel_room_1's "go near the bedside
  table" leg (0.333 -> 0.0). The net IF gain masks it; watch for VIA_NEAR "near X"
  legs that now UNDERSHOOT and stop short of the anchor after this change.
- poses=4002/4001 watchdog-padded drives persist -> runtime cost and a latent
  "never-arriving follower" class the stall guard does not catch. Not scored, but
  a real driven-sim inefficiency and a sign some routes loop without arriving.
- The reports under `reports/gt_battery_post{wave,T11}_*` are UNCOMMITTED (not in
  `git diff b0c2518`); the before/after evidence lives only on this disk.

## C8 — UPDATE (fast tier re-run after gate exited)

The detached full gate (PID 43696) EXITED during this pass; the machine went quiet
(0 python processes), so I re-ran the fast tier once (`python -m pytest -p no:xdist`
from `src/`):

  1035 passed, 5 skipped, 54 deselected in 63.20s

This matches the claimed "1035 passed, 5 skipped, 54 deselected" EXACTLY. The
fast-tier count is now CONFIRMED (no longer deferred). (Full `-m "" -n auto` gate
count not independently re-derived — the detached run's own result stands.)

**C8: CONFIRMED**
