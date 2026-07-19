# Issue #76 — wall derivation for home_building_1 / livingroom_1: investigation notes

Branch `if76-wall-derivation`, worktree `.claude/worktrees/if76-wall-derivation`.
Surface: `core/runner/gt_battery.py` wall-derivation path (`_synthetic_from_gt`,
`_scene_wall_cells`, `_derive_wall_cells`, `_gt_footprint_bounds`,
`WALL_FIT_MAX_RESIDUAL_M`) — function-disjoint from #75's `_resolve_anchor_rec`.

## Phase 1 — root cause per scene

Ran `collect_scene_fit_residuals` live (this worktree's own data) for all 15 scenes:

```
hotel_room_1    0.195      loft            0.295      livingroom_2   0.382
home_building_2 0.196      office_2        0.385      office_1       0.449
japanese_room   0.459      studio          0.559      arabic_room    0.574
chinese_room    0.661      livingroom_4    0.692
--- WALL_FIT_MAX_RESIDUAL_M gate (0.8) ---
home_building_1 0.879      livingroom_1    0.926      hotel_room_2   0.975
--- S._ALIGN_RESIDUAL_GATE_M scoring gate (1.0) ---
livingroom_3    1.810   (unfittable, meth-F11, out of scope)
```

All four flagged scenes ship `traversable_area.ply` (checked directly —
`_traversable_ply_path(...).exists()` is `True` for all four; not a missing-file
problem). All four have a fitted sim->object frame (frame is not `None`). The flag in
every case is **the residual gate**: `frame_for_walls = frame if residual <=
WALL_FIT_MAX_RESIDUAL_M else None` at gt_battery.py:1319-1322 — `home_building_1`
(0.879), `livingroom_1` (0.926), `hotel_room_2` (0.975) all clear the 1.0 m *scoring*
alignment gate (so Frechet/coverage/tolerance stay live) but sit just above the
tighter 0.8 m *wall-derivation* gate (issue #52), so `_scene_wall_cells` is never
called for them — same mechanism, same distance band, no per-scene variation. This
matches the note text captured in `reports/gt_battery_walls_2026-07-19/
gt_battery_report.md`/`wall_realism_analysis.md`.

`livingroom_3` (1.810) is the confirmed frame-fit-unfittable case (meth-F11) — outside
this issue's scope per the brief; not touched.

## Phase 1b — does relaxing the gate actually unblock the 6 legs? (empirical test)

Rather than assume "gate too tight -> relax it -> legs move" (issue #76's premise), I
monkeypatched `WALL_FIT_MAX_RESIDUAL_M` and `_WALL_DILATE_CELLS` in-process (no file
edits) and re-scored each flagged scene directly against `score_scene`, comparing
`rubric_score` / `n_legs_reached_in_order` at the gate's current 0.8 m against 1.0 m
(matching the scoring gate) across a dilation sweep (0.2 m .. 2.0 m pad, i.e.
`_WALL_DILATE_CELLS` 2..20 — well past the documented 0.2 m sampling-gap sizing, deep
into "large enough to absorb the alignment-gate residual" territory):

**livingroom_1** (residual 0.926, both IF questions):
| dilate pad | Q0 (potted plant, 2 legs) | Q1 (lamp/candle corridor, 3 legs) |
|---|---|---|
| walls declined (current, gate 0.8) | rubric 1.0, legs 2/2 | rubric 0.0, legs 1/3 |
| walls forced on, gate 1.0, any pad 0.2m-2.0m | rubric 0.0, legs 0/2 | rubric 0.0, legs 0/3 |

Turning walls on for livingroom_1 **breaks a currently-fully-correct question** (Q0:
2/2 -> 0/2) and also **regresses** Q1's one already-reached leg (1/3 -> 0/3), at every
dilation tested. This reproduces exactly the misplacement failure mode that
`WALL_FIT_MAX_RESIDUAL_M` was calibrated to prevent (see the constant's docstring,
gt_battery.py:200-209, and `reports/gt_battery_walls_2026-07-19/
wall_realism_analysis.md` "The one flipped question" — same scene, same 0.926
residual, same goal near `(1.782, -4.299)`). It is not fixed by more dilation: the
misprojection here isn't a narrow-doorway sampling gap, it's mesh coverage genuinely
displaced relative to the goal by more than 2 m at every pad tested.

**home_building_1** (residual 0.879, both IF questions):
| dilate pad | Q0 (coffee/dining table, 2 legs) | Q1 (nightstand/corridor, 3 legs) |
|---|---|---|
| walls declined (current, gate 0.8) | rubric 0.0, legs 0/2 | rubric 0.0, legs 1/3 |
| walls forced on, gate 1.0, any pad 0.2m-2.0m | rubric 0.0, legs 0/2 (unchanged) | rubric 0.0, legs 0/3 (regressed) |

Q0 fails identically with or without interior walls — it is **not a wall-derivation
problem at all**: the coffee-table leg goal sits at object-frame x ~ 7.55, but this
scene's `traversable_area.ply`, mapped through the fitted frame, only covers x in
[-16.04, 4.51] — a genuine ~3 m mesh-coverage gap that no dilation up to 2.0 m closes,
and unrelated to `_gt_footprint_bounds`'s (much larger, outdoor-instance-inflated —
id 81 "unknown" is a ~54x57 m yard/ground-plane mesh) outer rectangle, since that
rectangle already applies identically in the walls-off baseline (`_synthetic_from_gt`
always uses `_gt_footprint_bounds(gt, 0.0)` for the room's outer boundary,
independent of `wall_cells`) and Q0 fails there too. Q1 regresses the same way as
livingroom_1's Q1: its one currently-reached leg (leg 0, nightstand) is lost the
moment any interior wall cells are stamped, at every dilation tested.

**hotel_room_2** (residual 0.975, negative control per the issue): both IF questions
score identically (rubric 0.5 legs 2/2, rubric 1.0 legs 3/3) whether walls are
declined or forced on — confirms the issue's own framing that the wall-unavailable
flag is necessary-but-not-sufficient for a miss, and that this scene is unaffected
either way (0 legs at risk from any gate change).

## Verdict — no principled fix exists within wall-derivation scope

The premise of issue #76 (interior walls are simply missing for these scenes and
deriving them would unblock the 6 legs) does not hold under direct measurement:

- The residual gate (`WALL_FIT_MAX_RESIDUAL_M = 0.8`) is doing exactly the job its
  docstring describes — home_building_1/livingroom_1/hotel_room_2 sit in the same
  0.88-0.98 m band the constant's own history (`reports/gt_battery_walls_2026-07-19`)
  already identified as unsafe for wall stamping (misprojects onto real floor).
  Relaxing it to the 1.0 m scoring gate does **not** unblock any of the 6 legs named
  in the issue and **does** regress 2 already-correct legs (livingroom_1 Q0's full
  2-leg credit, home_building_1 Q1's leg 0) plus livingroom_1 Q1's leg 0 — net
  negative, not neutral.
- Increasing `_WALL_DILATE_CELLS` (the only other tunable in the wall-derivation path)
  from 0.2 m up to 2.0 m pad — an order of magnitude past its documented
  sampling-gap sizing — does not recover any of the regressed/blocked legs either;
  the coverage gaps involved are larger than any dilation defensible as "bridging
  point-cloud sampling gaps" rather than "papering over misalignment," which is
  exactly the tradeoff the existing docstring (gt_battery.py:192-197) already warns
  against.
- `home_building_1`'s Q0 (coffee/dining table leg) failure is **not a wall-derivation
  defect at all** — it fails identically with walls fully declined, so it is some
  other, out-of-scope navigation/planning gap (matches probe-v2's own note that 2 of
  home_building_1's 4 legs are "plain double-misses ... unexplained by any of
  (a)/(b)/(c)", i.e. already known to not have an identified mechanism).

No threshold or dilation change is justified by the fit statistics gathered here —
per `docs/calibration.md`'s generalization protocol, a residual-gate relaxation must
be justified by the fit statistics, and here the fit statistics (a scene-by-scene
before/after comparison across the flagged band) show the opposite of a
justification: they show the current 0.8 m gate is correctly placed, not
accidentally strict. **No code change is made in this worktree.**

## Per-leg ledger (baseline = current `main`, gate 0.8, unchanged)

| scene | question | leg | current state | walls-forced-on (gate 1.0, tested) | verdict |
|---|---|---|---|---|---|
| livingroom_1 | Q0 potted plant/candle holder | leg 0 | reached | NOT reached | forcing walls on regresses this leg |
| livingroom_1 | Q0 potted plant/candle holder | leg 1 | reached | NOT reached | forcing walls on regresses this leg |
| livingroom_1 | Q1 lamp/black chair (corridor) | leg 0 | reached | NOT reached | forcing walls on regresses this leg |
| livingroom_1 | Q1 lamp/black chair (corridor) | leg 1-2 | not reached | not reached | unchanged either way — not a wall-derivation fix target |
| home_building_1 | Q0 coffee/dining table | leg 0-1 | not reached | not reached | unchanged either way — NOT a wall-derivation problem (mesh coverage gap present regardless) |
| home_building_1 | Q1 nightstand/corridor | leg 0 | reached | NOT reached | forcing walls on regresses this leg |
| home_building_1 | Q1 nightstand/corridor | leg 1-2 | not reached | not reached | unchanged either way — not a wall-derivation fix target |
| hotel_room_2 | Q0 bench/bed/lamp (corridor) | leg 0-1 | reached (5/5 across both Qs) | reached (unchanged) | negative control confirmed — 0 legs at risk |
| hotel_room_2 | Q1 picture/door (corridor) | leg 0-2 | reached | reached (unchanged) | negative control confirmed |
| livingroom_3 | both IF questions | all legs | not reached | n/a (frame unfittable, out of scope) | meth-F11, unchanged, out of scope |

Net: forcing walls on for the flagged band **loses 3 currently-reached legs**
(livingroom_1 x2, home_building_1 x1) and **gains 0**. The 6 legs the issue names as
"blocked" stay blocked either way; they are not a wall-derivation-shaped problem.

## Phase 3 — measurement

No code change was made (Phase 2 found no principled fix), so no `--out
../reports/gt_battery_post76` battery run was needed to demonstrate a delta — the
in-process before/after comparisons above (run against this worktree's live data,
identical `score_scene` call the battery itself uses) already show, per-leg, that the
6 named legs do not move and that forcing the change through would cost 3 currently-
passing legs. Re-running the full battery would reproduce the unchanged `main`
baseline exactly (no lines touched).

## Recommendation for the issue tracker

Issue #76's diagnosis (wall derivation "fails" for these scenes due to a fixable gap)
does not hold up under measurement — the gate is intentionally, correctly declining
wall derivation for exactly the residual band where it has been shown (both
historically and in this investigation) to place walls on real floor. Suggest closing
#76 as "investigated, not a defect" with a pointer to this report, or re-scoping it to
the actual unexplained mechanism behind home_building_1's Q0 (2 legs, fails with or
without walls, no (a)/(b)/(c)/(d) signature per probe-v2) — that is a genuine,
currently-unowned gap, but it is a navigation/planning question, not a
wall-derivation one, and outside this worktree's function-ownership boundary
(`core/nav`, not `core/runner/gt_battery.py`'s wall-stamping helpers).
