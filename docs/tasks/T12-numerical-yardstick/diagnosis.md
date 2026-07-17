# T12 — Diagnosis of the 4 true numerical failures

**Task:** T12 Task 5 (diagnose + fix the 4 true numerical count failures).
**Date:** 2026-07-17 · **Branch:** `feat/t12-numerical-yardstick`
**Baseline:** `reports/gt_battery_T12pre_2026-07-17/` (commit `8f7333e`),
TRUE numerical accuracy 11/15. Reproduced in isolation this session
(`reports/scratch_t12diag_four`, uncommitted): the 4 false rows are
exactly arabic_room (0 vs 2), home_building_1 (11 vs 6),
home_building_2 (3 vs 2), loft (0 vs 2).

Method: per-question `probe_t12.py` (uncommitted scratch) traces the
counting decision over the GT `BasicSceneIndex` — noun pool →
attribute filter → per-clause per-candidate predicate result → final
count. Instance ids below are VLA-3D object ids from the loaded GTScene.

Cause buckets (T7 convention): resolver code defect · calibration
residue · GT-index / annotation-vocabulary artifact · question-semantics.

---

## arabic_room — "How many sofas are below a window?"

- **Ours 0 · true 2.**
- **Bucket: resolver code defect** (predicate-form incompleteness — the
  vertical mirror of the accepted D4/H5 `above()` fix).
- **Mechanism.** Parser routes "below" → `Pred.UNDER` → `under()`.
  Base pool = 3 sofas {40, 56, 74}; 4 windows {20, 65, 75, 76} resolve
  as the anchor. `under()` requires footprint IoM-over-min ≥ 0.50 in
  BOTH its strict and tuck branches. Windows are wall-mounted (z-span
  1.36–2.26 m, a thin sliver on the wall plane); sofas sit on the floor
  and extend into the room, so sofa↔window footprint IoM = 0% for every
  pair → every candidate FAILS → count 0. This is exactly the wall-hung
  blindness D4 identified for `above()` (pictures over a bed have zero
  footprint overlap), which H5/T8-C4 fixed for `above()` by REPLACING
  the overlap gate with a lateral-offset tolerance — but the same fix
  was never applied to the `below`/`under` direction.
- **True answer is consistent with the GT geometry** via the
  inverse-of-`above` form: `below(sofa, window) ≡ above(window, sofa)`
  (window's XY centre within the sofa's inflated footprint AND window
  bottom strictly above sofa top) passes for exactly
  sofa 40 (window 20 above it) and sofa 74 (window 75 above it) → 2.
  The scene-graph `below` edges contain NO sofa→window pair (the
  VLA-3D generator's own `relate_below` uses the same footprint-IoM
  gate and is blind to this case), so the true answer of 2 reflects the
  natural-language wall-relative meaning of "below a window", not a
  generator edge — the identical situation D4 documented for `above`.
- **Decision: FIXED** — add the lateral-offset (inverse-`above`) branch
  to `under()` (see home_building_1; one fix covers both). Commit + test
  recorded in the Fixes section below.

## home_building_1 — "How many pillows are on the sofa under the pictures?"

- **Ours 11 · true 6.**
- **Bucket: resolver code defect** (same wall-mounted-anchor root cause,
  surfacing through a dropped nested disambiguator).
- **Mechanism.** Target = pillow, clause = `on(sofa)` where the sofa
  anchor carries a nested disambiguator `under(pictures)`. There are 4
  sofas {200, 357, 361, 376} and 2 pictures. `under(sofa, picture)`
  fails for every sofa (footprint IoM 0% — pictures are wall-hung), so
  the disambiguator is DROPPED (`audit: drop_disambiguator "under
  dropped: no 'sofa' candidate satisfied it"`) and ALL 4 sofas become
  the `on` anchor. 11 pillows sit on some sofa (6 on sofa 376, 5 on
  sofa 357) → count 11.
- **True answer is consistent with the GT geometry** via the same
  inverse-`above` form: only sofa 376 has a picture (149) laterally
  above it (`above(picture, sofa)` passes for 376 alone). With the
  disambiguator resolving to sofa 376, the `on(sofa 376)` count is
  exactly its 6 pillows {54, 59, 300, 310, 368, 382} → 6.
- **Decision: FIXED by the same `under()` lateral-branch change** — the
  disambiguator now selects sofa 376. Commit + test below.

## home_building_2 — "How many red pillows are on the sofa?"

- **Ours 3 · true 2.**
- **Bucket: color-annotation / perception gap (calibration residue).
  DIAGNOSED-UNFIXED.**
- **Mechanism.** `red` matches via the colour bridge (red → {red,
  maroon}). Three pillows carry a `maroon` alias and all sit on the one
  sofa (134): 85, 213, and 94. Raw color-scheme proportions from the
  object CSV:
  - 85 — maroon 0.78 (dominant) → red ✓
  - 213 — maroon 0.78 (dominant) → red ✓
  - 94 — **gray 0.36 (dominant)**, brown 0.21, **maroon 0.18** (3rd)
  The human answer key counts only the two maroon-dominant pillows (85,
  213 → 2); 94 is a predominantly-gray pillow that also carries a minor
  (18%) maroon component.
- **Why unfixed.** 94's maroon bin (18%) clears the VLA-3D generator's
  own 10% dominance floor (`judge_color`, prior_art/vla_3d.md §Dominant
  colors), so by the *generator's* semantics 94 legitimately IS a maroon
  object and our top-3 alias matching is faithful to it. The gap is
  between generator top-3 semantics and the human answer key's
  dominant-color perception. Closing it requires a color-salience cutoff
  (dominant-only, or a percentage threshold > 18%) — a tunable that (a)
  the task defers to the real-sim sweep and forbids chasing per-question,
  and (b) is shared with OR/IF color disambiguation, where the generator
  deliberately uses any top-3 color (vla_3d.md §referential color rule),
  so a dominant-only restriction risks OR/IF regressions unmeasurable on
  this battery. This is the "question colors vs the 15-color scheme"
  calibration residue T7 already flagged for the sweep.

## loft — "How many black pillows are on the sofa?"

- **Ours 0 · true 2.**
- **Bucket: color-vocabulary / annotation artifact. DIAGNOSED-UNFIXED.**
- **Mechanism.** No loft pillow carries a `black` alias — the scheme
  colors present are gray, maroon, olive. `black` therefore matches
  nothing → 0. The two true "black" pillows on a sofa are 89 and 91,
  both raw RGB (47, 79, 79) = "dark slate gray" — visually black, but
  the VLA-3D LAB/CSS3 classifier bins them as `gray`, the same scheme
  label as the six lighter (112, 128, 144) "slate gray" pillows on the
  sofa. The 15-color scheme *has* a `black` bin, but these points never
  landed in it.
- **Why unfixed.** The distinction between the 2 dark (47,79,79) and the
  lighter gray pillows exists ONLY in raw RGB, which `InstanceRecord`
  does not carry (aliases/caption hold scheme-color NAMES only). A
  scheme-name bridge black→gray overshoots to all 6 gray-on-sofa pillows
  (→ 6, not 2). Reaching exactly 2 requires plumbing raw RGB into the
  index plus an invented luminance cutoff to separate (47,79,79) from
  (112,128,144) — a calibration threshold to chase one question, which
  the task forbids and adjudication defers to the sweep. This is the
  "loft's black unmapped" residue T7 identified.

---

## Fixes

### FIX-1 — `under()` wall-relative below (lateral inverse-`above`) branch

- **Fixes:** arabic_room (0→2) and home_building_1 (11→6).
- **Change:** `src/core/geometry/toolbox.py::under()` gains a third branch
  (iii): `below(a, b) ≡ above(b, a)` — the anchor's XY centre within the
  target's inflated footprint AND the anchor strictly above the target
  (`b.min_z > a.max_z`), carrying NO footprint-IoM requirement. This is
  the vertical mirror of the accepted D4/H5 `above()` lateral form; the
  IoM-gated strict/tuck branches are unchanged. Because it fires only
  when the anchor is strictly above the target, it never collides with
  tuck-under (a table extends to the floor, so it is not strictly above a
  stool it shelters — verified by `test_under_tuck_gated_to_under_relation_class`
  and the new lateral tests).
- **Test:** `src/tests/geometry/test_predicates.py::`
  `test_under_wall_relative_below_lateral_branch` (pins the lateral pass
  with IoM < 0.5, so it can only pass via the new branch) and
  `test_under_lateral_branch_requires_anchor_strictly_above` (guards
  against broadening "under" to "beside" — anchor overlapping the target
  in z must NOT pass).
- **Commit:** `c867fce`
- **No calibration values were changed.** The fix is a predicate-form
  completion, not a threshold tune.

## Final numerical-slice tally

Regression-gated full-battery run (`reports/scratch_t12diag_all`,
uncommitted; `python -m core.runner.gt_battery --groundtruth
../data/vla3d/Unity --no-drive-if`):

**TRUE accuracy 13/15 (up from 11/15).** All 11 prior true passes
remained true (hard regression gate satisfied); arabic_room and
home_building_1 flipped to correct; home_building_2 (3 vs 2) and loft
(0 vs 2) remain DIAGNOSED-UNFIXED as color-annotation gaps deferred to
the sweep (rationale above).

| Scene | our | true | match |
|---|---|---|---|
| arabic_room | 2 | 2 | ✓ (fixed) |
| chinese_room | 6 | 6 | ✓ |
| home_building_1 | 6 | 6 | ✓ (fixed) |
| home_building_2 | 3 | 2 | ✗ (color annotation gap) |
| hotel_room_1 | 4 | 4 | ✓ |
| hotel_room_2 | 3 | 3 | ✓ |
| japanese_room | 3 | 3 | ✓ |
| livingroom_1 | 8 | 8 | ✓ |
| livingroom_2 | 2 | 2 | ✓ |
| livingroom_3 | 2 | 2 | ✓ |
| livingroom_4 | 6 | 6 | ✓ |
| loft | 0 | 2 | ✗ (color vocab gap) |
| office_1 | 6 | 6 | ✓ |
| office_2 | 1 | 1 | ✓ |
| studio | 3 | 3 | ✓ |
