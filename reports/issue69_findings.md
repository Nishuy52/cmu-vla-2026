# Issue #69 findings — A1/A2/D1/A3 resolver-rule investigation

Baseline (fresh `gt_leg_ceiling`, current `main` @ cfc8f4b): **35/72 = 0.4861**,
matching `reports/ceiling_diagnosis_2026-07-19/classification.md`.

## Result summary

| Rule | Legs predicted | Legs actually fixable | Ceiling after | Verdict |
|---|---|---|---|---|
| A1 superlative-anchor resolution | 5 | 0 | 35/72 = 0.4861 (unchanged) | not a resolver bug — diagnosis artifact |
| A2 ON/WITH fallback ranking | 5 | 0 | 35/72 = 0.4861 (unchanged) | not a resolver bug — diagnosis artifact |
| D1 corridor gate-midpoint redefinition | 2 | 0 (structural fix implemented, does not close either leg) | 35/72 = 0.4861 (unchanged) | real defect fixed, but does not lift these 2 legs |
| A3 bare-noun category tie-break | 1 | 0 (not implemented) | 35/72 = 0.4861 (unchanged) | conflicts with existing validated contract (#21) |

**Final ceiling: 35/72 = 0.4861 — no change from baseline.** The cumulative 65/72
(0.9028) ceiling predicted by the classification is not achievable via any
principled, parameter-free resolver-side rule change confined to
`core/geometry/toolbox.py` / `core/geometry/primitives.py`, for the reasons below.
Battery: `if_rubric=0.200`, `if_thread_viol=7`, `if_avoid_viol=0`,
`or_instance_match=6/6`, `or_iou=1.000` — identical before and after (see
`reports/gt_battery_baseline_if69/` vs `reports/gt_battery_post69/`).

## A1 + A2: root cause is a diagnosis-script artifact, not a resolver bug

`classification.json`'s per-leg method computes, for every miss, the minimum
distance from a same-class alternative instance to the GT reference trajectory
— but that trajectory is the WHOLE multi-leg polyline, not the segment of it
that corresponds to the leg under test (`gt_leg_ceiling.measure` does the same:
`d = min(||gt_xy - goal||)` over every point of the full trajectory, for every
leg). A same-class instance that happens to sit near a LATER leg's own target is
therefore indistinguishable, by this method, from a same-class instance that was
actually mis-resolved for THIS leg.

Traced all 10 flagged legs (5 A1 + 5 A2) against the real resolver and the
trajectory-fraction where each candidate is nearest the GT path:

- **A1** (superlative `closest_to`/`farthest_from`): in every case the anchor
  noun ("hookah", "fan decoration", "pyramid candle holder", "black chair",
  "easel") resolves to a SINGLE unambiguous instance, and the resolver's
  chosen candidate is the objectively closest/farthest one by Euclidean
  distance (verified numerically, 2D and 3D agree) — the ranking is correct.
  The flagged "better" alternative sits at trajectory fraction 0.87–1.0 (the
  very end of the route, i.e. near a LATER leg's own destination), except one
  case (livingroom_1 q4) where the alternative sits at fraction 0.0 (the
  recorded route's spawn point, not a visited target).
- **A2** (ON/WITH relation clause): in every case the resolver's chosen
  candidate PASSES the clause cleanly (`on()`/`with()` score 1.0, full
  footprint overlap, correct height band), while the flagged "better"
  alternative FAILS it outright (0% footprint overlap, or the clause's own
  anchor is 2–13 m away) — not a marginal footprint/height edge case, a clean
  semantic mismatch. The alternative again correlates with a later leg
  (fraction 0.6–1.0) or, in `home_building_2`'s case, a magazine 1.5 m+ from
  the scene's only ottoman (there is no ambiguity to resolve).

Implementing either "fix" as literally described (rank a same-class instance
higher because it lies nearer the full GT path) would not repair a bug — the
resolver is already producing the semantically correct answer in all 10 cases
— it would make the resolver actively wrong (feeding an unrelated instance to
real navigation) purely to chase a scoring artifact. Per the generalization
protocol and the "never a per-scene special case" rule, no change was made.
The underlying diagnosis-method issue (whole-trajectory vs. per-leg-segment
distance) is a `gt_battery.py` / `core/groundtruth/scoring.py` concern, out of
this task's surface.

## D1: corridor gate-midpoint — real defect fixed, does not close the 2 legs

`corridor_gate()` computed the gate's face points from independent per-axis
footprint-OVERLAP projection (`aabb_face_points_2d`). For `livingroom_1` q5's
sofa/round-table leg the two footprints already touch in a 1.3 cm sliver, so
this projection collapsed to a **zero-width gate** (`p0 == p1`) — a physically
meaningless "between" point no route can be judged against, independent of any
scoring concern. `studio` q5's couch/table leg is not degenerate but the
per-axis midpoint pins to the smaller anchor's own centroid, ignoring the
larger anchor's actual facing edge.

**Fix implemented** (`core/geometry/primitives.py::centroid_axis_face_points_2d`,
wired into `core/geometry/toolbox.py::corridor_gate`): walk the CENTROID-to-
centroid line out to each anchor's own footprint boundary — the same axis
`between()` already uses to define "a between b1 and b2" (capsule along the
centroid segment) — instead of the independent per-axis overlap construction.
This is a genuine internal-consistency fix (two different geometric
definitions of "between" existed in the same module) and always yields a
non-degenerate two-point segment. Verified against the existing symmetric
`corridor_gate` tests (identical results there) plus new tests for the
previously-degenerate touching-footprint case
(`tests/geometry/test_primitives.py`, `tests/geometry/test_corridor_avoid.py`).

**Measured effect**: `livingroom_1` q5 leg1 goal moves from (-0.7105, -2.4325)
to (-0.7105, -2.3548) — 1.908 m from the GT nearest point (was 1.884 m).
`studio` q5 leg1 goal moves negligibly (1.733 m vs 1.736 m). **Neither leg
crosses the 0.8 m tolerance; the ceiling is unchanged at 35/72.** No other leg
in the 15-scene battery changed (`leg_ceiling_baseline_if69.json` vs.
`leg_ceiling_post69_d1.json` diff to zero).

Both cases show the SAME pattern: one anchor's own AABB sits at 0.0 m from the
GT path (the demonstrator brushed past it directly) while the true crossing
point is offset toward that anchor at an angle no anchor-pair-only geometric
construction can predict without either (a) GT-path knowledge, which
`corridor_gate` structurally cannot have (it is also the real-navigation
via-point, computed blind to any demonstration), or (b) cross-leg route
context (the previous leg's approach direction), which would require a
signature/call-site change to `gt_battery.py::_if_rubric_geometry` and
`core/heads/instruction.py` — both outside this task's surface (`gt_battery.py`
is explicitly owned by another agent). The classification's own suggested
scoring-side alternative ("score by whether the GT path crosses the gate
segment... or move the rubric goal to the path's actual closest crossing
point") is a `core/groundtruth/scoring.py` / `gt_battery.py` change, not a
`toolbox.py` one — also out of surface. The degenerate-gate fix is kept
because it corrects a real, independently-verifiable defect (an unthreadable
zero-width gate affects real navigation too, not just this metric), but D1's
predicted 2-leg ceiling lift does not materialize from anything achievable on
this surface.

## A3: not implemented — conflicts with an existing, deliberately-tested contract

`livingroom_3` q5 leg1 ("the cabinet", bare noun, no clause) resolves via
`_tier_priority_order`'s bare-noun branch, which is DELIBERATELY exempt from
exact/synonym-vs-head-noun-cousin tier preference (issue #21) and falls back to
plain instance-id order. The classification's suggested fix — prefer the exact
"cabinet" (id 88) over the head-noun cousin "tv cabinet" (id 41) — is exactly
the behaviour issue #21 turned OFF for bare queries, and is directly
contradicted by the existing, still-passing test
`test_resolve_bare_noun_keeps_instance_id_order_with_no_clause`
(`tests/geometry/test_resolve.py`), which asserts a bare query must NOT prefer
an exact match over a lower-id cousin (added specifically so "table"
legitimately means every table, cousins included, regardless of exactness).
Trajectory-fraction tracing of this leg is also ambiguous (chosen candidate at
fraction 0.998 — plausibly the terminal leg's own target, not this leg's — vs.
alternative at 0.714, itself later than this 3-leg route's expected leg-1
window). Given (a) the fix as described would regress a validated,
purpose-built contract, and (b) the evidence for what "correct" even means
here is inconclusive, no change was made — consistent with
classification.md's own hedge ("a single-leg symptom; not enough evidence here
to prescribe a general rule").

## Files touched

- `src/core/geometry/primitives.py` — new `_ray_footprint_exit_2d` +
  `centroid_axis_face_points_2d` helpers.
- `src/core/geometry/toolbox.py` — `corridor_gate` now calls
  `centroid_axis_face_points_2d` instead of `aabb_face_points_2d`.
- `src/tests/geometry/test_primitives.py` — 3 new tests for the new helper.
- `src/tests/geometry/test_corridor_avoid.py` — 1 new test for the
  previously-degenerate touching-footprint case.

## Reproduction

```
cd src
../.venv/bin/python -m core.runner.gt_leg_ceiling \
    --groundtruth ../data/vla3d/Unity --out ../reports/leg_ceiling_post69_d1.json
../.venv/bin/python -m core.runner.gt_battery \
    --groundtruth ../data/vla3d/Unity --out ../reports/gt_battery_post69
../.venv/bin/python -m pytest -q
```
