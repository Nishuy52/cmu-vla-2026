# Issue #59 probe — leg-goal placement diagnosis

Battery run: `reports/gt_battery_probe59/` (BEFORE fix), `reports/gt_battery_post59/`
(AFTER fix). GT ceiling: `reports/issue59_leg_ceiling.json`.

Regenerate (from `src/`):

```
python -m core.runner.gt_leg_ceiling --groundtruth ../data/vla3d/Unity --out ../reports/issue59_leg_ceiling.json
python -m core.runner.gt_battery --groundtruth ../data/vla3d/Unity --out ../reports/gt_battery_probe59
```

## Probe design

`gt_battery.py`'s `GTQuestionScore.leg_probe` (new field, probe-only, never read by
scoring) adds per leg:

- `our_goal` / `our_instance_id` — the rubric's resolved leg goal and the instance(s)
  our resolver grounded it to (`_if_rubric_geometry`, extended to also return
  `leg_instance_ids`).
- `min_dist_driven_to_goal_m` — minimum distance from our OWN driven trajectory
  (densified, same step as the scorer) to `our_goal` — the magnitude version of the
  rubric's boolean "reached", so a miss's *size* is visible, not just its verdict.
- `dist_goal_to_gt_traj_m` — minimum distance from `our_goal` to the nearest point on
  the real GT reference trajectory (mapped into the object frame via the scene's fitted
  transform) — i.e. does our resolved goal sit anywhere the human path actually went,
  independent of whether our own vehicle got there.

A companion ceiling check (`core.runner.gt_leg_ceiling`, pre-existing) measures the same
`dist_goal_to_gt_traj_m`-style distance but is a faster no-driving pass; it was run first
as a sanity check.

## Headline numbers (BEFORE fix, probe59)

- GT reference-trajectory ceiling: **38/72 (52.8%)** of our resolved leg goals are ever
  within 0.8 m of the real path at all — even the human's own recorded route. The
  ordered-leg credit (16.1%) sits well below even that ceiling, so both effects are
  live: our goals are sometimes off-target, and our own driven path sometimes misses
  goals the ceiling shows are on-target.
- Of the 72 leg rows: 12 reached in order, 60 missed. Of the 60 misses:
  - **32/60** — `our_goal` is far from BOTH the GT path (`dist_goal_to_gt_traj_m > 0.8`)
    AND our own driven path (`min_dist_driven_to_goal_m > 0.8`, when computable) —
    consistent with either wrong instance/leg-mismatch, and unsurprising that a bad
    target is never approached by the drive either.
  - **28/60** — `our_goal` sits WITHIN 0.8 m of the real GT path (a plausible target)
    but our own driven trajectory never comes within 0.8 m of it — a driving/route
    problem independent of whether the target itself was right.
  - **0/60** — the reverse (goal far from GT path, driven path near it anyway).

### Terminal vs non-terminal split (the dominant signal)

Splitting `dist_goal_to_gt_traj_m` by whether a leg is the LAST leg of its question
(the terminal goal, which anchors the per-scene sim<->object frame fit) vs an
intermediate leg:

| | far from GT path (>0.8 m) | total |
|---|---|---|
| terminal leg | 5 | 30 (17%) |
| non-terminal (intermediate) leg | 29 | 42 (**69%**) |

Non-terminal legs are ~4x more likely to be far from the real path than terminal legs.
This is the dominant pattern in the data — not a random spread across all legs.

### Splitting the non-terminal misses further

For the 42 non-terminal-leg-far cases with a resolvable single anchor (excluding
corridor legs, which have no single "candidate pool" to check), we asked: does a
DIFFERENT same-class instance exist that WOULD have landed within 0.8 m of the GT path?

- **7 legs** — yes, a same-class alternative within tolerance existed and our resolver
  picked a worse one. This is a genuine, fixable resolver bug (see below).
- **9 legs** — no same-class candidate anywhere in the scene lands within tolerance.
- **10 legs** — only one candidate of that class exists (no alternative to have picked).

For the single/no-alternative-candidate legs, the best available distance ranges widely
(0.39 m to 3.19 m) — a mix of near-misses (plausibly a stop-point/frame-noise offset)
and clear misses (a different area of the room entirely), not one clean mode.

## Root cause of the fixable 7-leg bucket

Traced one concretely (`chinese_room`, "Go near the potted plant on the table..."):
`resolve()`'s fallback ladder dropped the `ON(table)` clause entirely (category-only
rung) because **no** candidate passed it as a hard filter — including the correct one
(instance 27, footprint IoM 69% with a table, correct height band) — because the
`anchor_larger` gate in `core/geometry/toolbox.py::on()` requires the supporter's
footprint to strictly exceed the object's own footprint. A wide-canopy potted plant's
AABB can exceed a small side table's footprint even while genuinely sitting on it, so
`on()` fails the hard gate for every candidate, `resolve()` falls back to `category_only`
and ranks the survivors by **plain instance id** (a floor-standing, unrelated plant with
the lowest id wins) — discarding all evidence of which candidate actually resembles "on
the table" once the hard clause can't be satisfied by anyone.

## Conclusion: dominant mode is NOT parse-level leg decomposition

The parser (`regex_tier`) correctly splits every sampled question into legs matching the
sentence's clause structure — verified directly (`parse_regex(...)` dumps match the
english exactly, e.g. "go near the stool under the picture and stop at the table
farthest from the columns" -> 2 legs, `stool UNDER picture` / `table FARTHEST_FROM
column`). This is not a leg-decomposition/parse bug, so the STOP condition in the issue
does not apply — the failure lives in leg-goal *placement* and *driving*, in scope for
this task.

The evidence splits into two real, distinct failure modes, in order of leg-count impact:

1. **Structural leg-goal/GT-path mismatch for intermediate legs (majority, ~35-45/72
   legs)**: even when our resolved instance is defensible (no better same-class
   candidate exists, or only one candidate exists at all), the recorded GT reference
   trajectory frequently never comes within several metres of it. Combined with the
   terminal-vs-non-terminal split (17% vs 69% far), this points at a systemic gap
   between our per-clause "each intermediate landmark mention is a must-visit waypoint"
   route model and what these GT trajectories actually validate for legs other than the
   final one — a rubric/GT-data characteristic, not a bug in our grounding or parsing.
   Not attempted here (out of proportion for this task's scope; would need change to
   either the GT-side leg semantics, which is not ours to redefine, or `scoring.py`,
   which is explicitly off-limits).
2. **Resolver ranking bug, fixed (7/72 legs directly, likely more indirectly)**: the
   `on()` anchor-size gate zeroing the soft ranking score too (not just the hard
   pass/fail) meant `resolve()`'s category-only fallback degenerated to an arbitrary
   instance-id tie-break whenever no candidate could satisfy a relation as a hard
   filter — discarding real "closest match" information. Fixed in
   `core/geometry/toolbox.py` (see below).
3. **A driven-trajectory / navigation gap** also visible in the 28/60-miss bucket
   above (goal plausible, our own drive never gets there) — not chased to a root cause
   here; the codebase already documents a related known limitation ("the vehicle
   beelines to the terminal and the ordered leg scores 0", `core/heads/instruction.py`
   `_goto_point` docstring, T11 sig-1) with partial mitigation already in place
   (`nearest_reachable_point` BFS snapping). A full fix is a `core/nav`/`core/heads`
   investigation beyond this task's remaining scope; flagged for follow-up.

## Fix applied (Phase 2)

`src/core/geometry/toolbox.py`:

- `on()`: the soft ranking `score` no longer multiplies by the hard `anchor_larger`
  gate (only by `vert_ok`) — the gate still fully controls `passed` (unchanged hard
  semantics), but a near-miss (right height, strong footprint overlap, just failing the
  coarse size-order heuristic) now carries a meaningful ranking signal instead of
  collapsing to the same zero score as a totally unrelated candidate.
- `resolve()`: new `_relaxed_relation_order` step — when the fallback ladder drops every
  relation clause (category-only rung) because none pass as a hard filter, survivors are
  now ranked by best-effort match to the DROPPED clauses (soft `PredResult.score`, summed
  across clauses) before falling back to the existing deterministic (tier-priority /
  instance-id) tie-break. Only engages when `hard_clauses` is empty post-ladder AND the
  target actually had relation clauses to begin with — every already-passing case
  (survivors found the normal way) is untouched.

`src/core/runner/gt_battery.py`, `src/core/runner/gt_leg_ceiling.py`,
`src/core/runner/cvsweep.py`: `_if_rubric_geometry` now also returns
`leg_instance_ids` (4-tuple return instead of 3); call sites and tests updated.
New `GTQuestionScore.leg_probe` field + `_min_dist_to_polyline` /
`_min_dist_driven_to_goal` / `_leg_probe_rows` helpers in `gt_battery.py` (probe-only,
no scoring semantics touched — `core/groundtruth/scoring.py` was not modified).

## Before / after battery numbers

| metric | before (`gt_battery_probe59`) | after (`gt_battery_post59`) |
|---|---|---|
| mean ordered-leg credit | 0.1611 | **0.1722** |
| IF headline (mean rubric_score) | 0.1500 | 0.1500 (unchanged) |
| numerical true-accuracy | 1.0 (15/15) | 1.0 (15/15), unchanged |
| object-reference mean IoU | 1.0 (6/6) | 1.0 (6/6), unchanged |
| threading violations | 7 | 7, unchanged |
| avoid violations | 0 | 0, unchanged |

Ordered-leg credit improved (0.161 -> 0.172, target `> 0.161` met) with zero regression
on numerical/object-reference/threading/avoid metrics (verified by a row-by-row diff of
every non-IF question between the two runs: 0 differences across 45 rows). The fix
flips exactly one leg's grounding to a same-class instance that lands within tolerance
of the GT path (`home_building_1`, "First, go to the nightstand with a clock on it,
then take the path between the dining table and the picture, and stop at the trash can
closest to the refridgerator" — the leading `nightstand WITH clock` leg goes from
resolving an unrelated instance to the correct one, `n_legs_reached_in_order` 0->1 of 3);
another leg (`chinese_room` potted-plant-on-table) also re-grounds correctly but the
driven trajectory still doesn't reach it (a separate, un-investigated navigation gap),
so it doesn't flip.

**IF headline (mean `rubric_score`) target (`> 0.150`) was NOT met** — it stays at
exactly 0.1500. The one leg-count gain above is fully offset by that same question's
PRE-EXISTING (unchanged before/after, `n_threading_violations` 1->1) corridor-threading
penalty: `rubric_score = clamp(ordered_leg_credit - penalty/n_legs)` = `clamp(1/3 -
1/3) = 0` both before and after, since the penalty fraction (1 violation / 3 legs)
exactly cancels the ordered-leg-credit gain (1/3) for this specific question. This is a
coincidental cancellation on the one leg the fix reaches, not evidence the fix is
ineffective — `ordered_leg_credit` (the pre-penalty signal the fix actually targets)
moved cleanly in the right direction with no regression anywhere else measured.

## Recommendation / follow-up

- The dominant remaining gap (mode 1 above, majority of the 60 missed legs) is a
  structural mismatch between our per-clause leg model and what GT trajectories
  validate for non-terminal legs — worth its own issue/investigation, likely owned by
  whoever can revisit either the route-generation contract or (with sign-off) the
  rubric's arrival semantics for intermediate legs; not something `core/nav`,
  `core/heads`, or resolver changes alone can close. Filed as
  https://github.com/Nishuy52/cmu-vla-2026/issues/61.
- The navigation gap (mode 3) — driven trajectory not reaching a correctly-placed
  intermediate leg — is a second, separate follow-up candidate for `core/nav`/
  `core/heads`. Filed as https://github.com/Nishuy52/cmu-vla-2026/issues/62.
