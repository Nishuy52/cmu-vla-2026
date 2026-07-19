# Issue #75 notes -- salience reorder gating

Branch: if75-salience-gating (worktree
`.claude/worktrees/if75-salience-gating`)

## Root cause

`_resolve_anchor_rec` (gt_battery.py) and `_ranked_anchor` (instruction.py)
each end with a same-labeled-survivor-group raw-distance-to-previous-leg
reorder (issue #71, tightened #73). Both UNCONDITIONALLY ran that reorder
whenever the group had >1 member, discarding whatever real evidence
`resolve()`'s own ranking already encoded within that group (a `between()`
clause matched with different soft margins; the issue #59
`_relaxed_relation_order` category-only soft score) -- not just across
different labels, which #71/#73 already protected.

## Fix

Added `_same_label_group_is_tied(same, clause, idx, th, eval_clause)` in
both files (private helper, direct helper of the two owned functions).
Re-evaluates the anchor's own disambiguator clause (the exact clause
`resolve()` itself would score) independently against each same-labeled
survivor and only permits the raw-distance reorder when every survivor's
`PredResult.score` AND `PredResult.margin` for that clause are equal within
float tolerance (1e-9) -- or the anchor carries no disambiguator at all
(the pre-#75 genuine no-clause tie case, unchanged).

Margin had to be checked alongside score: `office_2` q1 leg1 (folder on
cabinet closest to whiteboard) has every survivor's `on()` score quantised
to 0.0 (all fail the footprint-IoM hard gate), which a score-only check
reads as tied -- but `margin` (continuous slack) cleanly separates the
z-band-passing survivors (69/70/76, margin ~+0.15..+0.18) from the
z-band-failing ones (117/118, margin ~-0.13) that the previous
unconditional reorder was wrongly promoting via raw distance to the prior
leg. `ResolveResult.pass_matrix` could NOT be reused for this check: the
category_only rung empties `hard_clauses` before building the pass matrix,
so the matrix carries zero rows for exactly the legs that needed gating.
Re-evaluating the clause directly (via `core.geometry.toolbox._eval_clause`,
already a private import pattern used elsewhere in these files) was the
only way to see the same evidence `resolve()`'s own soft-fallback ranking
used, from outside `resolve()`.

Files touched (per ownership -- only the two owned functions + their
direct helpers):
- `src/core/runner/gt_battery.py`: `_same_label_group_is_tied` helper
  (module scope, near `_if_rubric_geometry`) + gated call site inside
  `_resolve_anchor_rec`. Added `Sequence` import (typing) and
  `_eval_clause`/`DEFAULT_THRESHOLDS` to the existing local toolbox import.
- `src/core/heads/instruction.py`: mirrored `_same_label_group_is_tied`
  helper (module scope, after `_has_superlative`) + gated call site inside
  `_ranked_anchor`.

## Verification

`tools/resolver_parity.py` + `tools/resolver_parity_adjudicate.py`
(offline diagnostics, unmodified) against `data/vla3d/Unity` (15 scenes,
72 legs):

| metric | before | after |
|---|---|---|
| id-divergent legs | 3 | 1 |
| `home_building_2` q1 leg2 (potted plant between curtain/TV) | wrong: id=221 (dist_to_gt_traj 5.34) | correct: id=58 (dist_to_gt_traj 0.85) |
| `office_2` q1 leg1 (folder on cabinet closest to whiteboard) | wrong: id=118 (dist_to_gt_traj 1.05) | correct: id=69 (dist_to_gt_traj 0.77) |

Both target legs now converge to the GT-correct pick on BOTH the rubric
side (`_if_rubric_geometry`/`_resolve_anchor_rec`) and the head side
(`InstructionHead._ranked_anchor`) -- verified directly via
`_if_rubric_geometry` leg_instance_ids for `office_2` q1
(`[(16,), (69,), (81,)]`).

Side effects (not required by the task, harmless): two other
previously-id-divergent legs also converged as a side effect of gating
(`chinese_room` q0 leg0 -> 27/27, `studio` q0 leg0 -> 24/24), dropping the
divergent count to 1 (`japanese_room` q1 leg1, unrelated, still diverges:
rubric=1 vs head=0 on a `via_near` "wardrobe doors" leg with no clause
evidence at all -- a genuine tie, unaffected by this fix, out of scope).

Full fast test tier (`pytest` from `src/`): all green, no failures
(only the existing intentional skips).

## Battery

`gt_battery --groundtruth data/vla3d/Unity --out reports/gt_battery_post75`,
compared against the existing `reports/gt_battery_post73` baseline (same
75-question/15-scene set):

| metric | post73 (baseline) | post75 (this fix) |
|---|---|---|
| IF headline (rubric-proxy mean) | 0.461 | 0.417 |
| IF mean ordered-leg credit | 0.522 | 0.478 |
| threading violations (tv) | 7 | 7 (unchanged, as required) |
| avoid violations | 0 | 0 |
| non-IF rows (numerical/referential) | unchanged | unchanged (confirmed by diff) |

tv did not increase (met). IF headline/credit went DOWN, not up as
naively hoped -- traced to the DRIVEN-trajectory arrival check, not to
resolver correctness:

- `office_2` q1: leg1's resolved goal moved from the old (wrong) id=118 at
  (4.52, 0.21) to the new (GT-correct, id=69) goal at (5.13, 3.14).
  `n_legs_reached_in_order` dropped 3/3 -> 2/3 (`reached_in_order: false`
  for leg1) purely because the DRIVEN trajectory/arrival-tolerance check
  fails to register arrival at the new (farther, differently-placed) goal
  -- same pattern on `livingroom_1` and `studio` (both moved as a
  documented side effect of the same gate firing on other same-label
  groups; both also regressed 1.00->0.50 / 0.50->0.00 for the identical
  reason). `home_building_2` q1 stayed rubric=0.00 either way (already
  failing on a DIFFERENT leg's threading violation before this fix, so
  the leg2 recovery had no headline room to show).
- Full per-row diff (`diff <(grep '^| ' post73/report.md) <(grep '^| '
  post75/report.md)`) confirms every regressed row is one of the 3 legs
  whose resolved *goal position* moved (office_2, livingroom_1, studio);
  no other row changed at all.

This is arrival/driving behavior, not resolver behavior: the resolver is
now MORE correct (verified above via the GT-trajectory-distance proxy and
direct instance-id inspection), but the driven-path arrival check --
owned by the concurrent #74 lane (route assembly / BreadcrumbFollower /
threading, explicitly out of this lane's ownership) -- doesn't yet
reliably reach a goal that moved to a new, still-valid position. Per the
task brief: "your final numbers get re-measured on integrated main after
the concurrent #74 lane merges -- relative deltas are your success
metric" -- the resolver-parity delta (this lane's actual measure) is
strictly positive (3 -> 1 id-divergent legs, both target legs recovered
to the GT-correct id on both sides); the battery-credit regression is
expected to close once #74's arrival/threading fixes land on top of this.
