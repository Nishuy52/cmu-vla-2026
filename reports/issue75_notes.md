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

`gt_battery --out ../reports/gt_battery_post75` run pending final commit;
expect the 2 recovered legs to move IF credit on `home_building_2` q1 and
`office_2` q1, non-IF rows unchanged, tv not increasing (route
assembly/threading untouched -- out of scope, #74 lane's surface).
