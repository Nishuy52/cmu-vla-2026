# H5 — Geometry predicate-form fixes: implementation report

Reconciled per hardening_backlog.md H5, dossier_deltas.md §A + the T8
reconciliation appendix (appendix wins over §A on A1/A8/A4), T8 verdicts
C2–C6, T7 diagnosis D3/D4/S2, and attack_numerical.md F2. No coefficients
were tuned — forms were pinned; the (post-H2) sweep owns the coefficients.

## Outcome

All seven H5 predicate/attribute form changes plus the T7-S2 scorer companion
landed in `src/core/geometry/toolbox.py` (+ a helper primitive), with the one
allowed parsing edit and the required coordinated calibration/doc/sweep edits.
The NUM-F2 empirical failures (pillows-on-sofa, pictures-above-bed) now pass as
named regression tests; the required tuck-under, strict-betweenness, size-gap,
next_to→near, and with≡inverse-on regressions are added. No unexpected
regressions: the only two existing goldens changed both encoded the OLD
semantics the spec deliberately reverses.

## Files changed

- `src/core/geometry/toolbox.py` — `Thresholds` field set; `on`, `above`,
  `under`, `between`, `with_feature` predicate bodies; `_attrs_match` +
  new `_size_attr_match` size resolver; `UNDER_RELATION` class list +
  `_under_relation_anchor`; `_BINARY_PREDS[Pred.NEXT_TO]` → `near`; module
  docstring line.
- `src/core/geometry/primitives.py` — new `footprint_area`, `footprint_iom`
  (intersection-over-min), `largest_face_area`.
- `src/core/parsing/regex_tier.py` — near-synonym routing (`next to`,
  `beside`, new `adjacent to` / `close to`) → `Pred.NEAR`; `_split_continuation`
  masks the two new multi-word tokens so corridor-leg splitting is unaffected.
- `src/core/groundtruth/scoring.py` — T7-S2 surgical fix: `_scene_graph_count`
  expands `between` `[id,id]` pair entries into member ids instead of
  str()-garbling them.
- `src/core/runner/cvsweep.py` — `default_sweep_spec`: dropped `on_vert_tol`
  key (field removed), added `on_upper_span_frac`, `on_top_tol`,
  `above_lateral_infl`; retargeted `on_min_overlap_frac` grid around 0.5.
- `src/core/calibration.py` — no edit needed: `Calibration.geometry` composes
  `Thresholds` by dataclass reference, so new/removed fields flow through
  automatically. (Verified by the pinning test.)
- `docs/calibration.md` — geometry table rebuilt for the new field set;
  wiring-summary counts updated (geometry 11→15, total 55→59, wireable 41→45).
- Tests: `src/tests/geometry/test_predicates.py`,
  `src/tests/geometry/test_resolve.py`,
  `src/tests/groundtruth/test_scoring.py`, `src/tests/test_calibration.py`.

## Thresholds fields added / changed / removed

Removed:
- `on_vert_tol` (0.15) — the top-face-only band the support-semantics form
  replaces (T8-C3/D3).
- `above_gap_max` (3.0) — dead (declared, never consumed). Dropped rather than
  wired: an upper bound on the above() vertical gap has no generation-spec
  counterpart and would spuriously reject a high picture over a low headboard
  (justified inline in `above` and in dossier A8). Task said pick one — dropped.

Changed:
- `on_min_overlap_frac` 0.30 → **0.50**, semantics changed from
  overlap/target-area to footprint **intersection-over-min** (A2 + T8-C2).

Added:
- `on_upper_span_frac` = **0.25** — on() upper z-band lower edge at
  `anchor.zmin + this·height` (T8-C3/D3, sweepable).
- `on_top_tol` = **0.15** — on() allows `a.bottom` up to this far above the
  anchor AABB top (sweepable).
- `above_lateral_infl` = **0.50** — above() anchor-footprint inflation for the
  lateral-offset gate that REPLACES the overlap gate (T8-C4/D4, sweepable).
- `under_iom_min` = **0.50** — footprint IoM-over-min gate for both under()
  branches (DD-A7).
- `under_tuck_tol` = **0.15** — tuck-under floor tolerance
  (`target.min_z ≤ anchor.zmin + this`) and strict-branch slack.
- `size_sep_gap` = **1.20** — size resolver's min largest-face-area ratio for a
  "small"/"big"/"largest" extreme (DD-A12 = T8-C6).

Net geometry field count 11 → 15 (pinned in `test_calibration.py`
`test_field_counts_per_subsystem`, updated 11→15).

## Predicate/attribute changes (spec mapping)

1. **on()** (T8-C2/C3, D3): three gates — IoM-over-min ≥ `on_min_overlap_frac`;
   anchor-larger (`b_footprint > a_footprint`); target bottom in the upper
   z-band `[zmin + on_upper_span_frac·height, ztop + on_top_tol]`. Pillow bottom
   0.2–1.1 m below sofa/bed AABB top with 100% overlap ⇒ TRUE.
2. **above()** (T8-C4, D4; REVERSES §A A8): footprint-overlap gate REPLACED by a
   lateral-offset tolerance — XY centre within anchor footprint inflated by
   `above_lateral_infl`; positive vertical gap kept; dead `above_gap_max`
   dropped. Wall-hung picture, zero overlap, small lateral offset ⇒ TRUE.
3. **under()/below()** (DD-A7): two branches, both require IoM ≥ `under_iom_min`.
   (i) strict `a.top ≤ b.zmin + under_tuck_tol`; (ii) tuck-under, gated to
   `UNDER_RELATION` anchor classes (verbatim from vla_3d.md, stored
   space/underscore-tolerant), `a.min_z ≤ b.zmin + under_tuck_tol` AND
   `a.top ≤ b.ztop`. Stool under table passes via (ii).
4. **next_to / beside / adjacent to / close to → near** (DD-A5): regex_tier
   routes all four surface forms to `Pred.NEAR`; `_BINARY_PREDS[Pred.NEXT_TO]`
   also points at `near` (belt-and-braces for an LLM-emitted `next_to`). The
   tight `next_to()` function stays defined under its own name; nothing routes
   to it.
5. **with_feature ≡ inverse-on** (DD-A6): `with(a,b)` primary = `on(b,a)` under
   the new on(); the old footprint-pad test kept only as an explicitly-audited
   relaxation rung (score capped below any strict-on score; explanation names
   the rung). `allow_pad_rung=False` forces strict inverse-on.
6. **between()** strict-betweenness (T8-C5): added `0 < t < 1` on the capsule
   projection; capsule/radius unchanged; symmetry/IoM deferred to the sweep.
7. **Size resolver** (DD-A12 = T8-C6): `_size_attr_match` ranks the same-class
   pool by largest-face area with a 1.2× (`size_sep_gap`) separation gap; no
   separated extreme ⇒ the size attribute matches nothing (honest none). Wired
   into `_attrs_match` (new optional `pool`/`th` params) alongside the untouched
   colour bridge; all three call sites (`resolve`, `counting`, `_resolve_anchor`)
   pass the same-class pool.

Companion — **scorer T7-S2**: `_scene_graph_count` now expands nested
`between` `[id,id]` pairs into their member anchor ids (all other relations stay
flat), so between-anchor labels resolve instead of being str()-garbled.

## Existing tests modified (each justified)

- `test_predicates.py::test_on_boundary_vert_tol` → renamed
  `test_on_boundary_top_tol`. **Justification:** it encoded the OLD top-face-only
  `on_vert_tol` band (asserted `on()` toggles at `b.top ± on_vert_tol`). That
  band is exactly what T8-C3/D3 reverse. Rewritten to assert the new upper-band
  top edge (`b.top + on_top_tol`), which is the point the spec still keeps. Added
  two sibling tests (`test_on_lower_band_edge_rejects_floor_object`,
  `test_on_anchor_larger_gate`) so the two NEW gates are covered, not just the
  relabelled one. This test now demanding the new behaviour is the point, not a
  weakening.
- `test_predicates.py::test_between_collinear_endpoints` → renamed
  `test_between_at_endpoint_fails_strict`, assertion flipped
  `passed` → `not passed`. **Justification:** it asserted a target sitting
  exactly AT an anchor centroid (projection t=0, off the segment end) counts as
  "between". Strict betweenness (T8-C5) is defined precisely to reject that; the
  flip encodes the spec, it is not a convenience weakening.
- `test_calibration.py::test_field_counts_per_subsystem` geometry count 11 → 15.
  **Justification:** mechanical pin update for the new `Thresholds` field set;
  the test exists to notice a dropped/added field, so it must track it.

No other existing test was altered. `test_above_positive`,
`test_above_negative_no_overlap`, `test_under_positive/negative`,
`test_with_feature_*`, and every `resolve`/`counting` golden pass unchanged under
the new forms.

## New tests added

`test_predicates.py`:
- `test_on_pillow_on_sofa_num_f2`, `test_on_pillow_on_bed_num_f2` — NUM-F2
  pillow cases (bottom 0.2–1.1 m below AABB top, asserted in-range).
- `test_above_wall_picture_num_f2` — NUM-F2 wall-hung picture (zero overlap,
  passes) + `test_above_far_picture_still_rejected` (tolerance not unbounded).
- `test_under_stool_tucked_under_table`, `test_under_tuck_gated_to_under_relation_class`,
  `test_under_strict_branch_still_works` — tuck-under stool + class gating +
  strict branch.
- `test_between_off_segment_end_rejected` — strict-betweenness off-end rejection.
- `test_size_attr_match_largest_with_gap`, `test_size_attr_match_smallest_with_gap`
  — size-resolver 1.2× separation gap (and honest-none when not separated).

`test_resolve.py`:
- `test_next_to_clause_uses_near_predicate` — next_to routes to near
  (+ asserts the routing table).
- `test_with_clause_is_inverse_on`, `test_with_hanging_lamp_not_supported` —
  with ≡ inverse-on.
- `test_resolve_big_table_relative_per_class`, `test_resolve_size_no_separation_relaxes`,
  `test_counting_big_table_relative` — relative size in resolve/counting.

`test_scoring.py`:
- `test_scene_graph_between_pair_not_garbled` (+ `_sg_with_between_pair` helper)
  — T7-S2 between-pair scorer fix.

## Test-run tallies (from `src/`)

- Targeted tier — `pytest tests/geometry tests/heads tests/groundtruth tests/perception -q`:
  **340 passed, 20 skipped, 0 failed.**
- Fast tier — `pytest -q`: **774 passed, 28 skipped, 3 errors** — the 3 errors
  are the known `test_regex_full_set` worktree data-gate errors (missing
  `upstream/.../questions.json`), unrelated to this change.
- Full gate — `pytest -n auto -m ""`: **807 passed, 38 skipped, 5 failed,
  3 errors** — all 8 non-passes are the documented worktree data gates
  (4 `test_battery` + 1 `test_cli` failing on missing `DEFAULT_QUESTIONS`
  scene data; 3 `test_regex_full_set` errors on missing questions.json). No new
  failures introduced.
