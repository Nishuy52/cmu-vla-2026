# Task: Diagnose GT-battery numerical count disagreements

## Intent
The 2026-07-11 GT battery reports numerical pipeline
exact-match 100% but independent (referential) agreement only
15% and scene-graph agreement 27%. Decide, per disagreeing
question, whether the gap is an annotation-semantics /
comparison-granularity mismatch (calibration fix, or no fix)
or a real resolver defect (code fix) - so the planned k-fold
CV parameter sweep optimizes against real errors, not
granularity artifacts.

## Context
- Raw signal: reports/gt_battery_full_2026-07-11/
  gt_battery_report.md (per-question table) and
  gt_battery_results.json (topline counts only - no
  per-instance detail, so re-running the resolver with
  instrumentation is required for instance-level verdicts).
- Leading hypotheses: implementation-gaps task record
  (T4) holes #1 (near threshold object-size-scaled instead of
  region-volume-scaled) and #5 (relation semantics: pair-shaped
  between/hanging_on, closest/farthest suppression, big/small
  cutoffs).
- Key nuances already established (do not re-derive):
  - Disagreement is BIDIRECTIONAL: home_building_1 pillows
    pipeline=31 vs indep=18/sg=11 (over-count, smells like a
    defect); livingroom_3 photos pipeline=2 vs indep=10/sg=9
    (under-count).
  - `*_class_only` opinions are relation-AGNOSTIC (coarser);
    comparing a relation-filtered pipeline count against them
    is a comparison-granularity artifact, not a fusion bug
    (e.g. arabic_room sofas 1 vs 3).
  - This is a GT-battery run: "fusion/dedup" means the toolbox
    `counting()` + relation thresholds over GT AABB instances,
    not sensor fusion. Code: src/core/groundtruth/scoring.py
    (`_scene_graph_count`, numerical path) and
    src/core/geometry/toolbox.py (`counting`, `near`/
    `near_thresh`).
- Dataset state: data/ was absent on this machine at task
  start; Unity.zip (~2.0 GB, all 18 scenes) re-downloaded from
  the CMU AirLab bucket (docs/vla3d_notes.md section 1).

## Acceptance Criteria
- [x] Every disagreeing numerical question (13 of 15) gets a
      cause bucket: granularity artifact / near-threshold /
      over-segmentation / relation-predicate defect /
      GT-annotation quirk. (Actual buckets that emerged: D1-D4
      code defects, S1-S3 scorer artifacts, calibration
      residue - see docs/diagnosis.md.)
- [x] Each bucket gets a calibration-vs-code verdict with
      instance-level evidence (not just topline counts).
- [x] Verdicts state what the k-fold CV sweep may and may not
      optimize against.
- [x] Bonus if data permits: close T4 #2 (target_index
      id-space) and #5a (between/hanging_on pair shape)
      against the real scene-graph JSON. (Both closed: id join
      100% sound; between pair-shaped, hanging_on flat.)

## Artifacts
- Diagnosis: docs/diagnosis.md (per-question verdict table)
- Questions: questions.md (if user questions arrive)

## Stage Gates
| Gate | State | Evidence / notes |
| --- | --- | --- |
| Task | APPROVED | resumed from handoff; explicit "continue with the Next action (item 3 first)" |
| Spec | N/A | analysis-only deliverable, no code change yet |
| Plan | N/A | instrument -> dump -> bucket -> verdict |

## Todo
- [x] Verify branch/git state matches handoff
- [x] Read battery report + T4 hypotheses + scorer/toolbox code
- [x] Discover dataset missing locally; restart Unity.zip
      download (background)
- [x] Extract per-scene small files (object CSV, scene-graph
      JSON, referential JSON) for the 15 battery scenes
- [x] Instrumented resolver dump per disagreeing question
      (delegated to executor tier) -> docs/resolver-dump.md
- [x] Bucket each question + write calibration-vs-code verdict
      -> docs/diagnosis.md
- [x] Fold verdict implications into the k-fold sweep plan
      (diagnosis.md "What the k-fold sweep may optimize")
- [x] Bonus: T4 #2 and #5a checks against real JSON (both
      closed - see diagnosis.md "T4 closures")
- [ ] (follow-on, new task when picked up) implement D1-D4
      resolver fixes + S1-S3 scorer fixes, re-run battery

## Notes
- 2026-07-14: created from handoff item 3. Dataset absent
  (handoff open-question confirmed); single 2.09 GB Unity.zip
  download restarted - resumable, content-length verified
  against the bucket's header.
- 2026-07-14: diagnosis complete. Headline: NOT a calibration
  problem - 4 resolver code defects (relaxation leaking into
  counting, alias/typo noun-matcher pollution, on() z-band vs
  backrests, above() overlap gate vs wall-hung), 3 scorer
  measurement artifacts (relation never checked in
  "referential" opinion, sg direction inversion, class-only
  granularity), small calibration residue (color synonym map,
  near form). The battery's one "clean" scene (hotel_room_1)
  was right by luck - category_only fallback returned the
  correct 4. Full verdicts in docs/diagnosis.md.
