# T13 — issue-sweep: clear the open GitHub issue queue (#10–#13, #15–#19)

## Intent

Fix all nine open issues on one branch (`fix/issue-sweep`) so the
actionable queue from T12 + the PR #14 review is empty before the
Ubuntu phase. One PR to main closing all of them.

## Context

- #10, #15–#19 are polish/hygiene items (fixtures, provenance,
  CLI/report consistency, test hygiene) — low risk.
- #13 is a genuine latent resolver bug (terminal-goal mis-ranking,
  chinese_room q5 / home_building_2 q4) worked around for battery
  alignment only by the T12 frame-fit; the underlying `resolve()`
  drives object-reference scoring and real-sim GOTO.
- #11/#12 are the two remaining TRUE-numerical failures (13/15),
  deferred from T12 because they need a shared color-salience knob.
  They are attempted here under a hard regression gate: full
  15-scene battery, `true_accuracy` must not drop from 13/15, and
  OR instance-match / IF alignment toplines must hold. If the gate
  cannot be met, the attempt is reverted and the issues stay open
  with findings appended.

## Acceptance criteria

- [x] #10: tools fixtures use `pipeline_determinism_rate`; `pytest tools` green (29ab72a)
- [x] #19: livingroom_3 registry assertion in its own named test (65fe4dc)
- [x] #15: `write_report` cal param wired or removed (single documented path) — removed: battery has no non-default calibration path; cvsweep keeps its legit `cal` (65fe4dc)
- [x] #16: unreadable answer key is loud and distinguishable from missing — stderr WARNING + topline "present but UNREADABLE" (65fe4dc)
- [x] #17: CLI summary leads with instance-match, IoU demoted to diag (65fe4dc)
- [x] #18: dirty_digest covers untracked file contents (untracked dirs by name only, documented perf guard) (2021c37)
- [x] #13: terminal-goal rank0 correct for chinese_room q5 + home_building_2 q4; OR scores + IF fit hold (25fd955; verifier CONFIRMED — root cause: `_resolve_anchor` consumed flat `by_label`, head-noun tier admitted wrong-modifier cousins ("horse figurine" matched elephant figurine), existential clause passed for the distractor, ranking fell to instance-id order. Fix: `_match_anchor_noun` keeps strongest non-empty tier of `by_label_tiered`. Full battery aggregate identical to T12post. Follow-ups filed: #20 pre-existing speaker OR mismatch, #21 latent bare-label/cousin coexistence caveat)
- [x] #11/#12 attempted: loft black-pillow == 2 and home_building_2 red-pillow == 2 with true_accuracy ≥ 13/15; else reverted + findings filed — FIXED, 15/15 (68bbf3f): `ColorBin(name, rgb, fraction)` on `InstanceRecord.color_bins` from GT CSV r/g/b+percentage columns; knobs on `Thresholds`: `colour_dominance_floor=0.50` (cross-hue bridged bins), `dark_luma_max=96.0`/`light_luma_min=220.0` (Rec.601, neutral bins only); legacy text-match fallback when bins absent; OR/IF aggregates unchanged vs T12post; docs/calibration.md updated
- [x] Full test gate green (`pytest -m ""` from src/ + `pytest tools`) — exit 0 incl. slow sims (verifier-run); fast tier 1081 passed / 28 skipped; tools 33 passed
- [x] Battery re-run + `tools/battery_diff.py` vs T12post committed under reports/ — `reports/gt_battery_T13post_2026-07-17/` incl. `diff_vs_T12post.md` (68bbf3f)
- [x] Verifier gates CONFIRMED for #13 and #11/#12 — #13: independent battery re-run, aggregate == T12post; #11/#12: bins-on/off isolation run proves exactly the 2 target rows flip, every OR/IF row byte-identical
- [x] LOG.md one-liners, INDEX.md entry, PR with `Fixes #N` lines

## Waves

1. **W1 (parallel, disjoint):** #10 (mech-executor, tools/test_battery_diff.py) ∥ #18 (executor, src/core/runner/provenance.py)
2. **W2:** #15+#16+#17+#19 (executor, src/core/runner/gt_battery.py + src/tests/runner/test_gt_battery.py)
3. **W3a:** #13 resolver fix (executor) → verifier gate
4. **W3b:** #11+#12 color salience (executor, battery-gated) → verifier gate
5. Close-out: full gate, battery diff, docs, PR

## Notes

- 2026-07-17: task opened; branch `fix/issue-sweep` off main @ 658ffdd.
- 2026-07-17: W1+W2 landed (29ab72a, 2021c37, 65fe4dc); full fast tier
  1068 passed / 0 failed after W2. Environment fix along the way: the
  git-ignored `upstream/` clone was found EMPTY (likely casualty of the
  session-15 worktree cleanup) — re-cloned Yuxin916/CMU-VLN-Challenge-2026
  (shallow), parsing tests green again. Battery-diff baseline for close-out:
  `reports/gt_battery_T12post_2026-07-17`.
- 2026-07-17: W3a (#13 resolver) dispatched to executor with scouted code
  map (resolve toolbox.py:788, _rank ~559, anchor/disambiguator ~624/668);
  systematic-debugging + TDD, no single-question threshold tuning allowed.
- 2026-07-17: W3a landed (25fd955) + verifier gate CONFIRMED (independent
  battery re-run, aggregate == T12post; refutation probes on synonym and
  bare-noun anchors held). Issues #20 (speaker OR, pre-existing) and #21
  (anchor-tier latent caveat) filed. Branch pushed to origin.
- 2026-07-17: W3b (#11/#12) dispatched: raw RGB + luminance cutoff for
  black/white, dominance floor for bridged aliases; thresholds live with
  calibration constants for the future CV sweep; hard gate = fast tier
  green + full battery true_accuracy >= 13/15 (target 15/15) with OR/IF
  toplines unchanged, diffed vs gt_battery_T12post_2026-07-17.
- 2026-07-18: W3b landed (68bbf3f) — numerical 13/15 -> 15/15, only the
  two target rows flip vs T12post; fast tier 1081 passed, full gate exit 0.
  Implementer isolated an apparent non-target diff (chinese_room/hb2 IF
  drive lengths, livingroom_3 OR target 31->58) as PRE-EXISTING
  baseline-vs-HEAD drift: the committed T12post report was generated at
  dirty 1f0c99e, not current main. Verifier gate dispatched (semantic
  edge probes: black-bin identity, dark-saturated exclusion, gray identity
  retention, dominance floor; independent full battery re-run + diff).
- 2026-07-18: W3b verifier gate CONFIRMED. Key evidence: (1) fresh full
  battery reproduces committed T13post with 0 changed rows; (2) bins-on/off
  isolation on identical code — only the 2 target numerical rows flip,
  every OR/IF row byte-identical, proving the shared `_attr_present` path
  changed no OR/IF outcome; (3) the 4 non-numerical rows drifted vs T12post
  are provably the #13 fix (no color words, terminal-goal signature,
  identical under color toggle); (4) all semantic edge probes pass
  (black-bin identity regardless of luma, dark-saturated maroon excluded
  from black, gray identity retained, CSV percentages verified [0,1]);
  (5) loader degrades gracefully on missing/garbage color columns.
  Anomaly noted: stale LibreOffice lock file in
  data/vla3d/Unity/home_building_2/ (GT CSV open in a spreadsheet app) —
  harmless, close it to avoid accidental GT edits.
- 2026-07-18: close-out — scratch battery dirs removed, all 9 issues
  (#10-#13, #15-#19) fixed on this branch; follow-ups #20/#21 filed.
  PR opened to main.
