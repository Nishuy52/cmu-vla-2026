# Live object-reference analysis — why all three runs score `n/a` (27 Jul 2026)

**Verdict: all three `n/a` results are a SCORER-side (GT-resolution) gap, not a
pipeline failure.** The robot published a marker in every case; the scorer could not
identify which GT object the question refers to, so IoU is undefined (correctly
flagged rather than guessed — `tools/score_live_run.py:408`).

This is **not** the same failure mode as the offline battery's known "24 vocabulary-drift
non-matches": every referring noun here IS in the scene's GT vocabulary, directly or via
the scorer's existing bridging (`cup`↔`paper cup`, `plant`↔`potted plant`,
`file cabinet`↔`cabinet`) — verified by direct calls to `scoring._anchor_agrees`
(`src/core/groundtruth/scoring.py:161`).

Note: slot 3's actual `/challenge_question` is "Find the paper cup on the table closest
to the projector screen."; "Find the potted plant on the file cabinet." is slot 4.

| # | Question | Scene | Noun in GT vocab | Ladder outcome | Root cause |
|---|---|---|---|---|---|
| 1 | pillow on the sofa closest to the windows | livingroom_1 | yes — pillow(3), sofa(1), window(2) | `ambiguous` | The parser nests "closest to windows" as the **sofa anchor's** disambiguator, not a top-level clause on `pillow`, so the superlative-restriction path (`scoring.py:644`) never fires. The fallback `on`+sofa filter matches 2 pillows that tie exactly (Jaccard 0.545 vs 0.545) → fails the 0.05 margin |
| 2 | paper cup on the table closest to the projector screen | office_1 | yes — cup≈paper cup(3), projector screen(1) | `ambiguous` | Ordinal relation family ("closest"/"second closest"/… share one relation string by design, `scoring.py:618-624`) pulls in 2 distractors; the true match (id 84, 0.583) misses the 0.05 margin over the runner-up (0.538) by **0.045** |
| 3 | potted plant on the file cabinet | office_1 | yes — plant≈potted plant(4), file cabinet(1) | `none` | The referential-statement generator **never emits an `on`/`near`/`above` relation for target-class `plant` in this scene** (checked across 1054 plant statements) — zero candidate statements exist. A corpus coverage gap, not a matching bug |

## Interim signal (marker vs nearest GT instance, since IoU is undefined)

| Question | Marker centroid (object frame) | Nearest instance of target class | Distance |
|---|---|---|---|
| pillow | (-0.02, -4.16, 0.56) | id 41 (maroon pillow) | 1.63 m |
| paper cup | (-0.38, -3.59, 0.36) | id 84 — the GT ladder's own top-ranked (but rejected) candidate | 2.00 m |
| potted plant | (-1.42, -6.98, 0.53) | id 71 (closest of 4) | 8.02 m |

- **paper cup**: plausible — the marker's nearest GT cup is the same instance the GT
  ladder ranked first before the tie-break rejected it.
- **pillow**: inconclusive — right area, but the nearest pillow disagrees with the
  scorer's own regex-resolver top pick.
- **potted plant**: a flag, not a signal — marker y=-6.98 falls **outside** office_1's
  annotated GT extent (y ∈ [-5.33, 1.72]), and all 4 candidates are 8-10 m away.
  Worth checking this bag's frame-fit before concluding the live answer was wrong.

## Code references

- `tools/score_live_run.py:386` `score_object_reference_run`, `:408` the n/a note
- `src/core/groundtruth/scoring.py:566` `_gt_target_from_referential` (exact→fuzzy→relation ladder)
- `scoring.py:644` superlative clauses read only **top-level** `plan.target.clauses` (case 1)
- `scoring.py:691` `top_j >= 0.5 and top_j - second_j >= 0.05` — the margin that rejects case 2
- Prior context: `docs/redteam/attack_object_ref_checkpoints.md` F2 found the identical
  nested-disambiguator hole in the **live** resolver; case 1 shows the **scorer's** own
  `parse_regex` call has it too

## Proposed fixes (proposals only — all scorer/GT-corpus side)

1. Recurse superlative-clause collection (and `_question_relations`) into anchor
   `disambiguator` chains, not just top-level clauses — closes case 1 and mirrors the
   F2 fix already recommended for the live resolver (one shared helper could fix both).
2. Scale the tie-break margin (`scoring.py:691`) to statement length instead of a fixed
   0.05 absolute floor — closes case 2 without weakening tie protection.
3. Audit referential-statement generation so every target class gets `on`/`near`/`above`
   statements where the VLA-3D annotations support them (case 3).

None of these touch the live pipeline or the published markers.

## Caveat recorded

While investigating, a scorer invocation with a default `--out` overwrote the
`reports/live_baseline_2026-07-20/` scores; this was detected and the files were
restored to their committed state (`git checkout`), verified clean. Any future
single-run scoring must pass an explicit `--out`.
