# Analysis rules

How to compare live results without fooling yourself. Every rule below
comes from a measured mistake in this repo's campaign history.

## Scorer-matched comparison

The rubric changes over time (#100, #155, #162, #165, #170 all moved it).
A mean scored under one rubric version is not comparable to a mean scored
under another. Before any A/B:

1. Re-score the BASELINE bags with the current scorer
   (`tools/score_live_run.py`, `PYTHONHASHSEED=0`).
2. Match rows by (scene, question prefix). Compare per question.
3. Report the matched-subset means, with the n for each side.

Scorer churn alone moved the 30-question live mean by ~0.02 (0.5333 ->
0.5444 -> 0.5222 across three scorer versions on identical bags). Treat
~0.02 as the floor below which a delta is not a signal.

## Noise discipline

- One live run per question per condition. Run variance flips 3-5
  questions of 30 between identical-code runs. A single question is 1/n of
  the mean. Never attribute a single flip to a fix without artifact
  evidence for the mechanism.
- Ordered-leg counts aggregate roughly 2x more events than question means;
  quote them beside every mean.
- A question can be "credited" from 1.4 m away (#100 history). Coverage
  and closest-approach beat leg credit for behaviour questions.

## Row hygiene

- Exclude capture-flagged rows (`capture_issues` non-empty) from means, on
  both sides, and print the flags. The #158 guard fires per row, not per
  job; a job-level DEGRADED count of zero does not mean the rows are clean.
- Excluded-from-mean rows render as unevaluable, not 0.0 (#165). If you
  compute your own mean from rows, skip `headline_live: null` AND
  capture-flagged rows; a naive row average once mis-stated a result by
  0.03.
- Assert phantom labels are absent: no instance label may contain " . "
  (#172). Count them per slot in `instance_index.jsonl` on every harvest.

## Cross-run diagnosis

For a persistent failure, build the three-generation table first:
per generation, the score, legs, and the scorer's per-leg note. Then read
the mechanism from the pattern:

- Same leg fails every generation with a violation note: systematic
  geometry (corridor-gate displacement class, #175).
- Violation at the same coordinates every generation: an avoid anchor that
  never grounds (#173 class). Check the anchor class in every
  generation's final index.
- Score flips between generations with no mechanism: run variance. Do not
  file it; note it.
- A row that improves only in one generation and regresses back: suspect a
  capture flag or contamination before suspecting the code.

## Attribution honesty

- A stack of fixes landing together proves the stack, not any single fix.
  Say so. Per-fix attribution comes from artifacts (prompt diagnostics,
  index composition, replay), not from the headline.
- The offline battery under-predicts live impact of rubric-geometry
  changes (1 question offline vs 8 live for #162) and does not predict
  live behaviour at all (#163: IF 0.889 offline vs 0.533 live). The live
  sweep is the only adoption oracle; the battery is a regression net.
- When you find your own earlier claim wrong, correct the issue record
  first, then the analysis. The tracker is the memory that outlives
  sessions.
