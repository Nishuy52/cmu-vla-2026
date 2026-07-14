# CV Sweep — Rerun Brief

*Updated 14 Jul 2026 (written 11 Jul evening). The first k-fold calibration sweep run was killed
before completion; this is the how-and-why for the rerun. Delete this file once the sweep results
are committed and adopted.*

## What happened

The sweep (`core/runner/cvsweep.py`, committed) ran ~55 CPU-minutes as two detached processes,
contending with test-suite timing measurements on the same machine. A process suspension failed
to hold (psutil suspend on Windows — CPU kept accumulating), so both processes were killed on
11 Jul ~12:30. No results were written (output buffered; `reports/cvsweep_run.*` all 0 bytes).
Unknown whether the per-(config, scene) cache was in-memory or on-disk — check
`core/runner/cvsweep.py` before rerunning; if the cache persists to disk, the rerun resumes
cheaply.

## What the sweep is (context)

Leave-3-scenes-out cross-validation (5 folds × 3 scenes, mirroring the 3 hidden test scenes) over
8 high-sensitivity geometry calibration thresholds (near/on/above/next_to/in family), scored by
the ground-truth battery with the *repaired*, points-weighted objective. Outputs to
`reports/cvsweep_<date>/`: `report.md`, `results.json`, `recommended_calibration.json` +
per-parameter stability table + generalization gap.

## Done: objective re-pointed, min_obs dropped, disk cache added (NUM-F7 / H2, 14 Jul)

`core/runner/cvsweep.py` is DONE — no longer under the H2 hold. What changed:

- **Objective re-pointed at the repaired scorer columns.** NUMERICAL (1pt each) is now STRICT
  independent count agreement — our count vs. the `referential` (relation-aware) independent
  opinion, over questions that HAVE strict referential evidence only (`_strict_independent`).
  Relation-agnostic `*_class_only` rows and scene-graph-only rows are NOT evidence and are
  EXCLUDED from both numerator and denominator — this mirrors `gt_battery.aggregate`'s
  `independent_source == "referential"` filter exactly (the old class-only-inclusive
  `_best_independent` fallback is retired). OBJECT_REFERENCE (2pt each) stays IoU>=0.25 on
  scoreable questions. INSTRUCTION_FOLLOWING (6pt each) now scores the DRIVEN-trajectory RUBRIC
  (`score_instruction_rubric` — ordered per-leg arrival credit minus threading/avoid penalties)
  over the *aligned* questions, replacing the old planned-path coverage@1m the challenge rubric
  never paid for.
- **`counting.min_obs` dropped from the sweep space** (NUM-F7): every GT instance has `n_obs=3`,
  so the gate is provably inert on GT data — it was dead weight, never a tunable that could move
  the objective. The default spec is geometry-only now (8 keys, all live
  `core.geometry.toolbox.Thresholds` fields).
- **Per-(config_hash, scene) on-disk JSON cache added** under `<out>/cache/` (atomic tmp-then-
  replace write, corrupt-cell tolerant — a truncated cell from a killed process is treated as
  absent and recomputed, never a crash). A re-run against the same `--out` resumes: every cell
  already on disk is a hit (`disk_hits` reported alongside `cache_hits`/`cache_misses`) and is not
  recomputed. `--no-cache` disables disk persistence entirely (memory-only, no resume) for fast
  unit-test runs.

Tests for all of the above (objective weighting/exclusion, grid-vs-Thresholds consistency,
cache resume/no-cache/corrupt-tolerance) live in `src/tests/runner/test_cvsweep.py`.

## Rerun instructions (pick one)

**Option A — overnight on this machine (simplest):**
```powershell
cd "C:\Users\jyuc1\Documents\Uni\2026 CMU VLA\src"
python -m core.runner.cvsweep --groundtruth ..\data\vla3d\Unity --n-samples 60 --out ..\reports\cvsweep_2026-07-14 > ..\reports\cvsweep_run.log 2>&1
```
- `--groundtruth` is required (Unity root with per-scene folders matched to `questions.json`
  scenes). The on-disk cache lives under `<out>/cache/` automatically — an interrupted run
  resumes cheaply just by re-invoking with the same `--out`; add `--no-cache` only for a clean,
  from-scratch timing run.
- Run it when NOTHING else needs the machine (the twice-learned lesson: it pegs a core for
  1–3+ hours and corrupts any concurrent timing/benchmark work).
- First: smoke it with `--n-samples 3 --scenes-subset loft,office_1` (~minutes) to confirm the
  wiring still matches current code, and check whether progress prints line-buffered (if the log
  stays 0 bytes for >15 min, add `PYTHONUNBUFFERED=1`).

**Option B — SoC cluster (if local hours are precious):**
Pure-Python + numpy, no GPU needed — clone the repo + copy `data/vla3d/Unity` (or re-download
there, it's on a CMU server), `pip install numpy rosbags`, run the same command under `sbatch`;
the on-disk cache under `<out>/cache/` gives checkpoint-resume for free across `sbatch`
re-submissions in ≤3 h slices if the long partition is unavailable
(`docs/soc_cluster_guide.md`).

## Gate: still on hold pending the IF poses-collapse investigation

The sweep code is done and tested, but the RERUN itself stays on hold until the IF driven-sim
poses-collapse investigation confirms the rubric numbers it now optimises are trustworthy — an
IF objective driven by a collapsed/degenerate pose stream would calibrate thresholds against a
broken signal. Un-hold condition: that investigation closes with the rubric scores confirmed
sound on real driven trajectories; only then run the rerun instructions above for real.

**If wall time is the constraint either way:** `--n-samples 30` halves the cost; the stability
table (not the peak score) is the real product, and it converges with fewer samples.

## After it completes

1. Read `report.md`: fold scores, generalization gap, per-parameter stability.
2. Adopt ONLY parameters with cross-fold modal consensus into a committed
   `calibration_adopted.json` (via `core/calibration.py` `apply_overrides`); leave unstable
   parameters at defaults — that instability is a finding, not a nuisance.
3. Re-run the GT battery once with adopted values; confirm the numerical over-count
   (11-vs-9 pillows class of error) improves and nothing regresses.
4. Commit results + adopted calibration, update `LOG.md` (one-liner per the task-record
   workflow), push, and delete this brief.
5. Note for Phase 2: the sweep covers *reasoning* thresholds only — perception tunables
   (fusion/tracker/keyframe/overhead) get swept in the sim per `docs/phase2_playbook.md`.
