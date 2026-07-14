# CV Sweep — Rerun Brief

*12 Jul 2026 (written 11 Jul evening). The first k-fold calibration sweep run was killed before
completion; this is the how-and-why for the rerun. Delete this file once the sweep results are
committed and adopted.*

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
~10 high-sensitivity calibration thresholds (geometry near/on/between family, counting gates),
scored by the ground-truth battery with the non-circular, points-weighted objective (independent
count agreement ×1, OR IoU ×2, IF path coverage ×6). Outputs to `reports/cvsweep_<date>/`:
`report.md`, `results.json`, `recommended_calibration.json` + per-parameter stability table +
generalization gap.

## Rerun instructions (pick one)

**Option A — overnight on this machine (simplest):**
```powershell
cd "C:\Users\jyuc1\Documents\Uni\2026 CMU VLA\src"
python -m core.runner.cvsweep --n-samples 60 --out ..\reports\cvsweep_2026-07-12 > ..\reports\cvsweep_run.log 2>&1
```
- Run it when NOTHING else needs the machine (the twice-learned lesson: it pegs a core for
  1–3+ hours and corrupts any concurrent timing/benchmark work).
- First: smoke it with `--n-samples 3 --scenes-subset loft,office_1` (~minutes) to confirm the
  wiring still matches post-tiering-PR code, and check whether progress prints line-buffered
  (if the log stays 0 bytes for >15 min, add `PYTHONUNBUFFERED=1`).

**Option B — SoC cluster (if local hours are precious):**
Pure-Python + numpy, no GPU needed — clone the repo + copy `data/vla3d/Unity` (or re-download
there, it's on a CMU server), `pip install numpy rosbags`, run the same command under `sbatch`
with checkpoint-resume in ≤3 h slices if the long partition is unavailable
(`docs/soc_cluster_guide.md`).

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
