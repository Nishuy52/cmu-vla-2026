# Cluster mechanics

Exact commands and contamination rules for the NUS SoC cluster.
Access: `ssh xlogin` (needs a live ControlMaster; the user opens it
interactively once per boot with `ssh xlogin true`). User: cyuhsin.

## What the batch job reads, and when

| Item | Read at | Consequence |
|---|---|---|
| `~/cluster_verify_batch.sbatch` | submit (Slurm snapshot) | Safe to overwrite after submit. |
| `MATRIX` file | run time | Unique name per submission, always. |
| `~/vla/src` | run time, per question | Never write while any job runs. |
| `~/question_pub.py`, `~/live_monitor.py` | every question | Never scp while any job runs; scp truncates in place. |

A verify job occupies the single udocker container and scene overlay.
Chain with `afterany`; never run two at once.

## Build a matrix

From the PINNED tree (the script resolves questions.json through the repo):

```bash
cd <pin-dir>
QTYPES=inst QMAX=2 .venv/bin/python - <out.txt> <scene> <scene> ... <<'PYEOF'
# (the matrix builder from tools/cluster/live_run/push_and_verify_batch.sh,
#  lines 35-79, parameterized by QTYPES/QMAX env vars)
PYEOF
```

Group shapes that fit the 155-minute wall guard at ~15.7 min/question:
10 questions per job. The proven splits: IF 3x10 (2 per scene), OR 16+14,
numerical 15 (1 per scene).

For an A/B: `diff` the new matrix against the baseline submission's matrix
file, which stays on the cluster under its unique name. Byte-identical or
you do not have an A/B.

## Submit

```bash
scp -q <matrix> xlogin:verify_batch_matrix_<tag>_<group>.txt
ssh xlogin "cd ~ && sbatch --dependency=afterany:<prev> \
  --export=ALL,MATRIX=\$HOME/verify_batch_matrix_<tag>_<group>.txt \
  ~/cluster_verify_batch.sbatch"
ssh xlogin "squeue -u cyuhsin -o '%.10i %.2t %.10M %.30E'"
```

Confirm PD + the correct predecessor in DEPENDENCY before the next
submission. Extract job ids with an anchored pattern; MOTD digits on the
ssh channel once threatened a wrong `afterany` target.

## Sync the tree

```bash
rsync -az --delete --exclude=__pycache__ --exclude=.pytest_cache \
  <pin-dir>/src/ xlogin:vla/src/
# proof:
rsync -rcn --exclude=__pycache__ --exclude=.pytest_cache \
  <pin-dir>/src/ xlogin:vla/src/ | grep -vE '^$|/$' | wc -l   # must be 0
```

Only when no job runs. `-c` compares content; mtime comparisons lie here.

## Harvest

```bash
tools/cluster/live_run/harvest_verify.sh <jobid>
```

Run it from the repo root with the repo venv active. It pulls the job
stdout, per-slot logs, debug dirs, and bags, then scores. If scoring
reports nothing, score by hand:
`python -m tools.score_live_run <run-dir> --out <captures-dir>` with
`PYTHONHASHSEED=0`, one `timeout` per run, serialized.

## Per-job assertions (before any number is quoted)

```bash
ssh xlogin "grep -o 'VERIFY-BATCH-DONE.*' ~/verify_batch_<J>.out"
ssh xlogin "grep -c '^SLOT .*SUCCESS' ~/verify_batch_<J>.out"
ssh xlogin "grep -c DEGRADED ~/verify_batch_<J>.out"
rsync -rcn <pin-dir>/src/ xlogin:vla/src/ | grep -vE '^$|/$' | wc -l
```

The last one is the src-integrity check. Nonzero, or any file in
`~/vla/src` with an mtime after submit time, means contamination: discard
the group and resubmit. A cancelled submission from another session once
rewrote one file seven minutes into a run.

## Known cluster facts

- All 15 scenes live in `~/scenes`; no re-upload needed.
- Disk: `df -h ~` — the filesystem is large; quota has not bound.
- A DEGRADED slot burns its full 780 s; a hard failure exits in ~1 min.
- The wall guard truncates the matrix at WALL_MIN*60-1200 s; assert the
  attempted count on every harvest.
- The loft scene starves sensors more than any other (issue #174); treat a
  flagged loft slot as harness noise, not pipeline signal.
