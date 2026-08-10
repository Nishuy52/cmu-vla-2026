#!/bin/sh
#SBATCH --job-name=sim
#SBATCH --time=120                      # minutes; short-job partitions schedule faster
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
##SBATCH --gpus=1                       # uncomment for any GPU
##SBATCH --gpus=a100:1                  # or request a specific type (a100, t4, titan, ...)
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=cyuhsin@comp.nus.edu.sg

# Cluster home is NFS (500GB). For I/O-heavy sims, stage into node-local /tmp
# (fastest storage; wiped when the job ends — copy results back out).
TMPDIR=$(mktemp -d)
cp -r "$HOME/simdata" "$TMPDIR/" 2>/dev/null || true

srun echo "replace with your sim command, e.g.: srun ./run_sim --data $TMPDIR/simdata"

# Collect results before /tmp is wiped
mkdir -p "$HOME/results/$SLURM_JOB_ID"
cp -r "$TMPDIR"/out* "$HOME/results/$SLURM_JOB_ID/" 2>/dev/null || true
rm -rf "$TMPDIR"
