# NUS SoC Compute Cluster — Walkthrough

*For model training / Linux work over SSH from Windows. Verified against the [CS3210 SoC GPU guide](https://www.comp.nus.edu.sg/~cs3210/student-guide/soc-gpus/) and [ybchen97's cluster guide](https://github.com/ybchen97/soc_cluster_guide), 10 Jul 2026. Items marked TODO need verifying once you have access — official docs: [SoC DocHub compute-cluster guides](https://dochub.comp.nus.edu.sg/cf/guides/compute-cluster/hardware).*

## 1. Get access (one-time, ~10 min)

1. SoC UNIX account (if you don't have one): https://mysoc.nus.edu.sg/~newacct
2. Enable "SoC Compute Cluster" service: https://mysoc.nus.edu.sg/~myacct/services.cgi
3. Off-campus: install the **SoC VPN** (not the general NUS VPN — it doesn't reach the cluster).

## 2. Connect from Windows

```powershell
ssh <soc-unix-id>@xlogin.comp.nus.edu.sg    # or xlogin0/1/2 for a specific node
```

Windows 11 has OpenSSH built in. Set up `~/.ssh/config` on the login node for GitHub (port 22 is blocked):

```
Host github.com
    Hostname ssh.github.com
    Port 443
```

Quality of life: VS Code Remote-SSH (or `claude` CLI on the login node) makes the cluster feel local. Add key-based auth (`ssh-copy-id`) to skip passwords.

## 3. Slurm basics

```bash
sinfo                                   # what's up
squeue --me                             # my jobs
srun --gpus=1 --pty bash                # interactive GPU shell
sbatch train.sh                         # batch job
```

GPUs are deny-by-default — always request explicitly (`--gpus=1` / `#SBATCH --gpus=1`).

### GPU inventory

| Nodes | GPU | VRAM |
|---|---|---|
| xgpc[0-9] | Tesla V100 | 16 GB |
| xgpd[0-9] | Titan V | 12 GB |
| xgpe[0-11] | Titan RTX | 24 GB |
| xgpf[0-10] | Tesla T4 | 16 GB |
| xgpg[0-9] | A100 | 40 GB |
| xgph[0-9] | A100 | 80 GB |
| xgph[10-19] | A100 MIG slice | 40 GB |
| xgpi[0-9] | H100 | 96 GB |
| xgpi[10-19] | H100 MIG slice | 47 GB |

Target a model: `srun --gres="gpu:a100-80:1" ...` or `--gpus=h100-96`. MIG slices (xgph/xgpi 10–19) queue faster — fine for fine-tuning smaller models.

### Time limits — the catch

Default partition allows **15 min, extendable to 3 h** (`--time=03:00:00`). ⚠️ TODO once you have access: check `sinfo` for a `long`/`medium` partition and its ceiling — multi-hour training runs need it, otherwise design training to **checkpoint + resume** in ≤3 h chunks (good practice anyway).

### Batch job template

```bash
#!/bin/bash
#SBATCH --job-name=vla-train
#SBATCH --gpus=a100-80:1
#SBATCH --time=03:00:00
#SBATCH --output=logs/%j.out
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vla
python train.py --resume-from-latest   # must checkpoint: 3h wall limit
```

## 4. Environment on the cluster

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
conda create -n vla python=3.12 && conda activate vla
pip install torch numpy open3d ...
```

⚠️ TODO: home-dir quota is small (historically ~10–20 GB). Check `quota -s`; put datasets/checkpoints on the scratch/temp filesystem the DocHub guide names for your cluster. VLA-3D is large — download the subset you need, not all 7.6K scenes.

## 5. What the cluster is (and isn't) for, in this project

- ✅ Fine-tuning / evaluating grounding & referring-expression models on VLA-3D
- ✅ Batch perception experiments (detector comparisons over sample data)
- ✅ A general Linux box for smoke-testing the `ai_module` core (no ROS needed for `core/`)
- ❌ Running the Unity simulator (GPU rendering + GUI; not a cluster workload; also no Docker for users — jobs run bare or via Singularity/Apptainer, TODO verify availability)
- ❌ The final eval environment — that's your future Ubuntu machine + Docker
