# NUS SoC Compute Cluster — Walkthrough

*For model training / Linux work over SSH from Windows. Verified against the [CS3210 SoC GPU guide](https://www.comp.nus.edu.sg/~cs3210/student-guide/soc-gpus/) and [ybchen97's cluster guide](https://github.com/ybchen97/soc_cluster_guide), 10 Jul 2026. Refreshed 25–26 Jul 2026 via live sessions — items below are now verified against the real cluster (`xgpe5`, Slurm partitions, quotas, etc.) rather than the docs; official docs remain: [SoC DocHub compute-cluster guides](https://dochub.comp.nus.edu.sg/cf/guides/compute-cluster/hardware).*

## 1. Get access (one-time, ~10 min)

1. SoC UNIX account (if you don't have one): https://mysoc.nus.edu.sg/~newacct
2. Enable "SoC Compute Cluster" service: https://mysoc.nus.edu.sg/~myacct/services.cgi
3. Off-campus: install the **SoC VPN** (not the general NUS VPN — it doesn't reach the cluster).

**Verified 25 Jul 2026:** `ssh xlogin` works passwordless from the laptop — key auth + ControlMaster multiplexing through the SoC jump host, configured in `~/.ssh/config` (cluster user `cyuhsin`). All cluster work in this project goes through this alias; don't hand-type the full `xlogin.comp.nus.edu.sg` host each time.

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

**OS/toolchain (verified):** Ubuntu 24.04, Python 3.12.

**⚠️ Login-node trap (verified 25 Jul 2026):** the login node caps `ulimit -u 64` (max user processes). Go binaries, heavy `pip install`s, and anything thread/process-hungry **will crash there** with confusing errors — this is a login-node process-limit artifact, not a real dependency failure. Run all heavy work via `srun`/`sbatch` instead; compute nodes have full limits **and** outbound internet, so downloads work fine from inside jobs.

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

**GRES types (verified 25 Jul 2026):** `nv` (Titan RTX / V100 / T4 class), `a100-40`, `a100-80`, `h100-47`, `h100-96`, `h200-141`. Request with e.g. `--gpus=nv:1`.

**Broken/avoid nodes (verified 25 Jul 2026)** — always submit GPU jobs with `-x xgpe0,xgpe2,xgpe6`:
- `xgpe0` — driver mismatch, jobs fail
- `xgpe2` — no GPU present
- `xgpe6` — CUDA init failure (probed 26 Jul 2026: `nvidia-smi` fails, torch `cuda.is_available()` False with "CUDA unknown error")
- `xgpe5` — confirmed healthy: Titan RTX 24GB, driver 575.57.08

### Time limits — the catch

Default partition allows **15 min, extendable to 3 h** (`--time=03:00:00`). **Verified 25 Jul 2026:** Slurm partitions are `normal`/`gpu` (3 h walltime cap) and `long`/`gpu-long` (3 days). Multi-hour training runs should use the `long`/`gpu-long` partition; otherwise design training to **checkpoint + resume** in ≤3 h chunks (good practice anyway).

**Fairshare (verified 25 Jul 2026)** — billed on *allocated* walltime, not busy time; an idle interactive shell costs the same as one doing real work:
- CPU: 0.5 units/unit-hr
- GPU, per-hour: `nv` 4.0 (`gpu`) / 6.0 (`gpu-long`); `a100-40` 10 (`gpu`) / 15 (`gpu-long`); `h100` 12–30
- Usage decays with a 10-day half-life; check standing with `sshare -U`

Etiquette: start allocations on demand and tear them down the moment you're done; chain downloads/installs onto CPU partitions (never burn GPU hours on a download); prefer `nv` for serving where it's sufficient — `a100`/`h100` cost 2.5–7x more per hour.

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

**Storage (verified 25 Jul 2026):** home is 500 GB NFS quota, **not backed up**. `/mnt/scratch` is 500 GB, slower, and has no per-user directory precreated — create your own. VLA-3D is large — download the subset you need, not all 7.6K scenes.

**Containers (verified 25 Jul 2026):** docker/apptainer/podman are all unusable — AppArmor blocks user namespaces. Don't fight this; run bare venvs instead.

**venv gotcha (hit 25 Jul 2026):** `~/venv-gpu` carries `torch 2.13.0+cu126` installed from the cu126 wheel index. A plain `pip install torchvision` pulls the PyPI default-CUDA build, which is ABI-incompatible and fails at import with `RuntimeError: operator torchvision::nms does not exist` — this surfaces misleadingly as a transformers `Could not import module 'BertModel'` lazy-import error, not an obvious torchvision problem. Fix:
```bash
pip install --force-reinstall --no-deps --index-url https://download.pytorch.org/whl/cu126 "torchvision==0.28.0+cu126"
```

**Ollama on the cluster (verified 25 Jul 2026):** the official install URLs changed — both `ollama.com/download/*.tgz` and the GitHub `ollama-linux-amd64.tgz` asset now 404. As of v0.32.3 the release asset is `ollama-linux-amd64.tar.zst` (extract with `tar --zstd`). We run a user-space install at `~/ollama/bin/ollama`, with models under `~/ollama-models` (pull them from a CPU allocation, not the login node — see the login-node trap above).

**GroundingDINO offload assets (staged 25–26 Jul 2026):** SwinB checkpoint (938,057,991 bytes, release `v0.1.0-alpha2`) + `GroundingDINO_SwinB_cfg.py` under `~/models/gdino/`; `bert-base-uncased` prefetched into the default HF cache. Verified on a Titan RTX (`xgpe5`): SwinB fp32 load takes 7.5 s, ~1.26 GiB VRAM.

The dev detector/LLM offload workflow (servers + SSH tunnels) lives in `tools/cluster/` (`servers.sh start/stop/status`, `tunnel.sh`) — cross-reference that and the offload subsection of `docs/ubuntu_setup.md` rather than duplicating the steps here.

## 5. What the cluster is (and isn't) for, in this project

- ✅ Fine-tuning / evaluating grounding & referring-expression models on VLA-3D
- ✅ Batch perception experiments (detector comparisons over sample data)
- ✅ A general Linux box for smoke-testing the `ai_module` core (no ROS needed for `core/`)
- ✅ Offloading the GroundingDINO detector / LLM serving via `tools/cluster/` + SSH tunnels (see §4)
- ❌ Running the Unity simulator (GPU rendering + GUI; not a cluster workload)
- ❌ Docker/Apptainer/Podman for users — AppArmor blocks user namespaces (verified 25 Jul 2026); use bare venvs
- ❌ The final eval environment — that's your future Ubuntu machine + Docker
