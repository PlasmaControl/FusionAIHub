# Fusion AI Toolkit & Hub (FAITH)

## Environment Setup
Instead of Anaconda, we use Pixi (much faster) to manage our environment.
1. Install Pixi

```bash
curl -fsSL https://pixi.sh/install.sh | sh
```

1. Create a new environment

```bash
pixi install
```

1. Activate the environment (do this every time you want to use the environment)

```bash
pixi shell
```

## Training

Models can be trained with either single GPU or multi-GPU (DDP).

### Profiling on Princeton Clusters
Type `squeue -u <your_username>` to see your jobs.
Type `jobstats <job_id>` to see the job's statistics.
And you can view the interactive profile by clicking on the link in the output.

### Loggging
Wandb is set to offline by default. To sync it, run
```bash
wandb sync --sync-all --include-offline
```

## Data

Unprocessed data is stored in `/scratch/gpfs/EKOLEMEN/d3d_fusion_data/`.

Unprocessed videos are temporarily stored in `/scratch/gpfs/EKOLEMEN/big_d3d_data/images/`.

Model-ready files are stored in `/scratch/gpfs/EKOLEMEN/foundation_model/`.

Model-ready files should be set with `664` permissions at least.

## Flash Attention

O($N^2$)$\to$ O($N$)

Pytorch automatically uses this as a backend once you install it, so you don't need to do anything special to use it.

Installation depends on the CUDA, Python, and PyTorch versions.

DO NOT USE `pip install flash-attn` since building wheels will take a long time.
Instead, search for a matching wheel for your system from either of the following sources:

- [https://github.com/Dao-AILab/flash-attention/releases](https://github.com/Dao-AILab/flash-attention/releases)
- [https://github.com/mjun0812/flash-attention-prebuild-wheels/releases](https://github.com/mjun0812/flash-attention-prebuild-wheels/releases)

Make sure your GCC version is at least 9 or higher. On Princeton clusters, you should upgrade it via

```bash
module load gcc-toolset/10
```

Then, install flash-attn via

```bash
wget <url>
pip install ninja
pip install <wheel>
```

A pre-downloaded wheel will be made available soon. For now, the url for the wheel on Princeton clusters is:
[https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.6.3+cu128torch2.10-cp311-cp311-linux_x86_64.whl](https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.6.3+cu128torch2.10-cp311-cp311-linux_x86_64.whl)