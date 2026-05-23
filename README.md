# FusionAIHub (FAITH)

## Frontier setup

```bash
# 1. Clone to scratch
cd /lustre/orion/fus187/scratch/$USER
git clone git@github.com:PlasmaControl/FusionAIHub.git
cd FusionAIHub
git switch foundation_model

# 2. Install pixi
curl -fsSL https://pixi.sh/install.sh | bash
source ~/.bashrc

# 3. Install the Frontier env (~5 min)
pixi install -e frontier

# 4. Build flash-attention 2 (~2-5 min)
pixi run -e frontier setup-flash-attn


## Other platforms

- **NVIDIA/CUDA**: `pixi install` (default env), scripts in `scripts/slurm/`
- **della-milan (MI210)**: `bash scripts/slurm_della_milan/setup_rocm_env.sh`,
  scripts in `scripts/slurm_della_milan/`
