#!/bin/bash
# One-time bootstrap of the ROCm conda env on Frontier.
# Run on a login node:  bash scripts/slurm_frontier/setup_frontier_env.sh
#
# Uses Frontier's miniforge3 module to create a conda env at
# $PROJECT_DIR/envs/frontier-rocm with python 3.11 and PyTorch ROCm 7.1 wheels
# matching the loaded rocm/7.1.1 module.

set -euo pipefail

PROJECT_DIR=/lustre/orion/fus187/scratch/nchen/FusionAIHub
cd "$PROJECT_DIR"

# Modules + conda init for `conda activate`.
# shellcheck disable=SC1091
source scripts/slurm_frontier/_frontier_common.sh

if [ ! -d "$CONDA_ENV_PATH" ]; then
    echo "=== Creating conda env at $CONDA_ENV_PATH (python 3.11) ==="
    conda create -y -p "$CONDA_ENV_PATH" python=3.11 -c conda-forge
fi

conda activate "$CONDA_ENV_PATH"

echo "=== Installing PyTorch (ROCm 7.1 wheels, matches loaded rocm/7.1.1) ==="
pip install --upgrade pip
pip install --index-url https://download.pytorch.org/whl/rocm7.1 torch torchvision \
  || pip install --index-url https://download.pytorch.org/whl/rocm6.1 torch torchvision

echo "=== Installing project dependencies ==="
pip install -e ".[all]" 2>/dev/null || pip install -e .

echo
echo "=== ROCm Build Check ==="
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'HIP version: {getattr(torch.version, \"hip\", None)}')
print(f'ROCm available (cuda API): {torch.cuda.is_available()}')
print(f'GPU count visible to login node: {torch.cuda.device_count()}')
hip = getattr(torch.version, 'hip', None)
assert hip is not None, 'torch is not a ROCm build'
"

echo
echo "=== RCCL Check ==="
python -c "
import torch.distributed as dist
print(f'NCCL/RCCL available: {dist.is_nccl_available()}')
"

echo
echo "=== Project Import Check ==="
python -c "
from tokamak_foundation_model.models.model_factory import build_model, MODEL_REGISTRY
print(f'Model registry: {list(MODEL_REGISTRY.keys())}')
from tokamak_foundation_model.trainer.trainer import UnimodalTrainer
from tokamak_foundation_model.utils.distributed import DistributedManager
print('All imports OK')
"

echo
echo "=== Setup Complete ==="
echo "Login nodes have no GPUs — torch.cuda.device_count()==0 is expected here."
echo "Conda env: $CONDA_ENV_PATH"
echo "Activate manually with:"
echo "  source scripts/slurm_frontier/_frontier_common.sh && conda activate \$CONDA_ENV_PATH"
