#!/bin/bash
# Run this once on the AMD cluster (della-milan) to create a ROCm venv.
# Usage: bash scripts/slurm_rocm/setup_rocm_env.sh
set -euo pipefail

cd /scratch/gpfs/nc1514/FusionAIHub

VENV_DIR=".venv-rocm"

echo "=== Creating ROCm virtual environment ==="
python3.11 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

echo "=== Installing PyTorch (ROCm 6.3) ==="
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.3

echo "=== Installing project dependencies ==="
pip install -e ".[all]" 2>/dev/null || pip install -e .

echo ""
echo "=== ROCm GPU Check ==="
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'ROCm available (via torch.cuda): {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
hip = getattr(torch.version, 'hip', None)
print(f'HIP version: {hip}')
"

echo ""
echo "=== RCCL Check ==="
python -c "
import torch.distributed as dist
print(f'NCCL available: {dist.is_nccl_available()}')
"

echo ""
echo "=== Import Check ==="
python -c "
from tokamak_foundation_model.models.model_factory import build_model, MODEL_REGISTRY
print(f'Model registry: {list(MODEL_REGISTRY.keys())}')
from tokamak_foundation_model.trainer.trainer import UnimodalTrainer
print('All imports OK')
"

echo ""
echo "=== Setup Complete ==="
echo "Activate with: source $VENV_DIR/bin/activate"
