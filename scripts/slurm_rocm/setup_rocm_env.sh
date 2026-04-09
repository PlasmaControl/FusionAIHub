#!/bin/bash
# Run this once on della-milan to set up the ROCm pixi environment
# and validate GPU detection.
set -euo pipefail

cd /scratch/gpfs/nc1514/FusionAIHub

echo "=== Installing ROCm pixi environment ==="
pixi install -e rocm

echo ""
echo "=== ROCm GPU Check ==="
pixi run -e rocm python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'ROCm available (via torch.cuda): {torch.cuda.is_available()}')
print(f'GPU count: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
print(f'HIP version: {torch.version.hip}')
"

echo ""
echo "=== RCCL Check ==="
pixi run -e rocm python -c "
import torch.distributed as dist
print(f'NCCL available: {dist.is_nccl_available()}')
"

echo ""
echo "=== Import Check ==="
pixi run -e rocm python -c "
from tokamak_foundation_model.models.model_factory import build_model, MODEL_REGISTRY
print(f'Model registry: {list(MODEL_REGISTRY.keys())}')
from tokamak_foundation_model.trainer.trainer import UnimodalTrainer
print('All imports OK')
"

echo ""
echo "=== Setup Complete ==="
