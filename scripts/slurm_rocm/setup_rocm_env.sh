#!/bin/bash
# Run this once on della-milan to create a ROCm venv for MI210 (gfx90a).
# Usage: bash scripts/slurm_rocm/setup_rocm_env.sh
set -euo pipefail

PROJECT_DIR=/scratch/gpfs/EKOLEMEN/nc1514/FusionAIHub
cd "$PROJECT_DIR"

VENV_DIR=".venv-rocm"
PY="$PROJECT_DIR/.pixi/envs/default/bin/python3.11"

export PATH=/opt/rocm-7.2.1/bin:$PATH
export LD_LIBRARY_PATH=/opt/rocm-7.2.1/lib:${LD_LIBRARY_PATH:-}

if [ ! -x "$PY" ]; then
    echo "ERROR: pixi python not found at $PY. Run 'pixi install' first." >&2
    exit 1
fi

if [ ! -d "$VENV_DIR" ]; then
    echo "=== Creating ROCm virtual environment ($VENV_DIR) ==="
    "$PY" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

echo "=== Installing PyTorch (ROCm 6.2 wheels, ABI-compat with system ROCm 7.2) ==="
pip install --upgrade pip
pip install --index-url https://download.pytorch.org/whl/rocm6.2 torch torchvision \
  || pip install --index-url https://download.pytorch.org/whl/rocm6.1 torch torchvision

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
assert torch.cuda.is_available(), 'GPU not visible to torch'
assert hip is not None, 'torch is not a ROCm build'
"

echo ""
echo "=== RCCL Check ==="
python -c "
import torch.distributed as dist
print(f'NCCL/RCCL available: {dist.is_nccl_available()}')
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
