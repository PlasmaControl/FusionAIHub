#!/bin/bash
# Build the Composable Kernel (CK) flash-attention 2 wheel for OLCF Frontier
# (MI250X / gfx90a). Replaces the Triton-AMD backend currently installed by
# `scripts/slurm_rocm/setup_frontier_env.sh` with the real hipcc-compiled CK
# kernels — needed for a fair comparison against nn.MultiheadAttention in the
# profile_stage1_1x1 benchmark.
#
# This is a multi-hour compile (CK template explosion). Fits in 4 h batch.
#
# Usage:
#     sbatch scripts/slurm_frontier/build_flash_attn_ck.sh
#
#SBATCH -A fus187
#SBATCH -J flashattn_ck_build
#SBATCH -o logs/%j_flashattn_ck_build.out
#SBATCH -e logs/%j_flashattn_ck_build.err
#SBATCH -t 04:00:00
#SBATCH -p extended
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=closest
#SBATCH --cpus-per-task=56
set -uo pipefail

PROJECT_DIR=/lustre/orion/fus187/scratch/nchen/FusionAIHub
cd "$PROJECT_DIR"
mkdir -p logs

FLASH_ATTN_LOCAL="${PROJECT_DIR}/.build/flash-attention"
EXPECTED_SHA=5301a359f59ef8fa10f211618d9f7a69716a8898
ROCM_MODULE=rocm/7.1.1

# Module load — needs hipcc + ROCm headers on PATH for the CK compile.
# shellcheck disable=SC1091
source /etc/profile.d/lmod.sh 2>/dev/null || true
module load PrgEnv-gnu "${ROCM_MODULE}" craype-accel-amd-gfx90a
export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH:-}"

# CK backend — do NOT set FLASH_ATTENTION_TRITON_AMD_ENABLE. Restrict to
# gfx90a only so we don't compile MI300 kernels we'll never use.
unset FLASH_ATTENTION_TRITON_AMD_ENABLE || true
export PYTORCH_ROCM_ARCH=gfx90a
export GPU_ARCHS=gfx90a

# Parallel compile. Frontier compute nodes have 64 cores / 512 GB RAM, and
# hipcc on CK templates can use several GB per worker. 32 is a safe middle
# ground — see https://github.com/ROCm/flash-attention#installation
export MAX_JOBS="${MAX_JOBS:-32}"
export NINJA_STATUS="[%f/%t %es] "

PIXI_PY="${PROJECT_DIR}/.pixi/envs/frontier/bin/python"
if [ ! -x "$PIXI_PY" ]; then
    echo "ERROR: frontier pixi env not provisioned at $PIXI_PY." >&2
    echo "       Run \`pixi install -e frontier\` first." >&2
    exit 1
fi

# Verify the clone is at the pinned SHA. Reset submodules to a clean state
# in case a prior attempt left build artifacts.
echo "=== Source state ==="
echo "    source = ${FLASH_ATTN_LOCAL}"
HAVE_SHA="$(cd "$FLASH_ATTN_LOCAL" && git rev-parse HEAD)"
echo "    SHA    = ${HAVE_SHA}"
if [ "${HAVE_SHA}" != "${EXPECTED_SHA}" ]; then
    echo "ERROR: clone at wrong SHA (want ${EXPECTED_SHA})" >&2
    exit 1
fi
echo "    re-syncing submodules"
(cd "$FLASH_ATTN_LOCAL" && git submodule update --init --recursive)

# Wipe any stale build artifacts from prior Triton-only install.
echo "    cleaning prior build artifacts"
rm -rf "${FLASH_ATTN_LOCAL}/build" "${FLASH_ATTN_LOCAL}/dist" \
       "${FLASH_ATTN_LOCAL}/flash_attn.egg-info"

# Drop the existing Triton-backend flash_attn so pip will replace it.
echo ""
echo "=== Removing existing flash_attn install ==="
"$PIXI_PY" -m pip uninstall -y flash_attn || true

echo ""
echo "=== Build env ==="
echo "    host       = $(hostname)"
echo "    python     = ${PIXI_PY}"
echo "    PYTORCH_ROCM_ARCH=${PYTORCH_ROCM_ARCH}"
echo "    GPU_ARCHS=${GPU_ARCHS}"
echo "    MAX_JOBS=${MAX_JOBS}"
echo "    FLASH_ATTENTION_TRITON_AMD_ENABLE=${FLASH_ATTENTION_TRITON_AMD_ENABLE:-unset (CK backend)}"
which hipcc 2>/dev/null && hipcc --version 2>/dev/null | head -3 || echo "    WARN: hipcc not on PATH"
echo ""

echo "=== Building flash-attn 2 CK wheel (this takes 1-3 h) ==="
t_start=$(date +%s)
"$PIXI_PY" -m pip install --no-build-isolation -v "${FLASH_ATTN_LOCAL}"
build_status=$?
t_end=$(date +%s)
echo ""
echo "=== Build duration: $((t_end - t_start)) s ==="

if [ $build_status -ne 0 ]; then
    echo "FAILED with status $build_status" >&2
    exit $build_status
fi

# Smoke-verify the install — exercises the CK kernel on a small input.
echo ""
echo "=== Verifying install ==="
"$PIXI_PY" -c "import flash_attn; print('flash_attn', flash_attn.__version__, '->', flash_attn.__file__)"
"$PIXI_PY" scripts/slurm_rocm/verify_flash_attn.py

echo ""
echo "=== Done. ==="
echo "Re-run the comparison with:"
echo "    sbatch scripts/slurm_frontier/profile_stage1_1x1.sh"
