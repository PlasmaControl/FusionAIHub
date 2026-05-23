# Frontier-common environment for ROCm DDP jobs.
# Source from every Frontier SLURM script BEFORE activating the venv.
# Sets modules, RCCL/NCCL knobs, MIOpen cache, and MASTER_ADDR/PORT.
#
# Frontier hardware reminders (see docs.olcf.ornl.gov):
#   - 4x MI250X = 8 GCDs per node, each appears as a separate GPU.
#   - HSN is Slingshot via libfabric/cxi; RCCL needs hsn0 + kdreg2.
#   - MIOpen cache in $HOME is slow & contended; redirect to /tmp.

# shellcheck shell=bash

module load PrgEnv-gnu/8.7.0
module load cpe/26.03
module load rocm/7.1.1
module load craype-accel-amd-gfx90a
export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH:-}"

# Pixi env activation. One-time setup:
#     pixi install -e frontier
# We do NOT use `pixi shell-hook` here because it re-resolves the lockfile
# on every invocation, which hangs indefinitely on Frontier's autofs UV cache
# under contention (we saw 30s+ hangs in interactive testing). Instead we
# manually prepend the env's bin/lib to PATH/LD_LIBRARY_PATH — this is what
# pixi shell-hook would do anyway for a non-conda env.
export PATH="$HOME/.pixi/bin:$PATH"
_FRONTIER_COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_FRONTIER_REPO_ROOT="$(cd "${_FRONTIER_COMMON_DIR}/../.." && pwd)"
_FRONTIER_PIXI_ENV="${_FRONTIER_REPO_ROOT}/.pixi/envs/frontier"
if [ ! -x "${_FRONTIER_PIXI_ENV}/bin/python" ]; then
    echo "ERROR: frontier pixi env missing at ${_FRONTIER_PIXI_ENV}" >&2
    echo "       Run \`pixi install -e frontier\` once from a login node." >&2
    exit 1
fi
export PATH="${_FRONTIER_PIXI_ENV}/bin:${PATH}"
export LD_LIBRARY_PATH="${_FRONTIER_PIXI_ENV}/lib:${LD_LIBRARY_PATH:-}"
export CONDA_PREFIX="${_FRONTIER_PIXI_ENV}"

# Performance / correctness knobs
export PYTORCH_ROCM_ARCH=gfx90a
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export HSA_FORCE_FINE_GRAIN_PCIE=1

# flash-attn 2 on ROCm: the main_perf-branch install requires this env var
# at IMPORT time to take the Triton-AMD (aiter) code path. Without it, it
# tries to import `flash_attn_2_cuda` (the NVIDIA CUDA extension) and fails.
export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE

# RCCL over Slingshot HSN
export NCCL_SOCKET_IFNAME=hsn0
export NCCL_NET_GDR_LEVEL=3
export FI_MR_CACHE_MONITOR=kdreg2
export FI_CXI_DEFAULT_CQ_SIZE=131072

# MIOpen kernel cache: per-job, node-local
export MIOPEN_USER_DB_PATH="/tmp/${USER}-miopen-${SLURM_JOB_ID:-local}"
export MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_USER_DB_PATH"
mkdir -p "$MIOPEN_USER_DB_PATH"

# Distributed master endpoint derived from SLURM allocation
if [ -n "${SLURM_NODELIST:-}" ]; then
    MASTER_ADDR="$(scontrol show hostnames "$SLURM_NODELIST" | head -n1)"
else
    MASTER_ADDR="127.0.0.1"
fi
export MASTER_ADDR
export MASTER_PORT="${MASTER_PORT:-29500}"
