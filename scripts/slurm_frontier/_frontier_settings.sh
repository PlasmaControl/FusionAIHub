# shellcheck shell=bash
# Sourced by every Frontier SLURM wrapper. Wrappers cd to the FusionAIHub
# repo root before sourcing, so $PWD = repo root here.

module load PrgEnv-gnu/8.7.0
module load cpe/26.03
module load rocm/7.1.1
module load craype-accel-amd-gfx90a
export LD_LIBRARY_PATH="${CRAY_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH}"

PIXI_ENV="$PWD/.pixi/envs/frontier"
export PATH="${PIXI_ENV}/bin:${PATH}"
export LD_LIBRARY_PATH="${PIXI_ENV}/lib:${LD_LIBRARY_PATH}"
export CONDA_PREFIX="${PIXI_ENV}"

# Performance / correctness knobs
export PYTORCH_ROCM_ARCH=gfx90a
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export HSA_FORCE_FINE_GRAIN_PCIE=1

# flash-attn 2 on ROCm: main_perf branch requires this at IMPORT time to
# take the Triton-AMD (aiter) path; otherwise it tries `flash_attn_2_cuda`.
export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE

# RCCL over Slingshot HSN
export NCCL_SOCKET_IFNAME=hsn0
export NCCL_NET_GDR_LEVEL=3
export FI_MR_CACHE_MONITOR=kdreg2
export FI_CXI_DEFAULT_CQ_SIZE=131072

# MIOpen kernel cache: per-job, node-local
export MIOPEN_USER_DB_PATH="/tmp/${USER}-miopen-${SLURM_JOB_ID}"
export MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_USER_DB_PATH"
mkdir -p "$MIOPEN_USER_DB_PATH"

# Distributed master endpoint
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_NODELIST" | head -n1)"
export MASTER_PORT=29500
